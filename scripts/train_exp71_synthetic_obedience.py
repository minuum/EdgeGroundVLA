#!/usr/bin/env python3
"""
2026-07-07 사용자 요청: 조이스틱 이질 지시 데이터 수집 전, 합성으로 먼저 근사 테스트.

지금까지 언어 조건화가 counterfactual 0%였던 이유는 "같은 장면에 다른 지시가 오는"
경우가 데이터에 전혀 없었기 때문(장면<->지시가 1:1). 이 스크립트는 실물 수집 없이,
같은 (bbox,vis) 프레임에 상충하는 두 합성 지시를 강제로 붙이고 각각 다른 정답
(FWD+L/FWD+R)으로 라벨링한 합성 쌍을 실제 데이터에 섞어 학습 -> 이런 "의무적으로
텍스트를 읽어야만 풀리는" 신호가 존재할 때 counterfactual 반응이 살아나는지 확인.

주의: 실제 반사실적 장면(진짜 다른 촬영)이 아니라 같은 이미지에 라벨만 강제로 붙인
근사치라, "모델이 이런 신호가 있으면 학습할 능력 자체는 있다"를 확인하는 용도이지
실전 데이터 품질을 대체하지 않는다.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_FILE_NORM = ROOT / "docs" / "v5" / "closed_loop_eval" / "exp71_vis_cache_normfixed.pt"
OUT_FILE = ROOT / "docs" / "v5" / "closed_loop_eval" / "exp71_synthetic_obedience.json"

WINDOW = 6
BBOX_DIM = 4
VIS_DIM = 256
TEXT_DIM = 512
NUM_CLASSES = 8
LEFT_CLASSES = {2, 4, 6}
RIGHT_CLASSES = {3, 5, 7}
FORCE_LEFT_CLASS = 4   # FWD+L
FORCE_RIGHT_CLASS = 5  # FWD+R

SYNTH_LEFT_SENT = "curve left decisively, go toward the left side no matter what"
SYNTH_RIGHT_SENT = "curve right decisively, go toward the right side no matter what"


def embed_texts(sentences):
    from transformers import Owlv2Processor, Owlv2Model
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2Model.from_pretrained("google/owlv2-base-patch16-ensemble").to(DEVICE).eval()
    embs = {}
    with torch.no_grad():
        for s in sentences:
            inputs = proc(text=[s], return_tensors="pt").to(DEVICE)
            feat = model.get_text_features(**inputs)
            embs[s] = F.normalize(feat, dim=-1).squeeze(0).cpu()
    del model
    torch.cuda.empty_cache()
    return embs


class TransformerActionHeadText(nn.Module):
    def __init__(self, frame_dim, window=WINDOW, nhead=4, num_layers=2):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, frame_dim))
        self.pos_emb = nn.Embedding(window + 1, frame_dim)
        el = nn.TransformerEncoderLayer(d_model=frame_dim, nhead=nhead, dim_feedforward=512,
                                         dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(el, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(frame_dim), nn.Linear(frame_dim, 128), nn.ReLU(),
                                   nn.Dropout(0.1), nn.Linear(128, NUM_CLASSES))

    def forward(self, x):
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        pos = torch.arange(x.size(1), device=x.device)
        return self.head(self.encoder(x + self.pos_emb(pos))[:, 0])


def build_frame_feat(bboxes, vis, t, window, text_vec):
    seq = []
    for k in range(window):
        idx = max(0, t - (window - 1 - k))
        feat = list(bboxes[idx]) + vis[idx].tolist() + text_vec.tolist()
        seq.append(feat)
    return seq


def build_dataset(episodes, real_embs, synth_embs, include_synth, window=WINDOW):
    X, y, meta = [], [], []
    for ep in episodes:
        bboxes, vis, gts, pt = ep["bboxes"], ep["vis"], ep["gts"], ep["path_type"]
        n = len(gts)
        for t in range(n):
            # 1) 실제 샘플 (real instruction, real label)
            X.append(build_frame_feat(bboxes, vis, t, window, real_embs[pt]))
            y.append(gts[t])
            meta.append({"kind": "real", "path_type": pt})
            if include_synth:
                # 2) 합성 강제-좌 샘플: 같은 프레임, "왼쪽으로" 지시 -> FWD+L 강제
                X.append(build_frame_feat(bboxes, vis, t, window, synth_embs[SYNTH_LEFT_SENT]))
                y.append(FORCE_LEFT_CLASS)
                meta.append({"kind": "synth_left", "path_type": pt})
                # 3) 합성 강제-우 샘플: 같은 프레임, "오른쪽으로" 지시 -> FWD+R 강제
                X.append(build_frame_feat(bboxes, vis, t, window, synth_embs[SYNTH_RIGHT_SENT]))
                y.append(FORCE_RIGHT_CLASS)
                meta.append({"kind": "synth_right", "path_type": pt})
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), meta


def train_one(X_tr, y_tr, X_te, y_te, frame_dim, epochs=200, lr=5e-4, seed=42):
    torch.manual_seed(seed)
    X_tr_t = torch.tensor(X_tr, device=DEVICE)
    y_tr_t = torch.tensor(y_tr, device=DEVICE)
    X_te_t = torch.tensor(X_te, device=DEVICE)
    y_te_t = torch.tensor(y_te, device=DEVICE)
    model = TransformerActionHeadText(frame_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_acc, best_state = 0.0, None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_tr_t), device=DEVICE)
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            logits = model(X_tr_t[idx])
            loss = F.cross_entropy(logits, y_tr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 40 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                acc = (model(X_te_t).argmax(1) == y_te_t).float().mean().item()
            if acc >= best_acc:
                best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, best_acc


def counterfactual_eval(model, val_eps, real_embs, synth_embs, window=WINDOW):
    """real 테스트 프레임에 좌/우 강제 지시(합성 문장) 주고 예측 분포 확인."""
    X_real, _, meta = build_dataset(val_eps, real_embs, synth_embs, include_synth=False, window=window)
    left_vec = synth_embs[SYNTH_LEFT_SENT].numpy()
    right_vec = synth_embs[SYNTH_RIGHT_SENT].numpy()
    X_left = X_real.copy(); X_left[:, :, -TEXT_DIM:] = left_vec[None, None, :]
    X_right = X_real.copy(); X_right[:, :, -TEXT_DIM:] = right_vec[None, None, :]
    with torch.no_grad():
        pred_left = model(torch.tensor(X_left, device=DEVICE)).argmax(1).cpu().numpy()
        pred_right = model(torch.tensor(X_right, device=DEVICE)).argmax(1).cpu().numpy()
    return {
        "cf_left_forces_FWDL": float(np.mean(pred_left == FORCE_LEFT_CLASS)),
        "cf_left_to_LEFT_classes": float(np.mean([p in LEFT_CLASSES for p in pred_left])),
        "cf_right_forces_FWDR": float(np.mean(pred_right == FORCE_RIGHT_CLASS)),
        "cf_right_to_RIGHT_classes": float(np.mean([p in RIGHT_CLASSES for p in pred_right])),
        "cf_changed_rate": float(np.mean(pred_left != pred_right)),
    }


def main():
    episodes = torch.load(CACHE_FILE_NORM, weights_only=False)
    print(f"episodes={len(episodes)}")

    INSTRUCTIONS = {
        "center_straight": "go straight ahead to the basket in front of you",
        "center_left": "the basket is slightly left of center, curve left a bit",
        "center_right": "the basket is slightly right of center, curve right a bit",
        "left_straight": "the basket is on the left side, go forward along the left",
        "left_left": "the basket is on your left, curve left to reach it",
        "left_right": "starting from the left, curve right toward the basket",
        "right_straight": "the basket is on the right side, go forward along the right",
        "right_left": "starting from the right, curve left toward the basket",
        "right_right": "the basket is on your right, curve right to reach it",
    }
    all_sentences = list(INSTRUCTIONS.values()) + [SYNTH_LEFT_SENT, SYNTH_RIGHT_SENT]
    embs_all = embed_texts(all_sentences)
    real_embs = {pt: embs_all[s] for pt, s in INSTRUCTIONS.items()}
    synth_embs = {SYNTH_LEFT_SENT: embs_all[SYNTH_LEFT_SENT], SYNTH_RIGHT_SENT: embs_all[SYNTH_RIGHT_SENT]}

    rng = np.random.default_rng(42)
    idx = list(range(len(episodes)))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * 0.15))
    val_eps = [episodes[i] for i in idx[:n_val]]
    train_eps = [episodes[i] for i in idx[n_val:]]
    print(f"train={len(train_eps)} val={len(val_eps)}")

    frame_dim = BBOX_DIM + VIS_DIM + TEXT_DIM
    results = {}

    print("\n=== A) 대조군: 합성 없이(real만, 기존과 동일 조건 재확인) ===")
    X_tr, y_tr, _ = build_dataset(train_eps, real_embs, synth_embs, include_synth=False)
    X_te, y_te, _ = build_dataset(val_eps, real_embs, synth_embs, include_synth=False)
    model_a, acc_a = train_one(X_tr, y_tr, X_te, y_te, frame_dim)
    cf_a = counterfactual_eval(model_a, val_eps, real_embs, synth_embs)
    print(f"acc={acc_a:.1%}  counterfactual={cf_a}")
    results["A_no_synth"] = {"acc": acc_a, **cf_a}

    print("\n=== B) 합성 이질지시 포함 (같은 프레임에 상충 지시+다른 정답 강제) ===")
    X_tr2, y_tr2, meta_tr2 = build_dataset(train_eps, real_embs, synth_embs, include_synth=True)
    X_te2, y_te2, meta_te2 = build_dataset(val_eps, real_embs, synth_embs, include_synth=True)
    # 평가는 real 샘플만 대상으로 정확도 산출(합성 라벨은 학습신호일 뿐 "정답"이 아님)
    real_mask_te = np.array([m["kind"] == "real" for m in meta_te2])
    model_b, _ = train_one(X_tr2, y_tr2, X_te2, y_te2, frame_dim)
    with torch.no_grad():
        preds_te2 = model_b(torch.tensor(X_te2, device=DEVICE)).argmax(1).cpu().numpy()
    acc_b_real = float(np.mean(preds_te2[real_mask_te] == y_te2[real_mask_te]))
    cf_b = counterfactual_eval(model_b, val_eps, real_embs, synth_embs)
    print(f"real-only acc={acc_b_real:.1%}  counterfactual={cf_b}")
    results["B_with_synth"] = {"acc_real_only": acc_b_real, **cf_b}

    OUT_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nsaved -> {OUT_FILE}")

    print("\n=== 결론 ===")
    print(f"A(합성 없음) counterfactual 변화율: {cf_a['cf_changed_rate']:.1%}")
    print(f"B(합성 있음) counterfactual 변화율: {cf_b['cf_changed_rate']:.1%}")
    print("B가 크게 높다면: '헤드는 이런 신호가 있으면 배울 능력이 있다' 확인 (데이터 문제였다는 가설 지지)")


if __name__ == "__main__":
    main()

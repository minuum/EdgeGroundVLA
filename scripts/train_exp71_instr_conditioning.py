#!/usr/bin/env python3
"""
VLA 사다리 ② (언어->정책 조건화) 재검증 — 실제 exp71 레시피(150ep pg448, Kosmos-2
vision feature) 기준, 정규화 버그(2026-07-07 발견: 운영 서버는 vis_feat를 L2
normalize하는데 학습스크립트는 안 함)를 고친 상태로 다시 확인.

지난 43ep OWL 프록시 실험(instr_head_results.json) 결과를 실제 레시피로 재현:
  - no_text / with_text / shuffled_text 3비교군 PM
  - permutation test (다른 path_type 임베딩으로 교체 -> 정확도 하락폭)
  - counterfactual test (left_left vs right_right 임베딩 강제 -> 예측 방향 변화율)

instruction 임베딩: OWL-v2 내장 CLIP 텍스트 타워 재사용(지난 실험과 동일 소스, 512d).
"""
import importlib.util
import json
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("exp71", ROOT / "scripts" / "train_exp71_stage2_transformer.py")
exp71 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp71)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE_FILE_NORM = ROOT / "docs" / "v5" / "closed_loop_eval" / "exp71_vis_cache_normfixed.pt"
OUT_FILE = ROOT / "docs" / "v5" / "closed_loop_eval" / "exp71_instr_conditioning_real_recipe.json"

WINDOW = 6
BBOX_DIM = 4
VIS_DIM = 256
TEXT_DIM = 512
NUM_CLASSES = 8
LEFT_CLASSES = {2, 4, 6}
RIGHT_CLASSES = {3, 5, 7}

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


def build_norm_cache():
    """L2 정규화 버그 고쳐서 vis feature 캐시 재생성 (150ep, episode명 포함)."""
    with open(exp71.ANN_PATH) as f:
        ann = json.load(f)
    print("FrozenCLIPV2 로드...")
    enc = exp71.FrozenCLIPV2(exp71.VLM_PATH, exp71.STAGE1_PT, DEVICE).eval()

    episodes = []
    for ep in ann:
        h5_path = Path(ep["episode"])
        if not h5_path.exists():
            continue
        frames = [fr for fr in ep["frames"] if fr.get("gt_class") is not None]
        if not frames:
            continue
        with h5py.File(str(h5_path), "r") as f:
            imgs_np = f["observations"]["images"][:]
        pil_imgs = [Image.fromarray(imgs_np[fr["frame_idx"]].astype("uint8")) for fr in frames]
        vis = F.normalize(enc.encode_batch(pil_imgs, DEVICE), dim=-1).cpu()  # 정규화 수정
        bboxes = [(fr.get("cx_det", 0.5), fr.get("cy_det", 0.5),
                   fr.get("area_det", 0.05), float(fr.get("has_bbox", False))) for fr in frames]
        gts = [fr["gt_class"] for fr in frames]
        episodes.append({"stem": h5_path.stem, "path_type": ep["path_type"],
                          "bboxes": bboxes, "vis": vis, "gts": gts})
    torch.save(episodes, CACHE_FILE_NORM)
    del enc
    torch.cuda.empty_cache()
    print(f"저장 -> {CACHE_FILE_NORM} ({len(episodes)}ep)")
    return episodes


def embed_instructions():
    from transformers import Owlv2Processor, Owlv2Model
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2Model.from_pretrained("google/owlv2-base-patch16-ensemble").to(DEVICE).eval()
    embs = {}
    with torch.no_grad():
        for pt, sent in INSTRUCTIONS.items():
            inputs = proc(text=[sent], return_tensors="pt").to(DEVICE)
            feat = model.get_text_features(**inputs)
            embs[pt] = F.normalize(feat, dim=-1).squeeze(0).cpu()
    del model
    torch.cuda.empty_cache()
    return embs


class TransformerActionHeadText(nn.Module):
    """exp71 구조 + 옵션 text embedding concat."""
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
        x = x + self.pos_emb(pos)
        return self.head(self.encoder(x)[:, 0])


def build_windows(episodes, embs, text_mode, window=WINDOW, seed=0):
    """text_mode: none | real | shuffled"""
    rng = random.Random(seed)
    path_types = list(embs.keys())
    X, y, meta = [], [], []
    for ep in episodes:
        pt = ep["path_type"]
        if text_mode == "none":
            text_vec = None
        elif text_mode == "real":
            text_vec = embs[pt]
        else:  # shuffled
            other = rng.choice([p for p in path_types if p != pt] or path_types)
            text_vec = embs[other]

        bboxes, vis, gts = ep["bboxes"], ep["vis"], ep["gts"]
        n = len(gts)
        for t in range(n):
            seq = []
            for k in range(window):
                idx = max(0, t - (window - 1 - k))
                frame_feat = list(bboxes[idx]) + vis[idx].tolist()
                if text_vec is not None:
                    frame_feat = frame_feat + text_vec.tolist()
                seq.append(frame_feat)
            X.append(seq)
            y.append(gts[t])
            meta.append({"path_type": pt, "stem": ep["stem"]})
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), meta


def train_one(X_tr, y_tr, X_te, y_te, frame_dim, epochs=300, lr=5e-4, seed=42):
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
        if ep % 50 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                acc = (model(X_te_t).argmax(1) == y_te_t).float().mean().item()
            if acc >= best_acc:
                best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, best_acc


def causal_eval(model, test_eps, embs, window=WINDOW):
    path_types = list(embs.keys())

    # permutation: 각 테스트 프레임의 text를 "다른" path_type 임베딩으로 교체
    X_real, y_real, meta_real = build_windows(test_eps, embs, "real", window)
    rng = random.Random(1)
    X_perm = X_real.copy()
    frame_dim_total = X_perm.shape[-1]
    for i, m in enumerate(meta_real):
        other = rng.choice([p for p in path_types if p != m["path_type"]])
        other_vec = embs[other].numpy()
        X_perm[i, :, -TEXT_DIM:] = other_vec[None, :]

    X_real_t = torch.tensor(X_real, device=DEVICE)
    X_perm_t = torch.tensor(X_perm, device=DEVICE)
    y_real_t = torch.tensor(y_real, device=DEVICE)
    with torch.no_grad():
        acc_real = (model(X_real_t).argmax(1) == y_real_t).float().mean().item()
        acc_perm = (model(X_perm_t).argmax(1) == y_real_t).float().mean().item()

    # counterfactual: 동일 프레임에 left_left vs right_right 임베딩 강제
    left_vec = embs["left_left"].numpy()
    right_vec = embs["right_right"].numpy()
    X_left = X_real.copy(); X_left[:, :, -TEXT_DIM:] = left_vec[None, None, :]
    X_right = X_real.copy(); X_right[:, :, -TEXT_DIM:] = right_vec[None, None, :]
    with torch.no_grad():
        pred_left = model(torch.tensor(X_left, device=DEVICE)).argmax(1).cpu().numpy()
        pred_right = model(torch.tensor(X_right, device=DEVICE)).argmax(1).cpu().numpy()
    left_to_left = np.mean([p in LEFT_CLASSES for p in pred_left])
    right_to_right = np.mean([p in RIGHT_CLASSES for p in pred_right])
    changed = np.mean(pred_left != pred_right)

    return {
        "acc_real": acc_real, "acc_perm": acc_perm, "perm_drop": acc_real - acc_perm,
        "cf_left_to_left": float(left_to_left), "cf_right_to_right": float(right_to_right),
        "cf_changed_rate": float(changed),
    }


def main():
    if CACHE_FILE_NORM.exists():
        print(f"정규화-수정 캐시 로드: {CACHE_FILE_NORM}")
        episodes = torch.load(CACHE_FILE_NORM, weights_only=False)
    else:
        episodes = build_norm_cache()
    print(f"episodes={len(episodes)}")

    embs = embed_instructions()
    print(f"instruction embeddings: {len(embs)}개 path_type")

    rng = np.random.default_rng(42)
    idx = list(range(len(episodes)))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * 0.15))
    val_eps = [episodes[i] for i in idx[:n_val]]
    train_eps = [episodes[i] for i in idx[n_val:]]
    print(f"train={len(train_eps)} val={len(val_eps)}")

    results = {}
    frame_dim_notext = BBOX_DIM + VIS_DIM
    frame_dim_text = BBOX_DIM + VIS_DIM + TEXT_DIM

    for mode, fdim in [("none", frame_dim_notext), ("real", frame_dim_text), ("shuffled", frame_dim_text)]:
        accs = []
        model_last = None
        for seed in range(3):
            X_tr, y_tr, _ = build_windows(train_eps, embs, mode, seed=seed)
            X_te, y_te, _ = build_windows(val_eps, embs, mode, seed=seed)
            model, acc = train_one(X_tr, y_tr, X_te, y_te, fdim, seed=42 + seed)
            accs.append(acc)
            model_last = model
        results[f"{mode}_pm_mean"] = float(np.mean(accs))
        results[f"{mode}_pm_std"] = float(np.std(accs))
        print(f"[{mode:9s}] PM = {np.mean(accs):.1%} ± {np.std(accs):.1%}  (3 seeds)")
        if mode == "real":
            causal = causal_eval(model_last, val_eps, embs)
            results["causal"] = causal
            print(f"  permutation: {causal['acc_real']:.3f} -> {causal['acc_perm']:.3f}  "
                  f"(Δ{causal['perm_drop']*100:+.1f}%p)")
            print(f"  counterfactual: left->LEFT류={causal['cf_left_to_left']:.1%}  "
                  f"right->RIGHT류={causal['cf_right_to_right']:.1%}  변화율={causal['cf_changed_rate']:.1%}")

    OUT_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nsaved -> {OUT_FILE}")


if __name__ == "__main__":
    main()

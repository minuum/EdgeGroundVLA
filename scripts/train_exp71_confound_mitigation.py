#!/usr/bin/env python3
"""
2026-07-10: 재수집 전, 현재 150ep로 짜낼 수 있는 마지막 학습레벨 시도.

배경(CH61 61-13): obj_right에서 FWD+L이 반복되는 이유는 `right_left`
path_type(목표=우측, 접근경로=좌곡선)이 cx>0.6 구간에서 gt_class=FWD+L을
대량으로 제공하기 때문 — "cx 높음(오른쪽)인데 정답은 좌회전"이라는
모순적 신호가 특정 path_type 하나에 쏠려 있음.

재수집 없이 시도 가능한 두 가지 완화책을 window6+bbox_scale3(현재 배포
베스트, 61-9) 기준으로 비교:

A) baseline: 현재 배포 설정 그대로 (재현용 대조군)
B) confound reweight: right_left path_type의 cx>0.6 & gt_class==FWD+L
   프레임 손실 가중치를 낮춰(0.3x) "cx높음→FWD+L" 과확신을 완화
C) hybrid cx-rule overlay: 학습은 A와 동일하되, 평가 시 cx>0.75(강한우)/
   cx<0.25(강한좌) 구간만 MLP 예측을 CX_RULE_THRESHOLDS 기하 규칙
   (ROT_R/ROT_L)으로 덮어써서 - 그 구간에서 실제 obj_*와 유사한 정확도가
   나오는지 확인 (서버에 이미 VLA_CX_RULE 지원 있음, 배포 시 켜기만 하면 됨)

측정: 전체 val_acc, FWD+L/FWD+R recall, cx>0.75 subset 정확도(=obj_right
proxy), cx<0.25 subset 정확도(=obj_left proxy). 5-seed.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE = Path("docs/v5/closed_loop_eval/exp71_vis_cache_normfixed.pt")
OUT_FILE = Path("docs/v5/closed_loop_eval/confound_mitigation_20260710.json")

WINDOW = 6
BBOX_SCALE = 3.0
NUM_CLASSES = 8
FRAME_DIM = 4 + 256
SEEDS = [0, 1, 2, 3, 4]
VAL_RATIO = 0.15
CX_ROT_L, CX_ROT_R = 0.25, 0.75  # CX_RULE_THRESHOLDS와 동일


class Head(nn.Module):
    def __init__(self, frame_dim=FRAME_DIM, window=WINDOW, nhead=4, num_layers=2):
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


def build_windows(eps, window=WINDOW, bbox_scale=BBOX_SCALE, confound_reweight=False):
    X, y, w, cx_last = [], [], [], []
    for ep in eps:
        bboxes, vis, gts = ep["bboxes"], ep["vis"], ep["gts"]
        n = len(gts)
        for t in range(n):
            seq = []
            for k in range(window):
                idx = max(0, t - (window - 1 - k))
                scaled_bbox = [v * bbox_scale for v in bboxes[idx]]
                seq.append(scaled_bbox + vis[idx].tolist())
            X.append(seq)
            y.append(gts[t])
            cx_last.append(bboxes[t][0])
            sw = 1.0
            if confound_reweight and ep["path_type"] == "right_left" and bboxes[t][0] > 0.6 and gts[t] == 4:
                sw = 0.3
            w.append(sw)
    return (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64),
            np.asarray(w, dtype=np.float32), np.asarray(cx_last, dtype=np.float32))


def cx_rule_pred(cx):
    if cx < CX_ROT_L:
        return 6
    if cx > CX_ROT_R:
        return 7
    return None  # MLP에 위임


def train_one(X_tr, y_tr, w_tr, X_va, y_va, seed, epochs=300, lr=5e-4):
    torch.manual_seed(seed)
    cls_counts = np.bincount(y_tr, minlength=NUM_CLASSES).astype(np.float32)
    cls_counts = np.where(cls_counts == 0, 1.0, cls_counts)
    cls_w = 1.0 / cls_counts
    cls_w = cls_w / cls_w.sum() * NUM_CLASSES
    cls_w_t = torch.tensor(cls_w, dtype=torch.float32, device=DEVICE)

    X_tr_t = torch.tensor(X_tr, device=DEVICE)
    y_tr_t = torch.tensor(y_tr, device=DEVICE)
    w_tr_t = torch.tensor(w_tr, device=DEVICE)
    X_va_t = torch.tensor(X_va, device=DEVICE)
    y_va_t = torch.tensor(y_va, device=DEVICE)

    model = Head().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_acc, best_state = 0.0, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr_t), device=DEVICE)
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            logits = model(X_tr_t[b])
            per_sample = F.cross_entropy(logits, y_tr_t[b], weight=cls_w_t, reduction="none")
            loss = (per_sample * w_tr_t[b]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 50 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                acc = (model(X_va_t).argmax(1) == y_va_t).float().mean().item()
            if acc >= best_acc:
                best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return best_acc, model


def eval_subsets(model, X_va, y_va, cx_va, hybrid_rule=False):
    with torch.no_grad():
        logits = model(torch.tensor(X_va, device=DEVICE))
        preds = logits.argmax(1).cpu().numpy()
    if hybrid_rule:
        for i, cx in enumerate(cx_va):
            r = cx_rule_pred(cx)
            if r is not None:
                preds[i] = r
    acc = (preds == y_va).mean()
    fwdl_mask = y_va == 4
    fwdr_mask = y_va == 5
    fwdl_recall = (preds[fwdl_mask] == 4).mean() if fwdl_mask.sum() else None
    fwdr_recall = (preds[fwdr_mask] == 5).mean() if fwdr_mask.sum() else None
    right_mask = cx_va > 0.75
    left_mask = cx_va < 0.25
    right_acc = (preds[right_mask] == y_va[right_mask]).mean() if right_mask.sum() else None
    left_acc = (preds[left_mask] == y_va[left_mask]).mean() if left_mask.sum() else None
    return {
        "val_acc": float(acc),
        "fwdl_recall": float(fwdl_recall) if fwdl_recall is not None else None,
        "fwdr_recall": float(fwdr_recall) if fwdr_recall is not None else None,
        "cx_gt_0.75_acc": float(right_acc) if right_acc is not None else None,
        "cx_lt_0.25_acc": float(left_acc) if left_acc is not None else None,
        "cx_gt_0.75_n": int(right_mask.sum()),
        "cx_lt_0.25_n": int(left_mask.sum()),
    }


def main():
    episodes = torch.load(CACHE, weights_only=False)
    rng = np.random.default_rng(42)
    idx = list(range(len(episodes)))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * VAL_RATIO))
    val_eps = [episodes[i] for i in idx[:n_val]]
    train_eps = [episodes[i] for i in idx[n_val:]]
    print(f"train={len(train_eps)} val={len(val_eps)}")

    X_va, y_va, _, cx_va = build_windows(val_eps)

    configs = [
        ("A_baseline(deployed)", False, False),
        ("B_confound_reweight", True, False),
        ("C_hybrid_cxrule_eval", False, True),  # A와 동일 학습, 평가만 hybrid
    ]

    results = {}
    for name, reweight, hybrid in configs:
        X_tr, y_tr, w_tr, _ = build_windows(train_eps, confound_reweight=reweight)
        seed_metrics = []
        for seed in SEEDS:
            acc, model = train_one(X_tr, y_tr, w_tr, X_va, y_va, seed=seed)
            m = eval_subsets(model, X_va, y_va, cx_va, hybrid_rule=hybrid)
            seed_metrics.append(m)
            print(f"  [{name}] seed={seed} val_acc={m['val_acc']:.1%} "
                  f"FWD+L_recall={m['fwdl_recall']} FWD+R_recall={m['fwdr_recall']} "
                  f"cx>0.75_acc={m['cx_gt_0.75_acc']}(n={m['cx_gt_0.75_n']}) "
                  f"cx<0.25_acc={m['cx_lt_0.25_acc']}(n={m['cx_lt_0.25_n']})")
        agg = {}
        for key in ["val_acc", "fwdl_recall", "fwdr_recall", "cx_gt_0.75_acc", "cx_lt_0.25_acc"]:
            vals = [m[key] for m in seed_metrics if m[key] is not None]
            agg[key + "_mean"] = float(np.mean(vals)) if vals else None
            agg[key + "_std"] = float(np.std(vals)) if vals else None
        results[name] = agg
        print(f"[{name}] val_acc={agg['val_acc_mean']:.1%}±{agg['val_acc_std']:.1%}  "
              f"cx>0.75_acc={agg['cx_gt_0.75_acc_mean']}±{agg['cx_gt_0.75_acc_std']}\n")

    OUT_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"saved -> {OUT_FILE}")


if __name__ == "__main__":
    main()

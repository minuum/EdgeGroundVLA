#!/usr/bin/env python3
"""
2026-07-07 후속: diag_mult(대각클래스 가중치)와 bbox_scale(cx 등 bbox 4dim을 스케일업해서
모델이 더 강하게 참고하도록)을 조합, 단일시드 변동성 문제(89.8%->72.4% 흔들림 발견)를
피하기 위해 5-seed로 재검증.

bbox_scale: frame feature의 bbox(4dim) 부분에 배수를 곱해서 concat -- vis_feat(256dim,
L2정규화라 norm=1)에 비해 상대적으로 bbox 크기를 키워 attention/projection이 이를
더 크게 반영하도록 유도하는 가장 단순한 방법.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda")
CACHE = Path("docs/v5/closed_loop_eval/exp71_vis_cache_normfixed.pt")
OUT_FILE = Path("docs/v5/closed_loop_eval/ablate_diagweight_bboxscale_multiseed.json")
CLASS_NAMES = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "ROT_L", "ROT_R"]
WINDOW = 6
BBOX_DIM = 4
FRAME_DIM = BBOX_DIM + 256
DIAG_CLASSES = [4, 5]
SEEDS = [0, 1, 2, 3, 4]


class Head(nn.Module):
    def __init__(self, frame_dim=FRAME_DIM, window=WINDOW, nhead=4, num_layers=2):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, frame_dim))
        self.pos_emb = nn.Embedding(window + 1, frame_dim)
        el = nn.TransformerEncoderLayer(d_model=frame_dim, nhead=nhead, dim_feedforward=512,
                                         dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(el, num_layers=num_layers)
        self.head = nn.Sequential(nn.LayerNorm(frame_dim), nn.Linear(frame_dim, 128), nn.ReLU(),
                                   nn.Dropout(0.1), nn.Linear(128, 8))

    def forward(self, x):
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        pos = torch.arange(x.size(1), device=x.device)
        return self.head(self.encoder(x + self.pos_emb(pos))[:, 0])


def build_windows(eps, bbox_scale, window=WINDOW):
    X, y = [], []
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
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def train_and_eval(X_tr, y_tr, X_va, y_va, diag_mult, epochs=300, lr=5e-4, seed=42):
    torch.manual_seed(seed)
    cls_counts = np.bincount(y_tr, minlength=8).astype(np.float32)
    cls_counts = np.where(cls_counts == 0, 1.0, cls_counts)
    weights = 1.0 / cls_counts
    for c in DIAG_CLASSES:
        weights[c] *= diag_mult
    weights = weights / weights.sum() * 8
    weights_t = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    X_tr_t = torch.tensor(X_tr, device=DEVICE)
    y_tr_t = torch.tensor(y_tr, device=DEVICE)
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
            loss = F.cross_entropy(logits, y_tr_t[b], weight=weights_t)
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
    with torch.no_grad():
        preds = model(X_va_t).argmax(1).cpu().numpy()

    result = {"val_acc": float(best_acc)}
    for c in [1, 4, 5]:
        mask = y_va == c
        if mask.sum() == 0:
            continue
        result[f"{CLASS_NAMES[c]}_recall"] = float((preds[mask] == c).mean())
        result[f"{CLASS_NAMES[c]}_to_FORWARD"] = float((preds[mask] == 1).mean())
    return result


def main():
    episodes = torch.load(CACHE, weights_only=False)
    rng = np.random.default_rng(42)
    idx = list(range(len(episodes)))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * 0.15))
    val_eps = [episodes[i] for i in idx[:n_val]]
    train_eps = [episodes[i] for i in idx[n_val:]]

    configs = [
        ("diag1.0_bbox1x", 1.0, 1.0),
        ("diag2.0_bbox1x", 2.0, 1.0),
        ("diag1.0_bbox3x", 1.0, 3.0),
        ("diag2.0_bbox3x", 2.0, 3.0),
    ]

    results = {}
    for name, diag_mult, bbox_scale in configs:
        X_tr, y_tr = build_windows(train_eps, bbox_scale)
        X_va, y_va = build_windows(val_eps, bbox_scale)
        seed_results = []
        for seed in SEEDS:
            r = train_and_eval(X_tr, y_tr, X_va, y_va, diag_mult=diag_mult, seed=seed)
            seed_results.append(r)
        agg = {}
        for key in seed_results[0]:
            vals = [r[key] for r in seed_results]
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
        results[name] = agg
        print(f"[{name:16s}] val_acc={agg['val_acc_mean']:.1%}±{agg['val_acc_std']:.1%}  "
              f"FWD+L={agg.get('FWD+L_recall_mean',0):.1%}±{agg.get('FWD+L_recall_std',0):.1%}  "
              f"FWD+R={agg.get('FWD+R_recall_mean',0):.1%}±{agg.get('FWD+R_recall_std',0):.1%}")

    OUT_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nsaved -> {OUT_FILE}")


if __name__ == "__main__":
    main()

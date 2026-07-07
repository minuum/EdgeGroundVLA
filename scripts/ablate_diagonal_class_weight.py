#!/usr/bin/env python3
"""
2026-07-07: FWD+R(대각-우) recall이 FWD+L(대각-좌)보다 뚜렷이 낮고(67.6% vs 88.5%),
FORWARD로 오분류되는 비율도 높음(21.6% vs 5.8%). class-weight에서 대각방향(FWD+L/FWD+R,
클래스 4/5) 가중치를 배수로 스윕해서 recall/정확도 트레이드오프를 확인.

기본 class weight는 1/count 역빈도 정규화. 여기에 FWD+L/FWD+R에만 추가 배수를 곱해서
0.5(낮춤)~3.0(높임) 범위로 스윕.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda")
CACHE = Path("docs/v5/closed_loop_eval/exp71_vis_cache_normfixed.pt")
OUT_FILE = Path("docs/v5/closed_loop_eval/ablate_diagonal_class_weight.json")
CLASS_NAMES = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "ROT_L", "ROT_R"]
WINDOW = 6
FRAME_DIM = 4 + 256
DIAG_CLASSES = [4, 5]


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


def build_windows(eps, window=WINDOW):
    X, y = [], []
    for ep in eps:
        bboxes, vis, gts = ep["bboxes"], ep["vis"], ep["gts"]
        n = len(gts)
        for t in range(n):
            seq = []
            for k in range(window):
                idx = max(0, t - (window - 1 - k))
                seq.append(list(bboxes[idx]) + vis[idx].tolist())
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
    for c in [1, 4, 5]:  # FORWARD, FWD+L, FWD+R
        mask = y_va == c
        if mask.sum() == 0:
            continue
        recall = (preds[mask] == c).mean()
        to_fwd = (preds[mask] == 1).mean()
        result[f"{CLASS_NAMES[c]}_recall"] = float(recall)
        result[f"{CLASS_NAMES[c]}_to_FORWARD"] = float(to_fwd)
    return result


def main():
    episodes = torch.load(CACHE, weights_only=False)
    rng = np.random.default_rng(42)
    idx = list(range(len(episodes)))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * 0.15))
    val_eps = [episodes[i] for i in idx[:n_val]]
    train_eps = [episodes[i] for i in idx[n_val:]]

    X_tr, y_tr = build_windows(train_eps)
    X_va, y_va = build_windows(val_eps)
    print(f"train={len(X_tr)} val={len(X_va)}")

    results = {}
    for mult in [0.5, 1.0, 1.5, 2.0, 3.0]:
        r = train_and_eval(X_tr, y_tr, X_va, y_va, diag_mult=mult)
        results[f"diag_mult_{mult}"] = r
        print(f"[diag_mult={mult:.1f}] val_acc={r['val_acc']:.1%}  "
              f"FWD+L_recall={r.get('FWD+L_recall',0):.1%}  FWD+R_recall={r.get('FWD+R_recall',0):.1%}  "
              f"FWD+R->FORWARD={r.get('FWD+R_to_FORWARD',0):.1%}")

    OUT_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nsaved -> {OUT_FILE}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
2026-07-08 후속: bbox_scale=3.0 발견(72%->85% acc, 표준편차 27%p->4%p)을 두 방향으로 확장.

A) window6(현재 운영 window) + bbox_scale=3x — window3가 실로봇 A/B에서 안 좋게 나올
   경우를 대비한 대안 후보. window 자체는 안 바꾸고 bbox_scale만 적용.
B) window3 + bbox_scale=3x + sticky_aug(flicker 강건성) 조합 — 어제 sticky_aug 단독으로는
   별 효과 없었는데(train_exp71_flicker_robustness.py), 그건 bbox 신호가 약해서 sticky
   처리 자체의 이득도 묻혔을 가능성. bbox_scale로 신호를 키운 상태에서 재시도.

각 3-seed, val_acc + 상관형 flicker 진동율 측정.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE = Path("docs/v5/closed_loop_eval/exp71_vis_cache_normfixed.pt")
OUT_FILE = Path("docs/v5/closed_loop_eval/bboxscale_window_sticky_combo.json")
CLASS_NAMES = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "ROT_L", "ROT_R"]
NUM_CLASSES = 8
LEFT_CLASSES = {2, 4, 6}
RIGHT_CLASSES = {3, 5, 7}
FALLBACK = (0.5, 0.5, 0.05, 0.0)
SEEDS = [0, 1, 2]


class Head(nn.Module):
    def __init__(self, frame_dim, window, nhead=4, num_layers=2):
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


def apply_sticky(bboxes, p, rng):
    out, last_real = [], FALLBACK
    for b in bboxes:
        if b[3] > 0.5:
            last_real = b
        if rng.random() < p:
            out.append((last_real[0], last_real[1], last_real[2], 0.0))
        else:
            out.append(b)
    return out


def build_windows(eps, window, bbox_scale, sticky_p=0.0, rng=None):
    X, y = [], []
    for ep in eps:
        bboxes = ep["bboxes"]
        if sticky_p > 0:
            bboxes = apply_sticky(bboxes, sticky_p, rng)
        vis, gts = ep["vis"], ep["gts"]
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


def correlated_flicker(bboxes, rng, area_near=0.15, p_base=0.1, p_near=0.85):
    out, was_near = [], False
    for b in bboxes:
        p = p_near if was_near else p_base
        out.append(FALLBACK if rng.random() < p else b)
        was_near = (b[3] > 0.5 and b[2] > area_near)
    return out


def train_and_eval(train_eps, val_eps, window, bbox_scale, sticky_p, seed):
    frame_dim = 4 + 256
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X_tr, y_tr = build_windows(train_eps, window, bbox_scale, sticky_p, rng)
    X_va, y_va = build_windows(val_eps, window, bbox_scale, 0.0)

    cls_counts = np.bincount(y_tr, minlength=NUM_CLASSES).astype(np.float32)
    cls_counts = np.where(cls_counts == 0, 1.0, cls_counts)
    weights = 1.0 / cls_counts
    weights = weights / weights.sum() * NUM_CLASSES
    weights_t = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    X_tr_t = torch.tensor(X_tr, device=DEVICE)
    y_tr_t = torch.tensor(y_tr, device=DEVICE)
    X_va_t = torch.tensor(X_va, device=DEVICE)
    y_va_t = torch.tensor(y_va, device=DEVICE)

    model = Head(frame_dim, window).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 300)
    best_acc, best_state = 0.0, None
    for ep in range(300):
        model.train()
        perm = torch.randperm(len(X_tr_t), device=DEVICE)
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            logits = model(X_tr_t[b])
            loss = F.cross_entropy(logits, y_tr_t[b], weight=weights_t)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 50 == 0 or ep == 299:
            model.eval()
            with torch.no_grad():
                acc = (model(X_va_t).argmax(1) == y_va_t).float().mean().item()
            if acc >= best_acc:
                best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()

    # 상관형 flicker 진동율
    osc_rates = []
    for trial in range(3):
        rng2 = np.random.default_rng(700 + trial)
        for ep in val_eps:
            bboxes = correlated_flicker(ep["bboxes"], rng2)
            vis = ep["vis"]
            n = len(ep["gts"])
            preds = []
            for t in range(n):
                seq = []
                for k in range(window):
                    idx = max(0, t - (window - 1 - k))
                    scaled_bbox = [v * bbox_scale for v in bboxes[idx]]
                    seq.append(scaled_bbox + vis[idx].tolist())
                x = torch.tensor([seq], dtype=torch.float32, device=DEVICE)
                with torch.no_grad():
                    preds.append(model(x).argmax(dim=-1).item())
            flips = pairs = 0
            for a, b in zip(preds[:-1], preds[1:]):
                sa = "L" if a in LEFT_CLASSES else ("R" if a in RIGHT_CLASSES else None)
                sb = "L" if b in LEFT_CLASSES else ("R" if b in RIGHT_CLASSES else None)
                if sa and sb:
                    pairs += 1
                    flips += (sa != sb)
            if pairs:
                osc_rates.append(flips / pairs)

    return {"val_acc": float(best_acc), "osc_mean": float(np.mean(osc_rates)) if osc_rates else None}


def main():
    episodes = torch.load(CACHE, weights_only=False)
    rng = np.random.default_rng(42)
    idx = list(range(len(episodes)))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * 0.15))
    val_eps = [episodes[i] for i in idx[:n_val]]
    train_eps = [episodes[i] for i in idx[n_val:]]
    print(f"train={len(train_eps)} val={len(val_eps)}")

    configs = [
        ("A_window6_bbox1x(baseline)", 6, 1.0, 0.0),
        ("A_window6_bbox3x", 6, 3.0, 0.0),
        ("B_window3_bbox3x(no_sticky)", 3, 3.0, 0.0),
        ("B_window3_bbox3x+sticky0.5", 3, 3.0, 0.5),
    ]

    results = {}
    for name, window, bbox_scale, sticky_p in configs:
        seed_results = [train_and_eval(train_eps, val_eps, window, bbox_scale, sticky_p, s) for s in SEEDS]
        accs = [r["val_acc"] for r in seed_results]
        oscs = [r["osc_mean"] for r in seed_results if r["osc_mean"] is not None]
        agg = {
            "val_acc_mean": float(np.mean(accs)), "val_acc_std": float(np.std(accs)),
            "osc_mean": float(np.mean(oscs)) if oscs else None,
            "osc_std": float(np.std(oscs)) if oscs else None,
        }
        results[name] = agg
        print(f"[{name:30s}] val_acc={agg['val_acc_mean']:.1%}±{agg['val_acc_std']:.1%}  "
              f"진동율={agg['osc_mean']:.1%}±{agg['osc_std']:.1%}")

    OUT_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nsaved -> {OUT_FILE}")


if __name__ == "__main__":
    main()

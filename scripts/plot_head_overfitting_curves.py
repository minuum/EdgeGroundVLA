#!/usr/bin/env python3
"""헤드별 epoch에 따른 train_acc vs val_acc 곡선 — 과적합 진단용 (2026-08-26).

지금까지 train_one() 등은 val_acc만 25epoch마다 기록하고 train_acc는 아예
측정한 적이 없어서 "과적합됐는지"를 직접 볼 방법이 없었다. 이 스크립트는
mlp/cxgeom/film/deltacx/cxaux 5개 헤드를 exp77 캐시(무작위 15% split, seed 0)
로 다시 학습하면서 25epoch마다 (train_acc, val_acc)를 둘 다 기록한다.

출력:
  docs/v5/closed_loop_eval/head_overfitting_curves.json
  docs/v5/figures/head_overfitting_curves.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_exp73_stage1v3_heads import (
    MLPActionHead, CxGeomHead, FiLMHead, DeltaCxHead, CxAuxHead,
    build_windows, NUM_CLASSES, VAL_RATIO, SPLIT_SEED, DEVICE,
)

CACHE = ROOT / "docs/v5/closed_loop_eval/exp77_florence2_phrase_full_vis_cache.pt"
OUT_JSON = ROOT / "docs/v5/closed_loop_eval/head_overfitting_curves.json"
OUT_PNG = ROOT / "docs/v5/figures/head_overfitting_curves.png"
SEED = 0
EPOCHS = 300
EVAL_EVERY = 10  # 곡선 해상도를 위해 25→10으로 촘촘히

HEADS = {"mlp": MLPActionHead, "cxgeom": CxGeomHead, "film": FiLMHead, "deltacx": DeltaCxHead}


def train_with_curve(head_cls, X_tr, y_tr, X_va, y_va, seed, epochs=EPOCHS, lr=5e-4, is_cxaux=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cls_counts = np.bincount(y_tr, minlength=NUM_CLASSES).astype(np.float32)
    cls_counts = np.where(cls_counts == 0, 1.0, cls_counts)
    weights = 1.0 / cls_counts
    weights = weights / weights.sum() * NUM_CLASSES
    weights_t = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    X_tr_t = torch.tensor(X_tr, device=DEVICE); y_tr_t = torch.tensor(y_tr, device=DEVICE)
    X_va_t = torch.tensor(X_va, device=DEVICE); y_va_t = torch.tensor(y_va, device=DEVICE)

    model = (CxAuxHead() if is_cxaux else head_cls()).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    curve = {"epoch": [], "train_acc": [], "val_acc": []}
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr_t), device=DEVICE)
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            xb, yb = X_tr_t[b], y_tr_t[b]
            if is_cxaux:
                logit, cx_pred = model(xb)
                loss = F.cross_entropy(logit, yb, weight=weights_t) + 0.3 * F.mse_loss(cx_pred, xb[:, -1, 0])
            else:
                loss = F.cross_entropy(model(xb), yb, weight=weights_t)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % EVAL_EVERY == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                if is_cxaux:
                    tr_logit, _ = model(X_tr_t); va_logit, _ = model(X_va_t)
                else:
                    tr_logit, va_logit = model(X_tr_t), model(X_va_t)
                tr_acc = (tr_logit.argmax(1) == y_tr_t).float().mean().item()
                va_acc = (va_logit.argmax(1) == y_va_t).float().mean().item()
            curve["epoch"].append(ep)
            curve["train_acc"].append(tr_acc)
            curve["val_acc"].append(va_acc)
    return curve


def main():
    eps = torch.load(str(CACHE), weights_only=False)
    rng = np.random.default_rng(SPLIT_SEED)
    idx = list(range(len(eps)))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * VAL_RATIO))
    va_eps = [eps[i] for i in idx[:n_val]]
    tr_eps = [eps[i] for i in idx[n_val:]]
    X_tr, y_tr, _ = build_windows(tr_eps)
    X_va, y_va, _ = build_windows(va_eps)
    print(f"[SPLIT] train {X_tr.shape} / val {X_va.shape}")

    all_curves = {}
    for name, head_cls in HEADS.items():
        print(f"\n=== {name} ===", flush=True)
        curve = train_with_curve(head_cls, X_tr, y_tr, X_va, y_va, SEED)
        all_curves[name] = curve
        final_gap = curve["train_acc"][-1] - curve["val_acc"][-1]
        max_val = max(curve["val_acc"])
        max_val_ep = curve["epoch"][curve["val_acc"].index(max_val)]
        print(f"  final(epoch{curve['epoch'][-1]}): train={curve['train_acc'][-1]*100:.2f}% "
              f"val={curve['val_acc'][-1]*100:.2f}% gap={final_gap*100:.2f}%p")
        print(f"  val 최고점: epoch{max_val_ep} val={max_val*100:.2f}% "
              f"(train_acc@그시점={curve['train_acc'][curve['epoch'].index(max_val_ep)]*100:.2f}%)")

    # cxaux는 회귀 보조손실이 섞여 있어 별도(다른 함수 시그니처)
    print(f"\n=== cxaux ===", flush=True)
    curve = train_with_curve(None, X_tr, y_tr, X_va, y_va, SEED, is_cxaux=True)
    all_curves["cxaux"] = curve
    final_gap = curve["train_acc"][-1] - curve["val_acc"][-1]
    print(f"  final: train={curve['train_acc'][-1]*100:.2f}% val={curve['val_acc'][-1]*100:.2f}% "
          f"gap={final_gap*100:.2f}%p")

    OUT_JSON.write_text(json.dumps(all_curves, indent=2, ensure_ascii=False))
    print(f"\n저장 → {OUT_JSON}")

    # 그래프
    names = list(all_curves.keys())
    ncols = 3
    nrows = (len(names) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = axes.flatten()
    for i, name in enumerate(names):
        c = all_curves[name]
        ax = axes[i]
        ax.plot(c["epoch"], [v * 100 for v in c["train_acc"]], label="train_acc", color="#3b82f6", linewidth=2)
        ax.plot(c["epoch"], [v * 100 for v in c["val_acc"]], label="val_acc", color="#ef4444", linewidth=2)
        ax.fill_between(c["epoch"], [v * 100 for v in c["train_acc"]], [v * 100 for v in c["val_acc"]],
                         alpha=0.15, color="#f59e0b")
        gap = (c["train_acc"][-1] - c["val_acc"][-1]) * 100
        ax.set_title(f"{name}  (최종 gap={gap:.1f}%p)", fontsize=12, fontweight="bold")
        ax.set_xlabel("epoch"); ax.set_ylabel("accuracy(%)")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        ax.set_ylim(0, 100)
    for j in range(len(names), len(axes)):
        axes[j].axis("off")
    fig.suptitle("헤드별 train_acc vs val_acc (epoch별, exp77 캐시·seed0·무작위 15% split)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=150, facecolor="white")
    print(f"저장 → {OUT_PNG}")


if __name__ == "__main__":
    main()

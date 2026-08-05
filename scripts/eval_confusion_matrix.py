#!/usr/bin/env python3
"""배포 헤드의 val 혼동행렬 — 교수님 요청(학습결과: val/test confusion matrix) 대응.

exp73_pg448_trackF_v6_mlp_holdaware 체크포인트(3 seed)를 hold-aware val 분할에
그대로 forward pass해 클래스별 정확도와 혼동행렬을 계산한다. 학습 스크립트와
동일한 seed42 분할·stride5·window6을 사용해 apples-to-apples를 유지한다.

출력: docs/v5/detector/confusion_matrix.json, docs/v5/figures/confusion_matrix.png
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
from scripts.train_exp73_trackA_heads import (
    MLPActionHead, CACHE_V6, SPLIT_SEED, VAL_RATIO, NUM_CLASSES, DEVICE,
)
from scripts.exp73_held_aware_train import build_windows_hold_aware
from scripts.exp73_window_cadence import swap_bboxes, OWL_ANN

NAMES = ["STOP", "FWD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "ROT_L", "ROT_R"]
HEADS = sorted((ROOT / "runs/v5_nav/mlp/exp73").glob(
    "exp73_pg448_trackF_v6_mlp_holdaware_seed*.pt"))
OUT_JSON = ROOT / "docs/v5/detector/confusion_matrix.json"
OUT_PNG = ROOT / "docs/v5/figures/confusion_matrix.png"


def val_episodes(base):
    rng = np.random.default_rng(SPLIT_SEED)
    idx = list(range(len(base)))
    rng.shuffle(idx)
    nv = max(1, int(len(idx) * VAL_RATIO))
    vs = set(idx[:nv])
    return [e for i, e in enumerate(base) if i in vs]


def load_head(hp):
    head = MLPActionHead().to(DEVICE)
    sd = torch.load(str(hp), map_location=DEVICE, weights_only=False)
    head.load_state_dict(sd["model"])
    head.eval()
    return head


def main():
    base = torch.load(str(CACHE_V6), weights_only=False)
    va = swap_bboxes(val_episodes(base), OWL_ANN)
    X, y = build_windows_hold_aware(va)
    print(f"val 결정 시점 {len(X)}개, 헤드 {len(HEADS)} seed", flush=True)

    accs = []
    cms = []
    for hp in HEADS:
        head = load_head(hp)
        with torch.no_grad():
            pred = head(torch.from_numpy(X).float().to(DEVICE)).argmax(1).cpu().numpy()
        acc = float((pred == y).mean())
        accs.append(acc)
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        for t, p in zip(y, pred):
            cm[t, p] += 1
        cms.append(cm)
        print(f"  {hp.name}  val_acc={acc:.4f}")

    cm_sum = np.sum(cms, axis=0)
    cm_norm = cm_sum / np.maximum(cm_sum.sum(1, keepdims=True), 1)

    rep = {"n_val_decisions": int(len(X)), "n_seeds": len(HEADS),
           "val_acc_mean": float(np.mean(accs)), "val_acc_std": float(np.std(accs)),
           "class_names": NAMES, "confusion_matrix_sum": cm_sum.tolist(),
           "confusion_matrix_normalized": cm_norm.tolist(),
           "support": cm_sum.sum(1).tolist()}
    OUT_JSON.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"\nval_acc 평균 {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}%p (3 seed 합산 혼동행렬)")

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(NAMES, rotation=45, ha="right")
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(NAMES)
    ax.set_xlabel("예측 (Predicted)"); ax.set_ylabel("정답 (Ground Truth)")
    ax.set_title(f"Val 혼동행렬 (정규화) — acc {np.mean(accs)*100:.1f}%±{np.std(accs)*100:.1f}%p, n={len(X)}, 3 seed 합산")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v = cm_norm[i, j]
            if v > 0.005:
                ax.text(j, i, f"{v*100:.0f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="비율")
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=180, facecolor="white")
    print(f"저장: {OUT_PNG}")


if __name__ == "__main__":
    main()

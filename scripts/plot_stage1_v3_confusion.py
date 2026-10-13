#!/usr/bin/env python3
"""Stage 1 v3(image_proj, 5-class, 225ep, OWL-v2) val 혼동행렬 그래프.

train_stage1_v3_5cls_owl.py 실행 로그(logs/train_stage1_v3_5cls_owl.log)에 이미 찍힌
최종 혼동행렬 숫자를 그대로 시각화한다(재계산 아님 — 학습 스크립트 자체 출력).

출력: docs/v5/figures/stage1_v3_confusion.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT_PNG = ROOT / "docs/v5/figures/stage1_v3_confusion.png"

NAMES = ["strong_L", "weak_L", "center", "weak_R", "strong_R"]

# logs/train_stage1_v3_5cls_owl.log 최종 출력 그대로 (2026-08-07 완료, val_acc 94.09%)
CM = np.array([
    [797,   2,   3,   0,   3],
    [ 32, 635,  51,   3,   1],
    [  8,   4, 540,   7,   0],
    [  0,   0,   7, 665,  16],
    [  0,   0,   0,  69, 642],
], dtype=float)

N_CLASSES = len(NAMES)


def main():
    cm_norm = CM / CM.sum(axis=1, keepdims=True)
    val_acc = np.trace(CM) / CM.sum()

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(N_CLASSES)); ax.set_xticklabels(NAMES, rotation=45, ha="right")
    ax.set_yticks(range(N_CLASSES)); ax.set_yticklabels(NAMES)
    ax.set_xlabel("예측 (Predicted)")
    ax.set_ylabel("정답 (Ground Truth)")
    ax.set_title(f"image_proj Stage 1 학습(5-class, 225ep, OWL-v2)\n"
                 f"val 혼동행렬 — acc {val_acc*100:.1f}%, n={int(CM.sum())}")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            v = cm_norm[i, j]
            if v > 0.005:
                ax.text(j, i, f"{v*100:.0f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, label="ratio")
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=180, facecolor="white")
    print(f"저장: {OUT_PNG}")
    print(f"val_acc: {val_acc*100:.2f}%")


if __name__ == "__main__":
    main()

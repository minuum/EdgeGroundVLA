#!/usr/bin/env python3
"""배포 행동 헤드 val 혼동행렬을 '오류 심각도' 관점으로 시각화 — 2개 별도 이미지.

① action_head_error_severity_8x8.png — 8x8 혼동행렬을 셀 성격별로 색칠
   (정답=초록 / 같은 방향 성향 내 혼동=노랑, 대체 가능 / 좌우 반전=빨강, 치명적 / STOP 관련=회색)
② action_head_error_severity_3x3.png — 좌·우·중립 3방향으로 축약한 혼동행렬
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs/v5/detector/confusion_matrix_stage1v3_correct.json"
OUT_8X8 = ROOT / "docs/v5/figures/action_head_error_severity_8x8.png"
OUT_3X3 = ROOT / "docs/v5/figures/action_head_error_severity_3x3.png"

LEFTISH, RIGHTISH = {2, 4, 6}, {3, 5, 7}
C_OK, C_BENIGN, C_FLIP, C_STOP = "#bbf7d0", "#fef08a", "#fca5a5", "#e5e7eb"


def side(i):
    return "L" if i in LEFTISH else ("R" if i in RIGHTISH else "N")


def kind(i, j):
    if i == j:
        return "ok"
    if (side(i), side(j)) in (("L", "R"), ("R", "L")):
        return "flip"
    if i == 0 or j == 0:
        return "stop"
    return "benign"


def main():
    d = json.loads(SRC.read_text())
    cm = np.array(d["confusion_matrix_sum"])
    names = d["class_names"]
    n = len(names)
    total, correct = int(cm.sum()), int(np.trace(cm))
    err = total - correct
    agg_n = {"flip": 0, "benign": 0, "stop": 0}
    for i in range(n):
        for j in range(n):
            if i != j:
                agg_n[kind(i, j)] += int(cm[i][j])

    # ── ① 8x8 심각도 색칠 ──
    fig, ax = plt.subplots(figsize=(13, 12))
    colors = {"ok": C_OK, "benign": C_BENIGN, "flip": C_FLIP, "stop": C_STOP}
    for i in range(n):
        for j in range(n):
            v = int(cm[i][j])
            k = kind(i, j)
            face = colors[k] if v > 0 else "#ffffff"
            ax.add_patch(Rectangle((j, n - 1 - i), 1, 1, facecolor=face,
                                    edgecolor="#94a3b8", linewidth=0.8))
            if v > 0:
                bold = (i == j) or (k == "flip")
                ax.text(j + .5, n - 1 - i + .5, str(v), ha="center", va="center",
                        fontsize=22 if bold else 20,
                        fontweight="bold" if bold else "normal",
                        color="#0f172a")
    ax.set_xlim(0, n); ax.set_ylim(0, n)
    ax.set_xticks(np.arange(n) + .5); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=20)
    ax.set_yticks(np.arange(n) + .5); ax.set_yticklabels(names[::-1], fontsize=20)
    ax.set_xlabel("예측 (Predicted)", fontsize=25, fontweight="bold")
    ax.set_ylabel("정답 (Ground Truth)", fontsize=25, fontweight="bold")
    ax.set_title(f"val 혼동행렬 — 오류를 심각도로 구분\n정확도 {correct}/{total} = {correct/total*100:.2f}%",
                 fontsize=23, pad=14)
    ax.set_aspect("equal")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.legend(handles=[
        Patch(facecolor=C_OK, edgecolor="#94a3b8", label=f"정답  {correct}건"),
        Patch(facecolor=C_BENIGN, edgecolor="#94a3b8",
              label=f"같은 방향 성향 내 혼동 (대체 가능)  {agg_n['benign']}건 · 오류의 {agg_n['benign']/err*100:.1f}%"),
        Patch(facecolor=C_FLIP, edgecolor="#94a3b8",
              label=f"좌우 반전 (치명적)  {agg_n['flip']}건 · 전체의 {agg_n['flip']/total*100:.2f}%"),
        Patch(facecolor=C_STOP, edgecolor="#94a3b8",
              label=f"STOP 판정 시점 차이  {agg_n['stop']}건 · 오류의 {agg_n['stop']/err*100:.1f}%"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.20), fontsize=18, frameon=False)
    fig.tight_layout()
    OUT_8X8.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_8X8, dpi=170, facecolor="white", bbox_inches="tight")
    print(f"저장: {OUT_8X8.relative_to(ROOT)}")

    # ── ② 3x3 축약 ──
    lab3 = ["좌 계열\nLEFT·FWD+L·ROT_L", "우 계열\nRIGHT·FWD+R·ROT_R", "중립\nSTOP·FORWARD"]
    idx = {"L": 0, "R": 1, "N": 2}
    a = np.zeros((3, 3), dtype=int)
    for i in range(n):
        for j in range(n):
            a[idx[side(i)]][idx[side(j)]] += cm[i][j]
    acc3 = np.trace(a) / total
    rown = a / np.maximum(a.sum(1, keepdims=True), 1)

    fig2, ax2 = plt.subplots(figsize=(11, 10))
    im = ax2.imshow(rown, cmap="Greens", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, f"{a[i][j]}\n{rown[i][j]*100:.1f}%", ha="center", va="center",
                     fontsize=21, fontweight="bold" if i == j else "normal",
                     color="white" if rown[i][j] > .55 else "#0f172a")
    ax2.set_xticks(range(3)); ax2.set_xticklabels(lab3, fontsize=16, rotation=15, ha="right")
    ax2.set_yticks(range(3)); ax2.set_yticklabels(lab3, fontsize=16)
    ax2.set_xlabel("예측", fontsize=25, fontweight="bold"); ax2.set_ylabel("정답", fontsize=25, fontweight="bold")
    ax2.set_title(f"좌·우·중립 3방향으로 축약\n정확도 {acc3*100:.2f}%  (8-class 74.13% → +{acc3*100-74.13:.2f}%p)",
                  fontsize=23, pad=14)
    cbar = fig2.colorbar(im, ax=ax2, fraction=0.046, label="행 정규화 비율")
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label("행 정규화 비율", fontsize=17)
    fig2.tight_layout()
    fig2.savefig(OUT_3X3, dpi=170, facecolor="white", bbox_inches="tight")
    print(f"저장: {OUT_3X3.relative_to(ROOT)}")

    print(f"  정답 {correct} / 경미 {agg_n['benign']} / 좌우반전 {agg_n['flip']} / STOP {agg_n['stop']}")
    print(f"  3-way 축약 정확도 {acc3*100:.2f}%")


if __name__ == "__main__":
    main()

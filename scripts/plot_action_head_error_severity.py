#!/usr/bin/env python3
"""배포 행동 헤드 val 혼동행렬을 '오류 심각도' 관점으로 시각화.

좌: 8x8 혼동행렬을 셀 성격별로 색칠 — 정답(초록) / 같은 방향 성향 내 혼동(노랑,
    대체 가능) / 좌우 반전(빨강, 치명적) / STOP 관련(회색)
우: 좌·우·중립 3방향으로 축약한 혼동행렬 — 방향 성향만 보면 정확도가 얼마나 오르는지

출력: docs/v5/figures/action_head_error_severity.png
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
OUT = ROOT / "docs/v5/figures/action_head_error_severity.png"

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

    fig = plt.figure(figsize=(15.5, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1], wspace=0.28)

    # ── 좌: 8x8 심각도 색칠 ──
    ax = fig.add_subplot(gs[0])
    colors = {"ok": C_OK, "benign": C_BENIGN, "flip": C_FLIP, "stop": C_STOP}
    for i in range(n):
        for j in range(n):
            v = int(cm[i][j])
            k = kind(i, j)
            face = colors[k] if v > 0 else "#ffffff"
            ax.add_patch(Rectangle((j, n - 1 - i), 1, 1, facecolor=face,
                                    edgecolor="#94a3b8", linewidth=0.6))
            if v > 0:
                bold = (i == j) or (k == "flip")
                ax.text(j + .5, n - 1 - i + .5, str(v), ha="center", va="center",
                        fontsize=10 if bold else 9,
                        fontweight="bold" if bold else "normal",
                        color="#0f172a")
    ax.set_xlim(0, n); ax.set_ylim(0, n)
    ax.set_xticks(np.arange(n) + .5); ax.set_xticklabels(names, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(np.arange(n) + .5); ax.set_yticklabels(names[::-1], fontsize=9)
    ax.set_xlabel("예측 (Predicted)", fontsize=10)
    ax.set_ylabel("정답 (Ground Truth)", fontsize=10)
    ax.set_title(f"val 혼동행렬 — 오류를 심각도로 구분\n정확도 {correct}/{total} = {correct/total*100:.2f}%",
                 fontsize=11.5, pad=10)
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
    ], loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=8.5, frameon=False)

    # ── 우: 3x3 축약 ──
    ax2 = fig.add_subplot(gs[1])
    lab3 = ["좌 계열\nLEFT·FWD+L·ROT_L", "우 계열\nRIGHT·FWD+R·ROT_R", "중립\nSTOP·FORWARD"]
    idx = {"L": 0, "R": 1, "N": 2}
    a = np.zeros((3, 3), dtype=int)
    for i in range(n):
        for j in range(n):
            a[idx[side(i)]][idx[side(j)]] += cm[i][j]
    acc3 = np.trace(a) / total
    rown = a / np.maximum(a.sum(1, keepdims=True), 1)
    im = ax2.imshow(rown, cmap="Greens", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, f"{a[i][j]}\n{rown[i][j]*100:.1f}%", ha="center", va="center",
                     fontsize=10, fontweight="bold" if i == j else "normal",
                     color="white" if rown[i][j] > .55 else "#0f172a")
    ax2.set_xticks(range(3)); ax2.set_xticklabels(lab3, fontsize=8)
    ax2.set_yticks(range(3)); ax2.set_yticklabels(lab3, fontsize=8)
    ax2.set_xlabel("예측", fontsize=10); ax2.set_ylabel("정답", fontsize=10)
    ax2.set_title(f"좌·우·중립 3방향으로 축약\n정확도 {acc3*100:.2f}%  (8-class 74.13% → +{acc3*100-74.13:.2f}%p)",
                  fontsize=11.5, pad=10)
    fig.colorbar(im, ax=ax2, fraction=0.046, label="행 정규화 비율")

    fig.suptitle("행동 헤드 val 오류의 대부분은 '대체 가능한 행동' 혼동 — 주행을 실패시키는 좌우 반전은 전체의 2% 미만",
                 fontsize=12.5, y=1.0)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=170, facecolor="white", bbox_inches="tight")
    print(f"저장: {OUT.relative_to(ROOT)}")
    print(f"  정답 {correct} / 경미 {agg_n['benign']} / 좌우반전 {agg_n['flip']} / STOP {agg_n['stop']}")
    print(f"  3-way 축약 정확도 {acc3*100:.2f}%")


if __name__ == "__main__":
    main()

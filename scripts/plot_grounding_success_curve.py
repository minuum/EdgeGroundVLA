#!/usr/bin/env python3
"""그라운딩 성공률 vs threshold 그래프 — 교수님 요청 대응.

human_labels.json(296프레임, 사람이 "객체 없음(no_target)" 라벨링)과
owlv2_scores.json(같은 프레임의 OWL-v2 원점수)을 threshold별로 교차해
정탐률(객체 있는 프레임 중 검출)과 오탐률(객체 없는 프레임 중 오검출)을 계산한다.
CH64 64-17의 원본 데이터(owlv2_threshold_roc.py)를 재사용해 그래프로 시각화한다.

출력: docs/v5/figures/grounding_threshold_curve.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GDIR = ROOT / "docs/v5/hsv_owlv2_preview_20260704"
scores = json.loads((GDIR / "owlv2_scores.json").read_text())
labels = json.loads((GDIR / "human_labels.json").read_text())

present, absent = [], []
for k, s in scores.items():
    lb = labels.get(k, {})
    if lb.get("no_target") == "yes":
        absent.append(s)
    else:
        present.append(s)
present = np.array(present); absent = np.array(absent)
print(f"객체 있음(present) {len(present)}프레임 · 객체 없음(absent) {len(absent)}프레임")

THRESH = np.round(np.arange(0.05, 0.61, 0.025), 3)
tpr = [float((present >= t).mean()) for t in THRESH]   # 정탐률
fpr = [float((absent >= t).mean()) for t in THRESH]    # 오탐률(있다고 잘못 판단)

OP_THRESH = [0.20, 0.25]

fig, ax = plt.subplots(figsize=(7.2, 5.0))
ax.plot(THRESH, tpr, marker="o", ms=3.5, color="#16a34a", label="정탐률 (객체 있음 → 검출)")
ax.plot(THRESH, fpr, marker="o", ms=3.5, color="#dc2626", label="오탐률 (객체 없음 → 오검출)")
for t in OP_THRESH:
    ax.axvline(t, color="#94a3b8", linestyle="--", linewidth=1)
    ax.text(t + 0.003, 0.03, f"{t}", fontsize=9, color="#475569")
ax.set_xlabel("OWL-v2 confidence threshold")
ax.set_ylabel("비율")
ax.set_ylim(-0.02, 1.02)
ax.set_title(f"그라운딩 threshold vs 성공률 (n={len(present)+len(absent)}프레임, 사람 라벨 기준)")
ax.legend(loc="center left")
ax.grid(alpha=0.25)
fig.tight_layout()
OUT = ROOT / "docs/v5/figures/grounding_threshold_curve.png"
fig.savefig(OUT, dpi=180, facecolor="white")
print(f"저장: {OUT}")

# 확정값 지점 콘솔 출력
for t in (0.10, 0.20, 0.25, 0.30):
    i = int(round((t - 0.05) / 0.025))
    print(f"  threshold {t:.2f}: 정탐 {tpr[i]*100:.1f}%  오탐 {fpr[i]*100:.1f}%")

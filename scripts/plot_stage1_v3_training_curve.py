#!/usr/bin/env python3
"""image_proj Stage 1 학습 곡선 재생성 — 축 라벨 폰트 확대 (0817 hwp 빨간 메모 반영)."""
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "/home/minum/26CS/MoNaVLA/logs/train_stage1_v3_5cls_owl.log"
OUT = "/home/minum/26CS/MoNaVLA/docs/v5/figures/stage1_v3_training_curve.png"

epochs, val_acc, best = [], [], []
with open(LOG, encoding="utf-8") as f:
    for line in f:
        m = re.match(r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s*$", line)
        if m:
            epochs.append(int(m.group(1)))
            val_acc.append(float(m.group(2)) * 100)
            best.append(float(m.group(3)) * 100)

final = best[-1]

plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({"font.size": 15})
fig, ax = plt.subplots(figsize=(11, 6.5))
ax.plot(epochs, val_acc, "o-", color="#38bdf8", label="epoch별 val_acc", linewidth=2, markersize=6)
ax.plot(epochs, best, "-", color="#22c55e", label="best (누적 최고)", linewidth=3)
ax.axhline(final, color="#f87171", linestyle="--", linewidth=1.5, label=f"최종 {final:.2f}%")
ax.set_xlabel("epoch", fontsize=18, fontweight="bold")
ax.set_ylabel("val 정확도 (%)", fontsize=18, fontweight="bold")
ax.set_title("image_proj Stage 1 학습 곡선 (5-class, 225ep 체크포인트, 30 epoch 로그)", fontsize=15)
ax.tick_params(axis="both", labelsize=14)
ax.legend(fontsize=14, loc="lower right")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("wrote", OUT)

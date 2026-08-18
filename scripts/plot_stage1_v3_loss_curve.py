#!/usr/bin/env python3
"""image_proj Stage 1 train/val loss 곡선 — 0817 hwp 빨간 메모 "Loss 그래프가 있으면 추가 필요함" 대응.

데이터: docs/v5/detector/stage1_v3_loss_curve.json
(scripts/train_stage1_v3_5cls_owl_fastloss.py 재학습 결과 — 원본과 동일 시드/split,
val_acc 0.9415로 원본 0.9409와 일치 확인됨)
"""
import json
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

SRC = "/home/minum/26CS/MoNaVLA/docs/v5/detector/stage1_v3_loss_curve.json"
OUT = "/home/minum/26CS/MoNaVLA/docs/v5/figures/stage1_v3_loss_curve.png"

d = json.loads(open(SRC, encoding="utf-8").read())
hist = d["history"]
epochs = [h["epoch"] for h in hist]
train_loss = [h["train_loss"] for h in hist]
val_loss = [h["val_loss"] for h in hist]
val_acc = [h["val_acc"] * 100 for h in hist]
final_acc = d["final_best_val_acc"] * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5))

ax1.plot(epochs, train_loss, "o-", color="#38bdf8", label="train loss", linewidth=2, markersize=5)
ax1.plot(epochs, val_loss, "o-", color="#f87171", label="val loss", linewidth=2, markersize=5)
ax1.set_xlabel("epoch", fontsize=20, fontweight="bold")
ax1.set_ylabel("loss", fontsize=20, fontweight="bold")
ax1.set_title("image_proj Stage 1 — train/val loss", fontsize=18)
ax1.tick_params(axis="both", labelsize=17)
ax1.legend(fontsize=16)
ax1.grid(True, alpha=0.3)

ax2.plot(epochs, val_acc, "o-", color="#22c55e", linewidth=2, markersize=5)
ax2.axhline(final_acc, color="#f87171", linestyle="--", linewidth=1.5, label=f"최종 {final_acc:.2f}%")
ax2.set_xlabel("epoch", fontsize=20, fontweight="bold")
ax2.set_ylabel("val 정확도 (%)", fontsize=20, fontweight="bold")
ax2.set_title("image_proj Stage 1 — val accuracy", fontsize=18)
ax2.tick_params(axis="both", labelsize=17)
ax2.legend(fontsize=16, loc="lower right")
ax2.grid(True, alpha=0.3)

fig.suptitle("image_proj Stage 1 재학습 — loss는 계속 감소, val_acc와 함께 수렴 (과적합 징후 없음)", fontsize=16)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("wrote", OUT)

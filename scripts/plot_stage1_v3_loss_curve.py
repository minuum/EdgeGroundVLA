#!/usr/bin/env python3
"""image_proj Stage 1 Training/Validation curve — Accuracy/Loss 2개 별도 이미지.

Action Head(MLP) 학습 곡선과 동일한 색상 규칙: 초록(Training) / 하늘색(Validation).
데이터: docs/v5/detector/stage1_v3_loss_curve.json
(scripts/train_stage1_v3_5cls_owl_fastloss.py 재학습 결과 — 원본과 동일 시드/split)
"""
import json
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

SRC = "/home/minum/26CS/MoNaVLA/docs/v5/detector/stage1_v3_loss_curve.json"
OUT_LOSS = "/home/minum/26CS/MoNaVLA/docs/v5/figures/stage1_v3_loss_only.png"
OUT_ACC = "/home/minum/26CS/MoNaVLA/docs/v5/figures/stage1_v3_valacc_only.png"

GREEN, SKY = "#22c55e", "#38bdf8"

d = json.loads(open(SRC, encoding="utf-8").read())
hist = d["history"]
epochs = [h["epoch"] for h in hist]
train_loss = [h["train_loss"] for h in hist]
val_loss = [h["val_loss"] for h in hist]
train_acc = [h["train_acc"] * 100 for h in hist]
val_acc = [h["val_acc"] * 100 for h in hist]

# ── ① Loss 곡선 ──
fig1, ax1 = plt.subplots(figsize=(15, 9))
ax1.plot(epochs, train_loss, "-", color=GREEN, linewidth=3, label="Training")
ax1.plot(epochs, val_loss, "-", color=SKY, linewidth=3, label="Validation")
ax1.set_xlabel("Epoch", fontsize=30, fontweight="bold")
ax1.set_ylabel("Loss", fontsize=30, fontweight="bold")
ax1.set_title("image_proj Stage 1 학습 곡선 — Loss", fontsize=22, fontweight="bold")
ax1.tick_params(axis="both", labelsize=24)
for lbl in ax1.get_xticklabels() + ax1.get_yticklabels():
    lbl.set_fontweight("bold")
ax1.legend(fontsize=20)
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig(OUT_LOSS, dpi=150)
print("wrote", OUT_LOSS)

# ── ② Accuracy 곡선 ──
fig2, ax2 = plt.subplots(figsize=(15, 9))
ax2.plot(epochs, train_acc, "-", color=GREEN, linewidth=3, label="Training")
ax2.plot(epochs, val_acc, "-", color=SKY, linewidth=3, label="Validation")
ax2.set_xlabel("Epoch", fontsize=30, fontweight="bold")
ax2.set_ylabel("Accuracy (%)", fontsize=30, fontweight="bold")
ax2.set_title("image_proj Stage 1 학습 곡선 — Accuracy", fontsize=22, fontweight="bold")
ax2.tick_params(axis="both", labelsize=24)
for lbl in ax2.get_xticklabels() + ax2.get_yticklabels():
    lbl.set_fontweight("bold")
ax2.legend(fontsize=20, loc="lower right")
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(OUT_ACC, dpi=150)
print("wrote", OUT_ACC)

#!/usr/bin/env python3
"""image_proj Stage 1 train/val loss 곡선 + val accuracy 곡선 — 별도 이미지 2장으로 분리.

0817 hwp 빨간 메모 "Loss 그래프가 있으면 추가 필요함" 대응.
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
OUT_LOSS = "/home/minum/26CS/MoNaVLA/docs/v5/figures/stage1_v3_loss_only.png"
OUT_ACC = "/home/minum/26CS/MoNaVLA/docs/v5/figures/stage1_v3_valacc_only.png"

d = json.loads(open(SRC, encoding="utf-8").read())
hist = d["history"]
epochs = [h["epoch"] for h in hist]
train_loss = [h["train_loss"] for h in hist]
val_loss = [h["val_loss"] for h in hist]
val_acc = [h["val_acc"] * 100 for h in hist]
final_acc = d["final_best_val_acc"] * 100

# ── ① Loss 곡선 ──
fig1, ax1 = plt.subplots(figsize=(13, 7.5))
ax1.plot(epochs, train_loss, "o-", color="#38bdf8", label="train loss", linewidth=2.5, markersize=7)
ax1.plot(epochs, val_loss, "o-", color="#f87171", label="val loss", linewidth=2.5, markersize=7)
ax1.set_xlabel("epoch", fontsize=24, fontweight="bold")
ax1.set_ylabel("loss", fontsize=24, fontweight="bold")
ax1.set_title("image_proj Stage 1 — train/val loss", fontsize=19)
ax1.tick_params(axis="both", labelsize=18)
ax1.legend(fontsize=17)
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig(OUT_LOSS, dpi=150)
print("wrote", OUT_LOSS)

# ── ② val accuracy 곡선 ──
fig2, ax2 = plt.subplots(figsize=(13, 7.5))
ax2.plot(epochs, val_acc, "o-", color="#22c55e", linewidth=2.5, markersize=7, label="epoch별 val_acc")
ax2.axhline(final_acc, color="#f87171", linestyle="--", linewidth=1.5, label=f"최종 {final_acc:.2f}%")
ax2.set_xlabel("epoch", fontsize=24, fontweight="bold")
ax2.set_ylabel("val 정확도 (%)", fontsize=24, fontweight="bold")
ax2.set_title("image_proj Stage 1 — validation accuracy", fontsize=19)
ax2.tick_params(axis="both", labelsize=18)
ax2.legend(fontsize=17, loc="lower right")
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig(OUT_ACC, dpi=150)
print("wrote", OUT_ACC)

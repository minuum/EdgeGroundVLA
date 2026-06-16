#!/usr/bin/env python3
"""
교수님 보고서용 시각화 생성 — Exp46~51 종합
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "v5" / "bbox_nav_exp51" / "report_figs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})

CLASS_NAMES = ["STOP","FORWARD","LEFT","RIGHT","FWD+L","FWD+R","ROT_L","ROT_R"]

# ── 데이터 로드 ──────────────────────────────────────────────
exp46 = json.loads((ROOT/"docs/v5/bbox_nav_exp46/summary.json").read_text())
exp49 = json.loads((ROOT/"docs/v5/bbox_nav_exp49/summary.json").read_text())
exp49c= json.loads((ROOT/"docs/v5/bbox_nav_exp49/comprehensive_eval.json").read_text())
exp50 = json.loads((ROOT/"docs/v5/bbox_nav_exp50/summary.json").read_text())
exp51 = json.loads((ROOT/"docs/v5/bbox_nav_exp51/summary.json").read_text())
rob49 = json.loads((ROOT/"docs/v5/bbox_nav_exp49/image_robustness_results.json").read_text())
rob50 = json.loads((ROOT/"docs/v5/bbox_nav_exp50/image_robustness_results.json").read_text())
rob51 = json.loads((ROOT/"docs/v5/bbox_nav_exp51/image_robustness_results.json").read_text())


# ════════════════════════════════════════════════════════════
# Fig 1: 실험별 Val Accuracy 추이
# ════════════════════════════════════════════════════════════
def fig_exp_progression():
    exps  = ["Exp46\n(bbox+vis)", "Exp47\n(+text fp)", "Exp49\n(+goal)", "Exp50\n(+flip)", "Exp51\n(+crop)"]
    accs  = [93.2, 98.7, 96.4, 92.0, 93.3]
    notes = ["baseline", "fingerprint\n(paraphrase FAIL)", "grounded goal\n(paraphrase PASS)", "flip aug", "crop aug"]
    colors= ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(exps, accs, color=colors, width=0.55, zorder=3)
    ax.set_ylim(85, 102)
    ax.set_ylabel("Val Accuracy (%)", fontsize=12)
    ax.set_title("Experiment Progression — Val Accuracy", fontsize=14, fontweight="bold")
    ax.axhline(90, color="gray", ls="--", lw=1, zorder=2, label="90% threshold")
    ax.legend(fontsize=10)

    for bar, acc, note in zip(bars, accs, notes):
        ax.text(bar.get_x() + bar.get_width()/2, acc + 0.3,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width()/2, 86.0,
                note, ha="center", va="bottom", fontsize=8, color="gray")

    # paraphrase 결과 어노테이션
    ax.annotate("Paraphrase: 74.1% ❌", xy=(1, 98.7), xytext=(1.6, 100.5),
                arrowprops=dict(arrowstyle="-|>", color="red"), color="red", fontsize=9)
    ax.annotate("Paraphrase: 100% ✅", xy=(2, 96.4), xytext=(2.5, 99.5),
                arrowprops=dict(arrowstyle="-|>", color="green"), color="green", fontsize=9)

    plt.tight_layout()
    path = OUT_DIR / "fig1_exp_progression.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {path.name}")


# ════════════════════════════════════════════════════════════
# Fig 2: Robustness Heatmap (Exp49 vs 51)
# ════════════════════════════════════════════════════════════
def fig_robustness_heatmap():
    aug_labels = [
        "밝기 +40%", "밝기 −40%", "대비 +40%", "대비 −40%",
        "블러 σ=3", "블러 σ=6",
        "crop left 10%", "crop right 10%", "crop center 90%",
        "color jitter", "flip 대칭 반전"
    ]
    aug_keys = [
        "bright+40%","bright-40%","contrast+40%","contrast-40%",
        "blur_sigma3","blur_sigma6",
        "crop_left10%","crop_right10%","crop_center90%",
        "color_jitter","flip_horizontal"
    ]

    def get_rate(rob, key):
        for row in rob["aug_summary"]:
            if row["aug"] == key:
                return row["rate"] * 100
        return 0.0

    r49 = [get_rate(rob49, k) for k in aug_keys]
    r51 = [get_rate(rob51, k) for k in aug_keys]

    data = np.array([r49, r51])

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")

    ax.set_xticks(range(len(aug_labels)))
    ax.set_xticklabels(aug_labels, rotation=35, ha="right", fontsize=10)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Exp49\n(baseline)", "Exp51\n(flip+crop aug)"], fontsize=11)
    ax.set_title("Robustness Heatmap — Action 일치율 (%)", fontsize=13, fontweight="bold")

    for i in range(2):
        for j in range(len(aug_labels)):
            val = data[i, j]
            color = "white" if val < 40 else "black"
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="일치율 (%)")
    plt.tight_layout()
    path = OUT_DIR / "fig2_robustness_heatmap.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {path.name}")


# ════════════════════════════════════════════════════════════
# Fig 3: Crop Robustness 개선 (Exp49 → 51)
# ════════════════════════════════════════════════════════════
def fig_crop_improvement():
    categories = ["crop\nleft10%", "crop\nright10%", "crop\ncenter90%", "flip\n대칭반전"]
    keys       = ["crop_left10%","crop_right10%","crop_center90%","flip_horizontal"]

    def rates(rob):
        res = []
        for k in keys:
            for row in rob["aug_summary"]:
                if row["aug"] == k:
                    res.append(row["rate"] * 100)
        return res

    r49 = rates(rob49)
    r51 = rates(rob51)

    x   = np.arange(len(categories))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    b49 = ax.bar(x - w/2, r49, w, label="Exp49 (no aug)", color="#4C72B0", zorder=3)
    b51 = ax.bar(x + w/2, r51, w, label="Exp51 (flip+crop aug)", color="#55A868", zorder=3)

    ax.set_ylim(0, 115)
    ax.set_ylabel("Action 일치율 (%)", fontsize=12)
    ax.set_title("기하학적 Augmentation 효과 — Exp49 vs Exp51", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.axhline(60, color="orange", ls="--", lw=1.2, label="목표 60%")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    for bar in b49:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, v + 1, f"{v:.0f}%",
                ha="center", va="bottom", fontsize=10, color="#4C72B0", fontweight="bold")
    for bar in b51:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, v + 1, f"{v:.0f}%",
                ha="center", va="bottom", fontsize=10, color="#2a7a3b", fontweight="bold")

    plt.tight_layout()
    path = OUT_DIR / "fig3_crop_improvement.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {path.name}")


# ════════════════════════════════════════════════════════════
# Fig 4: Exp51 Confusion Matrix
# ════════════════════════════════════════════════════════════
def fig_confusion_matrix():
    cm = np.array(exp51["confusion"])
    labels = CLASS_NAMES

    # normalize by row
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sum > 0, cm / row_sum, 0.0)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Ground Truth", fontsize=11)
    ax.set_title("Exp51 Confusion Matrix (row-normalized)", fontsize=13, fontweight="bold")

    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j] > 0:
                color = "white" if cm_norm[i, j] > 0.6 else "black"
                ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]*100:.0f}%)",
                        ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized accuracy")
    plt.tight_layout()
    path = OUT_DIR / "fig4_confusion_matrix.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {path.name}")


# ════════════════════════════════════════════════════════════
# Fig 5: VLA 정의 비교 다이어그램 (텍스트 기반)
# ════════════════════════════════════════════════════════════
def fig_vla_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("VLA 아키텍처 비교", fontsize=14, fontweight="bold", y=1.01)

    # ── 왼쪽: True VLA (RT-2, OpenVLA 스타일) ──
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("True VLA (RT-2 / OpenVLA 스타일)", fontsize=11, fontweight="bold", color="#2c5f8a")

    boxes = [
        (5, 8.5, "Language Instruction\n\"왼쪽 바구니로 가\"", "#d4e8f7", "#2c5f8a"),
        (2, 6.5, "Camera Image\n(raw RGB)", "#d4f7d4", "#2a7a3b"),
        (5, 4.5, "LLM/VLM Backbone\n(e.g. PaLM-E, LLaMA)\n★ End-to-end joint\n  language+vision reasoning", "#fff3cd", "#856404"),
        (5, 2.0, "Action Tokens\n(8-class or continuous)\n★ Language directly\n  drives action", "#f8d7da", "#721c24"),
    ]
    arrows = [(5, 7.9, 5, 7.0), (2, 5.9, 4.0, 5.4), (5, 5.9, 5, 5.0), (5, 3.9, 5, 2.8)]

    for (x, y, text, fc, tc) in boxes:
        ax.add_patch(mpatches.FancyBboxPatch((x-2.2, y-0.55), 4.4, 1.1,
                     boxstyle="round,pad=0.1", fc=fc, ec=tc, lw=1.5, zorder=3))
        ax.text(x, y, text, ha="center", va="center", fontsize=8.5, color=tc, fontweight="bold", zorder=4)
    for (x1,y1,x2,y2) in arrows:
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle="-|>", color="gray", lw=1.5))

    ax.text(5, 0.6, "특징: 언어가 행동 생성에 직접 참여\n학습 시 language-vision 공동 fine-tuning",
            ha="center", fontsize=8.5, color="gray",
            bbox=dict(fc="white", ec="lightgray", boxstyle="round,pad=0.3"))

    # ── 오른쪽: 우리 모델 ──
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("MoNaVLA Exp51\n(Language-Conditioned Geo-Nav)", fontsize=11, fontweight="bold", color="#721c24")

    boxes2 = [
        (5, 9.0, "Language Instruction\n\"왼쪽 바구니로 가\"", "#d4e8f7", "#2c5f8a"),
        (5, 7.1, "Kosmos-2 Grounding\n(frozen, inference only)\n→ (cx=0.35, cy=0.42, area=0.08)", "#e8d4f7", "#5a2d8a"),
        (2, 5.0, "Camera Image\n→ Kosmos-2 Vision\n   Feature (1024-dim)\n   (frozen)", "#d4f7d4", "#2a7a3b"),
        (7, 5.0, "BBox History\n(32-dim)\ncx, cy, area,\nhas_bbox × 8 frames", "#fde8c8", "#8a4a00"),
        (5, 2.8, "MLP Classifier\nbbox(32)+vision(1024)+goal(3)\n→ 1059-dim → 8-class action\n★ 언어 정보 미포함", "#f8d7da", "#721c24"),
        (5, 1.0, "Action\n(8-class discrete)", "#fff3cd", "#856404"),
    ]
    arrows2 = [
        (5,8.4,5,7.7),(5,6.5,4.0,5.6),(5,6.5,6.0,5.6),
        (3.8,4.4,4.2,3.4),(6.2,4.4,5.8,3.4),(5,2.2,5,1.4),
    ]

    for (x, y, text, fc, tc) in boxes2:
        ax.add_patch(mpatches.FancyBboxPatch((x-2.1, y-0.6), 4.2, 1.2,
                     boxstyle="round,pad=0.1", fc=fc, ec=tc, lw=1.5, zorder=3))
        ax.text(x, y, text, ha="center", va="center", fontsize=8, color=tc, fontweight="bold", zorder=4)
    for (x1,y1,x2,y2) in arrows2:
        ax.annotate("", xy=(x2,y2), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle="-|>", color="gray", lw=1.5))

    # "Preprocessing" 라벨
    ax.add_patch(mpatches.FancyBboxPatch((0.3, 6.3), 9.4, 1.1,
                 boxstyle="round,pad=0.1", fc="none", ec="#8a4a00", lw=1.2, ls="--", zorder=2))
    ax.text(9.5, 6.85, "Preprocessing\n(언어는 여기서 소멸)", ha="right", fontsize=8,
            color="#8a4a00", style="italic")

    ax.text(5, 0.15, "특징: 언어 → 좌표 3개로 축소 후 소멸\nMLP는 기하학적 내비게이션 분류기",
            ha="center", fontsize=8.5, color="gray",
            bbox=dict(fc="white", ec="lightgray", boxstyle="round,pad=0.3"))

    plt.tight_layout()
    path = OUT_DIR / "fig5_vla_comparison.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {path.name}")


# ════════════════════════════════════════════════════════════
# Fig 6: 과적합 위험 분석 — 데이터 분포 & 분리 문제
# ════════════════════════════════════════════════════════════
def fig_overfitting_risk():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("과적합 위험 분석", fontsize=13, fontweight="bold")

    # 왼쪽: 클래스 불균형
    ax = axes[0]
    # Exp51 val confusion에서 GT 분포
    cm = np.array(exp51["confusion"])
    gt_counts = cm.sum(axis=1)
    labels_nonzero = [CLASS_NAMES[i] for i in range(8) if gt_counts[i] > 0]
    counts_nonzero = [gt_counts[i] for i in range(8) if gt_counts[i] > 0]
    colors = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2","#937860","#DA8BC3","#8C8C8C"]
    c_nz   = [colors[i] for i in range(8) if gt_counts[i] > 0]

    wedges, texts, autotexts = ax.pie(counts_nonzero, labels=labels_nonzero,
                                       autopct="%1.1f%%", colors=c_nz,
                                       startangle=140, pctdistance=0.8)
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title("Val 세트 Action 분포\n(FORWARD 지배적 → 쉬운 정확도 오해)", fontsize=10, fontweight="bold")

    # 오른쪽: 5-seed 안정성
    ax = axes[1]
    seeds = exp49c["random_seeds"]["seeds"]
    accs  = [a * 100 for a in exp49c["random_seeds"]["accs"]]
    mean_ = exp49c["random_seeds"]["mean"] * 100
    std_  = exp49c["random_seeds"]["std"]  * 100

    ax.bar([str(s) for s in seeds], accs, color="#4C72B0", zorder=3, width=0.55)
    ax.axhline(mean_, color="red", ls="--", lw=1.5, label=f"평균 {mean_:.1f}%")
    ax.fill_between([-0.5, 4.5], mean_-std_, mean_+std_,
                    color="red", alpha=0.12, label=f"±σ ({std_:.1f}%)")
    ax.set_ylim(88, 100)
    ax.set_xlabel("Random Seed", fontsize=11)
    ax.set_ylabel("Val Accuracy (%)", fontsize=11)
    ax.set_title(f"5-Seed 안정성 분석 (Exp49)\n평균 {mean_:.1f}% ± {std_:.1f}% — 분산 작음", fontsize=10, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    for i, (s, a) in enumerate(zip(seeds, accs)):
        ax.text(i, a + 0.15, f"{a:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.text(2, 88.5,
            "주의: 동일 환경 150개 에피소드 내 split\n"
            "→ 다른 환경/다른 카메라 미검증",
            ha="center", fontsize=8.5, color="#721c24",
            bbox=dict(fc="#f8d7da", ec="#721c24", boxstyle="round,pad=0.3"))

    plt.tight_layout()
    path = OUT_DIR / "fig6_overfitting_risk.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  ✅ {path.name}")


# ── 실행 ─────────────────────────────────────────────────────
print("보고서 시각화 생성 중...")
fig_exp_progression()
fig_robustness_heatmap()
fig_crop_improvement()
fig_confusion_matrix()
fig_vla_comparison()
fig_overfitting_risk()
print(f"\n모든 그림 저장: {OUT_DIR}")

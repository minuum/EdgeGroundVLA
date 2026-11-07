"""T4 어휘 확장 결과 막대그래프 (3분할×시드3 평균±sd). 출력 docs/v5/figures/ch79/text_cond_t4_bars.png"""
import json, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
plt.rcParams["font.family"] = "NanumGothic" if any("Nanum" in f.name for f in fm.fontManager.ttflist) else "DejaVu Sans"
ROOT = Path(__file__).resolve().parents[1]
d = json.load(open(ROOT / "docs/v5/egvla2_fm/text_cond_t4.json"))
VS = ["V0_img_only", "V3_old_vocab", "V4_big_vocab"]; LAB = ["V0\n이미지 전용", "V3\n기존 어휘(6+3)", "V4\n확장 어휘(40+40)"]; COL = ["#94a3b8", "#f87171", "#fbbf24"]
P = [("① 안 본 부정 문구 14개 → 정지율\n(높을수록 좋음, 기준 ≥90%)", "held_neg_stop_bbox", 90, "min"),
     ("② 같은 색·다른 물체 4개 → 정지율\n(gray chair 등, 높을수록 좋음)", "hard_neg_stop_bbox", None, None),
     ("③ 안 본 동의 표현 16개에서 정지\n(낮을수록 좋음, 기준 V0+3%p 이내)", "held_pos_stop", None, None),
     ("④ 속성만 다른 문구 2개에서 정지\n(red basket·blue bin, 정답 불명확·참고)", "ambig_stop", None, None)]
fig, axs = plt.subplots(1, 4, figsize=(17, 4.6))
for ax, (t, k, line, _) in zip(axs, P):
    m = [np.mean([r[k] for kk, r in d.items() if kk.split("|")[1] == v]) * 100 for v in VS]; s = [np.std([r[k] for kk, r in d.items() if kk.split("|")[1] == v]) * 100 for v in VS]
    ax.bar(range(3), m, yerr=s, color=COL, capsize=3, edgecolor="#1e293b")
    for i, (a, e) in enumerate(zip(m, s)): ax.text(i, a + e + 2, f"{a:.0f}%", ha="center", fontsize=11, fontweight="bold")
    if line: ax.axhline(line, ls="--", color="#16a34a", lw=1); ax.text(2.45, line + 1.5, f"기준 {line}%", ha="right", fontsize=8, color="#16a34a")
    if k == "held_pos_stop": ax.axhline(m[0] + 3, ls="--", color="#16a34a", lw=1); ax.text(2.45, m[0] + 4.5, "기준 V0+3%p", ha="right", fontsize=8, color="#16a34a")
    ax.set_ylim(0, 118); ax.set_xticks(range(3)); ax.set_xticklabels(LAB, fontsize=8.5); ax.set_title(t, fontsize=9.5); ax.grid(axis="y", alpha=.3); ax.set_ylabel("%")
fig.suptitle("T4 어휘 확장 — 학습에 전혀 없는 문구로만 평가 (3분할 × 시드 3, 평균 ± sd, 오프라인)", fontsize=12)
fig.tight_layout(); fig.savefig(ROOT / "docs/v5/figures/ch79/text_cond_t4_bars.png", dpi=110); print("ok")

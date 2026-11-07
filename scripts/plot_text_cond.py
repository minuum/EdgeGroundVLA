"""T1·T2 결과를 퍼센트 막대그래프로 (3분할×시드3 평균±sd). 출력: docs/v5/figures/ch79/text_cond_bars.png"""
import json, numpy as np
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
plt.rcParams["font.family"] = "NanumGothic" if any("Nanum" in f.name for f in fm.fontManager.ttflist) else "DejaVu Sans"
ROOT = Path(__file__).resolve().parents[1]
t2 = json.load(open(ROOT / "docs/v5/egvla2_fm/text_cond_t2.json"))
VS = ["V0_img_only", "V1_text_const", "V2_paraphrase_aug", "V3_aug_neg"]
LAB = ["V0\n이미지 전용", "V1\n문구 고정", "V2\n+패러프레이즈", "V3\n+부정 지시"]
COL = ["#94a3b8", "#38bdf8", "#4ade80", "#f87171"]
def agg(key):
    m, s = [], []
    for v in VS:
        x = [r[key] for k, r in t2.items() if k.split("|")[1] == v and key in r]
        m.append(np.mean(x)); s.append(np.std(x))
    return np.array(m), np.array(s)
panels = [("① 학습한 부정 문구 → 정지율 (높을수록 좋음)", "stop_neg_nobbox", True),
          ("② 안 본 부정 문구 → 정지율 (높을수록 좋음)", "unseen_stop_neg_nobbox", True),
          ("③ 정상 문구에서 정지 (낮을수록 좋음)", "stop_pos_bbox", True),
          ("④ 안 본 동의 표현에서 정지 (낮을수록 좋음)", "unseen_stop_pos", True)]
fig, axs = plt.subplots(1, 4, figsize=(17, 4.2))
for ax, (ttl, key, pct) in zip(axs, panels):
    m, s = agg(key); m, s = m * 100, s * 100
    b = ax.bar(range(4), m, yerr=s, color=COL, capsize=3, edgecolor="#1e293b")
    for i, (a, e) in enumerate(zip(m, s)): ax.text(i, a + e + 2, f"{a:.0f}%", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 118); ax.set_xticks(range(4)); ax.set_xticklabels(LAB, fontsize=8.5); ax.set_title(ttl, fontsize=10); ax.grid(axis="y", alpha=.3)
    ax.set_ylabel("%")
fig.suptitle("텍스트 조건화 T1·T2 — 퍼센트 지표 (3분할 × 시드 3 평균 ± 표준편차, 오프라인)", fontsize=12)
fig.tight_layout(); fig.savefig(ROOT / "docs/v5/figures/ch79/text_cond_bars.png", dpi=110); print("ok")

"""T4 문구별(학습에 없던 36개) 정지율 그림. 출력 docs/v5/figures/ch79/text_cond_t4_phrases.png, 표 데이터 docs/v5/egvla2_fm/text_cond_t4_phrases.json"""
import json, numpy as np, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.font_manager as fm
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "scripts")); import text_vocab as TV
plt.rcParams["font.family"] = "NanumGothic" if any("Nanum" in f.name for f in fm.fontManager.ttflist) else "DejaVu Sans"
d = json.load(open(ROOT / "docs/v5/egvla2_fm/text_cond_t4.json"))
def st(v, p, key="stop"): return float(np.mean([r["per"][p][key] for k, r in d.items() if k.split("|")[1] == v])) * 100
G = [("정상: 안 본 동의 표현·문장형 (멈추면 안 됨)", TV.POS_HELD, "#38bdf8"), ("부정: 새 물체 (멈춰야 함)", TV.NEG_HELD, "#f87171"),
     ("부정: 색은 같고 물체가 다름 (멈춰야 함)", TV.HARD_NEG, "#fb923c"), ("모호: 속성만 다름 (정답 불명확·참고)", TV.AMBIG, "#a78bfa")]
rows = []; labels = []; cols = []
for g, L, c in G:
    for p in L: rows.append(dict(group=g, phrase=p, V0=st("V0_img_only", p), V3=st("V3_old_vocab", p), V4=st("V4_big_vocab", p))); labels.append(p); cols.append(c)
json.dump(rows, open(ROOT / "docs/v5/egvla2_fm/text_cond_t4_phrases.json", "w"), ensure_ascii=False, indent=1)
fig, ax = plt.subplots(figsize=(11, 11)); y = np.arange(len(rows))[::-1]
ax.barh(y + .18, [r["V3"] for r in rows], .34, color="#f8717199", label="V3 기존 어휘(6+3)"); ax.barh(y - .18, [r["V4"] for r in rows], .34, color=cols, label="V4 확장 어휘(40+40)")
ax.scatter([r["V0"] for r in rows], y, marker="|", color="#475569", s=120, label="V0 이미지 전용(기준 정지율)", zorder=5)
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9); ax.set_xlim(0, 105); ax.set_xlabel("정지율 %  (3분할×시드3 평균)"); ax.grid(axis="x", alpha=.3)
i = 0
for g, L, c in G:
    i += len(L); ax.axhline(len(rows) - i - .5, color="#94a3b8", lw=.8)
    ax.text(103, len(rows) - i + len(L) / 2 - .5, g.split(":")[0], rotation=90, va="center", ha="left", fontsize=9, color=c)
ax.legend(loc="lower right", fontsize=8.5); ax.set_title("T4 — 학습에 전혀 없던 36개 문구별 정지율 (정상은 낮을수록, 부정은 높을수록 좋음)", fontsize=11)
fig.tight_layout(); fig.savefig(ROOT / "docs/v5/figures/ch79/text_cond_t4_phrases.png", dpi=105); print("ok")

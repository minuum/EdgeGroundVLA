#!/usr/bin/env python3
"""CH64 64-19/64-20 — 3체제 분해(체크포인트 순효과) + 젯슨-로컬 confidence gap."""
import json
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
_kf = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"
fm.fontManager.addfont(_kf)
plt.rcParams["font.family"] = fm.FontProperties(fname=_kf).get_name()
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/v5/ch64_figs"; OUT.mkdir(parents=True, exist_ok=True)
RECV = Path("/home/minum/MoNaVLA/inference_sessions_recv")
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "verm": "#D55E00",
     "pink": "#CC79A7", "sky": "#56B4E9", "grey": "#999999"}
plt.rcParams.update({"figure.dpi": 130, "font.size": 10.5, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True,
                     "axes.edgecolor": "#cccccc", "savefig.bbox": "tight",
                     "savefig.facecolor": "white", "figure.facecolor": "white"})
POS = {"trackA_strong_left": "강좌", "trackA_weak_left": "약좌",
       "trackF_center": "중앙", "trackA_weak_right": "약우",
       "trackA_strong_right": "강우"}


def wil(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0, c - h), 100 * min(1, c + h)


def regimes():
    df = pd.read_csv(RECV / "20260731" / "episode_log.csv")
    df.columns = [c.strip() for c in df.columns]
    df["dt"] = pd.to_datetime(df["날짜"])
    A = df[(df["#"] >= 125) & (df["#"] <= 147) & df["경로"].isin(POS)].copy()
    B = df[(df["#"] >= 149) & (df["#"] <= 179) & df["경로"].isin(POS)].copy()
    d = df[df["체크포인트"] == "exp73_owl_trackF_v6_mlp_holdaware_seed0.pt"].copy()
    Cc = d[(d["dt"] >= pd.Timestamp("2026-07-30 19:25")) & ~d["#"].between(200, 209)
           & (d["#"] != 230) & d["경로"].isin(POS)].copy()
    for x in (A, B, Cc):
        x["pos"] = x["경로"].map(POS); x["ok"] = x["결과"] == "성공"
    return A, B, Cc


# ============ 64-19: 3체제 분해 ============
A, B, Cc = regimes()
fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.5),
                         gridspec_kw={"width_ratios": [1.0, 1.35]})

ax = axes[0]
labs = ["A\n구 배포모델\nthr 0.25 · 가드X",
        "B\nexp73 챔피언\nthr 0.25 · 가드X",
        "C\nexp73 챔피언\nthr 0.20 · 가드O"]
kn = [(A.ok.sum(), len(A)), (B.ok.sum(), len(B)), (Cc.ok.sum(), len(Cc))]
ps = [100 * k / n for k, n in kn]
err = np.array([[p - wil(k, n)[0] for p, (k, n) in zip(ps, kn)],
                [wil(k, n)[1] - p for p, (k, n) in zip(ps, kn)]])
cols = [C["grey"], C["orange"], C["green"]]
ax.bar([0, 1, 2], ps, 0.56, color=cols, edgecolor="white", linewidth=0.6,
       yerr=err, capsize=4, error_kw={"lw": 1.1, "ecolor": "#444"})
for xi, (p, (k, n)) in zip([0, 1, 2], zip(ps, kn)):
    ax.text(xi, p + 7.5, f"{k}/{n}\n{p:.1f}%", ha="center", fontweight="bold",
            fontsize=10)
ax.annotate("", xy=(0.72, 44), xytext=(0.28, 44),
            arrowprops=dict(arrowstyle="->", color=C["verm"], lw=2.2))
ax.text(0.5, 47, "체크포인트 교체\n+47.2%p\n(p=0.0008)", ha="center", fontsize=9,
        color=C["verm"], fontweight="bold")
ax.annotate("", xy=(1.72, 78), xytext=(1.28, 78),
            arrowprops=dict(arrowstyle="->", color=C["blue"], lw=2.2))
ax.text(1.5, 73, "thr 0.20\n+회복로직\n+18.0%p\n(p=0.025)", ha="center", va="top",
        fontsize=9, color=C["blue"], fontweight="bold")
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(labs, fontsize=9)
ax.set_ylabel("실기 성공률 (%)"); ax.set_ylim(0, 118)
ax.set_title("(A) 3체제 분해 — A→B는 같은 날·같은 위치·threshold 고정\n"
             "즉 체크포인트 순효과가 분리 측정됨", fontsize=10.5, pad=8)

ax = axes[1]
x = np.arange(5); w = 0.27
names = ["강좌", "약좌", "중앙", "약우", "강우"]
for i, (lab, dfx, col) in enumerate([("A 구모델 (thr.25)", A, C["grey"]),
                                     ("B exp73 (thr.25)", B, C["orange"]),
                                     ("C exp73 (thr.20+가드)", Cc, C["green"])]):
    vals, txt = [], []
    for p in names:
        g = dfx[dfx.pos == p]; k, n = g.ok.sum(), len(g)
        vals.append(100 * k / n if n else 0); txt.append(f"{k}/{n}" if n else "")
    ax.bar(x + (i - 1) * w, vals, w, color=col, label=lab, edgecolor="white",
           linewidth=0.5)
    for xi, (v, t) in zip(x + (i - 1) * w, zip(vals, txt)):
        if t:
            ax.text(xi, v + 2, t, ha="center", fontsize=7.3, color="#333")
ax.set_xticks(x); ax.set_xticklabels(names)
ax.set_ylabel("실기 성공률 (%)"); ax.set_ylim(0, 122)
ax.legend(fontsize=8.3, ncol=1, loc="upper left", framealpha=0.95)
ax.set_title("(B) 위치별 — 전 위치에서 A<B<C 단조 개선\n"
             "(약좌만 B에서 n=3으로 표본 부족)", fontsize=10.5, pad=8)
fig.suptitle("64-19. 개선의 주동력은 threshold가 아니라 체크포인트 교체였다 "
             "— 64-11 정정",
             fontsize=12.5, y=1.04, fontweight="bold")
fig.savefig(OUT / "fig_64_19_regimes.png"); plt.close(fig)
print("saved fig_64_19_regimes.png")

# ============ 64-20: 젯슨-로컬 gap ============
d = json.load(open(ROOT / "docs/v5/grounding_analysis/jetson_local_gap.json"))
ms = np.array([r["local_score"] for r in d["frames"]["miss"]])
hs = np.array([r["local_score"] for r in d["frames"]["hit"]])
T = 0.20
fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3),
                         gridspec_kw={"width_ratios": [1.35, 1.0, 1.0]})

ax = axes[0]
bins = np.linspace(0, 0.9, 46)
ax.hist(ms, bins=bins, color=C["verm"], alpha=0.78,
        label=f"젯슨 미검출 {len(ms)}프레임\n(젯슨 score < 0.20 확정)")
ax.hist(hs, bins=bins, color=C["blue"], alpha=0.62,
        label=f"젯슨 검출 {len(hs)}프레임(대조군)")
ax.axvline(T, color="#222", ls="--", lw=1.8)
ax.text(T + 0.012, ax.get_ylim()[1] * 0.93, "threshold\n0.20", fontsize=9,
        fontweight="bold")
ax.set_xlabel("로컬 재실행 OWL-v2 max score"); ax.set_ylabel("프레임 수")
ax.legend(fontsize=8.2, framealpha=0.95)
ax.set_title("(A) 미검출 프레임은 로컬에서도 대부분 낮다\n"
             f"미검출군 중앙 {np.median(ms):.3f} vs 검출군 {np.median(hs):.3f}",
             fontsize=10.5, pad=8)

ax = axes[1]
up = int((ms >= T).sum()); down = int((hs < T).sum())
mat = np.array([[len(ms) - up, up], [down, len(hs) - down]])
im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max())
for i in range(2):
    for j in range(2):
        tot = mat[i].sum()
        ax.text(j, i, f"{mat[i, j]}\n({100*mat[i,j]/tot:.1f}%)", ha="center",
                va="center", fontsize=12, fontweight="bold",
                color="white" if mat[i, j] > mat.max() * 0.55 else "#222")
ax.set_xticks([0, 1]); ax.set_xticklabels(["로컬 미검출", "로컬 검출"], fontsize=9)
ax.set_yticks([0, 1]); ax.set_yticklabels(["젯슨\n미검출", "젯슨\n검출"], fontsize=9)
ax.grid(False)
ax.set_title(f"(B) 젯슨×로컬 일치도 86.6%\n"
             f"(참고: fp32×fp16 일치도 90.0%)", fontsize=10.5, pad=8)

ax = axes[2]
th = [0.20, 0.15, 0.10]
rec = [0.0] + [100 * ((ms >= t) & (ms < T)).sum() / len(ms) for t in th[1:]]
fpr = [0.0, 40.5, 74.7]
xx = np.arange(3); w2 = 0.36
ax.bar(xx - w2 / 2, rec, w2, color=C["green"], label="추가 회수되는\n미검출 비율",
       edgecolor="white", linewidth=0.6)
ax.bar(xx + w2 / 2, fpr, w2, color=C["verm"], label="오탐률\n(원본 로컬 ROC)",
       edgecolor="white", linewidth=0.6)
for xi, v in zip(xx - w2 / 2, rec):
    ax.text(xi, v + 1.8, f"+{v:.1f}%p", ha="center", fontsize=8.5, color=C["green"])
for xi, v in zip(xx + w2 / 2, fpr):
    ax.text(xi, v + 1.8, f"{v:.1f}%", ha="center", fontsize=8.5, color=C["verm"])
ax.set_xticks(xx); ax.set_xticklabels([f"{t:.2f}" for t in th])
ax.set_xlabel("threshold"); ax.set_ylabel("비율 (%)"); ax.set_ylim(0, 88)
ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
ax.set_title("(C) threshold를 더 내려도 남는다\n"
             "회수분보다 오탐이 훨씬 빠르게 증가", fontsize=10.5, pad=8)

fig.suptitle("64-20. 젯슨-로컬 gap은 부차적 — has_bbox=False의 78.7%는 로컬에서도 미검출",
             fontsize=12.5, y=1.035, fontweight="bold")
fig.savefig(OUT / "fig_64_20_jetson_gap.png"); plt.close(fig)
print("saved fig_64_20_jetson_gap.png")

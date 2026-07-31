#!/usr/bin/env python3
"""CH64 64-18 — 100개 스크리닝(89%) vs 7/23 baseline(33.3%) + 원인 분해 그림."""
from math import sqrt
from pathlib import Path
import numpy as np
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

C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "verm": "#D55E00",
     "pink": "#CC79A7", "sky": "#56B4E9", "grey": "#999999", "black": "#222222"}
plt.rcParams.update({"figure.dpi": 130, "font.size": 10.5, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True, "axes.edgecolor": "#cccccc",
                     "savefig.bbox": "tight", "savefig.facecolor": "white",
                     "figure.facecolor": "white"})


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


POS = ["강좌", "약좌", "중앙", "약우", "강우"]
OLD = [(0, 5), (1, 4), (0, 1), (1, 4), (3, 5)]        # 7/23 소다 스크리닝 21회
NEW = [(16, 20), (16, 20), (20, 20), (19, 20), (18, 20)]  # 7/31 100회

fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.3),
                         gridspec_kw={"width_ratios": [1.5, 1.0, 1.05]})

# ---------- (A) 위치별 7/23 vs 7/31 ----------
ax = axes[0]
x = np.arange(len(POS)); w = 0.38
old_p = [100 * k / n for k, n in OLD]
new_p = [100 * k / n for k, n in NEW]
new_err = np.array([[p - wilson(k, n)[0] for p, (k, n) in zip(new_p, NEW)],
                    [wilson(k, n)[1] - p for p, (k, n) in zip(new_p, NEW)]])

ax.bar(x - w / 2, old_p, w, color=C["grey"], label="7/23 baseline (n=21)",
       edgecolor="white", linewidth=0.6)
ax.bar(x + w / 2, new_p, w, color=C["green"], label="7/31 100개 (n=100)",
       edgecolor="white", linewidth=0.6,
       yerr=new_err, capsize=3, error_kw={"lw": 1.1, "ecolor": "#444"})
for xi, (p, (k, n)) in zip(x - w / 2, zip(old_p, OLD)):
    ax.text(xi, p + 2, f"{k}/{n}", ha="center", fontsize=8.5, color="#555")
for xi, (p, (k, n)) in zip(x + w / 2, zip(new_p, NEW)):
    ax.text(xi, p + 9, f"{k}/{n}", ha="center", fontsize=8.5,
            color=C["green"], fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(POS)
ax.set_ylabel("실기 성공률 (%)"); ax.set_ylim(0, 118); ax.set_xlim(-0.62, 5.45)
ax.axhline(89, color=C["green"], ls=":", lw=1.2, alpha=0.7)
ax.text(4.62, 89, "전체\n89%", fontsize=8.5, color=C["green"], ha="left",
        va="center", fontweight="bold")
ax.axhline(33.3, color=C["grey"], ls=":", lw=1.2, alpha=0.8)
ax.text(4.62, 33.3, "전체\n33.3%", fontsize=8.5, color="#666", ha="left",
        va="center")
ax.set_title("(A) 위치별 실기 성공률 — 전 위치 개선\n"
             "33.3% → 89.0% (Fisher p=3.3e-07)", fontsize=10.5, pad=8)
ax.legend(fontsize=8.5, loc="lower right", framealpha=0.95)

# ---------- (B) 성패를 가른 것: 그라운딩 ----------
ax = axes[1]
bars = ax.bar([0, 1], [86.2, 45.7], 0.55,
              color=[C["green"], C["verm"]], edgecolor="white", linewidth=0.6)
ax.set_xticks([0, 1])
ax.set_xticklabels(["성공 89건", "실패 11건"])
ax.set_ylabel("세션 평균 grounding 성공률 gnd% (%)")
ax.set_ylim(0, 100)
for xi, v in zip([0, 1], [86.2, 45.7]):
    ax.text(xi, v + 2.5, f"{v}%", ha="center", fontweight="bold",
            fontsize=11, color=C["green"] if v > 60 else C["verm"])
ax.annotate("", xy=(1, 45.7), xytext=(0, 86.2),
            arrowprops=dict(arrowstyle="->", color="#666", lw=1.3,
                            connectionstyle="arc3,rad=-0.25"))
ax.text(0.5, 68, "-40.5%p", ha="center", fontsize=10, color="#444",
        fontweight="bold")
ax.text(0.5, 12, "실패 11건 중 3건은 gnd%=0\n(세션 내내 한 번도 미검출)",
        ha="center", fontsize=8.5, color="#555",
        bbox=dict(boxstyle="round,pad=0.35", fc="#f6f6f6", ec="#ddd"))
ax.set_title("(B) 성패는 정책이 아니라\n그라운딩에서 갈린다", fontsize=10.5, pad=8)

# ---------- (C) threshold 하향 실측 효과 ----------
ax = axes[2]
th = ["0.25\n(기존)", "0.20\n(적용)", "0.15\n(배제)"]
det = [19.7, 33.8, 49.3]           # soda 실측: weak_left 실패세션 71프레임
fp = [0.0, 12.7, 40.5]             # minum 원본 로컬 ROC 오탐률
xx = np.arange(3); w2 = 0.36
ax.bar(xx - w2 / 2, det, w2, color=C["blue"], label="실측 검출률\n(약좌 실패 71프레임)",
       edgecolor="white", linewidth=0.6)
ax.bar(xx + w2 / 2, fp, w2, color=C["orange"], label="오탐률\n(원본 로컬 ROC)",
       edgecolor="white", linewidth=0.6)
for xi, v in zip(xx - w2 / 2, det):
    ax.text(xi, v + 1.5, f"{v}%", ha="center", fontsize=8.5, color=C["blue"])
for xi, v in zip(xx + w2 / 2, fp):
    ax.text(xi, v + 1.5, f"{v}%", ha="center", fontsize=8.5, color=C["orange"])
ax.set_xticks(xx); ax.set_xticklabels(th)
ax.set_ylabel("비율 (%)"); ax.set_ylim(0, 60)
ax.axvspan(0.62, 1.38, color=C["green"], alpha=0.09, zorder=0)
ax.text(1, 41, "채택\n(검출 1.71배↑)", ha="center", fontsize=8.5,
        color=C["green"], fontweight="bold")
ax.set_title("(C) threshold 0.25→0.20\n검출률 1.71배 (오탐 12.7% 감수)",
             fontsize=10.5, pad=8)
ax.legend(fontsize=7.8, loc="upper left", framealpha=0.95)

fig.suptitle("64-18. 100개 스크리닝 89% — 개선의 원인은 헤드가 아니라 그라운딩 가용성",
             fontsize=12.5, y=1.035, fontweight="bold")
fig.savefig(OUT / "fig_64_18_100test.png")
plt.close(fig)
print("saved fig_64_18_100test.png")

# ---------- 좌우 비대칭 별도 그림 ----------
fig, ax = plt.subplots(figsize=(5.6, 3.9))
groups = ["좌측 계열\n(강좌+약좌, n=40)", "우측 계열\n(강우+약우, n=40)"]
kn = [(32, 40), (37, 40)]
ps = [100 * k / n for k, n in kn]
err = np.array([[p - wilson(k, n)[0] for p, (k, n) in zip(ps, kn)],
                [wilson(k, n)[1] - p for p, (k, n) in zip(ps, kn)]])
ax.bar([0, 1], ps, 0.5, color=[C["pink"], C["sky"]], edgecolor="white",
       linewidth=0.6, yerr=err, capsize=5, error_kw={"lw": 1.2, "ecolor": "#444"})
for xi, (p, (k, n)) in zip([0, 1], zip(ps, kn)):
    lo, hi = wilson(k, n)
    ax.text(xi, p + 6, f"{k}/{n} = {p:.1f}%", ha="center", fontweight="bold",
            fontsize=10)
    ax.text(xi, lo - 8, f"95%CI\n{lo:.0f}~{hi:.0f}%", ha="center", fontsize=8.5,
            color="white", va="top")
ax.set_xticks([0, 1]); ax.set_xticklabels(groups)
ax.set_ylabel("실기 성공률 (%)"); ax.set_ylim(0, 120)
ax.axhspan(wilson(32, 40)[0], wilson(37, 40)[1], color=C["grey"], alpha=0.10, zorder=0)
ax.text(0.5, 111, "CI 겹침 · Fisher p=0.193 → 통계적으로 유의하지 않음",
        ha="center", fontsize=9, color=C["verm"], fontweight="bold")
ax.text(0.5, 20, "평균 스텝: 좌 17~19 vs 우 10~12 (1.6배)\n"
                 "→ 성공률 차이는 미미하나 '더 헤맨다'는 경향은 남음",
        ha="center", fontsize=8.5, color="#444",
        bbox=dict(boxstyle="round,pad=0.35", fc="#f8f8f8", ec="#ddd"))
ax.set_title("64-18b. 좌우 비대칭 — n=40에서도 아직 유의하지 않음", fontsize=10.5, pad=8)
fig.savefig(OUT / "fig_64_18_leftright.png")
plt.close(fig)
print("saved fig_64_18_leftright.png")

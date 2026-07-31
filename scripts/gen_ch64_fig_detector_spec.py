#!/usr/bin/env python3
"""CH64 64-21 — 백필 score로 본 "어떤 프레임이 어려운가" + 강우 예외 진단.

특화 소형 검출기의 스펙을 데이터로 정하기 위한 분석.
입력: soda 백필 사이드카(docs/inference_sessions/backfill_scores/*.json, 1084프레임)
"""
import glob
import json
from pathlib import Path

import h5py
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
plt.rcParams.update({"figure.dpi": 130, "font.size": 10.5, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True,
                     "axes.edgecolor": "#cccccc", "savefig.bbox": "tight",
                     "savefig.facecolor": "white", "figure.facecolor": "white"})
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/v5/ch64_figs"
RECV = Path("/home/minum/MoNaVLA/inference_sessions_recv")
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "verm": "#D55E00",
     "pink": "#CC79A7", "sky": "#56B4E9", "grey": "#999999"}
POS = {"trackA_strong_left": "강좌", "trackA_weak_left": "약좌",
       "trackF_center": "중앙", "trackA_weak_right": "약우",
       "trackA_strong_right": "강우"}
NAMES = ["중앙", "약우", "강우", "약좌", "강좌"]
K = "exp73_owl_trackF_v6_mlp_holdaware_seed0.pt"
IDX = {Path(f).stem.replace("session_", ""): f
       for f in glob.glob(str(RECV / "2026*" / "h5" / "*.h5"))}
SC = {Path(f).stem.replace("session_", ""): json.load(open(f))["scores"]
      for f in glob.glob(str(ROOT / "docs/inference_sessions/backfill_scores/*.json"))}


def build():
    log = pd.read_csv(RECV / "20260731" / "episode_log.csv")
    log.columns = [c.strip() for c in log.columns]
    d = log[(log["체크포인트"] == K) & log["경로"].isin(POS)].copy()
    d["dt"] = pd.to_datetime(d["날짜"])
    w = d[(d.dt >= pd.Timestamp("2026-07-30 19:25")) & ~d["#"].between(200, 209)
          & (d["#"] != 230)].copy()
    w["pos"] = w["경로"].map(POS); w["ok"] = w["결과"] == "성공"
    sess, frames = [], []
    for _, x in w.iterrows():
        sid = str(x["session_id"])
        with h5py.File(IDX[sid], "r") as h:
            bb = np.array(h["grounding/bbox"])
        s = np.array(SC.get(sid, [])[:len(bb)], dtype=float) if sid in SC else np.array([])
        det = bb[bb[:, 3] == 1]
        sess.append(dict(pos=x["pos"], ok=bool(x["ok"]), gnd=100 * bb[:, 3].mean(),
                         steps=float(x["steps"]), first_ok=bool(bb[0, 3]),
                         area=det[:, 2].mean() if len(det) else np.nan))
        for i, v in enumerate(s):
            if v >= 0:
                frames.append(dict(pos=x["pos"], score=v, area=bb[i, 2],
                                   has=bb[i, 3], prog=i / max(len(bb) - 1, 1)))
    return pd.DataFrame(sess), pd.DataFrame(frames)


se, fr = build()
fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.4),
                         gridspec_kw={"width_ratios": [1.15, 1.0, 1.15]})

# (A) 위치별 score 분포 — 어떤 배치가 검출기에게 어려운가
ax = axes[0]
data = [fr[fr.pos == p].score.values for p in NAMES]
bp = ax.boxplot(data, labels=NAMES, patch_artist=True, widths=0.55,
                medianprops=dict(color="#111", lw=2), showfliers=False)
for patch, p in zip(bp["boxes"], NAMES):
    patch.set_facecolor(C["verm"] if p == "강우" else C["sky"])
    patch.set_alpha(0.85)
ax.axhline(0.20, color="#222", ls="--", lw=1.6)
ax.text(0.62, 0.225, "threshold 0.20", fontsize=8.5, ha="left", fontweight="bold")
for i, p in enumerate(NAMES, 1):
    v = fr[fr.pos == p]
    ax.text(i, 0.86, f"미달\n{100*(v.score<0.20).mean():.0f}%", ha="center", fontsize=8,
            color=C["verm"] if p == "강우" else "#555",
            fontweight="bold" if p == "강우" else "normal")
ax.set_ylabel("OWL-v2 confidence score"); ax.set_ylim(0, 0.95)
ax.set_title("(A) 위치별 검출 confidence — 강우가 압도적으로 어렵다\n"
             "(백필 score, threshold 0.20 구간 593프레임)", fontsize=10.3, pad=8)

# (B) area vs score — 작으면(멀면) 못 잡는다
ax = axes[1]
d2 = fr[fr.has == 1]
ax.scatter(d2.area, d2.score, s=13, alpha=0.32, color=C["blue"], edgecolors="none")
z = np.polyfit(d2.area, d2.score, 1)
xs = np.linspace(d2.area.min(), d2.area.max(), 50)
ax.plot(xs, np.polyval(z, xs), color=C["verm"], lw=2.2)
r = np.corrcoef(d2.area, d2.score)[0, 1]
ax.axhline(0.20, color="#222", ls="--", lw=1.4)
ax.text(0.97, 0.06, f"상관 r = {r:+.3f}\nn = {len(d2)}", transform=ax.transAxes,
        ha="right", fontsize=9.5, fontweight="bold", color=C["verm"],
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#ddd"))
ax.set_xlabel("검출된 bbox area (객체가 클수록 = 가까울수록)")
ax.set_ylabel("confidence score")
ax.set_title("(B) 멀면 못 잡는다 — 특화 검출기의\n1차 개선 타겟", fontsize=10.3, pad=8)

# (C) 강우 예외 — 첫프레임 검출률 vs 성공률
ax = axes[2]
x = np.arange(5); w = 0.27
g = se.groupby("pos")
first = [100 * g.get_group(p).first_ok.mean() for p in NAMES]
gnd = [g.get_group(p).gnd.mean() for p in NAMES]
ok = [100 * g.get_group(p).ok.mean() for p in NAMES]
ax.bar(x - w, first, w, color=C["orange"], label="첫 프레임 검출률", edgecolor="white", lw=0.5)
ax.bar(x, gnd, w, color=C["blue"], label="세션 평균 gnd%", edgecolor="white", lw=0.5)
ax.bar(x + w, ok, w, color=C["green"], label="실기 성공률", edgecolor="white", lw=0.5)
for xi, v in zip(x - w, first):
    ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=7.6, color="#8a5a00")
for xi, v in zip(x, gnd):
    ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=7.6, color="#04527d")
for xi, v in zip(x + w, ok):
    ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", fontsize=7.6, color="#046b4d")
i = NAMES.index("강우")
ax.annotate("강우: 검출은 최악인데 성공률 90%\n(경로가 짧아 FORWARD prior로 커버)",
            xy=(i, 49), xytext=(1.25, 122), fontsize=8.3, color=C["verm"],
            fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=C["verm"], lw=1.5))
ax.set_xticks(x); ax.set_xticklabels(NAMES); ax.set_ylim(0, 158)
ax.set_ylabel("%"); ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
ax.set_title("(C) 강우 예외 — '검출되면 성공'은 충분조건일 뿐\n필요조건은 아니다",
             fontsize=10.3, pad=8)

fig.suptitle("64-21. 특화 검출기 스펙을 데이터로 정하기 — 어려운 프레임은 \"멀고, 세션 초반\"이다",
             fontsize=12.4, y=1.035, fontweight="bold")
fig.savefig(OUT / "fig_64_21_detector_spec.png"); plt.close(fig)
print("saved fig_64_21_detector_spec.png")

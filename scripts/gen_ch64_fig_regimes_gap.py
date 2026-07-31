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


def build():
    """H5 runtime_config(ground truth)로 3개 arm 구성 — 날짜창 추정 대신 메타데이터 사용."""
    import glob, json
    import h5py
    df = pd.read_csv(RECV / "20260731" / "episode_log.csv")
    df.columns = [c.strip() for c in df.columns]
    d = df[df["경로"].isin(POS)].copy()
    d["pos"] = d["경로"].map(POS); d["ok"] = d["결과"] == "성공"
    idx = {Path(f).stem.replace("session_", ""): f
           for f in glob.glob(str(RECV / "2026*" / "h5" / "*.h5"))}
    rec = []
    for _, r in d.iterrows():
        f = idx.get(str(r["session_id"])); dt = str(r["날짜"])[:16]
        if f:
            with h5py.File(f, "r") as h:
                c = json.loads(h.attrs["runtime_config"]); bb = h["grounding/bbox"][:]
            rec.append(dict(pos=r["pos"], ok=r["ok"], gnd=100 * bb[:, 3].mean(),
                            ckpt=Path(c.get("checkpoint_path", "?")).name,
                            th=c["owlv2_thresh"], date=dt))
        else:
            rec.append(dict(pos=r["pos"], ok=r["ok"], gnd=np.nan, ckpt="h5없음",
                            th=np.nan, date=dt))
    m = pd.DataFrame(rec)
    K = "exp73_owl_trackF_v6_mlp_holdaware_seed0.pt"
    A = m[(m.ckpt == "h5없음") & (m.date >= "2026-07-23") & (m.date < "2026-07-24")]
    B = m[(m.ckpt == K) & (m.th == 0.25)]
    Cc = m[(m.ckpt == K) & (m.th == 0.20)]
    return m, A, B, Cc


NAMES = ["강좌", "약좌", "중앙", "약우", "강우"]
m, A, B, Cc = build()
tab = {p: [(int(x[x.pos == p].ok.sum()), len(x[x.pos == p])) for x in (A, B, Cc)]
       for p in NAMES}


def std(i):
    num = den = 0
    for p in NAMES:
        k, n = tab[p][i]
        if n:
            num += k / n; den += 1
    return 100 * num / den


fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4),
                         gridspec_kw={"width_ratios": [1.4, 1.0, 1.0]})

# (A) 위치별 3 arm
ax = axes[0]
x = np.arange(5); w = 0.27
for i, (lab, col) in enumerate([("A 구모델 @0.25", C["grey"]),
                                ("B exp73 @0.25", C["orange"]),
                                ("C exp73 @0.20+가드", C["green"])]):
    vals = [100 * tab[p][i][0] / tab[p][i][1] if tab[p][i][1] else 0 for p in NAMES]
    ax.bar(x + (i - 1) * w, vals, w, color=col, label=lab, edgecolor="white", linewidth=0.5)
    for xi, (v, p) in zip(x + (i - 1) * w, zip(vals, NAMES)):
        k, n = tab[p][i]
        if n:
            ax.text(xi, v + 2, f"{k}/{n}", ha="center", fontsize=7.2,
                    color="#b45309" if n < 6 else "#333",
                    fontweight="bold" if n < 6 else "normal")
ax.set_xticks(x); ax.set_xticklabels(NAMES); ax.set_ylim(0, 124)
ax.set_ylabel("실기 성공률 (%)")
ax.legend(fontsize=8.2, loc="upper left", framealpha=0.95)
ax.set_title("(A) 위치별 3 arm — 주황 라벨은 n<6(신뢰 낮음)\n"
             "B의 중앙·약우·강우는 표본 3~10에 불과", fontsize=10.3, pad=8)

# (B) 두 추정량이 순위를 뒤집는다
ax = axes[1]
sa, sb, sc = std(0), std(1), std(2)
ka, na = A.ok.sum(), len(A); kb, nb = B.ok.sum(), len(B); kc, nc = Cc.ok.sum(), len(Cc)
hard = [(sum(tab[p][i][0] for p in ["강좌", "약좌"]),
         sum(tab[p][i][1] for p in ["강좌", "약좌"])) for i in range(3)]
ests = {
    "위치 동일가중\n표준화": (sb - sa, sc - sb),
    "조참 pooled\n(전체)": (100 * kb / nb - 100 * ka / na, 100 * kc / nc - 100 * kb / nb),
    "강좌+약좌만\n(표본 충분)": (100 * hard[1][0] / hard[1][1] - 100 * hard[0][0] / hard[0][1],
                          100 * hard[2][0] / hard[2][1] - 100 * hard[1][0] / hard[1][1]),
}
xs = np.arange(3); w2 = 0.36
ck = [v[0] for v in ests.values()]; th = [v[1] for v in ests.values()]
ax.bar(xs - w2 / 2, ck, w2, color=C["orange"], label="체크포인트 효과(A→B)",
       edgecolor="white", linewidth=0.6)
ax.bar(xs + w2 / 2, th, w2, color=C["green"], label="threshold+가드(B→C)",
       edgecolor="white", linewidth=0.6)
for xi, v in zip(xs - w2 / 2, ck):
    ax.text(xi, v + 1.2, f"+{v:.1f}", ha="center", fontsize=8.6, color="#b45309")
for xi, v in zip(xs + w2 / 2, th):
    ax.text(xi, v + 1.2, f"+{v:.1f}", ha="center", fontsize=8.6, color="#047857")
ax.set_xticks(xs); ax.set_xticklabels(list(ests.keys()), fontsize=8.4)
ax.set_ylabel("성공률 차이 (%p)"); ax.set_ylim(0, 62)
ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
ax.set_title("(B) 추정 방식에 따라 순위가 뒤집힌다\n"
             "→ 어느 쪽이 주동력인지 단정 불가", fontsize=10.3, pad=8)

# (C) 유일하게 견고한 불변량
ax = axes[2]
mm = m[m.gnd.notna()]
g1 = mm[mm.gnd >= 80]; g0 = mm[mm.gnd < 80]
vals = [100 * g1.ok.mean(), 100 * g0.ok.mean()]
ax.bar([0, 1], vals, 0.5, color=[C["green"], C["verm"]], edgecolor="white", linewidth=0.6)
for xi, (v, g) in zip([0, 1], zip(vals, [g1, g0])):
    ax.text(xi, v + 2.5, f"{int(g.ok.sum())}/{len(g)}\n{v:.1f}%", ha="center",
            fontweight="bold", fontsize=10.5)
ax.set_xticks([0, 1]); ax.set_xticklabels(["gnd% ≥ 80", "gnd% < 80"])
ax.set_ylabel("실기 성공률 (%)"); ax.set_ylim(0, 118)
ax.text(0.5, 26, f"실패 세션 gnd% 최대 {mm[~mm.ok].gnd.max():.1f}%\n"
                 f"gnd%≥80 실패는 {len(mm[(~mm.ok)&(mm.gnd>=80)])}건뿐",
        ha="center", fontsize=8.6, color="#444",
        bbox=dict(boxstyle="round,pad=0.35", fc="#f7f7f7", ec="#ddd"))
ax.set_title("(C) 견고한 불변량 — 그라운딩이 되면\n성패가 거의 결정된다 (n=159)",
             fontsize=10.3, pad=8)

fig.suptitle("64-19. 개선 요인 분해 — 두 효과 모두 크지만 순위는 단정 불가, "
             "견고한 것은 '그라운딩 가용성' 하나",
             fontsize=12.3, y=1.04, fontweight="bold")
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

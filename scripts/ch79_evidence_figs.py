"""CH79 실기 폐루프(10/28) 근거 그림: 성공률+구간, 마지막 bbox 분포, 시간별 bbox 궤적, 실제 프레임. 원본: ROS_action/dryrun_closedloop_20261028 (수동 결과 열 + H5)."""
import csv, json
from pathlib import Path
import numpy as np, h5py, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
D = ROOT / "ROS_action/dryrun_closedloop_20261028"
import os
LIM = int(os.environ.get("A3_LIMIT", 0))  # >0이면 A3를 시간순 앞 N회만(보고용 부분 집계, 나머지는 CH79 전체본)
OUT = ROOT / "docs/v5/figures/ch79" / (f"first{LIM}" if LIM else ""); OUT.mkdir(parents=True, exist_ok=True)
BG, FG, GR = "#0a1628", "#e2e8f0", "#334155"
COL = {"A1": "#f59e0b", "A2": "#60a5fa", "A3": "#34d399", "A0": "#94a3b8"}
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GR, "text.color": FG, "axes.labelcolor": FG,
                     "xtick.color": FG, "ytick.color": FG, "font.family": "NanumGothic" if any("Nanum" in f.name for f in matplotlib.font_manager.fontManager.ttflist) else "DejaVu Sans"})


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def cond(r):
    c = json.loads(r["런타임설정"]); ck = c["ckpt_name"]; b = c.get("servo_blend") or 0
    if "servolabel1.0" in ck: return "A3+blend" if b else "A3"
    if "servolabel0.5" in ck: return "A2"
    if ck == "C_pi0_adaln_drop30_e120.pt": return "A1" if b else "A0"
    return None


rows = []
for r in csv.DictReader(open(D / "episode_log_snapshot.csv")):
    if not r["날짜"].startswith("2026-10-28"): continue
    c = cond(r)
    if c is None or c == "A3+blend": continue
    ok = r["결과"] == "성공"; h = D / "h5" / f"session_{r['session_id']}.h5"
    bb = None
    if h.exists():
        with h5py.File(h) as f: bb = f["grounding/bbox"][:]
    rows.append(dict(sid=r["session_id"], c=c, ok=ok, bb=bb, tgt=r["경로"], h=h))
if LIM:
    a3 = sorted([r for r in rows if r["c"] == "A3"], key=lambda r: r["sid"])[:LIM]; keep = {r["sid"] for r in a3}
    rows = [r for r in rows if r["c"] != "A3" or r["sid"] in keep]
print(len(rows), {k: sum(1 for r in rows if r["c"] == k) for k in COL})

# 1. 성공률
fig, ax = plt.subplots(figsize=(7.5, 3.6))
for i, k in enumerate(["A1", "A2", "A3"]):
    rr = [r for r in rows if r["c"] == k]; s = sum(r["ok"] for r in rr); n = len(rr); lo, hi = wilson(s, n)
    ax.barh(i, s / n * 100, color=COL[k], alpha=.85); ax.errorbar(s / n * 100, i, xerr=[[max(0, s / n * 100 - lo * 100)], [max(0, hi * 100 - s / n * 100)]], color=FG, capsize=4)
    ax.text(103, i, f"{s}/{n}  ({lo*100:.0f}~{hi*100:.0f}%)", va="center", fontsize=10)
ax.set_yticks(range(3)); ax.set_yticklabels(["A1 기준+서보혼합", "A2 servolabel0.5", "A3 servolabel1.0"]); ax.set_xlim(0, 135); ax.set_xlabel("수동 성공률 (%) · 막대 끝 선 = 95% 구간(Wilson)")
ax.invert_yaxis(); ax.grid(axis="x", color=GR, ls=":"); fig.tight_layout(); fig.savefig(OUT / "success_ci.png", dpi=150); plt.close()

# 2. 마지막 프레임 bbox 분포 (면적 vs 소실 프레임)
def stats(r):
    b = r["bb"]; has = b[:, 3] > 0.5
    last = b[has][-1] if has.any() else None
    return (last[2] if last is not None else np.nan, int((~has).sum()), abs(last[0] - .5) if last is not None else np.nan)
fig, ax = plt.subplots(figsize=(7.5, 4.4)); rng = np.random.default_rng(0)
for k in ["A1", "A2", "A3"]:
    for ok, mk in [(True, "o"), (False, "X")]:
        pts = [stats(r) for r in rows if r["c"] == k and r["ok"] == ok and r["bb"] is not None]
        if not pts: continue
        a = np.array(pts); ax.scatter(a[:, 0], a[:, 1] + rng.uniform(-.12, .12, len(a)), c=COL[k], marker=mk, s=60 if ok else 90, alpha=.75, edgecolors="none", label=f"{k} {'성공' if ok else '실패'} ({len(a)})")
ax.axvline(.25, color=FG, ls="--", lw=1); ax.text(.255, ax.get_ylim()[1] * .92, "정지 기준 0.25", fontsize=9)
ax.set_xlabel("마지막 프레임 타깃 면적 (클수록 과접근)"); ax.set_ylabel("타깃 소실 프레임 수(위아래 흩뿌림)"); ax.legend(fontsize=8, frameon=False, ncol=2); ax.grid(color=GR, ls=":")
fig.tight_layout(); fig.savefig(OUT / "final_bbox_scatter.png", dpi=150); plt.close()

# 3. 시간 정규화 궤적 (면적, cx)
def resamp(v, m=30): return np.interp(np.linspace(0, 1, m), np.linspace(0, 1, len(v)), v)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for key, lab, ls in [("A3", "A3 성공", "-"), ("A1s", "A1 성공", "--"), ("A1f", "A1 실패", ":")]:
    sel = [r for r in rows if r["bb"] is not None and ((key == "A3" and r["c"] == "A3") or (key == "A1s" and r["c"] == "A1" and r["ok"]) or (key == "A1f" and r["c"] == "A1" and not r["ok"]))]
    col = COL["A3"] if key == "A3" else COL["A1"]
    for j, (ix, nm) in enumerate([(2, "타깃 면적"), (0, "타깃 cx (0.5=정면)")]):
        M = []
        for r in sel:
            b = r["bb"]; has = b[:, 3] > .5
            if has.sum() < 3: continue
            M.append(resamp(b[has, ix]))
        M = np.array(M); t = np.linspace(0, 1, M.shape[1]); m = M.mean(0); s = M.std(0)
        axs[j].plot(t, m, color=col, ls=ls, lw=2, label=f"{lab} (n={len(M)})"); axs[j].fill_between(t, m - s, m + s, color=col, alpha=.12 if key == "A3" else .06)
        axs[j].set_title(nm, fontsize=10); axs[j].set_xlabel("에피소드 진행(0~1)"); axs[j].grid(color=GR, ls=":")
axs[0].axhline(.25, color=FG, ls="--", lw=1); axs[0].legend(fontsize=8, frameon=False); axs[1].axhline(.5, color=FG, ls="--", lw=1)
fig.tight_layout(); fig.savefig(OUT / "bbox_trajectories.png", dpi=150); plt.close()

# 4. 실제 프레임 (처음·마지막): A3 성공 2, A1 실패 2, 소실 사례
def frame(r, i):
    with h5py.File(r["h"]) as f: im = f["observations/images"][i]
    im = cv2.resize(np.asarray(im), (480, 270)); b = r["bb"][i]
    if b[3] > .5:
        x, y = int(b[0] * 480), int(b[1] * 270); s = int(np.sqrt(b[2]) * 270 / 2)
        cv2.rectangle(im, (x - s, y - s), (x + s, y + s), (255, 80, 80), 2)
    return im

def pick(c, ok, n):
    rs = [r for r in rows if r["c"] == c and r["ok"] == ok and r["bb"] is not None]
    rs.sort(key=lambda r: r["sid"]); return [rs[int(i)] for i in np.linspace(len(rs) * .15, len(rs) * .85, n)] if rs else []
sel = [("A3 성공", r) for r in pick("A3", True, 2)] + [("A1 실패", r) for r in pick("A1", False, 2)]
fig, axs = plt.subplots(len(sel), 3, figsize=(9.6, 2.7 * len(sel)))
for i, (lab, r) in enumerate(sel):
    n = len(r["bb"]); st, ar = stats(r)[0], stats(r)[1]
    for j, (idx, t) in enumerate([(0, "시작"), (n // 2, "중간"), (n - 1, "마지막")]):
        axs[i, j].imshow(frame(r, idx)); axs[i, j].axis("off")
        axs[i, j].set_title(f"{lab} {r['sid'][-6:]} · {t} (면적 {r['bb'][idx, 2]:.2f})", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "frames.png", dpi=110); plt.close()
print("done")

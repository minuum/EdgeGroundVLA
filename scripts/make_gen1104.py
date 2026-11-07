#!/usr/bin/env python3
"""CH79 11/04 2차 일반화(pos4~6 x G6~G8) 분석: 요약 JSON + 그림 3종 + 필름스트립. 10/29 파이프라인(ch79_generalization_figs·ch79_alignment_figs·make_filmstrips_1029)과 같은 지표.
원본: docs/inference_sessions/session_20261104_*.h5 (RGB). 성공 열: 로봇 쪽 logs/episode_log.csv(origin/monavla-driving)의 2026-11-04 행. 환경변수 EP_LOG로 경로 지정.
출력: docs/v5/egvla2_fm/gen_1104_summary.json, filmstrips1104/, filmstrips1104.json, docs/v5/figures/ch79/{generalization,alignment,counts}_1104.png
실행: .venv/bin/python3 scripts/make_gen1104.py  (CPU만 사용)"""
import csv, glob, json, os, subprocess, sys
from pathlib import Path
import cv2, h5py, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
FM = ROOT / "docs/v5/egvla2_fm"; OUT = ROOT / "docs/v5/figures/ch79"; FS = FM / "filmstrips1104"; FS.mkdir(parents=True, exist_ok=True)
BG, FG, GR = "#0a1628", "#e2e8f0", "#334155"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GR, "text.color": FG, "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
                     "font.family": "NanumGothic" if any("Nanum" in f.name for f in matplotlib.font_manager.fontManager.ttflist) else "DejaVu Sans"})
TC = {"G6": "#34d399", "G7": "#60a5fa", "G8": "#f59e0b"}
PLAN = 33  # 계획: 3 pos x 3 목표 x 33 = 297

# --- 로그 ---
ep = os.environ.get("EP_LOG")
txt = open(ep, encoding="utf-8").read() if ep else subprocess.check_output(["git", "show", "origin/monavla-driving:logs/episode_log.csv"], cwd=ROOT).decode("utf-8")
log = [r for r in csv.DictReader(txt.splitlines()) if r["날짜"].startswith("2026-11-04")]
logok = {r["session_id"]: r for r in log}

rows = []
for p in sorted(glob.glob(str(ROOT / "docs/inference_sessions/session_20261104_*.h5"))):
    f = h5py.File(p); c = json.loads(f.attrs["runtime_config"]); bb = f["grounding/bbox"][:]; a = f["actions"][:]; sid = str(f.attrs["session_id"]); has = bb[:, 3] > .5
    lr = logok.get(sid)
    rows.append(dict(sid=sid, start=c["start_position"], target=c["target_position"], ok=bool(lr and lr["결과"] == "성공"), in_log=lr is not None, n=len(bb), lost_frames=int((~has).sum()),
                     cx0=float(bb[0, 0]), final_cx=float(bb[-1, 0]), final_area=float(bb[-1, 2]), ly0=float(a[0, 1]), az0=float(a[0, 2]), ckpt=c.get("ckpt_name")))
(FM / "gen_1104_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
nolog_h5 = [r["sid"] for r in rows if not r["in_log"]]
noh5 = [r for r in log if r["session_id"] not in {x["sid"] for x in rows}]
meta = dict(n_h5=len(rows), n_log=len(log), log_success=sum(r["결과"] == "성공" for r in log), h5_ok=sum(r["ok"] for r in rows), planned=297,
            log_without_h5=[dict(no=r["#"], sid=r["session_id"], route=r["경로"], steps=r["steps"], t=r["날짜"]) for r in noh5], h5_without_log=nolog_h5)
(FM / "gen_1104_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))

POS = ["pos4", "pos5", "pos6"]; TG = ["G6", "G7", "G8"]
cx0 = np.array([r["cx0"] for r in rows]); fcx = np.array([r["final_cx"] for r in rows]); far = np.array([r["final_area"] for r in rows])

def side(c): return "L" if c < .4 else ("R" if c > .6 else "C")
# 시작 방향 확인: 시작 cx가 좌(<0.4)/우(>0.6)일 때 첫 스텝 ly 부호(ly<0 = 왼쪽 이동)
dirchk = {}
for r in rows:
    s = side(r["cx0"]); dirchk.setdefault((r["start"], r["target"]), []).append((s, (r["ly0"] < 0) if s == "L" else (r["ly0"] > 0) if s == "R" else None))

# --- 그림 1: 건수·시작/마지막 지표 ---
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.1, 1, 1]})
a = ax[0]; w = .26
for j, t in enumerate(TG):
    v = [sum(1 for r in rows if r["start"] == p and r["target"] == t) for p in POS]
    b = a.bar(np.arange(3) + (j - 1) * w, [x / PLAN * 100 for x in v], w, color=TC[t], label=t)
    for x, n_, bb_ in zip(np.arange(3) + (j - 1) * w, v, b): a.text(x, bb_.get_height() + 1.5, f"{n_}/{PLAN}", ha="center", fontsize=8.5)
a.axhline(100, color="#94a3b8", ls="--", lw=1); a.set_xticks(range(3)); a.set_xticklabels(POS); a.set_ylim(0, 118); a.set_ylabel("계획 33회 대비 수집률 (%)"); a.legend(ncol=3, fontsize=9, facecolor=BG, edgecolor=GR, loc="upper right")
a.set_title(f"조합별 수집률: 252/297 = {252/297*100:.1f}% (pos6은 11/33)", fontsize=10.5)
a = ax[1]
for t in TG:
    m = [r for r in rows if r["target"] == t]; a.scatter([r["cx0"] for r in m], [r["final_cx"] for r in m], s=18, color=TC[t], alpha=.7, label=t)
a.plot([0, 1], [.5, .5], color="#f87171", ls="--", lw=1); a.set_xlim(0, 1); a.set_ylim(0, 1); a.set_xlabel("시작 cx"); a.set_ylabel("마지막 cx"); a.grid(color=GR, ls=":"); a.legend(fontsize=9, facecolor=BG, edgecolor=GR)
a.set_title("시작 cx → 마지막 cx (점선 = 화면 중앙)", fontsize=10.5)
a = ax[2]
for t in TG:
    m = [r for r in rows if r["target"] == t]; a.scatter([r["final_cx"] for r in m], [r["final_area"] for r in m], s=18, color=TC[t], alpha=.7, label=t)
a.set_xlim(0, 1); a.set_ylim(.15, .45); a.set_xlabel("마지막 프레임 cx"); a.set_ylabel("마지막 프레임 bbox 면적"); a.grid(color=GR, ls=":")
for r in rows:
    if r["final_area"] > .4 or r["final_area"] < .21 or abs(r["final_cx"] - .5) > .25: a.annotate(r["sid"][-6:], (r["final_cx"], r["final_area"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
a.set_title(f"마지막 프레임: 소실 {sum(r['lost_frames'] for r in rows)}프레임, 면적 {far.min():.2f}~{far.max():.2f}", fontsize=10.5)
fig.tight_layout(); fig.savefig(OUT / "generalization_1104.png", dpi=140); plt.close(fig)

# --- 그림 2: 시작 방향 확인 + 마지막 cx 정렬 비율 (퍼센트 막대) ---
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
a = ax[0]; labs = []; vals = []; cols = []; ns = []
for p in POS:
    for t in TG:
        d = [x for x in dirchk.get((p, t), []) if x[1] is not None]
        labs.append(f"{p}\n{t}"); ns.append(len(d)); vals.append(100 * np.mean([x[1] for x in d]) if d else 0); cols.append(TC[t])
bars = a.bar(range(9), vals, color=cols)
for i, (v, n_) in enumerate(zip(vals, ns)): a.text(i, v + 1.5, f"{v:.0f}%\n(n={n_})", ha="center", fontsize=8)
a.set_xticks(range(9)); a.set_xticklabels(labs, fontsize=8.5); a.set_ylim(0, 125); a.set_ylabel("첫 스텝 횡이동 명령이 바구니 쪽 (%)")
allD = [x for v in dirchk.values() for x in v if x[1] is not None]
a.set_title(f"시작 cx가 좌(<0.4)·우(>0.6)인 {len(allD)}회 중 첫 명령이 같은 쪽: {100*np.mean([x[1] for x in allD]):.1f}%", fontsize=10.5)
a = ax[1]; labs2 = ["중앙 ±0.1", "0.6 이상(우측 치우침)", "0.4 이하(좌측 치우침)"]
v2 = [(abs(fcx - .5) <= .1).mean() * 100, (fcx >= .6).mean() * 100, (fcx <= .4).mean() * 100]
a.bar(range(3), v2, color=["#34d399", "#f59e0b", "#60a5fa"])
for i, v in enumerate(v2): a.text(i, v + 1.5, f"{v:.0f}% ({int(round(v*len(fcx)/100))}회)", ha="center", fontsize=9)
a.set_xticks(range(3)); a.set_xticklabels(labs2); a.set_ylim(0, 100); a.set_ylabel("세션 비율 (%)")
a.set_title(f"마지막 cx(중앙값 {np.median(fcx):.2f}, 범위 {fcx.min():.2f}~{fcx.max():.2f}): 정렬은 성공 판정 기준이 아니었음", fontsize=10.5)
fig.tight_layout(); fig.savefig(OUT / "counts_1104.png", dpi=140); plt.close(fig)

# --- 필름스트립: (pos,target)마다 프레임 수 중앙값에 가까운 세션 1개(결과 안 보고, 동률은 이른 시각) + 이상치 ---
import make_filmstrips as MF  # noqa: E402
sel = []
for p in POS:
    for t in TG:
        rs = sorted([r for r in rows if r["start"] == p and r["target"] == t], key=lambda r: r["sid"])
        if not rs: continue
        med = np.median([r["n"] for r in rs]); sel.append((min(rs, key=lambda r: (abs(r["n"] - med), r["sid"])), "조합 대표(프레임 수 중앙값)"))
chosen = {s[0]["sid"] for s in sel}
def add(r, tag):
    if r["sid"] not in chosen: sel.append((r, tag)); chosen.add(r["sid"])
add(max(rows, key=lambda r: r["final_area"]), "이상치: 마지막 면적 최대")
add(min(rows, key=lambda r: r["final_area"]), "이상치: 마지막 면적 최소")
add(max(rows, key=lambda r: r["final_cx"]), "이상치: 마지막 cx 최우측")
add(min(rows, key=lambda r: r["final_cx"]), "이상치: 마지막 cx 최좌측")
add(max(rows, key=lambda r: r["n"]), "이상치: 스텝 최다")
idx = []
for r, tag in sel:
    f = h5py.File(ROOT / f"docs/inference_sessions/session_{r['sid']}.h5"); im = f["observations/images"]; act = f["actions"][:]; bb = f["grounding/bbox"][:]; n = len(bb)
    frames = [np.ascontiguousarray(np.asarray(im[t])) for t in range(n)]
    g = [MF.gray(x) for x in frames]; dxs = [cv2.phaseCorrelate(g[t], g[t + 1], MF.WIN)[0][0] for t in range(n - 1)] + [None]
    items = []
    for t in range(n):
        name = f"{r['sid']}_{t:02d}.jpg"; has = bb[t, 3] > .5
        MF.frame_img(frames[t], bb[t], act[t], dxs[t], (255, 80, 80), f"#{t}").convert("RGB").save(FS / name, "JPEG", quality=82)
        items.append(dict(f=name, t=t, cx=round(float(bb[t, 0]), 2) if has else None, area=round(float(bb[t, 2]), 3) if has else None,
                          lx=round(float(act[t, 0]), 2), ly=round(float(act[t, 1]), 2), az=round(float(act[t, 2]), 2), dx=None if dxs[t] is None else round(float(dxs[t]), 1)))
    idx.append(dict(sid=r["sid"], start=r["start"], target=r["target"], tag=tag, final_cx=round(r["final_cx"], 2), final_area=round(r["final_area"], 3), items=items))
(FM / "filmstrips1104.json").write_text(json.dumps(idx, ensure_ascii=False))

# --- 그림 3: 마지막 프레임 사진(정렬) ---
pick = [(r, tag) for r, tag in sel if "이상치" in tag] + [(r, "가장 중앙") for r in sorted(rows, key=lambda r: abs(r["final_cx"] - .5))[:2]]
fig = plt.figure(figsize=(13, 7.2)); ax = fig.add_axes([0.06, 0.69, 0.88, 0.25])
ax.hist(fcx, bins=np.arange(0.25, 0.86, 0.025), color="#60a5fa", alpha=.85); ax.axvline(.5, color="#f87171", ls="--"); ax.axvline(np.median(fcx), color="#34d399")
ax.set_xlabel("마지막 프레임에서 바구니 중심의 가로 위치 cx (빨강 점선 = 중앙 0.5, 초록 = 중앙값)"); ax.set_ylabel("세션 수")
ax.set_title(f"252회 마지막 cx: 중앙값 {np.median(fcx):.2f}, 중앙 ±0.1 안 {int((abs(fcx-.5)<=.1).sum())}회, 0.6 이상 {int((fcx>=.6).sum())}회, 0.4 이하 {int((fcx<=.4).sum())}회", fontsize=11)
for i, (r, tag) in enumerate(pick[:8]):
    f = h5py.File(ROOT / f"docs/inference_sessions/session_{r['sid']}.h5"); im = np.ascontiguousarray(f["observations/images"][-1]).astype(np.uint8); bb = f["grounding/bbox"][-1]
    a = fig.add_axes([0.02 + (i % 4) * 0.245, 0.345 - (i // 4) * 0.27, 0.235, 0.22]); a.imshow(im); a.axvline(im.shape[1] / 2, color="#f87171", ls="--", lw=1)
    a.plot(bb[0] * im.shape[1], bb[1] * im.shape[0], "+", color="#fbbf24", ms=14, mew=2); a.axis("off"); a.set_title(f"{tag.replace('이상치: ', '')} · {r['sid'][-6:]} · {r['start']}→{r['target']} · cx {r['final_cx']:.2f}", fontsize=8)
fig.text(0.02, 0.01, "점선 = 화면 가로 중앙, + = 모델이 잡은 바구니 중심. 원본 RGB 마지막 프레임. 성공 판정은 '근접·정지'였고 좌우 정렬은 기준이 아니었음.", fontsize=9, color="#94a3b8")
fig.savefig(OUT / "alignment_1104.png", dpi=130); plt.close(fig)
print(meta); print(len(idx), sum(len(x["items"]) for x in idx))
for p in POS:
    for t in TG:
        m = [r for r in rows if r["start"] == p and r["target"] == t]
        if m: print(p, t, len(m), "cx0", round(np.mean([r["cx0"] for r in m]), 2), "fcx", round(np.median([r["final_cx"] for r in m]), 2), "area", round(np.median([r["final_area"] for r in m]), 3), "n", np.median([r["n"] for r in m]))
print("dir", {k: (sum(1 for x in v if x[1]), sum(1 for x in v if x[1] is not None)) for k, v in dirchk.items()})
print("dir fails", [(r["sid"], r["cx0"], r["ly0"]) for r in rows if side(r["cx0"]) == "L" and r["ly0"] >= 0 or side(r["cx0"]) == "R" and r["ly0"] <= 0])
print("outliers", [(r["sid"], tag, r["final_cx"], r["final_area"], r["n"]) for r, tag in sel if "이상치" in tag])

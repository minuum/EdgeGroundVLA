"""CH79 5-3: 10/29 위치 일반화 96회(출발 pos1~6 × 목표 라벨) 요약 JSON + 그림.
원본: docs/inference_sessions/session_20261029_*.h5, 성공 열은 로봇 쪽 logs/episode_log.csv(환경변수 EP_LOG, 기본 runs/e2e_cache/ep_log_1029.csv).
단계: 1 = 출발 pos4~6(새 출발 위치, 목표 라벨 5개), 2 = 출발 pos1~3(기존 출발 위치, 임시 목표 G6~G8)."""
import csv, glob, json, os
from pathlib import Path
import numpy as np, h5py
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
EP = Path(os.environ.get("EP_LOG", ROOT / "runs/e2e_cache/ep_log_1029.csv"))
OUT = ROOT / "docs/v5/figures/ch79"; OUT.mkdir(parents=True, exist_ok=True)
BG, FG, GR = "#0a1628", "#e2e8f0", "#334155"
C1, C2, CR = "#34d399", "#60a5fa", "#f59e0b"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GR, "text.color": FG, "axes.labelcolor": FG,
                     "xtick.color": FG, "ytick.color": FG, "font.family": "NanumGothic" if any("Nanum" in f.name for f in matplotlib.font_manager.fontManager.ttflist) else "DejaVu Sans"})

ok = {}
for r in csv.DictReader(open(EP, encoding="utf-8")):
    ok[r["session_id"]] = r["결과"] == "성공" if r["session_id"] not in ok else ok[r["session_id"]] and r["결과"] == "성공"
rows = []
for p in sorted(glob.glob(str(ROOT / "docs/inference_sessions/session_20261029_*.h5"))):
    f = h5py.File(p); c = json.loads(f.attrs["runtime_config"]); bb = f["grounding/bbox"][:]; has = bb[:, 3] > 0.5; sid = str(f.attrs["session_id"])
    rows.append(dict(sid=sid, start=c["start_position"], target=c["target_position"], ok=bool(ok[sid]), n=len(bb), lost_frames=int((~has).sum()),
                     cx0=float(bb[0, 0]), final_cx=float(bb[-1, 0]), final_area=float(bb[-1, 2]), stage=1 if c["start_position"] in ("pos4", "pos5", "pos6") else 2))
assert len(rows) == 96 and all(r["ok"] for r in rows), (len(rows), sum(r["ok"] for r in rows))
(ROOT / "docs/v5/egvla2_fm/gen_1029_summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))

# 10/28 pos1 학습 5곳의 시작 cx 범위(A3 세션)
ref = {}
D = ROOT / "ROS_action/dryrun_closedloop_20261028"
for r in csv.DictReader(open(D / "episode_log_snapshot.csv", encoding="utf-8")):
    if "servolabel1.0" not in (r["체크포인트"] or ""): continue
    h = D / "h5" / f"session_{r['session_id']}.h5"
    if h.exists(): ref.setdefault(r["경로"], []).append(float(h5py.File(h)["grounding/bbox"][0, 0]))

fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
ys = ["10/28 기준\npos1 목표 5곳", "pos4", "pos5", "pos6", "pos1", "pos2", "pos3"]
for i, (k, v) in enumerate(sorted(ref.items(), key=lambda kv: np.mean(kv[1]))):
    a.plot([min(v), max(v)], [0, 0], color=CR, lw=9, alpha=.35 + .1 * i, solid_capstyle="butt")
for r in rows:
    y = ys.index(r["start"]); a.scatter(r["cx0"], y + np.random.RandomState(hash(r["sid"]) % 2**31).uniform(-.18, .18), s=22, color=C1 if r["stage"] == 1 else C2, alpha=.85)
a.set_yticks(range(len(ys))); a.set_yticklabels(ys); a.invert_yaxis(); a.set_xlim(0, 1)
a.set_xlabel("시작 프레임에서 타깃의 가로 위치(cx, 0=왼쪽 끝 1=오른쪽 끝)"); a.grid(axis="x", color=GR, ls=":")
a.set_title("시작 시야: 새 출발 위치(초록)·임시 목표 G6~8(파랑)이 10/28 pos1 목표 5곳(주황 띠) 범위와 겹침", fontsize=10.5)
for s, c, l in [(1, C1, "1단계 pos4~6 (60회)"), (2, C2, "2단계 pos1~3 × G6~8 (36회)")]:
    q = [r for r in rows if r["stage"] == s]; b.scatter([r["final_cx"] for r in q], [r["final_area"] for r in q], s=26, color=c, alpha=.85, label=l)
b.axhline(0.4, color="#f87171", ls="--", lw=1); b.text(0.02, 0.405, "면적 0.4 (10/28 실패 표식 영역 근처)", color="#f87171", fontsize=9)
for r in rows:
    if r["final_area"] > 0.4 or abs(r["final_cx"] - .5) > .2: b.annotate(r["sid"][-6:], (r["final_cx"], r["final_area"]), fontsize=7, color=FG, xytext=(3, 3), textcoords="offset points")
b.set_xlim(0, 1); b.set_ylim(0.15, 0.45); b.set_xlabel("마지막 프레임 cx"); b.set_ylabel("마지막 프레임 bbox 면적"); b.grid(color=GR, ls=":"); b.legend(loc="lower left", fontsize=9, facecolor=BG, edgecolor=GR)
b.set_title("마지막 프레임: 소실 0, 면적 0.22~0.41 (96개 전부)", fontsize=10.5)
fig.tight_layout(); fig.savefig(OUT / "generalization_1029.png", dpi=150); print("ok", len(rows))

"""CH79 5-3 보조: 마지막 프레임에서 바구니의 좌우 정렬. 96회 마지막 cx 분포 + 대표 마지막 프레임(중앙선 표시). cx는 H5 grounding bbox 중심(바구니 중심), 중앙선 = 화면 가로 0.5."""
import json
from pathlib import Path
import numpy as np, h5py, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
rows = json.load(open(ROOT / "docs/v5/egvla2_fm/gen_1029_summary.json"))
OUT = ROOT / "docs/v5/figures/ch79"
BG, FG, GR = "#0a1628", "#e2e8f0", "#334155"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GR, "text.color": FG, "axes.labelcolor": FG,
                     "xtick.color": FG, "ytick.color": FG, "font.family": "NanumGothic" if any("Nanum" in f.name for f in matplotlib.font_manager.fontManager.ttflist) else "DejaVu Sans"})
cx = np.array([r["final_cx"] for r in rows])
by = {r["sid"][-6:]: r for r in rows}
flag = ["200618", "200929", "212247"]
ctr = sorted(rows, key=lambda r: abs(r["final_cx"] - .5))[:3]
left = sorted(rows, key=lambda r: r["final_cx"])[:2]
pick = [(by[k], "재판정 대상") for k in flag] + [(r, "가장 중앙") for r in ctr] + [(r, "가장 왼쪽") for r in left]

fig = plt.figure(figsize=(13, 7.4))
ax = fig.add_axes([0.06, 0.69, 0.88, 0.25])
ax.hist(cx, bins=np.arange(0.3, 0.81, 0.025), color="#60a5fa", alpha=.85)
ax.axvline(0.5, color="#f87171", ls="--"); ax.text(0.503, ax.get_ylim()[1] * .9, "화면 중앙 0.5", color="#f87171", fontsize=9)
ax.axvline(np.median(cx), color="#34d399"); ax.text(np.median(cx) + .003, ax.get_ylim()[1] * .75, f"중앙값 {np.median(cx):.2f}", color="#34d399", fontsize=9, bbox=dict(fc=BG, ec="none", pad=1))
ax.set_xlabel("마지막 프레임에서 바구니 중심의 가로 위치 cx"); ax.set_ylabel("세션 수")
ax.set_title(f"96회 마지막 cx: 범위 {cx.min():.2f}~{cx.max():.2f}, 중앙 ±0.1 안 {int((abs(cx-.5)<=.1).sum())}회, 0.6 이상 {int((cx>=.6).sum())}회 → 전체적으로 오른쪽에 치우쳐 정지", fontsize=11)
for i, (r, tag) in enumerate(pick):
    f = h5py.File(ROOT / f"docs/inference_sessions/session_{r['sid']}.h5"); im = np.ascontiguousarray(f["observations/images"][-1][:, :, ::-1]).astype(np.uint8)
    bb = f["grounding/bbox"][-1]; a = fig.add_axes([0.02 + i * 0.1225, 0.04, 0.115, 0.46]) if False else fig.add_axes([0.02 + (i % 4) * 0.245, 0.345 - (i // 4) * 0.27, 0.235, 0.22])
    a.imshow(im); a.axvline(im.shape[1] / 2, color="#f87171", ls="--", lw=1)
    a.plot(bb[0] * im.shape[1], bb[1] * im.shape[0], "+", color="#fbbf24", ms=14, mew=2); a.axis("off")
    a.set_title(f"{tag} · {r['sid'][-6:]} · {r['start']}→{r['target']} · cx {r['final_cx']:.2f}", fontsize=8.5)
fig.text(0.02, 0.01, "점선 = 화면 가로 중앙, + = 모델이 잡은 바구니 중심. 재판정 대상 3건은 바구니가 오른쪽에 치우쳤고, 가장 중앙인 건 cx 0.49~0.50. 성공 판정은 '근접·정지'였고 좌우 정렬은 기준이 아니었음.", fontsize=9, color="#94a3b8")
fig.savefig(OUT / "alignment_1029.png", dpi=140); print(len(cx), np.median(cx), (abs(cx-.5)<=.1).sum(), (cx>=.6).sum())

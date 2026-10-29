"""목표 위치(경로)별 A3 성공 세션 1개씩: 진행 1/5~5/5 시점의 프레임. 세션 선택 규칙 = 앞 60회 중 해당 경로 성공 세션의 스텝 수 중앙값에 가장 가까운 것(H5 보유)."""
import csv, json
from pathlib import Path
import numpy as np, h5py, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent.parent; D = ROOT / "ROS_action/dryrun_closedloop_20261028"
OUT = ROOT / "docs/v5/figures/ch79"; plt.rcParams.update({"font.family": "NanumGothic", "figure.facecolor": "#0a1628", "text.color": "#e2e8f0"})
rows = [r for r in csv.DictReader(open(D / "episode_log_snapshot.csv")) if r["날짜"].startswith("2026-10-28") and "servolabel1.0" in r["체크포인트"] and not json.loads(r["런타임설정"]).get("servo_blend")]
rows = sorted(rows, key=lambda r: r["session_id"])[:60]
NAME = {"trackA_strong_left": "강한 왼쪽", "trackA_weak_left": "약한 왼쪽", "trackF_center": "중앙", "trackA_weak_right": "약한 오른쪽", "trackA_strong_right": "강한 오른쪽"}
fig, axs = plt.subplots(5, 5, figsize=(13, 8.4))
for i, (t, nm) in enumerate(NAME.items()):
    c = []
    for r in rows:
        h = D / "h5" / f"session_{r['session_id']}.h5"
        if r["경로"] == t and r["결과"] == "성공" and h.exists():
            with h5py.File(h) as f: c.append((f["actions"].shape[0], r["session_id"], h))
    med = np.median([x[0] for x in c]); n, sid, h = min(c, key=lambda x: abs(x[0] - med))
    with h5py.File(h) as f:
        bb = f["grounding/bbox"][:]
        for j, fr in enumerate([.2, .4, .6, .8, 1.0]):
            k = int(round(fr * (n - 1))); im = cv2.resize(np.asarray(f["observations/images"][k]), (480, 270))
            if bb[k, 3] > .5:
                x, y, s = int(bb[k, 0] * 480), int(bb[k, 1] * 270), int(np.sqrt(bb[k, 2]) * 135); cv2.rectangle(im, (x - s, y - s), (x + s, y + s), (255, 80, 80), 2)
            ax = axs[i, j]; ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title((f"[{nm} · {sid[-6:]}]  " if j == 0 else "") + f"{j+1}/5 · 면적 {bb[k,2]:.2f}", fontsize=10, loc="left" if j == 0 else "center", color="#6ee7b7" if j == 0 else "#e2e8f0")
fig.suptitle("A3 성공 세션 — 목표 위치별 진행 1/5~5/5 (시작 위치 pos1, 빨간 상자 = bbox)", fontsize=13)
fig.tight_layout(); fig.savefig(OUT / "path_frames.png", dpi=105); print("ok")

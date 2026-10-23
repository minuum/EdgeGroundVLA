#!/usr/bin/env python3
"""영상 기반 회전량 비교(bbox 불필요): 인접 프레임의 수평 이동(위상 상관)으로 회전 속도를 학습(사람 연속 회전)과 실기(SYNC 버스트)에서 같은 단위(px@폭 320)로 비교.
학습: 회전 구간(|az|>0.3, |lx,ly|<0.2, 3프레임 이상) 누적 dx / 지속시간. 실기: rotate 스텝 직후 프레임과의 dx(버스트 0.4초)와 그 다음 idle 스텝의 dx(정지 후 잔여).
실행: .venv_lerobot/bin/python3 scripts/analyze_rotation_odometry.py   출력: docs/v5/egvla2_fm/rotation_odometry.json
"""
import glob
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import train_egvla2_s1 as S1  # noqa: E402
T = S1.T
WIN = None


def prep(img):
    g = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2GRAY)
    g = cv2.resize(g, (320, 180), interpolation=cv2.INTER_AREA)[:110]            # 위쪽 벽 영역(바닥 이동 영향 줄임)
    return np.float32(g)


def dx_of(a, b):
    global WIN
    if WIN is None:
        WIN = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, WIN)
    return float(dx), float(resp)


def main():
    eps = S1.load_main(); train = []
    rng = np.random.default_rng(0); order = rng.permutation(len(eps))
    for i in order:
        e = eps[i]; a = e["chunk"][:, 0]; n = len(a)
        runs = []; t = 0
        while t < n:
            if abs(a[t, 2]) > 0.3 and np.linalg.norm(a[t, :2]) < 0.2:
                s = np.sign(a[t, 2]); k = t
                while k + 1 < n and np.sign(a[k + 1, 2]) == s and abs(a[k + 1, 2]) > 0.3 and np.linalg.norm(a[k + 1, :2]) < 0.2:
                    k += 1
                if k - t >= 2:
                    runs.append((t, k, s))
                t = k + 1
            else:
                t += 1
        if not runs:
            continue
        with h5py.File(T.DATA / e["name"], "r") as f:
            im = f["images"]
            for t, k, s in runs:
                end = min(k + 1, n - 1); frames = [prep(im[j][:, :, ::-1]) for j in range(t, end + 1)]
                tot, ok = 0.0, True
                for p, q in zip(frames[:-1], frames[1:]):
                    dx, r = dx_of(p, q); tot += dx
                    ok &= r > 0.05
                if ok:
                    train.append((float(s), tot, (end - t) * 0.128))
        if len(train) >= 160:
            break
    tr = np.array(train)
    out = {"train_runs": int(len(tr))}
    for s in (1.0, -1.0):
        m = tr[:, 0] == s
        out[f"train_az{'+' if s > 0 else '-'}"] = dict(n=int(m.sum()), dx_total_px=float(tr[m, 1].mean()), dur_s=float(tr[m, 2].mean()), px_per_s=float((tr[m, 1] / tr[m, 2]).mean()))
    # 실기 SYNC 세션
    robot = []; coast = []
    for rp in sorted(glob.glob(str(ROOT / "ROS_action/dryrun_closedloop_20261022/reports/*.json"))):
        d = json.load(open(rp))
        if d.get("instruction_mode") != "SYNC" or "egvla2" not in d["model_name"]:
            continue
        hp = ROOT / "ROS_action/dryrun_closedloop_20261022/h5" / f"session_{d['session_id']}.h5"
        with h5py.File(hp, "r") as f:
            imgs = f["observations/images"]; nimg = imgs.shape[0]; h = d["history"]
            for i in range(min(len(h), nimg) - 1):
                a = h[i]["action"]
                if abs(a[0]) < 0.05 and abs(a[1]) < 0.05 and abs(a[2]) > 0.1:
                    dx, r = dx_of(prep(imgs[i]), prep(imgs[i + 1]))
                    if r > 0.05:
                        robot.append((float(np.sign(a[2])), dx))
                        if i + 2 < min(len(h), nimg) and abs(h[i + 1]["action"][2]) < 0.05 and abs(h[i + 1]["action"][0]) < 0.05 and abs(h[i + 1]["action"][1]) < 0.05:
                            dx2, r2 = dx_of(prep(imgs[i + 1]), prep(imgs[i + 2]))
                            if r2 > 0.05:
                                coast.append((float(np.sign(a[2])), dx2))
    rb = np.array(robot); cb = np.array(coast)
    for s in (1.0, -1.0):
        m = rb[:, 0] == s
        out[f"robot_burst_az{'+' if s > 0 else '-'}"] = dict(n=int(m.sum()), dx_px_per_burst=float(rb[m, 1].mean()), px_per_s_if_0p4=float(rb[m, 1].mean() / 0.4)) if m.any() else dict(n=0)
        mc = cb[:, 0] == s if len(cb) else np.array([], bool)
        out[f"robot_coast_az{'+' if s > 0 else '-'}"] = dict(n=int(mc.sum()), dx_px=float(cb[mc, 1].mean())) if mc.any() else dict(n=0)
    (ROOT / "docs/v5/egvla2_fm/rotation_odometry.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

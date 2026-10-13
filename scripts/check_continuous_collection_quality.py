#!/usr/bin/env python3
"""연속 조이스틱 수집 데이터(egvla2_continuous_v1) 품질 점검 — 로봇 서버/학습 서버 어디서나 실행 가능.

연속 액션 학습(FM 등)에 쓸모 있는 데이터인지 보는 지표:
  - 중간값 비율: 비영 프레임 중 0.15<|v|<0.9 (스틱을 끝까지 안 민 프레임)  → 높을수록 좋음
  - 포화 비율: 비영 프레임 중 |v|>=0.99
  - 코너 비율: lx,ly 둘 다 비영인 프레임 중 둘 다 포화(45° 대각선 고정)
  - ang_z 단독 스파이크: 앞뒤 프레임이 0인 1프레임짜리 비영 ang_z (오입력 의심)
  - 에피소드 상대위치별 중간값 분포
사용: python3 scripts/check_continuous_collection_quality.py [dir=ROS_action/mobile_vla_dataset_egvla2] [--per-episode] [--set 세트명]

v3(egvla2_continuous_v3: actions_raw/actions_exec/frame_timestamps) 공식 정의 (2026-10-13 합의):
  - 기준 데이터: actions_raw (방사형 데드존만 통과한 원값). v1 파일은 actions 사용(구 기준).
  - 중간값: 비영 프레임 중 (lx,ly) 벡터 크기 0.15<|v|<0.9 비율 (±축별 값도 병기)
  - 코너: lx,ly 둘 다 비영인 프레임 중 둘 다 |v|>=0.99 비율 (raw 기준)
  - az 단독 스파이크: 앞뒤가 0인 1프레임 비영 az / 비영 az 프레임 수
  - 속력 단계(exec): |exec| 크기 분포, 각도 연속성(비영 이동 프레임의 고유 각도 수, 연속 프레임 각도 변화)
  - 합격(제안): 중간값(크기) ≥50% 이고 코너 ≤20% 이고 az 스파이크 비율 ≤25%
  --set: docs/v5/egvla2_fm/meta/manual_dataset_sets.json의 세트 멤버만 사용 (예: pilot_v2_gauge_20261013)
"""
import glob
import sys

import h5py
import numpy as np

_pos = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and sys.argv[i - 1] != "--set"]
d = _pos[0] if _pos else "ROS_action/mobile_vla_dataset_egvla2"
per_ep = "--per-episode" in sys.argv
set_name = sys.argv[sys.argv.index("--set") + 1] if "--set" in sys.argv else None
fs = sorted(glob.glob(f"{d}/*.h5"))
if set_name:
    import json
    import os
    members = json.load(open("docs/v5/egvla2_fm/meta/manual_dataset_sets.json"))["sets"][set_name]
    members = {m if m.endswith(".h5") else m + ".h5" for m in members}
    fs = [f for f in fs if os.path.basename(f) in members]
if not fs:
    sys.exit(f"h5 없음: {d}")


def stats(a):
    out = {}
    for i, n in enumerate(("lin_x", "lin_y", "ang_z")):
        v = np.abs(a[:, i]); nz = v[v > 1e-3]
        out[n] = dict(nz=len(nz), sat=float(np.mean(nz >= 0.99)) if len(nz) else float("nan"),
                      mid=float(np.mean((nz > 0.15) & (nz < 0.9))) if len(nz) else float("nan"))
    ax, ay = np.abs(a[:, 0]), np.abs(a[:, 1]); both = (ax > 1e-3) & (ay > 1e-3)
    out["corner"] = float(np.mean((ax[both] >= 0.99) & (ay[both] >= 0.99))) if both.any() else float("nan")
    z = a[:, 2]; nzz = np.where(np.abs(z) > 1e-3)[0]
    iso = sum(1 for t in nzz if (t == 0 or abs(z[t - 1]) < 1e-3) and (t == len(z) - 1 or abs(z[t + 1]) < 1e-3))
    out["az_isolated"] = (iso, len(nzz)); out["frames"] = len(a)
    return out


def run_v3(fs):
    """v3 공식 지표. 반환: 합격 여부."""
    raws, execs, dts, rows3 = [], [], [], []
    for p in fs:
        with h5py.File(p, "r") as h:
            r = h["actions_raw"][:].astype(float); e = h["actions_exec"][:].astype(float)
            t = h["frame_timestamps"][:]; ev = h["action_event_types"][:] if "action_event_types" in h else None
        m = np.array([x == b"joystick" for x in ev]) if ev is not None else np.ones(len(r), bool)
        raws.append(r[m]); execs.append(e[m]); dts.append(np.diff(t))
        mag = np.hypot(r[m][:, 0], r[m][:, 1]); nz = mag > 1e-3
        rows3.append((p.split("/")[-1], len(r[m]), float(np.mean((mag[nz] > .15) & (mag[nz] < .9))) if nz.any() else float("nan")))
    R = np.concatenate(raws); E = np.concatenate(execs)
    mag = np.hypot(R[:, 0], R[:, 1]); nz = mag > 1e-3
    mid_mag = float(np.mean((mag[nz] > .15) & (mag[nz] < .9)))
    ax, ay = np.abs(R[:, 0]), np.abs(R[:, 1]); both = (ax > 1e-3) & (ay > 1e-3)
    corner = float(np.mean((ax[both] >= .99) & (ay[both] >= .99))) if both.any() else 0.0
    iso = tot = 0
    for r in raws:
        z = r[:, 2]; idx = np.where(np.abs(z) > 1e-3)[0]; tot += len(idx)
        iso += sum(1 for t in idx if (t == 0 or abs(z[t - 1]) < 1e-3) and (t == len(z) - 1 or abs(z[t + 1]) < 1e-3))
    spike = iso / tot if tot else 0.0
    print(f"[v3] 에피소드 {len(fs)}개, joystick 프레임 {len(R)}개, 프레임 간격 중앙값 {np.median(np.concatenate(dts)):.3f}s")
    for i, n in enumerate(("lx", "ly", "az")):
        v = np.abs(R[:, i]); z = v[v > 1e-3]
        print(f"  raw {n}: 비영 {len(z):5d} | 포화 {np.mean(z >= .99):.0%} | 중간값(0.15~0.9) {np.mean((z > .15) & (z < .9)):.0%}")
    print(f"  (lx,ly) 크기 중간값(0.15~0.9): {mid_mag:.0%}   코너: {corner:.0%}   az 단독 스파이크: {iso}/{tot} ({spike:.0%})")
    em = np.hypot(E[:, 0], E[:, 1]); mv = em > 1e-3
    lv = np.round(em[mv], 2)
    u, c = np.unique(lv, return_counts=True)
    print("  exec 이동 크기 분포(속력 단계):", {float(a): f"{b / mv.sum():.0%}" for a, b in zip(u, c)})
    ang = np.degrees(np.arctan2(E[mv, 1], E[mv, 0]))
    print(f"  exec 이동 각도: 고유값 {len(np.unique(np.round(ang, 0)))}개 (1°단위), 표준편차 {ang.std():.1f}°")
    rm = np.hypot(R[:, 0], R[:, 1]); ang_r = np.degrees(np.arctan2(R[nz, 1], R[nz, 0]))
    print(f"  raw 이동 각도: 고유값 {len(np.unique(np.round(ang_r, 0)))}개 (1°단위)")
    ok = mid_mag >= .5 and corner <= .2 and spike <= .25
    print(f"\n공식 판정(중간값(크기)≥50% · 코너≤20% · az스파이크≤25%): {'합격' if ok else '미달'}")
    if per_ep:
        print("\n[에피소드별] 이름 | 프레임 | 크기 중간값")
        for n_, f_, m_ in rows3: print(f"  {n_[:60]:60s} {f_:4d} {m_:.0%}")
    return ok


with h5py.File(fs[0], "r") as _h:
    _v3 = "actions_raw" in _h and "actions_exec" in _h
if _v3 and all(("actions_raw" in h5py.File(f, "r")) for f in fs):
    sys.exit(0 if run_v3(fs) else 1)

allA, rows = [], []
for p in fs:
    with h5py.File(p, "r") as h:
        a = h["actions"][:].astype(float); ev = h["action_event_types"][:] if "action_event_types" in h else None
    if ev is not None:
        a = a[np.array([e == b"joystick" for e in ev])]
    allA.append(a); rows.append((p.split("/")[-1], stats(a)))
A = np.concatenate(allA); S = stats(A)
print(f"에피소드 {len(fs)}개, joystick 프레임 {len(A)}개")
for n in ("lin_x", "lin_y", "ang_z"):
    s = S[n]; print(f"  {n}: 비영 {s['nz']:5d} | 포화 {s['sat']:.0%} | 중간값(0.15~0.9) {s['mid']:.0%}")
print(f"  코너(lx,ly 동시 포화) 비율: {S['corner']:.0%}   ang_z 단독 스파이크: {S['az_isolated'][0]}/{S['az_isolated'][1]}")
mids = []
for a in allA:
    n = len(a)
    for t in range(n):
        v = np.abs(a[t, :2])
        if ((v > 0.15) & (v < 0.9)).any(): mids.append(t / max(1, n - 1))
if mids: print("  중간값 프레임 위치분포(0-.1/.1-.3/.3-.7/.7-.9/.9-1):", np.histogram(mids, bins=[0, .1, .3, .7, .9, 1.01])[0].tolist())
lx_mid, ly_mid = S["lin_x"]["mid"], S["lin_y"]["mid"]
ok = (lx_mid >= 0.5 or ly_mid >= 0.5) and S["corner"] <= 0.2
print("\n파일럿 합격 기준 예시: 중간값 비율 ≥50% (lx 또는 ly) 이고 코너 비율 ≤20%  →", "합격" if ok else "미달")
if per_ep:
    print("\n[에피소드별] 이름 | 프레임 | lx중간 | ly중간 | 코너")
    for name, s in rows:
        print(f"  {name[:60]:60s} {s['frames']:4d} {s['lin_x']['mid']:.0%} {s['lin_y']['mid']:.0%} {s['corner']:.0%}")

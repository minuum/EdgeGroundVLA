#!/usr/bin/env python3
"""연속 조이스틱 수집 데이터(egvla2_continuous_v1) 품질 점검 — 로봇 서버/학습 서버 어디서나 실행 가능.

연속 액션 학습(FM 등)에 쓸모 있는 데이터인지 보는 지표:
  - 중간값 비율: 비영 프레임 중 0.15<|v|<0.9 (스틱을 끝까지 안 민 프레임)  → 높을수록 좋음
  - 포화 비율: 비영 프레임 중 |v|>=0.99
  - 코너 비율: lx,ly 둘 다 비영인 프레임 중 둘 다 포화(45° 대각선 고정)
  - ang_z 단독 스파이크: 앞뒤 프레임이 0인 1프레임짜리 비영 ang_z (오입력 의심)
  - 에피소드 상대위치별 중간값 분포
사용: python3 scripts/check_continuous_collection_quality.py [dir=ROS_action/mobile_vla_dataset_egvla2] [--per-episode]
"""
import glob
import sys

import h5py
import numpy as np

d = next((a for a in sys.argv[1:] if not a.startswith("--")), "ROS_action/mobile_vla_dataset_egvla2")
per_ep = "--per-episode" in sys.argv
fs = sorted(glob.glob(f"{d}/*.h5"))
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

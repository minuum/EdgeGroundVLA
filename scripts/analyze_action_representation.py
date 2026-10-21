#!/usr/bin/env python3
"""이산(8-class) vs 연속 액션 표현의 데이터 해부(CPU): 사람 조작 방향의 분포, 8방향 양자화 오차와 경계 근접도, 정보량(엔트로피), 속력(크기) 분포와 고정 속력 가정의 오차, 에피소드별 계단(클래스 전환) 빈도.
대상: 본학습 세트 300건의 actions_exec(구동값: 속력 단계 × 방향)와 actions_raw(스틱 원값). 출력: docs/v5/egvla2_fm/action_repr_stats.json
실행: .venv/bin/python3 -I scripts/analyze_action_representation.py"""
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "ROS_action/mobile_vla_dataset_egvla2"
meta = json.load(open(ROOT / "docs/v5/egvla2_fm/meta/egvla2_v3_main.json"))["files"]
GO = 0.2
BIN_ANG = np.array([0, 45, -45, 90, -90])            # FORWARD, FWD+L, FWD+R, LEFT, RIGHT (전진=0°, 좌=+)


def ang(v):
    return np.degrees(np.arctan2(v[..., 1], v[..., 0]))


def snap(a):
    d = np.abs(((a[:, None] - BIN_ANG[None]) + 180) % 360 - 180)
    return BIN_ANG[d.argmin(1)], d.min(1)


ex_all, raw_all, eps = [], [], []
for d in meta:
    with h5py.File(DATA / d["name"], "r") as f:
        ex = np.array(f["actions_exec"], float); raw = np.array(f["actions_raw"], float)
    ex_all.append(ex); raw_all.append(raw); eps.append(d["name"])
EX = np.concatenate(ex_all); RAW = np.concatenate(raw_all)
mv = np.linalg.norm(EX[:, :2], axis=1) >= GO
a = ang(EX[mv, :2]); sn, qerr = snap(a)
out = dict(frames=int(len(EX)), moving=int(mv.sum()))
# 1) 방향 분포(10° 구간)와 8방향 점유
h, edges = np.histogram(a, bins=np.arange(-180, 181, 10)); out["angle_hist10"] = dict(edges=edges.tolist(), counts=h.tolist())
out["class_share"] = {str(k): float((sn == k).mean()) for k in BIN_ANG}
# 2) 양자화 오차
out["quant_err"] = dict(mean=float(qerr.mean()), median=float(np.median(qerr)), p90=float(np.percentile(qerr, 90)), hist=np.histogram(qerr, bins=[0, 2.5, 5, 10, 15, 22.6])[0].tolist())
bd = np.min(np.abs(((a[:, None] - np.array([22.5, 67.5, -22.5, -67.5])[None]) + 180) % 360 - 180), axis=1)
out["boundary_near"] = {f"within_{w}deg": float((bd < w).mean()) for w in (2, 5, 10)}
# 3) 정보량(엔트로피, bit)
def ent(x):
    p = x[x > 0] / x.sum(); return float(-(p * np.log2(p)).sum())
out["entropy_bits"] = dict(angle_5deg=ent(np.histogram(a, bins=np.arange(-180, 181, 5))[0].astype(float)), angle_1deg=ent(np.histogram(a, bins=np.arange(-180, 181, 1))[0].astype(float)),
                          class8=ent(np.array([(sn == k).sum() for k in BIN_ANG], float)))
# 4) 속력: raw 크기 분포, exec 단계, 고정 0.7 가정 오차
mr = np.linalg.norm(RAW[mv, :2], axis=1); me = np.linalg.norm(EX[mv, :2], axis=1)
out["speed"] = dict(raw_mag_hist=np.histogram(mr, bins=np.arange(0, 1.61, 0.1))[0].tolist(), exec_levels={str(round(k, 2)): float((np.abs(me - k) < 0.03).mean()) for k in (0.4, 0.7, 1.0)},
                    rel_err_fixed07=dict(mean=float(np.mean(np.abs(me - 0.7) / me)), median=float(np.median(np.abs(me - 0.7) / me))), speed_cmps_fixed=26.4 * 0 + (6.59 + 19.85 * 0.7))
# 5) 에피소드별 계단: 클래스 전환 빈도 vs 연속 방향 변화량
sw, tv = [], []
for ex in ex_all:
    m = np.linalg.norm(ex[:, :2], axis=1) >= GO; aa = ang(ex[:, :2]); s8, _ = snap(aa)
    idx = np.where(m)[0]
    if len(idx) > 3:
        sw.append(float((np.diff(s8[idx]) != 0).mean())); d_ = np.abs(((np.diff(aa[idx]) + 180) % 360) - 180); tv.append(float(d_.mean()))
out["staircase"] = dict(class_switch_per_frame=float(np.mean(sw)), mean_abs_angle_step_deg=float(np.mean(tv)))
# 6) 예시 에피소드(중앙 좌커브): 연속 각도 vs 양자화 각도
cand = [i for i, d in enumerate(meta) if d["position"] == "center" and d["path"] == "left_curve"]
for key, i in (("left_curve", cand[3]), ("straight", [j for j, d in enumerate(meta) if d["position"] == "center" and d["path"] == "straight"][3])):
    ex = ex_all[i]; m = np.linalg.norm(ex[:, :2], axis=1) >= GO; aa = ang(ex[:, :2]); s8, _ = snap(aa)
    out[f"example_{key}"] = dict(name=eps[i], angle=[round(float(x), 1) if mm else None for x, mm in zip(aa, m)], snapped=[int(x) if mm else None for x, mm in zip(s8, m)])
(ROOT / "docs/v5/egvla2_fm/action_repr_stats.json").write_text(json.dumps(out, ensure_ascii=False))
print(json.dumps({k: v for k, v in out.items() if not k.startswith("example") and k not in ("angle_hist10",)}, ensure_ascii=False)[:1500])

#!/usr/bin/env python3
"""방향 오차를 '성공 기준'에 가깝게 보는 보조 지표(CPU, 오프라인): 사람 데모가 서로 다른 정당한 선택을 한다는 점을 반영.
A) 다중 정답 방향 오차: 모델 방향과 '같은 셀(목표 위치+경로)의 다른 에피소드가 같은 진행도(±4%)에서 낸 방향들' 중 가장 가까운 것과의 각도(min-over-references). 사람 자신의 방향도 같은 방식으로 재서 기준선으로.
B) 종점 허용 범위: 모델/사람 종점(속력 보정으로 적분)이 같은 셀의 다른 사람 종점 구름에서 얼마나 떨어져 있나(최근접 거리).
C) 사람 데모의 종점 이미지 허용치: 마지막 프레임 bbox 가로 위치·크기의 분포(완전한 중앙 도착이 아니어도 수집에 포함된 범위).
입력: error_anatomy.json(fold0·LOPO strong_right 안 본 에피소드의 사람·모델 프레임별 값). 출력: docs/v5/egvla2_fm/outcome_metrics.json"""
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "ROS_action/mobile_vla_dataset_egvla2"
meta = json.load(open(ROOT / "docs/v5/egvla2_fm/meta/egvla2_v3_main.json"))["files"]
anat = json.load(open(ROOT / "docs/v5/egvla2_fm/error_anatomy.json")); cache = np.load(ROOT / "docs/v5/egvla2_fm/v3_florence2_cache.npz")
GO, WIN = 0.2, 0.04
VA, VB = 6.59, 19.85
ang = lambda v: np.degrees(np.arctan2(v[..., 1], v[..., 0]))
dd = lambda a, b: np.abs((a - b + 180) % 360 - 180)
H = {}
for d in meta:
    with h5py.File(DATA / d["name"], "r") as f:
        ex = np.array(f["actions_exec"], float)[:, :2]; ts = np.array(f["frame_timestamps"], float)
    n = len(ex); dt = np.clip(np.diff(ts, append=ts[-1]), 0, 1.0); dt[-1] = np.median(dt[:-1])
    H[d["name"]] = dict(cell=f'{d["position"]}|{d["path"]}', ex=ex, prog=np.arange(n) / max(n - 1, 1), dt=dt, mv=np.linalg.norm(ex, axis=1) >= GO)


def path_end(vel, dt):
    return np.cumsum(vel * dt[:, None], 0)[-1]


def human_vel(ex):
    m = np.linalg.norm(ex, axis=1, keepdims=True)
    return np.where(m > 1e-6, ex / np.maximum(m, 1e-9) * (VA + VB * m), 0)


ENDS = {nm: path_end(human_vel(h["ex"]), h["dt"]) for nm, h in H.items()}
out = {}
for sp, items in anat["splits"].items():
    me_m, me_h, one_m, one_h, dm, dh, ends_m = [], [], [], [], [], [], []
    for it in items:
        nm = it["name"]; h = H[nm]; refs = [o for k, o in H.items() if o["cell"] == h["cell"] and k != nm]
        P = np.array(it["pred"]); Hh = np.array(it["human"]); n = it["n"]
        for t in range(n):
            if np.linalg.norm(P[t]) < GO or np.linalg.norm(Hh[t]) < GO:
                continue
            ra = np.concatenate([ang(o["ex"][o["mv"] & (np.abs(o["prog"] - h["prog"][t]) <= WIN)]) for o in refs])
            if len(ra) < 3:
                continue
            me_m.append(float(dd(ang(P[t]), ra).min())); me_h.append(float(dd(ang(Hh[t]), ra).min())); one_m.append(float(dd(ang(P[t]), ang(Hh[t])))); one_h.append(0.0)
        pm = np.linalg.norm(P, axis=1, keepdims=True); vp = np.where(pm >= GO, P / np.maximum(pm, 1e-9) * (VA + VB * 0.7), 0); em = path_end(vp, np.array(it["dt"]))
        oth = np.array([ENDS[k] for k, o in H.items() if o["cell"] == h["cell"] and k != nm])
        dm.append(float(np.linalg.norm(oth - em, axis=1).min())); dh.append(float(np.linalg.norm(oth - ENDS[nm], axis=1).min())); ends_m.append((h["cell"], em.tolist(), ENDS[nm].tolist()))
    q = lambda x: dict(mean=float(np.mean(x)), median=float(np.median(x)), p90=float(np.percentile(x, 90)), le10=float((np.array(x) <= 10).mean()), le15=float((np.array(x) <= 15).mean()))
    out[sp] = dict(multi_ref_model=q(me_m), multi_ref_human=q(me_h), single_ref_model=q(one_m), endpoint_nn_model=q(dm), endpoint_nn_human=q(dh),
                   frac_model_end_within_human_p90=float((np.array(dm) <= np.percentile(dh, 90)).mean()), n_frames=len(me_m), n_eps=len(items), ends=ends_m,
                   hist=dict(model=np.histogram(me_m, bins=[0, 5, 10, 15, 20, 30, 45, 181])[0].tolist(), human=np.histogram(me_h, bins=[0, 5, 10, 15, 20, 30, 45, 181])[0].tolist()))
fin = np.array([[cache[d["name"] + "|bbox"][-1][0], cache[d["name"] + "|bbox"][-1][2], cache[d["name"] + "|bbox"][-1][3]] for d in meta])
ok = fin[:, 2] >= 0.5
out["final_frame"] = dict(n=int(len(fin)), detected=float(ok.mean()), cx_mean=float(fin[ok, 0].mean()), cx_std=float(fin[ok, 0].std()), cx_p05=float(np.percentile(fin[ok, 0], 5)), cx_p95=float(np.percentile(fin[ok, 0], 95)),
                          area_mean=float(fin[ok, 1].mean()), area_p05=float(np.percentile(fin[ok, 1], 5)), area_p95=float(np.percentile(fin[ok, 1], 95)), within_cx_0p1=float((np.abs(fin[ok, 0] - 0.5) <= 0.1).mean()), within_cx_0p2=float((np.abs(fin[ok, 0] - 0.5) <= 0.2).mean()))
cells = {}
for nm, e in ENDS.items():
    cells.setdefault(H[nm]["cell"], []).append(e)
out["cell_ends"] = {c: np.array(v).round(1).tolist() for c, v in cells.items()}
out["cell_end_spread_cm"] = {c: dict(std=float(np.std(np.array(v), axis=0).mean()), n=len(v)) for c, v in cells.items()}
(ROOT / "docs/v5/egvla2_fm/outcome_metrics.json").write_text(json.dumps(out, ensure_ascii=False))
for sp in ("fold0", "lopo_strong_right"):
    o = out[sp]; print(sp, "단일 정답 오차 %.1f° → 다중 정답 %.1f° (사람 자신 %.1f°)" % (o["single_ref_model"]["mean"], o["multi_ref_model"]["mean"], o["multi_ref_human"]["mean"]),
                       "| ≤10°: 모델 %.0f%% 사람 %.0f%%" % (100 * o["multi_ref_model"]["le10"], 100 * o["multi_ref_human"]["le10"]),
                       "| 종점 최근접 거리 모델 %.0fcm 사람 %.0fcm, 사람 p90 이내 %.0f%%" % (o["endpoint_nn_model"]["median"], o["endpoint_nn_human"]["median"], 100 * o["frac_model_end_within_human_p90"]))
print("마지막 프레임 bbox", {k: round(v, 2) for k, v in out["final_frame"].items()})

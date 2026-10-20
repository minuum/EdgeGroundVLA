#!/usr/bin/env python3
"""사람 조작 일치도 천장: 같은 상황에서 사람끼리 방향이 몇 도 다른가. 모델 방향 오차(약 10°)가 높은지 낮은지 판단하는 기준.
각 에피소드의 이동 프레임(exec 크기>0)에 대해 진행도 p=t/n ±WIN 안의 '다른 에피소드' 프레임들과 비교:
  A) 같은 셀(목표 위치+경로)의 다른 에피소드, B) 같은 목표 위치(경로 무관), C) 전체.
지표: (1) 쌍별 평균 각도 차(사람 vs 다른 사람 한 프레임), (2) 합의 방향(다른 에피소드들의 평균 단위벡터)과의 각도 = '진행도+셀을 아는 오라클 예측기'.
실행: .venv/bin/python3 -I scripts/eval_human_ceiling.py   출력: docs/v5/egvla2_fm/human_ceiling.json
"""
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "ROS_action/mobile_vla_dataset_egvla2"
meta = json.load(open(ROOT / "docs/v5/egvla2_fm/meta/egvla2_v3_main.json"))["files"]
WIN = 0.04
E = []
for d in meta:
    with h5py.File(DATA / d["name"], "r") as f:
        a = np.array(f["actions_exec"], float)[:, :2]
    n = len(a); E.append(dict(pos=d["position"], path=d["path"], a=a, p=np.arange(n) / max(n - 1, 1), mv=np.linalg.norm(a, axis=1) > 0.2))
ang = lambda u, v: np.degrees(np.arccos(np.clip(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))


def run(sel):
    pair, cons = [], []
    for i, e in enumerate(E):
        others = [o for j, o in enumerate(E) if j != i and sel(e, o)]
        if not others:
            continue
        for t in np.where(e["mv"])[0]:
            vs = []
            for o in others:
                m = o["mv"] & (np.abs(o["p"] - e["p"][t]) <= WIN)
                if m.any():
                    vs.append(o["a"][m])
            if len(vs) < 3:
                continue
            V = np.concatenate(vs); U = V / np.linalg.norm(V, axis=1, keepdims=True)
            pair.append(float(np.mean([ang(e["a"][t], v) for v in V[:: max(1, len(V) // 40)]])))
            c = U.mean(0)
            if np.linalg.norm(c) > 1e-6:
                cons.append(float(ang(e["a"][t], c)))
    return dict(pairwise_mean=float(np.mean(pair)), pairwise_median=float(np.median(pair)), consensus_mean=float(np.mean(cons)), consensus_median=float(np.median(cons)),
                consensus_p90=float(np.percentile(cons, 90)), n=len(cons))


res = {"같은 셀(목표 위치+경로)": run(lambda e, o: e["pos"] == o["pos"] and e["path"] == o["path"]),
       "같은 목표 위치(경로 무관)": run(lambda e, o: e["pos"] == o["pos"]), "전체": run(lambda e, o: True)}
for k, v in res.items():
    print(k, {a: round(b, 1) if isinstance(b, float) else b for a, b in v.items()})
(ROOT / "docs/v5/egvla2_fm/human_ceiling.json").write_text(json.dumps(dict(win=WIN, results=res), indent=1, ensure_ascii=False))

#!/usr/bin/env python3
"""bbox·특징·액션·영상 순서(정렬) 점검.
A1(학습): 캐시 bbox/특징[t]가 H5 images[t]에서 다시 계산한 값과 맞는지, 오프셋(t±1,2)이 더 잘 맞지 않는지.
A2(학습): 영상 회전량 dx(t→t+1)와 사람 az[t+k]의 상관이 가장 큰 지연 k (액션 라벨이 영상보다 앞/뒤인지).
A3(실기 SYNC): 스텝 i의 회전 명령과 영상 dx(i→i+1)·bbox 기록 순서.
실행: HF_HUB_OFFLINE=1 .venv/bin/python3 scripts/analyze_alignment.py   출력: docs/v5/egvla2_fm/alignment_check.json
"""
import glob
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
DATA = ROOT / "ROS_action/mobile_vla_dataset_egvla2"
CACHE = ROOT / "docs/v5/egvla2_fm/v3_florence2_cache.npz"
PART = sys.argv[1] if len(sys.argv) > 1 else "a1"


def gray(img):
    g = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2GRAY)
    return np.float32(cv2.resize(g, (320, 180), interpolation=cv2.INTER_AREA)[:110])


WIN = cv2.createHanningWindow((320, 110), cv2.CV_32F)


def dx_of(a, b):
    (dx, _), r = cv2.phaseCorrelate(a, b, WIN); return float(dx), float(r)


def main():
    from robovlm_nav.serve.egvla2_fm_policy import FusedF2Extractor
    cache = np.load(CACHE); out = {}
    rng = np.random.default_rng(0)
    if PART == "a1":
        names = sorted({k.split("|")[0] for k in cache.files if k.endswith("|bbox")})
        eps = [dict(name=n, bbox=cache[f"{n}|bbox"]) for n in names if (DATA / n).exists()]
    else:
        import train_egvla2_s1 as S1
        eps = S1.load_main()
        DATA_ = S1.T.DATA
    # A1: 캐시 정렬
    ex = FusedF2Extractor(dtype=torch.float16) if PART == "a1" else None
    if PART == "a1":
        best = []; sims = {k: [] for k in (-2, -1, 0, 1, 2)}; bb_err = {k: [] for k in (-2, -1, 0, 1, 2)}
        for i in rng.choice(len(eps), 14, replace=False):
            e = eps[i]; n = len(e["bbox"]); cb = cache[f"{e['name']}|bbox"]; cf = cache[f"{e['name']}|feat"]
            with h5py.File(DATA / e["name"], "r") as f:
                for t in rng.choice(np.arange(3, n - 3), 3, replace=False):
                    b, ft = ex(np.ascontiguousarray(f["images"][t][:, :, ::-1]))
                    errs = {}
                    for k in sims:
                        c = cf[t + k]; sims[k].append(float(np.dot(ft, c) / (np.linalg.norm(ft) * np.linalg.norm(c) + 1e-9)))
                        bb_err[k].append(float(np.abs(b[:3] - cb[t + k][:3]).sum())); errs[k] = bb_err[k][-1]
                    best.append(min(errs, key=errs.get))
        out["A1_cache_alignment"] = dict(n=len(best), best_offset_counts={int(k): int(best.count(k)) for k in sims},
                                         feat_cos_by_offset={int(k): round(float(np.mean(v)), 5) for k, v in sims.items()}, bbox_abs_err_by_offset={int(k): round(float(np.mean(v)), 4) for k, v in bb_err.items()})
    else:
        # A2: 학습 액션-영상 지연
        pairs = []
        for i in rng.choice(len(eps), 70, replace=False):
            e = eps[i]; a = e["chunk"][:, 0]
            with h5py.File(DATA / e["name"], "r") as f:
                im = f["images"]; n = im.shape[0]; g = [gray(im[t][:, :, ::-1]) for t in range(n)]
            dx = np.array([dx_of(g[t], g[t + 1])[0] for t in range(n - 1)])
            for t in range(n - 1):
                pairs.append((i, t, dx[t]))
            e["_dx"] = dx
        lag = {}
        for k in range(-3, 4):
            xs, ys = [], []
            for i in {p[0] for p in pairs}:
                e = eps[i]; a = e["chunk"][:, 0, 2]; dx = e["_dx"]
                for t in range(max(0, -k), min(len(dx), len(a) - k)):
                    xs.append(a[t + k]); ys.append(dx[t])
            lag[k] = round(float(np.corrcoef(xs, ys)[0, 1]), 4)
        out["A2_train_action_vs_image_motion_corr_by_lag"] = lag      # corr(az[t+k], dx(t→t+1))
        # A3: 실기
        xs0, ys0, lagl = [], [], {k: ([], []) for k in (-1, 0, 1)}
        for rp in sorted(glob.glob(str(ROOT / "ROS_action/dryrun_closedloop_20261022/reports/*.json"))):
            d = json.load(open(rp))
            if d.get("instruction_mode") != "SYNC" or "egvla2" not in d["model_name"]:
                continue
            with h5py.File(ROOT / f"ROS_action/dryrun_closedloop_20261022/h5/session_{d['session_id']}.h5", "r") as f:
                im = f["observations/images"]; n = min(im.shape[0], len(d["history"])); g = [gray(im[t]) for t in range(n)]
                ha = f["actions"][:n]
            dx = [dx_of(g[t], g[t + 1])[0] for t in range(n - 1)]
            for k in lagl:
                for t in range(max(0, -k), min(n - 1, n - 1 - k)):
                    lagl[k][0].append(d["history"][t + k]["action"][2]); lagl[k][1].append(dx[t])
            # H5 actions vs 리포트 history action 일치
            out.setdefault("A3_h5_vs_report_action_maxdiff", []).append(float(np.abs(ha - np.array([h["action"] for h in d["history"][:n]])).max()))
        out["A3_live_action_vs_image_motion_corr_by_lag"] = {k: round(float(np.corrcoef(*lagl[k])[0, 1]), 4) for k in lagl}     # corr(az[t+k], dx(t→t+1))
        out["A3_h5_vs_report_action_maxdiff"] = float(max(out["A3_h5_vs_report_action_maxdiff"]))
    (ROOT / f"docs/v5/egvla2_fm/alignment_check_{PART}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False)); print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

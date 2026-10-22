#!/usr/bin/env python3
"""서빙 샘플링 조합 평가: 기준 FM-π0(배포 구성)을 분할마다 한 번 학습하고 흐름 시작 노이즈 σ × 샘플 수 K × Euler 스텝을 같은 모델로 평가(재학습 없음).
σ=0이면 샘플이 모두 같아 K=1만 평가. 분할 10개, seed 0. 출력: docs/v5/egvla2_fm/sampling_grid.json (분할 단위 저장·재개).
실행: .venv_lerobot/bin/python3 scripts/eval_sampling_grid.py  (SMOKE=1: epoch 2·분할 1개)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_bold_ablation as BA  # noqa: E402
import eval_runtime_ablation as RA  # noqa: E402
import eval_subgoal_ablation as AB  # noqa: E402
import train_egvla2_s1 as S1  # noqa: E402
T = S1.T
SMOKE = bool(os.getenv("SMOKE"))
OUT = ROOT / "docs/v5/egvla2_fm" / ("sampling_grid_smoke.json" if SMOKE else "sampling_grid.json")
SIGMAS, KS, STEPS = (0.0, 0.25, 0.5, 1.0), (1, 4, 8, 16), (3, 5, 10)


def main():
    eps = S1.load_main(); cache = np.load(T.CACHE); E = AB.prep(eps, cache, 3); folds = S1.make_folds(eps)
    splits = {f"fold{i}": folds[i] for i in range(len(folds))}
    for pos in ("center", "weak_left", "strong_left", "weak_right", "strong_right"):
        splits[f"lopo_{pos}"] = [i for i, e in enumerate(eps) if e["cell"].startswith(pos + "|")]
    order = [x for x in os.getenv("SPLITS", "").split(",") if x in splits]
    if order:
        splits = {x: splits[x] for x in order}
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, te_idx in splits.items():
        if name in res:
            continue
        tr_idx = [i for i in range(len(E)) if i not in te_idx]
        tr, te = T.to_t(E, tr_idx), T.to_t(E, te_idx)
        gt = te[2][..., :2].cpu().numpy(); g_first, g_burst = gt[:, 0], gt[:, :3].mean(1)
        m, H = BA.fit(tr, "base")
        r = {}
        for sg in SIGMAS:
            for st in STEPS:
                S16 = BA.draw_sigma(m, te[0], te[1], 1 if sg == 0 else 16, st, sg, H)
                for k in ([1] if sg == 0 else KS):
                    p = S16[:, :k].mean(1)
                    r[f"s{sg}|K{k}|e{st}"] = dict(first=RA.angle(p[:, 0], g_first), n1_vs_burst=RA.angle(p[:, 0], g_burst))
        res[name] = r
        OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
        print(name, {k: round(v["first"], 2) for k, v in r.items() if k.endswith("e10") and k.startswith(("s0.0", "s0.5", "s1.0"))}, flush=True)
        if SMOKE:
            break


if __name__ == "__main__":
    main()

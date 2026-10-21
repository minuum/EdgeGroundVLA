#!/usr/bin/env python3
"""데이터량 곡선: FM-π0(서브골 드롭아웃30%)을 학습 데이터 크기·다양성을 바꿔 학습해 안 본 에피소드/안 본 목표 위치의 방향 오차를 본다. 최적화 스텝 수는 맞춤(epoch = 120/비율).
실험 1(밀도, 같은 목표 위치): 학습 에피소드의 비율 {0.1,0.2,0.4,0.7,1.0}(5-fold, seed 0·1). 실험 2(다양성, 안 본 목표 위치): 같은 에피소드 수(60개)를 k개 목표 위치에 나눠 뽑아(k=1~4) 남은 위치로 평가.
출력: docs/v5/egvla2_fm/data_scaling.json (실행 중 계속 저장). 실행: .venv_lerobot/bin/python3 scripts/eval_data_scaling.py"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB  # noqa: E402
import train_egvla2_s1 as S1  # noqa: E402
T = S1.T
OUT = ROOT / "docs/v5/egvla2_fm/data_scaling.json"


def run(E, tr_idx, te_idx, epochs, seed):
    tr, te = T.to_t(E, tr_idx), T.to_t(E, te_idx)
    m = AB.fit(S1.ARCHS["C_pi0_adaln"], tr, epochs, seed, drop=0.3); r = S1.direction_metrics(m, te)
    return dict(angle=r["angle_deg"], go_f1=r["go_f1"], n_ep=len(tr_idx), frames=int(len(tr[0])))


def main():
    eps = S1.load_main(); cache = np.load(T.CACHE); E = AB.prep(eps, cache, 3); folds = S1.make_folds(eps)
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    save = lambda: OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    cells = {}
    for i, e in enumerate(eps):
        cells.setdefault(e["cell"], []).append(i)
    # 실험 1: 밀도
    for seed in (0, 1):
        for fi, te_idx in enumerate(folds):
            tr_all = [i for i in range(len(E)) if i not in te_idx]
            for frac in (0.1, 0.2, 0.4, 0.7, 1.0):
                key = f"density|f{frac}|fold{fi}|seed{seed}"
                if key in res:
                    continue
                rng = np.random.RandomState(100 * seed + fi); sel = []
                for c, idx in cells.items():                      # 셀(목표 위치×경로)별로 같은 비율 → 구성 유지
                    pool = [i for i in idx if i in set(tr_all)]; k = max(1, int(round(frac * len(pool)))); sel += list(rng.choice(pool, k, replace=False))
                res[key] = run(E, sel, te_idx, int(round(120 / frac)), seed); save(); print(key, round(res[key]["angle"], 2), res[key]["n_ep"], flush=True)
    # 실험 2: 다양성(같은 에피소드 수 60)
    pos = sorted({e["cell"].split("|")[0] for e in eps}); byp = {p: [i for i, e in enumerate(eps) if e["cell"].startswith(p + "|")] for p in pos}
    N = 60
    for te_p in pos:
        others = [p for p in pos if p != te_p]
        for k in (1, 2, 3, 4):
            combos = list(itertools.combinations(others, k)); rng = np.random.RandomState(7)
            if len(combos) > 3:
                combos = [combos[j] for j in rng.choice(len(combos), 3, replace=False)] if k < 4 else combos
            for co in combos:
                key = f"diversity|{te_p}|k{k}|{'+'.join(co)}"
                if key in res:
                    continue
                sel = []
                for p in co:                                          # 위치당 N/k개(셀 균형은 무작위)
                    sel += list(rng.choice(byp[p], N // k, replace=False))
                res[key] = run(E, sel, byp[te_p], int(round(120 * 240 / len(sel))), 0); save(); print(key, round(res[key]["angle"], 2), flush=True)


if __name__ == "__main__":
    main()

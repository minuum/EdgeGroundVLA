#!/usr/bin/env python3
"""좌우 반전 증강 시험: 학습 에피소드를 수평 반전(이미지 반전 → Florence-2 특징·bbox 재계산 캐시, 라벨 ly·az 부호 반전, 목표 위치·경로 좌우 교환)해 학습 데이터를 2배로.
평가는 항상 실제(반전 안 한) hold-out 에피소드. 반전본은 학습 에피소드에서만 만든다(hold-out 에피소드의 반전본은 제외 → 누설 없음).
분할: LOPO 5개 목표 위치 + 5-fold fold0. 헤드: FM-π0(서브골 드롭아웃30%, 120 epoch), seed 0,1. 기준선(반전 없음)과 같은 seed로 쌍 비교.
실행: .venv_lerobot/bin/python3 scripts/eval_flip_aug.py   (선행: FLIP=1 precompute로 v3_florence2_cache_flip.npz)   출력: docs/v5/egvla2_fm/flip_aug.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB  # noqa: E402
import train_egvla2_s1 as S1  # noqa: E402
T = S1.T
FLIPC = ROOT / "docs/v5/egvla2_fm/v3_florence2_cache_flip.npz"
OUT = ROOT / "docs/v5/egvla2_fm/flip_aug.json"
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1").split(",")]
MIRP = {"weak_left": "weak_right", "weak_right": "weak_left", "strong_left": "strong_right", "strong_right": "strong_left", "center": "center"}
MIRR = {"left_curve": "right_curve", "right_curve": "left_curve", "straight": "straight"}


def flipped(eps, fc):
    out = []
    for e in eps:
        raw = fc[f"{e['name']}|bbox"].astype(np.float32); ch = e["chunk"].copy(); ch[..., 1] *= -1; ch[..., 2] *= -1
        p, q = e["cell"].split("|")
        out.append(dict(name=e["name"] + "#flip", cell=f"{MIRP.get(p, p)}|{MIRR.get(q, q)}", bbox=AB.window(raw, 3), feat=fc[f"{e['name']}|feat"].astype(np.float32), chunk=ch))
    return out


def evaluate(m, te):
    r = S1.direction_metrics(m, te); Sm = T.sample(m, te[0], te[1]).mean(1)
    r["mse_xy"] = float(((Sm[..., :2] - te[2][..., :2]) ** 2).mean().item()); return r


def main():
    eps = S1.load_main(); cache = np.load(T.CACHE); fc = np.load(FLIPC)
    E = AB.prep(eps, cache, 3); F = flipped(E, fc); folds = S1.make_folds(eps)
    pos = sorted({e["cell"].split("|")[0] for e in eps})
    splits = {"fold0": folds[0]}; splits.update({f"lopo_{p}": [i for i, e in enumerate(eps) if e["cell"].startswith(p + "|")] for p in pos})
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for seed in SEEDS:
        for sname, te_idx in splits.items():
            key = f"{sname}|seed{seed}"
            if key in res:
                continue
            tr_idx = [i for i in range(len(E)) if i not in te_idx]
            te = T.to_t(E, te_idx); base_tr = T.to_t(E, tr_idx)
            aug = E + F; aug_idx = tr_idx + [len(E) + i for i in tr_idx]
            aug_tr = T.to_t(aug, aug_idx)
            r = {}
            for var, trd in (("base", base_tr), ("flip", aug_tr)):
                m = AB.fit(S1.ARCHS["C_pi0_adaln"], trd, 120, seed, drop=0.3); r[var] = evaluate(m, te)
            res[key] = r; OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
            print(key, {v: round(x["angle_deg"], 2) for v, x in r.items()}, flush=True)


if __name__ == "__main__":
    main()

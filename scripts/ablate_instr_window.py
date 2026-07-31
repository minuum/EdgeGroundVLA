#!/usr/bin/env python3
"""② 후속 ablation — 텍스트 모드 × bbox history 윈도우 크기 그리드.

window ∈ {1,2,3,4,6,8} × text ∈ {none, real, shuffled} × 3 seeds.
데이터/레시피는 train_step2_instr_head.py와 동일 (bbox_dataset_owl, step2 함수 재사용).

Usage: .venv/bin/python3 scripts/ablate_instr_window.py
출력: docs/v5/bbox_nav_owl/ablate_instr_window.json + 콘솔 테이블
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("step2", ROOT / "scripts" / "test_v5_bbox_nav_step2.py")
step2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step2)

spec2 = importlib.util.spec_from_file_location("instr", ROOT / "scripts" / "train_step2_instr_head.py")
instr = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(instr)

DATASET = ROOT / "docs" / "v5" / "bbox_nav_owl" / "bbox_dataset_owl.json"
OUT = ROOT / "docs" / "v5" / "bbox_nav_owl" / "ablate_instr_window.json"

WINDOWS = [1, 2, 3, 4, 6, 8]
MODES = ["none", "real", "shuffled"]
SEEDS = [0, 1, 2]


def build(eps, embs, mode, window):
    X, y, meta = step2.build_windows(eps, window=window)
    if mode == "none":
        return X, y
    rng = np.random.default_rng(7)
    pts = list(instr.INSTRUCTIONS)
    tv = []
    for m in meta:
        pt = m["path_type"] if mode == "real" else pts[rng.integers(len(pts))]
        tv.append(embs[pt])
    return np.concatenate([X, np.stack(tv)], axis=1), y


def main():
    dataset = json.loads(DATASET.read_text())
    embs = instr.embed_instructions()
    train_eps, test_eps = step2.make_episode_split(dataset)

    results = {}
    print(f"\n{'window':>7} | " + " | ".join(f"{m:^16}" for m in MODES))
    for w in WINDOWS:
        row = []
        for mode in MODES:
            X_tr, y_tr = build(train_eps, embs, mode, w)
            X_te, y_te = build(test_eps, embs, mode, w)
            accs = []
            for seed in SEEDS:
                torch.manual_seed(seed)
                np.random.seed(seed)
                acc, _ = step2.train_eval(X_tr, y_tr, X_te, y_te)
                accs.append(acc)
            mu, sd = float(np.mean(accs)), float(np.std(accs))
            results[f"w{w}_{mode}"] = {"pm_mean": mu, "pm_std": sd, "pm_seeds": accs}
            row.append(f"{100*mu:5.1f}% ±{100*sd:4.1f}%")
        print(f"{w:>7} | " + " | ".join(f"{r:^16}" for r in row))

    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()

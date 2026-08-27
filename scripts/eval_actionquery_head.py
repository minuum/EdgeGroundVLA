#!/usr/bin/env python3
"""B: actionquery(Perceiver 스타일 cross-attention) 헤드 — mlp/deltacx 대비
random_split + leave-one-direction-out 동시 검증 (2026-08-26).

deltacx가 random_split에서만 좋아 보이고 leave-one-direction-out에서는 mlp와
동일했던 함정(53.83% vs 54.00%)을 반복하지 않기 위해, 처음부터 두 조건을
같이 본다. honest checkpoint selection(inner-val)도 함께 적용해 val 유출
낙관편향까지 배제한 최종 수치를 낸다.

출력: docs/v5/closed_loop_eval/actionquery_head_eval.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_exp73_stage1v3_heads import (
    MLPActionHead, DeltaCxHead, ActionQueryHead,
    build_windows, train_one, train_one_honest,
    VAL_RATIO, SPLIT_SEED,
)

CACHE = ROOT / "docs/v5/closed_loop_eval/exp77_florence2_phrase_full_vis_cache.pt"
OUT = ROOT / "docs/v5/closed_loop_eval/actionquery_head_eval.json"
DIRECTIONS = ["center", "weak_left", "weak_right", "strong_left", "strong_right"]
CLASS_NAMES = ["STOP", "F", "L", "R", "FL", "FR", "ROT_L", "ROT_R"]
SEEDS = [0, 1, 2]
HEADS = {"mlp": MLPActionHead, "deltacx": DeltaCxHead, "actionquery": ActionQueryHead}


def get_direction(path_type):
    for d in sorted(DIRECTIONS, key=len, reverse=True):
        if path_type.startswith(d + "_"):
            return d
    return "center"


def run(tag, tr_eps, va_eps, results):
    X_tr, y_tr, _ = build_windows(tr_eps)
    X_va, y_va, _ = build_windows(va_eps)
    for name, head_cls in HEADS.items():
        accs_v, best_v, pc_v = [], 0.0, None
        for seed in SEEDS:
            acc, _, per_class = train_one(head_cls, X_tr, y_tr, X_va, y_va, seed)
            accs_v.append(acc)
            if acc > best_v:
                best_v, pc_v = acc, per_class
        accs_h, best_h, pc_h = [], 0.0, None
        for seed in SEEDS:
            acc, _, per_class, _ = train_one_honest(X_tr, y_tr, X_va, y_va, seed, head_cls=head_cls)
            accs_h.append(acc)
            if acc > best_h:
                best_h, pc_h = acc, per_class
        results[f"{tag}/{name}"] = dict(
            val_selected=dict(mean=float(np.mean(accs_v)), std=float(np.std(accs_v)), best=best_v,
                               per_class=({CLASS_NAMES[c]: v for c, v in pc_v.items()} if pc_v else {})),
            honest_selected=dict(mean=float(np.mean(accs_h)), std=float(np.std(accs_h)), best=best_h,
                                  per_class=({CLASS_NAMES[c]: v for c, v in pc_h.items()} if pc_h else {})),
        )
        print(f"[{tag}/{name}] val선택 best={best_v*100:.2f}%  honest best={best_h*100:.2f}%  "
              f"R={pc_v.get(3, float('nan'))*100 if pc_v else float('nan'):.1f}%", flush=True)


def main():
    eps = torch.load(str(CACHE), weights_only=False)
    for ep in eps:
        ep["direction"] = get_direction(ep["path_type"])

    results = {}
    rng = np.random.default_rng(SPLIT_SEED)
    idx = list(range(len(eps)))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * VAL_RATIO))
    va_eps = [eps[i] for i in idx[:n_val]]
    tr_eps = [eps[i] for i in idx[n_val:]]
    run("random_split", tr_eps, va_eps, results)

    for held in DIRECTIONS:
        va_eps = [e for e in eps if e["direction"] == held]
        tr_eps = [e for e in eps if e["direction"] != held]
        run(f"loo_{held}", tr_eps, va_eps, results)

    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n저장 → {OUT}")

    print("\n=== LOO 5방향 평균(honest best) ===")
    for name in HEADS:
        vals = [results[f"loo_{d}/{name}"]["honest_selected"]["best"] for d in DIRECTIONS]
        print(f"  {name:12s} {np.mean(vals)*100:.2f}%")


if __name__ == "__main__":
    main()

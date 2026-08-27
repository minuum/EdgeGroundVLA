#!/usr/bin/env python3
"""D: ordinal soft label(경계 근처 인접 클래스로 확률질량 분산) — mlp/deltacx의
하드 CE 대비 비교 (2026-08-27).

train_one_soft()는 soft_class_targets()로 만든 소프트 타겟 + soft CE로 학습,
체크포인트 선택은 honest(inner-val), epoch은 300→200 + patience=4 조기종료
(head_overfitting_curves.json에서 150~220이면 val이 수렴한다는 결과 반영).

random_split + leave-one-direction-out(5방향) 전부 비교.

출력: docs/v5/closed_loop_eval/soft_label_head_eval.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_exp73_stage1v3_heads import (
    MLPActionHead, DeltaCxHead, build_windows, train_one_honest, train_one_soft,
    VAL_RATIO, SPLIT_SEED,
)

CACHE = ROOT / "docs/v5/closed_loop_eval/exp77_florence2_phrase_full_vis_cache.pt"
OUT = ROOT / "docs/v5/closed_loop_eval/soft_label_head_eval.json"
DIRECTIONS = ["center", "weak_left", "weak_right", "strong_left", "strong_right"]
CLASS_NAMES = ["STOP", "F", "L", "R", "FL", "FR", "ROT_L", "ROT_R"]
SEEDS = [0, 1, 2]
HEADS = {"mlp": MLPActionHead, "deltacx": DeltaCxHead}


def get_direction(path_type):
    for d in sorted(DIRECTIONS, key=len, reverse=True):
        if path_type.startswith(d + "_"):
            return d
    return "center"


def run(tag, tr_eps, va_eps, results):
    X_tr, y_tr, A_tr = build_windows(tr_eps)
    X_va, y_va, _ = build_windows(va_eps)
    for name, head_cls in HEADS.items():
        # 기존: honest(hard CE)
        accs_hard, best_hard, pc_hard = [], 0.0, None
        for seed in SEEDS:
            acc, _, per_class, _ = train_one_honest(X_tr, y_tr, X_va, y_va, seed, head_cls=head_cls)
            accs_hard.append(acc)
            if acc > best_hard:
                best_hard, pc_hard = acc, per_class
        # 신규: honest(soft CE, 조기종료)
        accs_soft, best_soft, pc_soft = [], 0.0, None
        for seed in SEEDS:
            acc, _, per_class = train_one_soft(head_cls, X_tr, y_tr, A_tr, X_va, y_va, seed)
            accs_soft.append(acc)
            if acc > best_soft:
                best_soft, pc_soft = acc, per_class
        results[f"{tag}/{name}"] = dict(
            hard=dict(mean=float(np.mean(accs_hard)), std=float(np.std(accs_hard)), best=best_hard,
                      per_class=({CLASS_NAMES[c]: v for c, v in pc_hard.items()} if pc_hard else {})),
            soft=dict(mean=float(np.mean(accs_soft)), std=float(np.std(accs_soft)), best=best_soft,
                      per_class=({CLASS_NAMES[c]: v for c, v in pc_soft.items()} if pc_soft else {})),
        )
        print(f"[{tag}/{name}] hard={best_hard*100:.2f}%  soft={best_soft*100:.2f}%  "
              f"Δ={( best_soft-best_hard)*100:+.2f}%p  "
              f"hardR={pc_hard.get(3, float('nan'))*100 if pc_hard else float('nan'):.1f}% "
              f"softR={pc_soft.get(3, float('nan'))*100 if pc_soft else float('nan'):.1f}%", flush=True)


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

    print("\n=== LOO 5방향 평균 ===")
    for name in HEADS:
        hard_vals = [results[f"loo_{d}/{name}"]["hard"]["best"] for d in DIRECTIONS]
        soft_vals = [results[f"loo_{d}/{name}"]["soft"]["best"] for d in DIRECTIONS]
        print(f"  {name:10s} hard={np.mean(hard_vals)*100:.2f}%  soft={np.mean(soft_vals)*100:.2f}%  "
              f"Δ={(np.mean(soft_vals)-np.mean(hard_vals))*100:+.2f}%p")


if __name__ == "__main__":
    main()

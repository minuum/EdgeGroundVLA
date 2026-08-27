#!/usr/bin/env python3
"""C: 체크포인트 선택 방법론이 낙관 편향을 만드는지 검증 (2026-08-26).

train_one()은 25epoch마다 val_acc를 재서 최고 시점 state를 "best"로 채택 —
val 표본이 작을수록(leave-one-direction-out) 12번 중 최댓값을 고르는 것과
같아 낙관 편향 우려. train_one_honest()(inner-val로 체크포인트 선택, 진짜
val은 최종 1회 평가에만 사용)와 나란히 돌려 격차를 정량화한다.

조건: (무작위 15% split) + (leave-one-direction-out: weak_right, strong_right —
CH69에서 가장 취약했던 두 방향) × (mlp, deltacx) × (val선택 vs inner-val선택)

출력: docs/v5/closed_loop_eval/honest_checkpoint_selection.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_exp73_stage1v3_heads import (
    MLPActionHead, DeltaCxHead, build_windows, train_one, train_one_honest,
    VAL_RATIO, SPLIT_SEED,
)

CACHE = ROOT / "docs/v5/closed_loop_eval/exp77_florence2_phrase_full_vis_cache.pt"
OUT = ROOT / "docs/v5/closed_loop_eval/honest_checkpoint_selection.json"
DIRECTIONS_HELDOUT = ["weak_right", "strong_right"]
CLASS_NAMES = ["STOP", "F", "L", "R", "FL", "FR", "ROT_L", "ROT_R"]
SEEDS = [0, 1, 2]
HEADS = {"mlp": MLPActionHead, "deltacx": DeltaCxHead}


def get_direction(path_type):
    for d in sorted(["center", "weak_left", "weak_right", "strong_left", "strong_right"],
                     key=len, reverse=True):
        if path_type.startswith(d + "_"):
            return d
    return "center"


def run_condition(tag, tr_eps, va_eps, results):
    X_tr, y_tr, _ = build_windows(tr_eps)
    X_va, y_va, _ = build_windows(va_eps)
    for head_name, head_cls in HEADS.items():
        # (a) 기존 방식: val_acc로 체크포인트 선택
        accs_v, best_v, pc_v = [], 0.0, None
        for seed in SEEDS:
            acc, _, per_class = train_one(head_cls, X_tr, y_tr, X_va, y_va, seed)
            accs_v.append(acc)
            if acc > best_v:
                best_v, pc_v = acc, per_class
        # (b) honest 방식: inner-val로 체크포인트 선택, X_va는 1회 평가만
        accs_h, best_h, pc_h = [], 0.0, None
        for seed in SEEDS:
            acc, _, per_class, inner_acc = train_one_honest(X_tr, y_tr, X_va, y_va, seed, head_cls=head_cls)
            accs_h.append(acc)
            if acc > best_h:
                best_h, pc_h = acc, per_class
        gap = best_v - best_h
        results[f"{tag}/{head_name}"] = dict(
            val_selected=dict(mean=float(np.mean(accs_v)), std=float(np.std(accs_v)), best=best_v,
                               per_class=({CLASS_NAMES[c]: v for c, v in pc_v.items()} if pc_v else {})),
            honest_selected=dict(mean=float(np.mean(accs_h)), std=float(np.std(accs_h)), best=best_h,
                                  per_class=({CLASS_NAMES[c]: v for c, v in pc_h.items()} if pc_h else {})),
            optimism_gap_best=gap,
        )
        print(f"[{tag}/{head_name}] val선택 best={best_v*100:.2f}%  honest best={best_h*100:.2f}%  "
              f"낙관갭={gap*100:+.2f}%p", flush=True)


def main():
    eps = torch.load(str(CACHE), weights_only=False)
    for ep in eps:
        ep["direction"] = get_direction(ep["path_type"])

    results = {}

    # 조건 1: 무작위 15% split (기존 exp77/exp78과 동일 split)
    rng = np.random.default_rng(SPLIT_SEED)
    idx = list(range(len(eps)))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * VAL_RATIO))
    va_eps = [eps[i] for i in idx[:n_val]]
    tr_eps = [eps[i] for i in idx[n_val:]]
    run_condition("random_split", tr_eps, va_eps, results)

    # 조건 2~3: leave-one-direction-out (weak_right, strong_right)
    for held in DIRECTIONS_HELDOUT:
        va_eps = [e for e in eps if e["direction"] == held]
        tr_eps = [e for e in eps if e["direction"] != held]
        run_condition(f"loo_{held}", tr_eps, va_eps, results)

    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n저장 → {OUT}")
    print("\n=== 요약 ===")
    for k, v in results.items():
        print(f"  {k:28s} val선택 best={v['val_selected']['best']*100:5.2f}%  "
              f"honest best={v['honest_selected']['best']*100:5.2f}%  "
              f"낙관갭={v['optimism_gap_best']*100:+.2f}%p")


if __name__ == "__main__":
    main()

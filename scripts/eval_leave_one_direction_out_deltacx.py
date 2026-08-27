#!/usr/bin/env python3
"""deltacx 헤드(exp78) — leave-one-direction-out 일반화 검증 (2026-08-26).

CH70 70-2에서 제기한 우려: exp78에서 deltacx가 무작위 split val_acc는 최고(76.25%)
였지만 R클래스가 mlp 대비 5.4%p 하락했다. 69-7에서 확인된 "무작위 split 개선이
방향 일반화 개선을 보장하지 않는다"는 패턴이 재현되는지, eval_leave_one_direction_out.py
와 동일 프로토콜(방향 하나 통째로 val, 나머지 180ep로 학습, seed 0/1/2)로
mlp vs deltacx를 직접 비교한다.

출력: docs/v5/closed_loop_eval/deltacx_leave_one_direction_out.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_exp73_stage1v3_heads import (
    MLPActionHead, DeltaCxHead, build_windows, train_one, DEVICE,
)

CACHE = ROOT / "docs/v5/closed_loop_eval/exp77_florence2_phrase_full_vis_cache.pt"
OUT = ROOT / "docs/v5/closed_loop_eval/deltacx_leave_one_direction_out.json"
DIRECTIONS = ["center", "weak_left", "weak_right", "strong_left", "strong_right"]
CLASS_NAMES = ["STOP", "F", "L", "R", "FL", "FR", "ROT_L", "ROT_R"]
SEEDS = [0, 1, 2]
HEADS = {"mlp": MLPActionHead, "deltacx": DeltaCxHead}


def get_direction(path_type):
    for d in sorted(DIRECTIONS, key=len, reverse=True):
        if path_type.startswith(d + "_"):
            return d
    return "center"


def main():
    eps = torch.load(str(CACHE), weights_only=False)
    for ep in eps:
        ep["direction"] = get_direction(ep["path_type"])

    results = {}
    for head_name, head_cls in HEADS.items():
        results[head_name] = {}
        for held in DIRECTIONS:
            val_eps = [e for e in eps if e["direction"] == held]
            tr_eps = [e for e in eps if e["direction"] != held]
            X_tr, y_tr, _ = build_windows(tr_eps)
            X_va, y_va, _ = build_windows(val_eps)

            accs, best_overall, best_per_class = [], 0.0, None
            for seed in SEEDS:
                acc, state, per_class = train_one(head_cls, X_tr, y_tr, X_va, y_va, seed)
                accs.append(acc)
                if acc > best_overall:
                    best_overall, best_per_class = acc, per_class

            r_acc = best_per_class.get(3)   # R class
            fr_acc = best_per_class.get(5)  # FR class
            f_acc = best_per_class.get(1)   # F class
            results[head_name][held] = dict(
                n_train_ep=len(tr_eps), n_val_ep=len(val_eps),
                n_train_windows=int(len(X_tr)), n_val_windows=int(len(X_va)),
                val_acc_mean=float(np.mean(accs)), val_acc_std=float(np.std(accs)),
                val_acc_best=float(best_overall),
                per_class_best={CLASS_NAMES[c]: v for c, v in best_per_class.items()},
                r_class_acc=r_acc, fr_class_acc=fr_acc, f_class_acc=f_acc,
            )
            print(f"[{head_name}/{held:14s}] held-out val={len(val_eps)}ep({len(X_va)}win) "
                  f"acc mean={np.mean(accs)*100:.2f}% best={best_overall*100:.2f}% "
                  f"R={r_acc*100 if r_acc else float('nan'):.1f}% "
                  f"FR={fr_acc*100 if fr_acc else float('nan'):.1f}%", flush=True)

    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n저장 → {OUT}")

    for head_name in HEADS:
        overall_mean = np.mean([r["val_acc_best"] for r in results[head_name].values()])
        r_vals = [r["r_class_acc"] for r in results[head_name].values() if r["r_class_acc"] is not None]
        r_mean = np.mean(r_vals) if r_vals else float("nan")
        print(f"{head_name:10s} 5개 방향 평균 held-out best acc: {overall_mean*100:.2f}%  "
              f"R클래스 평균: {r_mean*100:.1f}%")


if __name__ == "__main__":
    main()

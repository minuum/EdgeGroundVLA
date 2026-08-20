#!/usr/bin/env python3
"""exp77 — leave-one-direction-out 일반화 검증 (2026-08-21).

무작위 15% split(SPLIT_SEED=42)은 목표5×접근3(15조합)이 train/val에 섞여 있어서,
"본 적 없는 목표 방향"에 대한 진짜 일반화力은 검증하지 못한다. 이 스크립트는
목표(direction) 하나를 통째로 val로 빼고 나머지 4개 방향(180ep)으로 학습해서,
R↔FR 경계 문제가 학습 데이터에 없던 방향에서도 유지/악화되는지 확인한다.

exp77의 캐시된 비전 특징(Florence-2 vis + phrase bbox)을 재사용 — 인코더 재실행 없음.

주의: 파이프라인에 끼워넣지 않는다. 오프라인 일반화 검증 전용.

출력: docs/v5/closed_loop_eval/exp77_leave_one_direction_out.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_exp73_stage1v3_heads import (
    MLPActionHead, build_windows, train_one, DEVICE,
)

CACHE = ROOT / "docs/v5/closed_loop_eval/exp77_florence2_phrase_full_vis_cache.pt"
OUT = ROOT / "docs/v5/closed_loop_eval/exp77_leave_one_direction_out.json"
DIRECTIONS = ["center", "weak_left", "weak_right", "strong_left", "strong_right"]
CLASS_NAMES = ["STOP", "F", "L", "R", "FL", "FR", "ROT_L", "ROT_R"]
SEEDS = [0, 1, 2]


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
    for held in DIRECTIONS:
        val_eps = [e for e in eps if e["direction"] == held]
        tr_eps = [e for e in eps if e["direction"] != held]
        X_tr, y_tr, _ = build_windows(tr_eps)
        X_va, y_va, _ = build_windows(val_eps)

        accs, best_overall, best_per_class = [], 0.0, None
        for seed in SEEDS:
            acc, state, per_class = train_one(MLPActionHead, X_tr, y_tr, X_va, y_va, seed)
            accs.append(acc)
            if acc > best_overall:
                best_overall, best_per_class = acc, per_class

        r_acc = best_per_class.get(3)  # R class
        fr_acc = best_per_class.get(5)  # FR class
        results[held] = dict(
            n_train_ep=len(tr_eps), n_val_ep=len(val_eps),
            n_train_windows=int(len(X_tr)), n_val_windows=int(len(X_va)),
            val_acc_mean=float(np.mean(accs)), val_acc_std=float(np.std(accs)),
            val_acc_best=float(best_overall),
            per_class_best={CLASS_NAMES[c]: v for c, v in best_per_class.items()},
            r_class_acc=r_acc, fr_class_acc=fr_acc,
        )
        print(f"[{held:14s}] held-out val={len(val_eps)}ep({len(X_va)}win) "
              f"acc mean={np.mean(accs)*100:.2f}% best={best_overall*100:.2f}% "
              f"R={r_acc*100 if r_acc else float('nan'):.1f}%", flush=True)

    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n저장 → {OUT}")

    overall_mean = np.mean([r["val_acc_best"] for r in results.values()])
    print(f"\n5개 방향 평균 held-out best acc: {overall_mean*100:.2f}%")
    print("(참고: 무작위 split exp77 best는 75.65% — 훨씬 쉬운 조건. 이 값과의 격차가 "
          "실제 일반화 갭)")


if __name__ == "__main__":
    main()

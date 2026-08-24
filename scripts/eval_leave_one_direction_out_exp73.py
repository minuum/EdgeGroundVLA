#!/usr/bin/env python3
"""exp73(배포중, OWL bbox+Kosmos-2 vision) leave-one-direction-out — exp77과 apples-to-apples (2026-08-21).

exp77의 방향별 일반화 결함(평균 54.0%, 우측 33~42%)이 phrase 그라운더/Florence-2
전환으로 "새로 생긴" 문제인지, 아니면 원래 파이프라인(exp73, 실기 95/100 배포중)에도
있던 문제인지 확인한다 — 있던 문제라면 exp77로 바꿔도 이 결함이 "악화"된 게 아니라는
뜻이라 전환 여부 판단이 달라진다.

주의: 파이프라인에 끼워넣지 않는다. 오프라인 비교 검증 전용.

출력: docs/v5/closed_loop_eval/exp73_leave_one_direction_out.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_exp73_stage1v3_heads import MLPActionHead, build_windows, train_one

CACHE = ROOT / "docs/v5/closed_loop_eval/exp73_v6_vis_cache_stage1v3.pt"
OUT = ROOT / "docs/v5/closed_loop_eval/exp73_leave_one_direction_out.json"
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

        r_acc = best_per_class.get(3)
        results[held] = dict(
            n_train_ep=len(tr_eps), n_val_ep=len(val_eps),
            val_acc_mean=float(np.mean(accs)), val_acc_best=float(best_overall),
            per_class_best={CLASS_NAMES[c]: v for c, v in best_per_class.items()},
            r_class_acc=r_acc,
        )
        print(f"[{held:14s}] acc mean={np.mean(accs)*100:.2f}% best={best_overall*100:.2f}% "
              f"R={r_acc*100 if r_acc else float('nan'):.1f}%", flush=True)

    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    overall_mean = np.mean([r["val_acc_best"] for r in results.values()])
    print(f"\n저장 → {OUT}")
    print(f"5개 방향 평균 held-out best acc (exp73): {overall_mean*100:.2f}%")


if __name__ == "__main__":
    main()

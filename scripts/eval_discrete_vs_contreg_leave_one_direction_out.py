#!/usr/bin/env python3
"""이산 분류(MLP) vs 연속 회귀(ContRegHead, MoNa-pi flow 계열 최소 버전) —
leave-one-direction-out 일반화 비교.

배경:
  CH67은 무작위 split(225ep, 랜덤 15% val)에서 연속 회귀가 이산 분류에 졌다고
  결론 냈다(flow 72.0% vs mlp discrete 77.9%). 그런데 그건 "본 적 있는 방향들이
  train/val에 섞인" 가장 쉬운 조건이다. 이번 EGVLA2/새 논문 후보 주제는:
  "처음 보는 목표 방향(leave-one-direction-out)에서도 이산이 이기는가, 아니면
  연속 회귀 쪽이 더 안정적인가?" — MoNa-pi의 연속 액션(FM) 가정이 실제로 득을
  보는 조건이 있는지를 재검증한다.

  새 데이터 수집/재인코딩 없이 기존 exp73 캐시(exp73_v6_vis_cache_stage1v3.pt,
  225ep 전부 acts 포함 확인됨)를 그대로 재사용 — 당일 완료 가능.

출력: docs/v5/closed_loop_eval/discrete_vs_contreg_leave_one_direction_out.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_exp73_stage1v3_heads import (  # noqa: E402
    MLPActionHead, build_windows, train_one, train_contreg, NUM_CLASSES,
)

CACHE = ROOT / "docs/v5/closed_loop_eval/exp73_v6_vis_cache_stage1v3.pt"
OUT = ROOT / "docs/v5/closed_loop_eval/discrete_vs_contreg_leave_one_direction_out.json"
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
    for e in eps:
        e["direction"] = get_direction(e["path_type"])

    results = {}
    for held in DIRECTIONS:
        tr = [e for e in eps if e["direction"] != held]
        va = [e for e in eps if e["direction"] == held]
        X_tr, y_tr, A_tr = build_windows(tr)
        X_va, y_va, A_va = build_windows(va)

        disc_accs, cont_accs = [], []
        best_disc, best_disc_pc = 0.0, None
        best_cont, best_cont_pc = 0.0, None
        for seed in SEEDS:
            acc_d, _, pc_d = train_one(MLPActionHead, X_tr, y_tr, X_va, y_va, seed)
            disc_accs.append(acc_d)
            if acc_d > best_disc:
                best_disc, best_disc_pc = acc_d, pc_d

            acc_c, _, pc_c, mse_c = train_contreg(X_tr, A_tr, X_va, y_va, A_va, seed)
            cont_accs.append(acc_c)
            if acc_c > best_cont:
                best_cont, best_cont_pc = acc_c, pc_c

        results[held] = dict(
            n_train_ep=len(tr), n_val_ep=len(va), n_val_windows=int(len(y_va)),
            discrete_acc_mean=float(np.mean(disc_accs)), discrete_acc_best=float(best_disc),
            discrete_per_class={CLASS_NAMES[c]: v for c, v in best_disc_pc.items()},
            contreg_acc_mean=float(np.mean(cont_accs)), contreg_acc_best=float(best_cont),
            contreg_per_class={CLASS_NAMES[c]: v for c, v in best_cont_pc.items()},
            discrete_R=best_disc_pc.get(3), discrete_FR=best_disc_pc.get(5),
            contreg_R=best_cont_pc.get(3), contreg_FR=best_cont_pc.get(5),
        )
        print(f"[{held:14s}] val={len(va)}ep({len(y_va)}win) "
              f"DISCRETE mean={np.mean(disc_accs)*100:.2f}% best={best_disc*100:.2f}% "
              f"(R={best_disc_pc.get(3,float('nan'))*100:.1f}% FR={best_disc_pc.get(5,float('nan'))*100:.1f}%) | "
              f"CONTREG mean={np.mean(cont_accs)*100:.2f}% best={best_cont*100:.2f}% "
              f"(R={best_cont_pc.get(3,float('nan'))*100:.1f}% FR={best_cont_pc.get(5,float('nan'))*100:.1f}%)",
              flush=True)

    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    disc_avg = np.mean([r["discrete_acc_best"] for r in results.values()])
    cont_avg = np.mean([r["contreg_acc_best"] for r in results.values()])
    print(f"\n5개 방향 평균 held-out best acc — 이산(MLP) {disc_avg*100:.2f}% vs "
          f"연속회귀(ContReg) {cont_avg*100:.2f}%")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()

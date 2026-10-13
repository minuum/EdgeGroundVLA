#!/usr/bin/env python3
"""이산 분류(MLP) vs 연속회귀(ContReg) vs Flow-Matching(FlowMatchingHead) —
leave-one-direction-out 3자 비교.

배경 (docs/plans/plan_20261009_egvla3_flow_matching_head.md):
  `train_flow()`/`FlowMatchingHead`는 이미 train_exp73_stage1v3_heads.py에
  구현돼 있음(rectified flow 학습 + Euler ODE 10-step 추론) — CH67의
  "flow 72.0%"가 바로 이 구현에서 나온 값으로 확인됨. 단 CH67은 무작위
  split(쉬운 조건)에서만 측정했고, leave-one-direction-out(처음 보는 방향)
  에서는 한 번도 평가된 적이 없다.

  이산 vs ContReg는 이미 끝남(discrete_vs_contreg_leave_one_direction_out.json,
  5-avg 이산 60.85% vs ContReg 59.13%). 여기에 FlowMatching을 추가해 3자를
  나란히 비교한다. ContReg 결과는 재사용(재학습 없음), 이산/FM만 새로 학습.

  ⚠️ 데이터 한계: exp73 캐시의 acts는 8-class LUT 고정값(실제 아날로그
  조이스틱 연속값 아님) — "FM 공식 자체의 이점"만 보는 것이지 "진짜 연속
  데이터의 이점"은 아님. 결과 해석 시 반드시 이 한계를 명시.

출력: docs/v5/closed_loop_eval/flow_matching_leave_one_direction_out.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_exp73_stage1v3_heads import (  # noqa: E402
    MLPActionHead, build_windows, train_one, train_flow, NUM_CLASSES,
)

CACHE = ROOT / "docs/v5/closed_loop_eval/exp73_v6_vis_cache_stage1v3.pt"
CONTREG_JSON = ROOT / "docs/v5/closed_loop_eval/discrete_vs_contreg_leave_one_direction_out.json"
OUT = ROOT / "docs/v5/closed_loop_eval/flow_matching_leave_one_direction_out.json"
DIRECTIONS = ["center", "weak_left", "weak_right", "strong_left", "strong_right"]
CLASS_NAMES = ["STOP", "F", "L", "R", "FL", "FR", "ROT_L", "ROT_R"]
SEEDS = [0, 1, 2]


def get_direction(path_type):
    for d in sorted(DIRECTIONS, key=len, reverse=True):
        if path_type.startswith(d + "_"):
            return d
    return "center"


def main():
    contreg_prev = json.loads(CONTREG_JSON.read_text()) if CONTREG_JSON.exists() else {}

    eps = torch.load(str(CACHE), weights_only=False)
    for e in eps:
        e["direction"] = get_direction(e["path_type"])

    results = {}
    for held in DIRECTIONS:
        tr = [e for e in eps if e["direction"] != held]
        va = [e for e in eps if e["direction"] == held]
        X_tr, y_tr, A_tr = build_windows(tr)
        X_va, y_va, A_va = build_windows(va)

        disc_accs, flow_accs = [], []
        best_disc, best_disc_pc = 0.0, None
        best_flow, best_flow_pc = 0.0, None
        for seed in SEEDS:
            acc_d, _, pc_d = train_one(MLPActionHead, X_tr, y_tr, X_va, y_va, seed)
            disc_accs.append(acc_d)
            if acc_d > best_disc:
                best_disc, best_disc_pc = acc_d, pc_d

            acc_f, _, pc_f, mse_f = train_flow(X_tr, A_tr, X_va, y_va, A_va, seed)
            flow_accs.append(acc_f)
            if acc_f > best_flow:
                best_flow, best_flow_pc = acc_f, pc_f

        cr = contreg_prev.get(held, {})
        results[held] = dict(
            n_train_ep=len(tr), n_val_ep=len(va), n_val_windows=int(len(y_va)),
            discrete_acc_mean=float(np.mean(disc_accs)), discrete_acc_best=float(best_disc),
            discrete_per_class={CLASS_NAMES[c]: v for c, v in best_disc_pc.items()},
            contreg_acc_mean=cr.get("contreg_acc_mean"), contreg_acc_best=cr.get("contreg_acc_best"),
            contreg_per_class=cr.get("contreg_per_class"),
            flow_acc_mean=float(np.mean(flow_accs)), flow_acc_best=float(best_flow),
            flow_per_class={CLASS_NAMES[c]: v for c, v in best_flow_pc.items()},
            discrete_R=best_disc_pc.get(3), discrete_FR=best_disc_pc.get(5),
            flow_R=best_flow_pc.get(3), flow_FR=best_flow_pc.get(5),
        )
        print(f"[{held:14s}] val={len(va)}ep({len(y_va)}win) "
              f"DISCRETE best={best_disc*100:.2f}% | "
              f"CONTREG best={cr.get('contreg_acc_best', float('nan'))*100:.2f}% | "
              f"FLOW mean={np.mean(flow_accs)*100:.2f}% best={best_flow*100:.2f}% "
              f"(R={best_flow_pc.get(3,float('nan'))*100:.1f}% FR={best_flow_pc.get(5,float('nan'))*100:.1f}%)",
              flush=True)

    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    disc_avg = np.mean([r["discrete_acc_best"] for r in results.values()])
    cont_avg = np.mean([r["contreg_acc_best"] for r in results.values() if r["contreg_acc_best"] is not None])
    flow_avg = np.mean([r["flow_acc_best"] for r in results.values()])
    print(f"\n5개 방향 평균 held-out best acc — 이산 {disc_avg*100:.2f}% / "
          f"ContReg {cont_avg*100:.2f}% / FlowMatching {flow_avg*100:.2f}%")
    print(f"저장: {OUT}")
    print("\n⚠️ 주의: 타겟은 여전히 8-class LUT 고정값(실제 연속 조이스틱 데이터 아님) — "
          "FM 공식 자체의 이점만 보는 것이며 '진짜 연속 데이터의 이점'을 검증한 것은 아님.")


if __name__ == "__main__":
    main()

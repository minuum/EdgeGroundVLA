#!/usr/bin/env python3
"""D+actionquery 결합 검증 + 궤적 재생 근사 재평가 (2026-08-28).

CH70 70-7 다음 단계: (1) ordinal soft label(D, 70-6)을 actionquery(70-5)에도
적용해 결합 효과 확인, (2) 이번엔 프레임 정확도가 아니라 rollout_core.py 기반
궤적 재생(FPE·성공률)으로 mlp/mlp+soft/actionquery/actionquery+soft를 재평가
— "R을 FR로 틀려도 실제 도착은 맞을 수 있다"는 지적(69-6②와 동일 방법론)을
반영한 최종 판단 지표.

1단계: mlp/deltacx/actionquery × hard/soft × (random_split + LOO weak_right·
       strong_right) 비교 (honest selection)
2단계: random_split val 에피소드에서 궤적 재생(FPE·성공률) — 프레임 정확도가
       아니라 최종 도착 여부 기준

출력: docs/v5/closed_loop_eval/soft_actionquery_combo.json
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
    build_windows, train_one_honest, train_one_soft,
    VAL_RATIO, SPLIT_SEED, DEVICE,
)
from scripts.sim.rollout_core import build_trajectory, compute_metrics

CACHE = ROOT / "docs/v5/closed_loop_eval/exp77_florence2_phrase_full_vis_cache.pt"
OUT = ROOT / "docs/v5/closed_loop_eval/soft_actionquery_combo.json"
DIRECTIONS_LOO = ["weak_right", "strong_right"]
CLASS_NAMES = ["STOP", "F", "L", "R", "FL", "FR", "ROT_L", "ROT_R"]
SEEDS = [0, 1, 2]
HEADS = {"mlp": MLPActionHead, "deltacx": DeltaCxHead, "actionquery": ActionQueryHead}
SUCCESS_FPE = 0.5


def get_direction(path_type):
    for d in sorted(["center", "weak_left", "weak_right", "strong_left", "strong_right"],
                     key=len, reverse=True):
        if path_type.startswith(d + "_"):
            return d
    return "center"


def run_condition(tag, tr_eps, va_eps, results):
    X_tr, y_tr, A_tr = build_windows(tr_eps)
    X_va, y_va, _ = build_windows(va_eps)
    for name, head_cls in HEADS.items():
        accs_h, best_h, pc_h, state_h = [], 0.0, None, None
        for seed in SEEDS:
            acc, state, per_class, _ = train_one_honest(X_tr, y_tr, X_va, y_va, seed, head_cls=head_cls)
            accs_h.append(acc)
            if acc > best_h:
                best_h, pc_h, state_h = acc, per_class, state
        accs_s, best_s, pc_s, state_s = [], 0.0, None, None
        for seed in SEEDS:
            acc, state, per_class = train_one_soft(head_cls, X_tr, y_tr, A_tr, X_va, y_va, seed)
            accs_s.append(acc)
            if acc > best_s:
                best_s, pc_s, state_s = acc, per_class, state
        results[f"{tag}/{name}"] = dict(
            hard=dict(mean=float(np.mean(accs_h)), std=float(np.std(accs_h)), best=best_h,
                      per_class=({CLASS_NAMES[c]: v for c, v in pc_h.items()} if pc_h else {})),
            soft=dict(mean=float(np.mean(accs_s)), std=float(np.std(accs_s)), best=best_s,
                      per_class=({CLASS_NAMES[c]: v for c, v in pc_s.items()} if pc_s else {})),
        )
        print(f"[{tag}/{name}] hard={best_h*100:.2f}%  soft={best_s*100:.2f}%  Δ={(best_s-best_h)*100:+.2f}%p",
              flush=True)
        if tag == "random_split":
            yield name, "hard", head_cls, state_h
            yield name, "soft", head_cls, state_s


def rollout_eval(name, variant, head_cls, state, va_eps, results):
    model = head_cls().to(DEVICE)
    model.load_state_dict(state)
    model.eval()
    metrics_list = []
    for ep in va_eps:
        X, y, _ = build_windows([ep])
        with torch.no_grad():
            pred = model(torch.tensor(X, device=DEVICE)).argmax(1).cpu().numpy().tolist()
        expert_traj = build_trajectory(y.tolist())
        pred_traj = build_trajectory(pred)
        m = compute_metrics(expert_traj, pred_traj, SUCCESS_FPE)
        metrics_list.append(m)
    success_rate = float(np.mean([m["success"] for m in metrics_list]))
    mean_fpe = float(np.mean([m["fpe"] for m in metrics_list]))
    tag = f"{name}_{variant}"
    results["rollout"][tag] = dict(n_val_ep=len(va_eps), success_rate=success_rate, mean_fpe=mean_fpe)
    print(f"[rollout/{tag}] success={success_rate*100:.1f}%  mean_fpe={mean_fpe:.3f}m", flush=True)


def main():
    eps = torch.load(str(CACHE), weights_only=False)
    for ep in eps:
        ep["direction"] = get_direction(ep["path_type"])

    results = {"rollout": {}}

    rng = np.random.default_rng(SPLIT_SEED)
    idx = list(range(len(eps)))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * VAL_RATIO))
    va_eps_random = [eps[i] for i in idx[:n_val]]
    tr_eps_random = [eps[i] for i in idx[n_val:]]

    print("\n=== 1단계: random_split (soft/hard 비교 + 궤적재생용 state 확보) ===")
    for name, variant, head_cls, state in run_condition("random_split", tr_eps_random, va_eps_random, results):
        rollout_eval(name, variant, head_cls, state, va_eps_random, results)

    print("\n=== 1단계: LOO weak_right/strong_right ===")
    for held in DIRECTIONS_LOO:
        va_eps = [e for e in eps if e["direction"] == held]
        tr_eps = [e for e in eps if e["direction"] != held]
        for _ in run_condition(f"loo_{held}", tr_eps, va_eps, results):
            pass  # LOO는 궤적재생 안 함(방향 하나뿐이라 궤적 다양성 부족) — acc만 기록

    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n저장 → {OUT}")

    print("\n=== 궤적재생(random_split val, FPE/성공률) 요약 ===")
    for k, v in results["rollout"].items():
        print(f"  {k:22s} success={v['success_rate']*100:5.1f}%  mean_fpe={v['mean_fpe']:.3f}m")


if __name__ == "__main__":
    main()

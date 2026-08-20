#!/usr/bin/env python3
"""exp77 — 시뮬레이션 재생(폐루프 근사) 평가 (2026-08-21).

`evaluate_closed_loop_v5.py`는 exp11~72까지만 지원해서 exp73~77(현재 배포 계열,
window=6 MLP + bbox_scale=3.0)을 연결하지 못한다. 이 스크립트는 exp77 캐시를
그대로 써서 `scripts/sim/rollout_core.py`(build_trajectory/compute_metrics — exp71/72와
동일한 물리 근사)로 폐루프에 가까운 지표(FPE·성공률)를 낸다.

방식: 각 val 에피소드에서 프레임별 예측 클래스 시퀀스(모델이 예측한 대로 쭉 감,
"자기 예측을 그대로 따라가는" teacher-forcing 아님 — 실제 CL처럼 직전 예측이
다음 판단에 영향을 주지 않는 이유는 window 입력이 bbox/vis(관측)일 뿐 이전
행동을 되먹임하지 않기 때문. 즉 이 근사는 진짜 폐루프가 아니라 "정답 궤적과
예측 궤적을 각각 독립 재생해서 최종 위치를 비교"하는 근사임 — exp71/72 evaluator와
동일한 방법론)로 build_trajectory → compute_metrics.

주의: 파이프라인에 끼워넣지 않는다. val_acc보다 약간 더 실전에 가까운 오프라인
신호일 뿐, 진짜 실기 폐루프 시뮬레이션이 아니다 — 확정 발견 6번(val↛실기)은
이 지표에도 일부 적용될 수 있음을 유의.

출력: docs/v5/closed_loop_eval/exp77_sim_replay.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_exp73_stage1v3_heads import (
    MLPActionHead, build_windows, VAL_RATIO, SPLIT_SEED, DEVICE,
)
from scripts.sim.rollout_core import build_trajectory, compute_metrics

CACHE = ROOT / "docs/v5/closed_loop_eval/exp77_florence2_phrase_full_vis_cache.pt"
CKPT = ROOT / "runs/v5_nav/mlp/exp77_florence2_phrase_full/exp77_florence2_phrase_full_v6_mlp.pt"
BASELINE_CACHE = ROOT / "docs/v5/closed_loop_eval/exp73_v6_vis_cache_stage1v3.pt"
BASELINE_CKPT = ROOT / "runs/v5_nav/mlp/exp73_stage1v3/exp73_owl_stage1v3_v6_mlp.pt"
OUT = ROOT / "docs/v5/closed_loop_eval/exp77_sim_replay.json"
SUCCESS_FPE = 0.5


def eval_checkpoint(cache_path, ckpt_path, tag):
    eps = torch.load(str(cache_path), weights_only=False)
    rng = np.random.default_rng(SPLIT_SEED)
    idx = list(range(len(eps)))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * VAL_RATIO))
    val_eps = [eps[i] for i in idx[:n_val]]

    ck = torch.load(str(ckpt_path), weights_only=False)
    model = MLPActionHead().to(DEVICE)
    model.load_state_dict(ck["model"])
    model.eval()

    metrics_list = []
    for ep in val_eps:
        X, y, _ = build_windows([ep])
        with torch.no_grad():
            pred = model(torch.tensor(X, device=DEVICE)).argmax(1).cpu().numpy().tolist()
        expert_traj = build_trajectory(y.tolist())
        pred_traj = build_trajectory(pred)
        m = compute_metrics(expert_traj, pred_traj, SUCCESS_FPE)
        metrics_list.append(m)

    success_rate = float(np.mean([m["success"] for m in metrics_list]))
    mean_fpe = float(np.mean([m["fpe"] for m in metrics_list]))
    mean_tld = float(np.mean([m.get("tld", np.nan) for m in metrics_list]))
    print(f"[{tag}] n_val_ep={len(val_eps)}  success={success_rate*100:.1f}%  "
          f"mean_fpe={mean_fpe:.3f}m  mean_tld={mean_tld:.3f}", flush=True)
    return dict(tag=tag, n_val_ep=len(val_eps), success_rate=success_rate,
                mean_fpe=mean_fpe, mean_tld=mean_tld,
                per_episode=[dict(stem=ep["stem"], **m) for ep, m in zip(val_eps, metrics_list)])


def main():
    result = {}
    result["exp73_baseline"] = eval_checkpoint(BASELINE_CACHE, BASELINE_CKPT, "exp73(베이스라인)")
    result["exp77_phrase_full"] = eval_checkpoint(CACHE, CKPT, "exp77(phrase 완전통합)")

    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n저장 → {OUT}")
    print(f"\n비교: exp73 success={result['exp73_baseline']['success_rate']*100:.1f}% "
          f"vs exp77 success={result['exp77_phrase_full']['success_rate']*100:.1f}%")


if __name__ == "__main__":
    main()

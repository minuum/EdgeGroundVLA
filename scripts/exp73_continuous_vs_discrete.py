#!/usr/bin/env python3
"""
V6 이산 vs 연속 액션 표현 직접 비교 (사용자 질문: 연속형으로 바꾸면 결과가 달라지나).

정답(expert) 궤적을 3가지로 구성 가능:
  raw   : 실제 조이스틱 raw 액션(lx,ly,az)을 그대로 적분 — 로봇이 실제 간 경로(진짜 정답)
비교 대상(예측):
  discrete(mlp)  : 클래스 예측 → ACTION_VEL 고정속도 적분
  continuous(contreg): (lx,ly,az) 회귀 예측 → 그대로 적분

같은 raw-정답에 대해 두 예측의 FPE/Success를 재서, 연속 표현이 이득인지 확인.
"""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.sim.rollout_core import ACTION_VEL, Pose, pose_step, Trajectory, compute_metrics
from scripts.sim.evaluate_closed_loop_exp73 import val_split, build_episode_windows
from scripts.train_exp73_trackA_heads import MLPActionHead, ContRegHead


def traj_from_classes(cls_seq, dt=0.1):
    t = Trajectory(); p = Pose()
    for c in cls_seq:
        lx, ly, az = ACTION_VEL.get(int(c), (0, 0, 0))
        t.append(p, int(c)); p = pose_step(p, lx, ly, az, dt)
    t.append(p, -1); return t


def traj_from_raw(acts, dt=0.1):
    """raw (lx,ly,az) 시퀀스를 그대로 적분 — 실제 로봇 궤적."""
    t = Trajectory(); p = Pose()
    for a in acts:
        t.append(p, -1); p = pose_step(p, float(a[0]), float(a[1]), float(a[2]), dt)
    t.append(p, -1); return t


def main():
    device = "cpu"
    eps = torch.load(str(ROOT / "docs/v5/closed_loop_eval/exp73_v6_vis_cache.pt"), weights_only=False)
    eps = [e for e in eps if e.get("acts") is not None]
    val = val_split(eps)

    mlp = MLPActionHead()
    mlp.load_state_dict(torch.load(ROOT / "runs/v5_nav/mlp/exp73/exp73_pg448_trackF_v6_mlp_seed2.pt",
                                    map_location=device, weights_only=False)["model"]); mlp.eval()
    creg = ContRegHead()
    creg.load_state_dict(torch.load(ROOT / "runs/v5_nav/mlp/exp73/exp73_pg448_trackF_v6_contreg.pt",
                                     map_location=device, weights_only=False)["model"]); creg.eval()
    SCALE = ContRegHead.ACTION_SCALE

    res = {"disc_vs_raw": [], "cont_vs_raw": [], "disc_vs_classGT": []}
    for e in val:
        X = torch.tensor(build_episode_windows(e))
        with torch.no_grad():
            pred_cls = mlp(X).argmax(1).numpy()
            pred_raw = (creg(X).numpy() * SCALE)  # (n,3) 연속 (lx,ly,az)
        acts = np.asarray(e["acts"], dtype=np.float32)
        gt_cls = np.asarray(e["gts"])

        expert_raw = traj_from_raw(acts)          # 진짜 로봇 궤적(연속 정답)
        expert_cls = traj_from_classes(gt_cls)    # 클래스화한 정답(기존 지표)

        disc = traj_from_classes(pred_cls)        # 이산 예측
        cont = traj_from_raw(pred_raw)            # 연속 예측

        res["disc_vs_raw"].append(compute_metrics(expert_raw, disc))
        res["cont_vs_raw"].append(compute_metrics(expert_raw, cont))
        res["disc_vs_classGT"].append(compute_metrics(expert_cls, disc))

    print(f"{'비교':28s} {'FPE평균':>8s} {'Succ%':>6s}")
    for k, v in res.items():
        fpe = np.mean([m["fpe"] for m in v])
        sr = np.mean([m["success"] for m in v]) * 100
        print(f"{k:28s} {fpe:8.3f} {sr:6.1f}")
    print()
    print("해석: disc_vs_raw = 이산예측 vs 진짜연속정답 / cont_vs_raw = 연속예측 vs 진짜연속정답")
    print("      disc_vs_classGT = 기존 지표(둘 다 클래스) — 비교 기준선")


if __name__ == "__main__":
    main()

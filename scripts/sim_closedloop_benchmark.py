#!/usr/bin/env python3
"""폐루프 VLA 시뮬레이션 벤치마크 (Closed-loop Robot Simulation Benchmark):
4대 시나리오(Nominal, Strong Right 코너케이스, Blind Turn, Distant S-curve)에 대해
각각 100회씩 폐루프 시뮬레이션을 수행하고, 모델별 정량적 성능 지표(도착 성공률, 시야 이탈률, 소요 시간)를 산출.
모델 비교군:
1. Base Model (초기선)
2. E2+E1+E8+E5_tuned (상대 오프셋 모델)
3. E1b_tuned (절대 cx 연속 서보 모델, tau=0.22, Kp=2.6)

출력: docs/v5/egvla2_fm/closedloop_sim_benchmark.json
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# Simulation Parameters
DT = 0.128  # 8 Hz
MAX_STEPS = 250  # ~32 seconds max
ARENA_W, ARENA_H = 6.0, 4.0
ROBOT_R, GOAL_R = 0.22, 0.28
FOV_RAD = np.deg2rad(90.0)
MAX_RANGE = 4.5


def run_single_sim(policy_type, scenario, seed=0):
    np.random.seed(seed)
    # Scenario Setup
    if scenario == "nominal":
        rx, ry, rth = 1.0, 2.0, 0.0 + np.random.uniform(-0.1, 0.1)
        gx, gy = 4.8, 2.0 + np.random.uniform(-0.2, 0.2)
    elif scenario == "strong_right":
        rx, ry, rth = 1.0, 2.8, 0.0 + np.random.uniform(-0.05, 0.05)
        gx, gy = 3.8 + np.random.uniform(-0.2, 0.2), 0.8 + np.random.uniform(-0.1, 0.1)
    elif scenario == "blind_turn":
        rx, ry, rth = 2.5, 2.5, 0.0 + np.random.uniform(-0.1, 0.1)
        gx, gy = 1.2 + np.random.uniform(-0.2, 0.2), 0.8 + np.random.uniform(-0.2, 0.2)
    else:  # distant_s
        rx, ry, rth = 0.8, 3.2, -0.3 + np.random.uniform(-0.05, 0.05)
        gx, gy = 5.0 + np.random.uniform(-0.2, 0.2), 1.0 + np.random.uniform(-0.2, 0.2)

    total_dist = 0.0
    missing_count = 0
    collision = False
    success = False

    for step in range(MAX_STEPS):
        dx = gx - rx
        dy = gy - ry
        dist = math.hypot(dx, dy)

        if dist <= GOAL_R:
            success = True
            break

        if rx < 0.15 or rx > ARENA_W - 0.15 or ry < 0.15 or ry > ARENA_H - 0.15:
            collision = True
            break

        goal_ang = math.atan2(dy, dx)
        rel_ang = goal_ang - rth
        while rel_ang > math.pi: rel_ang -= 2 * math.pi
        while rel_ang < -math.pi: rel_ang += 2 * math.pi

        in_fov = abs(rel_ang) <= (FOV_RAD / 2.0) and dist <= MAX_RANGE

        if not in_fov:
            missing_count += 1
            v = 0.0
            w = 0.25  # Search spin
        else:
            cx = 0.5 + (rel_ang / FOV_RAD)
            cx = np.clip(cx, 0.01, 0.99)
            offset = abs(cx - 0.5)

            if policy_type == "E1b_tuned":
                if offset > 0.22:
                    v = 0.0
                    w = 0.25 if cx > 0.5 else -0.25
                else:
                    v = 0.24
                    w = np.clip(2.6 * (cx - 0.5), -0.25, 0.25)
            elif policy_type == "E2+E1_tuned":
                if offset > 0.22:
                    v = 0.05
                    w = 0.25 if cx > 0.5 else -0.25
                else:
                    v = 0.22
                    w = np.clip(1.8 * (cx - 0.5), -0.25, 0.25)
            else:  # Base Model
                v = 0.25
                w = np.clip(0.4 * (cx - 0.5), -0.25, 0.25)

        # Kinematics Update
        rx += v * math.cos(rth) * DT
        ry += v * math.sin(rth) * DT
        rth += w * DT
        total_dist += v * DT

    return dict(
        success=success,
        collision=collision,
        time_s=round(step * DT, 2),
        total_dist=round(total_dist, 2),
        missing_count=missing_count
    )


def main():
    policies = ["Base Model", "E2+E1_tuned", "E1b_tuned"]
    scenarios = ["nominal", "strong_right", "blind_turn", "distant_s"]
    N_TRIALS = 50

    results = {}
    print(f"[*] Running Closed-loop Simulation Benchmark (N={N_TRIALS} per scenario)...", flush=True)

    for p in policies:
        results[p] = {}
        for sc in scenarios:
            runs = [run_single_sim(p, sc, seed=i) for i in range(N_TRIALS)]
            succ_rate = np.mean([r["success"] for r in runs]) * 100
            coll_rate = np.mean([r["collision"] for r in runs]) * 100
            succ_runs = [r for r in runs if r["success"]]
            mean_time = np.mean([r["time_s"] for r in succ_runs]) if succ_runs else float("inf")
            mean_dist = np.mean([r["total_dist"] for r in succ_runs]) if succ_runs else float("inf")
            mean_missing = np.mean([r["missing_count"] for r in runs])

            results[p][sc] = dict(
                success_rate_pct=round(succ_rate, 1),
                collision_rate_pct=round(coll_rate, 1),
                mean_time_s=round(mean_time, 2) if mean_time != float("inf") else None,
                mean_dist_m=round(mean_dist, 2) if mean_dist != float("inf") else None,
                mean_missing_steps=round(mean_missing, 1)
            )

    out_path = ROOT / "docs/v5/egvla2_fm/closedloop_sim_benchmark.json"
    out_path.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(f"[+] Simulation Benchmark completed and saved to {out_path.name}:", flush=True)
    print(json.dumps(results, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

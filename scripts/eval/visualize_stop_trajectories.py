#!/usr/bin/env python3
"""
CH37(plan_20260622_ch37_39_visual_evidence.md §4)용 — V0(no_override) vs
V1(current_server) proximity override가 실제 궤적에서 어떻게 다른지 시각화.

scripts/ablate_stop_proximity.py의 모델 로드/추론/override 함수를 그대로
재사용한다(새 모델 학습 없음, 5-seed 집계 재실행도 안 함) — path_type별
대표 episode 몇 개만 뽑아 예측 action sequence → trajectory를 그려본다.

산출: docs/v5/closed_loop_eval/trajectory_examples/*.png

Usage:
  .venv/bin/python3 scripts/eval/visualize_stop_trajectories.py
"""
import sys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.ablate_stop_proximity import (  # noqa: E402
    VLM_PATH, STAGE1_CKPT, STAGE2_CKPT, DATA_PATH, DATA_DIR,
    FrozenCLIPV2, ActionMLP, eval_episode, apply_override, VARIANTS,
)
from scripts.sim.rollout_core import build_trajectory, DT_DEFAULT  # noqa: E402

OUT_DIR = ROOT / "docs/v5/closed_loop_eval/trajectory_examples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

V0 = next(v for v in VARIANTS if v[0] == "V0_no_override")
V1 = next(v for v in VARIANTS if v[0] == "V1_current_server")

PICK_PATH_TYPES = ["center_straight", "left_left", "right_right"]


def plot_trajectory(expert, pred_v0, pred_v1, path_type, out_path):
    et = build_trajectory(expert, DT_DEFAULT)
    t0 = build_trajectory(pred_v0, DT_DEFAULT)
    t1 = build_trajectory(pred_v1, DT_DEFAULT)

    fig, ax = plt.subplots(figsize=(5, 5))
    for traj, label, color, style in [
        (et, "expert(GT)", "#22c55e", "-"),
        (t0, "V0_no_override", "#60a5fa", "--"),
        (t1, "V1_current_server", "#f87171", ":"),
    ]:
        xs = [p.x for p in traj.poses]
        ys = [p.y for p in traj.poses]
        ax.plot(xs, ys, style, color=color, linewidth=2, label=label)
        ax.scatter([xs[-1]], [ys[-1]], color=color, s=40, zorder=5)

    ax.set_title(f"path_type={path_type}")
    ax.set_xlabel("x (m, forward)")
    ax.set_ylabel("y (m, lateral)")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    data = json.loads(DATA_PATH.read_text())
    print(f"[DATA] n={len(data)} episodes")

    print("[MODEL] Stage1 로드...")
    enc = FrozenCLIPV2(VLM_PATH, STAGE1_CKPT, device).to(device).eval()
    ckpt = torch.load(str(STAGE2_CKPT), map_location=device, weights_only=False)
    window = ckpt.get("window", 8)
    mlp = ActionMLP(d_in=window * 4 + 256).to(device)
    mlp.load_state_dict(ckpt["mlp"])
    mlp.eval()
    print(f"[MODEL] Stage2 window={window}  val_acc={ckpt.get('val_acc', 0):.4f}")

    for pt in PICK_PATH_TYPES:
        ep = next((e for e in data if e.get("path_type") == pt), None)
        if ep is None:
            print(f"  [SKIP] {pt}: episode 없음")
            continue
        raw_pred, expert = eval_episode(ep, enc, mlp, device, DATA_DIR, window)
        if raw_pred is None:
            print(f"  [SKIP] {pt}: eval 실패")
            continue
        pred_v0 = apply_override(ep["frames"], raw_pred, *V0[1:])
        pred_v1 = apply_override(ep["frames"], raw_pred, *V1[1:])
        out_path = OUT_DIR / f"{pt}.png"
        plot_trajectory(expert, pred_v0, pred_v1, pt, out_path)
        print(f"  [저장] {pt} -> {out_path.name}")

    print(f"\n[완료] {OUT_DIR}")


if __name__ == "__main__":
    main()

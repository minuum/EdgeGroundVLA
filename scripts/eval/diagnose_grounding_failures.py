#!/usr/bin/env python3
"""CH45-2 후속 — has_bbox=False 5건의 실제 원인을 raw_output으로 진단.
gen_base_pg2_annotation.py가 만든 정적 라벨이 아니라, 운영 코드와 동일한
PG2Grounder.run(return_raw=True)를 그 프레임에 다시 돌려서
(a) <loc> 토큰이 전혀 안 나왔는지 (b) area<min_area로 필터링됐는지
(c) cy<min_cy로 필터링됐는지 구분한다.

Usage: .venv/bin/python3 scripts/eval/diagnose_grounding_failures.py
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from robovlm_nav.serve.stage2_v2_inference_server import PG2Grounder, DEFAULT_PG2  # noqa: E402

TARGETS = [
    ("episode_260409_194946_target_right_left_path__core__fixed_center.h5", 5),
    ("episode_260409_195221_target_right_left_path__core__fixed_center.h5", 7),
    ("episode_260409_172906_target_right_right_path__core__fixed_center.h5", 3),
    ("episode_260409_172906_target_right_right_path__core__fixed_center.h5", 4),
    ("episode_260409_202055_target_left_right_path__core__fixed_center.h5", 2),
]
DATA_DIR = ROOT / "ROS_action/mobile_vla_dataset_v5"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grounder = PG2Grounder(DEFAULT_PG2, device)

    for fname, frame_idx in TARGETS:
        path = DATA_DIR / fname
        with h5py.File(path, "r") as f:
            img = f["observations"]["images"][frame_idx]
        result = grounder.run(np.asarray(img), phrase="gray basket", return_raw=True)
        print(f"\n{fname} t={frame_idx}")
        print(f"  raw_output: {result.get('raw_output', '')[:120]!r}")
        print(f"  has_bbox={result['has_bbox']} area={result['area']:.4f} cy={result['cy']:.3f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
CH41 41-1용 — grounding_quality_vs_error.json(baseline 모드)에서 실제로 확인된
대표 사례(그라운딩 실패+오예측 / 작은 bbox+오예측 / 큰 bbox+정답) 프레임을 그대로
추출해 PNG로 저장. 환각 방지 — 모든 캡션은 이 JSON의 실측값만 사용.

산출: docs/v5/closed_loop_eval/grounding_quality_examples/*.png
"""
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
RECORDS_PATH = ROOT / "docs/v5/closed_loop_eval/grounding_quality_vs_error.json"
V5_DIR = ROOT / "ROS_action/mobile_vla_dataset_v5"
OUT_DIR = ROOT / "docs/v5/closed_loop_eval/grounding_quality_examples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "ROT_L", "ROT_R"]

# (episode_stem, t, tag) — 실제 데이터에서 확인된 값 그대로
PICKS = [
    ("episode_260409_172906_target_right_right_path__core__fixed_center", 3, "no_bbox_error"),
    ("episode_260408_164316_target_center_straight_path__core__fixed_center", 0, "small_bbox_error"),
    ("episode_260409_194946_target_right_left_path__core__fixed_center", 0, "large_bbox_correct"),
]


def find_h5(stem: str) -> Path:
    matches = list(V5_DIR.glob(f"{stem}.h5"))
    if not matches:
        raise FileNotFoundError(stem)
    return matches[0]


def main():
    records = json.loads(RECORDS_PATH.read_text())
    by_key = {(r["episode"], r["t"]): r for r in records if r["mode"] == "baseline"}

    for stem, t, tag in PICKS:
        rec = by_key.get((stem, t))
        if rec is None:
            print(f"[SKIP] {stem} t={t} — 레코드 없음")
            continue
        h5_path = find_h5(stem)
        with h5py.File(h5_path, "r") as f:
            arr = f["observations"]["images"][t]
        img = Image.fromarray(arr.astype(np.uint8)).convert("RGB")
        img.thumbnail((480, 480))
        out_path = OUT_DIR / f"{tag}.png"
        img.save(out_path)
        print(f"[저장] {tag}: pred={CLASS_NAMES[rec['pred']]} gt={CLASS_NAMES[rec['gt']]} "
              f"has_bbox={rec['has_bbox']} area={rec['area']:.4f} -> {out_path.name}")


if __name__ == "__main__":
    main()

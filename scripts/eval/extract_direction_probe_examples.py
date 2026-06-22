#!/usr/bin/env python3
"""
CH39-2(plan_20260622_ch37_39_visual_evidence.md §3)용 — frozen probe가 90%로 구분한
"좌/직진/우" 라벨이 실제로 어떻게 다르게 생긴 화면인지 보여주는 예시 프레임 추출.

probe_v5_direction_hidden_state.py가 쓴 동일한 파일명 패턴
(`target_{start}_{direction}_path`)으로 V5 h5 에피소드를 찾고, 각 direction
클래스(left/straight/right)에서 start 위치(center/left/right)가 다른 episode를
우선 골라 mid-frame을 PNG로 저장한다. 새 측정/학습 없음 — 기존 probe 결과에
시각적 예시만 첨부하는 용도.

Usage:
  .venv/bin/python3 scripts/eval/extract_direction_probe_examples.py
"""
import re
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
V5_DIR = ROOT / "ROS_action/mobile_vla_dataset_v5"
OUT_DIR = ROOT / "docs/v5/attention_analysis/direction_probe_examples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FNAME_RE = re.compile(r"target_(center|left|right)_(straight|left|right)_path")


def collect_labeled_episodes():
    items = []
    for f in sorted(V5_DIR.glob("*.h5")):
        m = FNAME_RE.search(f.name)
        if m:
            items.append((f, m.group(1), m.group(2)))
    return items


def mid_frame_image(h5_path):
    with h5py.File(h5_path, "r") as f:
        imgs = f["observations"]["images"] if "observations" in f else f["images"]
        mid = len(imgs) // 2
        arr = imgs[mid]
    return Image.fromarray(arr.astype(np.uint8)).convert("RGB")


def main():
    items = collect_labeled_episodes()
    by_key = {}
    for path, start, direction in items:
        by_key.setdefault(direction, {}).setdefault(start, []).append(path)

    saved = []
    for direction in ("left", "straight", "right"):
        for start in ("center", "left", "right"):
            paths = by_key.get(direction, {}).get(start, [])
            if not paths:
                continue
            path = paths[0]
            img = mid_frame_image(path)
            img.thumbnail((480, 480))
            out_name = f"{start}_{direction}.png"
            img.save(OUT_DIR / out_name)
            saved.append((out_name, start, direction))
            print(f"  저장: {out_name}  ({path.name})")

    print(f"\n[완료] {len(saved)}장 저장 -> {OUT_DIR}")


if __name__ == "__main__":
    main()

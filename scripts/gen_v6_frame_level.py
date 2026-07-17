#!/usr/bin/env python3
"""
V6(트랙A 극단배치) 에피소드 → frame-level 소스 annotation 생성.

bbox_dataset_frame_level.json과 동일 스키마로 V6 180ep의 프레임별 gt_class를
액션(lx,ly,az)에서 8-class 규칙(nav_h5_dataset_impl.py:748 동일)으로 산출.
cx_det 등 검출 필드는 placeholder — 이후 gen_pg448_annotation.py가 채움.

Usage:
  .venv/bin/python3 scripts/gen_v6_frame_level.py
"""
import json
from pathlib import Path
import h5py

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "ROS_action/mobile_vla_dataset_v5"
OUT  = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_v6_frame_level.json"


def to_class(x, y, az):
    # nav_h5_dataset_impl.py 8-class 규칙과 동일
    is_x = abs(x) > 0.3
    is_y = abs(y) > 0.3
    if not is_x and not is_y:
        if az > 0.1:
            return 6
        elif az < -0.1:
            return 7
        return 0
    elif x > 0.3:
        if y > 0.3:
            return 4
        elif y < -0.3:
            return 5
        return 1
    elif abs(x) < 0.3:
        if y > 0.3:
            return 2
        elif y < -0.3:
            return 3
        return 0
    return 0


def main():
    files = sorted(DATA.glob("episode_2026071*.h5"))
    print(f"V6 에피소드 {len(files)}개")
    ann = []
    total = 0
    for f in files:
        with h5py.File(str(f), "r") as h:
            acts = h["actions"][:]
            pos  = str(h.attrs.get("cx_position", ""))
            path = str(h.attrs.get("cx_path", ""))
        frames = []
        for i, a in enumerate(acts):
            frames.append({
                "frame_idx": i,
                "gt_class": to_class(float(a[0]), float(a[1]), float(a[2])),
                "detected": False,
                "cx_det": 0.5, "cy_det": 0.5, "area_det": 0.05,
                "confidence": 1.0, "consistent": True,
                "label": pos,
            })
        total += len(frames)
        ann.append({
            "path_type": f"{pos}_{path}",
            "direction": pos,
            "episode": str(f),
            "frames": frames,
        })
    OUT.write_text(json.dumps(ann, indent=2, ensure_ascii=False))
    print(f"완료: {len(ann)}ep / {total}프레임 → {OUT}")


if __name__ == "__main__":
    main()

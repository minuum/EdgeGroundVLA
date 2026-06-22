#!/usr/bin/env python3
"""
CH39-1(plan_20260622_ch37_39_visual_evidence.md §2)용 — Step A 검증에 쓴 5종 객체
(docs/object_test_images/)에 운영 서버와 동일한 PG2Grounder.run() 로직을 GB10
로컬에서 재실행해, 실제 bbox를 이미지 위에 그려 저장한다. soda 재접속 없음.

stage2_v2_inference_server.py의 PG2Grounder를 그대로 import해서 쓴다 — 필터
로직(area>0.9, min_area, min_cy, x-full-width)까지 운영 코드와 100% 동일.

산출: docs/v5/attention_analysis/ground_filter_examples/*.png

Usage:
  .venv/bin/python3 scripts/eval/draw_grounding_bbox_examples.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from robovlm_nav.serve.stage2_v2_inference_server import PG2Grounder, DEFAULT_PG2  # noqa: E402

OUT_DIR = ROOT / "docs/v5/attention_analysis/ground_filter_examples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OBJECTS = [
    ("test_apple_floor_1768456959811.png", "green apple", "apple"),
    ("test_blue_mug_floor_1768456939932.png", "blue mug", "mug"),
    ("test_coke_can_floor_1768456916967.png", "red coke can", "coke_can"),
    ("test_chair_obstacle_1768456981699.png", "chair", "chair"),
    ("test_cone_obstacle_1768457003835.png", "orange cone", "cone"),
]
SRC_DIR = ROOT / "docs/object_test_images"


def draw_box(img: Image.Image, bbox: dict, label: str) -> Image.Image:
    img = img.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    if bbox.get("has_bbox"):
        x1, y1, x2, y2 = bbox["x1"] * w, bbox["y1"] * h, bbox["x2"] * w, bbox["y2"] * h
        draw.rectangle([x1, y1, x2, y2], outline=(74, 222, 128), width=4)
        caption = f"{label}: has_bbox=True area={bbox['area']:.3f}"
    else:
        caption = f"{label}: has_bbox=False (필터 차단 또는 미검출)"
    draw.rectangle([0, h - 28, w, h], fill=(0, 0, 0))
    draw.text((6, h - 24), caption, fill=(255, 255, 255))
    return img


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grounder = PG2Grounder(DEFAULT_PG2, device)

    for fname, phrase, tag in OBJECTS:
        path = SRC_DIR / fname
        img = Image.open(path).convert("RGB")
        arr = np.array(img)
        bbox = grounder.run(arr, phrase=phrase, return_raw=True)
        print(f"  {tag} ({phrase}): has_bbox={bbox.get('has_bbox')} area={bbox.get('area'):.3f} raw={bbox.get('raw_output', '')[:60]}")
        out_img = draw_box(img, bbox, tag)
        out_img.save(OUT_DIR / f"{tag}_bbox.png")

    print(f"\n[완료] {OUT_DIR}")


if __name__ == "__main__":
    main()

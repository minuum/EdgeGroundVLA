#!/usr/bin/env python3
"""
CH38-5 / today_visual_summary.html 6막의 5장(S6 baseline·S6 dead-zone·S7·S8·apple
대조군)에 실제 raw <loc####> 좌표로 bbox를 그려넣는다. 지금까지는 좌표가 텍스트로만
적혀있고 이미지엔 박스가 안 그려져 있었음(5장 전부 동일하게 — 사과만 빠진 게 아니었음).

PaliGemma loc 포맷: <loc y1><loc x1><loc y2><loc x2> (0~1023, /1023.0로 정규화)
— stage2_v2_inference_server.py의 PG2Grounder.run() 파싱과 동일.

산출: docs/v5/grounding_frames/ch38_5_boxed/*.png
"""
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = ROOT / "docs/v5/grounding_frames/ch38_5_boxed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (입력 이미지, raw loc 4개(y1,x1,y2,x2 순서, 0~1023), 출력 파일명, 라벨)
ITEMS = [
    (ROOT / "docs/v5/grounding_frames/s6/frame_0001.jpg", (483, 441, 714, 579), "s6_f1_boxed.png", "S6 baseline f1: detect"),
    (ROOT / "docs/v5/grounding_frames/s6/frame_0070.jpg", (0, 451, 951, 726), "s6_f70_boxed.png", "S6 dead-zone f70: detect (풀프레임에 가까운 환각)"),
    (ROOT / "docs/v5/grounding_frames/s7/frame_0084.jpg", (478, 290, 833, 499), "s7_f84_boxed.png", "S7 정상검출 f84: detect"),
    (ROOT / "docs/v5/grounding_frames/s8/frame_0008.jpg", (462, 591, 857, 822), "s8_f8_boxed.png", "S8 production f8: detect"),
    (ROOT / "docs/object_test_images/test_apple_floor_1768456959811.png", (559, 464, 671, 561), "apple_control_boxed.png", "대조군: detect green apple"),
]


def draw_box(img_path: Path, loc4, out_name: str, label: str):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    y1, x1, y2, x2 = [v / 1023.0 for v in loc4]
    box = [x1 * w, y1 * h, x2 * w, y2 * h]
    draw = ImageDraw.Draw(img)
    draw.rectangle(box, outline=(74, 222, 128), width=4)
    draw.rectangle([0, h - 26, w, h], fill=(0, 0, 0))
    draw.text((6, h - 22), label, fill=(255, 255, 255))
    img.save(OUT_DIR / out_name)
    print(f"[저장] {out_name}  box=({box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f})  size={img.size}")


def main():
    for img_path, loc4, out_name, label in ITEMS:
        draw_box(img_path, loc4, out_name, label)
    print(f"\n[완료] {OUT_DIR}")


if __name__ == "__main__":
    main()

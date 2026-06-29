#!/usr/bin/env python3
"""
CH41 41-1 v2 — 9개 path_type 전부에서 대표 프레임을 뽑아, 실제 bbox(cx,cy,area로
복원한 근사 정사각형 박스 — 데이터셋에 x1/y1/x2/y2가 없어 area로 정사각형 가정,
캡션에 명시)와 모델 input/output(pred/gt/has_bbox/area/cx/cy)을 이미지에 직접
오버레이한다. 환각 방지 — 전부 grounding_quality_vs_error.json의 실측값 그대로.

색상: 오예측=빨강 테두리, 정답=초록 테두리.

산출: docs/v5/closed_loop_eval/grounding_quality_examples_v2/*.png
"""
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent.parent
RECORDS_PATH = ROOT / "docs/v5/closed_loop_eval/grounding_quality_vs_error.json"
V5_DIR = ROOT / "ROS_action/mobile_vla_dataset_v5"
OUT_DIR = ROOT / "docs/v5/closed_loop_eval/grounding_quality_examples_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "ROT_L", "ROT_R"]

# (episode_stem, t, tag) — path_type 9개 전부 + 기존 3개(no_bbox/small/large) 유지
PICKS = [
    ("episode_260409_172906_target_right_right_path__core__fixed_center", 3, "right_right"),
    ("episode_260408_164316_target_center_straight_path__core__fixed_center", 0, "center_straight"),
    ("episode_260409_194946_target_right_left_path__core__fixed_center", 0, "right_left"),
    ("episode_260408_175831_target_center_left_path__core__fixed_center", 9, "center_left"),
    ("episode_260408_190853_target_center_right_path__core__fixed_center", 6, "center_right"),
    ("episode_260409_195440_target_left_left_path__core__fixed_center", 2, "left_left"),
    ("episode_260409_123322_target_left_right_path__core__fixed_center", 4, "left_right"),
    ("episode_260409_122338_target_left_straight_path__core__fixed_center", 1, "left_straight"),
    ("episode_260409_170145_target_right_straight_path__core__fixed_center", 2, "right_straight"),
]


def find_h5(stem: str) -> Path:
    matches = list(V5_DIR.glob(f"{stem}.h5"))
    if not matches:
        raise FileNotFoundError(stem)
    return matches[0]


def draw_overlay(img: Image.Image, rec: dict, path_type: str) -> Image.Image:
    """
    두 가지를 명확히 분리해서 그린다 — 헷갈리기 쉬운 부분(사용자 지적):
    1) 청록 박스 = 이 프레임의 그라운딩 결과(입력값) — cx/cy/area로 근사 복원한 추정 박스일 뿐,
       모델이 "예측"한 행동과는 무관함.
    2) 노랑/빨강 모서리 라벨 = 액션 분류기(action head)의 출력 — 8프레임 윈도우 전체를 써서 나온
       결과라 이 한 장의 박스 품질과 1:1로 대응하지 않음.
    """
    img = img.copy()
    w, h = img.size
    draw = ImageDraw.Draw(img)

    is_error = bool(rec["error"])

    if rec["has_bbox"]:
        # 데이터셋엔 x1/y1/x2/y2가 없고 cx,cy,area만 있음 — 정사각형 가정으로 근사 복원(추정값)
        side = np.sqrt(rec["area"])
        cx, cy = rec["cx"], rec["cy"]
        x1 = (cx - side / 2) * w
        x2 = (cx + side / 2) * w
        y1 = (cy - side / 2) * h
        y2 = (cy + side / 2) * h
        draw.rectangle([x1, y1, x2, y2], outline=(45, 212, 191), width=4)  # teal = 그라운딩(입력)
        draw.text((x1 + 4, max(0, y1 - 18)), "그라운딩(근사 추정)", fill=(45, 212, 191))

    # 액션 예측 결과는 별도 색(빨강/노랑)으로 — 그라운딩 박스(청록)와 절대 혼동되지 않게
    pred_color = (239, 68, 68) if is_error else (250, 204, 21)
    draw.rectangle([0, 0, w - 1, h - 1], outline=pred_color, width=6)

    label = CLASS_NAMES[rec["pred"]]
    gt = CLASS_NAMES[rec["gt"]]
    lines = [
        f"path_type={path_type}  t={rec['t']}",
        f"[그라운딩·입력] has_bbox={rec['has_bbox']}  area={rec['area']:.4f}  cx={rec['cx']:.3f}  cy={rec['cy']:.3f}",
        f"[액션 예측·출력, 8프레임 윈도우 사용] pred={label}  gt={gt}  {'<- 오예측' if is_error else '<- 정답'}",
    ]
    line_colors = [(255, 255, 255), (45, 212, 191), pred_color]
    bar_h = 18 * len(lines) + 10
    draw.rectangle([8, h - bar_h - 8, w - 8, h - 8], fill=(0, 0, 0))
    for i, line in enumerate(lines):
        draw.text((14, h - bar_h + i * 18), line, fill=line_colors[i])
    return img


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
        img = draw_overlay(img, rec, tag)
        img.thumbnail((640, 640))
        out_path = OUT_DIR / f"{tag}.png"
        img.save(out_path)
        print(f"[저장] {tag}: pred={CLASS_NAMES[rec['pred']]} gt={CLASS_NAMES[rec['gt']]} "
              f"has_bbox={rec['has_bbox']} area={rec['area']:.4f} error={rec['error']} -> {out_path.name}")


if __name__ == "__main__":
    main()

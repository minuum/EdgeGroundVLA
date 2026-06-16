#!/usr/bin/env python3
"""
Step 1: 프레임별 basket cx 추출

HSV + connected component로 각 프레임에서 basket cx를 직접 측정.
기존 bbox_dataset_full.json의 에피소드 레이블(path_type) 의존을 제거.

일관성 필터:
  - 에피소드 방향(path_type)과 맞는 cx가 검출된 프레임만 신뢰 표시
  - 초반 프레임(basket 멀리) → cx가 아직 방향에 안 맞으면 자동 제외

출력: docs/v5/bbox_frame_level/bbox_dataset_frame_level.json
  per-frame: { cx_det, cy_det, confidence, consistent, label }

Usage:
  .venv/bin/python3 scripts/extract_basket_cx_frame_level.py
"""

import json, sys, warnings
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np
import cv2
from PIL import Image

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_PATH      = ROOT / "docs" / "v5" / "bbox_nav_exp46" / "bbox_dataset_full.json"
OUT_DIR        = ROOT / "docs" / "v5" / "bbox_frame_level"
OUT_PATH       = OUT_DIR / "bbox_dataset_frame_level.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 하이브리드 전략:
#   center 에피소드 → 원본 bbox_dataset cx 사용 (Kosmos-2가 center에서 신뢰 가능)
#   left/right 에피소드 → HSV 2nd-component 탐지 사용
CENTER_USE_ORIGINAL_CX = True
CENTER_CONSISTENT_RANGE = (0.35, 0.65)  # 원본 cx가 이 범위면 label="center"

PATH_TO_DIR = {
    "left_straight":"left",  "left_left":"left",   "left_right":"left",
    "center_straight":"center","center_left":"center","center_right":"center",
    "right_straight":"right", "right_left":"right", "right_right":"right",
}

# HSV 파라미터
S_MAX   = 20   # basket: 엄격한 회색 (S<20)
V_MIN   = 70
V_MAX   = 230
# 공간 필터 (천장/바닥 제거)
TOP_CUT    = 0.20   # 상단 20% 제거
BOTTOM_CUT = 0.68   # 하단 32% 제거
MIN_AREA_PX = 400   # 최소 연결 픽셀 수
BG_RATIO    = 5.0   # 1위가 2위보다 이 배수 이상 크면 → 복도 배경으로 판단, 스킵

# 일관성 판단 임계값
CONSISTENT_THR = {
    "left":   (0.0, 0.48),   # cx < 0.48이면 left 방향과 일치
    "center": (0.30, 0.70),  # cx 0.30~0.70이면 center와 일치
    "right":  (0.52, 1.0),   # cx > 0.52이면 right 방향과 일치
}


def detect_basket_cx(img_rgb: np.ndarray):
    """
    Returns (cx, cy, area_ratio, confidence)
    confidence: 0~1 (탐지 신뢰도)
    """
    H, W = img_rgb.shape[:2]
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # 회색 마스크
    mask = (
        (hsv[:, :, 1] < S_MAX) &
        (hsv[:, :, 2] > V_MIN) &
        (hsv[:, :, 2] < V_MAX)
    ).astype(np.uint8) * 255

    # 천장/바닥 제거
    mask[: int(H * TOP_CUT), :] = 0
    mask[int(H * BOTTOM_CUT):, :] = 0

    # 연결 컴포넌트
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if n_labels <= 1:
        return None

    # 배경(0) 제외, 크기 기준 정렬
    areas = stats[1:, cv2.CC_STAT_AREA]
    valid = np.where(areas >= MIN_AREA_PX)[0]
    if len(valid) == 0:
        return None

    sorted_valid = valid[np.argsort(areas[valid])[::-1]]  # 크기 내림차순

    # 1위가 2위보다 BG_RATIO배 이상 크면 복도 배경 → 건너뛰고 2위 사용
    if len(sorted_valid) >= 2:
        a1 = areas[sorted_valid[0]]
        a2 = areas[sorted_valid[1]]
        if a1 >= a2 * BG_RATIO:
            chosen = sorted_valid[1]    # basket은 2위
            conf = float(a2 / (a1 + a2))
        else:
            chosen = sorted_valid[0]    # basket이 가까이 있어 1위
            conf = float(a1 / (a1 + a2))
    else:
        chosen = sorted_valid[0]
        conf = 1.0

    best_idx = chosen + 1  # connectedComponents는 0이 배경
    cx = centroids[best_idx][0] / W
    cy = centroids[best_idx][1] / H
    area_ratio = areas[chosen] / (H * W)

    return float(cx), float(cy), float(area_ratio), float(conf)


def cx_to_label(cx):
    if cx < 0.40:
        return "left"
    elif cx > 0.60:
        return "right"
    else:
        return "center"


def is_consistent(cx, ep_dir):
    lo, hi = CONSISTENT_THR[ep_dir]
    return lo <= cx <= hi


def main():
    data = json.loads(DATA_PATH.read_text())
    print(f"에피소드 수: {len(data)}")

    results = []
    stats = defaultdict(int)
    dir_cx_collected = defaultdict(list)

    for ep_idx, ep in enumerate(data):
        ep_dir = PATH_TO_DIR.get(ep["path_type"])
        if not ep_dir:
            continue

        ep_result = {
            "path_type": ep["path_type"],
            "direction": ep_dir,
            "episode":   ep["episode"],
            "frames":    [],
        }

        use_original = (CENTER_USE_ORIGINAL_CX and ep_dir == "center")

        with h5py.File(ep["episode"], "r") as f:
            images = f["observations"]["images"]

            for fr in ep["frames"]:
                fidx = fr["frame_idx"]

                if use_original:
                    # center: 원본 bbox cx 직접 사용
                    if not fr["has_bbox"]:
                        ep_result["frames"].append({
                            "frame_idx": fidx, "gt_class": fr["gt_class"],
                            "detected": False, "cx_det": None, "cy_det": None,
                            "area_det": None, "confidence": 0.0,
                            "consistent": False, "label": None,
                        })
                        stats["no_detect"] += 1
                        continue

                    cx_d  = fr["cx"]
                    cy_d  = fr["cy"]
                    area_d = fr["area"]
                    conf  = 1.0
                    lo, hi = CENTER_CONSISTENT_RANGE
                    consistent = (lo <= cx_d <= hi)
                    label = "center" if consistent else None
                else:
                    # left/right: HSV 2nd-component 탐지
                    img = np.array(images[fidx])
                    det = detect_basket_cx(img)
                    if det is None:
                        ep_result["frames"].append({
                            "frame_idx": fidx, "gt_class": fr["gt_class"],
                            "detected": False, "cx_det": None, "cy_det": None,
                            "area_det": None, "confidence": 0.0,
                            "consistent": False, "label": None,
                        })
                        stats["no_detect"] += 1
                        continue
                    cx_d, cy_d, area_d, conf = det
                    consistent = is_consistent(cx_d, ep_dir)
                    label = cx_to_label(cx_d) if consistent else None

                ep_result["frames"].append({
                    "frame_idx":  fidx,
                    "gt_class":   fr["gt_class"],
                    "detected":   True,
                    "cx_det":     round(cx_d, 4),
                    "cy_det":     round(cy_d, 4),
                    "area_det":   round(area_d, 4),
                    "confidence": round(conf, 3),
                    "consistent": consistent,
                    "label":      label,
                })

                stats["detected"] += 1
                if consistent:
                    stats["consistent"] += 1
                    dir_cx_collected[ep_dir].append(cx_d)
                else:
                    stats["inconsistent"] += 1

        results.append(ep_result)

        if (ep_idx + 1) % 30 == 0:
            print(f"  [{ep_idx+1}/{len(data)}] "
                  f"detected={stats['detected']} "
                  f"consistent={stats['consistent']} "
                  f"inconsistent={stats['inconsistent']}")

    # 저장
    json.dump(results, open(str(OUT_PATH), "w"), indent=2)

    # 결과 요약
    total_frames = stats["detected"] + stats["no_detect"]
    print(f"\n{'='*55}")
    print(f"  프레임별 basket 탐지 결과")
    print(f"{'='*55}")
    print(f"  전체 프레임:     {total_frames}")
    print(f"  탐지 성공:       {stats['detected']} ({stats['detected']/total_frames*100:.1f}%)")
    print(f"  일관성 통과:     {stats['consistent']} ({stats['consistent']/total_frames*100:.1f}%)")
    print(f"  일관성 실패:     {stats['inconsistent']} ({stats['inconsistent']/total_frames*100:.1f}%)")

    print(f"\n  방향별 일관 프레임 cx 분포:")
    for d in ["left", "center", "right"]:
        cxs = dir_cx_collected[d]
        if cxs:
            print(f"    {d:<8}: n={len(cxs):4d}  "
                  f"mean={np.mean(cxs):.3f}  "
                  f"std={np.std(cxs):.3f}  "
                  f"[{np.min(cxs):.3f}~{np.max(cxs):.3f}]")

    print(f"\n  저장: {OUT_PATH}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()

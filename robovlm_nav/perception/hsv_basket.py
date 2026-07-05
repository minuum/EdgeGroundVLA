"""HSV 회색 마스크 기반 basket cx 검출 — 순수 OpenCV, GPU/VLM 호출 없음.

scripts/extract_basket_cx_frame_level.py에서 이동 (V5 구조화 데이터 라벨링에서
이미 검증된 룰, plan_20260703_hsv_preview_align.md 참고). CH54 프리뷰 정렬 루프에서
재사용하기 위해 robovlm_nav/ 하위로 옮겨 sys.path 의존 없이 import 가능하게 함.
"""
import cv2
import numpy as np

# 회색 basket HSV 파라미터
S_MAX = 20      # basket: 엄격한 회색 (S<20)
V_MIN = 70
V_MAX = 230
# 공간 필터 (천장/바닥 제거)
TOP_CUT = 0.20      # 상단 20% 제거
BOTTOM_CUT = 0.68   # 하단 32% 제거
MIN_AREA_PX = 400   # 최소 연결 픽셀 수
BG_RATIO = 5.0      # 1위가 2위보다 이 배수 이상 크면 → 복도 배경으로 판단, 스킵


def detect_basket_cx(img_rgb: np.ndarray):
    """
    Returns (cx, cy, area_ratio, confidence) 또는 None (미검출).
    confidence: 0~1 (탐지 신뢰도)
    """
    H, W = img_rgb.shape[:2]
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    mask = (
        (hsv[:, :, 1] < S_MAX) &
        (hsv[:, :, 2] > V_MIN) &
        (hsv[:, :, 2] < V_MAX)
    ).astype(np.uint8) * 255

    mask[: int(H * TOP_CUT), :] = 0
    mask[int(H * BOTTOM_CUT):, :] = 0

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if n_labels <= 1:
        return None

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
            chosen = sorted_valid[1]
            conf = float(a2 / (a1 + a2))
        else:
            chosen = sorted_valid[0]
            conf = float(a1 / (a1 + a2))
    else:
        chosen = sorted_valid[0]
        conf = 1.0

    best_idx = chosen + 1  # connectedComponents는 0이 배경
    cx = centroids[best_idx][0] / W
    cy = centroids[best_idx][1] / H
    area_ratio = areas[chosen] / (H * W)

    return float(cx), float(cy), float(area_ratio), float(conf)

# -*- coding: utf-8 -*-
import json
import warnings
import sys
from pathlib import Path
import h5py
import numpy as np
import cv2

warnings.filterwarnings("ignore")
ROOT = Path("/home/minum/26CS/MoNaVLA")
sys.path.insert(0, str(ROOT))

ANN_PATH = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_pg2_cx.json"

# HSV 추출을 위한 파라미터 (extract_basket_cx_frame_level.py와 동일하게 설정)
S_MAX = 20
V_MIN = 70
V_MAX = 230
TOP_CUT = 0.20
BOTTOM_CUT = 0.68
MIN_AREA_PX = 400
BG_RATIO = 5.0

def detect_basket_cx(img_rgb):
    """
    HSV 및 connected component를 기반으로 바스켓의 cx, cy, area_ratio를 계산합니다.
    """
    H, W = img_rgb.shape[:2]
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # 회색 영역 마스크 생성
    mask = (
        (hsv[:, :, 1] < S_MAX) &
        (hsv[:, :, 2] > V_MIN) &
        (hsv[:, :, 2] < V_MAX)
    ).astype(np.uint8) * 255

    # 상/하단 불필요 영역 제거
    mask[: int(H * TOP_CUT), :] = 0
    mask[int(H * BOTTOM_CUT):, :] = 0

    # 연결 성분 분석
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if n_labels <= 1:
        return None

    # 면적 정렬 (배경 제외)
    areas = stats[1:, cv2.CC_STAT_AREA]
    valid = np.where(areas >= MIN_AREA_PX)[0]
    if len(valid) == 0:
        return None

    sorted_valid = valid[np.argsort(areas[valid])[::-1]]

    # 1순위 후보가 복도 배경 노이즈인지 체크하여 필터링
    if len(sorted_valid) >= 2:
        a1 = areas[sorted_valid[0]]
        a2 = areas[sorted_valid[1]]
        if a1 >= a2 * BG_RATIO:
            chosen = sorted_valid[1]  # 2순위 선택
        else:
            chosen = sorted_valid[0]  # 1순위 선택
    else:
        chosen = sorted_valid[0]

    best_idx = chosen + 1
    cx = centroids[best_idx][0] / W
    cy = centroids[best_idx][1] / H
    area_ratio = areas[chosen] / (H * W)

    return float(cx), float(cy), float(area_ratio)

def main():
    print(f"[START] {ANN_PATH} 로드 중...")
    with open(ANN_PATH, "r") as f:
        data = json.load(f)

    patched_count = 0
    total_frames = 0
    missing_hsv_eps = 0

    for ep_idx, ep in enumerate(data):
        h5_path = Path(ep["episode"])
        if not h5_path.exists():
            continue

        # 에피소드 내 프레임 중 HSV GT가 누락되었거나 None인 것이 있는지 파악
        need_patch = False
        for fr in ep["frames"]:
            if "cx_det_hsv" not in fr or fr["cx_det_hsv"] is None:
                need_patch = True
                break

        if need_patch:
            missing_hsv_eps += 1
            # H5 파일 열어서 비디오 이미지 로드
            try:
                with h5py.File(str(h5_path), "r") as f:
                    imgs = f["observations"]["images"][:]
            except Exception as e:
                print(f"  [ERROR] {h5_path} 로드 실패: {e}")
                continue

            for fr in ep["frames"]:
                total_frames += 1
                if "cx_det_hsv" not in fr or fr["cx_det_hsv"] is None:
                    fidx = fr["frame_idx"]
                    img_np = imgs[min(fidx, len(imgs)-1)].astype("uint8")
                    det = detect_basket_cx(img_np)
                    
                    if det is not None:
                        cx, cy, ar = det
                        fr["cx_det_hsv"] = round(cx, 4)
                        fr["cy_det_hsv"] = round(cy, 4)
                        fr["area_det_hsv"] = round(ar, 4)
                    else:
                        # 검출되지 않을 시 기본값/None 처리
                        fr["cx_det_hsv"] = None
                        fr["cy_det_hsv"] = None
                        fr["area_det_hsv"] = None
                    patched_count += 1

        if (ep_idx + 1) % 20 == 0:
            print(f"  [{ep_idx+1}/{len(data)}] 진행 중... 누적 패치된 프레임 수: {patched_count}")

    # 변경된 데이터를 다시 저장
    with open(ANN_PATH, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n[COMPLETE] 총 {missing_hsv_eps}개 에피소드에서 {patched_count}개 프레임 복구 완료.")
    print(f"[SAVE] {ANN_PATH} 업데이트 완료.")

if __name__ == "__main__":
    main()

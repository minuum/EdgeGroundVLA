#!/usr/bin/env python3
"""
임의의 그라운딩 세션(gnd_*.jsonl)에서 PG2가 fallback 처리한 프레임에
HSV 기반 classical detector(scratch/patch_hsv_annotations.py 동일 로직)를
재적용해 복구 가능한지 테스트.

목적: PG2 파인튜닝용 ground-truth를 HSV로 자동 보강할 수 있는지 검증.
PG2 self-labeling은 PG2 collapse 자체를 학습데이터에 포함시키는 순환참조라
HSV 같은 독립적인 검출기로 교차검증/대체 라벨링이 필요함.

⚠️ "진짜 복구" 분류도 100% 신뢰 불가 — 벽 얼룩/가구 등을 오탐하는 사례가
실제로 확인됨(S7 n=6, n=21). 출력된 genuine 리스트 중 최소 3~5장은
Read 툴로 프레임 직접 열어 cx 위치가 실제 바스켓과 맞는지 시각 검증할 것.

사용:
    python3 scripts/eval/test_hsv_recovery.py \
        --jsonl logs/grounding_sessions/<세션>.jsonl \
        --frames docs/v5/grounding_frames/<세션폴더>
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# patch_hsv_annotations.py와 동일 파라미터
S_MAX = 20
V_MIN = 70
V_MAX = 230
TOP_CUT = 0.20
BOTTOM_CUT = 0.68
MIN_AREA_PX = 400
BG_RATIO = 5.0


def detect_basket_cx(img_rgb: np.ndarray):
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

    sorted_valid = valid[np.argsort(areas[valid])[::-1]]

    if len(sorted_valid) >= 2:
        a1 = areas[sorted_valid[0]]
        a2 = areas[sorted_valid[1]]
        chosen = sorted_valid[1] if a1 >= a2 * BG_RATIO else sorted_valid[0]
    else:
        chosen = sorted_valid[0]

    best_idx = chosen + 1
    cx = centroids[best_idx][0] / W
    cy = centroids[best_idx][1] / H
    area_ratio = areas[chosen] / (H * W)
    return float(cx), float(cy), float(area_ratio)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="그라운딩 세션 jsonl 경로")
    ap.add_argument("--frames", required=True, help="frame_NNNN.jpg 디렉토리")
    args = ap.parse_args()

    records = [json.loads(l) for l in Path(args.jsonl).read_text().splitlines() if l.strip()]
    frame_dir = Path(args.frames)
    n_frames_with_image = len(list(frame_dir.glob("frame_*.jpg")))
    if len(records) > n_frames_with_image:
        print(f"⚠️  jsonl({len(records)}건)이 추출된 프레임({n_frames_with_image}장)보다 많음 — "
              f"앞 {n_frames_with_image}건만 검증, 라이브 세션이 계속 누적된 것으로 보임\n")
        records = records[:n_frames_with_image]

    fallback_idxs = [i + 1 for i, r in enumerate(records) if not r.get("has_bbox")]
    print(f"검증 대상 {len(records)}프레임, PG2 fallback {len(fallback_idxs)}건")
    print(f"HSV 재검출 테스트 시작...\n")

    genuine, suspect, nohsv = [], [], []
    for idx in fallback_idxs:
        fpath = frame_dir / f"frame_{idx:04d}.jpg"
        if not fpath.exists():
            continue
        img_bgr = cv2.imread(str(fpath))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        det = detect_basket_cx(img_rgb)
        if det is None:
            nohsv.append(idx)
            continue
        cx, cy, area = det
        # cx>0.85 & area<0.01 → 배경의 다른 회색 장비(우산 리플렉터/노트북 등)를
        # 오탐할 가능성이 높음. 실제 바스켓 위치(센터~좌측)와 불일치.
        if cx > 0.85 and area < 0.01:
            suspect.append((idx, cx, cy, area))
        else:
            genuine.append((idx, cx, cy, area))

    print(f"=== 결과 ===")
    print(f"진짜 바스켓으로 보이는 복구: {len(genuine)}/{len(fallback_idxs)} ({len(genuine)/len(fallback_idxs)*100:.1f}%)")
    print(f"의심(배경 장비 오탐 추정, cx>0.85 & area<0.01): {len(suspect)}/{len(fallback_idxs)} "
          f"({len(suspect)/len(fallback_idxs)*100:.1f}%)")
    print(f"HSV도 완전 미검출: {len(nohsv)}/{len(fallback_idxs)} ({len(nohsv)/len(fallback_idxs)*100:.1f}%)")
    print()

    if genuine:
        print("--- 진짜 복구 (라벨로 사용 가능) ---")
        for idx, cx, cy, area in genuine:
            print(f"  n={idx:03d}  cx={cx:.3f}  cy={cy:.3f}  area={area:.4f}")

    if suspect:
        print(f"\n--- 의심 프레임(배경 오탐 추정, 수동 확인 필요) ---\n  {[s[0] for s in suspect]}")

    if nohsv:
        print(f"\n--- HSV도 완전 실패 (수동 라벨링 필수) ---\n  {nohsv}")

    valid_pg2 = sum(1 for r in records if r.get("has_bbox"))
    print(f"\n전체 유효율 (PG2 단독): {valid_pg2}/{len(records)} ({valid_pg2/len(records)*100:.1f}%)")
    print(f"전체 유효율 (PG2 + HSV 진짜복구만): {valid_pg2+len(genuine)}/{len(records)} "
          f"({(valid_pg2+len(genuine))/len(records)*100:.1f}%)")
    print(f"※ 의심 {len(suspect)}건은 병합에서 제외 — 배경 장비 오탐 가능성 높음")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Florence-2 `<CAPTION_TO_PHRASE_GROUNDING>` 출력(위치토큰 디코딩 후 픽셀박스) →
OWLv2와 동일한 (cx,cy,area,has_bbox) bbox 스키마로 변환.

plan_20261012_egvla2_florence2_unification.md §2.2용 — 하위 MLP/FiLM
(SubgoalSteeredMLP 등)이 그라운딩 백엔드(OWLv2/Florence-2)와 무관하게 동일
입력 텐서 계약을 받도록 하는 어댑터. scripts/l1_florence2_grounder.py:96-110의
cx/cy/area 변환 패턴을 재사용한다.

has_bbox 재정의 (§2.1에서 발견한 블로커 대응):
  Florence-2 phrase grounding은 score를 주지 않고 대상이 없어도 항상 박스를
  반환한다(부재 대조군 "microwave oven" 검출률 100%, OWLv2는 5.0% — CH74-2).
  "박스 유무"만으로는 has_bbox를 못 정하므로, plan §2.2 대안 (a)를 채택:
  박스 면적이 너무 작거나(노이즈) 너무 크면(과거 PG2에서 반복 관찰된
  "full-frame 붕괴" 패턴과 동일 증상 — 화면 전체/절반 이상을 덮는 박스는
  실제 타겟이 아니라 grounding 실패의 신호로 간주) has_bbox=False로 판정한다.
  임계값(MIN_AREA_FRAC/MAX_AREA_FRAC)은 초기 추정치 — 실측 캘리브레이션은
  TODO(§2.2 체크리스트 참고).
"""
MIN_AREA_FRAC = 0.0008   # 너무 작으면 노이즈 박스로 간주 (16x16 그리드 1셀 미만 수준)
MAX_AREA_FRAC = 0.55     # 화면 절반 이상 덮으면 PG2류 full-frame 붕괴로 간주


def florence2_output_to_bbox(parsed_boxes, W, H,
                              min_area_frac=MIN_AREA_FRAC, max_area_frac=MAX_AREA_FRAC):
    """post_process_generation()이 반환한 bboxes 리스트 → OWLv2와 동일 스키마.

    Args:
        parsed_boxes: [[x1,y1,x2,y2], ...] 픽셀 좌표 (post_process_generation 출력)
        W, H: 원본 이미지 가로/세로 픽셀
    Returns:
        dict(cx, cy, area, has_bbox) — cx/cy/area는 0~1 정규화, OWLv2의
        cx_det/cy_det/area_det와 동일 스케일.
    """
    if not parsed_boxes:
        return dict(cx=0.0, cy=0.0, area=0.0, has_bbox=False)

    best = max(parsed_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    x1, y1, x2, y2 = best
    area = (x2 - x1) * (y2 - y1) / (W * H)
    cx = (x1 + x2) / 2 / W
    cy = (y1 + y2) / 2 / H
    has_bbox = min_area_frac <= area <= max_area_frac
    return dict(cx=cx, cy=cy, area=area, has_bbox=has_bbox)


if __name__ == "__main__":
    # 단위 테스트: 정상 박스 / 노이즈 박스 / full-frame 붕괴 박스 3종
    cases = [
        ("정상(화면 중앙 30%)", [[400, 300, 880, 660]], 1280, 720),
        ("노이즈(1px급)", [[640, 360, 641, 361]], 1280, 720),
        ("full-frame 붕괴", [[0, 0, 1270, 715]], 1280, 720),
        ("박스 없음", [], 1280, 720),
    ]
    for name, boxes, W, H in cases:
        r = florence2_output_to_bbox(boxes, W, H)
        print(f"{name:20s} -> cx={r['cx']:.3f} cy={r['cy']:.3f} area={r['area']:.4f} has_bbox={r['has_bbox']}")

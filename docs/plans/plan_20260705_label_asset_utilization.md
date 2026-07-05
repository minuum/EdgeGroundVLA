# 셀프라벨링 자산 활용 — 벤치마크 고정 + IoU 비교 + OWL 전용 헤드

> 작성일: 2026-07-05
> 상태: 승인됨 (사용자: "너 추천 순서대로 ㄱㄱ") — B → C → A 순서로 진행
> 배경: 296프레임(5모델 O/X + 객체없음 79) + 72프레임(정답 bbox 좌표) 사람 라벨 확보.
> OWL-v2 threshold 0.25에서 오탐 0%/정탐 95.3% 확정 (owlv2_threshold_roc.py).
> 남은 질문: "메인 그라운더도 OWL로 단일화 가능한가" — 유일한 장벽은 헤드가 PG2 bbox
> 분포로 학습됐다는 것. 이건 재학습으로 제거 가능한 제약.

---

## 리서치 요약

- `docs/v5/bbox_truth_mini.json` — 72 annotations, 18 에피소드 전부 로컬
  `ROS_action/mobile_vla_dataset_v5/*.h5`와 매칭 확인. 필드: `bbox_xyxy_norm`(사람 정답),
  `target_visible`(True 56 / partial 16), `coarse_position`(C38/L18/R16), `frame_idx`, `episode`
- `docs/v5/hsv_owlv2_preview_20260704/human_labels.json` — 296프레임, 모델별 O/X + no_target 79
- `docs/v5/hsv_owlv2_preview_20260704/owlv2_scores.json` — OWL confidence 296개
- Step2 레시피 (`scripts/test_v5_bbox_nav_step2.py`): `bbox_dataset.json`(45ep/794f,
  frames[].{cx,cy,area,has_bbox,gt_class}) → WINDOW=3 bbox history + 16×16 grayscale → MLP(8-class)

## B. 고정 그라운딩 벤치마크 (1단계)

**신규**: `scripts/eval/grounding_benchmark.py` + 결과 `docs/v5/grounding_benchmark/results.json`

임의 그라운더(callable: image→{cx,cy,area,has_bbox,x1..y2,score?})를 3지표로 채점:

1. **정확도(있음)**: 296셋 중 객체있음 217프레임 — 모델 bbox cx가 사람 O/X 라벨과 일치 여부는
   이미 라벨로 확정 → 신규 모델은 "사람 O 받은 프레임들의 합의 cx"(라벨 O인 모델들의 cx 중앙값)
   대비 |Δcx|<0.15면 정답으로 자동 채점
2. **오탐(없음)**: no_target 79프레임에서 has_bbox=True 비율 (score 모델은 threshold 적용 후)
3. **IoU(정밀)**: truth_mini 72프레임 — 사람 정답 bbox와의 IoU 평균/중앙값, cx MAE

기존 5모델(H1/H2/Ow/PG/Kr)은 저장된 예측(meta.json)으로 즉시 채점, IoU는 C에서 채움.

## C. 72프레임 IoU 정밀 비교 (2단계)

**신규**: `scripts/eval/eval_iou_truth_mini.py`

- 72프레임을 H5에서 로드 → PG2 / OWL-v2(th 0.25) / Kosmos-refexp 3모델 실행
- 지표: IoU mean/median, cx MAE, coarse_position(L/C/R) 일치율, partial(16개) 별도 집계
- 결과를 B의 results.json에 병합 → 벤치마크 완성

## A. OWL 전용 헤드 학습 (3단계, 본론)

**신규**: `scripts/gen_bbox_dataset_owl.py` + `scripts/train_step2_owl_head.py`

1. `bbox_dataset.json`의 45 에피소드/794프레임과 **동일한 프레임**을 OWL-v2(th 0.25)로
   재-그라운딩 → `docs/v5/bbox_nav_owl/bbox_dataset_owl.json` (gt_class는 원본 복사,
   cx/cy/area/has_bbox만 OWL로 교체. 미검출 시 has_bbox=False + cx 0.5 유지 — PG2 관례 동일)
2. Step2와 **동일 레시피**(WINDOW=3, 16×16 image, MLP, 동일 split/seed 5개)로 학습
   — 차이는 입력 bbox 소스뿐. apples-to-apples 보장 위해 PG2 기반 기존 dataset으로도
   같은 코드로 재학습해 baseline 재실측 ([[feedback_apples_to_apples_baseline]])
3. 판정: OWL-헤드 PM이 재실측 PG2-헤드 PM 대비 -2%p 이내면 "메인도 OWL 단일화 가능" 결론

## 완료 기준 (DoD)

- [ ] B: grounding_benchmark.py — 5모델 즉시 채점 결과 저장
- [ ] C: 72프레임 IoU 3모델 비교, results.json 병합
- [ ] A-1: bbox_dataset_owl.json 생성 (794프레임, OWL th 0.25)
- [ ] A-2: 동일 레시피 5-seed 학습 — PG2 재실측 vs OWL 헤드 PM 비교표
- [ ] 결론 문서: docs/v5/grounding_benchmark/CONCLUSION.md (단일화 가능 여부 판정)

## 리스크

- 296셋 자동채점(합의 cx 기준)은 사람 O/X와 완전히 같지 않음 — 신규 모델용 근사 지표로만 사용
- Step2 원본 bbox_dataset의 그라운딩 소스와 OWL의 검출 특성 차이(예: 원거리 미검출)로
  has_bbox=False 프레임 비율이 달라질 수 있음 — 학습 전에 분포 비교 리포트 출력
- truth_mini는 partial 16개 포함 — IoU 집계에서 visible/partial 분리

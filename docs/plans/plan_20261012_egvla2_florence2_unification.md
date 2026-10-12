# 계획: EGVLA2 — Florence-2 단일 백본 통합 (OWLv2+Kosmos-2 → 1모델)

> 작성일: 2026-10-12 | 리서치: [research_20261012_egvla2_text_subgoal.md](research_20261012_egvla2_text_subgoal.md) §5, 팀 미팅 섹션
> 상태: **승인 대기 — 아직 코드 작성 안 함**

## 0. 범위 재확인

- 팀 합의(2026-10-12, 동환님): Two models(G,V): OWLv2(그라운딩)+Kosmos-2(비전) → **One model(G,V): Florence-2**
- 이 plan은 **백본 통합(그라운딩+비전을 Florence-2 하나로)** 범위만 다룸. 액션 헤드를 MLP→FiLM(FM)으로 바꾸는 것과 텍스트 서브골 결합은 별도 plan(`plan_20261012_egvla2_text_subgoal_film.md`)에서 다룸 — 사용자 요청대로 분리.
- 새로운 실기 데이터 수집은 이 plan 범위 밖(기존 V6 225ep/h5 재사용).
- Florence-2로 실시간 추론을 돌리는 성능(지연/리소스) 검증은 포함하되, 로봇 실기 재배포는 범위 밖(오프라인 비교까지).

## 1. 현재 구조 대비 변경점

### 1.1 현재 (Two-model)
```
이미지 ──▶ OWLv2(그라운딩 전용, zero-shot detection) ──▶ bbox(cx,cy,area,confidence,detected)
이미지 ──▶ Kosmos-2(비전 인코더, frozen+LoRA) ──▶ image_feat(256~1024d)
                                                        │
                                   bbox + image_feat ──▶ MLP/FiLM ──▶ action
```
두 개의 독립 백본을 각각 forward — 추론 시 2회 모델 호출.

### 1.2 변경 후 (One-model: Florence-2)
```
이미지 ──▶ Florence-2 (단일 백본)
             ├─ <CAPTION_TO_PHRASE_GROUNDING> 태스크 ──▶ 위치토큰 <loc_xxx> ──▶ bbox(cx,cy,area,has_bbox)
             └─ 비전 인코더 hidden state ──▶ image_feat
                                                        │
                                   bbox + image_feat ──▶ MLP/FiLM ──▶ action
```
- 핵심 질문(팀 TODO에 명시): 그라운딩 forward와 비전 인코딩 forward를 **동시에 1회로 할지, 여전히 2회(태스크별) 호출할지** — 아래 §2.3에서 두 경로 모두 검증 계획 포함.

### 1.3 bbox 스키마 정합 (필수 선행 작업, 리서치 §5)
Florence-2 원시 출력은 위치 토큰이라 OWLv2의 (cx,cy,area) 직접 출력과 형식이 다름. 기존 `scripts/l1_florence2_grounder.py:96-110` 패턴을 재사용해 변환 함수를 분리 모듈로 뺀다:

```python
# scripts/florence2_bbox_adapter.py (신규)
def florence2_output_to_bbox(parsed_boxes, W, H):
    """post_process_generation() 결과를 OWLv2와 동일한 (cx,cy,area,has_bbox) 스키마로 변환."""
    if not parsed_boxes:
        return dict(cx=0.0, cy=0.0, area=0.0, has_bbox=False)
    best = max(parsed_boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
    x1, y1, x2, y2 = best
    return dict(
        cx=(x1+x2)/2/W, cy=(y1+y2)/2/H,
        area=(x2-x1)*(y2-y1)/(W*H),
        has_bbox=True,
    )
```
이렇게 해서 하위 MLP/FiLM 레이어는 그라운딩 백엔드가 OWLv2든 Florence-2든 **동일 입력 텐서 계약**을 받는다 — 재학습 없이 백본만 교체 비교 가능.

## 2. 작업 단계

### 2.1 사전 점검 (CH59 결함 재발 확인) — ✅ 완료 (2026-10-12)
- `scripts/l1_florence2_grounder.py`를 V6 200프레임 샘플에 재실행. 결과(`docs/v5/detector/l1_florence2_grounder.json`):
  - **cx 포화 결함 재현 안 됨** — 예측 cx 최대값 0.9955(CH59 때는 0.559에서 막힘) → 언어부/디코딩 경로 결함은 해소된 것으로 확인.
  - **신규 블로커 발견**: 부재 대조군("microwave oven") 검출률 **100%**(OWLv2는 5.0%) — Florence-2의 `<CAPTION_TO_PHRASE_GROUNDING>`은 score를 주지 않고 대상이 없어도 항상 박스를 반환함. 즉 **"박스 유무"만으로는 `has_bbox`/`detected`를 판단할 수 없음**.
  - 영향: §1.3의 "boxes 리스트 empty → has_bbox=False" 매핑은 성립 안 함(리스트가 거의 항상 non-empty). `has_bbox` 재정의가 §2.2에서 선행 과제로 추가됨(아래).

### 2.2 bbox 어댑터 구현 (§2.1에서 발견된 블로커 반영)
- `scripts/florence2_bbox_adapter.py` 작성 (위 1.3 코드 기반), 단위 테스트: OWLv2 bbox_dataset_v6_owl.json의 동일 프레임에 대해 Florence-2 변환 결과와 cx/cy/area 상관관계(correlation) 확인.
- **`has_bbox` 재정의 (신규 선행 과제)**: Florence-2가 score 없이 항상 박스를 반환하므로, 아래 대안 중 하나로 간접 신뢰도를 만들어야 함:
  - (a) 박스 크기/종횡비 이상치 필터 — 터무니없이 크거나 화면 전체를 덮는 박스는 `has_bbox=False`로 간주
  - (b) `gray basket` 박스와 부재 대조군("microwave oven" 등) 박스의 **IoU/거리 비교** — 너무 가까우면(=아무 phrase에나 같은 영역을 찍으면) 신뢰도 낮음으로 판단
  - (c) 동일 프레임에 대해 OWLv2 `detected` 플래그를 당분간 **공동 신뢰도 게이트**로 병행 사용(Florence-2 bbox 좌표 + OWLv2 detected 플래그) — 가장 간단하지만 "One model" 목표와 어긋나므로 임시방편으로만 표기
  - 셋 중 어느 걸 쓸지는 V6 200프레임 기준 (a)/(b) 각각의 정밀도·재현율을 뽑아보고 결정 (§2.1 데이터로 바로 분석 가능 — 추가 모델 호출 불필요)

### 2.3 동시 vs 분리 forward 비교 (팀 TODO 항목)
- (A) 분리: `<CAPTION_TO_PHRASE_GROUNDING>` 태스크 forward 1회 + 비전 전용 forward 1회 (현재 Kosmos-2 방식과 동일한 패턴)
- (B) 동시: 그라운딩 forward의 encoder hidden state를 비전 특징으로 재사용(추가 forward 없음)
- 측정: 2가지 방식 각각의 (지연시간, bbox 품질, PM) — Jetson 등 실제 추론 환경 기준 `scripts/measure_sequential_vs_threaded_grounding_florence2.py` 선례 패턴 재사용

### 2.4 리소스 비교 (팀 TODO 항목)
- Florence-2(base) 단일 모델 vs 현재 OWLv2+Kosmos-2 조합: 추론 latency, GPU 메모리, (가능하면) Jetson 전력 비교
- 측정 스크립트는 기존 `scripts/ablate_preview_ft_v2.py`/`detector_backbone_sweep.py` 계열 패턴 재사용, 신규 비교 스크립트 1개만 추가

### 2.5 Stage 2 학습 재현
- bbox 어댑터로 변환한 Florence-2 bbox + Florence-2 비전 특징으로, 기존 `eval_subgoal_steering_extended_v6.py`의 MLP(concat/steered) 비교를 **그대로** 재실행(그라운딩 백본만 교체) — 새 학습 코드 작성 안 함, `BBOX_DIM`/`IMG_DIM` 입력 소스만 바뀜

## 3. 수정/생성될 파일

| 파일 | 변경 |
|---|---|
| `scripts/florence2_bbox_adapter.py` (신규) | 위치토큰→bbox 스키마 변환 |
| `scripts/compare_florence2_vs_owlv2_kosmos2_resource.py` (신규) | 리소스/지연 비교 |
| `scripts/eval_subgoal_steering_extended_v6.py` | bbox/image feature 소스를 Florence-2 어댑터 출력으로 교체하는 옵션 플래그 추가 (기존 OWLv2 경로는 보존, 분기만 추가) |
| `docs/v5/detector/florence2_v6_cx_sanity_check.json` (신규 출력) | CH59 결함 재발 확인 결과 |

## 4. 트레이드오프

- **장점**: 추론 시 모델 호출 2회→1회(동시 forward 채택 시) 가능성, 유지보수 포인트 단일화, 경량화 논문 방향과도 부합(모델 수 축소)
- **리스크**:
  - Florence-2 그라운딩 디코딩 경로에 CH59 결함이 남아있으면 전체 방향 재검토 필요 — §2.1을 가장 먼저 해야 하는 이유
  - "동시 forward"가 bbox 품질을 떨어뜨릴 가능성(태스크 프롬프트 없이 뽑은 hidden state가 그라운딩 전용 forward보다 부정확할 수 있음) — §2.3에서 PM으로 직접 검증 필요, 품질 저하 시 "분리 호출 유지"로 결론 날 수 있음(이것도 유효한 결과)
  - Florence-2(base/large) 선택에 따라 리소스 이득이 상쇄될 수 있음 — base 기준으로 우선 측정

## 5. 체크리스트 (승인 후 진행)

- [ ] 2.1 CH59 결함 재발 확인 (V6 200프레임)
- [ ] 2.2 bbox 어댑터 구현 + OWLv2 대비 상관관계 확인
- [ ] 2.3 동시 vs 분리 forward 비교 (지연+PM)
- [ ] 2.4 리소스 비교 (Florence-2 단일 vs OWLv2+Kosmos-2)
- [ ] 2.5 Stage 2 PM 재현 (Florence-2 bbox/비전 기준)
- [ ] 결과 CH73(또는 신규 챕터)에 반영 + 로컬호스트 링크

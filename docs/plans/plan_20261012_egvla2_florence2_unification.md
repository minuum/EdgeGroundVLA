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

### 2.3 동시 vs 분리 forward 비교 (팀 TODO 항목) — 학습서버 1차 결과 확보 (2026-10-12)
- (A) 분리: `<CAPTION_TO_PHRASE_GROUNDING>` 태스크 forward 1회 + 비전 전용 forward 1회 (현재 Kosmos-2 방식과 동일한 패턴)
- (B) 동시: 그라운딩 forward의 encoder hidden state를 비전 특징으로 재사용(추가 forward 없음)
- 측정 스크립트: `scripts/bench_grounder_vs_florence2.py` (신규, 학습서버/젯슨 공용 — `torch.cuda.is_available()`로 디바이스 자동 분기)
- **학습서버(GPU) 결과** (2026-10-12, n_iters=20):

  | 측정 | 레이턴시 |
  |---|---|
  | OWLv2 (forward+postprocess) | 529.6±52.6 ms |
  | Florence-2 그라운더(generate, beam=3) | **247.6±5.9 ms** |
  | Florence-2 비전 인코더만(별도 forward) | 93.7±0.9 ms |

  → 학습서버 기준 Florence-2 그라운더가 OWLv2보다 2배 이상 빠름(지금까지 CH59/67에서 본 "Florence-2가 느리다"는 인상은 전부 **젯슨** 측정이었음 — 환경별로 분리 보고 필요).
  → `_encode_image`가 `generate()` 내부에 포함된 커스텀 오버라이드로 확인됨 — 분리 호출 시 247.6+93.7=341.3ms, **(B) 동시 재사용 시 247.6ms로 약 27% 절감** 가능.

- **젯슨(Orin NX) 결과** (2026-10-12, monavla-f0 세션에서 로봇 접속해 실측, `transformers==4.45.2`는
  `post_process_grounded_object_detection` 없어 운영 서버와 동일한 `post_process_object_detection`으로
  교체해 측정):

  | 측정 | 레이턴시 |
  |---|---|
  | ① 현재 방식: OWLv2(forward+postprocess) | 1915.6±25.7 ms (논문 Table10의 1901.7ms와 거의 일치) |
  | ① + Kosmos-2 비전(~3%) 포함 총합 | ≈ 1960 ms |
  | ② Florence-2 분리 호출: 그라운더(generate, beam=3) | 367.6±3.9 ms |
  | ② Florence-2 분리 호출: 비전 인코더만(별도 forward) | 측정치 수신 중 깨짐(재확인 필요) |
  | ② 분리 호출 합계 | 575.2 ms |
  | ③ Florence-2 **fused**(기존 `vision_tower.forward_features_unpool` 대신 `_encode_image` 직접 1회 호출해 bbox+비전특징 동시 획득) | **371.1±2.7 ms** (plain generate 367.6ms와 거의 동일 — 재사용에 추가 비용 없음) |

  → **중요 구현 디테일**: 공개 `generate()`는 내부에서 비전 특징을 계산하지만 지역변수로 버려져서 호출부에서
  재사용 불가능 — `_encode_image()`를 **직접 1회 호출**해서 (1) bbox 생성용 디코더 입력과 (2) action head용
  비전 특징을 동시에 얻는 방식으로 실제 fused 버전을 구현·검증함. 재사용 가능한 vision feature shape:
  `(1, 577, 768)`.
  → (정정 전 기재 — 위 비교는 dtype 불일치, 아래 **규격 벤치(§2.3-b 최종)** 참고)

- **젯슨 규격 벤치 최종 결과** (`scripts/bench_egvla2_unified.py --tag jetson`, 2026-10-12, 실제 V6 프레임 1장,
  warmup 10/iters 50, A=현재 프로덕션 순차(OWLv2→Kosmos-2), B=F2 분리 호출, C=F2 fused 1회):

  | dtype | A 현재(전체) | A-OWLv2 | A-Kosmos2비전 | B F2분리 | C F2 fused | fused vs A | fused vs B |
  |---|---|---|---|---|---|---|---|
  | **fp32 (메인 배포)** | 1970.98ms | 1914.39ms | 56.49ms | 1404.37ms | **817.36ms** | **2.41×** | 1.72× |
  | fp16 (보조) | 1017.97ms | — | — | 582.16ms | 372.11ms | 2.74× | 1.56× |

  결과 JSON: `docs/v5/detector/bench_egvla2_unified_jetson_fp32.json`, `..._jetson_fp16.json`
  **학습서버(GB10) 동일 규격** (`..._server_fp32.json`/`_fp16.json`, 2026-10-12):

  | dtype | A 현재(전체) | A-OWLv2 | A-Kosmos2비전 | B F2분리 | C F2 fused | fused vs A | fused vs B |
  |---|---|---|---|---|---|---|---|
  | **fp32 (메인 배포)** | 491.6ms | 433.0 | 58.6 | 315.1ms | **180.0ms** | **2.73×** | 1.75× |
  | fp16 (보조) | 360.3ms | 300.5 | 59.8 | 275.4ms | 201.1ms | 1.79× | 1.37× |

  → 서버에서는 F2가 fp16(201ms)이 fp32(180ms)보다 오히려 느림(디코딩 호출 오버헤드 지배) — 메인이 fp32라 영향 제한적.
  → 계산량(GFLOPs): A 1277 / B 847 / C 467 (`docs/v5/detector/egvla2_model_cost_breakdown.json`, CH74-8).

  → **메인 배포 조건(fp32) 기준 결론**: Florence-2 fused는 현재 방식(OWLv2+Kosmos-2) 대비 **2.41배** 빠름 —
  5.3배(dtype 불일치)도 994ms 비교(994→371, 2.7배, 하니스 fp16 강제)도 아닌, 동일 dtype·동일 하니스로 처음
  확정된 수치. fp16끼리 비교(2.74배)와 비슷한 비율대라 dtype보다 fused 구조 자체의 이득이 더 큰 비중임을
  시사. "OWLv2가 젯슨에서 유독 느리다"는 기존 관찰(CH59/67, 9/22 3종 비교)과는 별개로, 이번 수치가 가장
  신뢰할 수 있는 공식 비교값.
  → **fused(C)의 bbox 정확도는 분리(B)와 별도로 재측정할 필요가 없음을 코드+실측으로 확인** (2026-10-13,
  `scripts/verify_fused_bbox_equivalence.py`, 젯슨): Florence-2 공개 `generate()`의 내부 구현이 정확히
  `_encode_image() → _merge_input_ids_with_image_features() → language_model.generate(input_ids=None,
  inputs_embeds=..., 동일 kwargs)` 순서이고, fused 래퍼도 같은 함수를 같은 인자로 수동 호출할 뿐이라 bbox
  디코딩 경로가 구조적으로 100% 동일(결정적 빔서치, `do_sample=False`). 실제 V6 프레임 15장(서로 다른 5개
  에피소드 × 3구간)으로 B/C의 `(cx,cy,area,has_bbox)` 출력을 직접 비교 → **15/15 bit-exact 일치**. 따라서
  아래 분리(B) 기준 정확도 수치는 fused(C)에도 그대로 적용됨.
  → **정확도 검증은 기존 데이터로 이미 완료돼 있었음** — `docs/v5/detector/v6_phrase_grounding_human_labels.json`
  (OWL 실패 프레임 위주로 층화 샘플링된 사람 검증 329개)을 재집계:

  | | OWLv2 | Florence-2 |
  |---|---|---|
  | ok | 126 (38.3%) | **329 (100%)** |
  | ng | 203 | 0 |

  OWL이 실패한 203개 프레임을 Florence-2가 전부 회수(both_ok=126, f2_only=203, owl_only=0, both_fail=0) —
  CH-그라운딩 위키 요약의 "재현율 84.96%(실기)~100%(사람검증 329개)" 수치의 출처와 동일 데이터. 단, 표본이
  OWL 실패쪽에 의도적으로 편향돼 있어 "Florence-2 전체 정확도 100%"로 일반화하면 안 되고, **"OWL이 놓치는
  프레임을 Florence-2가 회수하는 능력"** 지표로 해석. 좌표 정밀도(cx MAE 등 연속값 비교)는 `florence2_backbone.json`
  (Florence-2 cx_mae 0.0015 vs Kosmos-2 0.002, 비전 백본 기준)로 별도 확보돼 있음 — OWLv2와의 cx MAE 직접
  비교는 아직 없으나, ok/ng 판정 결과와 종합하면 그리드 양자화 우려는 현재 근거로는 기각 가능(과거 Kosmos-2
  free-gen 변환 결함과 달리 `<CAPTION_TO_PHRASE_GROUNDING>` 방식은 CH59 결함도 재현 안 됨, §2.1).

### 2.4 리소스 비교 (팀 TODO 항목)
- Florence-2(base) 단일 모델 vs 현재 OWLv2+Kosmos-2 조합: 추론 latency(§2.3로 1차 확보), GPU 메모리, (가능하면) Jetson 전력 비교
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
  - ~~"동시 forward"가 bbox 품질을 떨어뜨릴 가능성~~ → **기각** (2026-10-13): fused는 생성 경로가 B와 코드 레벨로 동일, 15/15 bit-exact 검증 완료 (§2.3-b 하단, `scripts/verify_fused_bbox_equivalence.py`)
  - Florence-2(base/large) 선택에 따라 리소스 이득이 상쇄될 수 있음 — base 기준으로 우선 측정

## 5. 체크리스트 (승인 후 진행)

- [ ] 2.1 CH59 결함 재발 확인 (V6 200프레임)
- [x] 2.2 bbox 어댑터 구현 + OWLv2 대비 검증 — `scripts/florence2_bbox_adapter.py`, V6 15프레임 cx 상관 0.991 / has_bbox 일치 15/15 (면적 기반 휴리스틱, 임계값 캘리브레이션은 백로그)
- [x] 2.3-a 학습서버(GPU) 지연 벤치 — Florence-2 247.6ms < OWLv2 529.6ms, 동시 재사용 시 27% 절감
- [x] 2.3-b 젯슨 지연 벤치 — `bench_egvla2_unified.py` 규격 벤치로 최종 확정: **fp32(메인) A=1970.98ms vs C(fused)=817.36ms → 2.41×**, fp16(보조) A=1017.97ms vs C=372.11ms → 2.74× (2026-10-12). 이전 5.3×(dtype 불일치)/2.7×(994ms, fp16 강제 하니스) 기재는 본 수치로 대체.
- [x] 2.3-c bbox 정확도 비교 — 기존 V6 사람검증 329개 재집계로 완료: OWL ok 38.3% vs Florence-2 ok 100%(OWL 실패 203개 전부 회수), 그리드 양자화 우려는 §2.1 결과로 기각
- [x] 2.3-d fused(C) bbox 정확도가 분리(B)와 동일한지 검증 — 코드 구조상 동일 + 실측 15/15 bit-exact 일치 (2026-10-13, `scripts/verify_fused_bbox_equivalence.py`). §2.3-c 수치가 fused에도 그대로 적용됨
- [x] 2.4 리소스 비교 — 지연(서버·젯슨, fp32/fp16), 파라미터 458M→231M, 가중치 메모리, GFLOPs 1277→467 완료 (CH74-3/74-8). 젯슨 전력·실측 메모리는 미측정
- [x] 2.5 Stage 2 PM 재현 (Florence-2 bbox/비전 기준, 10-seed, 2026-10-13) — steered_reg 78.38±0.96 / steered 78.41±0.74 / concat 76.76±1.08 / concat_small 76.70±1.05. steered>concat 유의(p=1.6e-5) 유지, steered_reg 추가 이득 없음(vs steered p=0.79). CH73 OWLv2+16×16(72.37)보다 높으나 입력(768-d 풀링 특징)이 달라 그라운더 효과로 분리 불가. `summary_florence2_v6.json`
- [ ] 결과 CH73(또는 신규 챕터)에 반영 + 로컬호스트 링크

- [x] 후속① 그라운더 단독 효과 분리 (2026-10-13) — OWLv2 bbox + 동일 F2 특징: steered_reg 78.38% = F2 bbox 78.38%. CH73 대비 상승은 특징 효과. `summary_florence2_v6_owlbbox.json`
- [x] 후속② has_bbox 임계값 캘리브레이션 (2026-10-13) — 면적·가장자리·종횡비 조합 최고 balanced acc 0.73, 기하 규칙으로 부재 판별 불가 → 현행 유지, 신뢰도 신호(토큰 확률) 필요. `f2_has_bbox_calibration.json`

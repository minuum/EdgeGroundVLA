# Plan — "근본적인 VLA"로 가는 다음 단계

> 작성: 2026-06-22 · 상태: **검토 대기 (승인 전 구현 금지)**
> 동기: instruction-grounding 절충안(`plan_20260621_instruction_grounding.md`) 적용 후, 사용자가 "근본적인 VLA가 되게 해줘"라고 요청.
> 연관: `docs/v5/research_story.html` CH38(VLA 레퍼런스 아키텍처 비교, 38-5에서 방향 신호 ~0 정정), `plan_20260621_groundingdino_vs_pg2.md` §13-15(PG2 attention/hidden-state 측정)

---

## 0. 출발점 — 이미 알고 있는 것 (재인용, 재측정 안 함)

| 사실 | 근거 |
|---|---|
| Google-robot 백본: text attention 0.0000%, 구조적으로 죽음 | Exp15, `scripts/measure_attention.py` |
| base PG2: text attention 40~98%(26 layer 전체) — 구조적으로 살아있음 | `scripts/measure_attention_pg2.py`, CH38-1 |
| PG2 zero-shot grounding은 객체 일반화됨(바스켓 외 5종 5/5) | `plan_20260621_instruction_grounding.md` §7 |
| PG2에 "go left/right/forward" 류 **방향 텍스트**를 줘도 생성 결과(`detect`, `nav_*`)가 거의 동일 — 방향 신호는 출력 레벨에서 ~0 | CH38-5, `pg2_direction_output_test.json` |
| 현재 action head(ActionMLP 등)는 bbox 좌표/면적 + image feat만 입력 — 텍스트 입력 자체가 없음 | `plan_20260621_instruction_grounding.md` §1 |
| `research_story.html`이 정의한 "진짜 VLA" = "find the X" → grounding → navigation, 텍스트만 바꿔 같은 시스템이 다른 목표로 가는 것 | research_story.html:9167-9170 |
| 방금 발견: `PG2Grounder.run()`의 4개 후처리 필터(`area>0.9`, `area<0.01`, `cy<0.35`, x-full-width)가 **바스켓 전용으로 박혀있어** 다른 객체의 정상 bbox를 false negative 처리함(사과 cy=0.344로 탈락) | 이번 세션 soda 실측 |

---

## 1. "VLA-ness" 등급 재정의 — 이번 plan이 어디를 겨냥하는지

| Tier | 정의 | 현재 상태 |
|---|---|---|
| **T0** | 언어가 **목표(대상 객체) 선택**에 영향 — "basket" vs "mug" 등 | ✅ 구현됨(instruction-grounding), 단 필터 버그로 일부 객체 false negative |
| **T1** | T0 + 임의 객체에 대해 **신뢰성 있게** 동작(객체별 후처리 필터/임계값 보정) | ❌ 미완 — 필터가 바스켓 전용 |
| **T2** | 언어가 **경로/태도**(left/right/slow 등)에 영향 — 같은 목표, 다른 방식으로 도달 | ❌ 미검증, CH38-5가 PG2 출력 레벨에서는 신호가 거의 없음을 보임 |
| **T3** | 단일 통합 아키텍처(grounding+action 분리 없음), end-to-end 학습 | ❌ Exp01~16에서 시도 후 폐기(백본 구조적 붕괴) — 이 plan에서 재시도 안 함 |

**이번 plan의 목표: T0 → T1을 확정하고, T2가 "지금 백본으로 현실적으로 가능한지"를 기존 데이터로 저비용 검증한다.** T3(완전 통합 재학습)은 이미 실패 경험이 있고 비용이 크므로, 이 plan에서는 다루지 않고 T2 검증 결과에 따라 별도 plan으로 분리한다.

---

## 2. Step A — T1 확정: 객체별 그라운딩 필터 보정 (즉시, 저위험)

### 문제
`stage2_v2_inference_server.py:332-348`의 4개 필터가 바스켓 형태/위치 가정에 맞춰 하드코딩됨:
```python
if area > 0.9: ...          # full-frame collapse
elif area < 0.01: ...       # tiny noise
elif cy_val < 0.35: ...     # "바구니는 상단에 없다" 가정 — 다른 객체엔 부적합
elif x1 < 0.02 and x2 > 0.98: ...
```
사과(`cy=0.344`)처럼 화면 중상단에 작게 잡히는 객체는 정상 bbox인데도 걸러짐.

### 변경 방향
- `area>0.9`(full-frame 환각 차단), `x-full-width`(가로 전체 차단)는 **객체 무관 보편 규칙**으로 유지.
- `cy_val < 0.35`와 `area < 0.01`은 **바스켓 전용 휴리스틱** → §9(`plan_20260621_instruction_grounding.md`)에서 이미 만든 `GOAL_AREA_MAP`과 같은 패턴으로 **phrase별 오버라이드 가능한 파라미터**로 전환.
  - 기본값(바스켓 등 미등록 phrase)은 현재 값(`cy<0.35`, `area<0.01`) 유지 → 하위호환.
  - `configs/goal_area_map.json`과 별도로(혹은 같은 파일 확장으로) `min_cy`/`min_area` 같은 필드를 phrase별로 둠.
- 변경 파일: `robovlm_nav/serve/stage2_v2_inference_server.py` (PG2Grounder.run 필터 부분만), 필요 시 `configs/goal_area_map.json` 스키마 확장.

### 검증
- soda에서 서버 내리고(`go.sh --stop`) 단독 스크립트로 사과/머그/콜라캔/의자/콘 5종 재측정(직접 로드 — 운영 서버 동시 실행 금지, [[soda-pg2-concurrent-load-crash]] 메모리 준수) → 5종 모두 정상 bbox 인식되는지 확인.
- 통과 후에만 서버 재기동 + `/predict` 재테스트.

이 Step은 **새 학습 없음, 순수 코드 수정**이라 위험 낮음 — 승인되면 바로 구현 가능.

---

## 3. Step B — T2 저비용 사전검증: 기존 V5 데이터로 "방향 신호가 실제로 있는가" 확인

### 왜 새 데이터 수집부터 안 하는가
CH38-5에서 이미 PG2에 "go left"/"go right" 텍스트를 줘도 **생성 출력**이 거의 동일함을 확인했다. 그런데 이건 "텍스트 프롬프트로 직접 지시했을 때"의 결과이고, **이미 존재하는 V5 H5 에피소드들이 자연 주행 중 PG2가 보는 이미지(바스켓이 좌/중/우 어디 있는지)에 따라 hidden state가 갈라지는지**는 아직 따로 측정한 적이 없다. 이건 완전히 다른 질문이고, 답이 다를 수 있다.

### 확인된 기존 데이터
`ROS_action/mobile_vla_dataset_v5/`의 150개 에피소드 파일명에 시작위치×경로방향 라벨이 이미 있음:
```
center_straight ×21, center_left ×15, center_right ×15
left_straight   ×21, left_left   ×27, left_right   ×23
right_straight  ×34, right_left  ×33, right_right  ×31
```
**새 데이터 수집 없이** 이 라벨을 direction ground-truth로 재사용 가능.

### 검증 방법 (신규 스크립트, 학습 없음 — frozen probe만)
1. 각 에피소드의 대표 프레임(예: 중간 프레임) 이미지에 대해 PG2 `output_hidden_states=True`로 **bbox 영역 직전의 hidden state**(현재 grounding에서 실제로 쓰는 표현)를 추출.
2. 9개 그룹(시작위치×방향) 라벨로 **선형 probe**(로지스틱 회귀, frozen — PG2/액션헤드 둘 다 변경 없음) 분리 가능한지 확인.
   - Exp54가 "basket을 보는지"를 frozen probe(96.6%)로 증명한 것과 동일 방법론.
3. **결정 게이트**:
   - 분리 잘 됨(예: 방향 라벨 probe acc ≫ chance) → PG2 hidden state에 방향 신호가 실제로 있다는 뜻 → T2로 가는 길이 열림(다음 plan에서 "hidden state를 action head 입력에 추가 + 기존 라벨로 head만 재학습" 검토, **새 데이터 수집 불필요**).
   - 분리 안 됨(chance 수준) → CH38-5 결론이 hidden-state 레벨에서도 재확인됨 → **이 백본으로는 T2가 막혀있다**는 결론을 research_story.html에 추가하고, T2는 백본 교체(π0/TinyVLA류 액션 전문가 도입) 없이는 불가능하다고 명시 → 이 경우 새 데이터 수집/재학습에 투자하지 않는 게 맞다는 근거가 됨.

### 변경 파일 (신규, 기존 코드 수정 없음)
- `scripts/eval/probe_v5_direction_hidden_state.py` — 추출 + probe 학습/평가, GB10 로컬 실행(soda 불필요, 학습 데이터에 직접 접근 가능한 머신에서).

---

## 4. 실행 순서 및 위험도

| 순서 | 내용 | 위험 | 되돌릴 수 있는가 |
|---|---|---|---|
| 1 | Step A 코드 변경 (필터 파라미터화) | 낮음 — 순수 로직 수정, 새 학습 없음 | 쉬움 (git revert) |
| 2 | Step A soda 검증 (서버 내리고 단독 테스트 → 재기동) | 낮음 — 단, **서버 동시 로드 절대 금지**([[soda-pg2-concurrent-load-crash]]) | 서버 재기동으로 복구 |
| 3 | Step B probe 스크립트 (GB10 로컬, 기존 V5 데이터) | 없음 — 읽기 전용 분석, 학습/배포 없음 | 해당 없음 |
| 4 | Step B 결과에 따라 후속 plan 분기 | — | — |

**이번 plan은 1~3까지만 다룬다.** 4(T2 본 구현 또는 "포기하고 T1에 머문다"는 결론 문서화)는 Step B 결과를 보고 별도 plan으로 작성.

---

## 5. 완료 기준 (이번 plan 범위)

- [x] Step A: `PG2Grounder.run()` 필터 phrase별 파라미터화(`configs/ground_filter_map.json`), GB10 로컬 5/5 통과
- [x] Step A soda 배포: 서버 내리고 pull → 단독 검증 → 재기동
- [x] Step B: `probe_v5_direction_hidden_state.py` 작성·실행, 9-class(또는 3-class 방향만) probe 정확도 측정
- [x] Step B 결과를 `research_story.html`에 CH39로 기록
- [ ] 다음 plan 분기점 명시(T2 구현 plan)

### Step B 결과 (2026-06-22) — 강한 긍정 신호

V5 실주행 220개 에피소드(`target_{start}_{direction}_path` 패턴), 중간 프레임 1장씩,
PG2 grounding 프롬프트("detect gray basket", **텍스트 고정**, 이미지만 다름) 마지막
hidden state(2304-dim) → frozen 선형 probe(5-fold CV):

| probe | n_class | chance | CV acc | chance 대비 |
|---|---|---|---|---|
| direction(좌/직진/우) | 3 | 0.333 | **0.900 ± 0.031** | 2.70x |
| start(중앙/좌/우 출발) | 3 | 0.333 | 0.991 ± 0.018 | 2.97x |
| start×direction | 9 | 0.111 | 0.923 ± 0.027 | 8.30x |

**해석**: CH38-5("go left/right 텍스트를 줘도 PG2 생성 출력이 거의 동일")와 모순되지 않음 —
그건 "텍스트로 직접 지시했을 때 출력이 안 바뀐다"는 얘기였고, 이건 "텍스트는 고정인데
**이미지(실제 주행 장면)** 가 다를 때 hidden state가 갈리는가"라는 다른 질문이었다.
결과는 명확히 **갈린다** — 즉 PG2의 내부 표현(hidden state)에는 현재 action head가
쓰는 bbox 좌표/면적보다 훨씬 풍부한 장면 정보(경로방향과 상관된 시각 정보)가 이미
들어있다.

**의미**: T2(언어가 경로에 영향)로 가는 길은 "PG2에 방향을 텍스트로 명령"하는 방식이
아니라 — **action head의 입력을 bbox 좌표 대신(또는 추가로) PG2 hidden state로
바꾸고, 기존 V5 220개 에피소드 라벨로 head만 재학습**하는 방식이 유망하다는 근거.
**새 데이터 수집 불필요** — 라벨도 이미지도 이미 있음.

산출: `docs/v5/attention_analysis/v5_direction_probe.json`

**다음 plan (이번 plan 범위 밖)**: action head를 hidden-state 입력으로 바꿔 실제로
재학습했을 때 PM/closed-loop 성능이 오르는지 확인 — 이건 "새 학습"이 들어가므로
별도 plan으로 분리해 사용자 승인 받음.

### Step A soda 검증 결과 (2026-06-22)

| 객체 | phrase | has_bbox | 비고 |
|---|---|---|---|
| 콜라캔 | red coke can | ✅ True | area=0.124, 안정적 |
| 의자 | chair | ✅ True | area=0.385, 안정적 |
| 콘 | orange cone | ✅ True | area=0.149, 안정적 |
| 사과 | green apple | ❌ False | **필터 버그 아님** — full-frame 환각(`<loc0000>...<loc1006><loc1020>`), area>0.9 필터가 정상 차단. soda(Jetson)에서 5회 반복 모두 동일(결정적). |
| 머그컵 | blue mug | ❌ False | 사과와 동일 — full-frame 환각, area>0.9 필터 정상 동작 |

**추가 발견(범위 밖, 기록만)**: 같은 이미지·phrase에 대해 GB10에서는 valid box(또는 5/5 hit 기존 평가)가 나왔는데 soda(Jetson Orin)에서는 일관되게 full-frame 환각이 나옴 — bf16 연산이 하드웨어(GB10 vs Jetson Orin)에 따라 다른 결과로 갈리는, 신뢰도가 낮은(low-confidence) 객체에서 나타나는 현상으로 추정. Step A가 고치려던 "필터가 정상 박스를 거부하는" 문제와는 다른 종류의 문제(애초에 정상 박스가 안 나옴) — 별도 이슈로 분리, 이번 plan 범위 밖.

---

## 6. 트레이드오프 / 메모

- Step A는 사용자가 바로 승인해도 안전하다고 판단(새 학습 없음, 범위 작음).
- Step B는 "방향 신호가 진짜 있는지"를 가장 싸게 알아보는 방법이라, **새 데이터 수집(여러 객체×여러 방향 조이스틱 수집, CH38-3에서 언급된 로드맵)보다 먼저** 해야 한다 — Step B가 실패하면 그 수집 자체가 헛수고가 될 수 있음.
- T3(완전 통합 1개 모델, π0/TinyVLA처럼)는 이번 세션 비교 분석(CH38)에서 이미 "백본 교체 없이는 불가능"이라는 잠정 결론이 있었음 — 이 plan은 그 결론을 뒤집으려는 게 아니라, **T3까지 안 가도 어디까지 "VLA답게" 만들 수 있는지**를 확정하는 게 목적.

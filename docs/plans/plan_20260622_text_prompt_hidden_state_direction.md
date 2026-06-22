# Plan — 텍스트로 명령한 방향이 PG2 hidden state에 들어가는가 (CH38-5의 hidden-state판)

> 작성: 2026-06-22 · 상태: **승인됨 — 구현 진행**
> 동기: 사용자 요청 — "에피소드 이름(방향)이 instruction으로 들어가는지" 조합 실험. 명확화 결과: grounding prompt를 경로말(left/right)로 바꿔 hidden state를 추출하고, CH39 Step B(이미지로 갈리는 방향 신호 90%)와 비교하는 실험으로 확정.

---

## 0. 출발점 — 이미 아는 것

| 사실 | 근거 |
|---|---|
| CH38-5: "detect gray basket on the left/right" 같은 방향 텍스트를 줘도 **생성 출력**(bbox)이 거의 안 바뀜 | 4개 세션 실측, `pg2_direction_output_test.json` |
| CH38-4: 같은 방향 텍스트 변형의 **hidden state 코사인거리** 0.029~0.043(객체 차이 0.363~0.441의 1/10) — "약하지만 0은 아님" | `scripts/measure_hidden_state_pg2.py`, 세션 4장 사진만 |
| CH39 Step B: 텍스트는 고정("detect gray basket")인데 **이미지**(실제 주행 장면)가 다르면 hidden state가 방향별로 90~99% 분리됨 | `probe_v5_direction_hidden_state.py`, V5 220프레임 |
| 위 둘은 "텍스트로 명령" vs "이미지로 관찰"이라는 다른 질문이었음 — 아직 **같은 이미지 세트에, 텍스트만 바꿔서 체계적으로** 비교한 적은 없음(CH38-4는 세션 4장뿐) | — |

**이번 plan의 질문**: 같은 V5 이미지 220장에 대해 grounding prompt를 `"detect gray basket"` / `"...on the left"` / `"...on the right"`로 바꾸면, hidden state가 **이미지의 실제 방향**(CH39처럼 90%)을 따라가는가, 아니면 **prompt가 말한 방향**을 따라가는가, 아니면 둘 다 무시하는가?

---

## 1. 실험 설계

### 1-1. 데이터 (새 수집 없음)
CH39 Step B와 동일한 220개 V5 에피소드 중간 프레임. 단, **3가지 prompt 변형**으로 각각 hidden state 추출:
- P0: `"detect gray basket"` (기존 CH39 캐시 재사용 가능)
- P1: `"detect gray basket on the left"`
- P2: `"detect gray basket on the right"`

같은 이미지에 P0/P1/P2 3개 hidden state가 생긴다(220×3 = 660회 추출, 속도는 이미 §2에서 확인한 배치 방식으로 빠름 — 전체 30분 이내 예상).

### 1-2. 분석 — 3가지 probe를 따로 학습
1. **이미지 방향 vs P0 hidden state** — CH39 재확인용(이미 있음, 재추출 불필요).
2. **이미지 방향 vs P1/P2 hidden state** — "프롬프트가 틀린 방향을 말해도 이미지의 진짜 방향이 여전히 읽히는가?"(예: 우회전 이미지에 "on the left"라고 prompt를 줬을 때도 hidden state가 우회전으로 분리되는지)
3. **prompt 종류(P0/P1/P2, 라벨로 취급) vs hidden state** — "같은 이미지인데 prompt만 바꿔도 hidden state가 그 자체로 분리되는가?"(분리되면 prompt 정보가 일부 들어간다는 뜻, CH38-4의 코사인거리 결과와 같은 방향이지만 N=220으로 통계적으로 더 안정적)

### 1-3. 핵심 비교표 (산출 목표)
| 질문 | probe 정확도 | chance |
|---|---|---|
| 이미지 방향 (P0, 기존) | (CH39 재인용: 90.0%) | 33.3% |
| 이미지 방향 (P1 prompt 고정 "on the left") | ? | 33.3% |
| 이미지 방향 (P2 prompt 고정 "on the right") | ? | 33.3% |
| prompt 종류(P0/P1/P2) 자체 | ? | 33.3% |

이 표가 "어떤 feature가 key고 measurement가 높은지"에 대한 직접적인 답이 된다 — 이미지 방향 신호가 prompt를 압도하면 "그라운딩은 보는 대로 본다(명령 무시)"가 hidden-state 레벨에서도 정량 확정.

---

## 2. 변경 파일

| 파일 | 작업 |
|---|---|
| `scripts/eval/probe_v5_direction_text_prompt.py` (신규) | 위 3-probe 실험, `probe_v5_direction_hidden_state.py` 구조 재사용 |
| `docs/v5/attention_analysis/v5_direction_text_prompt_probe.json` (신규) | 산출 |
| `docs/v5/research_story.html` | CH42로 결과 기록(긍정/부정 모두) |

## 3. 위험도

- 새 데이터 수집 없음(기존 220 에피소드), 새 학습 없음(frozen probe만), soda 미접촉.
- 추출량이 3배(660회)로 늘지만 배치 처리 기준 30분 이내 예상 — 시간 초과 시 사용자에게 재보고.

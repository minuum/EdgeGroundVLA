# Plan — 그라운딩/인식 품질 진단 + 윈도우 크기 ablation

> 작성: 2026-06-22 · 상태: **승인됨 — §1부터 구현**
> 동기: CH40 hidden-state head를 실제 V5 에피소드로 순차 테스트해보니(right_right 초반) hidden state 버전이 오히려 방향을 헷갈리는 구간이 보였다. 사용자가 명시: "실주행이 안 되는 핵심은 head 구조가 아니라 그라운딩/시멘틱 인식이 객체를 제대로 못 잡아서"라는 가설을 우선 검증하라고 지시. 동시에 윈도우 크기 ablation 등 head-level 탐색도 같은 plan에 포함하라고 함.

---

## 0. 출발점

| 사실 | 근거 |
|---|---|
| right_right 에피소드 1~2번째 프레임에서 replace 모드가 FWD+L(반대 방향)로 오예측 | 이번 세션 hub 실측(순차 프레임 테스트) |
| CH40: PM 75.9%→89%(add)인데 closed-loop SR은 96.6%→93%(add, -3.5%p) | CH40, plan_20260622_hidden_state_action_head.md |
| action head는 `--window` 인자로 2/4/8/16 조정 가능(`train_exp54_stage2_v2_action.py`, `train_hidden_state_action.py`) | 기존 코드 |
| 그라운딩 신뢰도 지표(has_bbox, area, cx/cy)는 이미 매 프레임 `bbox_dataset_full.json`에 기록되어 있음 | Step A/B 작업에서 확인 |

---

## 1. 그라운딩/인식 품질 진단 (우선)

### 질문
"방향전환 실패(오예측)가 그라운딩 신뢰도가 낮은 프레임에서 더 많이 일어나는가?" — head 구조(bbox vs hidden state)와 무관하게, **애초에 객체를 잘 못 잡은 프레임에서 모든 모드가 같이 흔들리는지**를 본다.

### 방법 (새 데이터 수집 없음, 기존 V5 150개 에피소드 재사용)
1. val 29개 에피소드 전체에 대해 baseline/add/replace 3모드로 프레임별 예측을 순차로 뽑는다(이번 세션에 5개 에피소드만 수동으로 본 걸 전체로 확장 — `closed_loop_eval_hidden_state.py`를 약간 확장해 프레임별 pred/gt/bbox 신뢰도를 같이 저장).
2. 각 프레임에 대해 "오예측 여부"(pred != gt_class)와 "그라운딩 신뢰도"(has_bbox, area, cx가 0.5 근처인지 등)를 같이 테이블로 만든다.
3. 상관 확인: 오예측 비율이 `has_bbox=False`이거나 `area`가 매우 작은/큰(경계값) 프레임에서 유의하게 높은가? 3개 모드 공통으로 흔들리는 프레임과 모드별로만 흔들리는 프레임을 구분.
4. (가능하면) right_right류 "방향이 막 바뀌는 시작 지점" 프레임들만 따로 떼어서 그라운딩 신뢰도가 그 구간에서 유독 낮은지 확인 — "객체가 화면에서 움직이는 전환 구간일수록 그라운딩이 불안정해진다"는 가설.

### 산출
- `docs/v5/closed_loop_eval/grounding_quality_vs_error.json` — 프레임별 (mode, pred, gt, has_bbox, area, cx, cy, error) 레코드
- 표/요약을 CH41로 문서화(결과가 가설을 지지하든 안 하든 그대로 기록)

이 결과에 따라 다음 plan이 갈린다: 상관이 강하면 → "그라운딩 정확도 자체를 올리는" 작업(예: 필터/모델 교체, multi-frame consistency)으로 다음 plan 작성. 상관이 약하면 → head/시퀀스 모델링 쪽 문제일 가능성이 커짐.

---

## 2. 윈도우 크기 ablation (head-level, 보조)

### 방법
기존 `train_hidden_state_action.py --window {2,4,8,16}` 그대로 사용, baseline(`none`)/add/replace 3 모드 × 4개 윈도우 = 12개 조합 학습(각 ~2~3분, 총 30분 내). 동일 seed=42 split.

### 산출
- PM 비교표(12개 조합) → 어느 윈도우가 최선인지, hidden state 유무에 따라 최적 윈도우가 달라지는지 확인.
- closed-loop은 1차로는 PM만 보고, PM에서 뚜렷한 우위가 보이는 조합만 골라 closed-loop까지 확장(전부 다 closed-loop 돌리면 비용 큼).

---

## 3. 종합

§1 결과가 §2보다 먼저 나오고, §1이 "그라운딩 품질이 핵심"이라는 가설을 지지하면 — §2(윈도우 ablation)는 "참고용 보조 데이터"로 위치를 낮추고, CH41 결론에서 다음 우선순위를 명시적으로 "그라운딩 개선"으로 못박는다. [[project_focus_grounding_for_direction]] 메모와 일치하는 결론이 나오는지가 핵심 체크포인트.

---

## 4. 변경 파일

| 파일 | 작업 |
|---|---|
| `scripts/eval/closed_loop_eval_hidden_state.py` | 프레임별 상세 로그(pred/gt/bbox 신뢰도) 저장 옵션 추가(기존 집계 로직 변경 없음, 추가만) |
| `docs/v5/closed_loop_eval/grounding_quality_vs_error.json` (신규) | §1 산출물 |
| 윈도우 ablation은 기존 `train_hidden_state_action.py` 그대로 반복 실행, 코드 변경 없음 |
| `docs/v5/research_story.html` | CH41로 결과 기록 |

## 5. 위험도

- 둘 다 기존 150개 데이터 재사용, 새 학습 비용 적음(윈도우 ablation 12개 합쳐도 30분 내), soda 미접촉.
- §1 진단이 인과관계를 증명하진 못함(상관 분석) — "그라운딩이 약해서 방향이 틀린다"를 강하게 주장하려면 추후 그라운딩 개선판으로 직접 개입 실험이 필요(이번 plan 범위 밖, 결과 보고 후 별도 plan).

# 그라운더 단일화 판정 — OWL-v2로 프리뷰+메인 통합 가능

> 작성일: 2026-07-05
> 플랜: docs/plans/plan_20260705_label_asset_utilization.md (B→C→A 완료)
> 데이터: 사람 라벨 296프레임(hsv_owlv2_preview_20260704) + 72프레임(bbox_truth_mini)

## 판정: **가능 — 3지표 전부에서 OWL-v2가 PG2와 대등하거나 우세**

### B. 296프레임 벤치마크 (`results.json`)

| 모델 | 정확도(객체있음 217) | 오탐(객체없음 79) |
|---|---|---|
| **OWL-v2 th0.25** | 94.9% | **0.0%** |
| OWL-v2 th0.10 | 99.1% | 74.7% |
| PG2 | 96.3% | 100.0% |
| Kosmos-2 refexp | 97.7% | 97.5% |
| HSV 원본/재튜닝 | 0.9% / 2.8% | 94.9% |

### C. 72프레임 IoU (사람 정답 bbox, `eval_iou_truth_mini.py`)

| 모델 | IoU mean | IoU median | cx MAE | 놓침 |
|---|---|---|---|---|
| **OWL-v2 th0.25** | **0.932** | **0.943** | 0.002 | 1/72 |
| PG2 | 0.927 | 0.939 | 0.002 | 0/72 |
| Kosmos-2 | 0.885 | 0.895 | 0.005 | 0/72 |

### A. 헤드 학습 비교 (동일 레시피/split, 5-seed, `head_compare.json`)

공통 43 에피소드(billy 전용 2개 제외), train 600 / test 158, Step2 MLP 레시피 그대로.

| 입력 bbox 소스 | PM (5-seed) |
|---|---|
| PG2 (재실측 baseline) | 77.2% ± 0.9% |
| **OWL-v2 th0.25** | **78.4% ± 1.7% (Δ +1.1%p)** |

분포도 문제없음: has_bbox 99.6% vs 99.4%, cx μ 동일(0.500 vs 0.497), σ 오히려 타이트(0.110 vs 0.127).

## 판정 기준 대비

플랜의 기준은 "OWL 헤드가 PG2 재실측 대비 **-2%p 이내면** 단일화 가능"이었는데,
실제로는 **+1.1%p 우세** — 기준을 여유 있게 통과.

## 시사점 (운영 아키텍처)

- **프리뷰+메인 전부 OWL-v2 단일화 가능**: PG2(6.7GB, 로드 35s, latency 1.4~26s 가변)가
  통째로 빠지고 OWL(0.62GB, 6s, ~0.4s 고정)만 남음 → Jetson 메모리 여유 대폭 확보,
  soda의 PG2 동시로드 크래시 이력 리스크도 소멸
- **threshold 0.25 부재판정**이 메인 루프에도 그대로 적용 — "없는데 가짜 bbox로 주행" 실패모드
  (실주행 프레임의 27%) 제거
- **예외**: hidden-state head(exp71/72 계열)는 PG2 hidden 의존이라 이 경로에선 PG2 유지 필요.
  bbox 기반 head(MLP/Transformer) 운영 시에만 단일화 유효
- 실배포 전 필요한 검증: closed-loop 시뮬 평가(evaluate_closed_loop_v5.py에 OWL 헤드 연결)
  + soda Jetson에서 OWL 실측 latency (기존 기록 432ms)

## 추가 — VLA 사다리 ①·② 결과 (2026-07-05, plan_20260705_vla_ladder_step1_2.md)

### ① 언어→타겟 선택
- OwlV2Grounder gray-접두 버그 수정, 쿼리 스위칭 갤러리(`docs/v5/owl_query_switch/`)로
  같은 프레임에서 쿼리별 다른 객체 검출 실증

### ② 언어→정책 조건화 (instruction CLIP emb 512 → 헤드 concat)

| 비교군 | PM (5-seed) |
|---|---|
| no_text (baseline) | 78.4% ± 1.7% |
| **with_text** | **81.8% ± 1.0% (Δ +3.4%p)** |
| shuffled_text (대조군) | 77.5% ± 1.7% |

인과 검증:
- **Permutation**: 임베딩을 다른 path_type 것으로 교체 시 80.4% → 65.8% (**−14.6%p**)
  → 텍스트를 실질적으로 사용함 (기준 ≥5%p 통과). shuffled가 baseline 수준인 것도
  "진짜 대응관계일 때만 이득"을 뒷받침
- **Counterfactual**: left지시→LEFT류 23.4%(교차 19.6%), right지시→RIGHT류 20.3%(교차 18.4%),
  예측변화율 20.3% — 방향 순응은 **약함**(±3~4%p 비대칭). 해석: 합성 instruction이
  시각 정보와 완전 상관이라, 텍스트가 "경로 맥락(prior)"으로는 쓰이지만 "명령"으로는
  아직 작동 안 함. 진짜 명령 순응을 얻으려면 같은 장면에 다른 지시→다른 행동 데이터
  (조이스틱 좌/중/우 수집, 5/15 미팅 계획)가 필요 — 이것이 데이터 수집의 정량 근거

**의의**: frozen VLM attention 경로(Exp12/13/15/17~41C 전부 text 0%)와 달리, 명시적
text feature 주입은 **언어가 행동 예측에 인과적으로 개입하는 첫 실측 사례** (perm −14.6%p).

### ②-b. window × text ablation (3-seed, `ablate_instr_window.json`)

| window | none | real | shuffled |
|---|---|---|---|
| 1 | 74.3±0.3 | **81.2±0.8** | 75.3±0.5 |
| 2 | 77.0±1.7 | 81.0±2.1 | 75.9±1.0 |
| 3 | 77.6±1.7 | 81.6±1.0 | 76.8±1.1 |
| 4 | 77.8±0.0 | 82.3±0.5 | 77.2±0.5 |
| 6 | 78.5±0.9 | **82.9±1.4 (최고)** | 77.8±0.5 |
| 8 | 78.1±0.3 | 82.3±1.0 | 77.0±0.8 |

발견:
1. **텍스트 이득이 모든 윈도우에서 유지** (+4.4~+6.9%p) — bbox history가 길어져도
   instruction 정보가 대체되지 않음 (w1 +6.9 → w6 +4.4로 다소 감소하나 여전히 큼)
2. **w1+text(81.2%) > w6 no-text(78.5%)** — instruction 한 줄이 6프레임 히스토리보다
   더 많은 경로 정보를 담음
3. shuffled는 전 구간 none과 동급 — 대조군 일관 유지
4. 최적: **w6 + text = 82.9%** (히스토리 포화점 w6과 텍스트 이득이 독립적으로 합산)

## 추가 — CH60 closed-loop 검증 (2026-07-05)

`docs/v5/closed_loop_eval/CH60_OWL_TEXT_CLOSED_LOOP.md`: pg2_w3 / owl_w6 / owl_w6_text
리플레이 비교 — SR 3자 동률 59.3%, FPE는 **owl_w6 0.44m로 최저**(pg2 0.59m 대비 −25%).
text 헤드의 open-loop 이득(+4.4%p)은 closed-loop으로 전이 안 됨 (경로 prior 중복).
**최종 운영 권장: OWL-v2(th0.25) + w6 헤드.**

## 산출물

- `scripts/eval/grounding_benchmark.py` — 고정 벤치마크 (신규 그라운더 자동 채점 API 포함)
- `scripts/eval/eval_iou_truth_mini.py` — 72프레임 IoU (+ truth_mini_preds.json 캐시)
- `scripts/gen_bbox_dataset_owl.py` → `docs/v5/bbox_nav_owl/bbox_dataset_owl.json` (758프레임)
- `scripts/train_step2_owl_head.py` → `docs/v5/bbox_nav_owl/head_compare.json`

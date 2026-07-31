# CH60 — closed-loop 리플레이: PG2 vs OWL vs OWL+text 헤드

> 작성일: 2026-07-05
> 스크립트: `scripts/sim/evaluate_closed_loop_owl_text.py`
> 결과: `docs/v5/closed_loop_eval/owl_text_metrics.json`
> 선행: docs/v5/grounding_benchmark/CONCLUSION.md (open-loop PM: pg2 77.2 / owl_w6 78.5 / owl_w6+text 82.9)

## 설정

- 공통 43 에피소드(billy 전용 2개 제외), episode명 기준 동일 split → test 9 에피소드
- 3 seed × 3 variant, DT=0.1 / success_fpe=0.5m (기존 rollout_metrics.json step2 기록과 동일 조건)
- text emb: path_type instruction 9종 → OWL 내장 CLIP 텍스트 타워 512d (배포 시 사용자가 주는 입력)

## 결과

| variant | SR | FPE | TLD |
|---|---|---|---|
| pg2_w3 (기존 Step2 재실측) | 59.3% ± 5.2 | 0.59m | 0.97 |
| **owl_w6** | 59.3% ± 5.2 | **0.44m (−25%)** | 0.97 |
| owl_w6_text | 59.3% ± 13.9 | 0.54m | 0.99 |

(참고: 과거 기록 step2 SR 50~66.7%, FPE 0.55~0.70m — baseline 재실측이 그 범위 안에 재현됨)

## 해석

1. **OWL 전환은 closed-loop에서도 손해 없음 + FPE 25% 개선** (0.59→0.44m).
   open-loop(B/C/A) 결론 "단일화 가능"이 closed-loop에서도 유지됨.
2. **text 헤드는 closed-loop에서 이득 없음**: SR 동률, FPE는 owl_w6보다 오히려 0.1m 나쁘고
   seed 분산이 3배(±13.9). open-loop +4.4%p가 실주행 궤적으로는 전이되지 않음.
   - 원인 추정: counterfactual 검증(±3~4%p)에서 이미 드러났듯 텍스트는 "경로 맥락 prior"로만
     작동 — 리플레이 rollout에서는 bbox 궤적이 같은 정보를 제공하므로 중복.
     장면-지시 상관이 깨진 데이터(조이스틱 좌/중/우) 없이는 closed-loop 이득도 없을 것.
3. test 9 에피소드 × 3 seed의 소표본이므로 SR 동률은 "차이를 검출 못함"으로 읽어야 함.
   FPE는 연속 지표라 상대 비교 신뢰도가 더 높음.

## 주의 (재현 시)

- DT=0.35, success_fpe=0.8로 잘못 돌리면 SR 7~19%로 붕괴 — **rollout 파라미터는 반드시
  rollout_core.DT_DEFAULT(0.1)와 기존 기록 조건(0.5m)에 맞출 것** (이번에 한 번 밟은 함정)

## CH60-b — 운영 계보(exp66) 그라운더 drop-in 스왑 (2026-07-05 추가)

경량 bbox 헤드 계보가 아닌 **운영 계보**(FrozenCLIPV2 vision 256d + bbox w8, exp66 학습된
헤드 ckpt 그대로, **재학습 없음**)에서 bbox 입력만 교체 (`scripts/eval_exp66_owl_swap.py`):

| bbox 소스 | SR | FPE | n |
|---|---|---|---|
| PG2 (exp66 원본 재현) | 96.6% | 0.10m | 29 |
| **OWL-v2 th0.25 (drop-in)** | **96.6%** | 0.13m | 29 |

- baseline이 기록값(96.6%/0.10m)과 정확히 재현됨 → 하네스 신뢰 가능
- **재학습 없이 그라운더만 OWL로 바꿔도 SR 완전 유지** (FPE +0.03m는 오차 수준)
- → 운영 서버에서 `VLA_GROUNDER=owlv2`로 켜기만 하면 되는 수준의 호환성

**같은 기준(exp66 레시피, augment 포함)으로 OWL 데이터에서 헤드 재학습**
(`train_exp54_stage2_v2_action.py --data bbox_dataset_owl_150ep.json --augment --tag owl150`,
147ep/2572프레임, has_bbox 99.5%):

| 구성 | SR | FPE | n |
|---|---|---|---|
| exp66 헤드 + PG2 (원본) | 96.6% | **0.10m** | 29 |
| exp66 헤드 + OWL (drop-in) | 96.6% | 0.13m | 29 |
| **OWL 재학습 헤드 + OWL** | **100%** | 0.12m | 30 |
| exp71 헤드 + PG2 (held-out) | 100% | 0.07m | 21 |
| exp71 헤드 + OWL (drop-in, `exp71_owl_swap.json`) | **100%** | **0.07m** | 21 |

- OWL 재학습 헤드(per-frame val_acc 90.7%)는 CL에서 SR 100% — drop-in(96.6%)보다 오히려 상회
- exp71(운영 기본 Transformer)은 drop-in만으로 PG2와 **완전 동일**(100%/0.07m) — 재학습조차 불필요
- 단, 이 계보의 리플레이 CL은 포화 상태이므로 세부 차이(0.10 vs 0.12m)에 의미 부여 금지

## CH60-c — exp71/72 기록 오염 발견 및 정정 (2026-07-05 추가)

- rollout_metrics.json의 exp71/72 SR 100%/FPE 0.00은 **train/test 오염**: CL 평가의
  test 9 에피소드가 전부 exp71 학습 split에 포함 (seed 42/val 0.15 재구성으로 확정)
- 진짜 held-out(학습 val 21ep) 재평가 (`scripts/eval_exp71_holdout_cl.py`):
  **SR 100%, FPE 0.07m** — 오염과 무관하게 exp71은 리플레이에서 실제로 거의 만점
  (per-frame val_acc 97.6% → 에피소드 전부정답 확률 0.976^15≈70% ≈ 실측 14/21과 일치)
- **시사점: CLIP-vision 계보에서 오프라인 리플레이 CL은 포화된 벤치마크** — 변별력 없음.
  실로봇 실패는 리플레이가 재현 못 하는 요인(그라운딩 flicker/부재판정/latency)에서 발생
  → 헤드가 아니라 그라운더가 병목이라는 이번 벤치마크 전체 결론과 정합

## CH60-e — "방향을 어떻게 뽑을 것인가" 전체 접근법 비교 (2026-07-05 추가)

CH60 전체가 "그라운더/헤드를 뭘로 할까"를 다루는데, 그 전에 이미 결정났던 더 상위 질문
— **"zero-shot 유사도만으로 방향까지 직접 뽑을 수 있는가"** — 를 CH60 옆에 같이 놓고 보면
지금 아키텍처가 왜 이 형태인지 전체 그림이 보인다.

### 접근법 4갈래

| # | 접근 | 방식 | 방향 정보 출처 |
|---|---|---|---|
| 1 | **CLIP zero-shot 유사도 직접분류** | 이미지×4문장("left"/"center"/"right"/"없음") 유사도 비교, 가장 높은 문장 채택 | 텍스트-이미지 매칭 그 자체 |
| 2 | **OWL-v2/Kosmos-2 zero-shot bbox** | bbox 좌표를 뽑아 cx 연속값 획득 | 검출된 박스 위치 |
| 3 | **PG2 zero-shot bbox (oracle급)** | 위와 동일 방식, 정확도만 최상 | 검출된 박스 위치 |
| 4 | **학습된 헤드 (지금 운영 계보, exp50~72/CH60)** | frozen vision feature + bbox history → MLP/Transformer가 방향을 직접 학습 | 헤드가 학습한 매핑 |

### 36프레임 ablation — zero-shot 방식들의 성적 (`docs/v5/ablate_perception_models.json`)

| 모델 | 검출률 | cx 일치 | **방향 일치** |
|---|---|---|---|
| PG2 (oracle) | 100% | 100% | 100% |
| **CLIP 유사도 직접분류 (#1)** | 72.2% | 25.0% | **13.9% — 최하위권** |
| OWL-v2 (#2) | 100% | 83.3% | 52.8% |
| Kosmos-2 (#2) | 100% | 83.3% | 52.8% |
| Florence-2 | 100% | 11.1% | 2.8% |

### CH60(#4, 학습된 헤드)과 나란히 놓으면

| 접근 | 방향 일치/SR | 비고 |
|---|---|---|
| CLIP zero-shot 직접분류 (#1) | 13.9% | 좌표 없이 3칸(0.2/0.5/0.8) 강제 스냅 — 구조적 한계 |
| OWL/Kosmos zero-shot bbox (#2) | 52.8% | 좌표는 뽑지만 방향 판단까지 직결 안 됨(threshold/필터 없이는) |
| **OWL-v2 th0.25 + 학습된 헤드 (#4, CH60)** | **SR 96.6~100%** | bbox를 헤드가 히스토리와 함께 학습해서 방향으로 변환 |

**결론**: zero-shot 유사도만으로 방향을 직접 뽑는 방식(#1)은 애초에 후보에서 탈락할 만큼
성능이 낮았고(13.9%), bbox를 뽑는 zero-shot(#2)도 방향 일치 52.8%로 그 자체론 부족했다.
지금 아키텍처(#4)가 세 자리 수 SR을 내는 이유는 grounding 모델을 "방향 분류기"로 직접 쓰지
않고, **"위치 신호 공급원"**으로만 쓰고 방향으로의 변환은 학습된 헤드가 맡도록 역할을
분리했기 때문 — CH60에서 OWL-v2를 골랐던 것도 "OWL이 방향을 잘 맞혀서"가 아니라
"OWL이 PG2보다 안정적으로 위치 신호(bbox+부재판정)를 공급하고, 그 신호를 헤드가 그대로
잘 소화하기 때문"이라는 게 이 비교로 더 분명해진다.

## 결론 (CH60)

- **최종 운영 권장: 기존 exp66/71 헤드 그대로 + 그라운더만 OWL-v2(th 0.25)로 교체**
  (CH60-b: 재학습 불필요, SR 96.6% 유지) — soda는 `VLA_GROUNDER=owlv2` 켜는 것으로 충분
- 리플레이 CL은 CLIP-vision 계보에서 포화 — 이후 개선 검증은 실로봇 세션 지표
  (fallback률, preview latency, 실주행 SR)로 해야 함
- text 조건화는 open-loop 전용 이득으로 보류 — 조이스틱 이질 지시 데이터 수집 후 재평가
- exp71/72의 기존 rollout_metrics 기록은 오염으로 무효 처리, exp71_holdout.json이 정본

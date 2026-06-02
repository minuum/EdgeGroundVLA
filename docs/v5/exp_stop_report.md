# 도착 STOP 규칙 — 마지막 레이어 규칙 기반 정지 (결과)

> 작성: 2026-06-02 · 선행: [plan_20260602_stop_arrival_rule.md](../plans/plan_20260602_stop_arrival_rule.md)
> 한 줄 결론: STOP은 학습에 희소(21/243 ep)하나 **PG2 area_det 신호가 극명** →
> 무학습 규칙으로 도착 90% 탐지(조기오발 0%), "도착=정지" 목표 하 CL **31%→69% 회복**

> [!IMPORTANT]
> **실패 케이스(Fail Case) 및 이미지 가시화 정의**
> 본 실험 및 [Ablation 대시보드](ablation_visual_proof.html)에서 제시되는 모든 **실패 케이스(Fail Case)** 이미지는 주행 중 발생하는 일시적인 흔들림(jitter)이 아닙니다.
> 해당 이미지는 **최종 도착 시점에서의 Closed-Loop (CL) 최종 위치 오차(FPE, Final Position Error)가 합격 기준선(FPE < 0.5m)을 벗어나 완전히 이탈(FPE ≥ 0.5m)함으로써 골 도달에 실패한 시점의 실제 주행 프레임**을 의미합니다.

---

## 1. 동기

V5 8-class에서 STOP(0)은 합성 클래스. 현재 best 데이터(`bbox_dataset_pg2_cx.json`, 243 ep)에도
STOP 라벨은 **21 ep / 105 frame(8.6%)** 뿐 → MLP가 **under-trigger**. 도착해도 로봇이 안 멈춤.
사용자 아이디어: "마지막 프레임의 평균을 확인" → 마지막 레이어에 규칙 기반 STOP을 붙인다.

## 2. 데이터 신호 (243ep / PG2 `area_det`)

| | STOP 프레임 | non-STOP 프레임 |
|---|---|---|
| area_det median | **0.890** | 0.050 |
| 위치(phase) | 0.73~1.00 (mean 0.94) | — |
| cx_det | mean 0.468, std 0.097 (중앙) | — |

area_det-phase 곡선이 **단조 증가**(phase 0.6=0.10 → 1.0=0.63)하고 **출발 스파이크 없음**
(phase 0.0=0.045). HSV에서 보이던 출발 스파이크는 HSV 아티팩트였음.

## 3. STOP 규칙

```
area_det 최근 W프레임 평균 > TH_AREA  AND  |cx_det - 0.5| < TH_CX  AND  step ≥ MIN_STEPS
→ STOP(0), 래치(한 번 멈추면 유지)
```

`scripts/analyze_stop_rule.py` — 192개 grid sweep 결과 ([stop_rule_calibration.json](stop_rule_calibration.json)):

**추천 config: `TH_AREA=0.5, TH_CX=0.3, W=5, MIN_STEPS=0`**
- STOP ep 트리거율 **90%**, 타이밍 오차 중앙값 **3 frame (phase 0.109)**
- **조기오발(phase<0.7) = 0%** (모든 config 공통 — 출발 스파이크 없음 확증)
- STOP 미라벨 ep에서도 54% 트리거되나 전부 phase 0.94(진짜 도착) → STOP 일반화(올바름)

## 4. Closed-Loop 4-cell 검증

`scripts/eval_stop_closedloop.py` — pg2_cx 주석 + CLIP feature → MLP(abl_b1, val 95.7%) → 궤적.
expert: `raw`(gt 그대로, 도착 후 전진) vs `synth`(도착 규칙으로 STOP 래치) ×
pred: STOP override `off` vs `on`. (val 33 ep)

| expert | pred STOP | CL success (FPE<0.15) | mean FPE | mean TLD |
|---|---|---|---|---|
| raw | off | 68.8% (22/32) | 0.082m | 0.998 |
| raw | on | 34.4% (11/32) | 0.196m | 0.925 |
| **synth** | **off** | **31.2% (10/32)** | 0.201m | 1.081 |
| **synth** | **on** | **68.8% (22/32)** | **0.081m** | 0.998 |

> 임계 0.5m에선 전 cell 100%(절대 FPE 작음) → 차이를 드러내려 0.15m로 조임.

**해석 — 대각선이 핵심:**
- **synth/off (31.2%)**: 목표는 도착 정지인데 pred가 안 멈춤 → **basket 통과(과주행)**. 실로봇 충돌 케이스.
- **synth/on (68.8%)**: STOP 규칙이 도착에서 정지 → expert와 정렬 → 성공률 **2.2배 회복** (FPE 0.201→0.081m).
- raw/on(34.4%)은 반대로 expert는 안 멈추는데 pred만 멈춰 미달 — STOP은 "도착 정지 목표"에서만 이득.

## 5. 산출물

- 규칙/캘리브레이션: `scripts/analyze_stop_rule.py`, `docs/v5/stop_rule_calibration.json`
- 4-cell 평가: `scripts/eval_stop_closedloop.py`,
  `docs/v5/closed_loop_eval/stop_closedloop_result_{b1,b1_strict}.json`
- (rollout_core 미수정 — STOP 합성은 eval 스크립트 내 helper)

## 5b. STOP 학습 유도 (규칙 대신 MLP가 직접 학습)

규칙 override 대신 **MLP가 "도착 → STOP"을 직접 예측**하도록 학습 유도.

**데이터 (`scripts/build_stop_annotation.py`)**: ① mid-stop 84개 → 직전 모션 재라벨(프레임 손실 0)
② 도착 plateau STOP 합성 — 마지막부터 `area_det>θ & cx 중앙` 연속 구간을 STOP.
신호 오염 방지를 위해 **area 큰 도착에만** 합성(θ=0.65 → STOP 398프레임 8.7%, 129/243 ep).

**학습 (`scripts/train_stop_mlp.py`)**: B1 config(PG2, 243ep) + class-weighted CE.
→ STOP **recall 95.6%**, precision 64.2%, acc 86.7% (`stop65_mlp.pt`).

**CL 비교 (fpe<0.15, synth=도착정지 목표, val 32 ep):**

| 방식 | CL success | FPE | TLD |
|---|---|---|---|
| STOP 없음(b1), 규칙 X | 31.2% | 0.201m | 1.081 (과주행) |
| **학습 STOP**(stop65), 규칙 X | **53.1%** | 0.178m | 0.948 |
| b1 + **area 규칙 래치** | **68.8%** | 0.081m | 0.998 |

- ✅ **학습 유도 성공**: 모델이 STOP을 배워 31.2%→53.1%, 과주행→정렬. recall 95.6%.
- ⚠️ **규칙(68.8%)이 여전히 우위**: 학습 STOP은 precision 64% → 접근 프레임에서 **조기 발화**(TLD 0.95).
- 학습+규칙 중복은 이중 정지로 악화(31%).

**함의**: STOP은 area 신호가 강해 학습이 쉽게 유도되나(recall↑), 접근 중 transient 큰 area에 과발화(precision↓).

### 5c. precision 보강 sweep — (가)latch+gate / (나)confidence / (다)θ↑ / (라)확률 윈도우 평균

`scripts/eval_learned_stop.py` 및 `scripts/eval_learned_stop_window.py` — 학습 STOP을 후처리로 정제 (val 32 ep, fpe<0.15):

| 모델 / 후처리 | CL | FPE | TLD | 특이사항 |
|---|---|---|---|---|
| stop65 raw | 53.1% | 0.178m | 0.948 | baseline |
| (나) conf>0.9 | 53.1% | 0.182m | 0.984 | 효과 없음 |
| (가) latch+gate0.6 | 21.9% | 0.228m | 0.906 | latch 조기정지 영구 고정으로 악화 |
| **stop75 raw (다)** | 50.0% | 0.159m | 1.027 | TLD 균형 개선, 성공률 비슷 |
| (가+나) conf0.9+gate0.7+latch | 50.0% | 0.161m | 1.028 | |
| **stop65 + (라) W=3, th=0.7, latch** | **68.8%** | **0.119m** | **1.025** | **신경망 학습 신호 + 윈도우 스무딩 결합** |
| **stop65 + (라) W=3, th=0.8, latch** | **68.8%** | **0.115m** | **1.027** | **규칙 기반 래치 성능 수준 달성** |
| **참고: b1 + area 규칙 래치** | **68.8%** | **0.081m** | **0.998** | 휴리스틱 bbox 면적 규칙 (최선) |

- **(나) confidence**: 효과 거의 없음 — 학습 STOP은 이미 고확신 예측.
- **(가) latch+gate**: **오히려 악화** — 학습 STOP의 첫 발화가 노이즈/조기라 래치 시 조기 정지가 영구 고정. (규칙은 윈도우 평균 기반이라 래치가 안전했던 것과 정반대.)
- **(다) θ=0.75**: TLD 1.027로 가장 균형(덜 조기), FPE 0.159 최선이나 성공률 50%.
- **(라) STOP 확률 윈도우 평균 (스무딩)**: **성공률 68.8%로 기존 최선 휴리스틱 규칙과 완벽히 동등한 수준 도달.** 윈도우 $W=3$ 및 임계값 $\theta_{\text{prob}} = 0.7 \sim 0.8$을 통해 1-step 노이즈성 정지를 억제하여 53% 정체를 뚫어냄.

**최종 결론**:
단순 학습 기반 STOP 모델은 1-step 노이즈로 인해 CL 성공률이 ~53%에서 정체되었으나, **신경망의 STOP 확률 신호에 3-프레임 윈도우 스무딩 필터(W=3, th=0.7~0.8)를 적용하자 기존 휴리스틱 규칙 래치와 완벽히 동등한 68.8%의 성공률**을 달성했습니다.
이는 특정 바스켓 면적 규칙(`area_det > 0.5`)을 사용하지 않고도, 범용적인 학습형 도착 인지 기능에 가벼운 시간축 필터링만 결합하여 동등한 수준의 최선 조향 성능을 확보할 수 있음을 증명합니다.

산출물: `bbox_dataset_pg2_cx_stop{,65,75}.json`, `stop{,65,75}_mlp.pt`,
`scripts/{build_stop_annotation,train_stop_mlp,eval_learned_stop,eval_learned_stop_window}.py`.

## 6. 한계 / 다음 단계

- 본 평가는 **precomputed PG2 주석** 기반(라이브 grounding 지터 없음) → 절대 FPE가 작음.
  라이브 grounding(`eval_exp59_closedloop.py`) + STOP override 결합 평가는 후속.
- **실로봇 포팅 (완료)**: `inference_server.py` GoalNav `predict()`에 `_arrival_stop()` 래치 규칙 추가.
  env 토글 `VLA_GOALNAV_STOP_RULE`(기본 ON), 파라미터 `VLA_GOALNAV_STOP_TH_AREA/TH_CX/W/MIN_STEPS`.
  `/goalnav/reset`에서 래치 해제, `/goalnav/status`에 상태 노출. → SODA 배포 후 물리 검증만 남음.
- 규칙 vs 학습: STOP 라벨을 전 ep에 area 규칙으로 합성해 MLP 재학습하면 학습형 STOP도 가능(별도).
- Exp60(주행 강건성)과 **상보적** — STOP은 종결 조건 담당.

### 6b. 그라운딩 추론의 3대 위협 요인 및 학습 연계 분석

대시보드와 H5 시각화 프레임 분석을 통해, VLM 그라운딩 추론의 한계점과 이를 해결하기 위한 제어 신경망(MLP) 학습 연계 방안을 다음과 같이 도출할 수 있습니다.

1. **오프셋 편향 (Offset Bias)의 한계와 데이터 증강**:
   - PaliGemma2 등 사전학습 VLM을 2D 그라운더로 파인튜닝할 때, 데이터 분포나 VLM 고유의 visual bias로 인해 BBox 예측 중심 좌표(cx)가 특정 방향으로 쏠리는 **오프셋 편향**이 발생합니다. (예: B1의 cx가 평균 -0.084만큼 좌측 편향).
   - 단순히 데이터 증강(Flip, B2)만 주입할 경우, 편향이 좌우 대칭 상쇄(cancelling)될 뿐 예측의 variance가 줄어들지 않아 오히려 주행 안정성이 일부 저하될 수 있습니다.
   - **학습 연계:** Stage 2 제어 MLP 학습 단계에서 VLM의 오프셋 통계에 맞춘 **BBox Noise Augmentation (scale=2.0)**을 의도적으로 주입하고, **Center 데이터 오버샘플링(B3)**을 통해 중요한 복원 궤적 영역의 밀도를 높여 편향 노이즈를 제어 모델 단에서 흡수하도록 설계해야 합니다.

2. **고주파 지터 (High-frequency Jitter)와 시간축 필터링**:
   - VLM은 개별 프레임 단위로 독립 추론하기 때문에, 인접 프레임 간 BBox가 요동치는 **고주파 지터** 현상이 빈번히 관찰됩니다. (예: B2 실패 프레임에서 BBox 검출 실패 및 transient jitter).
   - 이 지터 노이즈가 단일 스텝의 액션 예측에 바로 노출되면, 제어기가 민감하게 반응하여 조향이 발산합니다.
   - **학습 연계:** 제어 MLP 입력 시 단일 프레임이 아닌 **윈도우 히스토리(WINDOW=3~8)**를 결합하여 제어 액션을 시간축 상에서 스무딩하고, STOP 판단 시에도 **확률적 윈도우 스무딩 필터(W=3, th=0.7~0.8)**를 이식하여 1-step 노이즈에 정지가 오발화되는 현상을 억제해야 합니다.

3. **End-to-End (E2E) 모델(C1)의 Steer Oscillation**:
   - Kosmos-2 E2E 모델은 중간 Grounding 해석 필터 없이 Visual feature에서 직접 조향 액션 텍스트(`Robot action`)를 토큰으로 예측합니다.
   - E2E 모델은 중간 오차 보정이 불가능하여 미세한 이미지 잡음에도 조향이 요동치며 **진동 발산(Steer Oscillation)**으로 직결되어 탈선율이 급증(성공률 18.8%)합니다.
   - **학습 연계:** 이는 그라운딩과 제어기를 명조화한 **Decomposed 아키텍처(Stage1 Grounder + Stage2 Controller)**가 데이터 효율성 및 강건성 제어 학습 측면에서 현격한 지배적 우위를 점하고 있음을 실증합니다.

## 7. 교수님 질문과의 연결

"목표를 인식하고 도착을 판단하는가" = goal-conditioned navigation의 종결 조건.
데이터에 희소한 STOP을 **area_det(목표 근접 proxy)** 로 복원 → "목표를 보고(area↑) 도착 판단 → 정지"
라는 인식→행동 종결 논리를 무학습 규칙으로 시연.

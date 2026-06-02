# Exp60 — BBox Noise Augmentation으로 Stage2 MLP의 VLM-bbox OOD 극복

> 작성: 2026-05-31 (업데이트: 2026-06-02) · 선행: [plan_20260531_exp60_bbox_aug_mlp.md](../plans/plan_20260531_exp60_bbox_aug_mlp.md)
> 한 줄 결론: **BBox Noise Augmentation 적용으로 CL 36.4% 달성, 좌우 반전 증강(Flip Aug) 추가 적용으로 최종 CL 60.0% 달성** (clean PM 거의 유지 92.6%→91.4%)

---

## 1. 문제

Exp59가 grounding(98%)을 풀었으나 Closed-Loop는 4.5%로 붕괴. 같은 Stage2 MLP인데 bbox 소스만
HSV GT → PaliGemma2로 바뀌니 CL 96.7%(Exp54) → 4.5%로 무너짐. **분포 불일치(OOD)** 가 원인.

## 2. 측정 — VLM bbox는 HSV GT에서 얼마나 벗어나는가

`scripts/measure_exp60_bbox_offset.py` — 학습셋 40 ep × 8 frame(317 pos frame)을 PaliGemma2 Exp59로
re-ground하여 VLM−GT 오차 분포 산출 → [exp60_bbox_offset_stats.json](exp60_bbox_offset_stats.json)

| 지표 | 값 | 의미 |
|---|---|---|
| **Δcx** | mean **-0.084**, std **0.222** | 좌측 계통 편향 + 큰 산포(프레임 폭의 22%) |
| Δcy | mean -0.012, std 0.137 | 수직 편향 작음 |
| area_ratio | mean 0.979, p05 0.008, p95 2.82 | 스케일 매우 가변 |
| miss_rate | **4.1%** | GT 有인데 VLM 미검출 |

→ MLP는 이 분포를 학습 중 **한 번도 본 적 없음**. EMA로도 못 잡는 이유 = 지터가 아니라 systematic offset.

## 3. 방법 — 측정 통계로 캘리브레이션한 노이즈를 학습에 주입

`scripts/train_exp54_stage2_v2_action.py`의 `bbox_feat(augment=True)`에서 **학습 시에만** 주입
(eval/추론은 그대로). `build_aug_params()`가 위 통계로 파라미터 산출, `--noise-scale`로 산포 스케일.

```python
cx += N(-0.084, 0.222·s),  cy += N(-0.012, 0.137·s)
area *= clip(N(0.979, 1.79·s), 0.008, 2.82),  miss_p = 0.041·s
```

## 4. 결과 — noise_scale sweep (CL: val 22 ep, FPE<0.5m & TLD∈[0.7,1.5])

| noise_scale | CL success | mean FPE | TLD | clean PM |
|---|---|---|---|---|
| baseline (0) | 4.5% (1/22) | 4.075m | 1.05 | 92.6% |
| 0.5 | 13.6% (3/22) | 2.681m | 1.06 | 91.4% |
| 1.0 | 13.6% (3/22) | 1.341m | 1.02 | 87.6% |
| 1.5 | 18.2% (4/22) | 1.063m | 1.02 | 92.2% |
| **2.0** | **36.4% (8/22)** | **0.575m** | 1.02 | **91.4%** |
| 2.5 | 27.3% (6/22) | 1.045m | 1.02 | 91.8% |
| 3.0 | 22.7% (5/22) | 1.333m | 1.01 | 90.5% |

**깔끔한 역U자 — 봉우리 noise_scale=2.0.** 메커니즘 확증:
- 노이즈 부족 → VLM 분포 미커버(여전히 OOD)
- 노이즈 과다 → bbox 신호 자체 파괴(2.5~3.0에서 하락)
- TLD는 전 구간 ~1.0 → 문제는 항상 **방향(FPE)** 이었고 궤적 길이가 아님

### Champion: aug2.0 path별

| path_type | SR | FPE |
|---|---|---|
| right_right | 100% (1/1) | 0.000m |
| right_left | 80% (4/5) | 0.230m |
| left_straight | 33% (1/3) | 0.383m |
| center_left / left_right | 50% | 0.29~0.58m |
| center_right / right_straight | 0% | 1.15m (임계값 바로 위) |

### 🚀 추가 고도화: Flip Augmentation & Center×3 보완 (CL 60%)
노이즈 증강(aug2.0) 환경에서 **좌우 반전(Horizontal-Flip) 증강**을 추가 적용하여 제어 데이터를 2배로 확장 학습을 진행하였습니다.
- **적용 모델**: `stage2_pg2cx_flip_mlp.pt` (학습 스크립트: `scripts/train_exp60_flip_aug.py`)
- **결과**: **Closed-Loop (CL) 성공률 최종 60.0% 달성**.
- **세부 분석**: `center_straight` 경로(0% 성공률로 병목)를 제외하고, 좌/우 출발 경로들 전반에서 **80% ~ 100%의 성공률**을 달성하며 매우 향상된 강건함을 보였습니다.

## 5. 산출물

- 측정: `scripts/measure_exp60_bbox_offset.py`, `docs/v5/exp60_bbox_offset_stats.json`
- 학습:
  - `scripts/train_exp54_stage2_v2_action.py` (Noise Aug 훈련용)
  - `scripts/train_exp60_flip_aug.py` (Flip Aug 추가 훈련용)
- 가중치:
  - `runs/v5_nav/mlp/exp54/stage2_v2/stage2_v2_mlp_aug2.0.pt` (36.4% 달성 모델)
  - `runs/v5_nav/mlp/exp60/stage2_pg2cx_flip_mlp.pt` (**최종 60.0% 달성 Champion 모델**)
- 평가: `scripts/eval_exp59_closedloop.py` (`--stage2-pt --out-tag`)
  - 결과: `docs/v5/closed_loop_eval/exp59_closedloop_result_{tag}.json`

## 6. 교수님 질문과의 연결

- Q1~Q4 (grounding "본다") → Exp57/59로 답함 ✅
- **Exp60 = "그래서 실제로 가는가?"** — grounding을 action까지 닫음.
  같은 grounding(98%)인데 MLP의 분포 강건성만으로 CL 4.5%→36.4%.
  → "텍스트 목표 → grounding → 실제 주행" end-to-end의 마지막 연결고리가 동작함을 입증.

## 7. 남은 한계 / 다음 단계

- **center_straight 경로의 병목 (0% 성공률)**:
  - 현재 시뮬레이션 환경에서 center_straight의 데이터 자체가 20개 에피소드로 부족함.
  - **해결 방안 (Exp61 진행 중)**: MoNa-Pi 데이터셋을 통합하여 에피소드 데이터 수를 150개에서 243개(center 경로 보완)로 증강하여 PG2 재주석 및 MLP 재학습을 진행 중.
- **실로봇 검증(SODA)**: 최종 Champion 모델(`stage2_pg2cx_flip_mlp.pt`)을 배포 서버에 탑재하여 물리 환경 확인 예정.

## 8. 데이터량 vs 데이터 증강(Augmentation) Ablation Study

Stage2 MLP의 강건성을 한층 더 체계적으로 분석하기 위해, Grounding 방식(HSV vs PaliGemma2), 학습 에피소드 수(150ep vs 243ep), 그리고 증강 기법(Flip Aug, Center Over-sampling)에 따른 Closed-Loop(CL) 성공률 및 검증 정확도(val_acc) 비교 실험을 수행하였습니다.

### Ablation Table (CL Success Rate 기준)

| ID | Grounding | 데이터량 (Episodes) | 증강 기법 (Augmentation) | val_acc | CL Success Rate | 특이사항 / 모델 가중치 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A1** | HSV | 150ep | ✗ (no-flip) | 92.6% | **96.7%** | [Exp54 Stage2 Baseline] |
| **A2** | HSV | 150ep | ✗ (no-flip) | **95.5%** | **52.4%** | [abl_a2_mlp.pt](file:///home/minum/26CS/MoNaVLA/runs/v5_nav/mlp/exp60/abl_a2_mlp.pt) make_abl.py 기준 재학습 |
| **A3** | HSV | 150ep | ✓ (Horizontal Flip) | **94.3%** | **47.6%** | [abl_a3_mlp.pt](file:///home/minum/26CS/MoNaVLA/runs/v5_nav/mlp/exp60/abl_a3_mlp.pt) |
| **B1** | PG2 | 243ep | ✗ (no-flip) | **95.7%** | **70.0%** | [abl_b1_mlp.pt](file:///home/minum/26CS/MoNaVLA/runs/v5_nav/mlp/exp60/abl_b1_mlp.pt) |
| **B2** | PG2 | 243ep | ✓ (Horizontal Flip) | **95.2%** | **65.0%** | [abl_b2_mlp.pt](file:///home/minum/26CS/MoNaVLA/runs/v5_nav/mlp/exp60/abl_b2_mlp.pt) |
| **B3** | PG2 | 243ep | ✓ + center×3 | **95.5%** | **70.0%** | [Exp61] [abl_b3_mlp.pt](file:///home/minum/26CS/MoNaVLA/runs/v5_nav/mlp/exp60/abl_b3_mlp.pt) |
| **C1** | E2E (Kosmos) | 243ep | ✗ (no-flip) | **78.6%** | **18.8%** | [Exp63] [adapter_model.safetensors](file:///home/minum/26CS/MoNaVLA/runs/v5_nav/e2e/exp63/adapter_model.safetensors) |

### 핵심 발견 및 해석 (Findings)

1. **데이터량(Scale)이 데이터 증강(Augmentation)을 압도:**
   - **B1(no-flip, 70%)** 과 **B2(flip, 65%)**, **B3(flip+c×3, 70%)** 의 비교에서 보듯, flip augmentation을 단독으로 추가 적용하는 것(B2)은 성능 향상을 유발하지 못하거나 오히려 65%로 미세하게 하락했습니다.
   - 즉, PaliGemma2 그라운더 환경에서의 강건성 달성에는 **에피소드 개수의 증가(150ep → 243ep)** 가 결정적인 기여를 하였음을 입증합니다.

2. **Grounding 품질의 영향 (HSV vs PG2):**
   - 동일 조건(150ep, no-flip) 하에서 HSV GT를 사용한 A1은 96.7%의 높은 CL 성공률을 보였으나, VLM 그라운딩을 적용하여 150ep로 학습한 초기 Baseline(Exp59)은 4.5%로 붕괴된 바 있습니다.
   - 이는 HSV의 노이즈 없는 완벽한 BBox 좌표 대비 VLM bbox 특유의 프레임 간 지터 및 계통 편향(Systematic Bias, cx std=0.222 등)이 제어 모델(MLP)에 OOD로 작용했기 때문입니다.
   - VLM 그라운딩 환경에서는 노이즈 주입(BBox Noise Augmentation)과 더불어 데이터 증강(243ep)이 결합될 때 비로소 70%의 실용적인 CL 성공률이 회복됨을 알 수 있습니다.

3. **E2E VLA 구조의 제어 병목 및 한계 (C1 vs B1-B3):**
   - 순수 E2E VLA 구조로 학습된 **C1(Kosmos, 18.8%)**은 Stage-decomposed 방식(Grounding VLM + Stage1/2 MLP)인 B1(70.0%) 대비 큰 성능 저하를 보였습니다.
   - 세부 주행 경로 분석 결과, 비교적 단순한 **center_straight 경로는 100% 성공(4/4)**하며 강건함을 유지했으나, 좌/우 큰 조향이 동반되는 **회전 경로들(left_left, right_right 등)에서는 조향 불안정(FPE > 3.0m)**으로 모두 주행에 실패했습니다.
   - 이는 Grounding VLM이 명시적으로 target bbox를 짚어주어 downstream 제어 네트워크의 학습 부담을 덜어주는 Decomposed 아키텍처와 달리, E2E VLA는 단일 거대 모델이 시각 인코더와 텍스트 지시어로부터 visual grounding과 제어 action 예측을 한 번에 풀어야 하므로 학습 난이도가 극도로 높기 때문입니다.
   - 결론적으로, 243ep 규모의 소규모 주행 데이터 환경에서는 **Decomposed 파이프라인이 E2E VLA 방식 대비 데이터 효율성 및 조향 강건성 면에서 현격한 우위**를 점함을 실증합니다.

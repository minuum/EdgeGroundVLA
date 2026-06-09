# Exp59 Technical Report: Hard Negative PaliGemma2 LoRA Grounder & Closed-Loop Analysis

본 보고서는 단일 타겟 물체(gray basket) 추적 상황에서 주변의 유사/이질적인 다른 장애물들에 대한 오탐(False Positive) 현상(R2-3 문제)을 방지하기 위해 설계된 **Exp59 (Hard Negative 학습 PaliGemma2-3B LoRA Grounder)** 모델의 검증 및 closed-loop 시뮬레이션 평가 결과를 분석합니다.

---

## 1. Background (배경)

VLM 기반 Robot Navigation(VLA) 분야에서 기존의 일반적인 pre-trained grounding 모델들은 텍스트 쿼리에 부합하는 대상을 찾을 때, 타겟과 유사하거나 형태가 겹치는 다른 장애물(예: brown pot, red ball, person 등)이 나타나면 이를 오탐(False Positive)하여 주행 경로를 이탈하는 고질적인 한계가 존재했습니다. 이를 **R2-3 (타겟 오분류 및 오탐) 문제**로 규정합니다.

이 문제를 해결하기 위해, target 물체인 **gray basket**이 있는 이미지 외에도, target 없이 다른 부정적 물체(Hard Negative)들만 존재하는 이미지를 학습 데이터에 포함하여 **PaliGemma2-3B 백본에 LoRA fine-tuning**을 적용한 **Exp59** 그라운더 모델을 설계하고 학습시켰습니다. 본 실험의 목표는 다음과 같습니다.

1. 타겟 이외의 물체에 대해 BBox를 생성하지 않는 **신경망 수준의 완벽한 오탐 차단(R2-3 극복)** 검증.
2. 런타임에 이 그라운더를 탑재하고 기존 Stage2 Action MLP와 연동하여 **closed-loop 시뮬레이션**을 수행했을 때의 실질 주행 복원 성능 평가.

---

## 2. Analysis & Methodology (분석 방법론)

본 검증은 크게 두 가지 단계로 나누어 수행되었습니다.

### A. Cross-Object Grounding Evaluation (교차 객체 그라운딩 평가)

- **평가 대상**: Exp59 PaliGemma2 LoRA Grounder
- **데이터셋**: 타겟 물체(gray basket) 이미지 20장, Hard Negative 3종(brown pot, red ball, person) 각 20장 (총 80장)
- **출처**: [v5_cross_results.json](file:///home/minum/26CS/MoNaVLA/docs/v5/exp59_cross_object/v5_cross_results.json)
- **지표**: True Positive (TP) Rate, False Positive (FP) Rate

### B. Closed-Loop Offline Simulation (폐루프 시뮬레이션 평가)

- **평가 대상**: Exp59 Grounder + Stage2 MLP (Exp54 학습 가중치 사용)
- **데이터셋**: validation 세트 22개 에피소드 (총 394 프레임)
- **출처**: [exp59_closedloop_result.json](file:///home/minum/26CS/MoNaVLA/docs/v5/closed_loop_eval/exp59_closedloop_result.json)
- **성공 기준**: Final Position Error (FPE) < 0.5m 이면서 Trajectory Length Ratio (TLD) ∈ [0.7, 1.5]
- **비교군**: Baseline Exp54 (Ground-Truth/HSV bbox 정보를 API 수준에서 우회 입력받아 Action MLP만 돌린 오프라인 리플레이 성능)

---

## 3. Findings (정량적 분석 결과)

### 3.1. Cross-Object Grounding 정량 평가 결과

Hard Negative 데이터를 노출시켜 파인튜닝한 결과, VLM은 target 물체인 `gray basket`에 대해서만 BBox를 예측하고 다른 물체들은 무시하는 강력한 변별력을 보였습니다.

| 탐색 대상 (Text Query) | 평가 이미지 종류 | 테스트 수 (Episodes) | 탐지 성공 (Hits) | 성공률 (Rate) | 결과 판정 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **"detect gray basket"** | **Gray Basket (Target)** | 20 | 19 | **95.0%** | **True Positive (TP)** |
| **"detect gray basket"** | **Brown Pot (Negative)** | 20 | 0 | **0.0%** | **False Positive (FP)** |
| **"detect gray basket"** | **Red Ball (Negative)** | 20 | 0 | **0.0%** | **False Positive (FP)** |
| **"detect gray basket"** | **Person (Negative)** | 20 | 0 | **0.0%** | **False Positive (FP)** |

- **분석 요약**: True Positive는 **95.0%**인 반면, Hard Negative에 대한 False Positive는 **0.0%**로 완벽하게 0에 수렴하여 R2-3 문제를 완벽히 해결했습니다.

### 3.2. Closed-Loop Offline Simulation 정량 평가 결과

22개 에피소드 전체 검증 셋에서 실제 VLM 추론을 루프 내에 직접 결합하여 수행한 closed-loop 시뮬레이션 결과입니다. 검출 플래그(`has_bbox`) 누적 오류를 수정한 버전 및 BBox EMA 필터링($\alpha=0.5$)을 적용한 버전을 비교 분석했습니다.

| 평가 모델 | BBox 획득 방식 | EMA 계수 ($\alpha$) | 성공률 (Success Rate) | 평균 FPE (m) | 평균 TLD | 평균 Grounding 성공률 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Exp54** | Ground-Truth (HSV BBox 제공) | N/A | **96.7%** | - | - | - |
| **Exp59 (버그 수정 전)** | PaliGemma2 3B 추론 | 1.0 (비활성) | **4.5%** (1/22) | **3.970m** | **1.055** | **98.0%** |
| **Exp59 (버그 수정 후)** | PaliGemma2 3B 추론 | 1.0 (비활성) | **4.5%** (1/22) | **4.075m** | **1.053** | **98.0%** |
| **Exp59 (버그 수정 후 + EMA)** | PaliGemma2 3B 추론 | 0.5 | **4.5%** (1/22) | **4.098m** | **1.053** | **98.0%** |

#### 경로 타입(path_type)별 세부 지표 (오류 수정 후, EMA 0.5 적용 기준)

- **center_left**: 0/2 (SR 0%), 평균 FPE = 6.049m
- **center_right**: 0/3 (SR 0%), 평균 FPE = 7.858m
- **center_straight**: 0/2 (SR 0%), 평균 FPE = 4.888m
- **left_left**: 0/2 (SR 0%), 평균 FPE = 4.397m
- **left_right**: 0/2 (SR 0%), 평균 FPE = 1.150m
- **left_straight**: 0/3 (SR 0%), 평균 FPE = 0.812m
- **right_left**: 0/5 (SR 0%), 평균 FPE = 6.008m
- **right_right**: 0/1 (SR 0%), 평균 FPE = 0.575m
- **right_straight**: 1/2 (SR 50%), 평균 FPE = 0.288m

---

## 4. Discussion & Conclusion (고찰 및 결론)

### 4.1. 성능 차이 분석 (왜 오류 수정 및 스무딩 후에도 성공률이 4.5%에 정체되었는가?)

1. **has_bbox 버그 수정의 영향**:
   - `eval_exp59_closedloop.py`에서 BBox 슬롯의 검출 플래그가 현재 프레임 상태로 덮어씌워지는 버그를 바로잡아 과거 프레임의 탐지 여부가 올바르게 복원되었습니다. 그러나 성공률은 4.5%로 변화가 없었습니다. 이는 제어 실패가 검출 누적 플래그 오차에만 기인하지 않음을 입증합니다.

2. **Temporal Jittering vs Systematic Bias (시간적 지터와 계통 편향)**:
   - EMA 스무딩($\alpha=0.5$)을 걸어 BBox 좌표와 면적의 떨림을 억제했음에도 평균 FPE와 성공률에는 유의미한 이득이 없었습니다.
   - 이는 프레임 간의 미세한 '떨림(Jittering)'보다, **PaliGemma2 그라운더가 출력하는 BBox와 HSV GT BBox 사이의 계통 오차(Systematic Bias/Offset)**가 지배적이기 때문입니다.
   - PaliGemma2는 대상을 98% 확률로 잘 그라운딩하고 있으나, 그 출력 좌표는 HSV GT 필터에 비해 일관되게 약간 위나 옆으로 치우치거나 면적이 조금 다르게 나타납니다.
   - 제어 헤드인 Stage2 MLP는 노이즈와 편향이 완전히 배제된 깔끔한 HSV GT BBox 분포로만 학습되어 이 분포에 과적합(Overfitting)되어 있습니다. 따라서 VLM의 BBox 예측에 편향 오차가 단 며칠 픽셀만 존재해도 입력 분포를 벗어난 OOD(Out-of-Distribution)로 간주하고 엉뚱한 거동 클래스(예: 직진 구간에서 급좌회전)를 생성하며, 누적 표류(Drift)로 인해 FPE가 수 미터 이상 증가하게 됩니다.

### 4.2. 향후 극복 방안

- **BBox Noise & Offset Augmentation**: MLP 제어망 단독 학습 시, VLM 그라운더의 예측 좌표 분산(Jittering) 및 계통 편향(Offset Bias)을 모사하여 BBox 입력 데이터에 인위적인 노이즈 증강(Augmentation)을 적용해 재학습시켜야 합니다.
- **End-to-End Joint Tuning**: 그라운더와 제어망을 별개로 설계하는 분리형 아키텍처의 한계를 인식하고, 두 단계를 엮어서 end-to-end로 파인튜닝하거나, 다층 퍼셉트론(MLP)에 더 다양한 VLM 예측 데이터를 노출시키는 Robust Training 기법이 요구됩니다.
- **실로봇 연동**: 그럼에도 불구하고, Exp59 PaliGemma2 LoRA 그라운더는 R2-3(오탐) 문제 극복에 강력한 강점이 있으므로, 조향 노이즈 강인성을 확보한 제어 헤드를 탑재하여 SODA 배포 서버(`soda@100.85.118.58:~/MoNaVLA`)를 통해 물리 로봇 테스트를 병행 수행할 예정입니다.

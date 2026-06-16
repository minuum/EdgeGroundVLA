# MoNaVLA V5 용어 사전 (Glossary)

**작성일**: 2026-05-12  
**대상**: 프로젝트 참여자 전원 — 새로 합류한 사람, 교수님 보고 준비, 논문 작성 시 기준 문서로 사용

---

## 목차

1. [평가 지표](#1-평가-지표)
2. [테스트 방식](#2-테스트-방식)
3. [모델 구조 및 접근법](#3-모델-구조-및-접근법)
4. [실험 체계](#4-실험-체계)
5. [데이터셋](#5-데이터셋)
6. [실패 패턴 및 병리](#6-실패-패턴-및-병리)
7. [학습 세부 용어](#7-학습-세부-용어)
8. [액션 공간](#8-액션-공간)
9. [인프라 및 파일 포맷](#9-인프라-및-파일-포맷)

---

## 1. 평가 지표

### PM (Perfect Match)

**한 줄 정의**: 모델이 예측한 action class가 ground truth class와 정확히 일치한 프레임의 비율.

```
PM = (pred_class == gt_class인 프레임 수) / (전체 프레임 수)
```

**왜 사용하는가**  
실제 로봇은 각 순간에 하나의 행동 명령을 받는다. 이산 action class(예: FORWARD, LEFT, ROT_R)를 단위로 정확히 일치하는지를 보는 것이 가장 직접적인 오프라인 정책 품질 척도이기 때문이다.

**주의사항**  
- PM은 교사-강제(teacher-forced) 평가이다. 즉 모델에게 매 프레임마다 실제 과거 이미지를 주고 예측을 받는다. 이것은 실제 로봇 주행(closed-loop)과 다르다.  
- V5 데이터에서 FORWARD class가 전체 프레임의 ~74%를 차지하므로, "항상 FORWARD를 예측"하면 PM ≈ 74%가 나온다. 이런 이유로 PM 단독 수치를 기준으로 쓰면 안 되고, per-path PM(경로별 분리)과 함께 봐야 한다.  
- eval_t(window 내 어느 시점 예측을 보는가)에 따라 수치가 달라진다. V5에서는 `eval_t=0`(window 첫 프레임)으로 통일했다.

**현재 수치 기준**  
- Exp11 (end-to-end baseline): **58.6%**  
- Exp14 Step 2 (decomposition): **75.9%** (5 seed 평균)

---

### DM (Directional Match)

**한 줄 정의**: V4 시절 연속 action regression에서 쓰던 방향 일치 지표.

**현재 상태**: V5에서는 이산 분류로 전환했기 때문에 DM은 더 이상 계산하지 않는다. 스크립트 이름에 `pm_dm`이 남아 있지만, DM 부분은 비어 있다. 혼동하지 말 것.

---

### val_loss

**한 줄 정의**: 학습 중 validation 데이터셋에서 측정한 cross-entropy loss.

**왜 사용하는가**  
학습이 잘 되고 있는지 빠르게 확인하는 지표. 수렴 여부, overfitting 진입 시점, 체크포인트 선택 기준으로 사용한다.

**왜 PM이나 CL Success와 다를 수 있는가**  
val_loss가 낮아도 PM이 0%가 나오는 일이 실제로 발생했다(Exp04: val_loss 0.776 → PM 0%). 이것은 loss가 낮아도 모델이 항상 같은 class를 예측하는 class collapse에 빠질 수 있기 때문이다. 따라서 val_loss는 필요조건이지 충분조건이 아니다.

---

### IoU (Intersection over Union)

**한 줄 정의**: 모델이 예측한 bounding box와 ground truth bounding box의 겹침 비율.

```
IoU = 교집합 면적 / 합집합 면적
```

**왜 사용하는가**  
Exp10처럼 VLM이 목표물(바구니)을 이미지에서 공간적으로 얼마나 정확히 찾아내는지를 측정할 때 사용한다. 지각(perception) 능력 자체를 직접 측정하는 지표다.

**현재 수치**: Exp10 grounding — mean IoU **0.87**, IoU@0.5 기준 성공률 높음.

---

### FPE (Final Position Error)

**한 줄 정의**: closed-loop 시뮬레이션에서 에피소드가 끝났을 때 로봇의 최종 위치와 목표 위치 사이의 거리(미터).

**왜 사용하는가**  
PM 같은 frame-level 지표는 "각 프레임에서 맞았는지"를 보지만, 실제 항법에서는 누적 오차가 쌓여 마지막에 목표와 얼마나 떨어졌는지가 핵심이다. FPE는 이 누적 오차를 최종 결과로 요약한다.

**현재 수치 기준 (closed-loop sim)**  
- Exp11: mean FPE **1.45m** (방향 오류 누적으로 큰 편차)  
- Exp14 Step 2: mean FPE **0.55m**  
- 성공 기준: FPE < **0.5m** (Phase 1 설계 기준)

---

### TLD (Total Linear Distance)

**한 줄 정의**: closed-loop 시뮬레이션에서 로봇이 에피소드 동안 이동한 총 직선 거리의 합(미터).

**왜 사용하는가**  
FPE가 크더라도 TLD가 작으면 "그냥 거의 안 움직인 것"이고, TLD가 크면 "열심히 움직였지만 방향이 틀렸다"는 의미가 된다. 두 수치를 함께 보면 실패 원인을 구분할 수 있다.

**현재 수치**: Exp11과 Exp14 Step 2의 TLD는 둘 다 ≈ **1.03m**로 유사하다. 즉 이동 거리는 비슷하지만 Exp11은 방향 오류로 FPE가 2.6배 더 크다.

---

### 성공률 (Closed-Loop Success Rate)

**한 줄 정의**: 에피소드 단위로, 목표 위치에 도달하고 적절히 정지한 에피소드의 비율.

**성공 기준 (V5 Phase 1)**: FPE < 0.5m AND TLD ∈ [0.7, 1.5m] 범위 내 정지

**현재 수치**  
- Exp11: **0%** (9 에피소드 모두 실패)  
- Exp14 Step 2: **66.7%** (9 에피소드 중 6개 성공)

---

### Prefix Success / prefix@N

**한 줄 정의**: 에피소드 시작 후 처음 N 프레임만 봤을 때 올바른 방향으로 출발하는지를 측정하는 단기 성공 지표.

**왜 사용하는가**  
V5 데이터에서 가장 흔한 실패 유형은 "초반에 방향을 잘못 잡는 것"이다. prefix@5(첫 5 프레임)나 prefix@10은 전체 rollout보다 훨씬 빠르게 계산할 수 있고, 초반 경로 설정(route commitment)이 맞았는지를 직접 측정한다.

---

### Per-path PM

**한 줄 정의**: 경로 유형(center_straight, left_left, center_right 등)별로 나눠서 측정한 PM.

**왜 사용하는가**  
전체 PM은 FORWARD가 74%이기 때문에 숫자가 높아도 실제로 곡선 경로를 못 다닐 수 있다. path_type별로 쪼개야 모델이 진짜로 방향 분기를 학습했는지 알 수 있다.

---

## 2. 테스트 방식

### Offline Evaluation (오프라인 평가)

**정의**: 저장된 H5 에피소드 이미지를 재생하면서 모델 예측을 ground truth action과 비교하는 방식. 로봇이 실제로 움직이지 않는다.

**사용하는 이유**: 빠르고 재현 가능하다. 실험 초기 선별 및 학습 중 체크포인트 선택에 사용한다.

**한계**: 모델 예측이 틀려도 다음 프레임에 항상 GT 이미지가 주어진다(teacher forcing). 따라서 오류가 누적되는 실제 상황을 반영하지 못한다.

---

### Teacher-Forced Evaluation (교사-강제 평가)

**정의**: 매 time step마다 모델에게 ground truth 이미지(전 프레임의 실제 관측)를 입력으로 준 채 예측을 받는 방식.

**offline evaluation과 같은 의미로 쓰인다.** 학습 시에는 teacher forcing을 통해 ground truth 다음 입력을 줘서 빠른 수렴을 유도하기도 한다.

---

### Free-Running Rollout (자유 실행 롤아웃)

**정의**: 모델이 내뱉은 action을 실제로 적용해 다음 상태를 만들고, 그 상태를 다시 입력으로 받아 예측을 이어가는 방식.

**offline evaluation과의 차이**: teacher forcing이 없으므로, 한 번 틀리면 오류가 누적된다. 이를 **exposure bias**(노출 편향)라 한다.

---

### Closed-Loop (CL) Simulation

**정의**: 모델이 예측한 action을 kinematics 시뮬레이터에 입력하여 가상의 로봇 trajectory를 만들고, 목표 도달 여부를 에피소드 단위로 판정하는 테스트.

**왜 중요한가**  
PM이 높아도 CL에서 실패할 수 있다(Exp04, Exp11 모두 경험). 반대로 PM이 낮아도 CL에서 성공하는 경우는 거의 없다. **CL success rate가 V5의 궁극적 판정 지표**다.

**사용 스크립트**: `scripts/sim/evaluate_closed_loop_v5.py`

---

### Sanity Check

**정의**: 학습 후 모델이 각 action class를 실제로 골고루 예측하는지 빠르게 확인하는 간단한 테스트.

**왜 사용하는가**  
class collapse 여부를 빠르게 잡아내기 위해. 모델이 특정 class(주로 FORWARD)만 예측하면 PM은 높아도 쓸모가 없다. sanity check에서 left/right/rotation 등 소수 class도 예측에 등장하는지 확인한다.

---

### Feature Ablation (특징 제거 실험)

**정의**: 입력 특징의 일부를 제거하거나 고정해서, 각 특징이 성능에 얼마나 기여하는지 측정하는 실험.

**V5에서의 결과** (Exp14 Step 1/2 ablation):  
- bbox만: 67.4% ±9.8%  
- 이미지만: 75.6% ±0.8%  
- bbox + 이미지: 76.7% ±1.3%  

→ 이미지가 핵심, bbox는 보조.

---

## 3. 모델 구조 및 접근법

### VLA (Vision-Language-Action)

**정의**: 카메라 이미지(Vision) + 언어 지시(Language)를 받아 로봇 행동(Action)을 출력하는 모델 계열의 총칭.

이 프로젝트의 MoNaVLA가 VLA 계열 모델이다. 네이게이션 전문 경량 VLA를 만드는 것이 목표.

---

### VLM (Vision-Language Model)

**정의**: 이미지와 언어를 동시에 처리할 수 있는 멀티모달 사전학습 모델. MoNaVLA에서는 **Kosmos-2**를 backbone VLM으로 사용한다.

**VLM vs VLA**: VLM은 텍스트/이미지 이해까지 하는 모델이고, VLA는 거기에 action 출력까지 추가된 것.

---

### Kosmos-2

**정의**: Microsoft가 개발한 VLM. 이미지에서 객체의 bounding box를 생성할 수 있는 **grounding 능력**이 내장되어 있다.

MoNaVLA에서 두 가지 버전이 존재한다:  
- **Pure HF Kosmos-2** (`.vlms/kosmos-2-patch14-224`): 텍스트 생성/grounding 정상 작동  
- **Google-robot Kosmos-2** (`.vlms/google_robot_pretrain/...`): robot navigation용으로 post-training된 버전. `generate()` 호출 금지 — "Tin Tin Tin Roof..." 무한반복 버그 있음. 텍스트 경로(text attention)가 구조적으로 붕괴됨.

---

### LoRA (Low-Rank Adaptation)

**정의**: 대형 모델을 fine-tuning할 때 전체 가중치를 건드리지 않고, 각 레이어에 작은 저랭크 행렬 A, B를 추가해 학습하는 기법.

**왜 사용하는가**  
Kosmos-2 전체 파라미터(수억 개)를 다 학습하면 GPU 메모리가 부족하고 overfitting 위험이 크다. LoRA는 학습 파라미터 수를 극적으로 줄이면서도 fine-tuning 효과를 낸다.

**혼동 주의**: LoRA가 text collapse의 원인이 아니다. Exp15(head-only, LoRA 없음)에서도 text attention = 0%로 확인됨. 붕괴는 Google-robot backbone에서 이미 일어났다.

---

### End-to-End Policy (엔드-투-엔드 정책)

**정의**: 이미지 + 언어 지시를 직접 입력받아 action class를 한 번에 출력하는 구조. 중간 표현(예: bbox 위치)을 명시적으로 꺼내지 않는다.

**문제점 (V5에서 확인)**: 큰 모델을 end-to-end로 학습시키면 spatial grounding 정보를 제대로 활용하지 못하고 shortcut learning이나 class collapse에 빠지기 쉬웠다.

**관련 실험**: Exp01~09, Exp11

---

### Decomposition (분리형 접근)

**정의**: 문제를 두 단계로 나누는 구조.  
1. **Perception step**: VLM이 목표물의 위치를 bbox로 추출  
2. **Policy step**: bbox + 이미지 특징을 작은 MLP가 받아 action class 예측

**왜 더 잘 되는가**  
큰 VLM이 spatial information을 이미 잘 인코딩하고 있지만, 그 정보를 end-to-end action head로 전달하는 과정에서 소실된다. 명시적 중간 표현(bbox)을 꺼내면 소실을 막을 수 있다.

**관련 실험**: Exp14 Step 1, Step 2 (현재 best)

---

### MLP (Multi-Layer Perceptron)

**정의**: 단순한 완전연결 신경망. 여러 개의 선형 변환(Linear)과 활성함수(ReLU 등)를 쌓은 구조.

**V5에서의 역할**: Decomposition의 policy step에서 사용. bbox history + 16×16 이미지 특징을 받아 8-class action을 예측하는 작은 head.  
거대한 VLM과 달리 파라미터 수가 적고 학습이 빠르다.

---

### LSTM (Long Short-Term Memory)

**정의**: 시계열 데이터를 처리하는 순환 신경망의 일종. 과거 hidden state를 기억하면서 순차적으로 처리한다.

**V5에서의 역할**: 초기 MobileVLA 설계에서 LSTM을 temporal policy decoder로 사용했다. 현재 V5 주 실험에서는 window 기반 처리로 대체.

---

### Grounding (그라운딩)

**정의**: 언어로 설명된 객체("gray basket")를 이미지에서 실제 위치(bounding box)로 매핑하는 능력.

**왜 중요한가**  
항법 정책이 제대로 작동하려면 모델이 목표물이 이미지 어디에 있는지 알아야 한다. grounding이 되지 않으면 언어 지시를 무시하고 시각적 패턴에만 의존하게 된다.

**Exp10**: grounding 학습에 특화된 실험. IoU 0.87 달성. 하지만 free-form generation을 통해 action에 연결하면 34.4%로 떨어짐.

---

### Grounding Aux (Grounding Auxiliary)

**정의**: 메인 action loss 외에 auxiliary loss로 grounding head를 추가해 VLM이 목표물 위치를 명시적으로 인식하도록 유도하는 학습 기법.

**사용 이유**: text path가 붕괴된 상황에서, auxiliary grounding 손실을 추가하면 spatial information이 action head까지 흘러오도록 유도할 수 있다는 가설로 시도됨.

---

### Text Attention / Text Path

**정의**: 모델이 언어 토큰에 얼마나 attention을 할당하는지를 측정한 값.

**V5의 핵심 발견**: Google-robot backbone에서 text attention = **0.000%**. 즉 모델이 언어 지시를 완전히 무시한다. 이것은 LoRA 학습 때문이 아니라 Google-robot post-training 단계에서 이미 발생한 구조적 문제다.  
측정 스크립트: `scripts/measure_attention.py`

---

## 4. 실험 체계

### Exp (Experiment 번호)

**정의**: V5 실험 각각에 붙인 고유 번호. Exp01부터 시작해 순차적으로 증가한다.

| 범위 | 특징 |
|------|------|
| Exp01~03 | V4 기반, FORWARD collapse |
| Exp04 | Google-robot backbone 첫 도입, val_loss 좋지만 PM 0% |
| Exp09 | 8-class 시도, bias 잔존 |
| Exp10 | BBox grounding 학습, IoU 0.87 |
| Exp11 | 현재 end-to-end baseline, PM 58.6% |
| Exp12~13 | instruction conditioning 시도, 폐기 |
| Exp14 Step1/2 | decomposition 접근, **현재 best** |
| Exp15 | VLM frozen, head만 학습 → text=0% 재확인 |
| Exp16 | 150ep 8-class, center_straight 포함 |
| Exp17+ | Phase B~D 시도 (text path 회복 시도) |
| Exp46~52 | 최근 grounding/feature 실험 |

---

### V4 / V5 (데이터셋 버전)

**V4**: 구버전. `ROS_action/basket_dataset_v2/` (528 H5 에피소드). 현재 학습 미사용.  
**V5**: 현재 버전. `ROS_action/mobile_vla_dataset_v5/` (150 H5 에피소드).

**V4 → V5 포맷 변경 주의**  
- V4: `f['images']`  
- V5: `f['observations']['images']`  
구버전 스크립트를 V5에 그대로 쓰면 KeyError 발생.

---

### Phase (실험 단계)

교수님 지시 아래 V5 실험을 큰 단계로 묶는 이름.

| Phase | 내용 | 결과 |
|-------|------|------|
| Phase A | Text path 회복 시도 (Exp17~41C) | **FAIL** — text attn 0% 유지 |
| Phase B | Decomposition 정교화 | Exp14 계열 |
| Phase C+ | grounding aux, counterfactual 등 | 진행 중 |

교수 프로토콜:  
- Step 1: 곡선만 학습 → 직선도 대응? (Exp11 완료, PM 58.6%)  
- Step 2: 50/50 비율 → 동작? (Exp16 학습 중)  
- Step 3: 33/33/33 → 완전 동작?

---

### Checkpoint / ckpt

**정의**: 특정 epoch에서 저장한 모델 가중치 파일. `.ckpt` 확장자.

**Exp11 best checkpoint**:  
`runs/v5_nav/kosmos/mobile_vla_v5_exp11/2026-04-16/v5-exp11-google-robot-8cls/epoch_epoch=epoch=14-val_loss=val_loss=1.010.ckpt`

---

### pretrained_vlm_path 인헤리턴스 함정

부모 config에서 `pretrained_vlm_path`를 설정했더라도 새 child config에서 **반드시 `null`을 명시**해야 한다. 명시하지 않으면 부모의 경로를 그대로 쓰는 인헤리턴스가 발생해 엉뚱한 backbone이 로드된다.

---

## 5. 데이터셋

### H5 / HDF5 (에피소드 파일)

**정의**: 각 주행 에피소드 데이터를 저장하는 파일 포맷. `.h5` 확장자.

각 에피소드 파일 안에는:
- `observations/images`: 각 프레임의 카메라 이미지
- `observations/instruction`: 언어 지시 텍스트
- `actions`: 각 프레임에서 실행한 action (`[linear_x, linear_y, angular_z]`)

---

### Episode (에피소드)

**정의**: 로봇이 시작 위치에서 목표(바구니)에 도달하거나 종료 조건이 될 때까지의 한 번의 주행.

V5에는 총 150개 에피소드가 있고, 평균 17.5 프레임(min 14, max 19).

---

### Path Type (경로 유형)

**정의**: 에피소드의 시작 위치와 방향에 따라 분류한 9가지 경로 유형.

| Path | 설명 | 에피소드 수 |
|------|------|------------|
| center_straight | 정면 직진 | 20 |
| left_straight | 왼쪽에서 직진 | 20 |
| right_straight | 오른쪽에서 직진 | 20 |
| center_left | 중앙에서 왼쪽으로 | 15 |
| center_right | 중앙에서 오른쪽으로 | 15 |
| left_left | 왼쪽에서 왼쪽으로 | 15 |
| left_right | 왼쪽에서 오른쪽으로 | 15 |
| right_left | 오른쪽에서 왼쪽으로 | 15 |
| right_right | 오른쪽에서 오른쪽으로 | 15 |

**왜 중요한가**: straight 경로에서는 FORWARD만 잘 맞춰도 PM이 높아 보이지만, curved 경로(center_left 등)에서 회전을 제때 하지 못하면 실제 항법이 실패한다.

---

### BBox (Bounding Box)

**정의**: 이미지 내 목표 객체(바구니) 주위에 그려진 직사각형 영역. `(cx, cy, width, height)` 또는 `(x1, y1, x2, y2)`로 표현.

**V5에서의 역할**: Grounding 모델이 바구니 위치를 추정한 결과. Decomposition 접근의 핵심 중간 표현.

**bbox_dataset.json**: grounding이 성공적으로 된 45개 에피소드의 프레임별 bbox 좌표 캐시. `docs/v5/bbox_nav_step1/bbox_dataset.json`

---

### Proxy Signal

**정의**: 직접 측정하기 어려운 신호 대신, 관련 있는 다른 신호로 대체해서 학습/평가에 사용하는 간접 지표.

**V5에서의 예**: STOP 레이블이 데이터에 없기 때문에, "목표까지 bbox area가 충분히 커졌는가"나 "center에서 특정 거리 이내인가" 같은 geometry 신호를 STOP 판단의 proxy로 사용.

---

### Grounding Cache

**정의**: 전체 150 에피소드 중 grounding이 성공한 45개 에피소드의 bbox 결과를 미리 저장해 둔 파일.

BBox nav step 1/2 학습에 사용. 전체 150 에피소드를 다 쓰지 못하는 이유는 일부 에피소드에서 Kosmos-2 grounding이 실패했기 때문.

---

## 6. 실패 패턴 및 병리

### FORWARD Bias (포워드 편향)

**정의**: 모델이 입력에 관계없이 FORWARD action만 예측하는 현상.

**왜 발생하는가**: V5 데이터에서 FORWARD가 ~74%를 차지한다. 모델이 class imbalance를 극복하지 못하면 "항상 FORWARD"가 최소 손실 전략이 된다.

**탐지 방법**: sanity check — per-class 예측 분포를 확인. FORWARD만 100%면 FORWARD bias.

---

### Class Collapse (클래스 붕괴)

**정의**: 모델이 하나 또는 소수의 action class만 예측하고 나머지를 전혀 예측하지 않는 현상.

FORWARD bias는 Class Collapse의 특수한 형태. 다른 형태로는 `FWD+R` collapse(항상 전진+오른쪽)도 관찰됐다.

**대표 사례**: Exp04 val_loss 0.776 → PM 0% (FORWARD collapse), Exp21 → FWD+R collapse

---

### Text Path Collapse (텍스트 경로 붕괴)

**정의**: 모델이 언어 지시를 완전히 무시하고 시각 정보에만 의존하는 현상. text attention = 0%로 측정.

**원인**: Google-robot post-training 단계에서 이미 발생. LoRA 학습 때문이 아님. Exp15(head-only, LoRA 없음)에서도 재확인.

---

### Trajectory Divergence (궤적 발산)

**정의**: closed-loop에서 초반에 한두 번의 잘못된 action이 누적되어 전체 경로가 목표에서 크게 벗어나는 현상.

**왜 위험한가**: frame-level PM이 높아도 한 번의 방향 오류가 로봇을 완전히 다른 방향으로 보낼 수 있다. Exp11의 0% CL success는 trajectory divergence가 주원인.

---

### Exposure Bias (노출 편향)

**정의**: 학습 시(teacher-forced)에는 GT 이미지를 주지만, 실제 추론(free-running)에서는 자신의 이전 예측 결과를 기반으로 행동해야 하는 간극.

**결과**: teacher-forced PM이 높아도 free-running에서는 오류가 누적되면서 성능이 떨어진다.

---

### Shortcut Learning (숏컷 학습)

**정의**: 모델이 과제의 본질적 패턴 대신, 데이터의 통계적 단서(예: 색상, 배경, 위치 편향)를 기반으로 예측하는 것.

**V5 맥락**: end-to-end policy가 언어 지시나 spatial reasoning 대신 "직진이 제일 많았으니 직진"이라는 단순 통계를 학습하는 경우.

---

### Action Attractor

**정의**: 모델이 특정 action에 "빨려들어가는" 강한 편향. 어떤 입력을 줘도 그 action만 출력함.

Class Collapse와 비슷한 현상을 더 동적으로 표현한 용어. 특히 sequence 추론에서 이전 hidden state가 특정 action으로 수렴해 고착되는 현상.

---

### Oscillation (진동)

**정의**: closed-loop에서 LEFT → RIGHT → LEFT를 반복하거나 ROT_L → ROT_R을 반복하는 비생산적 행동 패턴.

**failure taxonomy 코드**: `oscillation`

---

## 7. 학습 세부 용어

### Window (슬라이딩 윈도우)

**정의**: 에피소드에서 연속된 N 프레임을 묶어 하나의 학습/평가 단위로 만드는 방법.

V5에서는 `window_size=8`이 기본 설정. 8 프레임의 이미지 시퀀스를 받아 action 시퀀스를 예측.

---

### Action Chunking (액션 청킹)

**정의**: 한 번의 forward pass에서 현재 time step 하나만 예측하는 대신, 미래 N step의 action을 동시에 예측하는 방식.

V5에서는 `fwd_pred_next_n=5` 설정. 즉 현재 프레임에서 다음 5 step의 action을 동시에 예측한다.

**배열 형태**: `action_chunck.shape = (batch, window_size, fwd_pred_next_n)` = `(bs, 8, 5)`

---

### eval_t

**정의**: window 내의 어느 time step의 예측을 평가 지표 계산에 사용할지를 정하는 파라미터.

- `eval_t=0`: window 첫 프레임 예측 (V5 표준)  
- `eval_t=-1`: window 마지막 프레임 예측 (history를 가장 많이 활용)

**중요**: V5에서는 2026-04-16 버그 수정 이후 `eval_t=0`으로 통일. 과거 보고 수치가 `eval_t=-1`이었을 경우 직접 비교 불가.

---

### Counterfactual Training (반사실 학습)

**정의**: "이 instruction이 아닌 다른 instruction을 줬을 때 다른 action이 나와야 한다"는 negative example을 함께 학습에 사용하는 방식.

**왜 사용하는가**: 모델이 text를 무시하고 이미지만 보는 것을 막기 위해. text를 바꿨을 때 prediction도 바뀌어야 한다는 학습 신호를 추가.

**현재 상태**: config flag로 활성화 가능. Exp42에서 처음 사용됨.

---

### Stratified Split (층화 분할)

**정의**: 각 path_type이 train/test 분할에서 균등하게 포함되도록 분할하는 방법.

**왜 사용하는가**: 단순 random split 시 특정 path_type이 test에 몰릴 수 있음. BBox nav step 1/2에서는 9 path × 80/20 stratified split을 사용.

---

### Per-class Accuracy / Confusion Matrix

**정의**:  
- Per-class accuracy: 각 action class별로 정답률을 따로 계산한 것  
- Confusion matrix: 예측 class × 실제 class의 2D 행렬. 어떤 class를 어떤 class로 잘못 예측하는지 시각화

**왜 중요한가**: aggregate PM은 FORWARD bias를 숨긴다. confusion matrix에서 LEFT/RIGHT/ROT 계열의 recall이 낮은지 확인해야 한다.

---

## 8. 액션 공간

### 8-class Action Space (V5 표준)

V5 학습(Exp09 이후)에서 사용하는 이산 action class.

| Index | 이름 | linear_x | linear_y | angular_z | 비고 |
|-------|------|----------|----------|-----------|------|
| 0 | STOP | 0 | 0 | 0 | 데이터에 없음 — 에피소드 끝 프레임에 합성 |
| 1 | FORWARD | 1.0 | 0 | 0 | 전체 프레임의 ~74% |
| 2 | LEFT | 0 | 1.0 | 0 | 좌측 이동(strafe) |
| 3 | RIGHT | 0 | -1.0 | 0 | 우측 이동(strafe) |
| 4 | FWD+LEFT | 1.0 | 1.0 | 0 | 전진+좌측 대각선 |
| 5 | FWD+RIGHT | 1.0 | -1.0 | 0 | 전진+우측 대각선 |
| 6 | ROT_L | 0 | 0 | 1.0 | 제자리 왼쪽 회전, ~0.8% |
| 7 | ROT_R | 0 | 0 | -1.0 | 제자리 오른쪽 회전, ~0.8% |

---

### 9-class vs 8-class 혼동 주의

**inference_server.py**는 9-class 매핑을 사용한다. V5 학습은 8-class. 서버 배포 시 action class 인덱스 매핑을 반드시 확인해야 한다. 그냥 쓰면 index가 어긋난다.

---

### STOP 합성 레이블

실제 V5 데이터에는 STOP 레이블이 없다(`raw_action_counts['STOP'] = 0`). 학습 시에는 에피소드 마지막 프레임에 STOP class를 합성(synthetic label)해서 추가한다.

---

## 9. 인프라 및 파일 포맷

### ROS (Robot Operating System)

**정의**: 로봇 소프트웨어 개발을 위한 오픈소스 프레임워크. 데이터 수집(collector)과 실시간 inference 서버가 ROS로 작동한다.

`ROS_action/` 디렉토리 아래 로봇 제어 패키지와 데이터 수집 스크립트가 있다.

---

### Inference Server

**정의**: 실시간으로 카메라 이미지를 받아 모델 추론 결과(action class)를 로봇에 전달하는 서버.

**파일**: `robovlm_nav/serve/inference_server.py`  
**주의**: 9-class 매핑을 사용하고 있어 8-class 학습 모델과 클래스 매핑이 다를 수 있다.

---

### Gradio Data Collector

**정의**: 데이터 수집 시 사용하는 웹 UI 기반 도구. 로봇 주행을 모니터링하고 에피소드를 H5로 저장한다.

**파일**: `scripts/gradio_data_collector.py`

---

### menemory / MEMORY.md

**정의**:  
- **menemory**: 프로젝트 장기 기억 시스템. `.menemory/core/master_memory.md`에 저장. `menemory status`, `menemory show` 명령으로 조회.  
- **Claude auto-memory**: `~/.claude/projects/.../memory/MEMORY.md`에 저장. 사용자 프로필, 피드백, 작업 방식 선호도.

두 시스템은 별개다. menemory는 프로젝트 장기 목표/아키텍처 원칙, Claude memory는 사용자-Claude 협업 패턴을 저장한다.

---

### configs/ (설정 파일)

**정의**: 각 실험의 hyperparameter, 데이터 경로, 모델 구조 설정을 담은 JSON 파일.

**주의**: 부모 config를 상속하는 child config에서 `pretrained_vlm_path`를 반드시 `null`로 명시하지 않으면 잘못된 backbone이 로드될 수 있다.

---

### third_party/RoboVLMs/

**정의**: 외부 라이브러리로 가져온 RoboVLMs 코드. Kosmos-2 backbone 코드가 여기 있다.

**절대 수정 금지.** 이 디렉토리를 수정하면 upstream과의 동기화가 깨진다.

---

## 빠른 참조 — 핵심 수치 요약

| 지표 | Exp11 (end-to-end) | Exp14 Step 2 (decomposition) |
|------|-------------------|------------------------------|
| PM (offline, full val) | 58.6% | 75.9% |
| val_loss | 1.010 | — |
| CL Success Rate | **0%** | **66.7%** |
| mean FPE | 1.45m | 0.55m |
| mean TLD | 1.03m | 1.03m |

---

## 관련 문서

- PM 계산 방식 상세: [`docs/v5/PM_METHODOLOGY.md`](./PM_METHODOLOGY.md)  
- 평가 프로토콜 4단계: [`docs/v5/V5_EVALUATION_PROTOCOL.md`](./V5_EVALUATION_PROTOCOL.md)  
- Closed-loop 시뮬레이션 설계: [`docs/v5/V5_CLOSED_LOOP_SIM_PLAN.md`](./V5_CLOSED_LOOP_SIM_PLAN.md)  
- 전체 실험 이력: [`docs/ALL_EXPERIMENTS_MASTER_LIST.md`](../ALL_EXPERIMENTS_MASTER_LIST.md)  
- 프로젝트 최신 현황: [`CLAUDE.md`](../../CLAUDE.md)  
- 에이전트 진입점: [`docs/AGENT_ENTRYPOINT.md`](../AGENT_ENTRYPOINT.md)

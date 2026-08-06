# 📘 MoNaVLA 학습 방법론 및 Unseen 에피소드 검증 체계 종합 가이드

본 문서는 **MoNaVLA (EdgeGround-VLA)** 모델의 2단계 학습 파이프라인(Stage 1 & Stage 2), `Unseen Episodes` 독립 검증 체계(`SPLIT_SEED=42`), 및 `Hold-aware` 슬라이딩 윈도우 메커니즘을 총정리한 학술·기술 가이드입니다. 

---

## 📌 주요 용어 표준화 (Terminology Mapping)

교수님 미팅 및 논문(ICRA, IROS, CVPR 등) 작성 시 딱딱한 표현 대신 **글로벌 학술 표준 용어**로 통합하여 사용합니다:

- ❌ `Held-out Validation` ➔ ✅ **`Unseen Episodes Validation`** (학습 시 단 한 번도 관측되지 않은 33개 에피소드 검증셋)
- ❌ `Held-out Set` ➔ ✅ **`Independent Validation Set`** (학습 집합과 완벽히 독립된 검증셋)
- ❌ `Frame-level Split` ➔ ✅ **`Episode-disjoint Split`** (에피소드 간 프레임 겹침이 0.00%인 비중복 분할)

---

## 1. 🎯 `SPLIT_SEED=42` 및 `Unseen Episodes` 검증 체계

### 1.1 해결하고자 한 핵심 문제: 시간적 데이터 누출 (Temporal Leakage)
- **프레임 단위 무작위 Split의 문제**:
  - 로봇 주행 데이터는 연속된 프레임 간 시각적 배경, 조명, 타겟 위치가 거의 동일함.
  - 동일 에피소드 내 `t=1, 3` 프레임이 Train에, `t=2, 4` 프레임이 Val에 들어가면, 모델이 배경과 시퀀스를 미리 암기(Data Leakage)하여 **오프라인 검증 정확도가 94.5%로 심각하게 부풀려지는 오버피팅 착시**가 발생함.
  - 이 상태로 배포 시 **실시간 로봇 주행 성공률은 0%**로 실패함.
- **`Unseen Episodes` (Episode-disjoint) 격리 방식**:
  - 전체 225개 에피소드 중 **33개 에피소드 전체(약 2,500 프레임)를 통째로 떼어내어 Unseen Validation Set으로 격리**.
  - Train(192ep)과 Unseen Val(33ep) 사이의 시공간 프레임 중복률은 **0.00%**.

### 1.2 `SPLIT_SEED=42` 고정의 목적
- 난수 생성기(`np.random.default_rng(42)`) 시드를 42로 엄격히 고정함으로써, **백본 모델 변경(OWL-v2 vs Florence-2), 하이퍼파라미터 스위핑 등 어떤 실험을 하더라도 항상 100% 동일한 33개 에피소드가 Unseen Validation Set으로 유지**됨.
- 실험 간 공정한 비교(Fair Comparison) 및 학술적 재현성(Reproducibility) 완벽 확보.

---

## 2. ⏳ `Hold-aware` 슬라이딩 윈도우 메커니즘

### 2.1 서빙 런타임 제어와의 정합성 (Alignment)
- **서빙 런타임 환경**: `grounding_skip_n=3` (실효 1.3Hz) Zero-Order Hold (ZOH)로 바운딩 박스를 유지하며, 제어 루프도 약 5프레임 간격(\(t \sim t+5\))으로 조향 행동을 래치(Latch)함.
- **학습 시 Hold-aware 라벨링 (stride=5 다수결)**:
  - 매 프레임(1-stride) 단위 독립 라벨로 학습시키면 런타임 래치 제어 특성과 불일치하여 미세한 조향 떨림(Jittering) 발생.
  - 이를 해결하기 위해 `stride=5` 간격 이동 윈도우 내 연속 5개 Ground-Truth 라벨 중 **다수결(Majority Voting)**로 대표 행동을 채택하는 Hold-aware 샘플링 적용.
  - **결과**: 런타임 Zero-Order Hold 제어 환경과 학습 데이터셋을 일치시켜 89% 실주행 성공 시 조향 래칭(Latch) 안정성 확보.

---

## 3. 🏗️ 2단계 파이프라인 (Stage 1 & Stage 2)

```text
┌──────────────────────────────────────────────────────────────────┐
│ [Stage 1] 오프라인 피처 사전 추출 & 캐싱 (1회 수행)               │
│ - HDF5 225ep (16,599 프레임) 입력                                │
│ - Frozen OWL-v2 (0.155B) ➔ bbox 4ch (cx, cy, area, has_bbox)     │
│ - Frozen Kosmos-2 vision (0.303B) ➔ image_proj ➔ L2 norm 256ch   │
│ ➔ 260차원 특징 벡터 추출 후 에피소드 캐시 파일(*_vis_cache.pt) 저장 │
└──────────────────────────────────────────────────────────────────┘
                                │
                                ▼ (캐시 파일 저장 후 백본 연산 생략)
┌──────────────────────────────────────────────────────────────────┐
│ [Stage 2] 경량 행동 헤드 고속 반복 학습 (수초 내 완료)             │
│ - 저장된 260차원 캐시 로드 ➔ W=6 Hold-aware 윈도우 (1,560차원)    │
│ - 백본 인코더 전체 Frozen (추가 연산 0)                          │
│ - 오직 0.866M MLPActionHead (512➔128➔8) 만 300 Epoch 반복 학습    │
└──────────────────────────────────────────────────────────────────┘
```

### 3.1 Stage 2 경량 MLP 행동 헤드(0.866M)의 3대 장점
1. **극적 학습 속도 향상**: 백본 재계산 없이 1회 학습시간을 수십 분에서 **3~5초**로 단축하여 실험 회전율 극대화.
2. **Catastrophic Forgetting 방지**: 무거운 VLM/OVD 백본을 건드리지 않아 시각 표현 능력이 훼손되지 않음.
3. **sub-ms 연산 & 제어 안정성**: OVD 지연(1,901.7ms) 속에서도 sub-ms 추론으로 제어 병목 차단.

---

## 🗣️ 교수님 미팅용 1분 브리핑 스크립트

> **"교수님, 저희 학습 및 검증 체계는 3가지 핵심 설계로 완성되었습니다.**
> 
> **첫째, 데이터 검증셋은 프레임 무작위 셔플링 시 발생하는 배경 암기 착시(Data Leakage)를 막기 위해 전체 225개 에피소드 중 33개 에피소드를 통째로 떼어낸 'Unseen Episode Validation Set (독립 검증셋)' 방식입니다. `SPLIT_SEED=42`로 고정하여 어떤 백본 실험에서도 100% 동일한 미학습 33개 에피소드로 공정하게 평가됩니다.**
> 
> **둘째, 엣지 1.3Hz Zero-Order Hold 서빙 제어 환경과의 정합을 위해 `stride=5` 다수결 라벨링을 적용하는 'Hold-aware' 윈도우 기법으로 조향 래칭(Latch) 안정성을 확보했습니다.**
> 
> **셋째, 무거운 OVD와 VLM 백본 특징을 오프라인에서 1회만 pre-extract하고, 백본을 Frozen시킨 채 단 0.866M의 경량 MLP 헤드만 학습시키는 Stage 2 파이프라인으로 3초 만에 고속 학습을 수행하였습니다.**
> 
> **이 엄격한 Unseen 에피소드 검증 환경을 거쳤기에, 실제 로봇 100회 실주행 평가에서도 89%의 높은 성공률을 달성할 수 있었습니다."**

---

## 📝 논문 수록용 최종 공식 문장 (국문 / 영문)

### 🇰🇷 국문 (논문 5.2절)
> **"본 연구는 데이터 누출 차단과 연산 효율성을 극대화하기 위해 에피소드 비중복 분할(Episode-disjoint Split) 및 2단계(Two-stage) 파이프라인을 구축하였다.**
> 
> **첫째, 연속된 프레임 간 정보 누출(Temporal Data Leakage)을 차단하고자 225개 에피소드 중 33개 에피소드 전체를 훈련에서 완벽히 배제한 독립 검증 집합(Unseen Episode Validation Set, 15%)을 구축하였다. 난수 시드는 `SPLIT_SEED=42`로 고정하여 모든 비교 실험이 동일한 관측 이력이 없는 검증 에피소드 상에서 공정하게 평가되도록 하였다.**
> 
> **둘째, 런타임 Zero-Order Hold 제어 특성과의 정합을 위해 5-stride 이동 윈도우 내 다수결 라벨링(Majority Voting)을 적용하는 Hold-aware 샘플링 방식을 채택하였다.**
> 
> **셋째, 학습은 2단계로 수행된다. Stage 1에서 무거운 백본 인코더(OWL-v2 & Kosmos-2 vision)의 260차원 특징을 1회 사전 추출하여 캐싱하고, Stage 2에서는 백본 전체를 Frozen 상태로 유지한 채 1,560차원 입력에 대해 0.866M 파라미터의 경량 MLP 행동 헤드만을 3~5초 내에 고속 반복 학습시킴으로써 극적인 연산 효율성과 실시간 제어 안정성을 달성하였다."**

### 🇬🇧 영문 (Section 5.2 Training & Evaluation Methodology)
> **"To prevent temporal data leakage and maximize computational efficiency, we establish an episode-disjoint split alongside a two-stage training pipeline.**
> 
> **First, to prevent frame-level temporal leakage, we isolate 33 complete episodes (15% of the 225 episodes) as an independent Unseen Episode Validation Set. We fix the random seed at `SPLIT_SEED=42` to ensure that all model variants are benchmarked against identical, unobserved evaluation trajectories.**
> 
> **Second, to align with the runtime Zero-Order Hold control cadence, we adopt a Hold-aware sampling strategy using a 5-stride sliding window with majority voting labeling.**
> 
> **Third, Stage 1 pre-extracts 260-dimensional hybrid features (4-ch geometry + 256-ch vision) offline. In Stage 2, keeping the backbone frozen, we train a lightweight 0.866M MLP action head on the flattened 1,560-dimensional temporal window within seconds, achieving superior data efficiency and 89% real-robot navigation success."**

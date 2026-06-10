# [실험 계획서] PaliGemma LoRA 아키텍처 재설계 Ablation Study

## Background

* **일시:** 2026-06-04
* **작성자:** Speaker*2 (민우 이)
* **목적:** 6월 4일 회의 결과에 따라 PaliGemma 기반 Mobile VLA 모델의 LM(Language Model) 과적합 문제를 해소하고, OOD(Out-of-Distribution) 주행 강건성 및 조향 제어 안정성을 극대화하기 위한 4단계 Ablation Study 실험 세트를 설계함.

---

## Analysis (실험 조건 및 환경)

### 1. 학습 환경 및 데이터 사양

* **데이터셋:** 실물 로봇 주행 데이터 총 **224개 에피소드** (에피소드당 평균 18~20 프레임, 총 약 4,000 프레임)
* **학습 하이퍼파라미터:**
  * **Epochs:** 10 epochs
  * **Effective Batch Size:** 8 (Batch Size 1 × Accumulate Grad Batches 8)
  * **Learning Rate:** 1e-4
  * **Optimizer:** AdamW (Weight Decay 0.01)
  * **Precision:** bf16-mixed
* **하드웨어 제약:** A5000 24GB GPU 단일 환경에서 OOM(Out of Memory) 없이 학습 가능하도록 설정 조율.

### 2. 정량적 검증 메트릭 (Evaluation Metrics)

1. **Closed-Loop 주행 성공률 (Success Rate %):** OOD 환경(새로운 객체 배치 및 경로 방해물 등)을 포함하여 총 20회 실주행 테스트 진행 중, 목표 지점 오차 0.2~0.3m 이내 도달 비율 측정.
2. **조향 오버슈팅 및 이탈 빈도 (Overshoot Count):** 경로 추종 중 목표 각도 대비 15도 이상 오차가 누적되는 조향 지터(jitter) 발생 횟수.
3. **그라운딩 편차 (Grounding Robustness):** "grey basket" vs "grey container" vs "target object" 명령어 변경 시 BBox IoU 차이 및 검출 정확도 측정.
4. **추론 속도 및 메모리 점유율 (Inference Latency & VRAM):** 실물 배포 기준 지연속도(ms) 및 GPU 점유 메모리(GB) 측정 (실시간 제어 목표치: < 100ms).

---

## Proposed Ablation Experiments

제안하는 실험 ablation 조건은 아래와 같이 비전 인코더의 LoRA 적용 범위(상위 4개 vs 6개 레이어)를 포함해 세분화하여 설계합니다.

| 실험 ID | 실험 명칭 | LM (Gemma) LoRA | 비전 인코더 (SigLIP) LoRA | Projector 튜닝 | 비전 백본 구조 |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Exp 1** | **Baseline (기존 설정)** | Enabled (q,v,k,o_proj) | Enabled (q,v,k,o_proj) | Frozen | SigLIP 단일 |
| **Exp 2-A** | **LM Frozen + Vision Top-4 LoRA** | **Frozen** | **Enabled (상위 4개 레이어)** | Frozen | SigLIP 단일 |
| **Exp 2-B** | **LM Frozen + Vision Top-6 LoRA** | **Frozen** | **Enabled (상위 6개 레이어)** | Frozen | SigLIP 단일 |
| **Exp 3-A** | **LM Frozen + Vision Top-4 LoRA + Projector FT** | **Frozen** | **Enabled (상위 4개 레이어)** | **Full Fine-Tuned** | SigLIP 단일 |
| **Exp 3-B** | **LM Frozen + Vision Top-6 LoRA + Projector FT** | **Frozen** | **Enabled (상위 6개 레이어)** | **Full Fine-Tuned** | SigLIP 단일 |
| ~~Exp 4-A~~ | ~~Hybrid Vision (SigLIP+DINOv2) Top-4~~ | — | — | — | **❌ 폐기 (2026-06-10)** |
| ~~Exp 4-B~~ | ~~Hybrid Vision (SigLIP+DINOv2) Top-6~~ | — | — | — | **❌ 폐기 (2026-06-10)** |

### 세부 설정 및 구현 방식

#### Exp 1: Baseline (기존 설정)

* **목적:** 기존 구현체와 성능 대조군 마련.
* **설정:** `lora_target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]`를 지정하여 LM과 SigLIP 전체 attention에 LoRA 적용.

#### Exp 2: LM Frozen + Vision (SigLIP) LoRA (Top-4 vs Top-6)

* **목적:** LM 튜닝을 배제하여 특정 텍스트 프레이즈에 과적합되는 현상을 막고, 비전 피처 인코딩의 상위 수준 표상만 LoRA 튜닝.
* **설정:** LM 모듈 이름을 타깃에서 제외함.
  * **Exp 2-A (Top-4):** `vision_tower.vision_model.encoder.layers.23~26` (상위 4개, Layer 23, 24, 25, 26)에만 LoRA 적용.
  * **Exp 2-B (Top-6):** `vision_tower.vision_model.encoder.layers.21~26` (상위 6개, Layer 21, 22, 23, 24, 25, 26)에만 LoRA 적용.

#### Exp 3: LM Frozen + Vision LoRA (Top-4 vs Top-6) + Projector FT

* **목적:** LM을 Frozen 상태로 고정한 후, 비전 피처를 언어 모델 토큰 공간으로 사상하는 멀티모달 어댑터(`mm_projector`)를 Full Fine-tuning하여 비전-액션 간 최적의 의미론적 정렬 유도.
* **설정:**
  * **Exp 3-A (Top-4 + Projector FT):** Exp 2-A 설정 기반, `configs` 내 `"tune_mm_projector": true` 활성화.
  * **Exp 3-B (Top-6 + Projector FT):** Exp 2-B 설정 기반, `configs` 내 `"tune_mm_projector": true` 활성화.

#### ~~Exp 4: Hybrid Vision (SigLIP + DINOv2) LoRA~~ — ❌ 폐기 (2026-06-10)

> **폐기 사유:** 이 실험은 6/4 회의 기록의 "결정사항 1(SigLIP+DINOv2 적용)"에 근거했으나, 해당 결정은 **잘못된 전제**였음. PaliGemma2의 비전 인코더는 **SigLIP(SigLIP2) 단일 구조**이며 DINOv2를 포함하지 않는다. DINOv2+SigLIP 이중 인코더는 OpenVLA/Prismatic 계열 구조로 우리 backbone과 무관함. 따라서 Exp4(하이브리드)는 전면 폐기하고, 비전 LoRA는 SigLIP 단일(Exp2/Exp3) 범위로 한정한다.

---

## Conclusion & Action Items

* **우선순위 실행 계획:** **Exp 1 → Exp 2-A/B → Exp 3-A/B** 단일 백본(SigLIP) 기반 Ablation 실험으로 LM Frozen의 과적합 해결 효과를 검증함. (Exp4 하이브리드는 2026-06-10 폐기 — 상단 참조)
* **학습 로그 분석:** 각 실험별 TensorBoard 학습 로그(Loss curve) 및 Closed-Loop 주행 성공률 메트릭을 수집하여 `docs/v5/` 디렉토리에 정량 평가 결과 문서를 누적 및 최신화할 예정임.

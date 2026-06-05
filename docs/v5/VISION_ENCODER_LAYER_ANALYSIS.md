# [분석 리포트] SigLIP 및 DINOv2 이종 비전 인코더 레이어 수준 세부 분석 및 최적화 전략

## Background
* **일시:** 2026-06-04
* **작성자:** Speaker*2 (민우 이)
* **목적:** PaliGemma 및 OpenVLA 계열에서 사용되는 두 가지 비전 백본(SigLIP, DINOv2)의 레이어 계층 구조별 특징을 규명하고, 로봇 제어(Closed-Loop) 성능 극대화를 위해 LoRA로 튜닝해야 하는 최적의 레이어 영역을 정의하고자 함.

---

## Analysis (비전 인코더 레이어 수준 특징 구분)

Vision Transformer (ViT) 계열의 백본은 레이어 깊이에 따라 추출하는 정보의 물리적/의미론적 특성이 명확히 구분됩니다.

```
[카메라 입력]
     │
     ├──> Low-Level (하위 0~8층)   : 엣지, 로컬 텍스처, 색상 (Frozen 권장)
     │
     ├──> Mid-Level (중위 9~20층)  : 객체 파트, 중간 도형, 기하학 레이아웃 (Frozen 권장)
     │
     └──> High-Level (상위 21~26/39층) : 전역 의미론, Spatial 3D 관계, 태스크 정렬 (LoRA 튜닝 최적 영역)
```

---

## Findings

### 1. 인코더 레이어별 역할 구분 및 최적화 영역

| 레이어 수준 | 레이어 범위 (예: 24/27층 기준) | 추출되는 특징 (Features) | 최적화(LoRA) 추천 여부 | 이유 및 메커니즘 |
|:---|:---|:---|:---:|:---|
| **하위 레이어 (Low-Level)** | **Layer 0 ~ 8** | 로컬 엣지, 코너, 텍스처, 기본적인 색상 대비 및 패치 투영 | **Frozen (비추천)** | 일반 대용량 이미지 프리트레인 과정에서 학습된 범용적 시각 프리미티브가 잘 보존되어 있으며, 튜닝 시 OOD 일반화 성능이 붕괴될 위험이 큼. |
| **중위 레이어 (Mid-Level)** | **Layer 9 ~ 20** | 중간 규모의 모양(shapes), 물체 일부(parts), 객체 레이아웃 및 패치 간 기하학적 상관관계 | **Frozen (보통)** | 도메인이 극단적으로 바뀌지 않는 한 기존 피처를 유지하는 것이 파라미터 효율성과 학습 안정성 면에서 우수함. |
| **상위 레이어 (High-Level)** | **Layer 21 ~ 최종** | 전역적 의미 정보(Global Semantics), 물체 간 3차원 공간적 관계(Spatial Relations), 텍스트-비전 정렬 | **Trainable (LoRA 권장) ⭐** | 로봇의 실시간 Closed-Loop 제어 명령 및 물체 그라운딩 목적에 맞추어 태스크 정렬(Task-specific Alignment)이 이루어지는 핵심 영역임. |

---

### 2. SigLIP vs DINOv2 레이어 특성 및 튜닝 전략 비교

| 구분 | **SigLIP (Sigmoid CLIP)** | **DINOv2 (Self-Supervised)** |
|:---|:---|:---|
| **설계 사상** | **Contrastive Text-Image Alignment** | **Self-Supervised Dense Feature Learning** |
| **주요 강점** | 텍스트 프롬프트와 매핑되는 **고수준 의미(Semantic) 이해**, Zero-shot 일반화 | 픽셀 수준의 **기하학적/공간적 이해**, 정확한 깊이(Depth) 및 바운딩 박스(BBox) 검출 |
| **PaliGemma/V5 레이어 수** | So400m 기준: **27개 레이어** (Layer 0 ~ 26) | ViT-L/14 기준: **24개 레이어** (Layer 0 ~ 23) |
| **최적화 추천 레이어 슬라이스** | **Layer 23 ~ 26 (최상위 4개 레이어) 튜닝** | **Layer 20 ~ 23 (최상위 4개 레이어) 튜닝** |
| **LoRA 타깃 적용 효과** | "grey basket" 등 명령어 변경 시, 대상 물체의 **의미론적 그라운딩 영역**을 올바르게 고정함. | 물체와 로봇 그리퍼 간의 **물리적 3D 거리 및 조향 각도 오차**를 미세 제어하는 데 기여함. |

---

## Conclusion & 구체적 구현 액션 플랜

### 1. LoRA 적용을 위한 레이어 선별 필터링 규칙 (RegEx / Full Path)
LM을 완전히 제외하고 SigLIP과 DINOv2의 최상위 4개 레이어에만 LoRA 어댑터를 매핑하기 위해, PEFT의 `target_modules` 혹은 `_resolve_lora_target_modules`에서 다음과 같이 명확히 경로를 조립하여 인가합니다.

* **SigLIP 타깃 모듈:** `vision_tower.vision_model.encoder.layers.2[3-6].self_attn.(q_proj|v_proj)`
* **DINOv2 타깃 모듈 (하이브리드 결합 시):** `dino_tower.dino_model.encoder.layers.2[0-3].self_attn.(q_proj|v_proj)`
* **LM 영역 배제:** 어떠한 경우에도 `language_model` 혹은 `text_model` 키워드가 들어간 레이어는 타깃에서 제외합니다.

이와 같이 층 전체를 하위/중위/상위로 구분하고, 고수준의 의미론적 결합과 공간 인식이 일어나는 **상위 4개 레이어에만 선택적으로 LoRA를 적용**함으로써 최적의 학습 효율과 Closed-Loop 안정성을 얻을 수 있습니다.

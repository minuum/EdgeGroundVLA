# EGVLA2 vs EGVLA3 세대 분리 계획서 — 아키텍처 경계 · 핵심 강점 · 실기/논문 투트랙 전략

작성일: 2026-11-24 | 상태: **구상 완료 · 로드맵 확정안**  
관련 문서: [plan_20261122_vlm_decoder_e2e_and_dualrate.md](file:///home/minum/26CS/MoNaVLA/docs/plans/plan_20261122_vlm_decoder_e2e_and_dualrate.md), [CH83 E2E 분석](file:///home/minum/26CS/MoNaVLA/docs/v5/research_story_egvla2.html#ch83), [Phase 1~4 서빙 타임라인](file:///home/minum/26CS/MoNaVLA/docs/v5/phase14_serving_timeline.html)

---

## 1. Background: 왜 EGVLA2와 EGVLA3를 명확히 분리해야 하는가?

지금까지 한 저장소 안에서 **실기 안정화 실험(`A3`, `a3-s2`, `a3-ag23`, `V4R`)**과 **완전 미분 E2E 전환 실험(`B0/B1`, `Dec1Pass`, `Soft-Loc`)**이 동시에 진행되면서, 파트너(로봇 운용팀)와의 소통이나 논문 스토리라인에서 두 목표가 섞이는 문제가 있었다.

이를 해결하기 위해 **"미분 단절 유무(문자열 파싱 vs 텐서/확률 직결)"**와 **"검증 단계(실기 주행 완료형 vs 차세대 E2E 연구형)"**를 기준으로 **EGVLA2**와 **EGVLA3**의 경계·강점·역할을 명확히 분리한다.

---

## 2. 세대별 정체성 및 아키텍처 경계 (Generation Boundary)

```
[EGVLA1 (1세대, G1)] OWLv2 + Kosmos-2 (이중 백본) ➔ 이산 8방향 MLP (515ep, 실기 92%)
        │
        ▼  단일 경량 VLM(Florence-2 0.23B) 통합 + 연속 Flow Matching(PiZeroFM) 도입
[EGVLA2 (2세대, G2/G4)] 🔵 MOD-* 계열: 명시적 문자열 BBox 생성(generate) + 정규식 파싱 ➔ PiZeroFM
        │               (비미분 2단계 모듈러 · 소규모 300ep 실기 최강 안정성 · 언어 게이트 V4R)
        ▼  문자열 generate() 및 정규식 파싱 완전 제거 + VLM Decoder-Action 미분 가능 직결
[EGVLA3 (3세대, G3)]    🟣 VLM-E2E-* 계열: Prefix-Forced <loc> Softmax 기댓값(Soft-Loc) / 1-Pass Hidden ➔ PiZeroFM
                        (완전 미분 End-to-End VLA · 파싱 0% · Open-Vocab 공간 병목 보존)
```

| 구분 항목 | 🔵 **EGVLA2** (2세대 · `MOD-*` 계열) | 🟣 **EGVLA3** (3세대 · `VLM-E2E-*` 계열) |
| :--- | :--- | :--- |
| **핵심 정의** | **Explicit Modular Grounding + Flow-Matching VLA** | **Differentiable Spatial-Bottleneck End-to-End VLA** |
| **통일 네이밍 접두어** | `MOD-A3`, `MOD-A3-AG`, `MOD-DR-Fast/Safe`, `MOD-V4R` | `VLM-E2E-Hidden` (`EG3-08`), `VLM-E2E-SoftLoc+` (`EG3-09/10`) *(대조군: `ENC-E2E-*`)* |
| **VLM ↔ Action 연결부** | Florence-2 Decoder `generate()` → **문자열 `<loc>` 정규식 파싱** (`[cx, cy, area, has]`) | Florence-2 Decoder **`<loc_0~999>` 로짓 Softmax 기댓값** $\hat{u}=\sum p_i (i/1000)$ 또는 **1-Pass Hidden State** |
| **미분 가능 여부 (E2E)** | ❌ **비미분** (문자열 파싱 경계에서 Gradient 단절) | ✅ **완전 미분 가능** (Action Loss가 Decoder `<loc>` 로짓까지 역전파 가능) |
| **대응 SOTA 패러다임** | **NaVILA** (arXiv:2412.04453), **RT-H** (arXiv:2403.01823) | **$\pi_0$** (arXiv:2410.24164), **ECoT** (arXiv:2407.08693), **SpatialVLA** (arXiv:2501.15830) |
| **서빙 추출기 / 모드** | `FusedF2Extractor` (`beam=3`) / `FastF2Extractor(mode="beam1_fp16")` | `FastF2Extractor(mode="softloc_fp16")` |
| **종단 추론 지연 (GB10)** | `590.7ms` (`beam=3 fp32`) → **`285.3ms`** (`beam=1 fp16`, **2.09x**) | **`~70ms`** (`Hidden`) / **`296.6ms`** (`Soft-Loc fp16`, **2.08x**) |
| **A3 대비 조향 정밀도** | **`0.54°` (좌우 98.7%)** (`MOD-A3-AG`) | **`2.71°` (좌우 82.8%)** (`Soft-Loc + AG23`) / `5.52°` (`EG3-10`) |
| **11/13 다물체 중앙 출발** | **의자 100% · 책상 100%** (`a3-s2`, `a3-ag23`) | **의자 100% · 책상 100% · 화분 99~100%** |
| **11/16 근접 정지 (`≥0.25`)** | **바구니 90% · 의자 93%** (`MOD-A3-AG`) | **바구니 100% · 의자 93~100%** (`VLM-E2E-SoftLoc+`) |

---

## 3. 각각의 독보적 강점 (Core Strengths)

### 🔵 EGVLA2 (`MOD-*`)의 4대 강점 — *"실기 로봇 배포 & 엔지니어링 신뢰성 최강"*
1. **실기 폐루프 검증 완료 (Proven in Real World)**:
   - `A3`(455/455), `11/13` 다물체 65세션, `11/16` `a3-s2` 45/45 등 560회 이상의 실제 로봇 주행 로그로 검증된 아키텍처.
2. **최고 조향 정밀도 (`0.54°`) + `beam=1 fp16` 무손실 2배 가속 (`285.3ms`)**:
   - 이산 토큰 `argmax`/`beam=1`이 연속 확률 꼬리(tail noise)를 완벽히 잘라내므로 `beam=3` 대비 `cx MAE 0.0016` (일치율 `99.7%`)의 초정밀 좌표를 유지.
3. **Dual-Shortcut 해소 (`MOD-A3-AG`) + 언어 게이트 (`MOD-V4R`) 모듈화**:
   - 조향(`cx`)과 정지(`area≥0.23`)를 기하학적으로 동시 정합해 의자 중앙 비출발(`0% → 100%`)과 근접 미정지(`28% → 93%`)를 해결했고, `V4R`로 비목표 지시어 `99.5%` 정지 및 박스 반전 `94.3%` 방어 가능.
4. **이종 로봇(Serbot → YahBoom) 즉시 이식성 (`MOD-DR-Safe`)**:
   - 명시적 `bbox_area`와 `bbox_cx`가 노출되어 있어, الس2 연속 접근 감속(`0.14→0.25`), 스파이크 필터(`2.5x`), YahBoom 1프레임 카메라 면적 보정(`calibrate_near_area`)을 재학습 없이 즉시 적용 가능.

### 🟣 EGVLA3 (`VLM-E2E-*`)의 4대 강점 — *"학술적 독창성 & 진정한 미분 가능 E2E VLA"*
1. **"VLM을 진짜로 활용하는 완전 미분 E2E VLA"라는 학술적 정당성**:
   - 기존 `ENC-E2E`(`B0/B1`)처럼 Decoder를 잘라내 Open-Vocab 지식을 버리거나, EGVLA2처럼 정규식 파싱으로 미분을 끊지 않고 **Florence-2 Encoder + BART Decoder(6L) 전체를 미분 가능하게 관통**함.
2. **소규모(300ep) 데이터에서 $\pi_0$형 순수 Latent E2E의 한계를 규명하고 해결한 스토리**:
   - **결정적 Ablation 발견**: 외부 BBox 없이 Decoder Hidden만 넘긴 `VLM-E2E-Hidden`($\pi_0$형, `EG3-08`)은 학습 물체(바구니) 정지 `100%` 대비 **미학습 물체(의자) 정지 `18.6%`**로 무너짐.
   - 반면 Decoder `<loc_0~999>` 분포에 **Softmax 기댓값(미분가능 공간 병목)**을 씌운 `VLM-E2E-SoftLoc`(`EG3-09/10`)은 문자열 파싱 0%로 **미학습 의자 정지를 `18.6% → 100%` (+81.4%p)**로 끌어올림 → **"저데이터 환경의 E2E VLA에는 미분 가능한 명시적 공간 병목(Differentiable Spatial CoT)이 필수적이다"**라는 강력한 논문 기여 창출.
3. **Prefix-Forced KV-Cache로 문구 재생성 병목 제거 (`296.6ms`, 2.08배 가속)**:
   - `[</s>, <s>] + phrase_tokens`를 디코더 Prefix로 1-Pass 주입해 텍스트 토큰 생성 루프를 통째로 생략하고, 오직 4스텝 `<loc>` 확률 분포만 추출해 속도와 미분 가능성을 동시에 달성.
4. **End-to-End 공동 미세조정(Joint Action-Grounding Fine-Tuning) 확장성**:
   - 좌표 추출이 미분 가능(`probs * bins`)하므로, 주행 행동 오차(`loss_fm`)를 VLM Decoder LoRA까지 직접 역전파시켜 "주행에 유리한 그라운딩 표현"으로 VLM과 Action Expert를 공동 최적화(`EG3-11`)할 수 있음.

---

## 4. 코드 · 서빙 · 실험 관리 분리 원칙 (Implementation Governance)

동환 서버(`soda`)와 연구 서버(`minum`)가 혼선 없이 두 세대를 함께 다룰 수 있도록 아래 3가지 분리 규칙을 적용한다.

### 4.1 네이밍 & 레지스트리 분리 규칙
- **EGVLA2 (`G2`/`G4`)**: 파트너 호환 ID(`a3`, `a3-s2`, `a3-ag23`, `v4`, `v4r`) 유지 + **`MOD-*` 파란색 뱃지** 병기.
- **EGVLA3 (`G3`)**: `EG3-08~11` ID 사용 + **`VLM-E2E-*` 보라색 뱃지** (대조군인 인코더 단독 E2E는 `ENC-E2E-*` 노란색 뱃지) 병기.

### 4.2 서빙 서버 단일 코드 · 환경변수 스위치 분리 (`egvla2_inference_server.py`)
서버 코드를 두 벌로 쪼개지 않고, 이미 구현된 `VLA_EGVLA2_DECODE_MODE` 스위치 하나로 세대를 전환한다:

```bash
# [EGVLA2 표준 실기 모드] MOD-DR-Safe (beam=1 fp16 + A3C_AG23 + S2 감속, 285.3ms)
export VLA_EGVLA2_CKPT="runs/egvla2_fm/a3c/A3C_AG23_s0.pt"
export VLA_EGVLA2_DECODE_MODE="beam1_fp16"
export VLA_EGVLA2_DECEL_START_AREA="0.14"
export VLA_EGVLA2_JUMP_AREA_RATIO="2.5"

# [EGVLA3 미분 E2E 실험 모드] VLM-E2E-SoftLoc (문자열 파싱 0%, 296.6ms)
export VLA_EGVLA2_CKPT="runs/egvla2_fm/a3c/A3C_AG23_s0.pt"   # 또는 EG3_10_SoftLoc_v2.pt
export VLA_EGVLA2_DECODE_MODE="softloc_fp16"
export VLA_EGVLA2_DECEL_START_AREA="0.14"
```

### 4.3 단계별 실행 로드맵 (Two-Track Execution Plan — 확정)

> **세대 마일스톤 확정 (2026-11-24)**:
> - **🔵 EGVLA2 엔딩**: `A3C_AG23_s0` + **`beam=1 fp16` (`MOD-DR-Fast`, `285.3ms`)** + `MOD-DR-Safe`(S2 감속) + `V4/V4R` 적용 및 YahBoom 이식으로 **2세대 모듈러 트랙 완결**.
> - **🟣 EGVLA3 목표**: 컨트롤러 하드코딩(`near_area=0.25`)을 끄고(`VLA_EGVLA2_NEAR_AREA=off`, `STOP_DEBOUNCE=2`), **모델 자체가 멈춤(`[0, 0]`)을 직접 출력하는 완전 미분 E2E VLA (`EG3-11`)**로 실기 진행.

| 트랙 | 세대 | 단기 목표 (즉시 실행) | 중기 목표 (최종 마일스톤) | 담당 / 검증 환경 |
| :--- | :---: | :--- | :--- | :--- |
| **Track 1 (EGVLA2 엔딩)** | **EGVLA2 (`MOD-*`)** | 1) `V4`·`V4R` 언어 게이트 실기 2종 완료<br>2) `A3C_AG23_s0.pt` + **`beam=1 fp16`** + S2 감속 Serbot 실기 정지 확정 | **`yahboom_adapter.py` 기반 YahBoom 카 이식**으로 EGVLA2 실용화 엔딩 완료 | 동환 (`soda` Serbot) + 민우 |
| **Track 2 (EGVLA3 자율정지 E2E)** | **EGVLA3 (`VLM-E2E-*`)** | 1) **`EG3-11` (`VLM-E2E-SoftLoc-Joint`)** 오프라인 5축 검증 완료 (`2.90°`, `near_area` 없이 모델 자체 정지 **바구니 100%·의자 98%**, Open-Vocab 99.8%) | **`VLA_EGVLA2_NEAR_AREA=off` + `STOP_DEBOUNCE=2`** 환경에서 **모델 자체 멈춤 출력 기반 E2E 실기 15세션** 검증 및 논문 완성 | 민우 (`minum` GB10) → 파일럿 실기 |

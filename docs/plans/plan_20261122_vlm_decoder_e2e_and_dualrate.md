# 연구 계획서: VLM Decoder E2E (`VLM-E2E`) vs Modular Dual-Rate (`MOD-DR`)

작성일: 2026-11-22 | 상태: **Phase 1 완료 · Phase 2 실행 중**  
선행 근거: `plan_20261115_a3_improvement.md` (§10~12), `plan_20261117_stop_policy.md` (§10), `scripts/eval_e2e_on_real_a3.py` (기존 인코더 단독 E2E 실기 시뮬레이션 붕괴 확인)

---

## 0. 통일 네이밍 규칙 (파트너 호환 ID 유지 + 구조 축약명 병기)

파트너(`soda` / 로봇 운용팀)가 이미 사용 중인 기존 `A3` / `EG3-*` ID는 그대로 유지하되, **E2E 여부와 VLM Decoder 사용 여부**가 한눈에 구분되도록 아래 3계열 축약명을 병기한다.

| 계열 축약명 | 기존 호환 ID (유지) | E2E 여부 | VLM Decoder 사용 | 한 줄 요약 |
| :--- | :--- | :---: | :---: | :--- |
| **`MOD-A3`** | `A3` (`EG2-11`) | 아니오 (모듈러) | `generate()` (`beam=3 fp32`) | 현행 기준선 (`590.7ms`) |
| **`MOD-A3-AG`** | `a3-ag23` (`EG2-16`) | 아니오 (모듈러) | `generate()` (`beam=3 fp32`) | `A3` + Area-Grounded 정지 개선판 (배포 패키지) |
| **`MOD-DR-Fast`** | *(구 Track B1)* | 아니오 (모듈러) | `generate()` (`beam=1 fp16`) | **Phase 1 완료**: `283.0ms` (2.09배 가속, `cx MAE 0.0016`) |
| **`MOD-DR-Safe`** | *(구 Track B2)* | 아니오 (모듈러) | `beam=1 fp16` 비동기 Dual-Rate | `MOD-DR-Fast` + 접근 감속(S2) + 점프 필터 (실기/YahBoom용) |
| **`ENC-E2E-0`** | `EG3-03` (`B0_frozen_davit`) | 예 (인코더만) | **제거** (DaViT 동결) | 11/13·16 실기 시뮬 붕괴 (`26.8°`) |
| **`ENC-E2E-1`** | `EG3-04` (`B1_lora_last_stage`) | 예 (인코더만) | **제거** (DaViT LoRA) | 11/13·16 실기 시뮬 붕괴 (`25.1°`) |
| **`ENC-E2E-AG`** | `EG3-06` (`E2E-LoRA-AG`) | 예 (인코더만) | **제거** (DaViT LoRA + AG) | 시뮬 회복 (`8.05°`, 근접정지 `90%`) |
| **`ENC-E2E-Aux`** | `EG3-07` (`E2E-LoRA-SpatialAux`) | 예 (인코더만) | **제거** (+ 인코더 BBox 헤드) | 시뮬 `7.78°`, 근접정지 `90%` (VLM 디코더 미사용) |
| **`VLM-E2E-Hidden`** | *(구 Track A1 / `EG3-A1`)* | **예 (완전 E2E)** | **1-Pass Hidden State** (`π0`형) | Decoder 은닉 상태를 Action Expert가 직접 수신 (Phase 2) |
| **`VLM-E2E-SoftLoc`** | *(구 Track A2 / `EG3-A2`)* | **예 (완전 E2E)** | **Prefix-Forced Soft-Loc** (`fp16`) | Decoder `<loc>` 로짓 Softmax 기댓값 직접 연결 (`283.8ms`, Phase 2) |
| **`VLM-E2E-Distill`** | *(구 Track A3 / `EG3-A3`)* | **예 (완전 E2E)** | Decoder LoRA + `beam=3` 증류 | 소규모 데이터에서 다물체 Open-Vocab 보존 E2E (Phase 3) |

---

## 1. Background: 왜 `VLM-E2E`와 `MOD-DR` 두 트랙이 학술적으로 필요한가

1. **기존 순수 비전 E2E(`EG3-01`, `EG3-03 B0`, `EG3-04 B1`)의 한계 입증**:
   - 11/13(870장)·11/16(588장) 실기 프레임 시뮬레이션 결과, 기존 E2E 모델들은 `A3` 대비 방향이 `25.1°~39.9°` 어긋나고 좌우 일치율이 `36~51%`(반사실 민감도 `S = -3.77° ~ +0.01°`)로 붕괴함.
2. **Encoder-Only BBox Head의 학술적 한계**:
   - Florence-2의 Open-Vocabulary Grounding 지식(25억 개 레이블 사전학습)은 **BART Decoder(6 layers)**에 저장되어 있음. Decoder를 제거하고 Encoder 위에 새 BBox 헤드를 300ep(바구니 1종)로만 학습하면 Zero-Shot 다물체 능력이 소실됨.
3. **현행 서빙(`A3` / `V4`)의 속도 병목**:
   - 현재 지연(`~870ms`)의 94%는 Florence-2 Decoder에서 `num_beams=3` (`fp32`)으로 텍스트 좌표 토큰을 자기회귀 생성(`generate()`)하고 문자열로 파싱하는 구간에서 발생.

> **핵심 연구 질문 (Research Question)**:  
> *"소규모(300ep) 단일 물체 주행 데이터 환경에서, VLM Decoder의 사전학습 Open-Vocabulary 지식을 손실 없이 유지하면서도 지연(`~870ms`)과 비미분 병목을 동시에 극복하려면 **1-Pass Decoder Hidden State E2E(Track A)**와 **Beam=1 Dual-Rate(Track B)** 중 어느 구조가 정확도·일반화·실시간성의 최적 파레토 프론티어를 달성하는가?"*

---

## 2. Architecture: 두 트랙 상세 설계

### Track A — π0형 VLM Decoder 1-Pass E2E-VLA (`EG3-Dec1Pass`)
문자열 `generate()` 루프와 정규식 파싱을 제거하되, **Florence-2 Encoder + Decoder(6 layers)를 모두 통과**시켜 VLM을 100% 활용하는 완전 미분 가능 E2E 구조.

| 변형 | 메커니즘 | Open-Vocab 보존 장치 | 예상 지연 |
| :--- | :--- | :--- | :---: |
| **A1. Frozen Decoder 1-Pass Query + Action Expert** | Encoder 출력(`[Img 577 | Prompt]`)에 대해 Decoder에 `<bos>` + 학습 가능 Query $K=4$개를 **1-Pass 병렬 통과** → Decoder Last Hidden State $H_{\text{dec}} \in \mathbb{R}^{B \times 4 \times 768}$를 `PiZeroFM` Cross-Attn Prefix로 직접 주입 | Florence-2 Encoder+Decoder **100% 동결** (사전학습 표상 파괴 원천 차단) | **~60~90ms** |
| **A2. Differentiable Soft-Loc Bottleneck** | Decoder의 `<loc_0>~<loc_999>` 출력 로짓에 Softmax Expectation $\hat{u} = \sum_{i=0}^{999} p_i \cdot (i/1000)$을 적용해 연속 좌표 $(\hat{x}_1, \hat{y}_1, \hat{x}_2, \hat{y}_2)$를 **파싱 없이 미분 가능하게 추출** → `Area-Grounded PiZeroFM`으로 전달 | Florence-2 사전학습 `lm_head`의 `<loc>` 분포를 그대로 사용하므로 **의자·책상·화분 Zero-Shot 100% 유지** | **~90~130ms** |
| **A3. Low-LR Decoder LoRA + Open-Vocab Distillation** | A1/A2에 Decoder LoRA(`r=4`, `lr=5e-5`)를 추가하되, 300ep 바구니 + 11/13·11/16 다물체 `beam=3` 출력에 대한 **KL/SmoothL1 Distillation Loss**를 결합 | 행동 손실과 동시에 기존 `beam=3` 다물체 좌표를 증류해 Catastrophic Forgetting 방지 | **~60~90ms** |

### Track B — VLM Decoder (`beam=1, fp16`) Dual-Rate Asynchronous Architecture
기존 실기 검증된 `A3`/`A3C_AG23_s0`의 높은 다물체 일반화를 100% 유지하면서, 디코딩 경량화와 비동기 제어로 실시간성과 안전 정지를 확보하는 실전형·대조군 구조.

- **Slow Loop (VLM Grounder Thread, `num_beams=1`, `fp16`, ~4~5Hz)**:
  - `egvla2_fm_policy.py`의 `num_beams=3, fp32`를 `num_beams=1(greedy), fp16`으로 전환해 그라운딩 지연 3~4배 단축.
  - 단발성 BBox 점프(`cx`/`area` 스파이크) 제거용 **2프레임 평활화 + 아웃라이어 필터** 적용.
- **Fast Loop (Action Expert + Deceleration Controller, 15~20Hz)**:
  - 최신 유효 BBox 창 + **`A3C_AG23_s0`** (조향 `cx` + 정지 `area≥0.23` 기하 정합) + **연속 접근 감속(`plan_20261117_stop_policy.md` S2)** 구동.
  - Serbot → **YahBoom 카 이식 시** 구동계 어댑터(`[lx, ly] → [v, ω]`)와 카메라 1프레임 기준 면적 보정만 교체.

---

## 3. 공통 평가 기준 (3축 + 속도 + Open-Vocab)

오프라인 각도 오차(`dir_err`) 단일 지표의 함정(실기 역전)을 피하기 위해, 모든 실험은 아래 **5개 정량 지표**로 동일 평가한다:

| 평가 축 | 측정 데이터셋 | 합격 기준 (Go/No-Go Gate) |
| :--- | :--- | :--- |
| **① 조향 정렬** | 11/13 + 11/16 이동 프레임 (`A3` 기준) | 각도차 중앙값 `≤ 3.0°`, 좌우 부호 일치 `≥ 95%`, 반사실 `S_cx ≥ +15°` |
| **② 낯선 물체 중앙 출발** | 11/13 의자·책상·화분 가운데 세션 | 중앙 이동률 `≥ 95%`, 11/16 원거리(`<0.15`) 오정지 `≤ 5%` |
| **③ 근접 안전 정지** | 11/16 바구니·의자 `area ≥ 0.25` 프레임 | 정지율 `≥ 85%`, 반사실 면적 민감도 `S_area ≥ +60%p` |
| **④ Open-Vocab 유지** | 11/13 의자·책상·화분 65세션 (미학습 물체) | `beam=3` 기준 BBox 중심 오차 `≤ 0.03`, 검출률 `≥ 95%` |
| **⑤ 추론 지연 (Latency)** | GB10 / Jetson 실측 (ms/frame) | 현행 `beam=3 fp32` 대비 **Track B ≥ 3배**, **Track A ≥ 8배** 단축 |

---

## 4. 단계별 실행 계획 (Milestones)

| 단계 | 작업 내용 | 예상 시간 | 산출물 |
| :---: | :--- | :---: | :--- |
| **Phase 1**<br/>*(즉시 가능, 오프라인)* | **Track B1 벤치마크**: `beam=3 fp32` vs `beam=1 fp32` vs `beam=1 fp16`의 지연(ms), 11/13·11/16 박스 오차 및 `A3C_AG23_s0` 행동 일치율 전수 측정 | ~10분 | `scripts/bench_decoder_beams.py`, `docs/v5/egvla2_fm/beam_bench.json` |
| **Phase 2**<br/>*(오프라인 핵심)* | **Track A1/A2 프로토타입**: Florence-2 Frozen Decoder **1-Pass Hidden State (`A1`)** 및 **Soft-Loc Bottleneck (`A2`)** 추출 → `Area-Grounded PiZeroFM` 학습 및 11/13·11/16 5축 평가 | ~1~2시간 | `scripts/train_e2e_dec1pass.py`, `runs/egvla2_e2e_lora/EG3_Dec1Pass_*.pt` |
| **Phase 3**<br/>*(고도화)* | **Track A3 (Decoder LoRA + Open-Vocab Distillation)** 학습 및 **Track A vs Track B 파레토 프론티어 비교표·시각화(CH83)** 발행 | ~2~3시간 | `scripts/gen_ch83.py` → `research_story_egvla2.html#ch83` |
| **Phase 4**<br/>*(실기 · 이식)* | **Track B2 (Dual-Rate + 접근 감속 S2 + 아웃라이어 필터)** `soda` 패치 및 Serbot 실기 검증 → 정지 해결 확인 후 **YahBoom 카 이식** | 실기 일정 | `egvla2_only_logic.py` S2 패치, YahBoom 어댑터 |

---

## 5. 논문(EdgeGround-VLA)에서의 예상 기여 및 시사점

1. **왜 기존 E2E LoRA(`EG3-01~05`)는 300ep에서 실패했는가**: 비전 백본 과적합(`S_cx < 0`) + 디코더 제거 시 Open-Vocab 상실 메커니즘 규명.
2. **Decoder 1-Pass / Soft-Loc E2E(Track A) vs Beam=1 Dual-Rate(Track B) 정면 비교**: 동일 백본(Florence-2)과 동일 데이터(300ep) 위에서 **완전 미분 E2E와 경량 모듈러 Dual-Rate의 정확도–지연–다물체 일반화 파레토 트레이드오프**를 정량 제시.
3. **Dual-Shortcut 해소(`cx` + `area` 기하 정합) + 접근 감속 + 이종 로봇(Serbot→YahBoom) 실증**: 실제 엣지 로봇 환경에서의 재현 가능한 설계 원칙 확립.

---

## 6. 승인 요청 (다음 진행 선택)
1. **Phase 1 (`beam=3` vs `beam=1 fp16` 속도·정확도 벤치마크, ~10분)**부터 바로 실행할까요?
2. **Phase 2 (`Track A1/A2`: Florence-2 Decoder 1-Pass Hidden State / Soft-Loc E2E 프로토타입)**를 이어서 바로 구현·실험할까요?

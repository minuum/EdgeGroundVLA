# Grounding Center Bias 분석 — 2026-07-02

**질문:** 같은 모델과 같은 환경을 사용하는데, 왜 PG448 그라운딩이 실제 바스켓 위치 대신 이미지 중앙을 찍는가?

**분석 기반:** 실로봇 추론 세션 8개 (ep 37~47, 20260701~20260702, 64프레임) + V5 학습 데이터 비교

---

## 1. 핵심 결론

| 순위 | 원인 | 소스 파일 | 핵심 |
|------|------|----------|------|
| ★★★ | **PG448 주변부 탐지 실패 → 중앙 오탐** | PG2 모델 + 학습 데이터 편향 | 바스켓 cx>0.7 케이스 학습 데이터의 7.8% |
| ★★ | **Post-action 프레임 cx=0.5 기본값 주입** | `gradio_inference_dashboard.py:1577` | bbox 미전달로 직렬화 시 cx=0.5 초기화 |
| ★ | **skip_n=3 bbox 시간성 불일치** | `VLA_GROUNDING_SKIP_N=3` | 학습: W=6 전부 distinct / 추론: [a,a,a,d,d,d] |

**학습·추론 구조 대조 (train_exp71 ↔ stage2_v2_inference_server.py):**

| 항목 | 학습 | 추론 | 일치 |
|------|------|------|------|
| Padding 공식 | `max(0, t-(W-1-k))` (line 115) | `max(0, len(hist)-1-(W-1-k))` (line 700) | ✅ |
| Padding 방식 | 에피소드 첫 프레임 복제 | history 첫 프레임 복제 | ✅ |
| Feature 순서 | `[bbox 4d, vis 256d]` | `[bbox 4d, vis 256d]` | ✅ |
| vis_feat | 프레임마다 fresh | 프레임마다 fresh | ✅ |
| BBox 출처 | PG448 어노테이션 | PG448 실시간 grounding | ✅ |
| **BBox 시간성** | **W=6 전부 distinct** | **skip_n=3 → [a,a,a,d,d,d]** | **❌** |

**불일치는 bbox 시간성 하나.** 나머지 padding·feature 순서·vis_feat는 완전 동일.

**skip_n=3 완화 요인 (왜 실전에서 버텨지나):**
- vis_feat 6칸 모두 fresh → temporal 신호 주축은 온전, bbox 채널만 계단식
- CH49 실험: skip_n=3 SR/FPE 변화 없음 확정 (`server.py:519` 주석)
- feature ablation (CH49 기준): image_only 75.6% vs bbox+image 76.7% → bbox 기여 1.1%p

*이하 두 가지 주요 원인을 근거 중심으로 분석.*

---

## 2. 원인 1: PG448 주변부 탐지 실패

### 2.1 ep 1~36 vs 37~47 환경 차이

ep 1~36 (right_left, 대부분 성공):
- 바스켓이 학습 시작 시점부터 시야 정면에 위치
- 단순한 흰 벽 배경

ep 37~47 (obj_left/obj_center, 다수 실패):
- 바스켓을 찾기 위해 회전 → 바스켓이 프레임 가장자리로 이동
- 가구, 의자, 캐비닛이 있는 배경 (100143 t=7에서 확인)

### 2.2 PG448 학습 데이터의 cx 분포 편향

V5 학습 어노테이션 (2626프레임, has_bbox 2567개):

```
cx 구간 분포:
[0.00~0.20]    16개  (0.6%)
[0.20~0.35]   223개  (8.7%)
[0.35~0.45]   516개 (20.1%)
[0.45~0.55]  1138개 (44.3%)  ← 절반 가량
[0.55~0.65]   474개 (18.5%)
[0.65~0.80]   191개  (7.4%)
[0.80~1.00]     9개  (0.4%)

cx mean=0.495  std=0.110
cy mean=0.644  std=0.049  ← y 좌표는 균일 (카메라 높이 고정)
```

**cx > 0.7 구간: 200/2567 = 7.8%**

right_left 경로에서 실로봇 실측 cx: mean=0.551, std=0.073 → 학습 분포 정중앙.
바스켓이 cx=0.84로 이동하는 obj_left/obj_center 시나리오는 학습 데이터에서 매우 희귀(0.4%).

### 2.3 시각적 증거: session_20260702_100143 t=7

탐지값: cx=0.516, cy=0.682, area=0.1139 (has_bbox=True)

→ **실제 바스켓: 이미지 우측 cx≈0.84** (이미지에서 선명하게 보임)  
→ **PG2 탐지 위치: 이미지 중앙 빈 바닥 / 벽-바닥 경계** (bbox가 아무것도 없는 곳에 그려짐)

PG448이 `has_bbox=False`를 반환하지 않고 **False Positive를 중앙에 생성**했다는 점이 핵심.

### 2.4 PG448이 ep 1~36에서는 왜 잘 됐는가

| 조건 | ep 1~36 (성공) | ep 37~47 (실패) |
|------|----------------|----------------|
| 바스켓 cx 범위 | 0.33~0.68 (학습 핵심 구간) | 이탈 시 >0.75 |
| 배경 | 단순 흰 벽 | 가구/의자 존재 |
| 바스켓 시야 이탈 여부 | 이탈 없음 (정면 유지) | 회전 후 이탈 발생 |
| 오탐 위험 물체 | 없음 | 의자·캐비닛 등 유사색 객체 |

학습 데이터의 cx 분포와 배경 단순성 덕분에 PG448이 right_left에서 잘 동작했던 것이다.
obj_* 시나리오에서 바스켓이 시야 외각으로 이동하자, 학습 분포 외의 케이스에서 PG448이 중앙 False Positive를 생성.

### 2.5 서버 fallback 메커니즘의 맹점

서버 코드 (`stage2_v2_inference_server.py:466,485`):
```python
# PG2 parse 실패 시 fallback
_fallback = {"cx": 0.5, "cy": 0.6, "area": 0.06, "has_bbox": False, ...}
```

PG2가 **완전히 파싱 실패**하면 `has_bbox=False`가 반환된다.
그러나 PG2가 **틀린 위치를 탐지**하면 `has_bbox=True` + 잘못된 cx가 반환된다.

현재 시스템은 탐지 성공 여부만 체크하고 **탐지 신뢰도·위치 타당성은 검증하지 않는다**.

---

## 3. 원인 2: Post-action 프레임 cx=0.5 주입 (구조적 버그)

### 3.1 버그 소스 코드

`gradio_inference_dashboard.py:1577-1583` (최신 post-action 수집 로직):
```python
# 액션 완료 직후 캡처한 프레임 즉시 기록
_post = state.get("stable_frame")
if _post is not None:
    logger_instance.log_step(
        f"{current_step}p",
        result["action"],   # 이전 액션 그대로 복사
        0,
        image=_post,        # bbox, grounding_cached 미전달 ← BUG
    )
```

`bbox`와 `grounding_cached`가 `**extra` 인자로 전달되지 않음.

### 3.2 직렬화 체인에서 cx=0.5가 생성되는 과정

```
log_step() 호출
    │
    ├─ step_data["bbox"] = 없음 (skip — v is None 조건으로 누락)
    └─ step_data["grounding_cached"] = 없음

H5 저장 (end_session):
    │
    ├─ b = h.get("bbox") or {}  →  {}
    │   → cx=0.5, cy=0.6, area=0.0, has_bbox=False  (기본값)
    │
    └─ gc = h.get("grounding_cached")  →  None
        → float(gc) if gc is not None else -1.0  →  -1.0
```

결과: post-action 프레임 = `cached=-1, cx=0.5, cy=0.6, area=0.0`

### 3.3 정량 증거: session_20260702_100143

총 19프레임 구조:
```
t=0   NONE  cx=0.500  ← 초기 프레임 (step 1 이전)
t=1   LIVE  cx=0.485  ← 추론 step 1 (PG2 live)
t=2   NONE  cx=0.500  ← post-action step 1p (BUG)
t=3  CACHE  cx=0.485  ← 추론 step 2 (bbox 캐시 재사용)
t=4   NONE  cx=0.500  ← post-action step 2p (BUG)
t=5  CACHE  cx=0.485  ← 추론 step 3 (bbox 캐시)
t=6   NONE  cx=0.500  ← post-action step 3p (BUG)
t=7   LIVE  cx=0.516  ← 추론 step 4 (PG2 live — 오탐 포함)
...
총 10개 NONE (짝수 프레임 전부) = 전체의 52.6%
```

세션별 NONE 비율:
- 212540 (7.1%), 220400 (14.3%): 최신 post-action 로직 **이전** 세션
- 084101 (40%), 084555 (33.3%), 100143 (52.6%): 최신 로직 **이후** 세션

### 3.4 Transformer W=6 윈도우에 미치는 영향

NONE 비율 52.6%에서 WINDOW=6 안의 기대 구성:
- 추론 스텝 3개 (LIVE 또는 CACHE) + post-action 스텝 3개 (NONE, cx=0.5)
- 바스켓이 실제로 cx=0.84에 있어도, 추론 스텝의 PG2 오탐(cx=0.51) + NONE(cx=0.5)가 섞여
  모델 입력 window의 cx 평균: (0.51×3 + 0.5×3)/6 = **0.505** (중앙)
- 모델: "바스켓이 정면에 있다" → FORWARD 출력

---

## 4. 정량 요약

| 지표 | 학습 데이터 (V5 PG448) | 실로봇 live 탐지 | 실로봇 NONE 프레임 |
|------|----------------------|-----------------|-------------------|
| cx mean | 0.495 | 0.478~0.534 | 0.500 (고정) |
| cx std | 0.110 | 0.016 | 0.000 |
| cy mean | 0.644 | 0.470~0.682 | 0.600 (고정) |
| area mean | 0.117 | 0.069~0.114 | 0.000 (고정) |
| has_bbox% | 97.8% | 47~86% | 0% (고정) |

live 탐지 cx std=0.016 (학습 std=0.110의 **15% 수준**)  
→ PG448이 실제로 cx를 올바르게 변화시키지 못하고 center 근방에 고착되어 있음을 정량 확인.

---

## 5. 권장 조치

### P1 (즉시): Post-action 프레임 bbox 상속 버그 수정

`gradio_inference_dashboard.py:1577-1583` 수정:
```python
_post = state.get("stable_frame")
if _post is not None:
    logger_instance.log_step(
        f"{current_step}p",
        result["action"],
        0,
        image=_post,
        bbox=result.get("bbox"),                    # 추가
        grounding_cached=result.get("grounding_cached"),  # 추가
        grounding_latency_ms=0.0,                   # 추가 (캐시 재사용이므로 0)
    )
```
효과: NONE 프레임이 직전 valid bbox를 상속 → cx=0.5 오염 차단.

### P2 (단기): PG448 탐지 신뢰도 필터 추가

서버에 탐지 결과 후처리 추가:
```python
# 이전 탐지 대비 cx 급변 감지 → reject
if prev_bbox and abs(new_bbox["cx"] - prev_bbox["cx"]) > 0.30:
    return prev_bbox  # 직전 캐시 유지, has_bbox 그대로
```
효과: 바스켓이 갑자기 cx=0.5로 "점프"하는 False Positive 필터링.

### P3 (중기): 주변부 학습 데이터 보강

cx 0.70~0.85 구간 탐지 데이터 추가 수집 및 PG448 어노테이션:
- 현재 학습 비율 7.8% → 목표 15%+
- 바스켓이 우측/좌측 극단에 있는 에피소드 의도적 수집

### P4 (중기): PG2 프롬프트 구체화

`GROUNDING_PROMPT`: "gray laundry basket" 유지하되 fine-tuning 데이터에 배경 다양화 추가  
(현재 모든 학습 이미지가 단순 흰 벽 배경)

---

## 6. 기각된 가설

| 가설 | 기각 근거 |
|------|----------|
| 레이턴시 부족 | 484ms~5361ms 모두 동일 문제. skip_n=3 캐시로 레이턴시 영향 없음 |
| exp66→exp71 모델 교체 | ep 37~42 (2026-06-30)는 교체 이전부터 동일 현상 |
| skip_n=3 캐시 자체의 추론 오류 | 액션 모델은 매 스텝 새 추론. 캐시는 bbox 재사용만. bbox 영향은 학습 분포 이탈 문제와 별개 |

---

*분석: Claude Sonnet 4.6 / 2026-07-02*  
*데이터: `/home/minum/MoNaVLA/inference_sessions_recv/20260702/` (8세션 64프레임)*  
*시각화: `docs/v5/grounding_analysis/*.jpg`*

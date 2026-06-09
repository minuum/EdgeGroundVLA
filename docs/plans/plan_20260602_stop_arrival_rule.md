# Plan — 도착 STOP 알고리즘 (마지막 레이어 규칙 기반 정지)

> 작성: 2026-06-02 · 갱신: 2026-06-02 (243ep/PG2 현실 반영) · 상태: **구현 중 (승인됨)**
> 동기: STOP(class 0)이 학습에 **희소**(21/243 ep)해 MLP가 under-trigger → 도착 시 로봇이 잘 못 멈춤.
> 아이디어(사용자): "마지막 프레임의 평균을 확인해서" 마지막 레이어에 STOP 규칙을 붙인다.

> ⚠️ **데이터 갱신**: 현재 best 파이프라인은 `bbox_dataset_pg2_cx.json` (243 ep, PG2 grounding).
> 이전 분석(150ep HSV)과 달리 이 데이터엔 STOP(0)이 **21 ep, 105 frame** 존재하고
> 필드명은 `area_det`/`cx_det`(PG2), `area_det_hsv`(HSV). 아래 §2는 243ep/PG2 기준으로 재작성됨.

---

## 1. 문제

V5 8-class에서 **STOP(0)은 합성 클래스** — 데이터셋 gt_class 분포에 0이 단 하나도 없다:

```
gt_class 분포: {1:1955, 2:60, 3:46, 4:255, 5:270, 6:20, 7:20}   ← STOP(0) 없음
마지막 프레임 gt_class: {1: 150}   ← 150 ep 전부 도착 프레임이 FORWARD로 라벨됨
```

→ MLP는 STOP을 **구조적으로 학습 불가**. closed-loop·실로봇에서 로봇이 basket에 도착해도 **멈추지 않고 계속 전진**(overshoot)한다.

## 2. 리서치 — 도착 프레임은 어떻게 다른가 (243ep / PG2 area_det 실측)

`docs/v5/bbox_frame_level/bbox_dataset_pg2_cx.json` (243 ep) 분석:

- STOP(0) 라벨: **21 ep, 105 frame** (전 ep의 8.6%만) → MLP가 학습해도 **과소 대표**
- STOP 프레임 위치: phase **0.73~1.00** (mean 0.94) — 궤적 최후반
- STOP `area_det`: median **0.890** vs non-STOP median **0.050** → **극명한 분리**
- STOP `cx_det`: mean **0.468**, std 0.097 (중앙 정렬)

**area_det-phase 곡선 (PG2 grounding):**

```
phase 0.0: 0.045  (>0.5: 0%)   ← 출발 스파이크 없음! (HSV 아티팩트였음)
phase 0.5: 0.076               phase 0.8: 0.396  (>0.5: 43%)
phase 0.6: 0.102               phase 0.9: 0.544  (>0.5: 56%)
phase 0.7: 0.167  (>0.5: 12%)  phase 1.0: 0.626  (>0.5: 59%)
```

**threshold 분리도 (STOP recall vs non-STOP FP):**

| area_det > θ | STOP recall | non-STOP FP |
|---|---|---|
| 0.4 | 81% | 15% |
| **0.5** | **71%** | **15%** |
| 0.6 | 67% | 14% |
| 0.7 | 57% | 14% |

**핵심 통찰 (150ep HSV 대비 변경점):**
- **출발 스파이크 사라짐** — PG2 area_det는 phase 0.0에서 0.045. `rising`·`MIN_STEPS` 가드의 필요성 ↓ (안전장치로 유지는 가능)
- non-STOP "FP" 15%는 대부분 **21ep 외에서 미라벨된 진짜 도착 근처(phase 0.8~0.9) 프레임** → 규칙이 거기서 STOP 트리거하는 건 사실상 옳은 행동
- 도착 신호가 **area_det 단조 증가**로 매우 깨끗 → 단순 `area_det_avg > θ + centered` 만으로도 강건

## 3. 제안 알고리즘 — 마지막 레이어 STOP override

MLP 출력(8-class logit/argmax) **뒤에** 후처리 규칙을 둔다. MLP는 건드리지 않음.

```python
# 상태: 최근 W프레임 area 버퍼, stopped 래치
def stop_override(pred_class, area, cx, step, area_hist, stopped):
    if stopped:                                   # 래치: 한 번 멈추면 유지
        return 0
    area_hist.append(area); area_hist = area_hist[-W:]
    area_avg = mean(area_hist)                    # "마지막 프레임의 평균" — 사용자 아이디어
    rising   = area_hist[-1] >= area_hist[0]      # 후반 상승 추세 (출발 스파이크 배제)
    centered = abs(cx - 0.5) < TH_CX
    arrived  = (step >= MIN_STEPS               # 출발 직후 금지
                and area_avg > TH_AREA
                and rising and centered)
    if arrived:
        stopped = True
        return 0                                  # STOP
    return pred_class
```

**캘리브레이션 (243ep/PG2 `area_det` 기반 초기값, S1 sweep으로 확정):**
- `TH_AREA ≈ 0.50` (STOP recall 71% / non-STOP FP 15%; STOP median 0.89 vs non-STOP 0.05)
- `TH_CX  ≈ 0.20` (STOP cx std 0.097)
- `W = 3` (최근 3프레임 `area_det` 평균 — 지터 억제)
- `MIN_STEPS`: PG2엔 출발 스파이크 없어 완화 가능. 안전상 소수(예: 3) 유지
- `rising` 가드: 출발 스파이크 없으니 **선택**. 오발이 보이면 활성화
- 입력 필드: `area_det`/`cx_det` (PG2). HSV 비교용은 `area_det_hsv`

## 4. 평가 설계 — STOP이 의미 있으려면 expert도 멈춰야

현재 closed-loop는 expert(GT)가 도착 후에도 FORWARD라 끝까지 전진 → STOP을 넣어도 metric상 이득이 모호.
**expert에도 동일 규칙으로 STOP 합성**하여 "도착=정지"를 기준선으로 만든 뒤 비교:

```
[A] expert: GT 그대로(FORWARD 종료)        — 현재 baseline
[B] expert: 도착 프레임부터 STOP 합성        — 도착-정지 기준선
    pred:   ① STOP override 없음  vs  ② 있음
→ FPE/SR 비교. [B]②가 [B]①보다 overshoot 줄어 FPE↓ 기대
```

또한 **오발률(false STOP)** 측정: 앞부분 프레임에서 규칙이 잘못 STOP을 트리거하는 비율 (출발 스파이크 대응 검증).

## 5. 구현 단계

```
[x] S1. 캘리브레이션 sweep → docs/v5/stop_rule_calibration.json (추천 th_area=0.5,W=5)
[x] S2. STOP 합성 expert (eval 스크립트 내 helper, rollout_core 미수정)
[x] S3. STOP override (eval_stop_closedloop.py)
[x] S4. 4-cell 비교 CL → synth/on 31%→69% 회복 (FPE<0.15 기준)
[x] S5. inference_server.py GoalNav predict()에 _arrival_stop() 래치 + env 토글 (배포만 남음)
[x] S6. 결과 문서화 → docs/v5/exp_stop_report.md
```

> 결과 요약은 [exp_stop_report.md](../v5/exp_stop_report.md) 참조.

## 6. 수정/생성 파일

| 파일 | 변경 |
|---|---|
| `scripts/analyze_stop_rule.py` | **신규** — θ sweep, recall/FP 곡선 |
| `scripts/sim/rollout_core.py` | STOP 합성 expert 유틸 (옵션) |
| `scripts/eval_exp59_closedloop.py` | `--stop-rule`, `--synth-stop` 플래그 |
| `robovlm_nav/serve/inference_server.py` | GoalNav 추론에 stop_override 포팅 (실로봇) |
| `docs/v5/stop_rule_calibration.json`, `docs/v5/exp_stop_report.md` | **신규** |

## 7. 트레이드오프 / 리스크

- **출발 스파이크 오발**: phase 0.0의 큰 area → `MIN_STEPS` + `rising` 가드로 차단. sweep으로 검증 필수.
- **조기 정지 vs 과주행**: TH_AREA 낮으면 일찍 멈춰 미달(FPE↑), 높으면 못 멈춰 overshoot(FPE↑). 최적점 존재.
- **VLM area 노이즈**: 추론 시 area는 PaliGemma2 출력(area_ratio std 큼, Exp60). W프레임 평균으로 완화하되, 필요 시 EMA 병용.
- **규칙 vs 학습**: 근본적으로는 STOP 라벨을 합성해 MLP를 재학습하는 방법도 있음(별도). 본 플랜은 **무학습 규칙**으로 빠르게 실효성 확인이 목적.
- Exp60(aug2.0)과 **독립·상보적** — STOP은 도착 처리, aug는 주행 중 방향 강건성.

## 8. 교수님 질문과의 연결

- "도착을 인식하는가 / 멈출 줄 아는가" = goal-conditioned navigation의 종결 조건.
- 데이터에 없는 STOP을 **bbox area(=목표 근접 proxy)** 로 복원 → "목표를 보고(area↑) 도착을 판단" 이라는 인식→행동 논리 강화.

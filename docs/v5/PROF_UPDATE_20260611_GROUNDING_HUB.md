# 교수님 업데이트 (2026-06-11) — Grounding 전수 검증 & "객체 인식 실패"의 원인 규명

> 6/4 미팅 피드백("실주행 시 객체 인식 실패·궤적 붕괴") 후속.
> 결론 먼저: **오예측의 원인은 grounding 모델 exp59의 간헐적 붕괴였고, base PaliGemma2로 교체하면 해소된다. grounding은 데이터 부족 문제가 아니다.**

---

## 1. 핵심 발견 — 교수님이 보신 "객체 인식 못함"의 정체

실주행에 쓰인 grounding 모델 **exp59**(PaliGemma2 + LM LoRA, hard-negative)는
표준 프레임에선 멀쩡(full-frame 6%)해 보이지만, **실환경 변동에서 간헐적으로 박스가 화면 전체로 붕괴**(full-frame collapse)한다.

| 조건 | exp59 full-frame율 | base PG2 |
|------|:---:|:---:|
| 로봇 원/근거리 | **22%** | 0% |
| 저조도 | **17%** | 0% |
| 대각선 진입 | **14%** | 2% |
| 회전 ±12° | **8~16%** | 0% |
| 부분 가림 | **8%** | 0% |

- full-frame 붕괴 = "바구니가 화면 전체에 있다"는 박스 → action head가 위치 신호를 잃고 조향 붕괴 → **오예측**.
- 표준 정면 프레임만 보면 안 보이던 실패가, 거리·조명·각도가 바뀌는 **실주행 조건**에서 터진 것.
- → **이것이 교수님이 관찰하신 "객체 인식 실패"의 실제 메커니즘.**

**해결:** grounding을 LoRA(exp59) 대신 **base PaliGemma2(zero-shot)**로 고정. 같은 조건에서 full-frame 0~2%.

---

## 2. Grounding 전수 비교 (7모델 × 공통 세트)

표준 49프레임 + 의자 11장(OOD) 기준:

| 모델 | 백본/LoRA | hit | cx_MAE | full-frame | OOD 오탐 | 판정 |
|------|-----------|:---:|:---:|:---:|:---:|------|
| **base PG2** | PaliGemma2 / 없음 | 98% | 0.126 | **0%** | 9% | ✅ **채택 (최균형)** |
| pure Kosmos-2 | Kosmos2 / 없음 | 100% | 0.123 | 0% | **100%** | 바구니↑·OOD 구분 불가 |
| exp57 | PaliGemma1 / LM | 84% | 0.115 | 0% | 9% | hit 낮음 |
| exp58 | PaliGemma2 / LM | 100% | 0.191 | 57% | 27% | 부분 붕괴 |
| **exp59** | PaliGemma2 / LM(hard-neg) | 98% | 0.129 | 6%→실환경 8~22% | 0% | ⚠️ **실환경 붕괴(상기)** |
| exp64 | PaliGemma2 / **vision** | 94% | 0.150 | **92%** | 9% | ❌ full-frame 전면붕괴 |
| exp56 | Kosmos2 / LM | **0%** | — | — | — | LoRA가 grounding 파괴 |

**결론:** 7개 중 **base PG2를 종합적으로 이긴 모델은 없다.** 소규모 데이터 LoRA(exp56·58·64)는 grounding을 오히려 악화. pure Kosmos는 OOD 구분 불가.

---

## 3. base PG2 강건성 — "데이터셋이 적은가?"에 대한 답

데이터 표본을 늘리는 대신, 기존 프레임에 변형을 가해 강건성을 직접 측정.

- **측면 4경로**(좌좌/우우/좌우/우좌): 주행 구간(f3+) 추적률 **100%**, full-frame 0%. (miss는 출발 첫 프레임=바구니 화면 밖뿐)
- **free 극단 21ep**(바구니 극단위치·원근·대각선·저조도): hit 92~100%, full-frame 0~2%.
- **증강 11종**(밝기·대비·블러·노이즈·저해상도·회전·좌우반전·부분가림): hit 100%(저대비만 92%), 중심이동 ~0.

→ **grounding은 데이터 부족 문제가 아니다.** 사전학습(WebLI 10억쌍) 지식이 우리 소규모 데이터가 못 덮는 변형까지 일반화. 유일 약점 = **저대비(저조도/역광)** → 데이터 보강 1순위.

---

## 4. 결론 & 다음 단계

1. **grounding = base PaliGemma2 고정** (LoRA 미사용). 실환경 붕괴 해소.
2. **개선 노력은 action head로** — 오예측은 grounding이 아닌 action 문제이므로.
3. **객체 전환: basket → 의자(chair)** — `detect chair` 프롬프트 확정(인식 91%, stool 단어 금지), 색 무관.
4. **의자 데이터 350~500ep 재수집 + Stage2 action 재학습** (저조도 환경 포함).

---

## 5. 이전 판단 정정 (투명성)

| 항목 | 이전 | 정정 |
|------|------|------|
| exp64 (vision LoRA) | "vision LoRA로 grounding 개선 기대" | ❌ full-frame 92% 붕괴 — vision LoRA가 grounding 악화 (CH31/32) |
| 6/4 결정1 (SigLIP+DINOv2) | "DINOv2 추가 LoRA" | PaliGemma2엔 DINOv2 없음(SigLIP 단일) — 결정 폐기 |
| exp59 grounding | "hard-neg로 OOD 억제 성공" | OOD(0%)는 맞으나 **실환경 full-frame 붕괴**라는 더 큰 결함 — 미배포 |
| val TP/FP 지표 | "TP 99%/FP 0% → 성공" | val 지표는 박스 "유무"만 봄 — **품질(위치) 붕괴를 못 잡음**. 실측 평가로만 판정 가능 |

> 📂 근거: `docs/v5/grounding_hub.html` (전 모델 그리드·측면·free·증강·OOD), `scripts/eval_grounding_hub.py`, `scripts/grounding_augmentation_robustness.py`, `scripts/sideangle_per_model.py`

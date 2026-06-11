# Plan — Grounding Evaluation Hub 페이지 (exp50~64 + 측면 베이스라인)

> 작성: 2026-06-11 · 상태: **검토 대기 (승인 전 본구현 금지)**
> 동기: 교수님이 보신 "이상한 에피소드 + 오예측"을 grounding 관점에서 정면 해부.
> 연관: CH31/32 (exp64 full-frame collapse), `eval_exp64_grounding.py`, `grounding_sideangle_episodes.py`

---

## 0. 목적

교수님 우려(실주행 측면/OOD 진입 시 궤적 붕괴·오예측)가 **grounding 문제인가 action 문제인가**를 가른다.
- base PG2가 측면에서도 바구니를 안정적으로 잡으면 → **오예측은 grounding이 아니라 action head 문제**임을 입증.
- 동시에 exp56~64 grounding LoRA들이 base 대비 나은지/나쁜지를 **한 페이지에서 비교**.

산출: 리서치 히스토리(research_story.html)에 링크되는 **Grounding Evaluation Hub** 페이지.

---

## 1. 대상 모델 (조사 완료)

| 모델 | 백본 | LoRA 범위 | r | 상태 |
|---|---|---|---|---|
| **base PG2** | PaliGemma2-mix | 없음 | — | 기준선 |
| exp56 | Kosmos-2 | LM 2모듈 | 8 | dir_acc 0% (실패) |
| exp57 | PaliGemma-1 | LM 2모듈 | 8 | hit 100%, cx_err 0.286 |
| exp58 | PaliGemma-2 | LM 2모듈 | 8 | — |
| exp59 | PaliGemma-2 | LM 3모듈(hard-neg) | 16 | — |
| exp64 | PaliGemma-2 | **vision(SigLIP) 24모듈** | 16 | full-frame collapse (CH31/32) |

> 핵심 비교축: **LM-LoRA(56~59) vs vision-LoRA(64) vs base**, 그리고 백본 3종(Kosmos/PG1/PG2).
> ⚠️ Kosmos backbone `generate()` 주의 — exp56은 별도 처리 또는 제외 검토.

---

## 2. 평가 데이터 세트 (공통 프레임)

1. **표준 시점 49프레임** — `eval_exp64_grounding.py`의 viewpoint 9버킷 샘플 (이미 구현).
2. **측면 4경로** — left_left / right_right / left_right / right_left, 경로당 4ep × 6프레임 (이미 계산 완료, base PG2: ~92% hit / full-frame 0%).
3. **OOD 미학습 객체** — 의자 11장 (오탐 검증, 이미 구현).

각 모델 × 각 세트에서: hit / cx_MAE / cx_std / **full-frame율** / OOD FP.

---

## 3. 페이지 설계 — `docs/v5/grounding_hub.html`

```
[Hero] Grounding Evaluation Hub
  "오예측은 grounding이 아니라 action 문제" — 측면·OOD에서 base PG2 검증 + exp 비교

[Section A] 핵심 결론 카드
  - base PG2: 측면 92% / full-frame 0% → grounding 견고
  - exp64: full-frame 92% 붕괴 → vision-LoRA 실패
  - 결론: decomposition grounding은 base PG2

[Section B] 모델 비교 매트릭스 (표)
  모델 × [hit / cx_MAE / full-frame / OOD FP]  ← 표준 49프레임 기준

[Section C] 측면 경로 에피소드 갤러리 (교수님 오예측 대응)
  path type별 탭: left_left / right_right / left_right / right_left
  각 탭 = 에피소드별 시간순 프레임 그리드 (base PG2 bbox + cx 라벨)
  → "측면 꺾임에서도 base가 추적함" 시각 증거

[Section D] full-frame collapse 대조
  base(타이트) vs exp64(full-frame) 나란히 그리드 (CH31 이미지 재사용)

[Section E] 해석 & 다음 단계
  오예측 = action head 문제 → 의자 데이터 재수집 + Stage2 재학습으로 대응
```

---

## 4. 생성/수정 파일

| 파일 | 작업 |
|---|---|
| `scripts/eval_grounding_hub.py` | **신규** — exp56~64 + base를 공통 세트에 일괄 평가, JSON+그리드 산출 |
| `docs/v5/grounding_hub.html` | **신규** — 위 설계의 허브 페이지 |
| `docs/v5/grounding_hub/` | 모델별 그리드 PNG + 통합 JSON |
| `docs/v5/research_story.html` | 측면/허브 링크 추가 (CH31 또는 next-step에 진입 버튼) |
| `docs/index.html` | Hero 버튼 영역에 "Grounding Hub" 링크 (CLAUDE.md 공개문서 규칙) |

기존 자산 재사용: 측면 그리드 4장(완료), basket_compare_grid(완료), ood_chair_grid(완료).

---

## 5. 단계 (승인 후)

- **Phase 1 (완료):** 측면 4경로 base PG2 평가 + 그리드 4장 + summary JSON. ✅
- **Phase 2:** `eval_grounding_hub.py` — exp57/58/59/64 + base를 표준 49프레임 + OOD에 일괄 평가 (백본별 순차 로드). exp56(Kosmos)은 generate 안정성 확인 후 포함/제외.
- **Phase 3:** `grounding_hub.html` 작성 + 측면 갤러리 임베드.
- **Phase 4:** research_story + index hero 링크.

## 6. 결정 사항 (2026-06-11 확정)

- ✅ **대상 = exp56~64 grounding 전수** + base PG2 (exp50~55는 action 디코더라 grounding 테스트 불가, 제외).
- ✅ **exp56(Kosmos) = 시도 후 판단** — generate 정상이면 포함, 가비지/무한반복이면 "Kosmos generate 불가"로 표기하고 제외.
- ✅ **독립 페이지** `grounding_hub.html` + research_story·index hero 링크.
- 컴퓨팅: 백본 3종 순차 로드 → 약 20~30분 GPU (여유 충분).

## 7. 완료 기준

- [ ] exp57/58/59/64 + base 표준 세트 일괄 평가 JSON
- [ ] grounding_hub.html — 비교 매트릭스 + 측면 갤러리 + collapse 대조
- [ ] research_story + index hero 링크
- [ ] "오예측 = action 문제" 결론 명시

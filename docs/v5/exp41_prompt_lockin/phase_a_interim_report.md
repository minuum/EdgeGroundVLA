# Phase A 중간 보고서 — Exp41B 결과 (2026-05-06)

## 1줄 요약

Exp41B (Exp40 ckpt + path_type_aware, 1 epoch resume): **전 기준 실패 — 텍스트 경로 복구 불가**.  
Exp41C (Exp25 base scratch, 8 epoch) 진행 중 — 이쪽이 진짜 테스트.

---

## Exp41B 결과표

| 기준 | 목표 | 결과 | 판정 |
|---|---|---|---|
| text attention (24L 평균) | ≥ 5% | 0.000% | ❌ |
| action L1 diff (left↔right) | ≥ 1e-2 | 0.00000 | ❌ |
| mean softmax L1 (3-prompt avg) | ≥ 1e-2 | 0.00347 | ❌ |
| PM (val split, 235 samples) | ≥ 50% | 45.1% (106/235) | ❌ |
| pred change / 30 frames | > 0 | 0/30 (0%) | ❌ |

PM 혼동 행렬:
- FORWARD: 69% (87/126), 39개 → FWD+L로 새는 오류
- **LEFT: 0%** (15개 전부 FWD+L 또는 FWD+R로)
- **RIGHT: 0%** (12개 전부 FWD+L 또는 FWD+R로)
- FWD+L: 80%
- FWD+R: 18% (60개 중 40개 → FWD+L)

---

## 해석

- Exp41B는 Exp40 ckpt(grounding_aux로 인한 action collapse 경험) 위에서 1 epoch fine-tune한 것.
- grounding_aux의 `loss_balance_mode=learned`가 action loss를 잠식해 text 경로가 완전히 비활성화된 상태에서 출발함.
- path_type_aware preset 1 epoch만으로는 역전 불가 — backbone 구조 기인이 아닐 수 있음.

---

## Exp41C (진행 중)

config: `configs/mobile_vla_v5_exp41c_scratch_pta.json`
- base: Pure HF Kosmos-2 (`.vlms/kosmos-2-patch14-224`) — text attention 22.7% 보유
- grounding_aux **없음** — action loss가 온전히 살아있음
- 8 epoch scratch 학습
- path_type_aware preset 처음부터 적용

**의의**: Exp25는 const instruction으로 학습 → text 무시. Exp41C는 처음부터 path-dependent instruction → text 신호가 학습 신호로 들어감. 이것이 실패하면 Plan §6 "Phase D" (구조 변경 필요) 결론.

---

## 결과 파일 위치

- attention: `docs/v5/attention_analysis/summary.json`
- PM: `docs/v5/pm_eval/exp41b_results.json`
- sensitivity: `docs/v5/exp41_prompt_lockin/exp41b_sensitivity.json`

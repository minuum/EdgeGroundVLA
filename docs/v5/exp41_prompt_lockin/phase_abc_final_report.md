# Phase A~C 최종 보고서 — Prompt Lock-in 실험 (Exp41B / 41C / 42)

**작성일: 2026-05-07**  
**결론: Phase A 전 실험 FAIL → 구조적 한계 확인 → Exp43 설계 근거**

---

## 1. 실험 목적

"텍스트 프롬프트(left/right/forward)를 바꿨을 때 모델 출력이 달라지는가?"

- **Phase A 통과 기준:**
  | 기준 | 임계값 |
  |------|--------|
  | text attention (24-layer avg) | ≥ 5% |
  | action L1 diff (left↔right) | ≥ 1e-2 |
  | PM (val split, 235 samples) | ≥ 50% |
  | pred changes / 30 frames | > 0 |

---

## 2. 실험별 결과 요약

| 실험 | 기반 | 핵심 변경 | PM | text_attn | pred_changes | 판정 |
|------|------|----------|-----|-----------|-------------|------|
| **Exp41B** | Exp40 ckpt resume | path_type_aware (PTA), 1 epoch | 45.1% | — | 0 | FAIL |
| **Exp41C** | Exp25 scratch | PTA, 8 epoch | 53.6% | 0.000% | 0 | FAIL |
| **Exp42** | Exp25 scratch | counterfactual loss + PTA, 8 epoch | 53.6% | 0.000% | 0 | FAIL |

### Exp41B (PTA resume, 2026-05-05)
- Exp40 ckpt (grounding_aux, val_loss=0.575) 위에서 1 epoch resume
- LEFT 전체 collapse (15프레임 → 전부 FWD+L/FWD+R로)
- 모든 프롬프트에서 동일 출력: `mean_l1_left_vs_right = 0.0`

### Exp41C (PTA scratch, 2026-05-06)
- Exp25 baseline 위에서 path_type_aware prompt augmentation 8 epoch scratch 학습
- PM은 53.6%로 Exp25 수준 유지 (PM 기준만 통과)
- text attention layer-wise 측정: 전 24층 0.000%
- 6가지 프롬프트 포맷 테스트 → 전부 동일 출력

### Exp42 (counterfactual + PTA, 2026-05-06)
- counterfactual loss 추가: 반대 방향 프롬프트 입력 시 다른 출력을 강제하는 학습 신호
- PM 53.6%, text_attention 0.000%, pred_changes 0 — Exp41C와 동일

**Epoch 01 중간 점검 (6 프롬프트 포맷 전수):**

| 포맷 | pred_changes | 대표 출력 |
|------|-------------|---------|
| A_baseline (left/right/forward) | 0 | 전부 FORWARD |
| B_no_grounding | 0 | 전부 FORWARD |
| C_dir_at_end | 0 | 전부 FORWARD |
| D_phrase_tag | 0 | 전부 FORWARD |
| E_short_no_grounding | 0 | 전부 FORWARD |
| F_qa_style | 0 | 전부 FORWARD |

→ 프롬프트 포맷 자체는 문제가 아님.

---

## 3. 근본 원인 (확정)

**Google-robot post-training이 text 경로를 구조적으로 붕괴시킴.**

- Pure HF Kosmos-2: text 22.7% / image 77.3% (정상)
- Google-robot post-train 후: text **0.000%** / image 91.7%
- Exp15 (head-only), Exp41C/42 (full LoRA scratch) — 학습 방식 무관, 모두 0%

즉 word_embedding → backbone → [EOS] → action_head 경로에서 backbone이 text 정보를 이미지로 덮어씀. LoRA, PTA, counterfactual loss 모두 이 경로를 **복구하지 못함**.

---

## 4. Exp43 설계 근거

**골자:** backbone의 text attention 0%를 우회, action head 내에서 text를 직접 주입.

```
기존 경로 (깨진 것):
  lang_x → backbone → [EOS] → action_head
                 ↑
           text attn = 0%  (복구 불가)

Exp43 신규 경로:
  lang_x → word_embedding → token_seq (B, T, 2048)
                                  ↓
  vision_features → LSTM → cross-attention(Q=lstm_out, K/V=text_seq) → logits
                                  ↑
                            backbone 완전 우회
```

- `text_gate = 0.1` 초기화: 처음엔 비전 우선, text 유용할수록 gate가 커짐
- `detach 제거`: gradient가 word_embedding까지 흐름 (backbone weights는 frozen LoRA 경계 내)
- counterfactual loss 유지: text에 반응하는 학습 신호 보강

---

## 5. Exp43 Phase D 기대 기준

| 기준 | 목표 | 비고 |
|------|------|------|
| text_gate (학습 후) | > 0.1 (증가 확인) | 초기값 대비 상승 여부 |
| action L1 diff (left↔right) | ≥ 1e-2 | Phase A 기준 |
| pred_changes | > 0 | Phase A 기준 |
| PM | ≥ 50% | Exp25 수준 유지 |

결과 위치: `docs/v5/exp43_phase_d/` (학습 완료 후 자동 생성)

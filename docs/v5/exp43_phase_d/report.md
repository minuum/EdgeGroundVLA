# Exp43 Phase D 최종 보고서 — Cross-Attention Text Head

**작성일: 2026-05-07**  
**결론: Phase D FAIL → Phase A~D 전체 FAIL 확정**

---

## 1. 실험 설정

| 항목 | 내용 |
|------|------|
| 기반 | Exp42 (counterfactual + PTA, scratch) |
| 핵심 변경 | backbone text attention 0% 우회 — word embedding → cross-attention → action head |
| 구조 | `lang_x → word_embed → token_seq` + `vision → LSTM → cross-attn(Q=lstm, K/V=text) → logits` |
| text_gate 초기값 | 0.1 |
| 학습 | 8 epoch scratch, val_loss best: epoch=06 (10.830) |

---

## 2. Phase D 판정

| 기준 | 임계값 | 결과 | 판정 |
|------|--------|------|------|
| text_gate (학습 후) | > 0.1 상승 | **0.1034** (+0.34%) | ❌ |
| action L1 diff (left↔right) | ≥ 1e-2 | **2.8e-5** (max 6.3e-5) | ❌ |
| pred_changes / 30 frames | > 0 | **0/30 (0%)** | ❌ |
| PM (val 235 samples) | ≥ 50% | **53.6%** (epoch=03/06 동일) | ✅ |

**verdict: `TEXT_INSENSITIVE`**

---

## 3. Sensitivity 상세

- 30 프레임 전부: left / right / forward 프롬프트 → **전부 `TURN_R` 예측**
- 새로운 TURN_R collapse (이전 실험들은 FORWARD collapse)
- `mean_l1_left_vs_right = 2.8e-5` — 기준(1e-2)의 **1/360**
- cross-attention이 text 정보를 전혀 반영하지 못함

---

## 4. val_loss 추이

| epoch | val_loss |
|-------|----------|
| 03 | 10.870 |
| 05 | 10.864 |
| **06** | **10.830** (best) |
| 07 | 11.00 |

Exp25 baseline(10.117)보다 오히려 높음 → cross-attention head 추가가 학습을 방해.

---

## 5. 근본 원인

`text_gate`가 8 epoch 내내 0.1에서 안 움직임 = cross-attention gradient가 word_embedding까지 흘렀지만 **유의미한 학습 신호 없음**.

Google-robot post-train이 text token 자체를 죽인 상태 → K/V로 들어오는 text_seq가 이미 정보 없음 → cross-attention이 Q(vision)와 정렬할 게 없음.

**head 레벨 우회로는 복구 불가. backbone 기인 구조적 한계.**

---

## 6. Phase A~D 전체 요약

| Phase | 실험 | 접근 | 결과 |
|-------|------|------|------|
| A | Exp41B | PTA resume | FAIL |
| A | Exp41C | PTA scratch | FAIL |
| B | Exp42 | counterfactual + PTA | FAIL |
| D | **Exp43** | **cross-attn text head** | **FAIL** |

교수 프로토콜 "실패 시 TICVLA / MobilityVLA 대안 검토" 조건 도달.

---

## 7. 파일 위치

| 파일 | 내용 |
|------|------|
| `docs/v5/exp43_phase_d/sensitivity_epoch06.json` | 30-frame sensitivity 결과 |
| `docs/v5/exp43_phase_d/text_gate.json` | text_gate 추출값 |
| `docs/v5/pm_eval/exp43_epoch06_results.json` | PM eval (epoch=06, 53.6%) |
| `docs/v5/pm_eval/exp43_results.json` | PM eval (epoch=03, 53.6%) |
| `runs/v5_nav/kosmos/mobile_vla_v5_exp43/.../epoch=06-val_loss=10.830.ckpt` | best ckpt |

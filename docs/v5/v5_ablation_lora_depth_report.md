# v5 LoRA-depth Ablation — E2E PaliGemma VLA

> vision tower 상위 N 레이어만 LoRA(r=8) × mm_projector frozen/tuned. window=8, fwd_pred_next_n=5.
> 지표: action val_loss (낮을수록 좋음). 동일 243ep 데이터.

| # | 실험 | LoRA 레이어 | projector | best val_loss | 상태 |
|---|---|---|---|---|---|
| 1 | v5_ablation_top2_proj_frozen | 25-26 (top2) | frozen | **0.435** | ✅ 완료 |
| 2 | v5_ablation_top2_proj_tuned | 25-26 (top2) | tuned | **0.435** | ✅ 완료 |
| 3 | v5_ablation_top4_proj_frozen | 23-26 (top4) | frozen | **0.433** | ✅ 완료 |
| 4 | v5_ablation_top4_proj_tuned | 23-26 (top4) | tuned | **0.433** | ✅ 완료 |
| 5 | v5_ablation_top6_proj_frozen | 21-26 (top6) | frozen | **0.433** | ✅ 완료 |
| 6 | v5_ablation_top6_proj_tuned | 21-26 (top6) | tuned | **0.433** | ✅ 완료 |
| 7 | v5_ablation_top8_proj_frozen | 19-26 (top8) | frozen | **0.435** | ✅ 완료 |
| 8 | v5_ablation_top8_proj_tuned | 19-26 (top8) | tuned | **0.437** | ✅ 완료 |

## 핵심 발견

- best: **v5_ablation_top4_proj_frozen** (val_loss 0.433)
- 전체 스프레드: **0.004** (0.433~0.437)
- → 레이어 깊이(2→8)·projector tuning 모두 val_loss에 **유의미한 차이 없음**.
  E2E VLA는 LoRA 깊이가 병목이 아님(데이터 한계). C1(E2E exp63 CL 18.8%)과 정합.

## ⚠️ 중대 caveat — vision LoRA 미학습 (2026-06-08 검증)

`probe_e2e_grounding.py`로 ckpt를 점검한 결과, **모든 8개 config의 vision-tower LoRA가 미학습**:
- config `train_vision=False` → vision tower(및 그 LoRA) 동결.
- ckpt `lora_B 절대합 = 0.0000` (B·A=0 = 항등). `lora_A`만 랜덤 init 상태.
- 실제 학습된 파라미터: **action head + mm_projector**(proj_tuned 시) 뿐.

**함의:**
1. "LoRA 깊이(top2~8)" 축은 **사실상 no-op** — val_loss 평탄(0.433~0.437)의 근본 원인은
   "깊이 무의미"가 아니라 **vision LoRA가 애초에 학습되지 않았기 때문**.
2. 8개 모델의 grounding은 **base PG1과 동일**(hit 58%, full-frame 0%, cxMAE 0.108) →
   E2E action 학습은 vision 인코더를 건드리지 않아 **벽/의자 오트래킹이 없음**(CH27 ⑥).
3. 진짜 "vision LoRA 깊이" 효과를 보려면 `train_vision=True`로 재설계 필요.

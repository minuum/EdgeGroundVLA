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
| 8 | v5_ablation_top8_proj_tuned | 19-26 (top8) | tuned | **—** | ⏳ 대기 |

## 핵심 발견

- best: **v5_ablation_top4_proj_frozen** (val_loss 0.433)
- 전체 스프레드: **0.002** (0.433~0.435)
- → 레이어 깊이(2→8)·projector tuning 모두 val_loss에 **유의미한 차이 없음**.
  E2E VLA는 LoRA 깊이가 병목이 아님(데이터 한계). C1(E2E exp63 CL 18.8%)과 정합.

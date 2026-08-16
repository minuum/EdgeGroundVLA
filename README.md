# EdgeGround-VLA — Heterogeneous Grounding for On-Device Goal-Directed Navigation

> Open-vocabulary detection + lightweight action head for mobile robot basket navigation.
> **실기 95/100 (95.0%)** — OWL-v2 + Kosmos-2 vision + image_proj + MLP, 학습 파라미터 1.128M.

**마지막 업데이트**: <!-- SYNC:updated -->2026-08-16<!-- /SYNC:updated -->
**GitHub Pages**: https://minuum.github.io/EdgeGroundVLA/

---

## 핵심 결과

<!-- SYNC:results_table:start -->
| Method | Architecture | 실기 성공률 ↑ | val_acc | Note |
|---|---|---|---|---|
| E2E VLA (Exp11) | Kosmos-2 + LoRA | 0.0% (CL) | 58.6% PM | Text attn 0%, structural failure |
| Decomp v1 (Exp14) | CLIP + BBox MLP | 66.7% (CL) | 75.9% PM | First decomposition baseline |
| Simple MLP (Exp65b) | CLIP + plain MLP | 10.3% (CL) | — | No L2-norm, no aug → pipeline ablation |
| Ours (Exp66, legacy) | CLIP + L2-norm + aug | 96.6% (CL, 시뮬) | 93.5% | 구 SOTA · 시뮬 closed-loop 기준 |
| **Ours (exp73) ★** | **OWL-v2 + Kosmos-2 vision + image_proj + MLP** | **95.0% (실기 95/100)** | **74.1%** | 현재 배포 구성 |
<!-- SYNC:results_table:end -->

**Pipeline ablation**: Simple MLP 10.3% → L2+aug 96.6% (×9.4 gap).
Grounding source (HSV / base PG2 / LoRA cx) irrelevant once pipeline is correct.

---

## 아키텍처

```
[RGB Frame 224×224]
       ↓
[Kosmos-2 Vision Encoder] — frozen
       ↓ 1024-dim
[image_proj → 256-dim, L2-normalize]
       ↓
[Concatenate] ← BBox History (cx,cy,area,has_bbox × 8frames) = 32-dim
       ↓ 288-dim
[ActionMLP: 256→128→64→8]
       ↓
[8 Actions: STOP / FWD / LEFT / RIGHT / FWD+L / FWD+R / ROT_L / ROT_R]
       ↑
[Proximity Override: area≥0.50 AND |cx-0.5|≤0.30, 2 consecutive → STOP]
```

- **Stage 1 v2**: Kosmos-2 encoder + image_proj (val_acc 98.1%)
- **Stage 2 (Exp66)**: ActionMLP on frozen Stage 1 features (val_acc 93.5%, CL 96.6%)

---

## 주요 발견

1. **Text attention = 0%** — Google-robot post-trained Kosmos-2의 구조적 사망. LoRA/head-only 모두 복구 불가. E2E 실패의 근본 원인.
2. **Pipeline이 유일 결정 변수** — L2-norm + bbox augmentation이 성능의 전부. Grounding 소스 무관.
3. **Basket localization 이중 증명** — Zero-shot probe 96.6% + masking 9/9 flip (Exp66, base PG2). 이미지 경로가 basket을 독립적으로 인식.

---

## 데이터셋

| 경로 | 에피소드 | 비고 |
|---|---|---|
| `ROS_action/mobile_vla_dataset_v5/` | 244개 | basket, 9 path types + free 22개 |
| `ROS_action/mobile_vla_dataset_v5_add_free/` | 220개 | 리밸런싱 버전 (24~26개/type) |
| `ROS_action/mobile_vla_dataset_v5_2/` | 59개 | 의자 (별도 모델 필요) |

---

## 핵심 파일

| 파일 | 설명 |
|---|---|
| `scripts/train_exp54_stage2_v2_action.py` | Exp66 Stage2 학습 |
| `robovlm_nav/serve/stage2_v2_inference_server.py` | Stage2 v2 추론 서버 |
| `scripts/sim/evaluate_closed_loop_v5.py` | Closed-loop 평가 |
| `scripts/measure_attention.py` | Text attention 측정 |
| `docs/v5/bbox_frame_level/bbox_dataset_base_pg2_cx_243.json` | Stage2 bbox 레이블 (243 ep) |

## 체크포인트

<!-- SYNC:checkpoints:start -->
| 모델 | 경로 | 비고 |
|---|---|---|
| Stage 1 (image_proj, 5-class) ★ | `runs/v5_nav/mlp/stage1_v3_5cls/stage1_v3_5cls_owl_projs.pt` | val_acc 94.09% · 225ep · OWL-v2 라벨 |
| Stage 2 (MLP action head) ★ | `runs/v5_nav/mlp/exp73_stage1v3/exp73_owl_stage1v3_v6_mlp.pt` | val_acc 74.13% · window=6 · bbox_scale=3.0 |
| 비전 특징 캐시 | `docs/v5/closed_loop_eval/exp73_v6_vis_cache_stage1v3.pt` | 225ep 재인코딩 (Stage1 완료 후 생성) |
| Stage 1 v2 (legacy, CLIP) | `runs/v5_nav/mlp/shared/stage1_v2_projs.pt` | Exp66 계열 |
| Stage 2 MLP w=4 (legacy, Exp66) | `runs/v5_nav/mlp/exp66/action_mlp.pt` | Exp66 계열 |
<!-- SYNC:checkpoints:end -->

---

## 문서

- **전체 연구 여정 (CH1→CH36)**: [research_story.html](https://minuum.github.io/EdgeGroundVLA/v5/research_story.html)
- **시각 증거 (VIS)**: [research_story.html#vis](https://minuum.github.io/EdgeGroundVLA/v5/research_story.html#vis)
- **Grounding Hub**: [grounding_hub.html](https://minuum.github.io/EdgeGroundVLA/v5/grounding_hub.html)
- **구 실험 아카이브**: [legacy.html](https://minuum.github.io/EdgeGroundVLA/legacy.html)
- **에이전트 진입점**: `docs/AGENT_ENTRYPOINT.md`

---

> ⚠️ `third_party/RoboVLMs/` 수정 금지 — frozen backbone
> ⚠️ Google-robot backbone으로 `generate()` 절대 호출 금지 — 무한 반복

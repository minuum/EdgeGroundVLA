---
name: deploy-stage2-v2
description: Stage2 v2 체크포인트(stage1 encoder + best action head)와 추론 서버를 soda@100.85.118.58:~/MoNaVLA 로 rsync 전송. 같은 레포 구조 그대로 유지.
---

# Deploy Stage2 v2

Stage2 v2 모델 파일을 `soda@100.85.118.58:~/MoNaVLA` 로 전송. 같은 레포이므로 경로 구조 그대로.

## 실제 스크립트

```
scripts/deploy/rsync_stage2_v2.sh [--all]
```

## 전송 파일 목록

| 파일 | 크기 | 용도 |
|------|------|------|
| `runs/v5_nav/mlp/shared/stage1_v2_projs.pt` | 3.1MB | Stage1 vision encoder + image_proj |
| `runs/v5_nav/mlp/exp66/action_mlp.pt` | 456KB | Stage2 SOTA (96.6% CL, Exp66 ActionMLP w=8) |
| `robovlm_nav/serve/stage2_v2_inference_server.py` | ~15KB | FastAPI 추론 서버 |
| `scripts/train_exp54_stage2_v2_action.py` | — | 학습 스크립트 |
| `scripts/eval_exp54_stage2_v2_closedloop.py` | — | CL 평가 스크립트 |
| `docs/v5/DEPLOY_MANIFEST.json` | ~2KB | 모델 파라미터 요약 |

## 사용법

```bash
# best 모델만 (기본)
bash scripts/deploy/rsync_stage2_v2.sh

# ablation ckpt 전부 포함 (window 변형 등)
bash scripts/deploy/rsync_stage2_v2.sh --all
```

## 모델 파라미터 (최선 모델 기준)

| 항목 | 값 |
|------|-----|
| Head | ActionMLP |
| Window | 8 |
| d_in | 288 (8×4 + 256) |
| val_acc | 93.5% |
| CL (Closed-Loop) | 96.6% |
| FPE | 0.102m |
| cx source | base PG2 (HSV aug) |

## 주의사항

- Kosmos-2 그라운딩 모델 (`.vlms/kosmos-2-patch14-224`) 은 용량이 커서 별도 전송 필요
- Stage1 인코더는 Kosmos-2 vision_model 위에서 동작 — `VLA_GROUNDING_MODEL_PATH` 필수
- Google-robot pretrained (`inference_server.py`) 와 완전히 다른 파이프라인. 혼동 주의.

## 에이전트 사용 예시

"stage2 v2 soda 서버로 보내줘":
```bash
bash scripts/deploy/rsync_stage2_v2.sh
```

"stage2 v2 ablation ckpt 전부 포함해서 보내줘":
```bash
bash scripts/deploy/rsync_stage2_v2.sh --all
```

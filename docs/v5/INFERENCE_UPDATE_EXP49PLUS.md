# Inference Server 업데이트 노트 — Exp49+ 모델

작성일: 2026-05-22  
대상 브랜치: `inference-integration`, `monavla-driving`

---

## 현재 상태 (업데이트 전)

| 브랜치 | 배포 모델 | D_IN | 아키텍처 |
|--------|----------|------|---------|
| `inference-integration` | Exp47 MLP (InstructionMLPInference) | 3104 | bbox(32) + vis(1024) + instr_emb(2048) |
| `monavla-driving` | Exp47 MLP (InstructionMLPInference) | 3104 | 동일 |

현재 서버: `soda@100.85.118.58 ~/MoNaVLA`  
기본 배포: **Exp17** (primary, CL 11.1%) / **Exp18** (fallback, CL 11.1%)

---

## 전송된 체크포인트 목록

전체 상세 정보: `docs/v5/CHECKPOINT_REGISTRY_EXP49PLUS.json`

### 그룹 A — Pre-extracted feature 기반 MLP (VLM 불필요)

| 모델 | ckpt 경로 | D_IN | vision feature 소스 |
|------|----------|------|-------------------|
| **exp49** | `runs/v5_nav/mlp/exp49/exp49_mlp.pt` | 1056 | `bbox_nav_exp46/vision_features.npz` |
| **exp50** | `runs/v5_nav/mlp/exp50/exp50_mlp.pt` | 1056 | `bbox_nav_exp46/vision_features.npz` |
| **exp51** | `runs/v5_nav/mlp/exp51/exp51_mlp.pt` | 1056 | `bbox_nav_exp46/vision_features.npz` |
| **exp52** | `runs/v5_nav/mlp/exp52/exp52_mlp.pt` | 2080 | `bbox_nav_exp52/lang_vis_features.npz` |

> ⚠️ exp49~51은 inference 시 실시간 VLM 인코딩 필요 (pre-extracted feature는 eval 전용)  
> 실서버 배포 시 VLM forward pass 경로 추가 필요 (아래 섹션 참조)

### 그룹 B — CLIP LoRA 기반 MLP (VLM 필요)

| 모델 | ckpt 경로 | D_IN | 추가 필요 파일 |
|------|----------|------|-------------|
| **exp53** | `runs/v5_nav/mlp/exp53/exp53_clip_lora.pt` | 1056 | `exp53/clip_lora_adapter/` (PEFT, 2.1MB) |
| **exp54 stage2 v1** | `runs/v5_nav/mlp/exp54/stage2/stage2_mlp.pt` | 1056 | exp53 LoRA adapter 동일 |

### 그룹 C — FrozenCLIPV2 기반 (VLM + image_proj 필요, 학습 완료 후 전송 예정)

| 모델 | ckpt 경로 | D_IN | 추가 필요 파일 |
|------|----------|------|-------------|
| **exp54 stage2 v2** | `runs/v5_nav/mlp/exp54/stage2_v2/stage2_v2_mlp.pt` | **288** | `exp54/stage1_v2/stage1_v2_projs.pt` |

> 학습 중 (PID 1405772). 완료 후 별도 rsync 필요.

---

## Inference Server 수정 사항

### 1. exp49/50/51 실시간 추론 추가 (그룹 A)

현재 `InstructionMLPInference` (Exp47, D_IN=3104)와 **완전히 다른 아키텍처**.  
새 클래스 `GoalNavMLPInference` 추가 필요.

**MLP 구조** (exp49/50/51 공통):
```python
D_IN = 8 * 4 + 1024  # 1056: bbox(32) + vision_feat(1024)

class GoalNavMLP(nn.Module):
    def __init__(self):
        self.net = nn.Sequential(
            nn.Linear(1056, 512), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(512, 256),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 8),
        )
```

**Vision feature 추출** (실시간):
```python
# Kosmos-2 base vision model (google-robot 아님 — pure HF)
# .vlms/kosmos-2-patch14-224 의 vision_model 레이어만 사용
vision_feat = vision_model(pixel_values).last_hidden_state.mean(dim=1)  # (1, 1024)
```

**bbox feature 포맷** (WINDOW=8):
```python
# frames[-8:] 각각에서: [cx, cy, area, has_bbox]  → 32-dim
bbox_feat = np.array([
    [fr["cx"], fr["cy"], fr["area"], float(fr["has_bbox"])]
    for fr in history[-8:]
], dtype=np.float32).flatten()
```

---

### 2. exp53 실시간 추론 추가 (그룹 B)

exp53은 PEFT LoRA adapter를 Kosmos-2 vision_model에 적용한 버전.

```python
from peft import PeftModel
base = AutoModelForVision2Seq.from_pretrained(".vlms/kosmos-2-patch14-224", ...)
vision_model_lora = PeftModel.from_pretrained(
    base.vision_model,
    "runs/v5_nav/mlp/exp53/clip_lora_adapter/"
)
# 이후 encode는 exp49와 동일
```

**주의:** exp53 action head의 ckpt 키는 `"mlp"`. 로드 방법:
```python
ckpt = torch.load("runs/v5_nav/mlp/exp53/exp53_clip_lora.pt")
mlp.load_state_dict(ckpt["mlp"])
```

---

### 3. exp54 stage2 v2 실시간 추론 추가 (그룹 C, 학습 완료 후)

D_IN=288로 가장 작고 빠름. **image_proj (1024→256)** 레이어가 추가됨.

```python
# stage1_v2 checkpoint에서 image_proj 로드
stage1_ckpt = torch.load("runs/v5_nav/mlp/exp54/stage1_v2/stage1_v2_projs.pt")
image_proj = nn.Linear(1024, 256)
image_proj.load_state_dict(stage1_ckpt["image_proj"])

# encode:
import torch.nn.functional as F
raw_feat = vision_model(pixel_values).last_hidden_state.mean(dim=1)  # (1, 1024)
proj_feat = F.normalize(image_proj(raw_feat.float()), dim=-1)         # (1, 256)

# MLP D_IN = 32 + 256 = 288
x = torch.cat([bbox_feat_tensor, proj_feat], dim=-1)  # (1, 288)
action = mlp(x).argmax(dim=-1)
```

---

## 액션 클래스 매핑 (8-class, V5 공통)

```
0: STOP      → linear_x=0,    linear_y=0
1: FORWARD   → linear_x=0.3,  linear_y=0
2: LEFT      → linear_x=0,    linear_y=0.3
3: RIGHT     → linear_x=0,    linear_y=-0.3
4: FWD+L     → linear_x=0.3,  linear_y=0.3
5: FWD+R     → linear_x=0.3,  linear_y=-0.3
6: ROT_L     → linear_x=0,    linear_y=0.5   (제자리 회전)
7: ROT_R     → linear_x=0,    linear_y=-0.5  (제자리 회전)
```

> ⚠️ inference_server.py 에 9-class 매핑이 있을 수 있음 — **8-class만 사용할 것**

---

## Vision Feature 주의사항

### Pure HF Kosmos-2 vs Google-robot backbone

| 구분 | 경로 | 용도 |
|------|------|------|
| Pure HF | `.vlms/kosmos-2-patch14-224` | **exp49~54 전부** — `generate()` 정상 |
| Google-robot | `.vlms/google_robot_pretrain/kosmos_ph_google-robot-post-train.pt` | Exp11~18 — `generate()` 절대 금지 |

exp49+ 모델은 **모두 Pure HF Kosmos-2** vision encoder 기반.  
Google-robot backbone으로 feature를 추출하면 완전히 다른 feature 공간 → 예측 불가능.

### Vision model dtype

```python
# 학습 시: float16으로 인코딩
pv = inputs["pixel_values"].to(device, dtype=torch.float16)
out = vision_model(pixel_values=pv)
feat = out.last_hidden_state.mean(dim=1).float()  # float32로 변환 후 MLP
```

---

## 평가 우선순위 (SODA에서 실행 순서)

```bash
# 1. Pre-cached feature 기반 (빠름, VLM forward pass 없음)
python3 scripts/sim/evaluate_closed_loop_v5.py --model exp49
python3 scripts/sim/evaluate_closed_loop_v5.py --model exp50
python3 scripts/sim/evaluate_closed_loop_v5.py --model exp51
python3 scripts/sim/evaluate_closed_loop_v5.py --model exp52

# 2. CLIP LoRA 기반 PM 평가 (VLM 필요)
.venv/bin/python3 scripts/eval_exp54_stage2.py

# 3. Stage2 v2 (학습 완료 + rsync 후)
.venv/bin/python3 scripts/eval_exp54_stage2_v2.py
.venv/bin/python3 scripts/eval_exp54_stage2_v2_closedloop.py
```

---

## 배포 교체 판단 기준

| 기준 | 값 |
|------|---|
| 현재 서버 CL success | **11.1%** (Exp17/18) |
| 교체 고려 임계값 | **≥ 33.3%** (3배) |
| 참고 최선 기록 | 55.6% (Exp25) |

CL 평가 결과가 33% 이상인 모델이 나오면 서버 교체 진행.  
체크포인트 경로는 `CHECKPOINT_REGISTRY_EXP49PLUS.json` 참조.

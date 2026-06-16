# 모델 구조 분석 + 로봇서버 이관 플랜

작성: 2026-05-27

---

## 1. Google Robot Kosmos 구조 해부

### Kosmos-2 원형 구조

```
Kosmos2ForConditionalGeneration
├── vision_model         ← ViT (SigLIP 계열, 이미지 패치 → 1024D 임베딩)
├── image_to_text_projection  ← 1024D → text_model hidden size로 맵핑
├── text_model           ← LM (Transformer decoder)
│   └── lm_head          ← 최종 토큰 분포 → 텍스트 생성
└── generate()           ← text_model.generate() 호출
```

### RoboVLMs (Google Robot Post-train)이 쓰는 방식

generate()를 **절대 호출하지 않음**. 대신:

```python
# base_backbone.py:217-225
def model_encode_images(self, images):
    vision_outputs = self.model.vision_model(pixel_values=images)
    image_embeds = self.model.image_to_text_projection(
        vision_outputs.last_hidden_state
    )
    return image_embeds  # (B, N_patches, D) — 시각 특징만 뽑음
```

전체 forward 흐름:

```
이미지
  → vision_model (ViT)           [frozen or LoRA]
  → image_to_text_projection     [frozen or LoRA]
  → image_embeds (시각 토큰들)

텍스트 지시 ("basket is on the left")
  → word_embedding               [frozen]
  → text_embeds

[image_embeds + text_embeds + action_token]
  → text_model (LM decoder) — forward pass only
  → output.hidden_states[-1]     ← 마지막 레이어 hidden state

action_token 위치의 hidden state
  → action_head (LSTM/MLP/Linear)
  → 8-class 분류 or 연속 속도 출력
```

핵심: **lm_head(텍스트 생성)는 완전히 우회. hidden_state를 직접 action_head로 연결.**

---

## 2. 왜 텍스트 경로(text attention)가 0%가 됐나

### Google Post-training 데이터

- RT-1 (Google): "pick up the apple", "place on the plate" 등 **반복적·고정 패턴**
- Open-X Embodiment: 물체 조작 위주, 다양하지만 언어가 단순

### 수렴 과정

```
[학습 초기] 이미지 + 텍스트 → action

모델 입장: 이미지만 봐도 action이 충분히 예측됨
          텍스트는 항상 비슷한 패턴 → gradient 기여 거의 없음

[수렴 후] text attention → 0%
         이미지 경로만 살아남음
```

### 왜 generate()가 망가지나

학습 목표: `action_token` 위치에서 올바른 action 분포 예측  
lm_head의 의도: 원래 자연어 토큰 분포 → 이제 action 분포에 맞게 변형됨

```
generate() 호출 시:
  → lm_head가 action-biased distribution으로 토큰 샘플링
  → 실제 어휘와 맞지 않는 이상한 토큰 반복
  → "Tin Tin Tin Roof..." 무한 루프
```

### 우리 실험에서 확인 (Exp15)

```
vision_model + image_to_text_projection  → text attn = 0.000%  (변화 없음)
text_model (LM)                          → text attn = 0.000%  (구조적 사망)
LoRA fine-tuning 후에도                  → 동일 0.000%
```

**결론: Google post-training이 텍스트 경로를 구조적으로 닫아버림. 우리 학습과 무관.**

---

## 3. 현재 파이프라인 구조 (Exp54, 최선)

```
[Stage 1 — CLIP 특징 추출 + 위치 정렬]

이미지 (224×224)
  → Pure Kosmos-2 vision_model (frozen)
  → last_hidden_state (257, 1024D)
  → image_proj: 1024D → 256D
  
bbox (HSV 검출)
  → cx_det, cy_det, area_det
  → window=8 프레임 flatten → 32D

Stage 2 입력: [proj256 + bbox32] = 288D
  → ActionMLP
  → 8-class softmax → action

※ Google Robot Kosmos = 이 파이프라인에서 사용 안 함
※ Pure Kosmos-2 vision_model만 사용 (generate() 없이)
```

### 왜 Google Robot이 아닌 Pure Kosmos-2를 쓰나

| | Google Robot Kosmos | Pure Kosmos-2 |
|--|---------------------|---------------|
| vision_model | ✅ 로봇 환경 특화 | ✅ 일반 범용 |
| text 경로 | ❌ 구조적 사망 | ✅ 정상 |
| generate() | ❌ 망가짐 | ✅ 정상 |
| grounding | ❌ 불가 | ✅ bbox 출력 가능 |
| Exp54 Stage1 probe | — | **96.6% (frozen), 98.1% (LoRA)** |

Stage 1 contrastive로 Pure Kosmos-2가 이미 basket 위치를 충분히 인코딩함  
→ 굳이 Google Robot backbone 필요 없음

---

## 4. 로봇서버 이관 플랜

### 4-1. 현재 서버 구성

```
로봇서버 (SODA)
├── inference_server.py (FastAPI, port 8000)
│   └── Exp17/18 기반 (구버전) ← 교체 대상
├── H5 데이터: mobile_vla_dataset_v5/ (150 episodes)
└── GPU 있음
```

### 4-2. 이관할 모델

| 모델 | 역할 | 파일 |
|------|------|------|
| Pure Kosmos-2 vision_model | Stage1 특징 추출 | `.vlms/kosmos-2-patch14-224/` (1.6GB) |
| Stage1 v2 프로젝터 | 1024D→256D 맵핑 | `runs/v5_nav/mlp/exp54/stage1_v2/stage1_v2_projs.pt` |
| Stage2 v2 MLP | action 예측 | `runs/v5_nav/mlp/exp54/stage2_v2/stage2_v2_mlp.pt` |
| Grounding LoRA (Exp56) | 텍스트→bbox | `runs/v5_nav/grounding/exp56/` (학습 완료 후) |

### 4-3. 이관 명령어

```bash
# minum 머신에서 실행
SERVER="user@robot-server-ip"
REMOTE="/home/user/MoNaVLA"

# 1. 추론 스크립트
scp scripts/eval_exp54_stage2_v2_closedloop.py  $SERVER:$REMOTE/scripts/
scp scripts/run_grounding_realtime.py            $SERVER:$REMOTE/scripts/
scp docs/v5/ROBOT_SERVER_GROUNDING_TEST.md      $SERVER:$REMOTE/docs/

# 2. Stage2 v2 체크포인트 (최선 모델)
rsync -av runs/v5_nav/mlp/exp54/ $SERVER:$REMOTE/runs/v5_nav/mlp/exp54/

# 3. Grounding LoRA (Exp56 완료 후)
rsync -av runs/v5_nav/grounding/exp56/ $SERVER:$REMOTE/runs/v5_nav/grounding/exp56/

# ※ Pure Kosmos-2 모델은 서버에 이미 있을 가능성 높음
#   없으면: rsync -av .vlms/kosmos-2-patch14-224/ $SERVER:$REMOTE/.vlms/kosmos-2-patch14-224/
```

### 4-4. 서버에서 실행할 테스트 순서

**Step 1: Stage2 v2 closed-loop 재현**
```bash
python3 scripts/eval_exp54_stage2_v2_closedloop.py
# 목표: 96.67% 재현 (교수님 R2-1 대응)
```

**Step 2: Grounding 실시간 데모 (Pure Kosmos-2)**
```bash
# 여러 phrase 비교 — R2-3 핵심 데모
python3 scripts/run_grounding_realtime.py \
    --source /path/to/episode_xxx.h5 \
    --phrases "gray basket" "red ball" "person" \
    --serve
# http://server-ip:7860/realtime_test/live.html 에서 확인
```

**Step 3: Exp56 Grounding LoRA vs Pure Kosmos-2 비교**
```bash
# A. 순수 Kosmos-2 (baseline)
python3 scripts/run_grounding_realtime.py \
    --source /path/to/episode_xxx.h5 \
    --phrases "gray basket" "red ball" \
    --out-dir docs/v5/grounding_demo/pure

# B. Exp56 LoRA (학습된 버전)
python3 scripts/run_grounding_realtime.py \
    --source /path/to/episode_xxx.h5 \
    --adapter runs/v5_nav/grounding/exp56 \
    --phrases "gray basket" "red ball" \
    --out-dir docs/v5/grounding_demo/exp56
```

---

## 5. 모델별 역할 요약 (한눈에)

```
[현재 서버 — 교체 전]
Google Robot Kosmos E2E (Exp11/17/18)
  └─ vision_model + text_model → hidden_state → action_head (LoRA)
  └─ 결과: 58.6% PM, 0% CL  ← 실패

[이관 후 — 교체 타겟]
A. Navigation (Stage2 v2):
   Pure Kosmos-2 vision_model (frozen)
     └─ Stage1 v2 proj (1024D→256D)  ← basket 위치 정렬 학습됨
     └─ Stage2 v2 MLP (288D→8class) ← action 예측
   HSV bbox detector → cx/cy 입력
   결과: 96.67% CL  ← 현재 최선

B. Grounding (Exp56 LoRA):
   Pure Kosmos-2 + Exp56 adapter
     └─ "gray basket" → bbox (cx, cy) 출력
     └─ 텍스트 바꾸면 다른 bbox → 다른 action (R2-3 증거)
   generate() 사용 — 이 경우에만

C. (미래) Goal-Conditioned:
   텍스트 phrase → Grounding LoRA → cx/cy → Stage2 MLP → action
   HSV bbox 없이 순수 텍스트 기반 내비게이션
```

---

## 6. 체크리스트

- [ ] 서버에 Pure Kosmos-2 모델 존재 확인
- [ ] Stage2 v2 체크포인트 전송 (`exp54/stage2_v2/`)
- [ ] Exp56 Grounding LoRA 학습 완료 대기 (`logs/exp56_grounding_lora.log`)
- [ ] `run_grounding_realtime.py` 서버 전송
- [ ] Step 1: CL 96.67% 재현 확인
- [ ] Step 2: 실시간 grounding "gray basket" vs "red ball" 비교
- [ ] Step 3: 교수님 데모용 overlay 이미지 저장

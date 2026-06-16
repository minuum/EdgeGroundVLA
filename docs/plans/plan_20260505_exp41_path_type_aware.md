# Plan — Exp41: Path-Type-Aware Prompt Re-Conditioning (Phase A)

작성: 2026-05-05
브랜치: inference-integration
선행 보고서: `docs/v5/grounding_3tier_ablation.md`, `docs/v5/exp40_object_recognition_proof.md`
이전 계획: `docs/plans/plan_20260502_grounding_comparison.md` (4-model 비교 결과 텍스트 무감각 확정)

## 0. 목표 (one-liner)

V5 학습 데이터의 텍스트가 left/right/straight 모두 동일한 한 문장(`"Navigate straight forward to the gray basket. 바구니를 향해 직진해."`)으로 고정되어 있어 모델이 텍스트를 무시하도록 학습된 것을 확인. **`instruction_preset: "path_type_aware"` 한 줄만 추가한 Exp41**으로 텍스트가 학습 신호로 들어가는지 5 epoch 안에 판가름.

성공 조건: text attention 0% → 비-0%, 같은 이미지 + 다른 prompt 입력 시 액션 변화 (현재 4 모델 모두 < 1e-3, 목표 > 1e-2).

## 1. 근거 (research summary)

### 1.1 데이터 측 — H5 instruction 동일성 확인 (5/5)

```bash
# 150 episodes 전수 조사
left_path  → "<grounding>Navigate straight forward to the gray basket. 바구니를 향해 직진해."
right_path → "<grounding>Navigate straight forward to the gray basket. 바구니를 향해 직진해."
straight_path → "<grounding>Navigate straight forward to the gray basket. 바구니를 향해 직진해."
unique_count: 1
```

### 1.2 코드 측 — Exp17~Exp40 lineage가 H5 텍스트로 fall-through

`robovlm_nav/datasets/nav_h5_dataset_impl.py:619-639`:
```python
# preset 미설정 시 분기
elif 'language_instruction' in f:
    raw = f['language_instruction'][0]   # ← H5의 단일 문자열
    language_base = raw.decode(...)
    if self.grounding_prefix and not language_base.startswith('<grounding>'):
        language_base = f"<grounding>{language_base}"
```

`grep instruction_preset configs/mobile_vla_v5_exp{17,25,39,40}*.json` → **0 hits**. 즉 이 lineage 전부 텍스트가 const.

### 1.3 코드 측 — 이미 사용 가능한 preset

`nav_h5_dataset_impl.py:487-505` `_get_path_type_instruction()`:
- 파일명에서 `left_path` / `right_path` / `straight_path` 자동 감지
- `PATH_TYPE_INSTRUCTIONS` dict에서 10개 variation 무작위 선택
- 출력 형식: `"<grounding>Instruction: {variation}. Action:"`

### 1.4 검증 자산

- `scripts/measure_attention.py`:114-118에 이미 `INSTRUCTIONS = {left, right, forward}` 정의됨. Exp41 등록만 하면 즉시 비교 가능.
- `scripts/test_v5_pm_dm.py`:65에 `--instruction_preset` CLI 인자 이미 있음. 평가 시 path_type_aware로 바꿔 측정 가능.

## 2. 접근 (한 줄 수정 → resume from Exp25)

### Q1. 새 학습 vs Exp40 resume

| 옵션 | 비용 | 위험 |
|---|---|---|
| **A. Exp25 ckpt에서 resume + path_type_aware** (추천) | 5 epoch ~ 30분 | grounding_aux 영향 분리 명확 |
| B. Exp40 ckpt에서 resume + preset만 추가 | 5 epoch | grounding_aux의 액션 collapse와 prompt 효과가 섞임 |
| C. scratch (Exp25 config + preset) | 15+ epoch ~ 1.5시간 | 다른 변수 다 같음. 가장 정직 |

**추천: A**. Exp40의 액션 collapse가 grounding_aux 의심이라, 그건 분리해서 디버깅. Phase A는 prompt 효과만 본다.

### Q2. preset 선택

- `path_type_aware`: 파일명 기반 (left/right/straight). 매 step random variation.
- `action_aware_train`: 실제 action에서 라벨 추출 (forward/left/right/diag/stop). 더 정직하지만 첫 단계에서는 노이즈 큼.

**추천: path_type_aware (Phase A) → action_aware_train (Phase B에서)**

### Q3. resume 정합성

Exp25 ckpt는 const 텍스트로 학습된 가중치. path_type_aware로 변경하면 분포 shift 발생. 5 epoch fine-tune으로 충분히 적응할 것 (val_loss 추적으로 확인).

## 3. 변경 사항

### 3.1 두 config 병렬 (B: resume, C: scratch)

**B 변형 (`configs/mobile_vla_v5_exp41b_resume_exp40_pta.json`)** — Exp40 ckpt에서 5 epoch 추가 학습:
```json
{
    "_comment": "Exp41B: Exp40 ckpt resume + path_type_aware. grounding_aux 그대로, prompt 신호만 추가. 5 epoch fine-tune.",
    "parent": "configs/mobile_vla_v5_exp40_fix_chunking_grounding.json",
    "exp_name": "v5-exp41b-resume-exp40-pta",
    "task_name": "mobile_vla_v5_exp41b",
    "resume": "runs/v5_nav/kosmos/mobile_vla_v5_exp40/2026-05-04/v5-exp40-fix-chunking-grounding/last.ckpt",
    "trainer": {"max_epochs": 5, "check_val_every_n_epoch": 1},
    "train_dataset": {"instruction_preset": "path_type_aware"},
    "val_dataset":   {"instruction_preset": "path_type_aware"}
}
```

**C 변형 (`configs/mobile_vla_v5_exp41c_scratch_pta.json`)** — Exp25 config + preset, scratch 8 epoch:
```json
{
    "_comment": "Exp41C: Exp25 base + path_type_aware. grounding_aux 없음. scratch 8 epoch. 가장 정직한 prompt-only 비교.",
    "parent": "configs/mobile_vla_v5_exp25_step3_balanced_objective.json",
    "exp_name": "v5-exp41c-scratch-pta",
    "task_name": "mobile_vla_v5_exp41c",
    "trainer": {"max_epochs": 8, "check_val_every_n_epoch": 1},
    "vlm": {"pretrained_model_name_or_path": "/home/minum/26CS/MoNaVLA/.vlms/kosmos-2-patch14-224"},
    "tokenizer": {"pretrained_model_name_or_path": "/home/minum/26CS/MoNaVLA/.vlms/kosmos-2-patch14-224"},
    "train_dataset": {
        "data_dir": "/home/minum/minum/26CS/MoNa-pi/mobile_vla_dataset_v5",
        "instruction_preset": "path_type_aware"
    },
    "val_dataset": {
        "data_dir": "/home/minum/minum/26CS/MoNa-pi/mobile_vla_dataset_v5",
        "instruction_preset": "path_type_aware"
    }
}
```

### 3.2 measure_attention.py에 Exp41 등록

`scripts/measure_attention.py:69-112` MODELS dict에 추가:
```python
"exp41_path_type_aware": {
    "exp_dir": "runs/v5_nav/kosmos/mobile_vla_v5_exp41",
    "config": "configs/mobile_vla_v5_exp41_path_type_aware.json",
    "window_size": 8,
    "fwd_pred_next_n": 5,
    "num_classes": 8,
},
```

### 3.3 (선택) PM 평가 스크립트용 prompt sweep

`scripts/eval_prompt_sensitivity.py` 신규 (~80줄): 동일 이미지 20장 × 3 instruction (left/right/forward) → action 분포 비교. Exp25 vs Exp41 같이 돌려 차이 확인.

이건 Phase A 검증용. 본 plan 범위 안에 포함.

## 4. 수정 파일 요약 (구현 상태)

| 파일 | 변경 | 상태 |
|---|---|---|
| `configs/mobile_vla_v5_exp41b_resume_exp40_pta.json` | 신규 (15줄) | ✅ 완료 |
| `configs/mobile_vla_v5_exp41c_scratch_pta.json` | 신규 (24줄) | ✅ 완료 |
| `scripts/measure_attention.py` | MODELS dict에 14줄 추가 (exp41b, exp41c) | ✅ 완료 |
| `scripts/eval_prompt_sensitivity.py` | 신규 (~250줄) | ✅ 완료 |
| `runs/v5_nav/kosmos/mobile_vla_v5_exp41{b,c}/...` | 학습 결과 (자동 생성) | ⏳ 학습 대기 |
| `docs/v5/exp41_prompt_lockin/` | 결과 보고 | ⏳ 학습 후 |

기존 학습/추론 코드(`nav_trainer.py`, `inference_server.py`, `nav_h5_dataset_impl.py`) 수정 **없음**.

## 5. 실행 순서

```bash
# Step 1. 학습 (예상 30~40분, single GPU)
source .venv/bin/activate
python3 robovlm_nav/train.py configs/mobile_vla_v5_exp41_path_type_aware.json

# Step 2. attention 측정 (Exp25 vs Exp41)
python3 scripts/measure_attention.py
# → docs/v5/attention_analysis/summary.json + index.html
# → exp41이 left/right/forward 프롬프트로 다른 attention 분포 보이는지

# Step 3. PM 평가 (preset 일치)
python3 scripts/test_v5_pm_dm.py \
    --config configs/mobile_vla_v5_exp41_path_type_aware.json \
    --instruction_preset path_type_aware \
    --eval_split val --eval_t 0 \
    --output_json docs/v5/pm_eval/exp41_path_type_aware.json

# Step 4. prompt sensitivity (신규 스크립트)
python3 scripts/eval_prompt_sensitivity.py \
    --ckpt runs/v5_nav/kosmos/mobile_vla_v5_exp41/.../epoch_*.ckpt \
    --config configs/mobile_vla_v5_exp41_path_type_aware.json \
    --n-frames 20

# Step 5. 결과 확인 후 Phase B 진입 여부 결정
```

## 6. 검증 기준 (성공/실패)

### 성공 (→ Phase B 진입)
- [ ] text attention mean (24-layer 평균) ≥ 5% (현재 0.000%)
- [ ] same image + left/right/forward 프롬프트에 대한 action L1 차이 ≥ 1e-2 (현재 < 1e-3)
- [ ] PM ≥ 50% (Exp25 baseline 52%, 회귀 없음)
- [ ] confusion matrix가 단일 클래스로 collapse하지 않음

### 부분 성공 (→ Phase B 그대로 진행, but adjust)
- text attention 1~5% / action diff 5e-3~1e-2: prompt 신호 약함 → action_aware_train 더 강한 신호로 시도

### 실패 (→ Phase D 검토)
- text attention < 1% AND action diff < 1e-3: backbone이 text를 구조적으로 무시. LSTM decoder에 text token concat 등 구조 변경 필요.

## 7. 트레이드오프

- **장점**: 한 줄 변경, 5 epoch fine-tune. 비용 매우 낮음. 결과가 명확 (성공/실패 5분 안에 판단).
- **단점**: 
  - path_type_aware는 파일명 기반 — 실 운영에선 path_type 모름. Phase B에서 action_aware로 보강 필요.
  - resume이라 Exp25에 잠긴 attention pattern이 8 epoch fine-tune로 충분히 풀릴지 보장 없음. → 실패하면 scratch (옵션 C)로 재시도.
- **위험**: PATH_TYPE_INSTRUCTIONS 일부가 영문/한글 섞임 → 토크나이저가 한글을 unk로 처리하면 신호 약화. 측정 1회로 확인 가능 (`measure_attention.py`의 [debug] decoded 로그).

## 8. 일정

- Step 1 학습: 30~40분
- Step 2~4 평가: 20분
- Step 5 결과 정리/문서: 15분
- 총 ~1.5시간 → 같은 세션에서 Phase A 결정 가능

## 9. 사용자 결정 (2026-05-05 승인)

- **Q1: 옵션 B + C 병렬 실행** (A는 폐기)
  - **Exp41B** = Exp40 ckpt resume + path_type_aware (max_epochs 5)
  - **Exp41C** = scratch from Exp25 config + path_type_aware (max_epochs 8)
- **Q2**: B 5 / C 8 epochs
- **Q3**: optimizer state 기본값 (Trainer 기본 resume 동작 유지)
- **Q4**: 실패 시 결정 보류
- **Q5**: Exp40 액션 collapse는 B 결과 보고 분리 plan 결정

## 10. Phase B 미리보기 (이 plan 범위 밖)

Phase A 통과 시:
- Exp42: `instruction_preset: "action_aware_train"` (action 직접 라벨링)
- Exp43: 50% counterfactual prompt randomization (left 영상 + "go right" → right action 라벨로 합성 샘플)
- 추론 서버 prompt prefix를 학습 prefix와 일치시킴 (`<grounding>Instruction: {x}. Action:`)

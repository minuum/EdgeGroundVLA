# Plan: exp53 CL 평가 + 허브 CL 대시보드

**작성일**: 2026-05-27 (업데이트: 2026-06-03)  
**상태**: 승인 완료 (사용자 Option 1 선택)

---

## 배경

CL eval (`evaluate_closed_loop_v5.py`)은 pre-extracted `.npz` features를 사용함.  
exp53은 LoRA-enhanced vision_model이므로 새 npz 추출 필요.  
허브에 CL 결과 + 실로봇 주행 로그 저장 페이지 추가.

---

## 변경 범위

### 1. `scripts/extract_vis_features_exp53.py` (신규)

LoRA 적용 vision_model로 150 episodes 전체 features 추출 → npz 저장

```python
# 핵심 흐름
processor, model = load_kosmos2()                          # Pure HF Kosmos-2
model.vision_model = PeftModel.from_pretrained(            # LoRA 적용
    model.vision_model, ADAPTER_PATH
).eval()

# bbox_dataset_full.json의 150 에피소드 순회
# 각 프레임: vision_model(pixel_values) → mean pool → (1024,)
# 에피소드 단위 저장: vision_features.npz, vision_features_index.json

OUT_DIR = ROOT / "docs/v5/bbox_nav_exp53"
```

---

### 2. `scripts/sim/evaluate_closed_loop_v5.py` 수정

#### 2-a. exp53 등록

```python
GOAL_NAV_CKPTS["exp53"] = ROOT / "runs/v5_nav/mlp/exp53_clip_lora.pt"
GOAL_NAV_VIS_DIRS["exp53"] = ROOT / "docs/v5/bbox_nav_exp53"
GOAL_NAV_VIS_KEYS["exp53"] = "vision_features"
```

#### 2-b. `--model` choices에 exp53 추가

```python
ap.add_argument("--model", choices=[..., "exp53", ...])
```

#### 2-c. exp53 checkpoint format 처리

현재 코드:
```python
ckpt = torch.load(...)
d_in = ckpt["d_in"]                         # exp53엔 없음
mlp.load_state_dict(ckpt["model_state_dict"])  # exp53엔 없음
```

수정:
```python
if "mlp" in ckpt and "model_state_dict" not in ckpt:
    state = ckpt["mlp"]
    d_in = state["net.0.weight"].shape[1]
    mlp = nn.Sequential(...)
    # load full GoalNavMLP state
    tmp = {"net." + k if not k.startswith("net.") else k: v ... }
    # 아니면 그냥 net.* prefix 처리
else:
    d_in = ckpt["d_in"]
    mlp.load_state_dict(ckpt["model_state_dict"])
```

> 실제로는 GoalNavMLP 껍질 없이 Sequential만 load하므로 net.0.weight → 0.weight 변환 필요.

---

### 3. `scripts/gradio_cl_dashboard.py` (신규, port 7867)

세 탭 구성:

#### Tab 1: CL Results (결과 테이블)

- `rollout_metrics.json` 파싱 → 모델별 success_rate, FPE 표시
- per-path 상세 (path_type별 성공/실패)
- 새로고침 버튼

```
| 모델      | 성공률  | FPE   | 에피소드 수 |
|-----------|---------|-------|------------|
| exp49     | 96.7%   | 0.081 | 30         |
| exp51     | 96.7%   | 0.163 | 30         |
| exp54_s2v2| 96.7%   | 0.106 | 30         |
| exp53     | -       | -     | (미평가)    |
```

#### Tab 2: Run CL Eval (평가 실행)

- 모델 선택 드롭다운 (exp49/50/51/52/53)
- "Run" 버튼 → `evaluate_closed_loop_v5.py --model <선택>` 백그라운드 실행
- 실행 로그 스트리밍 textbox
- 완료 시 Tab 1 자동 갱신

#### Tab 3: Real Robot Log (실로봇 주행 기록)

로컬 JSON 파일(`docs/v5/real_robot_sessions.json`)에 저장:

```json
{
  "sessions": [
    {
      "date": "2026-05-27",
      "model": "exp49",
      "path_type": "center_straight",
      "success": true,
      "notes": "목표 바로 도달",
      "timestamp": "2026-05-27T14:30:00"
    }
  ]
}
```

UI:
- 모델 / path_type / 성공여부 / 메모 입력
- "기록 저장" 버튼
- 세션 히스토리 테이블 (최근 50건)

---

### 4. `scripts/gradio_hub.py` 수정

SERVICES에 CL Dashboard 추가:

```python
{
    "name":   "CL Dashboard",
    "port":   7867,
    "script": "scripts/gradio_cl_dashboard.py",
    "cmd":    "python3 scripts/gradio_cl_dashboard.py",
    "desc":   "Closed-Loop 평가 결과 + 실로봇 주행 로그",
    "group":  "Eval",
},
```

---

## 수정 파일 요약

| 파일 | 변경 | 비고 |
|------|------|------|
| `scripts/extract_vis_features_exp53.py` | 신규 | ~80줄 |
| `scripts/sim/evaluate_closed_loop_v5.py` | exp53 등록 + ckpt format 처리 | +20줄 |
| `scripts/gradio_cl_dashboard.py` | 신규 (3-tab Gradio) | ~250줄 |
| `scripts/gradio_hub.py` | SERVICES에 CL Dashboard 추가 | +8줄 |

---

## 실행 순서

1. feature 추출 (GPU, ~5분): `python3 scripts/extract_vis_features_exp53.py`
2. CL eval 실행: `python3 scripts/sim/evaluate_closed_loop_v5.py --model exp53`
3. 대시보드 실행: `python3 scripts/gradio_cl_dashboard.py`

---

## 완료 체크리스트

- [ ] extract_vis_features_exp53.py 작성
- [ ] evaluate_closed_loop_v5.py exp53 지원
- [ ] gradio_cl_dashboard.py 작성 (3탭)
- [ ] gradio_hub.py SERVICES 추가
- [ ] feature 추출 실행
- [ ] exp53 CL eval 실행

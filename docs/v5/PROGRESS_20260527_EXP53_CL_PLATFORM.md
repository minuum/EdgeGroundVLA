# 2026-05-27 진행 요약 — exp53 CL 평가 + CL 플랫폼 구축

## 오늘 한 것

### 1. exp53 CLIP-LoRA 추론 파이프라인

`robovlm_nav/serve/proxy_inference_server.py`에 exp53 지원 추가.

- `_GOAL_NAV_LORA_ADAPTERS["exp53"]` = `runs/v5_nav/mlp/clip_lora_adapter`
- PEFT 버전 충돌 해결: PEFT 0.19 저장 → 0.11 로드 시 `alora_invocation_tokens` unknown kwarg
  - `PeftModel.from_pretrained()` 대신 `get_peft_model()` + 수동 weight 로드
  - key remapping: `lora_A.weight` → `lora_A.default.weight`
- exp53 체크포인트 포맷: `{'mlp': state_dict, 'val_acc': float}` (d_in, model_state_dict 없음)
  - `state["net.0.weight"].shape[1]`로 d_in 자동 추론
- `scripts/gradio_inference_dashboard.py`에 GoalNav(Exp53, CLIP-LoRA) 모드 추가

### 2. exp53 CL 평가

`scripts/extract_vis_features_exp53.py`: LoRA-enhanced vision_model으로 150 ep 전체 feature 추출
- 출력: `docs/v5/bbox_nav_exp53/vision_features.npz` (6.3MB, 150 ep)
- 소요 시간: ~295s (GPU)

`scripts/sim/evaluate_closed_loop_v5.py` 수정:
- exp53 등록 (ckpt path, vis dir, vis key)
- H5 경로 fallback 추가: minum 절대경로(`/home/minum/...`) → soda 로컬 `DATA_DIR/<stem>.h5`

**결과 (30 episodes, test_size=0.2, random_state=42):**

| 모델 | 성공률 | FPE | TLD |
|------|--------|-----|-----|
| exp49 (baseline) | 96.7% | 0.08m | 1.00 |
| exp51 | 96.7% | 0.16m | 1.00 |
| exp54_s2v2 | 96.7% | 0.11m | 1.01 |
| **exp53 (CLIP-LoRA)** | **96.7%** | **0.13m** | **0.99** |
| exp52 | 93.3% | 0.13m | 1.00 |
| exp50 | 83.3% | 0.24m | 1.00 |

### 3. CL 플랫폼 구축

**`scripts/gradio_cl_dashboard.py`** (port 7867, 신규):
- Tab 1 "CL Results": rollout_metrics.json 파싱 → 모델별 성공률/FPE 컬러 테이블 + per-path 상세
- Tab 2 "Run CL Eval": 모델 선택 → evaluate_closed_loop_v5.py 실행 + 로그 스트리밍
- Tab 3 "Real Robot Log": 실로봇 주행 세션 기록 (docs/v5/real_robot_sessions.json)

**`scripts/gradio_hub.py`** 개선:
- Tailscale IP 자동 감지 (`tailscale0` NIC 우선, fallback UDP trick)
- 오프라인 서비스 카드 인라인 ▶ Start / ■ Stop 버튼
- Start 시 실시간 스피너 스트리밍 (최대 30s 대기, 포트 up 감지 시 즉시 완료)
- CL Dashboard 서비스 추가 (port 7867, group: Eval)

---

## 오프라인 CL 평가의 의미와 한계

### 무엇을 측정하는가

StratifiedShuffleSplit(test_size=0.2, random_state=42)로 나눈 hold-out 30 ep에서,
모델 예측 액션을 순서대로 쌓아 시뮬레이션 궤적을 만들고 전문가 궤적과 비교.

- **성공 기준**: FPE < 0.5m AND 0.7 ≤ TLD ≤ 1.5
- 9가지 경로 타입 균등 포함 (path_type당 3~4 ep)

### 오프라인이 가정하는 것 (실제론 틀림)

```
오프라인: 매 스텝 ground-truth 프레임 입력 → 오류 없이 시야 유지
실로봇:  t=0 예측 오류 → t=1 시야 달라짐 → compounding error
```

### 추가 변수 (실로봇에만 존재)

| 변수 | 오프라인 | 실로봇 |
|------|----------|--------|
| Vision feature | npz에서 읽음 | Kosmos-2 실시간 추론 (~300ms 루프) |
| BBox 검출 | 데이터 완벽 태깅 | 실패 시 cx=-1 처리 |
| Goal 초기화 | 에피소드 첫 프레임 정답 | 출발 시 grounding 실패하면 오염 |
| 액션 실행 | 이상적 물리 | 바퀴 슬립, 바닥 마찰 |
| 경로 반복성 | 고정 (동일 H5) | 매번 다름 |

### 해석

> **오프라인 CL = "이 모델이 망하지 않았음"을 검증하는 필터**
>
> 실로봇 세션 기록이 "실제로 쓸 수 있는가"를 판단하는 진짜 지표

exp53 96.7%는 exp49와 동등 수준임을 확인.
FPE는 exp49(0.08m) < exp53(0.13m)로 exp49가 더 정밀.
실로봇에서 어느 쪽이 더 강건한지는 동일 경로 반복 테스트로 비교 필요.

---

## 다음 할 일

1. **exp53 실로봇 테스트** — hub에서 GoalNav(Exp53) 모드로 실제 주행, Real Robot Log에 기록
2. **exp49 vs exp53 비교** — 동일 경로 각 3회 이상 반복, off-track 회복 여부 관찰
3. **grounding 안정성 모니터링** — 주행 중 bbox 검출 실패율 측정

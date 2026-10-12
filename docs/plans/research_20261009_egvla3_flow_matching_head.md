# 리서치 — EGVLA3 Flow-Matching 액션 헤드

## 기존 자산

### 1) `scripts/pilot_flow_matching_head.py` (이미 존재, 미사용 상태)
- `FlowMatchingActionHead(frame_dim, window, action_dim=3, hidden=256)`:
  조건부 flow matching 표준 구조. `cond_proj`(윈도 flatten 1560dim→256dim) +
  `time_proj`(스칼라 t→32dim) + `head`(cond+t+x_t → velocity, 3dim).
- `flow_matching_loss()`: `x0~N(0,I)`, `t~U[0,1]`, `x_t=(1-t)x0+t·x1`,
  타겟 `v=x1-x0`, MSE(pred_v, target_v). 표준 conditional FM loss 그대로.
- 지금까지는 **synthetic 데이터로 forward/backward sanity check만** 했고
  (`load_synthetic_data()`), 실데이터 학습·평가는 한 번도 안 돌아봤음.
  스크립트 자체 주석에 "실제 학습 결과 수치는 아직 없음 — 지어내지 말 것" 명시.

### 2) `train_exp73_stage1v3_heads.py`의 `ContRegHead` + `train_contreg()`
- 방금 전(`eval_discrete_vs_contreg_leave_one_direction_out.py`)에 썼던 것.
- Flow matching이 아니라 **직접 회귀**(transformer encoder → tanh(3dim), MSE
  직접 최소화) — 자기 docstring에도 "MoNa-pi flow 계열의 **최소 버전**"이라고
  명시돼 있어, 진짜 FM이 아님을 이미 인지하고 있었음.
- 방금 돌린 leave-one-direction-out 결과: center 방향에서 이산(MLP) 61.5% vs
  연속회귀(ContReg) 51.7% — 연속 쪽이 뚜렷이 열세 (나머지 4방향 실행 중).

### 3) CH67 기존 기록 (research_story.html)
- "flow 72.0% vs mlp discrete 77.9%"(225ep, 무작위 split) — 이 수치가
  `FlowMatchingActionHead`로 낸 것인지 `ContRegHead`류로 낸 것인지 스크립트
  상으로는 **추적 불가**(pilot 스크립트는 synthetic만 돌았다고 자체 명시하므로
  이 72.0%는 pilot이 아닌 별도의, 지금 레포에 없는 학습 코드에서 나온 값으로
  추정 — 확인 불가).

## 데이터 제약 (중요)

- `exp73_v6_vis_cache_stage1v3.pt`의 `acts`는 실제 아날로그 조이스틱 값이 아니라
  **8-class LUT 고정값**(예: `(0.0,-1.15,0.0)` 등 정확히 8종 조합만 존재).
  pilot 스크립트 주석: "진짜 아날로그 연속 액션은 soda가 dual/continuous
  모드로 재수집해야 얻을 수 있다."
- 즉 지금 가용한 데이터로 FM을 학습해도 **타겟 자체가 8개 점으로 이산화된
  분포**이므로, "진짜 연속 액션의 이점"을 검증하는 게 아니라 "같은 이산
  타겟을 다른 손실함수(FM velocity matching)로 맞췄을 때 분류 정확도가
  어떻게 나오는가"를 보는 것 — ContReg 실험과 근본적으로 같은 한계.
- 이 한계는 숨기지 않고 보고서/페이지에 명시해야 함 (프로젝트 규칙:
  "예측과 잘 맞으면 더 의심", "결과 지어내지 말 것").

## 추론(샘플링) 쪽 — 아직 구현된 적 없는 부분

- FM은 학습 손실만 있고, 평가하려면 `x0~N(0,I)`에서 출발해 ODE를 풀어
  `x1`(연속 액션 예측)을 얻는 **샘플링 루틴**이 따로 필요함 (pilot 스크립트엔
  없음). 가장 단순한 Euler 적분:
  ```python
  @torch.no_grad()
  def sample(model, frame_feat, n_steps=50):
      x = torch.randn(frame_feat.size(0), ACTION_DIM, device=frame_feat.device)
      dt = 1.0 / n_steps
      for i in range(n_steps):
          t = torch.full((frame_feat.size(0),), i * dt, device=frame_feat.device)
          x = x + model(frame_feat, x, t) * dt
      return x
  ```
- 샘플링 후 `cont_to_class_t()`(기존 `train_exp73_stage1v3_heads.py`에 이미
  구현됨, ContReg 평가에서 재사용 중)로 8-class 변환 → 이산/ContReg와
  apples-to-apples 비교 가능.

## 결론

- 아키텍처는 이미 있음(그대로 재사용 가능), **학습 루프·leave-one-direction-out
  평가·샘플링 루틴은 전부 새로 작성 필요** — "구현력이 더 필요한 부분"에 해당.
- 기존 ContReg(직접 회귀)와 이번 FM(velocity matching + ODE 샘플링)을
  **같은 프로토콜·같은 데이터·같은 5방향 held-out**으로 나란히 비교하면,
  "회귀 손실 자체보다 FM 정식 공식이 갖는 이점이 있는가"를 분리해서 볼 수 있음.

# 계획 — EGVLA3용 Flow-Matching 액션 헤드 실학습 + leave-one-direction-out 평가

리서치: `research_20261009_egvla3_flow_matching_head.md`

`pilot_flow_matching_head.py`의 `FlowMatchingActionHead`/`flow_matching_loss`는
아키텍처만 있고 synthetic sanity-check 외엔 한 번도 실데이터로 돌아간 적이
없다. 이번에 그걸 실제로 학습시키고, 지금까지와 동일한 leave-one-direction-out
프로토콜(exp73 캐시, 5방향 held-out, seed 0/1/2)로 이산(MLP)·연속회귀(ContReg)와
나란히 비교한다. **데이터 자체는 여전히 8-class LUT 고정값**이라는 한계는
결과 보고 시 반드시 명시한다(지어내지 않음).

## 접근 방식

새 스크립트 `scripts/eval_flow_matching_leave_one_direction_out.py` 작성.
기존 `eval_discrete_vs_contreg_leave_one_direction_out.py`와 같은 골격
(동일 캐시 로드, 동일 `get_direction`/`DIRECTIONS`, 동일 `build_windows`)을
재사용하고, 세 헤드(이산 MLP / ContReg / FlowMatching)를 한 번에 비교한다.

### 1) 학습 루프 (`train_flow_matching`)

```python
from scripts.pilot_flow_matching_head import FlowMatchingActionHead, flow_matching_loss

def train_flow_matching(X_tr, A_tr, X_va, y_va, A_va, seed, epochs=300, lr=1e-3, n_sample_steps=50):
    """A_tr/A_va: raw (lx,ly,az) 액션, scale은 ContRegHead.ACTION_SCALE(1.15)과 동일하게 맞춤."""
    torch.manual_seed(seed); np.random.seed(seed)
    scale = 1.15
    X_tr_t = torch.tensor(X_tr, device=DEVICE)                 # (N, window, frame_dim)
    A_tr_t = torch.tensor(A_tr / scale, device=DEVICE)         # (N, 3) in [-1,1]
    X_va_t = torch.tensor(X_va, device=DEVICE)
    y_va_t = torch.tensor(y_va, device=DEVICE)

    model = FlowMatchingActionHead(frame_dim=X_tr.shape[2], window=X_tr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    best_acc, best_state = 0.0, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr_t), device=DEVICE)
        for i in range(0, len(perm), 128):
            b = perm[i:i+128]
            opt.zero_grad()
            loss = flow_matching_loss(model, X_tr_t[b], A_tr_t[b])
            loss.backward(); opt.step()
        sched.step()
        if ep % 25 == 0 or ep == epochs - 1:
            model.eval()
            pred = sample(model, X_va_t, n_sample_steps) * scale        # (N_va, 3)
            acc = (cont_to_class_t(pred) == y_va_t).float().mean().item()
            if acc >= best_acc:
                best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    pred = cont_to_class_t(sample(model, X_va_t, n_sample_steps) * scale).cpu().numpy()
    per_class = {c: float((pred[y_va == c] == c).mean()) for c in range(NUM_CLASSES) if (y_va == c).sum() > 0}
    return best_acc, best_state, per_class
```

### 2) 샘플링 (ODE 적분, 평가 시에만 사용)

```python
@torch.no_grad()
def sample(model, frame_feat, n_steps=50):
    x = torch.randn(frame_feat.size(0), 3, device=frame_feat.device)
    dt = 1.0 / n_steps
    for i in range(n_steps):
        t = torch.full((frame_feat.size(0),), i * dt, device=frame_feat.device)
        x = x + model(frame_feat, x, t) * dt
    return x
```

`cont_to_class_t`는 `train_exp73_stage1v3_heads.py`에 이미 있는 것 그대로 import.

### 3) leave-one-direction-out 루프

`eval_discrete_vs_contreg_leave_one_direction_out.py`의 `main()` 구조를 그대로
복제하되, 각 held-out 방향·seed마다 이산(`train_one`)/ContReg(`train_contreg`,
이미 결과 있으면 재사용)/FlowMatching(`train_flow_matching`, 신규) 셋을 같이
학습·기록한다. 출력: `docs/v5/closed_loop_eval/flow_matching_leave_one_direction_out.json`.

### 4) 알려진 리스크 / 트레이드오프

- ODE 샘플링을 평가 때마다(25 epoch마다) 돌리면 이산/ContReg 대비 느림 —
  `n_sample_steps=50`은 품질·속도 타협값(필요시 20으로 낮춰 속도 우선 가능).
  **체크포인트 25epoch마다 추가 샘플링 비용은 train_contreg와 동일 패턴**이라
  실험 전체 시간은 조금 전 ContReg 실험(아직도 돌고 있음, 5방향에 오래 걸림)과
  비슷하거나 그보다 다소 길 것으로 예상.
- 타겟이 8-class LUT 고정값이라는 근본 한계는 그대로 — "FM 공식 자체의 이점"과
  "진짜 연속 데이터의 이점"을 분리해서 결론에 명시.

## 수정될 파일

- 신규: `scripts/eval_flow_matching_leave_one_direction_out.py`
- 읽기만: `scripts/pilot_flow_matching_head.py`, `scripts/train_exp73_stage1v3_heads.py`(import만, 수정 없음)
- 결과 출력: `docs/v5/closed_loop_eval/flow_matching_leave_one_direction_out.json`
- 완료 후: `docs/v5/project_roadmap_2026q4.html`의 EGVLA3 트랙/git-graph에 3자 비교(이산/ContReg/FM) 결과 반영

## 실행 순서

1. 지금 돌고 있는 `discrete_vs_contreg` LOO 작업이 끝날 때까지 대기(같은 GPU 점유)
2. 새 스크립트 작성 후 백그라운드 실행
3. 결과를 이산/ContReg와 한 표로 비교해 보고

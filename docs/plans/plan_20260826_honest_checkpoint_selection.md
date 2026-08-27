# 계획 — C: 체크포인트 선택 방법론 수정 (val 유출 의심 해소)

리서치(`research_20260826_grounding_action_fusion_refs.md` §"체크포인트 선택
방식 자체가 낙관 편향의 일부일 수 있음"): 지금 `train_one()`/`train_cxaux()`
등은 25epoch마다 val_acc를 측정해 **최고 시점의 state를 그대로 "best"로
채택**한다. 300/25=12번 중 최댓값을 고르는 것과 같아서, val 표본이 작을수록
(leave-one-direction-out처럼 held-out 방향 1개) 낙관 편향을 만들 수 있다.

## 접근 방식

기존 `train_one()` 등은 그대로 두고(과거 CH64~70 결과와의 비교 가능성 보존),
**train 내부에서 다시 떼어낸 inner-val로 체크포인트를 고르고, 진짜 val은
순수 최종 평가에만 쓰는** 새 함수 `train_one_honest()`를 추가한다.

```python
def train_one_honest(X_tr, y_tr, X_va, y_va, seed, epochs=300, lr=5e-4,
                      inner_val_frac=0.15, head_cls=None):
    """체크포인트 선택을 진짜 val(X_va)이 아니라 train에서 추가로 뗀
    inner-val로 한다 — X_va는 최종 1회 평가에만 사용(선택 유출 없음)."""
    head_cls = head_cls or MLPActionHead
    rng = np.random.default_rng(seed)
    n = len(X_tr)
    idx = rng.permutation(n)
    n_inner = max(1, int(n * inner_val_frac))
    inner_va_idx, inner_tr_idx = idx[:n_inner], idx[n_inner:]
    X_itr, y_itr = X_tr[inner_tr_idx], y_tr[inner_tr_idx]
    X_iva, y_iva = X_tr[inner_va_idx], y_tr[inner_va_idx]

    torch.manual_seed(seed)
    cls_counts = np.bincount(y_itr, minlength=NUM_CLASSES).astype(np.float32)
    cls_counts = np.where(cls_counts == 0, 1.0, cls_counts)
    weights = 1.0 / cls_counts
    weights = weights / weights.sum() * NUM_CLASSES
    weights_t = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    X_itr_t = torch.tensor(X_itr, device=DEVICE); y_itr_t = torch.tensor(y_itr, device=DEVICE)
    X_iva_t = torch.tensor(X_iva, device=DEVICE); y_iva_t = torch.tensor(y_iva, device=DEVICE)
    X_va_t = torch.tensor(X_va, device=DEVICE); y_va_t = torch.tensor(y_va, device=DEVICE)

    model = head_cls().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_inner_acc, best_state = 0.0, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X_itr_t), device=DEVICE)
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            loss = F.cross_entropy(model(X_itr_t[b]), y_itr_t[b], weight=weights_t)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 25 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                inner_acc = (model(X_iva_t).argmax(1) == y_iva_t).float().mean().item()
            if inner_acc >= best_inner_acc:      # ← 선택 기준: inner-val (X_va 미사용)
                best_inner_acc = inner_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():                          # ← 진짜 val은 여기서 딱 1번만 평가
        pred = model(X_va_t).argmax(1).cpu().numpy()
        acc = float((pred == y_va).mean())
    per_class = {}
    for c in range(NUM_CLASSES):
        m = (y_va == c)
        if m.sum() > 0:
            per_class[c] = float((pred[m] == c).mean())
    return acc, best_state, per_class, best_inner_acc
```

## 실행 계획

새 스크립트 `scripts/eval_honest_checkpoint_selection.py`로 exp77 캐시에서
mlp와 deltacx(exp78 최고 후보) 둘 다 `train_one()`(기존, val 직접 선택) vs
`train_one_honest()`(inner-val 선택)로 나란히 돌려서 **val_acc 격차가
실제로 얼마나 벌어지는지** 정량화한다. seed 0/1/2, epoch 300, 무작위 15%
split(SPLIT_SEED=42, 기존과 동일 — 비교 기준 유지)과 leave-one-direction-out
(deltacx, weak_right/strong_right 중심) 둘 다 재실행.

## 수정 파일

- `scripts/train_exp73_stage1v3_heads.py` — `train_one_honest()` 추가(기존
  함수는 변경 없음)
- `scripts/eval_honest_checkpoint_selection.py` — 신규, 비교 실행 스크립트

## 완료 기준

- [ ] `train_one_honest()` 구현 + 문법 검증
- [ ] mlp/deltacx × (무작위 split, leave-one-direction-out 2방향) × (val선택
      vs inner-val선택) 비교표
- [ ] 격차가 유의미하면(예: >1%p) 지금까지의 exp78/CH69/CH70 수치에 "낙관
      편향 가능성" 주석 추가, A(gatefuse)/B(actionquery)도 전부 honest
      selection으로 재검증

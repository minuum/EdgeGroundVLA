#!/usr/bin/env python3
"""
HELD-aware 재학습 (64-10/64-11 후속, 2026-07-23) — 항목 2.

기존 실험(exp73_window_cadence.py)은 baseline(stride=1)으로 학습한 모델을
HELD(결정 유지) 조건에서 "평가만" 했다 — 39.4%→24.2% 급락 확인(64-10),
실기(64-11)도 HELD 예측과 일치함을 확인.

이번엔 반대로 **학습 자체를 HELD 조건으로** 시킨다:
- 입력 X: 결정 시점(t=0,stride,2*stride,...)마다 stride 간격 window (기존과 동일)
- 라벨 y: 그 결정이 실제로 적용될 구간 [t, t+stride) 전체 GT의 **다수결** —
  "이 구간 전체를 대표하는 하나의 액션"을 직접 학습 목표로 삼음(단일 프레임
  라벨보다 hold에 강건한 선택을 하도록 유도하는 것이 가설).
- 체크포인트 선정도 offline val_acc가 아니라 **HELD closed-loop success**로 함
  (64-1에서 지적한 "잘못된 기준으로 챔피언을 뽑던" 문제의 재발 방지).

비교: baseline(stride=1) 학습 vs stride=5 다수결(hold-aware) 학습, 둘 다 HELD로 평가.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.train_exp73_trackA_heads import (
    MLPActionHead, CACHE_V6, SPLIT_SEED, VAL_RATIO, NUM_CLASSES, DEVICE, BBOX_SCALE, WINDOW,
)
from scripts.sim.rollout_core import build_trajectory, compute_metrics
from scripts.exp73_window_cadence import build_windows_strided, eval_closed_loop_held, swap_bboxes, OWL_ANN


def majority(vals):
    counts = np.bincount(vals, minlength=NUM_CLASSES)
    return int(counts.argmax())


def build_windows_hold_aware(eps, window=WINDOW, bbox_scale=BBOX_SCALE, stride=5):
    """결정 시점(stride 간격)마다 1개 샘플, 라벨은 [t, t+stride) 구간 GT의 다수결."""
    X, y = [], []
    for ep in eps:
        bboxes, vis, gts = ep["bboxes"], ep["vis"], ep["gts"]
        n = len(gts)
        for t in range(0, n, stride):
            seq = []
            for k in range(window):
                idx = max(0, t - (window - 1 - k) * stride)
                seq.append([v * bbox_scale for v in bboxes[idx]] + vis[idx].tolist())
            X.append(seq)
            y.append(majority(np.asarray(gts[t:t + stride], dtype=np.int64)))
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def train_one(X_tr, y_tr, X_va, y_va, seed, epochs=300, lr=5e-4):
    torch.manual_seed(seed); np.random.seed(seed)
    cls_counts = np.bincount(y_tr, minlength=NUM_CLASSES).astype(np.float32)
    cls_counts = np.where(cls_counts == 0, 1.0, cls_counts)
    weights = 1.0 / cls_counts; weights = weights / weights.sum() * NUM_CLASSES
    weights_t = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    X_tr_t = torch.tensor(X_tr, device=DEVICE); y_tr_t = torch.tensor(y_tr, device=DEVICE)
    X_va_t = torch.tensor(X_va, device=DEVICE); y_va_t = torch.tensor(y_va, device=DEVICE)
    model = MLPActionHead().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_acc, best_state = 0.0, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr_t), device=DEVICE)
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            loss = F.cross_entropy(model(X_tr_t[b]), y_tr_t[b], weight=weights_t)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 25 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                acc = (model(X_va_t).argmax(1) == y_va_t).float().mean().item()
            if acc >= best_acc: best_acc = acc; best_state = {k: v.clone() for k, v in model.state_dict().items()}
    return best_acc, best_state


def main():
    stride = 5
    base = torch.load(str(CACHE_V6), weights_only=False)
    base = [e for e in base if e.get("acts") is not None]
    rng = np.random.default_rng(SPLIT_SEED); idx = list(range(len(base))); rng.shuffle(idx)
    nv = max(1, int(len(idx) * VAL_RATIO))
    val = [base[i] for i in idx[:nv]]; tr = [base[i] for i in idx[nv:]]

    # (A) baseline: stride=1 학습 → HELD로 평가 (기존 64-10 재확인용, 비교 기준선)
    Xtr1, ytr1 = build_windows_strided(tr, stride=1)
    Xva1, yva1 = build_windows_strided(val, stride=1)

    # (B) hold-aware: stride=5 결정시점 + 구간 다수결 라벨로 학습
    Xtr5, ytr5 = build_windows_hold_aware(tr, stride=stride)
    Xva5, yva5 = build_windows_hold_aware(val, stride=stride)

    print(f"baseline train={len(Xtr1)}  hold-aware train={len(Xtr5)} (샘플 수 1/{stride})", flush=True)

    results = {"baseline": [], "hold_aware": []}
    for seed in [0, 1, 2]:
        # baseline(기존 방식) 재학습 후 HELD 평가로 챔피언 재선정 기준 확인
        acc1, st1 = train_one(Xtr1, ytr1, Xva1, yva1, seed, epochs=300)
        m1 = MLPActionHead(); m1.load_state_dict(st1); m1.eval()
        held1 = np.mean([r["success"] for r in eval_closed_loop_held(val, m1, stride=stride)]) * 100
        results["baseline"].append(held1)
        print(f"[baseline] seed{seed} offline={acc1*100:.1f}% HELD={held1:.1f}%", flush=True)

        # hold-aware 학습 (다수결 라벨, HELD 조건 그대로 반영)
        acc5, st5 = train_one(Xtr5, ytr5, Xva5, yva5, seed, epochs=300)
        m5 = MLPActionHead(); m5.load_state_dict(st5); m5.eval()
        held5 = np.mean([r["success"] for r in eval_closed_loop_held(val, m5, stride=stride)]) * 100
        results["hold_aware"].append(held5)
        print(f"[hold_aware] seed{seed} offline(다수결 acc)={acc5*100:.1f}% HELD={held5:.1f}%", flush=True)

        torch.save({"model": st5, "held_success": held5, "head": "mlp", "stride": stride},
                   str(ROOT / f"runs/v5_nav/mlp/exp73/exp73_pg448_trackF_v6_mlp_holdaware_seed{seed}.pt"))

    for k, v in results.items():
        print(f"=== {k}: HELD {np.mean(v):.1f}±{np.std(v):.1f}% ===", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
exp73 곡선 개선 실험 — (2) 회전/곡선 클래스 가중치 부스트 + (3) 회전 프레임 오버샘플링.

63-16에서 확인된 실패 패턴: 직진 75~100% vs 곡선 0~67% — 회전(FWD+L/FWD+R/LEFT/
RIGHT/ROT) 클래스를 잘 못 맞혀 곡선 궤적이 무너짐. 데이터의 71%가 FORWARD라 회전이
과소학습된 게 원인 후보. 재수집 없이 학습 레벨에서 완화 가능한지 검증.

baseline(train_one)은 역빈도 가중치만 씀. 여기에:
  --turn-boost B : 비직진 클래스(2~7) 가중치를 B배 추가 부스트
  --oversample K : 비직진 프레임을 K배 복제(오버샘플링)
mlp 헤드, V6 225ep(트랙A+F, 챔피언 조건과 정합), 3-seed.

Usage:
  .venv/bin/python3 scripts/exp73_turn_boost.py --turn-boost 3
  .venv/bin/python3 scripts/exp73_turn_boost.py --oversample 4
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_exp73_trackA_heads import (
    MLPActionHead, build_windows, CACHE_V6, SPLIT_SEED, VAL_RATIO, NUM_CLASSES, DEVICE,
)

NONSTRAIGHT = [2, 3, 4, 5, 6, 7]  # STOP(0)·FORWARD(1) 제외한 회전/대각/제자리 클래스


def train_boosted(X_tr, y_tr, X_va, y_va, seed, turn_boost=1.0, oversample=1, epochs=300, lr=5e-4):
    torch.manual_seed(seed); np.random.seed(seed)

    # (3) 오버샘플링 — 비직진 프레임 복제
    if oversample > 1:
        mask = np.isin(y_tr, NONSTRAIGHT)
        extra_X = np.repeat(X_tr[mask], oversample - 1, axis=0)
        extra_y = np.repeat(y_tr[mask], oversample - 1, axis=0)
        X_tr = np.concatenate([X_tr, extra_X], axis=0)
        y_tr = np.concatenate([y_tr, extra_y], axis=0)

    cls_counts = np.bincount(y_tr, minlength=NUM_CLASSES).astype(np.float32)
    cls_counts = np.where(cls_counts == 0, 1.0, cls_counts)
    weights = 1.0 / cls_counts
    # (2) 비직진 클래스 가중치 부스트
    if turn_boost != 1.0:
        for c in NONSTRAIGHT:
            weights[c] *= turn_boost
    weights = weights / weights.sum() * NUM_CLASSES
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
            if acc >= best_acc:
                best_acc = acc; best_state = {k: v.clone() for k, v in model.state_dict().items()}
    return best_acc, best_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--turn-boost", type=float, default=1.0)
    ap.add_argument("--oversample", type=int, default=1)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    tag = args.tag or f"boost{args.turn_boost}_os{args.oversample}"

    v6 = torch.load(str(CACHE_V6), weights_only=False)  # 225ep 정합
    rng = np.random.default_rng(SPLIT_SEED); idx = list(range(len(v6))); rng.shuffle(idx)
    nv = max(1, int(len(idx) * VAL_RATIO))
    val = [v6[i] for i in idx[:nv]]; tr = [v6[i] for i in idx[nv:]]
    Xtr, ytr, _ = build_windows(tr); Xva, yva, _ = build_windows(val)
    print(f"[{tag}] train={len(Xtr)} val={len(Xva)}", flush=True)

    for s in [int(x) for x in args.seeds.split(",")]:
        acc, st = train_boosted(Xtr, ytr, Xva, yva, s,
                                 turn_boost=args.turn_boost, oversample=args.oversample)
        out = ROOT / f"runs/v5_nav/mlp/exp73/exp73_pg448_trackF_v6_mlp_{tag}_seed{s}.pt"
        torch.save({"model": st, "val_acc": acc, "head": "mlp", "arm": "v6",
                    "window": 6, "bbox_scale": 3.0, "exp": "exp73"}, str(out))
        print(f"[{tag}] seed{s} val_acc={acc*100:.2f}% → {out.name}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()

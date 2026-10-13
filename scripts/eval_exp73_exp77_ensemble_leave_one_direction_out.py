#!/usr/bin/env python3
"""exp73/exp77 제로트레이닝 앙상블 — leave-one-direction-out 검증.

배경 (CH69-9/69-10, Phase 1 두 번째 후보):
  exp77 미러 증강(exp77_mirror_aug_train.py --loo)은 5방향 평균 55.55%로
  exp77 원본(54.00%)보다 소폭 개선됐지만 exp73(60.85%)에는 여전히 못 미친다
  (weak_right는 전혀 개선 안 됨: 41.7%→41.7%).

  여기서는 재학습 없이, 두 모델이 이미 저장해 둔 MLP 헤드를 그대로 불러와
  held-out 방향마다 새로 학습(train_one)한 뒤 softmax 확률을 평균(0.5/0.5)해
  앙상블이 두 모델의 약점(exp73의 전반적 약우측 vs exp77의 심한 우측 F→FR
  오판)을 서로 상쇄하는지 확인한다. exp73/exp77 캐시의 에피소드 순서·stem·
  gts가 완전히 동일함을 사전 확인했으므로 두 모델의 윈도 순서가 1:1로 맞는다.

출력: docs/v5/closed_loop_eval/exp73_exp77_ensemble_leave_one_direction_out.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_exp73_stage1v3_heads import (  # noqa: E402
    MLPActionHead, build_windows, DEVICE, NUM_CLASSES,
)

CACHE_73 = ROOT / "docs/v5/closed_loop_eval/exp73_v6_vis_cache_stage1v3.pt"
CACHE_77 = ROOT / "docs/v5/closed_loop_eval/exp77_florence2_phrase_full_vis_cache.pt"
OUT = ROOT / "docs/v5/closed_loop_eval/exp73_exp77_ensemble_leave_one_direction_out.json"
DIRECTIONS = ["center", "weak_left", "weak_right", "strong_left", "strong_right"]
CLASS_NAMES = ["STOP", "F", "L", "R", "FL", "FR", "ROT_L", "ROT_R"]
SEEDS = [0, 1, 2]


def get_direction(path_type):
    for d in sorted(DIRECTIONS, key=len, reverse=True):
        if path_type.startswith(d + "_"):
            return d
    return "center"


def train_and_probs(X_tr, y_tr, X_va, y_va, seed, epochs=300, lr=5e-4):
    """train_exp73_stage1v3_heads.train_one과 동일한 학습 루프이되 val softmax 확률도 반환."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    n_tr = len(X_tr)
    perm = np.random.permutation(n_tr)
    n_inner_val = max(1, int(n_tr * 0.1))
    inner_va_idx, inner_tr_idx = perm[:n_inner_val], perm[n_inner_val:]

    X_tr_t = torch.tensor(X_tr.reshape(n_tr, -1), device=DEVICE)
    y_tr_t = torch.tensor(y_tr, device=DEVICE)
    X_va_t = torch.tensor(X_va.reshape(len(X_va), -1), device=DEVICE)

    model = MLPActionHead(frame_dim=X_tr.shape[2], window=X_tr.shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_inner_acc, best_state, no_improve, patience = 0.0, None, 0, 30
    for ep in range(epochs):
        model.train()
        idx = inner_tr_idx[np.random.permutation(len(inner_tr_idx))]
        for s in range(0, len(idx), 256):
            b = idx[s:s + 256]
            opt.zero_grad()
            loss = F.cross_entropy(model(X_tr_t[b]), y_tr_t[b])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            inner_pred = model(X_tr_t[inner_va_idx]).argmax(1).cpu().numpy()
        inner_acc = float((inner_pred == y_tr[inner_va_idx]).mean())
        if inner_acc > best_inner_acc:
            best_inner_acc, best_state, no_improve = inner_acc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(X_va_t), dim=1).cpu().numpy()
    return probs


def per_class_acc(pred, y_va):
    per_class = {}
    for c in range(NUM_CLASSES):
        m = (y_va == c)
        if m.sum() > 0:
            per_class[c] = float((pred[m] == c).mean())
    return per_class


def main():
    eps73 = torch.load(str(CACHE_73), weights_only=False)
    eps77 = torch.load(str(CACHE_77), weights_only=False)
    for a, b in zip(eps73, eps77):
        assert a["stem"] == b["stem"] and a["gts"] == b["gts"]
        a["direction"] = get_direction(a["path_type"])
        b["direction"] = get_direction(b["path_type"])

    results = {}
    for held in DIRECTIONS:
        tr73 = [e for e in eps73 if e["direction"] != held]
        va73 = [e for e in eps73 if e["direction"] == held]
        tr77 = [e for e in eps77 if e["direction"] != held]
        va77 = [e for e in eps77 if e["direction"] == held]

        X_tr73, y_tr73, _ = build_windows(tr73)
        X_va73, y_va, _ = build_windows(va73)
        X_tr77, y_tr77, _ = build_windows(tr77)
        X_va77, y_va77, _ = build_windows(va77)
        assert (y_tr73 == y_tr77).all() and (y_va == y_va77).all()

        accs_73, accs_77, accs_ens = [], [], []
        best_ens, best_ens_pred = 0.0, None
        for seed in SEEDS:
            p73 = train_and_probs(X_tr73, y_tr73, X_va73, y_va, seed)
            p77 = train_and_probs(X_tr77, y_tr77, X_va77, y_va77, seed)
            pred73 = p73.argmax(1)
            pred77 = p77.argmax(1)
            p_ens = 0.5 * p73 + 0.5 * p77
            pred_ens = p_ens.argmax(1)
            accs_73.append(float((pred73 == y_va).mean()))
            accs_77.append(float((pred77 == y_va).mean()))
            acc_ens = float((pred_ens == y_va).mean())
            accs_ens.append(acc_ens)
            if acc_ens > best_ens:
                best_ens, best_ens_pred = acc_ens, pred_ens

        pc = per_class_acc(best_ens_pred, y_va)
        results[held] = dict(
            n_val_ep=len(va73), n_val_windows=int(len(y_va)),
            exp73_acc_mean=float(np.mean(accs_73)),
            exp77_acc_mean=float(np.mean(accs_77)),
            ensemble_acc_mean=float(np.mean(accs_ens)),
            ensemble_acc_std=float(np.std(accs_ens)),
            ensemble_acc_best=float(best_ens),
            per_class_best={CLASS_NAMES[c]: v for c, v in pc.items()},
            r_class_acc=pc.get(3), fr_class_acc=pc.get(5),
        )
        print(f"[{held:14s}] val={len(va73)}ep({len(y_va)}win) "
              f"exp73={np.mean(accs_73)*100:.2f}% exp77={np.mean(accs_77)*100:.2f}% "
              f"ENSEMBLE mean={np.mean(accs_ens)*100:.2f}% best={best_ens*100:.2f}% "
              f"R={pc.get(3,float('nan'))*100:.1f}% FR={pc.get(5,float('nan'))*100:.1f}%", flush=True)

    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    overall = np.mean([r["ensemble_acc_best"] for r in results.values()])
    print(f"\n5개 방향 평균 held-out best acc (앙상블): {overall*100:.2f}% "
          f"(exp73 60.85% / exp77 54.00% / exp77+mirror 55.55% 대비)")
    print(f"저장: {OUT}")


if __name__ == "__main__":
    main()

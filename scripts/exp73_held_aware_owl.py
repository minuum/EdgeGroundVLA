#!/usr/bin/env python3
"""HELD-aware 재학습 — OWL-v2 그라운더 버전 (64-12 확장, soda 요청 대응).
exp73_held_aware_train.py와 동일 로직, bbox만 OWL 주석으로 교체."""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.train_exp73_trackA_heads import MLPActionHead, CACHE_V6, SPLIT_SEED, VAL_RATIO
from scripts.exp73_window_cadence import eval_closed_loop_held, swap_bboxes, OWL_ANN
from scripts.exp73_held_aware_train import build_windows_hold_aware, train_one


def main():
    stride = 5
    base = torch.load(str(CACHE_V6), weights_only=False)
    base = [e for e in base if e.get("acts") is not None]
    eps = swap_bboxes(base, OWL_ANN)
    print(f"OWL bbox 교체: {len(eps)}ep", flush=True)

    rng = np.random.default_rng(SPLIT_SEED); idx = list(range(len(eps))); rng.shuffle(idx)
    nv = max(1, int(len(idx) * VAL_RATIO))
    val = [eps[i] for i in idx[:nv]]; tr = [eps[i] for i in idx[nv:]]

    Xtr5, ytr5 = build_windows_hold_aware(tr, stride=stride)
    Xva5, yva5 = build_windows_hold_aware(val, stride=stride)
    print(f"hold-aware train={len(Xtr5)}", flush=True)

    helds = []
    for seed in [0, 1, 2]:
        acc5, st5 = train_one(Xtr5, ytr5, Xva5, yva5, seed, epochs=300)
        m5 = MLPActionHead(); m5.load_state_dict(st5); m5.eval()
        held5 = np.mean([r["success"] for r in eval_closed_loop_held(val, m5, stride=stride)]) * 100
        helds.append(held5)
        print(f"[OWL hold_aware] seed{seed} offline(다수결)={acc5*100:.1f}% HELD={held5:.1f}%", flush=True)
        torch.save({"model": st5, "held_success": held5, "head": "mlp", "stride": stride, "grounder": "owl"},
                   str(ROOT / f"runs/v5_nav/mlp/exp73/exp73_owl_trackF_v6_mlp_holdaware_seed{seed}.pt"))

    print(f"=== OWL hold_aware: HELD {np.mean(helds):.1f}±{np.std(helds):.1f}% ===", flush=True)


if __name__ == "__main__":
    main()

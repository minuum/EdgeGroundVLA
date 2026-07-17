#!/usr/bin/env python3
"""
exp73 승자(mlp, pg448_trackF/v6) closed-loop(FPE/TLD/Success) 검증.

exp73_v6_vis_cache.pt(이미 인코딩된 vis+bbox+gt+raw action)를 재사용해 GPU 재인코딩 없이
train_exp73_trackA_heads.py와 동일한 window=6/bbox_scale=3.0 피처로 프레임별 예측 후,
rollout_core로 궤적을 구성해 exp11(0%)/step2(66.7%) 대비 위치를 확인한다.

Usage:
  .venv/bin/python3 scripts/sim/evaluate_closed_loop_exp73.py \
      --ckpt runs/v5_nav/mlp/exp73/exp73_pg448_trackF_v6_mlp.pt --head mlp
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.sim.rollout_core import ACTION_VEL, build_trajectory, compute_metrics
from scripts.train_exp73_trackA_heads import (
    MLPActionHead, CxGeomHead, TransformerActionHead, WINDOW, BBOX_SCALE, FRAME_DIM,
)

CACHE_V6 = ROOT / "docs/v5/closed_loop_eval/exp73_v6_vis_cache.pt"
OUT_DIR = ROOT / "docs/v5/closed_loop_eval"
VAL_RATIO = 0.15
SPLIT_SEED = 42

HEAD_CLS = {"mlp": MLPActionHead, "cxgeom": CxGeomHead, "transformer": TransformerActionHead}


def val_split(eps, seed=SPLIT_SEED, ratio=VAL_RATIO):
    idx = np.arange(len(eps))
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    n_val = max(1, int(len(eps) * ratio))
    return [eps[i] for i in idx[:n_val]]


def build_episode_windows(ep, window=WINDOW, bbox_scale=BBOX_SCALE):
    bboxes, vis = ep["bboxes"], ep["vis"]
    X = []
    for t in range(len(bboxes)):
        seq = []
        for k in range(window):
            idx = max(0, t - (window - 1 - k))
            seq.append([v * bbox_scale for v in bboxes[idx]] + vis[idx].tolist())
        X.append(seq)
    return np.asarray(X, dtype=np.float32)


@torch.no_grad()
def eval_episode(ep, model, device):
    X = torch.tensor(build_episode_windows(ep), device=device)
    pred = model(X).argmax(1).cpu().numpy()

    gt_classes = np.asarray(ep["gts"], dtype=np.int64)
    expert_traj = build_trajectory(gt_classes.tolist())
    pred_traj = build_trajectory(pred.tolist())
    m = compute_metrics(expert_traj, pred_traj)
    m["val_acc"] = float((pred == gt_classes).mean())
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "runs/v5_nav/mlp/exp73/exp73_pg448_trackF_v6_mlp.pt"))
    ap.add_argument("--head", default="mlp", choices=list(HEAD_CLS))
    ap.add_argument("--cache", default=str(CACHE_V6))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LOAD] cache {args.cache}", flush=True)
    eps = torch.load(args.cache, weights_only=False)
    eps = [e for e in eps if e.get("acts") is not None]  # V6만 (raw action 있는 것)
    val_eps = val_split(eps)
    print(f"  전체 {len(eps)}ep, val {len(val_eps)}ep", flush=True)

    model_cls = HEAD_CLS[args.head]
    model = model_cls().to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"[LOAD] ckpt {args.ckpt} (val_acc_at_save={ckpt.get('val_acc')})", flush=True)

    results = [eval_episode(ep, model, device) for ep in val_eps]

    fpe = np.array([r["fpe"] for r in results])
    tld = np.array([r["tld"] for r in results])
    acc = np.array([r["val_acc"] for r in results])
    succ = np.array([r["success"] for r in results])

    summary = {
        "head": args.head, "ckpt": args.ckpt, "n_episodes": len(val_eps),
        "fpe_mean": float(fpe.mean()), "fpe_std": float(fpe.std()),
        "tld_mean": float(tld.mean()),
        "val_acc_mean": float(acc.mean()),
        "success_rate": float(succ.mean()),
        "per_episode": [
            {"stem": ep["stem"], "path_type": ep["path_type"], **r}
            for ep, r in zip(val_eps, results)
        ],
    }
    out = OUT_DIR / f"exp73_closed_loop_{Path(args.ckpt).stem}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n=== exp73 {args.head} closed-loop ===")
    print(f"  FPE: {summary['fpe_mean']:.3f} ± {summary['fpe_std']:.3f} m")
    print(f"  TLD: {summary['tld_mean']:.3f}")
    print(f"  val_acc: {summary['val_acc_mean']*100:.1f}%")
    print(f"  Success@0.5m: {summary['success_rate']*100:.1f}%")
    print(f"  저장 → {out}")


if __name__ == "__main__":
    main()

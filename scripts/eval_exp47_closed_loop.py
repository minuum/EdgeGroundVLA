#!/usr/bin/env python3
"""
Exp47 Closed-Loop Offline Evaluation
  - Exp46 캐시(bbox + vision) + instruction embeddings → exp47_mlp.pt
  - 동일 80/20 val split (stratified, seed=42)
  - 에피소드별 offline rollout → FPE, TLD, success_rate
"""
import json, sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.sim.rollout_core import (
    CLASS_NAMES, DT_DEFAULT,
    build_trajectory, compute_metrics,
)

EXP46_DIR  = ROOT / "docs" / "v5" / "bbox_nav_exp46"
EXP47_DIR  = ROOT / "docs" / "v5" / "bbox_nav_exp47"
MLP47_DIR  = ROOT / "runs" / "v5_nav" / "mlp" / "exp47"
OUT_DIR    = ROOT / "docs" / "v5" / "closed_loop_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 8
WINDOW      = 8
VIS_DIM     = 1024
INSTR_DIM   = 2048
D_IN        = WINDOW * 4 + VIS_DIM + INSTR_DIM  # 3104


def build_mlp(d_in=D_IN):
    return nn.Sequential(
        nn.Linear(d_in, 512), nn.ReLU(), nn.Dropout(0.25),
        nn.Linear(512, 256),  nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 128),  nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(128, 64),   nn.ReLU(),
        nn.Linear(64, NUM_CLASSES),
    )


def load_model():
    ckpt = torch.load(str(MLP47_DIR / "exp47_mlp.pt"), map_location="cpu", weights_only=False)
    net  = build_mlp(ckpt["d_in"])
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    return net, ckpt["instr_map"]


def eval_episode(ep_data, vis_feats, instr_emb, net, device):
    frames = ep_data["frames"]
    pred_classes, expert_classes = [], []
    for t in range(len(frames)):
        bbox_feat = []
        for k in range(WINDOW):
            idx = max(0, t - (WINDOW - 1 - k))
            fr  = frames[idx]
            bbox_feat.extend([fr["cx"], fr["cy"], fr["area"], float(fr["has_bbox"])])

        feat = np.concatenate([
            np.array(bbox_feat, dtype=np.float32),
            vis_feats[t],
            instr_emb,
        ])
        x = torch.tensor([feat], dtype=torch.float32, device=device)
        with torch.no_grad():
            cls = int(net(x).argmax(1).item())
        pred_classes.append(min(cls, NUM_CLASSES - 1))
        expert_classes.append(frames[t]["gt_class"])
    return pred_classes, expert_classes


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    bbox_data  = json.loads((EXP46_DIR / "bbox_dataset_full.json").read_text())
    vis_index  = json.loads((EXP46_DIR / "vision_features_index.json").read_text())
    instr_raw  = json.loads((EXP47_DIR / "instruction_embeddings.json").read_text())
    npz        = np.load(str(EXP46_DIR / "vision_features.npz"))
    vis_cache  = {ep: npz[f"ep_{i}"] for ep, i in vis_index.items()}
    instr_embs = {pt: np.array(v, dtype=np.float32) for pt, v in instr_raw.items()}

    net, instr_map = load_model()
    net = net.to(device)

    path_labels = [ep["path_type"] for ep in bbox_data]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, te_idx = next(sss.split(np.zeros(len(bbox_data)), path_labels))
    val_eps = [bbox_data[i] for i in te_idx]

    print(f"\nVal episodes: {len(val_eps)}")

    results, per_path = {}, {}
    success_thresh = 1.0

    for ep_data in val_eps:
        ep_key    = ep_data["episode"]
        pt        = ep_data["path_type"]
        vis_feats = vis_cache.get(ep_key)
        instr_emb = instr_embs.get(pt)
        if vis_feats is None or instr_emb is None:
            print(f"  SKIP: {Path(ep_key).name}")
            continue

        pred_cls, expert_cls = eval_episode(ep_data, vis_feats, instr_emb, net, device)
        pred_traj   = build_trajectory(pred_cls)
        expert_traj = build_trajectory(expert_cls)
        metrics = compute_metrics(expert_traj, pred_traj)
        metrics["path_type"] = pt
        metrics["pm"] = float(np.mean([p == e for p, e in zip(pred_cls, expert_cls)]))
        results[Path(ep_key).stem] = metrics
        per_path.setdefault(pt, []).append(metrics)

    all_fpe = [m["fpe"] for m in results.values()]
    all_tld = [m["tld"] for m in results.values()]
    all_pm  = [m["pm"]  for m in results.values()]
    all_suc = [1 if m["fpe"] < success_thresh else 0 for m in results.values()]

    print(f"\n=== Exp47 Closed-Loop Eval ===")
    print(f"  Episodes:          {len(results)}")
    print(f"  PM (frame-level):  {np.mean(all_pm):.1%}")
    print(f"  FPE mean:          {np.mean(all_fpe):.3f} m")
    print(f"  TLD mean:          {np.mean(all_tld):.3f}")
    print(f"  Success (<1m FPE): {np.mean(all_suc):.1%}  ({sum(all_suc)}/{len(all_suc)})")

    print(f"\n--- Per path_type ---")
    for pt in sorted(per_path.keys()):
        ms  = per_path[pt]
        fpe = np.mean([m["fpe"] for m in ms])
        suc = np.mean([1 if m["fpe"] < success_thresh else 0 for m in ms])
        pm  = np.mean([m["pm"]  for m in ms])
        print(f"  {pt:<20}  n={len(ms):2d}  PM={pm:.1%}  FPE={fpe:.3f}m  suc={suc:.1%}")

    out = {
        "model": "exp47",
        "n_val_eps": len(results),
        "overall_pm": float(np.mean(all_pm)),
        "mean_fpe": float(np.mean(all_fpe)),
        "mean_tld": float(np.mean(all_tld)),
        "success_rate": float(np.mean(all_suc)),
        "per_path": {
            pt: {
                "n": len(ms),
                "pm": float(np.mean([m["pm"] for m in ms])),
                "mean_fpe": float(np.mean([m["fpe"] for m in ms])),
                "success_rate": float(np.mean([1 if m["fpe"] < success_thresh else 0 for m in ms])),
            }
            for pt, ms in per_path.items()
        },
        "episodes": results,
    }
    out_path = OUT_DIR / "exp47_closed_loop_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n저장: {out_path}")


if __name__ == "__main__":
    main()

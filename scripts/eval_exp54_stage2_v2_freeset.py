#!/usr/bin/env python3
"""
Exp54 Stage2-v2 CL 평가 — exp55 free set (추가 위치, 21 에피소드)

exp46 val split(9ep, fixed_center 출발)과 달리
exp55 free set은 다른 출발 위치/조건에서 수집한 데이터.

path_type: free_center / free_left / free_right (각 7개)

성공 기준: FPE < 0.5m AND TLD ∈ [0.7, 1.5]

Usage:
  .venv/bin/python3 scripts/eval_exp54_stage2_v2_freeset.py
  .venv/bin/python3 scripts/eval_exp54_stage2_v2_freeset.py --success_fpe 0.8
"""
import sys, json, argparse, warnings
import numpy as np
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from PIL import Image

from scripts.sim.rollout_core import (
    ACTION_VEL, CLASS_NAMES, DT_DEFAULT,
    build_trajectory, compute_metrics,
)

VLM_PATH    = ROOT / ".vlms" / "kosmos-2-patch14-224"
DATA_PATH   = ROOT / "docs" / "v5" / "bbox_nav_exp55" / "bbox_dataset_free.json"
STAGE1_V2   = ROOT / "runs" / "v5_nav" / "mlp" / "shared" / "stage1_v2_projs.pt"
STAGE2_CKPT = ROOT / "runs" / "v5_nav" / "mlp" / "exp54" / "stage2_v2" / "stage2_v2_mlp.pt"
OUT_DIR     = ROOT / "docs" / "v5" / "closed_loop_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 8
WINDOW      = 8
VIS_DIM     = 1024
PROJ_DIM    = 256
D_IN        = WINDOW * 4 + PROJ_DIM   # 288


class FrozenCLIPV2(nn.Module):
    def __init__(self, vlm_path, ckpt_path, device):
        super().__init__()
        from transformers import AutoModelForVision2Seq, AutoProcessor
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        print(f"[MODEL] Stage1 v2 val_acc={ckpt['val_acc']:.4f}")
        self.processor  = AutoProcessor.from_pretrained(str(vlm_path))
        base = AutoModelForVision2Seq.from_pretrained(str(vlm_path), torch_dtype=torch.float16)
        self.vision_model = base.vision_model.to(device)
        self.image_proj   = nn.Linear(VIS_DIM, PROJ_DIM).to(device)
        self.image_proj.load_state_dict(ckpt["image_proj"])
        for p in self.vision_model.parameters(): p.requires_grad = False
        for p in self.image_proj.parameters():   p.requires_grad = False

    @torch.no_grad()
    def encode_batch(self, pil_images, device, batch=16):
        all_feats = []
        for i in range(0, len(pil_images), batch):
            imgs = pil_images[i:i+batch]
            inputs = self.processor(images=imgs, return_tensors="pt")
            pv  = inputs["pixel_values"].to(device, dtype=torch.float16)
            out = self.vision_model(pixel_values=pv)
            feat = out.last_hidden_state.mean(dim=1).float()
            all_feats.append(F.normalize(self.image_proj(feat), dim=-1))
        return torch.cat(all_feats, dim=0)


class ActionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_IN, 256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 128),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),   nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, NUM_CLASSES),
        )
    def forward(self, x): return self.net(x)


def bbox_feat(frames, t):
    arr = []
    for k in range(WINDOW):
        fr = frames[max(0, t - (WINDOW - 1 - k))]
        arr.extend([fr["cx"], fr["cy"], fr["area"], float(fr["has_bbox"])])
    return np.array(arr, dtype=np.float32)


def eval_episode(ep_entry, enc, mlp, device):
    frames  = ep_entry["frames"]
    h5_path = Path(ep_entry["episode"])
    try:
        with h5py.File(str(h5_path), "r") as f:
            imgs = [Image.fromarray(f["observations"]["images"][i])
                    for i in range(len(frames))]
    except Exception as e:
        print(f"  [SKIP] {h5_path.name}: {e}")
        return None, None

    expert_classes = [fr["gt_class"] for fr in frames]
    vis_feats = enc.encode_batch(imgs, device)

    pred_classes = []
    mlp.eval()
    with torch.no_grad():
        for t in range(len(frames)):
            bf = torch.tensor(bbox_feat(frames, t), device=device)
            x  = torch.cat([bf, vis_feats[t]]).unsqueeze(0)
            pred_classes.append(mlp(x).argmax(1).item())

    return pred_classes, expert_classes


def compute_episode_metrics(pred_classes, expert_classes, dt, success_fpe):
    pred_traj   = build_trajectory(pred_classes, dt)
    expert_traj = build_trajectory(expert_classes, dt)

    pred_fp   = pred_traj.final_pos()
    expert_fp = expert_traj.final_pos()

    fpe = np.sqrt((pred_fp[0]-expert_fp[0])**2 + (pred_fp[1]-expert_fp[1])**2)
    pred_len   = pred_traj.total_length()
    expert_len = expert_traj.total_length()
    tld = pred_len / expert_len if expert_len > 1e-6 else float("inf")
    success = (fpe < success_fpe) and (0.7 <= tld <= 1.5)
    return {"fpe": round(float(fpe), 4), "tld": round(float(tld), 4), "success": bool(success)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dt",          type=float, default=DT_DEFAULT)
    p.add_argument("--success_fpe", type=float, default=0.5)
    p.add_argument("--ckpt",        type=str,   default=str(STAGE2_CKPT))
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"[ERROR] 체크포인트 없음: {ckpt_path}"); return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    data = json.loads(DATA_PATH.read_text())
    print(f"[DATA] {len(data)} episodes from exp55 free set")
    pt_counts = defaultdict(int)
    for ep in data: pt_counts[ep["path_type"]] += 1
    for pt, n in sorted(pt_counts.items()): print(f"  {pt}: {n}개")

    print("\n[MODEL] Stage1 v2 인코더 로드 중...")
    enc = FrozenCLIPV2(VLM_PATH, STAGE1_V2, device).to(device).eval()

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    mlp  = ActionMLP().to(device)
    mlp.load_state_dict(ckpt["mlp"])
    mlp.eval()
    print(f"[MODEL] Stage2 v2 MLP loaded (val_acc={ckpt['val_acc']:.4f})\n")

    results_by_path = defaultdict(list)
    all_metrics = []

    for i, ep in enumerate(data):
        pt = ep["path_type"]
        pred, expert = eval_episode(ep, enc, mlp, device)
        if pred is None:
            continue
        m = compute_episode_metrics(pred, expert, args.dt, args.success_fpe)
        m["path_type"] = pt
        m["episode"]   = Path(ep["episode"]).stem
        results_by_path[pt].append(m)
        all_metrics.append(m)
        tag = "✅" if m["success"] else "❌"
        print(f"  [{i+1:2d}/{len(data)}] {pt:<14} FPE={m['fpe']:.3f}m  TLD={m['tld']:.2f}  {tag}  ({Path(ep['episode']).stem[-30:]})")

    # 집계
    total   = len(all_metrics)
    success = sum(1 for m in all_metrics if m["success"])

    print(f"\n{'='*60}")
    print(f"  Exp54 Stage2-v2  |  exp55 free set  |  FPE<{args.success_fpe}m & TLD∈[0.7,1.5]")
    print(f"{'='*60}")
    print(f"  전체: {success}/{total} = {success/total*100:.1f}%   mean_FPE={np.mean([m['fpe'] for m in all_metrics]):.3f}m")
    print()
    for pt in sorted(results_by_path.keys()):
        ms = results_by_path[pt]
        sr  = sum(1 for m in ms if m["success"])
        fpe = np.mean([m["fpe"] for m in ms])
        tld = np.mean([m["tld"] for m in ms])
        print(f"  {pt:<14} {sr}/{len(ms)}  SR={sr/len(ms):.0%}  FPE={fpe:.3f}m  TLD={tld:.3f}")
    print(f"\n  참고 — exp46 val split: step2=66.7%(9ep)")
    print(f"{'='*60}")

    # JSON 저장
    metrics_path = OUT_DIR / "rollout_metrics.json"
    existing = json.loads(metrics_path.read_text()) if metrics_path.exists() else {"summary": {}, "per_path": {}}

    existing["summary"]["exp54_s2v2_free"] = {
        "dataset": "exp55_free_set",
        "n_episodes": total,
        "success_rate": round(success / total, 4) if total > 0 else 0,
        "mean_fpe": round(float(np.mean([m["fpe"] for m in all_metrics])), 4),
        "mean_tld": round(float(np.mean([m["tld"] for m in all_metrics])), 4),
        "success_fpe_threshold": args.success_fpe,
        "per_path_type": {
            pt: {
                "n": len(ms),
                "success_rate": round(sum(1 for m in ms if m["success"]) / len(ms), 4),
                "mean_fpe": round(float(np.mean([m["fpe"] for m in ms])), 4),
                "mean_tld": round(float(np.mean([m["tld"] for m in ms])), 4),
            }
            for pt, ms in results_by_path.items()
        }
    }
    existing["per_path"]["exp54_s2v2_free"] = {
        pt: [{"fpe": m["fpe"], "tld": m["tld"], "success": m["success"], "episode": m["episode"]}
             for m in ms]
        for pt, ms in results_by_path.items()
    }
    metrics_path.write_text(json.dumps(existing, indent=2))
    print(f"\n[SAVE] {metrics_path}")


if __name__ == "__main__":
    main()

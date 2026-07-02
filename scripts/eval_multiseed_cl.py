#!/usr/bin/env python3
"""
Multi-seed CL 평가 — exp67 / exp71 / exp72 × 5 seeds
FrozenCLIPV2 인코더를 한 번만 로드하고 15개 체크포인트 순차 평가.

결과: logs/multiseed_ablation/cl_summary.json
      (val_acc + CL_SR + CL_FPE mean±std)

Usage:
  .venv/bin/python3 scripts/eval_multiseed_cl.py [--seeds 0 1 2 3 4]
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
import h5py
from PIL import Image

from scripts.sim.rollout_core import (
    build_trajectory, continuous_to_class, compute_metrics, DT_DEFAULT
)

DATA_DIR   = ROOT / "ROS_action/mobile_vla_dataset_v5"
STEP1_DIR  = ROOT / "docs/v5/bbox_nav_step1"
ANN_PG448  = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_pg448_cx.json"
VLM_PATH   = ROOT / ".vlms/kosmos-2-patch14-224"
STAGE1_PT  = ROOT / "runs/v5_nav/mlp/shared/stage1_v2_projs.pt"
LOG_DIR    = ROOT / "logs/multiseed_ablation"
LOG_DIR.mkdir(parents=True, exist_ok=True)

CLIP_WINDOW = 8; CLIP_PROJ = 256; CLIP_VIS = 1024
NUM_CLASSES = 8


# ── 인코더 (한 번만 로드) ────────────────────────────────────
def load_encoder(device):
    from transformers import AutoModelForVision2Seq, AutoProcessor
    ckpt = torch.load(str(STAGE1_PT), map_location=device, weights_only=False)
    proc = AutoProcessor.from_pretrained(str(VLM_PATH))
    base = AutoModelForVision2Seq.from_pretrained(str(VLM_PATH), torch_dtype=torch.float16)
    vm   = base.vision_model.to(device).eval()
    proj = nn.Linear(CLIP_VIS, CLIP_PROJ).to(device)
    proj.load_state_dict(ckpt["image_proj"]); proj.eval()
    print("[ENCODER] FrozenCLIPV2 로드 완료")
    return proc, vm, proj

def encode_img(img_np, proc, vm, proj, device):
    img = Image.fromarray(img_np.astype("uint8"))
    inp = proc(images=[img], return_tensors="pt")
    pv  = inp["pixel_values"].to(device, dtype=torch.float16)
    with torch.no_grad():
        feat = vm(pixel_values=pv).last_hidden_state.mean(1).float()
        return proj(feat).squeeze(0)


# ── 테스트 에피소드 split (evaluate_closed_loop_v5.py와 동일) ─
def get_test_episodes():
    ann_pg448 = json.loads(ANN_PG448.read_text())
    ep_stems  = {Path(ep["episode"]).stem: ep for ep in ann_pg448}

    bbox_ds_ref = json.loads((STEP1_DIR / "bbox_dataset.json").read_text())
    by_path = defaultdict(list)
    for i, ep in enumerate(bbox_ds_ref):
        by_path[ep["path_type"]].append(i)
    rng = np.random.default_rng(42)
    test_idx = []
    for _, idxs in by_path.items():
        rng.shuffle(idxs)
        k = max(1, int(len(idxs) * 0.2))
        test_idx.extend(idxs[:k])
    test_refs = [bbox_ds_ref[i] for i in test_idx]

    test_eps = []
    for ref in test_refs:
        stem = ref["episode"]
        if not list(DATA_DIR.glob(f"{stem}.h5")):
            continue
        ann_ep = ep_stems.get(stem)
        if ann_ep is None:
            continue
        test_eps.append(ann_ep)
    print(f"[DATA] 테스트 에피소드: {len(test_eps)}개")
    return test_eps


# ── 모델 정의 ────────────────────────────────────────────────
class _MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d,256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128,64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, NUM_CLASSES))
    def forward(self, x): return self.net(x)

class _TransHead(nn.Module):
    FD = CLIP_PROJ + 4  # 260
    def __init__(self, window=CLIP_WINDOW):
        super().__init__()
        self.window = window
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.FD))
        self.pos_emb   = nn.Embedding(window + 1, self.FD)
        el = nn.TransformerEncoderLayer(
            d_model=self.FD, nhead=4, dim_feedforward=512,
            dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(el, num_layers=2)
        self.head = nn.Sequential(
            nn.LayerNorm(self.FD), nn.Linear(self.FD,128),
            nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, NUM_CLASSES))
    def forward(self, x):
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        pos = torch.arange(x.size(1), device=x.device)
        x = x + self.pos_emb(pos)
        return self.head(self.encoder(x)[:, 0])

class _GeomMLP(nn.Module):
    def __init__(self, hd):
        super().__init__()
        self.branch_a = nn.Sequential(
            nn.Linear(hd,256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.1))
        self.branch_b = nn.Sequential(nn.Linear(4,32), nn.ReLU())
        self.merge = nn.Sequential(
            nn.Linear(160,64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, NUM_CLASSES))
    def forward(self, h, g):
        return self.merge(torch.cat([self.branch_a(h), self.branch_b(g)], -1))


# ── 에피소드별 CL 평가 ───────────────────────────────────────
def eval_episode_exp67(ann_ep, head, proc, vm, proj, device):
    frames = ann_ep["frames"]
    path   = next(DATA_DIR.glob(f"{Path(ann_ep['episode']).stem}.h5"))
    with h5py.File(str(path), "r") as f:
        imgs = f["observations"]["images"][:]
        expert_actions = f["actions"][:]
    cache = {}
    def get_vis(fi):
        if fi not in cache:
            cache[fi] = encode_img(imgs[fi], proc, vm, proj, device)
        return cache[fi]
    preds = []
    for t, fr in enumerate(frames):
        hist = []
        for k in range(CLIP_WINDOW):
            fidx = max(0, t - (CLIP_WINDOW - 1 - k))
            f2 = frames[fidx]
            hist += [f2.get("cx_det",0.5), f2.get("cy_det",0.5),
                     f2.get("area_det",0.05), float(f2.get("has_bbox",False))]
        vis = get_vis(fr["frame_idx"]).cpu().tolist()
        x = torch.tensor([hist + vis], dtype=torch.float32, device=device)
        with torch.no_grad():
            preds.append(min(int(head(x).argmax(1).item()), NUM_CLASSES-1))
    return preds, expert_actions[:len(frames)]

def eval_episode_exp71(ann_ep, head, proc, vm, proj, device, window=CLIP_WINDOW):
    frames = ann_ep["frames"]
    path   = next(DATA_DIR.glob(f"{Path(ann_ep['episode']).stem}.h5"))
    with h5py.File(str(path), "r") as f:
        imgs = f["observations"]["images"][:]
        expert_actions = f["actions"][:]
    cache = {}
    def get_vis(fi):
        if fi not in cache:
            cache[fi] = encode_img(imgs[fi], proc, vm, proj, device)
        return cache[fi]
    preds = []
    for t, fr in enumerate(frames):
        seq = []
        for k in range(window):
            fidx = max(0, t - (window - 1 - k))
            f2 = frames[fidx]
            bbox = [f2.get("cx_det",0.5), f2.get("cy_det",0.5),
                    f2.get("area_det",0.05), float(f2.get("has_bbox",False))]
            seq.append(bbox + get_vis(frames[fidx]["frame_idx"]).cpu().tolist())
        x = torch.tensor([seq], dtype=torch.float32, device=device)
        with torch.no_grad():
            preds.append(min(int(head(x).argmax(1).item()), NUM_CLASSES-1))
    return preds, expert_actions[:len(frames)]

def eval_episode_exp72(ann_ep, head, proc, vm, proj, device):
    frames = ann_ep["frames"]
    path   = next(DATA_DIR.glob(f"{Path(ann_ep['episode']).stem}.h5"))
    with h5py.File(str(path), "r") as f:
        imgs = f["observations"]["images"][:]
        expert_actions = f["actions"][:]
    cache = {}
    def get_vis(fi):
        if fi not in cache:
            cache[fi] = encode_img(imgs[fi], proc, vm, proj, device)
        return cache[fi]
    preds = []
    for t, fr in enumerate(frames):
        hist = []
        for k in range(CLIP_WINDOW):
            fidx = max(0, t - (CLIP_WINDOW - 1 - k))
            f2 = frames[fidx]
            hist += [f2.get("cx_det",0.5), f2.get("cy_det",0.5),
                     f2.get("area_det",0.05), float(f2.get("has_bbox",False))]
        vis  = get_vis(fr["frame_idx"]).cpu().tolist()
        geom = [fr.get("cx_det",0.5), fr.get("cy_det",0.5),
                fr.get("area_det",0.05), float(fr.get("has_bbox",False))]
        xh = torch.tensor([hist + vis], dtype=torch.float32, device=device)
        xg = torch.tensor([geom],       dtype=torch.float32, device=device)
        with torch.no_grad():
            preds.append(min(int(head(xh, xg).argmax(1).item()), NUM_CLASSES-1))
    return preds, expert_actions[:len(frames)]


def run_cl(test_eps, head, eval_fn, device, proc, vm, proj, window=CLIP_WINDOW, dt=DT_DEFAULT, success_fpe=0.5):
    srs, fpes = [], []
    for ann_ep in test_eps:
        try:
            if eval_fn.__name__ == "eval_episode_exp71":
                preds, expert = eval_fn(ann_ep, head, proc, vm, proj, device, window=window)
            else:
                preds, expert = eval_fn(ann_ep, head, proc, vm, proj, device)
        except Exception as e:
            print(f"  [SKIP] {Path(ann_ep['episode']).stem}: {e}")
            continue
        expert_cls  = [continuous_to_class(*a[:3]) for a in expert]
        expert_traj = build_trajectory(expert_cls, dt)
        pred_traj   = build_trajectory(preds, dt)
        m = compute_metrics(expert_traj, pred_traj, success_fpe)
        srs.append(float(m["success"]))
        fpes.append(float(m["fpe"]))
    sr_mean  = np.mean(srs)  * 100 if srs else 0
    fpe_mean = np.mean(fpes)       if fpes else 0
    return sr_mean, fpe_mean, len(srs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0,1,2,3,4])
    ap.add_argument("--success-fpe", type=float, default=0.5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}  seeds={args.seeds}")

    proc, vm, proj = load_encoder(device)
    test_eps = get_test_episodes()

    results = {"exp67": [], "exp71": [], "exp72": []}
    val_acc_by_exp = json.loads((LOG_DIR / "summary.json").read_text()) if (LOG_DIR / "summary.json").exists() else {}

    TOTAL = len(args.seeds) * 3
    done  = 0

    for seed in args.seeds:
        # ── exp67 MLP ──────────────────────────────────────────
        done += 1
        pt = ROOT / f"runs/v5_nav/mlp/exp67_seed{seed}/action_mlp.pt"
        print(f"\n[{done}/{TOTAL}] exp67 MLP  seed={seed}")
        if pt.exists():
            ckpt = torch.load(str(pt), map_location=device, weights_only=False)
            head = _MLP(ckpt["d_in"]).to(device)
            head.load_state_dict(ckpt["mlp"]); head.eval()
            sr, fpe, n = run_cl(test_eps, head, eval_episode_exp67, device, proc, vm, proj,
                                 success_fpe=args.success_fpe)
            val_acc = ckpt.get("val_acc", 0) * 100
            results["exp67"].append({"seed": seed, "val_acc": val_acc, "cl_sr": sr, "cl_fpe": fpe, "n_eps": n})
            print(f"  val_acc={val_acc:.1f}%  CL_SR={sr:.1f}%  FPE={fpe:.3f}m  ({n} eps)")
        else:
            print(f"  [SKIP] {pt} 없음")

        # ── exp71 Transformer ──────────────────────────────────
        done += 1
        pt = ROOT / f"runs/v5_nav/mlp/exp71_seed{seed}/action_transformer.pt"
        print(f"[{done}/{TOTAL}] exp71 Trans  seed={seed}")
        if pt.exists():
            ckpt = torch.load(str(pt), map_location=device, weights_only=False)
            head = _TransHead(window=CLIP_WINDOW).to(device)
            head.load_state_dict(ckpt["model"]); head.eval()
            sr, fpe, n = run_cl(test_eps, head, eval_episode_exp71, device, proc, vm, proj,
                                 success_fpe=args.success_fpe)
            val_acc = ckpt.get("val_acc", 0) * 100
            results["exp71"].append({"seed": seed, "val_acc": val_acc, "cl_sr": sr, "cl_fpe": fpe, "n_eps": n})
            print(f"  val_acc={val_acc:.1f}%  CL_SR={sr:.1f}%  FPE={fpe:.3f}m  ({n} eps)")
        else:
            print(f"  [SKIP] {pt} 없음")

        # ── exp72 cx-Geom ──────────────────────────────────────
        done += 1
        pt = ROOT / f"runs/v5_nav/mlp/exp72_seed{seed}/action_cxgeom.pt"
        print(f"[{done}/{TOTAL}] exp72 Geom  seed={seed}")
        if pt.exists():
            ckpt = torch.load(str(pt), map_location=device, weights_only=False)
            head = _GeomMLP(hd=ckpt["hist_dim"]).to(device)
            head.load_state_dict(ckpt["model"]); head.eval()
            sr, fpe, n = run_cl(test_eps, head, eval_episode_exp72, device, proc, vm, proj,
                                 success_fpe=args.success_fpe)
            val_acc = ckpt.get("val_acc", 0) * 100
            results["exp72"].append({"seed": seed, "val_acc": val_acc, "cl_sr": sr, "cl_fpe": fpe, "n_eps": n})
            print(f"  val_acc={val_acc:.1f}%  CL_SR={sr:.1f}%  FPE={fpe:.3f}m  ({n} eps)")
        else:
            print(f"  [SKIP] {pt} 없음")

    # ── 집계 ─────────────────────────────────────────────────
    print("\n" + "="*60)
    print(" 최종 집계 (mean ± std, 5 seeds)")
    print("="*60)
    summary = {}
    for exp, rows in results.items():
        if not rows: continue
        va  = [r["val_acc"] for r in rows]
        sr  = [r["cl_sr"]   for r in rows]
        fpe = [r["cl_fpe"]  for r in rows]
        import statistics
        summary[exp] = {
            "val_acc_mean": np.mean(va),  "val_acc_std": statistics.stdev(va) if len(va)>1 else 0,
            "cl_sr_mean":   np.mean(sr),  "cl_sr_std":   statistics.stdev(sr)  if len(sr)>1  else 0,
            "cl_fpe_mean":  np.mean(fpe), "cl_fpe_std":  statistics.stdev(fpe) if len(fpe)>1 else 0,
            "n_seeds": len(rows), "raw": rows
        }
        print(f"  {exp}: val_acc={np.mean(va):.1f}±{summary[exp]['val_acc_std']:.1f}%  "
              f"CL_SR={np.mean(sr):.1f}±{summary[exp]['cl_sr_std']:.1f}%  "
              f"FPE={np.mean(fpe):.3f}±{summary[exp]['cl_fpe_std']:.3f}m")

    out = LOG_DIR / "cl_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n→ 저장: {out}")


if __name__ == "__main__":
    main()

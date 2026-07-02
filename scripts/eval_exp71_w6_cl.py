#!/usr/bin/env python3
"""
exp71 WINDOW=6 × 5 seeds CL 평가
runs/v5_nav/mlp/exp71_w6_seed{N}/action_transformer.pt 로드 → SR + FPE
"""
import sys, json, warnings
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

from scripts.sim.rollout_core import build_trajectory, continuous_to_class, compute_metrics, DT_DEFAULT

DATA_DIR  = ROOT / "ROS_action/mobile_vla_dataset_v5"
STEP1_DIR = ROOT / "docs/v5/bbox_nav_step1"
ANN_PG448 = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_pg448_cx.json"
VLM_PATH  = ROOT / ".vlms/kosmos-2-patch14-224"
STAGE1_PT = ROOT / "runs/v5_nav/mlp/shared/stage1_v2_projs.pt"
LOG_DIR   = ROOT / "logs/exp71_w6_multiseed"

CLIP_PROJ = 256; CLIP_VIS = 1024; NUM_CLASSES = 8; WINDOW = 6


def load_encoder(device):
    from transformers import AutoModelForVision2Seq, AutoProcessor
    ckpt = torch.load(str(STAGE1_PT), map_location=device, weights_only=False)
    proc = AutoProcessor.from_pretrained(str(VLM_PATH))
    base = AutoModelForVision2Seq.from_pretrained(str(VLM_PATH), torch_dtype=torch.float16)
    vm   = base.vision_model.to(device).eval()
    proj = nn.Linear(CLIP_VIS, CLIP_PROJ).to(device)
    proj.load_state_dict(ckpt["image_proj"]); proj.eval()
    return proc, vm, proj


def encode_img(img_np, proc, vm, proj, device):
    img = Image.fromarray(img_np.astype("uint8"))
    inp = proc(images=[img], return_tensors="pt")
    pv  = inp["pixel_values"].to(device, dtype=torch.float16)
    with torch.no_grad():
        feat = vm(pixel_values=pv).last_hidden_state.mean(1).float()
        return proj(feat).squeeze(0)


def get_test_episodes():
    ann_pg448 = json.loads(ANN_PG448.read_text())
    ep_stems  = {Path(ep["episode"]).stem: ep for ep in ann_pg448}
    bbox_ds   = json.loads((STEP1_DIR / "bbox_dataset.json").read_text())
    by_path   = defaultdict(list)
    for i, ep in enumerate(bbox_ds):
        by_path[ep["path_type"]].append(i)
    rng = np.random.default_rng(42)
    test_idx = []
    for _, idxs in by_path.items():
        rng.shuffle(idxs)
        test_idx.extend(idxs[:max(1, int(len(idxs)*0.2))])
    test_eps = []
    for i in test_idx:
        stem   = bbox_ds[i]["episode"]
        ann_ep = ep_stems.get(stem)
        if ann_ep and list(DATA_DIR.glob(f"{stem}.h5")):
            test_eps.append(ann_ep)
    return test_eps


class _TransHead(nn.Module):
    FD = CLIP_PROJ + 4
    def __init__(self, window=WINDOW):
        super().__init__()
        self.window = window
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.FD))
        self.pos_emb   = nn.Embedding(window + 1, self.FD)
        el = nn.TransformerEncoderLayer(
            d_model=self.FD, nhead=4, dim_feedforward=512,
            dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(el, num_layers=2)
        self.head = nn.Sequential(
            nn.LayerNorm(self.FD), nn.Linear(self.FD, 128),
            nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, NUM_CLASSES))

    def forward(self, x):
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        pos = torch.arange(x.size(1), device=x.device)
        x = x + self.pos_emb(pos)
        return self.head(self.encoder(x)[:, 0])


def eval_episode(ann_ep, head, window, proc, vm, proj, device, dt=DT_DEFAULT):
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
            f2   = frames[fidx]
            bbox = [f2.get("cx_det",0.5), f2.get("cy_det",0.5),
                    f2.get("area_det",0.05), float(f2.get("has_bbox",False))]
            seq.append(bbox + get_vis(frames[fidx]["frame_idx"]).cpu().tolist())
        x = torch.tensor([seq], dtype=torch.float32, device=device)
        with torch.no_grad():
            preds.append(min(int(head(x).argmax(1).item()), NUM_CLASSES-1))
    expert_cls  = [continuous_to_class(*a[:3]) for a in expert_actions[:len(frames)]]
    expert_traj = build_trajectory(expert_cls, dt)
    pred_traj   = build_trajectory(preds, dt)
    return compute_metrics(expert_traj, pred_traj, 0.5)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}  WINDOW={WINDOW}")
    proc, vm, proj = load_encoder(device)
    print("[ENCODER] 로드 완료")
    test_eps = get_test_episodes()
    print(f"[DATA] 테스트 에피소드: {len(test_eps)}개\n")

    all_results = []
    for seed in range(5):
        pt = ROOT / f"runs/v5_nav/mlp/exp71_w6_seed{seed}/action_transformer.pt"
        if not pt.exists():
            print(f"[SKIP] seed={seed}: {pt} 없음"); continue

        ckpt = torch.load(str(pt), map_location=device, weights_only=False)
        head = _TransHead(window=WINDOW).to(device)
        head.load_state_dict(ckpt["model"]); head.eval()
        val_acc = ckpt.get("val_acc", 0) * 100

        srs, fpes = [], []
        for ann_ep in test_eps:
            try:
                m = eval_episode(ann_ep, head, WINDOW, proc, vm, proj, device)
                srs.append(float(m["success"]))
                fpes.append(float(m["fpe"]))
            except Exception as e:
                print(f"  [SKIP] {Path(ann_ep['episode']).stem}: {e}")

        sr   = np.mean(srs)*100 if srs else 0
        fpe  = np.mean(fpes)    if fpes else 0
        all_results.append({"seed": seed, "val_acc": val_acc, "cl_sr": sr, "cl_fpe": fpe, "n_eps": len(srs)})
        print(f"  seed={seed}: val_acc={val_acc:.1f}%  CL_SR={sr:.1f}%  FPE={fpe:.3f}m  ({len(srs)} eps)")

    if all_results:
        accs  = [r["val_acc"] for r in all_results]
        srs   = [r["cl_sr"]   for r in all_results]
        fpes  = [r["cl_fpe"]  for r in all_results]
        summary = {
            "exp": "exp71_w6", "window": WINDOW,
            "val_acc_mean": float(np.mean(accs)), "val_acc_std": float(np.std(accs)),
            "cl_sr_mean":   float(np.mean(srs)),  "cl_sr_std":   float(np.std(srs)),
            "cl_fpe_mean":  float(np.mean(fpes)),  "cl_fpe_std":  float(np.std(fpes)),
            "n_seeds": len(all_results), "raw": all_results
        }
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        out = LOG_DIR / "cl_w6_summary.json"
        out.write_text(json.dumps(summary, indent=2))

        print(f"\n=== exp71 WINDOW=6 × {len(all_results)} seeds ===")
        print(f"  val_acc : {summary['val_acc_mean']:.1f} ± {summary['val_acc_std']:.1f}%")
        print(f"  CL_SR   : {summary['cl_sr_mean']:.1f} ± {summary['cl_sr_std']:.1f}%")
        print(f"  FPE     : {summary['cl_fpe_mean']:.3f} ± {summary['cl_fpe_std']:.3f} m")
        print(f"\n  → 저장: {out}")


if __name__ == "__main__":
    main()

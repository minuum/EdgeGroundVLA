#!/usr/bin/env python3
"""
exp72: cx-conditioned geometric MLP

exp67(MLP)과 동일 구조에서 현재 프레임의 cx/cy를 별도 geometric branch로 주입.
아이디어: MLP가 cx를 history 속에 묻어서 학습하는 것보다
         explicit geometric prior로 주입하면 방향 결정이 더 정확해질 수 있음.

Architecture:
  Branch A (temporal): hist(cx,cy,area,has_bbox × WINDOW=32) + vis_feat(256) = 288 → FC(128)
  Branch B (geometric): cx_now, cy_now, area_now, has_bbox_now = 4 → FC(32)
  Merge: concat(128, 32) = 160 → FC(64) → 8-class

Ablation axis: explicit geometric prior 주입이 implicit history 학습보다 유리한가?

Usage:
  .venv/bin/python3 scripts/train_exp72_stage2_cxgeom.py
"""
import sys, json, random, warnings
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
from PIL import Image

VLM_PATH  = ROOT / ".vlms" / "kosmos-2-patch14-224"
STAGE1_PT = ROOT / "runs/v5_nav/mlp/shared/stage1_v2_projs.pt"
ANN_PATH  = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_pg448_cx.json"
OUT_DIR   = ROOT / "runs/v5_nav/mlp/exp72"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW = 8; NUM_CLASSES = 8; PROJ_DIM = 256; VIS_DIM = 1024
GEOM_DIM = 4  # cx_now, cy_now, area_now, has_bbox_now


class FrozenCLIPV2(nn.Module):
    def __init__(self, vlm_path, stage1_pt, device):
        super().__init__()
        from transformers import AutoModelForVision2Seq, AutoProcessor
        ckpt = torch.load(str(stage1_pt), map_location=device, weights_only=False)
        self.processor = AutoProcessor.from_pretrained(str(vlm_path))
        base = AutoModelForVision2Seq.from_pretrained(str(vlm_path), torch_dtype=torch.float16)
        self.vm = base.vision_model.to(device).eval()
        self.proj = nn.Linear(VIS_DIM, PROJ_DIM).to(device)
        self.proj.load_state_dict(ckpt["image_proj"])
        self.proj.eval()
        self.device = device

    @torch.no_grad()
    def encode_batch(self, pil_imgs, device):
        results = []
        for img in pil_imgs:
            inp = self.processor(images=[img], return_tensors="pt")
            pv  = inp["pixel_values"].to(device, dtype=torch.float16)
            feat = self.vm(pixel_values=pv).last_hidden_state.mean(1).float()
            results.append(self.proj(feat).squeeze(0))
        return torch.stack(results)


class CxGeomMLP(nn.Module):
    """Two-branch: temporal history + explicit current-frame geometry."""
    def __init__(self, hist_dim=288, geom_dim=GEOM_DIM):
        super().__init__()
        # Branch A: temporal (same as exp67 MLP, first 2 layers)
        self.branch_a = nn.Sequential(
            nn.Linear(hist_dim, 256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1))
        # Branch B: geometric prior (current frame cx/cy)
        self.branch_b = nn.Sequential(
            nn.Linear(geom_dim, 32), nn.ReLU())
        # Merge
        self.merge = nn.Sequential(
            nn.Linear(128 + 32, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, NUM_CLASSES))

    def forward(self, hist, geom):
        a = self.branch_a(hist)
        b = self.branch_b(geom)
        return self.merge(torch.cat([a, b], dim=-1))


def build_dataset(ann, enc, device):
    X_hist, X_geom, y = [], [], []
    for ep in ann:
        h5_path = Path(ep["episode"])
        if not h5_path.exists():
            continue
        frames = [fr for fr in ep["frames"] if fr.get("gt_class") is not None]
        if not frames:
            continue
        try:
            with h5py.File(str(h5_path), "r") as f:
                imgs_np = f["observations"]["images"][:]
        except:
            continue
        pil_imgs = [Image.fromarray(imgs_np[fr["frame_idx"]].astype("uint8")) for fr in frames]
        vis = enc.encode_batch(pil_imgs, device)

        for t, fr in enumerate(frames):
            hist = []
            for k in range(WINDOW):
                fidx = max(0, t - (WINDOW - 1 - k))
                f2 = frames[fidx]
                hist.extend([f2.get("cx_det", 0.5), f2.get("cy_det", 0.5),
                             f2.get("area_det", 0.05), float(f2.get("has_bbox", False))])
            hist += vis[t].cpu().tolist()
            # geometric branch: 현재 프레임만
            geom = [fr.get("cx_det", 0.5), fr.get("cy_det", 0.5),
                    fr.get("area_det", 0.05), float(fr.get("has_bbox", False))]
            X_hist.append(hist)
            X_geom.append(geom)
            y.append(fr["gt_class"])
    return (np.array(X_hist, dtype=np.float32),
            np.array(X_geom, dtype=np.float32),
            np.array(y, dtype=np.int64))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    with open(ANN_PATH) as f:
        ann = json.load(f)
    random.shuffle(ann)
    n_val = max(1, int(len(ann) * args.val_ratio))
    val_eps, train_eps = ann[:n_val], ann[n_val:]

    print(f"[DATA] Train {len(train_eps)} / Val {len(val_eps)} eps")
    enc = FrozenCLIPV2(VLM_PATH, STAGE1_PT, device).eval()

    print("[DATA] 준비 중...")
    Xh_tr, Xg_tr, y_tr = build_dataset(train_eps, enc, device)
    Xh_va, Xg_va, y_va = build_dataset(val_eps, enc, device)
    print(f"  Train: {len(y_tr)} / Val: {len(y_va)} samples")

    Xh_tr_t = torch.from_numpy(Xh_tr).to(device)
    Xg_tr_t = torch.from_numpy(Xg_tr).to(device)
    y_tr_t  = torch.from_numpy(y_tr).to(device)
    Xh_va_t = torch.from_numpy(Xh_va).to(device)
    Xg_va_t = torch.from_numpy(Xg_va).to(device)
    y_va_t  = torch.from_numpy(y_va).to(device)

    model = CxGeomMLP(hist_dim=Xh_tr.shape[1]).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best_acc = 0.0
    print(f"\n[TRAIN] {args.epochs} epochs")
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(y_tr_t), device=device)
        loss_sum = 0.0
        for i in range(0, len(perm), 256):
            idx = perm[i:i+256]
            logits = model(Xh_tr_t[idx], Xg_tr_t[idx])
            loss = F.cross_entropy(logits, y_tr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item()
        sched.step()

        if ep % 50 == 0 or ep == args.epochs:
            model.eval()
            with torch.no_grad():
                acc = (model(Xh_va_t, Xg_va_t).argmax(1) == y_va_t).float().mean().item()
            print(f"  epoch {ep:4d}/{args.epochs}  loss={loss_sum:.2f}  val_acc={acc*100:.1f}%")
            if acc >= best_acc:
                best_acc = acc
                torch.save({"model": model.state_dict(), "val_acc": acc,
                            "hist_dim": Xh_tr.shape[1], "geom_dim": GEOM_DIM,
                            "source": "pg448", "exp": "exp72", "head": "cx_geom",
                            "seed": args.seed},
                           str(out_dir / "action_cxgeom.pt"))
                print(f"    [BEST] {acc*100:.1f}% → 저장")

    print(f"\n=== exp72 cx-Geom 결과 ===")
    print(f"  val_acc: {best_acc*100:.1f}%  seed={args.seed}")
    print(f"  체크포인트 → {out_dir}/action_cxgeom.pt")


if __name__ == "__main__":
    main()

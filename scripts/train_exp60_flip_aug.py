#!/usr/bin/env python3
"""
Exp60 flip augmentation — 좌우 반전으로 데이터 2배
train_exp60_stage2_pg2cx.py 기반, build_dataset_with_flip 사용

Usage:
  .venv/bin/python3 scripts/train_exp60_flip_aug.py --epochs 300
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

VLM_PATH  = ROOT / ".vlms/kosmos-2-patch14-224"
STAGE1_PT = ROOT / "runs/v5_nav/mlp/exp54/stage1_v2/stage1_v2_projs.pt"
ANN_PG2   = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_pg2_cx.json"
OUT_DIR   = ROOT / "runs/v5_nav/mlp/exp60"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW = 8; NUM_CLASSES = 8; PROJ_DIM = 256; VIS_DIM = 1024

# 액션 좌우 미러 매핑
ACTION_MIRROR = {0:0, 1:1, 2:3, 3:2, 4:5, 5:4, 6:7, 7:6}
# STOP↔STOP, FWD↔FWD, LEFT↔RIGHT, FWD+L↔FWD+R, ROT_L↔ROT_R


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


class ActionMLP(nn.Module):
    def __init__(self, d_in=288):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),  nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, NUM_CLASSES),
        )
    def forward(self, x): return self.net(x)


def build_dataset_with_flip(ann, enc, device):
    """원본 + 좌우 반전 = 2× 데이터"""
    X, y = [], []
    for ep in ann:
        h5_path = Path(ep["episode"])
        if not h5_path.exists(): continue
        frames = [fr for fr in ep["frames"] if fr.get("gt_class") is not None]
        if not frames: continue
        try:
            with h5py.File(str(h5_path), "r") as f:
                imgs_np = f["observations"]["images"][:]
        except: continue

        pil_orig = [Image.fromarray(imgs_np[fr["frame_idx"]].astype("uint8")) for fr in frames]
        pil_flip = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in pil_orig]

        vis_orig = enc.encode_batch(pil_orig, device)
        vis_flip = enc.encode_batch(pil_flip, device)

        for t, fr in enumerate(frames):
            gt_cls = fr["gt_class"]
            # 합성 STOP: 마지막 2프레임 AND area > 0.50 (진짜 도달 시점만)
            is_last_frames = (t >= len(frames) - 2)
            if is_last_frames and fr.get("area_det", 0) > 0.50:
                gt_cls = 0  # STOP (마지막 2프레임에서만)

            # 원본 히스토리
            hist_o, hist_f = [], []
            for k in range(WINDOW):
                fidx = max(0, t - (WINDOW-1-k))
                f2   = frames[fidx]
                cx   = f2.get("cx_det", 0.5)
                cy   = f2.get("cy_det", 0.5)
                area = f2.get("area_det", 0.05)
                hbb  = float(f2.get("has_bbox", f2.get("detected", False)))
                hist_o.extend([cx, cy, area, hbb])
                hist_f.extend([1.0-cx, cy, area, hbb])  # cx 반전

            X.append(hist_o + vis_orig[t].cpu().tolist())
            y.append(gt_cls)

            X.append(hist_f + vis_flip[t].cpu().tolist())
            y.append(ACTION_MIRROR.get(gt_cls, gt_cls))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",    type=int,   default=300)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    with open(ANN_PG2) as f:
        ann = json.load(f)

    random.shuffle(ann)
    n_val = max(1, int(len(ann)*args.val_ratio))
    val_eps, train_eps = ann[:n_val], ann[n_val:]

    print(f"[DATA] Train {len(train_eps)} eps / Val {len(val_eps)} eps")
    print("[MODEL] Stage1 v2 CLIP 로드...")
    enc = FrozenCLIPV2(VLM_PATH, STAGE1_PT, device).to(device).eval()

    print("[DATA] 반전 증강 데이터 준비 중...")
    X_tr, y_tr = build_dataset_with_flip(train_eps, enc, device)
    X_va, y_va = build_dataset_with_flip(val_eps,   enc, device)
    print(f"  Train: {len(X_tr)} (원본 {len(X_tr)//2} × 2)")
    print(f"  Val:   {len(X_va)}")

    X_tr_t = torch.from_numpy(X_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    X_va_t = torch.from_numpy(X_va).to(device)
    y_va_t = torch.from_numpy(y_va).to(device)

    mlp   = ActionMLP(d_in=X_tr.shape[1]).to(device)
    opt   = torch.optim.Adam(mlp.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best_acc = 0.0
    print(f"\n[TRAIN] {args.epochs} epochs (반전 증강)")
    for ep in range(1, args.epochs+1):
        mlp.train()
        perm = torch.randperm(len(X_tr_t), device=device)
        loss_sum = 0.0
        for i in range(0, len(perm), 256):
            idx = perm[i:i+256]
            logits = mlp(X_tr_t[idx])
            loss   = F.cross_entropy(logits, y_tr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item()
        sched.step()

        if ep % 50 == 0 or ep == args.epochs:
            mlp.eval()
            with torch.no_grad():
                acc = (mlp(X_va_t).argmax(1) == y_va_t).float().mean().item()
            print(f"  epoch {ep:4d}/{args.epochs}  loss={loss_sum:.2f}  val_acc={acc*100:.1f}%")
            if acc >= best_acc:
                best_acc = acc
                torch.save({"mlp": mlp.state_dict(), "val_acc": acc, "d_in": X_tr.shape[1],
                            "source": "pg2_cx_flip", "exp": "exp60_flip"},
                           str(OUT_DIR / "stage2_pg2cx_flip_mlp.pt"))
                print(f"    [BEST] {acc*100:.1f}% → saved")

    print(f"\n최종 val_acc: {best_acc*100:.1f}%")
    print(f"비교: Exp60 기본={50}% CL → flip 후 CL=?")


if __name__ == "__main__":
    main()

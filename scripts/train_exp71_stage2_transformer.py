#!/usr/bin/env python3
"""
exp71: Stage2 Transformer head (temporal self-attention over WINDOW)

exp67(MLP)과 동일 데이터(PG448 ann) + 동일 FrozenCLIPV2 인코더.
차이: 히스토리를 flatten하지 않고 TransformerEncoder로 시퀀스 처리.

Architecture:
  per-frame input: (cx, cy, area, has_bbox=4) + vis_feat(256) = 260-dim
  TransformerEncoder: d_model=260, nhead=4, num_layers=2, CLS token
  → 8-class action

Ablation axis: 시간적 순서/attention이 flat-concat보다 유리한가?

Usage:
  .venv/bin/python3 scripts/train_exp71_stage2_transformer.py
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
OUT_DIR   = ROOT / "runs/v5_nav/mlp/exp71"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW = 8; NUM_CLASSES = 8; PROJ_DIM = 256; VIS_DIM = 1024
BBOX_DIM = 4   # cx, cy, area, has_bbox
FRAME_DIM = BBOX_DIM + PROJ_DIM  # 260


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


class TransformerActionHead(nn.Module):
    """TransformerEncoder over WINDOW frames → 8-class action."""
    def __init__(self, frame_dim=FRAME_DIM, window=WINDOW, nhead=4, num_layers=2):
        super().__init__()
        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, frame_dim))
        # positional encoding (learned)
        self.pos_emb = nn.Embedding(window + 1, frame_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=frame_dim, nhead=nhead, dim_feedforward=512,
            dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(frame_dim),
            nn.Linear(frame_dim, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, NUM_CLASSES))

    def forward(self, x):
        # x: (B, WINDOW, FRAME_DIM)
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)              # (B, W+1, D)
        pos = torch.arange(x.size(1), device=x.device)
        x = x + self.pos_emb(pos)
        x = self.encoder(x)
        return self.head(x[:, 0])                    # CLS token output


def build_dataset(ann, enc, device, window=WINDOW):
    X, y = [], []
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
            seq = []
            for k in range(window):
                fidx = max(0, t - (window - 1 - k))
                f2 = frames[fidx]
                bbox = [f2.get("cx_det", 0.5), f2.get("cy_det", 0.5),
                        f2.get("area_det", 0.05), float(f2.get("has_bbox", False))]
                seq.append(bbox + vis[fidx].cpu().tolist())
            X.append(seq)   # (window, FRAME_DIM)
            y.append(fr["gt_class"])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-ratio", type=float, default=0.15)
    p.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    p.add_argument("--window", type=int, default=WINDOW)
    args = p.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    win = args.window
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    with open(ANN_PATH) as f:
        ann = json.load(f)
    random.shuffle(ann)
    n_val = max(1, int(len(ann) * args.val_ratio))
    val_eps, train_eps = ann[:n_val], ann[n_val:]

    print(f"[DATA] Train {len(train_eps)} / Val {len(val_eps)} eps  WINDOW={win}")
    enc = FrozenCLIPV2(VLM_PATH, STAGE1_PT, device).eval()

    print("[DATA] 준비 중 (시퀀스 형태)...")
    X_tr, y_tr = build_dataset(train_eps, enc, device, window=win)
    X_va, y_va = build_dataset(val_eps, enc, device, window=win)
    print(f"  Train: {len(X_tr)} / Val: {len(X_va)} samples")

    X_tr_t = torch.from_numpy(X_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    X_va_t = torch.from_numpy(X_va).to(device)
    y_va_t = torch.from_numpy(y_va).to(device)

    model = TransformerActionHead(window=win).to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best_acc = 0.0
    print(f"\n[TRAIN] {args.epochs} epochs")
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(X_tr_t), device=device)
        loss_sum = 0.0
        for i in range(0, len(perm), 128):
            idx = perm[i:i+128]
            logits = model(X_tr_t[idx])
            loss = F.cross_entropy(logits, y_tr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item()
        sched.step()

        if ep % 50 == 0 or ep == args.epochs:
            model.eval()
            with torch.no_grad():
                acc = (model(X_va_t).argmax(1) == y_va_t).float().mean().item()
            print(f"  epoch {ep:4d}/{args.epochs}  loss={loss_sum:.2f}  val_acc={acc*100:.1f}%")
            if acc >= best_acc:
                best_acc = acc
                torch.save({"model": model.state_dict(), "val_acc": acc,
                            "source": "pg448", "exp": "exp71", "head": "transformer",
                            "seed": args.seed, "window": win},
                           str(out_dir / "action_transformer.pt"))
                print(f"    [BEST] {acc*100:.1f}% → 저장")

    print(f"\n=== exp71 Transformer 결과 ===")
    print(f"  val_acc: {best_acc*100:.1f}%  seed={args.seed}  window={win}")
    print(f"  체크포인트 → {out_dir}/action_transformer.pt")


if __name__ == "__main__":
    main()

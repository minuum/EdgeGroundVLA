#!/usr/bin/env python3
"""
exp67: Stage2 MLP 재학습 — PG448 grounding cx 사용

exp65(PG224 cx)와 동일 구조, 어노테이션 소스만 변경.
셀프라벨링 185프레임에서 PG448 검출률 98.9% (PG224 73.5% 대비).
학습/추론 grounding 분포를 PG448로 통일.

Usage:
  # Step1: PG448 어노테이션 생성 (먼저 실행)
  .venv/bin/python3 scripts/gen_pg448_annotation.py

  # Step2: MLP 재학습
  .venv/bin/python3 scripts/train_exp67_stage2_pg448.py
"""
import sys, json, random, warnings
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

VLM_PATH  = ROOT / ".vlms" / "kosmos-2-patch14-224"
STAGE1_PT = ROOT / "runs/v5_nav/mlp/shared/stage1_v2_projs.pt"
ANN_PG448 = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_pg448_cx.json"
OUT_DIR   = ROOT / "runs/v5_nav/mlp/exp67"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW = 8; NUM_CLASSES = 8; PROJ_DIM = 256; VIS_DIM = 1024


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


def build_dataset(ann, enc, device, filter_no_bbox=False):
    X, y = [], []
    for ep in ann:
        h5_path = Path(ep["episode"])
        if not h5_path.exists():
            continue
        frames = [fr for fr in ep["frames"] if fr.get("gt_class") is not None]
        if filter_no_bbox:
            frames = [fr for fr in frames if fr.get("has_bbox", fr.get("detected", False))]
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
                fidx = max(0, t - (WINDOW-1-k))
                f2   = frames[fidx]
                cx   = f2.get("cx_det", 0.5)
                cy   = f2.get("cy_det", 0.5)
                area = f2.get("area_det", 0.05)
                hbb  = float(f2.get("has_bbox", f2.get("detected", False)))
                hist.extend([cx, cy, area, hbb])
            feat = hist + vis[t].cpu().tolist()
            X.append(feat)
            y.append(fr["gt_class"])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",    type=int,   default=300)
    parser.add_argument("--lr",        type=float, default=1e-3)
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--ann", type=str, default=str(ANN_PG448))
    parser.add_argument("--out", type=str, default=str(OUT_DIR))
    parser.add_argument("--filter-no-bbox", action="store_true",
                        help="has_bbox=False 프레임 학습 제외 (Axis3 어블레이션)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    ann_path = Path(args.ann); out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    if not ann_path.exists():
        print(f"ERROR: {ann_path} 없음. gen_pg448_annotation.py 먼저 실행하세요.")
        sys.exit(1)

    with open(ann_path) as f:
        ann = json.load(f)

    random.shuffle(ann)
    n_val = max(1, int(len(ann)*args.val_ratio))
    val_eps, train_eps = ann[:n_val], ann[n_val:]

    print(f"[DATA] Train {len(train_eps)} eps / Val {len(val_eps)} eps")
    print("[MODEL] Stage1 v2 CLIP 로드...")
    enc = FrozenCLIPV2(VLM_PATH, STAGE1_PT, device).to(device).eval()

    fnb = args.filter_no_bbox
    if fnb:
        print("[DATA] has_bbox=False 프레임 제외 (--filter-no-bbox)")
    print("[DATA] 학습 데이터 준비 중...")
    X_tr, y_tr = build_dataset(train_eps, enc, device, filter_no_bbox=fnb)
    X_va, y_va = build_dataset(val_eps,   enc, device, filter_no_bbox=fnb)
    print(f"  Train: {len(X_tr)}샘플 / Val: {len(X_va)}샘플")

    X_tr_t = torch.from_numpy(X_tr).to(device)
    y_tr_t = torch.from_numpy(y_tr).to(device)
    X_va_t = torch.from_numpy(X_va).to(device)
    y_va_t = torch.from_numpy(y_va).to(device)

    mlp   = ActionMLP(d_in=X_tr.shape[1]).to(device)
    opt   = torch.optim.Adam(mlp.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best_acc = 0.0
    print(f"\n[TRAIN] {args.epochs} epochs")
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
                            "source": "pg448", "exp": "exp67"},
                           str(out_dir / "action_mlp.pt"))
                print(f"    [BEST] {acc*100:.1f}% → 저장")

    print(f"\n최종 val_acc: {best_acc*100:.1f}%")
    print(f"체크포인트 → {out_dir}/action_mlp.pt")
    print(f"\n비교:")
    print(f"  exp65 (PG224 cx): val_acc=96.8%")
    print(f"  exp67 (PG448 cx): val_acc={best_acc*100:.1f}%")


if __name__ == "__main__":
    main()

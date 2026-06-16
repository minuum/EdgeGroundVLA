#!/usr/bin/env python3
"""
Exp54 Stage 2: Action Head (Stage 1 CLIP LoRA frozen)

Stage 1에서 basket 인식을 학습한 CLIP LoRA를 frozen으로 고정하고
navigation action만 별도로 학습한다.

입력: bbox_history(8×4=32) + clip_feat(1024) = 1056-dim
     (goal 벡터 제거 — Stage 1이 이미 시각적 위치 정보를 담고 있음)
출력: 8-class action

Usage:
  python3 scripts/train_exp54_stage2_action.py
  python3 scripts/train_exp54_stage2_action.py --epochs 300
"""

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VLM_PATH   = ROOT / ".vlms" / "kosmos-2-patch14-224"
DATA_PATH  = ROOT / "docs" / "v5" / "bbox_nav_exp46" / "bbox_dataset_full.json"
STAGE1_DIR = ROOT / "runs" / "v5_nav" / "mlp" / "exp54" / "stage1"
STAGE2_DIR = ROOT / "runs" / "v5_nav" / "mlp" / "exp54" / "stage2"
LORA_DIR   = STAGE1_DIR / "clip_lora_adapter"

CLASS_NAMES = ["STOP","FORWARD","LEFT","RIGHT","FWD+L","FWD+R","ROT_L","ROT_R"]
NUM_CLASSES = 8
WINDOW      = 8
VIS_DIM     = 1024
D_IN        = WINDOW * 4 + VIS_DIM   # 1056 (goal 없음)

PATH_TO_DIR = {
    "left_straight":"left",  "left_left":"left",   "left_right":"left",
    "center_straight":"center","center_left":"center","center_right":"center",
    "right_straight":"right", "right_left":"right", "right_right":"right",
}


# ──────────────────────────────────────────────
# 모델
# ──────────────────────────────────────────────

class FrozenCLIPLoRA(nn.Module):
    """Stage 1 LoRA 로드, 완전 frozen."""

    def __init__(self, vlm_path: Path, lora_dir: Path, device: torch.device):
        super().__init__()
        from transformers import AutoModelForVision2Seq, AutoProcessor
        from peft import PeftModel

        self.processor = AutoProcessor.from_pretrained(str(vlm_path))

        if not lora_dir.exists():
            raise FileNotFoundError(
                f"Stage 1 LoRA 없음: {lora_dir}\n"
                "먼저 Stage 1 학습 실행: python3 scripts/train_exp54_stage1_contrastive.py"
            )

        base = AutoModelForVision2Seq.from_pretrained(
            str(vlm_path),
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        self.vision_model = PeftModel.from_pretrained(base.vision_model, str(lora_dir))
        for p in self.vision_model.parameters():
            p.requires_grad = False
        print(f"[MODEL] Stage 1 LoRA 로드 완료 (frozen): {lora_dir}")

    @torch.no_grad()
    def encode(self, pil_images: list, device: torch.device) -> torch.Tensor:
        """PIL 이미지 리스트 → (N, 1024) float32."""
        inputs = self.processor(images=pil_images, return_tensors="pt")
        pv = inputs["pixel_values"].to(
            device,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        out = self.vision_model(pixel_values=pv)
        return out.last_hidden_state.mean(dim=1).float()  # (N, 1024)


class ActionMLP(nn.Module):
    def __init__(self, d_in: int = D_IN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 512), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(512, 256),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),  nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64),   nn.ReLU(),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────
# 데이터 유틸
# ──────────────────────────────────────────────

def load_images_from_h5(h5_path: str, indices: list) -> list:
    with h5py.File(h5_path, "r") as f:
        return [Image.fromarray(f["observations"]["images"][i]) for i in indices]


def build_bbox_feat(frames: list, t: int) -> np.ndarray:
    bbox = []
    for k in range(WINDOW):
        idx = max(0, t - (WINDOW - 1 - k))
        fr = frames[idx]
        bbox.extend([fr["cx"], fr["cy"], fr["area"], float(fr["has_bbox"])])
    return np.array(bbox, dtype=np.float32)


# ──────────────────────────────────────────────
# 학습
# ──────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[DEVICE] {device}")

    data = json.loads(DATA_PATH.read_text())
    ep_labels = [ep["path_type"] for ep in data]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(sss.split(np.zeros(len(data)), ep_labels))
    tr_eps = [data[i] for i in tr_idx]
    te_eps  = [data[i] for i in te_idx]
    print(f"Train: {len(tr_eps)} ep  Val: {len(te_eps)} ep")

    # 모델 초기화
    clip_enc = FrozenCLIPLoRA(VLM_PATH, LORA_DIR, device).to(device).eval()
    mlp      = ActionMLP(D_IN).to(device)

    # 클래스 가중치
    all_labels = [fr["gt_class"] for ep in tr_eps for fr in ep["frames"]]
    counts  = np.bincount(all_labels, minlength=NUM_CLASSES).astype(float)
    weights = np.where(counts > 0, 1.0 / (counts + 1e-6), 0.0)
    weights /= weights.sum() / NUM_CLASSES
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )

    optimizer = torch.optim.AdamW(mlp.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc, best_state = 0.0, None
    STAGE2_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        mlp.train()
        np.random.shuffle(tr_eps)

        batch_feats, batch_labels = [], []

        for ep in tr_eps:
            try:
                images = load_images_from_h5(ep["episode"], list(range(len(ep["frames"]))))
            except Exception as e:
                continue

            vis_feats = clip_enc.encode(images, device)  # (T, 1024)
            frames = ep["frames"]

            for t, frame in enumerate(frames):
                bbox = torch.tensor(build_bbox_feat(frames, t), device=device)
                feat = torch.cat([bbox, vis_feats[t]])
                batch_feats.append(feat)
                batch_labels.append(frame["gt_class"])

            if len(batch_feats) >= args.batch_size:
                _update(batch_feats, batch_labels, mlp, optimizer, criterion, device)
                batch_feats, batch_labels = [], []

        if batch_feats:
            _update(batch_feats, batch_labels, mlp, optimizer, criterion, device)

        scheduler.step()

        if epoch % 30 == 0 or epoch == args.epochs:
            acc = evaluate(clip_enc, mlp, te_eps, device)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
            print(f"  epoch {epoch:>3}  val_acc={acc:.4f}  best={best_acc:.4f}")

    mlp.load_state_dict(best_state)
    ckpt_path = STAGE2_DIR / "stage2_mlp.pt"
    torch.save({"mlp": best_state, "val_acc": best_acc, "d_in": D_IN}, str(ckpt_path))
    print(f"\n[SAVE] {ckpt_path}")
    print(f"최종 val_acc: {best_acc:.4f}")
    print(f"참고: Exp53={0.9468:.4f}  Exp49={0.9640:.4f}")
    return best_acc


def _update(feats, labels, mlp, optimizer, criterion, device):
    x = torch.stack(feats)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    optimizer.zero_grad()
    criterion(mlp(x), y).backward()
    optimizer.step()


@torch.no_grad()
def evaluate(clip_enc, mlp, eps, device) -> float:
    mlp.eval()
    correct = total = 0
    for ep in eps:
        try:
            images = load_images_from_h5(ep["episode"], list(range(len(ep["frames"]))))
        except:
            continue
        vis_feats = clip_enc.encode(images, device)
        frames = ep["frames"]
        for t, frame in enumerate(frames):
            bbox = torch.tensor(build_bbox_feat(frames, t), device=device)
            feat = torch.cat([bbox, vis_feats[t]]).unsqueeze(0)
            pred = mlp(feat).argmax(1).item()
            correct += int(pred == frame["gt_class"])
            total   += 1
    return correct / total if total > 0 else 0.0


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=300)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-3)
    args = p.parse_args()

    t0 = time.time()
    print("=" * 60)
    print("Exp54 Stage 2: Action Head (CLIP LoRA frozen)")
    print(f"d_in={D_IN}  epochs={args.epochs}")
    print("=" * 60)

    train(args)
    print(f"\n완료: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()

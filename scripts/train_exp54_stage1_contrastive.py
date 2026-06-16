#!/usr/bin/env python3
"""
Exp54 Stage 1: CLIP LoRA + Text Anchor Contrastive Learning

gray basket을 텍스트-이미지 정렬로 먼저 인식시킨다.
  - CLIP LoRA (layers 16-23) + image_proj(1024→256)
  - Kosmos-2 LM last-token → text_proj(2048→256) — LM frozen
  - 3개 고정 텍스트 앵커: left / center / right
  - loss: cosine similarity 기반 3-class CE (temperature=0.07)

검증: val 이미지가 올바른 텍스트 앵커에 가장 가까운지 (3-class retrieval acc)
목표: retrieval acc ≥ 80%  → "CLIP이 basket 위치를 본다" 근거

Usage:
  python3 scripts/train_exp54_stage1_contrastive.py
  python3 scripts/train_exp54_stage1_contrastive.py --epochs 200 --batch_size 16
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
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VLM_PATH  = ROOT / ".vlms" / "kosmos-2-patch14-224"
DATA_PATH = ROOT / "docs" / "v5" / "bbox_nav_exp46" / "bbox_dataset_full.json"
OUT_DIR   = ROOT / "runs" / "v5_nav" / "mlp" / "exp54" / "stage1"

LORA_R      = 16
LORA_ALPHA  = 32
LORA_LAYERS = list(range(16, 24))
LORA_TARGET = ["q_proj", "v_proj"]

VIS_DIM  = 1024
LM_DIM   = 2048
PROJ_DIM = 256
TEMPERATURE = 0.07

DIR_LABELS = {"left": 0, "center": 1, "right": 2}
DIRECTION_TEXTS = {
    "left":   "The gray basket is on the left side of the image",
    "center": "The gray basket is in the center of the image",
    "right":  "The gray basket is on the right side of the image",
}
PATH_TO_DIR = {
    "left_straight": "left",  "left_left": "left",   "left_right": "left",
    "center_straight":"center","center_left":"center","center_right":"center",
    "right_straight":"right",  "right_left":"right",  "right_right":"right",
}


# ──────────────────────────────────────────────
# 모델
# ──────────────────────────────────────────────

class CLIPLoRAEncoder(nn.Module):
    """Kosmos-2 vision encoder + LoRA(16-23) + projection head."""

    def __init__(self, vlm_path: Path, device: torch.device):
        super().__init__()
        from transformers import AutoModelForVision2Seq, AutoProcessor
        from peft import LoraConfig, get_peft_model

        self.processor = AutoProcessor.from_pretrained(str(vlm_path))
        base = AutoModelForVision2Seq.from_pretrained(
            str(vlm_path),
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        lora_cfg = LoraConfig(
            r=LORA_R, lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET,
            layers_to_transform=LORA_LAYERS,
            layers_pattern="layers",
            lora_dropout=0.05, bias="none",
        )
        self.vision_model = get_peft_model(base.vision_model, lora_cfg)
        self.vision_model.print_trainable_parameters()

        self.proj = nn.Linear(VIS_DIM, PROJ_DIM)

    def encode(self, pil_images: list, device: torch.device) -> torch.Tensor:
        """PIL 이미지 리스트 → (N, PROJ_DIM) L2-normalized."""
        inputs = self.processor(images=pil_images, return_tensors="pt")
        pv = inputs["pixel_values"].to(
            device,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        out = self.vision_model(pixel_values=pv)
        feat = out.last_hidden_state.mean(dim=1).float()  # (N, 1024)
        return F.normalize(self.proj(feat), dim=-1)       # (N, 256)


class TextAnchorEncoder(nn.Module):
    """Kosmos-2 LM last-token → projection head. LM은 frozen."""

    def __init__(self, vlm_path: Path, device: torch.device):
        super().__init__()
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(str(vlm_path))
        full_model = AutoModelForVision2Seq.from_pretrained(
            str(vlm_path),
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        )
        self.text_model = full_model.text_model.to(device)
        for p in self.text_model.parameters():
            p.requires_grad = False

        self.proj = nn.Linear(LM_DIM, PROJ_DIM)

    @torch.no_grad()
    def _raw_embed(self, text: str, device: torch.device) -> torch.Tensor:
        """텍스트 → (1, LM_DIM) float32 last-token hidden state."""
        inputs = self.processor.tokenizer(
            text, return_tensors="pt", add_special_tokens=True
        ).to(device)
        out = self.text_model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            output_hidden_states=True,
        )
        last_layer = out.hidden_states[-1]           # (1, seq, 2048)
        feat = last_layer[:, -1, :].float()          # (1, 2048)
        return feat

    def compute_anchors(self, device: torch.device) -> torch.Tensor:
        """3개 고정 앵커 raw embedding 계산. shape: (3, LM_DIM)."""
        raws = []
        for key in ("left", "center", "right"):
            raws.append(self._raw_embed(DIRECTION_TEXTS[key], device))
        return torch.cat(raws, dim=0)  # (3, 2048)

    def project(self, raw_anchors: torch.Tensor) -> torch.Tensor:
        """(3, LM_DIM) → (3, PROJ_DIM) L2-normalized."""
        return F.normalize(self.proj(raw_anchors), dim=-1)


# ──────────────────────────────────────────────
# 데이터 유틸
# ──────────────────────────────────────────────

def build_flat_records(data: list) -> list:
    """(h5_path, frame_idx, dir_label) flat 리스트."""
    records = []
    for ep in data:
        direction = PATH_TO_DIR.get(ep["path_type"])
        if direction is None:
            continue
        label = DIR_LABELS[direction]
        for fr in ep["frames"]:
            records.append((ep["episode"], fr["frame_idx"], label))
    return records


def load_image(h5_path: str, frame_idx: int) -> Image.Image:
    with h5py.File(h5_path, "r") as f:
        arr = f["observations"]["images"][frame_idx]
    return Image.fromarray(arr)


# ──────────────────────────────────────────────
# 학습
# ──────────────────────────────────────────────

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[DEVICE] {device}")

    data = json.loads(DATA_PATH.read_text())
    all_records = build_flat_records(data)
    print(f"[DATA] 전체 프레임: {len(all_records)}")

    # 에피소드 단위로 train/val 분할 (Exp53과 동일 split)
    ep_list = list({r[0] for r in all_records})
    ep_dir  = {}
    for ep in data:
        d = PATH_TO_DIR.get(ep["path_type"])
        if d:
            ep_dir[ep["episode"]] = d
    ep_labels = [ep_dir[e] for e in ep_list]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_ep_idx, te_ep_idx = next(sss.split(np.zeros(len(ep_list)), ep_labels))
    tr_eps = {ep_list[i] for i in tr_ep_idx}
    te_eps = {ep_list[i] for i in te_ep_idx}

    tr_records = [r for r in all_records if r[0] in tr_eps]
    te_records  = [r for r in all_records if r[0] in te_eps]
    print(f"Train: {len(tr_records)} frames  Val: {len(te_records)} frames")

    # 모델 초기화
    print("\n[MODEL] 초기화...")
    clip_enc  = CLIPLoRAEncoder(VLM_PATH, device).to(device)
    text_enc  = TextAnchorEncoder(VLM_PATH, device).to(device)

    print("[ANCHOR] 텍스트 앵커 계산...")
    anchor_raw = text_enc.compute_anchors(device)  # (3, 2048), frozen 이후 재사용
    anchor_raw = anchor_raw.detach()
    print(f"  anchor_raw shape: {anchor_raw.shape}")

    lora_params      = [p for p in clip_enc.vision_model.parameters() if p.requires_grad]
    proj_params      = list(clip_enc.proj.parameters()) + list(text_enc.proj.parameters())
    optimizer = torch.optim.AdamW([
        {"params": lora_params,  "lr": args.lr_lora},
        {"params": proj_params,  "lr": args.lr_proj},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc, best_state = 0.0, None
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        clip_enc.train(); text_enc.proj.train()

        np.random.shuffle(tr_records)
        total_loss, n_batches = 0.0, 0

        for i in range(0, len(tr_records), args.batch_size):
            batch = tr_records[i: i + args.batch_size]
            try:
                images = [load_image(h5, idx) for h5, idx, _ in batch]
            except Exception as e:
                continue

            labels = torch.tensor([lbl for _, _, lbl in batch],
                                   dtype=torch.long, device=device)

            # image → projected feat
            image_feats = clip_enc.encode(images, device)            # (B, 256)

            # text anchor → projected feat (재계산, text_proj 학습 중)
            anchor_feats = text_enc.project(anchor_raw.to(device))  # (3, 256)

            logits = image_feats @ anchor_feats.T / TEMPERATURE      # (B, 3)
            loss   = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item(); n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches if n_batches else 0.0

        if epoch % 20 == 0 or epoch == args.epochs:
            acc = evaluate(clip_enc, text_enc, anchor_raw, te_records, device)
            if acc > best_acc:
                best_acc = acc
                best_state = {
                    "clip_lora": {k: v.cpu().clone() for k, v in clip_enc.state_dict().items()},
                    "text_proj": {k: v.cpu().clone() for k, v in text_enc.proj.state_dict().items()},
                    "anchor_raw": anchor_raw.cpu().clone(),
                    "val_acc": best_acc,
                }
            print(f"  epoch {epoch:>3}  loss={avg_loss:.4f}  val_acc={acc:.4f}  best={best_acc:.4f}")

    # 저장
    lora_dir = OUT_DIR / "clip_lora_adapter"
    lora_dir.mkdir(parents=True, exist_ok=True)

    clip_enc.load_state_dict(best_state["clip_lora"])
    clip_enc.vision_model.save_pretrained(str(lora_dir))
    print(f"\n[SAVE] LoRA 어댑터: {lora_dir}")

    torch.save(best_state, str(OUT_DIR / "stage1_projs.pt"))
    print(f"[SAVE] proj heads + anchor: {OUT_DIR / 'stage1_projs.pt'}")
    print(f"\n최종 val_acc (retrieval): {best_acc:.4f}")
    return best_acc


@torch.no_grad()
def evaluate(clip_enc, text_enc, anchor_raw, records, device) -> float:
    clip_enc.eval(); text_enc.eval()
    anchor_feats = text_enc.project(anchor_raw.to(device))  # (3, 256)
    correct = total = 0

    for i in range(0, len(records), 32):
        batch = records[i: i + 32]
        try:
            images = [load_image(h5, idx) for h5, idx, _ in batch]
        except:
            continue
        labels = [lbl for _, _, lbl in batch]
        image_feats = clip_enc.encode(images, device)             # (B, 256)
        preds = (image_feats @ anchor_feats.T).argmax(dim=1)     # (B,)
        correct += (preds == torch.tensor(labels, device=device)).sum().item()
        total   += len(labels)

    return correct / total if total > 0 else 0.0


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=150)
    p.add_argument("--batch_size", type=int,   default=16)
    p.add_argument("--lr_lora",    type=float, default=2e-4)
    p.add_argument("--lr_proj",    type=float, default=1e-3)
    args = p.parse_args()

    t0 = time.time()
    print("=" * 60)
    print("Exp54 Stage 1: CLIP Contrastive — Gray Basket Recognition")
    print(f"epochs={args.epochs}  batch_size={args.batch_size}")
    print("=" * 60)

    val_acc = train(args)

    print(f"\n완료: {(time.time()-t0)/60:.1f}분")
    print(f"retrieval val_acc: {val_acc:.4f}  (목표: ≥0.80)")


if __name__ == "__main__":
    main()

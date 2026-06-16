#!/usr/bin/env python3
"""
Fine-tune Kosmos-2 grounding to reliably output 'gray basket' entity name + bbox.

Uses LoRA on text_model (q_proj, v_proj) so the vision features stay frozen.
Training data: bbox_truth_mini.json (72 manually annotated frames).
Optional: horizontal-flip augmentation (144 frames).

Target output format:
  " the {position} of the image.<phrase> gray basket</phrase>"
  "<object><patch_index_XXXX><patch_index_XXXX></object>"

Output: docs/v5/bbox_nav_step1/grounding_lora/

Usage:
    python3 scripts/finetune_kosmos2_grounding.py [--augment] [--epochs 50] [--lr 5e-5]
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers.models.kosmos2.processing_kosmos2 import coordinate_to_patch_index

BBOX_TRUTH_PATH = ROOT / "docs" / "v5" / "bbox_truth_mini.json"
ADAPTER_PATH    = ROOT / "docs" / "v5" / "bbox_nav_step1" / "grounding_lora"
GROUNDING_MODEL = ROOT / ".vlms" / "kosmos-2-patch14-224"
PROMPT_TEXT     = "<grounding>The gray basket is at"
NUM_PATCHES_SIDE = 32  # Kosmos-2: 32×32 patch grid for 224px input

_DATA_CANDIDATES = [
    Path("/home/billy/25-1kp/MoNaVLA/ROS_action/mobile_vla_dataset_v5"),
    Path("/home/minum/minum/26CS/MoNa-pi/mobile_vla_dataset_v5"),
    ROOT / "ROS_action" / "mobile_vla_dataset_v5",
]

POSITION_TEXT = {"left": "the left side of the image",
                 "center": "the center of the image",
                 "right": "the right side of the image"}


def resolve_data_dir() -> Path:
    import os
    ov = os.getenv("VLA_PROXY_DATA_DIR")
    if ov:
        return Path(ov)
    for c in _DATA_CANDIDATES:
        if c.exists() and any(c.glob("episode_*.h5")):
            return c
    return _DATA_CANDIDATES[-1]


def load_frame(data_dir: Path, episode: str, frame_idx: int) -> np.ndarray:
    matches = list(data_dir.glob(f"{episode}.h5"))
    if not matches:
        raise FileNotFoundError(f"H5 not found: {episode}")
    with h5py.File(matches[0], "r") as f:
        if "observations" in f and "images" in f["observations"]:
            return f["observations"]["images"][frame_idx].astype(np.uint8)
        return f["images"][frame_idx].astype(np.uint8)


def bbox_to_patch_tokens(bbox_xyxy: list[float]) -> tuple[str, str]:
    x1, y1, x2, y2 = bbox_xyxy
    # clamp to valid range and ensure x2>x1, y2>y1
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(1.0, x2), min(1.0, y2)
    x2 = max(x2, x1 + 1.0 / NUM_PATCHES_SIDE)
    y2 = max(y2, y1 + 1.0 / NUM_PATCHES_SIDE)
    i1, i2 = coordinate_to_patch_index((x1, y1, x2, y2), NUM_PATCHES_SIDE)
    return f"<patch_index_{str(i1).zfill(4)}>", f"<patch_index_{str(i2).zfill(4)}>"


def flip_bbox(bbox_xyxy: list[float]) -> list[float]:
    x1, y1, x2, y2 = bbox_xyxy
    return [1.0 - x2, y1, 1.0 - x1, y2]


def flip_position(pos: str) -> str:
    return {"left": "right", "center": "center", "right": "left"}[pos]


def build_target_text(coarse_position: str, bbox_xyxy: list[float]) -> str:
    pos_text = POSITION_TEXT[coarse_position]
    pt1, pt2 = bbox_to_patch_tokens(bbox_xyxy)
    return f" {pos_text}.<phrase> gray basket</phrase><object>{pt1}{pt2}</object>"


def prepare_sample(
    processor: AutoProcessor,
    pil_image: Image.Image,
    coarse_position: str,
    bbox_xyxy: list[float],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build full input_ids, labels, pixel_values for one training sample."""
    target_text = build_target_text(coarse_position, bbox_xyxy)

    # Encode prompt + image → prefix token IDs
    prefix_inputs = processor(text=PROMPT_TEXT, images=pil_image, return_tensors="pt")
    prefix_ids = prefix_inputs["input_ids"][0]  # (73,)
    pixel_values = prefix_inputs["pixel_values"][0]
    img_mask = prefix_inputs.get("image_embeds_position_mask", torch.zeros_like(prefix_ids.unsqueeze(0)))[0]

    # Tokenize target (no BOS, include EOS)
    target_ids = processor.tokenizer(
        target_text + processor.tokenizer.eos_token,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0]

    full_ids = torch.cat([prefix_ids, target_ids])
    labels = torch.cat([torch.full_like(prefix_ids, -100), target_ids])
    # Extend image_embeds_position_mask with zeros for target tokens
    full_img_mask = torch.cat([img_mask, torch.zeros(len(target_ids), dtype=img_mask.dtype)])

    return {
        "input_ids": full_ids.to(device),
        "labels": labels.to(device),
        "pixel_values": pixel_values.to(device),
        "image_embeds_position_mask": full_img_mask.to(device),
    }


def collate_fn(samples: list[dict]) -> dict[str, torch.Tensor]:
    """Pad a list of samples to the same length."""
    max_len = max(s["input_ids"].shape[0] for s in samples)
    device = samples[0]["input_ids"].device

    input_ids = torch.zeros(len(samples), max_len, dtype=torch.long, device=device)
    labels    = torch.full((len(samples), max_len), -100, dtype=torch.long, device=device)
    img_mask  = torch.zeros(len(samples), max_len, dtype=torch.long, device=device)
    pixel_values = torch.stack([s["pixel_values"] for s in samples])

    for i, s in enumerate(samples):
        L = s["input_ids"].shape[0]
        input_ids[i, :L] = s["input_ids"]
        labels[i, :L]    = s["labels"]
        img_mask[i, :L]  = s["image_embeds_position_mask"]

    return {
        "input_ids": input_ids,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_embeds_position_mask": img_mask,
        "attention_mask": (input_ids != 0).long(),
    }


@torch.no_grad()
def evaluate(model: nn.Module, processor: AutoProcessor,
             samples_raw: list[dict], device: torch.device) -> dict:
    """Run inference and check entity name + direction accuracy."""
    model.eval()
    basket_kw = ("basket", "gray box", "container", "bin", "laundry")
    correct_entity = 0
    correct_dir = 0
    n = len(samples_raw)

    for s in samples_raw:
        pil = Image.fromarray(s["image"]).convert("RGB")
        inp = processor(text=PROMPT_TEXT, images=pil, return_tensors="pt")
        inp = {k: v.to(device) for k, v in inp.items()}
        inp["pixel_values"] = inp["pixel_values"].to(
            torch.float16 if device.type == "cuda" else torch.float32
        )
        gen = model.generate(
            pixel_values=inp["pixel_values"],
            input_ids=inp["input_ids"],
            attention_mask=inp["attention_mask"],
            image_embeds=None,
            image_embeds_position_mask=inp.get("image_embeds_position_mask"),
            use_cache=True,
            max_new_tokens=64,
        )
        new_ids = gen[:, inp["input_ids"].shape[1]:]
        raw = processor.batch_decode(new_ids, skip_special_tokens=False)[0]
        caption, entities = processor.post_process_generation(raw)

        # Entity name check
        found_basket = any(
            any(k in ent.lower() for k in basket_kw)
            for ent, _, _ in entities
        )
        if found_basket:
            correct_entity += 1

        # Direction check from caption
        cap_lower = caption.lower()
        if s["coarse_position"] == "left" and ("left" in cap_lower):
            correct_dir += 1
        elif s["coarse_position"] == "center" and any(w in cap_lower for w in ("center", "middle", "front")):
            correct_dir += 1
        elif s["coarse_position"] == "right" and ("right" in cap_lower):
            correct_dir += 1

    return {
        "entity_acc": correct_entity / n,
        "direction_acc": correct_dir / n,
        "n": n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment", action="store_true", help="Add horizontal-flip augmentation")
    parser.add_argument("--epochs",  type=int,   default=50)
    parser.add_argument("--lr",      type=float, default=5e-5)
    parser.add_argument("--batch",   type=int,   default=4)
    parser.add_argument("--lora_r",  type=int,   default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--eval_only", action="store_true", help="Only evaluate existing adapter")
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    data_dir = resolve_data_dir()
    print(f"Data dir: {data_dir}")

    # ── Load annotations ───────────────────────────────────────────────────
    with open(BBOX_TRUTH_PATH) as f:
        truth = json.load(f)
    anns = truth["annotations"]

    raw_samples = []
    skipped = 0
    for ann in anns:
        cp = ann.get("coarse_position", "").lower()
        if cp not in POSITION_TEXT:
            skipped += 1
            continue
        try:
            img = load_frame(data_dir, ann["episode"], ann["frame_idx"])
        except FileNotFoundError:
            skipped += 1
            continue
        raw_samples.append({
            "image": img,
            "coarse_position": cp,
            "bbox_xyxy": ann["bbox_xyxy_norm"],
            "episode": ann["episode"],
        })
        if args.augment:
            raw_samples.append({
                "image": np.fliplr(img).copy(),
                "coarse_position": flip_position(cp),
                "bbox_xyxy": flip_bbox(ann["bbox_xyxy_norm"]),
                "episode": ann["episode"],
            })

    print(f"Loaded {len(raw_samples)} samples (skipped {skipped})")
    from collections import Counter
    print("Distribution:", Counter(s["coarse_position"] for s in raw_samples))

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"\nLoading Kosmos-2 from {GROUNDING_MODEL} ...")
    processor = AutoProcessor.from_pretrained(str(GROUNDING_MODEL))
    model = AutoModelForVision2Seq.from_pretrained(
        str(GROUNDING_MODEL),
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)

    # Eval-only mode: load existing adapter
    if args.eval_only:
        if not ADAPTER_PATH.exists():
            print("ERROR: No adapter found at", ADAPTER_PATH)
            sys.exit(1)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))
        model.eval()
        print("Loaded existing adapter. Running evaluation...")
        results = evaluate(model, processor, raw_samples, device)
        print(f"Entity match:  {results['entity_acc']:.3f}  ({results['n']} frames)")
        print(f"Direction acc: {results['direction_acc']:.3f}")
        return

    # ── Apply LoRA ─────────────────────────────────────────────────────────
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Freeze vision model and image-text projection
    for param in model.base_model.model.vision_model.parameters():
        param.requires_grad = False
    for param in model.base_model.model.image_to_text_projection.parameters():
        param.requires_grad = False

    # Cast LoRA parameters to float32 to avoid fp16 gradient overflow (NaN loss)
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()

    # ── Prepare dataset ────────────────────────────────────────────────────
    print("\nPreparing training samples...")
    # pixel_values stay fp16 (processed by frozen vision_model which is fp16)
    pv_dtype = torch.float16 if device.type == "cuda" else torch.float32
    prepared = []
    for i, s in enumerate(raw_samples):
        pil = Image.fromarray(s["image"]).convert("RGB")
        sample = prepare_sample(processor, pil, s["coarse_position"], s["bbox_xyxy"], device)
        sample["pixel_values"] = sample["pixel_values"].to(pv_dtype)
        prepared.append(sample)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(raw_samples)}", end="\r")
    print(f"  {len(prepared)}/{len(raw_samples)}")

    # ── Training loop ──────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    total_steps = args.epochs * math.ceil(len(prepared) / args.batch)
    warmup_steps = max(1, int(total_steps * 0.1))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\nTraining {args.epochs} epochs, batch={args.batch}, lr={args.lr} ...")
    import random
    model.train()
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        random.shuffle(prepared)
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, len(prepared), args.batch):
            batch = prepared[i : i + args.batch]
            feed = collate_fn(batch)

            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == "cuda")):
                outputs = model(
                    input_ids=feed["input_ids"],
                    attention_mask=feed["attention_mask"],
                    pixel_values=feed["pixel_values"],
                    image_embeds=None,
                    image_embeds_position_mask=feed["image_embeds_position_mask"],
                    labels=feed["labels"],
                )
            loss = outputs.loss.float()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        if epoch % 10 == 0 or epoch == args.epochs:
            lr_now = scheduler.get_last_lr()[0] * args.lr
            print(f"  epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  lr={lr_now:.2e}")

    # ── Evaluate ───────────────────────────────────────────────────────────
    print("\nEvaluating on training set...")
    results = evaluate(model, processor, raw_samples, device)
    print(f"Entity match:  {results['entity_acc']:.3f}  ({results['n']} frames)")
    print(f"Direction acc: {results['direction_acc']:.3f}")

    # ── Save adapter ───────────────────────────────────────────────────────
    ADAPTER_PATH.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(ADAPTER_PATH))
    print(f"\nSaved LoRA adapter → {ADAPTER_PATH}")


if __name__ == "__main__":
    main()

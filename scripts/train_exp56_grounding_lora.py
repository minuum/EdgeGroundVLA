#!/usr/bin/env python3
"""
Exp56: Grounding LoRA v2 — Scale-up with auto-detected bbox (2624 frames)

기존 finetune_kosmos2_grounding.py (72 수동 annotation)을
bbox_dataset_frame_level.json (2624 HSV auto-detected frames)로 확장.

주요 변경:
  - 데이터: cx_det/cy_det/area_det → xyxy 변환 (정사각형 근사)
  - 규모: ~750 frames (5 per episode, 150 ep) vs 72
  - 출력: runs/v5_nav/grounding/exp56/
  - val split: 에피소드 기준 20%

Usage:
    python3 scripts/train_exp56_grounding_lora.py
    python3 scripts/train_exp56_grounding_lora.py --frames-per-ep 10 --epochs 30
    python3 scripts/train_exp56_grounding_lora.py --eval-only
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers.models.kosmos2.processing_kosmos2 import coordinate_to_patch_index

FRAME_LEVEL_JSON = ROOT / "docs" / "v5" / "bbox_frame_level" / "bbox_dataset_frame_level.json"
GROUNDING_MODEL  = ROOT / ".vlms" / "kosmos-2-patch14-224"
OUT_DIR          = ROOT / "runs" / "v5_nav" / "grounding" / "exp56"
PROMPT_TEXT      = "<grounding>The gray basket is at"
NUM_PATCHES_SIDE = 32

_DATA_CANDIDATES = [
    Path("/home/billy/25-1kp/MoNaVLA/ROS_action/mobile_vla_dataset_v5"),
    Path("/home/minum/minum/26CS/MoNa-pi/mobile_vla_dataset_v5"),
    ROOT / "ROS_action" / "mobile_vla_dataset_v5",
]

POSITION_TEXT = {
    "left":   "the left side of the image",
    "center": "the center of the image",
    "right":  "the right side of the image",
}


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
    # episode may be a full path or just the stem name
    ep_path = Path(episode)
    if ep_path.exists():
        h5_path = ep_path
    else:
        # stem only — search in data_dir
        stem = ep_path.stem if ep_path.suffix == ".h5" else ep_path.name
        matches = list(data_dir.glob(f"{stem}.h5"))
        if not matches:
            raise FileNotFoundError(f"H5 not found: {episode}")
        h5_path = matches[0]
    with h5py.File(h5_path, "r") as f:
        if "observations" in f and "images" in f["observations"]:
            return f["observations"]["images"][frame_idx].astype(np.uint8)
        return f["images"][frame_idx].astype(np.uint8)


def cx_cy_area_to_xyxy(cx: float, cy: float, area: float) -> list[float]:
    side = math.sqrt(max(area, 1e-4))
    x1 = max(0.0, cx - side / 2)
    y1 = max(0.0, cy - side / 2)
    x2 = min(1.0, cx + side / 2)
    y2 = min(1.0, cy + side / 2)
    return [x1, y1, x2, y2]


def coarse_pos_from_cx(cx: float) -> str:
    if cx < 0.40:
        return "left"
    if cx > 0.60:
        return "right"
    return "center"


def bbox_to_patch_tokens(bbox_xyxy: list[float]) -> tuple[str, str]:
    x1, y1, x2, y2 = bbox_xyxy
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(1.0, x2), min(1.0, y2)
    x2 = max(x2, x1 + 1.0 / NUM_PATCHES_SIDE)
    y2 = max(y2, y1 + 1.0 / NUM_PATCHES_SIDE)
    i1, i2 = coordinate_to_patch_index((x1, y1, x2, y2), NUM_PATCHES_SIDE)
    return f"<patch_index_{str(i1).zfill(4)}>", f"<patch_index_{str(i2).zfill(4)}>"


def flip_bbox(bbox_xyxy: list[float]) -> list[float]:
    x1, y1, x2, y2 = bbox_xyxy
    return [1.0 - x2, y1, 1.0 - x1, y2]


def flip_pos(pos: str) -> str:
    return {"left": "right", "center": "center", "right": "left"}[pos]


def build_target_text(coarse_position: str, bbox_xyxy: list[float]) -> str:
    pos_text = POSITION_TEXT[coarse_position]
    pt1, pt2 = bbox_to_patch_tokens(bbox_xyxy)
    return f" {pos_text}.<phrase> gray basket</phrase><object>{pt1}{pt2}</object>"


def prepare_sample(processor, pil_image, coarse_position, bbox_xyxy, device):
    target_text = build_target_text(coarse_position, bbox_xyxy)
    prefix_inputs = processor(text=PROMPT_TEXT, images=pil_image, return_tensors="pt")
    prefix_ids  = prefix_inputs["input_ids"][0]
    pixel_values = prefix_inputs["pixel_values"][0]
    img_mask = prefix_inputs.get(
        "image_embeds_position_mask",
        torch.zeros_like(prefix_ids.unsqueeze(0))
    )[0]
    target_ids = processor.tokenizer(
        target_text + processor.tokenizer.eos_token,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0]
    full_ids = torch.cat([prefix_ids, target_ids])
    labels   = torch.cat([torch.full_like(prefix_ids, -100), target_ids])
    full_mask = torch.cat([img_mask, torch.zeros(len(target_ids), dtype=img_mask.dtype)])
    return {
        "input_ids": full_ids.to(device),
        "labels": labels.to(device),
        "pixel_values": pixel_values.to(device),
        "image_embeds_position_mask": full_mask.to(device),
    }


def collate_fn(samples):
    max_len = max(s["input_ids"].shape[0] for s in samples)
    device = samples[0]["input_ids"].device
    input_ids    = torch.zeros(len(samples), max_len, dtype=torch.long, device=device)
    labels       = torch.full((len(samples), max_len), -100, dtype=torch.long, device=device)
    img_mask     = torch.zeros(len(samples), max_len, dtype=torch.long, device=device)
    pixel_values = torch.stack([s["pixel_values"] for s in samples])
    for i, s in enumerate(samples):
        L = s["input_ids"].shape[0]
        input_ids[i, :L] = s["input_ids"]
        labels[i, :L]    = s["labels"]
        img_mask[i, :L]  = s["image_embeds_position_mask"]
    return {
        "input_ids":  input_ids,
        "labels":     labels,
        "pixel_values": pixel_values,
        "image_embeds_position_mask": img_mask,
        "attention_mask": (input_ids != 0).long(),
    }


@torch.no_grad()
def evaluate(model, processor, raw_samples, device):
    model.eval()
    basket_kw = ("basket", "gray box", "container", "bin", "laundry")
    correct_entity = 0
    correct_dir    = 0
    n = len(raw_samples)
    for s in raw_samples:
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

        found_basket = any(
            any(k in ent.lower() for k in basket_kw)
            for ent, _, _ in entities
        )
        if found_basket:
            correct_entity += 1

        cap_lower = caption.lower()
        cp = s["coarse_position"]
        if cp == "left" and "left" in cap_lower:
            correct_dir += 1
        elif cp == "center" and any(w in cap_lower for w in ("center", "middle", "front")):
            correct_dir += 1
        elif cp == "right" and "right" in cap_lower:
            correct_dir += 1

    return {
        "entity_acc":    correct_entity / max(n, 1),
        "direction_acc": correct_dir    / max(n, 1),
        "n": n,
    }


def load_dataset(data_dir: Path, frames_per_ep: int, augment: bool) -> list[dict]:
    with open(FRAME_LEVEL_JSON) as f:
        raw = json.load(f)

    samples = []
    skipped = 0
    for ep_data in raw:
        ep_name = ep_data["episode"]
        frames  = ep_data["frames"]

        # keep only detected frames
        det_frames = [fr for fr in frames if fr.get("detected")]
        if not det_frames:
            skipped += 1
            continue

        # evenly spaced sample
        if len(det_frames) <= frames_per_ep:
            chosen = det_frames
        else:
            idxs = [round(i * (len(det_frames) - 1) / (frames_per_ep - 1))
                    for i in range(frames_per_ep)]
            idxs = sorted(set(idxs))
            chosen = [det_frames[i] for i in idxs]

        for fr in chosen:
            cx   = fr["cx_det"]
            cy   = fr["cy_det"]
            area = fr["area_det"]
            if cx is None or cy is None or area is None:
                continue
            try:
                img_arr = load_frame(data_dir, ep_name, fr["frame_idx"])
            except (FileNotFoundError, OSError):
                skipped += 1
                continue

            cp   = coarse_pos_from_cx(cx)
            xyxy = cx_cy_area_to_xyxy(cx, cy, area)
            samples.append({
                "image":           img_arr,
                "coarse_position": cp,
                "bbox_xyxy":       xyxy,
                "episode":         ep_name,
            })
            if augment:
                samples.append({
                    "image":           np.fliplr(img_arr).copy(),
                    "coarse_position": flip_pos(cp),
                    "bbox_xyxy":       flip_bbox(xyxy),
                    "episode":         ep_name,
                })

    print(f"Loaded {len(samples)} samples (skipped {skipped} episodes)")
    from collections import Counter
    print("  position dist:", dict(Counter(s["coarse_position"] for s in samples)))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-per-ep", type=int, default=5,
                        help="Max frames sampled per episode (evenly spaced)")
    parser.add_argument("--augment",    action="store_true")
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--lr",         type=float, default=5e-5)
    parser.add_argument("--batch",      type=int,   default=6)
    parser.add_argument("--lora-r",     type=int,   default=8)
    parser.add_argument("--lora-alpha", type=int,   default=16)
    parser.add_argument("--val-ratio",  type=float, default=0.2)
    parser.add_argument("--eval-only",  action="store_true")
    parser.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(42)
    device   = torch.device(args.device)
    data_dir = resolve_data_dir()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Exp56: Grounding LoRA v2 (scale-up)")
    print(f"  Data dir : {data_dir}")
    print(f"  Frames/ep: {args.frames_per_ep}")
    print(f"  Augment  : {args.augment}")
    print(f"  Epochs   : {args.epochs}  LR={args.lr}  Batch={args.batch}")
    print("=" * 60)

    # ── Load dataset ───────────────────────────────────────────────────────
    all_samples = load_dataset(data_dir, args.frames_per_ep, args.augment)

    # Episode-level val split (deterministic)
    episodes = list({s["episode"] for s in all_samples})
    random.shuffle(episodes)
    n_val = max(1, int(len(episodes) * args.val_ratio))
    val_eps = set(episodes[:n_val])
    train_raw = [s for s in all_samples if s["episode"] not in val_eps]
    val_raw   = [s for s in all_samples if s["episode"] in val_eps]
    print(f"  Train={len(train_raw)}  Val={len(val_raw)}")

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"\nLoading Kosmos-2 from {GROUNDING_MODEL} ...")
    processor = AutoProcessor.from_pretrained(str(GROUNDING_MODEL))
    model = AutoModelForVision2Seq.from_pretrained(
        str(GROUNDING_MODEL),
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)

    if args.eval_only:
        if not (OUT_DIR / "adapter_config.json").exists():
            print("ERROR: No adapter at", OUT_DIR)
            sys.exit(1)
        model = PeftModel.from_pretrained(model, str(OUT_DIR))
        model.eval()
        print("\nEval on val set...")
        r = evaluate(model, processor, val_raw[:50], device)
        print(f"  entity_acc={r['entity_acc']:.3f}  direction_acc={r['direction_acc']:.3f}  n={r['n']}")
        return

    # ── LoRA ───────────────────────────────────────────────────────────────
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

    for param in model.base_model.model.vision_model.parameters():
        param.requires_grad = False
    for param in model.base_model.model.image_to_text_projection.parameters():
        param.requires_grad = False

    # fp32 for LoRA params to avoid fp16 gradient overflow
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.float()

    # ── Prepare training tensors ────────────────────────────────────────────
    print("\nPreparing training samples (this takes a few minutes)...")
    pv_dtype = torch.float16 if device.type == "cuda" else torch.float32
    prepared_train = []
    for i, s in enumerate(train_raw):
        pil = Image.fromarray(s["image"]).convert("RGB")
        samp = prepare_sample(processor, pil, s["coarse_position"], s["bbox_xyxy"], device)
        samp["pixel_values"] = samp["pixel_values"].to(pv_dtype)
        prepared_train.append(samp)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(train_raw)}", end="\r", flush=True)
    print(f"  {len(prepared_train)}/{len(train_raw)} prepared")

    # ── Training ───────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    total_steps  = args.epochs * math.ceil(len(prepared_train) / args.batch)
    warmup_steps = max(1, int(total_steps * 0.1))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\nTraining {args.epochs} epochs ...")
    model.train()
    global_step = 0
    best_dir_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        random.shuffle(prepared_train)
        epoch_loss = 0.0
        n_batches  = 0
        for i in range(0, len(prepared_train), args.batch):
            batch = prepared_train[i : i + args.batch]
            feed  = collate_fn(batch)
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
            n_batches  += 1
            global_step += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        if epoch % 5 == 0 or epoch == args.epochs:
            lr_now = scheduler.get_last_lr()[0] * args.lr
            val_r  = evaluate(model, processor, val_raw[:30], device)
            model.train()
            dir_acc = val_r["direction_acc"]
            print(
                f"  epoch {epoch:3d}/{args.epochs}"
                f"  loss={avg_loss:.4f}"
                f"  val_dir={dir_acc:.3f}"
                f"  val_entity={val_r['entity_acc']:.3f}"
                f"  lr={lr_now:.2e}"
            )
            if dir_acc >= best_dir_acc:
                best_dir_acc = dir_acc
                model.save_pretrained(str(OUT_DIR))
                print(f"    [BEST] saved → {OUT_DIR}")
        else:
            lr_now = scheduler.get_last_lr()[0] * args.lr
            print(f"  epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  lr={lr_now:.2e}")

    # ── Final eval ─────────────────────────────────────────────────────────
    print("\nFinal eval on full val set...")
    model.eval()
    final_r = evaluate(model, processor, val_raw, device)
    print(f"  entity_acc={final_r['entity_acc']:.3f}  direction_acc={final_r['direction_acc']:.3f}  n={final_r['n']}")

    result = {
        "exp": "exp56",
        "frames_per_ep": args.frames_per_ep,
        "augment": args.augment,
        "train_n": len(train_raw),
        "val_n":   len(val_raw),
        "best_dir_acc": best_dir_acc,
        "final_entity_acc": final_r["entity_acc"],
        "final_dir_acc":    final_r["direction_acc"],
    }
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved → {OUT_DIR}/results.json")
    print("=" * 60)
    print(f"Exp56 DONE  best_dir_acc={best_dir_acc:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

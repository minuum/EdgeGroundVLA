#!/usr/bin/env python3
"""
Exp57: PaliGemma Grounding LoRA — 복도 basket 위치 detection fine-tune

Zero-shot 65% → LoRA fine-tune → 90%+ 목표
Exp56(Kosmos-2) NaN 발산 문제 해결: SigLIP 기반 안정적 학습

형식:
  prefix : "detect gray basket"
  target : "<loc{y1}><loc{x1}><loc{y2}><loc{x2}> gray basket"
  (PaliGemma는 y1,x1,y2,x2 순, 0-1023 정수)

Usage:
    python3 scripts/train_exp57_paligemma_grounding.py
    python3 scripts/train_exp57_paligemma_grounding.py --frames-per-ep 8 --lr 1e-5
    python3 scripts/train_exp57_paligemma_grounding.py --eval-only
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
from PIL import Image

PALIGEMMA_PATH = Path.home() / ".cache/huggingface/hub" \
    / "models--google--paligemma-3b-pt-224" \
    / "snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c"

FRAME_LEVEL_JSON = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_frame_level.json"
OUT_DIR          = ROOT / "runs/v5_nav/grounding/exp57"
PROMPT           = "<image> detect gray basket"

_DATA_CANDIDATES = [
    Path("/home/minum/minum/26CS/MoNa-pi/mobile_vla_dataset_v5"),
    ROOT / "ROS_action/mobile_vla_dataset_v5",
]


def resolve_data_dir():
    import os
    ov = os.getenv("VLA_PROXY_DATA_DIR")
    if ov:
        return Path(ov)
    for c in _DATA_CANDIDATES:
        if c.exists() and any(c.glob("episode_*.h5")):
            return c
    return _DATA_CANDIDATES[-1]


def load_frame(episode: str, frame_idx: int, data_dir: Path) -> np.ndarray | None:
    ep = Path(episode)
    path = ep if ep.exists() else next(iter(data_dir.glob(f"{ep.stem}.h5")), None)
    if path is None:
        return None
    try:
        with h5py.File(path, "r") as f:
            if "observations" in f and "images" in f["observations"]:
                return f["observations"]["images"][frame_idx].astype(np.uint8)
            return f["images"][frame_idx].astype(np.uint8)
    except Exception:
        return None


def cx_cy_area_to_loc(cx, cy, area):
    """cx/cy/area → PaliGemma loc tokens (y1,x1,y2,x2 순서, 0-1023)"""
    side = math.sqrt(max(area, 1e-4))
    x1 = max(0.0, cx - side/2)
    y1 = max(0.0, cy - side/2)
    x2 = min(1.0, cx + side/2)
    y2 = min(1.0, cy + side/2)
    loc_y1 = int(y1 * 1023)
    loc_x1 = int(x1 * 1023)
    loc_y2 = int(y2 * 1023)
    loc_x2 = int(x2 * 1023)
    return f"<loc{loc_y1:04d}><loc{loc_x1:04d}><loc{loc_y2:04d}><loc{loc_x2:04d}> gray basket"


def flip_image(img: np.ndarray) -> np.ndarray:
    return np.fliplr(img).copy()


def flip_cx(cx: float) -> float:
    return 1.0 - cx


def load_dataset(data_dir: Path, frames_per_ep: int, augment: bool) -> list[dict]:
    with open(FRAME_LEVEL_JSON) as f:
        raw = json.load(f)

    samples = []
    skipped = 0
    for ep_data in raw:
        det = [fr for fr in ep_data["frames"] if fr.get("detected")
               and fr.get("cx_det") is not None]
        if not det:
            skipped += 1
            continue

        chosen = det if len(det) <= frames_per_ep else [
            det[round(i * (len(det)-1) / (frames_per_ep-1))]
            for i in range(frames_per_ep)
        ]
        chosen = list({fr["frame_idx"]: fr for fr in chosen}.values())  # dedup

        for fr in chosen:
            img = load_frame(ep_data["episode"], fr["frame_idx"], data_dir)
            if img is None:
                skipped += 1
                continue
            cx, cy, area = fr["cx_det"], fr["cy_det"], fr["area_det"]
            samples.append({
                "image": img,
                "target": cx_cy_area_to_loc(cx, cy, area),
                "cx": cx,
                "episode": ep_data["episode"],
            })
            if augment:
                samples.append({
                    "image": flip_image(img),
                    "target": cx_cy_area_to_loc(flip_cx(cx), cy, area),
                    "cx": flip_cx(cx),
                    "episode": ep_data["episode"],
                })

    print(f"  Loaded {len(samples)} samples (skipped {skipped} eps)")
    from collections import Counter
    pos = ["left" if s["cx"]<0.4 else ("right" if s["cx"]>0.6 else "center")
           for s in samples]
    print("  position:", dict(Counter(pos)))
    return samples


@torch.no_grad()
def evaluate(model, processor, raw_samples, device, max_eval=50):
    """hit rate + cx 오차 평가"""
    import re
    loc_re = re.compile(r"<loc(\d{4})>")
    model.eval()
    hits, cx_errs = 0, []
    samples = random.sample(raw_samples, min(max_eval, len(raw_samples)))

    for s in samples:
        pil = Image.fromarray(s["image"]).convert("RGB")
        inp = processor(text=PROMPT, images=pil, return_tensors="pt")
        inp = {k: v.to(device) for k, v in inp.items()}
        gen = model.generate(
            **inp,
            max_new_tokens=30,
            do_sample=False,
        )
        prefix_len = inp["input_ids"].shape[1]
        decoded = processor.decode(gen[0][prefix_len:], skip_special_tokens=False)

        vals = [int(v)/1023.0 for v in loc_re.findall(decoded)]
        if len(vals) >= 4:
            hits += 1
            # y1,x1,y2,x2 → cx
            pred_cx = (vals[1] + vals[3]) / 2
            cx_errs.append(abs(pred_cx - s["cx"]))

    n = len(samples)
    return {
        "hit_rate": hits / n,
        "cx_err":   sum(cx_errs) / max(len(cx_errs), 1),
        "n": n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-per-ep", type=int,   default=5)
    parser.add_argument("--augment",        action="store_true", default=True)
    parser.add_argument("--no-augment",     dest="augment", action="store_false")
    parser.add_argument("--epochs",         type=int,   default=20)
    parser.add_argument("--lr",             type=float, default=2e-5)
    parser.add_argument("--batch",          type=int,   default=4)
    parser.add_argument("--lora-r",         type=int,   default=8)
    parser.add_argument("--lora-alpha",     type=int,   default=16)
    parser.add_argument("--val-ratio",      type=float, default=0.15)
    parser.add_argument("--eval-only",      action="store_true")
    parser.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(42)
    device   = torch.device(args.device)
    data_dir = resolve_data_dir()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Exp57: PaliGemma Grounding LoRA")
    print(f"  backbone : paligemma-3b-pt-224")
    print(f"  data_dir : {data_dir}")
    print(f"  frames/ep: {args.frames_per_ep}  augment={args.augment}")
    print(f"  epochs   : {args.epochs}  lr={args.lr}  batch={args.batch}")
    print("=" * 60)

    # ── 데이터 로드 ────────────────────────────────────────────
    all_samples = load_dataset(data_dir, args.frames_per_ep, args.augment)

    eps = list({s["episode"] for s in all_samples})
    random.shuffle(eps)
    n_val   = max(1, int(len(eps) * args.val_ratio))
    val_eps = set(eps[:n_val])
    train_raw = [s for s in all_samples if s["episode"] not in val_eps]
    val_raw   = [s for s in all_samples if s["episode"] in val_eps]
    print(f"  Train={len(train_raw)}  Val={len(val_raw)}")

    # ── 모델 로드 ──────────────────────────────────────────────
    from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model

    print(f"\n[LOAD] {PALIGEMMA_PATH}")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    processor = PaliGemmaProcessor.from_pretrained(str(PALIGEMMA_PATH))
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        str(PALIGEMMA_PATH), torch_dtype=dtype
    ).to(device)

    if args.eval_only:
        if not (OUT_DIR / "adapter_config.json").exists():
            print("ERROR: No adapter at", OUT_DIR)
            sys.exit(1)
        model = PeftModel.from_pretrained(model, str(OUT_DIR))
        model.eval()
        print("\nEval-only mode")
        r = evaluate(model, processor, val_raw, device)
        print(f"  hit_rate={r['hit_rate']*100:.1f}%  cx_err={r['cx_err']:.3f}  n={r['n']}")
        return

    # ── LoRA 적용 ──────────────────────────────────────────────
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],   # Gemma 언어 모델 레이어
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 비전 타워·프로젝터 동결 (언어 모델만 LoRA)
    for p in model.base_model.model.vision_tower.parameters():
        p.requires_grad = False
    for p in model.base_model.model.multi_modal_projector.parameters():
        p.requires_grad = False

    # LoRA 파라미터 fp32로 업캐스트 (bfloat16 gradient 불안정 방지)
    for name, p in model.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()

    # ── 학습 샘플 텐서 준비 ────────────────────────────────────
    print("\n학습 샘플 준비 중...")
    pv_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    prepared = []
    for i, s in enumerate(train_raw):
        pil = Image.fromarray(s["image"]).convert("RGB")
        enc = processor(
            text=PROMPT,
            images=pil,
            suffix=s["target"],
            return_tensors="pt",
            padding="max_length",
            max_length=300,
            truncation=True,
        )
        prepared.append({
            "input_ids":      enc["input_ids"][0].to(device),
            "attention_mask": enc["attention_mask"][0].to(device),
            "pixel_values":   enc["pixel_values"][0].to(pv_dtype).to(device),
            "labels":         enc["labels"][0].to(device),
            "token_type_ids": enc.get("token_type_ids", torch.zeros_like(enc["input_ids"]))[0].to(device),
        })
        if (i+1) % 50 == 0:
            print(f"  {i+1}/{len(train_raw)}", end="\r", flush=True)
    print(f"  {len(prepared)}/{len(train_raw)} 준비 완료")

    # ── 학습 루프 ──────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    total_steps  = args.epochs * math.ceil(len(prepared) / args.batch)
    warmup_steps = max(1, int(total_steps * 0.1))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.05, 0.5 * (1 + math.cos(math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f"\n학습 시작: {args.epochs} epochs, batch={args.batch}, lr={args.lr}")
    model.train()
    best_hit = 0.0

    for epoch in range(1, args.epochs + 1):
        random.shuffle(prepared)
        ep_loss, n_batch = 0.0, 0

        for i in range(0, len(prepared), args.batch):
            batch = prepared[i: i+args.batch]
            max_len = max(b["input_ids"].shape[0] for b in batch)

            input_ids  = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
            attn_mask  = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
            labels     = torch.full((len(batch), max_len), -100, dtype=torch.long, device=device)
            tok_types  = torch.zeros(len(batch), max_len, dtype=torch.long, device=device)
            pv         = torch.stack([b["pixel_values"] for b in batch])

            for j, b in enumerate(batch):
                L = b["input_ids"].shape[0]
                input_ids[j, :L]  = b["input_ids"]
                attn_mask[j, :L]  = b["attention_mask"]
                labels[j, :L]     = b["labels"]
                tok_types[j, :L]  = b["token_type_ids"]

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    pixel_values=pv,
                    labels=labels,
                    token_type_ids=tok_types,
                )
            loss = outputs.loss.float()

            if torch.isnan(loss):
                print(f"\n  [WARN] NaN loss at epoch {epoch} batch {i} — skip")
                optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            ep_loss += loss.item()
            n_batch  += 1

        avg_loss = ep_loss / max(n_batch, 1)
        lr_now   = scheduler.get_last_lr()[0] * args.lr

        if epoch % 5 == 0 or epoch == args.epochs:
            r = evaluate(model, processor, val_raw, device)
            model.train()
            print(
                f"  epoch {epoch:3d}/{args.epochs}"
                f"  loss={avg_loss:.4f}"
                f"  hit={r['hit_rate']*100:.1f}%"
                f"  cx_err={r['cx_err']:.3f}"
                f"  lr={lr_now:.2e}"
            )
            if r["hit_rate"] >= best_hit:
                best_hit = r["hit_rate"]
                model.save_pretrained(str(OUT_DIR))
                print(f"    [BEST] hit={best_hit*100:.1f}% → saved")
        else:
            print(f"  epoch {epoch:3d}/{args.epochs}  loss={avg_loss:.4f}  lr={lr_now:.2e}")

    # ── 최종 평가 ──────────────────────────────────────────────
    print("\n최종 평가 (전체 val set)...")
    model.eval()
    final = evaluate(model, processor, val_raw, device, max_eval=len(val_raw))
    print(f"  hit_rate={final['hit_rate']*100:.1f}%  cx_err={final['cx_err']:.3f}  n={final['n']}")

    with open(OUT_DIR / "results.json", "w") as f:
        json.dump({
            "exp": "exp57",
            "backbone": "paligemma-3b-pt-224",
            "frames_per_ep": args.frames_per_ep,
            "augment": args.augment,
            "train_n": len(train_raw),
            "val_n":   len(val_raw),
            "best_hit_rate":     best_hit,
            "final_hit_rate":    final["hit_rate"],
            "final_cx_err":      final["cx_err"],
        }, f, indent=2)

    print("=" * 60)
    print(f"Exp57 완료  best_hit={best_hit*100:.1f}%  (zero-shot baseline: 65%)")
    print(f"adapter → {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

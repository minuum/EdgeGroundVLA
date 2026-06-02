#!/usr/bin/env python3
"""
Exp53 Vision Feature 추출 — LoRA-enhanced Kosmos-2 vision_model

Kosmos-2 vision_model에 clip_lora_adapter 적용 후
bbox_dataset_full.json의 150 에피소드 전체 프레임 feature 추출.
exp49와 동일한 1024-dim vis_feat, npz 포맷도 동일.

Usage:
  .venv/bin/python3 scripts/extract_vis_features_exp53.py
  .venv/bin/python3 scripts/extract_vis_features_exp53.py --dry_run
"""
import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HF_KOSMOS_PATH  = ROOT / ".vlms" / "kosmos-2-patch14-224"
ADAPTER_PATH    = ROOT / "runs" / "v5_nav" / "mlp" / "exp53" / "clip_lora_adapter"
EXP46_DIR       = ROOT / "docs" / "v5" / "bbox_nav_exp46"
DATA_DIR        = ROOT / "ROS_action" / "mobile_vla_dataset_v5"
OUT_DIR         = ROOT / "docs" / "v5" / "bbox_nav_exp53"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GROUNDING_PROMPT = "<grounding>The gray basket is at"


def load_model():
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from peft import get_peft_model, LoraConfig
    from safetensors.torch import load_file

    print(f"Loading Kosmos-2 from {HF_KOSMOS_PATH} ...")
    proc  = AutoProcessor.from_pretrained(str(HF_KOSMOS_PATH))
    model = AutoModelForVision2Seq.from_pretrained(
        str(HF_KOSMOS_PATH), torch_dtype=torch.float16
    ).cuda().eval()

    # PeftModel.from_pretrained fails on PEFT 0.11 vs adapter saved with PEFT 0.19
    # (unknown kwargs: alora_invocation_tokens, etc.)
    # Adapter was saved from Kosmos2VisionModel → apply to vision_model only.
    # Key remapping needed: PEFT 0.19 saves "lora_A.weight", PEFT 0.11 uses "lora_A.default.weight"
    print(f"Applying LoRA adapter from {ADAPTER_PATH} ...")
    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        layers_to_transform=list(range(16, 24)),
        layers_pattern="layers",
        inference_mode=True,
    )
    peft_vm = get_peft_model(model.vision_model, lora_cfg)

    weights_path = ADAPTER_PATH / "adapter_model.safetensors"
    raw_weights = load_file(str(weights_path))
    # remap: "lora_A.weight" → "lora_A.default.weight" (PEFT 0.19 → 0.11)
    remapped = {
        k.replace("lora_A.weight", "lora_A.default.weight")
         .replace("lora_B.weight", "lora_B.default.weight"): v
        for k, v in raw_weights.items()
    }
    missing, unexpected = peft_vm.load_state_dict(remapped, strict=False)
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:2]}")
    n_loaded = len(raw_weights) - len(unexpected)
    print(f"  Loaded {n_loaded}/{len(raw_weights)} adapter tensors")
    model.vision_model = peft_vm.eval()
    model.eval()

    return proc, model


def extract_feat(proc, model, pil_img: Image.Image) -> np.ndarray:
    """vision_model(image) → mean pool → (1024,) float32"""
    inputs = proc(text=GROUNDING_PROMPT, images=pil_img, return_tensors="pt")
    pv = inputs["pixel_values"].to("cuda").to(torch.float16)
    with torch.no_grad():
        vo = model.vision_model(pixel_values=pv)
        feat = vo.last_hidden_state[0].mean(0).float().cpu().numpy()
    return feat  # (1024,)


def resolve_h5(ep_key: str) -> Path:
    """minum 경로 → soda 로컬 경로 변환"""
    import os
    filename = os.path.basename(ep_key)
    stem, _ = os.path.splitext(filename)
    abs_data_dir = os.path.abspath(DATA_DIR)
    target_path = os.path.abspath(os.path.join(abs_data_dir, f"{stem}.h5"))
    if not target_path.startswith(abs_data_dir):
        raise ValueError(f"Path traversal detected: {ep_key}")
    return Path(target_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry_run", action="store_true", help="3 에피소드만 테스트")
    args = ap.parse_args()

    full_ds_path = EXP46_DIR / "bbox_dataset_full.json"
    if not full_ds_path.exists():
        print(f"ERROR: {full_ds_path} not found")
        sys.exit(1)

    dataset = json.loads(full_ds_path.read_text())
    if args.dry_run:
        dataset = dataset[:3]
        print(f"[dry_run] {len(dataset)} episodes only")
    else:
        print(f"Total episodes: {len(dataset)}")

    proc, model = load_model()

    all_feats: dict[str, np.ndarray] = {}  # ep_key → (n_frames, 1024)
    index: dict[str, int] = {}
    skipped = 0
    t0 = time.time()

    for i, ep_data in enumerate(dataset):
        ep_key = ep_data["episode"]
        h5_path = resolve_h5(ep_key)

        if not h5_path.exists():
            print(f"  SKIP [{i+1}/{len(dataset)}] {h5_path.name}: not found")
            skipped += 1
            continue

        try:
            with h5py.File(h5_path, "r") as f:
                images = f["observations"]["images"][:]  # (T, H, W, 3) uint8
        except Exception as e:
            print(f"  SKIP [{i+1}/{len(dataset)}] {h5_path.name}: {e}")
            skipped += 1
            continue

        n_frames = len(ep_data["frames"])
        ep_feats = np.zeros((n_frames, 1024), dtype=np.float32)

        for t, fr in enumerate(ep_data["frames"]):
            fidx = fr["frame_idx"]
            if fidx >= len(images):
                ep_feats[t] = 0.0
                continue
            pil = Image.fromarray(images[fidx].astype(np.uint8)).convert("RGB")
            ep_feats[t] = extract_feat(proc, model, pil)

        all_feats[ep_key] = ep_feats
        index[ep_key] = len(index)

        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (len(dataset) - i - 1)
        print(f"  [{i+1:3d}/{len(dataset)}] {h5_path.name} ({n_frames}f)  "
              f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    # Save
    out_npz = OUT_DIR / "vision_features.npz"
    out_idx = OUT_DIR / "vision_features_index.json"

    save_dict = {f"ep_{v}": all_feats[k] for k, v in index.items()}
    np.savez_compressed(str(out_npz), **save_dict)
    out_idx.write_text(json.dumps(index, indent=2))

    total = time.time() - t0
    print(f"\nDone. {len(index)} episodes, {skipped} skipped — {total:.1f}s")
    print(f"Saved: {out_npz}  ({out_npz.stat().st_size // 1024 // 1024}MB)")
    print(f"Saved: {out_idx}")


if __name__ == "__main__":
    main()

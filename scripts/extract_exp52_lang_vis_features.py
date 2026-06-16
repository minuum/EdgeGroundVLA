#!/usr/bin/env python3
"""
Exp52 lang-vis feature 추출
  HF Kosmos-2 + path_type별 instruction으로 joint forward
  → 마지막 레이어 image token hidden states 평균 (2048-dim)
  → docs/v5/bbox_nav_exp52/lang_vis_features.npz + lang_vis_features_index.json

Usage:
  .venv/bin/python3 scripts/extract_exp52_lang_vis_features.py
"""
import gc, json, sys, time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXP46_DIR = ROOT / "docs" / "v5" / "bbox_nav_exp46"
EXP52_DIR = ROOT / "docs" / "v5" / "bbox_nav_exp52"
HF_KOSMOS  = ROOT / ".vlms" / "kosmos-2-patch14-224"

EXP52_DIR.mkdir(parents=True, exist_ok=True)

OUT_NPZ   = EXP52_DIR / "lang_vis_features.npz"
OUT_IDX   = EXP52_DIR / "lang_vis_features_index.json"

INSTRUCTIONS = {
    "center_straight": "Navigate straight ahead to the basket in the center",
    "center_left":     "Navigate to the basket on the left",
    "center_right":    "Navigate to the basket on the right",
    "left_straight":   "Turn left and navigate straight to the basket",
    "left_left":       "Turn left and go to the basket on the left side",
    "left_right":      "Turn left then right to reach the basket",
    "right_straight":  "Turn right and navigate straight to the basket",
    "right_left":      "Turn right then left to reach the basket",
    "right_right":     "Turn right and go to the basket on the right side",
}


def extract_lang_vis(proc, model, pil_img, instruction, device):
    inputs = proc(text=instruction, images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
    hs   = out.hidden_states[-1]                            # (1, seq, 2048)
    mask = inputs["image_embeds_position_mask"][0].bool()   # (seq,)
    lv   = hs[0][mask].mean(0).float().cpu().numpy()        # (2048,)
    return lv


def main():
    t0 = time.time()
    print("=" * 60)
    print("Exp52: lang-vis feature 추출 (HF Kosmos-2 + instruction)")
    print("=" * 60)

    if OUT_NPZ.exists() and OUT_IDX.exists():
        idx = json.loads(OUT_IDX.read_text())
        print(f"이미 존재: {len(idx)}개 에피소드. 재추출하려면 파일 삭제 후 재실행.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print("HF Kosmos-2 로드 중...")
    proc  = AutoProcessor.from_pretrained(str(HF_KOSMOS))
    model = AutoModelForVision2Seq.from_pretrained(
        str(HF_KOSMOS), torch_dtype=torch.float16
    ).to(device).eval()
    print("모델 로드 완료.")

    bbox_data = json.loads((EXP46_DIR / "bbox_dataset_full.json").read_text())
    print(f"에피소드: {len(bbox_data)}")

    arrays = {}   # key: "ep_N"
    index  = {}   # ep_path → N

    for i, ep_data in enumerate(tqdm(bbox_data, desc="episodes")):
        ep_path = ep_data["episode"]
        pt      = ep_data["path_type"]
        instr   = INSTRUCTIONS.get(pt, "Navigate to the basket")
        frames  = ep_data["frames"]

        h5_path = Path(ep_path) if Path(ep_path).suffix == ".h5" else Path(ep_path + ".h5")
        if not h5_path.exists():
            print(f"  SKIP {ep_path}: H5 not found")
            continue

        with h5py.File(h5_path, "r") as f:
            if "observations" in f and "images" in f["observations"]:
                imgs = f["observations"]["images"][:]
            else:
                imgs = f["images"][:]

        n_frames = len(frames)
        lv_ep    = np.zeros((n_frames, 2048), dtype=np.float32)

        for t, fr in enumerate(frames):
            frame_idx = fr["frame_idx"]
            if frame_idx >= len(imgs):
                frame_idx = len(imgs) - 1
            pil = Image.fromarray(imgs[frame_idx].astype(np.uint8)).convert("RGB")
            lv_ep[t] = extract_lang_vis(proc, model, pil, instr, device)

        arrays[f"ep_{i}"] = lv_ep
        index[ep_path] = i

        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(bbox_data)}]  {elapsed/60:.1f}분 경과")

    np.savez_compressed(str(OUT_NPZ), **arrays)
    OUT_IDX.write_text(json.dumps(index, indent=2))

    elapsed = time.time() - t0
    print(f"\n완료: {len(index)}개 에피소드 ({elapsed/60:.1f}분)")
    print(f"  {OUT_NPZ}")
    print(f"  {OUT_IDX}")


if __name__ == "__main__":
    main()

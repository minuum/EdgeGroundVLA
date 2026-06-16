#!/usr/bin/env python3
"""
V5-2 chair 에피소드 annotation — base PaliGemma2로 `detect chair` grounding.

build_chair_dataset_v5_2.py 가 만든 template JSON을 읽어
cx_det/cy_det/area_det/has_bbox 필드를 base PG2 grounding으로 채운다.

full-frame 가드(area > 0.9) 포함.
vlen JPEG 이미지 처리.

Usage:
  .venv/bin/python3 scripts/gen_chair_pg2_annotation.py
  .venv/bin/python3 scripts/gen_chair_pg2_annotation.py \\
      --src docs/v5/bbox_chair/bbox_dataset_v5_2_template.json \\
      --out docs/v5/bbox_chair/bbox_dataset_chair_pg2_cx.json
"""
import json, re, io, sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import h5py

ROOT   = Path(__file__).resolve().parent.parent
PG2    = Path.home() / ".cache/huggingface/hub" \
         / "models--google--paligemma2-3b-mix-224" \
         / "snapshots/8e40ab4cc5df93dfb7fd2fff754bcdff8b62ee78"
LOC_RE = re.compile(r"<loc(\d{4})>")

DEFAULT_SRC = ROOT / "docs/v5/bbox_chair/bbox_dataset_v5_2_template.json"
DEFAULT_OUT = ROOT / "docs/v5/bbox_chair/bbox_dataset_chair_pg2_cx.json"


def load_model(device):
    from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
    dtype = torch.bfloat16
    print("[LOAD] base PaliGemma2 (LoRA 없음)...")
    proc  = PaliGemmaProcessor.from_pretrained(str(PG2))
    model = PaliGemmaForConditionalGeneration.from_pretrained(
                str(PG2), torch_dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
    return proc, model, dtype


@torch.no_grad()
def detect(model, proc, pil_img, device, dtype):
    inp = proc(text="detect chair", images=pil_img, return_tensors="pt").to(device)
    inp["pixel_values"] = inp["pixel_values"].to(dtype)
    gen = model.generate(**inp, max_new_tokens=48, do_sample=False)
    raw = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:],
                            skip_special_tokens=False)[0]
    locs = [int(v) / 1023.0 for v in LOC_RE.findall(raw)]
    if len(locs) >= 4:
        y1, x1, y2, x2 = locs[:4]
        cx   = (x1 + x2) / 2
        cy   = (y1 + y2) / 2
        area = (x2 - x1) * (y2 - y1)
        return cx, cy, area, True
    return 0.5, 0.5, 0.05, False


def read_frame(f, frame_idx: int) -> np.ndarray:
    """vlen JPEG bytes 또는 raw uint8 — 양쪽 처리."""
    raw = f["observations"]["images"][frame_idx]
    if hasattr(raw, "dtype") and raw.dtype != object and raw.ndim >= 2:
        return raw.astype(np.uint8)
    arr = np.frombuffer(bytes(raw), dtype=np.uint8)
    return np.array(Image.open(io.BytesIO(arr)).convert("RGB"))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC),
                    help="build_chair_dataset_v5_2.py 출력 template JSON")
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="grounding cx 주입 후 저장 경로")
    args = ap.parse_args()

    src_path = Path(args.src)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not src_path.exists():
        print(f"[ERROR] {src_path} 없음. build_chair_dataset_v5_2.py 먼저 실행하세요.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc, model, dtype = load_model(device)

    with open(src_path) as f:
        ann = json.load(f)

    total_frames = hit_frames = fullframe_filtered = 0
    new_ann = []

    for ep_i, ep in enumerate(ann):
        h5_path = Path(ep["episode"])
        if not h5_path.exists():
            print(f"  [SKIP] {h5_path.name} 없음")
            new_ann.append(ep)
            continue

        try:
            hf = h5py.File(str(h5_path), "r")
            n_imgs = len(hf["observations"]["images"])
        except Exception as e:
            print(f"  [SKIP] {h5_path.name}: {e}")
            new_ann.append(ep)
            continue

        new_frames = []
        seen_idx = {}  # frame_idx → (cx, cy, area, hit) 캐시 (STOP 주입 중복 감지)

        for fr in ep["frames"]:
            fidx = min(fr["frame_idx"], n_imgs - 1)

            if fidx not in seen_idx:
                img_np = read_frame(hf, fidx)
                pil = Image.fromarray(img_np).convert("RGB")
                cx, cy, area, hit = detect(model, proc, pil, device, dtype)

                # full-frame 가드
                if hit and area > 0.9:
                    hit = False
                    fullframe_filtered += 1

                seen_idx[fidx] = (cx, cy, area, hit)
                total_frames += 1
                hit_frames   += int(hit)
            else:
                cx, cy, area, hit = seen_idx[fidx]

            nf = dict(fr)
            nf["cx_det"]   = cx   if hit else 0.5
            nf["cy_det"]   = cy   if hit else 0.5
            nf["area_det"] = area if hit else 0.05
            nf["has_bbox"] = hit
            nf["detected"] = hit
            new_frames.append(nf)

        hf.close()

        new_ep = dict(ep)
        new_ep["frames"] = new_frames
        new_ann.append(new_ep)

        if (ep_i + 1) % 5 == 0 or (ep_i + 1) == len(ann):
            pct = hit_frames / total_frames * 100 if total_frames else 0
            print(f"  [{ep_i+1}/{len(ann)}] hit={hit_frames}/{total_frames} = {pct:.1f}%  "
                  f"fullframe_filtered={fullframe_filtered}")

    out_path.write_text(json.dumps(new_ann, ensure_ascii=False, indent=2))

    pct = hit_frames / total_frames * 100 if total_frames else 0
    print(f"\n완료: {hit_frames}/{total_frames} = {pct:.1f}% chair detected")
    print(f"full-frame 필터: {fullframe_filtered}프레임 제거")
    print(f"저장 → {out_path}")
    print(f"\n다음 단계 (학습):")
    print(f"  .venv/bin/python3 scripts/train_exp54_stage2_v2_action.py \\")
    print(f"      --augment --data {out_path} --tag chair_pg2_aug")


if __name__ == "__main__":
    main()

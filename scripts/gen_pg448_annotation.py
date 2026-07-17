#!/usr/bin/env python3
"""
exp67 Step1: PaliGemma2-448px로 전체 V5 에피소드 재주석.
gen_base_pg2_annotation.py(PG224 기반)와 동일 로직, 모델 경로만 448로 변경.
셀프라벨링 185프레임에서 PG448 검출률 98.9% vs PG224 73.5% 확인 → 어노테이션 품질 개선.

Usage:
  .venv/bin/python3 scripts/gen_pg448_annotation.py
"""
import json, re, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import h5py

ROOT   = Path(__file__).resolve().parent.parent
ANN    = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_frame_level.json"
PG448  = Path.home() / ".cache/huggingface/hub" \
         / "models--google--paligemma2-3b-mix-448" \
         / "snapshots/1406c92ec87d32cc6b983239278901b904ba7a51"
OUT    = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_pg448_cx.json"
LOC_RE = re.compile(r"<loc(\d{4})>")


def load_model(device):
    from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
    dtype = torch.bfloat16
    print(f"[LOAD] PaliGemma2-448 from {PG448}")
    proc  = PaliGemmaProcessor.from_pretrained(str(PG448))
    model = PaliGemmaForConditionalGeneration.from_pretrained(
                str(PG448), torch_dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
    print("  로드 완료")
    return proc, model, dtype


@torch.no_grad()
def detect(model, proc, img_np, device, dtype):
    pil = Image.fromarray(img_np).convert("RGB")
    inp = proc(text="detect gray basket", images=pil, return_tensors="pt").to(device)
    inp["pixel_values"] = inp["pixel_values"].to(dtype)
    gen = model.generate(**inp, max_new_tokens=48, do_sample=False)
    raw = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:],
                             skip_special_tokens=False)[0]
    locs = [int(v)/1023.0 for v in LOC_RE.findall(raw)]
    if len(locs) >= 4:
        y1, x1, y2, x2 = locs[:4]
        return (x1+x2)/2, (y1+y2)/2, (x2-x1)*(y2-y1), True
    return 0.5, 0.5, 0.05, False


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ANN))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out_path = Path(args.out)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc, model, dtype = load_model(device)

    with open(args.src) as f:
        ann = json.load(f)

    total_frames = hit_frames = 0
    new_ann = []

    for ep_i, ep in enumerate(ann):
        h5_path = Path(ep["episode"])
        if not h5_path.exists():
            new_ann.append(ep)
            continue

        with h5py.File(str(h5_path), "r") as f:
            # V5: observations/images, V6(트랙A): images 최상위
            imgs = (f["observations"]["images"] if "observations" in f else f["images"])[:]

        new_frames = []
        for fr in ep["frames"]:
            fidx = fr["frame_idx"]
            img_np = imgs[min(fidx, len(imgs)-1)].astype("uint8")
            cx_pg, cy_pg, area_pg, hit = detect(model, proc, img_np, device, dtype)

            # 오탐 필터 (gen_base_pg2_annotation.py 동일)
            if hit and (cy_pg < 0.35 or area_pg < 0.010 or area_pg > 0.9):
                hit = False

            new_fr = dict(fr)
            new_fr["cx_det_hsv"]   = fr.get("cx_det", 0.5)
            new_fr["cy_det_hsv"]   = fr.get("cy_det", 0.5)
            new_fr["area_det_hsv"] = fr.get("area_det", 0.05)
            new_fr["cx_det"]   = cx_pg if hit else 0.5
            new_fr["cy_det"]   = cy_pg if hit else 0.5
            new_fr["area_det"] = area_pg if hit else 0.05
            new_fr["detected"] = hit
            new_fr["has_bbox"] = hit

            total_frames += 1
            hit_frames   += int(hit)
            new_frames.append(new_fr)

        new_ep = dict(ep)
        new_ep["frames"] = new_frames
        new_ann.append(new_ep)

        if (ep_i+1) % 10 == 0:
            print(f"  [{ep_i+1}/{len(ann)}] hit={hit_frames}/{total_frames} ({hit_frames/total_frames*100:.1f}%)")

    with open(out_path, "w") as f:
        json.dump(new_ann, f, indent=2, ensure_ascii=False)

    print(f"\n완료: {hit_frames}/{total_frames} = {hit_frames/total_frames*100:.1f}% PG448 detected")
    print(f"저장 → {out_path}")
    print(f"\n비교:")
    print(f"  PG224 (exp65): 2519/2626 = 95.9%")
    print(f"  PG448 (exp67): {hit_frames}/{total_frames} = {hit_frames/total_frames*100:.1f}%")


if __name__ == "__main__":
    main()

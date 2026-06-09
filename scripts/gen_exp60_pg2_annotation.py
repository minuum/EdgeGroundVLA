#!/usr/bin/env python3
"""
Exp60 Step1: PaliGemma2 Exp59 LoRA로 전체 V5 에피소드 재주석
HSV cx_det → PG2 grounding cx로 교체

이후 Stage2 MLP를 이 annotation으로 재학습 → Exp60
"""
import json, re, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import h5py

ROOT   = Path(__file__).resolve().parent.parent
ANN    = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_frame_level.json"
PG2    = Path.home() / ".cache/huggingface/hub" \
         / "models--google--paligemma2-3b-mix-224" \
         / "snapshots/8e40ab4cc5df93dfb7fd2fff754bcdff8b62ee78"
EXP59  = ROOT / "runs/v5_nav/grounding/exp59"
OUT    = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_pg2_cx.json"
LOC_RE = re.compile(r"<loc(\d{4})>")


def load_model(device):
    from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
    from peft import PeftModel
    dtype = torch.bfloat16
    print("[LOAD] PaliGemma2 Exp59 LoRA...")
    proc  = PaliGemmaProcessor.from_pretrained(str(PG2))
    base  = PaliGemmaForConditionalGeneration.from_pretrained(
                str(PG2), torch_dtype=dtype, low_cpu_mem_usage=True).to(device)
    model = PeftModel.from_pretrained(base, str(EXP59)).eval()
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc, model, dtype = load_model(device)

    with open(ANN) as f:
        ann = json.load(f)

    total_frames = hit_frames = 0
    new_ann = []

    for ep_i, ep in enumerate(ann):
        h5_path = Path(ep["episode"])
        if not h5_path.exists():
            new_ann.append(ep)
            continue

        with h5py.File(str(h5_path), "r") as f:
            imgs = f["observations"]["images"][:]

        new_frames = []
        for fr in ep["frames"]:
            fidx = fr["frame_idx"]
            img_np = imgs[min(fidx, len(imgs)-1)].astype("uint8")
            cx_pg2, cy_pg2, area_pg2, hit = detect(model, proc, img_np, device, dtype)

            # ── 오탐 필터 A+B ────────────────────────────────────────────
            # A: cy < 0.35 → 이미지 상단 (basket은 항상 바닥에 있음)
            # B: area < 0.010 → bbox가 너무 작음 (노이즈/오탐)
            if hit and (cy_pg2 < 0.35 or area_pg2 < 0.010):
                hit = False  # 오탐으로 간주, 미검출 처리

            new_fr = dict(fr)
            # HSV 값을 pg2 값으로 교체 (원본도 보존)
            new_fr["cx_det_hsv"]   = fr.get("cx_det", 0.5)
            new_fr["cy_det_hsv"]   = fr.get("cy_det", 0.5)
            new_fr["area_det_hsv"] = fr.get("area_det", 0.05)
            new_fr["cx_det"]   = cx_pg2 if hit else 0.5
            new_fr["cy_det"]   = cy_pg2 if hit else 0.5
            new_fr["area_det"] = area_pg2 if hit else 0.05
            new_fr["detected"] = hit
            new_fr["has_bbox"] = hit

            total_frames += 1
            hit_frames   += int(hit)
            new_frames.append(new_fr)

        new_ep = dict(ep)
        new_ep["frames"] = new_frames
        new_ann.append(new_ep)

        if (ep_i+1) % 10 == 0:
            print(f"  [{ep_i+1}/{len(ann)}] hit={hit_frames}/{total_frames} = {hit_frames/total_frames*100:.1f}%")

    with open(OUT, "w") as f:
        json.dump(new_ann, f, indent=2, ensure_ascii=False)

    print(f"\n완료: {hit_frames}/{total_frames} = {hit_frames/total_frames*100:.1f}% PG2 detected")
    print(f"저장 → {OUT}")


if __name__ == "__main__":
    main()

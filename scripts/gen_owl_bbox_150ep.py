#!/usr/bin/env python3
"""운영 계보용 OWL bbox 데이터셋 — exp66 학습 데이터(150ep) 전체를 OWL-v2(th0.25)로 재-그라운딩.

원본: docs/v5/bbox_frame_level/bbox_dataset_base_pg2_cx.json (exp66/71 학습 소스)
출력: docs/v5/bbox_nav_owl/bbox_dataset_owl_150ep.json — 같은 스키마, cx_det/cy_det/area_det/has_bbox만 교체

Usage: .venv/bin/python3 scripts/gen_owl_bbox_150ep.py
"""
import copy
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_base_pg2_cx.json"
OUT = ROOT / "docs/v5/bbox_nav_owl/bbox_dataset_owl_150ep.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OWL_THRESH = 0.25


def main():
    src = json.loads(SRC.read_text())
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    print("[OWL-v2] 로딩...")
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble").to(DEVICE).eval()

    out = []
    n_frames = sum(len(ep["frames"]) for ep in src)
    done = 0
    skipped = 0
    t0 = time.time()
    for ep in src:
        h5p = Path(ep["episode"])
        if not h5p.exists():
            skipped += 1
            continue
        e = copy.deepcopy(ep)
        with h5py.File(str(h5p)) as f:
            imgs = f["observations"]["images"][:]
        for fr in e["frames"]:
            img = Image.fromarray(imgs[fr["frame_idx"]].astype("uint8")).convert("RGB")
            W, H = img.width, img.height
            inp = proc(text=[["gray laundry basket"]], images=img, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                o = model(**inp)
            res = proc.post_process_object_detection(
                o, threshold=OWL_THRESH, target_sizes=[(H, W)])[0]
            if len(res["boxes"]) == 0:
                fr["cx_det"], fr["cy_det"], fr["area_det"], fr["has_bbox"] = 0.5, 0.5, 0.05, False
            else:
                best = int(res["scores"].argmax())
                x1, y1, x2, y2 = res["boxes"][best].cpu().tolist()
                x1, x2, y1, y2 = x1 / W, x2 / W, y1 / H, y2 / H
                fr["cx_det"] = (x1 + x2) / 2
                fr["cy_det"] = (y1 + y2) / 2
                fr["area_det"] = (x2 - x1) * (y2 - y1)
                fr["has_bbox"] = True
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{n_frames} ({time.time()-t0:.0f}s)")
        out.append(e)

    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    hb = sum(1 for e in out for f in e["frames"] if f["has_bbox"])
    tot = sum(len(e["frames"]) for e in out)
    print(f"저장: {OUT}  ({len(out)}ep, 스킵 {skipped}, has_bbox {hb}/{tot} = {100*hb/tot:.1f}%)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Step2 학습 데이터(bbox_dataset.json 45ep/794프레임)를 OWL-v2(th 0.25)로 재-그라운딩.

원본과 동일한 에피소드/프레임/gt_class를 유지하고 cx/cy/area/has_bbox만 OWL 출력으로 교체
→ docs/v5/bbox_nav_owl/bbox_dataset_owl.json
미검출(score<0.25) 시 has_bbox=False, cx=0.5, cy=0.5, area=0.05 (원본 PG2 관례 동일).
마지막에 원본 vs OWL의 has_bbox/cx 분포 비교 리포트 출력 (plan 리스크 항목).

Usage: .venv/bin/python3 scripts/gen_bbox_dataset_owl.py
"""
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "ROS_action" / "mobile_vla_dataset_v5"
SRC = ROOT / "docs" / "v5" / "bbox_nav_step1" / "bbox_dataset.json"
OUT_DIR = ROOT / "docs" / "v5" / "bbox_nav_owl"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "bbox_dataset_owl.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OWL_THRESH = 0.25


def main():
    src = json.loads(SRC.read_text())
    n_frames = sum(len(ep["frames"]) for ep in src)
    print(f"원본: {len(src)} ep / {n_frames} frames")

    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    print("[OWL-v2] 로딩...")
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble").to(DEVICE).eval()

    out_eps = []
    skipped = []
    done = 0
    t0 = time.time()
    for ep in src:
        h5p = DATA_DIR / f"{ep['episode']}.h5"
        if not h5p.exists():
            # billy 머신에만 있는 에피소드 — 비교 시 양쪽 모두 제외 (train_step2_owl_head.py에서 교집합 처리)
            skipped.append(ep["episode"])
            continue
        new_frames = []
        with h5py.File(h5p, "r") as h:
            imgs = h["observations"]["images"]
            for fr in ep["frames"]:
                img = Image.fromarray(np.array(imgs[fr["frame_idx"]]).astype(np.uint8)).convert("RGB")
                W, H = img.width, img.height
                inp = proc(text=[["gray laundry basket"]], images=img, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    o = model(**inp)
                res = proc.post_process_object_detection(
                    o, threshold=OWL_THRESH, target_sizes=[(H, W)])[0]
                nf = {"frame_idx": fr["frame_idx"], "gt_class": fr["gt_class"]}
                if len(res["boxes"]) == 0:
                    nf.update({"cx": 0.5, "cy": 0.5, "area": 0.05, "has_bbox": False})
                else:
                    best = int(res["scores"].argmax())
                    x1, y1, x2, y2 = res["boxes"][best].cpu().tolist()
                    x1, x2, y1, y2 = x1 / W, x2 / W, y1 / H, y2 / H
                    nf.update({"cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
                               "area": (x2 - x1) * (y2 - y1), "has_bbox": True,
                               "score": float(res["scores"][best])})
                new_frames.append(nf)
                done += 1
                if done % 100 == 0:
                    print(f"  {done}/{n_frames} ({time.time()-t0:.0f}s)")
        out_eps.append({"path_type": ep["path_type"], "episode": ep["episode"], "frames": new_frames})

    OUT.write_text(json.dumps(out_eps, indent=2, ensure_ascii=False))
    print(f"저장: {OUT}  (스킵 {len(skipped)}개: {skipped})")

    # 분포 비교 리포트
    def stats(eps):
        fs = [f for e in eps for f in e["frames"]]
        hb = sum(1 for f in fs if f["has_bbox"])
        cxs = [f["cx"] for f in fs if f["has_bbox"]]
        return hb, len(fs), (np.mean(cxs) if cxs else 0), (np.std(cxs) if cxs else 0)

    for name, eps in [("원본(PG2계열)", src), ("OWL th0.25", out_eps)]:
        hb, n, mu, sd = stats(eps)
        print(f"{name}: has_bbox {hb}/{n} ({100*hb/n:.1f}%)  cx μ={mu:.3f} σ={sd:.3f}")


if __name__ == "__main__":
    main()

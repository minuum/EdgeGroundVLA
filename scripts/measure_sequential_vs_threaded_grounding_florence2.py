#!/usr/bin/env python3
"""Jetson Orin NX — grounder.run()(OWL-v2) vs Florence-2 vision_tower 순차/병렬 실측.

측정_sequential_vs_threaded_grounding_vision.py(Kosmos-2)의 짝 스크립트.
Kosmos-2 encode_image()를 Florence-2 vision_tower.forward_features_unpool()로
바꿔서 동일한 순차/병렬 비교를 한다 — "Florence-2로 백본을 바꿔도 그라운딩
뒤에 숨어서 cadence 저하가 거의 없어지는가"를 추정이 아니라 직접 실측.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("VLA_OWLV2_FP16", "1")
os.environ.setdefault("VLA_OWLV2_THRESH", "0.2")
os.environ.setdefault("VLA_OWLV2_AREA_SCALE", "3.0")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from robovlm_nav.serve.stage2_v2_inference_server import OwlV2Grounder

N_WARMUP = 10
N_ITERS = 100
PHRASE = "gray basket"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    print("Florence-2-base 로딩 (fp16)...", flush=True)
    florence = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base", trust_remote_code=True,
        torch_dtype=torch.float16).to(device).eval()
    florence_proc = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
    vision_tower = florence.vision_tower

    print("OwlV2Grounder 로딩...", flush=True)
    grounder = OwlV2Grounder(device)
    grounder._ensure_loaded()

    image_rgb = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
    pil = Image.fromarray(image_rgb).convert("RGB")
    pv = florence_proc(images=pil, text="<OD>", return_tensors="pt")["pixel_values"].to(device, torch.float16)

    def run_vision():
        with torch.no_grad():
            return vision_tower.forward_features_unpool(pv)

    print(f"워밍업 {N_WARMUP}회...", flush=True)
    for _ in range(N_WARMUP):
        grounder.run(image_rgb, phrase=PHRASE)
        run_vision()
    torch.cuda.synchronize()

    print(f"순차 측정 {N_ITERS}회...", flush=True)
    seq_ms, seq_ground_ms, seq_vis_ms = [], [], []
    for _ in range(N_ITERS):
        t0 = time.time()
        grounder.run(image_rgb, phrase=PHRASE)
        torch.cuda.synchronize()
        tg1 = time.time()
        run_vision()
        torch.cuda.synchronize()
        t1 = time.time()
        seq_ms.append((t1 - t0) * 1000)
        seq_ground_ms.append((tg1 - t0) * 1000)
        seq_vis_ms.append((t1 - tg1) * 1000)
    torch.cuda.synchronize()

    print(f"병렬 측정 {N_ITERS}회...", flush=True)
    par_ms = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        for _ in range(N_ITERS):
            t0 = time.time()
            fut_g = ex.submit(grounder.run, image_rgb, phrase=PHRASE)
            fut_v = ex.submit(run_vision)
            fut_g.result()
            fut_v.result()
            t1 = time.time()
            par_ms.append((t1 - t0) * 1000)
    torch.cuda.synchronize()

    def stats(xs):
        arr = np.array(xs)
        return {"mean": float(arr.mean()), "median": float(np.median(arr)),
                "p95": float(np.percentile(arr, 95)), "min": float(arr.min()), "max": float(arr.max())}

    seq_stats = stats(seq_ms)
    par_stats = stats(par_ms)
    ideal_max = max(stats(seq_ground_ms)["mean"], stats(seq_vis_ms)["mean"])

    out = {
        "device_str": "Jetson Orin NX",
        "n_warmup": N_WARMUP,
        "n_iters": N_ITERS,
        "grounder": "owlv2 (fp16, thresh=0.2)",
        "vision_encoder": "Florence-2-base vision_tower.forward_features_unpool (fp16)",
        "sequential_ms": seq_stats,
        "sequential_grounding_component_ms": stats(seq_ground_ms),
        "sequential_vision_component_ms": stats(seq_vis_ms),
        "threaded_ms": par_stats,
        "ideal_max_ab_ms": ideal_max,
        "ratio_threaded_over_sequential": par_stats["mean"] / seq_stats["mean"],
        "ratio_idealmax_over_sequential": ideal_max / seq_stats["mean"],
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))

    out_path = ROOT / "docs/v5/detector/sequential_vs_threaded_grounding_florence2_jetson.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()

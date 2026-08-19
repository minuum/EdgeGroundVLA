#!/usr/bin/env python3
"""Jetson Orin NX — Florence-2-base vision_tower.forward_features_unpool() 지연 측정.

minum 요청 (2026-08-19, docs/DATASET_V6_STATUS.md): Kosmos-2 vision_model 대비
Florence-2가 GB10에서 파라미터 3.35배 작음에도 11% 느렸음(576 vs 256 vision
tokens) — Jetson Orin NX 실측치 확인용. fp32/fp16 둘 다, 배치=1, 720x1280 입력
(minum 원 스크립트 scripts/detector_florence2_backbone.py의 measure_latency()와
동일 방식: <OD> task prompt, processor 기본 리사이즈).
"""
import json
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

DEV = "cuda"
N_WARMUP = 10
N_ITERS = 50


def measure(dtype):
    m = AutoModelForCausalLM.from_pretrained(
        "microsoft/Florence-2-base", trust_remote_code=True,
        torch_dtype=dtype).to(DEV).eval()
    proc = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
    img = Image.fromarray(np.zeros((720, 1280, 3), dtype=np.uint8))
    pv = proc(images=img, text="<OD>", return_tensors="pt")["pixel_values"].to(DEV, dtype)
    vt = m.vision_tower

    torch.cuda.reset_peak_memory_stats()
    for _ in range(N_WARMUP):
        with torch.no_grad():
            vt.forward_features_unpool(pv)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(N_ITERS):
        with torch.no_grad():
            vt.forward_features_unpool(pv)
    torch.cuda.synchronize()
    ms = (time.time() - t0) / N_ITERS * 1000
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    n_vision_tokens = pv.shape[-1] // 32 * (pv.shape[-2] // 32)  # sanity only, not used for report
    del m, vt
    torch.cuda.empty_cache()
    return {"ms": ms, "peak_mem_mb": peak_mem_mb, "pixel_values_shape": list(pv.shape)}


def main():
    results = {}
    for dtype_name, dtype in [("fp32", torch.float32), ("fp16", torch.float16)]:
        print(f"[{dtype_name}] 측정 중...", flush=True)
        r = measure(dtype)
        print(f"[{dtype_name}] {r['ms']:.1f}ms, peak_mem={r['peak_mem_mb']:.0f}MB, "
              f"pixel_values={r['pixel_values_shape']}", flush=True)
        results[dtype_name] = r

    out = {
        "device": "Jetson Orin NX",
        "model": "microsoft/Florence-2-base",
        "method": "vision_tower.forward_features_unpool(pixel_values)",
        "batch_size": 1,
        "n_warmup": N_WARMUP,
        "n_iters": N_ITERS,
        "torch_version": torch.__version__,
        **results,
    }
    out_path = "docs/v5/detector/florence2_backbone_jetson_latency.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()

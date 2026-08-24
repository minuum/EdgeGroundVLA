#!/usr/bin/env python3
"""Jetson Orin NX — grounder.run()(OWL-v2) vs enc.encode_image()(Kosmos-2 vision) 순차/병렬 지연 비교.

minum 요청 (2026-08-21, docs/DATASET_V6_STATUS.md): stage2_v2_inference_server.py
predict()가 두 독립 연산을 순차 실행 중(bbox = grounder.run(...) 후 vis_feat =
enc.encode_image(...)) — ThreadPoolExecutor로 동시 제출 시 총 지연이 A+B에서
max(A,B)에 가까워지는지, Jetson처럼 SM이 적은 GPU에서도 그 효과가 나는지 실측.

실제 서버 코드(stage2_v2_inference_server.py)의 OwlV2Grounder/Stage1Encoder를
그대로 import해서 사용 — 프로덕션 설정(grounder=owlv2, fp16, thresh=0.2)과 동일.
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# 프로덕션 런타임 상태와 동일한 env 설정 (logs/stage2_runtime_state.json 기준)
os.environ.setdefault("VLA_OWLV2_FP16", "1")
os.environ.setdefault("VLA_OWLV2_THRESH", "0.2")
os.environ.setdefault("VLA_OWLV2_AREA_SCALE", "3.0")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

from robovlm_nav.serve.stage2_v2_inference_server import (
    DEFAULT_VLM, OwlV2Grounder, Stage1Encoder,
)

STAGE1_PATH = ROOT / "runs/v5_nav/mlp/stage1_v3_5cls/stage1_v3_5cls_owl_projs.pt"
N_WARMUP = 10
N_ITERS = 100
PHRASE = "gray basket"


def make_frame():
    arr = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
    return arr, Image.fromarray(arr).convert("RGB")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    print("Stage1Encoder(Kosmos-2 vision) 로딩...", flush=True)
    enc = Stage1Encoder(DEFAULT_VLM, STAGE1_PATH, device)
    enc.eval()

    print("OwlV2Grounder 로딩...", flush=True)
    grounder = OwlV2Grounder(device)
    grounder._ensure_loaded()  # 첫 run() 지연에 로딩시간이 안 섞이게 미리 로드

    image_rgb, pil = make_frame()

    print(f"워밍업 {N_WARMUP}회...", flush=True)
    for _ in range(N_WARMUP):
        grounder.run(image_rgb, phrase=PHRASE)
        enc.encode_image(pil)
    torch.cuda.synchronize()

    # 1) 순차
    print(f"순차 측정 {N_ITERS}회...", flush=True)
    seq_ms = []
    seq_ground_ms = []
    seq_vis_ms = []
    for _ in range(N_ITERS):
        t0 = time.time()
        tg0 = time.time()
        grounder.run(image_rgb, phrase=PHRASE)
        torch.cuda.synchronize()
        tg1 = time.time()
        enc.encode_image(pil)
        torch.cuda.synchronize()
        t1 = time.time()
        seq_ms.append((t1 - t0) * 1000)
        seq_ground_ms.append((tg1 - tg0) * 1000)
        seq_vis_ms.append((t1 - tg1) * 1000)
    torch.cuda.synchronize()

    # 2) 병렬 (ThreadPoolExecutor)
    print(f"병렬 측정 {N_ITERS}회...", flush=True)
    par_ms = []
    with ThreadPoolExecutor(max_workers=2) as ex:
        for _ in range(N_ITERS):
            t0 = time.time()
            fut_g = ex.submit(grounder.run, image_rgb, phrase=PHRASE)
            fut_v = ex.submit(enc.encode_image, pil)
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
        "vision_encoder": "Kosmos-2 vision_model + image_proj (stage1_v3_5cls)",
        "sequential_ms": seq_stats,
        "sequential_grounding_component_ms": stats(seq_ground_ms),
        "sequential_vision_component_ms": stats(seq_vis_ms),
        "threaded_ms": par_stats,
        "ideal_max_ab_ms": ideal_max,
        "ratio_threaded_over_sequential": par_stats["mean"] / seq_stats["mean"],
        "ratio_idealmax_over_sequential": ideal_max / seq_stats["mean"],
    }

    print(json.dumps(out, indent=2, ensure_ascii=False))

    out_path = ROOT / "docs/v5/detector/sequential_vs_threaded_grounding_vision_jetson.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()

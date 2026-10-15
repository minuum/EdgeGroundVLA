#!/usr/bin/env python3
"""EGVLA2 FM 액션헤드용: v3 연속 수집 h5 → Florence-2 fused 캐시(bbox 4-d + 풀링 768-d).

precompute_v6_florence2_fused.py와 동일한 방식(_encode_image 1회 + phrase grounding)이며
대상만 ROS_action/mobile_vla_dataset_egvla2/ 의 v3 파일이다 (images는 최상위 키).
출력: docs/v5/egvla2_fm/v3_florence2_cache.npz
  "{파일명}|bbox" (n,4)[cx,cy,area,has_bbox], "{파일명}|feat" (n,768) float16, 프레임은 h5 전 프레임 순서.
이미 캐시에 있는 파일은 건너뛴다(증분 갱신). rsync 중인 임시 파일(.으로 시작)은 glob에서 제외됨.
실행: HF_HUB_OFFLINE=1 .venv/bin/python3 scripts/precompute_egvla2_continuous_features.py
"""
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from florence2_bbox_adapter import florence2_output_to_bbox  # noqa: E402
from precompute_v6_florence2_fused import MODEL, TASK, PHRASE, BATCH, DEV, run_batch  # noqa: E402,F401

DATA = ROOT / "ROS_action" / "mobile_vla_dataset_egvla2"
OUT_DIR = ROOT / "docs" / "v5" / "egvla2_fm"
OUT = OUT_DIR / "v3_florence2_cache.npz"


def main():
    from transformers import AutoModelForCausalLM, AutoProcessor
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = []
    for p in sorted(DATA.glob("*.h5")):
        with h5py.File(p, "r") as f:
            if f.attrs.get("action_space") == "egvla2_continuous_v3":
                files.append(p)
    print(f"v3 파일 {len(files)}개", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.float16).to(DEV).eval()
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    cache, t0 = (dict(np.load(OUT)) if OUT.exists() else {}), time.time()
    for i, p in enumerate(files):
        if f"{p.name}|feat" in cache:
            continue
        with h5py.File(p, "r") as f:
            imgs = f["images"][:]
        pils = [Image.fromarray(np.ascontiguousarray(im[:, :, ::-1]).astype(np.uint8)).convert("RGB") for im in imgs]
        bb, ft = [], []
        for s in range(0, len(pils), BATCH):
            b, x = run_batch(model, proc, pils[s:s + BATCH])
            bb.append(b); ft.append(x)
        cache[f"{p.name}|bbox"] = np.concatenate(bb)
        cache[f"{p.name}|feat"] = np.concatenate(ft)
        print(f"  [{i + 1}/{len(files)}] {p.name} ({len(pils)}f) elapsed={(time.time() - t0) / 60:.1f}m", flush=True)
    np.savez_compressed(OUT, **cache)
    print(f"저장: {OUT}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Stage 1: bbox_dataset_v6_owl.json의 모든 에피소드 이미지를 16x16 그레이스케일로
1회만 전처리해 캐시 저장 (Stage 2 다중 seed 학습에서 h5 재로드 I/O 병목 제거).

225 episodes / 16,599 frames — 각 h5를 정확히 1번씩만 연다.
"""
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "ROS_action" / "mobile_vla_dataset_v5"
OWL_DIR = ROOT / "docs" / "v5" / "bbox_nav_owl"
DATASET_FILE = OWL_DIR / "bbox_dataset_v6_owl.json"
CACHE_FILE = OWL_DIR / "v6_img_feat_cache_16x16.npz"

IMG_SIZE = 16


def resolve_episode_path(ep_field):
    p = Path(ep_field)
    if p.suffix == ".h5" and p.exists():
        return p
    return next(DATA_DIR.glob(f"{ep_field}.h5"))


def frame_to_image_feature(frame):
    img = Image.fromarray(frame.astype(np.uint8)).convert("L").resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(img, dtype=np.float32).reshape(-1) / 255.0


def main():
    dataset = json.loads(DATASET_FILE.read_text())
    print(f"에피소드 {len(dataset)}개 이미지 전처리 시작...")
    t0 = time.time()
    cache = {}
    for i, ep in enumerate(dataset):
        path = resolve_episode_path(ep["episode"])
        with h5py.File(path, "r") as f:
            imgs = f["observations"]["images"][:] if ("observations" in f and "images" in f["observations"]) else f["images"][:]
        frame_idxs = [fr["frame_idx"] for fr in ep["frames"]]
        feats = np.stack([frame_to_image_feature(imgs[idx]) for idx in frame_idxs]).astype(np.float32)
        cache[ep["episode"]] = feats
        if i % 20 == 0 or i == len(dataset) - 1:
            print(f"  [{i+1}/{len(dataset)}] {ep['episode'].split('/')[-1]} ({len(frame_idxs)} frames) "
                  f"elapsed={time.time()-t0:.1f}s")

    np.savez_compressed(CACHE_FILE, **{k: v for k, v in cache.items()})
    print(f"\n저장: {CACHE_FILE} ({time.time()-t0:.1f}s 소요, {len(cache)} episodes)")


if __name__ == "__main__":
    main()

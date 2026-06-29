#!/usr/bin/env python3
"""
plan_20260622_hidden_state_action_head.md §2-3 — bbox_dataset_full.json의
150개 에피소드(2,626프레임) 전체에 대해 PG2 hidden state(2304차원)를 추출해
캐시한다. 속도 테스트(time_hidden_state_extraction.py) 결과 배치 처리 시
프레임당 ~0.22s로 전체 약 10분 내 완료 예상 — 다운샘플링 불필요.

추출 방식은 Step B(probe_v5_direction_hidden_state.py)와 동일(같은 prompt,
같은 hidden_states[-1][:,-1,:]) — 단 에피소드 단위 배치로 처리해 속도 개선.

산출: docs/v5/hidden_state_cache/v5_hidden_states.npz
  - key: f"{episode_stem}__f{frame_idx}" -> (2304,) float16 array

Usage:
  .venv/bin/python3 scripts/eval/extract_v5_hidden_states_full.py
"""
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

ROOT = Path(__file__).resolve().parent.parent.parent
PG2 = Path.home() / ".cache/huggingface/hub/models--google--paligemma2-3b-mix-224/snapshots/8e40ab4cc5df93dfb7fd2fff754bcdff8b62ee78"
DATA_PATH = ROOT / "docs/v5/bbox_nav_exp46/bbox_dataset_full.json"
OUT_PATH = ROOT / "docs/v5/hidden_state_cache/v5_hidden_states.npz"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
PROMPT = "detect gray basket"


def main():
    data = json.loads(DATA_PATH.read_text())
    print(f"[데이터] {len(data)}개 에피소드")

    print(f"\n[로드] {PG2}")
    proc = PaliGemmaProcessor.from_pretrained(str(PG2))
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        str(PG2), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map={"": "cuda"},
    ).eval()

    cache = {}
    t_start = time.time()
    total_frames = 0

    skipped = []
    for i, ep in enumerate(data):
        h5_path = Path(ep["episode"])
        stem = h5_path.stem
        frames = ep["frames"]
        if not h5_path.exists():
            skipped.append(str(h5_path))
            continue
        with h5py.File(h5_path, "r") as f:
            imgs_ds = f["observations"]["images"]
            imgs = [Image.fromarray(imgs_ds[fr["frame_idx"]].astype(np.uint8)).convert("RGB")
                    for fr in frames]

        inp = proc(text=[PROMPT] * len(imgs), images=imgs, return_tensors="pt", padding=True).to(model.device)
        inp["pixel_values"] = inp["pixel_values"].to(model.dtype)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
            hs = out.hidden_states[-1][:, -1, :].float().cpu().numpy().astype(np.float16)

        for fr, h in zip(frames, hs):
            cache[f"{stem}__f{fr['frame_idx']}"] = h
        total_frames += len(frames)

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t_start
            print(f"  [{i+1}/{len(data)}] {total_frames}프레임 처리, 경과 {elapsed/60:.1f}분")

    elapsed = time.time() - t_start
    print(f"\n[완료] {len(data)-len(skipped)}/{len(data)}개 에피소드(스킵 {len(skipped)}), {total_frames}프레임, {elapsed/60:.1f}분 소요")
    if skipped:
        print("[스킵된 파일]")
        for s in skipped:
            print(f"  {s}")

    np.savez_compressed(OUT_PATH, **cache)
    print(f"[저장] {OUT_PATH} ({OUT_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

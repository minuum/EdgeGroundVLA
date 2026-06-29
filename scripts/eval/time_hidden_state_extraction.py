#!/usr/bin/env python3
"""
plan_20260622_hidden_state_action_head.md §2-1 — 본 추출 들어가기 전 속도 진단.
Step B가 220프레임에 68분(프레임당 ~18.5초) 걸린 원인이 CUDA그래프 동적 shape
재컴파일인지 확인하고, eager 모드 강제 / 배치 처리 완화책의 효과를 측정한다.
새 캐시 생성 없음 — 순수 타이밍 테스트.

Usage:
  .venv/bin/python3 scripts/eval/time_hidden_state_extraction.py
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
PROMPT = "detect gray basket"
N_FRAMES = 14  # 첫 episode 전체 프레임 (14개)


def load_frames():
    data = json.loads(DATA_PATH.read_text())
    ep = data[0]
    h5_path = Path(ep["episode"])
    with h5py.File(h5_path, "r") as f:
        imgs_ds = f["observations"]["images"]
        imgs = [Image.fromarray(imgs_ds[fr["frame_idx"]].astype(np.uint8)).convert("RGB")
                for fr in ep["frames"]]
    print(f"[데이터] {h5_path.name} — {len(imgs)} 프레임, size={imgs[0].size}")
    return imgs


def time_one_by_one(model, proc, imgs, label):
    torch.cuda.synchronize()
    t0 = time.time()
    for img in imgs:
        inp = proc(text=PROMPT, images=img, return_tensors="pt").to(model.device)
        inp["pixel_values"] = inp["pixel_values"].to(model.dtype)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)
            _ = out.hidden_states[-1][0, -1, :].float().cpu().numpy()
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"[{label}] {len(imgs)}프레임 1장씩 처리: {dt:.1f}s ({dt/len(imgs):.2f}s/frame)")
    return dt


def time_batched(model, proc, imgs, label):
    torch.cuda.synchronize()
    t0 = time.time()
    inp = proc(text=[PROMPT] * len(imgs), images=imgs, return_tensors="pt", padding=True).to(model.device)
    inp["pixel_values"] = inp["pixel_values"].to(model.dtype)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
        _ = out.hidden_states[-1][:, -1, :].float().cpu().numpy()
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"[{label}] {len(imgs)}프레임 배치 처리: {dt:.1f}s ({dt/len(imgs):.2f}s/frame)")
    return dt


def main():
    imgs = load_frames()

    print(f"\n[로드] {PG2}")
    proc = PaliGemmaProcessor.from_pretrained(str(PG2))
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        str(PG2), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map={"": "cuda"},
    ).eval()

    print("\n=== 1) 기본(현재 probe 스크립트와 동일 방식) — 1장씩, dynamo 기본 ===")
    dt1 = time_one_by_one(model, proc, imgs, "기본 1장씩")

    print("\n=== 2) eager 모드 강제 (torch._dynamo.config.disable=True) — 1장씩 ===")
    torch._dynamo.config.disable = True
    dt2 = time_one_by_one(model, proc, imgs, "eager 1장씩")
    torch._dynamo.config.disable = False

    print("\n=== 3) 배치 처리 (14프레임 한 번에) ===")
    dt3 = time_batched(model, proc, imgs, "배치")

    print(f"\n[추정] 전체 2,626프레임 기준 예상 시간:")
    print(f"  기본 방식: {dt1/len(imgs)*2626/60:.1f}분")
    print(f"  eager 모드: {dt2/len(imgs)*2626/60:.1f}분")
    print(f"  배치(에피소드=14프레임 가정): {dt3/len(imgs)*2626/60:.1f}분")


if __name__ == "__main__":
    main()

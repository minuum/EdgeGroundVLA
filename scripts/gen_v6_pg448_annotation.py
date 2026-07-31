#!/usr/bin/env python3
"""
V6(트랙A) 에피소드 PG448 배치 주석.

gen_pg448_annotation.py와 동일 검출/필터 로직에 두 가지 차이:
  1. batch 추론 (기본 64) — GB10에서 0.62s/frame
  2. stride 검출 + carry-forward — 배포 서버의 grounding cache(LIVE 33~36%,
     평균 3프레임 간격)와 동일한 시맨틱. stride 사이 프레임은 직전 LIVE 값 복사.

Usage:
  .venv/bin/python3 scripts/gen_v6_pg448_annotation.py \
      --src docs/v5/bbox_frame_level/bbox_dataset_v6_frame_level.json \
      --out docs/v5/bbox_frame_level/bbox_dataset_v6_pg448_cx.json \
      --stride 3 --batch 64
"""
import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json, re, argparse
from pathlib import Path
import numpy as np
import torch
torch._dynamo.config.disable = True
from PIL import Image
import h5py

ROOT   = Path(__file__).resolve().parent.parent
PG448  = Path.home() / ".cache/huggingface/hub" \
         / "models--google--paligemma2-3b-mix-448" \
         / "snapshots/1406c92ec87d32cc6b983239278901b904ba7a51"
LOC_RE = re.compile(r"<loc(\d{4})>")


def load_model(device):
    from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
    dtype = torch.bfloat16
    print(f"[LOAD] PaliGemma2-448 from {PG448}", flush=True)
    proc  = PaliGemmaProcessor.from_pretrained(str(PG448))
    model = PaliGemmaForConditionalGeneration.from_pretrained(
                str(PG448), torch_dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
    print("  로드 완료", flush=True)
    return proc, model, dtype


@torch.no_grad()
def detect_batch(model, proc, pil_imgs, device, dtype):
    """returns list of (cx, cy, area, hit)"""
    texts = ["<image>detect gray basket"] * len(pil_imgs)
    inp = proc(text=texts, images=pil_imgs, return_tensors="pt", padding=True).to(device)
    inp["pixel_values"] = inp["pixel_values"].to(dtype)
    gen = model.generate(**inp, max_new_tokens=24, do_sample=False)
    outs = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=False)
    results = []
    for raw in outs:
        locs = [int(v) / 1023.0 for v in LOC_RE.findall(raw)]
        if len(locs) >= 4:
            y1, x1, y2, x2 = locs[:4]
            results.append(((x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * (y2 - y1), True))
        else:
            results.append((0.5, 0.5, 0.05, False))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--resume", action="store_true",
                    help="--out에 부분 결과가 있으면 이미 처리된(has_bbox 필드 존재) 에피소드는 건너뜀")
    args = ap.parse_args()

    device = torch.device("cuda")
    proc, model, dtype = load_model(device)

    with open(args.src) as f:
        ann = json.load(f)

    done_stems = set()
    new_ann = []
    if args.resume and Path(args.out).exists():
        prev = json.loads(Path(args.out).read_text())
        for ep in prev:
            if ep["frames"] and "grounding_cached" in ep["frames"][0]:
                done_stems.add(Path(ep["episode"]).stem)
                new_ann.append(ep)
        print(f"[RESUME] 이미 완료된 {len(done_stems)}ep 건너뜀", flush=True)

    total_live = hit_live = 0
    for ep_i, ep in enumerate(ann):
        if Path(ep["episode"]).stem in done_stems:
            continue
        h5_path = Path(ep["episode"])
        if not h5_path.exists():
            new_ann.append(ep)
            continue
        with h5py.File(str(h5_path), "r") as f:
            imgs = (f["observations"]["images"] if "observations" in f else f["images"])[:]

        frames = ep["frames"]
        live_idx = list(range(0, len(frames), args.stride))
        # 마지막 프레임은 항상 LIVE (STOP 부근 정확도)
        if live_idx[-1] != len(frames) - 1:
            live_idx.append(len(frames) - 1)

        # V6 raw도 V5처럼 BGR 저장 (2026-07-16 시각 확인: 반전본이 회색 바구니/목재 테이블)
        # 배포 그라운더는 RGB 정상 경로이므로 학습 주석도 RGB로 생성해야 일치
        pil_imgs = [Image.fromarray(imgs[frames[i]["frame_idx"]][:, :, ::-1].astype("uint8"))
                    for i in live_idx]
        live_results = []
        for b in range(0, len(pil_imgs), args.batch):
            live_results.extend(detect_batch(model, proc, pil_imgs[b:b + args.batch], device, dtype))

        # 오탐 필터 (gen_pg448_annotation.py 동일)
        filtered = []
        for cx, cy, area, hit in live_results:
            if hit and (cy < 0.35 or area < 0.010 or area > 0.9):
                hit = False
            filtered.append((cx, cy, area, hit))
        total_live += len(filtered)
        hit_live   += sum(1 for r in filtered if r[3])

        live_map = dict(zip(live_idx, filtered))
        new_frames = []
        last = (0.5, 0.5, 0.05, False)
        for i, fr in enumerate(frames):
            if i in live_map:
                last = live_map[i]
                cached = False
            else:
                cached = True
            cx, cy, area, hit = last
            new_fr = dict(fr)
            new_fr["cx_det"]   = cx if hit else 0.5
            new_fr["cy_det"]   = cy if hit else 0.5
            new_fr["area_det"] = area if hit else 0.05
            new_fr["detected"] = hit
            new_fr["has_bbox"] = hit
            new_fr["grounding_cached"] = cached
            new_frames.append(new_fr)

        new_ep = dict(ep)
        new_ep["frames"] = new_frames
        new_ann.append(new_ep)

        if (ep_i + 1) % 10 == 0 or ep_i == len(ann) - 1:
            print(f"  [{ep_i+1}/{len(ann)}] LIVE hit={hit_live}/{total_live} "
                  f"({hit_live/max(total_live,1)*100:.1f}%)", flush=True)
            # 중간 저장 (GPU OOM 등으로 죽어도 진행분 보존)
            Path(args.out).write_text(json.dumps(new_ann, indent=2, ensure_ascii=False))
            torch.cuda.empty_cache()

    Path(args.out).write_text(json.dumps(new_ann, indent=2, ensure_ascii=False))
    print(f"\n완료: LIVE {hit_live}/{total_live} = {hit_live/max(total_live,1)*100:.1f}% detected")
    print(f"저장 → {args.out}")


if __name__ == "__main__":
    main()

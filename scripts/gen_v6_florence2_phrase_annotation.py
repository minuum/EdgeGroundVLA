#!/usr/bin/env python3
"""V6(225ep) Florence-2 그라운더 주석 — 명시적 phrase 그라운딩 버전 (2026-08-21).

gen_v6_florence2_annotation.py(열린 질문+키워드 매칭, V6 재현율 미검증)의 후속.
2026-08-20 발견: `<OD>`/`<DENSE_REGION_CAPTION>`(열린 질문) 대신
`<CAPTION_TO_PHRASE_GROUNDING>` + phrase="gray basket"(OWL-v2처럼 타겟 직접 지정)을
쓰면 0807 실기 배치 재현율 84.96%, V6 사람 검증(n=97) 100%로 압도적으로 높다
(scripts/label/serve_v6_phrase_grounding_verify.py 라벨링 결과).

이 스크립트는 이 방법으로 V6 전체(225ep, 16599프레임) 그라운더 주석을 다시 만든다.

기존 OWL 주석과의 차이:
  gen_v6_owl_annotation.py처럼 stride+carry-forward 시맨틱은 동일하게 유지하되,
  Florence-2 phrase grounding은 **거부 모드가 없다**(coverage 100%) — 즉 라이브
  샘플링된 모든 프레임에서 항상 박스가 나온다. 그래서 has_bbox는 항상 True가
  되며, 이는 OWL의 has_bbox(검출 성공/실패 신호)와 정보량이 다르다는 걸
  명심할 것 — carry-forward 구간만 실제로 "오래된 값"이라는 의미가 남는다.

주의: 파이프라인에 끼워넣지 않는다. Stage2 재학습용 오프라인 주석 생성 전용.

Usage:
  .venv/bin/python3 scripts/gen_v6_florence2_phrase_annotation.py \
      --src docs/v5/bbox_frame_level/bbox_dataset_v6_frame_level.json \
      --out docs/v5/bbox_nav_florence2/bbox_dataset_v6_florence2_phrase.json \
      --stride 3
"""
import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "microsoft/Florence-2-base"
TASK = "<CAPTION_TO_PHRASE_GROUNDING>"
PHRASE = "gray basket"


@torch.no_grad()
def detect_one(model, proc, pil):
    W, H = pil.width, pil.height
    text = TASK + PHRASE
    inp = proc(text=text, images=pil, return_tensors="pt")
    ids = model.generate(
        input_ids=inp["input_ids"].to(DEVICE),
        pixel_values=inp["pixel_values"].to(DEVICE, torch.float16),
        max_new_tokens=128, num_beams=3, do_sample=False)
    txt = proc.batch_decode(ids, skip_special_tokens=False)[0]
    parsed = proc.post_process_generation(txt, task=TASK, image_size=(W, H))[TASK]
    boxes = parsed.get("bboxes", []) or []
    if not boxes:
        return 0.5, 0.5, 0.05, False
    x1, y1, x2, y2 = boxes[0]
    cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
    area = (x2 - x1) * (y2 - y1) / (W * H)
    return cx, cy, area, True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=3)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoProcessor
    print("[Florence-2] 로딩...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.float16).to(DEVICE).eval()
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

    with open(args.src) as f:
        ann = json.load(f)

    total_live = hit_live = 0
    new_ann = []
    t0 = time.time()
    for ep_i, ep in enumerate(ann):
        h5_path = Path(ep["episode"])
        if not h5_path.exists():
            new_ann.append(ep)
            continue
        with h5py.File(str(h5_path), "r") as f:
            imgs = (f["observations"]["images"] if "observations" in f else f["images"])[:]

        frames = ep["frames"]
        live_idx = list(range(0, len(frames), args.stride))
        if live_idx[-1] != len(frames) - 1:
            live_idx.append(len(frames) - 1)

        live_results = []
        for i in live_idx:
            im = Image.fromarray(imgs[frames[i]["frame_idx"]][:, :, ::-1].astype("uint8")).convert("RGB")
            live_results.append(detect_one(model, proc, im))

        total_live += len(live_results)
        hit_live += sum(1 for r in live_results if r[3])

        live_map = dict(zip(live_idx, live_results))
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
            new_fr["cx_det"] = cx if hit else 0.5
            new_fr["cy_det"] = cy if hit else 0.5
            new_fr["area_det"] = area if hit else 0.05
            new_fr["detected"] = hit
            new_fr["has_bbox"] = hit
            new_fr["grounding_cached"] = cached
            new_frames.append(new_fr)

        new_ep = dict(ep)
        new_ep["frames"] = new_frames
        new_ann.append(new_ep)

        if (ep_i + 1) % 10 == 0 or ep_i == len(ann) - 1:
            elapsed = time.time() - t0
            print(f"  [{ep_i+1}/{len(ann)}] LIVE hit={hit_live}/{total_live} "
                  f"({hit_live/max(total_live,1)*100:.1f}%)  elapsed={elapsed/60:.1f}min", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(new_ann, indent=2, ensure_ascii=False))
    print(f"\n완료: LIVE {hit_live}/{total_live} = {hit_live/max(total_live,1)*100:.1f}% detected")
    print(f"저장 → {args.out}")


if __name__ == "__main__":
    main()

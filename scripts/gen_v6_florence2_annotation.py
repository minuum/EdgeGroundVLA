#!/usr/bin/env python3
"""V6(225ep) Florence-2 그라운더 주석 — 그라운더 ablation용 (exp73 OWL-v2 대체).

gen_v6_owl_annotation.py와 동일한 stride+carry-forward 시맨틱. 검출 방식만
Florence-2 `<OD>` + `<DENSE_REGION_CAPTION>` 키워드 매칭(정답 미사용, 단어 경계 regex)으로
교체 — florence2_keyword_match.py / florence2_grounding_0807_fullbatch.py와 동일 원칙.

DENSE_REGION_CAPTION을 우선 채택(정밀도 85.7% > OD 63.6%, 2026-08-17 n=80 측정),
DENSE가 못 찾으면 OD로 폴백(재현율 보강).

주의: 파이프라인에 끼워넣지 않는다. Stage2 재학습용 오프라인 주석 생성 전용.

Usage:
  .venv/bin/python3 scripts/gen_v6_florence2_annotation.py \
      --src docs/v5/bbox_frame_level/bbox_dataset_v6_frame_level.json \
      --out docs/v5/bbox_nav_florence2/bbox_dataset_v6_florence2.json \
      --stride 3
"""
import argparse
import json
import re
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL = "microsoft/Florence-2-base"

KEYWORDS = ["hamper", "basket", "trash can", "trash bin", "waste container",
            "waste bin", "wastebasket", "bin", "container"]
KW_RE = [re.compile(r"\b" + re.escape(k) + r"\b") for k in KEYWORDS]


@torch.no_grad()
def gen(model, proc, pil, task, beams=3):
    W, H = pil.width, pil.height
    inp = proc(text=task, images=pil, return_tensors="pt")
    ids = model.generate(
        input_ids=inp["input_ids"].to(DEVICE),
        pixel_values=inp["pixel_values"].to(DEVICE, torch.float16),
        max_new_tokens=256, num_beams=beams, do_sample=False)
    txt = proc.batch_decode(ids, skip_special_tokens=False)[0]
    parsed = proc.post_process_generation(txt, task=task, image_size=(W, H))[task]
    boxes = parsed.get("bboxes", []) or []
    labels = (parsed.get("labels") or parsed.get("bboxes_labels") or [])
    out = []
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b
        lb = str(labels[i]).lower().strip() if i < len(labels) else ""
        out.append(dict(label=lb, cx=(x1 + x2) / 2 / W, cy=(y1 + y2) / 2 / H,
                        area=(x2 - x1) * (y2 - y1) / (W * H)))
    return out


def pick_by_keyword(dets):
    hits = [d for d in dets if any(rx.search(d["label"]) for rx in KW_RE)]
    if not hits:
        return None
    return max(hits, key=lambda d: d["area"])


def detect_one(model, proc, pil):
    """DENSE 우선, 없으면 OD 폴백. returns (cx, cy, area, hit)."""
    dense = pick_by_keyword(gen(model, proc, pil, "<DENSE_REGION_CAPTION>"))
    if dense:
        return dense["cx"], dense["cy"], dense["area"], True
    od = pick_by_keyword(gen(model, proc, pil, "<OD>"))
    if od:
        return od["cx"], od["cy"], od["area"], True
    return 0.5, 0.5, 0.05, False


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

        # V6 raw는 BGR 저장 — RGB로 반전 (OWL 주석과 동일 처리)
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

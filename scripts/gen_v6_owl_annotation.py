#!/usr/bin/env python3
"""
V6(트랙A) 에피소드 OWL-v2 주석 — 그라운더 ablation용.

gen_v6_pg448_annotation.py와 동일한 stride+carry-forward 시맨틱,
검출기만 OWL-v2(th0.25, "gray laundry basket" — gen_owl_bbox_150ep.py 동일)로 교체.
BGR→RGB 반전 동일 적용.

Usage:
  .venv/bin/python3 scripts/gen_v6_owl_annotation.py \
      --src docs/v5/bbox_frame_level/bbox_dataset_v6_frame_level.json \
      --out docs/v5/bbox_nav_owl/bbox_dataset_v6_owl.json \
      --stride 3 --batch 16
"""
import json, argparse
from pathlib import Path
import torch
from PIL import Image
import h5py

ROOT = Path(__file__).resolve().parent.parent
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OWL_THRESH = 0.25
PROMPT = "gray laundry basket"


@torch.no_grad()
def detect_batch(model, proc, pil_imgs):
    """returns list of (cx, cy, area, hit)"""
    sizes = [(im.height, im.width) for im in pil_imgs]
    inp = proc(text=[[PROMPT]] * len(pil_imgs), images=pil_imgs, return_tensors="pt").to(DEVICE)
    o = model(**inp)
    results = []
    post = proc.post_process_object_detection(o, threshold=OWL_THRESH, target_sizes=sizes)
    for res, (H, W) in zip(post, sizes):
        if len(res["boxes"]) == 0:
            results.append((0.5, 0.5, 0.05, False))
        else:
            best = int(res["scores"].argmax())
            x1, y1, x2, y2 = res["boxes"][best].cpu().tolist()
            x1, x2, y1, y2 = x1 / W, x2 / W, y1 / H, y2 / H
            results.append(((x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * (y2 - y1), True))
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    print("[OWL-v2] 로딩...", flush=True)
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble").to(DEVICE).eval()

    with open(args.src) as f:
        ann = json.load(f)

    total_live = hit_live = 0
    new_ann = []
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

        # V6 raw는 BGR 저장 — RGB로 반전 (PG448 주석과 동일)
        pil_imgs = [Image.fromarray(imgs[frames[i]["frame_idx"]][:, :, ::-1].astype("uint8"))
                    for i in live_idx]
        live_results = []
        for b in range(0, len(pil_imgs), args.batch):
            live_results.extend(detect_batch(model, proc, pil_imgs[b:b + args.batch]))

        total_live += len(live_results)
        hit_live   += sum(1 for r in live_results if r[3])

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

    Path(args.out).write_text(json.dumps(new_ann, indent=2, ensure_ascii=False))
    print(f"\n완료: LIVE {hit_live}/{total_live} = {hit_live/max(total_live,1)*100:.1f}% detected")
    print(f"저장 → {args.out}")


if __name__ == "__main__":
    main()

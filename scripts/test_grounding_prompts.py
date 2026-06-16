#!/usr/bin/env python3
"""
Grounding prompt comparison test.
Tests different prompt formats on real episode frames to find which gives
the best bbox accuracy for the gray basket.
"""
import sys, os, glob, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import h5py
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

MODEL_PATH = ".vlms/kosmos-2-patch14-224"
FULLSCREEN_AREA_THRESHOLD = 0.85

PROMPTS = [
    ("<grounding>The gray basket is at",                 "current"),
    ("<grounding><phrase>gray basket</phrase>",           "phrase-tag"),
    ("<grounding>gray basket",                            "minimal"),
    ("<grounding>The gray laundry basket",                "descriptive"),
    ("<grounding><phrase>basket</phrase>",                "phrase-basket-only"),
]

# basket이 어느 쪽에 있는지 알 수 있는 에피소드 선택
# path_type으로 ground truth direction 추정 가능
DATA_DIR = "/home/minum/minum/26CS/MoNa-pi/mobile_vla_dataset_v5"
EPISODE_PATTERNS = {
    "LEFT":   ["*left_straight*", "*center_left*", "*left_left*"],
    "RIGHT":  ["*right_straight*", "*center_right*", "*right_right*"],
    "CENTER": ["*center_straight*"],
}

def cx_to_direction(cx):
    if cx < 0.35: return "LEFT"
    if cx > 0.65: return "RIGHT"
    return "CENTER"

CAPTION_DIRECTION_PATTERNS = [
    (["far left",  "extreme left",  "leftmost",    "bottom left",  "lower left",
      "front left", "left side",    "left corner",  "upper left",   "top left",
      "bottom-left", "lower-left",  "left-hand side"],               0.12),
    (["left"],                                                        0.25),
    (["far right", "extreme right", "rightmost",   "bottom right", "lower right",
      "front right", "right side",  "right corner", "upper right",  "top right",
      "bottom-right", "lower-right", "right-hand side"],             0.88),
    (["right"],                                                       0.75),
    (["center", "middle",  "straight ahead", "in front",
      "directly ahead",    "in the middle",  "front and center"],    0.5),
]
BASKET_KEYWORDS = ("basket", "gray box", "container", "bin", "laundry")

def parse_entities(entities, caption):
    """Extract bbox — mirrors updated proxy_inference_server._parse_basket_bbox."""
    candidates = []
    for entity_name, _span, boxes in entities:
        for box in boxes:
            x1, y1, x2, y2 = [float(v) for v in box]
            if max(x1, y1, x2, y2) > 1.5:
                x1, y1, x2, y2 = x1/1000, y1/1000, x2/1000, y2/1000
            area = (x2 - x1) * (y2 - y1)
            if area > FULLSCREEN_AREA_THRESHOLD:
                continue
            candidates.append({
                "entity": entity_name,
                "cx": (x1 + x2) / 2,
                "cy": (y1 + y2) / 2,
                "area": area,
                "is_basket": any(k in entity_name.lower() for k in BASKET_KEYWORDS),
            })

    matched = [b for b in candidates if b["is_basket"]]
    if matched:
        return matched[0], "entity_match"

    caption_lower = caption.lower()
    for phrases, cx in CAPTION_DIRECTION_PATTERNS:
        if any(p in caption_lower for p in phrases):
            return {"cx": cx, "entity": f"caption:{phrases[0]}"}, "caption_fallback"

    if candidates: return candidates[0], "any_entity"
    return None, "no_detection"


def run_test():
    print(f"Loading model from {MODEL_PATH} ...")
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16
    ).cuda().eval()
    print("Model loaded.\n")

    # 각 방향별로 에피소드 2개씩 선택
    test_cases = []  # (ep_file, gt_direction, frame_idx)
    for gt_dir, patterns in EPISODE_PATTERNS.items():
        found = []
        for pat in patterns:
            found.extend(sorted(glob.glob(f"{DATA_DIR}/{pat}")))
        for ep_file in found[:2]:  # 방향별 2개
            with h5py.File(ep_file, "r") as f:
                n = len(f["observations"]["images"][()])
            # 중간 프레임 (basket이 잘 보이는 시점)
            for fi in [n // 3, n // 2]:
                test_cases.append((ep_file, gt_dir, fi))

    print(f"Testing {len(test_cases)} frames × {len(PROMPTS)} prompts\n")
    print(f"{'EP':25s} {'GT':6s} | " + " | ".join(f"{label:18s}" for _, label in PROMPTS))
    print("-" * (25 + 8 + len(PROMPTS) * 21))

    # per-prompt stats
    stats = {label: {"correct": 0, "total": 0, "entity_match": 0, "no_det": 0}
             for _, label in PROMPTS}

    for ep_file, gt_dir, fi in test_cases:
        with h5py.File(ep_file, "r") as f:
            img_np = f["observations"]["images"][fi]
        pil_img = Image.fromarray(img_np.astype("uint8")).convert("RGB")
        ep_name = os.path.basename(ep_file)[:24]

        row_parts = []
        for prompt, label in PROMPTS:
            inp = processor(text=prompt, images=pil_img, return_tensors="pt")
            inp = {k: v.to("cuda") for k, v in inp.items()}
            inp["pixel_values"] = inp["pixel_values"].to(torch.float16)
            with torch.no_grad():
                gen = model.generate(
                    pixel_values=inp["pixel_values"],
                    input_ids=inp["input_ids"],
                    attention_mask=inp["attention_mask"],
                    image_embeds=None,
                    image_embeds_position_mask=inp.get("image_embeds_position_mask"),
                    use_cache=True, max_new_tokens=64,
                )
            new_ids = gen[:, inp["input_ids"].shape[1]:]
            raw = processor.batch_decode(new_ids, skip_special_tokens=False)[0]
            caption, entities = processor.post_process_generation(raw)

            bbox, src = parse_entities(entities, caption)
            if bbox is None:
                cx_str = "NONE"
                pred_dir = "NONE"
                stats[label]["no_det"] += 1
            else:
                cx = bbox["cx"]
                pred_dir = cx_to_direction(cx)
                cx_str = f"{cx:.2f}({pred_dir[0]})"
                if src == "entity_match":
                    stats[label]["entity_match"] += 1

            correct = (pred_dir == gt_dir.upper())
            if bbox is not None:
                stats[label]["total"] += 1
                if correct:
                    stats[label]["correct"] += 1

            mark = "✓" if correct else "✗"
            ent = bbox["entity"][:10] if bbox else "---"
            row_parts.append(f"{cx_str:6s} {ent:10s}{mark}")

        print(f"{ep_name:25s} {gt_dir[:5]:5s} | " + " | ".join(row_parts))

    # 최종 통계
    print("\n" + "=" * 70)
    print("SUMMARY")
    print(f"{'Prompt':20s} {'Acc':>6s} {'EntityMatch':>12s} {'NoDetect':>9s}")
    print("-" * 50)
    for _, label in PROMPTS:
        s = stats[label]
        acc = s["correct"] / s["total"] * 100 if s["total"] else 0
        em = s["entity_match"]
        nd = s["no_det"]
        total = s["total"] + nd
        print(f"{label:20s} {acc:5.1f}%  {em:>5}/{total:<5}  {nd:>5}/{total}")

if __name__ == "__main__":
    run_test()

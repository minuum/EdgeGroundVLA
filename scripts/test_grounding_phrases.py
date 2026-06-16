#!/usr/bin/env python3
"""
VLM Grounding Phrase 비교 테스트

"gray basket" 외 다양한 표현으로 grounding IoU 비교.
어떤 단어 조합이 우리 복도 이미지에서 가장 잘 작동하는지 측정.

Usage:
  .venv/bin/python3 scripts/test_grounding_phrases.py
  .venv/bin/python3 scripts/test_grounding_phrases.py --n-frames 60
"""

import argparse, json, sys, warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
import h5py, numpy as np, torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VLM_PATH  = ROOT / ".vlms" / "kosmos-2-patch14-224"
DATA_PATH = ROOT / "docs" / "v5" / "bbox_nav_exp46" / "bbox_dataset_full.json"
DATA_DIR  = ROOT / "ROS_action" / "mobile_vla_dataset_v5"

# 테스트할 phrase 목록 — 카테고리별로 묶음
PHRASES = {
    # 색상만
    "gray":               "gray",
    "grey":               "grey",

    # 형태만
    "basket":             "basket",
    "bin":                "bin",
    "container":          "container",
    "box":                "box",

    # 색상+형태 조합
    "gray basket":        "gray basket",   # 현재 사용 중
    "grey basket":        "grey basket",
    "gray bin":           "gray bin",
    "gray container":     "gray container",
    "gray box":           "gray box",
    "gray object":        "gray object",

    # VLM이 실제로 부른 이름
    "gray trash can":     "gray trash can",
    "gray trash bin":     "gray trash bin",
    "gray cylinder":      "gray cylinder",
    "gray laundry basket":"gray laundry basket",

    # 상황 묘사
    "target object":      "target object",
    "gray target":        "gray target",
}

IOU_THR = 0.3


def load_h5_frame(ep, frame_idx):
    ep_path = Path(ep["episode"])
    if ep_path.is_absolute() and ep_path.exists():
        h5_path = ep_path
    else:
        cands = list(DATA_DIR.glob(f"{ep_path.stem}.h5"))
        if not cands:
            cands = list(DATA_DIR.glob(f"**/{ep_path.stem}.h5"))
        if not cands:
            return None
        h5_path = cands[0]
    try:
        with h5py.File(str(h5_path), "r") as f:
            return Image.fromarray(f["observations"]["images"][frame_idx])
    except Exception:
        return None


def compute_iou(pred_box, cx, cy, area):
    side = area ** 0.5
    gx1, gy1 = cx - side/2, cy - side/2
    gx2, gy2 = cx + side/2, cy + side/2
    px1, py1, px2, py2 = pred_box
    ix1 = max(px1, gx1); iy1 = max(py1, gy1)
    ix2 = min(px2, gx2); iy2 = min(py2, gy2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    union = (px2-px1)*(py2-py1) + (gx2-gx1)*(gy2-gy1) - inter
    return inter/union if union > 0 else 0.0


def ground_phrase(proc, model, img, phrase, device):
    prompt = f"<grounding> An image of {phrase}."
    inputs = proc(text=prompt, images=img, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v,"to") else v for k,v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    raw = proc.decode(out[0], skip_special_tokens=False)
    _, entities = proc.post_process_generation(raw, cleanup_and_extract=True)
    boxes = []
    for ent_phrase, _span, ent_boxes in entities:
        boxes.extend(ent_boxes)
    return boxes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-frames", type=int, default=90,
                   help="테스트할 총 프레임 수 (basket_present만)")
    p.add_argument("--seed",     type=int, default=42)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device}")

    from transformers import AutoProcessor, AutoModelForVision2Seq
    proc  = AutoProcessor.from_pretrained(str(VLM_PATH))
    model = AutoModelForVision2Seq.from_pretrained(
        str(VLM_PATH), torch_dtype=torch.float16
    ).to(device).eval()
    print("[MODEL] Pure Kosmos-2 로드 완료\n")

    data = json.loads(DATA_PATH.read_text())
    rng  = np.random.RandomState(args.seed)

    # basket present 프레임만 샘플링
    present_samples = []
    for ep in data:
        for fr in ep["frames"]:
            if fr.get("has_bbox", False):
                present_samples.append((ep, fr))

    rng.shuffle(present_samples)
    samples = present_samples[:args.n_frames]
    print(f"[DATA] basket_present 프레임 {len(samples)}개 사용\n")

    # 각 phrase 테스트
    results = {}
    for phrase_key, phrase_text in PHRASES.items():
        hits, ious, total = 0, [], 0
        for ep, fr in samples:
            img = load_h5_frame(ep, fr.get("frame_idx", 0))
            if img is None:
                continue
            cx   = fr.get("cx",   0.5)
            cy   = fr.get("cy",   0.5)
            area = fr.get("area", 0.05)
            boxes = ground_phrase(proc, model, img, phrase_text, device)
            total += 1
            if boxes:
                best_iou = max(compute_iou(b, cx, cy, area) for b in boxes)
                ious.append(best_iou)
                if best_iou >= IOU_THR:
                    hits += 1
            else:
                ious.append(0.0)

        hit_rate = hits / total if total > 0 else 0
        mean_iou = np.mean(ious) if ious else 0.0
        any_box  = sum(1 for b in ious if b > 0) / max(1, total)
        results[phrase_key] = {
            "hit_rate": hit_rate, "mean_iou": mean_iou,
            "any_box": any_box, "n": total
        }

        bar = "█" * int(hit_rate * 20)
        print(f"  {phrase_key:<24} IoU≥0.3={hit_rate:.1%}  mean={mean_iou:.3f}  any={any_box:.0%}  {bar}")

    # 랭킹 출력
    print(f"\n{'='*60}")
    print("RANKING — IoU≥0.3 기준")
    print(f"{'='*60}")
    ranked = sorted(results.items(), key=lambda x: x[1]["hit_rate"], reverse=True)
    for rank, (phrase, v) in enumerate(ranked, 1):
        marker = " ← 현재 사용" if phrase == "gray basket" else ""
        print(f"  {rank:2d}. {phrase:<24} {v['hit_rate']:.1%}  (mean IoU {v['mean_iou']:.3f}){marker}")

    print(f"\n[결론]")
    best = ranked[0]
    if best[0] != "gray basket":
        diff = best[1]["hit_rate"] - results["gray basket"]["hit_rate"]
        print(f"  '{best[0]}' 가 '{results['gray basket']['hit_rate']:.1%}' → '{best[1]['hit_rate']:.1%}' (+{diff:.1%}p 향상)")
        print(f"  → Option C 프롬프트를 '{best[0]}'로 변경 고려")
    else:
        print(f"  현재 사용 중인 'gray basket'이 최적 표현")


if __name__ == "__main__":
    main()

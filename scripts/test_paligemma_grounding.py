#!/usr/bin/env python3
"""
PaliGemma Zero-shot Grounding 테스트

복도 이미지에서 "detect gray basket" zero-shot 작동 여부 확인.
여러 phrase 비교 → R2-3 데모 가능성 평가.

Usage:
    python3 scripts/test_paligemma_grounding.py
    python3 scripts/test_paligemma_grounding.py --max-frames 20 --phrases "gray basket" "person"
    python3 scripts/test_paligemma_grounding.py --save-overlays
"""
import argparse
import json
import re
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw

PALIGEMMA_PATH = Path.home() / ".cache/huggingface/hub" \
    / "models--google--paligemma-3b-pt-224" \
    / "snapshots/35e4f46485b4d07967e7e9935bc3786aad50687c"

DATA_DIR = Path("/home/minum/minum/26CS/MoNa-pi/mobile_vla_dataset_v5")
BBOX_JSON = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_frame_level.json"
OUT_DIR   = ROOT / "docs/v5/grounding_demo/paligemma_zeroshot"

# loc 토큰 파싱: <loc0389> → int
LOC_RE = re.compile(r"<loc(\d{4})>")

COLOR = {
    "gray basket": (50, 220, 80),
    "person":      (50, 150, 255),
    "red ball":    (220, 50,  50),
    "white wall":  (200, 200, 200),
}


def parse_locs(text: str) -> list[float]:
    """<loc####> 4개 추출 → [y1,x1,y2,x2] normalized (PaliGemma 기본 순서)"""
    vals = [int(v) / 1024.0 for v in LOC_RE.findall(text)]
    if len(vals) >= 4:
        y1, x1, y2, x2 = vals[:4]
        return [x1, y1, x2, y2]  # xyxy로 변환
    return []


def draw_result(img_arr: np.ndarray, phrase: str, decoded: str) -> Image.Image:
    pil = Image.fromarray(img_arr).convert("RGB")
    draw = ImageDraw.Draw(pil)
    W, H = pil.size
    color = COLOR.get(phrase.lower(), (255, 200, 0))

    bbox = parse_locs(decoded)
    if bbox:
        x1, y1, x2, y2 = bbox
        px1, py1 = int(x1*W), int(y1*H)
        px2, py2 = int(x2*W), int(y2*H)
        draw.rectangle([px1, py1, px2, py2], outline=color, width=3)
        label = f"{phrase} [{x1:.2f},{y1:.2f},{x2:.2f},{y2:.2f}]"
        draw.rectangle([px1, py1-18, px1+len(label)*7, py1], fill=color)
        draw.text((px1+2, py1-17), label, fill=(0, 0, 0))
    else:
        draw.text((10, 10), f"[NO DETECTION] {phrase}", fill=color)

    # 하단에 raw 출력
    short = decoded[:80].replace("\n", " ")
    draw.rectangle([0, H-22, W, H], fill=(0, 0, 0))
    draw.text((5, H-20), short, fill=(200, 200, 200))
    return pil


def load_test_frames(max_frames: int) -> list[dict]:
    """bbox_dataset_frame_level.json 에서 has_bbox 프레임 샘플링"""
    with open(BBOX_JSON) as f:
        dataset = json.load(f)

    frames = []
    # 에피소드별로 1프레임씩 (다양한 경로 타입 커버)
    for ep in dataset:
        det = [fr for fr in ep["frames"] if fr.get("detected")]
        if not det:
            continue
        # 중간 프레임 선택
        fr = det[len(det) // 2]
        frames.append({
            "episode": ep["episode"],
            "path_type": ep["path_type"],
            "direction": ep["direction"],
            "frame_idx": fr["frame_idx"],
            "cx_gt": fr.get("cx_det"),
            "label": fr.get("label"),
        })
        if max_frames and len(frames) >= max_frames:
            break
    return frames


def load_image(ep_path: str, frame_idx: int) -> np.ndarray | None:
    p = Path(ep_path)
    if not p.exists():
        # DATA_DIR에서 stem으로 검색
        matches = list(DATA_DIR.glob(f"{p.stem}.h5"))
        if not matches:
            return None
        p = matches[0]
    try:
        with h5py.File(p, "r") as f:
            if "observations" in f and "images" in f["observations"]:
                return f["observations"]["images"][frame_idx].astype(np.uint8)
            return f["images"][frame_idx].astype(np.uint8)
    except Exception:
        return None


EXP57_ADAPTER = Path(__file__).resolve().parent.parent / "runs/v5_nav/grounding/exp57"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames",   type=int, default=30,
                        help="테스트할 프레임 수 (기본 30)")
    parser.add_argument("--phrases",      nargs="+",
                        default=["gray basket", "red ball", "person"],
                        help="detect할 phrase 목록")
    parser.add_argument("--save-overlays", action="store_true",
                        help="오버레이 이미지 저장")
    parser.add_argument("--adapter",      default="",
                        help="LoRA adapter 경로. 'exp57'=Exp57 / 경로 직접")
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    adapter_path = None
    if args.adapter:
        adapter_path = EXP57_ADAPTER if args.adapter == "exp57" else Path(args.adapter)

    print("=" * 60)
    print("PaliGemma Grounding Test")
    print(f"  Model  : paligemma-3b-pt-224")
    print(f"  Adapter: {adapter_path or 'none (zero-shot)'}")
    print(f"  Phrases: {args.phrases}")
    print(f"  Frames : {args.max_frames}")
    print("=" * 60)

    # ── 모델 로딩 ──────────────────────────────────────────────
    from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration

    print(f"\n[LOAD] {PALIGEMMA_PATH}")
    processor = PaliGemmaProcessor.from_pretrained(str(PALIGEMMA_PATH))
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        str(PALIGEMMA_PATH),
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    ).to(device)

    if adapter_path is not None and adapter_path.exists():
        from peft import PeftModel
        print(f"[LOAD] LoRA adapter from {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
        print("  LoRA 로딩 완료")

    model.eval()
    print("  모델 로딩 완료")

    # ── 프레임 준비 ────────────────────────────────────────────
    test_frames = load_test_frames(args.max_frames)
    print(f"\n  테스트 프레임: {len(test_frames)}개")

    if args.save_overlays:
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 추론 ──────────────────────────────────────────────────
    stats = {p: {"hit": 0, "total": 0, "cx_errors": []} for p in args.phrases}
    results = []

    for i, fr_info in enumerate(test_frames):
        img_arr = load_image(fr_info["episode"], fr_info["frame_idx"])
        if img_arr is None:
            continue

        pil = Image.fromarray(img_arr).convert("RGB")
        row_imgs = []

        for phrase in args.phrases:
            prompt = f"detect {phrase}"
            inputs = processor(text=prompt, images=pil, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                )
            # prefix 길이만큼 자르기
            prefix_len = inputs["input_ids"].shape[1]
            decoded = processor.decode(gen[0][prefix_len:], skip_special_tokens=False)

            # bbox 파싱
            bbox = parse_locs(decoded)
            hit = len(bbox) > 0

            stats[phrase]["total"] += 1
            if hit:
                stats[phrase]["hit"] += 1
                # cx 오차 계산 (basket phrase인 경우)
                if "basket" in phrase and fr_info.get("cx_gt") is not None:
                    pred_cx = (bbox[0] + bbox[2]) / 2
                    err = abs(pred_cx - fr_info["cx_gt"])
                    stats[phrase]["cx_errors"].append(err)

            result_str = f"{'✅' if hit else '❌'} {phrase}: {decoded[:50]}"
            print(f"  [{i+1:2d}] {fr_info['path_type'][:12]:12s} | {result_str}")

            if args.save_overlays:
                overlay = draw_result(img_arr, phrase, decoded)
                row_imgs.append(overlay)

            results.append({
                "frame": i,
                "path_type": fr_info["path_type"],
                "phrase": phrase,
                "hit": hit,
                "decoded": decoded[:100],
                "bbox": bbox,
            })

        # 저장
        if args.save_overlays and row_imgs:
            combined = Image.new("RGB", (320 * len(row_imgs), 240))
            for j, img in enumerate(row_imgs):
                combined.paste(img.resize((320, 240)), (j * 320, 0))
            combined.save(str(OUT_DIR / f"frame_{i:03d}.jpg"))

    # ── 결과 요약 ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    for phrase, s in stats.items():
        hit_rate = s["hit"] / max(s["total"], 1)
        cx_info = ""
        if s["cx_errors"]:
            cx_info = f"  cx_err={sum(s['cx_errors'])/len(s['cx_errors']):.3f}"
        print(f"  '{phrase}': {s['hit']}/{s['total']} = {hit_rate*100:.1f}%{cx_info}")

    # basket vs others 비교
    print()
    basket_phrases = [p for p in args.phrases if "basket" in p]
    other_phrases  = [p for p in args.phrases if "basket" not in p]
    if basket_phrases and other_phrases:
        b_rate = stats[basket_phrases[0]]["hit"] / max(stats[basket_phrases[0]]["total"], 1)
        o_rates = [stats[p]["hit"] / max(stats[p]["total"], 1) for p in other_phrases]
        avg_o = sum(o_rates) / len(o_rates)
        print(f"  basket hit rate  : {b_rate*100:.1f}%")
        print(f"  others avg rate  : {avg_o*100:.1f}%")
        diff = b_rate - avg_o
        if diff > 0.2:
            print(f"  → 차이 {diff*100:.1f}%p ✅ 텍스트 phrase가 detection을 구분함 (R2-3 증거)")
        else:
            print(f"  → 차이 {diff*100:.1f}%p ⚠️  phrase 구분이 불분명")

    # JSON 저장
    out_json = OUT_DIR if args.save_overlays else ROOT / "docs/v5/grounding_demo"
    out_json.mkdir(parents=True, exist_ok=True)
    with open(out_json / "paligemma_results.json", "w") as f:
        json.dump({"stats": stats, "details": results}, f, indent=2)
    print(f"\n결과 저장 → {out_json}/paligemma_results.json")


if __name__ == "__main__":
    main()

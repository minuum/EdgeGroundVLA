#!/usr/bin/env python3
"""
PG2 fallback(has_bbox=False) 프레임 self-labeling 검증용 썸네일 생성.
gen_preview_label_thumbs_v2.py와 동일 포맷(640x360, L/C/R 존 오버레이, bbox 컬러 오버레이).

목적: 07-03 수신 세션의 fallback 프레임들에 대해 PG2 raw(필터 无) bbox를 그려서,
      실제로 basket이 화면에 있는데 못 찾은 건지 눈으로 검증.

Usage:
  .venv/bin/python3 scripts/gen_fallback_verify_thumbs.py
"""
import io
import json
import re
import random
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
RECV_DIRS = [Path("/home/minum/MoNaVLA/inference_sessions_recv/20260702"),
             Path("/home/minum/MoNaVLA/inference_sessions_recv/20260703")]
OUT_DIR = ROOT / "docs" / "v5" / "fallback_verify_20260703"
THUMB_W, THUMB_H = 640, 360
PG2_PATH = "google/paligemma2-3b-mix-448"
_LOC_RE = re.compile(r"<loc(\d{4})>")

MODEL_COLORS = {"PG": (80, 150, 255)}   # 파랑, gen_preview_label_thumbs_v2와 동일 컬러


def draw_thumb(img: Image.Image, pg_result: dict | None) -> Image.Image:
    thumb = img.resize((THUMB_W, THUMB_H), Image.LANCZOS).convert("RGBA")
    W, H = THUMB_W, THUMB_H
    L_X, R_X, MID = int(0.40 * W), int(0.60 * W), W // 2

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle([0, 0, L_X, H], fill=(80, 140, 255, 22))
    od.rectangle([R_X, 0, W, H], fill=(255, 180, 0, 22))
    od.line([(L_X, 0), (L_X, H)], fill=(80, 140, 255, 140), width=1)
    od.line([(MID, 0), (MID, H)], fill=(180, 180, 180, 100), width=1)
    od.line([(R_X, 0), (R_X, H)], fill=(255, 180, 0, 140), width=1)
    od.text((4, 3), "L", fill=(80, 140, 255, 180))
    od.text((L_X + 4, 3), "C", fill=(200, 200, 200, 150))
    od.text((R_X + 4, 3), "R", fill=(255, 180, 0, 180))
    thumb = Image.alpha_composite(thumb, overlay).convert("RGB")

    draw = ImageDraw.Draw(thumb)
    if pg_result is not None and "x1" in pg_result:
        color = MODEL_COLORS["PG"]
        x1, y1 = int(pg_result["x1"] * W), int(pg_result["y1"] * H)
        x2, y2 = int(pg_result["x2"] * W), int(pg_result["y2"] * H)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1 + 2, y1 + 1), "PG448", fill=color)
        label = f"has_bbox=True cx={pg_result['cx']:.2f} area={pg_result['area']:.3f}"
    else:
        label = "has_bbox=False (raw locs<4, 진짜 미탐지)"
    draw.text((4, H - 14), label, fill=(255, 255, 0))
    return thumb


def run_pg2_raw(model, proc, device, dtype, pil_img: Image.Image, phrase="gray basket"):
    inp = proc(text=f"detect {phrase}", images=pil_img, return_tensors="pt").to(device)
    inp["pixel_values"] = inp["pixel_values"].to(dtype)
    with torch.no_grad():
        gen = model.generate(**inp, max_new_tokens=48, min_new_tokens=1, do_sample=False)
    raw = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=False)[0]
    locs = [int(v) / 1023.0 for v in _LOC_RE.findall(raw)]
    if len(locs) < 4:
        return {"raw": raw, "locs": locs}
    y1, x1, y2, x2 = locs[:4]
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return {"raw": raw, "locs": locs, "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2, "area": (x2 - x1) * (y2 - y1)}


def main():
    random.seed(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    candidates = []
    for d in RECV_DIRS:
        for f in sorted(d.glob("session_*.h5")):
            with h5py.File(f, "r") as h:
                bbox = h["grounding/bbox"][:]
                for i in range(bbox.shape[0]):
                    if bbox[i, 3] == 0.0:
                        candidates.append((f, i))
    sample = random.sample(candidates, min(8, len(candidates)))
    sample.sort()
    print(f"fallback 프레임 {len(candidates)}개 중 {len(sample)}개 검증")

    print("PG2-448 로딩 중...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
    proc = PaliGemmaProcessor.from_pretrained(PG2_PATH)
    model = PaliGemmaForConditionalGeneration.from_pretrained(PG2_PATH, torch_dtype=dtype).to(device).eval()
    print("로딩 완료.")

    meta = []
    for f, i in sample:
        with h5py.File(f, "r") as h:
            img_arr = h["observations/images"][i]
        pil = Image.fromarray(img_arr.astype(np.uint8)).convert("RGB")
        result = run_pg2_raw(model, proc, device, dtype, pil)
        thumb = draw_thumb(pil, result if "x1" in result else None)

        key = f"{f.stem}_f{i:02d}"
        fname = f"thumb_{key}.jpg"
        thumb.save(OUT_DIR / fname, quality=88)
        meta.append({"key": key, "session": f.name, "frame_idx": i,
                      "raw": result.get("raw", ""), "locs": result.get("locs", []),
                      "pg_cx": result.get("cx"), "pg_area": result.get("area")})
        print(f"  [{key}] raw={result.get('raw','')[:60]!r}")

    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"\n완료: {len(meta)}개 → {OUT_DIR}")


if __name__ == "__main__":
    main()

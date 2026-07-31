#!/usr/bin/env python3
"""
PG2 fallback(has_bbox=False) 프레임을 PG2 / Kosmos-2 refexp(Kr) / OWL-v2(Ow) 3개 모델로
비교 그라운딩 해서 스크롤 갤러리 HTML로 보여주는 스크립트.
gen_preview_label_thumbs_v2.py와 동일한 draw_thumb 포맷(640x360, L/C/R 존, 모델별 컬러 bbox) +
docs/v5/exp54_viz/beforeafter/gallery.html과 동일한 다크 테마 스크롤 갤러리.

Usage:
  .venv/bin/python3 scripts/gen_fallback_multimodel_gallery.py [--limit 40]
"""
import argparse
import io
import json
import re
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
RECV_DIRS = [Path("/home/minum/MoNaVLA/inference_sessions_recv/20260702"),
             Path("/home/minum/MoNaVLA/inference_sessions_recv/20260703")]
OUT_DIR = ROOT / "docs" / "v5" / "fallback_multimodel_20260703"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THUMB_W, THUMB_H = 320, 180

MODEL_COLORS = {"PG": (80, 150, 255), "Kr": (60, 220, 60), "Ow": (240, 200, 0)}


def draw_thumb(img: Image.Image, model_results: dict) -> Image.Image:
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
    for mname, r in model_results.items():
        if r is None:
            continue
        color = MODEL_COLORS[mname]
        x1, y1 = int(r["x1"] * W), int(r["y1"] * H)
        x2, y2 = int(r["x2"] * W), int(r["y2"] * H)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1 + 2, y1 + 1), mname, fill=color)
    return thumb


EXCLUDE_SESSIONS = {"session_20260701_213204"}

def collect_fallback_frames(limit: int | None):
    frames = []
    for d in RECV_DIRS:
        for f in sorted(d.glob("session_*.h5")):
            if f.stem in EXCLUDE_SESSIONS:
                continue
            with h5py.File(f, "r") as h:
                bbox = h["grounding/bbox"][:]
                imgs = h["observations/images"]
                for i in range(bbox.shape[0]):
                    if bbox[i, 3] == 0.0:
                        frames.append({
                            "img": Image.fromarray(np.array(imgs[i]).astype(np.uint8)).convert("RGB"),
                            "episode": f.stem, "frame_idx": i,
                            "key": f"{f.stem}_f{i:02d}",
                            "server_cx": float(bbox[i, 0]), "server_area": float(bbox[i, 2]),
                        })
    if limit:
        frames = frames[:limit]
    return frames


# ── PG2-448 ──────────────────────────────────────────────────────────────────
_LOC_RE = re.compile(r"<loc(\d{4})>")

def run_pg2(frames):
    print("[PG2-448] 로딩...")
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
    proc = PaliGemmaProcessor.from_pretrained("google/paligemma2-3b-mix-448")
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        "google/paligemma2-3b-mix-448", torch_dtype=torch.bfloat16).to(DEVICE).eval()
    results = {}
    t0 = time.time()
    for i, fr in enumerate(frames):
        inp = proc(text="detect gray basket", images=fr["img"], return_tensors="pt").to(DEVICE)
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=48, min_new_tokens=1, do_sample=False)
        raw = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=False)[0]
        locs = [int(v) / 1023.0 for v in _LOC_RE.findall(raw)]
        if len(locs) >= 4:
            y1, x1, y2, x2 = locs[:4]
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            results[fr["key"]] = {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                                   "cx": (x1 + x2) / 2, "area": (x2 - x1) * (y2 - y1), "raw": raw}
        else:
            results[fr["key"]] = None
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(frames)} ({time.time()-t0:.0f}s)")
    del model; torch.cuda.empty_cache()
    return results


# ── Kosmos-2 refexp (Kr) ──────────────────────────────────────────────────────
_TARGET_KW = {"basket", "container", "bin", "laundry", "gray box", "pot"}

def _kosmos_parse_refexp(entities):
    for ename, _span, boxes in entities:
        for box in boxes:
            x1, y1, x2, y2 = [float(v) for v in box]
            if max(x1, y1, x2, y2) > 1.5:
                x1, y1, x2, y2 = x1/1000, y1/1000, x2/1000, y2/1000
            area = (x2 - x1) * (y2 - y1)
            if area > 0.85:
                continue
            if ename.startswith("<patch_index_"):
                return {"cx": (x1+x2)/2, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
    return None

def run_kosmos_refexp(frames):
    print("[Kosmos-2 refexp] 로딩...")
    from transformers import AutoProcessor, AutoModelForVision2Seq
    proc = AutoProcessor.from_pretrained(str(ROOT / ".vlms" / "kosmos-2-patch14-224"))
    model = AutoModelForVision2Seq.from_pretrained(str(ROOT / ".vlms" / "kosmos-2-patch14-224")).to(DEVICE).eval()
    results = {}
    t0 = time.time()
    for i, fr in enumerate(frames):
        prompt = "<grounding><phrase>gray laundry basket</phrase>"
        inp = proc(text=prompt, images=fr["img"], return_tensors="pt")
        inp = {k: v.to(DEVICE) for k, v in inp.items()}
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=64, use_cache=True)
        raw = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=False)[0]
        _caption, entities = proc.post_process_generation(raw)
        results[fr["key"]] = _kosmos_parse_refexp(entities)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(frames)} ({time.time()-t0:.0f}s)")
    del model; torch.cuda.empty_cache()
    return results


# ── OWL-v2 (Ow) ────────────────────────────────────────────────────────────────

def run_owlv2(frames):
    print("[OWL-v2] 로딩...")
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(DEVICE).eval()
    results = {}
    t0 = time.time()
    for i, fr in enumerate(frames):
        img = fr["img"]; W, H = img.width, img.height
        inp = proc(text=[["gray laundry basket"]], images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        res = proc.post_process_object_detection(out, threshold=0.1, target_sizes=[(H, W)])[0]
        boxes = res["boxes"]
        if len(boxes) == 0:
            results[fr["key"]] = None
        else:
            best = int(res["scores"].argmax())
            x1, y1, x2, y2 = boxes[best].cpu().tolist()
            results[fr["key"]] = {"cx": (x1+x2)/2/W, "x1": x1/W, "y1": y1/H, "x2": x2/W, "y2": y2/H}
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(frames)} ({time.time()-t0:.0f}s)")
    del model; torch.cuda.empty_cache()
    return results


HTML_HEAD = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>PG2 Fallback 프레임 — PG2 vs Kr vs Ow 비교</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ background: #0a0f1a; color: #e2e8f0; font-family: 'Segoe UI', sans-serif; padding: 32px 20px; }}
h1 {{ font-size: 1.3rem; margin-bottom: 6px; }}
.subtitle {{ color: #64748b; font-size: 0.85rem; margin-bottom: 24px; line-height: 1.6; }}
.legend {{ display: flex; gap: 16px; margin-bottom: 24px; font-size: 0.82rem; }}
.legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
.dot {{ width: 12px; height: 12px; border-radius: 3px; display: inline-block; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 10px; }}
.card {{ background: #0d1117; border: 1px solid #1e293b; border-radius: 8px; overflow: hidden; }}
.card-header {{ padding: 5px 8px; background: #111827; font-size: 0.65rem; color: #94a3b8; line-height: 1.5; }}
.card img {{ width: 100%; display: block; }}
</style>
</head>
<body>
<h1>PG2 Fallback 프레임 — PG2 / Kr(Kosmos refexp) / Ow(OWL-v2) 비교</h1>
<p class="subtitle">서버에서 has_bbox=False로 fallback 처리된 프레임 {n}개. 3개 모델 각각 독립적으로 그라운딩해서 겹쳐 표시.</p>
<div class="legend">
  <span><span class="dot" style="background:rgb(80,150,255)"></span>PG (PaliGemma2-448)</span>
  <span><span class="dot" style="background:rgb(60,220,60)"></span>Kr (Kosmos-2 refexp)</span>
  <span><span class="dot" style="background:rgb(240,200,0)"></span>Ow (OWL-v2)</span>
</div>
<div class="grid">
"""

CARD_TMPL = """  <div class="card">
    <div class="card-header">
      <span>{key}</span>
      <span>서버기록: cx={server_cx:.3f} area={server_area:.3f} has_bbox=False</span>
      <span>PG={pg_str}</span>
      <span>Kr={kr_str}</span>
      <span>Ow={ow_str}</span>
    </div>
    <img src="{fname}" loading="lazy">
  </div>
"""

HTML_TAIL = """</div>
</body>
</html>
"""


def fmt(r):
    if r is None:
        return "미검출"
    return f"cx={r['cx']:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = collect_fallback_frames(args.limit)
    print(f"대상 프레임: {len(frames)}개")

    pg_res = run_pg2(frames)
    kr_res = run_kosmos_refexp(frames)
    ow_res = run_owlv2(frames)

    cards = []
    for fr in frames:
        key = fr["key"]
        model_results = {"PG": pg_res.get(key), "Kr": kr_res.get(key), "Ow": ow_res.get(key)}
        thumb = draw_thumb(fr["img"], model_results)
        fname = f"thumb_{key}.jpg"
        thumb.save(OUT_DIR / fname, quality=85)
        cards.append(CARD_TMPL.format(
            key=key, server_cx=fr["server_cx"], server_area=fr["server_area"],
            pg_str=fmt(pg_res.get(key)), kr_str=fmt(kr_res.get(key)), ow_str=fmt(ow_res.get(key)),
            fname=fname,
        ))

    html = HTML_HEAD.format(n=len(frames)) + "".join(cards) + HTML_TAIL
    (OUT_DIR / "gallery.html").write_text(html)

    meta = [{"key": fr["key"], "episode": fr["episode"], "frame_idx": fr["frame_idx"],
             "server_cx": fr["server_cx"], "server_area": fr["server_area"],
             "pg": pg_res.get(fr["key"]), "kr": kr_res.get(fr["key"]), "ow": ow_res.get(fr["key"])}
            for fr in frames]
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"완료: {len(frames)}개 → {OUT_DIR}/gallery.html")


if __name__ == "__main__":
    main()

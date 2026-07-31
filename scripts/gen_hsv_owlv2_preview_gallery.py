#!/usr/bin/env python3
"""
CH54 프리뷰(콜드스타트 정렬) 대안 후보 셀프검증 갤러리 — HSV(원본) / HSV(재튜닝) / OWL-v2 /
PG2(참고용 기준) 4가지를 각 세션 초반 프레임(f0~f2)에 겹쳐 그려서 그리드로 비교.

배경: plan_20260703_hsv_preview_align.md 검증 1단계에서 HSV(원본 임계값)와 PG2가
15프레임 중 2개만 일치(13%) — 복도 배경(회색 벽)을 basket으로 오탐하는 것으로 추정.
재튜닝 변형(TOP_CUT/BOTTOM_CUT/BG_RATIO 조정)과 OWL-v2를 같이 놓고 사람이 직접
어느 게 실제로 basket을 잡는지 판단하기 위한 도구.

Usage:
  .venv/bin/python3 scripts/gen_hsv_owlv2_preview_gallery.py [--sessions N]
"""
import argparse
import glob
import json
import time
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
RECV_GLOB = "/home/minum/MoNaVLA/inference_sessions_recv/*/session_*.h5"
OUT_DIR = ROOT / "docs" / "v5" / "hsv_owlv2_preview_20260704"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THUMB_W, THUMB_H = 320, 180
FRAMES_PER_SESSION = 999  # 사실상 전체 프레임 (세션당 실제 길이로 자동 캡) — 최대 커버리지
EXCLUDE_SESSIONS = {"session_20260701_213204"}

MODEL_COLORS = {"H1": (60, 220, 60), "H2": (0, 200, 255), "Ow": (240, 200, 0),
                "PG": (255, 90, 90), "Kr": (200, 120, 255)}

# ── HSV 변형 1 (기존, robovlm_nav/perception/hsv_basket.py와 동일) ──────────────
H1_PARAMS = dict(S_MAX=20, V_MIN=70, V_MAX=230, TOP_CUT=0.20, BOTTOM_CUT=0.68,
                  MIN_AREA_PX=400, BG_RATIO=5.0)
# ── HSV 변형 2 (재튜닝 후보) — 벽/복도 배경 배제 강화: 상하단 더 좁게, 배경 판정 더 민감 ──
H2_PARAMS = dict(S_MAX=20, V_MIN=70, V_MAX=230, TOP_CUT=0.35, BOTTOM_CUT=0.60,
                  MIN_AREA_PX=400, BG_RATIO=2.5)


def detect_basket_cx_variant(img_rgb, p):
    H, W = img_rgb.shape[:2]
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask = (
        (hsv[:, :, 1] < p["S_MAX"]) & (hsv[:, :, 2] > p["V_MIN"]) & (hsv[:, :, 2] < p["V_MAX"])
    ).astype(np.uint8) * 255
    mask[: int(H * p["TOP_CUT"]), :] = 0
    mask[int(H * p["BOTTOM_CUT"]):, :] = 0
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if n_labels <= 1:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    valid = np.where(areas >= p["MIN_AREA_PX"])[0]
    if len(valid) == 0:
        return None
    sorted_valid = valid[np.argsort(areas[valid])[::-1]]
    if len(sorted_valid) >= 2:
        a1, a2 = areas[sorted_valid[0]], areas[sorted_valid[1]]
        if a1 >= a2 * p["BG_RATIO"]:
            chosen, conf = sorted_valid[1], float(a2 / (a1 + a2))
        else:
            chosen, conf = sorted_valid[0], float(a1 / (a1 + a2))
    else:
        chosen, conf = sorted_valid[0], 1.0
    best_idx = chosen + 1
    cx, cy = centroids[best_idx][0] / W, centroids[best_idx][1] / H
    area = areas[chosen] / (H * W)
    stat = stats[best_idx]
    x1 = stat[cv2.CC_STAT_LEFT] / W
    y1 = stat[cv2.CC_STAT_TOP] / H
    x2 = (stat[cv2.CC_STAT_LEFT] + stat[cv2.CC_STAT_WIDTH]) / W
    y2 = (stat[cv2.CC_STAT_TOP] + stat[cv2.CC_STAT_HEIGHT]) / H
    return {"cx": float(cx), "cy": float(cy), "area": float(area), "conf": float(conf),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2}


def collect_frames(max_sessions):
    files = sorted(glob.glob(RECV_GLOB))
    files = [f for f in files if Path(f).stem not in EXCLUDE_SESSIONS]
    if max_sessions:
        files = files[:max_sessions]
    frames = []
    for fp in files:
        stem = Path(fp).stem
        with h5py.File(fp, "r") as h:
            imgs = h["observations"]["images"]
            n_total = len(imgs)
            n = min(FRAMES_PER_SESSION, n_total)
            # 콜드스타트(f0)만 보지 않고 에피소드 전체에 고르게 퍼진 프레임을 샘플링
            idxs = sorted(set(int(round(x)) for x in np.linspace(0, n_total - 1, n)))
            for i in idxs:
                frames.append({
                    "img": Image.fromarray(np.array(imgs[i]).astype(np.uint8)).convert("RGB"),
                    "img_np": np.array(imgs[i]).astype(np.uint8),
                    "episode": stem, "frame_idx": i, "key": f"{stem}_f{i:02d}",
                })
    return frames


def run_pg2(frames):
    print("[PG2-448] 로딩...")
    from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
    import re
    loc_re = re.compile(r"<loc(\d{4})>")
    path = str(Path.home() / ".cache/huggingface/hub/models--google--paligemma2-3b-mix-448"
               "/snapshots/1406c92ec87d32cc6b983239278901b904ba7a51")
    proc = PaliGemmaProcessor.from_pretrained(path)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(DEVICE).eval()
    results = {}
    t0 = time.time()
    for i, fr in enumerate(frames):
        inp = proc(text="detect gray basket", images=fr["img"], return_tensors="pt").to(DEVICE)
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=48, min_new_tokens=1, do_sample=False)
        raw = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=False)[0]
        locs = [int(v) / 1023.0 for v in loc_re.findall(raw)]
        if len(locs) >= 4:
            y1, x1, y2, x2 = locs[:4]
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            results[fr["key"]] = {"cx": (x1 + x2) / 2, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
        else:
            results[fr["key"]] = None
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(frames)} ({time.time()-t0:.0f}s)")
    del model
    torch.cuda.empty_cache()
    return results


def run_kosmos_refexp(frames):
    """Kosmos-2 refexp (Kr) — gen_fallback_multimodel_gallery.py와 동일 방식."""
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
        found = None
        for ename, _span, boxes in entities:
            for box in boxes:
                x1, y1, x2, y2 = [float(v) for v in box]
                if max(x1, y1, x2, y2) > 1.5:
                    x1, y1, x2, y2 = x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000
                if (x2 - x1) * (y2 - y1) > 0.85:
                    continue
                if ename.startswith("<patch_index_"):
                    found = {"cx": (x1 + x2) / 2, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
                    break
            if found:
                break
        results[fr["key"]] = found
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(frames)} ({time.time()-t0:.0f}s)")
    del model
    torch.cuda.empty_cache()
    return results


def run_owlv2(frames):
    print("[OWL-v2] 로딩...")
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(DEVICE).eval()
    results = {}
    t0 = time.time()
    for i, fr in enumerate(frames):
        img = fr["img"]
        W, H = img.width, img.height
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
            results[fr["key"]] = {"cx": (x1 + x2) / 2 / W, "x1": x1 / W, "y1": y1 / H, "x2": x2 / W, "y2": y2 / H}
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(frames)} ({time.time()-t0:.0f}s)")
    del model
    torch.cuda.empty_cache()
    return results


def draw_thumb(img: Image.Image, model_results: dict) -> Image.Image:
    thumb = img.resize((THUMB_W, THUMB_H), Image.LANCZOS).convert("RGB")
    draw = ImageDraw.Draw(thumb)
    W, H = THUMB_W, THUMB_H
    for mname, r in model_results.items():
        if r is None:
            continue
        color = MODEL_COLORS[mname]
        x1, y1 = int(r["x1"] * W), int(r["y1"] * H)
        x2, y2 = int(r["x2"] * W), int(r["y2"] * H)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1 + 2, y1 + 1), mname, fill=color)
    return thumb


HTML_HEAD = """<!DOCTYPE html>
<html lang="ko"><head><meta charset="UTF-8"><title>CH54 프리뷰 후보 비교 — H1/H2/OWL-v2 vs PG2</title>
<style>
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ background:#0a0f1a; color:#e2e8f0; font-family:'Segoe UI',sans-serif; padding:28px 20px; }}
h1 {{ font-size:1.25rem; margin-bottom:6px; }}
.subtitle {{ color:#64748b; font-size:0.83rem; margin-bottom:18px; line-height:1.6; }}
.legend {{ display:flex; gap:16px; margin-bottom:20px; font-size:0.8rem; flex-wrap:wrap; }}
.legend span {{ display:inline-flex; align-items:center; gap:6px; }}
.dot {{ width:12px; height:12px; border-radius:3px; display:inline-block; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(320px,1fr)); gap:10px; }}
.card {{ background:#0d1117; border:1px solid #1e293b; border-radius:8px; overflow:hidden; }}
.card-header {{ padding:5px 8px; background:#111827; font-size:0.62rem; color:#94a3b8; line-height:1.5; }}
.card img {{ width:100%; display:block; }}
</style></head><body>
<h1>CH54 프리뷰 후보 비교 — H1(HSV 원본) / H2(HSV 재튜닝) / Ow(OWL-v2) vs PG(참고 기준)</h1>
<p class="subtitle">
H1: S_MAX=20 TOP_CUT=0.20 BOTTOM_CUT=0.68 BG_RATIO=5.0 (기존, robovlm_nav/perception/hsv_basket.py)<br>
H2: TOP_CUT=0.35 BOTTOM_CUT=0.60 BG_RATIO=2.5 (재튜닝 후보 — 벽/복도 배경 배제 강화)<br>
PG(빨강)는 참고용 기준선(느리지만 정확). H1/H2/Ow가 실제 basket을 잡는지 육안으로 확인.
</p>
<div class="legend">
  <span><span class="dot" style="background:rgb(60,220,60)"></span>H1 (HSV 원본)</span>
  <span><span class="dot" style="background:rgb(0,200,255)"></span>H2 (HSV 재튜닝)</span>
  <span><span class="dot" style="background:rgb(240,200,0)"></span>Ow (OWL-v2)</span>
  <span><span class="dot" style="background:rgb(255,90,90)"></span>PG (PaliGemma2, 참고 기준)</span>
</div>
<div class="grid">
"""
CARD_TMPL = """  <div class="card">
    <div class="card-header">{key}<br>H1={h1_str} H2={h2_str} Ow={ow_str} PG={pg_str}</div>
    <img src="{fname}" loading="lazy">
  </div>
"""
HTML_TAIL = "</div></body></html>"


def fmt(r):
    return "미검출" if r is None else f"cx={r['cx']:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=0, help="0=전체 세션")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = collect_frames(args.sessions or None)
    print(f"대상 프레임: {len(frames)}개")

    h1_res, h2_res = {}, {}
    for fr in frames:
        h1_res[fr["key"]] = detect_basket_cx_variant(fr["img_np"], H1_PARAMS)
        h2_res[fr["key"]] = detect_basket_cx_variant(fr["img_np"], H2_PARAMS)

    pg_res = run_pg2(frames)
    ow_res = run_owlv2(frames)
    kr_res = run_kosmos_refexp(frames)

    cards = []
    for fr in frames:
        key = fr["key"]
        model_results = {"H1": h1_res.get(key), "H2": h2_res.get(key),
                          "Ow": ow_res.get(key), "PG": pg_res.get(key),
                          "Kr": kr_res.get(key)}
        thumb = draw_thumb(fr["img"], model_results)
        fname = f"thumb_{key}.jpg"
        thumb.save(OUT_DIR / fname, quality=85)
        cards.append(CARD_TMPL.format(
            key=key, fname=fname,
            h1_str=fmt(h1_res.get(key)), h2_str=fmt(h2_res.get(key)),
            ow_str=fmt(ow_res.get(key)), pg_str=fmt(pg_res.get(key)),
        ))

    html = HTML_HEAD + "".join(cards) + HTML_TAIL
    (OUT_DIR / "gallery.html").write_text(html)

    meta = [{"key": fr["key"], "episode": fr["episode"], "frame_idx": fr["frame_idx"],
             "h1": h1_res.get(fr["key"]), "h2": h2_res.get(fr["key"]),
             "ow": ow_res.get(fr["key"]), "pg": pg_res.get(fr["key"]),
             "kr": kr_res.get(fr["key"])}
            for fr in frames]
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"완료: {len(frames)}개 → {OUT_DIR}/gallery.html")


if __name__ == "__main__":
    main()

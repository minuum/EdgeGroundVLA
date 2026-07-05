#!/usr/bin/env python3
"""truth_mini 72프레임(사람 정답 bbox) — PG2 / OWL-v2(th0.25) / Kosmos-refexp IoU 정밀 비교.

지표: IoU mean/median, cx MAE, coarse_position(L/C/R) 일치율.
visible(56) / partial(16) 분리 집계. 결과는 grounding_benchmark/results.json에 병합.

Usage: .venv/bin/python3 scripts/eval/eval_iou_truth_mini.py
"""
import json
import re
import statistics
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "ROS_action" / "mobile_vla_dataset_v5"
TRUTH = ROOT / "docs" / "v5" / "bbox_truth_mini.json"
OUT_DIR = ROOT / "docs" / "v5" / "grounding_benchmark"
RESULTS = OUT_DIR / "results.json"
PRED_CACHE = OUT_DIR / "truth_mini_preds.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OWL_THRESH = 0.25
PG2_PATH = str(Path.home() / ".cache/huggingface/hub/models--google--paligemma2-3b-mix-448"
               "/snapshots/1406c92ec87d32cc6b983239278901b904ba7a51")


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def cx_to_lcr(cx):
    return "left" if cx < 0.40 else ("right" if cx > 0.60 else "center")


def load_frames():
    truth = json.loads(TRUTH.read_text())["annotations"]
    frames = []
    for a in truth:
        h5p = DATA_DIR / f"{a['episode']}.h5"
        with h5py.File(h5p, "r") as h:
            img = np.array(h["observations"]["images"][a["frame_idx"]]).astype(np.uint8)
        frames.append({
            "key": f"{a['episode']}_f{a['frame_idx']:03d}",
            "img": Image.fromarray(img).convert("RGB"),
            "gt_bbox": a["bbox_xyxy_norm"],
            "gt_pos": a["coarse_position"],
            "visible": a["target_visible"],  # True | "partial"
        })
    return frames


# ── 모델 러너 (296 갤러리와 동일 설정) ─────────────────────────────────────────

def run_pg2(frames):
    from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
    loc_re = re.compile(r"<loc(\d{4})>")
    proc = PaliGemmaProcessor.from_pretrained(PG2_PATH)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        PG2_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(DEVICE).eval()
    out = {}
    for fr in frames:
        inp = proc(text="detect gray basket", images=fr["img"], return_tensors="pt").to(DEVICE)
        inp["pixel_values"] = inp["pixel_values"].to(torch.bfloat16)
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=48, min_new_tokens=1, do_sample=False)
        raw = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=False)[0]
        locs = [int(v) / 1023.0 for v in loc_re.findall(raw)]
        if len(locs) >= 4:
            y1, x1, y2, x2 = locs[:4]
            out[fr["key"]] = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
        else:
            out[fr["key"]] = None
    del model
    torch.cuda.empty_cache()
    return out


def run_owl(frames):
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble").to(DEVICE).eval()
    out = {}
    for fr in frames:
        img = fr["img"]
        W, H = img.width, img.height
        inp = proc(text=[["gray laundry basket"]], images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            o = model(**inp)
        res = proc.post_process_object_detection(o, threshold=OWL_THRESH, target_sizes=[(H, W)])[0]
        if len(res["boxes"]) == 0:
            out[fr["key"]] = None
        else:
            best = int(res["scores"].argmax())
            x1, y1, x2, y2 = res["boxes"][best].cpu().tolist()
            out[fr["key"]] = [x1 / W, y1 / H, x2 / W, y2 / H]
    del model
    torch.cuda.empty_cache()
    return out


def run_kr(frames):
    from transformers import AutoProcessor, AutoModelForVision2Seq
    proc = AutoProcessor.from_pretrained(str(ROOT / ".vlms" / "kosmos-2-patch14-224"))
    model = AutoModelForVision2Seq.from_pretrained(str(ROOT / ".vlms" / "kosmos-2-patch14-224")).to(DEVICE).eval()
    out = {}
    for fr in frames:
        inp = proc(text="<grounding><phrase>gray laundry basket</phrase>", images=fr["img"], return_tensors="pt")
        inp = {k: v.to(DEVICE) for k, v in inp.items()}
        with torch.no_grad():
            gen = model.generate(**inp, max_new_tokens=64, use_cache=True)
        raw = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=False)[0]
        _cap, entities = proc.post_process_generation(raw)
        found = None
        for ename, _s, boxes in entities:
            for box in boxes:
                x1, y1, x2, y2 = [float(v) for v in box]
                if max(x1, y1, x2, y2) > 1.5:
                    x1, y1, x2, y2 = x1 / 1000, y1 / 1000, x2 / 1000, y2 / 1000
                if (x2 - x1) * (y2 - y1) > 0.85:
                    continue
                if ename.startswith("<patch_index_"):
                    found = [x1, y1, x2, y2]
                    break
            if found:
                break
        out[fr["key"]] = found
    del model
    torch.cuda.empty_cache()
    return out


def summarize(frames, preds, name):
    per = {"all": [], "visible": [], "partial": []}
    cx_err, pos_hit, miss = [], 0, 0
    n_pos = 0
    for fr in frames:
        p = preds.get(fr["key"])
        if p is None:
            miss += 1
            continue
        v = iou(p, fr["gt_bbox"])
        per["all"].append(v)
        per["partial" if fr["visible"] == "partial" else "visible"].append(v)
        gt_cx = (fr["gt_bbox"][0] + fr["gt_bbox"][2]) / 2
        pcx = (p[0] + p[2]) / 2
        cx_err.append(abs(pcx - gt_cx))
        n_pos += 1
        if cx_to_lcr(pcx) == fr["gt_pos"]:
            pos_hit += 1
    return {
        "iou_mean": statistics.mean(per["all"]) if per["all"] else 0.0,
        "iou_median": statistics.median(per["all"]) if per["all"] else 0.0,
        "iou_mean_visible": statistics.mean(per["visible"]) if per["visible"] else 0.0,
        "iou_mean_partial": statistics.mean(per["partial"]) if per["partial"] else 0.0,
        "cx_mae": statistics.mean(cx_err) if cx_err else None,
        "lcr_match": pos_hit / n_pos if n_pos else 0.0,
        "n_detected": n_pos, "n_missed": miss, "n_total": len(frames),
    }


def main():
    frames = load_frames()
    print(f"truth_mini 프레임: {len(frames)}")

    if PRED_CACHE.exists():
        preds = json.loads(PRED_CACHE.read_text())
        print("기존 예측 캐시 재사용")
    else:
        preds = {}
        for name, fn in [("pg", run_pg2), ("ow_th025", run_owl), ("kr", run_kr)]:
            t0 = time.time()
            preds[name] = fn(frames)
            print(f"[{name}] {time.time()-t0:.0f}s")
        PRED_CACHE.write_text(json.dumps(preds, indent=2))

    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    print(f"\n{'model':<10}{'IoU_mean':>9}{'IoU_med':>9}{'vis':>7}{'part':>7}{'cxMAE':>8}{'L/C/R':>8}{'miss':>6}")
    for name in ["pg", "ow_th025", "kr"]:
        s = summarize(frames, preds[name], name)
        results.setdefault(name, {}).update({f"truthmini_{k}": v for k, v in s.items()})
        print(f"{name:<10}{s['iou_mean']:>9.3f}{s['iou_median']:>9.3f}{s['iou_mean_visible']:>7.3f}"
              f"{s['iou_mean_partial']:>7.3f}{s['cx_mae']:>8.3f}{100*s['lcr_match']:>7.1f}%{s['n_missed']:>6}")
    RESULTS.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n병합 저장: {RESULTS}")


if __name__ == "__main__":
    main()

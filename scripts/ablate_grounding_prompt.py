"""
CH58 — Kosmos-2 + OWL-v2 Grounding Prompt Ablation

39개 세션(2026-06-26) frame 1 이미지로 프롬프트/쿼리 변형별 bbox 검출 품질 측정.

Usage:
  .venv/bin/python3 -u scripts/ablate_grounding_prompt.py
  .venv/bin/python3 -u scripts/ablate_grounding_prompt.py --models kosmos
  .venv/bin/python3 -u scripts/ablate_grounding_prompt.py --models owlv2
"""
import argparse, io, json, time
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
import torch

ROOT      = Path(__file__).parent.parent
SESS_DIR  = ROOT / "docs" / "inference_sessions"
LABEL_F   = Path("/tmp/mona_labels.json")
OUT_FILE  = ROOT / "docs" / "v5" / "ablate_grounding_prompt.json"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

# ── 수동 레이블 (위치: L/C/R/NONE, 박스: FULL/PART_IN/PART_OUT/WRONG) ──────
MANUAL_POS = {
    "316.h5": "C",  "455.h5": "C",  "644.h5": "R",  "751.h5": "C",
    "013.h5": "L",  "622.h5": "C",  "837.h5": "L",  "247.h5": "C",
    "355.h5": "R",  "535.h5": "R",  "752.h5": "C",  "153.h5": "R",
    "337.h5": "R",  "536.h5": "C",  "658.h5": "C",  "823.h5": "L",
    "545.h5": "L",  "512.h5": "C",  "611.h5": "R",  "747.h5": "L",
    "853.h5": "L",  "643.h5": "L",  "814.h5": "R",
}  # 23개 위치 레이블 (NONE 제외)

MANUAL_BOX = {
    "316.h5": "PART_IN", "455.h5": "FULL",    "644.h5": "PART_IN",
    "751.h5": "FULL",    "013.h5": "PART_IN", "622.h5": "FULL",
    "837.h5": "PART_IN", "139.h5": "WRONG",   "247.h5": "WRONG",
    "355.h5": "WRONG",   "535.h5": "WRONG",   "752.h5": "PART_OUT",
    "928.h5": "WRONG",   "153.h5": "FULL",    "337.h5": "WRONG",
    "536.h5": "PART_OUT","658.h5": "FULL",    "823.h5": "PART_OUT",
    "109.h5": "WRONG",   "941.h5": "WRONG",   "011.h5": "PART_OUT",
    "058.h5": "PART_OUT","249.h5": "WRONG",   "600.h5": "WRONG",
    "935.h5": "WRONG",   "530.h5": "WRONG",   "545.h5": "FULL",
    "651.h5": "PART_OUT","758.h5": "PART_IN", "853.h5": "FULL",
    "945.h5": "WRONG",   "352.h5": "PART_IN", "512.h5": "FULL",
    "611.h5": "PART_OUT","747.h5": "PART_IN", "946.h5": "PART_IN",
    "643.h5": "PART_IN", "739.h5": "WRONG",   "814.h5": "PART_IN",
}

# ── Kosmos-2 변형 ──────────────────────────────────────────────────────────
KOSMOS_VARIANTS = {
    "K_current":        {"prompt": "<grounding>The basket is at",                         "mode": "completion"},
    "K_locate":         {"prompt": "<grounding>The gray basket is located at",            "mode": "completion"},
    "K_nav":            {"prompt": "<grounding>An image of a robot navigating toward the basket", "mode": "completion"},
    "K_refexp":         {"prompt": "<grounding><phrase>basket</phrase>",                   "mode": "refexp"},
    "K_refexp_gray":    {"prompt": "<grounding><phrase>gray basket</phrase>",              "mode": "refexp"},
    "K_refexp_laundry": {"prompt": "<grounding><phrase>gray laundry basket</phrase>",     "mode": "refexp"},
}

# ── OWL-v2 변형 ────────────────────────────────────────────────────────────
OWLV2_VARIANTS = {
    "O_current":   {"queries": [["gray basket"]]},
    "O_basket":    {"queries": [["basket"]]},
    "O_laundry":   {"queries": [["gray laundry basket"]]},
    "O_container": {"queries": [["gray container"]]},
    "O_multi":     {"queries": [["basket", "laundry basket", "gray container"]]},
}


def _raw_to_pil(raw):
    if raw.ndim == 1:
        return Image.open(io.BytesIO(bytes(raw))).convert("RGB")
    return Image.fromarray(raw.astype(np.uint8)).convert("RGB")


def load_sessions():
    """39개 세션 frame 1 이미지 + 기존 PG2 cx 로드"""
    sessions = []
    for sp in sorted(SESS_DIR.glob("session_20260626*.h5")):
        key = sp.name[-6:]  # e.g. "316.h5" (last 3 digits of HHMMSS + .h5)
        with h5py.File(sp) as f:
            n = f["observations/images"].shape[0]
            idx = min(1, n - 1)
            img = _raw_to_pil(f["observations/images"][idx])
            bbox = f["grounding/bbox"][idx]  # [cx, cy, area, has_bbox]
            pg2_cx = float(bbox[0])
            pg2_has = float(bbox[3]) > 0.5
        sessions.append({
            "key": key,
            "stem": sp.stem,
            "img": img,
            "pg2_cx": pg2_cx,
            "pg2_has": pg2_has,
            "manual_pos": MANUAL_POS.get(key),
            "manual_box": MANUAL_BOX.get(key),
        })
    print(f"[sessions] {len(sessions)}개 로드 완료")
    return sessions


def eval_variant(pred_cx_map: dict, sessions: list) -> dict:
    """예측 cx 딕셔너리 → 메트릭 계산"""
    det_n = sum(1 for s in sessions if pred_cx_map.get(s["key"]) is not None)
    det_rate = det_n / len(sessions)

    cxs = [v for v in pred_cx_map.values() if v is not None]
    cx_mean = float(np.mean(cxs)) if cxs else 0.5
    cx_std  = float(np.std(cxs))  if cxs else 0.0

    # dir_vs_manual: L/R 레이블 14개 기준
    dir_ok = dir_total = 0
    for s in sessions:
        pos = s["manual_pos"]
        if pos not in ("L", "R"):
            continue
        cx = pred_cx_map.get(s["key"])
        if cx is None:
            continue
        pred_side = "R" if cx > 0.5 else "L"
        dir_ok    += (pred_side == pos)
        dir_total += 1

    dir_vs_manual = dir_ok / dir_total if dir_total else 0.0

    return {
        "det_rate":       round(det_rate, 4),
        "det_n":          det_n,
        "cx_mean":        round(cx_mean, 4),
        "cx_std":         round(cx_std, 4),
        "dir_vs_manual":  round(dir_vs_manual, 4),
        "dir_n":          dir_total,
    }


def _print_r(tag, r, fb_rate=None):
    fb = f"  fallback={fb_rate:.1%}" if fb_rate is not None else ""
    print(f"  {tag:<28s} det={r['det_rate']:.1%}  cx={r['cx_mean']:.3f}±{r['cx_std']:.3f}"
          f"  dir_manual={r['dir_vs_manual']:.1%}({r['dir_n']}){fb}")


# ── Kosmos-2 블록 ──────────────────────────────────────────────────────────

def _parse_bbox_from_entities(entities, area_max=0.85):
    """entity 리스트에서 가장 큰 (바스켓) bbox 추출"""
    TARGET_KW = {"basket", "container", "bin", "laundry", "gray box", "pot"}
    candidates = []
    for ename, _span, boxes in entities:
        for box in boxes:
            x1, y1, x2, y2 = [float(v) for v in box]
            if max(x1, y1, x2, y2) > 1.5:
                x1, y1, x2, y2 = x1/1000, y1/1000, x2/1000, y2/1000
            area = (x2 - x1) * (y2 - y1)
            if area > area_max:
                continue
            is_target = any(k in ename.lower() for k in TARGET_KW)
            candidates.append({"cx": (x1+x2)/2, "area": area, "is_target": is_target, "entity": ename})
    matched = [c for c in candidates if c["is_target"]]
    if matched:
        return matched[0]["cx"], False
    if candidates:
        best = max(candidates, key=lambda c: c["area"])
        return best["cx"], False
    return None, False


_CAPTION_DIR = [
    (["far left","extreme left","leftmost","bottom left","lower left",
      "front left","left side","left corner","upper left","top left"], 0.12),
    (["left"], 0.25),
    (["far right","extreme right","rightmost","bottom right","lower right",
      "front right","right side","right corner","upper right","top right"], 0.88),
    (["right"], 0.75),
    (["center","middle","straight","front"], 0.50),
]

def _caption_to_cx(caption_lower):
    for phrases, cx in _CAPTION_DIR:
        if any(p in caption_lower for p in phrases):
            return cx
    return None


def run_kosmos(sessions):
    from transformers import AutoProcessor, AutoModelForVision2Seq
    print("\n[Kosmos-2] 로딩...")
    t0 = time.time()
    MODEL_PATH = str(ROOT / ".vlms" / "kosmos-2-patch14-224")
    proc  = AutoProcessor.from_pretrained(MODEL_PATH)
    model = AutoModelForVision2Seq.from_pretrained(MODEL_PATH).to(DEVICE).eval()
    print(f"[Kosmos-2] 로드 완료 {time.time()-t0:.1f}s")

    results = {}

    for variant_id, cfg in KOSMOS_VARIANTS.items():
        prompt = cfg["prompt"]
        mode   = cfg["mode"]
        t0 = time.time()
        pred_cx_map = {}
        fallback_n  = 0

        for s in sessions:
            img = s["img"]
            inputs = proc(text=prompt, images=img, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=64,
                    use_cache=True,
                )

            new_ids = generated[:, inputs["input_ids"].shape[1]:]
            raw = proc.batch_decode(new_ids, skip_special_tokens=False)[0]
            caption, entities = proc.post_process_generation(raw)

            cx, _ = _parse_bbox_from_entities(entities)
            if cx is None:
                # caption 폴백
                cx = _caption_to_cx(caption.lower())
                if cx is not None:
                    fallback_n += 1

            pred_cx_map[s["key"]] = cx

        lat = (time.time() - t0) * 1000 / len(sessions)
        fb_rate = fallback_n / len(sessions)
        r = eval_variant(pred_cx_map, sessions)
        r["lat_ms"]       = round(lat, 1)
        r["fallback_rate"]= round(fb_rate, 4)
        r["prompt"]       = prompt
        r["mode"]         = mode
        results[variant_id] = r
        _print_r(f"[{variant_id}]", r, fb_rate)

    return results


# ── OWL-v2 블록 ────────────────────────────────────────────────────────────

def run_owlv2(sessions):
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    print("\n[OWL-v2] 로딩...")
    t0 = time.time()
    proc  = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(DEVICE).eval()
    print(f"[OWL-v2] 로드 완료 {time.time()-t0:.1f}s")

    results = {}

    for variant_id, cfg in OWLV2_VARIANTS.items():
        queries = cfg["queries"]  # e.g. [["gray basket"]] or [["basket","laundry basket",...]]
        t0 = time.time()
        pred_cx_map = {}

        for s in sessions:
            img  = s["img"]
            W    = img.width
            inp  = proc(text=queries, images=img, return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                out = model(**inp)
            res  = proc.post_process_object_detection(out, threshold=0.1,
                       target_sizes=[(img.height, img.width)])[0]
            boxes = res["boxes"]
            if len(boxes) == 0:
                pred_cx_map[s["key"]] = None
            else:
                best = int(res["scores"].argmax())
                x1, _, x2, _ = boxes[best].cpu().tolist()
                pred_cx_map[s["key"]] = (x1 + x2) / 2 / W

        lat = (time.time() - t0) * 1000 / len(sessions)
        r = eval_variant(pred_cx_map, sessions)
        r["lat_ms"]  = round(lat, 1)
        r["queries"] = queries[0]
        results[variant_id] = r
        _print_r(f"[{variant_id}]", r)

    return results


# ── PG2 베이스라인 (세션 당시 저장값) ──────────────────────────────────────

def baseline_pg2(sessions):
    pred_cx_map = {}
    for s in sessions:
        pred_cx_map[s["key"]] = s["pg2_cx"] if s["pg2_has"] else None
    r = eval_variant(pred_cx_map, sessions)
    r["source"] = "session_h5_grounding/bbox[1]"
    _print_r("[PG2_baseline]", r)
    return r


# ── 결과 출력 테이블 ────────────────────────────────────────────────────────

def print_summary(all_results):
    print("\n" + "="*70)
    print(f"{'variant':<28} {'det':>5} {'cx':>6} {'std':>5} {'dir/14':>7} {'lat':>7} {'fb':>5}")
    print("-"*70)
    for vid, r in all_results.items():
        fb = f"{r['fallback_rate']:.0%}" if "fallback_rate" in r else "  —"
        print(f"{vid:<28} {r['det_rate']:>4.0%}  {r['cx_mean']:>5.3f} {r['cx_std']:>5.3f}"
              f"  {r['dir_vs_manual']:>5.0%}/{r['dir_n']}"
              f"  {r.get('lat_ms',0):>5.0f}ms  {fb}")
    print("="*70)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["kosmos","owlv2"],
                   choices=["kosmos","owlv2"])
    args = p.parse_args()

    sessions = load_sessions()
    all_results = {}

    print("\n--- PG2 베이스라인 (세션 당시 추론값) ---")
    all_results["PG2_baseline"] = baseline_pg2(sessions)

    if "kosmos" in args.models:
        k_res = run_kosmos(sessions)
        all_results.update(k_res)

    if "owlv2" in args.models:
        o_res = run_owlv2(sessions)
        all_results.update(o_res)

    print_summary(all_results)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    # 기존 파일과 머지 (img 제외 직렬화 가능한 값만)
    existing = {}
    if OUT_FILE.exists():
        try:
            existing = json.load(open(OUT_FILE))
        except Exception:
            pass
    existing["ch58_prompt_ablation"] = all_results
    json.dump(existing, open(OUT_FILE, "w"), indent=2, ensure_ascii=False)
    print(f"\n저장 → {OUT_FILE}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
VLM Text Recognition — 150 에피소드 배치 실행

Kosmos-2가 우리 복도 프레임을 보고:
  (1) Grounding: "gray basket" bbox 반환하는가
  (2) Free description: "basket" 키워드를 언급하는가

결과를 CSV + JSON으로 증분 저장 (중단돼도 이어서 실행 가능).

Usage:
  .venv/bin/python3 scripts/run_vlm_text_recognition.py
  .venv/bin/python3 scripts/run_vlm_text_recognition.py --frames-per-ep 5
  .venv/bin/python3 scripts/run_vlm_text_recognition.py --resume        # 기존 결과 이어서
  .venv/bin/python3 scripts/run_vlm_text_recognition.py --only-grounding  # grounding만
"""

import argparse, csv, json, sys, time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VLM_PATH  = ROOT / ".vlms" / "kosmos-2-patch14-224"
DATA_PATH = ROOT / "docs" / "v5" / "bbox_nav_exp46" / "bbox_dataset_full.json"
OUT_DIR   = ROOT / "docs" / "v5" / "vlm_recognition"

CSV_PATH  = OUT_DIR / "vlm_text_recognition.csv"
JSON_PATH = OUT_DIR / "vlm_text_recognition.json"

CLASS_NAMES = ["STOP","FORWARD","LEFT","RIGHT","FWD+L","FWD+R","ROT_L","ROT_R"]

KEYWORDS_BASKET   = ["basket", "baskets"]
KEYWORDS_GRAY     = ["gray", "grey"]
KEYWORDS_CORRIDOR = ["corridor", "hallway", "hall", "aisle", "passage"]
KEYWORDS_OBJECT   = ["container", "bin", "box", "cart", "trolley"]

CSV_FIELDS = [
    "ep_idx", "ep_stem", "path_type",
    "frame_t", "frame_idx", "has_bbox", "gt_class",
    "cx", "cy", "area",
    # grounding (전체 — ANY phrase)
    "grounding_any_success",   # 어떤 phrase든 bbox 나왔으면 True
    "grounding_n_boxes",
    "grounding_phrases",       # pipe-separated: "gray basket|a trash can"
    # grounding (basket-specific — "basket" 포함 phrase만)
    "basket_grounding_success", # "gray basket" 등 basket phrase에서만 bbox 나왔으면 True
    "basket_iou",               # basket phrase 박스 기준 IoU (신뢰 가능)
    "basket_bbox_str",          # basket phrase 박스 좌표
    # side-phrase (basket 아닌 것들)
    "side_phrases",             # basket 아닌 phrase들: "a white wall|a window"
    # free description
    "free_text",
    "mentions_basket", "mentions_gray", "mentions_corridor", "mentions_object",
    "text_len",
]


# ─── 샘플링 ──────────────────────────────────────────────

def sample_frames(ep, n_per_ep=5):
    """
    에피소드당 n_per_ep 프레임 샘플.
    has_bbox=True 위주로 60%, has_bbox=False 40% 비율.
    """
    frames = ep["frames"]
    present = [i for i, f in enumerate(frames) if f.get("has_bbox")]
    absent  = [i for i, f in enumerate(frames) if not f.get("has_bbox")]

    rng = np.random.default_rng(hash(ep["episode"]) % (2**31))

    n_pres = min(len(present), max(1, int(n_per_ep * 0.6)))
    n_abs  = min(len(absent),  n_per_ep - n_pres)
    n_pres = min(len(present), n_per_ep - n_abs)

    chosen = []
    if present:
        chosen += rng.choice(present, size=n_pres, replace=False).tolist()
    if absent:
        chosen += rng.choice(absent,  size=n_abs,  replace=False).tolist()

    return sorted(set(chosen))


# ─── 텍스트 분석 ─────────────────────────────────────────

def analyze_text(text):
    t = text.lower()
    return {
        "mentions_basket":   any(k in t for k in KEYWORDS_BASKET),
        "mentions_gray":     any(k in t for k in KEYWORDS_GRAY),
        "mentions_corridor": any(k in t for k in KEYWORDS_CORRIDOR),
        "mentions_object":   any(k in t for k in KEYWORDS_OBJECT),
        "text_len":          len(text.split()),
    }


def bbox_iou_from_area(cx, cy, area, pred_boxes):
    """
    GT bbox: square approximation (area = w*h 비율, assume square).
    pred_boxes: list of (x1,y1,x2,y2) in [0,1].
    """
    if not pred_boxes or area <= 0:
        return 0.0
    side = area ** 0.5
    gx1, gy1 = cx - side/2, cy - side/2
    gx2, gy2 = cx + side/2, cy + side/2

    best = 0.0
    for (px1, py1, px2, py2) in pred_boxes:
        ix1 = max(gx1, px1); iy1 = max(gy1, py1)
        ix2 = min(gx2, px2); iy2 = min(gy2, py2)
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        if inter == 0:
            continue
        union = (gx2-gx1)*(gy2-gy1) + (px2-px1)*(py2-py1) - inter
        best  = max(best, inter/union if union > 0 else 0.0)
    return round(best, 4)


# ─── VLM 추론 ────────────────────────────────────────────

def run_grounding(proc, model, pil_img, device):
    """
    Returns:
        any_success:     bool  — 어떤 phrase든 bbox 있으면 True
        n_boxes:         int   — 전체 박스 수
        basket_boxes:    list  — "basket" 포함 phrase의 박스만 (신뢰 가능)
        basket_phrases:  list  — basket phrase 이름들
        side_phrases:    list  — basket 아닌 phrase 이름들 (hallucination 후보)
        all_phrases:     list  — 전체 phrase 이름들
    """
    prompt = "<grounding><phrase>gray basket</phrase>"
    inputs = proc(text=prompt, images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}
    with torch.no_grad():
        gen = model.generate(**inputs, use_cache=True, max_new_tokens=64)
    raw = proc.batch_decode(gen, skip_special_tokens=False)[0]
    try:
        _, entities = proc.post_process_generation(raw, cleanup_and_extract=True)
    except Exception:
        entities = []

    basket_boxes, basket_phrases = [], []
    side_phrases  = []
    all_phrases   = []
    n_boxes = 0

    for phrase, _, bboxes in entities:
        ph = phrase.strip()
        if not bboxes:
            continue
        n_boxes += len(bboxes)
        all_phrases.append(ph)
        if "basket" in ph.lower():
            basket_boxes.extend(bboxes)
            basket_phrases.append(ph)
        else:
            side_phrases.append(ph)

    any_success = n_boxes > 0
    return any_success, n_boxes, basket_boxes, basket_phrases, side_phrases, all_phrases


def run_free_desc(proc, model, pil_img, device):
    prompt = "An image of"
    inputs = proc(text=prompt, images=pil_img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items() if v is not None}
    with torch.no_grad():
        gen = model.generate(**inputs, use_cache=True, max_new_tokens=128)
    clean = proc.batch_decode(gen, skip_special_tokens=True)[0].strip()
    return clean


# ─── 메인 ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-per-ep",  type=int, default=5)
    parser.add_argument("--resume",         action="store_true",
                        help="기존 CSV에서 처리된 에피소드 스킵")
    parser.add_argument("--only-grounding", action="store_true",
                        help="free description 생략 (속도 2배)")
    parser.add_argument("--ep-limit",       type=int, default=None,
                        help="디버그용: 처음 N 에피소드만")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 이미 처리된 에피소드 확인
    done_stems = set()
    if args.resume and CSV_PATH.exists():
        with open(CSV_PATH, newline="") as f:
            for row in csv.DictReader(f):
                done_stems.add(row["ep_stem"])
        print(f"[RESUME] 기존 완료: {len(done_stems)} 에피소드")

    # 데이터 로드
    data = json.loads(DATA_PATH.read_text())
    if args.ep_limit:
        data = data[:args.ep_limit]
    print(f"[DATA] {len(data)} 에피소드", flush=True)

    # VLM 로드
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    print("[VLM] Kosmos-2 로딩...", flush=True)
    from transformers import AutoProcessor, AutoModelForVision2Seq
    proc  = AutoProcessor.from_pretrained(str(VLM_PATH))
    model = AutoModelForVision2Seq.from_pretrained(
        str(VLM_PATH), torch_dtype=torch.float16
    ).to(device)
    model.eval()
    print("[VLM] 로드 완료\n", flush=True)

    # CSV 초기화 (resume이면 append, 아니면 새로 쓰기)
    csv_mode = "a" if args.resume and CSV_PATH.exists() else "w"
    csv_file = open(CSV_PATH, csv_mode, newline="", encoding="utf-8")
    writer   = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if csv_mode == "w":
        writer.writeheader()

    all_rows = []
    t0 = time.time()
    total_frames = 0

    for ep_i, ep in enumerate(data):
        ep_stem = Path(ep["episode"]).stem
        if ep_stem in done_stems:
            print(f"  [{ep_i+1}/{len(data)}] SKIP {ep_stem}", flush=True)
            continue

        frame_indices = sample_frames(ep, args.frames_per_ep)
        ep_rows = []
        ep_ok = True

        try:
            with h5py.File(ep["episode"], "r") as hf:
                images_dset = hf["observations"]["images"]

                for t in frame_indices:
                    fr       = ep["frames"][t]
                    frame_idx = fr.get("frame_idx", t)
                    pil_img   = Image.fromarray(images_dset[frame_idx])

                    row = {
                        "ep_idx":    ep_i,
                        "ep_stem":   ep_stem,
                        "path_type": ep["path_type"],
                        "frame_t":   t,
                        "frame_idx": frame_idx,
                        "has_bbox":  fr.get("has_bbox", False),
                        "gt_class":  CLASS_NAMES[fr["gt_class"]],
                        "cx":        round(fr.get("cx", 0.5), 4),
                        "cy":        round(fr.get("cy", 0.5), 4),
                        "area":      round(fr.get("area", 0.0), 5),
                    }

                    # ── Grounding ──
                    any_ok, g_n, basket_boxes, basket_phs, side_phs, all_phs = \
                        run_grounding(proc, model, pil_img, device)

                    basket_ok  = bool(basket_boxes)
                    basket_iou = (
                        bbox_iou_from_area(fr["cx"], fr["cy"], fr["area"], basket_boxes)
                        if fr.get("has_bbox") and basket_boxes else 0.0
                    )

                    row["grounding_any_success"]    = any_ok
                    row["grounding_n_boxes"]        = g_n
                    row["grounding_phrases"]        = "|".join(all_phs)
                    row["basket_grounding_success"] = basket_ok
                    row["basket_iou"]               = round(basket_iou, 4)
                    row["basket_bbox_str"]          = str(basket_boxes[:2]) if basket_boxes else ""
                    row["side_phrases"]             = "|".join(side_phs)

                    # ── Free description ──
                    if args.only_grounding:
                        free_text = ""
                        row.update({"free_text":"","mentions_basket":False,
                                    "mentions_gray":False,"mentions_corridor":False,
                                    "mentions_object":False,"text_len":0})
                    else:
                        free_text = run_free_desc(proc, model, pil_img, device)
                        row["free_text"] = free_text
                        row.update(analyze_text(free_text))

                    ep_rows.append(row)
                    total_frames += 1

                    b_icon     = "📦" if fr.get("has_bbox") else "  "
                    bsk_icon   = "✅" if basket_ok else "❌"
                    side_str   = f" +side=[{', '.join(side_phs)}]" if side_phs else ""
                    print(f"    {b_icon} t={t:3d} basket={bsk_icon} iou={basket_iou:.2f}{side_str}",
                          flush=True)

        except Exception as e:
            print(f"  [ERROR] {ep_stem}: {e}", flush=True)
            ep_ok = False

        if ep_ok and ep_rows:
            writer.writerows(ep_rows)
            csv_file.flush()
            all_rows.extend(ep_rows)

        elapsed = time.time() - t0
        fps = total_frames / elapsed if elapsed > 0 else 0
        remaining = (len(data) - ep_i - 1) * args.frames_per_ep / fps if fps > 0 else 0
        print(f"  [{ep_i+1}/{len(data)}] {ep_stem[:30]:30s}  "
              f"frames={len(ep_rows)}  "
              f"elapsed={elapsed/60:.1f}m  ETA={remaining/60:.1f}m",
              flush=True)

    csv_file.close()

    # JSON 저장
    json_rows = []
    if args.resume and JSON_PATH.exists():
        json_rows = json.loads(JSON_PATH.read_text())
    json_rows.extend(all_rows)
    JSON_PATH.write_text(json.dumps(json_rows, indent=2, ensure_ascii=False))

    # ── 요약 통계 ──
    print(f"\n{'='*60}")
    print(f"[완료] 총 {len(json_rows)} 프레임 처리됨")
    print(f"       결과: {CSV_PATH}")

    pres_rows = [r for r in json_rows if r["has_bbox"]]
    abs_rows  = [r for r in json_rows if not r["has_bbox"]]

    if pres_rows:
        gr = sum(1 for r in pres_rows if r.get("basket_grounding_success"))
        mk = sum(1 for r in pres_rows if r.get("mentions_basket"))
        avg_iou = sum(float(r.get("basket_iou", 0)) for r in pres_rows) / len(pres_rows)
        print(f"\n  [basket 있는 프레임 {len(pres_rows)}개]")
        print(f"    grounding 성공:    {gr}/{len(pres_rows)} = {gr/len(pres_rows)*100:.1f}%")
        print(f"    avg grounding IoU: {avg_iou:.3f}")
        print(f"    텍스트에서 'basket' 언급: {mk}/{len(pres_rows)} = {mk/len(pres_rows)*100:.1f}%")

    if abs_rows:
        fp = sum(1 for r in abs_rows if r.get("basket_grounding_success"))
        fm = sum(1 for r in abs_rows if r.get("mentions_basket"))
        print(f"\n  [basket 없는 프레임 {len(abs_rows)}개]")
        print(f"    grounding false positive: {fp}/{len(abs_rows)} = {fp/len(abs_rows)*100:.1f}%")
        print(f"    텍스트 false positive:   {fm}/{len(abs_rows)} = {fm/len(abs_rows)*100:.1f}%")

    # path_type별 grounding 성공률
    by_path = {}
    for r in json_rows:
        pt = r["path_type"]
        if pt not in by_path:
            by_path[pt] = {"pres_total":0,"pres_gr":0}
        if r["has_bbox"]:
            by_path[pt]["pres_total"] += 1
            if r.get("basket_grounding_success"):
                by_path[pt]["pres_gr"] += 1

    print(f"\n  [path_type별 grounding 성공률 (basket 있는 프레임)]")
    for pt in sorted(by_path):
        d = by_path[pt]
        if d["pres_total"] > 0:
            rate = d["pres_gr"]/d["pres_total"]*100
            print(f"    {pt:20s}: {d['pres_gr']:3d}/{d['pres_total']:3d} = {rate:.1f}%")

    print(f"\n  CSV: {CSV_PATH}")
    print(f"  JSON: {JSON_PATH}")


if __name__ == "__main__":
    main()

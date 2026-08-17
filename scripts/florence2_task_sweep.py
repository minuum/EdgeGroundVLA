#!/usr/bin/env python3
"""Florence-2 박스 생성기 — 태스크 프롬프트 스윕 (2026-08-17).

배경:
  l1_florence2_grounder.py는 <CAPTION_TO_PHRASE_GROUNDING> **한 가지만** 시도했고,
  그 결과 detect_rate가 없는 물체까지 전부 1.00으로 나와 "그라운더로 실패" 판정을 받았다.

  그런데 <CAPTION_TO_PHRASE_GROUNDING>은 설계상 "그 구절이 이미지에 존재한다"는 전제로
  캡션 내 구절을 이미지에 정렬시키는 태스크다 — **거부(not found) 출력 모드가 없다.**
  detect_rate 1.00은 모델 결함이라기보다 태스크 선택의 결과일 수 있다.

  반면 <OD> / <OPEN_VOCABULARY_DETECTION>은 "찾은 것만 반환"하는 의미론이라
  없는 물체에 대해 빈 결과를 낼 수 있다 → has_bbox=0 신호를 얻을 가능성.

  또한 백본으로서는 이미 통과했다(CH67 67-2: cx MAE 0.00152 vs Kosmos-2 0.0020).
  즉 피처는 멀쩡하고 문제는 출력부다. 이 스윕은 출력부만 겨냥한다.

측정:
  · 존재 객체(gray basket)  → detect_rate 높아야 정상
  · 부재 객체(person, microwave oven) → detect_rate 낮아야 정상 (이게 핵심)
  · 판별력 = present_rate - absent_rate. CAPTION_TO_PHRASE_GROUNDING은 0에 가까웠다.
  · cx 범위 — CH59의 우편향(cx max 0.559) 재현 여부

주의: 파이프라인에 끼워넣지 않는다. 출력 품질만 독립 측정한다.

출력: docs/v5/detector/florence2_task_sweep.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
ANN = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_v6_pg448_cx.json"
OUT = ROOT / "docs/v5/detector/florence2_task_sweep.json"
SPLIT_SEED, VAL_RATIO = 42, 0.15
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = "microsoft/Florence-2-base"

# 태스크별 (프롬프트 접미 필요 여부, 결과 키)
TASKS = {
    "<CAPTION_TO_PHRASE_GROUNDING>": True,   # 기존 실패본 — 대조군
    "<OPEN_VOCABULARY_DETECTION>":   True,   # 거부 가능성 있음
    "<OD>":                          False,  # 프롬프트 없음, 찾은 것 전부 라벨과 함께
}
PRESENT = "gray basket"
ABSENT = ["person", "microwave oven"]


def val_frames(n, seed=0):
    """l1_florence2_grounder.py와 동일 함수·동일 seed — 같은 프레임 보장."""
    ann = json.loads(ANN.read_text())
    rng = np.random.default_rng(SPLIT_SEED)
    idx = list(range(len(ann)))
    rng.shuffle(idx)
    val_eps = set(idx[:max(1, int(len(idx) * VAL_RATIO))])
    rows = [dict(ep=ep["episode"], fi=f["frame_idx"])
            for ei, ep in enumerate(ann) if ei in val_eps
            for f in ep["frames"] if not f["grounding_cached"] and f["detected"]]
    r2 = np.random.default_rng(seed)
    pick = r2.choice(len(rows), size=min(n, len(rows)), replace=False)
    return [rows[i] for i in sorted(pick)]


def run_task(model, proc, pil, task, phrase, needs_phrase):
    """한 프레임·한 태스크 실행 → 박스 리스트와 라벨."""
    W, H = pil.width, pil.height
    text = task + phrase if needs_phrase else task
    inp = proc(text=text, images=pil, return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(
            input_ids=inp["input_ids"].to(DEV),
            pixel_values=inp["pixel_values"].to(DEV, torch.float16),
            max_new_tokens=128, num_beams=3, do_sample=False)
    txt = proc.batch_decode(ids, skip_special_tokens=False)[0]
    parsed = proc.post_process_generation(txt, task=task, image_size=(W, H))[task]
    boxes = parsed.get("bboxes", []) or []
    labels = parsed.get("bboxes_labels", parsed.get("labels", [])) or []
    return boxes, labels, W, H


def norm_box(b, W, H):
    x1, y1, x2, y2 = b
    return dict(cx=(x1 + x2) / 2 / W, cy=(y1 + y2) / 2 / H,
                area=(x2 - x1) * (y2 - y1) / (W * H))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoProcessor
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.float16).to(DEV).eval()
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

    rows = val_frames(args.n)
    print(f"표본 {len(rows)} 프레임 · 태스크 {len(TASKS)}종", flush=True)

    # results[task][phrase] = list of (detected: bool, box or None)
    res = {t: defaultdict(list) for t in TASKS}
    od_labels = defaultdict(int)   # <OD>가 실제로 무슨 라벨을 뱉는지 수집

    cur, hf = None, None
    for n, r in enumerate(rows):
        if r["ep"] != cur:
            if hf is not None:
                hf.close()
            hf = h5py.File(r["ep"], "r"); cur = r["ep"]
        im = np.ascontiguousarray(np.array(hf["images"][r["fi"]])[:, :, ::-1])
        pil = Image.fromarray(im.astype(np.uint8)).convert("RGB")

        for task, needs_phrase in TASKS.items():
            if task == "<OD>":
                # 프롬프트 없이 1회 실행 → 라벨 중 basket 계열이 있는지로 판정
                boxes, labels, W, H = run_task(model, proc, pil, task, "", False)
                for lb in labels:
                    od_labels[str(lb).lower()] += 1
                hit = [b for b, lb in zip(boxes, labels)
                       if "basket" in str(lb).lower() or "bin" in str(lb).lower()]
                res[task]["gray basket"].append(
                    norm_box(max(hit, key=lambda b: (b[2]-b[0])*(b[3]-b[1])), W, H)
                    if hit else None)
                continue

            for ph in [PRESENT] + ABSENT:
                boxes, labels, W, H = run_task(model, proc, pil, task, ph, needs_phrase)
                if not boxes:
                    res[task][ph].append(None)
                else:
                    best = max(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                    res[task][ph].append(norm_box(best, W, H))

        if (n + 1) % 10 == 0:
            print(f"  {n+1}/{len(rows)}", flush=True)
    if hf is not None:
        hf.close()

    # ── 집계 ────────────────────────────────────────────────────────────
    rep = {"n_frames": len(rows), "model": MODEL, "tasks": {}}
    for task in TASKS:
        t = {}
        for ph, lst in res[task].items():
            got = [b for b in lst if b is not None]
            cxs = [b["cx"] for b in got]
            t[ph] = {
                "detect_rate": len(got) / len(lst) if lst else 0.0,
                "n": len(lst),
                "cx_mean": float(np.mean(cxs)) if cxs else None,
                "cx_std": float(np.std(cxs)) if cxs else None,
                "cx_min": float(np.min(cxs)) if cxs else None,
                "cx_max": float(np.max(cxs)) if cxs else None,
                "area_mean": float(np.mean([b["area"] for b in got])) if got else None,
            }
        # 판별력 = 존재 객체 검출률 − 부재 객체 평균 검출률
        if PRESENT in t:
            absent_rates = [t[a]["detect_rate"] for a in ABSENT if a in t]
            if absent_rates:
                t["_discrimination"] = t[PRESENT]["detect_rate"] - float(np.mean(absent_rates))
        rep["tasks"][task] = t
    rep["od_labels_seen"] = dict(sorted(od_labels.items(), key=lambda kv: -kv[1])[:25])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2, ensure_ascii=False))

    print("\n" + "=" * 68)
    for task, t in rep["tasks"].items():
        print(f"\n{task}")
        for ph in [PRESENT] + ABSENT:
            if ph not in t:
                continue
            d = t[ph]
            cxr = (f"cx {d['cx_min']:.3f}~{d['cx_max']:.3f}"
                   if d["cx_min"] is not None else "cx —")
            print(f"  {ph:18s} detect {d['detect_rate']:5.1%}  {cxr}")
        if "_discrimination" in t:
            print(f"  → 판별력(존재−부재): {t['_discrimination']:+.3f}")
    print(f"\n<OD>가 본 라벨 상위: {list(rep['od_labels_seen'])[:10]}")
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()

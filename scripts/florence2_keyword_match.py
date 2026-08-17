#!/usr/bin/env python3
"""Florence-2 — 정답 없이 박스를 고를 수 있는가 (2026-08-17).

왜 이 실험이 필요한가:
  florence2_od_recall_boost.py의 "재현율(라벨무관)"은 **오라클 상한**이다.
  정답 cx로 어느 박스가 맞는지 골랐기 때문이다. 실제 검출기는 정답 없이 골라야 한다.
  이 스크립트는 **키워드 매칭만으로** 박스를 고르고, 그 선택이 맞았는지를 잰다.

앞선 발견:
  <DENSE_REGION_CAPTION>은 타겟을 정확히 서술한다 —
  "gray plastic laundry hamper with perforated design"(25회),
  "a gray trash can with a perforated design on the front."(2회) 등.
  고정 어휘가 아니라 자유 캡션이므로 부분문자열 매칭이 필요하다.

측정 (정답을 선택에 쓰지 않는다):
  1. 키워드로 박스 선택 → 그 박스가 정답과 |Δcx|<=0.05면 적중
  2. recall    = 적중 프레임 / 전체 프레임
  3. precision = 적중 / 키워드가 뭐라도 고른 프레임   ← 오탐 여부
  4. OD ∪ DENSE 합집합 효과

비교 기준: OWL-v2 실기 검출률 90.5% (2026-08-07 100세션 실측)

주의: 파이프라인에 끼워넣지 않는다. 출력 품질만 독립 측정.

출력: docs/v5/detector/florence2_keyword_match.json
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
OUT = ROOT / "docs/v5/detector/florence2_keyword_match.json"
SPLIT_SEED, VAL_RATIO = 42, 0.15
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = "microsoft/Florence-2-base"
HIT_TOL = 0.05
OWL_BASELINE = 0.905          # 2026-08-07 100세션 실측 985/1088

# 타겟을 가리키는 표현들 — 앞선 실험에서 실제로 관측된 어휘에서 도출
KEYWORDS = ["hamper", "basket", "trash can", "trash bin", "waste container",
            "waste bin", "wastebasket", "bin", "container"]


def val_frames(n, seed=0):
    ann = json.loads(ANN.read_text())
    rng = np.random.default_rng(SPLIT_SEED)
    idx = list(range(len(ann)))
    rng.shuffle(idx)
    val_eps = set(idx[:max(1, int(len(idx) * VAL_RATIO))])
    rows = [dict(ep=ep["episode"], fi=f["frame_idx"], gt_cx=f.get("cx_det"))
            for ei, ep in enumerate(ann) if ei in val_eps
            for f in ep["frames"]
            if not f["grounding_cached"] and f["detected"] and f.get("cx_det") is not None]
    r2 = np.random.default_rng(seed)
    pick = r2.choice(len(rows), size=min(n, len(rows)), replace=False)
    return [rows[i] for i in sorted(pick)]


def gen(model, proc, pil, task, beams=3):
    W, H = pil.width, pil.height
    inp = proc(text=task, images=pil, return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(
            input_ids=inp["input_ids"].to(DEV),
            pixel_values=inp["pixel_values"].to(DEV, torch.float16),
            max_new_tokens=256, num_beams=beams, do_sample=False)
    txt = proc.batch_decode(ids, skip_special_tokens=False)[0]
    parsed = proc.post_process_generation(txt, task=task, image_size=(W, H))[task]
    boxes = parsed.get("bboxes", []) or []
    labels = (parsed.get("labels") or parsed.get("bboxes_labels") or [])
    out = []
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b
        lb = str(labels[i]).lower().strip() if i < len(labels) else ""
        out.append(dict(label=lb, cx=(x1 + x2) / 2 / W,
                        area=(x2 - x1) * (y2 - y1) / (W * H)))
    return out


def pick_by_keyword(dets):
    """정답을 쓰지 않고 키워드만으로 박스 선택. 여러 개면 최대 면적."""
    hits = [d for d in dets if any(k in d["label"] for k in KEYWORDS)]
    if not hits:
        return None
    return max(hits, key=lambda d: d["area"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoProcessor
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.float16).to(DEV).eval()
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

    rows = val_frames(args.n)
    print(f"표본 {len(rows)} 프레임 · 키워드 {len(KEYWORDS)}종 · 정답 미사용 선택", flush=True)

    TASKS = [("OD", "<OD>"), ("DENSE", "<DENSE_REGION_CAPTION>")]
    sel = {name: [] for name, _ in TASKS}      # 프레임별 선택 결과(dict or None)
    gts = []
    matched_labels = defaultdict(int)

    cur, hf = None, None
    for n, r in enumerate(rows):
        if r["ep"] != cur:
            if hf is not None:
                hf.close()
            hf = h5py.File(r["ep"], "r"); cur = r["ep"]
        im = np.ascontiguousarray(np.array(hf["images"][r["fi"]])[:, :, ::-1])
        pil = Image.fromarray(im.astype(np.uint8)).convert("RGB")
        gts.append(r["gt_cx"])

        for name, task in TASKS:
            dets = gen(model, proc, pil, task)
            p = pick_by_keyword(dets)
            sel[name].append(p)
            if p:
                matched_labels[p["label"]] += 1

        if (n + 1) % 20 == 0:
            print(f"  {n+1}/{len(rows)}", flush=True)
    if hf is not None:
        hf.close()

    N = len(rows)
    gts = np.array(gts)

    def score(picks):
        chosen = [i for i, p in enumerate(picks) if p is not None]
        hit = [i for i in chosen if abs(picks[i]["cx"] - gts[i]) <= HIT_TOL]
        errs = [abs(picks[i]["cx"] - gts[i]) for i in chosen]
        return {
            "recall": len(hit) / N,
            "precision": len(hit) / len(chosen) if chosen else None,
            "selected_rate": len(chosen) / N,
            "cx_mae_when_selected": float(np.mean(errs)) if errs else None,
        }

    rep = {"n_frames": N, "model": MODEL, "hit_tol_cx": HIT_TOL,
           "keywords": KEYWORDS, "owl_baseline_detect_rate": OWL_BASELINE,
           "per_task": {}, "matched_labels": dict(
               sorted(matched_labels.items(), key=lambda kv: -kv[1])[:15])}

    for name, _ in TASKS:
        rep["per_task"][name] = score(sel[name])

    # 합집합 — OD 우선, 없으면 DENSE
    union = [sel["OD"][i] or sel["DENSE"][i] for i in range(N)]
    rep["union_OD_then_DENSE"] = score(union)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2, ensure_ascii=False))

    print("\n" + "=" * 76)
    print(f"{'방식':22s} {'선택률':>8s} {'재현율':>8s} {'정밀도':>8s} {'cx MAE':>9s}")
    print("-" * 76)
    for k in list(rep["per_task"]) + ["union_OD_then_DENSE"]:
        d = rep["per_task"].get(k) or rep["union_OD_then_DENSE"]
        pr = f"{d['precision']:7.1%}" if d["precision"] is not None else "      —"
        mae = f"{d['cx_mae_when_selected']:9.4f}" if d["cx_mae_when_selected"] is not None else "        —"
        print(f"{k:22s} {d['selected_rate']:7.1%} {d['recall']:7.1%} {pr} {mae}")
    print(f"\n{'OWL-v2 실기 기준':22s} {'—':>8s} {OWL_BASELINE:7.1%}")
    print(f"\n키워드가 잡은 라벨 상위: {list(rep['matched_labels'])[:6]}")
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()

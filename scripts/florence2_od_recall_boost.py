#!/usr/bin/env python3
"""Florence-2 <OD> 재현율 개선 실험 (2026-08-17).

앞선 결과(florence2_od_label_probe.py, n=80):
  <OD>의 "waste container" 라벨이 우리 타겟을 **정확히** 추적한다
  — cx MAE 0.002, 상관 +1.00, cx 범위 0.291~0.736.
  그러나 **등장률 37.5%**로 OWL-v2(90.5%)에 크게 못 미친다.
  즉 남은 문제는 정확도가 아니라 재현율이다.

이 스크립트가 확인하는 것:
  A. 상호보완성 — "waste container"가 없는 프레임에서 다른 라벨(furniture 등)이
     타겟을 잡고 있는가? 잡고 있다면 라벨 합집합으로 재현율을 올릴 수 있다.
  B. 태스크 교체 — <DENSE_REGION_CAPTION>은 더 많은 영역을 캡션과 함께 반환하므로
     재현율이 높을 수 있다. 같은 프레임에서 <OD>와 직접 비교한다.
  C. beam 수 영향 — num_beams를 올리면 검출이 늘어나는가.

판정: "타겟 적중"은 예측 cx가 OWL 정답 cx와 |Δcx| <= 0.05 인 경우로 정의한다
      (0.05는 화면폭의 5% — 방향 판단에 영향을 주지 않는 수준).

주의: 파이프라인에 끼워넣지 않는다. 출력 품질만 독립 측정.

출력: docs/v5/detector/florence2_od_recall_boost.json
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
OUT = ROOT / "docs/v5/detector/florence2_od_recall_boost.json"
SPLIT_SEED, VAL_RATIO = 42, 0.15
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = "microsoft/Florence-2-base"
HIT_TOL = 0.05          # |Δcx| 허용치
PRIMARY = "waste container"


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


def gen(model, proc, pil, task, beams=3, max_new=256):
    W, H = pil.width, pil.height
    inp = proc(text=task, images=pil, return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(
            input_ids=inp["input_ids"].to(DEV),
            pixel_values=inp["pixel_values"].to(DEV, torch.float16),
            max_new_tokens=max_new, num_beams=beams, do_sample=False)
    txt = proc.batch_decode(ids, skip_special_tokens=False)[0]
    parsed = proc.post_process_generation(txt, task=task, image_size=(W, H))[task]
    boxes = parsed.get("bboxes", []) or []
    labels = (parsed.get("labels") or parsed.get("bboxes_labels") or [])
    out = []
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b
        lb = str(labels[i]).lower().strip() if i < len(labels) else ""
        out.append((lb, (x1 + x2) / 2 / W, (x2 - x1) * (y2 - y1) / (W * H)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoProcessor
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.float16).to(DEV).eval()
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

    rows = val_frames(args.n)
    print(f"표본 {len(rows)} 프레임 · 적중 기준 |Δcx| <= {HIT_TOL}", flush=True)

    CONFIGS = [
        ("OD_b3",   "<OD>",                     3),
        ("OD_b5",   "<OD>",                     5),
        ("DENSE_b3", "<DENSE_REGION_CAPTION>",  3),
    ]

    # cfg -> 프레임별 결과
    hit_primary = defaultdict(list)   # PRIMARY 라벨이 적중했나
    hit_anylabel = defaultdict(list)  # 아무 라벨이라도 적중했나
    best_label_when_primary_missing = defaultdict(lambda: defaultdict(int))
    n_boxes = defaultdict(list)

    cur, hf = None, None
    for n, r in enumerate(rows):
        if r["ep"] != cur:
            if hf is not None:
                hf.close()
            hf = h5py.File(r["ep"], "r"); cur = r["ep"]
        im = np.ascontiguousarray(np.array(hf["images"][r["fi"]])[:, :, ::-1])
        pil = Image.fromarray(im.astype(np.uint8)).convert("RGB")
        gt = r["gt_cx"]

        for name, task, beams in CONFIGS:
            dets = gen(model, proc, pil, task, beams=beams)
            n_boxes[name].append(len(dets))

            prim = [d for d in dets if PRIMARY in d[0]]
            p_hit = any(abs(d[1] - gt) <= HIT_TOL for d in prim)
            hit_primary[name].append(p_hit)

            hits = [d for d in dets if abs(d[1] - gt) <= HIT_TOL]
            hit_anylabel[name].append(bool(hits))

            if not p_hit and hits:
                # PRIMARY가 놓친 프레임에서 무엇이 타겟을 잡았나
                best = min(hits, key=lambda d: abs(d[1] - gt))
                best_label_when_primary_missing[name][best[0] or "(무라벨)"] += 1

        if (n + 1) % 20 == 0:
            print(f"  {n+1}/{len(rows)}", flush=True)
    if hf is not None:
        hf.close()

    N = len(rows)
    rep = {"n_frames": N, "model": MODEL, "hit_tol_cx": HIT_TOL,
           "primary_label": PRIMARY, "configs": {}}
    for name, task, beams in CONFIGS:
        rep["configs"][name] = {
            "task": task, "num_beams": beams,
            "recall_primary_label": float(np.mean(hit_primary[name])),
            "recall_any_label": float(np.mean(hit_anylabel[name])),
            "mean_boxes_per_frame": float(np.mean(n_boxes[name])),
            "rescue_labels": dict(sorted(
                best_label_when_primary_missing[name].items(),
                key=lambda kv: -kv[1])[:10]),
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2, ensure_ascii=False))

    print("\n" + "=" * 74)
    print(f"{'설정':12s} {'박스/프레임':>11s} {'재현율(waste)':>14s} {'재현율(라벨무관)':>17s}")
    print("-" * 74)
    for name in rep["configs"]:
        c = rep["configs"][name]
        print(f"{name:12s} {c['mean_boxes_per_frame']:11.1f} "
              f"{c['recall_primary_label']:13.1%} {c['recall_any_label']:16.1%}")
    for name in rep["configs"]:
        rl = rep["configs"][name]["rescue_labels"]
        if rl:
            print(f"\n[{name}] waste container가 놓쳤을 때 타겟을 잡은 라벨: {rl}")
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Florence-2 <OD> 라벨 탐침 — 어떤 라벨이 우리 타겟을 추적하는가 (2026-08-17).

발견(florence2_task_sweep.py, n=6):
  <CAPTION_TO_PHRASE_GROUNDING>과 <OPEN_VOCABULARY_DETECTION>은 둘 다 판별력 0.000 —
  없는 물체("person")도 항상 박스를 뱉는다. 거부 모드가 없다.

  그런데 <OD>는 다르다. 프롬프트 없이 "찾은 것만" 라벨과 함께 반환하므로 거부가 된다.
  그리고 실제로 바구니를 보고 있었다 — 다만 "basket"이 아니라
  **"waste container"** 라고 부른다(회색 빨래바구니 → 쓰레기통으로 인식, 외형상 타당).

이 스크립트가 하는 일:
  <OD>가 뱉는 **모든 라벨**을 수집하고, 각 라벨의 박스 cx를 OWL-v2 정답 cx와 대조해
  **어떤 라벨이 실제로 우리 타겟을 추적하는지** 찾는다. 라벨 이름을 미리 가정하지 않는다.

판정:
  · 라벨별 등장률(coverage) — 프레임 중 몇 %에서 나오는가
  · 라벨별 cx MAE vs OWL 정답 — 낮을수록 우리 타겟을 정확히 추적
  · cx 범위 — CH59 우편향(cx max 0.559)이 <OD>에서도 재현되는가

주의: 파이프라인에 끼워넣지 않는다. 출력 품질만 독립 측정.

출력: docs/v5/detector/florence2_od_label_probe.json
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
OUT = ROOT / "docs/v5/detector/florence2_od_label_probe.json"
SPLIT_SEED, VAL_RATIO = 42, 0.15
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = "microsoft/Florence-2-base"
TASK = "<OD>"


def val_frames(n, seed=0):
    """정답 cx(OWL 주석)를 함께 들고 온다."""
    ann = json.loads(ANN.read_text())
    rng = np.random.default_rng(SPLIT_SEED)
    idx = list(range(len(ann)))
    rng.shuffle(idx)
    val_eps = set(idx[:max(1, int(len(idx) * VAL_RATIO))])
    rows = [dict(ep=ep["episode"], fi=f["frame_idx"],
                 gt_cx=f.get("cx_det"), gt_area=f.get("area_det"))
            for ei, ep in enumerate(ann) if ei in val_eps
            for f in ep["frames"]
            if not f["grounding_cached"] and f["detected"] and f.get("cx_det") is not None]
    r2 = np.random.default_rng(seed)
    pick = r2.choice(len(rows), size=min(n, len(rows)), replace=False)
    return [rows[i] for i in sorted(pick)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoProcessor
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.float16).to(DEV).eval()
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

    rows = val_frames(args.n)
    print(f"표본 {len(rows)} 프레임 · {TASK} · 라벨 무가정 수집", flush=True)

    # label -> list of (cx, area, gt_cx)
    per_label = defaultdict(list)
    n_frames_with_any = 0

    cur, hf = None, None
    for n, r in enumerate(rows):
        if r["ep"] != cur:
            if hf is not None:
                hf.close()
            hf = h5py.File(r["ep"], "r"); cur = r["ep"]
        im = np.ascontiguousarray(np.array(hf["images"][r["fi"]])[:, :, ::-1])
        pil = Image.fromarray(im.astype(np.uint8)).convert("RGB")
        W, H = pil.width, pil.height

        inp = proc(text=TASK, images=pil, return_tensors="pt")
        with torch.no_grad():
            ids = model.generate(
                input_ids=inp["input_ids"].to(DEV),
                pixel_values=inp["pixel_values"].to(DEV, torch.float16),
                max_new_tokens=256, num_beams=3, do_sample=False)
        txt = proc.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = proc.post_process_generation(txt, task=TASK, image_size=(W, H))[TASK]
        boxes = parsed.get("bboxes", []) or []
        labels = parsed.get("labels", []) or []
        if boxes:
            n_frames_with_any += 1

        seen_this_frame = set()
        for b, lb in zip(boxes, labels):
            lb = str(lb).lower().strip()
            if lb in seen_this_frame:      # 프레임당 라벨 1회만(최대 박스 채택 효과)
                continue
            seen_this_frame.add(lb)
            x1, y1, x2, y2 = b
            per_label[lb].append((
                (x1 + x2) / 2 / W,
                (x2 - x1) * (y2 - y1) / (W * H),
                r["gt_cx"],
            ))

        if (n + 1) % 20 == 0:
            print(f"  {n+1}/{len(rows)}", flush=True)
    if hf is not None:
        hf.close()

    N = len(rows)
    stats = {}
    for lb, vals in per_label.items():
        cx = np.array([v[0] for v in vals])
        ar = np.array([v[1] for v in vals])
        gt = np.array([v[2] for v in vals])
        stats[lb] = {
            "coverage": len(vals) / N,
            "n": len(vals),
            "cx_mae_vs_owl": float(np.mean(np.abs(cx - gt))),
            "cx_corr": float(np.corrcoef(cx, gt)[0, 1]) if len(vals) > 2 and cx.std() > 1e-9 else None,
            "cx_mean": float(cx.mean()), "cx_std": float(cx.std()),
            "cx_min": float(cx.min()), "cx_max": float(cx.max()),
            "area_mean": float(ar.mean()),
        }

    rep = {
        "n_frames": N, "model": MODEL, "task": TASK,
        "frames_with_any_box": n_frames_with_any,
        "labels": dict(sorted(stats.items(), key=lambda kv: -kv[1]["coverage"])),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2, ensure_ascii=False))

    print("\n" + "=" * 78)
    print(f"박스가 하나라도 나온 프레임: {n_frames_with_any}/{N}")
    print(f"\n{'라벨':26s} {'등장률':>7s} {'cx MAE':>8s} {'상관':>6s} {'cx 범위':>16s}")
    print("-" * 78)
    for lb, d in list(rep["labels"].items())[:14]:
        corr = f"{d['cx_corr']:+.2f}" if d["cx_corr"] is not None else "  — "
        print(f"{lb:26s} {d['coverage']:6.1%} {d['cx_mae_vs_owl']:8.3f} {corr:>6s} "
              f"  {d['cx_min']:.3f}~{d['cx_max']:.3f}")
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()

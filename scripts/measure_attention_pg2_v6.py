#!/usr/bin/env python3
"""
PG2 방향(left/right/forward) text attention 구분력 — V6 트랙A/F(극단cx+오버슈트류) 데이터로 재현 확인.

배경: measure_attention_pg2.py는 고정 이미지 1장(s6/frame_0001.jpg)으로 spread=1.4%p를
측정했다. 이번엔 V6의 15개 path_type(weak/strong left/right + center × left/straight/right
curve) 각각의 대표 프레임으로 동일 측정을 반복해, 그 결과가 프레임 하나의 우연이 아니라
V6 전반에서 재현되는지 확인한다. 전체 재학습 없이 zero-shot PG2로만 측정.

Usage:
  .venv/bin/python3 scripts/measure_attention_pg2_v6.py
"""
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration

ROOT = Path(__file__).resolve().parent.parent
PG2 = Path.home() / ".cache/huggingface/hub/models--google--paligemma2-3b-mix-224/snapshots/8e40ab4cc5df93dfb7fd2fff754bcdff8b62ee78"
ANN_V6 = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_v6_pg448_cx.json"
OUT_DIR = ROOT / "docs/v5/attention_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
NUM_IMAGE_TOKENS = 256

INSTRUCTIONS = {
    "left":    "Navigate to the left toward the gray basket",
    "right":   "Navigate to the right toward the gray basket",
    "forward": "Navigate straight forward to the gray basket",
}


def analyze_layer(attn, num_image=NUM_IMAGE_TOKENS):
    layers = []
    for li, a in enumerate(attn):
        last_row = a[0, :, -1, :]
        seq_len = last_row.shape[-1]
        img_end = min(num_image, seq_len)
        img_part = last_row[:, :img_end].sum(dim=-1)
        text_part = last_row[:, img_end:].sum(dim=-1)
        total = last_row.sum(dim=-1)
        layers.append({
            "layer": li,
            "text_ratio_mean": float((text_part / (total + 1e-9)).mean()),
        })
    return layers


@torch.no_grad()
def measure(model, proc, img, prompt):
    inp = proc(text=prompt, images=img, return_tensors="pt").to(model.device)
    inp["pixel_values"] = inp["pixel_values"].to(model.dtype)
    out = model(**inp, output_attentions=True)
    return analyze_layer(out.attentions)


def pick_representative_frames():
    """path_type별 대표 프레임(중간 지점) — episode의 h5에서 이미지 로드."""
    with open(ANN_V6) as f:
        ann = json.load(f)
    reps = {}
    for ep in ann:
        pt = ep["path_type"]
        if pt in reps:
            continue
        h5_path = Path(ep["episode"])
        if not h5_path.exists():
            continue
        with h5py.File(str(h5_path), "r") as h:
            imgs = (h["observations"]["images"] if "observations" in h else h["images"])
            mid = len(imgs) // 2
            img_np = imgs[mid][:, :, ::-1].astype("uint8")  # BGR->RGB
        reps[pt] = Image.fromarray(img_np)
    return reps


def main():
    print(f"[로드] {PG2}")
    proc = PaliGemmaProcessor.from_pretrained(str(PG2))
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        str(PG2), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map={"": "cuda"},
    ).eval()

    reps = pick_representative_frames()
    print(f"[프레임] path_type {len(reps)}개 대표 프레임 로드 완료")

    results = {}
    spreads = []
    for pt, img in reps.items():
        per_instr = {}
        for key, prompt in INSTRUCTIONS.items():
            layers = measure(model, proc, img, prompt)
            mean_text = float(np.mean([l["text_ratio_mean"] for l in layers]))
            per_instr[key] = mean_text
        spread = max(per_instr.values()) - min(per_instr.values())
        spreads.append(spread)
        results[pt] = {"mean_text_ratio": per_instr, "spread": spread}
        print(f"  {pt:24s} L/R/F mean_text_ratio={per_instr}  spread={spread:.5f}")

    spreads = np.array(spreads)
    summary = {
        "per_path_type": results,
        "spread_mean": float(spreads.mean()),
        "spread_std": float(spreads.std()),
        "spread_min": float(spreads.min()),
        "spread_max": float(spreads.max()),
        "n_path_types": len(reps),
        "reference_single_frame_spread": 0.014,
    }
    print("\n=== 요약 ===")
    print(f"  path_type {len(reps)}개 평균 spread = {summary['spread_mean']:.5f} "
          f"± {summary['spread_std']:.5f}  (범위 {summary['spread_min']:.5f}~{summary['spread_max']:.5f})")
    print(f"  기존 단일 프레임 측정값(참고): 0.01400")

    out = OUT_DIR / "pg2_v6_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[저장] {out}")


if __name__ == "__main__":
    main()

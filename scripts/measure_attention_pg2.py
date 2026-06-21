#!/usr/bin/env python3
"""
PG2(PaliGemma2) frozen 상태에서 instruction별 text attention 측정 — Exp15와 동일한
방법론을 PG2에 적용한 저비용 사전 검증.

배경: Google-robot(Kosmos-2 post-train) 백본은 Exp15(VLM 완전 frozen + head-only)에서
text attention 0.0000%로 측정됨 — 백본 자체가 구조적으로 text를 무시한다는 증거였다
(scripts/measure_attention.py). PG2를 새 action backbone으로 검토하기 전에, 같은
측정을 PG2에 대해 먼저 해서 "PG2는 애초에 text에 민감한 표현을 갖고 있는가"를
학습/투자 없이 확인한다.

이미 알려진 사실(Exp57): PG2 zero-shot "detect gray basket"→100% vs "detect red ball"→0%
— 출력 레벨에서는 이미 text-conditioned임이 증명됨. 이 스크립트는 그 행동의 근거가
실제로 self-attention이 text 토큰에 가는 데서 오는지(=Google-robot과 다름을 수치로
확인), 그리고 동일 객체에 대해 방향성 instruction(left/right/forward)을 바꿔도
attention이 달라지는지(=향후 action backbone으로 썼을 때 instruction을 구분할 토대가
있는지)를 본다.

방법: 동일 이미지 + 다른 프롬프트로 forward(output_attentions=True, teacher-forced,
generate 호출 안 함 — Google-robot에서처럼 generate가 무한반복하는 위험과 무관).
마지막 시퀀스 위치(다음 토큰 생성 직전)의 attention row에서 image(0:256) vs
text(256:end) 영역 비율을 layer별로 집계.

산출: docs/v5/attention_analysis/pg2_summary.json, pg2_index.html

Usage:
  .venv/bin/python3 scripts/measure_attention_pg2.py
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration

ROOT = Path(__file__).resolve().parent.parent
PG2 = Path.home() / ".cache/huggingface/hub/models--google--paligemma2-3b-mix-224/snapshots/8e40ab4cc5df93dfb7fd2fff754bcdff8b62ee78"
IMG_PATH = ROOT / "docs/v5/grounding_frames/s6/frame_0001.jpg"
OUT_DIR = ROOT / "docs/v5/attention_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
NUM_IMAGE_TOKENS = 256  # PG2-224: 16x16 패치

# (1) 양성 대조군 — Exp57에서 출력 레벨로 이미 증명된 대비(있음 vs 없음 객체)
#     이게 text attention 차이로 안 보이면 측정 방법 자체가 문제.
CONTRAST_PROMPTS = {
    "basket": "detect gray basket",
    "ball": "detect red ball",
}

# (2) 본 실험 — measure_attention.py(Kosmos)와 동일 프레이밍, 동일 객체 + 방향만 변경.
#     PG2를 향후 action backbone으로 쓸 때 "instruction(방향)을 구분할 토대가 있는가".
INSTRUCTIONS = {
    "left":    "Navigate to the left toward the gray basket",
    "right":   "Navigate to the right toward the gray basket",
    "forward": "Navigate straight forward to the gray basket",
}


def analyze_layer(attn, num_image=NUM_IMAGE_TOKENS):
    """attn: tuple of (B, heads, S, S) per layer. 마지막 위치(다음 토큰 생성 직전) 기준."""
    layers = []
    for li, a in enumerate(attn):
        last_row = a[0, :, -1, :]  # (heads, S)
        seq_len = last_row.shape[-1]
        img_end = min(num_image, seq_len)
        img_part = last_row[:, :img_end].sum(dim=-1)
        text_part = last_row[:, img_end:].sum(dim=-1)
        total = last_row.sum(dim=-1)
        layers.append({
            "layer": li,
            "seq_len": int(seq_len),
            "image_ratio_mean": float((img_part / (total + 1e-9)).mean()),
            "text_ratio_mean": float((text_part / (total + 1e-9)).mean()),
        })
    return layers


@torch.no_grad()
def measure(model, proc, img, prompt):
    inp = proc(text=prompt, images=img, return_tensors="pt").to(model.device)
    inp["pixel_values"] = inp["pixel_values"].to(model.dtype)
    out = model(**inp, output_attentions=True)
    layers = analyze_layer(out.attentions)
    text_len = inp["input_ids"].shape[1] - NUM_IMAGE_TOKENS
    return layers, text_len


def main():
    print(f"[로드] {PG2}")
    proc = PaliGemmaProcessor.from_pretrained(str(PG2))
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        str(PG2), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map={"": "cuda"},
    ).eval()
    img = Image.open(IMG_PATH).convert("RGB")

    results = {}

    print("\n=== (1) 양성 대조군 — basket vs ball (출력 레벨로 이미 증명된 대비) ===")
    for key, prompt in CONTRAST_PROMPTS.items():
        layers, text_len = measure(model, proc, img, prompt)
        last = layers[-1]
        mean_text = float(np.mean([l["text_ratio_mean"] for l in layers]))
        mean_img = float(np.mean([l["image_ratio_mean"] for l in layers]))
        print(f"  {key:8s} text_len={text_len:2d}  last_layer[img={last['image_ratio_mean']:.3f} "
              f"text={last['text_ratio_mean']:.3f}]  mean[img={mean_img:.3f} text={mean_text:.3f}]")
        results[f"contrast_{key}"] = {"prompt": prompt, "text_len": text_len, "per_layer": layers,
                                       "mean_text_ratio": mean_text, "mean_image_ratio": mean_img}

    print("\n=== (2) 본 실험 — 동일 객체, 방향(left/right/forward)만 변경 ===")
    for key, prompt in INSTRUCTIONS.items():
        layers, text_len = measure(model, proc, img, prompt)
        last = layers[-1]
        mean_text = float(np.mean([l["text_ratio_mean"] for l in layers]))
        mean_img = float(np.mean([l["image_ratio_mean"] for l in layers]))
        print(f"  {key:8s} text_len={text_len:2d}  last_layer[img={last['image_ratio_mean']:.3f} "
              f"text={last['text_ratio_mean']:.3f}]  mean[img={mean_img:.3f} text={mean_text:.3f}]")
        results[f"instr_{key}"] = {"prompt": prompt, "text_len": text_len, "per_layer": layers,
                                    "mean_text_ratio": mean_text, "mean_image_ratio": mean_img}

    # 방향 간 last-layer text attention pattern이 실제로 다른지(=동일 위치별 분포 변화) 확인
    print("\n=== (3) 방향 간 차이 — 동일 레이어/포지션에서 attention이 실제로 변하는가 ===")
    last_layer_text_ratios = {k: results[f"instr_{k}"]["mean_text_ratio"] for k in INSTRUCTIONS}
    spread = max(last_layer_text_ratios.values()) - min(last_layer_text_ratios.values())
    print(f"  left/right/forward mean_text_ratio = {last_layer_text_ratios}")
    print(f"  spread(max-min) = {spread:.5f}  ({'서로 다름 — 텍스트에 반응' if spread > 0.01 else '거의 동일 — Google-robot처럼 무시 가능성'})")

    out_path = OUT_DIR / "pg2_summary.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n[저장] {out_path}")


if __name__ == "__main__":
    main()

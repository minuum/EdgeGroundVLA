#!/usr/bin/env python3
"""
PG2 hidden state가 instruction에 따라 실제로 구분되는 표현을 만드는가 —
"action head 입력을 bbox 대신 PG2 hidden state로 바꾸면 다를까"를 학습 없이
바로 확인하는 사전 평가 (plan §13/CH38-2의 후속 질문).

measure_attention_pg2.py는 attention 가중치(어디를 보는가)를 측정했다.
이 스크립트는 한 단계 더 나가 실제로 action head에 들어갈 **표현(벡터) 자체**가
instruction에 따라 충분히 달라지는지를 측정한다 — 표현이 안 갈리면 아무리 좋은
head를 붙여도 학습할 신호가 없다(Exp12/13처럼 무시당할 운명).

방법: 동일 이미지(여러 장) × 여러 prompt에 대해 PG2의 마지막 레이어 hidden state를
(마지막 시퀀스 위치, 다음 토큰 생성 직전 — attention 실험과 동일 위치) 추출,
프롬프트 쌍 간 코사인거리를 비교:
  - basket vs ball(객체 자체가 다름) — 강한 신호 기대
  - left vs right vs forward(같은 객체, 방향만 다름) — 약한 신호 기대(attention spread 1.4%p와 일관되게)
  - 동일 prompt 반복(노이즈 바닥선, deterministic forward라 0 근접 기대)

산출: docs/v5/attention_analysis/pg2_hidden_state_distances.json

Usage:
  .venv/bin/python3 scripts/measure_hidden_state_pg2.py
"""
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration

ROOT = Path(__file__).resolve().parent.parent
PG2 = Path.home() / ".cache/huggingface/hub/models--google--paligemma2-3b-mix-224/snapshots/8e40ab4cc5df93dfb7fd2fff754bcdff8b62ee78"
OUT_DIR = ROOT / "docs/v5/attention_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGES = [
    ROOT / "docs/v5/grounding_frames/s6/frame_0001.jpg",
    ROOT / "docs/v5/grounding_frames/s7/frame_0084.jpg",
]

PROMPTS = {
    "basket":  "detect gray basket",
    "ball":    "detect red ball",
    "left":    "Navigate to the left toward the gray basket",
    "right":   "Navigate to the right toward the gray basket",
    "forward": "Navigate straight forward to the gray basket",
}


@torch.no_grad()
def get_last_hidden(model, proc, img, prompt):
    inp = proc(text=prompt, images=img, return_tensors="pt").to(model.device)
    inp["pixel_values"] = inp["pixel_values"].to(model.dtype)
    out = model(**inp, output_hidden_states=True)
    h = out.hidden_states[-1][0, -1, :]  # 마지막 레이어, 마지막 위치 (다음 토큰 생성 직전)
    return h.float().cpu().numpy()


def cos_dist(a, b):
    return float(1.0 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main():
    print(f"[로드] {PG2}")
    proc = PaliGemmaProcessor.from_pretrained(str(PG2))
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        str(PG2), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map={"": "cuda"},
    ).eval()

    results = {}
    for img_path in IMAGES:
        img = Image.open(img_path).convert("RGB")
        tag = img_path.parent.name + "/" + img_path.name
        print(f"\n=== 이미지: {tag} ===")

        vecs = {k: get_last_hidden(model, proc, img, p) for k, p in PROMPTS.items()}
        # 동일 prompt 반복 — 노이즈 바닥선 (deterministic forward라 0 근접 기대)
        vecs["basket_repeat"] = get_last_hidden(model, proc, img, PROMPTS["basket"])

        pairs = {
            "basket_vs_ball (객체 다름)":      cos_dist(vecs["basket"], vecs["ball"]),
            "left_vs_right (방향만 다름)":     cos_dist(vecs["left"], vecs["right"]),
            "left_vs_forward (방향만 다름)":   cos_dist(vecs["left"], vecs["forward"]),
            "right_vs_forward (방향만 다름)":  cos_dist(vecs["right"], vecs["forward"]),
            "basket_vs_basket_repeat (노이즈 바닥선)": cos_dist(vecs["basket"], vecs["basket_repeat"]),
        }
        for k, v in pairs.items():
            print(f"  {k:38s} cos_dist={v:.5f}")

        dir_avg = np.mean([pairs["left_vs_right (방향만 다름)"],
                           pairs["left_vs_forward (방향만 다름)"],
                           pairs["right_vs_forward (방향만 다름)"]])
        ratio = pairs["basket_vs_ball (객체 다름)"] / max(dir_avg, 1e-9)
        print(f"  → 객체차이/방향차이 비율 = {ratio:.1f}x "
              f"(노이즈 바닥선={pairs['basket_vs_basket_repeat (노이즈 바닥선)']:.5f})")

        results[tag] = {**pairs, "object_vs_direction_ratio": float(ratio)}

    out_path = OUT_DIR / "pg2_hidden_state_distances.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n[저장] {out_path}")


if __name__ == "__main__":
    main()

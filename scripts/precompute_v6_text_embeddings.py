#!/usr/bin/env python3
"""EGVLA2 text-subgoal plan: V6 5-direction 영어 템플릿을 frozen CLIP text encoder로 임베딩해 캐시.

로컬 HF 캐시에 있는 openai/clip-vit-large-patch14 text tower(768-d)를 사용한다(plan의 B/32 512-d에서 변경).
CLIP은 영어 전용이라 한국어 템플릿은 쓰지 않는다.
출력: docs/v5/bbox_nav_owl/v6_text_embed_cache.npz  키 "{direction}" -> (n_templates, 768) float32 (L2 정규화 pooled)
실행: HF_HUB_OFFLINE=1 .venv/bin/python3 scripts/precompute_v6_text_embeddings.py
"""
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPModel, CLIPTokenizer

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs/v5/bbox_nav_owl/v6_text_embed_cache.npz"
MODEL = "openai/clip-vit-large-patch14"

V6_DIRECTION_INSTRUCTIONS = {
    "center": ["Go straight toward the basket", "Drive forward to the basket ahead", "Head directly at the basket"],
    "weak_left": ["Steer slightly left toward the basket", "Veer a little to the left toward the basket",
                  "Bear gently left to reach the basket"],
    "strong_left": ["Turn sharply left toward the basket", "Make a hard left turn toward the basket",
                    "Swerve far to the left to reach the basket"],
    "weak_right": ["Steer slightly right toward the basket", "Veer a little to the right toward the basket",
                   "Bear gently right to reach the basket"],
    "strong_right": ["Turn sharply right toward the basket", "Make a hard right turn toward the basket",
                     "Swerve far to the right to reach the basket"],
}


@torch.no_grad()
def main():
    tok = CLIPTokenizer.from_pretrained(MODEL)
    model = CLIPModel.from_pretrained(MODEL).eval()
    out = {}
    for d, texts in V6_DIRECTION_INSTRUCTIONS.items():
        t = tok(texts, padding=True, return_tensors="pt")
        e = model.get_text_features(**t)
        out[d] = torch.nn.functional.normalize(e, dim=-1).numpy().astype(np.float32)
        print(d, out[d].shape)
    np.savez(OUT, **out)
    # 방향 간 코사인 유사도(템플릿 평균) — 임베딩이 방향을 구분하는지 사전 확인
    keys = list(out)
    m = np.stack([out[k].mean(0) for k in keys])
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    print(keys)
    print(np.round(m @ m.T, 3))


if __name__ == "__main__":
    main()

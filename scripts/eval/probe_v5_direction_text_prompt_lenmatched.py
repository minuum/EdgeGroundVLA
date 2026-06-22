#!/usr/bin/env python3
"""
probe_v5_direction_text_prompt.py의 토큰-길이 통제판. 원래 실험은 P0="detect
gray basket"(3단어) vs P1="...on the left"(6단어)처럼 prompt 길이가 달라서,
"prompt 종류 99.2% 구분"이 방향 이해가 아니라 단순 길이/토큰수 차이일 수 있다는
지적(사용자) — gray를 left/right로 치환해 **단어 수를 동일하게** 맞춤:
  P0 = "detect gray basket"  (3단어)
  P1 = "detect left basket"  (3단어, gray->left)
  P2 = "detect right basket" (3단어, gray->right)

새 데이터 수집 없음(기존 220 에피소드), 새 학습 없음(frozen probe만).

산출: docs/v5/attention_analysis/v5_direction_text_prompt_probe_lenmatched.json

Usage:
  .venv/bin/python3 scripts/eval/probe_v5_direction_text_prompt_lenmatched.py
"""
import json
import re
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

ROOT = Path(__file__).resolve().parent.parent.parent
PG2 = Path.home() / ".cache/huggingface/hub/models--google--paligemma2-3b-mix-224/snapshots/8e40ab4cc5df93dfb7fd2fff754bcdff8b62ee78"
V5_DIR = ROOT / "ROS_action/mobile_vla_dataset_v5"
OUT_DIR = ROOT / "docs/v5/attention_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FNAME_RE = re.compile(r"target_(center|left|right)_(straight|left|right)_path")

PROMPTS = {
    "P0_basket": "detect gray basket",
    "P1_left": "detect left basket",
    "P2_right": "detect right basket",
}


def collect_labeled_episodes():
    items = []
    for f in sorted(V5_DIR.glob("*.h5")):
        m = FNAME_RE.search(f.name)
        if m:
            items.append((f, m.group(1), m.group(2)))
    return items


def mid_frame_image(h5_path):
    with h5py.File(h5_path, "r") as f:
        imgs = f["observations"]["images"] if "observations" in f else f["images"]
        mid = len(imgs) // 2
        arr = imgs[mid]
    return Image.fromarray(arr.astype(np.uint8)).convert("RGB")


@torch.no_grad()
def get_hidden_batch(model, proc, imgs, prompt):
    inp = proc(text=[prompt] * len(imgs), images=imgs, return_tensors="pt", padding=True).to(model.device)
    inp["pixel_values"] = inp["pixel_values"].to(model.dtype)
    out = model(**inp, output_hidden_states=True)
    return out.hidden_states[-1][:, -1, :].float().cpu().numpy()


def run_probe(X, y, name):
    y = np.array(y)
    n_class = len(set(y))
    chance = 1.0 / n_class
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    clf = LogisticRegression(max_iter=2000)
    scores = cross_val_score(clf, X, y, cv=cv)
    acc = float(scores.mean())
    print(f"=== {name} (n_class={n_class}, chance={chance:.3f}) ===")
    print(f"  5-fold CV acc = {acc:.3f} ± {scores.std():.3f}  (ratio={acc/chance:.2f}x)")
    return {"n_class": n_class, "chance": chance, "cv_acc_mean": acc,
            "cv_acc_std": float(scores.std()), "ratio_vs_chance": acc / chance}


def main():
    items = collect_labeled_episodes()
    print(f"[데이터] {len(items)}개 에피소드")

    print(f"\n[로드] {PG2}")
    proc = PaliGemmaProcessor.from_pretrained(str(PG2))
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        str(PG2), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map={"": "cuda"},
    ).eval()

    imgs = [mid_frame_image(path) for path, _, _ in items]
    directions = [d for _, _, d in items]

    hidden_by_prompt = {}
    for tag, prompt in PROMPTS.items():
        feats = []
        for i in range(0, len(imgs), 16):
            batch = imgs[i:i+16]
            h = get_hidden_batch(model, proc, batch, prompt)
            feats.append(h)
        hidden_by_prompt[tag] = np.concatenate(feats, axis=0)
        print(f"  [{tag}] 추출 완료 — shape={hidden_by_prompt[tag].shape}")

    results = {"n_episodes": len(items)}

    # 1. 이미지 방향 vs P0 (CH39 재확인)
    results["image_direction_via_P0"] = run_probe(hidden_by_prompt["P0_basket"], directions, "이미지방향 vs P0(기존 prompt)")

    # 2. 이미지 방향 vs P1/P2 (prompt가 틀려도 이미지 신호가 읽히는가)
    results["image_direction_via_P1_left"] = run_probe(hidden_by_prompt["P1_left"], directions, "이미지방향 vs P1(on the left로 고정)")
    results["image_direction_via_P2_right"] = run_probe(hidden_by_prompt["P2_right"], directions, "이미지방향 vs P2(on the right로 고정)")

    # 3. prompt 종류 자체 vs hidden state (같은 이미지, prompt만 다름)
    all_feats = np.concatenate([hidden_by_prompt[t] for t in PROMPTS], axis=0)
    all_prompt_labels = []
    for t in PROMPTS:
        all_prompt_labels.extend([t] * len(items))
    results["prompt_identity_same_image"] = run_probe(all_feats, all_prompt_labels, "prompt 종류(P0/P1/P2) vs hidden state(같은 이미지셋)")

    out_path = OUT_DIR / "v5_direction_text_prompt_probe_lenmatched.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n[저장] {out_path}")


if __name__ == "__main__":
    main()

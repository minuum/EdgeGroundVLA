#!/usr/bin/env python3
"""
Step B (plan_20260622_fundamental_vla.md §3) — "근본적인 VLA"로 가기 전 저비용 사전검증.

CH38-5는 PG2에 "go left/right" 텍스트를 직접 줬을 때 생성 출력이 거의 동일함을
보였다(명령 프롬프트 얘기). 이 스크립트는 다른 질문을 본다: 기존 V5 H5 에피소드들이
"자연 주행 중 보는 이미지"(바스켓이 좌/중/우 어디 있는지에 따라 달라지는 실제 화면)에
대해, PG2의 grounding-prompt hidden state가 경로방향(left/straight/right) 라벨로
선형분리 가능한가 — 즉 "방향 신호가 hidden state에 원래 있는데 출력 단계에서만
안 쓰이는 것인지" vs "hidden state 자체에 신호가 없는 것인지"를 구분한다.

새 데이터 수집 없음, 새 학습 없음(frozen probe만, Exp54 방법론과 동일).
파일명 패턴 `target_{start}_{direction}_path`(start, direction ∈ {center,left,right}×
{straight,left,right})로 라벨을 얻는다 — 9-class(start×direction)와 3-class(direction만)
둘 다 측정.

산출: docs/v5/attention_analysis/v5_direction_probe.json

Usage:
  .venv/bin/python3 scripts/eval/probe_v5_direction_hidden_state.py
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
GROUND_PROMPT = "detect gray basket"


def collect_labeled_episodes():
    """파일명에서 (start, direction) 라벨을 뽑아 (path, start, direction) 리스트로."""
    items = []
    for f in sorted(V5_DIR.glob("*.h5")):
        m = FNAME_RE.search(f.name)
        if m:
            items.append((f, m.group(1), m.group(2)))
    return items


@torch.no_grad()
def get_last_hidden(model, proc, img, prompt=GROUND_PROMPT):
    inp = proc(text=prompt, images=img, return_tensors="pt").to(model.device)
    inp["pixel_values"] = inp["pixel_values"].to(model.dtype)
    out = model(**inp, output_hidden_states=True)
    h = out.hidden_states[-1][0, -1, :]
    return h.float().cpu().numpy()


def mid_frame_image(h5_path):
    with h5py.File(h5_path, "r") as f:
        imgs = f["observations"]["images"] if "observations" in f else f["images"]
        mid = len(imgs) // 2
        arr = imgs[mid]
    return Image.fromarray(arr.astype(np.uint8)).convert("RGB")


def main():
    items = collect_labeled_episodes()
    print(f"[데이터] target_{{start}}_{{direction}}_path 패턴 매칭 에피소드: {len(items)}개")
    by_key = {}
    for _, s, d in items:
        by_key[(s, d)] = by_key.get((s, d), 0) + 1
    for k, v in sorted(by_key.items()):
        print(f"  {k}: {v}")

    print(f"\n[로드] {PG2}")
    proc = PaliGemmaProcessor.from_pretrained(str(PG2))
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        str(PG2), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        device_map={"": "cuda"},
    ).eval()

    feats, starts, directions = [], [], []
    for i, (path, start, direction) in enumerate(items):
        img = mid_frame_image(path)
        h = get_last_hidden(model, proc, img)
        feats.append(h)
        starts.append(start)
        directions.append(direction)
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(items)}] hidden state 추출 중...")

    X = np.stack(feats)  # (N, D)
    print(f"\n[특징] X shape={X.shape}")

    results = {"n_episodes": len(items), "by_start_direction": {f"{k[0]}_{k[1]}": v for k, v in by_key.items()}}

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    for name, y in [("direction_3class", directions), ("start_3class", starts),
                    ("start_x_direction_9class", [f"{s}_{d}" for s, d in zip(starts, directions)])]:
        y = np.array(y)
        n_class = len(set(y))
        chance = 1.0 / n_class
        clf = LogisticRegression(max_iter=2000, multi_class="auto")
        scores = cross_val_score(clf, X, y, cv=cv)
        acc = float(scores.mean())
        print(f"\n=== probe: {name} (n_class={n_class}, chance={chance:.3f}) ===")
        print(f"  5-fold CV acc = {acc:.3f} ± {scores.std():.3f}  (chance={chance:.3f}, ratio={acc/chance:.2f}x)")
        results[name] = {
            "n_class": n_class, "chance": chance,
            "cv_acc_mean": acc, "cv_acc_std": float(scores.std()),
            "ratio_vs_chance": acc / chance,
        }

    out_path = OUT_DIR / "v5_direction_probe.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n[저장] {out_path}")


if __name__ == "__main__":
    main()

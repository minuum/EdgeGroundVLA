#!/usr/bin/env python3
"""배포 헤드(신버전, stage1v3)의 val 혼동행렬 — 올바른 채점 방식(stride=1) 버전.

eval_confusion_matrix_stage1v3.py는 구버전과 apples-to-apples 비교를 위해
cadence-aligned(stride=5, 다수결 라벨) 채점을 그대로 썼으나, 실제 배포 헤드는
train_exp73_stage1v3_heads.py의 build_windows()(stride=1, 프레임 단위)로
학습되었다. 이 스크립트는 학습 때와 동일한 build_windows()로 val을 채점한다.

출력: docs/v5/detector/confusion_matrix_stage1v3_correct.json,
      docs/v5/figures/confusion_matrix_stage1v3_correct.png
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.train_exp73_stage1v3_heads import (
    MLPActionHead, CACHE_V6, SPLIT_SEED, VAL_RATIO, NUM_CLASSES, DEVICE,
    build_windows,
)
from scripts.exp73_window_cadence import swap_bboxes, OWL_ANN

NAMES = ["STOP", "FWD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "ROT_L", "ROT_R"]
HEAD_PT = ROOT / "runs/v5_nav/mlp/exp73_stage1v3/exp73_owl_stage1v3_v6_mlp.pt"
OUT_JSON = ROOT / "docs/v5/detector/confusion_matrix_stage1v3_correct.json"
OUT_PNG = ROOT / "docs/v5/figures/confusion_matrix_stage1v3_correct.png"


def val_episodes(base):
    rng = np.random.default_rng(SPLIT_SEED)
    idx = list(range(len(base)))
    rng.shuffle(idx)
    nv = max(1, int(len(idx) * VAL_RATIO))
    vs = set(idx[:nv])
    return [e for i, e in enumerate(base) if i in vs]


def load_head(hp):
    head = MLPActionHead().to(DEVICE)
    sd = torch.load(str(hp), map_location=DEVICE, weights_only=False)
    head.load_state_dict(sd["model"])
    head.eval()
    return head, sd.get("val_acc")


def main():
    base = torch.load(str(CACHE_V6), weights_only=False)
    va = swap_bboxes(val_episodes(base), OWL_ANN)
    X, y, _A = build_windows(va)
    print(f"val 결정 시점 {len(X)}개 (stride=1, 학습과 동일 채점), 신버전 헤드(단일 체크포인트, seed0)", flush=True)

    head, saved_val_acc = load_head(HEAD_PT)
    with torch.no_grad():
        pred = head(torch.from_numpy(X).float().to(DEVICE)).argmax(1).cpu().numpy()
    acc = float((pred == y).mean())
    print(f"  {HEAD_PT.name}  val_acc={acc:.4f} (저장된 val_acc={saved_val_acc})")

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    cm_norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)

    rep = {"n_val_decisions": int(len(X)), "n_seeds": 1,
           "val_acc": acc, "class_names": NAMES,
           "confusion_matrix_sum": cm.tolist(),
           "confusion_matrix_normalized": cm_norm.tolist(),
           "support": cm.sum(1).tolist(),
           "checkpoint": str(HEAD_PT), "cache": str(CACHE_V6),
           "scoring": "stride=1 (build_windows, matches training)"}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rep, indent=2, ensure_ascii=False))
    print(f"\nval_acc {acc*100:.2f}% (올바른 채점, stride=1)")

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(NAMES, rotation=45, ha="right")
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(NAMES)
    ax.set_xlabel("예측 (Predicted)"); ax.set_ylabel("정답 (Ground Truth)")
    ax.set_title(f"Val 혼동행렬(정규화) — image_proj v3+exp73_stage1v3\n"
                 f"acc {acc*100:.1f}%, n={len(X)}, stride=1(학습과 동일 채점)")
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            v = cm_norm[i, j]
            if v > 0.005:
                ax.text(j, i, f"{v*100:.0f}", ha="center", va="center",
                        color="white" if v > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label="ratio")
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=180, facecolor="white")
    print(f"저장: {OUT_PNG}")


if __name__ == "__main__":
    main()

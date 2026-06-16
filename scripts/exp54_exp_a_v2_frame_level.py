#!/usr/bin/env python3
"""
실험 A v2: Stage 1 v2 (frame-level label) 적용 후 에피소드 위치별 정확도

v1과의 비교:
  v1 결과: early/mid/late 모두 100% (격차 0%p) → 복도 패턴 암기
  v2 기대: late > early (basket 가까울수록 더 잘 맞춰야 함)

Ground truth: 프레임별 cx_det (detected basket position)
  - consistent=True 프레임만 평가
  - label: cx 기반 left/center/right

Usage:
  .venv/bin/python3 scripts/exp54_exp_a_v2_frame_level.py
"""

import json, sys, warnings
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VLM_PATH   = ROOT / ".vlms" / "kosmos-2-patch14-224"
DATA_PATH  = ROOT / "docs" / "v5" / "bbox_frame_level" / "bbox_dataset_frame_level.json"
STAGE1_V2  = ROOT / "runs" / "v5_nav" / "mlp" / "shared" / "stage1_v2_projs.pt"

PROJ_DIM = 256
LM_DIM   = 2048
VIS_DIM  = 1024
DIR_IDX  = {"left": 0, "center": 1, "right": 2}


def load_model(device):
    from transformers import AutoModelForVision2Seq, AutoProcessor
    ckpt = torch.load(str(STAGE1_V2), map_location=device, weights_only=False)
    print(f"[MODEL] Stage1 v2 val_acc={ckpt['val_acc']:.4f}")

    processor = AutoProcessor.from_pretrained(str(VLM_PATH))
    base = AutoModelForVision2Seq.from_pretrained(
        str(VLM_PATH), torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    )
    vision_model = base.vision_model.to(device).eval()

    image_proj = nn.Linear(VIS_DIM, PROJ_DIM).to(device)
    image_proj.load_state_dict(ckpt["image_proj"])
    image_proj.eval()

    text_proj = nn.Linear(LM_DIM, PROJ_DIM).to(device)
    text_proj.load_state_dict(ckpt["text_proj"])
    text_proj.eval()

    anchor_feats = F.normalize(text_proj(ckpt["anchor_raw"].to(device)), dim=-1)
    return processor, vision_model, image_proj, anchor_feats


@torch.no_grad()
def encode_images(vision_model, image_proj, processor, images, device):
    inputs = processor(images=images, return_tensors="pt")
    pv = inputs["pixel_values"].to(
        device, dtype=torch.float16 if device.type == "cuda" else torch.float32
    )
    out = vision_model(pixel_values=pv)
    feat = out.last_hidden_state.mean(dim=1).float()
    return F.normalize(image_proj(feat), dim=-1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    processor, vision_model, image_proj, anchor_feats = load_model(device)

    data = json.loads(DATA_PATH.read_text())

    seg_correct  = defaultdict(int)
    seg_total    = defaultdict(int)
    dir_seg_cor  = defaultdict(lambda: defaultdict(int))
    dir_seg_tot  = defaultdict(lambda: defaultdict(int))

    for ep in data:
        frames = [f for f in ep["frames"] if f["consistent"] and f["label"]]
        if not frames:
            continue

        n = len(frames)
        e_cut = max(1, n // 3)
        l_cut = n - max(1, n // 3)
        segs = {
            "early": frames[:e_cut],
            "mid":   frames[e_cut:l_cut],
            "late":  frames[l_cut:],
        }

        for seg_name, seg_frames in segs.items():
            if not seg_frames:
                continue
            images, labels = [], []
            for fr in seg_frames:
                try:
                    with h5py.File(ep["episode"], "r") as f:
                        images.append(Image.fromarray(f["observations"]["images"][fr["frame_idx"]]))
                    labels.append(DIR_IDX[fr["label"]])
                except:
                    pass
            if not images:
                continue

            feats = encode_images(vision_model, image_proj, processor, images, device)
            preds = (feats @ anchor_feats.T).argmax(dim=1).cpu().numpy()

            for pred, gt in zip(preds, labels):
                correct = int(pred == gt)
                seg_correct[seg_name]  += correct
                seg_total[seg_name]    += 1
                d = ["left","center","right"][gt]
                dir_seg_cor[d][seg_name] += correct
                dir_seg_tot[d][seg_name] += 1

    print(f"\n{'='*60}")
    print(f"  실험 A v2 — frame-level label 적용 후 에피소드 위치별 정확도")
    print(f"{'='*60}")
    print(f"\n  [v1 결과: early/mid/late 모두 100%  격차 0%p]")
    print(f"\n  {'구간':<8} {'정답':>6} {'전체':>6} {'정확도':>9}  해석")
    print("  " + "-" * 52)
    accs = {}
    for seg in ["early", "mid", "late"]:
        tot = seg_total[seg]
        cor = seg_correct[seg]
        acc = cor / tot * 100 if tot > 0 else 0.0
        accs[seg] = acc
        note = "(basket 멀리)" if seg == "early" else "(접근 중)" if seg == "mid" else "(basket 가까이)"
        print(f"  {seg:<8} {cor:>6} {tot:>6} {acc:>8.1f}%  {note}")

    gap = accs.get("late", 0) - accs.get("early", 0)
    print(f"\n  late - early 격차: {gap:+.1f}%p")
    if gap >= 10:
        verdict = "basket 위치를 본다 ✅  (격차 10%p↑)"
    elif gap >= 4:
        verdict = "약한 basket 의존성 (4~10%p)"
    else:
        verdict = "여전히 복도 패턴 암기 ⚠️  (격차 4%p 미만)"
    print(f"  → 판정: {verdict}")

    print(f"\n  방향별 early vs late:")
    print(f"  {'방향':<8} {'early':>8} {'mid':>8} {'late':>8}  격차")
    print("  " + "-" * 46)
    for d in ["left", "center", "right"]:
        row = []
        for seg in ["early", "mid", "late"]:
            tot = dir_seg_tot[d][seg]
            cor = dir_seg_cor[d][seg]
            row.append(cor / tot * 100 if tot > 0 else 0.0)
        g = row[2] - row[0]
        print(f"  {d:<8} {row[0]:>7.1f}% {row[1]:>7.1f}% {row[2]:>7.1f}%  {g:>+6.1f}%p")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

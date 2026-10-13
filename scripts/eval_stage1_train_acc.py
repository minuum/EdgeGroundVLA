#!/usr/bin/env python3
"""
Stage 1 (image_proj) 체크포인트의 train_acc 측정 — val_acc만 로그에 있고 train_acc는
학습 스크립트에 없어서 별도로 평가한다. 0521(v2, 3-class, 150ep)과 신규(v3, 5-class,
225ep, OWL) 둘 다 각자의 학습 데이터·클래스 스킴·split(seed=42, 동일 로직 재현)으로
train_eps를 복원해 train_acc를 계산한다.

Usage:
  .venv/bin/python3 scripts/eval_stage1_train_acc.py --version v2
  .venv/bin/python3 scripts/eval_stage1_train_acc.py --version v3
"""

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VLM_PATH = ROOT / ".vlms" / "kosmos-2-patch14-224"
PROJ_DIM = 256
VIS_DIM  = 1024

CONFIGS = {
    "v2": {
        "data_path": ROOT / "docs" / "v5" / "bbox_frame_level" / "bbox_dataset_frame_level.json",
        "ckpt_path": ROOT / "runs" / "v5_nav" / "mlp" / "exp54" / "stage1_v2" / "stage1_v2_projs.pt",
        "dir_idx": {"left": 0, "center": 1, "right": 2},
        "bgr_fix": False,   # 150ep H5는 observations 중첩 + 이미 정상 컬러였음
        "flat_h5": False,
    },
    "v3": {
        "data_path": ROOT / "docs" / "v5" / "bbox_nav_owl" / "bbox_dataset_v6_owl.json",
        "ckpt_path": ROOT / "runs" / "v5_nav" / "mlp" / "stage1_v3_5cls" / "stage1_v3_5cls_owl_projs.pt",
        "dir_idx": {"strong_left": 0, "weak_left": 1, "center": 2, "weak_right": 3, "strong_right": 4},
        "bgr_fix": True,    # 225ep(V6) H5는 BGR 저장 + flat 포맷
        "flat_h5": True,
    },
}


def load_frame_level_data(data_path):
    raw = json.loads(Path(data_path).read_text())
    episodes = []
    for ep in raw:
        frames = [f for f in ep["frames"] if f["consistent"] and f["label"] is not None]
        if frames:
            episodes.append({
                "episode": ep["episode"],
                "direction": ep["direction"],
                "frames": frames,
            })
    return episodes


def load_image(h5_path, frame_idx, flat_h5, bgr_fix, cache):
    f = cache.get(h5_path)
    if f is None:
        f = h5py.File(h5_path, "r")
        cache[h5_path] = f
    key = "images" if flat_h5 else None
    arr = (f["images"] if flat_h5 else f["observations"]["images"])[frame_idx]
    if bgr_fix:
        arr = arr[:, :, ::-1]
    return Image.fromarray(arr.astype("uint8").copy())


def encode_images(vision_model, image_proj, processor, images, device):
    inputs = processor(images=images, return_tensors="pt")
    pv = inputs["pixel_values"].to(
        device, dtype=torch.float16 if device.type == "cuda" else torch.float32
    )
    with torch.no_grad():
        out = vision_model(pixel_values=pv)
    feat = out.last_hidden_state.mean(dim=1).float()
    return F.normalize(image_proj(feat), dim=-1)


def evaluate(vision_model, image_proj, processor, anchor_proj, eps, dir_idx, device,
             flat_h5, bgr_fix, cache):
    image_proj.eval()
    n_cls = len(dir_idx)
    correct = total = 0
    confusion = np.zeros((n_cls, n_cls), dtype=int)
    BATCH = 32
    for ep in eps:
        images, gts = [], []
        for fr in ep["frames"]:
            try:
                images.append(load_image(ep["episode"], fr["frame_idx"], flat_h5, bgr_fix, cache))
                gts.append(dir_idx[fr["label"]])
            except Exception:
                continue
        # 에피소드 내에서도 BATCH개씩 끊어서 forward — 학습 스크립트와 달리 에피소드 크기가
        # 커도(V6는 최대 100+프레임) 메모리를 안정적으로 쓰기 위함
        for i in range(0, len(images), BATCH):
            chunk_imgs = images[i:i + BATCH]
            chunk_gts = gts[i:i + BATCH]
            feats = encode_images(vision_model, image_proj, processor, chunk_imgs, device)
            preds = (feats @ anchor_proj.T).argmax(dim=1).cpu().numpy()
            for pred, gt in zip(preds, chunk_gts):
                confusion[gt][pred] += 1
                correct += int(pred == gt)
                total += 1
    acc = correct / total if total > 0 else 0.0
    return acc, confusion, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=["v2", "v3"])
    args = ap.parse_args()
    cfg = CONFIGS[args.version]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}  [VERSION] {args.version}")

    all_eps = load_frame_level_data(cfg["data_path"])
    ep_dirs = [ep["direction"] for ep in all_eps]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(sss.split(np.zeros(len(all_eps)), ep_dirs))
    tr_eps = [all_eps[i] for i in tr_idx]
    val_eps = [all_eps[i] for i in te_idx]
    print(f"[DATA] train={len(tr_eps)}ep / val={len(val_eps)}ep "
          f"(전체 {len(all_eps)}ep, split seed=42 동일 재현)")

    from transformers import AutoModelForVision2Seq, AutoProcessor
    processor = AutoProcessor.from_pretrained(str(VLM_PATH))
    base_model = AutoModelForVision2Seq.from_pretrained(
        str(VLM_PATH), torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    vision_model = base_model.vision_model
    for p in vision_model.parameters():
        p.requires_grad = False

    ckpt = torch.load(cfg["ckpt_path"], map_location=device)
    image_proj = nn.Linear(VIS_DIM, PROJ_DIM).to(device)
    image_proj.load_state_dict(ckpt["image_proj"])
    text_proj = nn.Linear(2048, PROJ_DIM).to(device)
    text_proj.load_state_dict(ckpt["text_proj"])
    anchor_raw = ckpt["anchor_raw"].to(device)
    with torch.no_grad():
        anchor_proj = F.normalize(text_proj(anchor_raw), dim=-1)

    print(f"[CKPT] {cfg['ckpt_path']} (저장된 val_acc: {ckpt.get('val_acc', 'N/A')})")

    cache = {}
    t0 = time.time()
    tr_acc, tr_conf, tr_n = evaluate(
        vision_model, image_proj, processor, anchor_proj, tr_eps,
        cfg["dir_idx"], device, cfg["flat_h5"], cfg["bgr_fix"], cache,
    )
    print(f"[TRAIN] acc={tr_acc:.4f} (n={tr_n}, {time.time()-t0:.0f}s)")

    t0 = time.time()
    val_acc, val_conf, val_n = evaluate(
        vision_model, image_proj, processor, anchor_proj, val_eps,
        cfg["dir_idx"], device, cfg["flat_h5"], cfg["bgr_fix"], cache,
    )
    print(f"[VAL]   acc={val_acc:.4f} (n={val_n}, {time.time()-t0:.0f}s)")

    print(f"\n{'='*50}")
    print(f"  {args.version} — train_acc={tr_acc:.4f}  val_acc={val_acc:.4f}  "
          f"gap={tr_acc-val_acc:+.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

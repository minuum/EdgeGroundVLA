#!/usr/bin/env python3
"""
Exp54 Stage 1 v2: Frame-level Label로 재학습

기존 v1과의 차이:
  - 레이블 소스: path_type(에피소드 단위) → cx_det(프레임 단위 실제 basket 위치)
  - 데이터: bbox_dataset_frame_level.json (consistent=True 프레임만 사용)
  - class weight: center(68개) vs left(360개)/right(750개) 불균형 보정

Stage 1 목표:
  이미지에서 basket의 실제 위치(left/center/right)를 텍스트와 정렬.
  "복도 구도"가 아닌 "basket cx"를 기반으로 학습.

Usage:
  .venv/bin/python3 scripts/train_exp54_stage1_v2_frame_level.py
"""

import json, sys, time, warnings
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VLM_PATH  = ROOT / ".vlms" / "kosmos-2-patch14-224"
DATA_PATH = ROOT / "docs" / "v5" / "bbox_frame_level" / "bbox_dataset_frame_level.json"
OUT_DIR   = ROOT / "runs" / "v5_nav" / "mlp" / "exp54" / "stage1_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROJ_DIM   = 256
LM_DIM     = 2048
VIS_DIM    = 1024
EPOCHS     = 30
BATCH_SIZE = 16
LR         = 3e-4
TEMPERATURE = 0.07

DIR_IDX = {"left": 0, "center": 1, "right": 2}
ANCHOR_TEXTS = {
    "left":   "The gray basket is on the left side of the image",
    "center": "The gray basket is in the center of the image",
    "right":  "The gray basket is on the right side of the image",
}


# ─────────────────────────────────────────────────────────
# 모델
# ─────────────────────────────────────────────────────────

def load_base_model(device):
    from transformers import AutoModelForVision2Seq, AutoProcessor
    processor = AutoProcessor.from_pretrained(str(VLM_PATH))
    model = AutoModelForVision2Seq.from_pretrained(
        str(VLM_PATH),
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    return processor, model


@torch.no_grad()
def compute_text_anchors(model, processor, device):
    """3개 방향 텍스트 앵커 사전 계산 (frozen)."""
    text_model = model.text_model
    anchors = []
    for d in ["left", "center", "right"]:
        text = ANCHOR_TEXTS[d]
        inp = processor.tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
        out = text_model(
            input_ids=inp.input_ids,
            attention_mask=inp.attention_mask,
            output_hidden_states=True,
        )
        feat = out.hidden_states[-1][:, -1, :].float()  # (1, 2048)
        anchors.append(feat)
    return torch.cat(anchors, dim=0)  # (3, 2048)


# ─────────────────────────────────────────────────────────
# 데이터
# ─────────────────────────────────────────────────────────

def load_frame_level_data():
    """consistent=True 프레임만 추출. 에피소드 단위 split을 위해 구조 유지."""
    raw = json.loads(DATA_PATH.read_text())
    episodes = []
    for ep in raw:
        frames = [f for f in ep["frames"] if f["consistent"] and f["label"] is not None]
        if frames:
            episodes.append({
                "episode":   ep["episode"],
                "direction": ep["direction"],   # 에피소드 방향 (split 기준)
                "frames":    frames,
            })
    return episodes


def load_image(h5_path, frame_idx):
    with h5py.File(h5_path, "r") as f:
        return Image.fromarray(f["observations"]["images"][frame_idx])


# ─────────────────────────────────────────────────────────
# 학습
# ─────────────────────────────────────────────────────────

def encode_images(vision_model, image_proj, processor, images, device):
    inputs = processor(images=images, return_tensors="pt")
    pv = inputs["pixel_values"].to(
        device, dtype=torch.float16 if device.type == "cuda" else torch.float32
    )
    with torch.no_grad():
        out = vision_model(pixel_values=pv)
    feat = out.last_hidden_state.mean(dim=1).float()
    return F.normalize(image_proj(feat), dim=-1)  # (N, 256)


def evaluate(vision_model, image_proj, processor, anchor_proj, val_eps, device):
    image_proj.eval()
    correct = total = 0
    confusion = np.zeros((3, 3), dtype=int)
    for ep in val_eps:
        for fr in ep["frames"]:
            try:
                img = load_image(ep["episode"], fr["frame_idx"])
            except:
                continue
            feat = encode_images(vision_model, image_proj, processor, [img], device)
            pred = (feat @ anchor_proj.T).argmax(dim=1).item()
            gt   = DIR_IDX[fr["label"]]
            confusion[gt][pred] += 1
            correct += int(pred == gt)
            total   += 1
    acc = correct / total if total > 0 else 0.0
    return acc, confusion


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    # ── 데이터 로드
    all_eps = load_frame_level_data()
    ep_dirs = [ep["direction"] for ep in all_eps]

    # 방향별 프레임 통계
    from collections import Counter
    frame_label_counts = Counter(
        fr["label"] for ep in all_eps for fr in ep["frames"]
    )
    print(f"[DATA] 에피소드: {len(all_eps)}")
    print(f"       프레임: left={frame_label_counts['left']} "
          f"center={frame_label_counts['center']} "
          f"right={frame_label_counts['right']}")

    # 에피소드 단위 train/val split (80/20, stratify by direction)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(sss.split(np.zeros(len(all_eps)), ep_dirs))
    tr_eps = [all_eps[i] for i in tr_idx]
    val_eps = [all_eps[i] for i in te_idx]
    print(f"       train={len(tr_eps)} ep / val={len(val_eps)} ep")

    # ── 모델
    print("[MODEL] 로드 중...")
    processor, base_model = load_base_model(device)

    # CLIP vision model (LoRA 없이 순수 파인튜닝)
    vision_model = base_model.vision_model.to(device)
    for p in vision_model.parameters():
        p.requires_grad = False  # frozen — proj만 학습

    # text anchor 사전 계산
    anchor_raw = compute_text_anchors(base_model, processor, device)  # (3, 2048)
    print(f"[MODEL] text anchor 계산 완료")

    # proj layers
    image_proj = nn.Linear(VIS_DIM, PROJ_DIM).to(device)
    text_proj  = nn.Linear(LM_DIM,  PROJ_DIM).to(device)

    # anchor 투영 (고정)
    with torch.no_grad():
        anchor_proj = F.normalize(text_proj(anchor_raw), dim=-1)  # (3, 256)

    # ── class weight (center 보정)
    counts = np.array([
        frame_label_counts["left"],
        frame_label_counts["center"],
        frame_label_counts["right"],
    ], dtype=float)
    weights = (counts.sum() / (3 * counts))
    weights = weights / weights.sum() * 3
    class_weight = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    print(f"[LOSS] class weight: left={weights[0]:.2f} center={weights[1]:.2f} right={weights[2]:.2f}")

    optimizer = torch.optim.AdamW(
        list(image_proj.parameters()) + list(text_proj.parameters()),
        lr=LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc = 0.0
    best_state = None

    print(f"\n{'epoch':>6} {'val_acc':>9} {'best':>9}")
    print("-" * 30)

    for epoch in range(1, EPOCHS + 1):
        image_proj.train()
        text_proj.train()
        np.random.shuffle(tr_eps)

        batch_feats, batch_labels = [], []

        for ep in tr_eps:
            images = []
            labels = []
            for fr in ep["frames"]:
                try:
                    images.append(load_image(ep["episode"], fr["frame_idx"]))
                    labels.append(DIR_IDX[fr["label"]])
                except:
                    pass

            if not images:
                continue

            # 배치 단위 인코딩
            feats = encode_images(vision_model, image_proj, processor, images, device)
            batch_feats.append(feats)
            batch_labels.extend(labels)

            if len(batch_labels) >= BATCH_SIZE:
                x = torch.cat(batch_feats, dim=0)
                y = torch.tensor(batch_labels, dtype=torch.long, device=device)
                # anchor_proj를 현재 text_proj로 재계산
                ap = F.normalize(text_proj(anchor_raw), dim=-1)
                logits = (x @ ap.T) / TEMPERATURE
                loss = criterion(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_feats, batch_labels = [], []

        if batch_feats:
            x = torch.cat(batch_feats, dim=0)
            y = torch.tensor(batch_labels, dtype=torch.long, device=device)
            ap = F.normalize(text_proj(anchor_raw), dim=-1)
            logits = (x @ ap.T) / TEMPERATURE
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        # ── 평가 (anchor_proj 갱신)
        with torch.no_grad():
            anchor_proj = F.normalize(text_proj(anchor_raw), dim=-1)

        acc, confusion = evaluate(vision_model, image_proj, processor, anchor_proj, val_eps, device)

        if acc > best_acc:
            best_acc = acc
            best_state = {
                "image_proj": {k: v.cpu().clone() for k, v in image_proj.state_dict().items()},
                "text_proj":  {k: v.cpu().clone() for k, v in text_proj.state_dict().items()},
                "anchor_raw": anchor_raw.cpu().clone(),
                "val_acc":    best_acc,
            }

        print(f"{epoch:>6}  {acc:>8.4f}  {best_acc:>8.4f}")

    # ── 결과 출력
    print(f"\n{'='*50}")
    print(f"  Stage 1 v2 (frame-level label) 완료")
    print(f"  val_acc: {best_acc:.4f}")
    print(f"  v1(path_type): 1.0000  →  v2(frame-level): {best_acc:.4f}")
    print(f"{'='*50}")

    # 혼동 행렬 (best epoch 기준)
    image_proj.load_state_dict(best_state["image_proj"])
    text_proj.load_state_dict(best_state["text_proj"])
    with torch.no_grad():
        anchor_proj = F.normalize(text_proj(anchor_raw), dim=-1)
    _, confusion = evaluate(vision_model, image_proj, processor, anchor_proj, val_eps, device)

    dirs = ["left", "center", "right"]
    print(f"\n혼동 행렬 (행=실제, 열=예측):")
    print(f"{'':>9} {'left':>8} {'center':>8} {'right':>8}  {'정확도':>8}")
    for i, d in enumerate(dirs):
        row = confusion[i]
        acc_d = row[i] / row.sum() * 100 if row.sum() > 0 else 0
        print(f"  {d:>7} {row[0]:>8} {row[1]:>8} {row[2]:>8}  {acc_d:>7.1f}%")

    # 저장
    ckpt_path = OUT_DIR / "stage1_v2_projs.pt"
    torch.save(best_state, str(ckpt_path))
    print(f"\n[SAVE] {ckpt_path}")
    print(f"소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()

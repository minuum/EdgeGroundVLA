#!/usr/bin/env python3
"""
Exp55 Stage 1: Vision LoRA + Contrastive (Exp53 버그 수정판)

Exp53/Stage1 v2 문제점:
  - Exp53: get_peft_model(base.vision_model, ...) → standalone vision_model에 PEFT 적용
    → adapter key: base_model.model.encoder.layers.X
    → 풀 모델 로드 시 base_model.model.vision_model.model.encoder.layers.X 와 불일치
  - Stage1 v2: LoRA 없이 frozen CLIP + image_proj만 학습

이번 수정:
  - get_peft_model(full_model, ...) → 풀 모델 전체에 PEFT 적용
  - target_modules에 vision_model.model.encoder.layers.{i} 경로 명시
  - adapter key가 처음부터 풀 모델 경로로 저장됨 → Stage2에서 정상 로드

Usage:
  .venv/bin/python3 scripts/train_exp55_stage1_lora.py
  .venv/bin/python3 scripts/train_exp55_stage1_lora.py --epochs 50 --lora-rank 16
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
OUT_DIR   = ROOT / "runs" / "v5_nav" / "mlp" / "exp55" / "stage1_lora"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LORA_R      = 16
LORA_ALPHA  = 32
LORA_LAYERS = list(range(16, 24))

PROJ_DIM    = 256
LM_DIM      = 2048
VIS_DIM     = 1024
EPOCHS      = 40
BATCH_SIZE  = 16
LR          = 2e-4
TEMPERATURE = 0.07

DIR_IDX = {"left": 0, "center": 1, "right": 2}
ANCHOR_TEXTS = {
    "left":   "The gray basket is on the left side of the image",
    "center": "The gray basket is in the center of the image",
    "right":  "The gray basket is on the right side of the image",
}


# ─── 데이터 ───────────────────────────────────────────────

def load_frame_level_data():
    raw = json.loads(DATA_PATH.read_text())
    episodes = []
    for ep in raw:
        frames = [f for f in ep["frames"] if f.get("consistent") and f.get("label")]
        if frames:
            episodes.append({
                "episode":   ep["episode"],
                "direction": ep.get("direction", ep.get("path_type", "unknown")),
                "frames":    frames,
            })
    return episodes


def load_image(h5_path, frame_idx):
    with h5py.File(h5_path, "r") as f:
        return Image.fromarray(f["observations"]["images"][frame_idx])


# ─── 텍스트 앵커 사전 계산 ────────────────────────────────

@torch.no_grad()
def compute_text_anchors(text_model, processor, device):
    anchors = []
    for d in ["left", "center", "right"]:
        inp = processor.tokenizer(
            ANCHOR_TEXTS[d], return_tensors="pt", add_special_tokens=True
        ).to(device)
        out = text_model(
            input_ids=inp.input_ids,
            attention_mask=inp.attention_mask,
            output_hidden_states=True,
        )
        feat = out.hidden_states[-1][:, -1, :].float()  # (1, 2048)
        anchors.append(feat)
    return torch.cat(anchors, dim=0)  # (3, 2048)


# ─── 인코딩 ───────────────────────────────────────────────

def encode_images(vision_model, image_proj, processor, images, device):
    inputs = processor(images=images, return_tensors="pt")
    pv = inputs["pixel_values"].to(device, dtype=torch.float16)
    out = vision_model(pixel_values=pv)
    feat = out.last_hidden_state.mean(dim=1).float()
    return F.normalize(image_proj(feat), dim=-1)  # (N, 256)


@torch.no_grad()
def evaluate(vision_model, image_proj, processor, anchor_proj, val_eps, device):
    image_proj.eval()
    vision_model.eval()
    correct = total = 0
    confusion = np.zeros((3, 3), dtype=int)
    for ep in val_eps:
        for fr in ep["frames"]:
            try:
                img = load_image(ep["episode"], fr["frame_idx"])
            except Exception:
                continue
            feat = encode_images(vision_model, image_proj, processor, [img], device)
            pred = (feat @ anchor_proj.T).argmax(dim=1).item()
            gt   = DIR_IDX[fr["label"]]
            confusion[gt][pred] += 1
            correct += int(pred == gt)
            total   += 1
    return (correct / total if total > 0 else 0.0), confusion


# ─── 학습 ─────────────────────────────────────────────────

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",    type=int,   default=EPOCHS)
    p.add_argument("--lora-rank", type=int,   default=LORA_R)
    p.add_argument("--lr",        type=float, default=LR)
    args = p.parse_args()

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    print(f"[CONFIG] epochs={args.epochs}  lora_rank={args.lora_rank}  lr={args.lr}")

    # 데이터
    all_eps = load_frame_level_data()
    ep_dirs = [ep["direction"] for ep in all_eps]
    from collections import Counter
    label_counts = Counter(fr["label"] for ep in all_eps for fr in ep["frames"])
    print(f"[DATA] episodes={len(all_eps)}  "
          f"left={label_counts['left']} center={label_counts['center']} right={label_counts['right']}")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(sss.split(np.zeros(len(all_eps)), ep_dirs))
    tr_eps  = [all_eps[i] for i in tr_idx]
    val_eps = [all_eps[i] for i in te_idx]
    print(f"       train={len(tr_eps)} ep / val={len(val_eps)} ep")

    # 모델 로드
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from peft import LoraConfig, get_peft_model

    print("[MODEL] Kosmos-2 로드 중...", flush=True)
    processor  = AutoProcessor.from_pretrained(str(VLM_PATH))
    base_model = AutoModelForVision2Seq.from_pretrained(
        str(VLM_PATH), torch_dtype=torch.float16
    )

    # ── 핵심 수정: 풀 모델에 PEFT 적용 (vision_model 경로 명시) ──
    target_modules = []
    for i in LORA_LAYERS:
        target_modules.append(f"vision_model.model.encoder.layers.{i}.self_attn.q_proj")
        target_modules.append(f"vision_model.model.encoder.layers.{i}.self_attn.v_proj")

    lora_cfg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        # task_type 없음 → PeftModel (generic), CAUSAL_LM wrapper 아님
    )
    peft_model = get_peft_model(base_model, lora_cfg)
    peft_model.print_trainable_parameters()

    # vision_model 추출 (LoRA Linear 레이어가 in-place로 교체돼 있음)
    vision_model = peft_model.base_model.model.vision_model.to(device)

    # text_model frozen (LoRA 없음, 앵커 계산용)
    text_model = peft_model.base_model.model.text_model.to(device)
    for p in text_model.parameters():
        p.requires_grad = False

    # 텍스트 앵커 사전 계산
    anchor_raw = compute_text_anchors(text_model, processor, device)  # (3, 2048)
    print("[MODEL] 텍스트 앵커 계산 완료")

    # projection heads
    image_proj = nn.Linear(VIS_DIM, PROJ_DIM).to(device)
    text_proj  = nn.Linear(LM_DIM,  PROJ_DIM).to(device)

    # class weight
    counts = np.array([label_counts["left"], label_counts["center"], label_counts["right"]], dtype=float)
    weights = counts.sum() / (3 * counts)
    weights /= weights.sum() / 3
    class_weight = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    print(f"[LOSS] class weight: left={weights[0]:.2f} center={weights[1]:.2f} right={weights[2]:.2f}")

    # optimizer: LoRA params + projection heads
    lora_params  = [p for p in vision_model.parameters() if p.requires_grad]
    proj_params  = list(image_proj.parameters()) + list(text_proj.parameters())
    optimizer = torch.optim.AdamW(
        [{"params": lora_params, "lr": args.lr * 0.1},   # LoRA는 lr 낮게
         {"params": proj_params, "lr": args.lr}],
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc   = 0.0
    best_state = None

    print(f"\n{'epoch':>6} {'loss':>8} {'val_acc':>9} {'best':>9}")
    print("-" * 38)

    for epoch in range(1, args.epochs + 1):
        vision_model.train()
        image_proj.train()
        text_proj.train()
        np.random.shuffle(tr_eps)

        batch_feats, batch_labels = [], []
        epoch_loss = 0.0
        epoch_n    = 0

        for ep in tr_eps:
            images, labels = [], []
            for fr in ep["frames"]:
                try:
                    images.append(load_image(ep["episode"], fr["frame_idx"]))
                    labels.append(DIR_IDX[fr["label"]])
                except Exception:
                    pass
            if not images:
                continue

            feats = encode_images(vision_model, image_proj, processor, images, device)
            batch_feats.append(feats)
            batch_labels.extend(labels)

            if len(batch_labels) >= BATCH_SIZE:
                x  = torch.cat(batch_feats, dim=0)
                y  = torch.tensor(batch_labels, dtype=torch.long, device=device)
                ap = F.normalize(text_proj(anchor_raw), dim=-1)
                loss = criterion(x @ ap.T / TEMPERATURE, y)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(lora_params + proj_params, 1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(y)
                epoch_n    += len(y)
                batch_feats, batch_labels = [], []

        if batch_feats:
            x  = torch.cat(batch_feats, dim=0)
            y  = torch.tensor(batch_labels, dtype=torch.long, device=device)
            ap = F.normalize(text_proj(anchor_raw), dim=-1)
            loss = criterion(x @ ap.T / TEMPERATURE, y)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(lora_params + proj_params, 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(y)
            epoch_n    += len(y)

        scheduler.step()

        with torch.no_grad():
            anchor_proj = F.normalize(text_proj(anchor_raw), dim=-1)

        acc, confusion = evaluate(vision_model, image_proj, processor, anchor_proj, val_eps, device)
        avg_loss = epoch_loss / max(1, epoch_n)

        if acc > best_acc:
            best_acc = acc
            # LoRA 어댑터 저장 (풀 모델 경로 key → Stage2에서 정상 로드)
            lora_adapter_dir = OUT_DIR / "lora_adapter"
            peft_model.save_pretrained(str(lora_adapter_dir))
            # proj 저장
            best_state = {
                "image_proj":      {k: v.cpu().clone() for k, v in image_proj.state_dict().items()},
                "text_proj":       {k: v.cpu().clone() for k, v in text_proj.state_dict().items()},
                "anchor_raw":      anchor_raw.cpu().clone(),
                "val_acc":         best_acc,
                "lora_adapter_dir": str(lora_adapter_dir),
                "lora_rank":       args.lora_rank,
            }
            torch.save(best_state, str(OUT_DIR / "stage1_lora_projs.pt"))
            mark = "  ← best"
        else:
            mark = ""

        print(f"{epoch:>6}  {avg_loss:>8.4f}  {acc:>8.4f}  {best_acc:>8.4f}{mark}")

    # 최종 혼동 행렬
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

    print(f"\n[완료] best val_acc={best_acc:.4f}  ({(time.time()-t0)/60:.1f}분)")
    print(f"[저장]")
    print(f"  LoRA adapter : {OUT_DIR / 'lora_adapter'}")
    print(f"  proj weights : {OUT_DIR / 'stage1_lora_projs.pt'}")
    print(f"\n[다음 단계]")
    print(f"  .venv/bin/python3 scripts/train_exp55_stage2_lora.py")


if __name__ == "__main__":
    main()

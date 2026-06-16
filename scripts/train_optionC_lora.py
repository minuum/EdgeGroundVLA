#!/usr/bin/env python3
"""
Option C: Pure Kosmos-2 End-to-End VLA Fine-tuning (LoRA)

generate() 기반 액션 예측 — text path가 살아있는 Pure Kosmos-2를 활용.

다른 VLA와의 차이:
  - RT-2/OpenVLA: 연속 액션 256-bin 양자화, 수십억 파라미터
  - Option C: 8-class 텍스트 토큰, 150 ep, Kosmos-2 1.6B

프롬프트 전략 (--prompt):
  p1  blind:    이미지만 보고 액션 (text path 최대 활용)
  p2  bbox:     색상 감지 bbox 좌표 주입 (하이브리드)
  p3  grounding 모델 자체 grounding 후 액션 (순수 end-to-end, 느림)

Usage:
  .venv/bin/python3 scripts/train_optionC_lora.py --prompt p2
  .venv/bin/python3 scripts/train_optionC_lora.py --prompt p1 --epochs 150
  .venv/bin/python3 scripts/train_optionC_lora.py --prompt p2 --lora-rank 16
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VLM_PATH  = ROOT / ".vlms" / "kosmos-2-patch14-224"
DATA_PATH = ROOT / "docs" / "v5" / "bbox_nav_exp46" / "bbox_dataset_full.json"
DATA_DIR  = ROOT / "ROS_action" / "mobile_vla_dataset_v5"
OUT_DIR   = ROOT / "runs" / "v5_nav" / "optionC"

CLASS_NAMES = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "ROT_L", "ROT_R"]
NUM_CLASSES = 8
WINDOW      = 8


# ─── 프롬프트 빌더 ──────────────────────────────────────────────────────────

def build_prompt(frames, t, path_type, strategy="p2"):
    action_str = "/".join(CLASS_NAMES[1:])
    if strategy == "p1":
        return (
            f"<image> A mobile robot navigates a corridor following a gray basket. "
            f"Choose navigation action ({action_str}):"
        )
    elif strategy == "p2":
        fr = frames[t]
        cx, cy, area = fr.get("cx", 0.5), fr.get("cy", 0.5), fr.get("area", 0.05)
        has = bool(fr.get("has_bbox", False))
        if has:
            loc = "left" if cx < 0.4 else ("right" if cx > 0.6 else "center")
            return (
                f"<image> Gray basket detected at ({cx:.2f}, {cy:.2f}), "
                f"area={area:.3f} ({loc} side). "
                f"Navigation action ({action_str}):"
            )
        else:
            return (
                f"<image> No basket visible. "
                f"Navigation action ({action_str}):"
            )
    elif strategy == "p3":
        # p3은 추론 시 두 단계 — 학습은 p2와 동일하게 시작
        return build_prompt(frames, t, path_type, "p2")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ─── 데이터셋 ────────────────────────────────────────────────────────────────

class NavVLADataset(Dataset):
    def __init__(self, episodes, proc, prompt_strategy="p2", augment=False):
        self.proc     = proc
        self.strategy = prompt_strategy
        self.augment  = augment
        self.samples  = []

        for ep in episodes:
            ep_path = Path(ep["episode"])
            if ep_path.is_absolute() and ep_path.exists():
                h5_path = ep_path
            else:
                cands = list(DATA_DIR.glob(f"{ep_path.stem}.h5"))
                if not cands:
                    cands = list(DATA_DIR.glob(f"**/{ep_path.stem}.h5"))
                if not cands:
                    continue
                h5_path = cands[0]

            frames = ep["frames"]
            pt     = ep.get("path_type", "unknown")
            for t, fr in enumerate(frames):
                self.samples.append({
                    "h5_path":  str(h5_path),
                    "frame_idx": fr.get("frame_idx", t),
                    "frames":   frames,
                    "t":        t,
                    "path_type": pt,
                    "gt_class": fr.get("gt_class", 1),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        with h5py.File(s["h5_path"], "r") as f:
            arr = f["observations"]["images"][s["frame_idx"]]
        img = Image.fromarray(arr)

        if self.augment and np.random.rand() < 0.3:
            # 좌우 flip augment — gt_class도 반전
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            flip_map = {0:0, 1:1, 2:3, 3:2, 4:5, 5:4, 6:7, 7:6}
            gt = flip_map[s["gt_class"]]
        else:
            gt = s["gt_class"]

        prompt = build_prompt(s["frames"], s["t"], s["path_type"], self.strategy)
        target = CLASS_NAMES[gt]
        return img, prompt, target, gt


def collate_fn(batch, proc, device="cpu"):
    imgs, prompts, targets, gt_classes = zip(*batch)

    # 입력 인코딩 (prompt + image)
    enc = proc(text=list(prompts), images=list(imgs), return_tensors="pt", padding=True)

    # 타겟 토큰화 (action 텍스트만)
    target_enc = proc.tokenizer(
        list(targets), return_tensors="pt", padding=True, add_special_tokens=False
    )

    return {
        "input_ids":      enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "pixel_values":   enc.get("pixel_values"),
        "image_embeds":   enc.get("image_embeds"),
        "target_ids":     target_enc["input_ids"],
        "gt_classes":     torch.tensor(gt_classes, dtype=torch.long),
    }


# ─── 학습 ────────────────────────────────────────────────────────────────────

def train(args):
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from peft import LoraConfig, get_peft_model, TaskType

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    # 출력 디렉토리
    run_name = f"optionC_{args.prompt}_r{args.lora_rank}"
    ckpt_dir = OUT_DIR / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 데이터
    data = json.loads(DATA_PATH.read_text())
    labels = [ep["path_type"] for ep in data]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, va_idx = next(sss.split(np.zeros(len(data)), labels))
    tr_eps = [data[i] for i in tr_idx]
    va_eps = [data[i] for i in va_idx]
    print(f"[DATA] train={len(tr_eps)} val={len(va_eps)} episodes")

    # 프로세서 + 모델
    proc  = AutoProcessor.from_pretrained(str(VLM_PATH))
    model = AutoModelForVision2Seq.from_pretrained(str(VLM_PATH), torch_dtype=torch.float16)

    # LoRA 설정 — text decoder + vision encoder 상위 레이어
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        lora_dropout=0.05,
        target_modules=[
            # text decoder
            "text_model.model.layers.16.self_attn.q_proj",
            "text_model.model.layers.16.self_attn.v_proj",
            "text_model.model.layers.17.self_attn.q_proj",
            "text_model.model.layers.17.self_attn.v_proj",
            "text_model.model.layers.18.self_attn.q_proj",
            "text_model.model.layers.18.self_attn.v_proj",
            "text_model.model.layers.19.self_attn.q_proj",
            "text_model.model.layers.19.self_attn.v_proj",
            "text_model.model.layers.20.self_attn.q_proj",
            "text_model.model.layers.20.self_attn.v_proj",
            "text_model.model.layers.21.self_attn.q_proj",
            "text_model.model.layers.21.self_attn.v_proj",
            "text_model.model.layers.22.self_attn.q_proj",
            "text_model.model.layers.22.self_attn.v_proj",
            "text_model.model.layers.23.self_attn.q_proj",
            "text_model.model.layers.23.self_attn.v_proj",
            # vision encoder
            "vision_model.model.encoder.layers.20.self_attn.q_proj",
            "vision_model.model.encoder.layers.20.self_attn.v_proj",
            "vision_model.model.encoder.layers.21.self_attn.q_proj",
            "vision_model.model.encoder.layers.21.self_attn.v_proj",
            "vision_model.model.encoder.layers.22.self_attn.q_proj",
            "vision_model.model.encoder.layers.22.self_attn.v_proj",
            "vision_model.model.encoder.layers.23.self_attn.q_proj",
            "vision_model.model.encoder.layers.23.self_attn.v_proj",
        ],
        inference_mode=False,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model = model.to(device)

    # 데이터셋
    tr_ds = NavVLADataset(tr_eps, proc, args.prompt, augment=True)
    va_ds = NavVLADataset(va_eps, proc, args.prompt, augment=False)
    print(f"[DATA] train frames={len(tr_ds)}  val frames={len(va_ds)}")

    # 타겟 토큰 ID 사전 계산 (각 action 이름의 첫 토큰)
    action_token_ids = []
    for name in CLASS_NAMES:
        toks = proc.tokenizer(name, add_special_tokens=False)["input_ids"]
        action_token_ids.append(toks[0])
    action_token_ids = torch.tensor(action_token_ids, dtype=torch.long).to(device)
    print(f"[TOKEN] action ids: {action_token_ids.tolist()}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.cuda.amp.GradScaler()
    best_acc = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_correct = 0
        tr_total = 0
        t0 = time.time()

        for bi in range(0, len(tr_ds), args.batch_size):
            batch_items = [tr_ds[j] for j in range(bi, min(bi + args.batch_size, len(tr_ds)))]
            imgs, prompts, targets, gt_classes = zip(*batch_items)

            # 각 샘플 개별 처리 (패딩 이슈 회피)
            batch_loss = 0.0
            batch_preds = []
            for img, prompt, target, gt in zip(imgs, prompts, targets, gt_classes):
                enc = proc(text=prompt, images=img, return_tensors="pt")
                input_ids      = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)
                pixel_values   = enc.get("pixel_values")
                if pixel_values is not None:
                    pixel_values = pixel_values.to(device, dtype=torch.float16)
                iep_mask = enc.get("image_embeds_position_mask")
                if iep_mask is not None:
                    iep_mask = iep_mask.to(device)

                # 타겟 토큰
                target_ids = proc.tokenizer(
                    target, add_special_tokens=False, return_tensors="pt"
                )["input_ids"].to(device)

                # Forward: labels = 타겟 토큰만 (prompt는 -100)
                labels = torch.full((1, input_ids.shape[1] + target_ids.shape[1]),
                                    -100, dtype=torch.long, device=device)
                labels[0, input_ids.shape[1]:] = target_ids[0, :labels.shape[1]-input_ids.shape[1]]

                # input_ids에 타겟 추가
                full_input = torch.cat([input_ids, target_ids], dim=1)
                full_mask  = torch.cat([attention_mask,
                                        torch.ones_like(target_ids)], dim=1)

                with torch.cuda.amp.autocast():
                    kwargs = dict(input_ids=full_input, attention_mask=full_mask, labels=labels)
                    if pixel_values is not None:
                        kwargs["pixel_values"] = pixel_values
                    if iep_mask is not None:
                        ext = torch.zeros(1, target_ids.shape[1], dtype=iep_mask.dtype, device=device)
                        kwargs["image_embeds_position_mask"] = torch.cat([iep_mask, ext], dim=1)
                    out = model(**kwargs)
                    loss = out.loss

                batch_loss += loss
                tr_loss    += loss.item()

                # 예측: 입력만으로 next token
                with torch.no_grad(), torch.cuda.amp.autocast():
                    kw2 = dict(input_ids=input_ids, attention_mask=attention_mask,
                               max_new_tokens=3, do_sample=False)
                    if pixel_values is not None:
                        kw2["pixel_values"] = pixel_values
                    if iep_mask is not None:
                        kw2["image_embeds_position_mask"] = iep_mask
                    gen = model.generate(**kw2)
                    new_tok = gen[0][input_ids.shape[1]]
                    pred_class = (action_token_ids == new_tok).nonzero(as_tuple=True)
                    pred = pred_class[0][0].item() if len(pred_class[0]) > 0 else -1

                tr_correct += int(pred == gt)
                tr_total   += 1

            optimizer.zero_grad()
            scaler.scale(batch_loss / len(batch_items)).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0
            )
            scaler.step(optimizer)
            scaler.update()

        scheduler.step()
        tr_acc = tr_correct / max(1, tr_total)

        # 검증
        model.eval()
        va_correct = 0
        va_total   = 0
        with torch.no_grad():
            for i in range(min(len(va_ds), 200)):  # 속도를 위해 200개 제한
                img, prompt, target, gt = va_ds[i]
                enc = proc(text=prompt, images=img, return_tensors="pt")
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)
                pixel_values = enc.get("pixel_values")
                if pixel_values is not None:
                    pixel_values = pixel_values.to(device, dtype=torch.float16)
                iep_mask = enc.get("image_embeds_position_mask")
                if iep_mask is not None:
                    iep_mask = iep_mask.to(device)

                kw = dict(input_ids=input_ids, attention_mask=attention_mask,
                          max_new_tokens=3, do_sample=False)
                if pixel_values is not None:
                    kw["pixel_values"] = pixel_values
                if iep_mask is not None:
                    kw["image_embeds_position_mask"] = iep_mask
                with torch.cuda.amp.autocast():
                    gen = model.generate(**kw)
                new_tok = gen[0][input_ids.shape[1]]
                pred_class = (action_token_ids == new_tok).nonzero(as_tuple=True)
                pred = pred_class[0][0].item() if len(pred_class[0]) > 0 else -1
                va_correct += int(pred == gt)
                va_total   += 1

        va_acc = va_correct / max(1, va_total)
        elapsed = time.time() - t0

        print(f"[Epoch {epoch:3d}/{args.epochs}] "
              f"loss={tr_loss/max(1,tr_total):.4f}  "
              f"tr_acc={tr_acc:.3f}  va_acc={va_acc:.3f}  "
              f"({elapsed:.0f}s)", flush=True)

        history.append({"epoch": epoch, "tr_loss": tr_loss/max(1,tr_total),
                         "tr_acc": tr_acc, "va_acc": va_acc})

        if va_acc > best_acc:
            best_acc = va_acc
            # LoRA adapter 저장
            model.save_pretrained(str(ckpt_dir / "adapter_best"))
            torch.save({"epoch": epoch, "va_acc": va_acc, "tr_acc": tr_acc,
                        "prompt_strategy": args.prompt, "lora_rank": args.lora_rank},
                       ckpt_dir / "meta_best.pt")
            print(f"  [SAVE] best va_acc={va_acc:.3f}")

    # 최종 저장
    model.save_pretrained(str(ckpt_dir / "adapter_final"))
    json.dump(history, open(ckpt_dir / "history.json", "w"), indent=2)
    print(f"\n[완료] best val_acc={best_acc:.3f}")
    print(f"[CKPT] {ckpt_dir / 'adapter_best'}")
    print(f"\n[평가 실행]")
    print(f"  .venv/bin/python3 scripts/test_kosmos2_raw.py --mode all --ckpt {ckpt_dir/'adapter_best'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt",     choices=["p1", "p2", "p3"], default="p2")
    p.add_argument("--lora-rank",  type=int,   default=8)
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--lr",         type=float, default=5e-5)
    p.add_argument("--batch-size", type=int,   default=4)
    args = p.parse_args()

    print(f"[CONFIG] prompt={args.prompt}  lora_rank={args.lora_rank}  "
          f"epochs={args.epochs}  lr={args.lr}")
    train(args)


if __name__ == "__main__":
    main()

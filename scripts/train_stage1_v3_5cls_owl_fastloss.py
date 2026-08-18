#!/usr/bin/env python3
"""Stage 1 v3 재학습 — 비전 특징 캐싱으로 속도 개선 + train/val loss 로깅.

train_stage1_v3_5cls_owl.py의 원본 병목: frozen Kosmos-2 vision_model(0.303B)의
forward pass를 매 epoch마다 16,599프레임 전체에 대해 다시 계산함(총 277분).
vision_model은 학습되지 않으므로 그 출력(1024차원, mean-pooled)은 epoch에 무관하게
동일하다 — 한 번만 계산해 캐싱하고, 이후 epoch은 image_proj(작은 Linear)만 학습한다.

추가: 원본 스크립트는 val_acc만 기록하고 loss는 기록하지 않았음(0817 hwp 빨간
메모 "Loss 그래프가 있으면 추가 필요함" 대응) — 매 epoch train/val loss를 기록해
docs/v5/detector/stage1_v3_loss_curve.json으로 저장.

동일 시드(42)·동일 split·동일 하이퍼파라미터로 원본과 같은 학습 경로를 재현하는지
확인 가능(최종 val_acc가 기존 0.9409와 일치해야 캐싱이 정확함을 검증한 것).

Usage:
  python3 scripts/train_stage1_v3_5cls_owl_fastloss.py
"""
import json, sys, time, warnings
from pathlib import Path

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
DATA_PATH = ROOT / "docs" / "v5" / "bbox_nav_owl" / "bbox_dataset_v6_owl.json"
OUT_DIR   = ROOT / "runs" / "v5_nav" / "mlp" / "stage1_v3_5cls"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FEAT_CACHE_PATH = ROOT / "runs" / "v5_nav" / "mlp" / "stage1_v3_5cls" / "raw_vision_feat_cache.pt"
CURVE_OUT = ROOT / "docs" / "v5" / "detector" / "stage1_v3_loss_curve.json"

PROJ_DIM, LM_DIM, VIS_DIM = 256, 2048, 1024
EPOCHS, BATCH_SIZE, LR, TEMPERATURE = 30, 16, 3e-4, 0.07

DIR_IDX = {"strong_left": 0, "weak_left": 1, "center": 2, "weak_right": 3, "strong_right": 4}
N_CLASSES = len(DIR_IDX)
ANCHOR_TEXTS = {
    "strong_left":  "The gray basket is strongly on the left side of the image",
    "weak_left":    "The gray basket is slightly on the left side of the image",
    "center":       "The gray basket is in the center of the image",
    "weak_right":   "The gray basket is slightly on the right side of the image",
    "strong_right": "The gray basket is strongly on the right side of the image",
}
ANCHOR_ORDER = ["strong_left", "weak_left", "center", "weak_right", "strong_right"]


def load_base_model(device):
    from transformers import AutoModelForVision2Seq, AutoProcessor
    processor = AutoProcessor.from_pretrained(str(VLM_PATH))
    model = AutoModelForVision2Seq.from_pretrained(
        str(VLM_PATH), torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device)
    return processor, model


@torch.no_grad()
def compute_text_anchors(model, processor, device):
    text_model = model.text_model
    anchors = []
    for d in ANCHOR_ORDER:
        inp = processor.tokenizer(ANCHOR_TEXTS[d], return_tensors="pt", add_special_tokens=True).to(device)
        out = text_model(input_ids=inp.input_ids, attention_mask=inp.attention_mask, output_hidden_states=True)
        anchors.append(out.hidden_states[-1][:, -1, :].float())
    return torch.cat(anchors, dim=0)


def load_frame_level_data():
    raw = json.loads(DATA_PATH.read_text())
    episodes = []
    for ep in raw:
        frames = [f for f in ep["frames"] if f["consistent"] and f["label"] is not None]
        if frames:
            episodes.append({"episode": ep["episode"], "direction": ep["direction"], "frames": frames})
    return episodes


_IMG_CACHE = {}


def preload_all_episodes(episodes):
    import h5py
    total = sum(len(ep["frames"]) for ep in episodes)
    print(f"[PRELOAD] {total}개 프레임 미리 읽고 BGR→RGB 반전해서 RAM에 캐싱 중...")
    t0 = time.time()
    n = 0
    for ep in episodes:
        path = ep["episode"]
        with h5py.File(path, "r") as f:
            images = f["images"]
            for fr in ep["frames"]:
                idx = fr["frame_idx"]
                arr = images[idx][:, :, ::-1].astype("uint8").copy()
                _IMG_CACHE[(path, idx)] = arr
                n += 1
        if n % 3000 == 0:
            print(f"  ... {n}/{total} ({time.time()-t0:.0f}s)")
    print(f"[PRELOAD] 완료 — {n}개 프레임 캐싱됨 ({time.time()-t0:.0f}초 소요)")


def load_image(h5_path, frame_idx):
    return Image.fromarray(_IMG_CACHE[(h5_path, frame_idx)])


@torch.no_grad()
def build_raw_feat_cache(vision_model, processor, all_eps, device):
    """frozen vision_model의 raw 1024차원 출력을 프레임별로 한 번만 계산해 캐싱."""
    if FEAT_CACHE_PATH.exists():
        print(f"[FEAT-CACHE] 기존 캐시 재사용: {FEAT_CACHE_PATH}")
        return torch.load(FEAT_CACHE_PATH, map_location=device)

    print("[FEAT-CACHE] raw vision feature 캐시가 없음 — 최초 1회 계산 중...")
    t0 = time.time()
    cache = {}
    batch_keys, batch_imgs = [], []

    def flush():
        if not batch_imgs:
            return
        inputs = processor(images=batch_imgs, return_tensors="pt")
        pv = inputs["pixel_values"].to(device, dtype=torch.float16 if device.type == "cuda" else torch.float32)
        out = vision_model(pixel_values=pv)
        feat = out.last_hidden_state.mean(dim=1).float().cpu()
        for k, v in zip(batch_keys, feat):
            cache[k] = v
        batch_keys.clear(); batch_imgs.clear()

    n = 0
    for ep in all_eps:
        for fr in ep["frames"]:
            key = (ep["episode"], fr["frame_idx"])
            if key in cache:
                continue
            try:
                img = load_image(ep["episode"], fr["frame_idx"])
            except Exception:
                continue
            batch_keys.append(key)
            batch_imgs.append(img)
            n += 1
            if len(batch_imgs) >= 32:
                flush()
                if n % 1600 == 0:
                    print(f"  ... {n}장 처리 ({time.time()-t0:.0f}s)")
    flush()
    print(f"[FEAT-CACHE] 완료 — {len(cache)}개 프레임, {(time.time()-t0):.0f}초 소요")
    torch.save(cache, FEAT_CACHE_PATH)
    print(f"[FEAT-CACHE] 저장: {FEAT_CACHE_PATH}")
    return cache


def get_cached_feat(cache, h5_path, frame_idx, device):
    return cache[(h5_path, frame_idx)].to(device)


def evaluate(cache, image_proj, anchor_proj, val_eps, device, criterion):
    image_proj.eval()
    correct = total = 0
    confusion = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    losses = []
    with torch.no_grad():
        for ep in val_eps:
            for fr in ep["frames"]:
                key = (ep["episode"], fr["frame_idx"])
                if key not in cache:
                    continue
                raw = get_cached_feat(cache, ep["episode"], fr["frame_idx"], device).unsqueeze(0)
                feat = F.normalize(image_proj(raw), dim=-1)
                logits = (feat @ anchor_proj.T) / TEMPERATURE
                gt = DIR_IDX[fr["label"]]
                y = torch.tensor([gt], dtype=torch.long, device=device)
                losses.append(criterion(logits, y).item())
                pred = logits.argmax(dim=1).item()
                confusion[gt][pred] += 1
                correct += int(pred == gt)
                total += 1
    acc = correct / total if total > 0 else 0.0
    val_loss = float(np.mean(losses)) if losses else float("nan")
    return acc, val_loss, confusion


def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    all_eps = load_frame_level_data()
    preload_all_episodes(all_eps)
    ep_dirs = [ep["direction"] for ep in all_eps]

    from collections import Counter
    frame_label_counts = Counter(fr["label"] for ep in all_eps for fr in ep["frames"])
    print(f"[DATA] 에피소드: {len(all_eps)}")

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(sss.split(np.zeros(len(all_eps)), ep_dirs))
    tr_eps = [all_eps[i] for i in tr_idx]
    val_eps = [all_eps[i] for i in te_idx]
    print(f"       train={len(tr_eps)} ep / val={len(val_eps)} ep")

    print("[MODEL] 로드 중...")
    processor, base_model = load_base_model(device)
    vision_model = base_model.vision_model.to(device)
    for p in vision_model.parameters():
        p.requires_grad = False
    vision_model.eval()

    anchor_raw = compute_text_anchors(base_model, processor, device)
    print(f"[MODEL] text anchor 계산 완료 ({N_CLASSES}-class)")

    raw_feat_cache = build_raw_feat_cache(vision_model, processor, all_eps, device)
    del vision_model, base_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    image_proj = nn.Linear(VIS_DIM, PROJ_DIM).to(device)
    text_proj  = nn.Linear(LM_DIM,  PROJ_DIM).to(device)

    counts = np.array([frame_label_counts[d] for d in ANCHOR_ORDER], dtype=float)
    weights = (counts.sum() / (N_CLASSES * counts))
    weights = weights / weights.sum() * N_CLASSES
    class_weight = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    print(f"[LOSS] class weight: " + " ".join(f"{d}={w:.2f}" for d, w in zip(ANCHOR_ORDER, weights)))

    optimizer = torch.optim.AdamW(
        list(image_proj.parameters()) + list(text_proj.parameters()), lr=LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_acc, best_state = 0.0, None
    history = []

    print(f"\n{'epoch':>6} {'train_loss':>11} {'val_loss':>9} {'val_acc':>9} {'best':>9}")
    print("-" * 50)

    for epoch in range(1, EPOCHS + 1):
        image_proj.train(); text_proj.train()
        np.random.shuffle(tr_eps)

        batch_feats, batch_labels = [], []
        train_losses = []

        def step(feats, labels):
            x = torch.stack(feats, dim=0)
            y = torch.tensor(labels, dtype=torch.long, device=device)
            proj_x = F.normalize(image_proj(x), dim=-1)
            ap = F.normalize(text_proj(anchor_raw), dim=-1)
            logits = (proj_x @ ap.T) / TEMPERATURE
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            return loss.item()

        for ep in tr_eps:
            for fr in ep["frames"]:
                key = (ep["episode"], fr["frame_idx"])
                if key not in raw_feat_cache:
                    continue
                batch_feats.append(get_cached_feat(raw_feat_cache, ep["episode"], fr["frame_idx"], device))
                batch_labels.append(DIR_IDX[fr["label"]])
                if len(batch_labels) >= BATCH_SIZE:
                    train_losses.append(step(batch_feats, batch_labels))
                    batch_feats, batch_labels = [], []
        if batch_feats:
            train_losses.append(step(batch_feats, batch_labels))

        scheduler.step()
        with torch.no_grad():
            anchor_proj = F.normalize(text_proj(anchor_raw), dim=-1)
        acc, val_loss, confusion = evaluate(raw_feat_cache, image_proj, anchor_proj, val_eps, device, criterion)
        train_loss = float(np.mean(train_losses))

        if acc > best_acc:
            best_acc = acc
            best_state = {
                "image_proj": {k: v.cpu().clone() for k, v in image_proj.state_dict().items()},
                "text_proj":  {k: v.cpu().clone() for k, v in text_proj.state_dict().items()},
                "anchor_raw": anchor_raw.cpu().clone(),
                "val_acc": best_acc, "dir_idx": DIR_IDX,
            }

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": acc, "best_val_acc": best_acc})
        print(f"{epoch:>6}  {train_loss:>11.4f}  {val_loss:>9.4f}  {acc:>8.4f}  {best_acc:>8.4f}")

    print(f"\n{'='*50}\n  Stage 1 v3 재학습(fastloss) 완료 — val_acc: {best_acc:.4f}\n{'='*50}")

    if best_state is None:
        print("[ERROR] val_acc가 한 번도 개선되지 않음 — 저장 중단")
        return

    CURVE_OUT.parent.mkdir(parents=True, exist_ok=True)
    CURVE_OUT.write_text(json.dumps({
        "script": "train_stage1_v3_5cls_owl_fastloss.py",
        "final_best_val_acc": best_acc,
        "history": history,
    }, indent=2, ensure_ascii=False))
    print(f"[SAVE] {CURVE_OUT}")

    ckpt_path = OUT_DIR / "stage1_v3_5cls_owl_projs_fastloss.pt"
    torch.save(best_state, str(ckpt_path))
    print(f"[SAVE] {ckpt_path}")
    print(f"소요: {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()

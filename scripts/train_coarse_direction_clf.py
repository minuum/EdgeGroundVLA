#!/usr/bin/env python3
"""
Train a coarse direction classifier (left/center/right) for basket localization.

Uses frozen Kosmos-2 vision features (mean-pooled, 1024-dim) as input.

Data sources:
  --mini (default): bbox_truth_mini.json — 72 manually annotated frames
  --full:           bbox_truth_mini + bbox_dataset_full.json (1256 has-bbox frames)
                    + horizontal-flip augmentation to balance RIGHT class

Evaluation:
  mini mode:  leave-one-out cross-validation (no val set, only 72 frames)
  full mode:  stratified 80/20 episode-split (train on 80% of episodes)

Output: runs/v5_nav/mlp/step1/coarse_direction_clf.pt

Usage:
    python3 scripts/train_coarse_direction_clf.py           # mini, LOO eval
    python3 scripts/train_coarse_direction_clf.py --full    # expanded, val-split eval
"""
import argparse
import json
import random
import sys
import os
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

BBOX_TRUTH_PATH  = ROOT / "docs" / "v5" / "bbox_truth_mini.json"
BBOX_FULL_PATH   = ROOT / "docs" / "v5" / "bbox_nav_step1" / "bbox_dataset_full.json"
OUTPUT_PATH      = ROOT / "runs" / "v5_nav" / "mlp" / "step1" / "coarse_direction_clf.pt"
GROUNDING_MODEL  = ROOT / ".vlms" / "kosmos-2-patch14-224"

_DATA_PATH_CANDIDATES = [
    Path("/home/billy/25-1kp/MoNaVLA/ROS_action/mobile_vla_dataset_v5"),
    Path("/home/minum/minum/26CS/MoNa-pi/mobile_vla_dataset_v5"),
    ROOT / "ROS_action" / "mobile_vla_dataset_v5",
]

LABEL_MAP  = {"left": 0, "center": 1, "right": 2}
LABEL_CX   = {0: 0.25, 1: 0.5, 2: 0.75}
LABEL_NAME = {0: "LEFT", 1: "CENTER", 2: "RIGHT"}

# cx thresholds for direction from bbox_dataset_full
CX_LEFT_MAX   = 0.35
CX_RIGHT_MIN  = 0.65


def resolve_data_dir() -> Path:
    override = os.getenv("VLA_PROXY_DATA_DIR")
    if override:
        return Path(override)
    for cand in _DATA_PATH_CANDIDATES:
        if cand.exists() and any(cand.glob("episode_*.h5")):
            return cand
    return _DATA_PATH_CANDIDATES[-1]


def load_frame(data_dir: Path, episode: str, frame_idx: int) -> np.ndarray:
    matches = list(data_dir.glob(f"{episode}.h5"))
    if not matches:
        raise FileNotFoundError(f"H5 not found: {episode}")
    with h5py.File(matches[0], "r") as f:
        if "observations" in f and "images" in f["observations"]:
            return f["observations"]["images"][frame_idx].astype(np.uint8)
        return f["images"][frame_idx].astype(np.uint8)


def flip_label(label: int) -> int:
    return {0: 2, 1: 1, 2: 0}[label]


def build_mini_samples(data_dir: Path) -> list[dict]:
    """Load bbox_truth_mini.json — manually annotated, gold quality."""
    with open(BBOX_TRUTH_PATH) as f:
        truth = json.load(f)
    samples = []
    skipped = 0
    for ann in truth["annotations"]:
        cp = ann.get("coarse_position", "").lower()
        if cp not in LABEL_MAP:
            skipped += 1
            continue
        try:
            img = load_frame(data_dir, ann["episode"], ann["frame_idx"])
        except FileNotFoundError:
            skipped += 1
            continue
        samples.append({
            "image": img,
            "label": LABEL_MAP[cp],
            "episode": ann["episode"],
            "source": "gold",
            "flipped": False,
        })
    if skipped:
        print(f"  mini: skipped {skipped} frames")
    return samples


def build_full_samples(data_dir: Path, gold_episodes: set[str],
                       max_per_class: int = 300) -> list[dict]:
    """Load bbox_dataset_full.json has-bbox frames (silver quality).

    Applies horizontal-flip augmentation to balance RIGHT class:
      - each LEFT frame also produces a flipped RIGHT frame (and vice-versa)
    Caps each class at max_per_class before augmentation.
    Skips episodes already in gold set (to avoid leakage in val-split mode).
    """
    with open(BBOX_FULL_PATH) as f:
        full_data = json.load(f)

    # Collect one sample per (episode, frame) with a clear direction
    by_class: dict[int, list[dict]] = defaultdict(list)
    for ep_entry in full_data:
        ep_name = ep_entry["episode"]
        if ep_name in gold_episodes:
            continue
        for fr in ep_entry["frames"]:
            if not fr.get("has_bbox"):
                continue
            cx = float(fr["cx"])
            if cx < CX_LEFT_MAX:
                label = 0
            elif cx > CX_RIGHT_MIN:
                label = 2
            else:
                label = 1
            by_class[label].append({
                "episode": ep_name,
                "frame_idx": fr["frame_idx"],
                "label": label,
            })

    print(f"  full (before aug): "
          f"LEFT={len(by_class[0])}, CENTER={len(by_class[1])}, RIGHT={len(by_class[2])}")

    # Subsample to max_per_class (keep diversity by capping, not filtering by ep)
    selected: list[dict] = []
    for lbl, entries in by_class.items():
        random.shuffle(entries)
        selected.extend(entries[:max_per_class])

    # Augment: LEFT ↔ RIGHT via horizontal flip to balance
    flip_pairs = []
    left_pool  = [s for s in selected if s["label"] == 0]
    right_pool = [s for s in selected if s["label"] == 2]
    center_pool= [s for s in selected if s["label"] == 1]

    right_deficit = len(left_pool) - len(right_pool)
    left_deficit  = len(right_pool) - len(left_pool)

    if right_deficit > 0:
        # Flip some LEFT frames to create synthetic RIGHT
        donors = (left_pool * ((right_deficit // len(left_pool)) + 2))[:right_deficit]
        for d in donors:
            flip_pairs.append({**d, "label": 2, "flipped": True})
    elif left_deficit > 0:
        donors = (right_pool * ((left_deficit // max(len(right_pool), 1)) + 2))[:left_deficit]
        for d in donors:
            flip_pairs.append({**d, "label": 0, "flipped": True})

    all_entries = selected + flip_pairs
    print(f"  full (after aug):  "
          f"LEFT={sum(1 for s in all_entries if s['label']==0 and not s.get('flipped'))}"
          f"+{sum(1 for s in all_entries if s['label']==0 and s.get('flipped'))}flip, "
          f"CENTER={sum(1 for s in all_entries if s['label']==1)}, "
          f"RIGHT={sum(1 for s in all_entries if s['label']==2 and not s.get('flipped'))}"
          f"+{sum(1 for s in all_entries if s['label']==2 and s.get('flipped'))}flip")

    # Load images
    samples = []
    skipped = 0
    for entry in all_entries:
        try:
            img = load_frame(data_dir, entry["episode"], entry["frame_idx"])
        except (FileNotFoundError, Exception):
            skipped += 1
            continue
        if entry.get("flipped"):
            img = np.fliplr(img).copy()
        samples.append({
            "image": img,
            "label": entry["label"],
            "episode": entry["episode"],
            "source": "silver",
            "flipped": bool(entry.get("flipped")),
        })
    if skipped:
        print(f"  full: skipped {skipped} frames")
    return samples


@torch.no_grad()
def extract_features(
    vision_model: nn.Module,
    processor: AutoProcessor,
    images: list[np.ndarray],
    device: torch.device,
    batch_size: int = 8,
) -> torch.Tensor:
    """Mean-pool Kosmos-2 vision features — (N, 1024)."""
    all_feats = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        pil_imgs = [Image.fromarray(img).convert("RGB") for img in batch]
        inputs = processor(
            text=["<grounding>"] * len(pil_imgs),
            images=pil_imgs,
            return_tensors="pt",
            padding=True,
        )
        pv = inputs["pixel_values"].to(device)
        if device.type == "cuda":
            pv = pv.half()
        vo = vision_model(pixel_values=pv)
        feats = vo.last_hidden_state.mean(dim=1).float()
        all_feats.append(feats.cpu())
        print(f"  extracted {min(i + batch_size, len(images))}/{len(images)}", end="\r")
    print()
    return torch.cat(all_feats, dim=0)


def train_linear(
    features: torch.Tensor,
    labels: torch.Tensor,
    epochs: int,
    lr: float,
    verbose: bool = True,
) -> tuple[nn.Linear, torch.Tensor, torch.Tensor]:
    counts = torch.bincount(labels, minlength=3).float()
    weights = (1.0 / counts.clamp(min=1))

    mean = features.mean(dim=0)
    std  = features.std(dim=0).clamp(min=1e-6)
    feats_norm = (features - mean) / std

    clf = nn.Linear(features.shape[1], 3)
    opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss(weight=weights)

    clf.train()
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        loss = crit(clf(feats_norm), labels)
        loss.backward()
        opt.step()
        if verbose and (ep % 50 == 0 or ep == epochs):
            preds = clf(feats_norm).argmax(dim=1)
            acc = (preds == labels).float().mean().item()
            print(f"  epoch {ep:4d}/{epochs}  loss={loss.item():.4f}  acc={acc:.3f}")

    return clf, mean, std


def evaluate_set(clf: nn.Linear, features: torch.Tensor,
                 labels: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
                 tag: str = ""):
    clf.eval()
    with torch.no_grad():
        preds = clf((features - mean) / std).argmax(dim=1)
    overall = (preds == labels).float().mean().item()
    print(f"\n{tag} accuracy: {overall:.3f}  (n={len(labels)})")
    for c in range(3):
        m = labels == c
        if m.sum():
            acc = (preds[m] == labels[m]).float().mean().item()
            print(f"  {LABEL_NAME[c]:8s}: {acc:.3f}  ({m.sum().item()} samples)")
    return overall


def loo_accuracy(features: torch.Tensor, labels: torch.Tensor,
                 epochs: int, lr: float) -> float:
    n = len(features)
    preds = []
    for i in range(n):
        idx = [j for j in range(n) if j != i]
        Xt, yt = features[idx], labels[idx]
        clf, mean, std = train_linear(Xt, yt, epochs, lr, verbose=False)
        clf.eval()
        with torch.no_grad():
            xi = (features[i : i + 1] - mean) / std
            preds.append(clf(xi).argmax(dim=-1).item())
        if (i + 1) % 10 == 0:
            print(f"  LOO {i + 1}/{n}", end="\r")
    print()
    preds_t = torch.tensor(preds)
    overall = (preds_t == labels).float().mean().item()
    print(f"\nLOO accuracy: {overall:.3f}  (n={n})")
    for c in range(3):
        m = labels == c
        if m.sum():
            acc = (preds_t[m] == labels[m]).float().mean().item()
            print(f"  {LABEL_NAME[c]:8s}: {acc:.3f}  ({m.sum().item()} samples)")
    return overall


def episode_split(samples: list[dict], val_ratio: float = 0.2, seed: int = 42):
    """80/20 split by episode — prevents same-episode leakage between train/val."""
    eps = list({s["episode"] for s in samples})
    rng = random.Random(seed)
    rng.shuffle(eps)
    n_val = max(1, int(len(eps) * val_ratio))
    val_eps = set(eps[:n_val])
    train = [s for s in samples if s["episode"] not in val_eps]
    val   = [s for s in samples if s["episode"] in val_eps]
    return train, val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full",    action="store_true", help="Use bbox_dataset_full + augmentation")
    parser.add_argument("--epochs",  type=int,   default=300)
    parser.add_argument("--lr",      type=float, default=1e-3)
    parser.add_argument("--max_per_class", type=int, default=300,
                        help="Cap full-dataset samples per class before augmentation")
    parser.add_argument("--seed",    type=int,   default=42)
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    device = torch.device(args.device)
    data_dir = resolve_data_dir()
    print(f"Data dir: {data_dir}")

    # ── Build sample list ──────────────────────────────────────────────────
    print("\nLoading mini (gold) samples...")
    mini_samples = build_mini_samples(data_dir)
    print(f"  mini: {len(mini_samples)} frames  "
          f"L={sum(1 for s in mini_samples if s['label']==0)} "
          f"C={sum(1 for s in mini_samples if s['label']==1)} "
          f"R={sum(1 for s in mini_samples if s['label']==2)}")

    if args.full:
        gold_eps = {s["episode"] for s in mini_samples}
        print("\nLoading full (silver) samples...")
        full_samples = build_full_samples(data_dir, gold_eps, args.max_per_class)
    else:
        full_samples = []

    all_samples = mini_samples + full_samples
    total_dist = Counter(s["label"] for s in all_samples)
    print(f"\nTotal: {len(all_samples)} frames  "
          f"LEFT={total_dist[0]}, CENTER={total_dist[1]}, RIGHT={total_dist[2]}")

    # ── Load Kosmos-2 ──────────────────────────────────────────────────────
    print(f"\nLoading Kosmos-2 from {GROUNDING_MODEL} ...")
    processor  = AutoProcessor.from_pretrained(str(GROUNDING_MODEL))
    full_model = AutoModelForVision2Seq.from_pretrained(
        str(GROUNDING_MODEL),
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    ).to(device).eval()
    vision_model = full_model.vision_model
    print("Model loaded.\n")

    # ── Extract features ───────────────────────────────────────────────────
    if args.full:
        # Episode-split mode: extract features for train and val separately
        train_samples, val_samples = episode_split(all_samples, seed=args.seed)
        print(f"Episode split: {len(train_samples)} train / {len(val_samples)} val")

        print("Extracting train features...")
        train_feats  = extract_features(vision_model, processor,
                                        [s["image"] for s in train_samples], device)
        train_labels = torch.tensor([s["label"] for s in train_samples], dtype=torch.long)

        print("Extracting val features...")
        val_feats  = extract_features(vision_model, processor,
                                      [s["image"] for s in val_samples], device)
        val_labels = torch.tensor([s["label"] for s in val_samples], dtype=torch.long)

        del full_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(f"\nTraining on {len(train_labels)} frames ...")
        clf, mean, std = train_linear(train_feats, train_labels, args.epochs, args.lr)

        evaluate_set(clf, train_feats, train_labels, mean, std, "Train")
        evaluate_set(clf, val_feats, val_labels, mean, std, "Val")

        # Re-train on full data for the saved model
        print("\nRe-training on full dataset for final model...")
        all_feats  = torch.cat([train_feats, val_feats], dim=0)
        all_labels = torch.cat([train_labels, val_labels], dim=0)
        clf, mean, std = train_linear(all_feats, all_labels, args.epochs, args.lr)
        n_samples = len(all_samples)

    else:
        # Mini mode: extract all features, then LOO
        print("Extracting features...")
        all_feats  = extract_features(vision_model, processor,
                                      [s["image"] for s in all_samples], device)
        all_labels = torch.tensor([s["label"] for s in all_samples], dtype=torch.long)

        del full_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(f"\nLeave-one-out cross-validation ({len(all_labels)} frames) ...")
        loo_accuracy(all_feats, all_labels, args.epochs, args.lr)

        print(f"\nTraining final model on all {len(all_labels)} frames...")
        clf, mean, std = train_linear(all_feats, all_labels, args.epochs, args.lr)
        n_samples = len(all_samples)

    # ── Save ───────────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model":        clf.state_dict(),
            "mean":         mean,
            "std":          std,
            "label_map":    LABEL_MAP,
            "label_cx":     LABEL_CX,
            "n_samples":    n_samples,
            "feature_dim":  all_feats.shape[1] if not args.full else train_feats.shape[1],
            "mode":         "full" if args.full else "mini",
        },
        OUTPUT_PATH,
    )
    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

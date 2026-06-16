#!/usr/bin/env python3
"""
Exp46: Improved Step2 MLP
  A. 데이터 확장: 45 → 150 에피소드 (전체 V5)
  B. VLM vision 특징: 16×16 grayscale → Kosmos-2 vision encoder mean-pool (1024-dim)
  C. 히스토리 window: 3 → 8

Usage:
  python3 scripts/train_v5_exp46_improved_step2.py
  python3 scripts/train_v5_exp46_improved_step2.py --skip_grounding  # grounding 재사용
  python3 scripts/train_v5_exp46_improved_step2.py --skip_grounding --skip_vision  # 둘 다 재사용
"""
import argparse, json, sys, gc
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HF_KOSMOS_PATH = ROOT / ".vlms" / "kosmos-2-patch14-224"
DATA_DIR       = ROOT / "ROS_action" / "mobile_vla_dataset_v5"
OUT_DIR        = ROOT / "docs" / "v5" / "bbox_nav_exp46"
BBOX_CACHE     = OUT_DIR / "bbox_dataset_full.json"
VIS_CACHE      = OUT_DIR / "vision_features.npz"
MLP_DIR        = ROOT / "runs" / "v5_nav" / "mlp" / "exp46"

OUT_DIR.mkdir(parents=True, exist_ok=True)
MLP_DIR.mkdir(parents=True, exist_ok=True)

PATH_TYPES = [
    "center_straight","center_left","center_right",
    "left_straight","left_left","left_right",
    "right_straight","right_left","right_right",
]
CLASS_NAMES = ["STOP","FORWARD","LEFT","RIGHT","FWD+L","FWD+R","ROT_L","ROT_R"]
NUM_CLASSES  = 8
WINDOW       = 8    # C: 3 → 8
VIS_DIM      = 1024 # B: Kosmos-2 vision encoder mean-pool dim
GROUNDING_PROMPT = "<grounding>The gray basket is at"
MAX_NEW_TOKENS   = 48


# ──────────────────────────────────────────────
# 공통 유틸
# ──────────────────────────────────────────────

def gt_action_class(lx, ly, az):
    is_x = abs(lx) > 0.3
    is_y = abs(ly) > 0.3
    if not is_x and not is_y:
        if az > 0.1: return 6
        if az < -0.1: return 7
        return 0
    if lx > 0.3:
        if ly > 0.3: return 4
        if ly < -0.3: return 5
        return 1
    if abs(lx) < 0.3:
        if ly > 0.3: return 2
        if ly < -0.3: return 3
    return 0


def parse_basket_bbox(caption, entities):
    kw = ("basket", "gray box", "container", "gray")
    cands = []
    for ent_name, _span, boxes in entities:
        for box in boxes:
            x1, y1, x2, y2 = box
            area = (x2-x1)*(y2-y1)
            if area > 0.85: continue
            cands.append({"entity": ent_name,
                           "cx": (x1+x2)/2, "cy": (y1+y2)/2, "area": area,
                           "is_basket": any(k in ent_name.lower() for k in kw)})
    matched = [b for b in cands if b["is_basket"]]
    if matched: return matched[0]
    cap_low = caption.lower()
    if "far left"  in cap_low: return {"cx":0.10,"cy":0.5,"area":0.05,"entity":"caption:far_left"}
    if "far right" in cap_low: return {"cx":0.90,"cy":0.5,"area":0.05,"entity":"caption:far_right"}
    if "left"  in cap_low and "right" not in cap_low: return {"cx":0.25,"cy":0.5,"area":0.05,"entity":"caption:left"}
    if "right" in cap_low and "left"  not in cap_low: return {"cx":0.75,"cy":0.5,"area":0.05,"entity":"caption:right"}
    if "center" in cap_low: return {"cx":0.5,"cy":0.5,"area":0.05,"entity":"caption:center"}
    if cands: return cands[0]
    return None


def load_episode_images(ep_path):
    with h5py.File(ep_path, "r") as f:
        if "observations" in f and "images" in f["observations"]:
            return f["observations"]["images"][:]
        return f["images"][:]


def load_episode_actions(ep_path):
    with h5py.File(ep_path, "r") as f:
        return f["actions"][:]


# ──────────────────────────────────────────────
# A. BBox grounding — 전체 150 에피소드
# ──────────────────────────────────────────────

def extract_bbox_dataset_full():
    from transformers import AutoProcessor, AutoModelForImageTextToText

    print("\n[A] BBox grounding — 전체 V5 에피소드 (150개)")
    processor = AutoProcessor.from_pretrained(str(HF_KOSMOS_PATH))
    model = AutoModelForImageTextToText.from_pretrained(
        str(HF_KOSMOS_PATH), torch_dtype=torch.float16
    ).cuda().eval()

    dataset = []
    total_eps = 0
    for pt in PATH_TYPES:
        eps = sorted(DATA_DIR.glob(f"episode_*target_{pt}_path*.h5"))
        print(f"  {pt}: {len(eps)} episodes")
        for ep in eps:
            imgs    = load_episode_images(ep)
            actions = load_episode_actions(ep)
            frames_data = []
            for fi in range(len(imgs)):
                gt_cls = gt_action_class(*actions[fi])
                pil = Image.fromarray(imgs[fi].astype(np.uint8)).convert("RGB")
                inputs = processor(text=GROUNDING_PROMPT, images=pil, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                pv = inputs["pixel_values"].to(torch.float16)
                with torch.no_grad():
                    out = model.generate(
                        pixel_values=pv,
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        image_embeds=None,
                        image_embeds_position_mask=inputs.get("image_embeds_position_mask"),
                        max_new_tokens=MAX_NEW_TOKENS,
                    )
                new_ids = out[:, inputs["input_ids"].shape[1]:]
                raw = processor.batch_decode(new_ids, skip_special_tokens=False)[0]
                caption, entities = processor.post_process_generation(raw)
                bbox = parse_basket_bbox(caption, entities)
                frames_data.append({
                    "frame_idx": fi,
                    "gt_class":  gt_cls,
                    "cx":        float(bbox["cx"])   if bbox else 0.5,
                    "cy":        float(bbox["cy"])   if bbox else 0.5,
                    "area":      float(bbox["area"]) if bbox else 0.0,
                    "has_bbox":  bbox is not None,
                })
            dataset.append({"path_type": pt, "episode": str(ep), "frames": frames_data})
            total_eps += 1
            print(f"    [{total_eps}/150] {ep.name}  frames={len(frames_data)}", flush=True)

    del model; gc.collect(); torch.cuda.empty_cache()
    BBOX_CACHE.write_text(json.dumps(dataset, indent=2))
    print(f"  → 저장: {BBOX_CACHE} ({total_eps} episodes)")
    return dataset


# ──────────────────────────────────────────────
# B. Kosmos-2 vision encoder — mean-pool 특징 (1024-dim)
# ──────────────────────────────────────────────

def extract_vision_features(dataset):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    print("\n[B] Kosmos-2 vision encoder 특징 추출 (mean-pool 1024-dim)")
    proc  = AutoProcessor.from_pretrained(str(HF_KOSMOS_PATH))
    model = AutoModelForVision2Seq.from_pretrained(
        str(HF_KOSMOS_PATH), torch_dtype=torch.float16
    ).cuda().eval()

    # episode → frame → 1024-dim
    vis_cache = {}  # ep_path (str) → np.ndarray (N_frames, 1024)

    for i, ep_data in enumerate(dataset):
        ep_path = ep_data["episode"]
        imgs = load_episode_images(ep_path)
        feats = []
        for fi in range(len(imgs)):
            pil = Image.fromarray(imgs[fi].astype(np.uint8)).convert("RGB")
            inputs = proc(text="<grounding>", images=pil, return_tensors="pt")
            pv = inputs["pixel_values"].to(model.device, dtype=torch.float16)
            with torch.no_grad():
                vis_out = model.vision_model(pv)
                # (1, N_patches, 1024) → mean over patches → (1024,)
                feat = vis_out.last_hidden_state[0].mean(0).float().cpu().numpy()
            feats.append(feat)
        vis_cache[ep_path] = np.stack(feats)  # (N_frames, 1024)
        print(f"  [{i+1}/{len(dataset)}] {Path(ep_path).name}  shape={vis_cache[ep_path].shape}", flush=True)

    del model; gc.collect(); torch.cuda.empty_cache()

    # npz로 저장 (key=episode_path, value=features)
    np.savez_compressed(str(VIS_CACHE), **{
        f"ep_{i}": v for i, v in enumerate(vis_cache.values())
    })
    # episode 순서 index도 저장
    index = {ep: i for i, ep in enumerate(vis_cache.keys())}
    (OUT_DIR / "vision_features_index.json").write_text(json.dumps(index, indent=2))
    print(f"  → 저장: {VIS_CACHE}")
    return vis_cache


# ──────────────────────────────────────────────
# C. 데이터셋 빌드 — window=8, bbox + vision
# ──────────────────────────────────────────────

def build_dataset(bbox_data, vis_cache):
    print(f"\n[C] 윈도우 빌드 (WINDOW={WINDOW}) ...")
    X, y, path_labels = [], [], []

    for ep_data in bbox_data:
        ep_path = ep_data["episode"]
        frames  = ep_data["frames"]
        vis_feats = vis_cache.get(ep_path)
        if vis_feats is None:
            print(f"  WARNING: vision features missing for {Path(ep_path).name}, skip")
            continue

        for t in range(len(frames)):
            # bbox history: WINDOW frames × 4
            bbox_feat = []
            for k in range(WINDOW):
                idx = max(0, t - (WINDOW - 1 - k))
                fr  = frames[idx]
                bbox_feat.extend([fr["cx"], fr["cy"], fr["area"], float(fr["has_bbox"])])

            # vision feature: current frame mean-pool (1024-dim)
            vis_feat = vis_feats[t]  # (1024,)

            feat = np.concatenate([np.array(bbox_feat, dtype=np.float32), vis_feat])
            X.append(feat)
            y.append(frames[t]["gt_class"])
            path_labels.append(ep_data["path_type"])

    X = np.stack(X)
    y = np.array(y)
    print(f"  총 샘플: {len(X)}, 입력 dim: {X.shape[1]}")
    return X, y, path_labels


# ──────────────────────────────────────────────
# MLP 정의
# ──────────────────────────────────────────────

def build_mlp(d_in):
    return nn.Sequential(
        nn.Linear(d_in, 512), nn.ReLU(), nn.Dropout(0.25),
        nn.Linear(512, 256),  nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 128),  nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(128, 64),   nn.ReLU(),
        nn.Linear(64, NUM_CLASSES),
    )


def train_mlp(X_tr, y_tr, X_te, y_te, d_in, epochs=100, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = build_mlp(d_in).to(device)

    class_counts = np.bincount(y_tr, minlength=NUM_CLASSES).astype(float)
    weights = np.where(class_counts > 0, 1.0 / (class_counts + 1e-6), 0.0)
    weights /= weights.sum() / NUM_CLASSES
    crit = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    Xtr = torch.tensor(X_tr, dtype=torch.float32)
    ytr = torch.tensor(y_tr, dtype=torch.long)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=256, shuffle=True)

    best_acc, best_state = 0.0, None
    for ep in range(1, epochs + 1):
        net.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(net(xb), yb).backward()
            opt.step()
        sched.step()

        if ep % 10 == 0 or ep == epochs:
            net.eval()
            with torch.no_grad():
                Xte = torch.tensor(X_te, dtype=torch.float32).to(device)
                yte = torch.tensor(y_te, dtype=torch.long).to(device)
                preds = net(Xte).argmax(1)
                acc = (preds == yte).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            print(f"  epoch {ep:>3}  val_acc={acc:.3f}  best={best_acc:.3f}")

    net.load_state_dict(best_state)
    return net, best_acc


def evaluate(net, X_te, y_te):
    device = next(net.parameters()).device
    net.eval()
    with torch.no_grad():
        Xte = torch.tensor(X_te, dtype=torch.float32).to(device)
        preds = net(Xte).argmax(1).cpu().numpy()

    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for g, p in zip(y_te, preds):
        confusion[g, p] += 1

    print("\n=== Confusion Matrix ===")
    print("         " + "".join(f"{n:>8}" for n in CLASS_NAMES))
    for r in range(NUM_CLASSES):
        total = confusion[r].sum()
        if total == 0: continue
        acc = confusion[r, r] / total * 100
        print(f"{CLASS_NAMES[r]:<9}" + "".join(f"{v:>8}" for v in confusion[r]) + f"  {acc:.0f}%")

    overall = confusion.diagonal().sum() / confusion.sum()
    print(f"\n전체 정확도: {overall:.1%}")
    return confusion, overall


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_grounding", action="store_true", help="기존 bbox_dataset_full.json 재사용")
    ap.add_argument("--skip_vision",    action="store_true", help="기존 vision_features.npz 재사용")
    ap.add_argument("--epochs", type=int, default=150)
    args = ap.parse_args()

    # A. BBox grounding
    if args.skip_grounding and BBOX_CACHE.exists():
        print(f"[A] bbox cache 재사용: {BBOX_CACHE}")
        bbox_data = json.loads(BBOX_CACHE.read_text())
    else:
        bbox_data = extract_bbox_dataset_full()

    print(f"  에피소드 수: {len(bbox_data)}")

    # B. Vision features
    vis_index_path = OUT_DIR / "vision_features_index.json"
    if args.skip_vision and VIS_CACHE.exists() and vis_index_path.exists():
        print(f"\n[B] vision cache 재사용: {VIS_CACHE}")
        index = json.loads(vis_index_path.read_text())
        npz   = np.load(str(VIS_CACHE))
        vis_cache = {ep: npz[f"ep_{i}"] for ep, i in index.items()}
    else:
        vis_cache = extract_vision_features(bbox_data)

    # C. 데이터셋 빌드
    X, y, path_labels = build_dataset(bbox_data, vis_cache)

    # Train/val split (80/20 stratified by path_type)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(sss.split(X, path_labels))
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]
    print(f"  train: {len(X_tr)}, val: {len(X_te)}")

    # 학습
    print(f"\n[학습] MLP {X.shape[1]}→512→256→128→64→{NUM_CLASSES}, epochs={args.epochs}")
    net, best_acc = train_mlp(X_tr, y_tr, X_te, y_te, X.shape[1], epochs=args.epochs)

    # 평가
    confusion, overall = evaluate(net, X_te, y_te)

    # 저장
    ckpt_path = MLP_DIR / "exp46_mlp.pt"
    torch.save({
        "model_state_dict": net.state_dict(),
        "d_in":    X.shape[1],
        "window":  WINDOW,
        "vis_dim": VIS_DIM,
        "overall_acc": overall,
        "confusion": confusion.tolist(),
    }, str(ckpt_path))

    summary = {
        "overall_acc": float(overall),
        "best_val_acc": float(best_acc),
        "n_train": int(len(X_tr)),
        "n_val":   int(len(X_te)),
        "n_episodes": len(bbox_data),
        "window": WINDOW,
        "vis_dim": VIS_DIM,
        "d_in": int(X.shape[1]),
        "confusion": confusion.tolist(),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n✅ 완료! 정확도={overall:.1%}  ckpt={ckpt_path}")


if __name__ == "__main__":
    main()

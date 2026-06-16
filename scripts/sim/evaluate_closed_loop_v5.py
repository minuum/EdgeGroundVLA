#!/usr/bin/env python3
"""
V5 Closed-Loop Simulation Evaluator (Phase 1: Offline Replay)

두 모델 비교:
  --model exp11   : Kosmos-2 policy model (LSTM action head)
  --model step2   : BBox+Image MLP (decomposition approach)
  --model exp17   : Exp17 (33/33/34 balanced, end-to-end)
  --model exp18   : Exp18 (VLM + LoRA + text embedding fusion)

Usage:
  # Exp11
  python3 scripts/sim/evaluate_closed_loop_v5.py --model exp11 \
    --config configs/mobile_vla_v5_exp11_google_robot_8cls.json \
    --ckpt runs/v5_nav/kosmos/.../epoch=14.ckpt

  # Step 2 (자동으로 MLP 재학습 후 rollout)
  python3 scripts/sim/evaluate_closed_loop_v5.py --model step2

  # Exp19 (Step2 + proxy features)
  python3 scripts/sim/evaluate_closed_loop_v5.py --model exp19
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from scripts.sim.rollout_core import (
    ACTION_VEL, CLASS_NAMES, DT_DEFAULT,
    build_trajectory, continuous_to_class, compute_metrics,
)

DATA_DIR  = ROOT / "ROS_action" / "mobile_vla_dataset_v5"
STEP1_DIR = ROOT / "docs" / "v5" / "bbox_nav_step1"
EXP46_DIR = ROOT / "docs" / "v5" / "bbox_nav_exp46"
OUT_DIR   = ROOT / "docs" / "v5" / "closed_loop_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GOAL_NAV_CKPTS = {
    "exp49": ROOT / "runs" / "v5_nav" / "mlp" / "exp49" / "exp49_mlp.pt",
    "exp50": ROOT / "runs" / "v5_nav" / "mlp" / "exp50" / "exp50_mlp.pt",
    "exp51": ROOT / "runs" / "v5_nav" / "mlp" / "exp51" / "exp51_mlp.pt",
    "exp52": ROOT / "runs" / "v5_nav" / "mlp" / "exp52" / "exp52_mlp.pt",
    "exp53": ROOT / "runs" / "v5_nav" / "mlp" / "exp53" / "exp53_clip_lora.pt",
}
GOAL_NAV_VIS_DIRS = {
    "exp49": ROOT / "docs" / "v5" / "bbox_nav_exp46",   # vision_features.npz (1024-dim)
    "exp50": ROOT / "docs" / "v5" / "bbox_nav_exp46",
    "exp51": ROOT / "docs" / "v5" / "bbox_nav_exp46",
    "exp52": ROOT / "docs" / "v5" / "bbox_nav_exp52",   # lang_vis_features.npz (2048-dim)
    "exp53": ROOT / "docs" / "v5" / "bbox_nav_exp53",   # LoRA-enhanced vision_features.npz
}
GOAL_NAV_VIS_KEYS = {
    "exp49": "vision_features",
    "exp50": "vision_features",
    "exp51": "vision_features",
    "exp52": "lang_vis_features",
    "exp53": "vision_features",
}
GOAL_NAV_WINDOW = 8
GOAL_NAV_GOAL_DIM = 3

PATH_TYPES = [
    "center_straight", "center_left", "center_right",
    "left_straight", "left_left", "left_right",
    "right_straight", "right_left", "right_right",
]
NUM_CLASSES = 8
IMG_SIZE = 16
WINDOW = 3
IMG_SIZE_STEP3 = 32
WINDOW_STEP3 = 8
CONSISTENCY_K = 5
CX_TOL = 0.08
AREA_TOL = 0.08


def load_text_embedding_map():
    dataset_path = ROOT / "docs" / "v5" / "v5_dataset_with_text_embeddings.json"
    if not dataset_path.exists():
        return {}
    data = json.loads(dataset_path.read_text())
    return {
        item["episode"]: np.asarray(item["text_embedding"], dtype=np.float32)
        for item in data
        if "episode" in item and "text_embedding" in item
    }


def recent_bbox_consistency(frames, t, k=CONSISTENCY_K, cx_tol=CX_TOL, area_tol=AREA_TOL):
    start = max(0, t - k + 1)
    tail = frames[start:t + 1]
    valid = [fr for fr in tail if fr["has_bbox"]]
    if not valid:
        return 0.0
    if len(valid) == 1:
        return 1.0
    stable_pairs = 0
    total_pairs = 0
    for a, b in zip(valid[:-1], valid[1:]):
        total_pairs += 1
        if abs(float(b["cx"]) - float(a["cx"])) <= cx_tol and abs(float(b["area"]) - float(a["area"])) <= area_tol:
            stable_pairs += 1
    if total_pairs == 0:
        return 1.0
    return stable_pairs / total_pairs


def build_proxy_features(frames, t):
    cur = frames[t]
    prev = frames[t - 1] if t > 0 else None
    area = float(cur["area"])
    center_error_x = abs(float(cur["cx"]) - 0.5)
    abs_delta_cx = 0.0 if prev is None else abs(float(cur["cx"]) - float(prev["cx"]))
    recent_consistency = recent_bbox_consistency(frames, t)
    return [area, center_error_x, abs_delta_cx, recent_consistency]


# ── Exp11: policy model inference ───────────────────────────────────────────

def load_exp11_model(config_path: str, ckpt_path: str):
    import gc
    import lightning.fabric.plugins.environments.mpi as _mpi_env_mod
    _mpi_env_mod._MPI4PY_AVAILABLE = False

    import robovlms.model.backbone as backbone_mod
    import robovlms.model.policy_head as ph_mod
    import robovlms.train as train_mod
    import robovlms.train.base_trainer as base_trainer_mod

    from robovlm_nav.models.nav_robokosmos import NavRoboKosMos
    from robovlm_nav.models.policy_head.nav_policy_impl import (
        MobileVLAClassificationDecoder as NavClassificationDecoder,
        MobileVLALSTMDecoder as NavLSTMDecoder,
    )
    from robovlm_nav.models.policy_head.hybrid_action_head import HybridActionHead
    from robovlm_nav.trainer.nav_trainer import NavTrainer

    setattr(backbone_mod, "RoboKosMos", NavRoboKosMos)
    setattr(backbone_mod, "RoboVLM-Nav", NavRoboKosMos)
    setattr(ph_mod, "MobileVLAClassificationDecoder", NavClassificationDecoder)
    setattr(ph_mod, "MobileVLALSTMDecoder", NavLSTMDecoder)
    setattr(ph_mod, "NavPolicy", NavClassificationDecoder)
    setattr(ph_mod, "HybridActionHead", HybridActionHead)
    base_trainer_mod.BaseTrainer = NavTrainer
    setattr(train_mod, "NavTrainer", NavTrainer)
    setattr(train_mod, "BaseTrainer", NavTrainer)

    sys.path.insert(0, str(ROOT / "third_party" / "RoboVLMs"))
    from main import load_config, update_configs
    configs = load_config(config_path)

    vlm_path = str(ROOT / ".vlms" / "kosmos-2-patch14-224")
    if isinstance(configs.get("vlm"), dict):
        configs["vlm"]["pretrained_model_name_or_path"] = vlm_path
    if isinstance(configs.get("tokenizer"), dict):
        configs["tokenizer"]["pretrained_model_name_or_path"] = vlm_path

    from robovlms.train.mobile_vla_trainer import MobileVLATrainer
    model_wrapper = MobileVLATrainer(configs)
    train_ds_cfg = configs.get("train_dataset", {})
    eval_image_size = int(configs.get("image_size", 224))
    eval_resize_mode = train_ds_cfg.get("resize_mode", "direct")
    eval_image_mean = configs.get("image_mean", [0.48145466, 0.4578275, 0.40821073])
    eval_image_std = configs.get("image_std", [0.26862954, 0.26130258, 0.27577711])
    model_wrapper._eval_image_size = eval_image_size
    model_wrapper._eval_resize_mode = eval_resize_mode
    model_wrapper._eval_image_mean = eval_image_mean
    model_wrapper._eval_image_std = eval_image_std
    if hasattr(model_wrapper, "model"):
        model_wrapper.model._eval_image_size = eval_image_size
        model_wrapper.model._eval_resize_mode = eval_resize_mode
        model_wrapper.model._eval_image_mean = eval_image_mean
        model_wrapper.model._eval_image_std = eval_image_std

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    full_sd = ckpt.get("model_state_dict", ckpt.get("state_dict", {}))
    filtered = {k: v for k, v in full_sd.items()
                if not any(k.startswith(p) for p in ["train_dataset", "val_dataset"])}
    model_wrapper.load_state_dict(filtered, strict=False)
    del full_sd, ckpt; gc.collect()

    model_wrapper.eval().cuda().half()
    return model_wrapper


def preprocess_eval_frame(img, image_size, resize_mode, mean, std):
    from PIL import Image
    import torchvision.transforms as T

    if resize_mode == "letterbox":
        src_w, src_h = img.size
        scale = min(image_size / src_w, image_size / src_h)
        new_w = max(1, int(round(src_w * scale)))
        new_h = max(1, int(round(src_h * scale)))
        resized = img.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("RGB", (image_size, image_size), (0, 0, 0))
        off_x = (image_size - new_w) // 2
        off_y = (image_size - new_h) // 2
        canvas.paste(resized, (off_x, off_y))
        img = canvas
    else:
        img = img.resize((image_size, image_size), Image.BILINEAR)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])
    return transform(img)


def eval_exp11_episode(ep_path, model, processor, window_size=8, text_embedding_map=None):
    """Run policy model on every frame of one episode; return predicted class list.
    model = model_wrapper.model (RoboKosMos backbone), same as PM eval's forward_action call.
    """
    from PIL import Image

    with h5py.File(ep_path, "r") as f:
        if "observations" in f and "images" in f["observations"]:
            imgs = f["observations"]["images"][:]
        else:
            imgs = f["images"][:]
        expert_actions = f["actions"][:]

    image_size = int(getattr(model, "_eval_image_size", 224))
    resize_mode = getattr(model, "_eval_resize_mode", "direct")
    image_mean = getattr(model, "_eval_image_mean", [0.48145466, 0.4578275, 0.40821073])
    image_std = getattr(model, "_eval_image_std", [0.26862954, 0.26130258, 0.27577711])

    instr = "<grounding>Navigate toward the gray basket"
    lang_tokens = processor.tokenizer(
        instr, return_tensors="pt",
        padding="max_length", max_length=64, truncation=True,
    )
    lang_x = lang_tokens["input_ids"].cuda()
    lang_mask = lang_tokens["attention_mask"].bool().cuda()

    pred_classes = []
    text_embedding = None
    if text_embedding_map is not None:
        emb = text_embedding_map.get(ep_path.stem)
        if emb is not None:
            text_embedding = torch.tensor(emb, dtype=torch.float16, device="cuda").unsqueeze(0)

    for t in range(len(imgs)):
        window_imgs = []
        for k in range(window_size):
            idx = max(0, t - (window_size - 1 - k))
            img = Image.fromarray(imgs[idx].astype(np.uint8)).convert("RGB")
            window_imgs.append(preprocess_eval_frame(img, image_size, resize_mode, image_mean, image_std))
        vision_x = torch.stack(window_imgs).unsqueeze(0).half().cuda()  # (1, ws, 3, H, W)

        out = model.forward_action(
            vision_x=vision_x,
            lang_x=lang_x,
            attention_mask=lang_mask,
            text_embedding=text_embedding,
            vision_gripper=None,
            instr_and_action_ids=None,
            instr_and_action_labels=None,
            instr_and_action_mask=None,
            mode="test",
        )

        if out is None:
            pred_classes.append(1); continue
        if isinstance(out, (tuple, list)):
            logits = out[0]
        else:
            logits = out
        if logits is None:
            pred_classes.append(1); continue
        arr = logits.detach().cpu().float().numpy()
        if arr.ndim == 4:   cls = int(np.argmax(arr[0, 0, 0, :]))
        elif arr.ndim == 3: cls = int(np.argmax(arr[0, 0, :]))
        elif arr.ndim == 2: cls = int(np.argmax(arr[0, :]))
        else:               cls = int(np.argmax(arr))
        pred_classes.append(min(cls, NUM_CLASSES - 1))

    return pred_classes, expert_actions


# ── Step 2: BBox+Image MLP inference ─────────────────────────────────────────

def train_step2_mlp(bbox_dataset, train_eps, window=WINDOW, epochs=220, seed=0):
    import torch.nn as nn
    from PIL import Image
    from collections import defaultdict

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def frame_to_img_feat(ep_stem, frame_idx):
        path = next(DATA_DIR.glob(f"{ep_stem}.h5"))
        with h5py.File(path, "r") as f:
            imgs = f["observations"]["images"][:] if ("observations" in f and "images" in f["observations"]) else f["images"][:]
        frame = imgs[frame_idx]
        img = Image.fromarray(frame.astype(np.uint8)).convert("L").resize((IMG_SIZE, IMG_SIZE))
        return np.asarray(img, dtype=np.float32).reshape(-1) / 255.0

    X, y = [], []
    for ep in train_eps:
        frames = ep["frames"]
        img_feats = [frame_to_img_feat(ep["episode"], f["frame_idx"]) for f in frames]
        for t in range(len(frames)):
            feat = []
            for k in range(window):
                idx = max(0, t - (window - 1 - k))
                f = frames[idx]
                feat.extend([f["cx"], f["cy"], f["area"], float(f["has_bbox"])])
            feat.extend(img_feats[t].tolist())
            X.append(feat)
            y.append(frames[t]["gt_class"])

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    torch.manual_seed(seed)
    d_in = X.shape[1]
    model = torch.nn.Sequential(
        torch.nn.Linear(d_in, 256), torch.nn.ReLU(), torch.nn.Dropout(0.25),
        torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Dropout(0.2),
        torch.nn.Linear(128, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, NUM_CLASSES),
    ).to(DEVICE)

    cls_counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float32)
    cls_counts = np.where(cls_counts == 0, 1.0, cls_counts)
    weights = torch.tensor(1.0 / cls_counts, device=DEVICE)
    weights = weights / weights.sum() * NUM_CLASSES
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    X_t = torch.tensor(X, device=DEVICE)
    y_t = torch.tensor(y, device=DEVICE)
    for ep_i in range(epochs):
        model.train()
        idx = torch.randperm(len(X_t))
        for i in range(0, len(idx), 128):
            b = idx[i:i+128]
            loss = loss_fn(model(X_t[b]), y_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep_i % 55 == 0 or ep_i == epochs - 1:
            model.eval()
            with torch.no_grad():
                acc = (model(X_t).argmax(1) == y_t).float().mean().item()
            print(f"  MLP ep{ep_i:3d}: loss={loss.item():.3f} train_acc={acc:.3f}")
    model.eval()
    return model, DEVICE


def eval_step2_episode(ep_entry, mlp, device, window=WINDOW):
    """Predict action class for each frame in ep_entry using Step 2 MLP."""
    from PIL import Image

    frames = ep_entry["frames"]

    def frame_to_img_feat(frame_idx):
        path = next(DATA_DIR.glob(f"{ep_entry['episode']}.h5"))
        with h5py.File(path, "r") as f:
            imgs = f["observations"]["images"][:] if ("observations" in f and "images" in f["observations"]) else f["images"][:]
        frame = imgs[frame_idx]
        img = Image.fromarray(frame.astype(np.uint8)).convert("L").resize((IMG_SIZE, IMG_SIZE))
        return np.asarray(img, dtype=np.float32).reshape(-1) / 255.0

    img_feats = [frame_to_img_feat(f["frame_idx"]) for f in frames]
    pred_classes = []
    for t in range(len(frames)):
        feat = []
        for k in range(window):
            idx = max(0, t - (window - 1 - k))
            f = frames[idx]
            feat.extend([f["cx"], f["cy"], f["area"], float(f["has_bbox"])])
        feat.extend(img_feats[t].tolist())
        x = torch.tensor([feat], dtype=torch.float32, device=device)
        with torch.no_grad():
            cls = int(mlp(x).argmax(1).item())
        pred_classes.append(min(cls, NUM_CLASSES - 1))

    # Load expert actions from H5
    path = next(DATA_DIR.glob(f"{ep_entry['episode']}.h5"))
    with h5py.File(path, "r") as f:
        expert_actions = f["actions"][:]

    return pred_classes, expert_actions[:len(frames)]


def train_exp19_proxy_mlp(bbox_dataset, train_eps, window=WINDOW, epochs=220, seed=0):
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    def frame_to_img_feat(ep_stem, frame_idx):
        path = next(DATA_DIR.glob(f"{ep_stem}.h5"))
        with h5py.File(path, "r") as f:
            imgs = f["observations"]["images"][:] if ("observations" in f and "images" in f["observations"]) else f["images"][:]
        frame = imgs[frame_idx]
        img = Image.fromarray(frame.astype(np.uint8)).convert("L").resize((IMG_SIZE, IMG_SIZE))
        return np.asarray(img, dtype=np.float32).reshape(-1) / 255.0

    X, y = [], []
    for ep in train_eps:
        frames = ep["frames"]
        img_feats = [frame_to_img_feat(ep["episode"], f["frame_idx"]) for f in frames]
        for t in range(len(frames)):
            feat = []
            for k in range(window):
                idx = max(0, t - (window - 1 - k))
                f = frames[idx]
                feat.extend([f["cx"], f["cy"], f["area"], float(f["has_bbox"])])
            feat.extend(img_feats[t].tolist())
            feat.extend(build_proxy_features(frames, t))
            X.append(feat)
            y.append(frames[t]["gt_class"])

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)

    d_in = X.shape[1]
    model = torch.nn.Sequential(
        torch.nn.Linear(d_in, 256), torch.nn.ReLU(), torch.nn.Dropout(0.25),
        torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Dropout(0.2),
        torch.nn.Linear(128, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, NUM_CLASSES),
    ).to(device)

    cls_counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float32)
    cls_counts = np.where(cls_counts == 0, 1.0, cls_counts)
    weights = torch.tensor(1.0 / cls_counts, device=device)
    weights = weights / weights.sum() * NUM_CLASSES
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    X_t = torch.tensor(X, device=device)
    y_t = torch.tensor(y, device=device)

    best_acc, best_state = 0.0, None
    for ep_i in range(epochs):
        model.train()
        idx = torch.randperm(len(X_t))
        for i in range(0, len(idx), 128):
            b = idx[i:i + 128]
            loss = loss_fn(model(X_t[b]), y_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep_i % 40 == 0 or ep_i == epochs - 1:
            model.eval()
            with torch.no_grad():
                acc = (model(X_t).argmax(1) == y_t).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  Exp19 ep{ep_i:3d}: loss={loss.item():.3f} train_acc={acc:.3f} best={best_acc:.3f}")

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, device


def eval_exp19_episode(ep_entry, mlp, device, window=WINDOW):
    from PIL import Image

    frames = ep_entry["frames"]

    def frame_to_img_feat(frame_idx):
        path = next(DATA_DIR.glob(f"{ep_entry['episode']}.h5"))
        with h5py.File(path, "r") as f:
            imgs = f["observations"]["images"][:] if ("observations" in f and "images" in f["observations"]) else f["images"][:]
        frame = imgs[frame_idx]
        img = Image.fromarray(frame.astype(np.uint8)).convert("L").resize((IMG_SIZE, IMG_SIZE))
        return np.asarray(img, dtype=np.float32).reshape(-1) / 255.0

    img_feats = [frame_to_img_feat(f["frame_idx"]) for f in frames]
    pred_classes = []
    for t in range(len(frames)):
        feat = []
        for k in range(window):
            idx = max(0, t - (window - 1 - k))
            f = frames[idx]
            feat.extend([f["cx"], f["cy"], f["area"], float(f["has_bbox"])])
        feat.extend(img_feats[t].tolist())
        feat.extend(build_proxy_features(frames, t))
        x = torch.tensor([feat], dtype=torch.float32, device=device)
        with torch.no_grad():
            cls = int(mlp(x).argmax(1).item())
        pred_classes.append(min(cls, NUM_CLASSES - 1))

    path = next(DATA_DIR.glob(f"{ep_entry['episode']}.h5"))
    with h5py.File(path, "r") as f:
        expert_actions = f["actions"][:]

    return pred_classes, expert_actions[:len(frames)]


# ── Step 3: Full dataset MLP (WINDOW=8, IMG=32x32) ──────────────────────────

NUM_CLASSES_STEP3 = 7   # STOP 제거: 1=FWD...7=ROT_R → 0-6
STEP3_OFFSET = 1        # train label = orig-1, pred → pred+1

def train_step3_mlp(bbox_dataset, train_eps, window=WINDOW_STEP3, img_size=IMG_SIZE_STEP3,
                    epochs=300, seed=0):
    from PIL import Image
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    def get_img_feat(ep_stem, frame_idx):
        path = next(DATA_DIR.glob(f"{ep_stem}.h5"))
        with h5py.File(path, "r") as f:
            imgs = f["observations"]["images"][:] if ("observations" in f and "images" in f["observations"]) else f["images"][:]
        img = Image.fromarray(imgs[frame_idx].astype(np.uint8)).convert("L").resize((img_size, img_size))
        return np.asarray(img, dtype=np.float32).reshape(-1) / 255.0

    X, y = [], []
    for ep in train_eps:
        frames = ep["frames"]
        img_feats = [get_img_feat(ep["episode"], f["frame_idx"]) for f in frames]
        for t in range(len(frames)):
            feat = []
            for k in range(window):
                idx = max(0, t - (window - 1 - k))
                f = frames[idx]
                feat.extend([f["cx"], f["cy"], f["area"], float(f["has_bbox"])])
            feat.extend(img_feats[t])
            X.append(feat)
            y.append(frames[t]["gt_class"])

    # STOP(0) 제거, label remapping: orig-1
    X_list, y_list = [], []
    for xi, yi in zip(X, y):
        if yi == 0: continue
        X_list.append(xi); y_list.append(yi - STEP3_OFFSET)
    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.int64)

    d_in = X.shape[1]
    model = torch.nn.Sequential(
        torch.nn.Linear(d_in, 512), torch.nn.ReLU(), torch.nn.Dropout(0.25),
        torch.nn.Linear(512, 256), torch.nn.ReLU(), torch.nn.Dropout(0.2),
        torch.nn.Linear(256, 128), torch.nn.ReLU(),
        torch.nn.Linear(128, NUM_CLASSES_STEP3),
    ).to(DEVICE)

    cls_counts = np.bincount(y, minlength=NUM_CLASSES_STEP3).astype(np.float32)
    cls_counts = np.where(cls_counts == 0, 1.0, cls_counts)
    weights = torch.tensor(1.0 / cls_counts, device=DEVICE)
    weights = weights / weights.sum() * NUM_CLASSES_STEP3
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    X_t = torch.tensor(X, device=DEVICE)
    y_t = torch.tensor(y, device=DEVICE)

    best_acc, best_state = 0.0, None
    for ep_i in range(epochs):
        model.train()
        idx = torch.randperm(len(X_t))
        for i in range(0, len(idx), 128):
            b = idx[i:i+128]
            loss = loss_fn(model(X_t[b]), y_t[b])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep_i % 50 == 0 or ep_i == epochs - 1:
            model.eval()
            with torch.no_grad():
                acc = (model(X_t).argmax(1) == y_t).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  MLP ep{ep_i:3d}: loss={loss.item():.3f} train_acc={acc:.3f} best={best_acc:.3f}")

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model, DEVICE, STEP3_OFFSET


def eval_step3_episode(ep_entry, mlp, device, offset=STEP3_OFFSET, window=WINDOW_STEP3, img_size=IMG_SIZE_STEP3):
    from PIL import Image
    frames = ep_entry["frames"]

    path = next(DATA_DIR.glob(f"{ep_entry['episode']}.h5"))
    with h5py.File(path, "r") as f:
        imgs = f["observations"]["images"][:] if ("observations" in f and "images" in f["observations"]) else f["images"][:]
        expert_actions = f["actions"][:]

    img_feats = []
    for fr in frames:
        img = Image.fromarray(imgs[fr["frame_idx"]].astype(np.uint8)).convert("L").resize((img_size, img_size))
        img_feats.append(np.asarray(img, dtype=np.float32).reshape(-1) / 255.0)

    pred_classes = []
    for t in range(len(frames)):
        feat = []
        for k in range(window):
            idx = max(0, t - (window - 1 - k))
            f = frames[idx]
            feat.extend([f["cx"], f["cy"], f["area"], float(f["has_bbox"])])
        feat.extend(img_feats[t])
        x = torch.tensor([feat], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred_classes.append(int(mlp(x).argmax(1).item()) + offset)  # 0-6 → 1-7

    return pred_classes, expert_actions[:len(frames)]


# ── HTML builder ─────────────────────────────────────────────────────────────

def build_html(results_by_model, summary_by_model):
    model_names = list(results_by_model.keys())
    colors = {"exp11": "#60a5fa", "step2": "#22c55e"}

    def color_pm(v, good=0.3, warn=0.15):
        if v >= good: return "#22c55e"
        if v >= warn: return "#fbbf24"
        return "#ef4444"

    # Summary boxes
    boxes = ""
    for name in model_names:
        s = summary_by_model[name]
        col = colors.get(name, "#94a3b8")
        boxes += f"""
    <div class="box">
      <div style="color:#94a3b8; font-size:0.85rem; text-transform:uppercase;">{name}</div>
      <div class="num" style="color:{col};">{s['success_rate']:.1%}</div>
      <div style="color:#64748b;">success rate</div>
      <div style="margin-top:8px; font-size:0.85rem;">
        FPE: {s['mean_fpe']:.2f}m &nbsp; TLD: {s['mean_tld']:.2f}
      </div>
    </div>"""

    # Per-path table
    all_paths = PATH_TYPES
    header_parts = ["<th>Path Type</th>"]
    for n in model_names:
        col = colors.get(n, "#fff")
        header_parts.append(f"<th style='color:{col}'>{n} success</th>")
        header_parts.append(f"<th style='color:{col}'>{n} FPE</th>")
    header = "".join(header_parts)
    rows = ""
    for pt in all_paths:
        cells = f"<td>{pt}</td>"
        for name in model_names:
            pt_data = results_by_model[name].get(pt, [])
            if not pt_data:
                cells += "<td>—</td><td>—</td>"
                continue
            sr = sum(1 for m in pt_data if m["success"]) / len(pt_data)
            mfpe = np.mean([m["fpe"] for m in pt_data])
            cells += f"<td style='color:{color_pm(sr)}'>{sr:.0%} ({sum(1 for m in pt_data if m['success'])}/{len(pt_data)})</td>"
            cells += f"<td>{mfpe:.2f}m</td>"
        rows += f"<tr>{cells}</tr>"

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>V5 Closed-Loop Simulation (Phase 1)</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; padding: 24px; background: #0f172a; color: #e2e8f0; }}
  h1 {{ font-size: 2rem; margin-bottom: 8px; }}
  .sub {{ color: #94a3b8; margin-bottom: 24px; max-width: 900px; line-height: 1.6; }}
  .back {{ color: #60a5fa; text-decoration: none; display: inline-block; margin-bottom: 16px; }}
  .grid {{ display: grid; grid-template-columns: repeat({len(model_names)}, 1fr); gap: 16px; margin-bottom: 20px; }}
  .box {{ background: #1e293b; padding: 16px 20px; border-radius: 10px; }}
  .num {{ font-size: 2.5rem; font-weight: 800; }}
  table {{ border-collapse: collapse; background: #1e293b; border-radius: 8px; overflow: hidden; width: 100%; margin-top: 16px; }}
  th, td {{ padding: 8px 14px; border-bottom: 1px solid #334155; text-align: left; font-size: 0.9rem; }}
  th {{ background: #0b1220; }}
  .diag {{ background: #172554; border-left: 4px solid #60a5fa; padding: 14px 18px; border-radius: 6px; color: #dbeafe; margin-top: 20px; line-height: 1.7; }}
</style>
</head>
<body>
  <a class="back" href="../../index.html">← Back to main</a>
  <h1>V5 Closed-Loop Simulation (Phase 1: Offline Replay)</h1>
  <p class="sub">
    각 프레임의 예측 action을 누적해 kinematic trajectory를 생성하고 expert trajectory와 비교.
    성공 기준: FPE &lt; 0.5m AND TLD ∈ [0.7, 1.5].
  </p>

  <div class="grid">{boxes}
  </div>

  <h2>Per Path Type</h2>
  <table>
    <tr>{header}</tr>
    {rows}
  </table>

  <div class="diag">
    <strong>설정</strong><br>
    Phase 1: offline replay — 원본 H5 이미지로 예측, 예측 action을 kinematic model에 적분.<br>
    속도 매핑: lx/ly=1.15 m/s, az=±0.25 rad/s (데이터 실측). dt=0.1s.<br>
    성공 기준: FPE &lt; 0.5m AND TLD ∈ [0.7, 1.5].
  </div>
</body>
</html>"""
    return html


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["exp11", "step2", "step3", "step3_ablated", "exp17", "exp18", "exp19", "exp49", "exp50", "exp51", "exp52", "exp53", "both"], default="step2")
    ap.add_argument("--config", default="configs/mobile_vla_v5_exp11_google_robot_8cls.json")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--dt", type=float, default=DT_DEFAULT)
    ap.add_argument("--success_fpe", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="Seeds for multi-run evaluation (step2/step3 only)")
    args = ap.parse_args()

    run_exp11 = args.model in ("exp11", "exp17", "exp18", "both")
    run_step2 = args.model in ("step2", "both")
    run_step3 = args.model == "step3"
    run_exp19 = args.model == "exp19"

    # Load existing results so partial runs merge rather than overwrite
    existing_json_path = OUT_DIR / "rollout_metrics.json"
    if existing_json_path.exists():
        existing = json.loads(existing_json_path.read_text())
        summary_by_model = existing.get("summary", {})
        results_by_model = existing.get("per_path", {})
    else:
        results_by_model = {}
        summary_by_model = {}

    # ── Step 2 evaluation ────────────────────────────────────────────────────
    if run_step2:
        print("\n=== Step 2 (BBox+Image MLP) ===")
        bbox_ds = json.loads((STEP1_DIR / "bbox_dataset.json").read_text())

        # Episode-level split (same seed=42 as original step2 script)
        rng = np.random.default_rng(42)
        by_path = defaultdict(list)
        for i, ep in enumerate(bbox_ds):
            by_path[ep["path_type"]].append(i)
        train_idx, test_idx = [], []
        for _, idxs in by_path.items():
            rng.shuffle(idxs)
            k = max(1, int(len(idxs) * 0.2))
            test_idx.extend(idxs[:k])
            train_idx.extend(idxs[k:])
        train_eps = [bbox_ds[i] for i in train_idx]
        test_eps  = [bbox_ds[i] for i in test_idx]
        print(f"  MLP train={len(train_eps)} test={len(test_eps)} episodes")

        print("  Training MLP...")
        mlp, device = train_step2_mlp(bbox_ds, train_eps)

        ep_results = defaultdict(list)
        for ep_entry in test_eps:
            pred_classes, expert_actions = eval_step2_episode(ep_entry, mlp, device)
            expert_cls = [continuous_to_class(*a[:3]) for a in expert_actions]
            expert_traj = build_trajectory(expert_cls, args.dt)
            pred_traj   = build_trajectory(pred_classes, args.dt)
            m = compute_metrics(expert_traj, pred_traj, args.success_fpe)
            m["episode"] = ep_entry["episode"]
            m["path_type"] = ep_entry["path_type"]
            ep_results[ep_entry["path_type"]].append(m)
            print(f"  {ep_entry['path_type']:20s} FPE={m['fpe']:.2f}m TLD={m['tld']:.2f} {'✅' if m['success'] else '❌'}")

        all_m = [m for ms in ep_results.values() for m in ms]
        summary_by_model["step2"] = {
            "n_episodes": len(all_m),
            "success_rate": sum(m["success"] for m in all_m) / max(len(all_m), 1),
            "mean_fpe": float(np.mean([m["fpe"] for m in all_m])),
            "mean_tld": float(np.mean([m["tld"] for m in all_m])),
        }
        results_by_model["step2"] = dict(ep_results)
        print(f"\n  Step 2 success: {summary_by_model['step2']['success_rate']:.1%}"
              f"  FPE: {summary_by_model['step2']['mean_fpe']:.2f}m"
              f"  TLD: {summary_by_model['step2']['mean_tld']:.2f}")

    # ── Exp11/Exp17/Exp18 evaluation ────────────────────────────────────────────
    if run_exp11:
        model_name = args.model.upper() if args.model in ("exp11", "exp17") else "Exp11"
        print(f"\n=== {model_name} (policy model) ===")
        if not args.ckpt:
            # Try to find best ckpt automatically
            if args.model == "exp17":
                ckpt_dir = ROOT / "runs/v5_nav/kosmos/mobile_vla_v5_exp17/2026-04-20/v5-exp17-step3-balanced"
            elif args.model == "exp18":
                ckpt_dir = ROOT / "runs/v5_nav/kosmos/mobile_vla_v5_exp18/2026-04-21/v5-exp18-vla-text-fusion"
            else:
                ckpt_dir = ROOT / "runs/v5_nav/kosmos/mobile_vla_v5_exp11/2026-04-16/v5-exp11-google-robot-8cls"
            ckpts = sorted(ckpt_dir.glob("epoch_epoch*.ckpt"))
            if not ckpts:
                print(f"  No {model_name} checkpoint found, skipping.")
                run_exp11 = False
            else:
                args.ckpt = str(ckpts[-1])
                print(f"  Auto-selected ckpt: {Path(args.ckpt).name}")
        if args.model == "exp18" and args.config == "configs/mobile_vla_v5_exp11_google_robot_8cls.json":
            args.config = "configs/mobile_vla_v5_exp18_vla_finetuned.json"

    if run_exp11:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(str(ROOT / ".vlms/kosmos-2-patch14-224"))
        text_embedding_map = load_text_embedding_map() if args.model == "exp18" else None

        model_wrapper = load_exp11_model(args.config, args.ckpt)
        model_backbone = model_wrapper.model  # RoboKosMos, same as PM eval
        model_backbone.eval()
        print("  Model loaded.")

        # Same test split as Step 2 for fair comparison
        bbox_ds = json.loads((STEP1_DIR / "bbox_dataset.json").read_text())
        rng2 = np.random.default_rng(42)
        by_path2 = defaultdict(list)
        for i, ep in enumerate(bbox_ds):
            by_path2[ep["path_type"]].append(i)
        train_idx2, test_idx2 = [], []
        for _, idxs in by_path2.items():
            rng2.shuffle(idxs)
            k = max(1, int(len(idxs) * 0.2))
            test_idx2.extend(idxs[:k])
            train_idx2.extend(idxs[k:])
        test_ep_stems = {bbox_ds[i]["episode"] for i in test_idx2}

        from transformers import AutoProcessor as AP
        processor = AP.from_pretrained(str(ROOT / ".vlms/kosmos-2-patch14-224"))

        ep_results = defaultdict(list)
        for pt in PATH_TYPES:
            all_eps = sorted(DATA_DIR.glob(f"episode_*target_{pt}_path*.h5"))
            test_eps = [e for e in all_eps if e.stem in test_ep_stems]
            if not test_eps:
                test_eps = all_eps[-1:]
            for ep_path in test_eps:
                try:
                    pred_classes, expert_actions = eval_exp11_episode(
                        ep_path, model_backbone, processor, window_size=8,
                        text_embedding_map=text_embedding_map,
                    )
                    expert_cls = [continuous_to_class(*a[:3]) for a in expert_actions]
                    expert_traj = build_trajectory(expert_cls, args.dt)
                    pred_traj   = build_trajectory(pred_classes, args.dt)
                    m = compute_metrics(expert_traj, pred_traj, args.success_fpe)
                    m["episode"] = ep_path.stem
                    m["path_type"] = pt
                    ep_results[pt].append(m)
                    print(f"  {pt:20s} FPE={m['fpe']:.2f}m TLD={m['tld']:.2f} {'✅' if m['success'] else '❌'}")
                except Exception as e:
                    print(f"  ERROR on {ep_path.name}: {e}")

        all_m = [m for ms in ep_results.values() for m in ms]
        model_key = args.model if args.model in ("exp11", "exp17", "exp18") else "exp11"
        summary_by_model[model_key] = {
            "n_episodes": len(all_m),
            "success_rate": sum(m["success"] for m in all_m) / max(len(all_m), 1),
            "mean_fpe": float(np.mean([m["fpe"] for m in all_m])),
            "mean_tld": float(np.mean([m["tld"] for m in all_m])),
        }
        results_by_model[model_key] = dict(ep_results)
        print(f"\n  {model_key.upper()} success: {summary_by_model[model_key]['success_rate']:.1%}"
              f"  FPE: {summary_by_model[model_key]['mean_fpe']:.2f}m"
              f"  TLD: {summary_by_model[model_key]['mean_tld']:.2f}")

    # ── Step 3 multi-seed evaluation ─────────────────────────────────────────
    if run_step3:
        print("\n=== Step 3 (Full Dataset BBox+Image MLP, WINDOW=8, 32x32) ===")
        full_ds_path = STEP1_DIR / "bbox_dataset_full.json"
        if not full_ds_path.exists():
            print(f"  ERROR: {full_ds_path} not found. Run step1 --full first.")
        else:
            bbox_ds = json.loads(full_ds_path.read_text())
            print(f"  Dataset: {len(bbox_ds)} episodes")

            seeds = args.seeds
            print(f"  Seeds: {seeds}  ({len(seeds)} runs)")

            # Collect per-seed results
            seed_results = []  # list of ep_results dicts per seed
            for seed in seeds:
                print(f"\n  --- seed={seed} ---")
                rng = np.random.default_rng(seed)
                by_path = defaultdict(list)
                for i, ep in enumerate(bbox_ds):
                    by_path[ep["path_type"]].append(i)
                train_idx, test_idx = [], []
                for _, idxs in by_path.items():
                    rng.shuffle(idxs)
                    k = max(1, int(len(idxs) * 0.2))
                    test_idx.extend(idxs[:k])
                    train_idx.extend(idxs[k:])
                train_eps = [bbox_ds[i] for i in train_idx]
                test_eps  = [bbox_ds[i] for i in test_idx]
                print(f"  train={len(train_eps)} test={len(test_eps)} episodes")

                mlp, device, offset = train_step3_mlp(bbox_ds, train_eps, seed=seed)

                ep_results = defaultdict(list)
                for ep_entry in test_eps:
                    pred_classes, expert_actions = eval_step3_episode(ep_entry, mlp, device, offset=offset)
                    expert_cls = [continuous_to_class(*a[:3]) for a in expert_actions]
                    expert_traj = build_trajectory(expert_cls, args.dt)
                    pred_traj   = build_trajectory(pred_classes, args.dt)
                    m = compute_metrics(expert_traj, pred_traj, args.success_fpe)
                    m["episode"] = ep_entry["episode"]
                    m["path_type"] = ep_entry["path_type"]
                    ep_results[ep_entry["path_type"]].append(m)
                    print(f"    {ep_entry['path_type']:20s} FPE={m['fpe']:.2f}m TLD={m['tld']:.2f} {'✅' if m['success'] else '❌'}")

                all_m = [m for ms in ep_results.values() for m in ms]
                sr = sum(m["success"] for m in all_m) / max(len(all_m), 1)
                fpe = float(np.mean([m["fpe"] for m in all_m]))
                tld = float(np.mean([m["tld"] for m in all_m]))
                print(f"  seed={seed}: success={sr:.1%}  FPE={fpe:.2f}m  TLD={tld:.2f}")
                seed_results.append({"seed": seed, "success_rate": sr, "mean_fpe": fpe,
                                     "mean_tld": tld, "ep_results": dict(ep_results)})

            # Aggregate across seeds
            srs  = [r["success_rate"] for r in seed_results]
            fpes = [r["mean_fpe"] for r in seed_results]
            tlds = [r["mean_tld"] for r in seed_results]
            print(f"\n  === Step 3 Multi-seed Summary ({len(seeds)} seeds) ===")
            print(f"  Success:  {np.mean(srs):.1%} ± {np.std(srs):.1%}  {[f'{s:.1%}' for s in srs]}")
            print(f"  FPE:      {np.mean(fpes):.3f} ± {np.std(fpes):.3f}m")
            print(f"  TLD:      {np.mean(tlds):.3f} ± {np.std(tlds):.3f}")

            summary_by_model["step3"] = {
                "n_seeds": len(seeds),
                "success_rate": float(np.mean(srs)),
                "success_rate_std": float(np.std(srs)),
                "mean_fpe": float(np.mean(fpes)),
                "mean_fpe_std": float(np.std(fpes)),
                "mean_tld": float(np.mean(tlds)),
                "per_seed": seed_results,
            }
            # Use last seed's ep_results for HTML per-path table
            results_by_model["step3"] = seed_results[-1]["ep_results"]

            # Save step3-specific results
            step3_out = OUT_DIR / "step3_multiseed.json"
            step3_out.write_text(json.dumps({"summary": summary_by_model["step3"],
                                             "seeds": seed_results}, indent=2))
            print(f"  Saved: {step3_out}")

    # ── GoalNav (Exp49/50/51) evaluation ────────────────────────────────────
    if args.model in GOAL_NAV_CKPTS:
        exp_name = args.model
        print(f"\n=== {exp_name.upper()} (GoalNav MLP: bbox+vis+goal) ===")

        ckpt_path = GOAL_NAV_CKPTS[exp_name]
        if not ckpt_path.exists():
            print(f"  ERROR: checkpoint not found at {ckpt_path}")
        else:
            # Load model
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            import torch.nn as nn
            if "mlp" in ckpt and "model_state_dict" not in ckpt:
                # exp53 format: {'mlp': full GoalNavMLP state_dict, 'val_acc': float}
                state = ckpt["mlp"]
                d_in = state["net.0.weight"].shape[1]
                mlp = nn.Sequential(
                    nn.Linear(d_in, 512), nn.ReLU(), nn.Dropout(0.25),
                    nn.Linear(512, 256),  nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(256, 128),  nn.ReLU(), nn.Dropout(0.1),
                    nn.Linear(128, 64),   nn.ReLU(),
                    nn.Linear(64, NUM_CLASSES),
                )
                # strip "net." prefix to match bare Sequential keys
                mlp.load_state_dict({k[len("net."):]: v for k, v in state.items()})
            else:
                d_in = ckpt["d_in"]
                mlp = nn.Sequential(
                    nn.Linear(d_in, 512), nn.ReLU(), nn.Dropout(0.25),
                    nn.Linear(512, 256),  nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(256, 128),  nn.ReLU(), nn.Dropout(0.1),
                    nn.Linear(128, 64),   nn.ReLU(),
                    nn.Linear(64, NUM_CLASSES),
                )
                mlp.load_state_dict(ckpt["model_state_dict"])
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            mlp = mlp.to(device).eval()
            print(f"  Loaded {exp_name} (d_in={d_in}) from {ckpt_path.name}")

            # Load data
            vis_dir      = GOAL_NAV_VIS_DIRS[exp_name]
            vis_key      = GOAL_NAV_VIS_KEYS[exp_name]
            full_ds_path = EXP46_DIR / "bbox_dataset_full.json"
            vis_npz_path = vis_dir / f"{vis_key}.npz"
            vis_idx_path = vis_dir / f"{vis_key}_index.json"
            if not full_ds_path.exists() or not vis_npz_path.exists():
                print(f"  ERROR: vis cache not found at {vis_npz_path}")
            else:
                bbox_data = json.loads(full_ds_path.read_text())
                vis_npz   = np.load(str(vis_npz_path))
                vis_idx   = json.loads(vis_idx_path.read_text())
                vis_cache = {ep: vis_npz[f"ep_{i}"] for ep, i in vis_idx.items()}
                print(f"  Dataset: {len(bbox_data)} episodes, vis cache: {len(vis_cache)}")

                # Same split as training (StratifiedShuffleSplit, random_state=42)
                from sklearn.model_selection import StratifiedShuffleSplit
                ep_labels = [ep["path_type"] for ep in bbox_data]
                sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                tr_ep_idx, te_ep_idx = next(sss.split(np.zeros(len(bbox_data)), ep_labels))
                test_eps = [bbox_data[i] for i in te_ep_idx]
                print(f"  Test episodes: {len(test_eps)}")

                W = GOAL_NAV_WINDOW
                ep_results = defaultdict(list)
                for ep_data in test_eps:
                    ep_full  = ep_data["episode"]  # full path or stem
                    _ep_p = Path(ep_full)
                    if _ep_p.suffix == ".h5":
                        h5_path = _ep_p if _ep_p.exists() else DATA_DIR / _ep_p.name
                    else:
                        h5_path = DATA_DIR / f"{ep_full}.h5"
                    ep_key   = ep_full  # key into vis_cache (full path as stored)
                    pt       = ep_data["path_type"]
                    frames   = ep_data["frames"]
                    vis_feats = vis_cache.get(ep_key)
                    if vis_feats is None:
                        print(f"  SKIP {Path(ep_full).stem}: no vis cache")
                        continue

                    fr0  = frames[0]
                    goal = np.array(
                        [fr0["cx"], fr0["cy"], fr0["area"]] if fr0["has_bbox"] else [0.5, 0.5, 0.0],
                        dtype=np.float32,
                    )

                    pred_classes = []
                    for t in range(len(frames)):
                        bbox_feat = []
                        for k in range(W):
                            idx = max(0, t - (W - 1 - k))
                            fr  = frames[idx]
                            bbox_feat.extend([fr["cx"], fr["cy"], fr["area"], float(fr["has_bbox"])])
                        feat = np.concatenate([
                            np.array(bbox_feat, dtype=np.float32),
                            vis_feats[t],
                            goal,
                        ])
                        x = torch.tensor(feat[None], dtype=torch.float32, device=device)
                        with torch.no_grad():
                            pred_classes.append(int(mlp(x).argmax(1).item()))

                    # Expert actions from H5
                    if not h5_path.exists():
                        print(f"  SKIP {h5_path.stem}: no H5 file")
                        continue
                    with h5py.File(h5_path, "r") as f:
                        expert_actions = f["actions"][:]

                    n = min(len(pred_classes), len(frames), len(expert_actions))
                    expert_cls  = [continuous_to_class(*a[:3]) for a in expert_actions[:n]]
                    expert_traj = build_trajectory(expert_cls, args.dt)
                    pred_traj   = build_trajectory(pred_classes[:n], args.dt)
                    m = compute_metrics(expert_traj, pred_traj, args.success_fpe)
                    m["episode"]   = h5_path.stem
                    m["path_type"] = pt
                    ep_results[pt].append(m)
                    print(f"  {pt:20s} FPE={m['fpe']:.2f}m TLD={m['tld']:.2f} {'✅' if m['success'] else '❌'}")

                all_m = [m for ms in ep_results.values() for m in ms]
                summary_by_model[exp_name] = {
                    "n_episodes": len(all_m),
                    "success_rate": sum(m["success"] for m in all_m) / max(len(all_m), 1),
                    "mean_fpe": float(np.mean([m["fpe"] for m in all_m])),
                    "mean_tld": float(np.mean([m["tld"] for m in all_m])),
                }
                results_by_model[exp_name] = dict(ep_results)
                s = summary_by_model[exp_name]
                print(f"\n  {exp_name.upper()} success: {s['success_rate']:.1%}"
                      f"  FPE: {s['mean_fpe']:.2f}m  TLD: {s['mean_tld']:.2f}")

    if run_exp19:
        print("\n=== Exp19 (Step2 + proxy features) ===")
        bbox_ds = json.loads((STEP1_DIR / "bbox_dataset.json").read_text())
        rng = np.random.default_rng(42)
        by_path = defaultdict(list)
        for i, ep in enumerate(bbox_ds):
            by_path[ep["path_type"]].append(i)
        train_idx, test_idx = [], []
        for _, idxs in by_path.items():
            rng.shuffle(idxs)
            k = max(1, int(len(idxs) * 0.2))
            test_idx.extend(idxs[:k])
            train_idx.extend(idxs[k:])
        train_eps = [bbox_ds[i] for i in train_idx]
        test_eps = [bbox_ds[i] for i in test_idx]
        print(f"  Exp19 train={len(train_eps)} test={len(test_eps)} episodes")

        print("  Training Exp19 proxy MLP...")
        mlp, device = train_exp19_proxy_mlp(bbox_ds, train_eps)

        ep_results = defaultdict(list)
        for ep_entry in test_eps:
            pred_classes, expert_actions = eval_exp19_episode(ep_entry, mlp, device)
            expert_cls = [continuous_to_class(*a[:3]) for a in expert_actions]
            expert_traj = build_trajectory(expert_cls, args.dt)
            pred_traj = build_trajectory(pred_classes, args.dt)
            m = compute_metrics(expert_traj, pred_traj, args.success_fpe)
            m["episode"] = ep_entry["episode"]
            m["path_type"] = ep_entry["path_type"]
            ep_results[ep_entry["path_type"]].append(m)
            print(f"  {ep_entry['path_type']:20s} FPE={m['fpe']:.2f}m TLD={m['tld']:.2f} {'✅' if m['success'] else '❌'}")

        all_m = [m for ms in ep_results.values() for m in ms]
        summary_by_model["exp19"] = {
            "n_episodes": len(all_m),
            "success_rate": sum(m["success"] for m in all_m) / max(len(all_m), 1),
            "mean_fpe": float(np.mean([m["fpe"] for m in all_m])),
            "mean_tld": float(np.mean([m["tld"] for m in all_m])),
        }
        results_by_model["exp19"] = dict(ep_results)
        print(f"\n  EXP19 success: {summary_by_model['exp19']['success_rate']:.1%}"
              f"  FPE: {summary_by_model['exp19']['mean_fpe']:.2f}m"
              f"  TLD: {summary_by_model['exp19']['mean_tld']:.2f}")

    # ── Save & HTML ──────────────────────────────────────────────────────────
    out_json = {
        "summary": summary_by_model,
        "per_path": results_by_model,
    }
    (OUT_DIR / "rollout_metrics.json").write_text(json.dumps(out_json, indent=2))
    html = build_html(results_by_model, summary_by_model)
    (OUT_DIR / "index.html").write_text(html)

    print(f"\nSummary JSON : {OUT_DIR / 'rollout_metrics.json'}")
    print(f"HTML         : {OUT_DIR / 'index.html'}")
    print("\nFINAL SUMMARY:")
    for name, s in summary_by_model.items():
        print(f"  {name:8s}: success={s['success_rate']:.1%}  FPE={s['mean_fpe']:.2f}m  TLD={s['mean_tld']:.2f}")


if __name__ == "__main__":
    main()

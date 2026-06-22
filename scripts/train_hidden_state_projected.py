#!/usr/bin/env python3
"""
plan_20260622_hidden_state_projection_weighting.md — CH40/41이 원시 2304차원
hidden state를 그대로 concat/대체해서 baseline을 못 넘었던 것의 재시도(C안).
hidden state 앞에 학습 가능한 linear projection(2304 -> proj_dim)을 둬서
"어떤 차원/조합을 얼마나 쓸지"를 역전파로 배우게 한다 — 명시적 가중치 스칼라보다
표현력이 높고 구현이 단순.

train_hidden_state_action.py의 복제본, 차이는 ProjectedHiddenHead 도입뿐.

Usage:
  .venv/bin/python3 scripts/train_hidden_state_projected.py --use_hidden_state add --proj_dim 64
  .venv/bin/python3 scripts/train_hidden_state_projected.py --use_hidden_state replace --proj_dim 128
"""

import argparse, json, sys, time
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

from robovlm_nav.image_preprocess import resize_for_vlm  # noqa: E402

VLM_PATH     = ROOT / ".vlms" / "kosmos-2-patch14-224"
DATA_PATH    = ROOT / "docs" / "v5" / "bbox_nav_exp46" / "bbox_dataset_full.json"
STAGE1_V2    = ROOT / "runs" / "v5_nav" / "mlp" / "shared" / "stage1_v2_projs.pt"
STAGE2_DIR   = ROOT / "runs" / "v5_nav" / "mlp" / "exp_hidden_state" / "stage2_v2_projected"
HIDDEN_CACHE = ROOT / "docs" / "v5" / "hidden_state_cache" / "v5_hidden_states.npz"

CLASS_NAMES = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "ROT_L", "ROT_R"]
NUM_CLASSES = 8
WINDOW      = 8
VIS_DIM     = 1024
PROJ_DIM    = 256
HIDDEN_DIM  = 2304


class FrozenCLIPV2(nn.Module):
    def __init__(self, vlm_path, ckpt_path, device):
        super().__init__()
        from transformers import AutoModelForVision2Seq, AutoProcessor
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        print(f"[MODEL] Stage1 v2 val_acc={ckpt['val_acc']:.4f}", flush=True)
        self.processor = AutoProcessor.from_pretrained(str(vlm_path))
        base = AutoModelForVision2Seq.from_pretrained(str(vlm_path), torch_dtype=torch.float16)
        self.vision_model = base.vision_model.to(device)
        self.image_proj   = nn.Linear(VIS_DIM, PROJ_DIM).to(device)
        self.image_proj.load_state_dict(ckpt["image_proj"])
        for p in self.vision_model.parameters():
            p.requires_grad = False
        for p in self.image_proj.parameters():
            p.requires_grad = False
        print("[MODEL] frozen 완료 (vision_model + image_proj)", flush=True)

    @torch.no_grad()
    def encode_batch(self, pil_images, device, batch=32):
        all_feats = []
        for i in range(0, len(pil_images), batch):
            imgs = pil_images[i:i+batch]
            inputs = self.processor(images=imgs, return_tensors="pt")
            pv = inputs["pixel_values"].to(device, dtype=torch.float16)
            out = self.vision_model(pixel_values=pv)
            feat = out.last_hidden_state.mean(dim=1).float()
            all_feats.append(F.normalize(self.image_proj(feat), dim=-1))
        return torch.cat(all_feats, dim=0)


class ActionMLP(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 128),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),   nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64,  NUM_CLASSES),
        )
    def forward(self, x):
        return self.net(x)


class LinearHead(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Linear(d_in, NUM_CLASSES)
    def forward(self, x):
        return self.net(x)


class FCHead(nn.Module):
    """RoboVLMs FCDecoder 스타일 — deep MLP, ActionMLP보다 넓고 깊음."""
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 1024), nn.ReLU(),
            nn.Linear(1024, 512),  nn.ReLU(),
            nn.Linear(512, 256),   nn.ReLU(),
            nn.Linear(256, NUM_CLASSES),
        )
    def forward(self, x):
        return self.net(x)


HEAD_REGISTRY = {"linear": LinearHead, "mlp": ActionMLP, "fc": FCHead}


class ProjectedHiddenHead(nn.Module):
    """hidden state(2304) -> 학습 가능한 linear projection(proj_dim) -> bbox_img와 concat -> head(교체 가능)."""
    def __init__(self, bbox_img_dim, proj_dim, mode, head_type="mlp"):
        super().__init__()
        self.mode = mode
        self.proj = nn.Linear(HIDDEN_DIM, proj_dim)
        d_in = (bbox_img_dim + proj_dim) if mode == "add" else proj_dim + PROJ_DIM
        self.head = HEAD_REGISTRY[head_type](d_in=d_in)

    def forward(self, bbox_img, hidden_raw):
        h = self.proj(hidden_raw)
        x = torch.cat([bbox_img, h], dim=-1)
        return self.head(x)


def bbox_feat(frames, t, window=WINDOW):
    arr = []
    for k in range(window):
        fr = frames[max(0, t - (window - 1 - k))]
        cx = fr.get("cx", 0.5); cy = fr.get("cy", 0.5)
        area = fr.get("area", 0.05); has = float(fr.get("has_bbox", False))
        arr.extend([cx, cy, area, has])
    return np.array(arr, dtype=np.float32)


def load_images(h5_path, indices):
    import io as _io
    with h5py.File(h5_path, "r") as f:
        imgs_ds = f["observations"]["images"]
        result = []
        for i in indices:
            raw = imgs_ds[i]
            if hasattr(raw, "dtype") and raw.dtype != object and raw.ndim >= 2:
                img = Image.fromarray(raw.astype("uint8"))
            else:
                arr = np.frombuffer(bytes(raw), dtype=np.uint8)
                img = Image.open(_io.BytesIO(arr)).convert("RGB")
            result.append(resize_for_vlm(img))
        return result


def load_hidden_cache():
    npz = np.load(HIDDEN_CACHE)
    return {k: npz[k] for k in npz.files}


def hidden_lookup(hidden_cache, ep_path, frame_idx):
    stem = Path(ep_path).stem
    return hidden_cache.get(f"{stem}__f{frame_idx}")


def filter_episodes_with_hidden(eps, hidden_cache):
    kept, dropped = [], []
    for ep in eps:
        stem = Path(ep["episode"]).stem
        has_any = any(f"{stem}__f{fr['frame_idx']}" in hidden_cache for fr in ep["frames"])
        (kept if has_any else dropped).append(ep)
    return kept, dropped


def precompute_features(enc, eps, device, label):
    cache = {}
    n = len(eps)
    print(f"[CACHE] {label} feature 사전 추출 중 ({n} episodes)...", flush=True)
    t0 = time.time()
    for i, ep in enumerate(eps):
        try:
            imgs = load_images(ep["episode"], [fr["frame_idx"] for fr in ep["frames"]])
        except Exception as e:
            print(f"  skip {ep['episode']}: {e}", flush=True)
            cache[ep["episode"]] = None
            continue
        feats = enc.encode_batch(imgs, device)
        cache[ep["episode"]] = feats.cpu()
        if (i + 1) % 20 == 0 or (i + 1) == n:
            print(f"  {i+1}/{n} done ({time.time()-t0:.0f}s)", flush=True)
    print(f"[CACHE] {label} 완료 — {time.time()-t0:.1f}초", flush=True)
    return cache


def build_bbox_img(mode, bf, img_feat):
    return img_feat if mode == "replace" else torch.cat([bf, img_feat])


def _step(bi_batch, h_batch, labels, model, opt, criterion, device):
    bi = torch.stack(bi_batch).to(device)
    h = torch.stack(h_batch).to(device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    opt.zero_grad()
    criterion(model(bi, h), y).backward()
    opt.step()


@torch.no_grad()
def evaluate(te_cache, te_eps, model, device, hidden_cache, mode, window=WINDOW):
    model.eval()
    correct = total = 0
    from collections import defaultdict
    per_class = defaultdict(lambda: [0, 0])
    for ep in te_eps:
        feats = te_cache.get(ep["episode"])
        if feats is None:
            continue
        for t, fr in enumerate(ep["frames"]):
            bf = torch.tensor(bbox_feat(ep["frames"], t, window=window), dtype=torch.float32)
            bi = build_bbox_img(mode, bf, feats[t]).unsqueeze(0).to(device)
            hv = hidden_lookup(hidden_cache, ep["episode"], fr["frame_idx"])
            h = torch.from_numpy(hv.astype(np.float32)) if hv is not None else torch.zeros(HIDDEN_DIM, dtype=torch.float32)
            h = h.unsqueeze(0).to(device)
            p = model(bi, h).argmax(1).item()
            g = fr["gt_class"]
            per_class[g][0] += int(p == g)
            per_class[g][1] += 1
            correct += int(p == g)
            total += 1
    acc = correct / total if total > 0 else 0.0
    return acc, per_class


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[DEVICE] {device}", flush=True)
    print(f"[MODE] use_hidden_state={args.use_hidden_state}  proj_dim={args.proj_dim}", flush=True)

    data = json.loads(Path(args.data).read_text())
    ep_labels = [ep["path_type"] for ep in data]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(sss.split(np.zeros(len(data)), ep_labels))
    tr_eps = [data[i] for i in tr_idx]
    te_eps = [data[i] for i in te_idx]
    print(f"Train: {len(tr_eps)} ep  Val: {len(te_eps)} ep  (baseline과 동일 seed=42 split)", flush=True)

    hidden_cache = load_hidden_cache()
    print(f"[HIDDEN] 캐시 로드 — {len(hidden_cache)}개 프레임", flush=True)
    tr_before, te_before = len(tr_eps), len(te_eps)
    tr_eps, _ = filter_episodes_with_hidden(tr_eps, hidden_cache)
    te_eps, _ = filter_episodes_with_hidden(te_eps, hidden_cache)
    print(f"[HIDDEN] hidden state 없는 episode 제외: train {tr_before}->{len(tr_eps)}, "
          f"val {te_before}->{len(te_eps)}", flush=True)

    enc = FrozenCLIPV2(VLM_PATH, STAGE1_V2, device).to(device).eval()
    tr_cache = precompute_features(enc, tr_eps, device, "train")
    te_cache = precompute_features(enc, te_eps, device, "val")
    del enc
    torch.cuda.empty_cache() if device.type == "cuda" else None
    print("[CACHE] VLM 해제 완료 — MLP만 학습", flush=True)

    window = args.window
    bbox_img_dim = (window * 4 + PROJ_DIM) if args.use_hidden_state == "add" else PROJ_DIM
    model = ProjectedHiddenHead(bbox_img_dim, args.proj_dim, args.use_hidden_state, args.head_type).to(device)
    print(f"[HEAD] {args.head_type}  window={window}  proj_dim={args.proj_dim}  mode={args.use_hidden_state}", flush=True)

    all_labels = [fr["gt_class"] for ep in tr_eps for fr in ep["frames"]]
    counts = np.bincount(all_labels, minlength=NUM_CLASSES).astype(float)
    weights = np.where(counts > 0, 1.0 / (counts + 1e-6), 0.0)
    weights /= weights.sum() / NUM_CLASSES
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc, best_state = 0.0, None
    STAGE2_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'epoch':>6} {'val_acc':>9} {'best':>9}", flush=True)
    print("-" * 30, flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        np.random.shuffle(tr_eps)
        bi_batch, h_batch, lb_batch = [], [], []

        for ep in tr_eps:
            feats = tr_cache.get(ep["episode"])
            if feats is None:
                continue
            for t, fr in enumerate(ep["frames"]):
                bf = torch.tensor(bbox_feat(ep["frames"], t, window=window), dtype=torch.float32)
                bi = build_bbox_img(args.use_hidden_state, bf, feats[t])
                hv = hidden_lookup(hidden_cache, ep["episode"], fr["frame_idx"])
                h = torch.from_numpy(hv.astype(np.float32)) if hv is not None else torch.zeros(HIDDEN_DIM, dtype=torch.float32)
                bi_batch.append(bi); h_batch.append(h); lb_batch.append(fr["gt_class"])
                if len(lb_batch) >= args.batch_size:
                    _step(bi_batch, h_batch, lb_batch, model, opt, criterion, device)
                    bi_batch, h_batch, lb_batch = [], [], []

        if lb_batch:
            _step(bi_batch, h_batch, lb_batch, model, opt, criterion, device)
        sched.step()

        if epoch % 10 == 0 or epoch == args.epochs:
            acc, per_class = evaluate(te_cache, te_eps, model, device, hidden_cache, args.use_hidden_state, window=window)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"{epoch:>6}  {acc:>8.4f}  {best_acc:>8.4f}", flush=True)

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    final_acc, per_class = evaluate(te_cache, te_eps, model, device, hidden_cache, args.use_hidden_state, window=window)

    print(f"\n{'='*55}", flush=True)
    print(f"  ProjectedHiddenHead 완료 (mode={args.use_hidden_state}, proj_dim={args.proj_dim})")
    print(f"  val_acc: {final_acc:.4f}")
    print(f"{'='*55}", flush=True)

    ckpt_path = STAGE2_DIR / f"stage2_proj{args.proj_dim}_{args.use_hidden_state}_{args.head_type}.pt"
    torch.save({"model": best_state, "val_acc": final_acc, "proj_dim": args.proj_dim,
                "use_hidden_state": args.use_hidden_state, "window": window,
                "head_type": args.head_type}, str(ckpt_path))
    print(f"\n[SAVE] {ckpt_path}", flush=True)
    return final_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",          type=int,   default=300)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--window",          type=int,   default=WINDOW)
    p.add_argument("--data",            default=str(DATA_PATH))
    p.add_argument("--use_hidden_state", default="add", choices=["add", "replace"])
    p.add_argument("--proj_dim",        type=int,   default=64)
    p.add_argument("--head_type",       default="mlp", choices=["linear", "mlp", "fc"])
    args = p.parse_args()

    t0 = time.time()
    print("=" * 55, flush=True)
    print("ProjectedHiddenHead (plan_20260622_hidden_state_projection_weighting.md)")
    print(f"mode={args.use_hidden_state}  proj_dim={args.proj_dim}  epochs={args.epochs}", flush=True)
    print("=" * 55, flush=True)
    train(args)
    print(f"\n소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
RoboVLMs LSTM head를 hidden-state ablation에 추가(사용자 요청, CH43 후속).
원래 LSTMHead(stage2_v2_inference_server.py)는 윈도우 8프레임 시퀀스를
[img_feat(256), cx,cy,area,has(4)]로 만들어 LSTM에 넣는 구조 — 여기에 매
윈도우 스텝마다 그 프레임의 hidden state(2304 -> proj_dim 학습 가능 projection)를
추가해서 시퀀스 입력을 확장한다.

mode:
  none    [img(256), bbox(4)] x window — 원래 LSTMHead 그대로(baseline 재현)
  add     [img(256), proj(hidden)(proj_dim), bbox(4)] x window
  replace [img(256), proj(hidden)(proj_dim)] x window (bbox 제외)

Usage:
  .venv/bin/python3 scripts/train_hidden_state_lstm.py --use_hidden_state none
  .venv/bin/python3 scripts/train_hidden_state_lstm.py --use_hidden_state add --proj_dim 32
  .venv/bin/python3 scripts/train_hidden_state_lstm.py --use_hidden_state replace --proj_dim 32
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
STAGE2_DIR   = ROOT / "runs" / "v5_nav" / "mlp" / "exp_hidden_state" / "stage2_v2_lstm"
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
        print("[MODEL] frozen 완료", flush=True)

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


class LSTMHidden(nn.Module):
    def __init__(self, seq_dim, proj_dim, mode, hidden=256, num_layers=2):
        super().__init__()
        self.mode = mode
        self.proj = nn.Linear(HIDDEN_DIM, proj_dim) if mode != "none" else None
        self.lstm = nn.LSTM(seq_dim, hidden, num_layers, batch_first=True, dropout=0.1)
        self.classifier = nn.Linear(hidden, NUM_CLASSES)

    def forward(self, img_seq, bbox_seq, hidden_seq):
        # img_seq: (B, W, 256), bbox_seq: (B, W, 4) or None, hidden_seq: (B, W, 2304) or None
        parts = [img_seq]
        if self.mode != "replace" and bbox_seq is not None:
            parts.append(bbox_seq)
        if self.mode != "none":
            parts.append(self.proj(hidden_seq))
        x = torch.cat(parts, dim=-1)
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1])


def bbox_one(fr):
    cx = fr.get("cx", 0.5); cy = fr.get("cy", 0.5)
    area = fr.get("area", 0.05); has = float(fr.get("has_bbox", False))
    return np.array([cx, cy, area, has], dtype=np.float32)


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
    kept = []
    for ep in eps:
        stem = Path(ep["episode"]).stem
        if any(f"{stem}__f{fr['frame_idx']}" in hidden_cache for fr in ep["frames"]):
            kept.append(ep)
    return kept


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


def build_seq(frames, feats, t, hidden_cache, ep_path, mode, window=WINDOW):
    img_seq, bbox_seq, hidden_seq = [], [], []
    for k in range(window):
        ti = max(0, t - (window - 1 - k))
        ti = min(ti, len(frames) - 1)
        fr = frames[ti]
        img_seq.append(feats[ti])
        bbox_seq.append(torch.tensor(bbox_one(fr)))
        if mode != "none":
            hv = hidden_lookup(hidden_cache, ep_path, fr["frame_idx"])
            h = torch.from_numpy(hv.astype(np.float32)) if hv is not None else torch.zeros(HIDDEN_DIM, dtype=torch.float32)
            hidden_seq.append(h)
    img_seq = torch.stack(img_seq)
    bbox_seq = torch.stack(bbox_seq)
    hidden_seq = torch.stack(hidden_seq) if mode != "none" else None
    return img_seq, bbox_seq, hidden_seq


def _step(img_b, bbox_b, hidden_b, labels, model, opt, criterion, device, mode):
    img_b = torch.stack(img_b).to(device)
    bbox_b = torch.stack(bbox_b).to(device)
    hidden_b = torch.stack(hidden_b).to(device) if mode != "none" else None
    y = torch.tensor(labels, dtype=torch.long, device=device)
    opt.zero_grad()
    criterion(model(img_b, bbox_b, hidden_b), y).backward()
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
            img_seq, bbox_seq, hidden_seq = build_seq(ep["frames"], feats, t, hidden_cache, ep["episode"], mode, window)
            img_seq = img_seq.unsqueeze(0).to(device)
            bbox_seq = bbox_seq.unsqueeze(0).to(device)
            hidden_seq = hidden_seq.unsqueeze(0).to(device) if hidden_seq is not None else None
            p = model(img_seq, bbox_seq, hidden_seq).argmax(1).item()
            g = fr["gt_class"]
            per_class[g][0] += int(p == g)
            per_class[g][1] += 1
            correct += int(p == g)
            total += 1
    return (correct / total if total > 0 else 0.0), per_class


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
    print(f"Train: {len(tr_eps)} ep  Val: {len(te_eps)} ep", flush=True)

    hidden_cache = {}
    if args.use_hidden_state != "none":
        hidden_cache = load_hidden_cache()
        tr_eps = filter_episodes_with_hidden(tr_eps, hidden_cache)
        te_eps = filter_episodes_with_hidden(te_eps, hidden_cache)
        print(f"[HIDDEN] 캐시 로드 — train {len(tr_eps)} ep, val {len(te_eps)} ep", flush=True)

    enc = FrozenCLIPV2(VLM_PATH, STAGE1_V2, device).to(device).eval()
    tr_cache = precompute_features(enc, tr_eps, device, "train")
    te_cache = precompute_features(enc, te_eps, device, "val")
    del enc
    torch.cuda.empty_cache() if device.type == "cuda" else None

    seq_dim = PROJ_DIM
    if args.use_hidden_state != "replace":
        seq_dim += 4
    if args.use_hidden_state != "none":
        seq_dim += args.proj_dim
    model = LSTMHidden(seq_dim, args.proj_dim, args.use_hidden_state).to(device)
    print(f"[HEAD] LSTMHidden  seq_dim={seq_dim}  mode={args.use_hidden_state}", flush=True)

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

    for epoch in range(1, args.epochs + 1):
        model.train()
        np.random.shuffle(tr_eps)
        img_b, bbox_b, hidden_b, lb_b = [], [], [], []

        for ep in tr_eps:
            feats = tr_cache.get(ep["episode"])
            if feats is None:
                continue
            for t, fr in enumerate(ep["frames"]):
                img_seq, bbox_seq, hidden_seq = build_seq(ep["frames"], feats, t, hidden_cache, ep["episode"], args.use_hidden_state, args.window)
                img_b.append(img_seq); bbox_b.append(bbox_seq)
                if args.use_hidden_state != "none":
                    hidden_b.append(hidden_seq)
                lb_b.append(fr["gt_class"])
                if len(lb_b) >= args.batch_size:
                    _step(img_b, bbox_b, hidden_b, lb_b, model, opt, criterion, device, args.use_hidden_state)
                    img_b, bbox_b, hidden_b, lb_b = [], [], [], []

        if lb_b:
            _step(img_b, bbox_b, hidden_b, lb_b, model, opt, criterion, device, args.use_hidden_state)
        sched.step()

        if epoch % 10 == 0 or epoch == args.epochs:
            acc, per_class = evaluate(te_cache, te_eps, model, device, hidden_cache, args.use_hidden_state, args.window)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"{epoch:>6}  {acc:>8.4f}  {best_acc:>8.4f}", flush=True)

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    final_acc, per_class = evaluate(te_cache, te_eps, model, device, hidden_cache, args.use_hidden_state, args.window)

    print(f"\n{'='*55}", flush=True)
    print(f"  LSTMHidden 완료 (mode={args.use_hidden_state}, proj_dim={args.proj_dim})")
    print(f"  val_acc: {final_acc:.4f}", flush=True)
    print(f"{'='*55}", flush=True)

    ckpt_path = STAGE2_DIR / f"stage2_lstm_{args.use_hidden_state}.pt"
    torch.save({"model": best_state, "val_acc": final_acc, "proj_dim": args.proj_dim,
                "use_hidden_state": args.use_hidden_state}, str(ckpt_path))
    print(f"[SAVE] {ckpt_path}", flush=True)
    return final_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--window", type=int, default=WINDOW)
    p.add_argument("--data", default=str(DATA_PATH))
    p.add_argument("--use_hidden_state", default="none", choices=["none", "add", "replace"])
    p.add_argument("--proj_dim", type=int, default=32)
    args = p.parse_args()

    t0 = time.time()
    print("=" * 55, flush=True)
    print("LSTMHidden (RoboVLMs LSTM head + hidden-state ablation)")
    print(f"mode={args.use_hidden_state}  proj_dim={args.proj_dim}", flush=True)
    print("=" * 55, flush=True)
    train(args)
    print(f"\n소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()

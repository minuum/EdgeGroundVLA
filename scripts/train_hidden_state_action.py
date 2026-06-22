#!/usr/bin/env python3
"""
plan_20260622_hidden_state_action_head.md §3 — Exp54 Stage2 v2 Action Head를
PG2 hidden state(2304차원)로 재학습. train_exp54_stage2_v2_action.py의 복제본
(베이스라인 재현 가능성 보존) — 변경점은 --use_hidden_state 플래그 하나뿐,
나머지(seed=42 split, optimizer, scheduler, epoch, batch)는 100% 동일.

Step B(probe_v5_direction_hidden_state.py)가 검증한 가정과 동일하게,
현재 타임스텝 1프레임의 hidden state만 사용(윈도우 전체 아님) — 미리 추출된
캐시(docs/v5/hidden_state_cache/v5_hidden_states.npz)를 그냥 로드만 한다.
PG2 재로딩 없음, 학습 중 추가 forward 없음.

--use_hidden_state:
  none    기존과 동일(bbox 32 + img 256, baseline 재현용)
  add     기존 288 + hidden 2304 = 2592
  replace bbox 32 제거, img 256 + hidden 2304 = 2560

Usage:
  .venv/bin/python3 scripts/train_hidden_state_action.py --use_hidden_state add
  .venv/bin/python3 scripts/train_hidden_state_action.py --use_hidden_state replace
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

VLM_PATH     = ROOT / ".vlms" / "kosmos-2-patch14-224"
DATA_PATH    = ROOT / "docs" / "v5" / "bbox_nav_exp46" / "bbox_dataset_full.json"
STAGE1_V2    = ROOT / "runs" / "v5_nav" / "mlp" / "shared" / "stage1_v2_projs.pt"
STAGE2_DIR   = ROOT / "runs" / "v5_nav" / "mlp" / "exp_hidden_state" / "stage2_v2"
HIDDEN_CACHE = ROOT / "docs" / "v5" / "hidden_state_cache" / "v5_hidden_states.npz"

CLASS_NAMES = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "ROT_L", "ROT_R"]
NUM_CLASSES = 8
WINDOW      = 8
VIS_DIM     = 1024
PROJ_DIM    = 256
HIDDEN_DIM  = 2304
D_IN_BASE   = WINDOW * 4 + PROJ_DIM   # 288


# ─── 인코더 (train_exp54_stage2_v2_action.py와 동일) ───

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


# ─── 데이터 유틸 ───

def load_images(h5_path, indices):
    import io as _io
    with h5py.File(h5_path, "r") as f:
        imgs_ds = f["observations"]["images"]
        result = []
        for i in indices:
            raw = imgs_ds[i]
            if hasattr(raw, "dtype") and raw.dtype != object and raw.ndim >= 2:
                result.append(Image.fromarray(raw.astype("uint8")))
            else:
                arr = np.frombuffer(bytes(raw), dtype=np.uint8)
                result.append(Image.open(_io.BytesIO(arr)).convert("RGB"))
        return result


def bbox_feat(frames, t, window=WINDOW):
    arr = []
    for k in range(window):
        fr = frames[max(0, t - (window - 1 - k))]
        cx   = fr.get("cx",   fr.get("cx_det",   0.5))
        cy   = fr.get("cy",   fr.get("cy_det",   0.5))
        area = fr.get("area", fr.get("area_det", 0.05))
        has  = float(fr.get("has_bbox", fr.get("detected", False)))
        arr.extend([cx, cy, area, has])
    return np.array(arr, dtype=np.float32)


def load_hidden_cache():
    if not HIDDEN_CACHE.exists():
        raise FileNotFoundError(f"hidden state 캐시 없음: {HIDDEN_CACHE} — "
                                 "scripts/eval/extract_v5_hidden_states_full.py 먼저 실행")
    npz = np.load(HIDDEN_CACHE)
    return {k: npz[k] for k in npz.files}


def build_x(mode, bf, img_feat, hidden_vec):
    """mode별 최종 입력 벡터 구성. hidden_vec=None이면 0벡터로 대체(스킵된 episode 대비)."""
    if mode == "none":
        return torch.cat([bf, img_feat])
    h = torch.from_numpy(hidden_vec.astype(np.float32)) if hidden_vec is not None \
        else torch.zeros(HIDDEN_DIM, dtype=torch.float32)
    if mode == "add":
        return torch.cat([bf, img_feat, h])
    if mode == "replace":
        return torch.cat([img_feat, h])
    raise ValueError(mode)


def d_in_for(mode, window):
    base = window * 4 + PROJ_DIM
    if mode == "none":
        return base
    if mode == "add":
        return base + HIDDEN_DIM
    if mode == "replace":
        return PROJ_DIM + HIDDEN_DIM
    raise ValueError(mode)


# ─── Feature Pre-caching ───

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


def hidden_lookup(hidden_cache, ep_path, frame_idx):
    stem = Path(ep_path).stem
    key = f"{stem}__f{frame_idx}"
    return hidden_cache.get(key)


def filter_episodes_with_hidden(eps, hidden_cache):
    """hidden state가 하나도 없는 episode(추출 스킵분)는 mode!='none'일 때 제외."""
    kept, dropped = [], []
    for ep in eps:
        stem = Path(ep["episode"]).stem
        has_any = any(f"{stem}__f{fr['frame_idx']}" in hidden_cache for fr in ep["frames"])
        (kept if has_any else dropped).append(ep)
    return kept, dropped


# ─── 학습 ───

def _step(feats, labels, mlp, opt, criterion, device):
    x = torch.stack(feats).to(device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    opt.zero_grad()
    criterion(mlp(x), y).backward()
    opt.step()


@torch.no_grad()
def evaluate(te_cache, te_eps, mlp, device, hidden_cache, mode, window=WINDOW):
    mlp.eval()
    correct = total = 0
    from collections import defaultdict
    per_class = defaultdict(lambda: [0, 0])
    for ep in te_eps:
        feats = te_cache.get(ep["episode"])
        if feats is None:
            continue
        for t, fr in enumerate(ep["frames"]):
            bf = torch.tensor(bbox_feat(ep["frames"], t, window=window), dtype=torch.float32)
            hv = hidden_lookup(hidden_cache, ep["episode"], fr["frame_idx"]) if mode != "none" else None
            x = build_x(mode, bf, feats[t], hv).unsqueeze(0).to(device)
            p = mlp(x).argmax(1).item()
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
    print(f"[MODE] use_hidden_state={args.use_hidden_state}", flush=True)

    data = json.loads(Path(args.data).read_text())
    ep_labels = [ep["path_type"] for ep in data]

    from collections import Counter
    label_counts = Counter(ep_labels)
    can_stratify = all(c >= 2 for c in label_counts.values())
    if can_stratify:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_idx, te_idx = next(sss.split(np.zeros(len(data)), ep_labels))
    else:
        from sklearn.model_selection import ShuffleSplit
        ss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        tr_idx, te_idx = next(ss.split(np.zeros(len(data))))
    tr_eps = [data[i] for i in tr_idx]
    te_eps = [data[i] for i in te_idx]
    print(f"Train: {len(tr_eps)} ep  Val: {len(te_eps)} ep  (baseline과 동일 seed=42 split)", flush=True)

    hidden_cache = {}
    if args.use_hidden_state != "none":
        hidden_cache = load_hidden_cache()
        print(f"[HIDDEN] 캐시 로드 — {len(hidden_cache)}개 프레임", flush=True)
        tr_before, te_before = len(tr_eps), len(te_eps)
        tr_eps, tr_dropped = filter_episodes_with_hidden(tr_eps, hidden_cache)
        te_eps, te_dropped = filter_episodes_with_hidden(te_eps, hidden_cache)
        if tr_dropped or te_dropped:
            print(f"[HIDDEN] hidden state 없는 episode 제외: train {tr_before}->{len(tr_eps)}, "
                  f"val {te_before}->{len(te_eps)}", flush=True)

    enc = FrozenCLIPV2(VLM_PATH, STAGE1_V2, device).to(device).eval()
    tr_cache = precompute_features(enc, tr_eps, device, "train")
    te_cache = precompute_features(enc, te_eps, device, "val")
    del enc
    torch.cuda.empty_cache() if device.type == "cuda" else None
    print("[CACHE] VLM 해제 완료 — MLP만 학습", flush=True)

    window = args.window
    d_in = d_in_for(args.use_hidden_state, window)
    mlp = ActionMLP(d_in=d_in).to(device)
    print(f"[HEAD] ActionMLP  window={window}  d_in={d_in}", flush=True)

    all_labels = [fr["gt_class"] for ep in tr_eps for fr in ep["frames"]]
    counts = np.bincount(all_labels, minlength=NUM_CLASSES).astype(float)
    weights = np.where(counts > 0, 1.0 / (counts + 1e-6), 0.0)
    weights /= weights.sum() / NUM_CLASSES
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    print(f"[LOSS] class weights: {[f'{w:.2f}' for w in weights]}", flush=True)

    opt = torch.optim.AdamW(mlp.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc, best_state = 0.0, None
    STAGE2_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'epoch':>6} {'val_acc':>9} {'best':>9}", flush=True)
    print("-" * 30, flush=True)

    for epoch in range(1, args.epochs + 1):
        mlp.train()
        np.random.shuffle(tr_eps)
        bf_batch, lb_batch = [], []

        for ep in tr_eps:
            feats = tr_cache.get(ep["episode"])
            if feats is None:
                continue
            for t, fr in enumerate(ep["frames"]):
                bf = torch.tensor(bbox_feat(ep["frames"], t, window=window), dtype=torch.float32)
                hv = hidden_lookup(hidden_cache, ep["episode"], fr["frame_idx"]) if args.use_hidden_state != "none" else None
                x_t = build_x(args.use_hidden_state, bf, feats[t], hv)
                bf_batch.append(x_t)
                lb_batch.append(fr["gt_class"])
                if len(lb_batch) >= args.batch_size:
                    _step(bf_batch, lb_batch, mlp, opt, criterion, device)
                    bf_batch, lb_batch = [], []

        if bf_batch:
            _step(bf_batch, lb_batch, mlp, opt, criterion, device)
        sched.step()

        if epoch % 10 == 0 or epoch == args.epochs:
            acc, per_class = evaluate(te_cache, te_eps, mlp, device, hidden_cache, args.use_hidden_state, window=window)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
            print(f"{epoch:>6}  {acc:>8.4f}  {best_acc:>8.4f}", flush=True)

    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
    mlp.load_state_dict(best_state)
    final_acc, per_class = evaluate(te_cache, te_eps, mlp, device, hidden_cache, args.use_hidden_state, window=window)

    print(f"\n{'='*55}", flush=True)
    print(f"  Hidden-State Action Head 완료 (mode={args.use_hidden_state})")
    print(f"  val_acc: {final_acc:.4f}")
    print(f"  참고: Exp54 Step2(bbox+image) baseline PM=0.759")
    print(f"{'='*55}", flush=True)
    print(f"\n  클래스별 정확도:")
    for i, name in enumerate(CLASS_NAMES):
        c, t = per_class[i]
        a = c / t * 100 if t > 0 else 0.0
        print(f"    {name:<8}: {a:>6.1f}%  ({c}/{t})")

    ckpt_path = STAGE2_DIR / f"stage2_hidden_{args.use_hidden_state}.pt"
    torch.save({"mlp": best_state, "val_acc": final_acc, "d_in": d_in,
                "use_hidden_state": args.use_hidden_state, "window": window}, str(ckpt_path))
    print(f"\n[SAVE] {ckpt_path}", flush=True)
    return final_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",          type=int,   default=300)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--lr",              type=float, default=1e-3)
    p.add_argument("--window",          type=int,   default=WINDOW)
    p.add_argument("--data",            default=str(DATA_PATH))
    p.add_argument("--use_hidden_state", default="add", choices=["none", "add", "replace"])
    args = p.parse_args()

    t0 = time.time()
    print("=" * 55, flush=True)
    print("Hidden-State Action Head (plan_20260622_hidden_state_action_head.md)")
    print(f"mode={args.use_hidden_state}  epochs={args.epochs}", flush=True)
    print("=" * 55, flush=True)
    train(args)
    print(f"\n소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()

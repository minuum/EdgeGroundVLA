#!/usr/bin/env python3
"""
Exp54 Stage 2 v2: Action Head (Stage 1 v2 frozen)

Stage 1 v2 (frozen base Kosmos-2 + image_proj)를 고정하고 action head만 학습.
v1 Stage 2와의 차이:
  - encoder: LoRA-CLIP(1024) → frozen CLIP + image_proj(256)
  - D_IN: 1056 → 288 (bbox 32 + proj_feat 256)
  - 더 작고 깔끔한 표현 (basket 위치 정렬된 feature)

Usage:
  .venv/bin/python3 scripts/train_exp54_stage2_v2_action.py
  .venv/bin/python3 scripts/train_exp54_stage2_v2_action.py --epochs 300
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

VLM_PATH   = ROOT / ".vlms" / "kosmos-2-patch14-224"
DATA_PATH  = ROOT / "docs" / "v5" / "bbox_nav_exp46" / "bbox_dataset_full.json"
STAGE1_V2  = ROOT / "runs" / "v5_nav" / "mlp" / "exp54" / "stage1_v2" / "stage1_v2_projs.pt"
STAGE2_DIR = ROOT / "runs" / "v5_nav" / "mlp" / "exp54" / "stage2_v2"

CLASS_NAMES = ["STOP","FORWARD","LEFT","RIGHT","FWD+L","FWD+R","ROT_L","ROT_R"]
NUM_CLASSES = 8
WINDOW      = 8
VIS_DIM     = 1024
PROJ_DIM    = 256
D_IN        = WINDOW * 4 + PROJ_DIM   # 288


# ─── 인코더 ─────────────────────────────────────────

class FrozenCLIPV2(nn.Module):
    """Stage 1 v2: frozen base Kosmos-2 + trained image_proj (완전 frozen)."""

    def __init__(self, vlm_path, ckpt_path, device):
        super().__init__()
        from transformers import AutoModelForVision2Seq, AutoProcessor

        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        print(f"[MODEL] Stage1 v2 val_acc={ckpt['val_acc']:.4f}", flush=True)

        self.processor = AutoProcessor.from_pretrained(str(vlm_path))
        base = AutoModelForVision2Seq.from_pretrained(
            str(vlm_path), torch_dtype=torch.float16
        )
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
        """PIL 리스트 → (N, 256) L2-normalized. 배치 처리."""
        all_feats = []
        for i in range(0, len(pil_images), batch):
            imgs = pil_images[i:i+batch]
            inputs = self.processor(images=imgs, return_tensors="pt")
            pv = inputs["pixel_values"].to(device, dtype=torch.float16)
            out = self.vision_model(pixel_values=pv)
            feat = out.last_hidden_state.mean(dim=1).float()
            all_feats.append(F.normalize(self.image_proj(feat), dim=-1))
        return torch.cat(all_feats, dim=0)  # (N, 256)


# ─── Action MLP ──────────────────────────────────────

class ActionMLP(nn.Module):
    def __init__(self, d_in=D_IN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 128),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),   nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64,  NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x)


# ─── 데이터 유틸 ─────────────────────────────────────

def load_images(h5_path, indices):
    with h5py.File(h5_path, "r") as f:
        return [Image.fromarray(f["observations"]["images"][i]) for i in indices]


def bbox_feat(frames, t, augment=False, rng=None, aug=None):
    """WINDOW개 프레임의 (cx,cy,area,has_bbox) → (WINDOW*4,) 벡터.

    augment=True 이면 VLM(PaliGemma2) grounding bbox 분포를 모사하는
    노이즈를 주입한다 (학습 전용). Exp60: HSV GT만 학습 → VLM bbox OOD 문제 완화.
    aug = {mu_x, sd_x, mu_y, sd_y, area_mu, area_sd, area_lo, area_hi, miss_p}
    """
    arr = []
    for k in range(WINDOW):
        fr = frames[max(0, t - (WINDOW - 1 - k))]
        cx, cy, area, has = fr["cx"], fr["cy"], fr["area"], float(fr["has_bbox"])
        if augment and aug is not None and has > 0.5:
            cx += rng.normal(aug["mu_x"], aug["sd_x"])          # 계통 오프셋 + 지터
            cy += rng.normal(aug["mu_y"], aug["sd_y"])
            ratio = rng.normal(aug["area_mu"], aug["area_sd"])  # area 스케일 모사
            ratio = float(np.clip(ratio, aug["area_lo"], aug["area_hi"]))
            area  = max(0.0, area * ratio)
            cx = float(np.clip(cx, 0.0, 1.0))
            cy = float(np.clip(cy, 0.0, 1.0))
            if rng.random() < aug["miss_p"]:                    # 가끔 미검출 모사
                has = 0.0
        arr.extend([cx, cy, area, has])
    return np.array(arr, dtype=np.float32)


def build_aug_params(stats_path, noise_scale):
    """측정 통계(exp60_bbox_offset_stats.json) → 증강 파라미터.
    noise_scale: 산포(std)와 miss_p를 스케일 (계통 오프셋 mean은 유지).
    """
    s = json.loads(Path(stats_path).read_text())
    dcx, dcy, ar = s["delta_cx"], s["delta_cy"], s["area_ratio"]
    aug = {
        "mu_x":    dcx["mean"],
        "sd_x":    dcx["std"] * noise_scale,
        "mu_y":    dcy["mean"],
        "sd_y":    dcy["std"] * noise_scale,
        "area_mu": ar["mean"],
        "area_sd": ar["std"] * noise_scale,
        "area_lo": ar["p05"],
        "area_hi": ar["p95"],
        "miss_p":  s["miss_rate"] * noise_scale,
    }
    return aug


# ─── Feature Pre-caching ─────────────────────────────

def precompute_features(enc, eps, device, label):
    """모든 에피소드 이미지를 한 번만 encode → dict 캐시 반환.
    캐시 키: episode path (str)
    캐시 값: Tensor (N_frames, PROJ_DIM) on CPU
    """
    cache = {}
    n = len(eps)
    print(f"[CACHE] {label} feature 사전 추출 중 ({n} episodes)...", flush=True)
    t0 = time.time()
    for i, ep in enumerate(eps):
        try:
            imgs = load_images(ep["episode"], list(range(len(ep["frames"]))))
        except Exception as e:
            print(f"  skip {ep['episode']}: {e}", flush=True)
            cache[ep["episode"]] = None
            continue
        feats = enc.encode_batch(imgs, device)  # (N, 256) on device
        cache[ep["episode"]] = feats.cpu()       # CPU에 보관
        if (i + 1) % 20 == 0 or (i + 1) == n:
            print(f"  {i+1}/{n} done ({time.time()-t0:.0f}s)", flush=True)
    print(f"[CACHE] {label} 완료 — {time.time()-t0:.1f}초", flush=True)
    return cache


# ─── 학습 ─────────────────────────────────────────────

def _step(feats, labels, mlp, opt, criterion, device):
    x = torch.stack(feats).to(device)
    y = torch.tensor(labels, dtype=torch.long, device=device)
    opt.zero_grad()
    criterion(mlp(x), y).backward()
    opt.step()


@torch.no_grad()
def evaluate(tr_cache, te_eps, mlp, device):
    mlp.eval()
    correct = total = 0
    from collections import defaultdict
    per_class = defaultdict(lambda: [0, 0])
    for ep in te_eps:
        feats = tr_cache.get(ep["episode"])
        if feats is None:
            continue
        for t, fr in enumerate(ep["frames"]):
            bf = torch.tensor(bbox_feat(ep["frames"], t), dtype=torch.float32)
            x  = torch.cat([bf, feats[t]]).unsqueeze(0).to(device)
            p  = mlp(x).argmax(1).item()
            g  = fr["gt_class"]
            per_class[g][0] += int(p == g)
            per_class[g][1] += 1
            correct += int(p == g)
            total   += 1
    acc = correct / total if total > 0 else 0.0
    return acc, per_class


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[DEVICE] {device}", flush=True)

    data = json.loads(DATA_PATH.read_text())
    ep_labels = [ep["path_type"] for ep in data]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(sss.split(np.zeros(len(data)), ep_labels))
    tr_eps = [data[i] for i in tr_idx]
    te_eps = [data[i] for i in te_idx]
    print(f"Train: {len(tr_eps)} ep  Val: {len(te_eps)} ep", flush=True)

    enc = FrozenCLIPV2(VLM_PATH, STAGE1_V2, device).to(device).eval()

    # ── 핵심: 한 번만 encode ──────────────────────────
    tr_cache = precompute_features(enc, tr_eps, device, "train")
    te_cache = precompute_features(enc, te_eps, device, "val")
    # VLM 더 이상 불필요 — GPU/CPU 메모리 해제
    del enc
    torch.cuda.empty_cache() if device.type == "cuda" else None
    print("[CACHE] VLM 해제 완료 — MLP만 학습", flush=True)
    # ─────────────────────────────────────────────────

    mlp = ActionMLP(D_IN).to(device)

    # ── Exp60: VLM bbox 분포 모사 증강 ────────────────
    aug, aug_rng = None, None
    if args.augment:
        aug = build_aug_params(args.aug_stats, args.noise_scale)
        aug_rng = np.random.default_rng(args.seed)
        print(f"[AUG] noise_scale={args.noise_scale} aug={ {k: round(v,4) for k,v in aug.items()} }", flush=True)
    # ──────────────────────────────────────────────────

    all_labels = [fr["gt_class"] for ep in tr_eps for fr in ep["frames"]]
    counts  = np.bincount(all_labels, minlength=NUM_CLASSES).astype(float)
    weights = np.where(counts > 0, 1.0 / (counts + 1e-6), 0.0)
    weights /= weights.sum() / NUM_CLASSES
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
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
                bf = torch.tensor(
                    bbox_feat(ep["frames"], t, augment=args.augment, rng=aug_rng, aug=aug),
                    dtype=torch.float32,
                )
                bf_batch.append(torch.cat([bf, feats[t]]))
                lb_batch.append(fr["gt_class"])
                if len(lb_batch) >= args.batch_size:
                    _step(bf_batch, lb_batch, mlp, opt, criterion, device)
                    bf_batch, lb_batch = [], []

        if bf_batch:
            _step(bf_batch, lb_batch, mlp, opt, criterion, device)
        sched.step()

        if epoch % 10 == 0 or epoch == args.epochs:
            acc, per_class = evaluate(te_cache, te_eps, mlp, device)
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
            print(f"{epoch:>6}  {acc:>8.4f}  {best_acc:>8.4f}", flush=True)

    # 최종 평가 (best 모델)
    mlp.load_state_dict(best_state)
    final_acc, per_class = evaluate(te_cache, te_eps, mlp, device)

    print(f"\n{'='*55}", flush=True)
    print(f"  Exp54 Stage 2 v2 완료")
    print(f"  val_acc: {final_acc:.4f}")
    print(f"  참고: Exp49={0.9640:.4f}  Exp53={0.9468:.4f}")
    print(f"{'='*55}", flush=True)
    print(f"\n  클래스별 정확도:")
    for i, name in enumerate(CLASS_NAMES):
        c, t = per_class[i]
        a = c / t * 100 if t > 0 else 0.0
        print(f"    {name:<8}: {a:>6.1f}%  ({c}/{t})")

    if args.augment:
        tag = args.tag or f"aug{args.noise_scale:g}"
        ckpt_path = STAGE2_DIR / f"stage2_v2_mlp_{tag}.pt"   # 기존 baseline 덮어쓰지 않음
    else:
        ckpt_path = STAGE2_DIR / "stage2_v2_mlp.pt"
    torch.save({"mlp": best_state, "val_acc": final_acc, "d_in": D_IN,
                "augment": args.augment, "noise_scale": args.noise_scale if args.augment else 0.0,
                "aug_params": aug}, str(ckpt_path))
    print(f"\n[SAVE] {ckpt_path}", flush=True)
    return final_acc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=300)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-3)
    # ── Exp60: VLM bbox 분포 증강 ──
    p.add_argument("--augment",     action="store_true",
                   help="VLM grounding bbox 분포 모사 노이즈 주입 (학습 전용)")
    p.add_argument("--aug-stats",   default=str(ROOT / "docs/v5/exp60_bbox_offset_stats.json"),
                   help="measure_exp60_bbox_offset.py 산출 통계 JSON")
    p.add_argument("--noise-scale", type=float, default=1.0,
                   help="산포(std)·miss_p 스케일 (sweep용: 0.5/1.0/1.5)")
    p.add_argument("--seed",        type=int,   default=42, help="증강 rng 시드")
    p.add_argument("--tag",         default=None, help="ckpt suffix (기본: aug{noise_scale})")
    args = p.parse_args()

    t0 = time.time()
    print("=" * 55, flush=True)
    print("Exp54 Stage 2 v2: Action Head (Stage 1 v2 frozen)")
    print(f"D_IN={D_IN} (bbox32 + proj256)  epochs={args.epochs}")
    print("=" * 55, flush=True)
    train(args)
    print(f"\n소요: {(time.time()-t0)/60:.1f}분", flush=True)


if __name__ == "__main__":
    main()

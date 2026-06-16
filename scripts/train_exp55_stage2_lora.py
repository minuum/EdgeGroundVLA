#!/usr/bin/env python3
"""
Exp55 Stage 2: LoRA-enhanced CLIP + Action MLP

Stage1 LoRA 체크포인트를 불러와 시각 특징을 추출하고
Stage2 MLP(bbox + visual → action)를 학습.

Stage2 v2와의 차이:
  - FrozenCLIPWithLoRA: LoRA가 merge된 vision encoder 사용
  - LoRA 학습으로 basket 방향에 정렬된 시각 특징 → MLP 성능 향상 기대

Usage:
  .venv/bin/python3 scripts/train_exp55_stage2_lora.py
  .venv/bin/python3 scripts/train_exp55_stage2_lora.py --epochs 300
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

VLM_PATH      = ROOT / ".vlms" / "kosmos-2-patch14-224"
DATA_PATH     = ROOT / "docs" / "v5" / "bbox_nav_exp46" / "bbox_dataset_full.json"
STAGE1_CKPT   = ROOT / "runs" / "v5_nav" / "mlp" / "exp55" / "stage1_lora" / "stage1_lora_projs.pt"
LORA_ADAPTER  = ROOT / "runs" / "v5_nav" / "mlp" / "exp55" / "stage1_lora" / "lora_adapter"
STAGE2_DIR    = ROOT / "runs" / "v5_nav" / "mlp" / "exp55" / "stage2_lora"

CLASS_NAMES = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "ROT_L", "ROT_R"]
NUM_CLASSES = 8
WINDOW      = 8
VIS_DIM     = 1024
PROJ_DIM    = 256
D_IN        = WINDOW * 4 + PROJ_DIM  # 288


# ─── LoRA-enhanced 인코더 ─────────────────────────────────

class FrozenCLIPWithLoRA(nn.Module):
    """Stage1 LoRA adapter + image_proj 로드, 모두 frozen."""

    def __init__(self, vlm_path, stage1_ckpt_path, lora_adapter_dir, device):
        super().__init__()
        from transformers import AutoModelForVision2Seq, AutoProcessor
        from peft import PeftModel

        ckpt = torch.load(str(stage1_ckpt_path), map_location=device, weights_only=False)
        print(f"[MODEL] Stage1 LoRA val_acc={ckpt['val_acc']:.4f}", flush=True)

        self.processor = AutoProcessor.from_pretrained(str(vlm_path))

        # 풀 모델에 LoRA adapter 로드 (키 경로가 stage1 저장과 일치)
        print("[MODEL] LoRA adapter 로드 중...", flush=True)
        base = AutoModelForVision2Seq.from_pretrained(str(vlm_path), torch_dtype=torch.float16)
        peft_model = PeftModel.from_pretrained(base, str(lora_adapter_dir))

        # LoRA 가중치를 base에 merge → 단순한 일반 모델로 변환 (추론 속도 향상)
        merged = peft_model.merge_and_unload()
        self.vision_model = merged.vision_model.to(device)

        self.image_proj = nn.Linear(VIS_DIM, PROJ_DIM).to(device)
        self.image_proj.load_state_dict(ckpt["image_proj"])

        for p in self.vision_model.parameters():
            p.requires_grad = False
        for p in self.image_proj.parameters():
            p.requires_grad = False

        print("[MODEL] frozen 완료 (LoRA-merged vision + image_proj)", flush=True)

    @torch.no_grad()
    def encode_batch(self, pil_images, device, batch=32):
        """PIL 리스트 → (N, 256) L2-normalized."""
        all_feats = []
        for i in range(0, len(pil_images), batch):
            imgs = pil_images[i:i+batch]
            inputs = self.processor(images=imgs, return_tensors="pt")
            pv = inputs["pixel_values"].to(device, dtype=torch.float16)
            out = self.vision_model(pixel_values=pv)
            feat = out.last_hidden_state.mean(dim=1).float()
            all_feats.append(F.normalize(self.image_proj(feat), dim=-1))
        return torch.cat(all_feats, dim=0)  # (N, 256)


# ─── Action MLP ───────────────────────────────────────────

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


# ─── 데이터 유틸 ──────────────────────────────────────────

def load_images(h5_path, indices):
    with h5py.File(h5_path, "r") as f:
        return [Image.fromarray(f["observations"]["images"][i]) for i in indices]


def bbox_feat(frames, t):
    arr = []
    for k in range(WINDOW):
        fr = frames[max(0, t - (WINDOW - 1 - k))]
        arr.extend([fr["cx"], fr["cy"], fr["area"], float(fr["has_bbox"])])
    return np.array(arr, dtype=np.float32)


def precompute_features(enc, eps, device, label):
    cache = {}
    n = len(eps)
    print(f"[CACHE] {label} feature 추출 중 ({n} episodes)...", flush=True)
    t0 = time.time()
    for i, ep in enumerate(eps):
        try:
            imgs = load_images(ep["episode"], list(range(len(ep["frames"]))))
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


# ─── 학습/평가 ────────────────────────────────────────────

@torch.no_grad()
def evaluate(te_cache, te_eps, mlp, device):
    mlp.eval()
    correct = total = 0
    from collections import defaultdict
    per_class = defaultdict(lambda: [0, 0])
    for ep in te_eps:
        feats = te_cache.get(ep["episode"])
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

    if not STAGE1_CKPT.exists():
        print(f"[ERROR] Stage1 checkpoint 없음: {STAGE1_CKPT}")
        print(f"  먼저 실행: .venv/bin/python3 scripts/train_exp55_stage1_lora.py")
        return
    if not LORA_ADAPTER.exists():
        print(f"[ERROR] LoRA adapter 없음: {LORA_ADAPTER}")
        return

    data = json.loads(DATA_PATH.read_text())
    ep_labels = [ep["path_type"] for ep in data]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(sss.split(np.zeros(len(data)), ep_labels))
    tr_eps = [data[i] for i in tr_idx]
    te_eps = [data[i] for i in te_idx]
    print(f"Train: {len(tr_eps)} ep  Val: {len(te_eps)} ep", flush=True)

    enc = FrozenCLIPWithLoRA(VLM_PATH, STAGE1_CKPT, LORA_ADAPTER, device).to(device).eval()

    tr_cache = precompute_features(enc, tr_eps, device, "train")
    te_cache = precompute_features(enc, te_eps, device, "val")
    del enc
    torch.cuda.empty_cache()
    print("[CACHE] VLM 해제 완료 — MLP만 학습", flush=True)

    mlp = ActionMLP(D_IN).to(device)

    all_labels = [fr["gt_class"] for ep in tr_eps for fr in ep["frames"]]
    counts  = np.bincount(all_labels, minlength=NUM_CLASSES).astype(float)
    weights = np.where(counts > 0, 1.0 / (counts + 1e-6), 0.0)
    weights /= weights.sum() / NUM_CLASSES
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )
    print(f"[LOSS] class weights: {[f'{w:.2f}' for w in weights]}", flush=True)

    opt   = torch.optim.AdamW(mlp.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_acc, best_state = 0.0, None
    STAGE2_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'epoch':>6} {'val_acc':>9} {'best':>9}", flush=True)
    print("-" * 30)

    for epoch in range(1, args.epochs + 1):
        mlp.train()
        batch_feats, batch_labels = [], []

        for ep in tr_eps:
            feats = tr_cache.get(ep["episode"])
            if feats is None:
                continue
            for t, fr in enumerate(ep["frames"]):
                bf = torch.tensor(bbox_feat(ep["frames"], t), dtype=torch.float32)
                x  = torch.cat([bf, feats[t]])
                batch_feats.append(x)
                batch_labels.append(fr["gt_class"])

                if len(batch_labels) >= args.batch_size:
                    xs = torch.stack(batch_feats).to(device)
                    ys = torch.tensor(batch_labels, dtype=torch.long, device=device)
                    opt.zero_grad()
                    criterion(mlp(xs), ys).backward()
                    opt.step()
                    batch_feats, batch_labels = [], []

        if batch_feats:
            xs = torch.stack(batch_feats).to(device)
            ys = torch.tensor(batch_labels, dtype=torch.long, device=device)
            opt.zero_grad()
            criterion(mlp(xs), ys).backward()
            opt.step()

        sched.step()

        acc, per_class = evaluate(te_cache, te_eps, mlp, device)

        if acc > best_acc:
            best_acc   = acc
            best_state = {
                "mlp":       {k: v.cpu().clone() for k, v in mlp.state_dict().items()},
                "val_acc":   acc,
                "epoch":     epoch,
            }
            torch.save(best_state, str(STAGE2_DIR / "stage2_lora_mlp.pt"))
            mark = "  ← best"
        else:
            mark = ""

        if epoch % 10 == 0 or epoch <= 5:
            print(f"{epoch:>6}  {acc:>8.4f}  {best_acc:>8.4f}{mark}", flush=True)
        elif mark:
            print(f"{epoch:>6}  {acc:>8.4f}  {best_acc:>8.4f}{mark}", flush=True)

    # 최종 결과
    print(f"\n{'='*55}")
    print(f"Exp55 Stage2 LoRA 결과")
    print(f"{'='*55}")
    print(f"  LoRA 없는 Stage2 v2 val_acc: 0.9259  (92.6%)")
    print(f"  LoRA 있는 Stage2 LoRA val_acc: {best_acc:.4f}  ({best_acc:.1%})")
    diff = best_acc - 0.9259
    print(f"  차이: {diff:+.4f}  ({diff:+.1%})")
    print(f"\n  per class:")
    for cls_idx, name in enumerate(CLASS_NAMES):
        v = per_class.get(cls_idx, [0, 0])
        if v[1] > 0:
            print(f"    {name:<10} {v[0]/v[1]:.1%}  ({v[0]}/{v[1]})")
    print(f"\n[CKPT] {STAGE2_DIR / 'stage2_lora_mlp.pt'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs",     type=int,   default=200)
    p.add_argument("--lr",         type=float, default=5e-4)
    p.add_argument("--batch-size", type=int,   default=128)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()

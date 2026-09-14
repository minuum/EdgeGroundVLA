#!/usr/bin/env python3
"""
심사위원1 의견1 / 심사위원3 의견4 대응: OWLv2 단독 vs Kosmos-2 단독 vs 융합(배포) ablation.

exp73 배포 조건과 완전히 동일한 데이터·분할·헤드 구조(MLP, window=6, bbox_scale=3.0)에서
입력 특징만 3갈래로 나눠 비교한다:
  bbox_only  — OWLv2 bbox 4채널만 (cx, cy, area, has_bbox) × window=6 = 24-dim
  image_only — Kosmos-2 vision_model → image_proj 256채널만 × window=6 = 1536-dim
  fused      — 배포와 동일한 260채널(4+256) × window=6 = 1560-dim

캐시(exp73_v6_vis_cache_stage1v3.pt)의 vis 특징은 그라운더와 무관(Kosmos-2 전용)하므로 그대로
재사용하고, bbox만 bbox_dataset_v6_owl.json(OWLv2 검출)으로 교체한다 — 이는 실제 배포 체크포인트
(exp73_owl_stage1v3_v6_mlp.pt) 학습 시와 동일한 절차(train_exp73_stage1v3_heads.py --ann-v6 참조).

Usage:
  .venv/bin/python3 scripts/ablate_owlv2_kosmos2_fusion.py
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CACHE_V6 = ROOT / "docs/v5/closed_loop_eval/exp73_v6_vis_cache_stage1v3.pt"
ANN_OWL = ROOT / "docs/v5/bbox_nav_owl/bbox_dataset_v6_owl.json"
OUT_JSON = ROOT / "docs/v5/bbox_nav_owl/ablation_owlv2_kosmos2_fusion.json"

WINDOW = 6
BBOX_SCALE = 3.0
NUM_CLASSES = 8
PROJ_DIM = 256
VAL_RATIO = 0.15
SPLIT_SEED = 42
SEEDS = [0, 1, 2]
EPOCHS = 300
LR = 5e-4
CLASS_NAMES = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "ROT_L", "ROT_R"]


def load_episodes_with_owl_bbox():
    """CACHE_V6의 vis(Kosmos-2, 그라운더 무관)는 그대로 두고 bbox만 OWLv2로 교체."""
    v6_eps = torch.load(str(CACHE_V6), weights_only=False)
    with open(ANN_OWL) as f:
        ann = json.load(f)
    ann_by_stem = {Path(e["episode"]).stem: e for e in ann}
    replaced = 0
    for ep in v6_eps:
        src = ann_by_stem.get(ep["stem"])
        if src is None:
            continue
        frames = [fr for fr in src["frames"] if fr.get("gt_class") is not None]
        ep["bboxes"] = [(fr.get("cx_det", 0.5), fr.get("cy_det", 0.5),
                         fr.get("area_det", 0.05), float(fr.get("has_bbox", False)))
                        for fr in frames]
        replaced += 1
    print(f"[ANN] OWLv2 bbox 교체: {replaced}/{len(v6_eps)}ep")
    return v6_eps


def build_windows(eps, window=WINDOW, bbox_scale=BBOX_SCALE):
    """exp73 배포와 동일 규격의 260-dim(4 bbox + 256 vision) 윈도우 특징."""
    X, y = [], []
    for ep in eps:
        bboxes, vis, gts = ep["bboxes"], ep["vis"], ep["gts"]
        for t in range(len(gts)):
            seq = []
            for k in range(window):
                idx = max(0, t - (window - 1 - k))
                seq.append([v * bbox_scale for v in bboxes[idx]] + vis[idx].tolist())
            X.append(seq)
            y.append(gts[t])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


class MLPHead(nn.Module):
    """배포 MLPActionHead와 동일 구조(512->128->8), 입력 차원만 가변."""
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, NUM_CLASSES))

    def forward(self, x):
        return self.net(x.flatten(1))


def slice_condition(X, condition):
    """X: (N, window, 260) -> bbox_only(24), image_only(1536), fused(1560)."""
    if condition == "bbox_only":
        return X[:, :, :4]
    if condition == "image_only":
        return X[:, :, 4:]
    return X  # fused


def train_one(X_tr, y_tr, X_va, y_va, seed, in_dim):
    torch.manual_seed(seed)
    np.random.seed(seed)
    cls_counts = np.bincount(y_tr, minlength=NUM_CLASSES).astype(np.float32)
    cls_counts = np.where(cls_counts == 0, 1.0, cls_counts)
    weights = 1.0 / cls_counts
    weights = weights / weights.sum() * NUM_CLASSES
    weights_t = torch.tensor(weights, dtype=torch.float32, device=DEVICE)

    X_tr_t = torch.tensor(X_tr, device=DEVICE)
    y_tr_t = torch.tensor(y_tr, device=DEVICE)
    X_va_t = torch.tensor(X_va, device=DEVICE)
    y_va_t = torch.tensor(y_va, device=DEVICE)

    model = MLPHead(in_dim).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    best_acc, best_pred = 0.0, None
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(X_tr_t), device=DEVICE)
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            loss = F.cross_entropy(model(X_tr_t[b]), y_tr_t[b], weight=weights_t)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 25 == 0 or ep == EPOCHS - 1:
            model.eval()
            with torch.no_grad():
                pred = model(X_va_t).argmax(1)
                acc = (pred == y_va_t).float().mean().item()
            if acc >= best_acc:
                best_acc, best_pred = acc, pred.cpu().numpy()
    per_class = {}
    for c in range(NUM_CLASSES):
        m = (y_va == c)
        if m.sum() > 0:
            per_class[CLASS_NAMES[c]] = float((best_pred[m] == c).mean())
    return best_acc, per_class


def main():
    eps = load_episodes_with_owl_bbox()
    rng = np.random.default_rng(SPLIT_SEED)
    idx = list(range(len(eps)))
    rng.shuffle(idx)
    n_val = max(1, int(len(idx) * VAL_RATIO))
    val_eps = [eps[i] for i in idx[:n_val]]
    train_eps = [eps[i] for i in idx[n_val:]]
    print(f"[SPLIT] train={len(train_eps)} / val={len(val_eps)} episodes (동일 seed=42, exp73과 공통 val)")

    X_tr_full, y_tr = build_windows(train_eps)
    X_va_full, y_va = build_windows(val_eps)
    print(f"[DATA] train frames={len(X_tr_full)} / val frames={len(X_va_full)}")

    conditions = {
        "bbox_only": 4 * WINDOW,      # OWLv2 단독
        "image_only": PROJ_DIM * WINDOW,  # Kosmos-2 vision 단독
        "fused": (4 + PROJ_DIM) * WINDOW,  # 배포 구성(현재)
    }

    results = {}
    for cond, in_dim in conditions.items():
        X_tr = slice_condition(X_tr_full, cond)
        X_va = slice_condition(X_va_full, cond)
        accs, per_class_list = [], []
        for seed in SEEDS:
            acc, per_class = train_one(X_tr, y_tr, X_va, y_va, seed, in_dim)
            accs.append(acc)
            per_class_list.append(per_class)
            print(f"  [{cond}] seed={seed} val_acc={acc*100:.2f}%", flush=True)
        results[cond] = {
            "in_dim": in_dim,
            "val_acc_mean": float(np.mean(accs)),
            "val_acc_std": float(np.std(accs)),
            "val_acc_seeds": accs,
            "per_class_best_seed": per_class_list[int(np.argmax(accs))],
        }
        print(f"[{cond}] mean={np.mean(accs)*100:.2f}±{np.std(accs)*100:.2f}%")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n결과 저장 → {OUT_JSON}")
    print("\n=== 요약 (심사위원1 의견1 / 심사위원3 의견4 대응) ===")
    for cond, r in results.items():
        print(f"  {cond:12s} in_dim={r['in_dim']:5d}  val_acc={r['val_acc_mean']*100:5.2f}±{r['val_acc_std']*100:.2f}%")


if __name__ == "__main__":
    main()

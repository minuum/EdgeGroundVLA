#!/usr/bin/env python3
"""
Exp49: Language-Grounded Goal Navigation
  Exp46 캐시 재사용 + goal_pos (cx0, cy0, area0) 추가
  instruction_emb(2048) 제거 → grounded goal(3)으로 교체

  입력: bbox(8×4=32) + vision(1024) + goal(3) = 1059-dim
  출력: 8-class action

  goal = 에피소드 시작 프레임의 grounded 바구니 위치 (cx0, cy0, area0)
  → 다른 언어 표현도 같은 물체 grounding → 동일 cx0 → 동일 행동 (paraphrase-robust)

Usage:
  python3 scripts/train_v5_exp49_goal_nav.py
"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXP46_DIR = ROOT / "docs" / "v5" / "bbox_nav_exp46"
OUT_DIR   = ROOT / "docs" / "v5" / "bbox_nav_exp49"
MLP_DIR   = ROOT / "runs" / "v5_nav" / "mlp" / "exp49"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MLP_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PATH = MLP_DIR / "exp49_mlp.pt"

CLASS_NAMES = ["STOP","FORWARD","LEFT","RIGHT","FWD+L","FWD+R","ROT_L","ROT_R"]
NUM_CLASSES = 8
WINDOW      = 8
VIS_DIM     = 1024
GOAL_DIM    = 3          # (cx0, cy0, area0)
D_IN        = WINDOW * 4 + VIS_DIM + GOAL_DIM  # 1059


# ──────────────────────────────────────────────
# 모델
# ──────────────────────────────────────────────

def build_mlp(d_in=D_IN):
    return nn.Sequential(
        nn.Linear(d_in, 512), nn.ReLU(), nn.Dropout(0.25),
        nn.Linear(512, 256),  nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 128),  nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(128, 64),   nn.ReLU(),
        nn.Linear(64, NUM_CLASSES),
    )


# ──────────────────────────────────────────────
# 데이터셋 빌드
# ──────────────────────────────────────────────

def build_dataset(bbox_data, vis_cache):
    print(f"\n[DATA] 데이터셋 빌드 (WINDOW={WINDOW}, D_IN={D_IN})...")
    X, y, path_labels = [], [], []

    skipped = 0
    for ep_data in bbox_data:
        ep_path = ep_data["episode"]
        pt      = ep_data["path_type"]
        frames  = ep_data["frames"]

        vis_feats = vis_cache.get(ep_path)
        if vis_feats is None:
            skipped += 1
            continue

        # goal = frame 0의 grounded 바구니 위치 (에피소드 전체에서 고정)
        fr0 = frames[0]
        if fr0["has_bbox"]:
            goal = np.array([fr0["cx"], fr0["cy"], fr0["area"]], dtype=np.float32)
        else:
            goal = np.array([0.5, 0.5, 0.0], dtype=np.float32)  # fallback

        for t in range(len(frames)):
            bbox_feat = []
            for k in range(WINDOW):
                idx = max(0, t - (WINDOW - 1 - k))
                fr  = frames[idx]
                bbox_feat.extend([fr["cx"], fr["cy"], fr["area"], float(fr["has_bbox"])])

            feat = np.concatenate([
                np.array(bbox_feat, dtype=np.float32),
                vis_feats[t],
                goal,
            ])
            X.append(feat)
            y.append(frames[t]["gt_class"])
            path_labels.append(pt)

    print(f"  총 프레임: {len(X)}  (에피소드 skip: {skipped})")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), path_labels


# ──────────────────────────────────────────────
# 학습
# ──────────────────────────────────────────────

def train_mlp(X_tr, y_tr, X_te, y_te, epochs=300):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = build_mlp(X_tr.shape[1]).to(device)

    class_counts = np.bincount(y_tr, minlength=NUM_CLASSES).astype(float)
    weights = np.where(class_counts > 0, 1.0 / (class_counts + 1e-6), 0.0)
    weights /= weights.sum() / NUM_CLASSES
    crit = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32).to(device)
    )
    opt   = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    Xtr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    ytr_t = torch.tensor(y_tr, dtype=torch.long).to(device)
    Xte_t = torch.tensor(X_te, dtype=torch.float32).to(device)
    yte_t = torch.tensor(y_te, dtype=torch.long).to(device)

    best_acc, best_state = 0.0, None
    for ep in range(1, epochs + 1):
        net.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), 128):
            b = perm[i:i+128]
            opt.zero_grad()
            crit(net(Xtr_t[b]), ytr_t[b]).backward()
            opt.step()
        sched.step()

        if ep % 30 == 0 or ep == epochs:
            net.eval()
            with torch.no_grad():
                acc = (net(Xte_t).argmax(1) == yte_t).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            print(f"  epoch {ep:>3}  val_acc={acc:.4f}  best={best_acc:.4f}")

    net.load_state_dict(best_state)
    return net, best_acc


# ──────────────────────────────────────────────
# 평가
# ──────────────────────────────────────────────

def evaluate(net, X_te, y_te):
    device = next(net.parameters()).device
    net.eval()
    with torch.no_grad():
        preds = net(
            torch.tensor(X_te, dtype=torch.float32).to(device)
        ).argmax(1).cpu().numpy()

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for g, p in zip(y_te, preds):
        cm[g, p] += 1

    print("\n=== Confusion Matrix ===")
    print("         " + "".join(f"{n:>8}" for n in CLASS_NAMES))
    for r in range(NUM_CLASSES):
        total = cm[r].sum()
        if total == 0:
            continue
        acc = cm[r, r] / total * 100
        print(f"{CLASS_NAMES[r]:<9}" + "".join(f"{v:>8}" for v in cm[r]) + f"  {acc:.0f}%")

    overall = cm.diagonal().sum() / cm.sum()
    print(f"\n전체 정확도: {overall:.1%}")
    return cm, overall


# ──────────────────────────────────────────────
# Ablation: goal 제거 시 성능 (Exp46 재현 확인)
# ──────────────────────────────────────────────

def ablation_no_goal(X_tr, y_tr, X_te, y_te):
    print("\n[ABLATION] goal(3) 제거 → Exp46 재현 확인...")
    d_in_no_goal = D_IN - GOAL_DIM
    X_tr_ng = X_tr[:, :d_in_no_goal]
    X_te_ng = X_te[:, :d_in_no_goal]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = build_mlp(d_in_no_goal).to(device)

    class_counts = np.bincount(y_tr, minlength=NUM_CLASSES).astype(float)
    weights = np.where(class_counts > 0, 1.0 / (class_counts + 1e-6), 0.0)
    weights /= weights.sum() / NUM_CLASSES
    crit = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(device))
    opt   = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100)

    Xtr_t = torch.tensor(X_tr_ng, dtype=torch.float32).to(device)
    ytr_t = torch.tensor(y_tr, dtype=torch.long).to(device)
    Xte_t = torch.tensor(X_te_ng, dtype=torch.float32).to(device)
    yte_t = torch.tensor(y_te, dtype=torch.long).to(device)

    best_acc = 0.0
    for ep in range(1, 101):
        net.train()
        perm = torch.randperm(len(Xtr_t))
        for i in range(0, len(perm), 128):
            b = perm[i:i+128]
            opt.zero_grad()
            crit(net(Xtr_t[b]), ytr_t[b]).backward()
            opt.step()
        sched.step()
        if ep == 100:
            net.eval()
            with torch.no_grad():
                acc = (net(Xte_t).argmax(1) == yte_t).float().mean().item()
            best_acc = acc

    print(f"  no-goal acc: {best_acc:.4f}  (Exp46 참고값: 0.9316)")
    return best_acc


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 60)
    print("Exp49: Language-Grounded Goal Navigation")
    print(f"D_IN = {D_IN}  (bbox={WINDOW*4} + vis={VIS_DIM} + goal={GOAL_DIM})")
    print("=" * 60)

    # ── 데이터 로드 ──
    bbox_data = json.loads((EXP46_DIR / "bbox_dataset_full.json").read_text())
    vis_npz   = np.load(str(EXP46_DIR / "vision_features.npz"))
    vis_idx   = json.loads((EXP46_DIR / "vision_features_index.json").read_text())
    vis_cache = {ep: vis_npz[f"ep_{i}"] for ep, i in vis_idx.items()}
    print(f"\n에피소드: {len(bbox_data)}, 비전 캐시: {len(vis_cache)}")

    # ── 데이터셋 빌드 ──
    X, y, path_labels = build_dataset(bbox_data, vis_cache)

    # ── Train/Val split (Exp47과 동일) ──
    ep_labels = [ep["path_type"] for ep in bbox_data]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_ep_idx, te_ep_idx = next(sss.split(np.zeros(len(bbox_data)), ep_labels))

    # 에피소드 인덱스 → 프레임 인덱스 매핑
    ep_frame_counts = []
    offset = 0
    for ep_data in bbox_data:
        n = len(ep_data["frames"])
        ep_frame_counts.append((offset, offset + n))
        offset += n

    tr_frame_idx = np.concatenate([np.arange(*ep_frame_counts[i]) for i in tr_ep_idx])
    te_frame_idx = np.concatenate([np.arange(*ep_frame_counts[i]) for i in te_ep_idx])

    X_tr, y_tr = X[tr_frame_idx], y[tr_frame_idx]
    X_te, y_te = X[te_frame_idx], y[te_frame_idx]
    print(f"Train: {len(X_tr)} frames  Val: {len(X_te)} frames")

    # ── goal 통계 출력 ──
    print("\n[GOAL] 에피소드별 goal(cx0) 분포:")
    from collections import defaultdict
    cx_by_type = defaultdict(list)
    for ep_data in bbox_data:
        fr0 = ep_data["frames"][0]
        cx_by_type[ep_data["path_type"]].append(fr0["cx"] if fr0["has_bbox"] else 0.5)
    for pt, cxs in sorted(cx_by_type.items()):
        print(f"  {pt:<18}: cx0 = {np.mean(cxs):.3f} ± {np.std(cxs):.3f}")

    # ── 학습 ──
    print("\n[TRAIN] MLP 학습 시작 (300 epochs)...")
    net, best_acc = train_mlp(X_tr, y_tr, X_te, y_te, epochs=300)

    # ── 평가 ──
    cm, overall = evaluate(net, X_te, y_te)

    # ── Ablation ──
    no_goal_acc = ablation_no_goal(X_tr, y_tr, X_te, y_te)

    # ── 저장 ──
    torch.save({
        "model_state_dict": net.state_dict(),
        "d_in":      D_IN,
        "window":    WINDOW,
        "vis_dim":   VIS_DIM,
        "goal_dim":  GOAL_DIM,
        "overall_acc": overall,
    }, str(CKPT_PATH))

    summary = {
        "model":         "exp49",
        "overall_acc":   float(overall),
        "best_val_acc":  float(best_acc),
        "no_goal_acc":   float(no_goal_acc),
        "goal_delta":    float(overall - no_goal_acc),
        "n_train":       int(len(X_tr)),
        "n_val":         int(len(X_te)),
        "d_in":          D_IN,
        "window":        WINDOW,
        "vis_dim":       VIS_DIM,
        "goal_dim":      GOAL_DIM,
        "confusion":     cm.tolist(),
        "comparison": {
            "exp46_acc": 0.9316,
            "exp47_acc": 0.9867,
            "exp49_acc": float(overall),
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Exp49 acc:     {overall:.1%}")
    print(f"  no-goal acc:   {no_goal_acc:.1%}  (Exp46 재현)")
    print(f"  goal delta:    {(overall - no_goal_acc):+.1%}")
    print(f"  Exp47 비교:    {0.9867:.1%} → {overall:.1%}")
    print(f"  총 소요: {elapsed/60:.1f}분")
    print(f"  ckpt: {CKPT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()

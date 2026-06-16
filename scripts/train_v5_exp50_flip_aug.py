#!/usr/bin/env python3
"""
Exp50: Flip Augmentation — 기하 Robustness 개선
  Exp49 + 좌우 반전 데이터 추가 (2626 → 5252 프레임)

  flip 변환:
    cx        → 1 - cx
    action    → FLIP_ACTION_MAP (LEFT↔RIGHT, FWD+L↔FWD+R, ROT_L↔ROT_R)
    goal_cx0  → 1 - cx0
    vis_feat  → 재추출 필요 (flipped image)

Usage:
  python3 scripts/train_v5_exp50_flip_aug.py
  python3 scripts/train_v5_exp50_flip_aug.py --skip_vis  # 비전 캐시 재사용
"""
import argparse, gc, json, sys, time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HF_KOSMOS   = ROOT / ".vlms" / "kosmos-2-patch14-224"
EXP46_DIR   = ROOT / "docs" / "v5" / "bbox_nav_exp46"
EXP49_DIR   = ROOT / "docs" / "v5" / "bbox_nav_exp49"
EXP50_DIR   = ROOT / "docs" / "v5" / "bbox_nav_exp50"
MLP_DIR     = ROOT / "runs" / "v5_nav" / "mlp" / "exp50"
EXP50_DIR.mkdir(parents=True, exist_ok=True)
MLP_DIR.mkdir(parents=True, exist_ok=True)

FLIP_VIS_CACHE = EXP50_DIR / "flipped_vision_features.npz"
CKPT_PATH      = MLP_DIR / "exp50_mlp.pt"

CLASS_NAMES = ["STOP","FORWARD","LEFT","RIGHT","FWD+L","FWD+R","ROT_L","ROT_R"]
NUM_CLASSES = 8
WINDOW      = 8
VIS_DIM     = 1024
GOAL_DIM    = 3
D_IN        = WINDOW * 4 + VIS_DIM + GOAL_DIM  # 1059

FLIP_ACTION_MAP = {0:0, 1:1, 2:3, 3:2, 4:5, 5:4, 6:7, 7:6}


# ─────────────────────────────────────────────
# 모델
# ─────────────────────────────────────────────

def build_mlp(d_in=D_IN):
    return nn.Sequential(
        nn.Linear(d_in, 512), nn.ReLU(), nn.Dropout(0.25),
        nn.Linear(512, 256),  nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 128),  nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(128, 64),   nn.ReLU(),
        nn.Linear(64, NUM_CLASSES),
    )


# ─────────────────────────────────────────────
# Step 1: Flipped vision feature 추출
# ─────────────────────────────────────────────

def extract_flipped_vision_features(bbox_data):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    total_eps   = len(bbox_data)
    total_frames = sum(len(ep["frames"]) for ep in bbox_data)

    # 시간 추정: Exp46 원본 추출 기준 ~0.3s/프레임
    est_sec = total_frames * 0.3
    print(f"\n[STEP 1] Flipped vision feature 추출")
    print(f"  에피소드: {total_eps}개  |  프레임: {total_frames}개")
    print(f"  예상 소요: ~{est_sec/60:.0f}분 ({est_sec:.0f}초)")
    print(f"  저장: {FLIP_VIS_CACHE}")

    proc  = AutoProcessor.from_pretrained(str(HF_KOSMOS), trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        str(HF_KOSMOS), trust_remote_code=True, torch_dtype=torch.float16
    ).cuda().eval()

    vis_cache = {}
    t_start   = time.time()

    pbar = tqdm(enumerate(bbox_data), total=total_eps,
                desc="Flip vis", ncols=80, unit="ep")

    for ep_idx, ep_data in pbar:
        ep_path = ep_data["episode"]
        frames  = ep_data["frames"]

        with h5py.File(ep_path, "r") as f:
            if "observations" in f and "images" in f["observations"]:
                imgs = f["observations"]["images"][:]
            else:
                imgs = f["images"][:]

        feats = []
        for fi in range(len(frames)):
            img_orig = Image.fromarray(imgs[fi].astype(np.uint8)).convert("RGB")
            img_flip = img_orig.transpose(Image.FLIP_LEFT_RIGHT)

            inputs = proc(text="<grounding>", images=img_flip, return_tensors="pt")
            pv = inputs["pixel_values"].to("cuda", dtype=torch.float16)
            with torch.no_grad():
                out  = model.vision_model(pv)
                feat = out.last_hidden_state[0].mean(0).float().cpu().numpy()
            feats.append(feat)

        vis_cache[ep_path] = np.stack(feats)  # (N_frames, 1024)

        # ETA 업데이트
        elapsed = time.time() - t_start
        done_frames = sum(len(bbox_data[i]["frames"]) for i in range(ep_idx+1))
        fps = done_frames / elapsed if elapsed > 0 else 0
        rem_frames = total_frames - done_frames
        eta = rem_frames / fps if fps > 0 else 0
        pbar.set_postfix({"fps": f"{fps:.1f}", "ETA": f"{eta:.0f}s"})

    del model
    gc.collect()
    torch.cuda.empty_cache()

    np.savez_compressed(str(FLIP_VIS_CACHE),
                        **{f"ep_{i}": v for i, v in enumerate(vis_cache.values())})
    elapsed = time.time() - t_start
    print(f"\n  ✅ 완료: {elapsed:.0f}초  ({FLIP_VIS_CACHE.stat().st_size/1e6:.1f} MB)")
    return vis_cache


# ─────────────────────────────────────────────
# Step 2: 데이터셋 빌드 (원본 + flip)
# ─────────────────────────────────────────────

def build_dataset(bbox_data, orig_vis_cache, flip_vis_cache):
    print(f"\n[STEP 2] 데이터셋 빌드 (원본 + flip, D_IN={D_IN})")
    X, y, path_labels = [], [], []
    skipped = 0

    for variant in ["original", "flipped"]:
        vis_cache = orig_vis_cache if variant == "original" else flip_vis_cache
        for ep_data in bbox_data:
            ep_path = ep_data["episode"]
            pt      = ep_data["path_type"]
            frames  = ep_data["frames"]

            vis_feats = vis_cache.get(ep_path)
            if vis_feats is None:
                skipped += 1
                continue

            fr0 = frames[0]
            if fr0["has_bbox"]:
                cx0, cy0, a0 = fr0["cx"], fr0["cy"], fr0["area"]
            else:
                cx0, cy0, a0 = 0.5, 0.5, 0.0

            if variant == "flipped":
                cx0 = 1.0 - cx0   # goal_cx 반전

            goal = np.array([cx0, cy0, a0], dtype=np.float32)

            for t in range(len(frames)):
                fr = frames[t]
                bbox_feat = []
                for k in range(WINDOW):
                    idx = max(0, t - (WINDOW - 1 - k))
                    fk  = frames[idx]
                    cx  = (1.0 - fk["cx"]) if variant == "flipped" else fk["cx"]
                    bbox_feat.extend([cx, fk["cy"], fk["area"], float(fk["has_bbox"])])

                feat = np.concatenate([
                    np.array(bbox_feat, dtype=np.float32),
                    vis_feats[t],
                    goal,
                ])
                action = fr["gt_class"]
                if variant == "flipped":
                    action = FLIP_ACTION_MAP[action]

                X.append(feat)
                y.append(action)
                path_labels.append(pt)

    total = len(X)
    orig_n = total // 2
    print(f"  원본: {orig_n}  |  flip: {orig_n}  |  합계: {total}  |  skip: {skipped}")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), path_labels


# ─────────────────────────────────────────────
# Step 3: 학습
# ─────────────────────────────────────────────

def train_mlp(X_tr, y_tr, X_te, y_te, epochs=300):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = build_mlp(X_tr.shape[1]).to(device)

    cc = np.bincount(y_tr, minlength=NUM_CLASSES).astype(float)
    w  = np.where(cc > 0, 1.0 / (cc + 1e-6), 0.0)
    w /= w.sum() / NUM_CLASSES
    crit  = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
    opt   = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    Xtr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(y_tr, dtype=torch.long,    device=device)
    Xte_t = torch.tensor(X_te, dtype=torch.float32, device=device)
    yte_t = torch.tensor(y_te, dtype=torch.long,    device=device)

    best_acc, best_state = 0.0, None
    t0 = time.time()

    pbar = tqdm(range(1, epochs + 1), desc="Train", ncols=80, unit="ep")
    for ep in pbar:
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
            elapsed = time.time() - t0
            eta     = elapsed / ep * (epochs - ep)
            pbar.set_postfix({"val": f"{acc:.4f}", "best": f"{best_acc:.4f}", "ETA": f"{eta:.0f}s"})

    net.load_state_dict(best_state)
    return net, best_acc


# ─────────────────────────────────────────────
# 평가
# ─────────────────────────────────────────────

def evaluate(net, X_te, y_te):
    device = next(net.parameters()).device
    net.eval()
    with torch.no_grad():
        preds = net(torch.tensor(X_te, dtype=torch.float32, device=device)).argmax(1).cpu().numpy()
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


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_vis", action="store_true", help="flipped vis cache 재사용")
    args = parser.parse_args()

    t_total = time.time()
    print("=" * 65)
    print("Exp50: Flip Augmentation (기하 Robustness 개선)")
    print(f"D_IN = {D_IN}  |  flip 포함 총 학습 프레임: ~5252")
    print("=" * 65)

    # ── 데이터 로드 ──
    bbox_data = json.loads((EXP46_DIR / "bbox_dataset_full.json").read_text())
    orig_npz  = np.load(str(EXP46_DIR / "vision_features.npz"))
    orig_idx  = json.loads((EXP46_DIR / "vision_features_index.json").read_text())
    orig_vis  = {ep: orig_npz[f"ep_{i}"] for ep, i in orig_idx.items()}

    total_frames = sum(len(ep["frames"]) for ep in bbox_data)
    print(f"\n에피소드: {len(bbox_data)}  |  원본 프레임: {total_frames}")

    # ── Step 1: Flipped vision feature ──
    if args.skip_vis and FLIP_VIS_CACHE.exists():
        print(f"\n[STEP 1] 캐시 재사용: {FLIP_VIS_CACHE}")
        flip_npz = np.load(str(FLIP_VIS_CACHE))
        flip_vis = {ep: flip_npz[f"ep_{i}"] for ep, i in orig_idx.items()}
    else:
        raw_flip = extract_flipped_vision_features(bbox_data)
        flip_vis = raw_flip

    # ── Step 2: 데이터셋 빌드 ──
    X, y, path_labels = build_dataset(bbox_data, orig_vis, flip_vis)

    # ── Train/Val split — 원본 에피소드 기준 (flip은 train에만 추가) ──
    # val은 원본만 사용 → Exp49와 동일 조건으로 PM 비교 가능
    ep_labels = [ep["path_type"] for ep in bbox_data]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_ep_idx, te_ep_idx = next(sss.split(np.zeros(len(bbox_data)), ep_labels))

    ep_frame_counts, offset = [], 0
    for ep_data in bbox_data:
        n = len(ep_data["frames"])
        ep_frame_counts.append((offset, offset + n))
        offset += n

    orig_total = sum(len(ep["frames"]) for ep in bbox_data)  # 2626

    # val: 원본 에피소드의 원본 프레임만
    te_frame_idx_orig = np.concatenate([np.arange(*ep_frame_counts[i]) for i in te_ep_idx])
    X_te, y_te = X[te_frame_idx_orig], y[te_frame_idx_orig]

    # train: 원본 train 프레임 + flip된 train 프레임
    tr_orig = np.concatenate([np.arange(*ep_frame_counts[i]) for i in tr_ep_idx])
    # flip 에피소드는 원본 뒤에 붙어있음 (offset = orig_total)
    tr_flip = tr_orig + orig_total
    X_tr = np.concatenate([X[tr_orig], X[tr_flip]])
    y_tr = np.concatenate([y[tr_orig], y[tr_flip]])

    print(f"\n[SPLIT] train: {len(X_tr)} (원본 {len(tr_orig)} + flip {len(tr_flip)})  |  val(원본): {len(X_te)}")

    # ── Step 3: 학습 ──
    print(f"\n[STEP 3] MLP 학습 (300 epochs)")
    est_train = 300 * len(X_tr) / 2100 * 0.2 / 60  # 대략적 추정
    print(f"  예상 소요: ~{est_train:.0f}분")
    net, best_acc = train_mlp(X_tr, y_tr, X_te, y_te, epochs=300)

    # ── 평가 ──
    cm, overall = evaluate(net, X_te, y_te)

    # ── 저장 ──
    torch.save({
        "model_state_dict": net.state_dict(),
        "d_in":     D_IN,
        "window":   WINDOW,
        "vis_dim":  VIS_DIM,
        "goal_dim": GOAL_DIM,
        "overall_acc": overall,
    }, str(CKPT_PATH))

    summary = {
        "model":        "exp50",
        "overall_acc":  float(overall),
        "best_val_acc": float(best_acc),
        "n_train":      int(len(X_tr)),
        "n_val":        int(len(X_te)),
        "comparison": {
            "exp46": 0.9316,
            "exp49": 0.9639,
            "exp50": float(overall),
        },
        "confusion": cm.tolist(),
    }
    (EXP50_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    elapsed = time.time() - t_total
    print(f"\n{'='*65}")
    print(f"  Exp49 기준: 96.4%  →  Exp50: {overall:.1%}")
    print(f"  총 소요: {elapsed/60:.1f}분")
    print(f"  ckpt: {CKPT_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    main()

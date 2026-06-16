#!/usr/bin/env python3
"""
Exp51: Crop Augmentation — 카메라 위치 Robustness
  Exp50(flip) + crop_left10% + crop_right10% 데이터 추가

  crop 변환:
    crop_left10%  : 이미지 왼쪽 10% 잘라내고 stretch, cx → max(0, cx-0.10)
    crop_right10% : 이미지 오른쪽 10% 잘라내고 stretch, cx → min(1, cx+0.10)
    action        : 변환 없음 (카메라 위치만 달라짐)

  학습 데이터: 원본(2626) + flip(2626) + crop_L(2626) + crop_R(2626) = 10504
  val 데이터:  원본만(526) — Exp49/50과 동일 조건

Usage:
  python3 scripts/train_v5_exp51_crop_aug.py
  python3 scripts/train_v5_exp51_crop_aug.py --skip_vis  # 캐시 재사용
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
EXP50_DIR   = ROOT / "docs" / "v5" / "bbox_nav_exp50"
EXP51_DIR   = ROOT / "docs" / "v5" / "bbox_nav_exp51"
MLP_DIR     = ROOT / "runs" / "v5_nav" / "mlp" / "exp51"
EXP51_DIR.mkdir(parents=True, exist_ok=True)
MLP_DIR.mkdir(parents=True, exist_ok=True)

FLIP_VIS_CACHE  = EXP50_DIR / "flipped_vision_features.npz"
CROPL_VIS_CACHE = EXP51_DIR / "crop_left10_vision_features.npz"
CROPR_VIS_CACHE = EXP51_DIR / "crop_right10_vision_features.npz"
CKPT_PATH       = MLP_DIR / "exp51_mlp.pt"

CLASS_NAMES = ["STOP","FORWARD","LEFT","RIGHT","FWD+L","FWD+R","ROT_L","ROT_R"]
NUM_CLASSES = 8
WINDOW      = 8
VIS_DIM     = 1024
GOAL_DIM    = 3
D_IN        = WINDOW * 4 + VIS_DIM + GOAL_DIM  # 1059

FLIP_ACTION_MAP = {0:0, 1:1, 2:3, 3:2, 4:5, 5:4, 6:7, 7:6}

CROP_VARIANTS = {
    "crop_left10":  {"cx_delta": -0.10},
    "crop_right10": {"cx_delta": +0.10},
}


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
# Step 1: Crop vision feature 추출
# ─────────────────────────────────────────────

def _crop_shift(img, direction, ratio):
    w, h = img.size
    shift = int(w * ratio)
    if direction == "left":
        return img.crop((shift, 0, w, h)).resize((w, h), Image.BILINEAR)
    else:
        return img.crop((0, 0, w - shift, h)).resize((w, h), Image.BILINEAR)


def extract_crop_vision_features(bbox_data, direction, out_path):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    total_eps    = len(bbox_data)
    total_frames = sum(len(ep["frames"]) for ep in bbox_data)
    est_sec      = total_frames * 0.3

    print(f"\n[STEP 1] Crop vision feature 추출 — {direction}")
    print(f"  에피소드: {total_eps}  |  프레임: {total_frames}")
    print(f"  예상 소요: ~{est_sec/60:.0f}분 ({est_sec:.0f}초)")
    print(f"  저장: {out_path}")

    proc  = AutoProcessor.from_pretrained(str(HF_KOSMOS), trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        str(HF_KOSMOS), trust_remote_code=True, torch_dtype=torch.float16
    ).cuda().eval()

    vis_cache = {}
    t_start   = time.time()

    pbar = tqdm(enumerate(bbox_data), total=total_eps,
                desc=f"crop_{direction}", ncols=80, unit="ep")

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
            img_crop = _crop_shift(img_orig, direction, 0.10)

            inputs = proc(text="<grounding>", images=img_crop, return_tensors="pt")
            pv = inputs["pixel_values"].to("cuda", dtype=torch.float16)
            with torch.no_grad():
                out  = model.vision_model(pv)
                feat = out.last_hidden_state[0].mean(0).float().cpu().numpy()
            feats.append(feat)

        vis_cache[ep_path] = np.stack(feats)

        elapsed    = time.time() - t_start
        done_frames = sum(len(bbox_data[i]["frames"]) for i in range(ep_idx + 1))
        fps        = done_frames / elapsed if elapsed > 0 else 0
        rem_frames = total_frames - done_frames
        eta        = rem_frames / fps if fps > 0 else 0
        pbar.set_postfix({"fps": f"{fps:.1f}", "ETA": f"{eta:.0f}s"})

    del model
    gc.collect()
    torch.cuda.empty_cache()

    np.savez_compressed(str(out_path),
                        **{f"ep_{i}": v for i, v in enumerate(vis_cache.values())})
    elapsed = time.time() - t_start
    print(f"\n  ✅ 완료: {elapsed:.0f}초  ({out_path.stat().st_size/1e6:.1f} MB)")
    return vis_cache


# ─────────────────────────────────────────────
# Step 2: 데이터셋 빌드
# ─────────────────────────────────────────────

def build_dataset(bbox_data, orig_vis, flip_vis, cropl_vis, cropr_vis):
    print(f"\n[STEP 2] 데이터셋 빌드 (원본+flip+crop_L+crop_R, D_IN={D_IN})")
    X, y, path_labels = [], [], []
    skipped = 0

    variants = [
        ("original",    orig_vis,  0.0,   False),
        ("flipped",     flip_vis,  0.0,   True),
        ("crop_left10", cropl_vis, -0.10, False),
        ("crop_right10",cropr_vis, +0.10, False),
    ]

    for variant_name, vis_cache, cx_delta, do_flip in variants:
        for ep_data in bbox_data:
            ep_path = ep_data["episode"]
            pt      = ep_data["path_type"]
            frames  = ep_data["frames"]

            vis_feats = vis_cache.get(ep_path)
            if vis_feats is None:
                skipped += 1
                continue

            fr0 = frames[0]
            cx0 = fr0["cx"]   if fr0["has_bbox"] else 0.5
            cy0 = fr0["cy"]   if fr0["has_bbox"] else 0.5
            a0  = fr0["area"] if fr0["has_bbox"] else 0.0

            if do_flip:
                cx0 = 1.0 - cx0
            else:
                cx0 = float(np.clip(cx0 + cx_delta, 0.0, 1.0))

            goal = np.array([cx0, cy0, a0], dtype=np.float32)

            for t in range(len(frames)):
                fr = frames[t]
                bbox_feat = []
                for k in range(WINDOW):
                    idx = max(0, t - (WINDOW - 1 - k))
                    fk  = frames[idx]
                    cx  = fk["cx"]
                    if do_flip:
                        cx = 1.0 - cx
                    else:
                        cx = float(np.clip(cx + cx_delta, 0.0, 1.0))
                    bbox_feat.extend([cx, fk["cy"], fk["area"], float(fk["has_bbox"])])

                feat = np.concatenate([
                    np.array(bbox_feat, dtype=np.float32),
                    vis_feats[t],
                    goal,
                ])
                action = fr["gt_class"]
                if do_flip:
                    action = FLIP_ACTION_MAP[action]

                X.append(feat)
                y.append(action)
                path_labels.append(pt)

    total  = len(X)
    n_each = total // 4
    print(f"  원본: {n_each}  flip: {n_each}  crop_L: {n_each}  crop_R: {n_each}")
    print(f"  합계: {total}  |  skip: {skipped}")
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
    parser.add_argument("--skip_vis", action="store_true", help="crop vis cache 재사용")
    args = parser.parse_args()

    t_total = time.time()
    print("=" * 65)
    print("Exp51: Crop Augmentation (카메라 위치 Robustness)")
    print(f"D_IN = {D_IN}  |  총 학습 프레임: ~10504 (4× 원본)")
    print("=" * 65)

    # ── 데이터 로드 ──
    bbox_data = json.loads((EXP46_DIR / "bbox_dataset_full.json").read_text())
    orig_npz  = np.load(str(EXP46_DIR / "vision_features.npz"))
    orig_idx  = json.loads((EXP46_DIR / "vision_features_index.json").read_text())
    orig_vis  = {ep: orig_npz[f"ep_{i}"] for ep, i in orig_idx.items()}

    total_frames = sum(len(ep["frames"]) for ep in bbox_data)
    print(f"\n에피소드: {len(bbox_data)}  |  원본 프레임: {total_frames}")

    # ── flip cache (Exp50 재사용) ──
    flip_npz = np.load(str(FLIP_VIS_CACHE))
    flip_vis = {ep: flip_npz[f"ep_{i}"] for ep, i in orig_idx.items()}
    print(f"[FLIP]  캐시 재사용: {FLIP_VIS_CACHE}")

    # ── Step 1: Crop vision features ──
    if args.skip_vis and CROPL_VIS_CACHE.exists() and CROPR_VIS_CACHE.exists():
        print(f"\n[STEP 1] crop vis 캐시 재사용")
        cropl_npz = np.load(str(CROPL_VIS_CACHE))
        cropr_npz = np.load(str(CROPR_VIS_CACHE))
        cropl_vis = {ep: cropl_npz[f"ep_{i}"] for ep, i in orig_idx.items()}
        cropr_vis = {ep: cropr_npz[f"ep_{i}"] for ep, i in orig_idx.items()}
    else:
        raw_l = extract_crop_vision_features(bbox_data, "left",  CROPL_VIS_CACHE)
        raw_r = extract_crop_vision_features(bbox_data, "right", CROPR_VIS_CACHE)
        cropl_vis = raw_l
        cropr_vis = raw_r

    # ── Step 2: 데이터셋 빌드 ──
    X, y, path_labels = build_dataset(bbox_data, orig_vis, flip_vis, cropl_vis, cropr_vis)

    # ── Train/Val split — 원본 에피소드 기준, val=원본만 ──
    ep_labels = [ep["path_type"] for ep in bbox_data]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_ep_idx, te_ep_idx = next(sss.split(np.zeros(len(bbox_data)), ep_labels))

    ep_frame_counts, offset = [], 0
    for ep_data in bbox_data:
        n = len(ep_data["frames"])
        ep_frame_counts.append((offset, offset + n))
        offset += n

    orig_total = sum(len(ep["frames"]) for ep in bbox_data)  # 2626

    # val: 원본만
    te_frame_idx = np.concatenate([np.arange(*ep_frame_counts[i]) for i in te_ep_idx])
    X_te, y_te = X[te_frame_idx], y[te_frame_idx]

    # train: 4개 variant의 train 프레임
    tr_orig = np.concatenate([np.arange(*ep_frame_counts[i]) for i in tr_ep_idx])
    X_tr_parts, y_tr_parts = [], []
    for v_offset in [0, orig_total, orig_total * 2, orig_total * 3]:
        idx = tr_orig + v_offset
        X_tr_parts.append(X[idx])
        y_tr_parts.append(y[idx])
    X_tr = np.concatenate(X_tr_parts)
    y_tr = np.concatenate(y_tr_parts)

    print(f"\n[SPLIT] train: {len(X_tr)} (원본+flip+cropL+cropR each {len(tr_orig)})  |  val(원본): {len(X_te)}")

    # ── Step 3: 학습 ──
    print(f"\n[STEP 3] MLP 학습 (300 epochs)")
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
        "model":        "exp51",
        "overall_acc":  float(overall),
        "best_val_acc": float(best_acc),
        "n_train":      int(len(X_tr)),
        "n_val":        int(len(X_te)),
        "comparison": {
            "exp49": 0.9639,
            "exp50": 0.9200,
            "exp51": float(overall),
        },
        "confusion": cm.tolist(),
    }
    (EXP51_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    elapsed = time.time() - t_total
    print(f"\n{'='*65}")
    print(f"  Exp49: 96.4%  Exp50: 92.0%  →  Exp51: {overall:.1%}")
    print(f"  총 소요: {elapsed/60:.1f}분")
    print(f"  ckpt: {CKPT_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    main()

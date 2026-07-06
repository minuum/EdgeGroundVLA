#!/usr/bin/env python3
"""
2026-07-06 실로봇 발견(soda): exp71 학습데이터 has_bbox=95.9%인데 실전 OWL은 40~60%
flicker → window 입력이 fallback(cx=0.5,area=0.06)으로 자주 덮여서 조건 무관하게
RIGHT<->FWD+L 왕복으로 수렴하는 것으로 의심됨.

4개 변형 비교 (Step2 MLP 레시피, bbox_nav_step1/bbox_dataset.json, window=6 통일):
  1. baseline      : 원본 그대로 (대조군)
  2. dropout_aug   : 학습 시 각 프레임을 p=0.5로 fallback(cx=0.5,cy=0.6,area=0.06,has=0) 치환
  3. sticky_aug    : 위와 같은 dropout 이벤트지만 fallback 대신 마지막 실제 검출값 유지
  4. window3       : window만 6->3로 축소 (증강 없음)
  5. cx_aux_loss   : 현재 프레임 cx가 실제 검출(has_bbox=1)일 때, cx 버킷(좌/중/우) ->
                     방향클래스 쏠림을 유도하는 보조 loss 추가 (has_bbox=0 프레임은 미적용)

평가: 각 변형을 (a) 원본 test set, (b) test set에 합성 flicker(p=0.5) 적용한
"실전 유사" 조건 둘 다에서 PM 측정 -> 어떤 변형이 실전 flicker에 가장 강건한지 비교.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "ROS_action" / "mobile_vla_dataset_v5"
DATASET_FILE = ROOT / "docs" / "v5" / "bbox_nav_step1" / "bbox_dataset.json"
OUT_FILE = ROOT / "docs" / "v5" / "closed_loop_eval" / "flicker_robustness_compare.json"

NUM_CLASSES = 8
IMG_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FALLBACK = (0.5, 0.6, 0.06, 0.0)
LEFT_CLASSES = {2, 4, 6}
RIGHT_CLASSES = {3, 5, 7}
CX_LEFT_TH, CX_RIGHT_TH = 0.35, 0.65
SEED = 0


def load_dataset():
    dataset = json.loads(DATASET_FILE.read_text())
    return [ep for ep in dataset if list(DATA_DIR.glob(f"{ep['episode']}.h5"))]


def frame_to_small_feature(frame):
    img = Image.fromarray(frame.astype(np.uint8)).convert("L").resize((IMG_SIZE, IMG_SIZE))
    return (np.asarray(img, dtype=np.float32) / 255.0).reshape(-1)


def load_episode_frames(stem):
    path = next(DATA_DIR.glob(f"{stem}.h5"))
    with h5py.File(path, "r") as f:
        if "observations" in f and "images" in f["observations"]:
            return f["observations"]["images"][:]
        return f["images"][:]


def make_episode_split(dataset, seed=42):
    rng = np.random.default_rng(seed)
    by_path = defaultdict(list)
    for i, ep in enumerate(dataset):
        by_path[ep["path_type"]].append(i)
    train_idx, test_idx = [], []
    for _, idxs in by_path.items():
        rng.shuffle(idxs)
        k = max(1, int(len(idxs) * 0.2))
        test_idx.extend(idxs[:k])
        train_idx.extend(idxs[k:])
    return [dataset[i] for i in train_idx], [dataset[i] for i in test_idx]


def build_raw_windows(dataset, window):
    """에피소드별 (bbox_seq, img_feats, gt) 원본 시퀀스를 만들어둔다. 증강은 학습 루프에서."""
    episodes = []
    for ep in dataset:
        imgs = load_episode_frames(ep["episode"])
        frames = ep["frames"]
        img_feats = [frame_to_small_feature(imgs[f["frame_idx"]]) for f in frames]
        bboxes = [(f["cx"], f["cy"], f["area"], float(f["has_bbox"])) for f in frames]
        gts = [f["gt_class"] for f in frames]
        episodes.append({"bboxes": bboxes, "img_feats": img_feats, "gts": gts, "path_type": ep["path_type"]})
    return episodes


def apply_flicker(bboxes, p, rng, sticky):
    """p 확률로 각 프레임을 fallback 처리. sticky=True면 fallback 대신 마지막 실검출값 유지."""
    out = []
    last_real = FALLBACK
    for b in bboxes:
        if b[3] > 0.5:
            last_real = b
        drop = rng.random() < p
        if not drop:
            out.append(b)
        elif sticky:
            out.append((last_real[0], last_real[1], last_real[2], 0.0))  # has_bbox는 0으로 유지(진짜 미검출 표시)
        else:
            out.append(FALLBACK)
    return out


def windows_from_episodes(episodes, window, flicker_p=0.0, sticky=False, rng=None):
    X, y, meta = [], [], []
    for ep in episodes:
        bboxes = ep["bboxes"]
        if flicker_p > 0:
            bboxes = apply_flicker(bboxes, flicker_p, rng, sticky)
        img_feats = ep["img_feats"]
        gts = ep["gts"]
        n = len(gts)
        for t in range(n):
            feat = []
            for k in range(window):
                idx = max(0, t - (window - 1 - k))
                feat.extend(list(bboxes[idx]))
            feat.extend(img_feats[t].tolist())
            X.append(feat)
            y.append(gts[t])
            meta.append({"path_type": ep["path_type"]})
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64), meta


def cx_bucket_target(cx):
    if cx < CX_LEFT_TH:
        return torch.tensor([1.0 if c in LEFT_CLASSES else 0.0 for c in range(NUM_CLASSES)])
    if cx > CX_RIGHT_TH:
        return torch.tensor([1.0 if c in RIGHT_CLASSES else 0.0 for c in range(NUM_CLASSES)])
    return None


def make_model(d_in):
    return nn.Sequential(
        nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(0.25),
        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, NUM_CLASSES),
    ).to(DEVICE)


def train_variant(name, train_eps, test_eps, window, aug_p, sticky, cx_aux, epochs=180):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    X_tr, y_tr, meta_tr = windows_from_episodes(train_eps, window, flicker_p=aug_p, sticky=sticky, rng=rng)
    X_te, y_te, meta_te = windows_from_episodes(test_eps, window, flicker_p=0.0)

    d_in = X_tr.shape[1]
    model = make_model(d_in)
    cls_counts = np.bincount(y_tr, minlength=NUM_CLASSES).astype(np.float32)
    cls_counts = np.where(cls_counts == 0, 1.0, cls_counts)
    weights = torch.tensor(1.0 / cls_counts, dtype=torch.float32, device=DEVICE)
    weights = weights / weights.sum() * NUM_CLASSES
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    X_tr_t = torch.tensor(X_tr, device=DEVICE)
    y_tr_t = torch.tensor(y_tr, device=DEVICE)
    X_te_t = torch.tensor(X_te, device=DEVICE)
    y_te_t = torch.tensor(y_te, device=DEVICE)

    # cx_aux: 현재 프레임(윈도우 마지막 슬라이스)의 cx/has_bbox 위치를 학습용으로도 보관
    cur_cx = X_tr[:, (window - 1) * 4 + 0]
    cur_has = X_tr[:, (window - 1) * 4 + 3]

    best_acc, best_state = 0.0, None
    for ep in range(epochs):
        model.train()
        idx = torch.randperm(len(X_tr_t))
        for i in range(0, len(idx), 128):
            b = idx[i:i + 128]
            logits = model(X_tr_t[b])
            loss = loss_fn(logits, y_tr_t[b])
            if cx_aux:
                b_np = b.cpu().numpy()
                aux_mask = cur_has[b_np] > 0.5
                if aux_mask.any():
                    aux_idx = np.where(aux_mask)[0]
                    cxs = cur_cx[b_np][aux_idx]
                    targets = []
                    keep = []
                    for j, cx in zip(aux_idx, cxs):
                        t = cx_bucket_target(cx)
                        if t is not None:
                            targets.append(t)
                            keep.append(j)
                    if targets:
                        tgt = torch.stack(targets).to(DEVICE)
                        logp = F.log_softmax(logits[keep], dim=-1)
                        aux_loss = -(tgt * logp).sum(dim=-1).mean() / tgt.sum(dim=-1).mean().clamp(min=1)
                        loss = loss + 0.15 * aux_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            preds = model(X_te_t).argmax(dim=-1)
            acc = (preds == y_te_t).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()

    # 실전 유사 조건: test set에 flicker(p=0.5) 걸어서 재평가 (증강 없이 학습된 baseline까지 공통 조건으로)
    flicker_accs = []
    for trial in range(3):
        rng2 = np.random.default_rng(100 + trial)
        X_fl, y_fl, _ = windows_from_episodes(test_eps, window, flicker_p=0.5, sticky=False, rng=rng2)
        with torch.no_grad():
            preds = model(torch.tensor(X_fl, device=DEVICE)).argmax(dim=-1)
            flicker_accs.append((preds.cpu().numpy() == y_fl).mean())

    # 액션 다양성(붕괴 여부) 측정: flicker 조건에서 예측 클래스 분포의 엔트로피
    with torch.no_grad():
        preds_fl = model(torch.tensor(X_fl, device=DEVICE)).argmax(dim=-1).cpu().numpy()
    dist = np.bincount(preds_fl, minlength=NUM_CLASSES) / len(preds_fl)
    entropy = -np.sum([p * np.log(p + 1e-9) for p in dist])

    result = {
        "clean_pm": float(best_acc),
        "flicker_pm_mean": float(np.mean(flicker_accs)),
        "flicker_pm_std": float(np.std(flicker_accs)),
        "flicker_pred_class_dist": dist.tolist(),
        "flicker_pred_entropy": float(entropy),
    }
    print(f"[{name:14s}] clean={best_acc:.1%}  flicker(p=0.5)={np.mean(flicker_accs):.1%}"
          f"±{np.std(flicker_accs):.1%}  액션분포엔트로피={entropy:.2f}")
    return result


def main():
    dataset = load_dataset()
    train_ds, test_ds = make_episode_split(dataset)
    print(f"episodes: train={len(train_ds)} test={len(test_ds)}")
    train_eps = build_raw_windows(train_ds, window=6)
    test_eps = build_raw_windows(test_ds, window=6)
    train_eps3 = build_raw_windows(train_ds, window=3)
    test_eps3 = build_raw_windows(test_ds, window=3)

    results = {}
    results["1_baseline"] = train_variant("baseline", train_eps, test_eps, window=6, aug_p=0.0, sticky=False, cx_aux=False)
    results["2_dropout_aug"] = train_variant("dropout_aug", train_eps, test_eps, window=6, aug_p=0.5, sticky=False, cx_aux=False)
    results["3_sticky_aug"] = train_variant("sticky_aug", train_eps, test_eps, window=6, aug_p=0.5, sticky=True, cx_aux=False)
    results["4_window3"] = train_variant("window3", train_eps3, test_eps3, window=3, aug_p=0.0, sticky=False, cx_aux=False)
    results["5_cx_aux_loss"] = train_variant("cx_aux_loss", train_eps, test_eps, window=6, aug_p=0.0, sticky=False, cx_aux=True)
    results["6_dropout+cx_aux"] = train_variant("dropout+cx_aux", train_eps, test_eps, window=6, aug_p=0.5, sticky=False, cx_aux=True)

    OUT_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nsaved -> {OUT_FILE}")


if __name__ == "__main__":
    main()

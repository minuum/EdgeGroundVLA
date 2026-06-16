#!/usr/bin/env python3
"""
Track 2: Zero-shot Linear Probe

Stage 1 학습 전 완전 frozen Kosmos-2 CLIP feature만으로
basket 위치(left/center/right) 분류 가능한가?

- 학습 없음 — raw CLIP feature 그대로
- logistic regression (sklearn) 5-fold CV (에피소드 단위 split)
- 결과 해석:
    80%↑  → "CLIP이 이미 basket 위치를 인코딩한다" (Stage 1은 꺼내는 것)
    60~80% → "부분 인코딩, Stage 1이 강화"
    60%↓  → "Stage 1이 능력을 만들어준다"

Usage:
  .venv/bin/python3 scripts/exp54_zeroshot_linear_probe.py
"""

import json, sys, warnings
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

VLM_PATH  = ROOT / ".vlms" / "kosmos-2-patch14-224"
DATA_PATH = ROOT / "docs" / "v5" / "bbox_frame_level" / "bbox_dataset_frame_level.json"

DIR_IDX = {"left": 0, "center": 1, "right": 2}
DIRS    = ["left", "center", "right"]


def load_vision_encoder(device):
    from transformers import AutoModelForVision2Seq, AutoProcessor
    processor = AutoProcessor.from_pretrained(str(VLM_PATH))
    base = AutoModelForVision2Seq.from_pretrained(
        str(VLM_PATH), torch_dtype=torch.float16
    )
    vm = base.vision_model.to(device).eval()
    for p in vm.parameters():
        p.requires_grad = False
    return processor, vm


@torch.no_grad()
def extract_features(processor, vm, images, device):
    inputs = processor(images=images, return_tensors="pt")
    pv = inputs["pixel_values"].to(device, dtype=torch.float16)
    out = vm(pixel_values=pv)
    feat = out.last_hidden_state.mean(dim=1).float()  # (N, 1024)
    return feat.cpu().numpy()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")
    print(f"[MODEL] frozen Kosmos-2 CLIP 로드 중...")
    processor, vm = load_vision_encoder(device)
    print(f"[MODEL] 로드 완료 (완전 frozen, 학습 없음)\n")

    data = json.loads(DATA_PATH.read_text())

    # 에피소드별 feature 추출
    print("[DATA] feature 추출 중 (에피소드 단위)...")
    ep_feats  = []   # per-episode: (N_frames, 1024)
    ep_labels = []   # per-episode: (N_frames,)  0/1/2
    ep_dirs   = []   # per-episode: direction string (split stratify용)

    BATCH = 16
    for ep_idx, ep in enumerate(data):
        frames = [f for f in ep["frames"] if f["consistent"] and f["label"]]
        if not frames:
            continue
        d = ep["direction"]

        images, labels = [], []
        for fr in frames:
            try:
                with h5py.File(ep["episode"], "r") as f:
                    img = Image.fromarray(f["observations"]["images"][fr["frame_idx"]])
                images.append(img)
                labels.append(DIR_IDX[fr["label"]])
            except:
                pass

        if not images:
            continue

        # 배치 단위 인코딩
        all_feats = []
        for i in range(0, len(images), BATCH):
            batch = images[i:i+BATCH]
            feats = extract_features(processor, vm, batch, device)
            all_feats.append(feats)

        ep_feats.append(np.concatenate(all_feats, axis=0))
        ep_labels.append(np.array(labels))
        ep_dirs.append(d)

        if (ep_idx + 1) % 30 == 0:
            total = sum(len(l) for l in ep_labels)
            print(f"  [{ep_idx+1}/{len(data)}] 에피소드, {total} 프레임 완료")

    n_eps = len(ep_feats)
    total_frames = sum(len(l) for l in ep_labels)
    print(f"\n[DATA] 에피소드: {n_eps}, 총 프레임: {total_frames}")

    # 에피소드 단위 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ep_arr = np.arange(n_eps)
    ep_dir_arr = np.array(ep_dirs)

    fold_accs   = []
    all_preds   = []
    all_gts     = []
    dir_fold_accs = defaultdict(list)

    print(f"\n[CV] 에피소드 단위 5-fold 시작...\n")
    for fold, (tr_idx, te_idx) in enumerate(skf.split(ep_arr, ep_dir_arr)):
        X_tr = np.concatenate([ep_feats[i] for i in tr_idx], axis=0)
        y_tr = np.concatenate([ep_labels[i] for i in tr_idx], axis=0)
        X_te = np.concatenate([ep_feats[i] for i in te_idx], axis=0)
        y_te = np.concatenate([ep_labels[i] for i in te_idx], axis=0)

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)

        acc = (preds == y_te).mean() * 100
        fold_accs.append(acc)
        all_preds.extend(preds)
        all_gts.extend(y_te)

        # 방향별 acc
        dir_str = ""
        for d in DIRS:
            mask = (y_te == DIR_IDX[d])
            if mask.sum() > 0:
                d_acc = (preds[mask] == y_te[mask]).mean() * 100
                dir_fold_accs[d].append(d_acc)
                dir_str += f"  {d}={d_acc:.1f}%"
        print(f"  fold {fold+1}  acc={acc:.1f}%  |{dir_str}")

    mean_acc = np.mean(fold_accs)
    std_acc  = np.std(fold_accs)

    # 혼동 행렬
    cm = confusion_matrix(all_gts, all_preds)

    print(f"\n{'='*60}")
    print(f"  Track 2: Zero-shot Linear Probe 결과")
    print(f"{'='*60}")
    print(f"\n  [설정] frozen Kosmos-2 CLIP (LoRA 없음, 학습 없음)")
    print(f"         logistic regression, 에피소드 단위 5-fold CV")
    print(f"\n  5-fold 정확도: {mean_acc:.1f}% ± {std_acc:.1f}%")
    print(f"  (각 fold: {', '.join(f'{a:.1f}%' for a in fold_accs)})")

    print(f"\n  방향별 평균 정확도:")
    for d in DIRS:
        accs = dir_fold_accs[d]
        if accs:
            print(f"    {d:<8}: {np.mean(accs):.1f}% ± {np.std(accs):.1f}%")

    print(f"\n  혼동 행렬 (행=실제, 열=예측):")
    print(f"  {'':>9} {'left':>8} {'center':>8} {'right':>8}  {'정확도':>8}")
    for i, d in enumerate(DIRS):
        row = cm[i]
        d_acc = row[i] / row.sum() * 100 if row.sum() > 0 else 0
        print(f"  {d:>9} {row[0]:>8} {row[1]:>8} {row[2]:>8}  {d_acc:>7.1f}%")

    print(f"\n  v1 (path_type 레이블) Stage 1:  100.0%")
    print(f"  v2 (frame-level 레이블) Stage 1:  98.1%")
    print(f"  zero-shot linear probe:          {mean_acc:.1f}%")

    if mean_acc >= 80:
        verdict = "CLIP이 이미 basket 위치를 인코딩 ✅ (Stage 1은 꺼내는 것)"
    elif mean_acc >= 60:
        verdict = "부분 인코딩 — Stage 1이 강화하는 구조"
    else:
        verdict = "Stage 1이 실제로 능력을 만들어줌 (linear 불가)"
    print(f"  → 판정: {verdict}")

    gap = 98.1 - mean_acc
    print(f"\n  Stage 1 v2 - zero-shot 격차: +{gap:.1f}%p")
    print(f"  (Stage 1이 이 격차만큼 CLIP feature를 정렬해줌)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

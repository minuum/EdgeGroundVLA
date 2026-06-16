#!/usr/bin/env python3
"""
Exp52: True VLA — Language-Conditioned Visual Features

Exp49 대비 변경:
  vision(1024) → lang_vis(2048)  ← Kosmos-2 joint forward image token hidden
  d_in: 1059 → 2083

입력: bbox(8×4=32) + lang_vis(2048) + goal(3) = 2083-dim
출력: 8-class action

Usage:
  .venv/bin/python3 scripts/train_v5_exp52_true_vla.py
  .venv/bin/python3 scripts/train_v5_exp52_true_vla.py --paraphrase_test  # paraphrase만
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXP46_DIR = ROOT / "docs" / "v5" / "bbox_nav_exp46"
EXP52_DIR = ROOT / "docs" / "v5" / "bbox_nav_exp52"
MLP_DIR   = ROOT / "runs" / "v5_nav" / "mlp" / "exp52"
EXP52_DIR.mkdir(parents=True, exist_ok=True)
MLP_DIR.mkdir(parents=True, exist_ok=True)

CKPT_PATH = MLP_DIR / "exp52_mlp.pt"

CLASS_NAMES = ["STOP","FORWARD","LEFT","RIGHT","FWD+L","FWD+R","ROT_L","ROT_R"]
NUM_CLASSES = 8
WINDOW      = 8
LANG_VIS_DIM = 2048
GOAL_DIM    = 3           # (cx0, cy0, area0)
D_IN        = WINDOW * 4 + LANG_VIS_DIM + GOAL_DIM  # 2083

INSTRUCTIONS = {
    "center_straight": "Navigate straight ahead to the basket in the center",
    "center_left":     "Navigate to the basket on the left",
    "center_right":    "Navigate to the basket on the right",
    "left_straight":   "Turn left and navigate straight to the basket",
    "left_left":       "Turn left and go to the basket on the left side",
    "left_right":      "Turn left then right to reach the basket",
    "right_straight":  "Turn right and navigate straight to the basket",
    "right_left":      "Turn right then left to reach the basket",
    "right_right":     "Turn right and go to the basket on the right side",
}

PARAPHRASES = {
    "center_straight": ["Go straight forward toward the basket in the middle",
                        "Move directly ahead to the central basket",
                        "Proceed straight to the basket at center",
                        "Drive forward to reach the basket in front"],
    "center_left":     ["Go to the container on the left side",
                        "Move toward the basket positioned on the left",
                        "Head to the left basket",
                        "Approach the left-side basket"],
    "center_right":    ["Head toward the container on the right",
                        "Move to the right-side basket",
                        "Go to the basket on the right",
                        "Navigate to the basket positioned right"],
    "left_straight":   ["Make a left turn and go straight to the basket",
                        "Turn left then proceed forward to the basket",
                        "Go left and drive straight toward the basket",
                        "Rotate left and navigate forward to the basket"],
    "left_left":       ["Rotate left and head to the left basket",
                        "Turn left and reach the basket on the left",
                        "Go left and then move to the left-side container",
                        "Navigate left and find the basket on the left"],
    "left_right":      ["Go left then navigate to the right basket",
                        "Turn left and move toward the right-side basket",
                        "Rotate left then head to the right container",
                        "Turn left and approach the basket on the right"],
    "right_straight":  ["Make a right turn and go straight to the basket",
                        "Turn right then proceed forward to the basket",
                        "Go right and drive straight toward the basket",
                        "Rotate right and navigate forward to the basket"],
    "right_left":      ["Go right then navigate to the left basket",
                        "Turn right and move toward the left-side basket",
                        "Rotate right then head to the left container",
                        "Turn right and approach the basket on the left"],
    "right_right":     ["Rotate right and head to the right basket",
                        "Turn right and reach the basket on the right",
                        "Go right and then move to the right-side container",
                        "Navigate right and find the basket on the right"],
}


# ──────────────────────────────────────────────
# 모델 (Exp49와 동일 구조)
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

def build_dataset(bbox_data, lv_cache):
    """
    lv_cache: ep_path → np.ndarray (N_frames, 2048)  lang-vis features
    """
    print(f"\n[DATA] 데이터셋 빌드 (WINDOW={WINDOW}, D_IN={D_IN})...")
    X, y, path_labels = [], [], []

    skipped = 0
    for ep_data in bbox_data:
        ep_path = ep_data["episode"]
        pt      = ep_data["path_type"]
        frames  = ep_data["frames"]

        lv_feats = lv_cache.get(ep_path)
        if lv_feats is None:
            skipped += 1
            continue

        fr0 = frames[0]
        if fr0["has_bbox"]:
            goal = np.array([fr0["cx"], fr0["cy"], fr0["area"]], dtype=np.float32)
        else:
            goal = np.array([0.5, 0.5, 0.0], dtype=np.float32)

        for t in range(len(frames)):
            bbox_feat = []
            for k in range(WINDOW):
                idx = max(0, t - (WINDOW - 1 - k))
                fr  = frames[idx]
                bbox_feat.extend([fr["cx"], fr["cy"], fr["area"], float(fr["has_bbox"])])

            feat = np.concatenate([
                np.array(bbox_feat, dtype=np.float32),
                lv_feats[t],
                goal,
            ])
            X.append(feat)
            y.append(frames[t]["gt_class"])
            path_labels.append(pt)

    print(f"  총 프레임: {len(X)}  (에피소드 skip: {skipped})")
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), path_labels


# ──────────────────────────────────────────────
# 학습 (Exp49와 동일)
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
# Paraphrase robustness 테스트
# ──────────────────────────────────────────────

def paraphrase_test(net, bbox_data):
    """
    9 path_type × 4 paraphrases = 36개 표현에 대해
    on-the-fly feature 추출 → 동일 action 예측 여부 확인
    """
    import gc
    from transformers import AutoModelForVision2Seq, AutoProcessor
    import h5py
    from PIL import Image

    print("\n" + "=" * 60)
    print("Paraphrase Robustness Test (9 path_type × 4 paraphrases)")
    print("=" * 60)

    HF_KOSMOS_PATH = ROOT / ".vlms" / "kosmos-2-patch14-224"
    proc  = AutoProcessor.from_pretrained(str(HF_KOSMOS_PATH))
    model_vlm = AutoModelForVision2Seq.from_pretrained(
        str(HF_KOSMOS_PATH), torch_dtype=torch.float16
    ).cuda().eval()

    device = next(net.parameters()).device
    net.eval()

    def extract_lv(pil_img, instruction):
        inputs = proc(text=instruction, images=pil_img, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        with torch.no_grad():
            out = model_vlm(**inputs, output_hidden_states=True)
        hs   = out.hidden_states[-1]
        mask = inputs["image_embeds_position_mask"][0].bool()
        return hs[0][mask].mean(0).float().cpu().numpy()

    def load_ep_imgs(ep_path):
        with h5py.File(ep_path, "r") as f:
            if "observations" in f and "images" in f["observations"]:
                return f["observations"]["images"][:]
            return f["images"][:]

    results = {}
    total_match = 0
    total_tests = 0

    for pt, paras in PARAPHRASES.items():
        ep_data = next((e for e in bbox_data if e["path_type"] == pt), None)
        if ep_data is None:
            continue

        frames  = ep_data["frames"]
        imgs    = load_ep_imgs(ep_data["episode"])
        instr_orig = INSTRUCTIONS[pt]

        # goal (frame 0 grounding) — 원본 instruction으로 추출된 goal 사용
        fr0  = frames[0]
        goal = np.array([fr0["cx"], fr0["cy"], fr0["area"]], dtype=np.float32) if fr0["has_bbox"] \
               else np.array([0.5, 0.5, 0.0], dtype=np.float32)

        # frame t=5 (중간 프레임)을 대표로 사용
        t = min(5, len(frames) - 1)
        pil = Image.fromarray(imgs[t].astype(np.uint8)).convert("RGB")

        bbox_feat = []
        for k in range(WINDOW):
            idx = max(0, t - (WINDOW - 1 - k))
            fr  = frames[idx]
            bbox_feat.extend([fr["cx"], fr["cy"], fr["area"], float(fr["has_bbox"])])
        bbox_arr = np.array(bbox_feat, dtype=np.float32)

        # 원본 instruction 예측
        lv_orig = extract_lv(pil, instr_orig)
        feat_orig = np.concatenate([bbox_arr, lv_orig, goal])
        with torch.no_grad():
            pred_orig = net(
                torch.tensor(feat_orig, dtype=torch.float32).unsqueeze(0).to(device)
            ).argmax(1).item()

        para_matches = []
        for para in paras:
            lv_para = extract_lv(pil, para)
            feat_para = np.concatenate([bbox_arr, lv_para, goal])
            with torch.no_grad():
                pred_para = net(
                    torch.tensor(feat_para, dtype=torch.float32).unsqueeze(0).to(device)
                ).argmax(1).item()
            match = (pred_para == pred_orig)
            para_matches.append(match)
            total_tests += 1
            if match:
                total_match += 1

        match_rate = sum(para_matches) / len(para_matches) * 100
        results[pt] = {
            "pred_orig": CLASS_NAMES[pred_orig],
            "paraphrase_match_rate": match_rate,
            "n_paraphrases": len(paras),
            "matches": para_matches,
        }
        status = "✅" if match_rate == 100 else ("⚠️" if match_rate >= 75 else "❌")
        print(f"  {status} {pt:<20}: pred={CLASS_NAMES[pred_orig]:<8}  paraphrase={match_rate:.0f}%")

    overall_para = total_match / total_tests * 100 if total_tests > 0 else 0.0
    print(f"\n  전체 paraphrase robustness: {overall_para:.1f}%  ({total_match}/{total_tests})")
    print(f"  Exp49 비교: 100% / Exp47: 74.1%")

    del model_vlm; gc.collect(); torch.cuda.empty_cache()

    return results, overall_para


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paraphrase_test", action="store_true", help="paraphrase 테스트만 실행")
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 60)
    print("Exp52: True VLA — Language-Conditioned Visual Features")
    print(f"D_IN = {D_IN}  (bbox={WINDOW*4} + lang_vis={LANG_VIS_DIM} + goal={GOAL_DIM})")
    print("=" * 60)

    # ── 데이터 로드 ──
    bbox_data = json.loads((EXP46_DIR / "bbox_dataset_full.json").read_text())

    # lang-vis features (Exp52)
    lv_npz  = np.load(str(EXP52_DIR / "lang_vis_features.npz"))
    lv_idx  = json.loads((EXP52_DIR / "lang_vis_features_index.json").read_text())
    lv_cache = {ep: lv_npz[f"ep_{i}"] for ep, i in lv_idx.items()}
    print(f"\n에피소드: {len(bbox_data)}, lang-vis 캐시: {len(lv_cache)}")

    if args.paraphrase_test:
        net = build_mlp()
        ckpt = torch.load(str(CKPT_PATH), map_location="cpu")
        net.load_state_dict(ckpt["model_state_dict"])
        net = net.cuda()
        para_results, para_overall = paraphrase_test(net, bbox_data)
        (EXP52_DIR / "paraphrase_results.json").write_text(
            json.dumps({"overall": para_overall, "by_path": para_results}, indent=2)
        )
        return

    # ── 데이터셋 빌드 ──
    X, y, path_labels = build_dataset(bbox_data, lv_cache)

    # ── Train/Val split (Exp49와 동일 seed) ──
    ep_labels = [ep["path_type"] for ep in bbox_data]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_ep_idx, te_ep_idx = next(sss.split(np.zeros(len(bbox_data)), ep_labels))

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

    # ── 학습 ──
    print("\n[TRAIN] MLP 학습 시작 (300 epochs)...")
    net, best_acc = train_mlp(X_tr, y_tr, X_te, y_te, epochs=300)

    # ── 평가 ──
    cm, overall = evaluate(net, X_te, y_te)

    # ── 저장 ──
    torch.save({
        "model_state_dict": net.state_dict(),
        "d_in":          D_IN,
        "window":        WINDOW,
        "lang_vis_dim":  LANG_VIS_DIM,
        "goal_dim":      GOAL_DIM,
        "overall_acc":   overall,
    }, str(CKPT_PATH))

    summary = {
        "model":        "exp52",
        "overall_acc":  float(overall),
        "best_val_acc": float(best_acc),
        "n_train":      int(len(X_tr)),
        "n_val":        int(len(X_te)),
        "d_in":         D_IN,
        "window":       WINDOW,
        "lang_vis_dim": LANG_VIS_DIM,
        "goal_dim":     GOAL_DIM,
        "confusion":    cm.tolist(),
        "comparison": {
            "exp46_acc": 0.9316,
            "exp47_acc": 0.9867,
            "exp49_acc": 0.9637,
            "exp52_acc": float(overall),
        },
    }
    (EXP52_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Exp52 acc:   {overall:.1%}")
    print(f"  Exp49 비교:  96.4% → {overall:.1%}")
    print(f"  총 소요: {elapsed/60:.1f}분")
    print(f"  ckpt: {CKPT_PATH}")
    print("=" * 60)
    print("\n다음: .venv/bin/python3 scripts/train_v5_exp52_true_vla.py --paraphrase_test")


if __name__ == "__main__":
    main()

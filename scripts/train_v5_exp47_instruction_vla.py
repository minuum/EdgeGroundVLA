#!/usr/bin/env python3
"""
Exp47: Instruction-Conditioned VLA MLP
  Exp46 기반 + instruction embedding 추가 → 진짜 VLA

  입력: bbox(8×4=32) + vision(1024) + instruction_emb(2048) = 3104-dim
  출력: 8-class action

  Synthetic label: path_type → 고정 instruction 문장
  → Kosmos-2 text encoder로 2048-dim 임베딩 추출 후 MLP 입력

Usage:
  python3 scripts/train_v5_exp47_instruction_vla.py
  python3 scripts/train_v5_exp47_instruction_vla.py --skip_instr   # 캐시 재사용
  python3 scripts/train_v5_exp47_instruction_vla.py --epochs 300
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HF_KOSMOS   = ROOT / ".vlms" / "kosmos-2-patch14-224"
BBOX_CACHE  = ROOT / "docs" / "v5" / "bbox_nav_exp46" / "bbox_dataset_full.json"
VIS_CACHE   = ROOT / "docs" / "v5" / "bbox_nav_exp46" / "vision_features.npz"
VIS_IDX     = ROOT / "docs" / "v5" / "bbox_nav_exp46" / "vision_features_index.json"
CANDS_PATH  = ROOT / "docs" / "v5" / "instruction_candidates.json"
OUT_DIR     = ROOT / "docs" / "v5" / "bbox_nav_exp47"
MLP_DIR     = ROOT / "runs" / "v5_nav" / "mlp" / "exp47"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MLP_DIR.mkdir(parents=True, exist_ok=True)

INSTR_CACHE = OUT_DIR / "instruction_embeddings.json"
CKPT_PATH   = MLP_DIR / "exp47_mlp.pt"

CLASS_NAMES = ["STOP","FORWARD","LEFT","RIGHT","FWD+L","FWD+R","ROT_L","ROT_R"]
NUM_CLASSES = 8
WINDOW      = 8
VIS_DIM     = 1024
INSTR_DIM   = 2048
D_IN        = WINDOW * 4 + VIS_DIM + INSTR_DIM  # 3104

# path_type → synthetic instruction (top-1 from candidates)
INSTR_MAP = {
    "center_straight": "Drive forward along the center path to the basket ahead.",
    "center_left":     "Move toward the basket by swinging left from the center.",
    "center_right":    "Approach the basket by gradually turning right while moving forward.",
    "left_straight":   "Approach the basket on your left by first turning to face it, then going straight.",
    "left_left":       "Navigate to the gray basket on the left side with a left-curving path.",
    "left_right":      "Navigate to the left-side basket by curving to the right.",
    "right_straight":  "Approach the basket on your right by first turning to face it, then going straight.",
    "right_left":      "Navigate to the right-side basket by curving to the left.",
    "right_right":     "Navigate to the gray basket on the right side with a right-curving path.",
}


# ──────────────────────────────────────────────
# Instruction embedding 추출
# ──────────────────────────────────────────────

def extract_instruction_embeddings():
    print("\n[INSTR] Kosmos-2 text encoder로 instruction 임베딩 추출...")
    from transformers import AutoModelForVision2Seq, AutoProcessor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proc  = AutoProcessor.from_pretrained(str(HF_KOSMOS), trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        str(HF_KOSMOS), trust_remote_code=True
    ).to(device).eval()

    embeddings = {}
    for pt, instr in INSTR_MAP.items():
        inputs   = proc(text=instr, images=None, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        with torch.no_grad():
            out = model.text_model(input_ids=input_ids, output_hidden_states=True)
            emb = out.hidden_states[-1][0].mean(0).cpu().float().numpy()
        embeddings[pt] = emb.tolist()
        print(f"  [{pt}] dim={len(emb)}  '{instr[:55]}...'")

    # 저장
    INSTR_CACHE.write_text(json.dumps(
        {pt: v for pt, v in embeddings.items()}, indent=2
    ))
    print(f"  → 저장: {INSTR_CACHE}")

    # GPU 해제
    del model
    torch.cuda.empty_cache()

    return {pt: np.array(v, dtype=np.float32) for pt, v in embeddings.items()}


# ──────────────────────────────────────────────
# 데이터셋 빌드
# ──────────────────────────────────────────────

def build_dataset(bbox_data, vis_cache, instr_embs):
    print(f"\n[DATA] 데이터셋 빌드 (WINDOW={WINDOW}, D_IN={D_IN})...")
    X, y, path_labels = [], [], []

    for ep_data in bbox_data:
        ep_path = ep_data["episode"]
        pt      = ep_data["path_type"]
        frames  = ep_data["frames"]

        vis_feats  = vis_cache.get(ep_path)
        instr_emb  = instr_embs.get(pt)
        if vis_feats is None or instr_emb is None:
            print(f"  SKIP: {Path(ep_path).name}")
            continue

        for t in range(len(frames)):
            bbox_feat = []
            for k in range(WINDOW):
                idx = max(0, t - (WINDOW - 1 - k))
                fr  = frames[idx]
                bbox_feat.extend([fr["cx"], fr["cy"], fr["area"], float(fr["has_bbox"])])

            feat = np.concatenate([
                np.array(bbox_feat, dtype=np.float32),
                vis_feats[t],   # (1024,)
                instr_emb,      # (2048,)
            ])
            X.append(feat)
            y.append(frames[t]["gt_class"])
            path_labels.append(pt)

    X = np.stack(X)
    y = np.array(y)
    print(f"  총 샘플: {len(X)}, 입력 dim: {X.shape[1]}")
    return X, y, path_labels


# ──────────────────────────────────────────────
# MLP
# ──────────────────────────────────────────────

def build_mlp(d_in=D_IN):
    return nn.Sequential(
        nn.Linear(d_in, 512), nn.ReLU(), nn.Dropout(0.25),
        nn.Linear(512, 256),  nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 128),  nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(128, 64),   nn.ReLU(),
        nn.Linear(64, NUM_CLASSES),
    )


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

        if ep % 10 == 0 or ep == epochs:
            net.eval()
            with torch.no_grad():
                acc = (net(Xte_t).argmax(1) == yte_t).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_state = {k: v.cpu().clone() for k, v in net.state_dict().items()}
            print(f"  epoch {ep:>3}  val_acc={acc:.3f}  best={best_acc:.3f}")

    net.load_state_dict(best_state)
    return net, best_acc


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
# Sensitivity test: instruction 교체 시 action 변화
# ──────────────────────────────────────────────

def sensitivity_test(net, bbox_data, vis_cache, instr_embs):
    print("\n=== Sensitivity Test: instruction 교체 효과 ===")
    device = next(net.parameters()).device
    net.eval()

    # val 에피소드 30개에서 무작위 10개 추출
    from sklearn.model_selection import StratifiedShuffleSplit
    path_labels = [ep["path_type"] for ep in bbox_data]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, te_idx = next(sss.split(np.zeros(len(bbox_data)), path_labels))
    val_eps = [bbox_data[i] for i in te_idx[:10]]

    results = []
    for ep_data in val_eps:
        ep_path = ep_data["episode"]
        pt      = ep_data["path_type"]
        frames  = ep_data["frames"]
        vis_feats = vis_cache.get(ep_path)
        if vis_feats is None:
            continue

        correct_instr = instr_embs[pt]

        # 반대 방향 instruction 선택
        opposites = {
            "center_left": "center_right", "center_right": "center_left",
            "left_left":   "left_right",   "left_right":   "left_left",
            "right_left":  "right_right",  "right_right":  "right_left",
            "left_straight": "right_straight", "right_straight": "left_straight",
            "center_straight": "center_left",
        }
        opp_pt     = opposites.get(pt, "center_right")
        wrong_instr = instr_embs[opp_pt]

        # 첫 프레임으로 테스트
        t = 0
        bbox_feat = []
        for k in range(WINDOW):
            idx = max(0, t - (WINDOW - 1 - k))
            fr  = frames[idx]
            bbox_feat.extend([fr["cx"], fr["cy"], fr["area"], float(fr["has_bbox"])])

        vis = vis_feats[t]
        bbox_arr = np.array(bbox_feat, dtype=np.float32)

        def predict(instr_emb):
            feat = np.concatenate([bbox_arr, vis, instr_emb])
            x    = torch.tensor([feat], dtype=torch.float32, device=device)
            with torch.no_grad():
                cls = int(net(x).argmax(1).item())
            return cls

        pred_correct = predict(correct_instr)
        pred_wrong   = predict(wrong_instr)
        changed      = pred_correct != pred_wrong

        results.append({
            "path_type": pt,
            "opposite":  opp_pt,
            "pred_correct_instr": CLASS_NAMES[pred_correct],
            "pred_wrong_instr":   CLASS_NAMES[pred_wrong],
            "changed": changed,
        })

    n_changed = sum(r["changed"] for r in results)
    print(f"\n  테스트 에피소드: {len(results)}개")
    print(f"  instruction 교체 시 action 변화: {n_changed}/{len(results)} ({n_changed/max(len(results),1):.1%})")
    print()
    for r in results:
        mark = "✅" if r["changed"] else "❌"
        print(f"  {mark} [{r['path_type']}→{r['opposite']}]  "
              f"correct:{r['pred_correct_instr']}  wrong:{r['pred_wrong_instr']}")

    return results


# ──────────────────────────────────────────────
# main
# ──────────────────────────────────────────────

def main():
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_instr", action="store_true", help="instruction 임베딩 캐시 재사용")
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()

    # Instruction embeddings
    if args.skip_instr and INSTR_CACHE.exists():
        print(f"[INSTR] 캐시 재사용: {INSTR_CACHE}")
        raw = json.loads(INSTR_CACHE.read_text())
        instr_embs = {pt: np.array(v, dtype=np.float32) for pt, v in raw.items()}
    else:
        instr_embs = extract_instruction_embeddings()

    print(f"\n  instruction 종류: {len(instr_embs)}")
    for pt, instr in INSTR_MAP.items():
        print(f"  [{pt}] \"{instr}\"")

    # Exp46 캐시 로드
    print("\n[DATA] Exp46 캐시 로드...")
    bbox_data = json.loads(BBOX_CACHE.read_text())
    vis_index = json.loads(VIS_IDX.read_text())
    npz       = np.load(str(VIS_CACHE))
    vis_cache = {ep: npz[f"ep_{i}"] for ep, i in vis_index.items()}
    print(f"  에피소드: {len(bbox_data)}, vision cache: {len(vis_cache)}")

    # 데이터셋 빌드
    X, y, path_labels = build_dataset(bbox_data, vis_cache, instr_embs)

    # Train/val split (동일 seed=42)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(sss.split(X, path_labels))
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]
    print(f"  train: {len(X_tr)}, val: {len(X_te)}")

    # 학습
    print(f"\n[TRAIN] MLP {X.shape[1]}→512→256→128→64→{NUM_CLASSES}, epochs={args.epochs}")
    net, best_acc = train_mlp(X_tr, y_tr, X_te, y_te, epochs=args.epochs)

    # 평가
    cm, overall = evaluate(net, X_te, y_te)

    # Sensitivity test
    sens = sensitivity_test(net, bbox_data, vis_cache, instr_embs)

    # 저장
    torch.save({
        "model_state_dict": net.state_dict(),
        "d_in":      X.shape[1],
        "window":    WINDOW,
        "vis_dim":   VIS_DIM,
        "instr_dim": INSTR_DIM,
        "instr_map": INSTR_MAP,
        "overall_acc": overall,
    }, str(CKPT_PATH))

    summary = {
        "model":        "exp47",
        "overall_acc":  float(overall),
        "best_val_acc": float(best_acc),
        "n_train":      int(len(X_tr)),
        "n_val":        int(len(X_te)),
        "d_in":         int(X.shape[1]),
        "window":       WINDOW,
        "vis_dim":      VIS_DIM,
        "instr_dim":    INSTR_DIM,
        "instr_map":    INSTR_MAP,
        "sensitivity":  {
            "n_tested":  len(sens),
            "n_changed": sum(r["changed"] for r in sens),
            "change_rate": sum(r["changed"] for r in sens) / max(len(sens), 1),
            "details":   sens,
        },
        "confusion": cm.tolist(),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    elapsed = time.time() - t0
    print(f"\n✅ 완료!  acc={overall:.1%}  ckpt={CKPT_PATH}")
    print(f"   총 소요: {elapsed/60:.1f}분")


if __name__ == "__main__":
    main()

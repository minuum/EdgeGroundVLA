#!/usr/bin/env python3
"""
2026-07-06 실로봇 obj_right 실패 세션 2개의 실제 bbox/이미지 궤적을
Step2 decomposition MLP(bbox history + 16x16 image feature)에 통과시켜,
exp71(운영 헤드, 이 세션에서 FORWARD/ROT_R만 냄)과 다른 방향을 냈을지 확인.

목적: 방향 편향이 "데이터 문제"인지 "헤드 구조(exp71 단일 softmax) 문제"인지
서버에서 로봇 없이 1차로 가른다.
"""
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("step2", ROOT / "scripts" / "test_v5_bbox_nav_step2.py")
step2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(step2)

SESS_DIR = Path("/home/minum/MoNaVLA/inference_sessions_recv/20260706")
SESSIONS = ["session_20260706_171922.h5", "session_20260706_172030.h5"]

CLASS_NAMES = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "ROT_L", "ROT_R"]


def classify_action(vx, vy, wz):
    if abs(vx) < 0.1 and abs(vy) < 0.1 and abs(wz) < 0.1:
        return 0
    if abs(wz) > 0.15 and abs(vx) < 0.2 and abs(vy) < 0.2:
        return 6 if wz > 0 else 7
    if vx > 0.3 and vy > 0.3:
        return 4
    if vx > 0.3 and vy < -0.3:
        return 5
    if vx > 0.3:
        return 1
    if vy > 0.3:
        return 2
    if vy < -0.3:
        return 3
    return 1


def build_live_windows(h5_path, window=step2.WINDOW):
    with h5py.File(h5_path, "r") as f:
        bbox = f["grounding/bbox"][:]  # (N,4) cx,cy,area,has
        imgs = f["observations/images"][:]  # (N,H,W,3)
        acts = f["actions"][:]

    n = len(acts)
    img_feats = [step2.frame_to_small_feature(imgs[t]) for t in range(n)]
    X = []
    for t in range(n):
        feat = []
        for k in range(window):
            idx = max(0, t - (window - 1 - k))
            cx, cy, area, has = bbox[idx]
            feat.extend([float(cx), float(cy), float(area), float(has)])
        feat.extend(img_feats[t].tolist())
        X.append(feat)
    exp71_classes = [classify_action(*acts[t]) for t in range(n)]
    return np.asarray(X, dtype=np.float32), exp71_classes, bbox


def main():
    print("=== Step2 decomposition MLP 학습 (기존 레시피 그대로, bbox_dataset.json) ===")
    dataset = step2.load_dataset()
    dataset = [ep for ep in dataset if list(step2.DATA_DIR.glob(f"{ep['episode']}.h5"))]
    train_ds, test_ds = step2.make_episode_split(dataset)
    X_tr, y_tr, _ = step2.build_windows(train_ds)
    X_te, y_te, _ = step2.build_windows(test_ds)
    print(f"train={len(X_tr)} test={len(X_te)}  (D={X_tr.shape[1]})")

    d_in = X_tr.shape[1]
    import torch.nn as nn
    model = nn.Sequential(
        nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(0.25),
        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, step2.NUM_CLASSES),
    ).to(step2.DEVICE)

    cls_counts = np.bincount(y_tr, minlength=step2.NUM_CLASSES).astype(np.float32)
    cls_counts = np.where(cls_counts == 0, 1.0, cls_counts)
    weights = torch.tensor(1.0 / cls_counts, dtype=torch.float32, device=step2.DEVICE)
    weights = weights / weights.sum() * step2.NUM_CLASSES
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    X_tr_t = torch.tensor(X_tr, device=step2.DEVICE)
    y_tr_t = torch.tensor(y_tr, device=step2.DEVICE)
    X_te_t = torch.tensor(X_te, device=step2.DEVICE)
    y_te_t = torch.tensor(y_te, device=step2.DEVICE)

    best_acc, best_state = 0.0, None
    for ep in range(220):
        model.train()
        idx = torch.randperm(len(X_tr_t))
        for i in range(0, len(idx), 128):
            b = idx[i:i + 128]
            logits = model(X_tr_t[b])
            loss = loss_fn(logits, y_tr_t[b])
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
    print(f"학습 완료: test PM={best_acc:.1%} (기존 기록 75.9% 대와 재현성 확인용)")

    results = {}
    for sess in SESSIONS:
        h5p = SESS_DIR / sess
        X_live, exp71_cls, bbox = build_live_windows(h5p)
        with torch.no_grad():
            logits = model(torch.tensor(X_live, device=step2.DEVICE))
            step2_cls = logits.argmax(dim=-1).cpu().numpy().tolist()

        print(f"\n=== {sess} ===")
        print(f"{'t':>3} {'cx':>6} {'has':>4}  {'exp71(실제)':<10} {'step2(재예측)':<10}")
        rows = []
        for t in range(len(exp71_cls)):
            cx, cy, area, has = bbox[t]
            e = CLASS_NAMES[exp71_cls[t]]
            s = CLASS_NAMES[step2_cls[t]]
            mark = "  <-- 다름" if e != s else ""
            print(f"{t:>3} {cx:6.3f} {int(has):>4}  {e:<10} {s:<10}{mark}")
            rows.append({"t": t, "cx": float(cx), "has_bbox": int(has),
                         "exp71": e, "step2": s})
        diff = sum(1 for r in rows if r["exp71"] != r["step2"])
        print(f"exp71과 다른 예측: {diff}/{len(rows)}")
        results[sess] = {"rows": rows, "diff_count": diff, "n": len(rows)}

    out = ROOT / "docs" / "v5" / "closed_loop_eval" / "live_20260706_step2_replay.json"
    out.write_text(json.dumps({"step2_test_pm": best_acc, "sessions": results}, ensure_ascii=False, indent=2))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()

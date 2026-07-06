#!/usr/bin/env python3
"""
soda가 보낸 실제 7세션(72~78, 진짜 카메라 피드백이 있던 라이브 데이터, 진짜 OWL-v2
flicker)의 bbox 스트림을 그대로 꺼내서, 운영 중인 window6(exp71_window6 체크포인트
원본) vs 재학습한 window3(격리 재학습본, held-out truth_mini 98.6%) 헤드에 각각
순서대로 넣어보고 실제로 다르게 예측하는지 비교.

합성 flicker가 아니라 진짜 세션의 실측 bbox라 지금까지 중 가장 신뢰도 높은 검증.
단, 카메라 자체는 이미 녹화된 그대로라 "다르게 예측했으면 실제로 궤적이 나아졌을지"는
알 수 없음 — 어디까지나 "같은 입력에 다른 헤드가 다르게 반응하는가"만 확인.
"""
import importlib.util
import json
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("exp71", ROOT / "scripts" / "train_exp71_stage2_transformer.py")
exp71 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp71)

SESS_DIR = Path("/home/minum/MoNaVLA/inference_sessions_recv/20260706")
SESSIONS = ["231153", "231407", "231523", "231638", "233159", "233327", "233424"]
PROD_CKPT = ROOT / "runs/v5_nav/mlp/exp71_window6/action_transformer.pt"
CACHE_FILE = ROOT / "docs" / "v5" / "closed_loop_eval" / "exp71_vis_cache.pt"
TRUTHMINI = ROOT / "docs" / "v5" / "bbox_truth_mini.json"
CLASS_NAMES = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "ROT_L", "ROT_R"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def load_prod_w6():
    model = exp71.TransformerActionHead(window=6).to(DEVICE)
    ckpt = torch.load(PROD_CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"운영 window6 체크포인트 로드: val_acc={ckpt.get('val_acc'):.4f}")
    return model


def retrain_window3_isolated(enc):
    """truth_mini 18ep 격리 + window3 — 지난 실험(held-out acc 98.6%)과 동일 레시피 재현."""
    import torch.nn as nn
    import torch.nn.functional as F

    with open(exp71.ANN_PATH) as f:
        ann = json.load(f)
    random.seed(42)
    random.shuffle(ann)
    n_val = max(1, int(len(ann) * 0.15))
    val_ann, train_ann = ann[:n_val], ann[n_val:]

    tm = json.loads(TRUTHMINI.read_text())
    truthmini_stems = set(a["episode"] for a in tm["annotations"])

    def filt_exclude(lst):
        out = []
        for ep in lst:
            h5p = Path(ep["episode"])
            if h5p.stem in truthmini_stems:
                continue
            frames = [fr for fr in ep["frames"] if fr.get("gt_class") is not None]
            if h5p.exists() and frames:
                out.append(ep)
        return out

    train_ann2 = filt_exclude(train_ann)
    val_ann2 = filt_exclude(val_ann)
    print(f"window3 재학습: train={len(train_ann2)} val={len(val_ann2)} (truth_mini 18ep 제외)")

    X_tr, y_tr = exp71.build_dataset(train_ann2, enc, DEVICE, window=3)
    X_va, y_va = exp71.build_dataset(val_ann2, enc, DEVICE, window=3)
    X_tr_t = torch.from_numpy(X_tr).to(DEVICE)
    y_tr_t = torch.from_numpy(y_tr).to(DEVICE)
    X_va_t = torch.from_numpy(X_va).to(DEVICE)
    y_va_t = torch.from_numpy(y_va).to(DEVICE)

    model = exp71.TransformerActionHead(window=3).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 300)
    best_acc, best_state = 0.0, None
    for ep in range(1, 301):
        model.train()
        perm = torch.randperm(len(X_tr_t), device=DEVICE)
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            logits = model(X_tr_t[idx])
            loss = F.cross_entropy(logits, y_tr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 50 == 0 or ep == 300:
            model.eval()
            with torch.no_grad():
                acc = (model(X_va_t).argmax(1) == y_va_t).float().mean().item()
            if acc >= best_acc:
                best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    print(f"window3 재학습 완료: val_acc={best_acc:.1%}")
    return model


def predict_seq(model, window, bboxes, vis_feats):
    n = len(bboxes)
    preds = []
    for t in range(n):
        seq = []
        for k in range(window):
            idx = max(0, t - (window - 1 - k))
            seq.append(list(bboxes[idx]) + vis_feats[idx].tolist())
        x = torch.tensor([seq], dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            preds.append(model(x).argmax(dim=-1).item())
    return preds


def main():
    print(f"device={DEVICE}")
    model_w6 = load_prod_w6()

    print("FrozenCLIPV2 로드...")
    enc = exp71.FrozenCLIPV2(exp71.VLM_PATH, exp71.STAGE1_PT, DEVICE).eval()

    model_w3 = retrain_window3_isolated(enc)

    print("\n" + "=" * 70)
    print("soda 실제 7세션 실측 bbox로 window6(운영) vs window3(재학습) 비교")
    print("=" * 70)

    LEFT_CLASSES, RIGHT_CLASSES = {2, 4, 6}, {3, 5, 7}
    summary = []
    for s in SESSIONS:
        fp = SESS_DIR / f"session_20260706_{s}.h5"
        with h5py.File(fp, "r") as f:
            bbox_raw = f["grounding/bbox"][:]  # cx,cy,area,has
            imgs = f["observations/images"][:]
            acts = f["actions"][:]
        n = len(acts)
        bboxes = [tuple(float(v) for v in bbox_raw[t]) for t in range(n)]
        from PIL import Image
        pil_imgs = [Image.fromarray(imgs[t].astype("uint8")) for t in range(n)]
        vis = enc.encode_batch(pil_imgs, DEVICE).cpu()

        actual_cls = [classify_action(*acts[t]) for t in range(n)]
        pred_w6 = predict_seq(model_w6, 6, bboxes, vis)
        pred_w3 = predict_seq(model_w3, 3, bboxes, vis)

        def osc_rate(seq):
            flips = pairs = 0
            for a, b in zip(seq[:-1], seq[1:]):
                sa = "L" if a in LEFT_CLASSES else ("R" if a in RIGHT_CLASSES else None)
                sb = "L" if b in LEFT_CLASSES else ("R" if b in RIGHT_CLASSES else None)
                if sa and sb:
                    pairs += 1
                    flips += (sa != sb)
            return (flips / pairs) if pairs else 0.0

        print(f"\n--- session {s} (n={n}) ---")
        print(f"{'t':>3} {'cx':>6} {'has':>4}  {'실제(운영w6 로그)':<10} {'재현w6':<10} {'재현w3':<10}")
        for t in range(n):
            mark = ""
            if pred_w6[t] != pred_w3[t]:
                mark = "  <-- w3가 다르게 예측"
            print(f"{t:>3} {bboxes[t][0]:6.3f} {int(bboxes[t][3]):>4}  "
                  f"{CLASS_NAMES[actual_cls[t]]:<10} {CLASS_NAMES[pred_w6[t]]:<10} {CLASS_NAMES[pred_w3[t]]:<10}{mark}")

        rec = {
            "session": s, "n": n,
            "osc_actual": osc_rate(actual_cls),
            "osc_pred_w6_repro": osc_rate(pred_w6),
            "osc_pred_w3": osc_rate(pred_w3),
            "agree_w6_actual": float(np.mean([a == b for a, b in zip(actual_cls, pred_w6)])),
            "diff_w3_vs_w6": sum(1 for a, b in zip(pred_w6, pred_w3) if a != b),
        }
        summary.append(rec)
        print(f"진동율: 실제={rec['osc_actual']:.1%}  w6재현={rec['osc_pred_w6_repro']:.1%}  "
              f"w3={rec['osc_pred_w3']:.1%}   w6재현이 실제 로그와 일치={rec['agree_w6_actual']:.1%}   "
              f"w3가 다르게 예측한 프레임={rec['diff_w3_vs_w6']}/{n}")

    out = ROOT / "docs" / "v5" / "closed_loop_eval" / "live_sessions_window_compare.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nsaved -> {out}")

    print("\n=== 요약 ===")
    print(f"평균 진동율: 실제={np.mean([r['osc_actual'] for r in summary]):.1%}  "
          f"w6재현={np.mean([r['osc_pred_w6_repro'] for r in summary]):.1%}  "
          f"w3={np.mean([r['osc_pred_w3'] for r in summary]):.1%}")
    print(f"w6 재현이 실제 운영 로그와 일치한 비율(평균): "
          f"{np.mean([r['agree_w6_actual'] for r in summary]):.1%}  (검증 sanity check)")


if __name__ == "__main__":
    main()

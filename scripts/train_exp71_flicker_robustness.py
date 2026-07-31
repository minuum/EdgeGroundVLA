#!/usr/bin/env python3
"""
진짜 exp71 레시피(FrozenCLIPV2 vision + Transformer, 150ep pg448 데이터)로
flicker 강건성 변형을 재학습/비교한다.

45ep 경량 MLP 프록시(train_owl_flicker_robustness.py)로 먼저 해봤지만, 그건
exp71과 다른 모델/데이터라 결과를 그대로 못 씀 (2026-07-06 사용자 지적) — 이번엔
실제 exp71 학습 스크립트(scripts/train_exp71_stage2_transformer.py)를 그대로 재사용.

변형:
  1. baseline      : 원본 재현 (기존 exp71_window6와 동일 레시피/시드)
  2. dropout_aug   : 학습 시 bbox를 p=0.5로 fallback(cx=0.5,cy=0.5,area=0.05,has=0) 치환
  3. sticky_aug    : 위와 같되 fallback 대신 마지막 실검출값 유지
  4. window3       : window만 6->3

비전 인코딩(FrozenCLIPV2, 가장 비싼 연산)은 에피소드당 1회만 수행해서 캐시하고,
증강은 캐시된 (bbox, vis_feat, gt) 시퀀스 위에서 수행 — 변형 4개를 다 인코딩
다시 안 하고 재사용.
"""
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("exp71", ROOT / "scripts" / "train_exp71_stage2_transformer.py")
exp71 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp71)

CACHE_FILE = ROOT / "docs" / "v5" / "closed_loop_eval" / "exp71_vis_cache.pt"
OUT_FILE = ROOT / "docs" / "v5" / "closed_loop_eval" / "exp71_flicker_robustness_compare.json"
FALLBACK = (0.5, 0.5, 0.05, 0.0)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_episode_cache(ann, enc, device):
    """에피소드별 (bboxes, vis_feats(N,256), gts) 캐시. 한 번만 인코딩."""
    episodes = []
    for ep in ann:
        h5_path = Path(ep["episode"])
        if not h5_path.exists():
            continue
        frames = [fr for fr in ep["frames"] if fr.get("gt_class") is not None]
        if not frames:
            continue
        try:
            with h5py.File(str(h5_path), "r") as f:
                imgs_np = f["observations"]["images"][:]
        except Exception:
            continue
        pil_imgs = [Image.fromarray(imgs_np[fr["frame_idx"]].astype("uint8")) for fr in frames]
        vis = enc.encode_batch(pil_imgs, device).cpu()  # (n, 256)
        bboxes = [(fr.get("cx_det", 0.5), fr.get("cy_det", 0.5),
                   fr.get("area_det", 0.05), float(fr.get("has_bbox", False))) for fr in frames]
        gts = [fr["gt_class"] for fr in frames]
        episodes.append({"bboxes": bboxes, "vis": vis, "gts": gts})
    return episodes


def apply_aug(bboxes, p, sticky, rng):
    if p <= 0:
        return bboxes
    out = []
    last_real = FALLBACK
    for b in bboxes:
        if b[3] > 0.5:
            last_real = b
        if rng.random() < p:
            out.append((last_real[0], last_real[1], last_real[2], 0.0) if sticky else FALLBACK)
        else:
            out.append(b)
    return out


def build_windows(episodes, window, aug_p=0.0, sticky=False, rng=None):
    X, y = [], []
    for ep in episodes:
        bboxes = apply_aug(ep["bboxes"], aug_p, sticky, rng) if aug_p > 0 else ep["bboxes"]
        vis = ep["vis"]
        gts = ep["gts"]
        n = len(gts)
        for t in range(n):
            seq = []
            for k in range(window):
                idx = max(0, t - (window - 1 - k))
                seq.append(list(bboxes[idx]) + vis[idx].tolist())
            X.append(seq)
            y.append(gts[t])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def train_one(name, train_eps, val_eps, window, aug_p, sticky, epochs=300, lr=5e-4, seed=42):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    X_tr, y_tr = build_windows(train_eps, window, aug_p=aug_p, sticky=sticky, rng=rng)
    X_va, y_va = build_windows(val_eps, window, aug_p=0.0)

    X_tr_t = torch.from_numpy(X_tr).to(DEVICE)
    y_tr_t = torch.from_numpy(y_tr).to(DEVICE)
    X_va_t = torch.from_numpy(X_va).to(DEVICE)
    y_va_t = torch.from_numpy(y_va).to(DEVICE)

    model = exp71.TransformerActionHead(window=window).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)

    best_acc, best_state = 0.0, None
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(X_tr_t), device=DEVICE)
        for i in range(0, len(perm), 128):
            idx = perm[i:i + 128]
            logits = model(X_tr_t[idx])
            loss = F.cross_entropy(logits, y_tr_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 50 == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                acc = (model(X_va_t).argmax(1) == y_va_t).float().mean().item()
            if acc >= best_acc:
                best_acc = acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    # 상관형 flicker(근접 직후 집중 dropout) + 순차예측 진동율 측정
    LEFT_CLASSES, RIGHT_CLASSES = {2, 4, 6}, {3, 5, 7}
    AREA_NEAR, P_BASE, P_NEAR = 0.15, 0.1, 0.85

    def correlated_flicker(bboxes, rng2):
        out, was_near = [], False
        for b in bboxes:
            p = P_NEAR if was_near else P_BASE
            out.append(FALLBACK if rng2.random() < p else b)
            was_near = (b[3] > 0.5 and b[2] > AREA_NEAR)
        return out

    osc_rates = []
    for trial in range(5):
        rng2 = np.random.default_rng(500 + trial)
        for ep in val_eps:
            bboxes = correlated_flicker(ep["bboxes"], rng2)
            vis = ep["vis"]
            n = len(ep["gts"])
            preds = []
            for t in range(n):
                seq = []
                for k in range(window):
                    idx = max(0, t - (window - 1 - k))
                    seq.append(list(bboxes[idx]) + vis[idx].tolist())
                x = torch.tensor([seq], dtype=torch.float32, device=DEVICE)
                with torch.no_grad():
                    preds.append(model(x).argmax(dim=-1).item())
            flips = pairs = 0
            for a, b in zip(preds[:-1], preds[1:]):
                sa = "L" if a in LEFT_CLASSES else ("R" if a in RIGHT_CLASSES else None)
                sb = "L" if b in LEFT_CLASSES else ("R" if b in RIGHT_CLASSES else None)
                if sa and sb:
                    pairs += 1
                    flips += (sa != sb)
            if pairs:
                osc_rates.append(flips / pairs)

    result = {
        "val_acc": float(best_acc),
        "oscillation_rate_mean": float(np.mean(osc_rates)) if osc_rates else None,
        "oscillation_rate_std": float(np.std(osc_rates)) if osc_rates else None,
        "n_osc_samples": len(osc_rates),
    }
    print(f"[{name:14s}] val_acc={best_acc:.1%}  진동율={result['oscillation_rate_mean']:.1%}"
          f"±{result['oscillation_rate_std']:.1%}  (n={len(osc_rates)})")
    return result


def main():
    device = DEVICE
    print(f"device={device}")

    if CACHE_FILE.exists():
        print(f"캐시 로드: {CACHE_FILE}")
        cache = torch.load(CACHE_FILE, weights_only=False)
        train_eps, val_eps = cache["train"], cache["val"]
    else:
        import random
        with open(exp71.ANN_PATH) as f:
            ann = json.load(f)
        random.seed(42)
        random.shuffle(ann)
        n_val = max(1, int(len(ann) * 0.15))
        val_ann, train_ann = ann[:n_val], ann[n_val:]

        print("FrozenCLIPV2 로드 중...")
        enc = exp71.FrozenCLIPV2(exp71.VLM_PATH, exp71.STAGE1_PT, device).eval()
        print(f"[DATA] Train {len(train_ann)} / Val {len(val_ann)} eps — 인코딩 중 (한 번만)...")
        train_eps = build_episode_cache(train_ann, enc, device)
        val_eps = build_episode_cache(val_ann, enc, device)
        torch.save({"train": train_eps, "val": val_eps}, CACHE_FILE)
        print(f"캐시 저장 -> {CACHE_FILE}")
        del enc
        torch.cuda.empty_cache()

    print(f"train episodes={len(train_eps)}  val episodes={len(val_eps)}")

    results = {}
    if OUT_FILE.exists():
        results = json.loads(OUT_FILE.read_text())
        print(f"기존 결과 로드({len(results)}개) — 조합 실험만 추가")
    else:
        results["1_baseline_w6"] = train_one("baseline_w6", train_eps, val_eps, window=6, aug_p=0.0, sticky=False)
        results["2_dropout_aug_w6"] = train_one("dropout_aug_w6", train_eps, val_eps, window=6, aug_p=0.5, sticky=False)
        results["3_sticky_aug_w6"] = train_one("sticky_aug_w6", train_eps, val_eps, window=6, aug_p=0.5, sticky=True)
        results["4_window3"] = train_one("window3", train_eps, val_eps, window=3, aug_p=0.0, sticky=False)

    # 조합 ablation: sticky x window, dropout x window(대조), sticky 확률 sweep
    results["5_sticky_w3"] = train_one("sticky_w3", train_eps, val_eps, window=3, aug_p=0.5, sticky=True)
    results["6_dropout_w3"] = train_one("dropout_w3", train_eps, val_eps, window=3, aug_p=0.5, sticky=False)
    results["7_sticky_w6_p0.3"] = train_one("sticky_w6_p0.3", train_eps, val_eps, window=6, aug_p=0.3, sticky=True)
    results["8_sticky_w6_p0.6"] = train_one("sticky_w6_p0.6", train_eps, val_eps, window=6, aug_p=0.6, sticky=True)
    results["9_sticky_w3_p0.3"] = train_one("sticky_w3_p0.3", train_eps, val_eps, window=3, aug_p=0.3, sticky=True)

    OUT_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nsaved -> {OUT_FILE}")


if __name__ == "__main__":
    main()

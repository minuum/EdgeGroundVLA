#!/usr/bin/env python3
"""
2026-07-07 사용자 지적 반영:
A) truth_mini(18ep)를 학습에서 완전히 제외하고 재학습 -> 진짜 held-out으로 clean-bbox 검증
B) 실제 궤적 리플레이(rollout_core, FPE/SR/TLD)로 baseline_w6/window3/sticky_aug_w6 비교
   (soda 관측 패턴 그대로: 근접 직후 집중 dropout)

이미 만든 exp71_vis_cache.pt(에피소드별 vis feature)를 재사용해서 재인코딩 없이 진행.
"""
import importlib.util
import json
import random
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

spec = importlib.util.spec_from_file_location("exp71", ROOT / "scripts" / "train_exp71_stage2_transformer.py")
exp71 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp71)

CACHE_FILE = ROOT / "docs" / "v5" / "closed_loop_eval" / "exp71_vis_cache.pt"
TRUTHMINI = ROOT / "docs" / "v5" / "bbox_truth_mini.json"
STEP1_DATASET = ROOT / "docs" / "v5" / "bbox_nav_step1" / "bbox_dataset.json"
DATA_DIR = ROOT / "ROS_action" / "mobile_vla_dataset_v5"
OUT_FILE = ROOT / "docs" / "v5" / "closed_loop_eval" / "truthmini_holdout_and_cl.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FALLBACK = (0.5, 0.5, 0.05, 0.0)

sys.path.insert(0, str(ROOT / "scripts" / "sim"))
import rollout_core as rc


def reconstruct_stems():
    """cache 빌드 당시와 동일한 순서로 train_ann/val_ann stem 목록 복원 (필터링 동일하게)."""
    with open(exp71.ANN_PATH) as f:
        ann = json.load(f)
    random.seed(42)
    random.shuffle(ann)
    n_val = max(1, int(len(ann) * 0.15))
    val_ann, train_ann = ann[:n_val], ann[n_val:]

    def filt(lst):
        stems = []
        for ep in lst:
            h5p = Path(ep["episode"])
            frames = [fr for fr in ep["frames"] if fr.get("gt_class") is not None]
            if h5p.exists() and frames:
                stems.append(h5p.stem)
        return stems

    return filt(train_ann), filt(val_ann)


def build_windows(episodes, window):
    X, y = [], []
    for ep in episodes:
        bboxes, vis, gts = ep["bboxes"], ep["vis"], ep["gts"]
        n = len(gts)
        for t in range(n):
            seq = []
            for k in range(window):
                idx = max(0, t - (window - 1 - k))
                seq.append(list(bboxes[idx]) + vis[idx].tolist())
            X.append(seq)
            y.append(gts[t])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def apply_aug(bboxes, p, sticky, rng):
    if p <= 0:
        return bboxes
    out, last_real = [], FALLBACK
    for b in bboxes:
        if b[3] > 0.5:
            last_real = b
        if rng.random() < p:
            out.append((last_real[0], last_real[1], last_real[2], 0.0) if sticky else FALLBACK)
        else:
            out.append(b)
    return out


def train_variant(name, train_eps, val_eps, window, aug_p=0.0, sticky=False, epochs=300, lr=5e-4, seed=42):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_eps_aug = train_eps
    if aug_p > 0:
        train_eps_aug = [{**ep, "bboxes": apply_aug(ep["bboxes"], aug_p, sticky, rng)} for ep in train_eps]
    X_tr, y_tr = build_windows(train_eps_aug, window)
    X_va, y_va = build_windows(val_eps, window)
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
                best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    print(f"[{name:14s}] val_acc={best_acc:.1%}  (train={len(train_eps)}ep val={len(val_eps)}ep)")
    return model, best_acc


def xyxy_to_cx_cy_area(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2, max(0.0, x2 - x1) * max(0.0, y2 - y1)


def part_a(train_eps_all, val_eps_all, train_stems, val_stems, truthmini_stems):
    """truth_mini 18ep를 학습풀에서 완전히 제외하고 재학습 -> 진짜 held-out 검증."""
    print("\n" + "=" * 60)
    print("A) truth_mini 완전 격리 재학습 + 진짜 held-out clean-bbox 검증")
    print("=" * 60)

    # stem -> cache entry 매핑
    stem_to_ep = dict(zip(train_stems, train_eps_all)) | dict(zip(val_stems, val_eps_all))
    remain_stems = [s for s in (train_stems + val_stems) if s not in truthmini_stems]
    print(f"truth_mini 제외 후 남은 에피소드: {len(remain_stems)} (원래 {len(train_stems)+len(val_stems)})")

    rng = np.random.default_rng(42)
    shuffled = remain_stems.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * 0.15))
    new_val_stems, new_train_stems = shuffled[:n_val], shuffled[n_val:]
    new_train_eps = [stem_to_ep[s] for s in new_train_stems]
    new_val_eps = [stem_to_ep[s] for s in new_val_stems]

    model_w6, acc_w6 = train_variant("baseline_w6(격리)", new_train_eps, new_val_eps, window=6)
    model_w3, acc_w3 = train_variant("window3(격리)", new_train_eps, new_val_eps, window=3)

    with open(exp71.ANN_PATH) as f:
        pg = json.load(f)
    pg_stems = {Path(ep["episode"]).stem: ep for ep in pg}

    tm = json.loads(TRUTHMINI.read_text())
    by_ep = {}
    for a in tm["annotations"]:
        by_ep.setdefault(a["episode"], []).append(a)

    results_a = {"baseline_w6_val_acc": acc_w6, "window3_val_acc": acc_w3}
    for window, model, tag in [(6, model_w6, "baseline_w6"), (3, model_w3, "window3")]:
        X, y = [], []
        for ep_stem, items in by_ep.items():
            pg_ep = pg_stems.get(ep_stem)
            if pg_ep is None or ep_stem not in stem_to_ep:
                continue
            cache_ep = stem_to_ep[ep_stem]
            orig_bbox = cache_ep["bboxes"]
            vis = cache_ep["vis"]
            for item in items:
                t = item["frame_idx"]
                if t >= len(orig_bbox):
                    continue
                cx, cy, area = xyxy_to_cx_cy_area(item["bbox_xyxy_norm"])
                clean_bbox = (cx, cy, area, 1.0)
                seq = []
                for k in range(window):
                    idx = max(0, t - (window - 1 - k))
                    bbox = clean_bbox if idx == t else orig_bbox[idx]
                    seq.append(list(bbox) + vis[idx].tolist())
                X.append(seq)
                y.append(item["gt_action_class"])
        if not X:
            print(f"[{tag}] truth_mini 매칭 샘플 없음 (전부 학습에 있었거나 stem 불일치)")
            continue
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        y_t = torch.tensor(y, dtype=torch.int64, device=DEVICE)
        with torch.no_grad():
            preds = model(X_t).argmax(dim=-1)
            acc = (preds == y_t).float().mean().item()
        results_a[f"{tag}_truthmini_holdout_acc"] = acc
        results_a[f"{tag}_truthmini_n"] = len(y)
        print(f"[{tag}] truth_mini 진짜 held-out acc={acc:.1%}  (n={len(y)})")

    return results_a, model_w6, model_w3


def correlated_flicker(bboxes, rng, area_near=0.15, p_base=0.1, p_near=0.85):
    out, was_near = [], False
    for b in bboxes:
        p = p_near if was_near else p_base
        out.append(FALLBACK if rng.random() < p else b)
        was_near = (b[3] > 0.5 and b[2] > area_near)
    return out


def part_b(models_by_window, stem_to_ep):
    """실제 궤적(bbox_nav_step1, 45ep 전부 exp71 vis cache와 stem 일치 확인됨)에 상관형
    flicker 주입 -> 모델 예측 액션으로 궤적 계산 -> rollout_core로 FPE/SR/TLD
    (gt_class 시퀀스로 만든 expert 궤적과 비교)."""
    print("\n" + "=" * 60)
    print("B) 실제 궤적 리플레이 + 상관형 flicker -> FPE/SR/TLD")
    print("=" * 60)

    step1 = json.loads(STEP1_DATASET.read_text())
    step1 = [ep for ep in step1 if ep["episode"] in stem_to_ep]
    print(f"bbox_nav_step1 에피소드: {len(step1)}개 (vis cache와 stem 일치)")

    results_b = {}
    for window, model in models_by_window.items():
        n_trials = 3
        fpe_all, sr_all, tld_all = [], [], []
        for trial in range(n_trials):
            rng = np.random.default_rng(300 + trial)
            for ep in step1:
                stem = ep["episode"]
                cache_ep = stem_to_ep[stem]
                vis = cache_ep["vis"]
                gt_seq = [fr["gt_class"] for fr in ep["frames"]]
                orig_bboxes = cache_ep["bboxes"]
                if len(orig_bboxes) != len(gt_seq):
                    continue  # step1과 pg448 프레임 필터링 차이로 길이 안 맞으면 스킵
                flick_bboxes = correlated_flicker(orig_bboxes, rng)

                preds = []
                for t in range(len(gt_seq)):
                    seq = []
                    for k in range(window):
                        idx = max(0, t - (window - 1 - k))
                        seq.append(list(flick_bboxes[idx]) + vis[idx].tolist())
                    x = torch.tensor([seq], dtype=torch.float32, device=DEVICE)
                    with torch.no_grad():
                        preds.append(model(x).argmax(dim=-1).item())

                expert_traj = rc.build_trajectory(gt_seq)
                pred_traj = rc.build_trajectory(preds)
                m = rc.compute_metrics(expert_traj, pred_traj, success_fpe=0.5)
                fpe_all.append(m["fpe"]); sr_all.append(m["success"]); tld_all.append(m["tld"])

        results_b[f"window{window}"] = {
            "sr": float(np.mean(sr_all)),
            "fpe_mean": float(np.mean(fpe_all)),
            "tld_mean": float(np.mean(tld_all)),
            "n": len(fpe_all),
        }
        print(f"[window={window}] SR={np.mean(sr_all):.1%}  FPE={np.mean(fpe_all):.3f}m  "
              f"TLD={np.mean(tld_all):.2f}  (n={len(fpe_all)}, {n_trials}trial x {len(step1)}ep)")
    return results_b


def main():
    cache = torch.load(CACHE_FILE, weights_only=False)
    train_eps_all, val_eps_all = cache["train"], cache["val"]
    train_stems, val_stems = reconstruct_stems()
    assert len(train_stems) == len(train_eps_all) and len(val_stems) == len(val_eps_all), \
        "stem 복원과 cache 순서가 안 맞음 -- 필터링 로직 재확인 필요"

    tm = json.loads(TRUTHMINI.read_text())
    truthmini_stems = set(a["episode"] for a in tm["annotations"])

    results = {}
    results["A"], model_w6, model_w3 = part_a(train_eps_all, val_eps_all, train_stems, val_stems, truthmini_stems)

    print("\n[주의] Part B는 45개 step1 에피소드 전부 사용 — Part A 재학습 시 이 중 일부가 학습셋에")
    print("포함됐을 수 있음(순수 held-out 아님). 정확도 절대값보다 baseline_w6 vs window3의")
    print("*상대 비교*(같은 조건에서 flicker에 더 강건한지)로만 해석할 것.")
    stem_to_ep_all = dict(zip(train_stems, train_eps_all)) | dict(zip(val_stems, val_eps_all))
    results["B"] = part_b({6: model_w6, 3: model_w3}, stem_to_ep_all)

    OUT_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nsaved -> {OUT_FILE}")


if __name__ == "__main__":
    main()

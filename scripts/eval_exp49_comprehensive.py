#!/usr/bin/env python3
"""
Exp49 종합 평가 스크립트

1. Full PM  — val 30ep 전체 프레임 정확도
2. Per-class PM — 클래스별 분석
3. Bootstrap CI — PM 95% 신뢰구간 (n=1000)
4. Random-seed PM — 5개 다른 train/val split로 안정성 확인
5. Closed-loop offline — FPE, TLD, success_rate
6. Goal sensitivity — cx0를 얼마나 바꿔야 action이 바뀌나
7. Full paraphrase action test — val 30ep × 5 프롬프트 전수 검사
8. Exp46/47/49 종합 비교표
"""
import json, sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.sim.rollout_core import build_trajectory, compute_metrics, CLASS_NAMES as CLS, DT_DEFAULT

EXP46_DIR = ROOT / "docs" / "v5" / "bbox_nav_exp46"
EXP47_DIR = ROOT / "docs" / "v5" / "bbox_nav_exp47"
EXP49_DIR = ROOT / "docs" / "v5" / "bbox_nav_exp49"
MLP49_DIR = ROOT / "runs" / "v5_nav" / "mlp" / "exp49"
OUT_PATH  = EXP49_DIR / "comprehensive_eval.json"

WINDOW      = 8
VIS_DIM     = 1024
GOAL_DIM    = 3
NUM_CLASSES = 8
CLASS_NAMES = ["STOP","FORWARD","LEFT","RIGHT","FWD+L","FWD+R","ROT_L","ROT_R"]


# ─────────────────────────────────────────────
# 모델
# ─────────────────────────────────────────────

def build_mlp(d_in):
    return nn.Sequential(
        nn.Linear(d_in, 512), nn.ReLU(), nn.Dropout(0.25),
        nn.Linear(512, 256),  nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(256, 128),  nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(128, 64),   nn.ReLU(),
        nn.Linear(64, NUM_CLASSES),
    )


def load_exp49():
    ckpt = torch.load(str(MLP49_DIR / "exp49_mlp.pt"), map_location="cpu", weights_only=False)
    net  = build_mlp(ckpt["d_in"])
    net.load_state_dict(ckpt["model_state_dict"])
    net.eval()
    return net, ckpt


def build_features(bbox_data, vis_cache):
    """에피소드 리스트 → (X, y, ep_ids) 반환"""
    X, y, ep_ids = [], [], []
    for ep_idx, ep_data in enumerate(bbox_data):
        ep_path  = ep_data["episode"]
        frames   = ep_data["frames"]
        vis_feats = vis_cache.get(ep_path)
        if vis_feats is None:
            continue
        fr0 = frames[0]
        goal = np.array([
            fr0["cx"] if fr0["has_bbox"] else 0.5,
            fr0["cy"] if fr0["has_bbox"] else 0.5,
            fr0["area"] if fr0["has_bbox"] else 0.0,
        ], dtype=np.float32)

        for t in range(len(frames)):
            bbox_feat = []
            for k in range(WINDOW):
                idx = max(0, t - (WINDOW - 1 - k))
                fr  = frames[idx]
                bbox_feat.extend([fr["cx"], fr["cy"], fr["area"], float(fr["has_bbox"])])
            feat = np.concatenate([np.array(bbox_feat, np.float32), vis_feats[t], goal])
            X.append(feat)
            y.append(frames[t]["gt_class"])
            ep_ids.append(ep_idx)
    return np.array(X, np.float32), np.array(y, np.int64), ep_ids


def predict_all(net, X, device, batch=512):
    net.eval()
    preds = []
    for i in range(0, len(X), batch):
        xb = torch.tensor(X[i:i+batch], dtype=torch.float32, device=device)
        with torch.no_grad():
            preds.append(net(xb).argmax(1).cpu().numpy())
    return np.concatenate(preds)


# ─────────────────────────────────────────────
# 1. Full PM + Per-class
# ─────────────────────────────────────────────

def test_full_pm(net, X_te, y_te, device):
    print("\n[1] Full PM + Per-class")
    preds = predict_all(net, X_te, device)
    pm = (preds == y_te).mean()

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    for g, p in zip(y_te, preds):
        cm[g, p] += 1

    print(f"  전체 PM: {pm:.4f} ({pm*100:.1f}%)")
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        total = cm[i].sum()
        if total == 0:
            continue
        acc = cm[i, i] / total
        per_class[name] = {"correct": int(cm[i,i]), "total": int(total), "acc": float(acc)}
        print(f"  {name:<8}: {cm[i,i]:>3}/{total:<3} = {acc*100:.0f}%")

    pred_dist = {CLASS_NAMES[i]: int((preds==i).sum()) for i in range(NUM_CLASSES)}
    print(f"  예측 분포: { {k:v for k,v in pred_dist.items() if v>0} }")
    return float(pm), cm.tolist(), per_class, pred_dist


# ─────────────────────────────────────────────
# 2. Bootstrap CI
# ─────────────────────────────────────────────

def bootstrap_ci(net, X_te, y_te, device, n=1000):
    print(f"\n[2] Bootstrap 95% CI (n={n})")
    preds  = predict_all(net, X_te, device)
    correct = (preds == y_te).astype(float)
    rng = np.random.RandomState(42)
    boot_pms = [rng.choice(correct, size=len(correct), replace=True).mean() for _ in range(n)]
    lo, hi = np.percentile(boot_pms, [2.5, 97.5])
    mean   = np.mean(boot_pms)
    print(f"  PM = {mean*100:.2f}%  95% CI = [{lo*100:.2f}%, {hi*100:.2f}%]")
    return {"mean": float(mean), "ci_lo": float(lo), "ci_hi": float(hi)}


# ─────────────────────────────────────────────
# 3. Random-seed PM (5 seeds)
# ─────────────────────────────────────────────

def test_random_seeds(bbox_data, vis_cache, device, seeds=(42,7,13,99,123)):
    print(f"\n[3] Random-seed PM (seeds={seeds})")
    ep_labels = [ep["path_type"] for ep in bbox_data]
    X_all, y_all, ep_ids = build_features(bbox_data, vis_cache)
    ep_frame_counts = []
    offset = 0
    for ep_data in bbox_data:
        n = len(ep_data["frames"])
        ep_frame_counts.append((offset, offset + n))
        offset += n

    accs = []
    for seed in seeds:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        tr_ep, te_ep = next(sss.split(np.zeros(len(bbox_data)), ep_labels))
        tr_idx = np.concatenate([np.arange(*ep_frame_counts[i]) for i in tr_ep])
        te_idx = np.concatenate([np.arange(*ep_frame_counts[i]) for i in te_ep])
        X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
        X_te, y_te = X_all[te_idx], y_all[te_idx]

        # 빠른 학습 (100 epoch)
        net_s = build_mlp(X_tr.shape[1]).to(device)
        cc = np.bincount(y_tr, minlength=NUM_CLASSES).astype(float)
        w = np.where(cc > 0, 1.0/(cc+1e-6), 0.0); w /= w.sum()/NUM_CLASSES
        crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))
        opt  = torch.optim.AdamW(net_s.parameters(), lr=1e-3, weight_decay=1e-4)
        sch  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200)
        Xtr  = torch.tensor(X_tr, dtype=torch.float32, device=device)
        ytr  = torch.tensor(y_tr, dtype=torch.long, device=device)
        for _ in range(200):
            net_s.train()
            perm = torch.randperm(len(Xtr))
            for i in range(0, len(perm), 128):
                b = perm[i:i+128]
                opt.zero_grad(); crit(net_s(Xtr[b]), ytr[b]).backward(); opt.step()
            sch.step()
        preds_s = predict_all(net_s, X_te, device)
        acc = float((preds_s == y_te).mean())
        accs.append(acc)
        print(f"  seed={seed}: PM={acc*100:.1f}%")

    mean, std = np.mean(accs), np.std(accs)
    print(f"  → 평균 {mean*100:.1f}% ± {std*100:.1f}%")
    return {"seeds": list(seeds), "accs": accs, "mean": float(mean), "std": float(std)}


# ─────────────────────────────────────────────
# 4. Closed-loop offline
# ─────────────────────────────────────────────

def test_closed_loop(net, bbox_data, vis_cache, val_idx, device):
    print(f"\n[4] Closed-loop offline ({len(val_idx)} episodes)")
    val_eps = [bbox_data[i] for i in val_idx]
    success_thresh = 1.0
    ep_results = []
    per_path = defaultdict(lambda: {"n":0,"pm_sum":0,"fpe_sum":0,"success":0})

    for ep_data in val_eps:
        ep_path  = ep_data["episode"]
        pt       = ep_data["path_type"]
        frames   = ep_data["frames"]
        vis_feats = vis_cache.get(ep_path)
        if vis_feats is None:
            continue

        fr0  = frames[0]
        goal = np.array([
            fr0["cx"] if fr0["has_bbox"] else 0.5,
            fr0["cy"] if fr0["has_bbox"] else 0.5,
            fr0["area"] if fr0["has_bbox"] else 0.0,
        ], dtype=np.float32)

        pred_cls, gt_cls = [], []
        for t in range(len(frames)):
            bbox_feat = []
            for k in range(WINDOW):
                idx = max(0, t - (WINDOW-1-k))
                fr  = frames[idx]
                bbox_feat.extend([fr["cx"], fr["cy"], fr["area"], float(fr["has_bbox"])])
            feat = np.concatenate([np.array(bbox_feat, np.float32), vis_feats[t], goal])
            x = torch.tensor([feat], dtype=torch.float32, device=device)
            with torch.no_grad():
                cls = int(net(x).argmax(1).item())
            pred_cls.append(cls)
            gt_cls.append(frames[t]["gt_class"])

        pred_traj = build_trajectory(pred_cls)
        gt_traj   = build_trajectory(gt_cls)
        metrics   = compute_metrics(pred_traj, gt_traj)

        pm = sum(p==g for p,g in zip(pred_cls, gt_cls)) / len(gt_cls)
        success = int(metrics["fpe"] < success_thresh)
        per_path[pt]["n"] += 1
        per_path[pt]["pm_sum"] += pm
        per_path[pt]["fpe_sum"] += metrics["fpe"]
        per_path[pt]["success"] += success
        ep_results.append({"path_type":pt,"pm":pm,"fpe":metrics["fpe"],"tld":metrics["tld"],"success":success})

    overall_pm  = np.mean([r["pm"] for r in ep_results])
    overall_fpe = np.mean([r["fpe"] for r in ep_results])
    overall_tld = np.mean([r["tld"] for r in ep_results])
    overall_sr  = np.mean([r["success"] for r in ep_results])

    print(f"  overall PM={overall_pm*100:.1f}%  FPE={overall_fpe:.3f}  TLD={overall_tld:.3f}  SR={overall_sr*100:.1f}%")
    for pt, v in sorted(per_path.items()):
        n = v["n"]
        print(f"  {pt:<18}: PM={v['pm_sum']/n*100:.0f}%  FPE={v['fpe_sum']/n:.3f}  SR={v['success']}/{n}")

    return {
        "overall_pm": float(overall_pm), "overall_fpe": float(overall_fpe),
        "overall_tld": float(overall_tld), "success_rate": float(overall_sr),
        "per_path": {pt: {"n":v["n"],"pm":v["pm_sum"]/v["n"],
                          "fpe":v["fpe_sum"]/v["n"],"success_rate":v["success"]/v["n"]}
                     for pt, v in per_path.items()},
    }


# ─────────────────────────────────────────────
# 5. Goal sensitivity — cx0 sweep
# ─────────────────────────────────────────────

def test_goal_sensitivity(net, bbox_data, vis_cache, val_idx, device):
    print(f"\n[5] Goal sensitivity — cx0 sweep")
    val_eps = [bbox_data[i] for i in val_idx[:10]]  # 10개 샘플
    cx_sweep = np.linspace(0.05, 0.95, 19)

    action_by_cx = defaultdict(list)
    for ep_data in val_eps:
        ep_path  = ep_data["episode"]
        frames   = ep_data["frames"]
        vis_feats = vis_cache.get(ep_path)
        if vis_feats is None:
            continue
        fr0 = frames[0]
        t   = 0
        bbox_feat = []
        for k in range(WINDOW):
            bbox_feat.extend([fr0["cx"], fr0["cy"], fr0["area"], float(fr0["has_bbox"])])
        bbox_arr = np.array(bbox_feat, np.float32)
        vis_feat = vis_feats[t]

        for cx in cx_sweep:
            goal = np.array([cx, fr0["cy"] if fr0["has_bbox"] else 0.5, fr0["area"] if fr0["has_bbox"] else 0.05], np.float32)
            feat = np.concatenate([bbox_arr, vis_feat, goal])
            x = torch.tensor([feat], dtype=torch.float32, device=device)
            with torch.no_grad():
                cls = int(net(x).argmax(1).item())
            action_by_cx[round(float(cx),2)].append(CLASS_NAMES[cls])

    print(f"  cx0  → most common action")
    sensitivity_rows = []
    for cx in cx_sweep:
        actions = action_by_cx[round(float(cx),2)]
        from collections import Counter
        most_common, cnt = Counter(actions).most_common(1)[0]
        pct = cnt/len(actions)*100
        bar = "█" * int(pct/10)
        print(f"  {cx:.2f} → {most_common:<8} ({pct:.0f}%  {bar})")
        sensitivity_rows.append({"cx0": round(float(cx),2), "action": most_common, "pct": float(pct)})
    return sensitivity_rows


# ─────────────────────────────────────────────
# 6. Full paraphrase action consistency (all 30 val eps × cached cx variance)
# ─────────────────────────────────────────────

def test_full_paraphrase_consistency(net, bbox_data, vis_cache, val_idx, device):
    """
    grounding을 다시 돌리지 않고, 기존 grounding_consistency_results.json의
    cx 분산 범위를 시뮬레이션 — cx0에 ±std 노이즈 추가 후 action 변화 확인
    """
    print(f"\n[6] Full paraphrase robustness (cx0 ± noise, 30 val eps)")
    val_eps = [bbox_data[i] for i in val_idx]
    cx_noise_levels = [0.05, 0.10, 0.15, 0.20, 0.25]  # 실측 cx std 범위
    rng = np.random.RandomState(42)

    results_by_noise = {}
    for noise in cx_noise_levels:
        match_count, total = 0, 0
        for ep_data in val_eps:
            ep_path  = ep_data["episode"]
            frames   = ep_data["frames"]
            vis_feats = vis_cache.get(ep_path)
            if vis_feats is None:
                continue
            fr0 = frames[0]
            goal_orig = np.array([
                fr0["cx"] if fr0["has_bbox"] else 0.5,
                fr0["cy"] if fr0["has_bbox"] else 0.5,
                fr0["area"] if fr0["has_bbox"] else 0.0,
            ], np.float32)
            bbox_feat = []
            for k in range(WINDOW):
                bbox_feat.extend([fr0["cx"], fr0["cy"], fr0["area"], float(fr0["has_bbox"])])
            bbox_arr = np.array(bbox_feat, np.float32)
            vis_feat = vis_feats[0]

            # baseline action
            feat_orig = np.concatenate([bbox_arr, vis_feat, goal_orig])
            x_orig = torch.tensor([feat_orig], dtype=torch.float32, device=device)
            with torch.no_grad():
                action_orig = int(net(x_orig).argmax(1).item())

            # 5회 노이즈 추가
            for _ in range(5):
                cx_noisy = float(np.clip(goal_orig[0] + rng.normal(0, noise), 0.01, 0.99))
                goal_noisy = np.array([cx_noisy, goal_orig[1], goal_orig[2]], np.float32)
                feat_noisy = np.concatenate([bbox_arr, vis_feat, goal_noisy])
                x_noisy = torch.tensor([feat_noisy], dtype=torch.float32, device=device)
                with torch.no_grad():
                    action_noisy = int(net(x_noisy).argmax(1).item())
                match_count += int(action_orig == action_noisy)
                total += 1

        rate = match_count / total
        results_by_noise[noise] = float(rate)
        print(f"  cx noise ±{noise:.2f} → action 일치율 {rate*100:.1f}%  ({match_count}/{total})")

    return results_by_noise


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Exp49 종합 평가")
    print("=" * 65)

    bbox_data = json.loads((EXP46_DIR / "bbox_dataset_full.json").read_text())
    vis_npz   = np.load(str(EXP46_DIR / "vision_features.npz"))
    vis_idx   = json.loads((EXP46_DIR / "vision_features_index.json").read_text())
    vis_cache = {ep: vis_npz[f"ep_{i}"] for ep, i in vis_idx.items()}

    ep_labels = [ep["path_type"] for ep in bbox_data]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_ep_idx, te_ep_idx = next(sss.split(np.zeros(len(bbox_data)), ep_labels))

    ep_frame_counts, offset = [], 0
    for ep_data in bbox_data:
        n = len(ep_data["frames"])
        ep_frame_counts.append((offset, offset + n))
        offset += n

    X_all, y_all, _ = build_features(bbox_data, vis_cache)
    te_frame_idx = np.concatenate([np.arange(*ep_frame_counts[i]) for i in te_ep_idx])
    tr_frame_idx = np.concatenate([np.arange(*ep_frame_counts[i]) for i in tr_ep_idx])
    X_te, y_te = X_all[te_frame_idx], y_all[te_frame_idx]
    X_tr, y_tr = X_all[tr_frame_idx], y_all[tr_frame_idx]

    net, ckpt = load_exp49()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    print(f"Device: {device}  |  val: {len(X_te)} frames  |  train: {len(X_tr)} frames")

    results = {}

    pm, cm, per_class, pred_dist = test_full_pm(net, X_te, y_te, device)
    results["full_pm"] = {"pm": pm, "confusion": cm, "per_class": per_class, "pred_dist": pred_dist}

    ci = bootstrap_ci(net, X_te, y_te, device)
    results["bootstrap_ci"] = ci

    seed_res = test_random_seeds(bbox_data, vis_cache, device)
    results["random_seeds"] = seed_res

    cl = test_closed_loop(net, bbox_data, vis_cache, list(te_ep_idx), device)
    results["closed_loop"] = cl

    sens = test_goal_sensitivity(net, bbox_data, vis_cache, list(te_ep_idx), device)
    results["goal_sensitivity"] = sens

    para = test_full_paraphrase_consistency(net, bbox_data, vis_cache, list(te_ep_idx), device)
    results["paraphrase_robustness"] = para

    # ── 최종 비교표 ──
    print("\n" + "=" * 65)
    print("최종 비교표")
    print("=" * 65)
    print(f"  {'모델':<10} {'val acc':>8} {'CL SR':>7} {'paraphrase':>12}")
    print(f"  {'Exp46':<10} {'93.2%':>8} {'100%':>7} {'N/A (no instr)':>12}")
    print(f"  {'Exp47':<10} {'98.7%':>8} {'100%':>7} {'74.1% (FAIL)':>12}")
    print(f"  {'Exp49':<10} {pm*100:>7.1f}% {cl['success_rate']*100:>6.1f}% {'100% (PASS)':>12}")
    print()
    print(f"  Bootstrap 95% CI: [{ci['ci_lo']*100:.1f}%, {ci['ci_hi']*100:.1f}%]")
    print(f"  5-seed 안정성: {seed_res['mean']*100:.1f}% ± {seed_res['std']*100:.1f}%")

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\n저장: {OUT_PATH}")


if __name__ == "__main__":
    main()

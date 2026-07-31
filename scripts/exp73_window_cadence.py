#!/usr/bin/env python3
"""
학습 window의 실시간 폭 vs 실제 서빙 cadence 불일치 검증 (soda 실측, 2026-07-23).

soda 발견: H5 수집 ~6.0Hz 연속 vs 실추론 ~1.3Hz 버스트(grounding_skip_n=3마다
PG2/OWL 그라운딩 2.2s 소요) → window=6이 학습 시 ~1.0초 폭인데 실기에선 ~4.6초 폭.
즉 학습 때 "최근 1초"로 배운 시퀀스 패턴을, 실기에선 "최근 4.6초"에 해당하는
훨씬 느리게 변한 장면에 적용하는 셈 — 학습/서빙 분포 불일치.

가설: window 프레임 간격(stride)을 실제 서빙 cadence에 맞게 늘려 학습하면(=논리적
"1초"가 아니라 "4.6초"짜리 window로 학습) 실기 조건에 더 맞는 모델이 되어
closed-loop(오프라인 재생이지만 stride 늘린 채 평가)에서 성능이 달라지는가.

stride=1 : 기존(연속 6Hz 그대로, ~1.0초 폭) — baseline
stride=5 : soda 실측 비율(~4.6배)에 맞춰 window 6프레임이 ~4.6초 폭이 되도록
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.train_exp73_trackA_heads import (
    MLPActionHead, CACHE_V6, SPLIT_SEED, VAL_RATIO, NUM_CLASSES, DEVICE, BBOX_SCALE, WINDOW,
)
from scripts.sim.rollout_core import build_trajectory, compute_metrics


def build_windows_strided(eps, window=WINDOW, bbox_scale=BBOX_SCALE, stride=1):
    """build_windows와 동일하나 window 슬롯 간 간격을 stride배로 늘림
    (stride=1이면 원본 build_windows와 100% 동일)."""
    X, y = [], []
    for ep in eps:
        bboxes, vis, gts = ep["bboxes"], ep["vis"], ep["gts"]
        for t in range(len(gts)):
            seq = []
            for k in range(window):
                idx = max(0, t - (window - 1 - k) * stride)
                seq.append([v * bbox_scale for v in bboxes[idx]] + vis[idx].tolist())
            X.append(seq); y.append(gts[t])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def train_one(X_tr, y_tr, X_va, y_va, seed, epochs=300, lr=5e-4):
    torch.manual_seed(seed); np.random.seed(seed)
    cls_counts = np.bincount(y_tr, minlength=NUM_CLASSES).astype(np.float32)
    cls_counts = np.where(cls_counts == 0, 1.0, cls_counts)
    weights = 1.0 / cls_counts; weights = weights / weights.sum() * NUM_CLASSES
    weights_t = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    X_tr_t = torch.tensor(X_tr, device=DEVICE); y_tr_t = torch.tensor(y_tr, device=DEVICE)
    X_va_t = torch.tensor(X_va, device=DEVICE); y_va_t = torch.tensor(y_va, device=DEVICE)
    model = MLPActionHead().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best_acc, best_state = 0.0, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr_t), device=DEVICE)
        for i in range(0, len(perm), 128):
            b = perm[i:i + 128]
            loss = F.cross_entropy(model(X_tr_t[b]), y_tr_t[b], weight=weights_t)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        if ep % 25 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                acc = (model(X_va_t).argmax(1) == y_va_t).float().mean().item()
            if acc >= best_acc: best_acc = acc; best_state = {k: v.clone() for k, v in model.state_dict().items()}
    return best_acc, best_state


@torch.no_grad()
def eval_closed_loop_held(eps, model, stride, window=WINDOW, bbox_scale=BBOX_SCALE):
    """soda 확인: 서빙은 stride마다만 새 결정을 내고, 그 결정을 다음 결정까지
    그대로 유지(hold)한 채 재발행(무보간, 속도는 방향만 쓰고 크기는 상수).
    즉 입력 cadence뿐 아니라 '결정 자체가 stride마다만 갱신'까지 재현."""
    results = []
    for ep in eps:
        bboxes, vis, gts = ep["bboxes"], ep["vis"], ep["gts"]
        n = len(gts)
        X = []
        for t in range(0, n, stride):  # stride 지점에서만 결정
            seq = []
            for k in range(window):
                idx = max(0, t - (window - 1 - k) * stride)
                seq.append([v * bbox_scale for v in bboxes[idx]] + vis[idx].tolist())
            X.append(seq)
        Xt = torch.tensor(np.asarray(X, dtype=np.float32))
        dec = model(Xt).argmax(1).numpy()  # stride 지점마다 1개 결정
        pred = np.repeat(dec, stride)[:n]  # 다음 결정까지 hold(그대로 유지)
        gt = np.asarray(gts, dtype=np.int64)
        et = build_trajectory(gt.tolist()); pt = build_trajectory(pred.tolist())
        m = compute_metrics(et, pt); m["val_acc"] = float((pred == gt).mean())
        results.append(m)
    return results


@torch.no_grad()
def eval_closed_loop_strided(eps, model, stride, window=WINDOW, bbox_scale=BBOX_SCALE):
    results = []
    for ep in eps:
        bboxes, vis, gts = ep["bboxes"], ep["vis"], ep["gts"]
        X = []
        for t in range(len(gts)):
            seq = []
            for k in range(window):
                idx = max(0, t - (window - 1 - k) * stride)
                seq.append([v * bbox_scale for v in bboxes[idx]] + vis[idx].tolist())
            X.append(seq)
        Xt = torch.tensor(np.asarray(X, dtype=np.float32))
        pred = model(Xt).argmax(1).numpy()
        gt = np.asarray(gts, dtype=np.int64)
        et = build_trajectory(gt.tolist()); pt = build_trajectory(pred.tolist())
        m = compute_metrics(et, pt); m["val_acc"] = float((pred == gt).mean())
        results.append(m)
    return results


OWL_ANN = ROOT / "docs/v5/bbox_nav_owl/bbox_dataset_v6_owl.json"


def swap_bboxes(eps, ann_path):
    """vis(그라운더 무관)는 그대로 두고 bbox만 지정 그라운더 주석으로 교체."""
    import json
    with open(ann_path) as f:
        alt = json.load(f)
    alt_by_stem = {Path(e["episode"]).stem: e for e in alt}
    out = []
    for ep in eps:
        src = alt_by_stem.get(ep["stem"])
        if src is None:
            continue
        frames = [fr for fr in src["frames"] if fr.get("gt_class") is not None]
        if len(frames) != len(ep["vis"]):
            continue
        ep = dict(ep)
        ep["bboxes"] = [(fr.get("cx_det", 0.5), fr.get("cy_det", 0.5), fr.get("area_det", 0.05),
                         float(fr.get("has_bbox", False))) for fr in frames]
        out.append(ep)
    return out


def main():
    base = torch.load(str(CACHE_V6), weights_only=False)
    base = [e for e in base if e.get("acts") is not None]

    for grounder, eps in [("PG2-448", base), ("OWL-v2", swap_bboxes(base, OWL_ANN))]:
        print(f"\n########## 그라운더={grounder} ({len(eps)}ep) ##########", flush=True)
        run_grounder(eps, grounder)


def run_grounder(eps, grounder_name):
    rng = np.random.default_rng(SPLIT_SEED); idx = list(range(len(eps))); rng.shuffle(idx)
    nv = max(1, int(len(idx) * VAL_RATIO))
    val = [eps[i] for i in idx[:nv]]; tr = [eps[i] for i in idx[nv:]]

    for stride, label in [(1, "baseline(1.0s, 학습때와 동일)"), (5, "실서빙cadence(~4.6s 근사)")]:
        Xtr, ytr = build_windows_strided(tr, stride=stride)
        Xva, yva = build_windows_strided(val, stride=stride)
        accs, succs, succs_held = [], [], []
        for seed in [0, 1, 2]:
            acc, st = train_one(Xtr, ytr, Xva, yva, seed, epochs=300)
            model = MLPActionHead(); model.load_state_dict(st); model.eval()
            res = eval_closed_loop_strided(val, model, stride=stride)
            succ = np.mean([r["success"] for r in res]) * 100
            accs.append(acc * 100); succs.append(succ)
            if stride > 1:
                res_h = eval_closed_loop_held(val, model, stride=stride)
                succ_h = np.mean([r["success"] for r in res_h]) * 100
                succs_held.append(succ_h)
                print(f"[{grounder_name}/stride={stride} {label}] seed{seed} HELD(결정도 stride마다만)={succ_h:.1f}%", flush=True)
            print(f"[{grounder_name}/stride={stride} {label}] seed{seed} offline={acc*100:.1f}% closed-loop={succ:.1f}%", flush=True)
        print(f"=== {grounder_name}/stride={stride}({label}): offline {np.mean(accs):.1f}±{np.std(accs):.1f}%  "
              f"closed-loop {np.mean(succs):.1f}±{np.std(succs):.1f}% ===", flush=True)
        if succs_held:
            print(f"=== {grounder_name}/stride={stride} HELD(실서빙 완전재현: 입력+결정 모두 stride마다): "
                  f"closed-loop {np.mean(succs_held):.1f}±{np.std(succs_held):.1f}% ===\n", flush=True)


if __name__ == "__main__":
    main()

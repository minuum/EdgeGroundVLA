#!/usr/bin/env python3
"""EGVLA2 S1: 본학습 세트(egvla2_v3_main)에서 MLP 회귀 vs FM(A/B2/C) 재시험. inner-val 조기 종료, 5-fold(위치×경로 stratified), 다중 seed.

plan: docs/plans/plan_20261015_egvla2_300ep_pipeline.md §2~4
환경변수: SEEDS(기본 0,1,2,3,4) EPOCHS(100) AZMASK(1이면 고립 az 스파이크 마스크 타깃 사용) TAG SAVE(1이면 전체 데이터로 A/C 체크포인트 저장)
실행: .venv_lerobot/bin/python3 scripts/train_egvla2_s1.py
출력: docs/v5/egvla2_fm/summary_s1{TAG}.json
"""
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
os.environ.setdefault("SET", "pilot_v2_gauge_20261013")  # T 모듈 import용(사용 안 함)
import train_egvla2_fm_head as T  # noqa: E402

TAG = os.environ.get("TAG", "")
AZMASK = os.environ.get("AZMASK", "0") == "1"
EPOCHS = int(os.environ.get("EPOCHS", 100))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2,3,4").split(",")]
EVAL_EVERY, BS = 5, 256
ARCHS = {k: T.ARCH[k] for k in ("A_mlp_fm", "B2_film_res_fm", "C_pi0_adaln")}
DEV = T.DEV
GO_M = 0.2


def mask_isolated_az(a):
    """이동 없이 az만 단독으로 눌린 고립 프레임(앞뒤 프레임 az=0)을 0으로. 원본 actions_exec는 건드리지 않는다."""
    a = a.copy()
    for t in range(len(a)):
        if abs(a[t, 2]) > 0.1 and np.hypot(a[t, 0], a[t, 1]) < 1e-6:
            prev = abs(a[t - 1, 2]) if t > 0 else 0
            nxt = abs(a[t + 1, 2]) if t + 1 < len(a) else 0
            if prev <= 0.1 and nxt <= 0.1:
                a[t, 2] = 0
    return a


def load_main():
    meta = json.loads((ROOT / "docs/v5/egvla2_fm/meta/egvla2_v3_main.json").read_text())
    cache = np.load(T.CACHE)
    eps = []
    for d in meta["files"]:
        name = d["name"]
        with h5py.File(T.DATA / name, "r") as f:
            a = np.array(f["actions_exec"], dtype=np.float32)
        if AZMASK:
            a = mask_isolated_az(a)
        if f"{name}|bbox" not in cache.files:
            raise SystemExit(f"캐시에 없음: {name} (precompute 먼저)")
        bb, ft = cache[f"{name}|bbox"].astype(np.float32), cache[f"{name}|feat"].astype(np.float32)
        n = len(a)
        assert len(bb) == n and len(ft) == n, name
        win = np.stack([np.concatenate([bb[max(0, t - T.WINDOW + 1 + k)] for k in range(T.WINDOW)]) for t in range(n)])
        chunk = np.zeros((n, T.H, T.ADIM), np.float32)
        for t in range(n):
            seg = a[t:t + T.H]
            chunk[t, :len(seg)] = seg
        eps.append(dict(name=name, cell=f"{d['position']}|{d['path']}", bbox=win, feat=ft, chunk=chunk))
    return eps


def make_folds(eps, k=5):
    cells = {}
    for i, e in enumerate(eps):
        cells.setdefault(e["cell"], []).append(i)
    folds = [[] for _ in range(k)]
    j = 0
    for c in sorted(cells):
        for i in cells[c]:
            folds[j % k].append(i)
            j += 1
    return folds


def direction_metrics(m, te):
    """첫 스텝 평균 액션의 방향 오차(°, 둘 다 이동 중일 때)와 GO/STOP F1."""
    S = T.sample(m, te[0], te[1]).mean(1)[:, 0, :2]
    y = te[2][:, 0, :2]
    pm, ym = S.norm(dim=-1), y.norm(dim=-1)
    both = (pm > GO_M) & (ym > GO_M)
    cos = (S[both] * y[both]).sum(-1) / (pm[both] * ym[both])
    ang = torch.rad2deg(torch.acos(cos.clamp(-1, 1))).mean().item() if both.any() else float("nan")
    pg, yg = pm > GO_M, ym > GO_M
    tp = (pg & yg).sum().item(); fp = (pg & ~yg).sum().item(); fn = (~pg & yg).sum().item()
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    return dict(angle_deg=ang, go_f1=f1, mag_mae=float((pm[both] - ym[both]).abs().mean().item()) if both.any() else float("nan"))


def train_inner(cls, tr, va, seed):
    """inner-val flow loss가 최저인 epoch 가중치를 반환."""
    torch.manual_seed(seed)
    m = cls(tr[0].shape[1], tr[1].shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    n, best, best_state, best_ep = len(tr[0]), 1e9, None, 0
    for ep in range(1, EPOCHS + 1):
        m.train()
        perm = torch.randperm(n, device=DEV)
        for s in range(0, n, BS):
            idx = perm[s:s + BS]
            loss = T.flow_loss(m, tr[0][idx], tr[1][idx], tr[2][idx])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % EVAL_EVERY == 0:
            m.eval()
            with torch.no_grad():
                g = torch.Generator(device=DEV).manual_seed(7)
                v = float(np.mean([T.flow_loss(m, va[0], va[1], va[2], g).item() for _ in range(3)]))
            if v < best:
                best, best_ep = v, ep
                best_state = {k: x.detach().clone() for k, x in m.state_dict().items()}
    m.load_state_dict(best_state)
    return m, best_ep, best


def main():
    eps = load_main()
    nfr = sum(len(e["bbox"]) for e in eps)
    folds = make_folds(eps)
    print(f"{len(eps)}ep {nfr}프레임 AZMASK={AZMASK} device={DEV} epochs={EPOCHS} seeds={SEEDS}", flush=True)
    res = {k: [] for k in list(ARCHS) + ["MLP_regression"]}
    t0 = time.time()
    for seed in SEEDS:
        rng = np.random.RandomState(seed)
        for fi, te_idx in enumerate(folds):
            rest = [i for i in range(len(eps)) if i not in te_idx]
            rng.shuffle(rest)
            nva = max(1, int(0.15 * len(rest)))
            va_idx, tr_idx = rest[:nva], rest[nva:]
            tr, va, te = T.to_t(eps, tr_idx), T.to_t(eps, va_idx), T.to_t(eps, te_idx)
            y_mean = tr[2].mean((0, 1))
            for name, cls in ARCHS.items():
                m, bep, bv = train_inner(cls, tr, va, seed)
                r = T.evaluate(m, te, y_mean)
                r.update(direction_metrics(m, te), seed=seed, fold=fi, best_epoch=bep, inner_val_flow=bv)
                res[name].append(r)
            res["MLP_regression"].append(dict(seed=seed, fold=fi,
                                              mse_axis=T.train_reg(T.to_t(eps, tr_idx), te, seed, epochs=60)))
            print(f"  seed {seed} fold {fi} ({(time.time() - t0) / 60:.1f}분)", flush=True)
    agg = {}
    for name in ARCHS:
        L = res[name]
        agg[name] = {k: np.nanmean([r[k] for r in L], 0).tolist() if isinstance(L[0][k], list) else float(np.nanmean([r[k] for r in L]))
                     for k in ("flow_loss", "mse_mean_axis", "r2_mean_axis", "base_mean_mse_axis", "diversity", "angle_deg", "go_f1", "mag_mae", "best_epoch")}
        agg[name]["fold_seed_std_r2"] = np.std([r["r2_mean_axis"] for r in L], 0).tolist()
    agg["MLP_regression"] = dict(mse_axis=np.mean([r["mse_axis"] for r in res["MLP_regression"]], 0).tolist())
    summary = dict(set="egvla2_v3_main", episodes=len(eps), frames=nfr, azmask=AZMASK, epochs=EPOCHS, seeds=SEEDS,
                   folds=[[eps[i]["name"] for i in f] for f in folds], aggregate=agg, runs=res)
    (T.OUT_DIR / f"summary_s1{TAG}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(agg, indent=1, ensure_ascii=False))
    if os.environ.get("SAVE") == "1":
        allt = T.to_t(eps, range(len(eps)))
        ck = ROOT / "runs" / "egvla2_fm"
        ck.mkdir(parents=True, exist_ok=True)
        for name in ("A_mlp_fm", "C_pi0_adaln"):
            m, bep, _ = train_inner(ARCHS[name], allt, T.to_t(eps, folds[0]), 0)  # 전체 학습, 조기 종료 기준은 fold0(참고용)
            torch.save(dict(arch=name, state=m.state_dict(), bdim=allt[0].shape[1], fdim=allt[1].shape[1], window=T.WINDOW,
                            H=T.H, epoch=bep), ck / f"{name}.pt")
            print("saved", ck / f"{name}.pt")


if __name__ == "__main__":
    main()

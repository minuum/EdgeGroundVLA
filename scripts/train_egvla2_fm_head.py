#!/usr/bin/env python3
"""EGVLA2: 연속 액션(v3)으로 Flow-Matching 액션헤드(A MLP-FM / B FiLM-FM / C π0형 AdaLN-Zero)를 실제로 학습해 본다.

plan: docs/plans/plan_20261013_egvla2_fm_action_head.md §8. 우열 비교가 아니라 "학습되는가"를 본다.
데이터: logs/manual_dataset_sets.json 의 pilot_v2_gauge_20261013(기본) 15ep, 5-fold 에피소드 hold-out.
입력: bbox 3프레임 창(12-d) + F2 풀링 특징 768-d (docs/v5/egvla2_fm/v3_florence2_cache.npz).
타깃: TARGET=exec(기본)|scaled|raw 의 H=8 청크(끝은 0 패딩).
실행: .venv_lerobot/bin/python3 scripts/train_egvla2_fm_head.py   (환경변수: TARGET, SET, EPOCHS, SEEDS)
출력: docs/v5/egvla2_fm/summary_{TARGET}.json
"""
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "ROS_action" / "mobile_vla_dataset_egvla2"
OUT_DIR = ROOT / "docs" / "v5" / "egvla2_fm"
CACHE = OUT_DIR / "v3_florence2_cache.npz"
TARGET = os.environ.get("TARGET", "exec")
SET = os.environ.get("SET", "pilot_v2_gauge_20261013")
EPOCHS = int(os.environ.get("EPOCHS", 150))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0,1,2").split(",")]
H, ADIM, WINDOW, HID = 8, 3, 3, 128
STEPS_EULER = 10
K_SAMPLES = 8
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_episodes():
    sets = json.loads((ROOT / "logs" / "manual_dataset_sets.json").read_text())["sets"][SET]
    cache = np.load(CACHE)
    eps = []
    for name in sorted(n if n.endswith(".h5") else n + ".h5" for n in sets):
        with h5py.File(DATA / name, "r") as f:
            a = np.array(f[{"exec": "actions_exec", "scaled": "actions_scaled", "raw": "actions_raw"}[TARGET]], dtype=np.float32)
        bb, ft = cache[f"{name}|bbox"].astype(np.float32), cache[f"{name}|feat"].astype(np.float32)
        n = len(a)
        assert len(bb) == n and len(ft) == n, (name, len(bb), n)
        win = np.stack([np.concatenate([bb[max(0, t - WINDOW + 1 + k)] for k in range(WINDOW)]) for t in range(n)])
        chunk = np.zeros((n, H, ADIM), np.float32)
        for t in range(n):
            seg = a[t:t + H]
            chunk[t, :len(seg)] = seg
        eps.append(dict(name=name, bbox=win, feat=ft, chunk=chunk))
    return eps


def temb(t, dim=64):
    half = dim // 2
    fr = torch.exp(-np.log(10000) * torch.arange(half, device=t.device) / half)
    ang = t[:, None] * 1000 * fr[None]
    return torch.cat([ang.sin(), ang.cos()], -1)


class MLPFM(nn.Module):                      # 구조 A
    def __init__(s, bdim, fdim):
        super().__init__()
        s.fp = nn.Sequential(nn.LayerNorm(fdim), nn.Linear(fdim, HID), nn.GELU())
        s.net = nn.Sequential(nn.Linear(HID + bdim + 64 + H * ADIM, 256), nn.GELU(), nn.Dropout(0.1),
                              nn.Linear(256, 256), nn.GELU(), nn.Linear(256, H * ADIM))

    def forward(s, bbox, feat, xt, t):
        return s.net(torch.cat([s.fp(feat), bbox, temb(t), xt.flatten(1)], -1)).view(-1, H, ADIM)


class FiLMFM(nn.Module):                     # 구조 B: 서브골(bbox) + flow 시간을 FiLM으로 주입
    def __init__(s, bdim, fdim):
        super().__init__()
        s.fp = nn.Sequential(nn.LayerNorm(fdim), nn.Linear(fdim, HID))
        s.film = nn.Sequential(nn.Linear(bdim + 64, 128), nn.GELU(), nn.Linear(128, 2 * HID))
        s.trunk = nn.Sequential(nn.GELU(), nn.Linear(HID + H * ADIM, 256), nn.GELU(), nn.Dropout(0.1),
                                nn.Linear(256, 256), nn.GELU(), nn.Linear(256, H * ADIM))

    def forward(s, bbox, feat, xt, t):
        g, b = s.film(torch.cat([bbox, temb(t)], -1)).chunk(2, -1)
        h = (1 + g) * s.fp(feat) + b
        return s.trunk(torch.cat([h, xt.flatten(1)], -1)).view(-1, H, ADIM)


class AdaBlock(nn.Module):
    def __init__(s, d, heads=4):
        super().__init__()
        s.n1, s.n2 = nn.LayerNorm(d, elementwise_affine=False), nn.LayerNorm(d, elementwise_affine=False)
        s.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        s.ada = nn.Linear(d, 6 * d)
        nn.init.zeros_(s.ada.weight); nn.init.zeros_(s.ada.bias)   # AdaLN-Zero

    def forward(s, h, prefix, c):
        a1, b1, g1, a2, b2, g2 = s.ada(c)[:, None].chunk(6, -1)
        x = s.n1(h) * (1 + a1) + b1
        kv = torch.cat([prefix, x], 1)
        h = h + g1 * s.attn(x, kv, kv, need_weights=False)[0]
        return h + g2 * s.mlp(s.n2(h) * (1 + a2) + b2)


class PiZeroFM(nn.Module):                   # 구조 C: π0형 action expert (경량)
    def __init__(s, bdim, fdim, layers=2, d=HID):
        super().__init__()
        s.sub, s.vis = nn.Linear(bdim, d), nn.Sequential(nn.LayerNorm(fdim), nn.Linear(fdim, d))
        s.inp, s.pos = nn.Linear(ADIM, d), nn.Parameter(torch.zeros(1, H, d))
        s.tm = nn.Sequential(nn.Linear(64, d), nn.GELU(), nn.Linear(d, d))
        s.blocks = nn.ModuleList(AdaBlock(d) for _ in range(layers))
        s.out = nn.Linear(d, ADIM)

    def forward(s, bbox, feat, xt, t):
        prefix = torch.stack([s.sub(bbox), s.vis(feat)], 1)
        h, c = s.inp(xt) + s.pos, s.tm(temb(t))
        for blk in s.blocks:
            h = blk(h, prefix, c)
        return s.out(h)


class MLPReg(nn.Module):                     # 기준선(sanity): 동일 입력 회귀
    def __init__(s, bdim, fdim):
        super().__init__()
        s.net = nn.Sequential(nn.LayerNorm(fdim + bdim), nn.Linear(fdim + bdim, 256), nn.GELU(), nn.Dropout(0.1),
                              nn.Linear(256, H * ADIM))

    def forward(s, bbox, feat):
        return s.net(torch.cat([feat, bbox], -1)).view(-1, H, ADIM)


ARCH = {"A_mlp_fm": MLPFM, "B_film_fm": FiLMFM, "C_pi0_adaln": PiZeroFM}


def flow_loss(m, bbox, feat, a, g=None):
    x0 = torch.randn_like(a, generator=g) if g is not None else torch.randn_like(a)
    t = torch.rand(len(a), device=a.device, generator=g) if g is not None else torch.rand(len(a), device=a.device)
    xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * a
    return F.mse_loss(m(bbox, feat, xt, t), a - x0)


@torch.no_grad()
def sample(m, bbox, feat, k=K_SAMPLES):
    m.eval()
    B = len(bbox)
    bb, ft = bbox.repeat_interleave(k, 0), feat.repeat_interleave(k, 0)
    x = torch.randn(B * k, H, ADIM, device=bbox.device)
    for i in range(STEPS_EULER):
        t = torch.full((B * k,), i / STEPS_EULER, device=bbox.device)
        x = x + m(bb, ft, x, t) / STEPS_EULER
    return x.view(B, k, H, ADIM)


def to_t(eps, idx):
    cat = lambda key: torch.tensor(np.concatenate([eps[i][key] for i in idx])).to(DEV)
    return cat("bbox"), cat("feat"), cat("chunk")


def evaluate(m, te, y_tr_mean):
    bb, ft, y = te
    S = sample(m, bb, ft)
    mean_s = S.mean(1)
    mse_ax = ((S - y[:, None]) ** 2).mean((0, 1, 2)).cpu().numpy()          # 축별(개별 샘플 기준)
    mse_mean = ((mean_s - y) ** 2).mean((0, 1)).cpu().numpy()
    var_y = y.var((0, 1)).cpu().numpy()
    r2 = 1 - mse_mean / np.maximum(var_y, 1e-8)
    base = ((y - y_tr_mean) ** 2).mean((0, 1)).cpu().numpy()
    div = S.std(1).mean((0, 1, 2)).item()                                    # 같은 조건 샘플 간 표준편차
    g = torch.Generator(device=DEV).manual_seed(123)
    fl = np.mean([flow_loss(m, bb, ft, y, g).item() for _ in range(5)])
    return dict(flow_loss=float(fl), mse_sample_axis=mse_ax.tolist(), mse_mean_axis=mse_mean.tolist(),
                r2_mean_axis=r2.tolist(), base_mean_mse_axis=base.tolist(), diversity=float(div))


def train_one(cls, tr, te, seed, epochs=EPOCHS):
    torch.manual_seed(seed); np.random.seed(seed)
    m = cls(tr[0].shape[1], tr[1].shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    n, curve = len(tr[0]), []
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(n, device=DEV)
        for s in range(0, n, 64):
            idx = perm[s:s + 64]
            loss = flow_loss(m, tr[0][idx], tr[1][idx], tr[2][idx])
            opt.zero_grad(); loss.backward(); opt.step()
        if (ep + 1) % 25 == 0 or ep == 0:
            m.eval()
            with torch.no_grad():
                g = torch.Generator(device=DEV).manual_seed(7)
                curve.append((ep + 1, float(np.mean([flow_loss(m, tr[0], tr[1], tr[2], g).item() for _ in range(3)])),
                              float(np.mean([flow_loss(m, te[0], te[1], te[2], g).item() for _ in range(3)]))))
    return m, curve


def train_reg(tr, te, seed, epochs=EPOCHS):
    torch.manual_seed(seed)
    m = MLPReg(tr[0].shape[1], tr[1].shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(tr[0]), device=DEV)
        for s in range(0, len(perm), 64):
            idx = perm[s:s + 64]
            loss = F.mse_loss(m(tr[0][idx], tr[1][idx]), tr[2][idx]); opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        p = m(te[0], te[1])
        mse = ((p - te[2]) ** 2).mean((0, 1)).cpu().numpy()
    return mse.tolist()


def overfit_test(eps):
    """1-batch 과적합: 32프레임에 600스텝 → flow loss와 샘플 MSE가 0 근처로 가는가."""
    out = {}
    tr = to_t(eps, range(len(eps)))
    idx = torch.randperm(len(tr[0]), generator=torch.Generator().manual_seed(0))[:32].to(DEV)
    sub = tuple(x[idx] for x in tr)
    for name, cls in ARCH.items():
        torch.manual_seed(0)
        m = cls(sub[0].shape[1], sub[1].shape[1]).to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0)
        for step in range(600):
            m.train(); loss = flow_loss(m, *sub); opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            g = torch.Generator(device=DEV).manual_seed(1)
            fl = float(np.mean([flow_loss(m, *sub, g).item() for _ in range(10)]))
        S = sample(m, sub[0], sub[1])
        out[name] = dict(final_flow_loss=fl, sample_mse=float(((S.mean(1) - sub[2]) ** 2).mean()))
    return out


def main():
    eps = load_episodes()
    nfr = sum(len(e["bbox"]) for e in eps)
    print(f"세트 {SET}: {len(eps)}ep, {nfr}프레임, TARGET={TARGET}, device={DEV}", flush=True)
    summary = dict(set=SET, target=TARGET, episodes=[e["name"] for e in eps], frames=nfr, epochs=EPOCHS, seeds=SEEDS,
                   chunk=H, euler_steps=STEPS_EULER)
    summary["one_batch_overfit"] = overfit_test(eps)
    print("1-batch 과적합:", json.dumps(summary["one_batch_overfit"]), flush=True)
    folds = [list(range(i, len(eps), 5)) for i in range(5)]
    res = {k: [] for k in list(ARCH) + ["MLP_regression"]}
    curves = {k: [] for k in ARCH}
    t0 = time.time()
    for seed in SEEDS:
        for fi, te_idx in enumerate(folds):
            tr_idx = [i for i in range(len(eps)) if i not in te_idx]
            tr, te = to_t(eps, tr_idx), to_t(eps, te_idx)
            y_mean = tr[2].mean((0, 1))
            for name, cls in ARCH.items():
                m, curve = train_one(cls, tr, te, seed)
                r = evaluate(m, te, y_mean); r.update(seed=seed, fold=fi)
                res[name].append(r); curves[name].append(curve)
            res["MLP_regression"].append(dict(seed=seed, fold=fi, mse_axis=train_reg(tr, te, seed)))
            print(f"  seed {seed} fold {fi} 완료 ({(time.time() - t0) / 60:.1f}분)", flush=True)
    agg = {}
    for name in ARCH:
        L = res[name]
        agg[name] = dict(
            flow_loss=float(np.mean([r["flow_loss"] for r in L])),
            mse_mean_axis=np.mean([r["mse_mean_axis"] for r in L], 0).tolist(),
            mse_sample_axis=np.mean([r["mse_sample_axis"] for r in L], 0).tolist(),
            r2_mean_axis=np.mean([r["r2_mean_axis"] for r in L], 0).tolist(),
            base_mean_mse_axis=np.mean([r["base_mean_mse_axis"] for r in L], 0).tolist(),
            diversity=float(np.mean([r["diversity"] for r in L])),
            train_flow_final=float(np.mean([c[-1][1] for c in curves[name]])),
            test_flow_final=float(np.mean([c[-1][2] for c in curves[name]])),
            test_flow_min=float(np.mean([min(x[2] for x in c) for c in curves[name]])))
    agg["MLP_regression"] = dict(mse_axis=np.mean([r["mse_axis"] for r in res["MLP_regression"]], 0).tolist())
    summary["aggregate"] = agg
    summary["runs"] = res
    (OUT_DIR / f"summary_{TARGET}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    print(json.dumps(agg, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

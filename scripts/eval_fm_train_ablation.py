#!/usr/bin/env python3
"""FM-π0 학습 설정 ablation (π0 논문과 다른 점을 하나씩 바꿔 봄). 분할: 5-fold + 안 본 목표 위치 5종, seed 0, 서브골 드롭아웃 30%, 120 epoch.
변형: flow 시간 샘플링(균등=현재 / π0식 Beta 노이즈 강조 / logit-normal), 트랜스포머 층수(1·2·4), 폭(64·128·256).
평가: 첫 스텝 방향 오차, 이동 구간(3스텝) 평균 방향 오차(청크 앞 3스텝 평균), 서브골 결측 시 방향 오차, 파라미터 수.
출력: docs/v5/egvla2_fm/fm_train_ablation.json (분할 단위 저장·재개). 실행: .venv_lerobot/bin/python3 scripts/eval_fm_train_ablation.py  (SMOKE=1: epoch 2)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_runtime_ablation as RA  # noqa: E402
import eval_subgoal_ablation as AB  # noqa: E402
import train_egvla2_s1 as S1  # noqa: E402
from robovlm_nav.models.egvla2_fm_heads import PiZeroFM  # noqa: E402
T = S1.T
DEV, BS = AB.DEV, AB.BS
EPOCHS = 2 if os.getenv("SMOKE") else 120
OUT = ROOT / "docs/v5/egvla2_fm" / ("fm_train_ablation_smoke.json" if os.getenv("SMOKE") else "fm_train_ablation.json")
VARIANTS = {"uniform|L2|d128": dict(tau="uniform", layers=2, d=128), "beta|L2|d128": dict(tau="beta", layers=2, d=128), "logitnormal|L2|d128": dict(tau="logitnormal", layers=2, d=128),
            "uniform|L1|d128": dict(tau="uniform", layers=1, d=128), "uniform|L4|d128": dict(tau="uniform", layers=4, d=128),
            "uniform|L2|d64": dict(tau="uniform", layers=2, d=64), "uniform|L2|d256": dict(tau="uniform", layers=2, d=256)}


def sample_tau(kind, n):
    if kind == "uniform":
        return torch.rand(n, device=DEV)
    if kind == "beta":                                   # π0: p(τ)=Beta((s-τ)/s; 1.5, 1), s=0.999 → 노이즈(τ 작음) 쪽 강조
        b = torch.distributions.Beta(1.5, 1.0).sample((n,)).to(DEV)
        return 0.999 * (1 - b)
    return torch.sigmoid(torch.randn(n, device=DEV))      # logit-normal(SD3)


def fit(cfg, tr, seed=0, drop=0.3):
    torch.manual_seed(seed)
    m = PiZeroFM(tr[0].shape[1], tr[1].shape[1], layers=cfg["layers"], d=cfg["d"]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2); n = len(tr[0])
    for _ in range(EPOCHS):
        m.train(); perm = torch.randperm(n, device=DEV)
        for s in range(0, n, BS):
            i = perm[s:s + BS]; bb = tr[0][i] * (torch.rand(len(i), 1, device=DEV) > drop).float(); a = tr[2][i]
            x0 = torch.randn_like(a); t = sample_tau(cfg["tau"], len(i))
            xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * a
            l = torch.nn.functional.mse_loss(m(bb, tr[1][i], xt, t), a - x0); opt.zero_grad(); l.backward(); opt.step()
    m.eval(); return m


def main():
    eps = S1.load_main(); cache = np.load(T.CACHE); E = AB.prep(eps, cache, 3); folds = S1.make_folds(eps)
    splits = {f"fold{i}": folds[i] for i in range(len(folds))}
    for pos in ("center", "weak_left", "strong_left", "weak_right", "strong_right"):
        splits[f"lopo_{pos}"] = [i for i, e in enumerate(eps) if e["cell"].startswith(pos + "|")]
    order = [x for x in os.getenv("SPLITS", "").split(",") if x in splits]
    if order:
        splits = {x: splits[x] for x in order}
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, te_idx in splits.items():
        res.setdefault(name, {})
        tr_idx = [i for i in range(len(E)) if i not in te_idx]
        tr, te = T.to_t(E, tr_idx), T.to_t(E, te_idx)
        gt = te[2][..., :2].cpu().numpy(); g_first, g_burst = gt[:, 0], gt[:, :3].mean(1)
        for vname, cfg in VARIANTS.items():
            if os.getenv("VARIANTS") and vname not in os.getenv("VARIANTS").split(","):
                continue
            if vname in res[name]:
                continue
            m = fit(cfg, tr)
            S = RA.draw(m, te[0], te[1], 8, 10).mean(1)                       # (N,H,2)
            Sm = RA.draw(m, torch.zeros_like(te[0]), te[1], 8, 10).mean(1)
            res[name][vname] = dict(first=RA.angle(S[:, 0], g_first), n3_vs_burst=RA.angle(S[:, :3].mean(1), g_burst), n1_vs_burst=RA.angle(S[:, 0], g_burst),
                                    miss_first=RA.angle(Sm[:, 0], g_first), params=int(sum(p.numel() for p in m.parameters())))
            OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
            print(name, vname, {k: round(v, 2) for k, v in res[name][vname].items()}, flush=True)


if __name__ == "__main__":
    main()

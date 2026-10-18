#!/usr/bin/env python3
"""서브골 ablation의 과적합 점검: 변형(full / nobbox / drop30) × 헤드(B2, C, MLP 회귀)의 학습·hold-out 곡선(epoch별).
기록: train flow loss(학습 4000프레임 부분집합), hold-out flow loss, hold-out 방향 오차(°), GO/STOP F1, (회귀는 MSE). 250 epoch까지, 1·5·10·20…마다.
분할: fold0, lopo_center, lopo_strong_right (seed 0). 출력: docs/v5/egvla2_fm/ablation_curves.json (분할마다 저장)
실행: .venv_lerobot/bin/python3 scripts/eval_ablation_curves.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB  # noqa: E402
import train_egvla2_s1 as S1  # noqa: E402
T, DEV = S1.T, S1.DEV
EPOCHS = 250
GRID = {1, 5} | set(range(10, EPOCHS + 1, 10))
OUT = ROOT / "docs/v5/egvla2_fm/ablation_curves.json"
BS = 256


def reg_metrics(m, te):
    with torch.no_grad():
        p = m(te[0], te[1])[:, 0, :2]; y = te[2][:, 0, :2]
        pm, ym = p.norm(dim=-1), y.norm(dim=-1); both = (pm > 0.2) & (ym > 0.2)
        cos = (p[both] * y[both]).sum(-1) / (pm[both] * ym[both])
        ang = torch.rad2deg(torch.acos(cos.clamp(-1, 1))).mean().item()
        pg, yg = pm > 0.2, ym > 0.2
        tp = (pg & yg).sum().item(); fp = (pg & ~yg).sum().item(); fn = (~pg & yg).sum().item()
        return dict(angle=ang, go_f1=2 * tp / max(2 * tp + fp + fn, 1), val=F.mse_loss(m(te[0], te[1]), te[2]).item())


def run(cls, tr, te, drop, reg=False, seed=0):
    torch.manual_seed(seed)
    m = cls(tr[0].shape[1], tr[1].shape[1]).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    n = len(tr[0]); sub = torch.randperm(n, generator=torch.Generator().manual_seed(1))[:4000].to(DEV); curve = []
    for ep in range(1, EPOCHS + 1):
        m.train(); perm = torch.randperm(n, device=DEV)
        for s in range(0, n, BS):
            i = perm[s:s + BS]; bb = tr[0][i]
            if drop > 0:
                bb = bb * (torch.rand(len(i), 1, device=DEV) > drop).float()
            l = F.mse_loss(m(bb, tr[1][i]), tr[2][i]) if reg else T.flow_loss(m, bb, tr[1][i], tr[2][i])
            opt.zero_grad(); l.backward(); opt.step()
        if ep in GRID:
            m.eval()
            with torch.no_grad():
                if reg:
                    r = reg_metrics(m, te); r["train"] = F.mse_loss(m(tr[0][sub], tr[1][sub]), tr[2][sub]).item()
                else:
                    g = torch.Generator(device=DEV).manual_seed(7)
                    val = float(np.mean([T.flow_loss(m, te[0], te[1], te[2], g).item() for _ in range(3)]))
                    trn = float(np.mean([T.flow_loss(m, tr[0][sub], tr[1][sub], tr[2][sub], g).item() for _ in range(3)]))
                    d = S1.direction_metrics(m, te); r = dict(angle=d["angle_deg"], go_f1=d["go_f1"], val=val, train=trn)
            r["ep"] = ep; curve.append(r)
    return curve


def main():
    eps = S1.load_main(); cache = np.load(T.CACHE); E = AB.prep(eps, cache, 3); folds = S1.make_folds(eps)
    splits = {"fold0": folds[0]}
    for p in ("center", "strong_right"):
        splits[f"lopo_{p}"] = [i for i, e in enumerate(eps) if e["cell"].startswith(p + "|")]
    allr = json.loads(OUT.read_text()) if OUT.exists() else {}
    t0 = time.time()
    for sname, te_idx in splits.items():
        if sname in allr:
            continue
        tr_idx = [i for i in range(len(E)) if i not in te_idx]; tr0, te0 = T.to_t(E, tr_idx), T.to_t(E, te_idx); res = {}
        for var in ("full", "nobbox", "drop30"):
            tr, te = tr0, te0
            if var == "nobbox":
                tr = (torch.zeros_like(tr0[0]), tr0[1], tr0[2]); te = (torch.zeros_like(te0[0]), te0[1], te0[2])
            for h in AB.HEADS:
                res[f"{h}|{var}"] = run(S1.ARCHS[h], tr, te, 0.3 if var == "drop30" else 0.0)
                print(sname, h, var, "완료", flush=True)
            res[f"MLP_regression|{var}"] = run(T.MLPReg, tr, te, 0.3 if var == "drop30" else 0.0, reg=True)
        allr[sname] = res; OUT.write_text(json.dumps(allr)); print("==", sname, f"{(time.time() - t0) / 60:.1f}분", flush=True)


if __name__ == "__main__":
    main()

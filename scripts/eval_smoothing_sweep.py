#!/usr/bin/env python3
"""서빙 평활 파라미터 오프라인 스윕: 안 본 에피소드에서 C(서브골 드롭아웃30%, 120 epoch) 예측을 K(샘플 수)·EMA α별로 시간 순서대로 평활해
방향 오차·프레임 저크·GO/STOP F1·반응 지연(프레임)을 비교한다. 분할 2종: fold0, LOPO strong_right. seed 0.
저크 = 연속 프레임 첫 스텝 (lx,ly) 변화량 평균, 반응 지연 = 예측이 사람 조작을 s프레임 늦게 따라간다고 볼 때 오차가 최소인 s.
실행: .venv_lerobot/bin/python3 scripts/eval_smoothing_sweep.py   출력: docs/v5/egvla2_fm/smoothing_sweep.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB  # noqa: E402
import train_egvla2_s1 as S1  # noqa: E402
T = S1.T
KS, ALPHAS = (1, 4, 8), (0.0, 0.3, 0.5, 0.7, 0.85)
GO = 0.2


def metrics(preds, gts, lens):
    """preds/gts: (N,2) 이어 붙인 에피소드들, lens: 에피소드 프레임 수."""
    ang, jit, dl, tp, fp, fn, o = [], [], [], 0, 0, 0, 0
    best = {s: [] for s in range(0, 7)}
    for n in lens:
        p, g = preds[o:o + n], gts[o:o + n]; o += n
        pm, gm = np.linalg.norm(p, axis=1), np.linalg.norm(g, axis=1)
        both = (pm > GO) & (gm > GO)
        if both.any():
            c = (p[both] * g[both]).sum(1) / (pm[both] * gm[both]); ang += list(np.degrees(np.arccos(np.clip(c, -1, 1))))
        jit += list(np.linalg.norm(np.diff(p, axis=0), axis=1))
        pg, yg = pm > GO, gm > GO
        tp += int((pg & yg).sum()); fp += int((pg & ~yg).sum()); fn += int((~pg & yg).sum())
        for s in best:
            if n > s + 8:
                best[s].append(float(np.linalg.norm(p[s:] - g[:n - s], axis=1).mean()))
    dl = int(min(best, key=lambda s: np.mean(best[s]) if best[s] else 1e9))
    return dict(angle=float(np.mean(ang)), jitter=float(np.mean(jit)), go_f1=2 * tp / max(2 * tp + fp + fn, 1), delay_frames=dl)


def ema(x, a, lens):
    if a == 0:
        return x
    out, o = x.copy(), 0
    for n in lens:
        for t in range(1, n):
            out[o + t] = a * out[o + t - 1] + (1 - a) * x[o + t]      # α = 과거 가중(서버의 ema_alpha = 새 값 가중 = 1-α)
        o += n
    return out


def main():
    eps = S1.load_main(); cache = np.load(T.CACHE); E = AB.prep(eps, cache, 3); folds = S1.make_folds(eps)
    splits = {"fold0": folds[0], "lopo_strong_right": [i for i, e in enumerate(eps) if e["cell"].startswith("strong_right|")]}
    res = {}
    for sname, te_idx in splits.items():
        tr_idx = [i for i in range(len(E)) if i not in te_idx]; tr, te = T.to_t(E, tr_idx), T.to_t(E, te_idx)
        m = AB.fit(S1.ARCHS["C_pi0_adaln"], tr, 120, 0, drop=0.3)
        with torch.no_grad():
            torch.manual_seed(0); S = T.sample(m, te[0], te[1], k=8)[:, :, 0, :2].cpu().numpy()      # (N,8,2)
        gts = te[2][:, 0, :2].cpu().numpy(); lens = [len(E[i]["bbox"]) for i in te_idx]
        res[sname] = {}
        for k in KS:
            base = S[:, :k].mean(1)
            for a in ALPHAS:
                res[sname][f"K{k}|a{a}"] = metrics(ema(base, a, lens), gts, lens)
        print(sname, {k: round(v["angle"], 1) for k, v in res[sname].items() if k.startswith("K8")}, flush=True)
    (ROOT / "docs/v5/egvla2_fm/smoothing_sweep.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

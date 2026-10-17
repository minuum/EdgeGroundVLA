#!/usr/bin/env python3
"""EGVLA2 서브골(bbox) 형태 ablation: FM 헤드(B2, C)에서 서브골 유무·창 길이·학습 중 서브골 드롭아웃의 영향을 hold-out으로 비교.
변형: full(창3) / nobbox(서브골 0) / w1(현재 프레임만) / w5 / drop30(학습 시 30% 확률로 서브골을 0으로 가림).
평가: 서브골 정상 입력, 그리고 서브골 결측(전부 0) 입력 두 조건. 지표: 방향 오차, GO/STOP F1, 청크 이동 MSE(lx,ly).
분할: 5-fold(stratified) + leave-one-position-out. 환경변수 SEEDS(기본 0), MODE(fold|lopo|both). 헤드별 고정 epoch(B2 220, C 120).
실행: .venv_lerobot/bin/python3 scripts/eval_subgoal_ablation.py   출력: docs/v5/egvla2_fm/subgoal_ablation.json(분할마다 저장)
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import train_egvla2_s1 as S1  # noqa: E402
T, DEV = S1.T, S1.DEV
HEADS = {"B2_film_res_fm": 220, "C_pi0_adaln": 120}
VARS = ["full", "nobbox", "w1", "w5", "drop30"]
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
MODE = os.environ.get("MODE", "both")
OUT = ROOT / "docs/v5/egvla2_fm/subgoal_ablation.json"
BS = 256


def window(raw, W):
    n = len(raw)
    return np.stack([np.concatenate([raw[max(0, t - W + 1 + k)] for k in range(W)]) for t in range(n)]).astype(np.float32)


def prep(eps, cache, W):
    out = []
    for e in eps:
        e2 = dict(e); e2["bbox"] = window(cache[f"{e['name']}|bbox"].astype(np.float32), W); out.append(e2)
    return out


def fit(cls, tr, epochs, seed, drop=0.0):
    torch.manual_seed(seed)
    m = cls(tr[0].shape[1], tr[1].shape[1]).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    n = len(tr[0])
    for _ in range(epochs):
        m.train(); perm = torch.randperm(n, device=DEV)
        for s in range(0, n, BS):
            i = perm[s:s + BS]; bb = tr[0][i]
            if drop > 0:
                bb = bb * (torch.rand(len(i), 1, device=DEV) > drop).float()
            l = T.flow_loss(m, bb, tr[1][i], tr[2][i]); opt.zero_grad(); l.backward(); opt.step()
    m.eval(); return m


def score(m, te, missing):
    bb = torch.zeros_like(te[0]) if missing else te[0]
    t2 = (bb, te[1], te[2])
    r = S1.direction_metrics(m, t2)
    S = T.sample(m, bb, te[1]).mean(1)
    r["mse_xy"] = float(((S[..., :2] - te[2][..., :2]) ** 2).mean().item())
    return r


def run_split(name, eps, cache, te_idx, seed):
    tr_idx = [i for i in range(len(eps)) if i not in te_idx]
    res = {}
    cache_w = {W: prep(eps, cache, W) for W in (1, 3, 5)}
    for var in VARS:
        W = {"w1": 1, "w5": 5}.get(var, 3)
        E = cache_w[W]; tr, te = T.to_t(E, tr_idx), T.to_t(E, te_idx)
        if var == "nobbox":
            tr = (torch.zeros_like(tr[0]), tr[1], tr[2])
        for h, ep in HEADS.items():
            m = fit(S1.ARCHS[h], tr, ep, seed, drop=0.3 if var == "drop30" else 0.0)
            ev = {"normal": score(m, (torch.zeros_like(te[0]) if var == "nobbox" else te[0], te[1], te[2]), False), "missing": score(m, te, True)}
            res[f"{h}|{var}"] = ev
            print(name, seed, h, var, {k: round(v["angle_deg"], 1) for k, v in ev.items()}, flush=True)
    return res


def main():
    eps = S1.load_main(); cache = np.load(T.CACHE)
    folds = S1.make_folds(eps)
    pos = sorted({e["cell"].split("|")[0] for e in eps})
    splits = {}
    if MODE in ("fold", "both"):
        splits.update({f"fold{i}": f for i, f in enumerate(folds)})
    if MODE in ("lopo", "both"):
        splits.update({f"lopo_{p}": [i for i, e in enumerate(eps) if e["cell"].startswith(p + "|")] for p in pos})
    allr = json.loads(OUT.read_text()) if OUT.exists() else {}
    t0 = time.time()
    for seed in SEEDS:
        for sname, te_idx in splits.items():
            key = f"{sname}|seed{seed}"
            if key in allr:
                continue
            allr[key] = run_split(sname, eps, cache, te_idx, seed)
            OUT.write_text(json.dumps(allr, indent=1, ensure_ascii=False))
            print(f"== {key} 완료 ({(time.time() - t0) / 60:.1f}분)", flush=True)


if __name__ == "__main__":
    main()

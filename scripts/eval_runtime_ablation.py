#!/usr/bin/env python3
"""서빙 런타임 요인 오프라인 ablation (CH78 ⚙️ 탭 A묶음). 분할: 5-fold(fold0~4) + 안 본 목표 위치 5종(LOPO), seed 0.
모델: C(FM-π0) 서브골 드롭아웃 30%와 50%, 120 epoch(배포와 같은 조건). 학습 없이 평가만 바꾸는 항목:
  (1) 청크 앞 n스텝 평균(n=1~8) × 정답 2종(현재 첫 스텝 / 이동 구간 3스텝 평균)
  (2) K 샘플 수 1·2·4·8·16, (3) Euler 스텝 1·2·3·5·10·20, (4) EMA α, (5) 서브골 결측 견고성(드롭아웃 30 vs 50).
출력: docs/v5/egvla2_fm/runtime_ablation.json (분할 단위로 저장·재개). 실행: .venv_lerobot/bin/python3 scripts/eval_runtime_ablation.py  (SMOKE=1: epoch 2)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB  # noqa: E402
import train_egvla2_s1 as S1  # noqa: E402
from robovlm_nav.models.egvla2_fm_heads import sample_chunk  # noqa: E402
T = S1.T
EPOCHS = 2 if os.getenv("SMOKE") else 120
OUT = ROOT / "docs/v5/egvla2_fm" / ("runtime_ablation_smoke.json" if os.getenv("SMOKE") else "runtime_ablation.json")
GO = 0.2
NS, KS, STEPS, ALPHAS = range(1, 9), (1, 2, 4, 8, 16), (1, 2, 3, 5, 10, 20), (0.0, 0.3, 0.5, 0.7, 0.85)


def angle(p, g):
    pm, gm = np.linalg.norm(p, axis=1), np.linalg.norm(g, axis=1)
    both = (pm > GO) & (gm > GO)
    if not both.any():
        return float("nan")
    c = (p[both] * g[both]).sum(1) / (pm[both] * gm[both])
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))).mean())


def ema(x, a, lens):
    if a == 0:
        return x
    out, o = x.copy(), 0
    for n in lens:
        for t in range(1, n):
            out[o + t] = a * out[o + t - 1] + (1 - a) * x[o + t]
        o += n
    return out


def jitter(p, lens):
    j, o = [], 0
    for n in lens:
        j += list(np.linalg.norm(np.diff(p[o:o + n], axis=0), axis=1)); o += n
    return float(np.mean(j))


@torch.no_grad()
def draw(m, bb, ft, k, steps, seed=0):
    torch.manual_seed(seed)
    return sample_chunk(m, bb, ft, k, steps)[..., :2].cpu().numpy()          # (N,k,H,2)


def run_split(E, tr_idx, te_idx):
    tr, te = T.to_t(E, tr_idx), T.to_t(E, te_idx)
    lens = [len(E[i]["bbox"]) for i in te_idx]
    gt = te[2][..., :2].cpu().numpy()                                       # (N,H,2)
    g_first, g_burst = gt[:, 0], gt[:, :3].mean(1)
    r = {}
    for drop in (0.3, 0.5):
        m = AB.fit(S1.ARCHS["C_pi0_adaln"], tr, EPOCHS, 0, drop=drop)
        key = f"drop{int(drop * 100)}"
        r[key] = {}
        for miss in (False, True):
            bb = torch.zeros_like(te[0]) if miss else te[0]
            S16 = draw(m, bb, te[1], 16, 10)
            r[key][f"miss{int(miss)}|first"] = angle(S16[:, :8].mean(1)[:, 0], g_first)
            if miss:
                continue
            mean8 = S16[:, :8].mean(1)                                       # (N,H,2)
            for n in NS:
                p = mean8[:, :n].mean(1)
                r[key][f"chunk_n{n}|vs_first"] = angle(p, g_first)
                r[key][f"chunk_n{n}|vs_burst3"] = angle(p, g_burst)
            if drop != 0.3:
                continue
            for k in KS:
                p = S16[:, :k].mean(1)
                r[key][f"K{k}|first"] = angle(p[:, 0], g_first); r[key][f"K{k}|n3_vs_burst"] = angle(p[:, :3].mean(1), g_burst)
            for st in STEPS:
                p = draw(m, te[0], te[1], 8, st).mean(1)
                r[key][f"euler{st}|first"] = angle(p[:, 0], g_first); r[key][f"euler{st}|n3_vs_burst"] = angle(p[:, :3].mean(1), g_burst)
            for a in ALPHAS:
                for n in (1, 3):
                    p = ema(mean8[:, :n].mean(1), a, lens)
                    r[key][f"ema{a}|n{n}"] = dict(vs_first=angle(p, g_first), vs_burst3=angle(p, g_burst), jitter=jitter(p, lens))
    return r


def main():
    eps = S1.load_main(); cache = np.load(T.CACHE); E = AB.prep(eps, cache, 3); folds = S1.make_folds(eps)
    splits = {f"fold{i}": folds[i] for i in range(len(folds))}
    for pos in ("center", "weak_left", "strong_left", "weak_right", "strong_right"):
        splits[f"lopo_{pos}"] = [i for i, e in enumerate(eps) if e["cell"].startswith(pos + "|")]
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, te_idx in splits.items():
        if name in res:
            continue
        tr_idx = [i for i in range(len(E)) if i not in te_idx]
        res[name] = run_split(E, tr_idx, te_idx)
        OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
        r = res[name]["drop30"]
        print(name, "first", round(r["chunk_n1|vs_first"], 2), "n3|burst", round(r["chunk_n3|vs_burst3"], 2), "n1|burst", round(r["chunk_n1|vs_burst3"], 2),
              "miss d30/d50", round(r["miss1|first"], 1), round(res[name]["drop50"]["miss1|first"], 1), flush=True)


if __name__ == "__main__":
    main()

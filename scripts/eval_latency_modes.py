#!/usr/bin/env python3
"""두 가지 오프라인 점검(배포 설정: FM-π0, 서브골 드롭아웃30%, 120 epoch, K=8):
① 지연 보상: 프레임 t의 예측 청크 k번째 스텝 vs 사람의 t+k 조작(보상) / 청크 0번째 스텝을 k프레임 늦게 쓰는 경우(현재 서빙) 비교.
② 다봉·경로 모호성: 에피소드 첫 10% vs 나머지에서 K 평균 / 단일 샘플 / 가장 가까운 샘플(best-of-K, 오라클) 오차, 샘플 간 방향 퍼짐, 이전 선택에 가까운 샘플을 잇는 '모드 유지' 방식.
분할: fold0(같은 목표 위치), LOPO strong_right(안 본 목표 위치). 출력: docs/v5/egvla2_fm/latency_modes.json
실행: .venv_lerobot/bin/python3 scripts/eval_latency_modes.py
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
GO = 0.2


def ang(a, b):
    na, nb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    ok = (na > GO) & (nb > GO)
    c = (a * b).sum(-1) / np.maximum(na * nb, 1e-9)
    return np.where(ok, np.degrees(np.arccos(np.clip(c, -1, 1))), np.nan)


def main():
    eps = S1.load_main(); cache = np.load(T.CACHE); E = AB.prep(eps, cache, 3); folds = S1.make_folds(eps)
    splits = {"fold0": folds[0], "lopo_strong_right": [i for i, e in enumerate(eps) if e["cell"].startswith("strong_right|")]}
    out = {}
    for sname, te_idx in splits.items():
        tr_idx = [i for i in range(len(E)) if i not in te_idx]; tr, te = T.to_t(E, tr_idx), T.to_t(E, te_idx)
        m = AB.fit(S1.ARCHS["C_pi0_adaln"], tr, 120, 0, drop=0.3)
        with torch.no_grad():
            torch.manual_seed(0); S = T.sample(m, te[0], te[1], k=8)[..., :2].cpu().numpy()     # (N,K,H,2)
        Y = te[2][:, 0, :2].cpu().numpy(); lens = [len(E[i]["bbox"]) for i in te_idx]
        mean = S.mean(1)                                                                       # (N,H,2)
        # ① 지연
        comp, stale = {}, {}
        for k in range(8):
            ca, sa = [], []; o = 0
            for n in lens:
                if n > k:
                    ca.append(ang(mean[o:o + n - k, k], Y[o + k:o + n])); sa.append(ang(mean[o:o + n - k, 0], Y[o + k:o + n]))
                o += n
            comp[k], stale[k] = float(np.nanmean(np.concatenate(ca))), float(np.nanmean(np.concatenate(sa)))
        # ② 다봉
        first = np.concatenate([np.arange(n) < max(1, int(0.1 * n)) for n in lens])
        s0 = S[:, :, 0]                                                                        # (N,K,2) 첫 스텝 샘플
        e_mean = ang(mean[:, 0], Y); e_single = ang(s0[:, 0], Y)
        e_each = np.stack([ang(s0[:, j], Y) for j in range(8)], 1); e_best = np.nanmin(e_each, 1)
        dirs = np.degrees(np.arctan2(s0[..., 1], s0[..., 0])); mag = np.linalg.norm(s0, axis=-1)
        spread = np.array([np.nanmax(np.abs(((d[mg > GO][:, None] - d[mg > GO][None]) + 180) % 360 - 180)) if (mg > GO).sum() > 1 else np.nan for d, mg in zip(dirs, mag)])
        chain = np.zeros((len(Y), 2)); o = 0
        for n in lens:                                                                         # 모드 유지: 이전 선택에 가장 가까운 샘플
            prev = None
            for t in range(o, o + n):
                if prev is None or np.linalg.norm(prev) < GO:
                    j = 0
                else:
                    j = int(np.argmin(np.linalg.norm(s0[t] - prev, axis=1)))
                chain[t] = s0[t, j]; prev = chain[t]
            o += n
        e_chain = ang(chain, Y)
        def grp(x):
            return dict(first10=float(np.nanmean(x[first])), rest=float(np.nanmean(x[~first])), all=float(np.nanmean(x)))
        out[sname] = dict(latency=dict(compensated=comp, stale=stale), modes=dict(mean_of_k=grp(e_mean), single=grp(e_single), best_of_k=grp(e_best), chained=grp(e_chain),
                          spread_deg=grp(spread), split45_frac=dict(first10=float(np.nanmean(spread[first] > 45)), rest=float(np.nanmean(spread[~first] > 45)))))
        print(sname, json.dumps(out[sname], ensure_ascii=False)[:900], flush=True)
    (ROOT / "docs/v5/egvla2_fm/latency_modes.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

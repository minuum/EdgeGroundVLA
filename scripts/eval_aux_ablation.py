#!/usr/bin/env python3
"""미래 특징/미래 bbox를 '학습할 때만' 보조 목표로 쓰는 ablation (추론 비용 0: 보조 헤드는 평가·배포에서 쓰지 않음).
FM-π0(층2·폭128) 액션 토큰 풀링 표현에서 (a) 3프레임 뒤 Florence-2 특징(코사인 손실) (b) 8프레임 뒤 특징 (c) 3프레임 뒤 bbox(cx,cy,면적, MSE)를 함께 예측.
분할: 5-fold + 안 본 위치, seed 0, 서브골 드롭아웃 30%, 120 epoch. 기준은 bold_ablation.json의 base(같은 설정)와 같은 분할끼리 비교.
출력: docs/v5/egvla2_fm/aux_ablation.json (분할 단위 저장·재개). 실행: .venv_lerobot/bin/python3 scripts/eval_aux_ablation.py  (SMOKE=1: epoch 2·분할 1개)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_runtime_ablation as RA  # noqa: E402
import eval_subgoal_ablation as AB  # noqa: E402
import train_egvla2_s1 as S1  # noqa: E402
import robovlm_nav.models.egvla2_fm_heads as HD  # noqa: E402
T = S1.T
DEV, BS = AB.DEV, AB.BS
SMOKE = bool(os.getenv("SMOKE"))
EPOCHS = 2 if SMOKE else 120
OUT = ROOT / "docs/v5/egvla2_fm" / ("aux_ablation_smoke.json" if SMOKE else "aux_ablation.json")
VARIANTS = {"aux_f3_w0.2": ("feat", 3, 0.2), "aux_f8_w0.5": ("feat", 8, 0.5), "aux_bbox3_w0.5": ("bbox", 3, 0.5)}


class AuxPiZero(HD.PiZeroFM):
    """PiZeroFM + 학습 전용 보조 헤드. forward 출력은 동일하고 풀링 표현을 self._pool에 남긴다."""
    def __init__(s, bdim, fdim, layers=2, d=HD.HID):
        super().__init__(bdim, fdim, layers, d)
        s.aux_f, s.aux_b = nn.Linear(d, fdim), nn.Linear(d, 3)

    def forward(s, bbox, feat, xt, t):
        prefix = torch.stack([s.sub(bbox), s.vis(feat)], 1)
        h, c = s.inp(xt) + s.pos, s.tm(HD.temb(t))
        for blk in s.blocks:
            h = blk(h, prefix, c)
        s._pool = h.mean(1)
        return s.out(h)


def future_targets(E, idx, k):
    f, b = [], []
    for i in idx:
        ft, win = E[i]["feat"], E[i]["bbox"]; n = len(ft); j = np.minimum(np.arange(n) + k, n - 1)
        f.append(ft[j]); b.append(win[:, 8:11][j])
    cat = lambda x: torch.tensor(np.concatenate(x)).float().to(DEV)
    ff = cat(f)
    return F.normalize(ff, dim=-1), cat(b)


def fit(tr, fut, kind, w, seed=0, drop=0.3):
    torch.manual_seed(seed)
    m = AuxPiZero(tr[0].shape[1], tr[1].shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2); n = len(tr[0])
    for _ in range(EPOCHS):
        m.train(); perm = torch.randperm(n, device=DEV)
        for s in range(0, n, BS):
            i = perm[s:s + BS]; bb = tr[0][i] * (torch.rand(len(i), 1, device=DEV) > drop).float(); a = tr[2][i]
            x0 = torch.randn_like(a); t = torch.rand(len(i), device=DEV)
            xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * a
            l = F.mse_loss(m(bb, tr[1][i], xt, t), a - x0)
            if kind == "feat":
                l = l + w * (1 - F.cosine_similarity(m.aux_f(m._pool), fut[0][i], dim=-1)).mean()
            else:
                l = l + w * F.mse_loss(m.aux_b(m._pool), fut[1][i])
            opt.zero_grad(); l.backward(); opt.step()
    m.eval()
    return m


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
        for vn, (kind, k, w) in VARIANTS.items():
            if vn in res[name]:
                continue
            fut = future_targets(E, tr_idx, k)
            m = fit(tr, fut, kind, w)
            S = RA.draw(m, te[0], te[1], 8, 10).mean(1)
            res[name][vn] = dict(first=RA.angle(S[:, 0], g_first), n1_vs_burst=RA.angle(S[:, 0], g_burst), params=int(sum(p.numel() for p in m.parameters())))
            OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
            print(name, vn, {a: round(b, 2) for a, b in res[name][vn].items() if not isinstance(b, int)}, flush=True)
        if SMOKE:
            break


if __name__ == "__main__":
    main()

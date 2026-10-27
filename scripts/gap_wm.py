#!/usr/bin/env python3
"""월드 모델식 보조 손실(Δbbox·Δfeat 예측) 1단계: plan_20261028_world_model_aux_head.md §2·§5.
조건: W0(보조 없음) W1(Δbbox) W2(Δfeat) W3(둘 다) W3k8(k=8) W3_prefix(prefix에서 예측). 분할: fold0·lopo_center·lopo_strong_right, 120 epoch, 서브골 드롭아웃 30%.
환경변수: SMOKE=1(epoch 2·분할 1개), ONLY=W0,W3(조건 선택), SEEDS=0,1,2, LAM=0.3
출력: docs/v5/egvla2_fm/gap_wm.json (조건·seed 단위 저장·재개; seed≠0이면 키 '조건@s시드'). 실행: .venv_lerobot/bin/python3 scripts/gap_wm.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import gap_suite as GS  # noqa: E402
import eval_subgoal_ablation as AB  # noqa: E402
import train_egvla2_s1 as S1  # noqa: E402
from robovlm_nav.models.egvla2_fm_wm import N_BB, PiZeroFMWM  # noqa: E402
T = S1.T
DEV, BS = GS.DEV, GS.BS
SMOKE = bool(os.getenv("SMOKE")); EPOCHS = 2 if SMOKE else 120
LAM = float(os.getenv("LAM", "0.3"))
SEEDS = [int(x) for x in os.getenv("SEEDS", "0").split(",")]
OUT = ROOT / "docs/v5/egvla2_fm" / ("gap_wm_smoke.json" if SMOKE else "gap_wm.json")
VARIANTS = {
    "W0": dict(k=4, bb=0, ft=0, src="h"),
    "W1": dict(k=4, bb=1, ft=0, src="h"),
    "W2": dict(k=4, bb=0, ft=1, src="h"),
    "W3": dict(k=4, bb=1, ft=1, src="h"),
    "W3k8": dict(k=8, bb=1, ft=1, src="h"),
    "W3_prefix": dict(k=4, bb=1, ft=1, src="prefix"),
}


def add_targets(E, k):
    """각 프레임 t에 대해 t+k와의 변화량. 에피소드 끝 k프레임은 마스크 0."""
    out = []
    for e in E:
        cur = e["bbox"][:, 8:8 + N_BB]; ft = e["feat"]; n = len(cur)
        dbb = np.zeros((n, N_BB), np.float32); dft = np.zeros_like(ft); m = np.zeros(n, np.float32)
        if n > k:
            dbb[:n - k] = cur[k:] - cur[:n - k]; dft[:n - k] = ft[k:] - ft[:n - k]; m[:n - k] = 1.0
        e2 = dict(e); e2.update(dbb=dbb, dft=dft, wm=m); out.append(e2)
    return out


def tens(E, idx):
    cat = lambda key: torch.tensor(np.concatenate([E[i][key] for i in idx])).to(DEV)
    return cat("bbox"), cat("feat"), cat("chunk"), cat("dbb"), cat("dft"), cat("wm")


def fit(tr, cfg, seed, drop=0.3, lam=LAM):
    torch.manual_seed(seed)
    bb_all, ft_all, ch, dbb, dft, wm = tr
    m = PiZeroFMWM(bb_all.shape[1], ft_all.shape[1], src=cfg["src"]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2); n = len(bb_all)
    sd = (dbb[wm > 0].std(0) + 1e-6)                                            # Δbbox 표준화(차원별 스케일 차이)
    use_aux = lam > 0 and (cfg["bb"] or cfg["ft"])
    for _ in range(EPOCHS):
        m.train(); perm = torch.randperm(n, device=DEV)
        for s in range(0, n, BS):
            i = perm[s:s + BS]; bb = bb_all[i] * (torch.rand(len(i), 1, device=DEV) > drop).float(); a = ch[i]
            x0 = torch.randn_like(a); t = torch.rand(len(i), device=DEV); xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * a
            if use_aux:
                out, aux = m(bb, ft_all[i], xt, t, a_true=a)
            else:
                out = m(bb, ft_all[i], xt, t)
            l = F.mse_loss(out, a - x0)
            if use_aux:
                w = wm[i]; den = w.sum().clamp(min=1.0); la = 0.0
                if cfg["bb"]:
                    la = la + (F.smooth_l1_loss(aux[:, :N_BB], dbb[i] / sd, reduction="none").mean(1) * w).sum() / den
                if cfg["ft"]:
                    la = la + ((1 - F.cosine_similarity(aux[:, N_BB:], dft[i], dim=-1)) * w).sum() / den
                l = l + lam * la
            opt.zero_grad(); l.backward(); opt.step()
    m.eval(); m.sd = sd; return m


@torch.no_grad()
def aux_diag(m, te):
    """안 본 에피소드에서 Δbbox 예측 오차(모델 vs Δ=0 복사 기준). 모델이 변화를 배웠는지 확인."""
    bb, ft, ch, dbb, dft, wm = te; msk = wm > 0
    _, aux = m(bb[msk], ft[msk], torch.zeros_like(ch[msk]), torch.zeros(int(msk.sum()), device=DEV), a_true=ch[msk])
    tgt = dbb[msk] / m.sd; pred = aux[:, :N_BB]
    return dict(dbb_l1_model=float((pred - tgt).abs().mean()), dbb_l1_zero=float(tgt.abs().mean()),
                dft_cos=float(F.cosine_similarity(aux[:, N_BB:], dft[msk], dim=-1).mean()))


def main():
    eps = S1.load_main(); cache = np.load(T.CACHE); E0 = AB.prep(eps, cache, 3); folds = S1.make_folds(eps)
    splits = {"fold0": folds[0], "lopo_center": [i for i, e in enumerate(eps) if e["cell"].startswith("center|")],
              "lopo_strong_right": [i for i, e in enumerate(eps) if e["cell"].startswith("strong_right|")]}
    only = os.getenv("ONLY"); names = only.split(",") if only else list(VARIANTS)
    live = GS.live_inputs(); res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for sp, te_idx in splits.items():
        res.setdefault(sp, {}); tr_idx = [i for i in range(len(E0)) if i not in te_idx]
        for vn in names:
            cfg = VARIANTS[vn]; E = add_targets(E0, cfg["k"]); tr, te = tens(E, tr_idx), tens(E, te_idx)
            for sd_ in SEEDS:
                key = vn if sd_ == 0 else f"{vn}@s{sd_}"
                if LAM != 0.3:
                    key += f"|lam{LAM}"
                if key in res[sp]:
                    continue
                m = fit(tr, cfg, sd_); r = GS.split_metrics(m, te[:3], live)
                if cfg["bb"] or cfg["ft"]:
                    r["aux"] = aux_diag(m, te)
                res[sp][key] = r; OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
                print(sp, key, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k not in ("live", "aux")}, "aux", r.get("aux"), flush=True)
        if SMOKE:
            break


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""E1(yaw-shift 증강 + 서보 재라벨)·E8(밝기 증강) 학습·평가. 라벨은 하드웨어 의미(E2): 회전→(0,0,±1), 이동→단위 방향, 정지→0.
E1 재라벨 규칙: 이동한 타깃 cx_new가 |cx-0.5|>0.30이면 회전(타깃 쪽), 아니면 기본 방향 + 서보 보정(ly += 2Δ, Δ=이동량/320)을 정규화. 기본이 회전·정지면 유지(단 |cx_new-0.5|>0.30이면 회전).
변형: E2+E8 / E2+E1 / E2+E1+E8+E5. 평가는 gap_suite와 같은 지표·분할. 시험 에피소드의 기준 프레임에서 만든 증강 샘플은 제외.
출력: docs/v5/egvla2_fm/gap_e1.json. 실행: .venv_lerobot/bin/python3 scripts/gap_e1.py (SMOKE=1: epoch 2·분할 1개)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import gap_suite as GS  # noqa: E402
import eval_subgoal_ablation as AB  # noqa: E402
import train_egvla2_s1 as S1  # noqa: E402
import robovlm_nav.models.egvla2_fm_heads as HD  # noqa: E402
T = S1.T
DEV, BS = GS.DEV, GS.BS
SMOKE = bool(os.getenv("SMOKE")); EPOCHS = 2 if SMOKE else 120
OUT = ROOT / "docs/v5/egvla2_fm" / ("gap_e1_smoke.json" if SMOKE else "gap_e1.json")
VARIANTS = {
    "E2+E8": dict(shift=False, bright=True, edge_w=1.0, thresh=0.30, gain=2.0),
    "E2+E1": dict(shift=True, bright=False, edge_w=1.0, thresh=0.30, gain=2.0),
    "E2+E1+E8+E5": dict(shift=True, bright=True, edge_w=3.0, thresh=0.30, gain=2.0),
    "E2+E1_tuned": dict(shift=True, bright=False, edge_w=1.0, thresh=0.22, gain=2.6),
    "E2+E1+E8+E5_tuned": dict(shift=True, bright=True, edge_w=3.0, thresh=0.22, gain=2.6)
}


def build_aug(E, eps_idx_by_name, thresh=0.30, gain=2.0):
    z = np.load(ROOT / "docs/v5/egvla2_fm/aug_features.npz", allow_pickle=True)
    base = [str(x) for x in z["base_names"]]; kinds = z["kind"]; bidx = z["base_idx"]; par = z["param"]; ab = z["bbox"]; af = z["feat"].astype(np.float32)
    rows = []
    for k in range(len(kinds)):
        n, t = base[int(bidx[k])].rsplit("|", 1); t = int(t); ei = eps_idx_by_name.get(n)
        if ei is None:
            continue
        e = E[ei]; win = e["bbox"][t].copy(); a0 = e["chunk"][t, 0].copy(); kind = str(kinds[k]); p = float(par[k])
        if kind == "shift":
            d = p / 320.0
            for s_ in (0, 4):
                win[s_] = np.clip(win[s_] + d, 0, 1)
            win[8:12] = ab[k]
            cx_new = float(ab[k][0]) if ab[k][3] > 0.5 else float(np.clip(e["bbox"][t][8] + d, 0, 1))
            a0t = torch.from_numpy(a0)[None]; hw = GS.hw_relabel(a0t)[0].numpy(); lab = hw.copy()
            if abs(cx_new - 0.5) > thresh:
                lab = np.array([0, 0, 1.0 if cx_new > 0.5 else -1.0], np.float32)
            elif np.hypot(hw[0], hw[1]) > 0.5:
                u = np.array([hw[0], hw[1] + gain * d]); u = u / max(np.linalg.norm(u), 1e-6); lab = np.array([u[0], u[1], 0.0], np.float32)
        else:
            win[8:12] = ab[k]; lab = GS.hw_relabel(torch.from_numpy(a0)[None])[0].numpy()
        rows.append((ei, kind, win.astype(np.float32), af[k], np.repeat(lab[None], HD.H, 0).astype(np.float32)))
    return rows


def fit(tr_real, aug, edge_w, seed=0, drop=0.3):
    torch.manual_seed(seed)
    bb = torch.cat([tr_real[0]] + ([torch.from_numpy(np.stack([r[2] for r in aug])).to(DEV)] if aug else []))
    ft = torch.cat([tr_real[1]] + ([torch.from_numpy(np.stack([r[3] for r in aug])).to(DEV)] if aug else []))
    y = torch.cat([GS.hw_relabel(tr_real[2])] + ([torch.from_numpy(np.stack([r[4] for r in aug])).to(DEV)] if aug else []))
    cx = bb[:, 8]; w = torch.where((cx < 0.25) | (cx > 0.75), torch.tensor(edge_w, device=DEV), torch.tensor(1.0, device=DEV))
    m = HD.PiZeroFM(bb.shape[1], ft.shape[1]).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2); n = len(bb)
    for _ in range(EPOCHS):
        m.train(); perm = torch.randperm(n, device=DEV)
        for s in range(0, n, BS):
            i = perm[s:s + BS]; b = bb[i] * (torch.rand(len(i), 1, device=DEV) > drop).float(); a = y[i]
            x0 = torch.randn_like(a); t = torch.rand(len(i), device=DEV); xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * a
            per = ((m(b, ft[i], xt, t) - (a - x0)) ** 2).mean((1, 2)); l = (per * w[i]).mean() / w[i].mean(); opt.zero_grad(); l.backward(); opt.step()
    m.eval(); return m


def main():
    eps = S1.load_main(); cache = np.load(T.CACHE); E = AB.prep(eps, cache, 3); folds = S1.make_folds(eps)
    by_name = {e["name"]: i for i, e in enumerate(eps)}
    splits = {"fold0": folds[0], "lopo_center": [i for i, e in enumerate(eps) if e["cell"].startswith("center|")], "lopo_strong_right": [i for i, e in enumerate(eps) if e["cell"].startswith("strong_right|")]}
    aug_cache = {}
    for vn, kw in VARIANTS.items():
        cfg = (kw["thresh"], kw["gain"])
        if cfg not in aug_cache:
            aug_cache[cfg] = build_aug(E, by_name, thresh=cfg[0], gain=cfg[1])
    live = GS.live_inputs(); res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, te_idx in splits.items():
        res.setdefault(name, {}); tr_idx = [i for i in range(len(E)) if i not in te_idx]; tr, te = T.to_t(E, tr_idx), T.to_t(E, te_idx); tes = set(te_idx)
        for vn, kw in VARIANTS.items():
            if vn in res[name]:
                continue
            allaug = aug_cache[(kw["thresh"], kw["gain"])]
            aug = [r for r in allaug if r[0] not in tes and ((r[1] == "shift" and kw["shift"]) or (r[1] == "bright" and kw["bright"]))]
            m = fit(tr, aug, kw["edge_w"]); res[name][vn] = GS.split_metrics(m, te, live); res[name][vn]["n_aug"] = len(aug)
            OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False)); r = res[name][vn]
            print(name, vn, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k != "live"}, "live", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r["live"].items()}, flush=True)
        if SMOKE:
            break


if __name__ == "__main__":
    main()

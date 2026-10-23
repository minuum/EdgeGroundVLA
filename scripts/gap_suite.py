#!/usr/bin/env python3
"""간극 지표 5종 + 저비용 후보(E2 하드웨어 의미 라벨, E5 가장자리 가중) 학습·평가. 분할: fold0, lopo_center, lopo_strong_right, seed 0, 서브골 드롭아웃 30%, 120 epoch.
지표(분할 모델로 측정): S(반사실 민감도), 모드 정확도(이동/회전/정지)와 회전 방향 부호 일치, 시작 15% 구간 회전 분류, 가장자리 프레임 방향 오차, 사람 방향 오차, 실기 프레임 453장에서 규칙 전문가와의 일치(M3).
규칙 전문가(M3): 면적≥0.25 정지, |cx−0.5|>0.30 회전(cx>0.5면 +az), 그 외 서보 방향(sy=clip(2(cx−0.5),±0.7), 각=atan2(sy, √(1−sy²))).
출력: docs/v5/egvla2_fm/gap_suite.json (분할 단위 저장·재개). 실행: .venv_lerobot/bin/python3 scripts/gap_suite.py  (SMOKE=1: epoch 2·분할 1개)
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
import robovlm_nav.models.egvla2_fm_heads as HD  # noqa: E402
from robovlm_nav.models.egvla2_fm_heads import sample_chunk  # noqa: E402
T = S1.T
DEV, BS = AB.DEV, AB.BS
SMOKE = bool(os.getenv("SMOKE"))
EPOCHS = 2 if SMOKE else 120
OUT = ROOT / "docs/v5/egvla2_fm" / ("gap_suite_smoke.json" if SMOKE else "gap_suite.json")
STOP_M = 0.2
VARIANTS = {"base": dict(), "E2_hw_labels": dict(e2=True), "E5_edge_w": dict(edge_w=3.0), "E2+E5": dict(e2=True, edge_w=3.0)}


def hw_relabel(a):
    """하드웨어 의미 라벨: 회전(|az|>0.1)→(0,0,±1), 이동→단위 방향, 정지→0. 크기·속력 단계 제거."""
    xy, az = a[..., :2], a[..., 2:]
    mag = xy.norm(dim=-1, keepdim=True); spin = az.abs() > 0.1
    u = xy / mag.clamp_min(1e-6)
    out_xy = torch.where(spin | (mag < 0.01), torch.zeros_like(xy), u)
    out_az = torch.where(spin, torch.sign(az), torch.zeros_like(az))
    return torch.cat([out_xy, out_az], -1)


def fit(tr, e2=False, edge_w=1.0, seed=0, drop=0.3):
    torch.manual_seed(seed)
    m = HD.PiZeroFM(tr[0].shape[1], tr[1].shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2); n = len(tr[0])
    y = hw_relabel(tr[2]) if e2 else tr[2]
    cx = tr[0][:, 8]; w = torch.where((cx < 0.25) | (cx > 0.75), torch.tensor(edge_w, device=DEV), torch.tensor(1.0, device=DEV))
    for _ in range(EPOCHS):
        m.train(); perm = torch.randperm(n, device=DEV)
        for s in range(0, n, BS):
            i = perm[s:s + BS]; bb = tr[0][i] * (torch.rand(len(i), 1, device=DEV) > drop).float(); a = y[i]
            x0 = torch.randn_like(a); t = torch.rand(len(i), device=DEV); xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * a
            per = ((m(bb, tr[1][i], xt, t) - (a - x0)) ** 2).mean((1, 2))
            l = (per * w[i]).mean() / w[i].mean(); opt.zero_grad(); l.backward(); opt.step()
    m.eval(); return m


@torch.no_grad()
def predict(m, bb, ft):
    torch.manual_seed(0)
    return sample_chunk(m, bb, ft, 8).mean(1)[:, 0, :].cpu().numpy()          # (N,3)


def mode_of(P, az_thr=0.2):
    mag = np.hypot(P[:, 0], P[:, 1])
    return np.where(mag >= STOP_M, 1, np.where(np.abs(P[:, 2]) >= az_thr, 2, 0))   # 0 정지, 1 이동, 2 회전


def gt_mode(G):
    mag = np.hypot(G[:, 0], G[:, 1])
    return np.where(np.abs(G[:, 2]) > 0.1, 2, np.where(mag >= 0.01, 1, 0))


def ang(x, y):
    return np.degrees(np.arctan2(y, x))


def dang(a, b):
    return np.abs((a - b + 180) % 360 - 180)


def live_inputs():
    z = np.load(ROOT / "docs/v5/egvla2_fm/live_feats.npz"); lb, lf, meta = z["live_b"], z["live_f"], z["meta"]; sess = [m.split("|")[0] for m in meta]; win = []
    for i in range(len(lb)):
        idx = [j for j in range(max(0, i - 2), i + 1) if sess[j] == sess[i]]; idx = [idx[0]] * (3 - len(idx)) + idx; win.append(np.concatenate([lb[j] for j in idx]))
    return lb, lf.astype(np.float32), np.array(win, np.float32)


def expert_live(lb):
    mode = np.full(len(lb), -1); angv = np.full(len(lb), np.nan); spin = np.zeros(len(lb)); sy = np.zeros(len(lb))
    for i, (cx, cy, area, has) in enumerate(lb):
        if has < 0.5:
            continue
        sy[i] = np.clip(2 * (cx - 0.5), -0.7, 0.7)
        if area >= 0.25:
            mode[i] = 0
        elif abs(cx - 0.5) > 0.30:
            mode[i] = 2; spin[i] = 1.0 if cx > 0.5 else -1.0
        else:
            mode[i] = 1; angv[i] = ang(np.sqrt(1 - sy[i] ** 2), sy[i])
    return mode, angv, spin, sy


def live_metrics(m, live):
    lb, lf, win = live; P = predict(m, torch.from_numpy(win).to(DEV), torch.from_numpy(lf).to(DEV))
    em, ea, es, sy = expert_live(lb); pm = mode_of(P); ok = em >= 0
    both_move = (em == 1) & (pm == 1); off = ok & (np.abs(lb[:, 0] - 0.5) > 0.1) & (pm == 1)
    return dict(mode_acc=float((pm[ok] == em[ok]).mean()), spin_dir_acc=float((np.sign(P[(em == 2) & (pm == 2), 2]) == es[(em == 2) & (pm == 2)]).mean()) if ((em == 2) & (pm == 2)).any() else None,
                move_angle_err=float(dang(ang(P[both_move, 0], P[both_move, 1]), ea[both_move]).mean()) if both_move.any() else None,
                corrective_sign=float((np.sign(P[off, 1]) == np.sign(sy[off])).mean()) if off.any() else None, n_ok=int(ok.sum()))


def split_metrics(m, te, live):
    bb, ft, ch = te; G = ch[:, 0, :].cpu().numpy(); P = predict(m, bb, ft)
    gm, pm = gt_mode(G), mode_of(P); cx = bb[:, 8].cpu().numpy()
    spin = gm == 2; mv = gm == 1; both = mv & (pm == 1)
    n = len(G); lens_first = np.zeros(n, bool)                                  # 시작 15%: 에피소드 경계 정보가 없어 cx 기반 대신 spin 프레임 전체로 대체
    def shift(dx):
        b = bb.clone()
        for k in (0, 4, 8):
            b[:, k] = (b[:, k] + dx).clamp(0, 1)
        return b
    ap = ang(*predict(m, shift(0.2), ft)[:, :2].T); am = ang(*predict(m, shift(-0.2), ft)[:, :2].T)
    S = float(np.mean(((ap - am + 180) % 360 - 180)[mv] / 2))
    edge = both & ((cx < 0.25) | (cx > 0.75))
    r = dict(S=S, mode_acc=float((pm == gm).mean()), spin_recall=float((pm[spin] == 2).mean()) if spin.any() else None,
             spin_sign_acc=float((np.sign(P[spin, 2]) == np.sign(G[spin, 2])).mean()) if spin.any() else None,
             dir_err=float(dang(ang(P[both, 0], P[both, 1]), ang(G[both, 0], G[both, 1])).mean()) if both.any() else None,
             edge_dir_err=float(dang(ang(P[edge, 0], P[edge, 1]), ang(G[edge, 0], G[edge, 1])).mean()) if edge.any() else None, n_edge=int(edge.sum()))
    r["live"] = live_metrics(m, live)
    return r


def main():
    eps = S1.load_main(); cache = np.load(T.CACHE); E = AB.prep(eps, cache, 3); folds = S1.make_folds(eps)
    splits = {"fold0": folds[0], "lopo_center": [i for i, e in enumerate(eps) if e["cell"].startswith("center|")], "lopo_strong_right": [i for i, e in enumerate(eps) if e["cell"].startswith("strong_right|")]}
    live = live_inputs(); res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, te_idx in splits.items():
        res.setdefault(name, {}); tr_idx = [i for i in range(len(E)) if i not in te_idx]; tr, te = T.to_t(E, tr_idx), T.to_t(E, te_idx)
        for vn, kw in VARIANTS.items():
            if vn in res[name]:
                continue
            m = fit(tr, **kw); res[name][vn] = split_metrics(m, te, live)
            OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
            r = res[name][vn]; print(name, vn, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k != "live"}, "live", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r["live"].items()}, flush=True)
        if SMOKE:
            break


if __name__ == "__main__":
    main()

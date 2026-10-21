#!/usr/bin/env python3
"""통제 실험: 같은 Florence-2 특징·bbox 창으로 이산 8-class 헤드(교차 엔트로피)를 학습해 연속 헤드(FM-π0, MLP 회귀)와 같은 분할·같은 지표로 비교.
분할: 5-fold(seed 0) + LOPO(목표 위치 5개). 지표(첫 스텝): 8-class 일치율, 방향 오차(정확 방향 대비), 이동 방향의 양자화 후 오차, GO/STOP F1, 종점 오차(속력 0.7 고정·같은 적분).
연속 헤드의 예측은 8방향으로 양자화한 값도 함께 평가(표현 vs 학습 방식 분리). 출력: docs/v5/egvla2_fm/discrete_head.json
실행: .venv_lerobot/bin/python3 scripts/eval_discrete_head.py"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB  # noqa: E402
import train_egvla2_s1 as S1  # noqa: E402
T, DEV = S1.T, S1.DEV
GO = 0.2
DIRS = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [0., -1.], [.7071, .7071], [.7071, -.7071], [0., 0.], [0., 0.]])   # STOP FWD L R FWD+L FWD+R ROT_L ROT_R


def cls_of(ex):
    """(N,3) exec 첫 스텝 → 8-class. 회전 우선 규칙(az>0.1이고 이동 거의 0)."""
    lx, ly, az = ex[:, 0], ex[:, 1], ex[:, 2]; m = np.hypot(lx, ly); c = np.zeros(len(ex), int)
    rot = (np.abs(az) > 0.1) & (m < 0.05); c[rot & (az > 0)] = 6; c[rot & (az <= 0)] = 7
    mov = (~rot) & (m >= GO)
    if mov.any():
        sc = (lx[mov, None] * DIRS[None, 1:6, 0].numpy() + ly[mov, None] * DIRS[None, 1:6, 1].numpy()) / m[mov, None]
        c[mov] = 1 + sc.argmax(1)
    return c


def snap_vec(v):
    """연속 (lx,ly) → 8방향 단위 벡터(정지·크기<0.2면 0)."""
    m = np.linalg.norm(v, axis=1); c = np.zeros(len(v), int); mov = m >= GO
    if mov.any():
        sc = (v[mov, None, 0] * DIRS[None, 1:6, 0].numpy() + v[mov, None, 1] * DIRS[None, 1:6, 1].numpy()) / m[mov, None]
        c[mov] = 1 + sc.argmax(1)
    return c, DIRS.numpy()[c]


class DHead(nn.Module):
    def __init__(s, bd, fd, hid=256):
        super().__init__(); s.net = nn.Sequential(nn.LayerNorm(fd + bd), nn.Linear(fd + bd, hid), nn.GELU(), nn.Dropout(0.1), nn.Linear(hid, 8))

    def forward(s, bb, ft):
        return s.net(torch.cat([ft, bb], -1))


def fit_cls(tr, y, epochs=60, seed=0):
    torch.manual_seed(seed); m = DHead(tr[0].shape[1], tr[1].shape[1]).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    yt = torch.tensor(y).to(DEV); n = len(yt)
    for _ in range(epochs):
        m.train(); perm = torch.randperm(n, device=DEV)
        for s in range(0, n, 256):
            i = perm[s:s + 256]; l = F.cross_entropy(m(tr[0][i], tr[1][i]), yt[i]); opt.zero_grad(); l.backward(); opt.step()
    m.eval(); return m


def path(vel, dt):
    return np.vstack([[0, 0], np.cumsum(vel * dt[:, None], 0)])


def metrics(vec_pred, cls_pred, ex, lens, dts):
    """vec_pred: (N,2) 예측 이동 방향 벡터(단위 또는 크기 포함), cls_pred: (N,) 8-class, ex: (N,3) 사람 exec."""
    cls_h = cls_of(ex); hm = np.linalg.norm(ex[:, :2], axis=1); pm = np.linalg.norm(vec_pred, axis=1)
    both = (hm >= GO) & (pm >= GO)
    cs = (vec_pred[both] * ex[both, :2]).sum(1) / (pm[both] * hm[both]); ang = float(np.degrees(np.arccos(np.clip(cs, -1, 1))).mean())
    pg, hg = pm >= GO, hm >= GO; tp, fp, fn = (pg & hg).sum(), (pg & ~hg).sum(), (~pg & hg).sum(); f1 = float(2 * tp / max(2 * tp + fp + fn, 1))
    o, fpe = 0, []
    for n, dt in zip(lens, dts):
        ph = ex[o:o + n, :2]; pp = vec_pred[o:o + n]; hh = np.linalg.norm(ph, axis=1, keepdims=True); pn = np.linalg.norm(pp, axis=1, keepdims=True)
        vh = np.where(hh > 1e-6, ph / np.maximum(hh, 1e-9) * (6.59 + 19.85 * hh), 0); vp = np.where(pn >= GO, pp / np.maximum(pn, 1e-9) * (6.59 + 19.85 * 0.7), 0)
        fpe.append(np.linalg.norm(path(vh, dt)[-1] - path(vp, dt)[-1])); o += n
    return dict(acc=float((cls_pred == cls_h).mean()), angle=ang, go_f1=f1, fpe=float(np.mean(fpe)))


def main():
    import h5py
    eps = S1.load_main(); cache = np.load(T.CACHE); E = AB.prep(eps, cache, 3); folds = S1.make_folds(eps)
    for e in E:
        with h5py.File(T.DATA / e["name"], "r") as f:
            ts = np.array(f["frame_timestamps"], float)
        e["dt"] = np.clip(np.diff(ts, append=ts[-1]), 0, 1.0); e["dt"][-1] = np.median(e["dt"][:-1])
    pos = sorted({e["cell"].split("|")[0] for e in eps})
    splits = {f"fold{i}": f for i, f in enumerate(folds)}; splits.update({f"lopo_{p}": [i for i, e in enumerate(eps) if e["cell"].startswith(p + "|")] for p in pos})
    OUT = ROOT / "docs/v5/egvla2_fm/discrete_head.json"; res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for sname, te_idx in splits.items():
        if sname in res:
            continue
        tr_idx = [i for i in range(len(E)) if i not in te_idx]; tr, te = T.to_t(E, tr_idx), T.to_t(E, te_idx)
        ex_tr = tr[2][:, 0, :].cpu().numpy(); ex_te = te[2][:, 0, :].cpu().numpy(); lens = [len(E[i]["bbox"]) for i in te_idx]; dts = [E[i]["dt"] for i in te_idx]
        r = {}
        dm = fit_cls(tr, cls_of(ex_tr))
        with torch.no_grad():
            lg = dm(te[0], te[1]); cp = lg.argmax(1).cpu().numpy()
        r["이산 8-class 헤드"] = metrics(DIRS.numpy()[cp], cp, ex_te, lens, dts)
        fm = AB.fit(S1.ARCHS["C_pi0_adaln"], tr, 120, 0, drop=0.3)
        with torch.no_grad():
            torch.manual_seed(0); v = T.sample(fm, te[0], te[1], k=8).mean(1)[:, 0, :2].cpu().numpy()
        sc, sv = snap_vec(v)
        r["FM-π0 (연속 그대로)"] = metrics(v, sc, ex_te, lens, dts); r["FM-π0 (8방향으로 양자화)"] = metrics(sv, sc, ex_te, lens, dts)
        rm = T.train_reg  # 회귀는 모델이 아니라 MSE만 반환하므로 직접 학습
        reg = T.MLPReg(tr[0].shape[1], tr[1].shape[1]).to(DEV); opt = torch.optim.AdamW(reg.parameters(), lr=1e-3, weight_decay=1e-2); torch.manual_seed(0)
        for _ in range(60):
            reg.train(); perm = torch.randperm(len(tr[0]), device=DEV)
            for s in range(0, len(perm), 256):
                i = perm[s:s + 256]; l = F.mse_loss(reg(tr[0][i], tr[1][i]), tr[2][i]); opt.zero_grad(); l.backward(); opt.step()
        reg.eval()
        with torch.no_grad():
            v = reg(te[0], te[1])[:, 0, :2].cpu().numpy()
        sc, sv = snap_vec(v)
        r["MLP 회귀 (연속 그대로)"] = metrics(v, sc, ex_te, lens, dts); r["MLP 회귀 (8방향으로 양자화)"] = metrics(sv, sc, ex_te, lens, dts)
        res[sname] = r; OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1)); print(sname, {k: round(v["angle"], 1) for k, v in r.items()}, flush=True)


if __name__ == "__main__":
    main()

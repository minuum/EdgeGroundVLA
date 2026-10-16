#!/usr/bin/env python3
"""FM 헤드 보조 평가(오프라인): ① 학습 곡선(train/hold-out flow loss) ② 액션 청크·시간 부드러움(저크) FM vs MLP 회귀
③ 샘플 불확실도(K샘플 표준편차)가 오차를 예측하는가 ④ 다봉 후보 프레임(이웃 장면의 사람 조작 방향이 갈림)에서 FM vs 회귀.
hold-out 2종: 5-fold 중 fold0, leave-one-position-out(strong_right). 헤드별 고정 epoch(A 240, B2 220, C 120, MLP 60), seed 0.
실행: .venv_lerobot/bin/python3 scripts/eval_fm_head_extras.py   출력: docs/v5/egvla2_fm/fm_head_extras.json
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
import train_egvla2_s1 as S1  # noqa: E402
T, DEV = S1.T, S1.DEV
EP = {"A_mlp_fm": 240, "B2_film_res_fm": 220, "C_pi0_adaln": 120}
BS = 256


def fit_flow(cls, tr, te, epochs, seed=0):
    torch.manual_seed(seed)
    m = cls(tr[0].shape[1], tr[1].shape[1]).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    n, curve = len(tr[0]), []
    for ep in range(1, epochs + 1):
        m.train(); perm = torch.randperm(n, device=DEV)
        for s in range(0, n, BS):
            i = perm[s:s + BS]; l = T.flow_loss(m, tr[0][i], tr[1][i], tr[2][i]); opt.zero_grad(); l.backward(); opt.step()
        if ep % 10 == 0 or ep == 1:
            m.eval()
            with torch.no_grad():
                g = torch.Generator(device=DEV).manual_seed(7)
                curve.append((ep, *[float(np.mean([T.flow_loss(m, d[0], d[1], d[2], g).item() for _ in range(3)])) for d in (tr, te)]))
    m.eval(); return m, curve


def fit_reg(tr, te, epochs=60, seed=0):
    torch.manual_seed(seed)
    m = T.MLPReg(tr[0].shape[1], tr[1].shape[1]).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    curve = []
    for ep in range(1, epochs + 1):
        m.train(); perm = torch.randperm(len(tr[0]), device=DEV)
        for s in range(0, len(perm), BS):
            i = perm[s:s + BS]; l = F.mse_loss(m(tr[0][i], tr[1][i]), tr[2][i]); opt.zero_grad(); l.backward(); opt.step()
        if ep % 10 == 0 or ep == 1:
            m.eval()
            with torch.no_grad():
                curve.append((ep, *[F.mse_loss(m(d[0], d[1]), d[2]).item() for d in (tr, te)]))
    m.eval(); return m, curve


def rank(x):
    return np.argsort(np.argsort(x)).astype(float)


def spear(a, b):
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def evaluate(models, tr, te, ep_lens):
    out = {}
    bb, ft, y = te
    ytr = tr[2][:, 0, :2].cpu().numpy(); ftr = torch.nn.functional.normalize(tr[1], dim=-1)
    fte = torch.nn.functional.normalize(ft, dim=-1)
    # 다봉 후보: 학습 프레임 중 최근접 10개의 사람 조작(이동 중) 방향 평균 결과 길이 R
    sim = fte @ ftr.T; nn_idx = sim.topk(10, dim=1).indices.cpu().numpy()
    R = np.zeros(len(te[0]))
    for i, ix in enumerate(nn_idx):
        v = ytr[ix]; mg = np.linalg.norm(v, axis=1); v = v[mg > 0.2] / mg[mg > 0.2, None]
        R[i] = np.linalg.norm(v.mean(0)) if len(v) >= 3 else 1.0
    multi = R < 0.8
    valid = np.concatenate([np.arange(s, e - T.H) for s, e in ep_lens if e - s > T.H])   # 청크 끝 0패딩 제외
    gt = y.cpu().numpy()
    def jit(p, idx):  # 같은 에피소드 안 연속 프레임 첫 스텝 이동 변화량
        d = []
        for s, e in ep_lens:
            ii = [i for i in idx if s < i < e]
            if ii: d.append(np.linalg.norm(p[ii, 0, :2] - p[[i - 1 for i in ii], 0, :2], axis=1))
        return float(np.concatenate(d).mean())
    def tv(p): return float(np.linalg.norm(np.diff(p[valid][..., :2], axis=1), axis=-1).mean())
    for name, m in models.items():
        with torch.no_grad():
            if name == "MLP_regression":
                mean = m(bb, ft).cpu().numpy(); single = mean; u = None
            else:
                Sm = T.sample(m, bb, ft); mean = Sm.mean(1).cpu().numpy(); single = Sm[:, 0].cpu().numpy(); u = Sm.std(1).mean((1, 2)).cpu().numpy()
        err = np.linalg.norm(mean[:, 0, :2] - gt[:, 0, :2], axis=1)
        r = dict(mse_xy=float(((mean[..., :2] - gt[..., :2]) ** 2).mean()), chunk_tv=tv(mean), chunk_tv_single=tv(single),
                 frame_jitter=jit(mean, valid), frame_jitter_single=jit(single, valid),
                 mse_xy_multimodal=float(((mean[multi][..., :2] - gt[multi][..., :2]) ** 2).mean()) if multi.any() else None,
                 mse_xy_other=float(((mean[~multi][..., :2] - gt[~multi][..., :2]) ** 2).mean()))
        if u is not None:
            q = np.quantile(u, [0.25, 0.75]); r.update(unc_err_spearman=spear(u, err), err_low_unc=float(err[u <= q[0]].mean()), err_high_unc=float(err[u >= q[1]].mean()),
                                                         unc_multimodal=float(u[multi].mean()) if multi.any() else None, unc_other=float(u[~multi].mean()))
        out[name] = r
    out["gt"] = dict(chunk_tv=tv(gt), frame_jitter=jit(gt, valid), multimodal_frac=float(multi.mean()), frames=int(len(te[0])))
    return out


def main():
    eps = S1.load_main(); folds = S1.make_folds(eps)
    lopo = {e["cell"].split("|")[0]: None for e in eps}
    splits = {"fold0": folds[0], "lopo_strong_right": [i for i, e in enumerate(eps) if e["cell"].startswith("strong_right|")]}
    res = {}
    for sname, te_idx in splits.items():
        tr_idx = [i for i in range(len(eps)) if i not in te_idx]
        tr, te = T.to_t(eps, tr_idx), T.to_t(eps, te_idx)
        lens, o = [], 0
        for i in te_idx:
            n = len(eps[i]["bbox"]); lens.append((o, o + n)); o += n
        models, curves = {}, {}
        for name, cls in S1.ARCHS.items():
            models[name], curves[name] = fit_flow(cls, tr, te, EP[name]); print(sname, name, "done", flush=True)
        models["MLP_regression"], curves["MLP_regression"] = fit_reg(tr, te); print(sname, "MLP done", flush=True)
        res[sname] = dict(n_train_ep=len(tr_idx), n_test_ep=len(te_idx), metrics=evaluate(models, tr, te, lens), curves=curves)
    (ROOT / "docs/v5/egvla2_fm/fm_head_extras.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print("saved")


if __name__ == "__main__":
    main()

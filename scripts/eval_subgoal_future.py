#!/usr/bin/env python3
"""π0.7식 '미래 서브골' 단순화 실험: 서브골 = 약 K프레임 뒤의 bbox. 현재 장면에서 미래 bbox를 생성(예측)하고 그걸로 액션 헤드를 조향.
변형: fut_oracle(정답 미래 bbox 입력 — 액션의 결과가 새어 들어가는 누설 상한, 비교용), fut_pred(생성기 예측 미래 bbox: 학습은 2-fold 교차 예측값, 테스트는 전체 학습 생성기).
기준: eval_subgoal_ablation.py의 full(창3, 서브골 현재 bbox만). 생성기 품질: 미래 bbox 예측 MSE vs '현재 bbox 유지' 기준선.
환경변수: SEEDS, K(기본 8), SMOKE=1(빠른 점검). 출력: docs/v5/egvla2_fm/subgoal_future.json
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB  # noqa: E402
import train_egvla2_s1 as S1  # noqa: E402
T, DEV = S1.T, S1.DEV
K = int(os.environ.get("K", 8)); SMOKE = os.environ.get("SMOKE") == "1"
HEADS = {h: (2 if SMOKE else ep) for h, ep in AB.HEADS.items()}
OUT = ROOT / "docs/v5/egvla2_fm/subgoal_future.json"


def build(eps, cache):
    out = []
    for e in eps:
        raw = cache[f"{e['name']}|bbox"].astype(np.float32); n = len(raw)
        fut = raw[np.minimum(np.arange(n) + K, n - 1)]
        e2 = dict(e); e2["bbox"] = AB.window(raw, 3); e2["fut"] = fut; out.append(e2)
    return out


def tens(E, idx):
    bb, ft, ch = T.to_t(E, idx)
    fu = torch.tensor(np.concatenate([E[i]["fut"] for i in idx])).to(DEV)
    return bb, ft, ch, fu


def fit_gen(bb, ft, fu, epochs=40, seed=0):
    torch.manual_seed(seed)
    g = torch.nn.Sequential(torch.nn.LayerNorm(ft.shape[1] + bb.shape[1]), torch.nn.Linear(ft.shape[1] + bb.shape[1], 256), torch.nn.GELU(), torch.nn.Linear(256, 4)).to(DEV)
    opt = torch.optim.AdamW(g.parameters(), lr=1e-3, weight_decay=1e-2); x = torch.cat([ft, bb], 1)
    for _ in range(epochs):
        perm = torch.randperm(len(x), device=DEV)
        for s in range(0, len(x), 256):
            i = perm[s:s + 256]; l = F.mse_loss(g(x[i]), fu[i]); opt.zero_grad(); l.backward(); opt.step()
    g.eval(); return g


def gen_pred(g, bb, ft):
    with torch.no_grad():
        return g(torch.cat([ft, bb], 1))


def run_split(name, E, te_idx, seed):
    tr_idx = [i for i in range(len(E)) if i not in te_idx]
    tr, te = tens(E, tr_idx), tens(E, te_idx)
    # 생성기: 전체 학습으로 하나 + 학습 에피소드 2-fold 교차 예측
    g_all = fit_gen(tr[0], tr[1], tr[3], 2 if SMOKE else 40, seed)
    half = [tr_idx[::2], tr_idx[1::2]]
    oof = torch.zeros_like(tr[3]); lens = [len(E[i]["bbox"]) for i in tr_idx]; off = np.concatenate([[0], np.cumsum(lens)])
    pos = {i: (off[k], off[k + 1]) for k, i in enumerate(tr_idx)}
    for a, b in ((0, 1), (1, 0)):
        ta = tens(E, half[a]); g = fit_gen(ta[0], ta[1], ta[3], 2 if SMOKE else 40, seed)
        for i in half[b]:
            s, e = pos[i]; oof[s:e] = gen_pred(g, tr[0][s:e], tr[1][s:e])
    pred_te = gen_pred(g_all, te[0], te[1])
    cur_te = te[0][:, -4:]   # 창의 마지막 4 = 현재 프레임 bbox
    gq = dict(gen_mse=float(F.mse_loss(pred_te[:, :3], te[3][:, :3]).item()), persist_mse=float(F.mse_loss(cur_te[:, :3], te[3][:, :3]).item()))
    res = {"generator": gq}
    for var, trf, tef in (("fut_oracle", tr[3], te[3]), ("fut_pred", oof, pred_te)):
        trx = (torch.cat([tr[0], trf], 1), tr[1], tr[2]); tex = (torch.cat([te[0], tef], 1), te[1], te[2])
        for h, ep in HEADS.items():
            m = AB.fit(S1.ARCHS[h], trx, ep, seed)
            r = S1.direction_metrics(m, tex); Sm = T.sample(m, tex[0], tex[1]).mean(1)
            r["mse_xy"] = float(((Sm[..., :2] - tex[2][..., :2]) ** 2).mean().item())
            res[f"{h}|{var}"] = {"normal": r}
            print(name, seed, h, var, round(r["angle_deg"], 1), flush=True)
    return res


def main():
    eps = S1.load_main(); cache = np.load(T.CACHE); E = build(eps, cache)
    folds = S1.make_folds(eps); pos = sorted({e["cell"].split("|")[0] for e in eps})
    splits = {f"fold{i}": f for i, f in enumerate(folds)}
    splits.update({f"lopo_{p}": [i for i, e in enumerate(eps) if e["cell"].startswith(p + "|")] for p in pos})
    if SMOKE:
        splits = {"fold0": folds[0]}
    allr = json.loads(OUT.read_text()) if (OUT.exists() and not SMOKE) else {}
    t0 = time.time()
    for seed in AB.SEEDS:
        for sname, te_idx in splits.items():
            key = f"{sname}|seed{seed}"
            if key in allr:
                continue
            allr[key] = run_split(sname, E, te_idx, seed)
            if not SMOKE:
                OUT.write_text(json.dumps(allr, indent=1, ensure_ascii=False))
            print(f"== {key} 완료 ({(time.time() - t0) / 60:.1f}분)", flush=True)


if __name__ == "__main__":
    main()

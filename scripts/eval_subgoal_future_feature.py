#!/usr/bin/env python3
"""'다음 이미지 예측' 형태 서브골의 잠재 공간 버전: 서브골 = K프레임(약 1초) 뒤 프레임의 Florence-2 풀링 특징(768-d, 이미지 임베딩).
fut_feat_oracle(정답 미래 특징 입력 — 로봇이 움직인 결과가 새어 들어가는 누설 상한), fut_feat_pred(현재 특징+bbox 창으로 미래 특징을 예측하는 생성기, 학습은 2-fold 교차 예측값, 테스트는 전체 학습 생성기).
생성기 품질: 미래 특징 예측 코사인 유사도 vs '현재 특징 그대로'(정지 가정) 기준선. 헤드: FM-π0(서브골 드롭아웃30%, 120 epoch). 분할: fold0 + LOPO 5개, seed 0. 출력: docs/v5/egvla2_fm/subgoal_future_feature.json
실행: .venv_lerobot/bin/python3 scripts/eval_subgoal_future_feature.py"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB  # noqa: E402
import train_egvla2_s1 as S1  # noqa: E402
T, DEV = S1.T, S1.DEV
K = 8
OUT = ROOT / "docs/v5/egvla2_fm/subgoal_future_feature.json"


def tens(E, idx):
    bb, ft, ch = T.to_t(E, idx)
    fu = torch.tensor(np.concatenate([E[i]["fut"] for i in idx])).to(DEV)
    return bb, ft, ch, fu


def fit_gen(bb, ft, fu, epochs=40):
    torch.manual_seed(0); din = ft.shape[1] + bb.shape[1]
    g = torch.nn.Sequential(torch.nn.LayerNorm(din), torch.nn.Linear(din, 512), torch.nn.GELU(), torch.nn.Linear(512, ft.shape[1])).to(DEV)
    opt = torch.optim.AdamW(g.parameters(), lr=1e-3, weight_decay=1e-2); x = torch.cat([ft, bb], 1)
    for _ in range(epochs):
        perm = torch.randperm(len(x), device=DEV)
        for s in range(0, len(x), 256):
            i = perm[s:s + 256]; l = F.mse_loss(g(x[i]), fu[i]); opt.zero_grad(); l.backward(); opt.step()
    g.eval(); return g


def gp(g, bb, ft):
    with torch.no_grad():
        return g(torch.cat([ft, bb], 1))


def main():
    eps = S1.load_main(); cache = np.load(T.CACHE); E = AB.prep(eps, cache, 3)
    for e in E:
        f = cache[f"{e['name']}|feat"].astype(np.float32); n = len(f); e["fut"] = f[np.minimum(np.arange(n) + K, n - 1)]
    folds = S1.make_folds(eps); pos = sorted({e["cell"].split("|")[0] for e in eps})
    splits = {"fold0": folds[0]}; splits.update({f"lopo_{p}": [i for i, e in enumerate(eps) if e["cell"].startswith(p + "|")] for p in pos})
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for sname, te_idx in splits.items():
        if sname in res:
            continue
        tr_idx = [i for i in range(len(E)) if i not in te_idx]; tr, te = tens(E, tr_idx), tens(E, te_idx)
        g_all = fit_gen(tr[0], tr[1], tr[3]); half = [tr_idx[::2], tr_idx[1::2]]; oof = torch.zeros_like(tr[3])
        lens = [len(E[i]["bbox"]) for i in tr_idx]; off = np.concatenate([[0], np.cumsum(lens)]); pos_ = {i: (off[k], off[k + 1]) for k, i in enumerate(tr_idx)}
        for a, b in ((0, 1), (1, 0)):
            ta = tens(E, half[a]); g = fit_gen(ta[0], ta[1], ta[3])
            for i in half[b]:
                s, e_ = pos_[i]; oof[s:e_] = gp(g, tr[0][s:e_], tr[1][s:e_])
        pred_te = gp(g_all, te[0], te[1])
        cos = lambda a, b: float(F.cosine_similarity(a, b, dim=1).mean().item())
        r = {"generator": dict(cos_pred=cos(pred_te, te[3]), cos_stay=cos(te[1], te[3]))}
        for var, trf, tef in (("fut_feat_oracle", tr[3], te[3]), ("fut_feat_pred", oof, pred_te)):
            trx = (torch.cat([tr[0], trf], 1), tr[1], tr[2]); tex = (torch.cat([te[0], tef], 1), te[1], te[2])
            m = AB.fit(S1.ARCHS["C_pi0_adaln"], trx, 120, 0, drop=0.3); d = S1.direction_metrics(m, tex); r[var] = dict(angle=d["angle_deg"], go_f1=d["go_f1"])
        res[sname] = r; OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1)); print(sname, {k: (round(v.get("angle", v.get("cos_pred", 0)), 3)) for k, v in r.items()}, flush=True)


if __name__ == "__main__":
    main()

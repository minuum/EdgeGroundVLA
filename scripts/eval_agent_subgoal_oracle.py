#!/usr/bin/env python3
"""에이전트 서브골 상한 실험: bbox 외에 '상위 에이전트가 주는 경로 의도'(좌커브/직진/우커브 원핫)를 두 번째 서브골 채널로 FM-π0에 넣는다(학습 시 이 채널만 30% 확률로 0).
평가 조건 3가지(같은 모델): 정답 의도 / 의도 없음(0) / 틀린 의도(무작위 다른 경로). 분할: fold0 + LOPO 5개, seed 0. 지표: 방향 오차 전체·에피소드 첫 10% 구간·나머지, 다중 정답 오차는 별도 스크립트.
출력: docs/v5/egvla2_fm/agent_subgoal.json. 실행: .venv_lerobot/bin/python3 scripts/eval_agent_subgoal_oracle.py"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB  # noqa: E402
import train_egvla2_s1 as S1  # noqa: E402
T, DEV = S1.T, S1.DEV
PATHS = ["left_curve", "straight", "right_curve"]
GO = 0.2


def onehot(cell):
    v = np.zeros(3, np.float32); v[PATHS.index(cell.split("|")[1])] = 1; return v


def fit2(tr, ep, seed, bdim_agent=3, p_bbox=0.3, p_agent=0.3):
    torch.manual_seed(seed)
    m = S1.ARCHS["C_pi0_adaln"](tr[0].shape[1], tr[1].shape[1]).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2); n = len(tr[0])
    for _ in range(ep):
        m.train(); perm = torch.randperm(n, device=DEV)
        for s in range(0, n, 256):
            i = perm[s:s + 256]; bb = tr[0][i].clone()
            bb[:, :-bdim_agent] *= (torch.rand(len(i), 1, device=DEV) > p_bbox).float(); bb[:, -bdim_agent:] *= (torch.rand(len(i), 1, device=DEV) > p_agent).float()
            l = T.flow_loss(m, bb, tr[1][i], tr[2][i]); opt.zero_grad(); l.backward(); opt.step()
    m.eval(); return m


def frame_err(m, te, agent_vec, lens):
    bb = te[0].clone(); bb[:, -3:] = agent_vec
    with torch.no_grad():
        torch.manual_seed(0); S = T.sample(m, bb, te[1], k=8).mean(1)[:, 0, :2].cpu().numpy()
    Y = te[2][:, 0, :2].cpu().numpy(); pm, ym = np.linalg.norm(S, axis=1), np.linalg.norm(Y, axis=1); both = (pm > GO) & (ym > GO)
    c = (S * Y).sum(1) / np.maximum(pm * ym, 1e-9); e = np.where(both, np.degrees(np.arccos(np.clip(c, -1, 1))), np.nan)
    first = np.concatenate([np.arange(n) < max(1, int(0.1 * n)) for n in lens])
    return dict(all=float(np.nanmean(e)), first10=float(np.nanmean(e[first])), rest=float(np.nanmean(e[~first])))


def main():
    eps = S1.load_main(); cache = np.load(T.CACHE); E = AB.prep(eps, cache, 3); folds = S1.make_folds(eps)
    for e, ep in zip(E, eps):
        e["bbox"] = np.concatenate([e["bbox"], np.tile(onehot(ep["cell"]), (len(e["bbox"]), 1))], 1)
    pos = sorted({e["cell"].split("|")[0] for e in eps})
    splits = {"fold0": folds[0]}; splits.update({f"lopo_{p}": [i for i, e in enumerate(eps) if e["cell"].startswith(p + "|")] for p in pos})
    OUT = ROOT / "docs/v5/egvla2_fm/agent_subgoal.json"; res = json.loads(OUT.read_text()) if OUT.exists() else {}
    rng = np.random.RandomState(3)
    for sname, te_idx in splits.items():
        if sname in res:
            continue
        tr_idx = [i for i in range(len(E)) if i not in te_idx]; tr, te = T.to_t(E, tr_idx), T.to_t(E, te_idx); lens = [len(E[i]["bbox"]) for i in te_idx]
        m = fit2(tr, 120, 0)
        true = te[0][:, -3:].clone(); none = torch.zeros_like(true)
        idx = true.argmax(1).cpu().numpy(); wrong = torch.zeros_like(true); w = np.array([rng.choice([k for k in range(3) if k != j]) for j in idx]); wrong[torch.arange(len(w)), torch.tensor(w)] = 1
        # 기준선: 같은 구조에서 에이전트 채널을 항상 0으로 학습한 모델은 별도 학습 없이 'none' 조건으로 근사(드롭아웃 30% 학습분)
        res[sname] = {"정답 의도": frame_err(m, te, true, lens), "의도 없음": frame_err(m, te, none, lens), "틀린 의도": frame_err(m, te, wrong.to(DEV), lens)}
        OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1)); print(sname, {k: round(v["all"], 1) for k, v in res[sname].items()}, flush=True)


if __name__ == "__main__":
    main()

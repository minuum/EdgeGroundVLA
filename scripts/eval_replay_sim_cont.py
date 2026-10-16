#!/usr/bin/env python3
"""리플레이 시뮬레이션(오프라인 궤적 적분): 안 본 에피소드의 기록 프레임마다 모델 예측을 내고, 예측 이동을 적분해 사람 궤적과 누적 위치 오차(FPE/ADE)를 비교.
변형: 사람(GT) / 연속-섀도(방향 연속, 속력 0.7 고정) / 연속-모델속력(s=clamp(|m|,0.4,1)) / 이산화(예측을 8방향으로 반올림, 속력 0.7 고정 = EGVLA1식 액션 표현) / GT 이산화(라벨 양자화만의 영향).
주의: 같은 지각(Florence-2)·같은 헤드에서 액션 표현만 바꾼 비교이며 EGVLA1 실제 모델(OWLv2+Kosmos-2)이 아니다. 영상은 원본이라 진짜 폐루프가 아니다(오차가 다음 프레임 입력을 바꾸지 않음).
실행: .venv_lerobot/bin/python3 scripts/eval_replay_sim_cont.py   출력: docs/v5/egvla2_fm/replay_sim_cont.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_fm_head_extras as X  # noqa: E402
import train_egvla2_s1 as S1  # noqa: E402
T = S1.T
V_A, V_B, RULE_S, STOP_M = 6.59, 19.85, 0.7, 0.2
DIRS = np.array([[1, 0], [0, 1], [0, -1], [.7071, .7071], [.7071, -.7071]])   # FORWARD LEFT RIGHT FWD+L FWD+R (lx 전진, ly 좌)


def speed(s):
    return V_A + V_B * s


def snap(v):
    m = np.linalg.norm(v, axis=-1, keepdims=True); u = v / np.maximum(m, 1e-9)
    k = (u @ DIRS.T).argmax(-1); out = DIRS[k].copy(); out[m[:, 0] < STOP_M] = 0
    return out


def integ(vel, dt):   # vel (n,2) cm/s [전진, 좌], dt (n,) -> 경로 (n+1,2)
    return np.vstack([[0, 0], np.cumsum(vel * dt[:, None], 0)])


def trajs(gt_exec, pred, dt):
    mag_gt = np.linalg.norm(gt_exec, axis=1)
    lv = np.where(mag_gt > 1e-6, mag_gt, 0)   # exec 크기 = 속력 단계(0.4/0.7/1.0), 회전 프레임은 0
    gt_v = np.where(lv[:, None] > 0, gt_exec / np.maximum(mag_gt[:, None], 1e-9) * (V_A + V_B * lv)[:, None], 0)
    m = np.linalg.norm(pred, axis=1, keepdims=True); u = pred / np.maximum(m, 1e-9); go = (m[:, 0] >= STOP_M)[:, None]
    s_model = np.clip(m, 0.4, 1.0)
    out = dict(GT=gt_v,
               cont_shadow=np.where(go, u * speed(RULE_S), 0),
               cont_model=np.where(go, u * speed(s_model), 0),
               disc=snap(pred) * speed(RULE_S),
               gt_disc=snap(gt_exec) * speed(RULE_S))
    return {k: integ(v, dt) for k, v in out.items()}


def main():
    eps = S1.load_main(); folds = S1.make_folds(eps)
    raw = {}
    import h5py
    for e in eps:
        with h5py.File(T.DATA / e["name"], "r") as f:
            t = np.array(f["frame_timestamps"], float)
        dt = np.clip(np.diff(t, append=t[-1]), 0, 1.0); dt[-1] = np.median(dt[:-1]); e["dt"] = dt
    splits = {"fold0": folds[0], "lopo_strong_right": [i for i, e in enumerate(eps) if e["cell"].startswith("strong_right|")]}
    res = {}
    for sname, te_idx in splits.items():
        tr_idx = [i for i in range(len(eps)) if i not in te_idx]
        tr, te = T.to_t(eps, tr_idx), T.to_t(eps, te_idx)
        models = {"C_pi0_adaln": X.fit_flow(S1.ARCHS["C_pi0_adaln"], tr, te, X.EP["C_pi0_adaln"])[0], "MLP_regression": X.fit_reg(tr, te)[0]}
        res[sname] = {}
        for name, m in models.items():
            with torch.no_grad():
                P = (T.sample(m, te[0], te[1]).mean(1) if name != "MLP_regression" else m(te[0], te[1]))[:, 0, :2].cpu().numpy()
            o, rows = 0, {k: [] for k in ("cont_shadow", "cont_model", "disc", "gt_disc")}
            for i in te_idx:
                n = len(eps[i]["bbox"]); gt_exec = eps[i]["chunk"][:, 0, :2]; dt = eps[i]["dt"]
                tj = trajs(gt_exec, P[o:o + n], dt); o += n
                g = tj["GT"]
                for k in rows:
                    p = tj[k]
                    rows[k].append((np.linalg.norm(p[-1] - g[-1]), np.linalg.norm(p - g, axis=1).mean(), abs(p[-1][1] - g[-1][1]), np.linalg.norm(np.diff(p, axis=0), axis=1).sum() - np.linalg.norm(np.diff(g, axis=0), axis=1).sum()))
            res[sname][name] = {k: dict(fpe_cm=float(np.mean([r[0] for r in v])), fpe_median=float(np.median([r[0] for r in v])), ade_cm=float(np.mean([r[1] for r in v])),
                                         lateral_err_cm=float(np.mean([r[2] for r in v])), path_len_diff_cm=float(np.mean([r[3] for r in v])), n=len(v)) for k, v in rows.items()}
            print(sname, name, {k: round(v["fpe_cm"], 1) for k, v in res[sname][name].items()}, flush=True)
    (ROOT / "docs/v5/egvla2_fm/replay_sim_cont.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

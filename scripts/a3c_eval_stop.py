#!/usr/bin/env python3
"""A3 vs A3C_S30(F30): 기록 세션 끝부분에서 정지(명령<0.2)하는지. 11/13 세션 전체 + T7 기록 세션(A3 452) 중 80개 무작위.
지표: 세션 마지막 3프레임 정지 비율, 세션 전체 이동 비율. 출력 docs/v5/egvla2_fm/a3c/eval_stop.json"""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB
import robovlm_nav.models.egvla2_fm_heads as HD
from robovlm_nav.serve.egvla2_fm_policy import EgVla2FMPolicy
C13 = np.load(ROOT / "runs/egvla2_fm/t13_obj_cache.npz"); C7 = np.load(ROOT / "runs/egvla2_fm/t7_session_cache.npz")
D = json.load(open(ROOT / "docs/v5/egvla2_fm/objtest_data.json"))
s13 = sorted({k.split("|")[0] for k in C13.files if k.endswith("|vis")})
s7 = sorted({k.split("|")[0] for k in C7.files if k.endswith("|vis")}); s7 = list(np.random.RandomState(0).choice(s7, 80, replace=False))
def get(C, sid):
    vis = C[f"{sid}|vis"].astype(np.float32); raw = C[f"{sid}|bbox"][:len(vis)].astype(np.float32); bw = AB.window(raw, 3); bw[raw[:, 3] < .5] = 0; return bw, vis
def run(mod, dev, bw, vis):
    o = []
    for b, v in zip(bw, vis):
        torch.manual_seed(0); o.append(HD.sample_chunk(mod, torch.tensor(b[None], device=dev), torch.tensor(v[None], device=dev), 8).mean(1)[0, 0, :2].cpu().numpy())
    return np.linalg.norm(np.array(o), axis=1)
R = {}
for name, p in (("A3", "deploy/C_pi0_servolabel1.0_drop30_e120.pt"), ("A3C_S30", "a3c/A3C_S30_s0.pt"), ("A3C_F30", "a3c/A3C_F30_s0.pt")):
    pol = EgVla2FMPolicy(ROOT / "runs/egvla2_fm" / p, k_samples=8); r = {}
    for tag, C, ss in (("11/13", C13, s13), ("T7(A3 기록 80)", C7, s7)):
        endstop, mv = [], []
        for s in ss:
            n = run(pol.model, pol.device, *get(C, s)); endstop.append(float((n[-3:] < .2).mean())); mv.append(float((n > .2).mean()))
        r[tag] = dict(마지막3프레임_정지=float(np.mean(endstop)), 전체_이동=float(np.mean(mv)), n=len(ss))
    R[name] = r; print(name, r, flush=True)
json.dump(R, open(ROOT / "docs/v5/egvla2_fm/a3c/eval_stop.json", "w"), ensure_ascii=False, indent=1)

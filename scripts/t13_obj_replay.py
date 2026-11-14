#!/usr/bin/env python3
"""11/13 물체 실험 65세션 열린고리 재생: V0·V4·V4R(시드 0~9)이 기록된 프레임·박스를 받으면 각 프레임에 어떤 명령을 냈을지.
V4·V4R은 (a) 그 세션의 실제 instruction (b) 'go to the gray basket' 두 문구로 돌린다. 출력 runs/egvla2_fm/t13_obj_replay.json"""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB
import robovlm_nav.models.egvla2_fm_heads as HD
DEV = AB.DEV; GO = 0.2
C = np.load(ROOT / "runs/egvla2_fm/t13_obj_cache.npz"); PH = list(C["_phrases"]); names = sorted({k.split("|")[0] for k in C.files if k.endswith("|vis")})
import h5py
INS = {n: str(h5py.File(ROOT / f"docs/inference_sessions/session_{n}.h5").attrs["instruction"]) for n in names}
KIND = {"V0": "EG4-00_V0_full", "V4": "EG4-04_V4_full", "V4R": "EG4-07_V4R_full"}
@torch.no_grad()
def step1(m, bb, ft): torch.manual_seed(0); return HD.sample_chunk(m, bb, ft, 8).mean(1)[:, 0, :2].cpu().numpy()
res = {n: {"ins": INS[n], "a3": C[f"{n}|act"][:len(C[f"{n}|vis"]), :2].round(3).tolist()} for n in names}
for v, base in KIND.items():
    for sd in range(10):
        sfx = "" if sd == 0 else f"_s{sd}"; ck = torch.load(ROOT / f"runs/egvla2_fm/textcond/{base}{sfx}.pt", map_location=DEV, weights_only=False)
        m = HD.PiZeroFM(12, ck["fdim"]).to(DEV); m.load_state_dict(ck["state"]); m.eval()
        for n in names:
            m_ = len(C[f"{n}|vis"]); raw = C[f"{n}|bbox"][:m_]; bbw = AB.window(raw.astype(np.float32), 3); bbw[raw[:, 3] < 0.5] = 0; bt = torch.tensor(bbw, device=DEV)
            vis = torch.tensor(C[f"{n}|vis"].astype(np.float32), device=DEV); txt = C[f"{n}|txt"].astype(np.float32)[:m_]
            for tag, k in (("ins", PH.index(INS[n])), ("gray", 0)):
                if v == "V0" and tag == "gray": continue
                f = vis if v == "V0" else torch.cat([vis, torch.tensor(txt[:, k], device=DEV)], -1)
                res[n].setdefault(f"{v}_{tag}", []).append(step1(m, bt, f).round(3).tolist())
    print(v, flush=True)
json.dump(res, open(ROOT / "runs/egvla2_fm/t13_obj_replay.json", "w")); print("saved")

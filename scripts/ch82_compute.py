#!/usr/bin/env python3
"""CH82용 시뮬레이션(재생) 계산: 11/13 65세션 프레임에서 a3·a3-s1·a3-s2·앙상블(3) 헤드의 첫 명령을 구해 박스 면적과 함께 저장.
출력 docs/v5/egvla2_fm/a3c/ch82_sim.json = {sid: {obj,pos,res,area:[...],cmd:{head:[[lx,ly],...]}}}. 열린 루프 재생이며 실기 결과를 예측하지 않는다."""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB
import robovlm_nav.models.egvla2_fm_heads as HD
from robovlm_nav.serve.egvla2_fm_policy import EgVla2FMPolicy
C = np.load(ROOT / "runs/egvla2_fm/t13_obj_cache.npz"); D = json.load(open(ROOT / "docs/v5/egvla2_fm/objtest_data.json"))
H = {"a3": ROOT / "runs/egvla2_fm/deploy/C_pi0_servolabel1.0_drop30_e120.pt", "a3-s1": ROOT / "runs/egvla2_fm/a3c/A3C_D30_s1.pt", "a3-s2": ROOT / "runs/egvla2_fm/a3c/A3C_D30_s2.pt"}
pols = {k: EgVla2FMPolicy(v, k_samples=8) for k, v in H.items()}
def run(pol, bw, vis):
    o = []
    for b, v in zip(bw, vis):
        torch.manual_seed(0); o.append(HD.sample_chunk(pol.model, torch.tensor(b[None], device=pol.device), torch.tensor(v[None], device=pol.device), 8).mean(1)[0, 0, :2].cpu().numpy())
    return np.array(o)
OUT = {}
for x in D:
    sid = x["sid"]; vis = C[f"{sid}|vis"].astype(np.float32); raw = C[f"{sid}|bbox"][:len(vis)].astype(np.float32); bw = AB.window(raw, 3); bw[raw[:, 3] < .5] = 0
    cmd = {k: run(p, bw, vis) for k, p in pols.items()}; cmd["a3-ens3"] = np.mean([cmd[k] for k in H], 0)
    OUT[sid] = dict(obj=x["obj"], pos=x["pos"], res=x["res"], n=len(vis), area=[round(float(a), 4) for a in raw[:, 2]], has=[int(h > .5) for h in raw[:, 3]], cmd={k: np.round(v, 3).tolist() for k, v in cmd.items()})
json.dump(OUT, open(ROOT / "docs/v5/egvla2_fm/a3c/ch82_sim.json", "w"), ensure_ascii=False); print("saved", len(OUT))

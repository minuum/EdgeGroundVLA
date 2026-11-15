#!/usr/bin/env python3
"""배포 폴더의 기존 체크포인트들이 11/13 가운데 그룹에서 어떻게 움직이는지(동환 가설: 박스 드롭아웃 30%가 원인인가). 이동 비율=첫 명령 크기>0.2 프레임 비율.
출력 docs/v5/egvla2_fm/a3c/eval_deploy_heads.json"""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB
import robovlm_nav.models.egvla2_fm_heads as HD
from robovlm_nav.serve.egvla2_fm_policy import EgVla2FMPolicy
C = np.load(ROOT / "runs/egvla2_fm/t13_obj_cache.npz"); D = json.load(open(ROOT / "docs/v5/egvla2_fm/objtest_data.json"))
GRP = {"바구니·가운데": ("가운데(G3)", "바구니"), "의자·가운데": ("가운데(G3)", "의자"), "화분·가운데": ("가운데(G3)", "화분"), "의자·오른쪽": ("오른쪽(G4로 추정)", "의자"), "책상·왼쪽": ("왼쪽(G2로 추정)", "작은 책상")}
SS = {k: [x["sid"] for x in D if (x["pos"], x["obj"]) == v][:5] for k, v in GRP.items()}
def get(s):
    vis = C[f"{s}|vis"].astype(np.float32); raw = C[f"{s}|bbox"][:len(vis)].astype(np.float32); bw = AB.window(raw, 3); bw[raw[:, 3] < .5] = 0; return bw, vis
H = {"A3(서보1.0·드롭30)": "C_pi0_servolabel1.0_drop30_e120.pt", "서보0.5·드롭30": "C_pi0_servolabel0.5_drop30_e120.pt", "서보없음·드롭30": "C_pi0_adaln_drop30_e120.pt", "서보없음·드롭0": "C_pi0_adaln_e120.pt"}
R = {}
for n, f in H.items():
    pol = EgVla2FMPolicy(ROOT / "runs/egvla2_fm/deploy" / f, k_samples=8); r = {}
    for k, ss in SS.items():
        a = []
        for s in ss:
            bw, vis = get(s); o = []
            for b, v in zip(bw, vis):
                torch.manual_seed(0); o.append(np.linalg.norm(HD.sample_chunk(pol.model, torch.tensor(b[None], device=pol.device), torch.tensor(v[None], device=pol.device), 8).mean(1)[0, 0, :2].cpu().numpy()))
            a.append(float((np.array(o) > .2).mean()))
        r[k] = round(float(np.mean(a)), 2)
    R[n] = r; print(n, r, flush=True)
json.dump(R, open(ROOT / "docs/v5/egvla2_fm/a3c/eval_deploy_heads.json", "w"), ensure_ascii=False, indent=1)

#!/usr/bin/env python3
"""S2-b: A3 헤드의 '가운데에서 0 명령'이 박스 때문인지 이미지 특징 때문인지 교차 시험.
11/13 세션 캐시(runs/egvla2_fm/t13_obj_cache.npz)의 (vis, 지시어 기준 기록 박스)를 세션 사이에서 맞바꿔 헤드에 넣는다. 박스 창은 학습과 같은 3프레임 창.
출력 docs/v5/egvla2_fm/a3c/s2_swap.json"""
import json, sys, collections
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB
import robovlm_nav.models.egvla2_fm_heads as HD
from robovlm_nav.serve.egvla2_fm_policy import EgVla2FMPolicy
C = np.load(ROOT / "runs/egvla2_fm/t13_obj_cache.npz"); D = json.load(open(ROOT / "docs/v5/egvla2_fm/objtest_data.json"))
pol = EgVla2FMPolicy(ROOT / "runs/egvla2_fm/deploy/C_pi0_servolabel1.0_drop30_e120.pt", k_samples=8); mod = pol.model; mod.eval()
def sess(group):
    return [x["sid"] for x in D if (x["pos"], x["obj"]) == group]
G = {"바구니·가운데": ("가운데(G3)", "바구니"), "의자·가운데": ("가운데(G3)", "의자"), "화분·가운데": ("가운데(G3)", "화분"), "의자·오른쪽": ("오른쪽(G4로 추정)", "의자"), "책상·왼쪽": ("왼쪽(G2로 추정)", "작은 책상"), "바구니·왼쪽": ("왼쪽(G2로 추정)", "바구니")}
def get(sid):
    vis = C[f"{sid}|vis"].astype(np.float32); raw = C[f"{sid}|bbox"][:len(vis)].astype(np.float32)
    bw = AB.window(raw, 3); bw[raw[:, 3] < .5] = 0; return vis, bw, raw
def run(bw, vis):
    n = []
    for b, v in zip(bw, vis):
        torch.manual_seed(0)
        ch = HD.sample_chunk(mod, torch.tensor(b[None], device=pol.device), torch.tensor(v[None], device=pol.device), 8).mean(1)[0, 0, :2].cpu().numpy(); n.append(float(np.linalg.norm(ch)))
    n = np.array(n); return float((n > .2).mean())
S = {k: sess(v) for k, v in G.items()}
ref_vis = {k: [get(s)[0] for s in v[:3]] for k, v in S.items()}; ref_box = {k: [get(s)[1] for s in v[:3]] for k, v in S.items()}
R = {}
for k, sids in S.items():
    a = []
    for s in sids[:5]:
        vis, bw, raw = get(s); a.append(run(bw, vis))
    R[k] = dict(real_box_real_vis=float(np.mean(a)), n=len(sids[:5]))
print("실제 박스+실제 이미지 특징 이동비율", {k: round(v["real_box_real_vis"], 2) for k, v in R.items()}, flush=True)
# 교차: 의자·가운데 이미지 특징 + 다른 그룹 박스(길이 맞춰 자름)
X = {}
for vk in ("의자·가운데", "바구니·가운데", "화분·가운데"):
    for bk in ("바구니·가운데", "의자·가운데", "화분·가운데"):
        r = []
        for vv in ref_vis[vk]:
            for bb in ref_box[bk]:
                n = min(len(vv), len(bb)); r.append(run(bb[:n], vv[:n]))
        X[f"vis={vk}|box={bk}"] = float(np.mean(r))
# 박스 없음(0 창)
Z = {k: float(np.mean([run(np.zeros_like(ref_box[k][i]), ref_vis[k][i]) for i in range(len(ref_vis[k]))])) for k in ("의자·가운데", "바구니·가운데", "화분·가운데")}
json.dump(dict(real=R, cross=X, zero_box=Z), open(ROOT / "docs/v5/egvla2_fm/a3c/s2_swap.json", "w"), ensure_ascii=False, indent=1)
print("교차(행=이미지 특징, 열=박스)"); [print(k, round(v, 2)) for k, v in X.items()]
print("박스 0창 + 실제 이미지:", {k: round(v, 2) for k, v in Z.items()})

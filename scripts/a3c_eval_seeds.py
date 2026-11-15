#!/usr/bin/env python3
"""박스 가림 비율(0/10/20/30%)별 시드 편차: A3C_D*_s{0,1,2}.pt(+A3=D30 시드 0 배포본)를 11/13 가운데 그룹·세션 끝 정지·박스 반전으로 비교.
출력 docs/v5/egvla2_fm/a3c/eval_seeds.json (헤드별 값과 가림 비율별 평균·최소·최대)"""
import json, sys, glob, re
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB
import robovlm_nav.models.egvla2_fm_heads as HD
from robovlm_nav.serve.egvla2_fm_policy import EgVla2FMPolicy
C13 = np.load(ROOT / "runs/egvla2_fm/t13_obj_cache.npz"); D = json.load(open(ROOT / "docs/v5/egvla2_fm/objtest_data.json"))
GRP = {"바구니·가운데": ("가운데(G3)", "바구니"), "의자·가운데": ("가운데(G3)", "의자"), "화분·가운데": ("가운데(G3)", "화분"), "의자·오른쪽": ("오른쪽(G4로 추정)", "의자"), "책상·왼쪽": ("왼쪽(G2로 추정)", "작은 책상"), "바구니·왼쪽": ("왼쪽(G2로 추정)", "바구니")}
SS = {k: [x["sid"] for x in D if (x["pos"], x["obj"]) == v] for k, v in GRP.items()}; s13 = [x["sid"] for x in D]
def get(sid):
    vis = C13[f"{sid}|vis"].astype(np.float32); raw = C13[f"{sid}|bbox"][:len(vis)].astype(np.float32); bw = AB.window(raw, 3); bw[raw[:, 3] < .5] = 0; return bw, vis
def run(mod, dev, bw, vis):
    o = []
    for b, v in zip(bw, vis):
        torch.manual_seed(0); o.append(HD.sample_chunk(mod, torch.tensor(b[None], device=dev), torch.tensor(v[None], device=dev), 8).mean(1)[0, 0, :2].cpu().numpy())
    return np.array(o)
mv = lambda a: float((np.linalg.norm(a, axis=1) > .2).mean())
heads = {"D30_s0(=A3 배포본)": ROOT / "runs/egvla2_fm/deploy/C_pi0_servolabel1.0_drop30_e120.pt"}
for p in sorted(glob.glob(str(ROOT / "runs/egvla2_fm/a3c/A3C_D*_s*.pt"))):
    m = re.search(r"A3C_D(\d\d)_s(\d)", p); heads[f"D{m.group(1)}_s{m.group(2)}"] = Path(p)
R = {}
for name, path in heads.items():
    pol = EgVla2FMPolicy(path, k_samples=8); mod, dev = pol.model, pol.device
    g = {k: round(float(np.mean([mv(run(mod, dev, *get(s))) for s in ss[:5]])), 2) for k, ss in SS.items()}
    e = [float((np.linalg.norm(run(mod, dev, *get(s)), axis=1)[-3:] < .2).mean()) for s in s13]
    R[name] = dict(그룹별_이동비율=g, 세션끝_정지_11_13=round(float(np.mean(e)), 2)); print(name, json.dumps(R[name], ensure_ascii=False), flush=True)
agg = {}
for dr in ("00", "10", "20", "30"):
    ks = [k for k in R if k.startswith(f"D{dr}_")]
    if not ks: continue
    agg[f"드롭{int(dr)}%"] = {"n_시드": len(ks), "의자·가운데_이동": [round(float(np.mean([R[k]["그룹별_이동비율"]["의자·가운데"] for k in ks])), 2), min(R[k]["그룹별_이동비율"]["의자·가운데"] for k in ks), max(R[k]["그룹별_이동비율"]["의자·가운데"] for k in ks)],
                              "바구니·가운데_이동_평균": round(float(np.mean([R[k]["그룹별_이동비율"]["바구니·가운데"] for k in ks])), 2), "세션끝_정지_평균": round(float(np.mean([R[k]["세션끝_정지_11_13"] for k in ks])), 2)}
json.dump(dict(heads=R, agg=agg), open(ROOT / "docs/v5/egvla2_fm/a3c/eval_seeds.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(agg, ensure_ascii=False, indent=1))

#!/usr/bin/env python3
"""
일반화 매트릭스 (논문 P1 자산) — train{V5,V6,V5+V6} × test{V5,V6}, 같은 PG448 그라운더.

bbox confound 제거: V5 vis 캐시는 그대로 두되(vis는 그라운더 무관, stage1_v2 동일),
bbox만 pg448 주석으로 교체해 V6(pg448)와 동일 그라운더 조건으로 맞춤.
각 셋 85/15 분할 → train-half로 학습, val-half로 closed-loop 평가(누수 없음). mlp, 3-seed 평균.
"""
import sys, json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.train_exp73_trackA_heads import (MLPActionHead, train_one, build_windows,
                                               CACHE_V6, SPLIT_SEED, VAL_RATIO)
from scripts.sim.evaluate_closed_loop_exp73 import build_episode_windows, eval_episode

CACHE_V5 = ROOT / "docs/v5/closed_loop_eval/exp71_vis_cache_colorfixed.pt"
V5_PG448 = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_pg448_cx.json"

def load_v5_pg448():
    eps = torch.load(str(CACHE_V5), weights_only=False)
    ann = {Path(e["episode"]).stem: e for e in json.load(open(V5_PG448))}
    out = []
    for e in eps:
        src = ann.get(e["stem"])
        if src is None: continue
        frames = [fr for fr in src["frames"] if fr.get("gt_class") is not None]
        if len(frames) != len(e["vis"]): continue  # 길이 불일치 스킵
        e = dict(e)
        e["bboxes"] = [(fr.get("cx_det",0.5),fr.get("cy_det",0.5),fr.get("area_det",0.05),
                        float(fr.get("has_bbox",fr.get("detected",False)))) for fr in frames]
        out.append(e)
    return out

def split(eps):
    rng = np.random.default_rng(SPLIT_SEED); idx=list(range(len(eps))); rng.shuffle(idx)
    nv=max(1,int(len(eps)*VAL_RATIO)); return [eps[i] for i in idx[nv:]],[eps[i] for i in idx[:nv]]

def cl_success(model, eps):
    r=[eval_episode(e,model,"cpu") for e in eps]
    return np.mean([x["success"] for x in r])*100

def main():
    v6=torch.load(str(CACHE_V6),weights_only=False); v6=[e for e in v6 if e.get("acts") is not None]
    v5=load_v5_pg448()
    print(f"V5(pg448)={len(v5)}ep  V6={len(v6)}ep",flush=True)
    v5tr,v5va=split(v5); v6tr,v6va=split(v6)
    Xva5,yva5,_=build_windows(v5va); Xva6,yva6,_=build_windows(v6va)

    trainsets={"V5":v5tr,"V6":v6tr,"V5+V6":v5tr+v6tr}
    mat={}
    for name,tr in trainsets.items():
        Xtr,ytr,_=build_windows(tr)
        s5=[];s6=[]
        for seed in [0,1,2]:
            _,st,_=train_one(MLPActionHead,Xtr,ytr,Xva6,yva6,seed,epochs=300)  # val monitor=V6
            m=MLPActionHead(); m.load_state_dict(st); m.eval()
            s5.append(cl_success(m,v5va)); s6.append(cl_success(m,v6va))
        mat[name]=(np.mean(s5),np.std(s5),np.mean(s6),np.std(s6))
        print(f"[{name}] →V5 {np.mean(s5):.1f}±{np.std(s5):.1f}  →V6 {np.mean(s6):.1f}±{np.std(s6):.1f}",flush=True)

    print("\n=== 일반화 매트릭스 (closed-loop Success%, 3-seed 평균) ===")
    hdr = "train\\test"
    print(f"{hdr:10s} {'→V5(쉬움)':>14s} {'→V6(어려움)':>14s}")
    for k,(m5,s5,m6,s6) in mat.items():
        print(f"{k:10s} {m5:6.1f}±{s5:.1f}     {m6:6.1f}±{s6:.1f}")
    out=ROOT/"docs/v5/closed_loop_eval/exp73_generalization_matrix.json"
    out.write_text(json.dumps({k:{"to_V5_mean":v[0],"to_V5_std":v[1],"to_V6_mean":v[2],"to_V6_std":v[3]} for k,v in mat.items()},indent=2))
    print("저장 →",out)

if __name__=="__main__": main()

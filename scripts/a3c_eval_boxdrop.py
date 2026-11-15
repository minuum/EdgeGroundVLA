#!/usr/bin/env python3
"""박스 가림 비율을 낮춘 A3C(D00·D10·D20)와 A3(0.3)를 같은 기준으로 비교(오프라인 재생, 성공률 아님).
지표: (a) 11/13 그룹별 이동 비율(첫 명령>0.2) (b) 의자 이미지+바구니 박스 교차 (c) 세션 끝 3프레임 정지(11/13 65세션, T7 A3 기록 80세션)
(d) 박스를 없앴을 때(창 전체 0) 바구니 세션 이동 비율 (e) 박스 좌우 반전 시 방향 부호가 뒤집히는 비율(박스를 따라가는 정도, 이동 프레임 기준)
(f) A3 헤드와의 방향 차이 중앙값(바구니 세션). 출력 docs/v5/egvla2_fm/a3c/eval_boxdrop.json"""
import json, sys, glob
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB
import robovlm_nav.models.egvla2_fm_heads as HD
from robovlm_nav.serve.egvla2_fm_policy import EgVla2FMPolicy
C13 = np.load(ROOT / "runs/egvla2_fm/t13_obj_cache.npz"); C7 = np.load(ROOT / "runs/egvla2_fm/t7_session_cache.npz"); D = json.load(open(ROOT / "docs/v5/egvla2_fm/objtest_data.json"))
GRP = {"바구니·가운데": ("가운데(G3)", "바구니"), "의자·가운데": ("가운데(G3)", "의자"), "화분·가운데": ("가운데(G3)", "화분"), "의자·오른쪽": ("오른쪽(G4로 추정)", "의자"), "책상·왼쪽": ("왼쪽(G2로 추정)", "작은 책상"), "바구니·왼쪽": ("왼쪽(G2로 추정)", "바구니")}
SS = {k: [x["sid"] for x in D if (x["pos"], x["obj"]) == v] for k, v in GRP.items()}
s13 = [x["sid"] for x in D]; s7 = sorted({k.split("|")[0] for k in C7.files if k.endswith("|vis")}); s7 = list(np.random.RandomState(0).choice(s7, 80, replace=False))
def get(C, sid):
    vis = C[f"{sid}|vis"].astype(np.float32); raw = C[f"{sid}|bbox"][:len(vis)].astype(np.float32); bw = AB.window(raw, 3); bw[raw[:, 3] < .5] = 0; return bw, vis
def run(mod, dev, bw, vis):
    o = []
    for b, v in zip(bw, vis):
        torch.manual_seed(0); o.append(HD.sample_chunk(mod, torch.tensor(b[None], device=dev), torch.tensor(v[None], device=dev), 8).mean(1)[0, 0, :2].cpu().numpy())
    return np.array(o)
mv = lambda a: float((np.linalg.norm(a, axis=1) > .2).mean())
heads = {"A3(드롭30)": ROOT / "runs/egvla2_fm/deploy/C_pi0_servolabel1.0_drop30_e120.pt"}
for k in ("D20", "D10", "D00"): heads[f"A3C-{k}(드롭{int(k[1:])})"] = ROOT / f"runs/egvla2_fm/a3c/A3C_{k}_s0.pt"
def flip(bw): b = bw.copy(); b[:, [0, 4, 8]] = np.where(bw[:, [3, 7, 11]] > .5, 1 - bw[:, [0, 4, 8]], bw[:, [0, 4, 8]]); return b
R = {}; REF = {}
for name, path in heads.items():
    pol = EgVla2FMPolicy(path, k_samples=8); mod, dev = pol.model, pol.device; r = {}
    r["그룹별_이동비율"] = {k: round(float(np.mean([mv(run(mod, dev, *get(C13, s)[::1])) for s in ss[:5]])), 2) for k, ss in SS.items()}
    vc = [get(C13, s)[1] for s in SS["의자·가운데"][:3]]; bb = [get(C13, s)[0] for s in SS["바구니·가운데"][:3]]
    r["의자이미지+바구니박스"] = round(float(np.mean([mv(run(mod, dev, b[:min(len(v), len(b))], v[:min(len(v), len(b))])) for v in vc for b in bb])), 2)
    e13, e7 = [], []
    for s in s13: n = np.linalg.norm(run(mod, dev, *get(C13, s)), axis=1); e13.append(float((n[-3:] < .2).mean()))
    for s in s7: n = np.linalg.norm(run(mod, dev, *get(C7, s)), axis=1); e7.append(float((n[-3:] < .2).mean()))
    r["세션끝_정지_11/13"] = round(float(np.mean(e13)), 2); r["세션끝_정지_T7"] = round(float(np.mean(e7)), 2)
    bas = SS["바구니·왼쪽"] + SS["바구니·가운데"]
    nob, fl, ang = [], [], []
    for s in bas:
        bw, vis = get(C13, s); o = run(mod, dev, bw, vis); REF.setdefault(name, {})[s] = o
        nob.append(mv(run(mod, dev, np.zeros_like(bw), vis)))
        of = run(mod, dev, flip(bw), vis); m_ = (np.linalg.norm(o, axis=1) > .2) & (np.linalg.norm(of, axis=1) > .2)
        if m_.any(): fl.append(float((np.sign(o[m_, 1]) != np.sign(of[m_, 1])).mean()))
    r["박스없음_이동비율(바구니)"] = round(float(np.mean(nob)), 2); r["박스반전_방향뒤집힘"] = round(float(np.mean(fl)), 2)
    if name != "A3(드롭30)":
        for s in bas:
            a, b = REF["A3(드롭30)"][s], REF[name][s]; k = (np.linalg.norm(a, axis=1) > .2) & (np.linalg.norm(b, axis=1) > .2)
            ang += [float(np.degrees(np.arccos(np.clip(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)), -1, 1)))) for x, y in zip(a[k], b[k])]
        r["A3와_방향차이_중앙값(도)"] = round(float(np.median(ang)), 1) if ang else None
    R[name] = r; print(name, json.dumps(r, ensure_ascii=False), flush=True)
json.dump(R, open(ROOT / "docs/v5/egvla2_fm/a3c/eval_boxdrop.json", "w"), ensure_ascii=False, indent=1)

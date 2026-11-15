#!/usr/bin/env python3
"""A3 및 A3C 변형을 같은 기준으로 평가(오프라인, 성공률 예측 아님).
(1) 11/13 세션 재생: 그룹별(바구니·가운데/의자·가운데/화분·가운데/의자·오른쪽/책상·왼쪽/바구니·왼쪽) 헤드가 움직이는 프레임 비율(명령 크기>0.2)
(2) 교차: 의자·가운데 이미지 특징 + 바구니·가운데 박스 (이미지 지름길 지표)
(3) 학습 에피소드 정지 보존: 학습 에피소드 마지막 프레임에서 정지(명령<0.2) 비율, 시작 프레임에서 이동 비율
(4) A3 대비 유사도: 11/13 바구니 세션에서 A3 첫 명령과 각도 차이 중앙값
(5) S1: 전처리 4종(sq768/lb768/sq1024/sq512) 이미지 특징에서 의자·가운데 이동 비율
출력 docs/v5/egvla2_fm/a3c/eval.json"""
import json, sys, glob
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB, train_egvla2_s1 as S1
import robovlm_nav.models.egvla2_fm_heads as HD
from robovlm_nav.serve.egvla2_fm_policy import EgVla2FMPolicy
T = S1.T
C = np.load(ROOT / "runs/egvla2_fm/t13_obj_cache.npz"); D = json.load(open(ROOT / "docs/v5/egvla2_fm/objtest_data.json"))
GRP = {"바구니·가운데": ("가운데(G3)", "바구니"), "의자·가운데": ("가운데(G3)", "의자"), "화분·가운데": ("가운데(G3)", "화분"), "의자·오른쪽": ("오른쪽(G4로 추정)", "의자"), "책상·왼쪽": ("왼쪽(G2로 추정)", "작은 책상"), "바구니·왼쪽": ("왼쪽(G2로 추정)", "바구니")}
SS = {k: [x["sid"] for x in D if (x["pos"], x["obj"]) == v] for k, v in GRP.items()}
def get(sid):
    vis = C[f"{sid}|vis"].astype(np.float32); raw = C[f"{sid}|bbox"][:len(vis)].astype(np.float32); bw = AB.window(raw, 3); bw[raw[:, 3] < .5] = 0; return vis, bw
def first(mod, dev, bw, vis, k=8):
    out = []
    for b, v in zip(bw, vis):
        torch.manual_seed(0); out.append(HD.sample_chunk(mod, torch.tensor(b[None], device=dev), torch.tensor(v[None], device=dev), k).mean(1)[0, 0, :2].cpu().numpy())
    return np.array(out)
mv = lambda a: float((np.linalg.norm(a, axis=1) > .2).mean())
eps = S1.load_main(); cache = np.load(T.CACHE); E = AB.prep(eps, cache, 3)
rng = np.random.RandomState(0); pick = rng.choice(len(E), 40, replace=False)
S1V = np.load(ROOT / "runs/egvla2_fm/a3c/s1_vis.npz", allow_pickle=True) if (ROOT / "runs/egvla2_fm/a3c/s1_vis.npz").exists() else None
S1G = json.loads(str(S1V["_groups"])) if S1V is not None else None
heads = {"A3": ROOT / "runs/egvla2_fm/deploy/C_pi0_servolabel1.0_drop30_e120.pt"}
for p in sorted(glob.glob(str(ROOT / "runs/egvla2_fm/a3c/A3C_*_s0.pt"))): heads[Path(p).stem.replace("_s0", "")] = Path(p)
R = {}; ref = {}
for name, path in heads.items():
    pol = EgVla2FMPolicy(path, k_samples=8); mod, dev = pol.model, pol.device; r = {}
    g = {}
    for k, ss in SS.items():
        a = [first(mod, dev, *get(s)[::-1]) for s in ss[:5]]; g[k] = float(np.mean([mv(x) for x in a]))
        if k == "바구니·왼쪽" or k == "바구니·가운데": ref.setdefault(name, {})[k] = a
    r["이동비율_11/13"] = g
    vc = [get(s)[0] for s in SS["의자·가운데"][:3]]; bb = [get(s)[1] for s in SS["바구니·가운데"][:3]]
    r["의자이미지+바구니박스"] = float(np.mean([mv(first(mod, dev, b[:min(len(v), len(b))], v[:min(len(v), len(b))])) for v in vc for b in bb]))
    # 학습 에피소드 40개: 마지막 3프레임 정지 비율 / 처음 3프레임 이동 비율(학습 라벨 기준과 비교)
    stp, mvs, lab_stp, lab_mv = [], [], [], []
    for ei in pick:
        e = E[ei]; bw = np.asarray(e["bbox"], np.float32); ft = np.asarray(e["feat"], np.float32); ch = np.asarray(e["chunk"])
        a = first(mod, dev, bw, ft)
        stp += list(np.linalg.norm(a[-3:], axis=1) < .2); mvs += list(np.linalg.norm(a[:3], axis=1) > .2)
        lab_stp += list(np.linalg.norm(ch[-3:, 0, :2], axis=1) < .2); lab_mv += list(np.linalg.norm(ch[:3, 0, :2], axis=1) > .2)
    r["학습에피소드_마지막3프레임_정지"] = float(np.mean(stp)); r["학습에피소드_처음3프레임_이동"] = float(np.mean(mvs))
    r["_라벨기준_정지_이동"] = [float(np.mean(lab_stp)), float(np.mean(lab_mv))]
    if S1G:
        r["S1_의자가운데_이동비율"] = {}
        for setting in ("sq768", "lb768", "sq1024", "sq512"):
            a = []
            for sid in S1G["의자·가운데"]:
                vis = S1V[f"{sid}|{setting}"]; _, bw = get(sid); n = min(len(vis), len(bw)); a.append(mv(first(mod, dev, bw[:n], vis[:n])))
            r["S1_의자가운데_이동비율"][setting] = float(np.mean(a))
        r["S1_바구니·가운데_이동비율"] = {}
        for setting in ("sq768", "lb768", "sq1024", "sq512"):
            a = []
            for sid in S1G["바구니·가운데"]:
                vis = S1V[f"{sid}|{setting}"]; _, bw = get(sid); n = min(len(vis), len(bw)); a.append(mv(first(mod, dev, bw[:n], vis[:n])))
            r["S1_바구니·가운데_이동비율"][setting] = float(np.mean(a))
        r["S1_화분·가운데_이동비율"] = {}
        for setting in ("sq768", "lb768", "sq1024", "sq512"):
            a = []
            for sid in S1G["화분·가운데"]:
                vis = S1V[f"{sid}|{setting}"]; _, bw = get(sid); n = min(len(vis), len(bw)); a.append(mv(first(mod, dev, bw[:n], vis[:n])))
            r["S1_화분·가운데_이동비율"][setting] = float(np.mean(a))
    R[name] = r; print(name, json.dumps(r, ensure_ascii=False), flush=True)
json.dump(R, open(ROOT / "docs/v5/egvla2_fm/a3c/eval.json", "w"), ensure_ascii=False, indent=1)

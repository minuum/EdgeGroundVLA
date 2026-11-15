#!/usr/bin/env python3
"""A3(시드0, 배포) 대비 '첫 명령 방향 차이'를 비교 가능한 모든 헤드로 같은 프레임·같은 방식으로 잰다(T7 A3 기록 세션, 짝수 번째 226개).
지표: A3와 해당 헤드가 둘 다 움직이는 프레임에서 두 (lx,ly) 벡터의 사잇각 중앙값·평균·상위10%, 이동/정지 일치율, 좌우 부호 일치율.
헤드: A3 시드 1·2(같은 레시피) / 서보 라벨 0.5 / 서보 없음(드롭 30·드롭 0·e1b tuned) / V0(이미지만) / V4 · V4R(문구 gray basket). 출력 docs/v5/egvla2_fm/a3c/dirdiff_all.json"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB
import robovlm_nav.models.egvla2_fm_heads as HD
from robovlm_nav.serve.egvla2_fm_policy import EgVla2FMPolicy
C7 = np.load(ROOT / "runs/egvla2_fm/t7_session_cache.npz"); dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sids = sorted({k.split("|")[0] for k in C7.files if k.endswith("|vis")})[::2]
def load(path):
    ck = torch.load(path, map_location=dev, weights_only=False); m = HD.PiZeroFM(int(ck["bdim"]), int(ck["fdim"])).to(dev); m.load_state_dict(ck["state"]); m.eval(); return m, int(ck["fdim"])
D = ROOT / "runs/egvla2_fm"
HEADS = {"A3(시드0,배포)": (D / "deploy/C_pi0_servolabel1.0_drop30_e120.pt", "img"), "A3 레시피 시드1": (D / "a3c/A3C_D30_s1.pt", "img"), "A3 레시피 시드2": (D / "a3c/A3C_D30_s2.pt", "img"),
         "서보 라벨 0.5(A2)": (D / "deploy/C_pi0_servolabel0.5_drop30_e120.pt", "img"), "서보 없음·드롭30(A1)": (D / "deploy/C_pi0_adaln_drop30_e120.pt", "img"),
         "서보 없음·드롭0": (D / "deploy/C_pi0_adaln_e120.pt", "img"), "e1b tuned": (D / "deploy/C_pi0_e1b_tuned_drop30_e120.pt", "img"),
         "V0 이미지만(텍스트 조건화 계열)": (D / "textcond/EG4-00_V0_full.pt", "img"), "V4(문구 gray basket)": (D / "textcond/EG4-04_V4_full.pt", "txt"), "V4R(문구 gray basket)": (D / "textcond/EG4-07_V4R_full.pt", "txt")}
MODELS = {}
for n, (p, kind) in HEADS.items():
    try: MODELS[n] = (*load(p), kind)
    except Exception as e: print("skip", n, e)
def run(m, kind, bw, vis, txt):
    o = []
    for i, (b, v) in enumerate(zip(bw, vis)):
        f = np.concatenate([v, txt[i]]) if kind == "txt" else v
        torch.manual_seed(0); o.append(HD.sample_chunk(m, torch.tensor(b[None], device=dev), torch.tensor(f[None].astype(np.float32), device=dev), 8).mean(1)[0, 0, :2].cpu().numpy())
    return np.array(o)
OUT = {n: [] for n in MODELS}; t0 = time.time()
for i, sid in enumerate(sids):
    vis = C7[f"{sid}|vis"].astype(np.float32); raw = C7[f"{sid}|bbox"][:len(vis)].astype(np.float32); bw = AB.window(raw, 3); bw[raw[:, 3] < .5] = 0; txt = C7[f"{sid}|txt"].astype(np.float32)[:, 0]
    for n, (m, fd, kind) in MODELS.items(): OUT[n].append(run(m, kind, bw, vis, txt))
    if i % 40 == 0: print(i, len(sids), f"{time.time()-t0:.0f}s", flush=True)
nr = lambda a: np.linalg.norm(a, axis=1); ref = "A3(시드0,배포)"; R = {}
def stats(a_list, b_list):
    ang, ag, sg = [], [], []
    for a, b in zip(a_list, b_list):
        ma, mb = nr(a) > .2, nr(b) > .2; ag += list(ma == mb); k = ma & mb
        if k.any(): ang += list(np.degrees(np.arccos(np.clip((a[k] * b[k]).sum(1) / (nr(a[k]) * nr(b[k])), -1, 1)))); sg += list(np.sign(a[k, 1]) == np.sign(b[k, 1]))
    return dict(중앙값=round(float(np.median(ang)), 2), 평균=round(float(np.mean(ang)), 2), 상위10퍼=round(float(np.percentile(ang, 90)), 2), 이동정지_일치=round(float(np.mean(ag)), 3), 좌우부호_일치=round(float(np.mean(sg)), 3), n_프레임=len(ang))
for n in MODELS:
    if n != ref: R[n] = stats(OUT[ref], OUT[n]); print(n, R[n], flush=True)
R["(참고) 시드1 vs 시드2"] = stats(OUT["A3 레시피 시드1"], OUT["A3 레시피 시드2"]); print("시드1 vs 시드2", R["(참고) 시드1 vs 시드2"])
json.dump(R, open(ROOT / "docs/v5/egvla2_fm/a3c/dirdiff_all.json", "w"), ensure_ascii=False, indent=1)

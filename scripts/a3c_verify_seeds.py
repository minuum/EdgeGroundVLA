#!/usr/bin/env python3
"""배포 후보 검증: A3(시드0, 배포본) vs 같은 레시피 시드 1·2 vs 앙상블(평균). 의자·가운데 5세션으로 시드를 고르는 과적합을 피하려고
(1) 선택에 쓰지 않은 데이터: T7 A3 기록 세션 452개 — 기록된 실제 A3의 이동/정지와 일치하는지, 세션 끝 정지, 방향이 A3 시드0과 얼마나 같은지
(2) 11/13의 다른 그룹(의자·옆, 책상, 화분, 바구니)과 의자·가운데
앙상블 = 세 헤드의 첫 명령(K=8 평균)을 평균. 출력 docs/v5/egvla2_fm/a3c/verify_seeds.json"""
import json, sys, time
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
s7 = sorted({k.split("|")[0] for k in C7.files if k.endswith("|vis")})
def get(C, sid):
    vis = C[f"{sid}|vis"].astype(np.float32); raw = C[f"{sid}|bbox"][:len(vis)].astype(np.float32); bw = AB.window(raw, 3); bw[raw[:, 3] < .5] = 0; return bw, vis
HEADS = {"A3(시드0,배포)": ROOT / "runs/egvla2_fm/deploy/C_pi0_servolabel1.0_drop30_e120.pt", "시드1": ROOT / "runs/egvla2_fm/a3c/A3C_D30_s1.pt", "시드2": ROOT / "runs/egvla2_fm/a3c/A3C_D30_s2.pt"}
pols = {k: EgVla2FMPolicy(v, k_samples=8) for k, v in HEADS.items()}
def run(pol, bw, vis):
    o = []
    for b, v in zip(bw, vis):
        torch.manual_seed(0); o.append(HD.sample_chunk(pol.model, torch.tensor(b[None], device=pol.device), torch.tensor(v[None], device=pol.device), 8).mean(1)[0, 0, :2].cpu().numpy())
    return np.array(o)
OUT = {}; t0 = time.time()
for tag, C, sids in (("T7", C7, s7), ("11/13", C13, [x["sid"] for x in D])):
    for i, sid in enumerate(sids):
        bw, vis = get(C, sid); OUT[(tag, sid)] = {k: run(p, bw, vis) for k, p in pols.items()}
        if i % 50 == 0: print(tag, i, len(sids), f"{time.time()-t0:.0f}s", flush=True)
for v in OUT.values(): v["앙상블(3)"] = np.mean([v[k] for k in HEADS], 0); v["앙상블(시드1·2)"] = np.mean([v["시드1"], v["시드2"]], 0)
names = list(HEADS) + ["앙상블(3)", "앙상블(시드1·2)"]
nr = lambda a: np.linalg.norm(a, axis=1); R = {}
for n in names:
    mvA, agree, end, ang, lat, mvall = [], [], [], [], [], []
    for sid in s7:
        o = OUT[("T7", sid)][n]; ref = OUT[("T7", sid)]["A3(시드0,배포)"]; rec = np.linalg.norm(np.asarray(C7[f"{sid}|act"])[:len(o), :2], axis=1) > .2 if f"{sid}|act" in C7.files else None
        mv = nr(o) > .2; mvall.append(float(mv.mean())); end.append(float((nr(o)[-3:] < .2).mean()))
        if rec is not None: agree.append(float((mv == rec[:len(mv)]).mean()))
        k = mv & (nr(ref) > .2)
        if k.any():
            ang += list(np.degrees(np.arccos(np.clip((o[k] * ref[k]).sum(1) / (nr(o[k]) * nr(ref[k])), -1, 1)))); lat += list(np.sign(o[k, 1]) == np.sign(ref[k, 1]))
    r = {"T7_이동비율": round(float(np.mean(mvall)), 3), "T7_기록(실제 A3)과_이동정지_일치": round(float(np.mean(agree)), 3) if agree else None, "T7_세션끝_정지": round(float(np.mean(end)), 3),
         "T7_A3시드0와_방향차이_중앙값": round(float(np.median(ang)), 2) if ang else None, "T7_좌우부호_일치": round(float(np.mean(lat)), 3) if lat else None}
    r["11/13_그룹별_이동"] = {g: round(float(np.mean([float((nr(OUT[("11/13", s)][n]) > .2).mean()) for s in ss[:5]])), 2) for g, ss in SS.items()}
    r["11/13_세션끝_정지"] = round(float(np.mean([float((nr(OUT[("11/13", x["sid"])][n])[-3:] < .2).mean()) for x in D])), 3); R[n] = r; print(n, json.dumps(r, ensure_ascii=False), flush=True)
json.dump(R, open(ROOT / "docs/v5/egvla2_fm/a3c/verify_seeds.json", "w"), ensure_ascii=False, indent=1)

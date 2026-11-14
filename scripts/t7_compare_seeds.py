#!/usr/bin/env python3
"""T7 최종: V0·V0R·V1·V2·V3·V4·V5 × 시드 0~9를 452 A3 세션에서 같은 지표로 비교(A3 출력은 한 번만 계산해 캐시).
있는 체크포인트만 쓴다(이어하기). 출력 docs/v5/egvla2_fm/t7_all_seeds.json = {종류: {지표: {mean,std,min,max,n,per_seed}}}"""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB
import eval_text_cond_t5 as E5
import robovlm_nav.models.egvla2_fm_heads as HD
import text_vocab as TV
from robovlm_nav.serve.egvla2_fm_policy import EgVla2FMPolicy
DEV = AB.DEV; GO = 0.2
C = np.load(ROOT / "runs/egvla2_fm/t7_session_cache.npz"); PH = list(C["_phrases"]); TK = np.load(ROOT / "runs/egvla2_fm/t7_session_tokens.npz"); PT = list(TK["_phrases"])
names = sorted({k.split("|")[0] for k in C.files if k.endswith("|vis")})
KIND = {"V0": "EG4-00_V0_full", "V0R": "EG4-05_V0R_full", "V1": "EG4-01_V1_full", "V2": "EG4-02_V2_full", "V3": "EG4-03_V3_full", "V4": "EG4-04_V4_full", "V4R": "EG4-07_V4R_full", "V5": "EG4-06_V5_full"}
neg_i = [PH.index(p) for p in TV.NEG_HELD]; pos_i = [PH.index(p) for p in TV.POS_HELD]; negt = [PT.index(p) for p in TV.NEG_HELD]; post = [PT.index(p) for p in TV.POS_HELD]
def raw_of(n):
    m_ = len(C[f"{n}|vis"]); return C[f"{n}|bbox"][:m_]
A3c = ROOT / "runs/egvla2_fm/t7_a3_cache.npz"
if A3c.exists(): A3O = dict(np.load(A3c))
else:
    A3 = EgVla2FMPolicy(str(ROOT / "runs/egvla2_fm/deploy/C_pi0_servolabel1.0_drop30_e120.pt"), speed_source="shadow", s_max=1.0); A3O = {}
    for n in names:
        raw = raw_of(n); vis = C[f"{n}|vis"].astype(np.float32)
        for tag, rb in [("n", raw), ("m", np.where(np.arange(4) == 0, 1 - raw, raw))]:
            A3.reset(); A3O[f"{n}|{tag}"] = np.array([A3.step(rb[t], vis[t])["first"][:2] for t in range(len(raw))])
    np.savez(A3c, **A3O); print("a3 cached", flush=True)
def load(pn, v):
    ck = torch.load(ROOT / f"runs/egvla2_fm/textcond/{pn}.pt", map_location=DEV, weights_only=False)
    m = (E5.AttnWrap() if v == "V5" else HD.PiZeroFM(12, ck["fdim"])).to(DEV); m.load_state_dict(ck["state"]); m.eval(); return m
@torch.no_grad()
def step1(m, bb, ft):
    torch.manual_seed(0); return HD.sample_chunk(m, bb, ft, 8).mean(1)[:, 0, :2].cpu().numpy()
def ang(a, b):
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1); m = (na > GO) & (nb > GO)
    return np.degrees(np.arccos(np.clip((a[m] * b[m]).sum(1) / (na[m] * nb[m]), -1, 1)))
lat = lambda x, y: float((np.sign(x[:, 1]) == np.sign(y[:, 1]))[(np.linalg.norm(x, axis=1) > GO) & (np.linalg.norm(y, axis=1) > GO)].mean())
A = {t: np.concatenate([A3O[f"{n}|{t}"] for n in names]) for t in ("n", "m")}
import os
ONLY = os.environ.get("ONLY")
res = {}
for v, base in KIND.items():
    if ONLY and v != ONLY: continue
    for sd in range(10):
        sfx = "" if sd == 0 else f"_s{sd}"
        if not (ROOT / f"runs/egvla2_fm/textcond/{base}{sfx}.pt").exists(): continue
        M = load(base + sfx, v); out = {t: [] for t in ("n", "m")}; ns, ps = [], []
        for n in names:
            raw = raw_of(n); vis = torch.tensor(C[f"{n}|vis"].astype(np.float32), device=DEV); txt = C[f"{n}|txt"].astype(np.float32)[:len(raw)]; tk = TK[f"{n}|tok"].astype(np.float32)
            T = lambda a: torch.tensor(a, device=DEV)
            feat = lambda k: vis if v in ("V0", "V0R") else (torch.cat([vis, T(tk[:, k]).flatten(1)], -1) if v == "V5" else torch.cat([vis, T(txt[:, k])], -1))
            for tag, rb in [("n", raw), ("m", np.where(np.arange(4) == 0, 1 - raw, raw))]:
                bbw = AB.window(rb.astype(np.float32), 3); bbw[raw[:, 3] < 0.5] = 0; bt = torch.tensor(bbw, device=DEV)
                out[tag].append(step1(M, bt, feat(0)))
                if tag == "n" and v in ("V2", "V3", "V4", "V4R", "V5"):
                    g = lambda ks: float(np.mean([np.linalg.norm(step1(M, bt, feat(j)), axis=1) < GO for j in ks]))
                    ns.append(g(negt if v == "V5" else neg_i)); ps.append(g(post if v == "V5" else pos_i))
        o = {t: np.concatenate(out[t]) for t in out}
        r = dict(ang=float(ang(o["n"], A["n"]).mean()), lat=lat(o["n"], A["n"]), m_ang=float(ang(o["m"], A["n"]).mean()), m_lat=lat(o["m"], A["n"]))
        if ns: r.update(neg_stop=float(np.mean(ns)), pos_stop=float(np.mean(ps)))
        res.setdefault(v, {})[sd] = r; print(v, sd, {k: round(x, 3) for k, x in r.items()}, flush=True)
OUTP = ROOT / "docs/v5/egvla2_fm/t7_all_seeds.json"
final = json.loads(OUTP.read_text()) if (ONLY and OUTP.exists()) else {}
for v, d in res.items():
    final[v] = {}
    for m in next(iter(d.values())):
        xs = [d[s][m] for s in sorted(d)]; final[v][m] = dict(mean=float(np.mean(xs)), std=float(np.std(xs)), min=float(np.min(xs)), max=float(np.max(xs)), n=len(xs), per_seed=xs)
(ROOT / "docs/v5/egvla2_fm/t7_all_seeds.json").write_text(json.dumps(final, indent=1)); print("saved")

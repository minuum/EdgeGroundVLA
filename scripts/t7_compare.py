#!/usr/bin/env python3
"""T7 (3): A3 세션 프레임에서 A3 체크포인트(서빙 정책 코드) vs V0/V4(저장 체크포인트)의 첫 스텝 (lx,ly) 비교. 기준은 A3의 명령이므로 유사도만 잰다.
입력: runs/egvla2_fm/t7_session_cache.npz, deploy A3 ckpt, runs/egvla2_fm/textcond/*.pt. 출력: docs/v5/egvla2_fm/t7_replay[_sN].json"""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB
import robovlm_nav.models.egvla2_fm_heads as HD
import text_vocab as TV
from robovlm_nav.serve.egvla2_fm_policy import EgVla2FMPolicy
import os
SD = int(os.environ.get("SEED", 0)); SFX = "" if SD == 0 else f"_s{SD}"
DEV = AB.DEV; GO = 0.2
C = np.load(ROOT / "runs/egvla2_fm/t7_session_cache.npz"); PH = list(C["_phrases"])
names = sorted({k.split("|")[0] for k in C.files if k.endswith("|vis")})
A3 = EgVla2FMPolicy(str(ROOT / "runs/egvla2_fm/deploy/C_pi0_servolabel1.0_drop30_e120.pt"), speed_source="shadow", s_max=1.0)


def load(pn, fdim):
    ck = torch.load(ROOT / f"runs/egvla2_fm/textcond/{pn}.pt", map_location=DEV, weights_only=False)
    m = HD.PiZeroFM(12, ck["fdim"]).to(DEV); m.load_state_dict(ck["state"]); m.eval(); return m
V0, V4 = load("EG4-00_V0_full" + SFX, 768), load("EG4-04_V4_full" + SFX, 1536)


@torch.no_grad()
def step1(m, bb, ft):
    torch.manual_seed(0); return HD.sample_chunk(m, bb, ft, 8).mean(1)[:, 0, :2].cpu().numpy()


def zero_miss(bbw, raw):
    z = bbw.copy(); z[raw[:, 3] < 0.5] = 0; return z


def ang(a, b):
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1); m = (na > GO) & (nb > GO)
    return np.degrees(np.arccos(np.clip((a[m] * b[m]).sum(1) / (na[m] * nb[m]), -1, 1)))


rows = {k: [] for k in ["stored", "a3", "v0", "v4", "v4_pos", "v0_mirror", "v4_mirror", "a3_mirror", "v4_neg_stop", "v4_hard_stop", "v4_heldpos_stop"]}
neg_i = [PH.index(p) for p in TV.NEG_HELD]; hard_i = [PH.index(p) for p in TV.HARD_NEG]; pos_i = [PH.index(p) for p in TV.POS_HELD]
sess = []
for n in names:
    raw = C[f"{n}|bbox"]; vis = C[f"{n}|vis"].astype(np.float32); txt = C[f"{n}|txt"].astype(np.float32); act = C[f"{n}|act"]
    m_ = len(vis); raw, act, txt = raw[:m_], act[:m_], txt[:m_]   # 13개 세션은 박스·액션이 이미지보다 1행 많음 → 이미지 길이에 맞춤
    for tag, rb in [("", raw), ("_mirror", np.where(np.arange(4) == 0, 1 - raw, raw))]:   # 거울: cx 반전(박스 반대편 오검출 모사)
        bbw = zero_miss(AB.window(rb.astype(np.float32), 3), raw)
        bt, ft = torch.tensor(bbw, device=DEV), torch.tensor(vis, device=DEV)
        a3 = []; A3.reset()
        for t in range(len(raw)):
            a3.append(A3.step(rb[t], vis[t])["first"][:2])
        a3 = np.array(a3)
        v0 = step1(V0, bt, ft); v4 = step1(V4, bt, torch.cat([ft, torch.tensor(txt[:, 0], device=DEV)], -1))
        rows["a3" + tag].append(a3); rows["v0" + tag].append(v0); rows["v4" + tag].append(v4)
        if tag == "":
            rows["stored"].append(act[:, :2])
            sg = lambda ks: float(np.mean([np.linalg.norm(step1(V4, bt, torch.cat([ft, torch.tensor(txt[:, k], device=DEV)], -1)), axis=1) < GO for k in ks]))
            rows["v4_neg_stop"].append(sg(neg_i)); rows["v4_hard_stop"].append(sg(hard_i)); rows["v4_heldpos_stop"].append(sg(pos_i))
    sess.append(n)
cat = lambda k: np.concatenate(rows[k])
S, A, V0r, V4r = cat("stored"), cat("a3"), cat("v0"), cat("v4")
st = lambda x: float((np.linalg.norm(x, axis=1) < GO).mean())
lat = lambda x, y: float((np.sign(x[:, 1]) == np.sign(y[:, 1]))[(np.linalg.norm(x, axis=1) > GO) & (np.linalg.norm(y, axis=1) > GO)].mean())
out = dict(sessions=len(sess), frames=int(len(S)),
           a3_replay_vs_stored=dict(ang_mean=float(ang(A, S).mean()), ang_p90=float(np.percentile(ang(A, S), 90)), stop_a3=st(A), stop_stored=st(S), lat_agree=lat(A, S)),
           v0_vs_a3=dict(ang_mean=float(ang(V0r, A).mean()), ang_p90=float(np.percentile(ang(V0r, A), 90)), stop=st(V0r), lat_agree=lat(V0r, A)),
           v4_vs_a3=dict(ang_mean=float(ang(V4r, A).mean()), ang_p90=float(np.percentile(ang(V4r, A), 90)), stop=st(V4r), lat_agree=lat(V4r, A)),
           mirror=dict(a3_vs_a3=float(ang(cat("a3_mirror"), A).mean()), v0_vs_a3=float(ang(cat("v0_mirror"), A).mean()), v4_vs_a3=float(ang(cat("v4_mirror"), A).mean()),
                       a3_lat_agree=lat(cat("a3_mirror"), A), v0_lat_agree=lat(cat("v0_mirror"), A), v4_lat_agree=lat(cat("v4_mirror"), A)),
           v4_text=dict(neg_phrases_stop=float(np.mean(rows["v4_neg_stop"])), hard_neg_stop=float(np.mean(rows["v4_hard_stop"])), held_pos_stop=float(np.mean(rows["v4_heldpos_stop"]))))
(ROOT / ("docs/v5/egvla2_fm/t7_replay" + SFX + ".json")).write_text(json.dumps(out, indent=1, ensure_ascii=False)); print(json.dumps(out, indent=1, ensure_ascii=False))

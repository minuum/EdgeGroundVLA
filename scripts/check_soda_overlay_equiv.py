#!/usr/bin/env python3
"""V4·V4R 서빙 경로 동등성 점검(plan_20261115_v4_serving.md §4). 기록 세션 프레임을 JPEG q95 왕복 후 서빙 추출기·정책에 넣고
학습·재생 캐시 경로(같은 체크포인트, 캐시된 vis·txt·기록 박스)와 비교한다. 두 경로 모두 프레임마다 torch.manual_seed(0).
기준(사전 고정): 문구특징 코사인≥0.999 · 박스 중심오차≤0.02 · 명령 방향 중앙값≤1° · 정지판정 일치≥95%.
실행: HF_HUB_OFFLINE=1 .venv/bin/python3 scripts/check_text_serving_equiv.py [fp32|fp16]. 출력 docs/v5/egvla2_fm/text_serving_equiv_{dtype}.json"""
import io, json, sys
from pathlib import Path
import h5py, numpy as np, torch
from PIL import Image
ROOT = Path("/home/minum/26CS/MoNaVLA")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, '/tmp/claude-1001/-home-minum-26CS-MoNaVLA/2f52c5c7-46bd-47bc-8139-16c898788d30/scratchpad/sodaov')
import robovlm_nav.models.egvla2_fm_heads as HD
import eval_subgoal_ablation as AB
import os, robovlm_nav.serve.egvla2_inference_server as SRV
assert 'sodaov' in SRV.__file__, SRV.__file__
print('SERVER FILE', SRV.__file__)
DT = sys.argv[1] if len(sys.argv) > 1 else "fp32"; dtype = torch.float32 if DT == "fp32" else torch.float16; GO = 0.2
C13 = np.load(ROOT / "runs/egvla2_fm/t13_obj_cache.npz"); P13 = list(C13["_phrases"]); C7 = np.load(ROOT / "runs/egvla2_fm/t7_session_cache.npz")
def sess():
    out = []; S13 = sorted({k.split("|")[0] for k in C13.files if k.endswith("|vis")})
    for sid in S13:
        ins = str(h5py.File(ROOT / f"docs/inference_sessions/session_{sid}.h5").attrs["instruction"]); out.append(("13", sid, ins))
    pick = {"go to the brown wooden chair": 2, "go to the gray basket": 2, "flower pot": 2, "go to the small desk": 1}; sel = []
    for k, n in pick.items(): sel += [x for x in out if x[2] == k][:n]
    n7 = sorted({k.split("|")[0] for k in C7.files if k.endswith("|vis")})[::150][:2]
    return sel + [("7", n, "gray basket") for n in n7]
R = {}
for tag, base in (("V4", "EG4-04_V4_full_s1"), ("V4R", "EG4-07_V4R_full_s1")):
    os.environ.update(VLA_EGVLA2_TEXT='1', VLA_EGVLA2_CKPT=str(ROOT / f"runs/egvla2_fm/textcond/{base}.pt"), VLA_EGVLA2_DTYPE=DT)
    svc = SRV.load_service_from_env(); pol = svc.policy; ext = svc.extractor; m = pol.model; cs, bx, ag, st, ph = [], [], [], [], []
    for src, sid, ins in sess():
        if src == "13": path = ROOT / f"docs/inference_sessions/session_{sid}.h5"; C = C13; key = sid; k = P13.index(ins)
        else: path = ROOT / f"docs/inference_sessions/{sid}"; C = C7; key = sid; k = 0
        with h5py.File(path, "r") as h: imgs = h["observations/images"][:len(C[f"{key}|vis"])]
        vis = C[f"{key}|vis"].astype(np.float32); txt = C[f"{key}|txt"].astype(np.float32)[:, k]; raw = C[f"{key}|bbox"][:len(vis)].astype(np.float32)
        bbw = AB.window(raw, 3); bbw[raw[:, 3] < 0.5] = 0
        pol.reset(); PH = ins if src == "13" else None   # soda 방식: 지시어는 인자, 없으면 None
        for t, im in enumerate(imgs):
            buf = io.BytesIO(); Image.fromarray(np.ascontiguousarray(im).astype(np.uint8)).save(buf, "JPEG", quality=95); img = np.array(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
            bb, ft = (ext(img, PH) if PH is not None else ext(img)); torch.manual_seed(0); r = pol.step(bb, ft); a = np.array(r["first"][:2])
            fe = torch.tensor(np.concatenate([vis[t], txt[t]]), device=pol.device)[None]; torch.manual_seed(0)
            ref = HD.sample_chunk(m, torch.tensor(bbw[t][None], device=pol.device), fe, 8).mean(1)[0, 0, :2].cpu().numpy()
            cos = lambda x, y: float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-9))
            cs.append((cos(ft[:768], vis[t]), cos(ft[768:], txt[t]))); bx.append(float(np.hypot(bb[0] - raw[t][0], bb[1] - raw[t][1])) if (bb[3] > 0.5 and raw[t][3] > 0.5) else (0.0 if bb[3] == raw[t][3] else 9.0))
            na, nr = np.linalg.norm(a), np.linalg.norm(ref); st.append((na < GO) == (nr < GO))
            if na > GO and nr > GO: ag.append(float(np.degrees(np.arccos(np.clip(np.dot(a, ref) / (na * nr), -1, 1)))))
    cs = np.array(cs); R[tag] = dict(n_frames=len(cs), vis_cos_min=float(cs[:, 0].min()), txt_cos_min=float(cs[:, 1].min()), txt_cos_mean=float(cs[:, 1].mean()), box_err_max=float(max(bx)), box_err_median=float(np.median(bx)),
                                     ang_median=float(np.median(ag)) if ag else None, ang_p90=float(np.percentile(ag, 90)) if ag else None, n_ang=len(ag), stop_agree=float(np.mean(st)))
    print(tag, DT, R[tag], flush=True)
R["_criteria"] = dict(txt_cos=0.999, box_err=0.02, ang_median=1.0, stop_agree=0.95); R["_dtype"] = DT
(Path("/tmp/claude-1001/-home-minum-26CS-MoNaVLA/2f52c5c7-46bd-47bc-8139-16c898788d30/scratchpad") / f"soda_overlay_equiv_{DT}.json").write_text(json.dumps(R, indent=1)); print("saved")

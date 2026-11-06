#!/usr/bin/env python3
"""phrase 변형이 Florence-2 bbox와 A3 헤드 명령에 미치는 영향(오프라인). plan: docs/plans/plan_20261106_text_robustness_offline.md"""
import glob, json, sys
from pathlib import Path
import h5py, numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from robovlm_nav.serve.egvla2_fm_policy import FusedF2Extractor, EgVla2FMPolicy
import precompute_v6_florence2_fused as P

PHR = {"ref": "gray basket",
       "para_bin": "gray bin", "para_container": "gray container", "para_the": "the gray basket", "para_sentence": "go to the gray bin",
       "generic_basket": "basket", "wrong_chair": "red chair", "wrong_person": "person", "wrong_box": "white box"}
CKPT = ROOT / "runs/egvla2_fm/deploy/C_pi0_servolabel1.0_drop30_e120.pt"

def frames():
    fs = sorted(glob.glob(str(ROOT / "docs/inference_sessions/session_20261029_*.h5")))[::4][:24]
    out = []
    for f in fs:
        with h5py.File(f, "r") as h:
            n = h["observations/images"].shape[0]
            for q in (0.25, 0.5, 0.75):
                j = int(n * q); out.append((Path(f).stem[-6:], j, h["observations/images"][j]))
    return out

def iou(a, b):
    return 0.0

def main():
    ex = FusedF2Extractor(); pol = EgVla2FMPolicy(CKPT, k_samples=8)
    fr = frames(); print(len(fr), "frames", flush=True)
    res = {k: [] for k in PHR}
    for k, ph in PHR.items():
        P.PHRASE = ph
        for sid, j, img in fr:
            bb, ft = ex(img)
            pol.reset(); torch.manual_seed(0)
            o = pol.step(bb, ft)
            res[k].append(dict(sid=sid, j=j, bbox=bb.tolist(), first=o["first"], miss=bool(bb[3] < 0.5), feat_norm=float(np.linalg.norm(ft))))
        print(k, ph, "done", flush=True)
    ref = res["ref"]; summ = {}
    for k, r in res.items():
        det = float(np.mean([not x["miss"] for x in r]))
        both = [(a, b) for a, b in zip(ref, r) if not a["miss"] and not b["miss"]]
        dcx = float(np.mean([abs(a["bbox"][0] - b["bbox"][0]) for a, b in both])) if both else None
        dar = float(np.mean([abs(a["bbox"][2] - b["bbox"][2]) for a, b in both])) if both else None
        dlx = float(np.mean([abs(a["first"][1] - b["first"][1]) for a, b in zip(ref, r)]))  # ly(횡) 첫 스텝 차이
        daz = float(np.mean([abs(a["first"][2] - b["first"][2]) for a, b in zip(ref, r)]))
        same = float(np.mean([np.sign(a["first"][2]) == np.sign(b["first"][2]) for a, b in zip(ref, r)]))
        summ[k] = dict(phrase=PHR[k], det_rate=det, n_both=len(both), d_cx=dcx, d_area=dar, d_first_ly=dlx, d_first_az=daz, az_sign_agree=same,
                       feat_identical=bool(all(abs(a["feat_norm"] - b["feat_norm"]) < 1e-3 for a, b in zip(ref, r))))
    (ROOT / "docs/v5/egvla2_fm/text_robustness.json").write_text(json.dumps(dict(summary=summ, n_frames=len(fr), raw=res), ensure_ascii=False, indent=1))
    for k, s in summ.items(): print(k, {a: (round(b, 3) if isinstance(b, float) else b) for a, b in s.items()})

main()

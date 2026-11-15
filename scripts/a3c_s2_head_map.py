#!/usr/bin/env python3
"""S2: A3 헤드가 (박스 중심 cx, 면적)에 따라 어떤 명령을 내는지 지도 + 학습 라벨의 같은 구간 분포.
plan_20261115_a3_improvement.md S2. 박스 창은 3프레임 같은 값(정지 상태), 이미지 특징은 학습 캐시 프레임의 실제 특징 N개 평균 결과.
출력 docs/v5/egvla2_fm/a3c/s2_head_map.json"""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB, train_egvla2_s1 as S1
import robovlm_nav.models.egvla2_fm_heads as HD
from robovlm_nav.serve.egvla2_fm_policy import EgVla2FMPolicy
T = S1.T; DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eps = S1.load_main(); cache = np.load(T.CACHE); E = AB.prep(eps, cache, 3)
# 학습 라벨: 프레임별 박스(마지막 창 프레임) · 첫 명령 크기
cx, ar, hs, fv, ln = [], [], [], [], []
for e in E:
    bb = np.asarray(e["bbox"]); ch = np.asarray(e["chunk"])
    cx += list(bb[:, 0]); ar += list(bb[:, 2]); hs += list(bb[:, 3]); ln += list(np.linalg.norm(ch[:, 0, :2], axis=1))
    fv.append(np.asarray(e["feat"])[::7])
cx, ar, hs, ln = map(np.array, (cx, ar, hs, ln)); fv = np.concatenate(fv)[:64].astype(np.float32)
m = (hs > .5)
bins_a = [0, .01, .02, .035, .05, .08, .12, .2, 1.01]; bins_c = [0, .35, .45, .55, .65, 1.01]
lab = {}
for i in range(len(bins_a) - 1):
    for j in range(len(bins_c) - 1):
        s = m & (ar >= bins_a[i]) & (ar < bins_a[i + 1]) & (cx >= bins_c[j]) & (cx < bins_c[j + 1])
        lab[f"a{bins_a[i]}-{bins_a[i+1]}|c{bins_c[j]}-{bins_c[j+1]}"] = dict(n=int(s.sum()), move=float((ln[s] > .2).mean()) if s.sum() else None, mean=float(ln[s].mean()) if s.sum() else None)
pol = EgVla2FMPolicy(ROOT / "runs/egvla2_fm/deploy/C_pi0_servolabel1.0_drop30_e120.pt", k_samples=8)
mod = pol.model; mod.eval(); out = {}
ga = [.005, .008, .012, .02, .03, .045, .06, .08, .1, .15, .25]; gc = [.1, .23, .35, .48, .6, .76, .9]
for a in ga:
    for c in gc:
        bb = torch.tensor([[c, .5, a, 1.0] * 3], device=pol.device, dtype=torch.float32)
        norms = []
        for f in fv[:16]:
            torch.manual_seed(0)
            ft = torch.tensor(f, device=pol.device)[None]
            ch = HD.sample_chunk(mod, bb, ft, 8).mean(1)[0, 0, :2].cpu().numpy(); norms.append(float(np.linalg.norm(ch)))
        out[f"{a}|{c}"] = dict(norm_mean=float(np.mean(norms)), move_frac=float(np.mean(np.array(norms) > .2)))
json.dump(dict(label_bins=lab, head_map=out, ga=ga, gc=gc), open(ROOT / "docs/v5/egvla2_fm/a3c/s2_head_map.json", "w"), indent=1)
print("A3 head move fraction (rows area, cols cx)", gc)
for a in ga: print("%.3f" % a, " ".join("%4.2f" % out[f"{a}|{c}"]["move_frac"] for c in gc))
print("label move fraction (rows area bins, cols cx bins)", bins_c)
for i in range(len(bins_a) - 1):
    print("%.3f-%.3f" % (bins_a[i], bins_a[i + 1]), " ".join(("%4.2f(%5d)" % (lab[f"a{bins_a[i]}-{bins_a[i+1]}|c{bins_c[j]}-{bins_c[j+1]}"]["move"] or 0, lab[f"a{bins_a[i]}-{bins_a[i+1]}|c{bins_c[j]}-{bins_c[j+1]}"]["n"])) for j in range(len(bins_c) - 1)))

#!/usr/bin/env python3
"""학습 에피소드(300개 중 짝수 150개, 프레임 2개 간격)에서 헤드가 '이동 후 처음 정지(<0.2)'를 내는 박스 면적을 a3·a3-s1·a3-s2·정답 라벨로 비교.
11/13 기록 프레임은 a3가 멈춘 지점(면적 약 0.23)에서 끝나 그 너머를 볼 수 없으므로(검열) 면적이 더 큰 구간이 있는 학습 에피소드로 확인한다(학습에 쓴 데이터라 일반화 지표는 아님).
출력 docs/v5/egvla2_fm/a3c/ch82_train_stoparea.json"""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB, train_egvla2_s1 as S1
import robovlm_nav.models.egvla2_fm_heads as HD
from robovlm_nav.serve.egvla2_fm_policy import EgVla2FMPolicy
T = S1.T; eps = S1.load_main(); cache = np.load(T.CACHE); E = AB.prep(eps, cache, 3)
H = {"a3": ROOT / "runs/egvla2_fm/deploy/C_pi0_servolabel1.0_drop30_e120.pt", "a3-s1": ROOT / "runs/egvla2_fm/a3c/A3C_D30_s1.pt", "a3-s2": ROOT / "runs/egvla2_fm/a3c/A3C_D30_s2.pt"}
pols = {k: EgVla2FMPolicy(v, k_samples=8) for k, v in H.items()}
def run(pol, bw, ft):
    o = []
    for b, v in zip(bw, ft):
        torch.manual_seed(0); o.append(HD.sample_chunk(pol.model, torch.tensor(b[None], device=pol.device), torch.tensor(v[None], device=pol.device), 8).mean(1)[0, 0, :2].cpu().numpy())
    return np.array(o)
res = {k: [] for k in list(H) + ["라벨"]}; areas_end = []
for ei in range(0, len(E), 2):
    e = E[ei]; bw = np.asarray(e["bbox"], np.float32)[::2]; ft = np.asarray(e["feat"], np.float32)[::2]; ch = np.asarray(e["chunk"])[::2]; a = bw[:, 10]; has = bw[:, 11] > .5
    seqs = {"라벨": np.linalg.norm(ch[:, 0, :2], axis=1)}
    for k, p in pols.items(): seqs[k] = np.linalg.norm(run(p, bw, ft), axis=1)
    areas_end.append(float(a[has][-1]) if has.any() else np.nan)
    for k, n in seqs.items():
        mv = np.where(n > .2)[0]
        if len(mv) == 0: res[k].append((np.nan, 0)); continue
        later = np.where((n[mv[0]:] < .2) & has[mv[0]:])[0]
        res[k].append((float(a[mv[0] + later[0]]) if len(later) else np.nan, int(len(later) > 0)))
out = {}
for k, v in res.items():
    v = np.array(v); ok = np.isfinite(v[:, 0]); out[k] = dict(n=len(v), 정지_있는_에피소드=int(ok.sum()), 정지면적_중앙값=round(float(np.nanmedian(v[:, 0])), 3), 정지면적_평균=round(float(np.nanmean(v[:, 0])), 3), 이전_정지_비율_면적025=round(float(np.mean(v[ok, 0] < .25)), 2) if ok.any() else None)
out["_에피소드_마지막_박스면적_중앙값"] = round(float(np.nanmedian(areas_end)), 3)
json.dump(out, open(ROOT / "docs/v5/egvla2_fm/a3c/ch82_train_stoparea.json", "w"), ensure_ascii=False, indent=1); print(json.dumps(out, ensure_ascii=False, indent=1))

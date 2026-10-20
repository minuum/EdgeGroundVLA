#!/usr/bin/env python3
"""방향 오차 해부: 안 본 에피소드(fold0, LOPO strong_right)의 프레임별 사람 vs 모델(C, 드롭아웃30%, 120ep, K=8, EMA 새 값 0.5) 방향을 저장하고,
오차 분포·진행도별 오차·'사람 방향 전환 근처' 비중·최선/중앙/최악 에피소드와 썸네일을 만든다. 출력: docs/v5/egvla2_fm/error_anatomy.json, anatomy/*.jpg
실행: .venv_lerobot/bin/python3 scripts/eval_error_anatomy.py
"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import eval_subgoal_ablation as AB  # noqa: E402
import train_egvla2_s1 as S1  # noqa: E402
T = S1.T
DATA = ROOT / "ROS_action/mobile_vla_dataset_egvla2"; OUT = ROOT / "docs/v5/egvla2_fm/anatomy"; OUT.mkdir(parents=True, exist_ok=True)
GO, ALPHA_NEW = 0.2, 0.5


def ema_new(x, a):
    o = x.copy()
    for t in range(1, len(x)):
        o[t] = a * x[t] + (1 - a) * o[t - 1]
    return o


def angdeg(v):
    return np.degrees(np.arctan2(v[..., 1], v[..., 0]))


def diff_deg(a, b):
    d = (a - b + 180) % 360 - 180
    return np.abs(d)


def main():
    eps = S1.load_main(); cache = np.load(T.CACHE); E = AB.prep(eps, cache, 3); folds = S1.make_folds(eps)
    splits = {"fold0": folds[0], "lopo_strong_right": [i for i, e in enumerate(eps) if e["cell"].startswith("strong_right|")]}
    res = {}
    for sname, te_idx in splits.items():
        tr_idx = [i for i in range(len(E)) if i not in te_idx]; tr, te = T.to_t(E, tr_idx), T.to_t(E, te_idx)
        m = AB.fit(S1.ARCHS["C_pi0_adaln"], tr, 120, 0, drop=0.3)
        with torch.no_grad():
            torch.manual_seed(0); P = T.sample(m, te[0], te[1], k=8).mean(1)[:, 0, :2].cpu().numpy()
        gts = te[2][:, 0, :2].cpu().numpy(); o = 0; items = []
        for i in te_idx:
            n = len(E[i]["bbox"]); p = ema_new(P[o:o + n], ALPHA_NEW); h = gts[o:o + n]; o += n
            pm, hm = np.linalg.norm(p, axis=1), np.linalg.norm(h, axis=1); both = (pm > GO) & (hm > GO)
            err = np.where(both, diff_deg(angdeg(p), angdeg(h)), np.nan)
            ha = angdeg(h); chg = np.zeros(n, bool)
            for t in range(n):                      # 사람 방향 전환: 5프레임 전보다 30° 이상 달라짐(둘 다 이동 중)
                if t >= 5 and hm[t] > GO and hm[t - 5] > GO and diff_deg(ha[t], ha[t - 5]) > 30:
                    chg[max(0, t - 3):t + 4] = True
            with h5py.File(DATA / E[i]["name"], "r") as f:
                ts = np.array(f["frame_timestamps"], float)
            items.append(dict(name=E[i]["name"], cell=E[i]["cell"], n=n, dt=np.diff(ts, append=ts[-1]).clip(0, 1).round(3).tolist(), human=h.round(3).tolist(), pred=p.round(3).tolist(),
                              err=[None if np.isnan(x) else round(float(x), 1) for x in err], chg=chg.tolist(), mean_err=float(np.nanmean(err)) if both.any() else None, n_moving=int(both.sum())))
        res[sname] = items
        print(sname, "에피소드", len(items), "평균 오차", np.nanmean([x["mean_err"] for x in items if x["mean_err"] is not None]), flush=True)
    # 예시 에피소드 선정(이동 프레임 ≥50): 최선/중앙/최악
    ex = {}
    for sname, items in res.items():
        ok = sorted([x for x in items if x["n_moving"] >= 50 and x["mean_err"] is not None], key=lambda x: x["mean_err"])
        ex[sname] = dict(best=ok[0]["name"], median=ok[len(ok) // 2]["name"], worst=ok[-1]["name"])
        for role, nm in ex[sname].items():
            it = next(x for x in items if x["name"] == nm); e = np.array([np.nan if v is None else v for v in it["err"]])
            frames = [0, int(np.nanargmax(e)) if role == "worst" else it["n"] // 2, it["n"] - 1]
            with h5py.File(DATA / nm, "r") as f:
                for k, fr in enumerate(frames):
                    Image.fromarray(np.ascontiguousarray(f["images"][fr][:, :, ::-1])).resize((288, 162)).save(OUT / f"{sname}_{role}_{k}.jpg", quality=75)
            it["thumb_frames"] = frames
    (ROOT / "docs/v5/egvla2_fm/error_anatomy.json").write_text(json.dumps(dict(settings="C drop30 120ep, K=8, EMA new-weight 0.5, seed 0", splits=res, examples=ex), ensure_ascii=False))
    print(ex)


if __name__ == "__main__":
    main()

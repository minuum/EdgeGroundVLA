#!/usr/bin/env python3
"""E1b: E1 설계 수정. E1(상대 보정 라벨: 사람 방향 + 2Δ)은 모델이 Δ를 알 수 없어(이동한 장면의 특징·bbox만 보임) 사람 라벨과 충돌해 S가 안 올랐다. E1b는 라벨을 **타깃 절대 위치(cx)의 함수**로 만든다.
실제 프레임: 사람 방향과 서보 방향(sy=clip(gain*(cx-0.5),±0.7))의 혼합(λ=0.5, 이동 프레임만, 단위 크기). 합성(이동) 프레임: 순수 서보(λ=1), |cx-0.5|>thresh 이면 회전(타깃 쪽 ±1).
추가 지표 S_img: 시험 에피소드의 기준 프레임에서 이미지 특징·bbox를 함께 ±45px 옮긴 입력의 예측 방향 차/2(교정 방향 양수).
변형: E1b / E1b+E8 / E1b+E8+E5 및 튜닝 변형 E1b_tuned / E1b+E8+E5_tuned (thresh=0.22, gain=2.6).
출력: docs/v5/egvla2_fm/gap_e1b.json. 실행: .venv_lerobot/bin/python3 scripts/gap_e1b.py (SMOKE=1: epoch 2·분할 1개)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
import gap_e1 as G1  # noqa: E402
import gap_suite as GS  # noqa: E402
import eval_subgoal_ablation as AB  # noqa: E402
import train_egvla2_s1 as S1  # noqa: E402
import robovlm_nav.models.egvla2_fm_heads as HD  # noqa: E402
T = S1.T
DEV, BS = GS.DEV, GS.BS
SMOKE = bool(os.getenv("SMOKE")); EPOCHS = 2 if SMOKE else 120
OUT = ROOT / "docs/v5/egvla2_fm" / ("gap_e1b_smoke.json" if SMOKE else "gap_e1b.json")
VARIANTS = {
    "E1b": dict(bright=False, edge_w=1.0, thresh=0.30, gain=2.0),
    "E1b+E8": dict(bright=True, edge_w=1.0, thresh=0.30, gain=2.0),
    "E1b+E8+E5": dict(bright=True, edge_w=3.0, thresh=0.30, gain=2.0),
    "E1b_tuned": dict(bright=False, edge_w=1.0, thresh=0.22, gain=2.6),
    "E1b+E8+E5_tuned": dict(bright=True, edge_w=3.0, thresh=0.22, gain=2.6),
}
LAM_REAL = 0.5


def servo_unit(cx, gain=2.0):
    sy = float(np.clip(gain * (cx - 0.5), -0.7, 0.7)); return np.array([np.sqrt(max(0.0, 1 - sy * sy)), sy], np.float32)


def real_label(a_chunk, cx, gain=2.0):
    """실제 프레임: 이동이면 사람 단위 방향과 서보 방향 혼합, 회전·정지는 하이브리드 의미 라벨 유지."""
    hw = GS.hw_relabel(torch.from_numpy(a_chunk))                              # (H,3)
    out = hw.clone().numpy()
    for h in range(len(out)):
        if np.hypot(out[h, 0], out[h, 1]) > 0.5:
            u = (1 - LAM_REAL) * out[h, :2] + LAM_REAL * servo_unit(cx, gain); out[h, :2] = u / max(np.linalg.norm(u), 1e-6)
    return out.astype(np.float32)


def aug_label(cx_new, thresh=0.30, gain=2.0):
    if abs(cx_new - 0.5) > thresh:
        return np.array([0, 0, 1.0 if cx_new > 0.5 else -1.0], np.float32)
    u = servo_unit(cx_new, gain); return np.array([u[0], u[1], 0.0], np.float32)


def build(E, by_name, test_names, thresh=0.30, gain=2.0):
    z = np.load(ROOT / "docs/v5/egvla2_fm/aug_features.npz", allow_pickle=True)
    base = [str(x) for x in z["base_names"]]; kinds = z["kind"]; bidx = z["base_idx"]; par = z["param"]; ab = z["bbox"]; af = z["feat"].astype(np.float32)
    rows = []
    for k in range(len(kinds)):
        n, t = base[int(bidx[k])].rsplit("|", 1); t = int(t); ei = by_name.get(n)
        if ei is None:
            continue
        e = E[ei]; win = e["bbox"][t].copy(); kind = str(kinds[k]); p = float(par[k])
        if kind == "shift":
            d = p / 320.0
            for s_ in (0, 4):
                win[s_] = np.clip(win[s_] + d, 0, 1)
            win[8:12] = ab[k]; cx_new = float(ab[k][0]) if ab[k][3] > 0.5 else float(np.clip(e["bbox"][t][8] + d, 0, 1)); lab = aug_label(cx_new, thresh=thresh, gain=gain)
        else:
            win[8:12] = ab[k]; lab = real_label(e["chunk"][t], float(e["bbox"][t][8]), gain=gain)[0]
        rows.append(dict(ei=ei, kind=kind, p=p, win=win.astype(np.float32), feat=af[k], lab=np.repeat(lab[None], HD.H, 0).astype(np.float32), name=n, t=t))
    return rows


def fit(tr, aug, edge_w, gain=2.0, seed=0, drop=0.3):
    torch.manual_seed(seed)
    # 실제 프레임 라벨(절대 cx 기반 서보 혼합)
    cx = tr[0][:, 8].cpu().numpy(); ch = tr[2].cpu().numpy(); yr = np.stack([real_label(ch[i], cx[i], gain=gain) for i in range(len(ch))]); yr = torch.from_numpy(yr).to(DEV)
    bb = torch.cat([tr[0]] + ([torch.from_numpy(np.stack([r["win"] for r in aug])).to(DEV)] if aug else []))
    ft = torch.cat([tr[1]] + ([torch.from_numpy(np.stack([r["feat"] for r in aug])).to(DEV)] if aug else []))
    y = torch.cat([yr] + ([torch.from_numpy(np.stack([r["lab"] for r in aug])).to(DEV)] if aug else []))
    c = bb[:, 8]; w = torch.where((c < 0.25) | (c > 0.75), torch.tensor(edge_w, device=DEV), torch.tensor(1.0, device=DEV))
    m = HD.PiZeroFM(bb.shape[1], ft.shape[1]).to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2); n = len(bb)
    for _ in range(EPOCHS):
        m.train(); perm = torch.randperm(n, device=DEV)
        for s in range(0, n, BS):
            i = perm[s:s + BS]; b = bb[i] * (torch.rand(len(i), 1, device=DEV) > drop).float(); a = y[i]
            x0 = torch.randn_like(a); t = torch.rand(len(i), device=DEV); xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * a
            per = ((m(b, ft[i], xt, t) - (a - x0)) ** 2).mean((1, 2)); l = (per * w[i]).mean() / w[i].mean(); opt.zero_grad(); l.backward(); opt.step()
    m.eval(); return m


def s_img(m, aug_test):
    """시험 에피소드 기준 프레임: 이미지 특징·bbox를 함께 +45/-45px 옮긴 입력의 예측 방향 차/2."""
    by = {}
    for r in aug_test:
        if r["kind"] == "shift" and r["p"] in (45.0, -45.0):
            by.setdefault((r["name"], r["t"]), {})[r["p"]] = r
    pairs = [(v[45.0], v[-45.0]) for v in by.values() if 45.0 in v and -45.0 in v]
    if not pairs:
        return None
    def ang_of(rs):
        P = GS.predict(m, torch.from_numpy(np.stack([r["win"] for r in rs])).to(DEV), torch.from_numpy(np.stack([r["feat"] for r in rs])).to(DEV)); return GS.ang(P[:, 0], P[:, 1]), np.hypot(P[:, 0], P[:, 1])
    ap, mp = ang_of([a for a, b in pairs]); am, mm = ang_of([b for a, b in pairs]); ok = (mp > 0.2) & (mm > 0.2)
    return float(np.mean(((ap - am + 180) % 360 - 180)[ok] / 2)) if ok.any() else None


def main():
    eps = S1.load_main(); cache = np.load(T.CACHE); E = AB.prep(eps, cache, 3); folds = S1.make_folds(eps); by_name = {e["name"]: i for i, e in enumerate(eps)}
    splits = {"fold0": folds[0], "lopo_center": [i for i, e in enumerate(eps) if e["cell"].startswith("center|")], "lopo_strong_right": [i for i, e in enumerate(eps) if e["cell"].startswith("strong_right|")]}
    
    aug_cache = {}
    for vn, kw in VARIANTS.items():
        cfg = (kw["thresh"], kw["gain"])
        if cfg not in aug_cache:
            aug_cache[cfg] = build(E, by_name, None, thresh=cfg[0], gain=cfg[1])

    live = GS.live_inputs(); res = json.loads(OUT.read_text()) if OUT.exists() else {}
    for name, te_idx in splits.items():
        res.setdefault(name, {}); tr_idx = [i for i in range(len(E)) if i not in te_idx]; tr, te = T.to_t(E, tr_idx), T.to_t(E, te_idx); tes = set(te_idx)
        for vn, kw in VARIANTS.items():
            if vn in res[name]:
                continue
            allrows = aug_cache[(kw["thresh"], kw["gain"])]
            aug = [r for r in allrows if r["ei"] not in tes and (r["kind"] == "shift" or kw["bright"])]
            m = fit(tr, aug, kw["edge_w"], gain=kw["gain"]); r = GS.split_metrics(m, te, live); r["S_img"] = s_img(m, [x for x in allrows if x["ei"] in tes]); r["n_aug"] = len(aug)
            res[name][vn] = r; OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
            print(name, vn, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k != "live"}, "live", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r["live"].items()}, flush=True)
        if SMOKE:
            break


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""plan §8.4: V6(225ep, 이산 명령 타깃)에서 '명령 적분 → 상대 waypoint' 타깃으로 FM을 학습해 본다.

같은 분할·같은 입력에서 두 타깃을 비교한다 (우열 주장이 아니라 waypoint 타깃이 학습 가능한지 확인):
  - chunk:    다음 H=8 프레임의 명령 (lx,ly,az)/1.15
  - waypoint: 위 명령의 누적합 (프레임 단위 dead-reckoning, 회전-이동 결합 없음) / (H*1.0)  → 값 범위 [-1,1] 근처
공통 평가 지표: 마지막(H번째) 상대 waypoint(누적합)의 축별 R² 와 MSE (chunk 타깃 예측은 누적합으로 변환해 같은 척도로 비교).
한계: V6 명령은 8-class LUT 값(±1.15)이라 연속성 정보는 없다. 속도 단위/시간 간격은 미보정(프레임 단위).
입력: bbox 3프레임 창 + F2 풀링 768-d (docs/v5/bbox_nav_owl/v6_florence2_cache.npz). 5-fold 에피소드 hold-out.
실행: .venv_lerobot/bin/python3 scripts/train_v6_fm_waypoint.py   (EPOCHS, SEEDS 환경변수)
"""
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import train_egvla2_fm_head as T  # noqa: E402

CACHE = ROOT / "docs/v5/bbox_nav_owl/v6_florence2_cache.npz"
DATASET = ROOT / "docs/v5/bbox_nav_owl/bbox_dataset_v6_owl.json"
OUT = ROOT / "docs/v5/egvla2_fm/summary_v6_waypoint.json"
EPOCHS = int(os.environ.get("EPOCHS", 60))
SEEDS = [int(x) for x in os.environ.get("SEEDS", "0").split(",")]
H, W = T.H, T.WINDOW
ARCHS = {k: T.ARCH[k] for k in ("A_mlp_fm", "C_pi0_adaln")}


def load(mode):
    ds, cache, eps = json.loads(DATASET.read_text()), np.load(CACHE), []
    for e in ds:
        with h5py.File(e["episode"], "r") as f:
            a = np.array(f["actions"], dtype=np.float32) / 1.15
        key = e["episode"]
        bb, ft = cache[f"{key}|bbox"].astype(np.float32), cache[f"{key}|feat"].astype(np.float32)
        idx = [fr["frame_idx"] for fr in e["frames"]]
        a = a[idx]; n = len(a)
        win = np.stack([np.concatenate([bb[max(0, t - W + 1 + k)] for k in range(W)]) for t in range(n)])
        chunk = np.zeros((n, H, 3), np.float32)
        for t in range(n):
            seg = a[t:t + H]; chunk[t, :len(seg)] = seg
        tgt = chunk if mode == "chunk" else np.cumsum(chunk, axis=1) / H
        eps.append(dict(name=Path(key).stem, bbox=win, feat=ft, chunk=tgt, raw_chunk=chunk))
    return eps


def final_wp(arr, mode):        # (N,H,3) 예측/정답 → 마지막 waypoint (N,3) (chunk는 누적합 후 /H)
    return arr[:, -1] if mode == "waypoint" else arr.sum(1) / H


def main():
    T.EPOCHS = EPOCHS
    res = {}
    for mode in ("chunk", "waypoint"):
        eps = load(mode)
        n = len(eps); folds = [list(range(i, n, 5)) for i in range(5)]
        print(f"[{mode}] {n}ep, {sum(len(e['bbox']) for e in eps)}프레임", flush=True)
        for name, cls in ARCHS.items():
            rows, t0 = [], time.time()
            for seed in SEEDS:
                for fi, te_idx in enumerate(folds):
                    tr_idx = [i for i in range(n) if i not in te_idx]
                    tr, te = T.to_t(eps, tr_idx), T.to_t(eps, te_idx)
                    m, curve = T.train_one(cls, tr, te, seed, epochs=EPOCHS)
                    S = T.sample(m, te[0], te[1]).mean(1)
                    pw, gw = final_wp(S, mode).cpu().numpy(), final_wp(te[2], mode).cpu().numpy()
                    base = final_wp(tr[2].mean(0, keepdim=True).expand(len(te[2]), -1, -1), mode).cpu().numpy()
                    mse = ((pw - gw) ** 2).mean(0); var = gw.var(0)
                    rows.append(dict(seed=seed, fold=fi, mse=mse.tolist(), r2=(1 - mse / np.maximum(var, 1e-8)).tolist(),
                                     base_mse=((base - gw) ** 2).mean(0).tolist(),
                                     test_flow=curve[-1][2], train_flow=curve[-1][1]))
            res[f"{mode}|{name}"] = dict(
                mse=np.mean([r["mse"] for r in rows], 0).tolist(), r2=np.mean([r["r2"] for r in rows], 0).tolist(),
                base_mse=np.mean([r["base_mse"] for r in rows], 0).tolist(),
                test_flow=float(np.mean([r["test_flow"] for r in rows])), train_flow=float(np.mean([r["train_flow"] for r in rows])))
            print(f"  {name}: R²(lx,ly,az)={np.round(res[f'{mode}|{name}']['r2'], 3)} base_mse={np.round(res[f'{mode}|{name}']['base_mse'], 3)}"
                  f" mse={np.round(res[f'{mode}|{name}']['mse'], 3)} ({(time.time() - t0) / 60:.1f}분)", flush=True)
    OUT.write_text(json.dumps(dict(epochs=EPOCHS, seeds=SEEDS, horizon=H, results=res), indent=1, ensure_ascii=False))
    print("저장:", OUT)


if __name__ == "__main__":
    main()

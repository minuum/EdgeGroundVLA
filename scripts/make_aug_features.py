#!/usr/bin/env python3
"""E1(yaw-shift)·E8(밝기) 증강 특징 추출: 학습 프레임 일부를 변환한 이미지에서 Florence-2 bbox·특징을 다시 뽑는다(서빙과 같은 FusedF2Extractor, fp16=학습 캐시와 동일).
E1: 가로 이동 s ∈ {-90,-45,+45,+90} px@폭320(실기 회전 한 번 ≈ -68px ≈ Δcx -0.21로 보정, 1280 폭에서는 4배 픽셀, 가장자리 복제). E8: 밝기 ×{0.9, 1.12, 1.25}.
기준 프레임: 에피소드별로 균등 추출한 1,500장(이동 프레임 우선). 출력: docs/v5/egvla2_fm/aug_features.npz (kind, base_idx, param, bbox, feat). 재개 가능(체크포인트 저장).
실행: HF_HUB_OFFLINE=1 .venv/bin/python3 scripts/make_aug_features.py    (.venv는 train_egvla2_s1 import 불가 → 캐시 키와 H5에서 직접 읽음)
"""
import io
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from robovlm_nav.serve.egvla2_fm_policy import FusedF2Extractor  # noqa: E402

DATA = ROOT / "ROS_action/mobile_vla_dataset_egvla2"
CACHE = ROOT / "docs/v5/egvla2_fm/v3_florence2_cache.npz"
OUT = ROOT / "docs/v5/egvla2_fm/aug_features.npz"
SHIFTS = (-90, -45, 45, 90); BRIGHT = (0.9, 1.12, 1.25); N_BASE = 1500


def shift_img(rgb, s_px320):
    s = int(round(s_px320 * rgb.shape[1] / 320)); out = np.empty_like(rgb)
    if s > 0:
        out[:, s:] = rgb[:, :-s]; out[:, :s] = rgb[:, :1]
    elif s < 0:
        out[:, :s] = rgb[:, -s:]; out[:, s:] = rgb[:, -1:]
    else:
        out = rgb.copy()
    return out


def main():
    cache = np.load(CACHE); names = sorted({k.split("|")[0] for k in cache.files if k.endswith("|bbox")}); names = [n for n in names if (DATA / n).exists()]
    rng = np.random.default_rng(0); per = max(1, N_BASE // len(names)); base = []
    for n in names:
        nfr = len(cache[f"{n}|bbox"]); idx = rng.choice(nfr, min(per, nfr), replace=False); base += [(n, int(t)) for t in sorted(idx)]
    base = base[:N_BASE]; print("기준 프레임", len(base), "에피소드", len(names), flush=True)
    done = {}
    if OUT.exists():
        z = np.load(OUT, allow_pickle=True); done = {(str(k), int(b), float(p)): (bb, f) for k, b, p, bb, f in zip(z["kind"], z["base_idx"], z["param"], z["bbox"], z["feat"])}
    ex = FusedF2Extractor(dtype=torch.float16); kinds, bidx, params, bbs, fts = [], [], [], [], []
    def save():
        allk = list(done.keys()); np.savez_compressed(OUT, kind=np.array([k[0] for k in allk]), base_idx=np.array([k[1] for k in allk]), param=np.array([k[2] for k in allk], np.float32),
                                                      bbox=np.array([done[k][0] for k in allk], np.float32), feat=np.array([done[k][1] for k in allk], np.float16), base_names=np.array([f"{n}|{t}" for n, t in base]))
    cnt = 0; cur = None
    for bi, (n, t) in enumerate(base):
        todo = [("shift", float(s)) for s in SHIFTS if ("shift", bi, float(s)) not in done] + [("bright", float(b)) for b in BRIGHT if ("bright", bi, float(b)) not in done]
        if not todo:
            continue
        with h5py.File(DATA / n, "r") as f:
            rgb = np.ascontiguousarray(np.asarray(f["images"][t])[:, :, ::-1])
        pils = []
        for kind, p in todo:
            img = shift_img(rgb, p) if kind == "shift" else np.clip(rgb.astype(np.float32) * p, 0, 255).astype(np.uint8)
            pils.append(Image.fromarray(np.ascontiguousarray(img)).convert("RGB"))
        bb_, ff_ = ex._P.run_batch(ex.model, ex.proc, pils, dtype=ex.dtype)           # 변형 7개를 한 번에(배치)
        for (kind, p), b, ft in zip(todo, bb_, ff_):
            done[(kind, bi, p)] = (b.astype(np.float32), ft.astype(np.float32)); cnt += 1
        if bi % 50 == 0:
            save(); print(bi, "/", len(base), "추출", cnt, flush=True)
    save(); print("완료", len(done))


if __name__ == "__main__":
    main()

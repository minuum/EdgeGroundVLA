#!/usr/bin/env python3
"""로봇 서버가 보낸 11/13 실기 프레임(docs/v5/figures/objtest/{sid}_{i}.jpg, 1280×720)이 이 서버의 H5 프레임과 같은지 전수 확인(PSNR). 출력 docs/v5/egvla2_fm/a3c/real_frames_1113_verify.json"""
import glob, json, os
from pathlib import Path
import h5py, numpy as np
from PIL import Image
ROOT = Path(__file__).resolve().parent.parent; F = ROOT / "docs/v5/figures/objtest"; ps = []; miss = 0; sizes = set()
for f in sorted(glob.glob(str(ROOT / "docs/inference_sessions/session_20261113_*.h5"))):
    sid = os.path.basename(f)[8:-3]
    with h5py.File(f, "r") as h: imgs = h["observations/images"][:]
    for i, im in enumerate(imgs):
        p = F / f"{sid}_{i}.jpg"
        if not p.exists(): miss += 1; continue
        a = Image.open(p).convert("RGB"); sizes.add(a.size); b = Image.fromarray(np.ascontiguousarray(im)).convert("RGB")
        if a.size != b.size: b = b.resize(a.size)
        mse = np.mean((np.asarray(a, float) - np.asarray(b, float)) ** 2); ps.append(99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse))
r = dict(frames=len(ps), missing=miss, sizes=[list(s) for s in sizes], psnr_min=round(float(min(ps)), 1), psnr_median=round(float(np.median(ps)), 1), bytes=sum(os.path.getsize(p) for p in glob.glob(str(F / "*.jpg"))))
json.dump(r, open(ROOT / "docs/v5/egvla2_fm/a3c/real_frames_1113_verify.json", "w"), indent=1); print(r)

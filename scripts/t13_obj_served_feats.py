#!/usr/bin/env python3
"""11/13 65세션을 '서빙 설계 그대로'(박스=gray basket 고정 문구, JPEG q95 왕복, fp32) 추출기에 통과시켜 프레임별
(gray basket 박스, 이미지 풀링, 지시어 문구 특징, gray basket 문구 특징)을 저장. 출력 runs/egvla2_fm/t13_obj_served_cache.npz"""
import io, sys
from pathlib import Path
import h5py, numpy as np, torch
from PIL import Image
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from robovlm_nav.serve.egvla2_text_policy import FusedF2TextExtractor
ext = FusedF2TextExtractor(dtype=torch.float32); out = {}
for f in sorted(ROOT.glob("docs/inference_sessions/session_20261113_*.h5")):
    sid = f.stem.replace("session_", "")
    with h5py.File(f, "r") as h: imgs = h["observations/images"][:]; ins = str(h.attrs["instruction"])
    B, V, TI, TG = [], [], [], []
    for im in imgs:
        buf = io.BytesIO(); Image.fromarray(np.ascontiguousarray(im).astype(np.uint8)).save(buf, "JPEG", quality=95); img = np.array(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"))
        ext.set_phrase(ins); bb, ft = ext(img); B.append(bb); V.append(ft[:768]); TI.append(ft[768:])
        ext.set_phrase("gray basket"); _, ft2 = ext(img); TG.append(ft2[768:])
    out[f"{sid}|bbox"] = np.array(B); out[f"{sid}|vis"] = np.array(V); out[f"{sid}|txt_ins"] = np.array(TI); out[f"{sid}|txt_gray"] = np.array(TG); print(sid, flush=True)
np.savez(ROOT / "runs/egvla2_fm/t13_obj_served_cache.npz", **out); print("saved")

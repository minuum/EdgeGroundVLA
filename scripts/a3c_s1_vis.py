#!/usr/bin/env python3
"""S1: 입력 전처리(해상도·가로세로비)를 바꾸면 이미지 특징(vis)이 어떻게 변하는지, 그리고 A3 헤드의 '가운데 0 명령'이 바뀌는지.
11/13 가운데 세션(의자 5·바구니 5·화분 5) 프레임을 4가지 전처리로 Florence-2 인코더에 넣어 풀링 특징 저장: sq768(현재, 학습과 동일·대조군) / lb768(가로세로비 유지, 검은 여백) / sq1024 / sq512.
JPEG q95 왕복 후 입력. 출력 runs/egvla2_fm/a3c/s1_vis.npz (헤드 평가는 a3c_eval.py)."""
import io, json, sys
from pathlib import Path
import h5py, numpy as np, torch
from PIL import Image
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
from robovlm_nav.serve.egvla2_fm_policy import FusedF2Extractor
D = json.load(open(ROOT / "docs/v5/egvla2_fm/objtest_data.json"))
G = {"의자·가운데": ("가운데(G3)", "의자"), "바구니·가운데": ("가운데(G3)", "바구니"), "화분·가운데": ("가운데(G3)", "화분")}
sids = {k: [x["sid"] for x in D if (x["pos"], x["obj"]) == v][:5] for k, v in G.items()}
ext = FusedF2Extractor(dtype=torch.float32); proc = ext.proc; model = ext.model
def pil_of(im):
    b = io.BytesIO(); Image.fromarray(np.ascontiguousarray(im).astype(np.uint8)).save(b, "JPEG", quality=95); return Image.open(io.BytesIO(b.getvalue())).convert("RGB")
def letterbox(p):
    s = max(p.size); c = Image.new("RGB", (s, s), (0, 0, 0)); c.paste(p, ((s - p.width) // 2, (s - p.height) // 2)); return c
@torch.no_grad()
def vis_of(pil, size):
    proc.image_processor.size = {"height": size, "width": size}
    pv = proc(text=["<CAPTION>"], images=[pil], return_tensors="pt")["pixel_values"].to(ext.device, torch.float32)
    return model._encode_image(pv).float().mean(1)[0].cpu().numpy()
out = {}
for k, ss in sids.items():
    for sid in ss:
        with h5py.File(ROOT / f"docs/inference_sessions/session_{sid}.h5", "r") as h: imgs = h["observations/images"][:]
        for name, f, size in (("sq768", lambda p: p, 768), ("lb768", letterbox, 768), ("sq1024", lambda p: p, 1024), ("sq512", lambda p: p, 512)):
            out[f"{sid}|{name}"] = np.stack([vis_of(f(pil_of(im)), size) for im in imgs]).astype(np.float32)
        print(k, sid, len(imgs), flush=True)
np.savez(ROOT / "runs/egvla2_fm/a3c/s1_vis.npz", **out, _groups=json.dumps(sids))
print("saved")

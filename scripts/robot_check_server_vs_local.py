#!/usr/bin/env python3
"""로봇/젯슨용 서버 경로 점검(구동 없음): 같은 프레임을 (a) 이 프로세스에서 직접 Florence-2+정책으로, (b) 실행 중인 서버(/predict, JPEG q95)로 보내 bbox와 방향을 비교한다.
기준: 서버 vs 로컬 차이가 '로컬 두 번(다른 seed) 차이'(K=8 샘플링 잡음)와 비슷하면 합격. 서버는 VLA_EGVLA2_EMA=0 으로 띄운다(평활이 점검을 흐림).
사용: python3 scripts/robot_check_server_vs_local.py --h5 <에피소드.h5 또는 폴더> --ckpt runs/egvla2_fm/deploy/C_pi0_adaln_drop30_e120.pt --server http://127.0.0.1:8001 [--n 20] [--quality 95] [--dtype fp32] [--api-key KEY]
H5는 BGR 저장(수집 포맷)로 보고 [:,:,::-1] RGB 변환(학습과 동일). 출력: 표 + --out JSON.
"""
import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import requests
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from robovlm_nav.serve.egvla2_fm_policy import EgVla2FMPolicy, FusedF2Extractor  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--h5", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--server", default="http://127.0.0.1:8001")
ap.add_argument("--n", type=int, default=20); ap.add_argument("--quality", type=int, default=95); ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32")
ap.add_argument("--api-key", default=__import__("os").getenv("VLA_API_KEY", "")); ap.add_argument("--out", default="/tmp/server_vs_local.json")
A = ap.parse_args()


def frames():
    p = Path(A.h5); fs = [p] if p.is_file() else sorted(p.glob("*.h5"))[:6]; out = []
    for f in fs:
        with h5py.File(f, "r") as h:
            im = h["observations"]["images"] if "observations" in h else h["images"]
            idx = np.linspace(5, len(im) - 6, max(1, A.n // len(fs))).astype(int)
            out += [(f.name, int(i), np.ascontiguousarray(im[i][:, :, ::-1])) for i in idx]
    return out[:A.n]


def ang(a, b):
    a, b = np.array(a[:2], float), np.array(b[:2], float)
    if np.linalg.norm(a) < 0.2 or np.linalg.norm(b) < 0.2:
        return None
    return float(np.degrees(np.arccos(np.clip(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1))))


hdr = {"X-API-Key": A.api_key} if A.api_key else {}
ext = FusedF2Extractor(dtype=torch.float32 if A.dtype == "fp32" else torch.float16)
pol = EgVla2FMPolicy(A.ckpt, k_samples=8, ema_alpha=0.0)
rows = []
for name, i, rgb in frames():
    bbox, feat = ext(rgb)
    la, lb = [], []
    for seed, store in ((0, la), (1, lb)):
        pol.reset(); torch.manual_seed(seed); store.append(pol.step(bbox, feat)["first"])
    buf = io.BytesIO(); Image.fromarray(rgb).save(buf, "JPEG", quality=A.quality)
    requests.post(A.server + "/reset", headers=hdr, timeout=10)
    t0 = time.time(); r = requests.post(A.server + "/predict", json={"image": base64.b64encode(buf.getvalue()).decode(), "instruction": "basket"}, headers=hdr, timeout=60); ms = (time.time() - t0) * 1000
    r.raise_for_status(); j = r.json(); ac = j.get("action_cont") or {}
    sb = ac.get("bbox_f2") or [j["bbox"]["cx"], j["bbox"]["cy"], j["bbox"]["area"], float(j["bbox"]["has_bbox"])]
    sf = ac.get("chunk_first") or j.get("chunk_first")
    rows.append(dict(file=name, frame=i, bbox_maxdiff=float(np.abs(np.array(sb) - bbox).max()), noise_deg=ang(la[0], lb[0]), diff_deg=ang(la[0], sf) if sf else None,
                     server_ms=ms, server_latency_ms=j.get("latency_ms"), subgoal_missing=j.get("subgoal_missing")))
    print(rows[-1], flush=True)
nz = [r["noise_deg"] for r in rows if r["noise_deg"] is not None]; df = [r["diff_deg"] for r in rows if r["diff_deg"] is not None]
summ = dict(n=len(rows), bbox_maxdiff_max=max(r["bbox_maxdiff"] for r in rows), local_sampling_noise_deg=float(np.mean(nz)) if nz else None, server_vs_local_deg=float(np.mean(df)) if df else None,
            server_latency_ms_mean=float(np.mean([r["server_latency_ms"] for r in rows if r["server_latency_ms"]])) if rows else None, quality=A.quality, dtype=A.dtype)
summ["pass"] = bool(summ["server_vs_local_deg"] is not None and summ["bbox_maxdiff_max"] < 0.01 and summ["server_vs_local_deg"] <= 1.5 * (summ["local_sampling_noise_deg"] or 0) + 0.5)
print(json.dumps(summ, indent=1, ensure_ascii=False)); Path(A.out).write_text(json.dumps(dict(summary=summ, rows=rows), indent=1, ensure_ascii=False))

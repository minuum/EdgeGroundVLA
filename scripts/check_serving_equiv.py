#!/usr/bin/env python3
"""서빙 입력 경로 동등성 점검: 같은 H5 프레임을 (dtype fp16/fp32) × (원본 / JPEG q80 왕복(대시보드 전송 경로))로 Florence-2에 넣어,
학습 캐시(fp16·원본 프레임)와 bbox·특징·최종 액션 방향을 비교한다. BGR→RGB는 학습(precompute)과 같은 [:,:,::-1].
실행: HF_HUB_OFFLINE=1 .venv/bin/python3 -I scripts/check_serving_equiv.py [--ckpt runs/egvla2_fm/deploy/C_pi0_adaln_drop30_e120.pt] [--n 20]
출력: docs/v5/egvla2_fm/serving_equiv.json
"""
import argparse
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
import precompute_v6_florence2_fused as P  # noqa: E402
from transformers import AutoModelForCausalLM, AutoProcessor  # noqa: E402
from robovlm_nav.serve.egvla2_fm_policy import EgVla2FMPolicy  # noqa: E402
from florence2_bbox_adapter import florence2_output_to_bbox  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default="runs/egvla2_fm/deploy/C_pi0_adaln_drop30_e120.pt"); ap.add_argument("--n", type=int, default=20); ap.add_argument("--q", default="80"); ap.add_argument("--fp16only", action="store_true"); ap.add_argument("--out", default="docs/v5/egvla2_fm/serving_equiv.json")
A = ap.parse_args(); QS = [int(x) for x in A.q.split(",")]
DT = [("fp16", torch.float16)] + ([] if A.fp16only else [("fp32", torch.float32)])
DATA = ROOT / "ROS_action/mobile_vla_dataset_egvla2"
cache = np.load(ROOT / "docs/v5/egvla2_fm/v3_florence2_cache.npz")
meta = json.load(open(ROOT / "docs/v5/egvla2_fm/meta/egvla2_v3_main.json"))["files"]
rng = np.random.RandomState(0); eps = [meta[i]["name"] for i in rng.choice(len(meta), 5, replace=False)]


def run(model, proc, pils, dt):
    inp = proc(text=[P.TASK + P.PHRASE] * len(pils), images=pils, return_tensors="pt")
    pv = inp["pixel_values"].to(P.DEV, dt); ids = inp["input_ids"].to(P.DEV)
    with torch.no_grad():
        f = model._encode_image(pv); e = model.get_input_embeddings()(ids); e, _ = model._merge_input_ids_with_image_features(f, e)
        out = model.language_model.generate(input_ids=None, inputs_embeds=e, max_new_tokens=128, num_beams=3, do_sample=False)
    texts = proc.batch_decode(out, skip_special_tokens=False); boxes = []
    for t, pil in zip(texts, pils):
        parsed = proc.post_process_generation(t, task=P.TASK, image_size=(pil.width, pil.height))[P.TASK]
        b = florence2_output_to_bbox(parsed.get("bboxes", []), pil.width, pil.height); boxes.append([b["cx"], b["cy"], b["area"], float(b["has_bbox"])])
    return np.array(boxes, np.float32), f.float().mean(1).cpu().numpy()


def jpeg(pil, q=80):
    b = io.BytesIO(); pil.save(b, "JPEG", quality=q); return Image.open(io.BytesIO(b.getvalue())).convert("RGB")


frames = []
for n in eps:
    with h5py.File(DATA / n, "r") as f:
        idx = np.linspace(5, len(f["images"]) - 6, A.n // 5).astype(int)
        for i in idx:
            frames.append((n, int(i), np.ascontiguousarray(f["images"][i][:, :, ::-1])))
pils = [Image.fromarray(x[2]).convert("RGB") for x in frames]
ref_b = np.stack([cache[f"{n}|bbox"][i] for n, i, _ in frames]); ref_f = np.stack([cache[f"{n}|feat"][i].astype(np.float32) for n, i, _ in frames])
pol = EgVla2FMPolicy(ROOT / A.ckpt, k_samples=8)
res = {}
for dtn, dt in DT:
    model = AutoModelForCausalLM.from_pretrained(P.MODEL, trust_remote_code=True, torch_dtype=dt).to(P.DEV).eval(); proc = AutoProcessor.from_pretrained(P.MODEL, trust_remote_code=True)
    for src, ps in ((("raw", pils),) + tuple((f"jpeg{q}", [jpeg(p, q) for p in pils]) for q in QS)):
        bb, ft = [], []
        for k in range(0, len(ps), 4):
            b, f_ = run(model, proc, ps[k:k + 4], dt); bb.append(b); ft.append(f_)
        bb, ft = np.concatenate(bb), np.concatenate(ft)
        cos = (ft * ref_f).sum(1) / (np.linalg.norm(ft, axis=1) * np.linalg.norm(ref_f, axis=1))
        ang = []
        for j in range(len(frames)):
            torch.manual_seed(0); pol.reset(); a = pol.step(ref_b[j], ref_f[j])["first"]
            torch.manual_seed(0); pol.reset(); b = pol.step(bb[j], ft[j].astype(np.float32))["first"]
            va, vb = np.array(a[:2]), np.array(b[:2])
            if np.linalg.norm(va) > 0.2 and np.linalg.norm(vb) > 0.2:
                ang.append(float(np.degrees(np.arccos(np.clip(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)), -1, 1)))))
        res[f"{dtn}|{src}"] = dict(bbox_maxdiff=float(np.abs(bb - ref_b).max()), bbox_cx_mad=float(np.abs(bb[:, 0] - ref_b[:, 0]).mean()), has_bbox_mismatch=int((bb[:, 3] != ref_b[:, 3]).sum()),
                                   feat_cos_min=float(cos.min()), feat_cos_mean=float(cos.mean()), action_angle_diff_mean=float(np.mean(ang)) if ang else None, action_angle_diff_max=float(np.max(ang)) if ang else None, n=len(frames))
        print(dtn, src, res[f"{dtn}|{src}"], flush=True)
    del model
(ROOT / A.out).write_text(json.dumps(dict(ckpt=A.ckpt, episodes=eps, results=res), indent=1, ensure_ascii=False))

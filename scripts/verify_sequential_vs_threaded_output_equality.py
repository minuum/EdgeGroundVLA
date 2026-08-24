#!/usr/bin/env python3
"""grounder.run()(OWL-v2) / enc.encode_image()(Kosmos-2 vision) 순차 vs
ThreadPoolExecutor 병렬 실행의 출력값이 완전히 동일한지 검증.

minum 지적(2026-08-24): 지금까지 측정은 속도(지연)만 쟀고, 두 실행 방식의
출력값(bbox 좌표, vis_feat)이 같은지는 확인 안 했음. 이론상 두 호출이
서로 결과를 안 쓰고 공유 가변 상태 갱신은 둘 다 끝난 뒤에만 일어나므로 같아야
하지만, 실제로 검증(assert 비교)해야 확신 가능.
"""
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("VLA_OWLV2_FP16", "1")
os.environ.setdefault("VLA_OWLV2_THRESH", "0.2")
os.environ.setdefault("VLA_OWLV2_AREA_SCALE", "3.0")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from PIL import Image

from robovlm_nav.serve.stage2_v2_inference_server import (
    DEFAULT_VLM, OwlV2Grounder, Stage1Encoder,
)

STAGE1_PATH = ROOT / "runs/v5_nav/mlp/stage1_v3_5cls/stage1_v3_5cls_owl_projs.pt"
N_FRAMES = 30
PHRASE = "gray basket"


def make_frame(seed):
    rng = np.random.RandomState(seed)
    arr = (rng.rand(720, 1280, 3) * 255).astype(np.uint8)
    return arr, Image.fromarray(arr).convert("RGB")


def bbox_equal(a, b):
    if a.keys() != b.keys():
        return False, f"key mismatch: {a.keys()} vs {b.keys()}"
    for k in a:
        va, vb = a[k], b[k]
        if isinstance(va, float) and isinstance(vb, float):
            if va != vb:  # 완전 동일성(부동소수 비트까지) 요구 — 결정적이어야 함
                return False, f"[{k}] {va!r} != {vb!r}"
        elif va != vb:
            return False, f"[{k}] {va!r} != {vb!r}"
    return True, ""


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    print("Stage1Encoder 로딩...", flush=True)
    enc = Stage1Encoder(DEFAULT_VLM, STAGE1_PATH, device)
    enc.eval()

    print("OwlV2Grounder 로딩...", flush=True)
    grounder = OwlV2Grounder(device)
    grounder._ensure_loaded()

    frames = [make_frame(i) for i in range(N_FRAMES)]

    # 워밍업
    for image_rgb, pil in frames[:3]:
        grounder.run(image_rgb, phrase=PHRASE)
        enc.encode_image(pil)
    torch.cuda.synchronize()

    print(f"{N_FRAMES}개 프레임에 대해 순차 실행 기준값 계산...", flush=True)
    seq_bbox, seq_vis = [], []
    for image_rgb, pil in frames:
        bbox = grounder.run(image_rgb, phrase=PHRASE)
        torch.cuda.synchronize()
        vis = enc.encode_image(pil).clone()
        torch.cuda.synchronize()
        seq_bbox.append(bbox)
        seq_vis.append(vis)

    print(f"{N_FRAMES}개 프레임에 대해 병렬(ThreadPoolExecutor) 실행...", flush=True)
    par_bbox, par_vis = [], []
    with ThreadPoolExecutor(max_workers=2) as ex:
        for image_rgb, pil in frames:
            fut_g = ex.submit(grounder.run, image_rgb, phrase=PHRASE)
            fut_v = ex.submit(enc.encode_image, pil)
            bbox = fut_g.result()
            vis = fut_v.result().clone()
            par_bbox.append(bbox)
            par_vis.append(vis)
    torch.cuda.synchronize()

    n_bbox_mismatch = 0
    n_vis_mismatch = 0
    max_vis_abs_diff = 0.0

    for i in range(N_FRAMES):
        ok, msg = bbox_equal(seq_bbox[i], par_bbox[i])
        if not ok:
            n_bbox_mismatch += 1
            print(f"  [frame {i}] bbox MISMATCH: {msg}")

        # 완전 동일성(bit-exact) 확인 + 참고용 max abs diff
        exact = torch.equal(seq_vis[i], par_vis[i])
        diff = (seq_vis[i] - par_vis[i]).abs().max().item()
        max_vis_abs_diff = max(max_vis_abs_diff, diff)
        if not exact:
            n_vis_mismatch += 1
            print(f"  [frame {i}] vis_feat NOT bit-exact, max_abs_diff={diff:.3e}")

    print()
    print(f"검증 프레임 수: {N_FRAMES}")
    print(f"bbox 불일치: {n_bbox_mismatch}/{N_FRAMES}")
    print(f"vis_feat bit-exact 불일치: {n_vis_mismatch}/{N_FRAMES} (max_abs_diff={max_vis_abs_diff:.3e})")
    print()
    if n_bbox_mismatch == 0 and n_vis_mismatch == 0:
        print("✅ PASS — 순차/병렬 출력값 완전 동일 (bit-exact)")
    else:
        print("❌ FAIL — 출력값 불일치 발견, 병렬화 프로덕션 적용 전 원인 조사 필요")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""공간 정보 보존 실험용: V6 전 프레임의 Florence-2 비전 토큰(577 = 전역 1 + 24x24 격자 576)을
격자 부분만 4x4로 평균풀링해 (n,16,768) fp16으로 저장. bbox는 기존 v6_florence2_cache.npz 재사용.
출력: docs/v5/bbox_nav_owl/v6_florence2_spatial.npz  키 "{episode}|sp"  (gitignore 대상, 약 400MB)
실행: HF_HUB_OFFLINE=1 .venv/bin/python3 scripts/precompute_v6_florence2_spatial.py
"""
import json, time
from pathlib import Path
import h5py, numpy as np, torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OWL = ROOT / "docs/v5/bbox_nav_owl"
OUT = OWL / "v6_florence2_spatial.npz"; PART = OWL / "v6_florence2_spatial.partial.npz"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = "microsoft/Florence-2-base"; BATCH = 16


@torch.no_grad()
def main():
    from transformers import AutoModelForCausalLM, AutoProcessor
    model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.float16).to(DEV).eval()
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
    ds = json.loads((OWL / "bbox_dataset_v6_owl.json").read_text())
    cache = dict(np.load(PART)) if PART.exists() else {}
    t0 = time.time()
    for i, ep in enumerate(ds):
        key = ep["episode"]
        if f"{key}|sp" in cache: continue
        with h5py.File(key, "r") as f:
            imgs = f["observations"]["images"][:] if "observations" in f else f["images"][:]
        pils = [Image.fromarray(np.ascontiguousarray(imgs[fr["frame_idx"]][:, :, ::-1]).astype(np.uint8)).convert("RGB") for fr in ep["frames"]]
        outs = []
        for s in range(0, len(pils), BATCH):
            pv = proc(text=["<CAPTION_TO_PHRASE_GROUNDING>gray basket"] * len(pils[s:s + BATCH]), images=pils[s:s + BATCH],
                      return_tensors="pt")["pixel_values"].to(DEV, torch.float16)
            feat = model._encode_image(pv)                      # (B,577,768)
            grid = feat[:, 1:, :].reshape(-1, 24, 24, 768).permute(0, 3, 1, 2).float()
            outs.append(torch.nn.functional.adaptive_avg_pool2d(grid, 4).flatten(2).permute(0, 2, 1).cpu().numpy().astype(np.float16))
        cache[f"{key}|sp"] = np.concatenate(outs)
        if (i + 1) % 5 == 0:
            np.savez(PART, **cache); print(f"[{i+1}/{len(ds)}] {(time.time()-t0)/60:.1f}m", flush=True)
    np.savez(OUT, **cache); print("저장", OUT, flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stage 1 (EGVLA2 §2.5): V6 전 프레임을 Florence-2 fused 방식으로 1회만 통과시켜 캐시.

_encode_image를 1회 호출해 (1) <CAPTION_TO_PHRASE_GROUNDING> bbox 생성과
(2) 비전 특징(577토큰 평균풀링 768-d)을 동시에 얻는다 (젯슨 실측 371ms 방식과 동일).
bbox는 florence2_bbox_adapter로 OWLv2 스키마(cx,cy,area,has_bbox)로 변환.

출력: docs/v5/bbox_nav_owl/v6_florence2_cache.npz
  키 "{episode}|bbox" -> (n,4) [cx,cy,area,has_bbox], "{episode}|feat" -> (n,768) float16
  프레임 순서는 bbox_dataset_v6_owl.json의 ep["frames"]와 동일.
이어하기: 부분 캐시(.partial.npz)가 있으면 완료된 에피소드는 건너뜀.

실행: HF_HUB_OFFLINE=1 .venv/bin/python3 scripts/precompute_v6_florence2_fused.py
"""
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from florence2_bbox_adapter import florence2_output_to_bbox  # noqa: E402

OWL_DIR = ROOT / "docs" / "v5" / "bbox_nav_owl"
DATASET_FILE = OWL_DIR / "bbox_dataset_v6_owl.json"
OUT = OWL_DIR / "v6_florence2_cache.npz"
PARTIAL = OWL_DIR / "v6_florence2_cache.partial.npz"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = "microsoft/Florence-2-base"
TASK = "<CAPTION_TO_PHRASE_GROUNDING>"
PHRASE = "gray basket"
BATCH = 8


def load_images(path):
    with h5py.File(path, "r") as f:
        return f["observations"]["images"][:] if "observations" in f else f["images"][:]


@torch.no_grad()
def run_batch(model, proc, pils):
    inp = proc(text=[TASK + PHRASE] * len(pils), images=pils, return_tensors="pt")
    pv = inp["pixel_values"].to(DEV, torch.float16)
    ids = inp["input_ids"].to(DEV)
    img_feats = model._encode_image(pv)                      # (B,577,768) — 1회만
    embeds = model.get_input_embeddings()(ids)
    embeds, _ = model._merge_input_ids_with_image_features(img_feats, embeds)
    out = model.language_model.generate(
        input_ids=None, inputs_embeds=embeds,
        max_new_tokens=128, num_beams=3, do_sample=False)
    texts = proc.batch_decode(out, skip_special_tokens=False)
    pooled = img_feats.float().mean(dim=1).cpu().numpy().astype(np.float16)
    boxes = []
    for t, pil in zip(texts, pils):
        parsed = proc.post_process_generation(t, task=TASK, image_size=(pil.width, pil.height))[TASK]
        b = florence2_output_to_bbox(parsed.get("bboxes", []), pil.width, pil.height)
        boxes.append([b["cx"], b["cy"], b["area"], float(b["has_bbox"])])
    return np.array(boxes, dtype=np.float32), pooled


def main():
    from transformers import AutoModelForCausalLM, AutoProcessor
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.float16).to(DEV).eval()
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

    dataset = json.loads(DATASET_FILE.read_text())
    cache = dict(np.load(PARTIAL)) if PARTIAL.exists() else {}
    done = {k.split("|")[0] for k in cache}
    print(f"에피소드 {len(dataset)}개 (완료 {len(done)}개 이어하기)", flush=True)

    t0 = time.time()
    for i, ep in enumerate(dataset):
        key = ep["episode"]
        if key in done:
            continue
        imgs = load_images(key)
        idxs = [fr["frame_idx"] for fr in ep["frames"]]
        pils = [Image.fromarray(np.ascontiguousarray(imgs[j][:, :, ::-1]).astype(np.uint8)).convert("RGB")
                for j in idxs]
        bb, ft = [], []
        for s in range(0, len(pils), BATCH):
            tb = time.time()
            b, f = run_batch(model, proc, pils[s:s + BATCH])
            if time.time() - tb > 20:
                print(f"    ! 느린 배치 {Path(key).stem} s={s} {time.time()-tb:.0f}s", flush=True)
            bb.append(b)
            ft.append(f)
        cache[f"{key}|bbox"] = np.concatenate(bb)
        cache[f"{key}|feat"] = np.concatenate(ft)
        done.add(key)
        el = time.time() - t0
        print(f"  [{len(done)}/{len(dataset)}] {Path(key).stem[-40:]} ({len(idxs)}f) elapsed={el/60:.1f}m", flush=True)
        if len(done) % 5 == 0:
            np.savez(PARTIAL, **cache)

    np.savez_compressed(OUT, **cache)
    print(f"저장: {OUT} ({(time.time()-t0)/60:.1f}분)", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""<CAPTION_TO_PHRASE_GROUNDING> + 명시적 phrase("gray basket") 재현율 실측 (2026-08-20).

지금까지 시도 안 해본 방식: OWL-v2처럼 타겟 문구를 직접 주고 그 문구에 맞는
위치만 찾으라고 지시. 이전에 이 태스크를 버린 이유는 "열린 질문"(뭐가 있는지
다 말해봐)으로 썼을 때 거부 모드가 없어서였다 — 명시적 phrase를 주는 이번
방식은 그 문제와 무관할 수 있다. 5개 샘플 눈대중 확인(4/5 근접)에서 가능성을
보고 전체 배치로 검증.

같은 0807 100세션 1087프레임, OWL-v2 정답 기준.

주의: 파이프라인에 끼워넣지 않는다. 출력 품질만 독립 측정.

출력: docs/v5/detector/florence2_phrase_grounding_0807.json
"""
import glob
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
H5_DIR = "/home/minum/MoNaVLA/inference_sessions_recv/20260807/h5"
OUT = ROOT / "docs/v5/detector/florence2_phrase_grounding_0807.json"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = "microsoft/Florence-2-base"
HIT_TOL = 0.05
PHRASE = "gray basket"
TASK = "<CAPTION_TO_PHRASE_GROUNDING>"


def iter_frames():
    for path in sorted(glob.glob(f"{H5_DIR}/*.h5")):
        with h5py.File(path, "r") as hf:
            imgs = hf["observations/images"]
            bbox = hf["grounding/bbox"][:]
            n = imgs.shape[0]
            for i in range(n):
                yield imgs[i], float(bbox[i, 0]), float(bbox[i, 3])


def main():
    from transformers import AutoModelForCausalLM, AutoProcessor
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.float16).to(DEV).eval()
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

    frames = list(iter_frames())
    N = len(frames)
    print(f"총 {N} 프레임 · {TASK} phrase='{PHRASE}'", flush=True)

    pred_cx, gt_cx, owl_ok = [], [], []
    t0 = time.time()
    for n, (im, cx, hb) in enumerate(frames):
        pil = Image.fromarray(im.astype(np.uint8)).convert("RGB")
        W, H = pil.width, pil.height
        text = TASK + PHRASE
        inp = proc(text=text, images=pil, return_tensors="pt")
        with torch.no_grad():
            ids = model.generate(
                input_ids=inp["input_ids"].to(DEV),
                pixel_values=inp["pixel_values"].to(DEV, torch.float16),
                max_new_tokens=128, num_beams=3, do_sample=False)
        txt = proc.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = proc.post_process_generation(txt, task=TASK, image_size=(W, H))[TASK]
        boxes = parsed.get("bboxes", []) or []
        if boxes:
            x1, y1, x2, y2 = boxes[0]
            pred_cx.append((x1 + x2) / 2 / W)
        else:
            pred_cx.append(np.nan)
        gt_cx.append(cx)
        owl_ok.append(hb)

        if (n + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (n + 1) * (N - n - 1)
            print(f"  {n+1}/{N}  elapsed={elapsed/60:.1f}min  eta={eta/60:.1f}min", flush=True)

    pred_cx = np.array(pred_cx)
    gt_cx = np.array(gt_cx)
    owl_ok = np.array(owl_ok).astype(bool)

    selected = ~np.isnan(pred_cx)
    both = owl_ok & selected
    diff = np.abs(pred_cx[both] - gt_cx[both])
    hits = diff <= HIT_TOL
    n_owl_success = int(owl_ok.sum())

    result = dict(
        n_frames=N,
        task=TASK, phrase=PHRASE,
        owl_success_frames=n_owl_success,
        coverage=float(selected.mean()),  # 이 태스크는 거부모드가 없어 거의 100%일 것
        n_both=int(both.sum()),
        recall_vs_owl=float(hits.sum() / n_owl_success) if n_owl_success else 0.0,
        cx_mae_vs_owl=float(diff.mean()) if both.sum() else None,
        cx_median_ae_vs_owl=float(np.median(diff)) if both.sum() else None,
        elapsed_min=(time.time() - t0) / 60,
        note=("명시적 phrase 지정 CAPTION_TO_PHRASE_GROUNDING 첫 전체 실측. "
              "거부 모드가 없어 coverage는 항상 높지만, cx 정확도(재현율)가 핵심 지표."),
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

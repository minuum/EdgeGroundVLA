#!/usr/bin/env python3
"""Florence-2 프롬프트 종류별 실제 결과 비교 갤러리 (2026-08-21).

CH69의 "재현율 34.7%→84.96%→100%" 서사를 실제 이미지로 보여준다. 같은 8개
샘플 프레임에서 OD(beam3)·DENSE_REGION_CAPTION(beam3)·CAPTION_TO_PHRASE_GROUNDING
("gray basket") 세 프롬프트의 실제 예측을 한 화면에서 비교한다.

OD/DENSE는 기존 raw 결과 재사용(florence2_grounding_0807_raw.json), phrase는
이 스크립트에서 새로 실행(가벼움, 8프레임뿐).

주의: 파이프라인에 끼워넣지 않는다. 문서용 시각 자료 생성 전용.

출력: docs/v5/ch64_figs/fig_florence2_prompt_comparison.png
"""
import glob
import json
import re
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "NanumGothic"
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
H5_DIR = "/home/minum/MoNaVLA/inference_sessions_recv/20260807/h5"
RAW = ROOT / "docs/v5/detector/florence2_grounding_0807_raw.json"
OUT = ROOT / "docs/v5/ch64_figs/fig_florence2_prompt_comparison.png"
MODEL_ID = "microsoft/Florence-2-base"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

KEYWORDS = ["hamper", "basket", "trash can", "trash bin", "waste container",
            "waste bin", "wastebasket", "bin", "container"]
KW_RE = [re.compile(r"\b" + re.escape(k) + r"\b") for k in KEYWORDS]

SAMPLE_INDICES = [0, 2, 5, 8, 12, 300, 500, 700]  # 다양한 위치 샘플


def pick_kw(dets):
    hits = [d for d in dets if any(rx.search(d["label"]) for rx in KW_RE)]
    if not hits:
        return None
    return max(hits, key=lambda d: d["area"])


def iter_paths():
    for path in sorted(glob.glob(f"{H5_DIR}/*.h5")):
        with h5py.File(path, "r") as hf:
            n = hf["observations/images"].shape[0]
            for i in range(n):
                yield path, i


def run_phrase(model, proc, pil):
    W, H = pil.width, pil.height
    task = "<CAPTION_TO_PHRASE_GROUNDING>"
    text = task + "gray basket"
    inp = proc(text=text, images=pil, return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(
            input_ids=inp["input_ids"].to(DEV),
            pixel_values=inp["pixel_values"].to(DEV, torch.float16),
            max_new_tokens=128, num_beams=3, do_sample=False)
    txt = proc.batch_decode(ids, skip_special_tokens=False)[0]
    parsed = proc.post_process_generation(txt, task=task, image_size=(W, H))[task]
    boxes = parsed.get("bboxes", []) or []
    if not boxes:
        return None
    x1, y1, x2, y2 = boxes[0]
    return dict(label="gray basket", cx=(x1 + x2) / 2 / W, area=(x2 - x1) * (y2 - y1) / (W * H))


def main():
    rows = json.loads(RAW.read_text())
    paths = list(iter_paths())
    assert len(paths) == len(rows)

    from transformers import AutoModelForCausalLM, AutoProcessor
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.float16).to(DEV).eval()
    proc = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

    samples = []
    h5_cache = {}
    for i in SAMPLE_INDICES:
        path, fi = paths[i]
        row = rows[i]
        if path not in h5_cache:
            h5_cache[path] = h5py.File(path, "r")
        im = h5_cache[path]["observations/images"][fi]
        pil = Image.fromarray(im.astype(np.uint8)).convert("RGB")
        od = pick_kw(row["OD_b3"])
        dense = pick_kw(row["DENSE_b3"])
        phrase = run_phrase(model, proc, pil)
        samples.append(dict(im=im, gt_cx=row["gt_cx"], owl_ok=row["owl_success"],
                            od=od, dense=dense, phrase=phrase))
        print(f"idx={i} gt_cx={row['gt_cx']:.3f} OD={od} DENSE={dense} PHRASE={phrase}", flush=True)

    for f in h5_cache.values():
        f.close()

    ncols = len(samples)
    nrows = 3  # OD, DENSE, PHRASE
    # 가독성 우선: 셀 크게, 폰트 크게, 고해상도로 저장 (2026-08-24 요청 반영)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.6, nrows * 4.0))
    row_labels = [("<OD>\n(beam3, 열린질문)", "#fbbf24"),
                  ("<DENSE_REGION_CAPTION>\n(beam3, 열린질문)", "#60a5fa"),
                  ('<CAPTION_TO_PHRASE_\nGROUNDING>+"gray basket"', "#4ade80")]

    for r, (title, color) in enumerate(row_labels):
        key = ["od", "dense", "phrase"][r]
        for c, s in enumerate(samples):
            ax = axes[r][c]
            ax.axis("off")
            H, W = s["im"].shape[:2]
            ax.imshow(s["im"])
            ax.axvline(s["gt_cx"] * W, color="#22c55e", linewidth=4.5, alpha=0.95)
            pred = s[key]
            if pred is not None:
                ax.axvline(pred["cx"] * W, color="#ef4444", linewidth=4.5, linestyle="--", alpha=0.95)
                err = abs(pred["cx"] - s["gt_cx"])
                hit = "O" if err <= 0.05 else "X"
                lbl = pred["label"][:16]
                ax.set_title(f"{hit}  {lbl}\nΔ={err:.3f}", fontsize=15, fontweight="bold",
                             color="#16a34a" if err <= 0.05 else "#dc2626", pad=8)
            else:
                ax.set_title("(미검출)", fontsize=15, fontweight="bold", color="#6b7280", pad=8)
        axes[r][0].set_ylabel(title, fontsize=17, fontweight="bold", color=color, rotation=0,
                              ha="right", va="center", labelpad=110, linespacing=1.5)
        axes[r][0].axis("on")
        axes[r][0].set_xticks([]); axes[r][0].set_yticks([])
        for spine in axes[r][0].spines.values():
            spine.set_visible(False)

    fig.suptitle("Florence-2 프롬프트 종류별 실제 예측 비교 — 같은 8개 프레임 (2026-08-07 배치)\n"
                  "초록 실선 = OWL-v2 정답    ·    빨강 점선 = Florence-2 예측    ·    Δ = 중심좌표 오차 (0.05 이하면 적중)",
                  fontsize=20, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0.14, 0, 1, 0.91])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, facecolor="white")
    print(f"\n저장 → {OUT}")


if __name__ == "__main__":
    main()

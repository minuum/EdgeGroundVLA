#!/usr/bin/env python3
"""Florence-2 그라운딩 — 95% 모델(exp73)과 '같은 조건'(2026-08-07 실기 100세션)으로 재측정.

배경:
  florence2_keyword_match.py는 학습용 val 주석(n=80, bbox_dataset_v6_pg448_cx.json)에서
  뽑은 표본이었다. OWL-v2의 90.5%(985/1088)는 논문 헤드라인 95/100 모델이 실제로 돈
  2026-08-07 100세션 실기 배치(inference_sessions_recv/20260807/h5/) 전량에서 나온
  수치라 apples-to-apples가 아니었다. 이 스크립트는 **같은 100세션, 같은 프레임**에서
  Florence-2를 돌려 직접 비교한다.

정답(ground truth) 정의:
  세션 H5의 grounding/bbox[:, 0]=cx, [:, 3]=has_bbox(OWL-v2가 이 프레임에서 성공했는지).
  OWL-v2 헤드라인 90.5% = has_bbox==1 프레임 비율(캐시 재사용 포함, 실제 배포 정의 그대로).

측정 (정답을 선택에 쓰지 않는다 — florence2_keyword_match.py와 동일 원칙):
  1. 키워드로 박스 선택(단어 경계 매칭으로 "bin"이 "cabinetry"에 걸리는
     이전 버그를 수정 — \\b(keyword)\\b regex)
  2. Florence-2 자체 검출률 = 키워드로 뭐라도 골라낸 프레임 / 전체(1088) — OWL 90.5%와
     직접 비교 가능한 정의
  3. OWL-v2가 성공한 프레임(has_bbox==1)만 대상으로, Florence-2가 고른 박스가
     |Δcx| <= HIT_TOL이면 "일치"로 카운트 → 일치율(agreement)

주의: 파이프라인에 끼워넣지 않는다. 출력 품질만 독립 측정.

출력: docs/v5/detector/florence2_grounding_0807_fullbatch.json
"""
import glob
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
H5_DIR = "/home/minum/MoNaVLA/inference_sessions_recv/20260807/h5"
OUT = ROOT / "docs/v5/detector/florence2_grounding_0807_fullbatch.json"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = "microsoft/Florence-2-base"
HIT_TOL = 0.05
OWL_BASELINE = 0.905  # 985/1088, 2026-08-07 100세션 실측 (has_bbox==1 비율)

KEYWORDS = ["hamper", "basket", "trash can", "trash bin", "waste container",
            "waste bin", "wastebasket", "bin", "container"]
KW_RE = [re.compile(r"\b" + re.escape(k) + r"\b") for k in KEYWORDS]


def gen(model, proc, pil, task, beams=3):
    W, H = pil.width, pil.height
    inp = proc(text=task, images=pil, return_tensors="pt")
    with torch.no_grad():
        ids = model.generate(
            input_ids=inp["input_ids"].to(DEV),
            pixel_values=inp["pixel_values"].to(DEV, torch.float16),
            max_new_tokens=256, num_beams=beams, do_sample=False)
    txt = proc.batch_decode(ids, skip_special_tokens=False)[0]
    parsed = proc.post_process_generation(txt, task=task, image_size=(W, H))[task]
    boxes = parsed.get("bboxes", []) or []
    labels = (parsed.get("labels") or parsed.get("bboxes_labels") or [])
    out = []
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = b
        lb = str(labels[i]).lower().strip() if i < len(labels) else ""
        out.append(dict(label=lb, cx=(x1 + x2) / 2 / W,
                        area=(x2 - x1) * (y2 - y1) / (W * H)))
    return out


def pick_by_keyword(dets):
    hits = [d for d in dets if any(rx.search(d["label"]) for rx in KW_RE)]
    if not hits:
        return None
    return max(hits, key=lambda d: d["area"])


def iter_frames():
    for path in sorted(glob.glob(f"{H5_DIR}/*.h5")):
        with h5py.File(path, "r") as hf:
            imgs = hf["observations/images"]
            bbox = hf["grounding/bbox"][:]
            n = imgs.shape[0]
            for i in range(n):
                yield path, i, imgs[i], float(bbox[i, 0]), float(bbox[i, 3])


def main():
    from transformers import AutoModelForCausalLM, AutoProcessor
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.float16).to(DEV).eval()
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

    TASKS = [("OD", "<OD>"), ("DENSE", "<DENSE_REGION_CAPTION>")]
    sel = {name: [] for name, _ in TASKS}
    gt_cx, gt_success = [], []
    matched_labels = defaultdict(lambda: defaultdict(int))

    t0 = time.time()
    frames = list(iter_frames())
    N = len(frames)
    print(f"총 {N} 프레임 (100세션) · OWL-v2 baseline 90.5%", flush=True)

    for n, (path, fi, im, cx, has_bbox) in enumerate(frames):
        pil = Image.fromarray(im.astype(np.uint8)).convert("RGB")
        gt_cx.append(cx)
        gt_success.append(has_bbox)

        for name, task in TASKS:
            dets = gen(model, proc, pil, task)
            p = pick_by_keyword(dets)
            sel[name].append(p)
            if p:
                matched_labels[name][p["label"]] += 1

        if (n + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (n + 1) * (N - n - 1)
            print(f"  {n+1}/{N}  elapsed={elapsed/60:.1f}min  eta={eta/60:.1f}min", flush=True)

    gt_cx = np.array(gt_cx)
    gt_success = np.array(gt_success)
    N = len(gt_cx)
    n_owl_success = int(gt_success.sum())

    def score(picks):
        picks_arr = np.array([p["cx"] if p else np.nan for p in picks])
        selected = ~np.isnan(picks_arr)
        coverage = selected.mean()  # OWL 90.5%와 직접 비교 가능한 정의

        owl_ok = gt_success.astype(bool)
        both = owl_ok & selected
        agree = np.abs(picks_arr[both] - gt_cx[both]) <= HIT_TOL if both.sum() else np.array([])
        agreement_rate = agree.mean() if len(agree) else 0.0
        cx_mae = float(np.abs(picks_arr[both] - gt_cx[both]).mean()) if both.sum() else None

        return dict(
            coverage=float(coverage),
            n_selected=int(selected.sum()),
            n_owl_success=n_owl_success,
            n_both=int(both.sum()),
            agreement_rate=float(agreement_rate),
            cx_mae_vs_owl=cx_mae,
        )

    result = dict(
        n_frames=N,
        n_sessions=100,
        owl_baseline_coverage=OWL_BASELINE,
        owl_success_frames=n_owl_success,
        hit_tol=HIT_TOL,
        elapsed_min=(time.time() - t0) / 60,
        by_task={name: score(sel[name]) for name in sel},
        top_labels={name: dict(sorted(matched_labels[name].items(), key=lambda x: -x[1])[:10])
                    for name in sel},
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

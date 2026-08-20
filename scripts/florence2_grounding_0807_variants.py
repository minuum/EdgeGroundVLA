#!/usr/bin/env python3
"""Florence-2 그라운딩 재현율 개선 실험 — beam5 + 키워드 확장 + 합집합 (2026-08-19).

florence2_grounding_0807_fullbatch.py(B안, beam3, 고정 키워드 9개)의 후속.
같은 100세션 1087프레임에서 raw 검출(라벨+bbox 전체, beam3/beam5 × OD/DENSE)을
**한 번만** 생성해서 저장한다. 키워드 목록/선택 규칙은 이 raw 파일에서 얼마든지
재계산 가능 — beam 재실행(가장 비싼 부분) 없이 여러 변형을 빠르게 시도하기 위함.

베이스라인(참고, florence2_grounding_0807_fullbatch.json):
  OD(beam3) 재현율 19.4% · DENSE(beam3) 재현율 28.4% · OWL-v2 90.5%

이 스크립트가 생성하는 것: docs/v5/detector/florence2_grounding_0807_raw.json
  프레임별 {gt_cx, owl_success, OD_b3: [...], DENSE_b3: [...], OD_b5: [...], DENSE_b5: [...]}
  각 항목은 검출된 모든 박스(라벨 포함) — 키워드 필터링 전 raw 상태.

그 다음 EXPANDED_KEYWORDS로 재계산한 결과도 이 스크립트 안에서 바로 출력한다
(원본 9개 키워드 vs 확장 키워드, beam3 vs beam5, OD 단독/DENSE 단독/합집합 — 총 조합 비교).

주의: 파이프라인에 끼워넣지 않는다. 출력 품질만 독립 측정.
"""
import glob
import json
import re
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
H5_DIR = "/home/minum/MoNaVLA/inference_sessions_recv/20260807/h5"
RAW_OUT = ROOT / "docs/v5/detector/florence2_grounding_0807_raw.json"
REPORT_OUT = ROOT / "docs/v5/detector/florence2_grounding_0807_variants.json"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = "microsoft/Florence-2-base"
HIT_TOL = 0.05
OWL_BASELINE = 0.905

# 원본(baseline) — florence2_keyword_match.py / fullbatch와 동일
KEYWORDS_ORIG = ["hamper", "basket", "trash can", "trash bin", "waste container",
                 "waste bin", "wastebasket", "bin", "container"]
# 확장 — 관찰된 실제 라벨 어휘 + 인접 의미어 추가
KEYWORDS_EXPANDED = KEYWORDS_ORIG + [
    "laundry", "garbage", "recycling", "dustbin", "rubbish", "bucket",
    "storage box", "crate", "box", "basket case",  # 관대한 후보 포함(오탐 위험 감수)
]


def kw_regexes(words):
    return [re.compile(r"\b" + re.escape(w) + r"\b") for w in words]


def gen(model, proc, pil, task, beams):
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


def iter_frames():
    for path in sorted(glob.glob(f"{H5_DIR}/*.h5")):
        with h5py.File(path, "r") as hf:
            imgs = hf["observations/images"]
            bbox = hf["grounding/bbox"][:]
            n = imgs.shape[0]
            for i in range(n):
                yield imgs[i], float(bbox[i, 0]), float(bbox[i, 3])


def build_raw():
    if RAW_OUT.exists():
        print(f"[RAW] 재사용 {RAW_OUT.name}")
        return json.loads(RAW_OUT.read_text())

    from transformers import AutoModelForCausalLM, AutoProcessor
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, trust_remote_code=True, torch_dtype=torch.float16).to(DEV).eval()
    proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)

    frames = list(iter_frames())
    N = len(frames)
    print(f"총 {N} 프레임 · raw 검출 생성(OD/DENSE × beam3/beam5)", flush=True)

    rows = []
    t0 = time.time()
    for n, (im, cx, hb) in enumerate(frames):
        pil = Image.fromarray(im.astype(np.uint8)).convert("RGB")
        row = dict(gt_cx=cx, owl_success=hb)
        row["OD_b3"] = gen(model, proc, pil, "<OD>", 3)
        row["DENSE_b3"] = gen(model, proc, pil, "<DENSE_REGION_CAPTION>", 3)
        row["OD_b5"] = gen(model, proc, pil, "<OD>", 5)
        row["DENSE_b5"] = gen(model, proc, pil, "<DENSE_REGION_CAPTION>", 5)
        rows.append(row)
        if (n + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (n + 1) * (N - n - 1)
            print(f"  {n+1}/{N}  elapsed={elapsed/60:.1f}min  eta={eta/60:.1f}min", flush=True)

    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
    RAW_OUT.write_text(json.dumps(rows))
    print(f"[RAW] 저장 → {RAW_OUT}")
    return rows


def pick(dets, regexes):
    hits = [d for d in dets if any(rx.search(d["label"]) for rx in regexes)]
    if not hits:
        return None
    return max(hits, key=lambda d: d["area"])


def score_variant(rows, keys, keywords, mode="single"):
    """keys: 리스트, mode='single'이면 keys[0]만, 'union'이면 keys 중 먼저 hit하는 것 채택
    (우선순위 = keys 순서, DENSE 먼저가 기본)."""
    regexes = kw_regexes(keywords)
    gt = np.array([r["gt_cx"] for r in rows])
    owl_ok = np.array([r["owl_success"] for r in rows]).astype(bool)
    picks_cx = []
    for r in rows:
        chosen = None
        for k in keys:
            p = pick(r[k], regexes)
            if p:
                chosen = p
                break
        picks_cx.append(chosen["cx"] if chosen else np.nan)
    picks_cx = np.array(picks_cx)
    selected = ~np.isnan(picks_cx)
    both = owl_ok & selected
    agree = np.abs(picks_cx[both] - gt[both]) <= HIT_TOL if both.sum() else np.array([])
    n_owl_success = int(owl_ok.sum())
    hits = float(agree.sum())
    return dict(
        coverage=float(selected.mean()),
        n_selected=int(selected.sum()),
        agreement_rate=float(agree.mean()) if len(agree) else 0.0,
        recall_vs_owl=hits / n_owl_success if n_owl_success else 0.0,
        cx_mae_vs_owl=float(np.abs(picks_cx[both] - gt[both]).mean()) if both.sum() else None,
    )


def main():
    rows = build_raw()

    VARIANTS = {
        "OD_b3_kwOrig":            (["OD_b3"], KEYWORDS_ORIG),
        "DENSE_b3_kwOrig":         (["DENSE_b3"], KEYWORDS_ORIG),
        "OD_b5_kwOrig":            (["OD_b5"], KEYWORDS_ORIG),
        "DENSE_b5_kwOrig":         (["DENSE_b5"], KEYWORDS_ORIG),
        "union_b3_DENSEfirst_kwOrig": (["DENSE_b3", "OD_b3"], KEYWORDS_ORIG),
        "union_b5_DENSEfirst_kwOrig": (["DENSE_b5", "OD_b5"], KEYWORDS_ORIG),
        "DENSE_b3_kwExpanded":     (["DENSE_b3"], KEYWORDS_EXPANDED),
        "DENSE_b5_kwExpanded":     (["DENSE_b5"], KEYWORDS_EXPANDED),
        "union_b5_DENSEfirst_kwExpanded": (["DENSE_b5", "OD_b5"], KEYWORDS_EXPANDED),
    }

    result = dict(n_frames=len(rows), owl_baseline=OWL_BASELINE,
                  baseline_ref="florence2_grounding_0807_fullbatch.json (OD_b3=19.4%, DENSE_b3=28.4%)")
    result["variants"] = {}
    for name, (keys, kws) in VARIANTS.items():
        s = score_variant(rows, keys, kws)
        result["variants"][name] = s
        print(f"{name:35s} coverage={s['coverage']*100:5.1f}%  recall_vs_owl={s['recall_vs_owl']*100:5.1f}%  "
              f"cx_mae={s['cx_mae_vs_owl']}")

    REPORT_OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\n저장 → {REPORT_OUT}")


if __name__ == "__main__":
    main()

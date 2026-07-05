#!/usr/bin/env python3
"""OWL-v2 confidence threshold 튜닝 — human_labels.json(296프레임, 객체없음 79) 기반.

각 프레임의 OWL-v2 최고 confidence score를 뽑아서, threshold를 스윕하며
  - 정탐 유지율: 객체 있음+OWL박스 정답(ow=ok) 프레임에서 score>=t 비율
  - 오탐률: 객체 없음 프레임에서 score>=t 비율
을 계산. 목적: "없는 걸 없다고 말하는" threshold 찾기.

출력: docs/v5/hsv_owlv2_preview_20260704/owlv2_scores.json + 콘솔 ROC 테이블
"""
import glob
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent.parent
GALLERY_DIR = ROOT / "docs" / "v5" / "hsv_owlv2_preview_20260704"
RECV_GLOB = "/home/minum/MoNaVLA/inference_sessions_recv/*/session_*.h5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SCORES_PATH = GALLERY_DIR / "owlv2_scores.json"


def collect_scores():
    if SCORES_PATH.exists():
        print(f"기존 스코어 재사용: {SCORES_PATH}")
        return json.loads(SCORES_PATH.read_text())

    meta = json.loads((GALLERY_DIR / "meta.json").read_text())
    files = {Path(f).stem: f for f in glob.glob(RECV_GLOB)}

    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    print("[OWL-v2] 로딩...")
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble").to(DEVICE).eval()

    scores = {}
    t0 = time.time()
    opened = {}
    for i, m in enumerate(meta):
        ep = m["episode"]
        if ep not in opened:
            opened[ep] = h5py.File(files[ep], "r")
        img = Image.fromarray(
            np.array(opened[ep]["observations"]["images"][m["frame_idx"]]).astype(np.uint8)
        ).convert("RGB")
        inp = proc(text=[["gray laundry basket"]], images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        # threshold=0 으로 전체 박스 score 확보 → 최고값 저장
        res = proc.post_process_object_detection(
            out, threshold=0.0, target_sizes=[(img.height, img.width)])[0]
        s = res["scores"]
        scores[m["key"]] = float(s.max()) if len(s) else 0.0
        if (i + 1) % 30 == 0:
            print(f"  {i+1}/{len(meta)} ({time.time()-t0:.0f}s)")
    for f in opened.values():
        f.close()
    SCORES_PATH.write_text(json.dumps(scores, indent=2))
    print(f"스코어 저장: {SCORES_PATH}")
    return scores


def main():
    scores = collect_scores()
    labels = json.loads((GALLERY_DIR / "human_labels.json").read_text())

    absent = [k for k, v in labels.items() if v.get("no_target") == "yes"]
    present_ok = [k for k, v in labels.items()
                  if v.get("no_target") != "yes" and v.get("ow") == "ok"]
    present_all = [k for k, v in labels.items() if v.get("no_target") != "yes"]

    print(f"\n객체없음 {len(absent)} / 객체있음(OWL 정답) {len(present_ok)} / 객체있음 전체 {len(present_all)}")
    print(f"\n{'thresh':>7} {'정탐유지(ok기준)':>16} {'정탐유지(전체)':>14} {'오탐률(absent)':>14}")
    best = None
    for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        tp = sum(1 for k in present_ok if scores.get(k, 0) >= t) / len(present_ok)
        tpa = sum(1 for k in present_all if scores.get(k, 0) >= t) / len(present_all)
        fp = sum(1 for k in absent if scores.get(k, 0) >= t) / len(absent)
        # Youden J (정탐-오탐 최대화)
        j = tp - fp
        marker = ""
        if best is None or j > best[0]:
            best = (j, t, tp, fp)
        print(f"{t:>7.2f} {100*tp:>15.1f}% {100*tpa:>13.1f}% {100*fp:>13.1f}%")
    j, t, tp, fp = best
    print(f"\n권장 threshold (Youden J 최대): {t:.2f} — 정탐 {100*tp:.1f}% 유지, 오탐 {100*fp:.1f}%")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""① 언어→타겟 선택 실증 — 같은 프레임에 다른 텍스트 쿼리 → 다른 객체 그라운딩.

실주행 세션 프레임 6장 × 쿼리 4종("gray laundry basket"/"door"/"chair"/"trash can")을
OWL-v2에 넣고, 쿼리별 bbox를 색으로 겹쳐 그린 갤러리 생성.
언어가 검출 타겟을 실제로 바꾼다는 시각 증거 (VLA 사다리 ①).

Usage: .venv/bin/python3 scripts/demo_owl_query_switch.py
출력: docs/v5/owl_query_switch/gallery.html
"""
import glob
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs" / "v5" / "owl_query_switch"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESH = 0.15  # 데모용 — basket 외 객체는 conf가 낮을 수 있어 0.25보다 완화

QUERIES = {
    "gray laundry basket": (240, 200, 0),
    "door": (60, 220, 60),
    "chair": (0, 200, 255),
    "trash can": (255, 90, 90),
}
# 서로 다른 장면의 프레임 6장 (세션 다양하게)
FRAME_PICKS = [
    ("20260702/session_20260701_220400.h5", 0),
    ("20260702/session_20260702_100143.h5", 10),
    ("20260702/session_20260702_131204.h5", 4),
    ("20260703/session_20260703_012829.h5", 5),
    ("20260703/session_20260703_065407.h5", 0),
    ("20260702/session_20260702_212834.h5", 15),
]
RECV = Path("/home/minum/MoNaVLA/inference_sessions_recv")


def main():
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    print("[OWL-v2] 로딩...")
    proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained(
        "google/owlv2-base-patch16-ensemble").to(DEVICE).eval()

    cards = []
    results_log = []
    for rel, fi in FRAME_PICKS:
        fp = RECV / rel
        if not fp.exists():
            print(f"skip (없음): {fp}")
            continue
        with h5py.File(fp, "r") as h:
            img = Image.fromarray(np.array(h["observations"]["images"][fi]).astype(np.uint8)).convert("RGB")
        W, H = img.width, img.height
        key = f"{fp.stem}_f{fi:02d}"

        # 쿼리 4종 동시 입력 — OWL은 멀티쿼리 네이티브 지원
        texts = list(QUERIES.keys())
        inp = proc(text=[texts], images=img, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model(**inp)
        res = proc.post_process_object_detection(out, threshold=THRESH, target_sizes=[(H, W)])[0]

        thumb = img.resize((480, 270), Image.LANCZOS)
        draw = ImageDraw.Draw(thumb)
        found = {}
        for q_idx, q in enumerate(texts):
            mask = res["labels"] == q_idx
            if not mask.any():
                found[q] = None
                continue
            scores_q = res["scores"][mask]
            boxes_q = res["boxes"][mask]
            best = int(scores_q.argmax())
            x1, y1, x2, y2 = boxes_q[best].cpu().tolist()
            sx, sy = 480 / W, 270 / H
            c = QUERIES[q]
            draw.rectangle([x1 * sx, y1 * sy, x2 * sx, y2 * sy], outline=c, width=2)
            draw.text((x1 * sx + 2, y1 * sy + 1), f"{q} {scores_q[best]:.2f}", fill=c)
            found[q] = {"cx": (x1 + x2) / 2 / W, "score": float(scores_q[best])}
        fname = f"qs_{key}.jpg"
        thumb.save(OUT_DIR / fname, quality=88)
        results_log.append({"key": key, "found": found})

        badges = " ".join(
            f'<span style="color:rgb{QUERIES[q]}">{q}: '
            + (f"cx={v['cx']:.2f}({v['score']:.2f})" if v else "미검출") + "</span>"
            for q, v in found.items())
        cards.append(f'<div class="card"><div class="hd">{key}<br>{badges}</div>'
                     f'<img src="{fname}"></div>')

    html = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="UTF-8">
<title>OWL-v2 쿼리 스위칭 — 언어가 타겟을 바꾼다</title><style>
body {{ background:#0a0f1a; color:#e2e8f0; font-family:sans-serif; padding:24px; }}
h1 {{ font-size:1.2rem; }} .sub {{ color:#64748b; font-size:0.85rem; margin-bottom:16px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(480px,1fr)); gap:12px; }}
.card {{ background:#0d1117; border:1px solid #1e293b; border-radius:8px; overflow:hidden; }}
.hd {{ padding:6px 10px; font-size:0.68rem; background:#111827; line-height:1.6; }}
.card img {{ width:100%; display:block; }}
</style></head><body>
<h1>OWL-v2 쿼리 스위칭 실증 — 같은 프레임, 다른 텍스트 쿼리 → 다른 객체</h1>
<p class="sub">4개 쿼리를 동시에 넣고 쿼리별 최고 score 박스를 겹쳐 표시 (th {THRESH}).
언어→타겟 선택(VLA 사다리 ①)의 시각 증거.</p>
<div class="grid">{''.join(cards)}</div></body></html>"""
    (OUT_DIR / "gallery.html").write_text(html)
    (OUT_DIR / "results.json").write_text(json.dumps(results_log, indent=2, ensure_ascii=False))
    print(f"완료: {len(cards)}프레임 → {OUT_DIR}/gallery.html")


if __name__ == "__main__":
    main()

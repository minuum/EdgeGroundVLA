#!/usr/bin/env python3
"""plan_20260624_zoom_regrounding_small_objects.md — area<0.05(작은/먼 객체) 프레임만
현재 (cx,cy) 중심 2배 줌 크롭으로 재그라운딩. 실패(has_bbox=False)하면 원본 유지.

기존 bbox_dataset_full_pg2.json은 보존, 새 파일로 저장.

Usage:
  .venv/bin/python3 scripts/eval/regroun_zoom_small.py
"""
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from robovlm_nav.serve.stage2_v2_inference_server import PG2Grounder, DEFAULT_PG2  # noqa: E402

SRC = ROOT / "docs/v5/bbox_nav_exp46/bbox_dataset_full_pg2.json"
OUT = ROOT / "docs/v5/bbox_nav_exp46/bbox_dataset_full_pg2_zoomsmall.json"
AREA_THRESH = 0.05
ZOOM = 2.0


def zoom_crop(img: np.ndarray, cx: float, cy: float, zoom: float = ZOOM):
    H, W = img.shape[:2]
    cw, ch = W / zoom, H / zoom
    x0 = int(np.clip(cx * W - cw / 2, 0, W - cw))
    y0 = int(np.clip(cy * H - ch / 2, 0, H - ch))
    cw_i, ch_i = int(cw), int(ch)
    crop = img[y0:y0 + ch_i, x0:x0 + cw_i]
    return crop, x0, y0, cw_i, ch_i, W, H


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grounder = PG2Grounder(DEFAULT_PG2, device)
    grounder._ensure_loaded()

    data = json.loads(SRC.read_text())
    print(f"[시작] {time.strftime('%Y-%m-%d %H:%M:%S')} — {len(data)} episodes")

    t0 = time.time()
    n_total = n_small = n_improved = 0
    for ei, ep in enumerate(data):
        h5_path = Path(ep["episode"])
        if not h5_path.exists():
            continue
        with h5py.File(h5_path, "r") as f:
            imgs = f["observations"]["images"]
            for fr in ep["frames"]:
                n_total += 1
                if not fr.get("has_bbox") or fr["area"] >= AREA_THRESH:
                    continue
                n_small += 1
                img = np.asarray(imgs[fr["frame_idx"]])
                crop, x0, y0, cw, ch, W, H = zoom_crop(img, fr["cx"], fr["cy"])
                bbox = grounder.run(crop, phrase="gray basket")
                if bbox["has_bbox"]:
                    x1 = (x0 + bbox["x1"] * cw) / W
                    y1 = (y0 + bbox["y1"] * ch) / H
                    x2 = (x0 + bbox["x2"] * cw) / W
                    y2 = (y0 + bbox["y2"] * ch) / H
                    fr["cx"], fr["cy"] = (x1 + x2) / 2, (y1 + y2) / 2
                    fr["area"] = (x2 - x1) * (y2 - y1)
                    n_improved += 1
        if (ei + 1) % 30 == 0:
            print(f"  [{ei+1}/{len(data)} ep] 경과 {(time.time()-t0)/60:.1f}분, "
                  f"small={n_small}, 재그라운딩성공={n_improved}")

    elapsed = time.time() - t0
    print(f"[완료] 총 {elapsed/60:.1f}분 — 전체 {n_total}프레임 중 small(area<{AREA_THRESH})={n_small}, "
          f"재그라운딩 성공(has_bbox 유지)={n_improved}")
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(f"[저장] {OUT}")


if __name__ == "__main__":
    main()

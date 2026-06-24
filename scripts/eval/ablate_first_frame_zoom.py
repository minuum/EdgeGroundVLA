#!/usr/bin/env python3
"""plan_20260624_first_frame_zoom_crop_ablation.md §2 — 에피소드 시작 후 첫 N프레임 중
area<threshold(또는 has_bbox=False)인 프레임만 2배 줌 크롭으로 재그라운딩, 교체 전/후 비교.

운영 코드(stage2_v2_inference_server.py)는 건드리지 않음 — 저장된 세션의 raw 이미지를
오프라인으로 리플레이해서 효과만 측정(읍 단계, ablation).

대상: docs/inference_sessions/*.h5 중 grounding/bbox 필드가 있는 세션 전체(현재 4개,
6/24 조이스틱 체감 테스트 — 그 이전 세션들은 grounding 필드 없음, 별도 처리 필요).

frame0(에피소드 첫 프레임)은 직전 bbox가 없으므로 화면 중앙(0.5, 0.5)을 줌 크롭 anchor로 사용.

Usage:
  .venv/bin/python3 scripts/eval/ablate_first_frame_zoom.py
"""
import glob
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
from scripts.eval.regroun_zoom_small import zoom_crop  # noqa: E402

SESSIONS_DIR = ROOT / "docs/inference_sessions"
OUT = ROOT / "docs/v5/first_frame_zoom_ablation.json"
FIRST_N_ZOOM = 5
AREA_THRESHOLD = 0.05


def remap_zoom_bbox(bbox, x0, y0, cw, ch, W, H):
    x1 = (x0 + bbox["x1"] * cw) / W
    y1 = (y0 + bbox["y1"] * ch) / H
    x2 = (x0 + bbox["x2"] * cw) / W
    y2 = (y0 + bbox["y2"] * ch) / H
    return {"cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2, "area": (x2 - x1) * (y2 - y1), "has_bbox": True}


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grounder = PG2Grounder(DEFAULT_PG2, device)
    grounder._ensure_loaded()

    session_paths = sorted(glob.glob(str(SESSIONS_DIR / "*.h5")))
    results = []
    n_candidates = n_has_bbox_flip = 0
    area_deltas = []
    cxcy_deltas = []

    for sp in session_paths:
        with h5py.File(sp, "r") as f:
            if "grounding" not in f or "bbox" not in f["grounding"]:
                continue  # 구버전 세션(grounding 필드 없음) — 스킵
            bbox = f["grounding/bbox"][:]  # (n, 4) = cx,cy,area,has_bbox
            cached = f["grounding/cached"][:]
            imgs = f["observations/images"]
            session_id = dict(f.attrs).get("session_id", Path(sp).stem)
            instruction = dict(f.attrs).get("instruction", "the gray basket")

            real_seen = 0  # placeholder(cached=-1) 제외하고 센 "실제 그라운딩 프레임" 카운트
            for t in range(len(bbox)):
                if cached[t] < 0:  # frame0 placeholder(아직 실제 grounding 호출 전) — 윈도우 카운트에서도 제외
                    continue
                real_seen += 1
                if real_seen > FIRST_N_ZOOM:
                    break  # "첫 N개 실제 그라운딩 프레임" 윈도우를 벗어남

                cx, cy, area, has = bbox[t]
                is_candidate = not (has > 0.5 and area >= AREA_THRESHOLD)  # 작거나 has_bbox=False
                if not is_candidate:
                    continue  # 이미 충분히 크게 잡힌 정상 프레임 — 재시도 대상 아님

                n_candidates += 1
                # anchor: 직전 프레임의 cx,cy (없으면 화면 중앙)
                anchor_cx, anchor_cy = (bbox[t - 1][0], bbox[t - 1][1]) if t > 0 else (0.5, 0.5)
                img = np.asarray(imgs[t])
                crop, x0, y0, cw, ch, W, H = zoom_crop(img, anchor_cx, anchor_cy, zoom=2.0)
                zb = grounder.run(crop, phrase=instruction if "gray basket" in instruction else "gray basket")

                before = {"cx": float(cx), "cy": float(cy), "area": float(area), "has_bbox": bool(has > 0.5)}
                if zb["has_bbox"]:
                    after = remap_zoom_bbox(zb, x0, y0, cw, ch, W, H)
                else:
                    after = dict(before)  # 줌 재시도도 실패 -> 원본 유지(악화 방지)

                flipped = (not before["has_bbox"]) and after["has_bbox"]
                if flipped:
                    n_has_bbox_flip += 1
                area_deltas.append(after["area"] - before["area"])
                cxcy_deltas.append(((after["cx"] - before["cx"]) ** 2 + (after["cy"] - before["cy"]) ** 2) ** 0.5)

                results.append({
                    "session": session_id, "frame_idx": t,
                    "before": before, "after": after, "has_bbox_flip": flipped,
                })

    summary = {
        "n_candidates": n_candidates,
        "n_has_bbox_flip_false_to_true": n_has_bbox_flip,
        "area_delta_mean": float(np.mean(area_deltas)) if area_deltas else None,
        "area_delta_std": float(np.std(area_deltas)) if area_deltas else None,
        "cxcy_shift_mean": float(np.mean(cxcy_deltas)) if cxcy_deltas else None,
        "n_sessions_with_grounding": len({r["session"] for r in results}) if results else 0,
        "n_sessions_total": len(session_paths),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    OUT.write_text(json.dumps({"summary": summary, "frames": results}, indent=2, ensure_ascii=False))
    print(f"\n[저장] {OUT}")


if __name__ == "__main__":
    main()

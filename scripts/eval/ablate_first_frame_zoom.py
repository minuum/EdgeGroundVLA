#!/usr/bin/env python3
"""plan_20260624_first_frame_zoom_crop_ablation.md §2 — 에피소드 시작 후 첫 N프레임 중
area<threshold(또는 has_bbox=False)인 프레임만 2배 줌 크롭으로 재그라운딩, 교체 전/후 비교.

운영 코드(stage2_v2_inference_server.py)는 건드리지 않음 — 오프라인 ablation만.

표본을 4개 조이스틱 체감 테스트 세션(n=2, 결론 보류)에서 확장(2026-06-25, 사용자 지시
"표본세션들말고 그동안 그라운딩이나 다른 테스트했던 여러것들에 대해서 진행"):

  소스 A) docs/inference_sessions/*.h5 중 grounding 필드가 있는 세션 — 저장된 bbox 그대로 사용
  소스 B) docs/inference_sessions/*.h5 중 grounding 필드가 없는 구버전 세션(12개) — raw 이미지로
          PG2Grounder를 처음부터 다시 돌려 첫 5프레임 baseline 그라운딩 새로 생성
  소스 C) docs/v5/bbox_nav_exp46/bbox_dataset_full_pg2.json (CH46/50에서 이미 PG2로 재주석한
          150개 V5 학습 에피소드) — 각 에피소드의 "시작 첫 5프레임"만 추출(실주행 episode-start와
          동일한 상황: 사람이 막 카메라 앞에 로봇을 세팅한 시점). baseline은 이미 계산되어 있으므로
          재그라운딩 없이 재사용, 후보(area<threshold)만 줌 재시도.

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
TRAIN_DATA = ROOT / "docs/v5/bbox_nav_exp46/bbox_dataset_full_pg2.json"
OUT = ROOT / "docs/v5/first_frame_zoom_ablation.json"
FIRST_N_ZOOM = 5
AREA_THRESHOLD = 0.05


def remap_zoom_bbox(bbox, x0, y0, cw, ch, W, H):
    x1 = (x0 + bbox["x1"] * cw) / W
    y1 = (y0 + bbox["y1"] * ch) / H
    x2 = (x0 + bbox["x2"] * cw) / W
    y2 = (y0 + bbox["y2"] * ch) / H
    return {"cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2, "area": (x2 - x1) * (y2 - y1), "has_bbox": True}


def try_zoom(grounder, img, anchor_cx, anchor_cy, before, phrase="gray basket"):
    crop, x0, y0, cw, ch, W, H = zoom_crop(img, anchor_cx, anchor_cy, zoom=2.0)
    zb = grounder.run(crop, phrase=phrase)
    if zb["has_bbox"]:
        return remap_zoom_bbox(zb, x0, y0, cw, ch, W, H)
    return dict(before)  # 줌도 실패하면 원본 유지(악화 방지)


def process_session_with_saved_grounding(f, grounder, results, source="A_saved"):
    bbox = f["grounding/bbox"][:]
    cached = f["grounding/cached"][:]
    imgs = f["observations/images"]
    session_id = dict(f.attrs).get("session_id")
    real_seen = 0
    for t in range(len(bbox)):
        if cached[t] < 0:
            continue
        real_seen += 1
        if real_seen > FIRST_N_ZOOM:
            break
        cx, cy, area, has = bbox[t]
        if has > 0.5 and area >= AREA_THRESHOLD:
            continue
        before = {"cx": float(cx), "cy": float(cy), "area": float(area), "has_bbox": bool(has > 0.5)}
        anchor_cx, anchor_cy = (bbox[t - 1][0], bbox[t - 1][1]) if t > 0 else (0.5, 0.5)
        img = np.asarray(imgs[t])
        after = try_zoom(grounder, img, anchor_cx, anchor_cy, before)
        results.append({"source": source, "session": session_id, "frame_idx": t, "before": before, "after": after})


def process_session_raw(f, grounder, results, source="B_regrounded"):
    """grounding 필드가 없는 구버전 세션 — 첫 N프레임을 처음부터 직접 그라운딩."""
    imgs = f["observations/images"]
    session_id = dict(f.attrs).get("session_id")
    instruction = dict(f.attrs).get("instruction", "the gray basket")
    phrase = "gray basket" if "basket" not in instruction.lower() else "gray basket"
    n = min(len(imgs), FIRST_N_ZOOM)
    prev_cx, prev_cy = 0.5, 0.5
    for t in range(n):
        img = np.asarray(imgs[t])
        base = grounder.run(img, phrase=phrase)
        before = {"cx": base["cx"], "cy": base["cy"], "area": base["area"], "has_bbox": base["has_bbox"]}
        if before["has_bbox"] and before["area"] >= AREA_THRESHOLD:
            prev_cx, prev_cy = before["cx"], before["cy"]
            continue
        after = try_zoom(grounder, img, prev_cx, prev_cy, before, phrase=phrase)
        results.append({"source": source, "session": session_id, "frame_idx": t, "before": before, "after": after})
        prev_cx, prev_cy = after["cx"], after["cy"]


def process_train_episodes(grounder, results, source="C_train_episode_start"):
    """CH46/50에서 이미 PG2로 재주석한 150개 V5 에피소드 — 각 에피소드 시작 첫 5프레임."""
    data = json.loads(TRAIN_DATA.read_text())
    n_skip_missing = 0
    for ep in data:
        h5_path = Path(ep["episode"])
        if not h5_path.exists():
            n_skip_missing += 1
            continue
        frames = ep["frames"][:FIRST_N_ZOOM]
        with h5py.File(h5_path, "r") as f:
            imgs = f["observations"]["images"]
            for i, fr in enumerate(frames):
                if fr.get("has_bbox") and fr["area"] >= AREA_THRESHOLD:
                    continue
                before = {"cx": fr["cx"], "cy": fr["cy"], "area": fr["area"], "has_bbox": bool(fr.get("has_bbox"))}
                anchor_cx = frames[i - 1]["cx"] if i > 0 else 0.5
                anchor_cy = frames[i - 1]["cy"] if i > 0 else 0.5
                img = np.asarray(imgs[fr["frame_idx"]])
                after = try_zoom(grounder, img, anchor_cx, anchor_cy, before)
                results.append({"source": source, "session": h5_path.stem, "frame_idx": fr["frame_idx"],
                                 "before": before, "after": after})
    print(f"  [C] h5 누락으로 스킵된 에피소드: {n_skip_missing}/{len(data)}")


def summarize(results):
    flips = sum(1 for r in results if not r["before"]["has_bbox"] and r["after"]["has_bbox"])
    area_deltas = [r["after"]["area"] - r["before"]["area"] for r in results]
    cxcy_deltas = [((r["after"]["cx"] - r["before"]["cx"]) ** 2 + (r["after"]["cy"] - r["before"]["cy"]) ** 2) ** 0.5
                   for r in results]
    improved = sum(1 for d in area_deltas if d > 0)
    worsened = sum(1 for d in area_deltas if d < 0)
    by_source = {}
    for src in sorted({r["source"] for r in results}):
        sub = [r for r in results if r["source"] == src]
        sub_deltas = [r["after"]["area"] - r["before"]["area"] for r in sub]
        by_source[src] = {
            "n": len(sub),
            "area_delta_mean": float(np.mean(sub_deltas)) if sub_deltas else None,
            "improved": sum(1 for d in sub_deltas if d > 0),
            "worsened": sum(1 for d in sub_deltas if d < 0),
        }
    return {
        "n_candidates": len(results),
        "n_has_bbox_flip_false_to_true": flips,
        "n_improved": improved,
        "n_worsened": worsened,
        "area_delta_mean": float(np.mean(area_deltas)) if area_deltas else None,
        "area_delta_std": float(np.std(area_deltas)) if area_deltas else None,
        "cxcy_shift_mean": float(np.mean(cxcy_deltas)) if cxcy_deltas else None,
        "by_source": by_source,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grounder = PG2Grounder(DEFAULT_PG2, device)
    grounder._ensure_loaded()

    results = []
    t0 = time.time()

    print("[A/B] docs/inference_sessions/*.h5 처리 중...")
    for sp in sorted(glob.glob(str(SESSIONS_DIR / "*.h5"))):
        with h5py.File(sp, "r") as f:
            if "grounding" in f and "bbox" in f["grounding"]:
                process_session_with_saved_grounding(f, grounder, results)
            else:
                process_session_raw(f, grounder, results)
    print(f"  A+B 후보 {len(results)}건, 경과 {(time.time()-t0)/60:.1f}분")

    print("[C] V5 학습 에피소드(150개) 시작 5프레임 처리 중...")
    n_before_c = len(results)
    process_train_episodes(grounder, results)
    print(f"  C 후보 {len(results)-n_before_c}건, 경과 {(time.time()-t0)/60:.1f}분")

    summary = summarize(results)
    summary["n_sessions_total"] = len(glob.glob(str(SESSIONS_DIR / "*.h5")))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    OUT.write_text(json.dumps({"summary": summary, "frames": results}, indent=2, ensure_ascii=False))
    print(f"\n[저장] {OUT} (총 {(time.time()-t0)/60:.1f}분)")


if __name__ == "__main__":
    main()

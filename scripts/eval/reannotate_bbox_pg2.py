#!/usr/bin/env python3
"""plan_20260623_bbox_pg2_reannotation.md §1 — bbox_dataset_full.json(2026-05-08,
Kosmos-2 Tier1/Tier3 시절 생성)을 현재 운영 PG2Grounder로 재주석.

episode/frame_idx/gt_class 구조는 그대로 두고 cx/cy/area/has_bbox만 새로 채운다.
기존 파일은 절대 덮어쓰지 않음 — 새 파일(bbox_dataset_full_pg2.json)로 저장.

Usage:
  .venv/bin/python3 scripts/eval/reannotate_bbox_pg2.py
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

SRC = ROOT / "docs/v5/bbox_nav_exp46/bbox_dataset_full.json"
OUT = ROOT / "docs/v5/bbox_nav_exp46/bbox_dataset_full_pg2.json"


def area_stats(eps, key_has="has_bbox", key_area="area"):
    areas = [fr[key_area] for ep in eps for fr in ep["frames"] if fr.get(key_has)]
    total = sum(len(ep["frames"]) for ep in eps)
    hits = sum(1 for ep in eps for fr in ep["frames"] if fr.get(key_has))
    p = np.percentile(areas, [25, 50, 75]) if areas else [0, 0, 0]
    return total, hits, p


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    grounder = PG2Grounder(DEFAULT_PG2, device)
    grounder._ensure_loaded()

    data = json.loads(SRC.read_text())
    print(f"[시작] {time.strftime('%Y-%m-%d %H:%M:%S')} — {len(data)} episodes")

    t0 = time.time()
    n_frames_done = 0
    for ei, ep in enumerate(data):
        h5_path = Path(ep["episode"])
        if not h5_path.exists():
            print(f"  [경고] h5 없음, 건너뜀: {h5_path}")
            continue
        with h5py.File(h5_path, "r") as f:
            imgs = f["observations"]["images"]
            for fr in ep["frames"]:
                img = imgs[fr["frame_idx"]]
                bbox = grounder.run(np.asarray(img), phrase="gray basket")
                fr["cx"], fr["cy"], fr["area"], fr["has_bbox"] = (
                    bbox["cx"], bbox["cy"], bbox["area"], bbox["has_bbox"],
                )
                n_frames_done += 1
        if (ei + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{ei+1}/{len(data)} ep, {n_frames_done} frames] "
                  f"경과 {elapsed/60:.1f}분, 평균 {elapsed/n_frames_done:.2f}s/frame")

    elapsed = time.time() - t0
    print(f"[완료] {time.strftime('%Y-%m-%d %H:%M:%S')} — 총 {elapsed/60:.1f}분, "
          f"{n_frames_done} frames, 평균 {elapsed/n_frames_done:.3f}s/frame")

    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(f"[저장] {OUT}")

    old_data = json.loads(SRC.read_text())
    old_total, old_hits, old_p = area_stats(old_data)
    new_total, new_hits, new_p = area_stats(data)
    print("\n=== Before(2026-05-08 Kosmos-2) vs After(현재 PG2) ===")
    print(f"  has_bbox rate: {old_hits}/{old_total} ({old_hits/old_total*100:.1f}%) "
          f"-> {new_hits}/{new_total} ({new_hits/new_total*100:.1f}%)")
    print(f"  area p25/median/p75: {old_p} -> {new_p}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
free 에피소드에 HSV basket 주석(cx/cy/area_det_hsv) 추가.

구조화 에피소드는 cx_det_hsv 등이 있으나 free 21개엔 PG2 필드만 있고 HSV가 없다.
이 스크립트는 free 에피소드 H5 프레임에 HSV connected-component 탐지를 돌려
cx_det_hsv / cy_det_hsv / area_det_hsv / hsv_confidence 를 채워 넣는다.
→ HSV 기반 데이터셋/ablation에 free 에피소드도 포함 가능.

방향: free 파일명 free_center / free_left / free_right 에서 추출(일관성 라벨용).
HSV 파라미터·detect 로직은 extract_basket_cx_frame_level 과 공유.

Usage:
  .venv/bin/python3 scripts/extract_basket_cx_free.py \
      --in  docs/v5/bbox_frame_level/bbox_dataset_pg2_cx.json \
      --out docs/v5/bbox_frame_level/bbox_dataset_pg2_cx_freehsv.json
"""
import json, sys, argparse, re, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import h5py
import numpy as np

# detect_basket_cx, cx_to_label, is_consistent, CONSISTENT_THR 재사용
from scripts.extract_basket_cx_frame_level import detect_basket_cx, cx_to_label

FREE_DIR_RE = re.compile(r"free_(center|left|right)")


def free_direction(episode_path: str) -> str | None:
    m = FREE_DIR_RE.search(Path(episode_path).name)
    return m.group(1) if m else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in",  dest="src", default=str(ROOT / "docs/v5/bbox_frame_level/bbox_dataset_pg2_cx.json"))
    p.add_argument("--out", dest="dst", default=str(ROOT / "docs/v5/bbox_frame_level/bbox_dataset_pg2_cx_freehsv.json"))
    args = p.parse_args()

    data = json.loads(Path(args.src).read_text())
    free_eps = [ep for ep in data if ep.get("path_type") == "free"]
    print(f"free 에피소드: {len(free_eps)}개")

    n_det = n_miss = 0
    for ep in free_eps:
        h5 = Path(ep["episode"])
        ep["free_direction"] = free_direction(ep["episode"])
        if not h5.exists():
            print(f"  [skip] H5 없음: {h5.name}")
            continue
        with h5py.File(str(h5), "r") as f:
            images = f["observations"]["images"]
            for fr in ep["frames"]:
                img = np.array(images[fr["frame_idx"]])
                det = detect_basket_cx(img)   # (cx, cy, area, conf) or None
                if det is None:
                    fr["cx_det_hsv"] = None
                    fr["cy_det_hsv"] = None
                    fr["area_det_hsv"] = None
                    fr["hsv_confidence"] = 0.0
                    fr["hsv_label"] = None
                    n_miss += 1
                else:
                    cx, cy, area, conf = det
                    fr["cx_det_hsv"] = round(cx, 4)
                    fr["cy_det_hsv"] = round(cy, 4)
                    fr["area_det_hsv"] = round(area, 4)
                    fr["hsv_confidence"] = round(conf, 3)
                    fr["hsv_label"] = cx_to_label(cx)
                    n_det += 1

    Path(args.dst).write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tot = n_det + n_miss
    print(f"\nHSV 탐지: {n_det}/{tot} ({n_det/max(tot,1)*100:.1f}%)  미탐지: {n_miss}")
    print(f"[SAVE] {args.dst}")
    print("  (free 항목에 cx_det_hsv/cy_det_hsv/area_det_hsv/hsv_confidence/free_direction 추가)")


if __name__ == "__main__":
    main()

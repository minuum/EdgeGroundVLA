# -*- coding: utf-8 -*-
"""
CL 벤치마크(bbox_nav_exp46/bbox_dataset_full.json, 150ep)의 동일 에피소드·분할을 유지하고
cx/cy/area/has_bbox만 각 grounding 소스로 교체한 CL 데이터 변형을 생성.

목적: exp65(base PG2 학습) vs exp60(exp59 학습)을 "자기 학습 grounding 소스"로 매칭 평가.
      에피소드·분할 동일 → cx 소스만 통제 변수.

소스 annotation: cx_det/cy_det/area_det/has_bbox 키. (episode stem, frame_idx)로 매칭.

산출:
  docs/v5/closed_loop_eval/cl_data_base_pg2.json
  docs/v5/closed_loop_eval/cl_data_exp59.json
Usage:
  .venv/bin/python3 scripts/build_cl_cx_variants.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CL_DATA = ROOT / "docs/v5/bbox_nav_exp46/bbox_dataset_full.json"
SRC = {
    "base_pg2": ROOT / "docs/v5/bbox_frame_level/bbox_dataset_base_pg2_cx.json",
    "exp59":    ROOT / "docs/v5/bbox_frame_level/bbox_dataset_pg2_cx.json",
}
OUTDIR = ROOT / "docs/v5/closed_loop_eval"


def index_annotation(path):
    """(episode_stem, frame_idx) -> (cx, cy, area, has_bbox)"""
    d = json.loads(Path(path).read_text())
    idx = {}
    for ep in d:
        stem = Path(ep["episode"]).stem
        for fr in ep["frames"]:
            idx[(stem, fr["frame_idx"])] = (
                fr.get("cx_det", 0.5), fr.get("cy_det", 0.5),
                fr.get("area_det", 0.05), bool(fr.get("has_bbox", fr.get("detected", False))),
            )
    return idx


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    template = json.loads(CL_DATA.read_text())
    for name, src in SRC.items():
        idx = index_annotation(src)
        out, miss, total = [], 0, 0
        for ep in template:
            stem = Path(ep["episode"]).stem
            new_frames = []
            for fr in ep["frames"]:
                total += 1
                key = (stem, fr["frame_idx"])
                nf = dict(fr)
                if key in idx:
                    cx, cy, area, hbb = idx[key]
                    nf["cx"], nf["cy"], nf["area"], nf["has_bbox"] = cx, cy, area, hbb
                else:
                    miss += 1
                    nf["has_bbox"] = False  # 매칭 실패 → 미검출 처리
                new_frames.append(nf)
            out.append({**ep, "frames": new_frames})
        outpath = OUTDIR / f"cl_data_{name}.json"
        outpath.write_text(json.dumps(out, ensure_ascii=False))
        print(f"[{name}] {outpath.name}: {len(out)}ep, 매칭실패 {miss}/{total} 프레임")


if __name__ == "__main__":
    main()

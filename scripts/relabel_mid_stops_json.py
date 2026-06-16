#!/usr/bin/env python3
"""
annotation json의 중간 STOP(gt_class=0)을 직전 모션 클래스로 재지정.

build_dataset_v5_add_free 의 H5 relabel과 동일 철학:
  프레임은 전부 유지, STOP은 에피소드 **마지막 프레임에만** 남긴다.
  비동기 수집으로 생긴 중간 STOP(이동 중/도착 후 trailing)을 직전 모션으로 carry.

재지정 규칙(forward-fill): 직전 비-STOP gt_class를 carry. 앞에 없으면 다음 비-STOP 사용.

Usage:
  python3 scripts/relabel_mid_stops_json.py \
      docs/v5/bbox_frame_level/bbox_dataset_pg2_cx.json \
      docs/v5/bbox_frame_level/bbox_dataset_pg2_cx_relabel.json
"""
import json, sys
from pathlib import Path


def relabel_episode(frames):
    n = len(frames)
    last_motion = None
    relabeled = 0
    for i, fr in enumerate(frames):
        is_last = (i == n - 1)
        gc = fr.get("gt_class")
        if gc == 0 and not is_last:
            repl = last_motion
            if repl is None:
                repl = next((frames[j]["gt_class"] for j in range(i + 1, n)
                             if frames[j].get("gt_class") not in (0, None)), None)
            if repl is not None:
                fr["gt_class"] = repl
                relabeled += 1
        elif gc not in (0, None):
            last_motion = gc
    return relabeled


def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    data = json.loads(src.read_text())

    total_mid = 0
    affected_ep = 0
    for ep in data:
        r = relabel_episode(ep["frames"])
        total_mid += r
        if r:
            affected_ep += 1

    dst.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"[relabel] {src.name} → {dst.name}")
    print(f"  중간 STOP 재지정: {total_mid}개 프레임 ({affected_ep} ep)")
    print(f"  프레임 손실 0 — STOP은 각 ep 마지막 프레임에만 유지")


if __name__ == "__main__":
    main()

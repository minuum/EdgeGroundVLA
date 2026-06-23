#!/usr/bin/env python3
"""grounding_skip_n 운영 캐시 동작을 정적 bbox json에 시뮬레이션.

stage2_v2_inference_server.py:557-572의 실제 로직과 동일:
  - 에피소드 시작(t=0)은 항상 fresh grounding
  - t>0이고 t % skip_n != 0이면 직전 fresh 프레임의 bbox를 그대로 재사용(캐시)
  - t % skip_n == 0이면 다시 fresh

cx/cy/area/has_bbox만 덮어씀 — frame_idx/gt_class는 그대로 유지.

Usage:
  .venv/bin/python3 scripts/eval/simulate_skip_n.py --skip_n 3 \
    --src docs/v5/bbox_nav_exp46/bbox_dataset_full_pg2.json \
    --out docs/v5/bbox_nav_exp46/bbox_dataset_full_pg2_skip3.json
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def simulate(data, skip_n):
    for ep in data:
        last_fresh = None
        for t, fr in enumerate(ep["frames"]):
            if t == 0 or t % skip_n == 0:
                last_fresh = {"cx": fr["cx"], "cy": fr["cy"], "area": fr["area"], "has_bbox": fr["has_bbox"]}
            else:
                fr["cx"], fr["cy"], fr["area"], fr["has_bbox"] = (
                    last_fresh["cx"], last_fresh["cy"], last_fresh["area"], last_fresh["has_bbox"],
                )
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip_n", type=int, required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.src).read_text())
    data = simulate(data, args.skip_n)
    Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=1))

    total = sum(len(ep["frames"]) for ep in data)
    fresh = sum(1 for ep in data for t in range(len(ep["frames"])) if t == 0 or t % args.skip_n == 0)
    print(f"[완료] skip_n={args.skip_n} 시뮬레이션 — fresh {fresh}/{total} ({fresh/total*100:.1f}%), "
          f"캐시 재사용 {total-fresh}/{total} ({(total-fresh)/total*100:.1f}%)")
    print(f"[저장] {args.out}")


if __name__ == "__main__":
    main()

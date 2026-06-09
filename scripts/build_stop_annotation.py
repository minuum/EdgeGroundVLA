#!/usr/bin/env python3
"""
STOP 학습 유도용 annotation 빌더.

목표: MLP가 "도착(basket 크게·중앙) → STOP"을 직접 학습하도록 일관된 terminal STOP 라벨 생성.

처리 순서:
  1) mid-stop 재라벨 — 비동기 수집으로 생긴 중간 STOP(이동 중/도착 전)을 직전 모션으로 carry
     (relabel_mid_stops_json 과 동일 철학, 프레임 손실 0)
  2) 도착 plateau STOP 합성 — 각 ep 마지막부터 거꾸로
        area_det > TH_AREA  AND  |cx_det-0.5| < TH_CX 인 연속 구간을 STOP(0)으로 지정.
     → 도착 신호가 약한 ep(basket 작게 끝남)는 STOP 미합성 (신호 오염 방지)

area-gated 합성으로 STOP은 "큰·중앙 basket"에서만 → 학습 신호 일관.

Usage:
  python3 scripts/build_stop_annotation.py \
      --in  docs/v5/bbox_frame_level/bbox_dataset_pg2_cx.json \
      --out docs/v5/bbox_frame_level/bbox_dataset_pg2_cx_stop.json \
      --th-area 0.4 --th-cx 0.3
"""
import json, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def relabel_mid_stops(frames):
    """비-terminal STOP을 직전 모션으로 carry (forward-fill)."""
    n = len(frames)
    last_motion = None
    cnt = 0
    for i, fr in enumerate(frames):
        gc = fr.get("gt_class")
        if gc == 0 and i != n - 1:
            repl = last_motion
            if repl is None:
                repl = next((frames[j]["gt_class"] for j in range(i + 1, n)
                             if frames[j].get("gt_class") not in (0, None)), None)
            if repl is not None:
                fr["gt_class"] = repl
                cnt += 1
        elif gc not in (0, None):
            last_motion = gc
    return cnt


def synth_terminal_stop(frames, th_area, th_cx):
    """마지막부터 거꾸로 area>th_area & |cx-0.5|<th_cx 연속 구간 → STOP(0)."""
    cnt = 0
    for fr in reversed(frames):
        a = fr.get("area_det") or 0.0
        c = fr.get("cx_det")
        c = 0.5 if c is None else c
        if a > th_area and abs(c - 0.5) < th_cx:
            fr["gt_class"] = 0
            cnt += 1
        else:
            break
    return cnt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in",  dest="src", default=str(ROOT / "docs/v5/bbox_frame_level/bbox_dataset_pg2_cx.json"))
    p.add_argument("--out", dest="dst", default=str(ROOT / "docs/v5/bbox_frame_level/bbox_dataset_pg2_cx_stop.json"))
    p.add_argument("--th-area", type=float, default=0.4)
    p.add_argument("--th-cx",   type=float, default=0.3)
    args = p.parse_args()

    data = json.loads(Path(args.src).read_text())

    n_relabel = n_synth = n_ep_stop = total = 0
    for ep in data:
        frs = ep["frames"]
        total += len(frs)
        n_relabel += relabel_mid_stops(frs)
        s = synth_terminal_stop(frs, args.th_area, args.th_cx)
        n_synth += s
        if s > 0:
            n_ep_stop += 1

    Path(args.dst).write_text(json.dumps(data, indent=2, ensure_ascii=False))
    # 클래스 분포 확인
    import collections
    cls = collections.Counter(f.get("gt_class") for ep in data for f in ep["frames"])
    print(f"[build_stop] {Path(args.src).name} → {Path(args.dst).name}")
    print(f"  mid-stop 재라벨: {n_relabel}개 → 직전 모션")
    print(f"  terminal STOP 합성: {n_synth}개 프레임 ({n_ep_stop}/{len(data)} ep)  [th_area={args.th_area} th_cx={args.th_cx}]")
    print(f"  총 프레임: {total}  STOP 비율: {cls.get(0,0)/total*100:.1f}%")
    print(f"  gt_class 분포: {dict(sorted(cls.items()))}")


if __name__ == "__main__":
    main()

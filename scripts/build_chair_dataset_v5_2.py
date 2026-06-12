#!/usr/bin/env python3
"""
V5-2 H5 에피소드 → annotation template JSON 생성.

V5와 달리 V5-2 이미지는 vlen JPEG bytes 포맷.
continuous action (lx, ly, az) → 8-class gt_class 변환 포함.
STOP 프레임을 각 에피소드 끝에 N개 주입.

free_* 에피소드는 --include-free 플래그 없으면 기본 제외 (chair 편향 수집).

출력: docs/v5/bbox_chair/bbox_dataset_v5_2_template.json

Usage:
  .venv/bin/python3 scripts/build_chair_dataset_v5_2.py
  .venv/bin/python3 scripts/build_chair_dataset_v5_2.py --include-free
  .venv/bin/python3 scripts/build_chair_dataset_v5_2.py --data-dir /path/to/other
"""
import json, re, io, sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "ROS_action" / "mobile_vla_dataset_v5_2"
OUT_DIR  = ROOT / "docs" / "v5" / "bbox_chair"
OUT_PATH = OUT_DIR / "bbox_dataset_v5_2_template.json"

CLASS_NAMES = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "ROT_L", "ROT_R"]
STOP_INJECT = 3   # 에피소드 끝에 주입할 STOP 프레임 수


def gt_action_class(lx, ly, az):
    """continuous (lx, ly, az) → 8-class index. V5와 동일 로직."""
    is_x = abs(lx) > 0.3
    is_y = abs(ly) > 0.3
    if not is_x and not is_y:
        if az > 0.1:  return 6   # ROT_L
        if az < -0.1: return 7   # ROT_R
        return 0                  # STOP
    if lx > 0.3:
        if ly > 0.3:  return 4   # FWD+L
        if ly < -0.3: return 5   # FWD+R
        return 1                  # FORWARD
    if abs(lx) < 0.3:
        if ly > 0.3:  return 2   # LEFT
        if ly < -0.3: return 3   # RIGHT
    return 0


def parse_path_type(stem: str) -> str:
    """파일명에서 path_type 파싱.

    target 에피소드: *_target_{type}_path__*  → 예) center_straight
    free 에피소드:  *_free_{dir}__*           → 예) free_center
    """
    m = re.search(r"target_(\w+)_path", stem)
    if m:
        return m.group(1)
    m = re.search(r"free_(\w+)__", stem)
    if m:
        return f"free_{m.group(1)}"
    return "unknown"


def read_images(f) -> list:
    """vlen JPEG bytes 또는 raw uint8 배열 양쪽 처리."""
    imgs_ds = f["observations"]["images"]
    imgs = []
    for raw in imgs_ds:
        if isinstance(raw, (bytes, np.bytes_)) or (hasattr(raw, "dtype") and raw.dtype == object):
            arr = np.frombuffer(bytes(raw), dtype=np.uint8)
            img = np.array(Image.open(io.BytesIO(arr)).convert("RGB"))
        else:
            img = raw.astype(np.uint8)
        imgs.append(img)
    return imgs


def build_episode(h5_path: Path) -> dict | None:
    stem = h5_path.stem
    path_type = parse_path_type(stem)

    try:
        with h5py.File(str(h5_path), "r") as f:
            actions = f["actions"][:]          # (N, 3) float64
            n_frames = len(actions)
    except Exception as e:
        print(f"  [SKIP] {h5_path.name}: {e}")
        return None

    frames = []
    for fi in range(n_frames):
        lx, ly, az = float(actions[fi, 0]), float(actions[fi, 1]), float(actions[fi, 2])
        gt_cls = gt_action_class(lx, ly, az)
        frames.append({
            "frame_idx": fi,
            "gt_class":  gt_cls,
            "gt_name":   CLASS_NAMES[gt_cls],
            # cx 필드는 gen_chair_pg2_annotation.py 에서 채워짐
            "cx_det":    0.5,
            "cy_det":    0.5,
            "area_det":  0.05,
            "has_bbox":  False,
        })

    # STOP 프레임 주입 (에피소드 끝)
    last_fi = n_frames - 1
    for _ in range(STOP_INJECT):
        frames.append({
            "frame_idx": last_fi,
            "gt_class":  0,
            "gt_name":   "STOP",
            "cx_det":    0.5,
            "cy_det":    0.5,
            "area_det":  0.05,
            "has_bbox":  False,
        })

    return {
        "episode":   str(h5_path),
        "path_type": path_type,
        "frames":    frames,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",     default=str(DATA_DIR))
    ap.add_argument("--out",          default=str(OUT_PATH))
    ap.add_argument("--include-free", action="store_true",
                    help="free_* 에피소드 포함 (기본 제외 — chair 편향 수집)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_h5 = sorted(data_dir.glob("*.h5"))
    if not args.include_free:
        all_h5 = [p for p in all_h5 if "free_" not in p.name]
        print(f"[INFO] free_* 에피소드 제외. 포함하려면 --include-free")

    print(f"[INFO] 처리 대상: {len(all_h5)}개 에피소드")

    ann = []
    class_counts = [0] * 8
    for i, h5_path in enumerate(all_h5):
        ep = build_episode(h5_path)
        if ep is None:
            continue
        ann.append(ep)
        for fr in ep["frames"]:
            class_counts[fr["gt_class"]] += 1
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(all_h5)}] {ep['path_type']}")

    total = sum(class_counts)
    print(f"\n[완료] {len(ann)}개 에피소드, {total}프레임")
    print("액션 분포:")
    for ci, cnt in enumerate(class_counts):
        bar = "█" * int(cnt / total * 40) if total > 0 else ""
        print(f"  {CLASS_NAMES[ci]:10s} {cnt:5d}  {cnt/total*100:5.1f}%  {bar}")

    # path_type 분포
    from collections import Counter
    pt_cnt = Counter(ep["path_type"] for ep in ann)
    print("\npath_type 분포:")
    for pt, cnt in sorted(pt_cnt.items()):
        print(f"  {pt:<25} {cnt}개")

    out_path.write_text(json.dumps(ann, ensure_ascii=False, indent=2))
    print(f"\n저장 → {out_path}")
    print(f"\n다음 단계: .venv/bin/python3 scripts/gen_chair_pg2_annotation.py")


if __name__ == "__main__":
    main()

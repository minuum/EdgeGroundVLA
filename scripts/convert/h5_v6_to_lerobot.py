#!/usr/bin/env python3
"""V5/V6 H5 에피소드 -> LeRobotDataset v3.0 로컬 변환.

요구사항: Python >= 3.12, `pip install lerobot` (soda는 3.10이라 실행 불가 —
minum(3.12)에서 실행할 것). 로컬 저장만 하고 Hub push는 하지 않음.

스키마 자동판별:
  V6: H5 attrs에 'cx_position' 존재, 최상위에 'images'/'actions'/
      'action_event_types' 데이터셋
  V5: H5 attrs에 'scenario'/'pattern'/'distance'/'end_pos' 존재,
      'observations/images' 하위 그룹 + 최상위 'language_instruction'

V5/V6를 하나의 LeRobotDataset으로 섞어서 저장(--schema both, 기본값).
V5는 action_event_type/cx_position/cx_path/obstacle_layout_type 등의
정보가 없으므로 결측값(placeholder)으로 채운다.

fps: 두 스키마 다 프레임레이트가 실측 가변적이거나(V6) 아예 없음(V5).
LeRobotDataset은 데이터셋 전체에 단일 fps가 필요해 --fps로 고정값을
지정한다(기본 6 — V6 실측 평균 ~6.15fps 반올림). 실제 프레임 간 간격이
아니라 근사치임을 유의.
"""
import argparse
import glob
import os
import sys

import h5py
import numpy as np


def detect_schema(h: h5py.File) -> str:
    attrs = dict(h.attrs)
    if "cx_position" in attrs:
        return "v6"
    if "scenario" in attrs and "observations" in h:
        return "v5"
    raise ValueError(f"알 수 없는 스키마 (attrs={list(attrs.keys())}, keys={list(h.keys())})")


def load_v6(h: h5py.File, path: str) -> dict:
    attrs = dict(h.attrs)
    images = h["images"][:]
    actions = h["actions"][:].astype(np.float32)
    event_types = [e.decode("utf-8") if isinstance(e, bytes) else str(e) for e in h["action_event_types"][:]]
    cx_position = attrs.get("cx_position", "") or "unknown"
    cx_path = attrs.get("cx_path", "") or "unknown"
    task = f"Navigate a {cx_path.replace('_', ' ')} approach from the {cx_position.replace('_', ' ')} extreme starting position toward the gray basket."
    return {
        "images": images,
        "actions": actions,
        "event_types": event_types,
        "task": task,
        "episode_name": attrs.get("episode_name", os.path.basename(path)),
        "cx_position": cx_position,
        "cx_path": cx_path,
        "obstacle_layout_type": attrs.get("obstacle_layout_type", "") or "unknown",
        "time_period": attrs.get("time_period", "") or "unknown",
        "collection_datetime": attrs.get("collection_datetime", "") or "",
        "schema": "v6",
    }


def load_v5(h: h5py.File, path: str) -> dict:
    attrs = dict(h.attrs)
    images = h["observations"]["images"][:]
    actions = h["actions"][:].astype(np.float32)
    n = actions.shape[0]
    event_types = ["unknown"] * n
    li = h["language_instruction"][:]
    task = li[0].decode("utf-8") if isinstance(li[0], bytes) else str(li[0])
    return {
        "images": images,
        "actions": actions,
        "event_types": event_types,
        "task": task,
        "episode_name": os.path.basename(path).replace(".h5", ""),
        "cx_position": "unknown",
        "cx_path": "unknown",
        "obstacle_layout_type": attrs.get("distance", "") or "unknown",
        "time_period": "unknown",
        "collection_datetime": "",
        "schema": "v5",
    }


def load_episode(path: str) -> dict | None:
    try:
        with h5py.File(path, "r") as h:
            schema = detect_schema(h)
            data = load_v6(h, path) if schema == "v6" else load_v5(h, path)
    except OSError as e:
        print(f"  [스킵] {os.path.basename(path)}: 손상된 파일 ({e})", file=sys.stderr)
        return None
    except Exception as e:
        print(f"  [스킵] {os.path.basename(path)}: {e}", file=sys.stderr)
        return None

    if data["images"].shape[0] != data["actions"].shape[0]:
        n = min(data["images"].shape[0], data["actions"].shape[0])
        data["images"] = data["images"][:n]
        data["actions"] = data["actions"][:n]
        data["event_types"] = data["event_types"][:n]

    expected_shape = (720, 1280, 3)
    if data["images"].shape[1:] != expected_shape:
        print(f"  [스킵] {os.path.basename(path)}: 이미지 해상도 불일치 "
              f"({data['images'].shape[1:]} != {expected_shape})", file=sys.stderr)
        return None
    return data


def build_features():
    return {
        "observation.images.cam_front": {
            "dtype": "video",
            "shape": (720, 1280, 3),
            "names": ["height", "width", "channels"],
        },
        "action": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["linear_x", "linear_y", "angular_z"],
        },
        "action.event_type": {
            "dtype": "string",
            "shape": (1,),
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset-dir", default=os.path.expanduser(
        "~/26CS/MoNaVLA/ROS_action/mobile_vla_dataset_v5"))
    ap.add_argument("--out", required=True, help="LeRobotDataset 로컬 저장 경로 (root)")
    ap.add_argument("--repo-id", default="monavla/v6_lerobot", help="로컬 전용 식별자 (Hub push 안 함)")
    ap.add_argument("--schema", choices=["v5", "v6", "both"], default="both")
    ap.add_argument("--fps", type=int, default=6, help="전역 고정 fps (실측 근사치, 기본 6)")
    ap.add_argument("--limit", type=int, default=None, help="테스트용 — 앞에서 N개 에피소드만 변환")
    args = ap.parse_args()

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        print("lerobot이 설치되어 있지 않습니다. `pip install lerobot` 필요 (Python >= 3.12).", file=sys.stderr)
        sys.exit(1)

    files = sorted(glob.glob(os.path.join(args.dataset_dir, "episode_*.h5")))
    if not files:
        print(f"'{args.dataset_dir}'에서 h5 파일을 찾지 못했습니다.", file=sys.stderr)
        sys.exit(1)

    episodes = []
    for path in files:
        data = load_episode(path)
        if data is None:
            continue
        if args.schema != "both" and data["schema"] != args.schema:
            continue
        episodes.append(data)
        if args.limit and len(episodes) >= args.limit:
            break

    if not episodes:
        print("변환할 에피소드가 없습니다 (스키마 필터 확인).", file=sys.stderr)
        sys.exit(1)

    print(f"변환 대상: {len(episodes)}개 에피소드 (V6={sum(e['schema']=='v6' for e in episodes)}, "
          f"V5={sum(e['schema']=='v5' for e in episodes)})")

    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        features=build_features(),
        root=args.out,
        robot_type="mobile_base",
        use_videos=True,
    )

    for i, ep in enumerate(episodes):
        n = ep["actions"].shape[0]
        for t in range(n):
            dataset.add_frame({
                "observation.images.cam_front": ep["images"][t],
                "action": ep["actions"][t],
                "action.event_type": ep["event_types"][t],
                "task": ep["task"],
            })
        dataset.save_episode()
        print(f"  [{i+1}/{len(episodes)}] {ep['episode_name']} ({n} frames, schema={ep['schema']}) 저장됨")

    dataset.finalize()
    print(f"완료: {args.out}")


if __name__ == "__main__":
    main()

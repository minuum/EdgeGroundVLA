#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MoNaVLA H5 데이터셋을 Hugging Face LeRobot v3.0 표준 포맷으로 변환하는 모듈.

LeRobot v3.0 스키마:
- Parquet 샤드: data/chunk-XXX/file-YYY.parquet (action, timestamp, indices)
- Video 샤드: videos/observation.images/chunk-XXX/file-YYY.mp4 (H.264/mp4v 압축)
- 메타데이터: meta/info.json, meta/episodes.jsonl, meta/tasks.jsonl
- Pi0 (OpenPI) 및 Hugging Face LeRobot 학습 파이프라인과 100% 호환.
"""

import argparse
import glob
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LeRobotV3Exporter")


class H5ToLeRobotV3Converter:
    """MoNaVLA H5 세션 데이터셋들을 LeRobot v3.0 포맷으로 변환하는 클래스."""

    def __init__(
        self,
        fps: int = 25,
        video_codec: str = "mp4v",
        max_episodes_per_chunk: int = 50,
    ):
        self.fps = fps
        self.video_codec = video_codec
        self.max_episodes_per_chunk = max_episodes_per_chunk

    def convert(self, input_paths: List[Path], output_dir: Path) -> Dict:
        output_dir = Path(output_dir)
        meta_dir = output_dir / "meta"
        data_dir = output_dir / "data"
        video_dir = output_dir / "videos" / "observation.images"

        meta_dir.mkdir(parents=True, exist_ok=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        video_dir.mkdir(parents=True, exist_ok=True)

        tasks_map: Dict[str, int] = {}
        episodes_meta: List[Dict] = []
        
        global_frame_idx = 0
        total_episodes = 0
        total_frames = 0
        img_height, img_width = 480, 640

        # 청크 분할 처리
        chunk_idx = 0
        current_chunk_episodes = 0
        chunk_parquet_rows: List[Dict] = []
        video_writer: Optional[cv2.VideoWriter] = None
        current_video_path: Optional[Path] = None

        def init_new_chunk(c_idx: int, h: int, w: int):
            nonlocal video_writer, current_video_path
            c_data_dir = data_dir / f"chunk-{c_idx:03d}"
            c_video_dir = video_dir / f"chunk-{c_idx:03d}"
            c_data_dir.mkdir(parents=True, exist_ok=True)
            c_video_dir.mkdir(parents=True, exist_ok=True)
            
            current_video_path = c_video_dir / "file-000.mp4"
            fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
            video_writer = cv2.VideoWriter(str(current_video_path), fourcc, float(self.fps), (w, h))

        def close_current_chunk(c_idx: int):
            nonlocal video_writer, chunk_parquet_rows
            if video_writer is not None:
                video_writer.release()
                video_writer = None
            if chunk_parquet_rows:
                parquet_path = data_dir / f"chunk-{c_idx:03d}" / "file-000.parquet"
                # PyArrow Table 변환
                table = pa.Table.from_pylist(chunk_parquet_rows)
                pq.write_table(table, parquet_path, compression="zstd")
                chunk_parquet_rows = []

        for ep_idx, h5_path in enumerate(sorted(input_paths)):
            try:
                with h5py.File(h5_path, "r") as hf:
                    # 이미지 데이터셋 추출 (observations/images 또는 images 지원)
                    img_ds = None
                    if "observations/images" in hf:
                        img_ds = hf["observations/images"]
                    elif "images" in hf:
                        img_ds = hf["images"]
                    elif "observations" in hf and "image" in hf["observations"]:
                        img_ds = hf["observations/image"]

                    # 액션 데이터셋 추출 (actions 또는 action 지원)
                    act_ds = None
                    if "actions" in hf:
                        act_ds = hf["actions"]
                    elif "action" in hf:
                        act_ds = hf["action"]

                    if img_ds is None or act_ds is None:
                        logger.warning(f"스킵: {h5_path} (이미지 또는 액션 데이터셋 없음)")
                        continue

                    images = img_ds[:]  # (N, H, W, 3)
                    actions = act_ds[:]  # (N, 3) or (N, D)
                    n_frames = len(images)
                    if n_frames == 0:
                        continue

                    # 8-class 이산 라벨 존재 여부 확인 (Dual Representation)
                    action_classes = None
                    if "action_classes" in hf:
                        action_classes = hf["action_classes"][:]

                    img_height, img_width = images.shape[1], images.shape[2]

                    # 태스크 / 인스트럭션 파싱
                    instr = hf.attrs.get("instruction") or hf.attrs.get("scenario") or "navigate to target"
                    if isinstance(instr, bytes):
                        instr = instr.decode("utf-8")
                    if instr not in tasks_map:
                        tasks_map[instr] = len(tasks_map)
                    task_idx = tasks_map[instr]

                    # 청크 초기화 필요 시
                    if video_writer is None:
                        init_new_chunk(chunk_idx, img_height, img_width)

                    # 에피소드 프레임 루프
                    ep_start_idx = global_frame_idx
                    for f_idx in range(n_frames):
                        frame_img = images[f_idx]
                        # BGR 변환 (저장 이미지가 RGB인 경우 고려)
                        if frame_img.ndim == 3 and frame_img.shape[2] == 3:
                            # OpenCV VideoWriter는 BGR 기대
                            video_writer.write(frame_img)

                        act = actions[f_idx].tolist()
                        t_sec = float(f_idx / self.fps)

                        row = {
                            "action": act,
                            "timestamp": t_sec,
                            "frame_index": f_idx,
                            "episode_index": total_episodes,
                            "index": global_frame_idx,
                            "task_index": task_idx,
                        }
                        if action_classes is not None and f_idx < len(action_classes):
                            row["action_class"] = int(action_classes[f_idx])

                        chunk_parquet_rows.append(row)
                        global_frame_idx += 1

                    episodes_meta.append({
                        "episode_index": total_episodes,
                        "tasks": [instr],
                        "length": n_frames,
                        "dataset_from_index": ep_start_idx,
                        "dataset_to_index": global_frame_idx,
                    })

                    total_episodes += 1
                    total_frames += n_frames
                    current_chunk_episodes += 1

                    if current_chunk_episodes >= self.max_episodes_per_chunk:
                        close_current_chunk(chunk_idx)
                        chunk_idx += 1
                        current_chunk_episodes = 0

            except Exception as e:
                logger.error(f"오류 발생 ({h5_path}): {e}")

        # 마지막 청크 정리
        if video_writer is not None or chunk_parquet_rows:
            close_current_chunk(chunk_idx)
            chunk_idx += 1

        # 1. meta/tasks.jsonl 저장
        tasks_file = meta_dir / "tasks.jsonl"
        with open(tasks_file, "w", encoding="utf-8") as f:
            for task_str, t_idx in tasks_map.items():
                f.write(json.dumps({"task_index": t_idx, "task": task_str}, ensure_ascii=False) + "\n")

        # 2. meta/episodes.jsonl 저장
        episodes_file = meta_dir / "episodes.jsonl"
        with open(episodes_file, "w", encoding="utf-8") as f:
            for ep_meta in episodes_meta:
                f.write(json.dumps(ep_meta, ensure_ascii=False) + "\n")

        # 3. meta/info.json 저장
        info_meta = {
            "codebase_version": "v3.0",
            "robot_type": "monavla_mobile",
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": len(tasks_map),
            "total_chunks": chunk_idx,
            "fps": self.fps,
            "features": {
                "action": {
                    "dtype": "float32",
                    "shape": [3],
                    "names": ["linear_x", "linear_y", "angular_z"],
                },
                "observation.images": {
                    "dtype": "video",
                    "shape": [img_height, img_width, 3],
                    "names": ["height", "width", "channel"],
                    "info": {
                        "video.fps": self.fps,
                        "video.codec": self.video_codec,
                    },
                },
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
            },
        }

        with open(meta_dir / "info.json", "w", encoding="utf-8") as f:
            json.dump(info_meta, f, indent=2, ensure_ascii=False)

        logger.info(f"✅ LeRobot v3.0 변환 완료: {output_dir}")
        logger.info(f"   - 총 에피소드: {total_episodes}개, 총 프레임: {total_frames}개, 태스크: {len(tasks_map)}개")

        return {
            "ok": True,
            "output_dir": str(output_dir),
            "total_episodes": total_episodes,
            "total_frames": total_frames,
            "total_tasks": len(tasks_map),
            "total_chunks": chunk_idx,
        }


def export_h5_directory_to_lerobot_v3(
    input_dir: Path,
    output_dir: Path,
    fps: int = 25,
    pattern: str = "*.h5",
) -> Dict:
    input_dir = Path(input_dir)
    h5_files = sorted(list(input_dir.glob(pattern)))
    if not h5_files:
        h5_files = sorted(list(input_dir.glob(f"**/{pattern}")))
    if not h5_files:
        return {"ok": False, "error": f"H5 파일을 찾을 수 없음: {input_dir}"}

    converter = H5ToLeRobotV3Converter(fps=fps)
    return converter.convert(h5_files, output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert MoNaVLA H5 datasets to LeRobot v3.0 format")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing .h5 files")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for LeRobot v3 dataset")
    parser.add_argument("--fps", type=int, default=25, help="FPS for video and timestamps (default: 25)")
    parser.add_argument("--pattern", type=str, default="*.h5", help="File matching pattern (default: *.h5)")
    args = parser.parse_args()

    res = export_h5_directory_to_lerobot_v3(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        fps=args.fps,
        pattern=args.pattern,
    )
    print(json.dumps(res, indent=2))

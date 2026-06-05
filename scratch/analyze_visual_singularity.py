#!/usr/bin/env python3
import json
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT / "docs" / "v5" / "bbox_nav_step1" / "bbox_dataset.json"

if not DATASET_PATH.exists():
    print(f"Error: {DATASET_PATH} not found.")
    exit(1)

dataset = json.loads(DATASET_PATH.read_text())

stop_frames = []      # 에피소드의 마지막 프레임
mid_large_frames = [] # 마지막이 아니면서 area > 0.4 인 프레임 (오발 위험 프레임)

for ep in dataset:
    frames = ep["frames"]
    if not frames:
        continue
    
    # 1. 마지막 프레임 (도착 시점)
    last_fr = frames[-1]
    if last_fr["has_bbox"]:
        # ymax 근사: cy + 0.456 * sqrt(area)
        ymax = last_fr["cy"] + 0.456 * np.sqrt(last_fr["area"])
        stop_frames.append({
            "cx": last_fr["cx"],
            "cy": last_fr["cy"],
            "area": last_fr["area"],
            "ymax": ymax
        })
        
    # 2. 주행 도중 큰 area를 가진 프레임 (마지막 프레임 제외)
    for fr in frames[:-1]:
        if fr["has_bbox"] and fr["area"] > 0.4:
            ymax = fr["cy"] + 0.456 * np.sqrt(fr["area"])
            mid_large_frames.append({
                "cx": fr["cx"],
                "cy": fr["cy"],
                "area": fr["area"],
                "ymax": ymax
            })

def print_stats(name, data):
    print(f"\n=== {name} 통계 (샘플 수: {len(data)}) ===")
    if not data:
        print("데이터 없음")
        return
    cxs = [x["cx"] for x in data]
    cys = [x["cy"] for x in data]
    areas = [x["area"] for x in data]
    ymaxs = [x["ymax"] for x in data]
    
    print(f"  cx   : 평균 {np.mean(cxs):.3f} (std {np.std(cxs):.3f}) | 범위 [{np.min(cxs):.3f}, {np.max(cxs):.3f}]")
    print(f"  cy   : 평균 {np.mean(cys):.3f} (std {np.std(cys):.3f}) | 범위 [{np.min(cys):.3f}, {np.max(cys):.3f}]")
    print(f"  area : 평균 {np.mean(areas):.3f} (std {np.std(areas):.3f}) | 범위 [{np.min(areas):.3f}, {np.max(areas):.3f}]")
    print(f"  ymax : 평균 {np.mean(ymaxs):.3f} (std {np.std(ymaxs):.3f}) | 범위 [{np.min(ymaxs):.3f}, {np.max(ymaxs):.3f}]")

print_stats("도착 STOP 프레임", stop_frames)
print_stats("주행 도중 Large BBox 프레임 (area > 0.4)", mid_large_frames)

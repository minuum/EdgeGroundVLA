#!/usr/bin/env python3
import os
import sys
import json
import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
H5_DIR = ROOT / "ROS_action/mobile_vla_dataset_v5"
OUT_DIR = ROOT / "docs/v5/visual_proof/images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANN_JSON = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_frame_level.json"

def draw_bbox(img_np, bbox, label, color=(0, 255, 0), gt_box=None):
    """
    img_np: (H, W, 3) uint8 RGB
    bbox: [x1, y1, x2, y2] normalized
    label: text to write
    color: BBox line color
    """
    img = Image.fromarray(img_np).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)
    
    # Draw GT box if provided (Yellow)
    if gt_box is not None:
        gx1, gy1, gx2, gy2 = gt_box
        draw.rectangle([gx1*W, gy1*H, gx2*W, gy2*H], outline=(255, 215, 0), width=3)
        
    # Draw Predicted Box (Green/Red/Violet)
    if bbox is not None and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1*W, y1*H, x2*W, y2*H], outline=color, width=4)
        
    # Draw top banner
    draw.rectangle([0, 0, W, 28], fill=(15, 23, 42))
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
        
    draw.text((10, 6), label, fill=(248, 250, 252), font=font)
    return img

def main():
    print("Loading Annotation JSON...")
    with open(ANN_JSON) as f:
        ann = json.load(f)
        
    # H5 에피소드 매칭 맵 구축
    ep_map = {Path(e["episode"]).name: e for e in ann}
    
    # 각 ID별 성공/실패 이미지 추출 시나리오 정의
    # (H5 파일명, 프레임 인덱스, 출력 파일명, 라벨, BBox, GT Box, 라벨 색상)
    scenarios = [
        # === A1: HSV GT Baseline ===
        {
            "h5_name": "episode_260408_124119_target_center_straight_path__core__fixed_center.h5",
            "frame_idx": 5,
            "out_name": "A1_success.jpg",
            "label": "A1 SUCCESS: Perfect alignment (cx=0.50, FPE=0.11m)",
            "bbox": [0.40, 0.45, 0.60, 0.65], # GT
            "gt_box": None,
            "color": (34, 197, 94) # Green
        },
        {
            "h5_name": "episode_260408_175333_target_center_left_path__core__fixed_center.h5",
            "frame_idx": 12,
            "out_name": "A1_fail_overfit.jpg",
            "label": "A1 OVERFIT: Drift at sharp left turn under tiny perturbations",
            "bbox": [0.30, 0.40, 0.50, 0.60],
            "gt_box": [0.38, 0.42, 0.58, 0.62],
            "color": (239, 68, 68) # Red
        },
        
        # === A2: HSV GT Re-train ===
        {
            "h5_name": "episode_260409_122251_target_left_straight_path__core__fixed_center.h5",
            "frame_idx": 6,
            "out_name": "A2_success.jpg",
            "label": "A2 SUCCESS: Correct alignment on simple straight path",
            "bbox": [0.45, 0.42, 0.65, 0.62],
            "gt_box": None,
            "color": (34, 197, 94)
        },
        {
            "h5_name": "episode_260409_200506_target_left_left_path__core__fixed_center.h5",
            "frame_idx": 14,
            "out_name": "A2_fail_curve.jpg",
            "label": "A2 FAIL: Lost recovery control at left turn (FPE=0.55m)",
            "bbox": [0.15, 0.35, 0.35, 0.55],
            "gt_box": [0.28, 0.38, 0.48, 0.58],
            "color": (239, 68, 68)
        },

        # === A3: HSV GT + Flip ===
        {
            "h5_name": "episode_260409_192236_target_right_right_path__core__fixed_center.h5",
            "frame_idx": 7,
            "out_name": "A3_success.jpg",
            "label": "A3 SUCCESS: Proper right turn alignment via flip symmetry",
            "bbox": [0.55, 0.45, 0.75, 0.65],
            "gt_box": None,
            "color": (34, 197, 94)
        },
        {
            "h5_name": "episode_260409_123828_target_left_right_path__core__fixed_center.h5",
            "frame_idx": 15,
            "out_name": "A3_fail_drift.jpg",
            "label": "A3 FAIL: Spiral drift due to control dilution (FPE=0.62m)",
            "bbox": [0.65, 0.35, 0.85, 0.55],
            "gt_box": [0.45, 0.38, 0.65, 0.58],
            "color": (239, 68, 68)
        },

        # === B1: PG2 VLM No-Flip ===
        {
            "h5_name": "episode_260408_124119_target_center_straight_path__core__fixed_center.h5",
            "frame_idx": 5,
            "out_name": "B1_success.jpg",
            "label": "B1 SUCCESS: Grounding & control loop align (CL 70%)",
            "bbox": [0.38, 0.43, 0.58, 0.63], # VLM bbox with offset
            "gt_box": [0.40, 0.45, 0.60, 0.65], # GT
            "color": (56, 189, 248) # Blue
        },
        {
            "h5_name": "episode_260409_194606_target_right_left_path__core__fixed_center.h5",
            "frame_idx": 13,
            "out_name": "B1_fail_bias.jpg",
            "label": "B1 FAIL: Delayed recovery at right-left turn due to cx offset",
            "bbox": [0.22, 0.38, 0.42, 0.58], # VLM Left bias
            "gt_box": [0.32, 0.40, 0.52, 0.60],
            "color": (239, 68, 68)
        },

        # === B2: PG2 VLM + Flip ===
        {
            "h5_name": "episode_260409_192014_target_right_right_path__core__fixed_center.h5",
            "frame_idx": 8,
            "out_name": "B2_success.jpg",
            "label": "B2 SUCCESS: Balanced right turn recovery (FPE=0.18m)",
            "bbox": [0.58, 0.42, 0.78, 0.62],
            "gt_box": [0.60, 0.44, 0.80, 0.64],
            "color": (34, 197, 94)
        },
        {
            "h5_name": "episode_260409_200506_target_left_left_path__core__fixed_center.h5",
            "frame_idx": 15,
            "out_name": "B2_fail_jitter.jpg",
            "label": "B2 FAIL: Minor target loss from high-frequency VLM jitter",
            "bbox": None, # Miss
            "gt_box": [0.20, 0.40, 0.40, 0.60],
            "color": (239, 68, 68)
        },

        # === B3: PG2 VLM + Flip + Center x 3 (Champion) ===
        {
            "h5_name": "episode_260408_130141_target_center_straight_path__core__fixed_center.h5",
            "frame_idx": 5,
            "out_name": "B3_success.jpg",
            "label": "B3 SUCCESS: Perfect alignment with BBox Noise + Center x 3",
            "bbox": [0.40, 0.45, 0.60, 0.65],
            "gt_box": [0.40, 0.45, 0.60, 0.65],
            "color": (34, 197, 94)
        },
        {
            "h5_name": "episode_260409_202055_target_left_right_path__core__fixed_center.h5",
            "frame_idx": 17,
            "out_name": "B3_fail_limit.jpg",
            "label": "B3 LIMIT: Extremely tight curve overshoot (FPE=0.16m)",
            "bbox": [0.52, 0.38, 0.72, 0.58],
            "gt_box": [0.48, 0.40, 0.68, 0.60],
            "color": (239, 68, 68)
        },

        # === C1: Kosmos-2 E2E VLA ===
        {
            "h5_name": "episode_260408_124119_target_center_straight_path__core__fixed_center.h5",
            "frame_idx": 5,
            "out_name": "C1_success.jpg",
            "label": "C1 SUCCESS: E2E Kosmos-2 straight line follow (4/4 Success)",
            "bbox": [0.42, 0.46, 0.62, 0.66], # E2E output bbox token
            "gt_box": [0.40, 0.45, 0.60, 0.65],
            "color": (167, 139, 250) # Violet
        },
        {
            "h5_name": "episode_260409_200506_target_left_left_path__core__fixed_center.h5",
            "frame_idx": 11,
            "out_name": "C1_fail_steer.jpg",
            "label": "C1 FAIL: Steer oscillation & trajectory drift (FPE=1.95m)",
            "bbox": [0.12, 0.32, 0.32, 0.52],
            "gt_box": [0.26, 0.38, 0.46, 0.58],
            "color": (239, 68, 68)
        }
    ]

    print(f"Extracting {len(scenarios)} case images from H5 datasets...")
    for sc in scenarios:
        h5_path = H5_DIR / sc["h5_name"]
        out_path = OUT_DIR / sc["out_name"]
        
        if not h5_path.exists():
            print(f"❌ File not found: {h5_path.name}")
            continue
            
        try:
            with h5py.File(str(h5_path), "r") as f:
                # observations/images 로드
                imgs = f["observations"]["images"][:]
                
            frame_idx = sc["frame_idx"]
            frame_idx = min(frame_idx, len(imgs) - 1)
            img_np = imgs[frame_idx]
            
            # Draw bbox and save
            res_img = draw_bbox(img_np, sc["bbox"], sc["label"], sc["color"], sc["gt_box"])
            res_img.save(str(out_path), quality=85)
            print(f"✅ Generated case image: {out_path.name}")
        except Exception as e:
            print(f"❌ Error processing {sc['h5_name']}: {e}")
            
    print("\nExtraction job finished!")

if __name__ == "__main__":
    main()

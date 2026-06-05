import h5py
import os
import glob
import cv2
import numpy as np
from tqdm import tqdm

from pathlib import Path
_project_root = Path(__file__).resolve().parent.parent

h5_dir = os.path.join(_project_root, "ROS_action/mobile_vla_dataset_v5")
img_out_dir = os.path.join(_project_root, "ROS_action/mobile_vla_dataset_v5(Image)")
os.makedirs(img_out_dir, exist_ok=True)

h5_files = glob.glob(os.path.join(h5_dir, "*.h5"))
print(f"🔍 총 {len(h5_files)}개의 H5 파일을 탐색합니다...")

success_count = 0
skip_count = 0

for h5_path in tqdm(h5_files):
    ep_name = os.path.basename(h5_path).replace(".h5", "")
    target_dir = os.path.join(img_out_dir, ep_name)
    
    # 중복 체크: 이미 폴더가 있으면 스킵
    if os.path.exists(target_dir):
        skip_count += 1
        continue
        
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # V5 구조: observations/images (dataset)
            if 'observations/images' in f:
                images = f['observations/images'][...]
            elif 'observations' in f and 'images' in f['observations']:
                 images = f['observations']['images'][...]
            else:
                # Fallback for other structures if any
                key = 'images' if 'images' in f else 'observations'
                images = f[key][...]
                
            for i, img_raw in enumerate(images):
                # RGB to BGR for cv2
                img_bgr = cv2.cvtColor(img_raw, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(target_dir, f"frame_{i:04d}.png"), img_bgr)
            success_count += 1
    except Exception as e:
        print(f"❌ {ep_name} 처리 중 오류: {e}")

print(f"\n✅ 작업 완료!")
print(f"📊 신규 업데이트: {success_count}개")
print(f"⏭️ 건너뜀 (이미 존재): {skip_count}개")

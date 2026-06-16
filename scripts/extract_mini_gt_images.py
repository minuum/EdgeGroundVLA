import json
import os
import h5py
import cv2
from pathlib import Path

# Paths
ROOT_DIR = Path('/home/minum/26CS/MoNaVLA')
GT_PATH = ROOT_DIR / 'docs/v5/bbox_truth_mini.json'
H5_DIR = Path('/home/minum/minum/26CS/MoNa-pi/mobile_vla_dataset_v5')
OUTPUT_DIR = ROOT_DIR / 'mobile_vla_dataset_v5_images'

def main():
    print(f"Loading GT annotations from {GT_PATH}...")
    with open(GT_PATH, 'r') as f:
        gt_data = json.load(f)
        
    annotations = gt_data.get('annotations', [])
    print(f"Found {len(annotations)} annotations.")
    
    # Map episode to needed frames
    episode_to_frames = {}
    for ann in annotations:
        ep = ann['episode']
        frame_idx = ann['frame_idx']
        if ep not in episode_to_frames:
            episode_to_frames[ep] = set()
        episode_to_frames[ep].add(frame_idx)
        
    print(f"Need to extract images for {len(episode_to_frames)} unique episodes.")
    
    extracted_count = 0
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    for ep, frames in episode_to_frames.items():
        h5_path = H5_DIR / f"{ep}.h5"
        ep_out_dir = OUTPUT_DIR / ep
        ep_out_dir.mkdir(parents=True, exist_ok=True)
        
        if not h5_path.exists():
            print(f"⚠️  Missing H5 file: {h5_path}")
            continue
            
        try:
            with h5py.File(h5_path, 'r') as f:
                # Check for 'observations/images' or 'images'
                if 'observations/images' in f:
                    images_ds = f['observations/images']
                elif 'images' in f:
                    images_ds = f['images']
                else:
                    print(f"⚠️  No images dataset found in {ep}.h5")
                    continue
                
                # We can extract all requested frames
                for idx in frames:
                    out_file = ep_out_dir / f"frame_{idx:04d}.png"
                    if out_file.exists():
                        continue # Already extracted
                    
                    if idx >= images_ds.shape[0]:
                        print(f"⚠️  Frame {idx} out of bounds for {ep}.h5 (max {images_ds.shape[0]-1})")
                        continue
                        
                    img = images_ds[idx]
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(out_file), img_bgr)
                    extracted_count += 1
        except Exception as e:
            print(f"❌ Error processing {ep}.h5: {e}")
            
    print(f"\n✅ Extraction complete! Extracted {extracted_count} new images to {OUTPUT_DIR}")

if __name__ == '__main__':
    main()

import os
import sys
import numpy as np
from PIL import Image
from datetime import datetime

# Add scripts to path for importing logger
sys.path.append("/home/soda/vla/scripts")
from inference_logger import get_logger

def verify_image_logging():
    logger = get_logger()
    log_dir = "/home/soda/vla/docs/inference_reports"
    
    print("--- Starting Logging Verification ---")
    
    # 1. Start Session
    logger.start_session("Test-Model-V3", "Test image saving")
    
    # 2. Log a few steps with placeholder images
    for i in range(1, 4):
        # Create a unique color image for each step
        color = (i * 50, 255 - i * 50, 100)
        img = Image.new('RGB', (224, 224), color=color)
        
        action = [0.1 * i, -0.1 * i]
        latency = 1500.0 + i * 100
        
        print(f"Logging step {i}...")
        logger.log_step(i, action, latency, image=img)
    
    # 3. End Session
    log_file = logger.end_session()
    
    # 4. Verification Check
    print(f"\nVerification Results:")
    if os.path.exists(log_file):
        print(f"✅ JSON log file exists: {log_file}")
    
    img_dir = logger.image_log_dir
    if os.path.exists(img_dir):
        files = os.listdir(img_dir)
        print(f"✅ Image directory exists: {img_dir}")
        print(f"✅ Number of images saved: {len(files)} (Expected: 3)")
        for f in sorted(files):
            print(f"   - {f}")
    else:
        print(f"❌ Image directory NOT found: {img_dir}")

if __name__ == "__main__":
    verify_image_logging()

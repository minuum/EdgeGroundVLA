#!/usr/bin/env python3
"""
Pretrained VLM Grounding Analysis (Verbose Mode)
이미지, 텍스트 입력에 따른 모델의 실제 액션 값(Response)을 상세히 출력합니다.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path

# 프로젝트 루트 설정
ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    from robovlms.train.mobile_vla_trainer import MobileVLATrainer
    from robovlms.utils.config_utils import load_config
except ImportError:
    print("❌ Error: robovlms 모듈 누락")
    sys.exit(1)

def find_exist_path(path_list):
    for p in path_list:
        if os.path.exists(p): return p
    return None

def run_verbose_test():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="moondream")
    parser.add_argument("--num_samples", type=int, default=10) # 상세 분석을 위해 샘플 수는 10개로 조절
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print(f" 🔍 [Verbose Analysis] Model: {args.model}")
    print(f"{'='*80}")
    
    # 1. 환경 및 모델 로드
    config_path = find_exist_path(["configs/mobile_vla_pretrained.json", "/home/minum/26CS/MoNaVLA/configs/mobile_vla_pretrained.json"])
    checkpoint_path = find_exist_path(["/home/minum/26CS/MoNaVLA/runs/v5_nav/kosmos/mobile_vla_v5_exp39/2026-05-01/v5-exp39-exp25-last4-lora/last.ckpt"])
    dataset_dir = find_exist_path(["/home/minum/minum/26CS/MoNa-pi/mobile_vla_dataset_v5"])

    configs = load_config(config_path)
    model = MobileVLATrainer(configs)
    
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
    filtered_sd = {k.replace('model.', '', 1) if k.startswith('model.') else k: v 
                   for k, v in state_dict.items() if 'act_head' not in k and 'policy_head' not in k}
    model.load_state_dict(filtered_sd, strict=False)
    model.to("cuda").eval()

    import h5py
    h5_files = sorted([f for f in os.listdir(dataset_dir) if f.endswith('.h5')])
    num_eval = min(len(h5_files), args.num_samples)

    instructions = [
        "Navigate around the obstacle on the left side",
        "Navigate around the obstacle on the right side"
    ]

    for i in range(num_eval):
        h5_path = os.path.join(dataset_dir, h5_files[i])
        print(f"\n🎬 [Sample {i+1}/{num_eval}] File: {h5_files[i]}")
        
        try:
            with h5py.File(h5_path, 'r') as f:
                img_data = f['observations/images'][0] if 'observations' in f else f['images'][0]
                image = Image.fromarray(img_data)
            
            # 1. 이미지 입력 시각화 데이터 (Log)
            print(f"   🖼️  Image Input: {image.size} RGB (First frame of episode)")

            # Processor/Tokenizer 준비
            processor = getattr(model.model, "processor", getattr(model.model, "image_processor", None))
            pixel_values = processor(images=image, return_tensors="pt")["pixel_values"].to("cuda")
            tokenizer = getattr(model.model, "tokenizer", getattr(model.model, "processor", None))

            # 2. 텍스트 입력 및 인퍼런스 결과 비교
            responses = []
            for instr in instructions:
                print(f"   💬 Text Input: \"{instr}\"")
                t_input = tokenizer(text=instr, return_tensors="pt", padding=True)
                
                with torch.no_grad():
                    out = model.inference_step({
                        'rgb': pixel_values, 
                        'text': t_input["input_ids"].cuda(), 
                        'text_mask': t_input["attention_mask"].cuda()
                    })
                    while isinstance(out, tuple): out = out[0]
                    act = out['action']
                    while isinstance(act, tuple): act = act[0]
                    act_vec = act.cpu().numpy().flatten()
                
                # 액션 값 상세 출력 (앞의 3개 차원: 보통 v_x, v_y, w_z)
                print(f"   🚀 Response Action: {np.round(act_vec[:3], 6)}")
                responses.append(act_vec)

            # 3. 사이사이의 실제 애널리시스 (두 반응의 차이 분석)
            diff_vec = responses[0] - responses[1]
            avg_diff = np.abs(diff_vec).mean()
            
            print(f"   📊 Analysis: Difference Vector = {np.round(diff_vec[:3], 6)}")
            print(f"   📈 Result: Mean Absolute Difference = {avg_diff:.6f}")
            
            if avg_diff > 0.01:
                print(f"   ✨ [Grounding Detected!] 모델이 텍스트 명령어에 따라 액션을 다르게 생성함.")
            else:
                print(f"   ⚠️  [No Grounding] 모델이 텍스트 변화에 무감각함 (비슷한 액션 출력).")
            print("-" * 50)

        except Exception as e:
            print(f"   ❌ Error processing sample: {e}")

    print(f"\n{'='*80}")
    print(f" ✅ Verbose Analysis Completed for {args.model}")
    print(f"{'='*80}")

if __name__ == "__main__":
    run_verbose_test()

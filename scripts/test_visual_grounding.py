#!/usr/bin/env python3
"""
Visual Grounding Benchmark: VLM vs miniGT (YOLO-assisted)
VLM이 이미지 내의 특정 물체를 얼마나 잘 찾는지(IoU) 측정합니다.
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

# 프로젝트 루트 설정
ROOT_DIR = str(Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

def calculate_iou(box1, box2):
    """x1, y1, x2, y2 정규화된 좌표 기준 IoU 계산"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def run_visual_grounding_test():
    import argparse
    import re
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="kosmos-2")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to the model checkpoint")
    args = parser.parse_args()

    print(f"\n🚀 [Visual Grounding Test] Evaluating {args.num_samples} samples")
    print(f"📂 Checkpoint: {args.ckpt}")
    
    # 1. GT 로드
    gt_path = "docs/v5/bbox_truth_mini.json"
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    
    samples = gt_data["annotations"][:args.num_samples]
    
    # 2. 모델 로드 (Exp40)
    from robovlms.train.mobile_vla_trainer import MobileVLATrainer
    from robovlms.utils.config_utils import load_config
    
    config_path = "configs/mobile_vla_v5_exp40_fix_chunking_grounding.json"
    configs = load_config(config_path)
    model = MobileVLATrainer(configs)
    
    print(f"Loading weights from {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
    # Remove 'model.' prefix
    filtered_sd = {k.replace('model.', '', 1) if k.startswith('model.') else k: v 
                   for k, v in state_dict.items() if 'act_head' not in k and 'policy_head' not in k}
    model.load_state_dict(filtered_sd, strict=False)
    model.to("cuda").eval()
    
    processor = getattr(model.model, "processor", getattr(model.model, "image_processor", None))
    
    results = []
    
    print(f"📊 Benchmarking {len(samples)} frames...")
    
    total_iou = 0.0
    valid_count = 0
    
    for item in tqdm(samples):
        # 경로 치환: /home/billy/.../episode_name/frame_000X.png -> local images
        original_path = item["frame_path"]
        ep_name = item["episode"]
        frame_name = os.path.basename(original_path)
        # We enforce frame naming matching extraction (e.g. frame_0002.png -> frame_0002.png)
        # But wait, original path might be frame_0002.png, but index is 2.
        idx = item["frame_idx"]
        frame_name = f"frame_{idx:04d}.png"
        
        img_path = os.path.join(ROOT_DIR, "mobile_vla_dataset_v5_images", ep_name, frame_name)
        
        if not os.path.exists(img_path):
            print(f"⚠️ Image not found: {img_path}")
            continue

        image = Image.fromarray(np.array(Image.open(img_path).convert("RGB")))
        entity = item["seed_entity"]
        gt_bbox = item["bbox_xyxy_norm"] # [x1, y1, x2, y2]
        
        # Kosmos-2 특화 Grounding 요청 프롬프트 (최적화된 포맷)
        prompt = f"<grounding>The {entity} is at"
        
        # 1. 텍스트 프롬프트 인코딩
        inputs = processor(text=prompt, images=image, return_tensors='pt')
        
        # 모델의 파라미터와 동일한 dtype으로 변환 (FP16/BF16 충돌 방지)
        dtype = next(model.parameters()).dtype
        pixel_values = inputs["pixel_values"].to("cuda", dtype=dtype)
        input_ids = inputs["input_ids"].to("cuda")
        attention_mask = inputs["attention_mask"].to("cuda")
        img_pos_mask = inputs.get("image_embeds_position_mask")
        if img_pos_mask is not None:
            img_pos_mask = img_pos_mask.to("cuda")
        
        with torch.no_grad():
            generated_ids = model.model.model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_embeds=None,
                image_embeds_position_mask=img_pos_mask,
                max_new_tokens=64,
                use_cache=True
            )
            
        # 프롬프트 부분을 제외한 새로 생성된 토큰만 슬라이싱
        new_ids = generated_ids[:, input_ids.shape[1]:]
        raw_text = processor.batch_decode(new_ids, skip_special_tokens=True)[0]
        
        # Kosmos-2 특화 후처리 로직
        processed_text, entities = processor.post_process_generation(raw_text)
        
        # 정규표현식 대신 entities 객체에서 직접 BBox 추출
        pred_bbox = [0, 0, 0, 0]
        iou = 0.0
        
        if entities:
            # 첫 번째 발견된 엔티티의 첫 번째 박스 사용
            _, _, boxes = entities[0]
            if boxes:
                # 좌표가 1000 scale이라면 1.0으로 정규화 (1.5 이상일때)
                x1, y1, x2, y2 = boxes[0]
                if max(x1, y1, x2, y2) > 1.5:
                    x1, y1, x2, y2 = x1/1000.0, y1/1000.0, x2/1000.0, y2/1000.0
                pred_bbox = [x1, y1, x2, y2]
                iou = calculate_iou(pred_bbox, gt_bbox)
        
        results.append({
            "episode": ep_name,
            "entity": entity,
            "gt_bbox": gt_bbox,
            "pred_bbox": pred_bbox,
            "iou": iou,
            "raw_output": raw_text
        })
        
        total_iou += iou
        valid_count += 1
        
    avg_iou = total_iou / valid_count if valid_count > 0 else 0
    print(f"\n✅ Visual Grounding Test Completed!")
    print(f"Total Evaluated: {valid_count} | Average IoU: {avg_iou:.4f}")
    
    # 상위/하위 예제 출력
    results.sort(key=lambda x: x["iou"], reverse=True)
    print("\n[Top 3 Recognitions]")
    for r in results[:3]:
        print(f" - [{r['entity']}] IoU: {r['iou']:.3f} | Pred: {[round(x,3) for x in r['pred_bbox']]} | Raw: {r['raw_output'].replace('<grounding>', '')[:60]}...")
        
    print("\n[Bottom 3 Recognitions]")
    for r in results[-3:]:
        print(f" - [{r['entity']}] IoU: {r['iou']:.3f} | Pred: {[round(x,3) for x in r['pred_bbox']]} | Raw: {r['raw_output'].replace('<grounding>', '')[:60]}...")

if __name__ == "__main__":
    run_visual_grounding_test()

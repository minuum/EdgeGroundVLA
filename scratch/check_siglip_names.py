# -*- coding: utf-8 -*-
# SigLIP 비전 인코더의 실제 레이어 이름 구조를 파악하기 위한 검증 스크립트

import sys
import os

# RoboVLMs의 라이브러리 경로를 추가합니다.
sys.path.insert(0, os.path.abspath("third_party/RoboVLMs"))

try:
    from transformers import AutoModelForVision2Seq
    print("Transformers 라이브러리 로드 성공")
    
    # PaliGemma 모델을 헤드 없이 구조만 로드해봅니다 (GPU 사용 안하고 빠르게 CPU로 로드)
    # 로컬 경로 .vlms/paligemma-3b-pt-224 가 있을 수 있으므로 확인 후 시도
    model_id = "google/paligemma-3b-pt-224"
    local_path = ".vlms/paligemma-3b-pt-224"
    if os.path.exists(local_path):
        model_id = local_path
        print(f"로컬 경로 {local_path} 에서 모델 로드 시도")

    model = AutoModelForVision2Seq.from_pretrained(
        model_id, 
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    print("\n[비전 타워 모듈 분석]")
    vision_tower = model.vision_tower
    print(f"Vision Tower Type: {type(vision_tower)}")
    
    # 0번 레이어의 하위 linear 레이어들의 이름과 타입을 출력합니다.
    first_layer = vision_tower.vision_model.encoder.layers[0]
    for name, module in first_layer.named_modules():
        if "Linear" in str(type(module)):
            print(f"Module Name: {name} | Type: {type(module)}")
            
except Exception as e:
    print(f"오류 발생: {e}")

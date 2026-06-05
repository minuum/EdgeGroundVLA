# -*- coding: utf-8 -*-
import sys
import torch
from pathlib import Path
from PIL import Image
import numpy as np

warnings_filter = True
ROOT = Path("/home/minum/26CS/MoNaVLA")
sys.path.insert(0, str(ROOT))

from transformers import AutoModelForVision2Seq, AutoProcessor

VLM_PATH = ROOT / ".vlms/kosmos-2-patch14-224"

def test_forward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 모델 및 프로세서 로드
    proc = AutoProcessor.from_pretrained(str(VLM_PATH))
    model = AutoModelForVision2Seq.from_pretrained(
        str(VLM_PATH), torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    ).to(device)
    
    # 임의의 이미지와 텍스트 구성
    img = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
    prompt = "Navigate to the gray basket. Robot action:"
    
    # 프로세서 인코딩
    inp = proc(text=prompt, images=img, return_tensors="pt").to(device)
    if device.type == "cuda":
        inp["pixel_values"] = inp["pixel_values"].to(torch.float16)
        
    print("inp keys:", inp.keys())
    for k, v in inp.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k} shape: {v.shape}, dtype: {v.dtype}")
            
    # 개별 인자를 풀어서 넣었을 때 에러 재현 여부 확인
    try:
        print("\n--- 1. 명시적 인자 포워딩 시도 ---")
        out1 = model(
            input_ids=inp["input_ids"],
            attention_mask=inp["attention_mask"],
            pixel_values=inp["pixel_values"]
        )
        print("1. 성공! logits shape:", out1.logits.shape)
    except Exception as e:
        print("1. 실패! 에러:", str(e))
        import traceback
        traceback.print_exc()

    # dict 언패킹 포워딩 시도
    try:
        print("\n--- 2. dict 언패킹 포워딩 (**inp) 시도 ---")
        out2 = model(**inp)
        print("2. 성공! logits shape:", out2.logits.shape)
    except Exception as e:
        print("2. 실패! 에러:", str(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_forward()

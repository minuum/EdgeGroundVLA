# -*- coding: utf-8 -*-
"""패치 검증: 모델 빌드 후 vision LoRA가 trainable인지 카운트 (학습 X)."""
import sys, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT/"scripts/train_active"))
import _vlora_patch  # 패치 적용
sys.path.insert(0, str(ROOT/"third_party/RoboVLMs"))
from robovlms.train.base_trainer import BaseTrainer
from robovlms.utils.config_utils import load_config

cfg = sys.argv[1]
variant = load_config(cfg)
model = BaseTrainer(variant)
vis_lora_tr = [n for n,p in model.named_parameters() if "lora_" in n and "vision_tower" in n and p.requires_grad]
vis_lora_all = [n for n,p in model.named_parameters() if "lora_" in n and "vision_tower" in n]
print(f"\n[VERIFY] vision LoRA 총 {len(vis_lora_all)}개 중 trainable {len(vis_lora_tr)}개")
print("→ 패치 성공" if len(vis_lora_tr) > 0 else "→ 패치 실패(여전히 frozen)")

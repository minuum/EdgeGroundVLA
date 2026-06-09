# -*- coding: utf-8 -*-
"""train_vision=False가 vision LoRA까지 얼리는 버그(base_backbone.py:604) 우회 패치.
third_party 비수정 — _trainable_params_setup 후 vision LoRA만 requires_grad 복구."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "third_party/RoboVLMs"))
import robovlms.model.backbone.base_backbone as _bb

_orig = _bb.BaseRoboVLM._trainable_params_setup
def _patched(self):
    _orig(self)
    n = 0
    for name, p in self.named_parameters():
        if "lora_" in name and "vision_tower" in name and not p.requires_grad:
            p.requires_grad_(True); n += 1
    print(f"[VLORA-PATCH] vision LoRA requires_grad 복구: {n} params", flush=True)
_bb.BaseRoboVLM._trainable_params_setup = _patched
print("[VLORA-PATCH] _trainable_params_setup 패치 적용됨", flush=True)

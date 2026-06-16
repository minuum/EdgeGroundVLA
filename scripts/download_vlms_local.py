#!/usr/bin/env python3
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM, AutoTokenizer
except ImportError as exc:
    print("transformers or torch is required to run this script")
    sys.path.append(str(ROOT / ".venv/lib/python3.10/site-packages"))
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM, AutoTokenizer

VLMS_DIR = ROOT / ".vlms"
VLMS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "paligemma-3b-mix-224": "google/paligemma-3b-mix-224",
    "moondream2": "vikhyatk/moondream2"
}

def main():
    print("Starting VLM downloads...")
    for folder, model_id in MODELS.items():
        target_path = VLMS_DIR / folder
        if target_path.exists() and any(target_path.iterdir()):
            print(f"[EXISTS] {folder} is already downloaded at {target_path}")
            continue
            
        print(f"\n[DOWNLOAD] Downloading {model_id} into {target_path}...")
        try:
            if "paligemma" in folder:
                from transformers import PaliGemmaForConditionalGeneration
                processor = AutoProcessor.from_pretrained(model_id)
                model = PaliGemmaForConditionalGeneration.from_pretrained(
                    model_id, torch_dtype=torch.float32
                )
                processor.save_pretrained(str(target_path))
                model.save_pretrained(str(target_path))
            elif "moondream" in folder:
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id, trust_remote_code=True, revision="2025-01-09", torch_dtype=torch.float32
                )
                tokenizer.save_pretrained(str(target_path))
                model.save_pretrained(str(target_path), safe_serialization=False)
            print(f"[OK] Successfully saved {folder} to {target_path}")
        except Exception as e:
            print(f"[ERROR] Failed to download {model_id}: {e}")

if __name__ == "__main__":
    main()

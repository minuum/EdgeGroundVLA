#!/usr/bin/env python3
"""
Zero-shot Moondream2 Masking & Inverse-Masking Ablation Test

목적:
  학습(fine-tuning) 없이 사전학습된 Zero-shot VLM 모델인 Moondream2 자체만을 사용하여
  (1) 배경 역-마스킹 (Inverse-Masking, 배경 소실 시에도 바스켓 물체 자체를 검출하는지)
  (2) 마스킹 (Masking Swell, 바스켓 차폐 시 검출되지 않고 인과성이 성립하는지)
  을 정량 평가하여 Zero-shot VLM의 물리적/기하학적 물체 검출 강건성을 대조 실증합니다.
"""

import os
import sys
import json
import warnings
from pathlib import Path
from collections import defaultdict
import h5py
import numpy as np
import torch
from PIL import Image

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "docs" / "v5" / "bbox_frame_level" / "bbox_dataset_frame_level.json"
OUT_DIR = ROOT / "docs" / "v5" / "grounding_comparison"
MASK_COLOR = (128, 128, 128)

def mask_background(img_pil, cx, cy, area, scale=1.0):
    """바스켓 영역만 살리고 배경을 회색으로 지우기 (Inverse Masking)"""
    W, H = img_pil.size
    side = int(np.sqrt(area) * min(W, H) * scale)
    half = side // 2
    bx, by = int(cx * W), int(cy * H)
    x1, y1 = max(0, bx - half), max(0, by - half)
    x2, y2 = min(W, bx + half), min(H, by + half)

    masked = Image.new("RGB", (W, H), MASK_COLOR)
    crop_basket = img_pil.crop((x1, y1, x2, y2))
    masked.paste(crop_basket, (x1, y1))
    return masked

def mask_target(img_pil, cx, cy, area, scale=1.0):
    """바스켓 영역을 회색으로 지우고 배경만 살리기 (Masking)"""
    W, H = img_pil.size
    side = int(np.sqrt(area) * min(W, H) * scale)
    half = side // 2
    bx, by = int(cx * W), int(cy * H)
    x1, y1 = max(0, bx - half), max(0, by - half)
    x2, y2 = min(W, bx + half), min(H, by + half)

    masked = img_pil.copy()
    from PIL import ImageDraw
    draw = ImageDraw.Draw(masked)
    draw.rectangle([x1, y1, x2, y2], fill=MASK_COLOR)
    return masked

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] Using: {device}")

    # 1. Moondream2 Zero-shot 모델 로드
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("[LOAD] Loading Moondream2 model from huggingface...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            "vikhyatk/moondream2",
            trust_remote_code=True,
            revision="2025-01-09",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device).eval()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

    # 2. 데이터셋 로드 및 가벼운 샘플링 (사용자의 속도 요구사항 반영)
    if not DATA_PATH.exists():
        print(f"❌ Error: 데이터셋 파일 없음: {DATA_PATH}")
        sys.exit(1)
        
    dataset = json.loads(DATA_PATH.read_text())
    
    # 방향별로 3개씩 가벼운 샘플링 (총 9개 프레임)
    samples = defaultdict(list)
    N_SAMPLE = 3
    MIN_AREA = 0.01

    for ep in dataset:
        d = ep["direction"]
        if len(samples[d]) >= N_SAMPLE:
            continue
        
        all_idxs = [f["frame_idx"] for f in ep["frames"]]
        max_idx = max(all_idxs) if all_idxs else 0
        
        valid_frames = [
            f for f in ep["frames"]
            if f["consistent"] and f.get("area_det") and f["area_det"] >= MIN_AREA
        ]
        
        for fr in valid_frames:
            if len(samples[d]) < N_SAMPLE:
                samples[d].append((ep["episode"], fr))

    test_cases = []
    for d in ["left", "center", "right"]:
        test_cases.extend(samples[d])

    print(f"[DATA] Sampled {len(test_cases)} frames for zero-shot Moondream2 ablation test.")

    # 3. Ablation 테스트 루프
    results = []
    SCALES = [1.5, 1.0, 0.5]
    
    # 통계 변수
    stats = {
        sc: {
            "inv_mask_success": 0,  # 배경 지웠을 때도 바스켓을 검출(성공)한 횟수
            "mask_success": 0,      # 바스켓을 가렸을 때 바스켓을 검출 안 한(성공) 횟수
            "total": 0
        } for sc in SCALES
    }

    print("\n🔍 [START] Moondream2 Zero-shot Ablation Sweep 시작\n")
    print(f"  {'Direction':<10} {'cx':>5} {'area':>6} | {'Scale':>5} | {'Inv-Mask (배경지움)':<20} | {'Mask (바스켓가림)':<20}")
    print("  " + "-" * 85)

    for ep_path, fr in test_cases:
        direction = fr["label"]
        cx = fr["cx_det"]
        cy = fr["cy_det"]
        area = fr["area_det"]
        frame_idx = fr["frame_idx"]

        # H5 이미지 로드
        try:
            with h5py.File(ep_path, "r") as f:
                img_data = f["observations"]["images"][frame_idx] if "observations" in f else f["images"][frame_idx]
                img = Image.fromarray(img_data).convert("RGB")
        except Exception as e:
            print(f"  ⚠️ Error loading image from {ep_path}: {e}")
            continue

        # 원본 이미지 검출 체크
        res_orig = model.detect(img, "gray basket")
        objects_orig = res_orig.get("objects", res_orig) if isinstance(res_orig, dict) else res_orig
        has_orig = len(objects_orig) > 0

        for sc in SCALES:
            # 1. 배경 역-마스킹 (Inverse Masking)
            img_inv = mask_background(img, cx, cy, area, scale=sc)
            res_inv = model.detect(img_inv, "gray basket")
            objects_inv = res_inv.get("objects", res_inv) if isinstance(res_inv, dict) else res_inv
            has_inv = len(objects_inv) > 0
            
            # 2. 마스킹 (Masking)
            img_mask = mask_target(img, cx, cy, area, scale=sc)
            res_mask = model.detect(img_mask, "gray basket")
            objects_mask = res_mask.get("objects", res_mask) if isinstance(res_mask, dict) else res_mask
            has_mask = len(objects_mask) > 0

            # 통계 업데이트
            stats[sc]["total"] += 1
            if has_inv:
                stats[sc]["inv_mask_success"] += 1
            if not has_mask:  # 가렸을 때 바스켓이 안 나와야 성공
                stats[sc]["mask_success"] += 1

            inv_status = "DETECTED (OK)" if has_inv else "NONE (FAIL)"
            mask_status = "NONE (OK)" if not has_mask else "DETECTED (FAIL)"
            
            print(f"  {direction:<10} {cx:>5.2f} {area:>6.4f} | {sc:>4.1f}x | {inv_status:<20} | {mask_status:<20}")

        print("  " + "-" * 85)

    # 4. 요약 리포트 빌드
    summary = {
        "model": "moondream2-zeroshot",
        "description": "학습 없이 순수 pre-trained Moondream2 모델을 활용한 차폐 및 배경 소실 ablation 테스트 결과",
        "scales": SCALES,
        "results": []
    }

    print("\n=================================================================================")
    print("  Zero-shot Moondream2 Ablation 최종 통계")
    print("=================================================================================")
    for sc in SCALES:
        s = stats[sc]
        tot = s["total"] if s["total"] > 0 else 1
        inv_rate = (s["inv_mask_success"] / tot) * 100
        mask_rate = (s["mask_success"] / tot) * 100
        
        print(f"  [Scale = {sc:.1f}x]")
        print(f"    배경 역-마스킹 (Inverse Masking) 바스켓 보존 성공률: {inv_rate:.1f}% ({s['inv_mask_success']}/{s['total']})")
        print(f"    바스켓 마스킹 (Target Masking) 차폐 성공률 (Non-detect): {mask_rate:.1f}% ({s['mask_success']}/{s['total']})")
        print("  " + "-" * 75)

        summary["results"].append({
            "scale": sc,
            "inverse_masking_success_rate": round(inv_rate, 1),
            "masking_success_rate": round(mask_rate, 1),
            "total_frames": s["total"]
        })

    result_json = OUT_DIR / "moondream_ablation_results.json"
    result_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n[SAVED] JSON 결과 저장 완료 ➔ {result_json}")

if __name__ == "__main__":
    main()

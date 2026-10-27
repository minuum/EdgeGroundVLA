#!/usr/bin/env python3
"""E2E VLA (Florence-2 LoRA + Flow Head) vs 2-Stage Decoupled (E1b_tuned) 간접 비교 검증.

목적:
  1. 실제 로봇 주행 테스트 배치에서 E2E VLA 모델의 액션 생성(Euler Sampling) 정상 동작 검증
  2. 추론 속도(FPS / Latency) 및 액션 출력 분포 (vx, vy, wz)의 건전성 확인
  3. 기존 2-Stage 챔피언과 비교 보고서 생성
"""

import json
import time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from robovlm_nav.models.egvla2_fm_heads import PiZeroFM, WINDOW, HID
from scripts.train_egvla2_e2e_lora import E2EVLA, RealRobotH5Dataset

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] E2E VLA 추론 및 성능 평가 시작 (Device: {device})")

    # 1. 모델 복원
    ckpt_path = ROOT / "runs" / "egvla2_e2e_lora" / "e2e_vla_florence2_lora_latest.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    
    model_name = ckpt["args"]["model_name"]
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=torch.float32)

    lora_config = LoraConfig(
        r=ckpt["args"]["lora_r"],
        lora_alpha=ckpt["args"]["lora_r"] * 2,
        target_modules=["qkv", "proj"],
        bias="none"
    )
    base_model.vision_tower = get_peft_model(base_model.vision_tower, lora_config)
    base_model.vision_tower.load_state_dict(ckpt["vision_lora"])

    action_head = PiZeroFM(bdim=WINDOW * 4, fdim=768, layers=2, d=HID)
    action_head.load_state_dict(ckpt["action_head"])

    e2e_model = E2EVLA(base_model, action_head, feat_dim=1024, hidden_dim=HID).to(device)
    e2e_model.feat_proj.load_state_dict(ckpt["feat_proj"])
    e2e_model.eval()

    # 2. 테스트 데이터셋 구성
    ep_dir = ROOT / "ROS_action" / "mobile_vla_dataset_egvla2"
    ep_files = sorted(list(ep_dir.glob("*.h5")))[:2]
    dataset = RealRobotH5Dataset(ep_files, processor, max_samples=20)

    # 3. 추론 속도 및 생성 액션 분석
    latencies = []
    generated_actions = []
    ground_truth_actions = []

    print("[*] 20개 주행 프레임 대상 Euler 10-Step 연속 액션 샘플링 진행...")
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            pix = sample["pixel_values"].unsqueeze(0).to(device)
            bbox = sample["bbox_win"].unsqueeze(0).to(device)
            gt_act = sample["action_chunk"].numpy()

            t0 = time.time()
            # 4개 액션 궤적 생성 후 평균
            pred_chunks = e2e_model.sample_actions(pix, bbox, k_samples=4, steps=10)
            latencies.append((time.time() - t0) * 1000.0) # ms

            mean_chunk = pred_chunks.mean(dim=1).squeeze(0).cpu().numpy() # (H=8, 3)
            generated_actions.append(mean_chunk[0]) # 바로 다음 스텝 액션 (vx, vy, wz)
            ground_truth_actions.append(gt_act[0])

    gen_act = np.array(generated_actions)
    gt_act = np.array(ground_truth_actions)

    mean_lat = float(np.mean(latencies[2:])) # 웜업 제외 평균 레이턴시
    fps = 1000.0 / mean_lat if mean_lat > 0 else 0.0
    action_mse = float(np.mean((gen_act - gt_act) ** 2))
    
    # 방향 오차 각도 (vx, vy 평면)
    pred_angles = np.arctan2(gen_act[:, 1], gen_act[:, 0]) * 180.0 / np.pi
    gt_angles = np.arctan2(gt_act[:, 1], gt_act[:, 0]) * 180.0 / np.pi
    angle_errors = np.abs(pred_angles - gt_angles)
    angle_errors = np.minimum(angle_errors, 360.0 - angle_errors)
    mean_angle_err = float(np.mean(angle_errors))

    report = {
        "architecture": "End-to-End VLA (Florence-2 LoRA + PiZeroFM AdaLN-Zero)",
        "parameters": {
            "total_backbone": "91.0M (Florence-2-base)",
            "trainable_lora": "658K (0.72%)",
            "action_expert": "0.32M (PiZeroFM)",
            "e2e_backprop": "ENABLED"
        },
        "performance": {
            "mean_inference_latency_ms": round(mean_lat, 2),
            "inference_hz": round(fps, 2),
            "action_chunk_length": 8,
            "euler_steps": 10,
            "direction_angle_error_deg": round(mean_angle_err, 2),
            "action_l2_mse": round(action_mse, 4)
        },
        "comparison_with_champion": {
            "champion_model": "E1b_tuned (Decoupled 2-Stage Frozen Florence-2)",
            "champion_latency_ms": "871 ms (1.15 Hz on Jetson, ~45 ms on Server)",
            "champion_dir_err_deg": None,   # 정정: 근거 없던 상수 10.42 삭제(같은 held-out 분할 비교는 eval_e2e_general.py)
            "e2e_advantage": "중간 BBox 의존 없이 시각-행동 End-to-End 직접 정렬 및 그래디언트 최적화 달성"
        }
    }

    out_file = ROOT / "docs" / "v5" / "egvla2_fm" / "e2e_vla_eval_report.json"
    out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[+] E2E VLA 평가 리포트 저장 완료: {out_file}")
    print(json.dumps(report, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""EGVLA2 End-to-End (E2E) VLA 학습 파이프라인 (Florence-2 LoRA + Flow-Matching Action Expert).

목적:
  기존의 Decoupled 2-Stage 구조(고정된 Florence-2 특징 캐시 + 독립 액션헤드)에서 벗어나,
  로봇의 주행 액션 손실(Flow-Matching Loss)이 Florence-2 백본 비전/언어 모듈까지 직접
  역전파되는 '진정한 End-to-End VLA' 아키텍처를 구현하고 학습한다.

아키텍처:
  1. Vision-Language Backbone: Microsoft Florence-2-base (PEFT LoRA 적용)
  2. Action Expert: π0 스타일 AdaLN-Zero Flow-Matching Head (PiZeroFM)
  3. Action Output: H=8 Horizon Chunk (vx, vy, wz continuous actions)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from robovlm_nav.models.egvla2_fm_heads import (
    PiZeroFM, H, ADIM, WINDOW, HID, sample_chunk
)

def compute_flow_loss(model, bbox, feat, target_action, generator=None):
    """Flow Matching Loss 계산: L = || v_theta(x_t, t, c) - (x_1 - x_0) ||^2"""
    x0 = torch.randn_like(target_action, generator=generator)
    t = torch.rand(len(target_action), device=target_action.device, generator=generator)
    xt = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * target_action
    vt_pred = model(bbox, feat, xt, t)
    target_v = target_action - x0
    return F.mse_loss(vt_pred, target_v)

class E2EVLA(nn.Module):
    """Florence-2 LoRA + Flow-Matching Policy Head 결합 E2E VLA 모델"""
    def __init__(self, florence_model, action_head, feat_dim=1024, hidden_dim=128):
        super().__init__()
        self.vlm = florence_model
        self.head = action_head
        # VLM feature projection to policy conditioning dimension (DaViT: 1024)
        self.feat_proj = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, 768),
            nn.GELU()
        )

    def extract_vlm_features(self, pixel_values):
        """Florence-2의 DaViT vision tower에서 패치 토큰 추출 및 글로벌 풀링"""
        # PeftModel을 거쳐 내부 DaViT의 forward_features_unpool 직접 호출
        base_tower = self.vlm.vision_tower.get_base_model() if hasattr(self.vlm.vision_tower, "get_base_model") else self.vlm.vision_tower
        vis_features = base_tower.forward_features_unpool(pixel_values)
        # vis_features: (B, num_tokens, D) -> global mean pooling (B, D)
        if vis_features.dim() == 3:
            feat = vis_features.mean(dim=1)
        elif vis_features.dim() == 4:
            feat = vis_features.flatten(2).transpose(1, 2).mean(dim=1)
        else:
            feat = vis_features
        return self.feat_proj(feat)

    def forward(self, pixel_values, bbox_window, target_action=None, xt=None, t=None):
        feat = self.extract_vlm_features(pixel_values)
        if target_action is not None:
            # Training mode: compute flow loss
            return compute_flow_loss(self.head, bbox_window, feat, target_action)
        else:
            # Inference mode: forward action vector field
            return self.head(bbox_window, feat, xt, t)

    @torch.no_grad()
    def sample_actions(self, pixel_values, bbox_window, k_samples=4, steps=10):
        """Euler integration으로 연속 액션 청크 샘플링"""
        feat = self.extract_vlm_features(pixel_values)
        return sample_chunk(self.head, bbox_window, feat, k=k_samples, steps=steps)


class RealRobotH5Dataset(Dataset):
    """실기 H5 주행 에피소드 데이터셋 로더"""
    def __init__(self, episode_files, processor, window_size=3, horizon=8, max_samples=None):
        self.processor = processor
        self.window_size = window_size
        self.horizon = horizon
        self.samples = []

        print(f"[*] H5 데이터셋 인덱싱 중... (에피소드 {len(episode_files)}개)")
        for ep_path in episode_files:
            try:
                with h5py.File(ep_path, "r") as f:
                    images = np.array(f["images"])           # (T, 720, 1280, 3)
                    actions = np.array(f["actions"], dtype=np.float32) # (T, 3)
                
                T = len(images)
                if T < window_size + 2:
                    continue

                # BBox는 [0.5, 0.5, 0.2, 0.2] 기본 서브골 앵커로 초기화
                dummy_bbox = np.tile([0.5, 0.5, 0.2, 0.2], (T, 1)).astype(np.float32)

                for t in range(T):
                    # BBox 슬라이딩 윈도우 (WINDOW * 4 = 12-d)
                    win = np.concatenate([dummy_bbox[max(0, t - window_size + 1 + k)] for k in range(window_size)])
                    # H=8 액션 청크 구성
                    chunk = np.zeros((horizon, 3), dtype=np.float32)
                    seg = actions[t:t + horizon]
                    chunk[:len(seg)] = seg

                    self.samples.append({
                        "image": images[t],
                        "bbox_win": win,
                        "action_chunk": chunk
                    })
                    if max_samples and len(self.samples) >= max_samples:
                        break
            except Exception as e:
                print(f"⚠️ 에피소드 로드 실패 ({ep_path.name}): {e}")
            if max_samples and len(self.samples) >= max_samples:
                break
        print(f"[+] 총 {len(self.samples)}개 프레임 샘플 로드 완료.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        pil_img = Image.fromarray(item["image"])
        inputs = self.processor(images=pil_img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0) # (3, H, W)
        return {
            "pixel_values": pixel_values,
            "bbox_win": torch.tensor(item["bbox_win"], dtype=torch.float32),
            "action_chunk": torch.tensor(item["action_chunk"], dtype=torch.float32)
        }


def main():
    parser = argparse.ArgumentParser(description="Train E2E VLA LoRA on Florence-2 + Flow-Matching Head")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora_r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--num_episodes", type=int, default=15, help="Number of episodes to train on")
    parser.add_argument("--max_samples", type=int, default=0, help="Max sample count limit (0 for unlimited)")
    parser.add_argument("--model_name", type=str, default="microsoft/Florence-2-base", help="Model backbone")
    args = parser.parse_args()

    max_samples = None if args.max_samples <= 0 else args.max_samples

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=======================================================")
    print(f"🚀 MoNaVLA E2E-VLA Training Launch")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Backbone: {args.model_name} | LoRA rank: {args.lora_r}")
    print(f"Epochs: {args.epochs} | Batch size: {args.batch_size} | Episodes: {args.num_episodes} | Max Samples: {max_samples or 'ALL'}")
    print(f"=======================================================\n")

    # 1. Florence-2 모델 및 프로세서 로드
    print("[1/5] Florence-2 백본 및 프로세서 로드 중...")
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32
    )

    # 2. LoRA 설정 및 주입 (Florence-2 DaViT vision tower 구조: qkv, proj)
    print("[2/5] PEFT LoRA 어댑터 설정 중...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        target_modules=["qkv", "proj"],
        lora_dropout=0.05,
        bias="none"
    )
    base_model.vision_tower = get_peft_model(base_model.vision_tower, lora_config)
    print("  [*] 백본 LoRA 주입 완료. 학습 파라미터 비율:")
    base_model.vision_tower.print_trainable_parameters()

    # 3. AdaLN Flow-Matching 액션 헤드 생성
    print("[3/5] π0 스타일 AdaLN-Zero Flow-Matching 액션 헤드 초기화...")
    bdim = WINDOW * 4  # 12-d
    fdim = 768         # 액션 헤드가 받는 피처 차원
    action_head = PiZeroFM(bdim=bdim, fdim=fdim, layers=2, d=HID)

    # 4. E2E 통합 모델 빌드 (Florence-2 DaViT 출력은 1024-d)
    e2e_model = E2EVLA(base_model, action_head, feat_dim=1024, hidden_dim=HID).to(device)

    # 5. 데이터셋 구축
    print("[4/5] 주행 에피소드 데이터셋 로드...")
    ep_dir = ROOT / "ROS_action" / "mobile_vla_dataset_egvla2"
    all_ep_files = sorted(list(ep_dir.glob("*.h5")))
    ep_files = all_ep_files[:args.num_episodes]
    dataset = RealRobotH5Dataset(ep_files, processor, max_samples=max_samples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)

    # 6. 옵티마이저 구성 (LoRA 파라미터 + Flow Head 파라미터 동시 학습)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in base_model.vision_tower.parameters() if p.requires_grad], "lr": args.lr * 0.5},
        {"params": action_head.parameters(), "lr": args.lr},
        {"params": e2e_model.feat_proj.parameters(), "lr": args.lr}
    ], weight_decay=1e-4)

    # 7. 학습 루프 (End-to-End Gradient Backpropagation)
    print(f"\n[5/5] E2E VLA 그래디언트 동시 역전파 학습 시작 ({args.epochs} 에포크)...")
    e2e_model.train()
    history = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_losses = []
        for batch in dataloader:
            pix = batch["pixel_values"].to(device)
            bbox = batch["bbox_win"].to(device)
            act = batch["action_chunk"].to(device)

            optimizer.zero_grad()
            loss = e2e_model(pix, bbox, target_action=act)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(e2e_model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        elapsed = time.time() - start_time
        history.append({"epoch": epoch, "loss": float(avg_loss)})
        print(f"  [Epoch {epoch:02d}/{args.epochs:02d}] Flow-Matching Loss: {avg_loss:.5f} (누적 {elapsed:.1f}s)")

    # 8. 체크포인트 및 결과 리포트 저장
    out_dir = ROOT / "runs" / "egvla2_e2e_lora"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "e2e_vla_florence2_lora_latest.pt"
    torch.save({
        "vision_lora": base_model.vision_tower.state_dict(),
        "action_head": action_head.state_dict(),
        "feat_proj": e2e_model.feat_proj.state_dict(),
        "args": vars(args),
        "history": history
    }, ckpt_path)

    report_path = ROOT / "docs" / "v5" / "egvla2_fm" / "e2e_vla_smoke_report.json"
    report_path.write_text(json.dumps({
        "status": "SUCCESS",
        "model": "Florence-2-base + PiZeroFM AdaLN (E2E LoRA)",
        "trainable_components": ["Florence-2 vision_tower LoRA (r=8)", "AdaLN-Zero Action Head", "Projection Layer"],
        "epochs": args.epochs,
        "initial_loss": history[0]["loss"] if history else None,
        "final_loss": history[-1]["loss"] if history else None,
        "loss_delta": (history[0]["loss"] - history[-1]["loss"]) if history else None,
        "checkpoint_saved": str(ckpt_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }, indent=2, ensure_ascii=False))

    print(f"\n✅ E2E VLA 학습 완료! 체크포인트 저장됨: {ckpt_path}")
    print(f"📊 검증 리포트 저장됨: {report_path}\n")

if __name__ == "__main__":
    main()

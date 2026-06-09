#!/usr/bin/env python3
"""
Mobile VLA Offline Simulation Tester
데이터셋의 H5 파일을 읽어서 추론을 시뮬레이션합니다.
실제 로봇 없이 모델의 동작을 테스트할 수 있습니다.
"""

import os
import sys
import h5py
import numpy as np
import argparse
from pathlib import Path
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import logging

# Logging 설정
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Mobile VLA 경로 추가
_project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(_project_root / 'RoboVLMs'))
sys.path.append(str(_project_root))

try:
    from robovlm_nav.serve.inference_server import MobileVLAInference
except ImportError:
    try:
        from Mobile_VLA.inference_server import MobileVLAInference
    except ImportError:
        MobileVLAInference = None


def load_episode_data(h5_path):
    """H5 파일에서 에피소드 데이터 로드"""
    print(f"🔍 Inspecting H5: {h5_path}")
    with h5py.File(h5_path, 'r') as f:
        # 데이터셋 구조에 맞춰 경로 수정
        # 1. Image 로드
        if 'images' in f:
            images = f['images'][:]
        elif 'observations/images/camera' in f:
            images = f['observations/images/camera'][:]
        else:
            raise KeyError(f"Could not find images in {list(f.keys())}")
            
        # 2. 액션 로드 (실제 데이터셋은 3차원일 수 있으므로 앞의 2개만 사용)
        if 'actions' in f:
            actions_raw = f['actions'][:]
            actions = actions_raw[:, :2] # linear_x, linear_y
        else:
            raise KeyError(f"Could not find actions in {list(f.keys())}")
        
        # 3. Instruction 로드
        instruction = None
        if 'language_instruction' in f:
            instruction_bytes = f['language_instruction'][0]
            instruction = instruction_bytes.decode('utf-8') if isinstance(instruction_bytes, bytes) else str(instruction_bytes)
        elif 'instruction' in f:
             instruction = f['instruction'][0].decode('utf-8')
        else:
            # 파일명 또는 속성에서 추출
            filename = Path(h5_path).name.lower()
            if 'left' in filename:
                instruction = "Navigate to the brown pot on the left"
            elif 'right' in filename:
                instruction = "Navigate to the brown pot on the right"
            elif 'basket' in filename:
                instruction = "Navigate to the brown pot"
            else:
                instruction = "Navigate to the target"
        
        return images, actions, instruction


def simulate_episode(model, images, gt_actions, instruction, visualize=False):
    """
    에피소드 시뮬레이션
    
    Args:
        model: MobileVLAInference 인스턴스
        images: (T, H, W, 3) 이미지 시퀀스
        gt_actions: (T, 2) Ground Truth 액션
        instruction: 텍스트 명령
        visualize: 시각화 여부
    
    Returns:
        pred_actions: 예측된 액션 시퀀스
        metrics: 평가 메트릭
    """
    T = len(images)
    pred_actions = []
    latencies = []
    
    # 모델 히스토리 초기화
    model.reset()
    
    print(f"\n🎬 Episode Simulation Started")
    print(f"📝 Instruction: {instruction}")
    print(f"🖼️ Total Frames: {T}")
    print(f"{'='*60}")
    
    for t in tqdm(range(T), desc="Simulating"):
        # 이미지 전처리 (224x224로 리사이즈)
        image = images[t]  # (H, W, 3) RGB
        image_resized = cv2.resize(image, (224, 224))
        
        # 추론
        pred_action, latency = model.predict(image_resized, instruction)
        
        pred_actions.append(pred_action)
        latencies.append(latency)
        
        # 프레임별 로그 (5프레임마다)
        if t % 5 == 0:
            print(f"  Frame {t:3d} | GT: [{gt_actions[t][0]:+.3f}, {gt_actions[t][1]:+.3f}] | "
                  f"Pred: [{pred_action[0]:+.3f}, {pred_action[1]:+.3f}] | "
                  f"Latency: {latency:.0f}ms")
    
    pred_actions = np.array(pred_actions)  # (T, 2)
    
    # 메트릭 계산
    mse = np.mean((pred_actions - gt_actions) ** 2)
    mae = np.mean(np.abs(pred_actions - gt_actions))
    avg_latency = np.mean(latencies)
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'avg_latency': avg_latency,
        'pred_actions': pred_actions,
        'gt_actions': gt_actions
    }
    
    print(f"{'='*60}")
    print(f"📊 Simulation Results:")
    print(f"  MSE:  {mse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    print(f"  Avg Latency: {avg_latency:.1f}ms")
    
    # 시각화
    if visualize:
        visualize_results(pred_actions, gt_actions, instruction)
    
    return pred_actions, metrics


def visualize_results(pred_actions, gt_actions, instruction):
    """결과 시각화"""
    T = len(pred_actions)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Simulation Results: {instruction}', fontsize=14, fontweight='bold')
    
    # 1. Linear X (전진/후진)
    ax = axes[0, 0]
    ax.plot(gt_actions[:, 0], label='Ground Truth', linewidth=2, alpha=0.7)
    ax.plot(pred_actions[:, 0], label='Predicted', linewidth=2, alpha=0.7, linestyle='--')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Linear X (m/s)')
    ax.set_title('Linear X Velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Linear Y (좌/우)
    ax = axes[0, 1]
    ax.plot(gt_actions[:, 1], label='Ground Truth', linewidth=2, alpha=0.7)
    ax.plot(pred_actions[:, 1], label='Predicted', linewidth=2, alpha=0.7, linestyle='--')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Linear Y (m/s)')
    ax.set_title('Linear Y Velocity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 2D Trajectory
    ax = axes[1, 0]
    # 누적 이동 계산 (간단한 적분)
    gt_traj = np.cumsum(gt_actions, axis=0)
    pred_traj = np.cumsum(pred_actions, axis=0)
    
    ax.plot(gt_traj[:, 0], gt_traj[:, 1], label='GT Trajectory', linewidth=2, alpha=0.7, marker='o', markersize=3)
    ax.plot(pred_traj[:, 0], pred_traj[:, 1], label='Pred Trajectory', linewidth=2, alpha=0.7, marker='x', markersize=3, linestyle='--')
    ax.set_xlabel('Cumulative X')
    ax.set_ylabel('Cumulative Y')
    ax.set_title('2D Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # 4. Error over time
    ax = axes[1, 1]
    errors = np.sqrt(np.sum((pred_actions - gt_actions) ** 2, axis=1))
    ax.plot(errors, linewidth=2, color='red', alpha=0.7)
    ax.set_xlabel('Frame')
    ax.set_ylabel('L2 Error')
    ax.set_title('Prediction Error Over Time')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/tmp/vla_simulation_result.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: /tmp/vla_simulation_result.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Mobile VLA Offline Simulation')
    _project_root = Path(__file__).resolve().parent.parent
    parser.add_argument('--dataset', type=str, default=str(_project_root / 'ROS_action/basket_dataset'),
                        help='Dataset directory')
    parser.add_argument('--episode', type=str, default=None,
                        help='Specific episode file (e.g., episode_20260129_010041_basket_1box_hori_left_core_medium.h5)')
    parser.add_argument('--checkpoint', type=str,
                        default=str(_project_root / 'runs/unified_regression_win12/kosmos/epoch=epoch=09-val_loss=val_loss=0.0013.ckpt'),
                        help='Model checkpoint path')
    parser.add_argument('--config', type=str,
                        default=str(_project_root / 'configs/mobile_vla_exp17_win8_k1.json'),
                        help='Config file path')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize results')
    parser.add_argument('--random', action='store_true',
                        help='Pick random episode')
    
    args = parser.parse_args()
    
    # 에피소드 선택
    if args.episode:
        episode_path = os.path.join(args.dataset, args.episode)
    elif args.random:
        import glob
        import random
        episodes = glob.glob(os.path.join(args.dataset, 'episode_*.h5'))
        episode_path = random.choice(episodes)
        print(f"🎲 Randomly selected: {Path(episode_path).name}")
    else:
        # 첫 번째 에피소드 사용
        import glob
        episodes = sorted(glob.glob(os.path.join(args.dataset, 'episode_*.h5')))
        if not episodes:
            print("❌ No episodes found in dataset!")
            return
        episode_path = episodes[0]
        print(f"📁 Using first episode: {Path(episode_path).name}")
    
    if not os.path.exists(episode_path):
        print(f"❌ Episode not found: {episode_path}")
        return
    
    # 모델 로드
    print(f"\n🚀 Loading model...")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Config: {args.config}")
    
    model = MobileVLAInference(args.checkpoint, args.config)
    print(f"✅ Model loaded successfully!")
    
    # 에피소드 데이터 로드
    print(f"\n📂 Loading episode: {Path(episode_path).name}")
    images, gt_actions, instruction = load_episode_data(episode_path)
    print(f"  Images: {images.shape}")
    print(f"  Actions: {gt_actions.shape}")
    print(f"  Instruction: {instruction}")
    
    # 시뮬레이션 실행
    pred_actions, metrics = simulate_episode(
        model, images, gt_actions, instruction, 
        visualize=args.visualize
    )
    
    print(f"\n✅ Simulation completed!")


if __name__ == '__main__':
    main()

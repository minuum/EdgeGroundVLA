# GitHub Citation: https://github.com/Robot-VLAs/RoboVLMs/blob/main/robovlms/data/calvin_dataset.py
# Mobile VLA용 HDF5 데이터셋 로더 (RoboVLMs CALVIN 데이터셋 구조 참고)

import glob
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from robovlms.utils.model_utils import build_tokenizer


class MobileVLAH5Dataset(Dataset):
    """
    Mobile VLA HDF5 데이터셋 로더
    
    Args:
        data_dir: HDF5 파일들이 있는 디렉토리
        episode_pattern: 에피소드 파일 패턴 (예: "episode_20251106_*.h5")
        window_size: 히스토리 윈도우 크기
        action_chunk_size: 액션 청크 크기 (fwd_pred_next_n)
        model_name: 모델 이름 (토크나이저용)
        image_size: 이미지 크기
        rgb_pad: RGB 증강 패딩
        train_split: 학습/검증 분할 비율
        is_validation: 검증 데이터셋 여부
    """
    
    def __init__(
        self,
        data_dir,
        episode_pattern="episode_*.h5",
        window_size=8,
        action_chunk_size=10,
        model_name="kosmos",
        image_size=224,
        rgb_pad=10,
        train_split=0.8,
        is_validation=False,
        shift_first=False,
        abs_action=False,  # 액션 절대값 사용 (방향 제거)
        augment=False,     # 데이터 증강 (Mirroring 등)
        discrete_action=False, # 분류 방식 사용 여부 (6개 클래스)
        **kwargs
    ):
        self.data_dir = data_dir
        self.episode_pattern = episode_pattern
        self.window_size = window_size
        self.action_chunk_size = action_chunk_size
        self.model_name = model_name
        self.image_size = image_size
        self.rgb_pad = rgb_pad
        self.train_split = train_split
        self.is_validation = is_validation
        self.shift_first = shift_first
        self.abs_action = abs_action  # 방향 제거 옵션
        self.augment = augment and (not is_validation)  # 검증셋에는 증강 미적용
        self.is_training = not is_validation
        
        # 에피소드 파일 로드
        episode_files = sorted(glob.glob(f"{data_dir}/{episode_pattern}"))
        if len(episode_files) == 0:
            raise ValueError(f"No episodes found in {data_dir} with pattern {episode_pattern}")
        
        # Train/Val 분할
        split_idx = int(len(episode_files) * train_split)
        if is_validation:
            self.episode_files = episode_files[split_idx:]
        else:
            self.episode_files = episode_files[:split_idx]
        
        # 각 에피소드의 프레임 수 계산
        self.episode_lengths = []
        self.cumulative_lengths = [0]
        for ep_file in self.episode_files:
            with h5py.File(ep_file, 'r') as f:
                # 구형 및 신형 V5 이미지 데이터셋 경로 분기 처리
                if 'images' in f:
                    img_dataset = f['images']
                elif 'observations' in f and 'images' in f['observations']:
                    img_dataset = f['observations/images']
                else:
                    raise KeyError(f"No images dataset found in {Path(ep_file).name}")
                length = len(img_dataset)
                self.episode_lengths.append(length)
                self.cumulative_lengths.append(self.cumulative_lengths[-1] + length)
        
        self.total_frames = self.cumulative_lengths[-1]
        
        print(f"MobileVLAH5Dataset initialized:")
        print(f"  data_dir: {data_dir}")
        print(f"  episode_pattern: {episode_pattern}")
        print(f"  num_episodes: {len(self.episode_files)}")
        print(f"  total_frames: {self.total_frames}")
        print(f"  window_size: {window_size}")
        print(f"  action_chunk_size: {action_chunk_size}")
        print(f"  shift_first: {shift_first}")
        print(f"  model_name: {model_name}")
        print(f"  rgb_pad: {rgb_pad}")
        print(f"  train_split: {train_split}")
        print(f"  is_training: {not is_validation}")
        
        # 토크나이저 및 text_fn 초기화
        self.tokenizer = kwargs.get("tokenizer", None)
        self.tokenizer_config = kwargs.get("tokenizer_config", None)
        self.model_name = model_name
        self.window_size = window_size
        self.action_chunk_size = action_chunk_size  # 사용하지 않음 (호환성 유지)
        self.fwd_pred_next_n = kwargs.get("fwd_pred_next_n", action_chunk_size)  # kwargs에서 가져오기
        self.discrete_action = discrete_action
        print(f"DEBUG: MobileVLAH5Dataset discrete_action: {self.discrete_action}")
        
        # text_fn 초기화 (tokenizer_config가 있으면)
        if self.tokenizer_config is not None and self.tokenizer is not None:
            from robovlms.data.data_utils import get_text_function
            tokenizer_type = self.tokenizer_config.get("tokenizer_type", "kosmos")
            max_text_len = self.tokenizer_config.get("max_text_len", 256)
            self.text_fn = get_text_function(self.tokenizer, tokenizer_type, max_text_len)
        else:
            self.text_fn = None
    
    def __len__(self):
        # 각 에피소드에서 window_size + fwd_pred_next_n 만큼의 프레임이 필요
        # RoboVLMs 구조: 8 + 10 = 18 프레임
        total_frames_needed = self.window_size + self.fwd_pred_next_n
        valid_frames = 0
        for length in self.episode_lengths:
            if length >= total_frames_needed:
                valid_frames += length - total_frames_needed + 1
        return max(1, valid_frames)
    
    def _find_episode_and_frame(self, idx):
        """전체 인덱스를 에피소드와 프레임 인덱스로 변환"""
        for ep_idx, (start, end) in enumerate(zip(self.cumulative_lengths[:-1], self.cumulative_lengths[1:])):
            if idx < end - start:
                frame_idx = idx
                return ep_idx, frame_idx
            idx -= (end - start)
        return len(self.episode_files) - 1, 0
    
    def __getitem__(self, idx):
        # Random Temporal Sampling for better diversity
        # Instead of sequential indexing, randomly select episode and start frame
        
        total_frames_needed = self.window_size + self.fwd_pred_next_n
        
        # Random episode selection (with reproducibility for validation)
        if self.is_validation:
            # Validation: deterministic sampling for reproducibility
            ep_idx = idx % len(self.episode_files)
            np.random.seed(idx)  # Deterministic random for this sample
        else:
            # Training: truly random episode selection
            ep_idx = np.random.randint(0, len(self.episode_files))
        
        # HDF5 파일 로드
        with h5py.File(self.episode_files[ep_idx], 'r') as f:
            # 구형 및 신형 V5 이미지 데이터셋 경로 분기 처리
            if 'images' in f:
                img_dataset = f['images']
            elif 'observations' in f and 'images' in f['observations']:
                img_dataset = f['observations/images']
            else:
                raise KeyError(f"No images dataset found in {Path(self.episode_files[ep_idx]).name}")
            total_len = len(img_dataset)
            
            # Random start frame within valid range
            max_start = max(0, total_len - total_frames_needed)
            if self.is_validation:
                # Validation: deterministic start
                start_frame = (idx // len(self.episode_files)) % (max_start + 1) if max_start > 0 else 0
            else:
                # Training: random start
                start_frame = np.random.randint(0, max_start + 1) if max_start > 0 else 0
            
            # 이미지 로드 (random start부터 total_frames_needed 프레임 - 슬라이싱 리드로 속도 최적화)
            images = []
            end_frame = min(start_frame + total_frames_needed, total_len)
            
            # 한 번에 슬라이싱으로 로드 (디스크 IO 횟수를 획기적으로 감축)
            img_arrays = img_dataset[start_frame:end_frame]
            
            # Normalization 텐서 상수 최적화
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
            
            for img_array in img_arrays:
                img = Image.fromarray(img_array.astype(np.uint8))
                img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
                img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
                img_tensor = img_tensor.permute(2, 0, 1)
                img_tensor = (img_tensor - mean) / std
                images.append(img_tensor)
            
            # Padding if needed (edge case)
            while len(images) < total_frames_needed:
                images.append(torch.zeros_like(images[-1]) if images else torch.zeros(3, self.image_size, self.image_size))
            
            # 액션 로드 (한 번에 슬라이싱 로드)
            actions_raw = f['actions'][start_frame:end_frame]
            actions = [a[:2].copy() for a in actions_raw]
            
            # Padding for actions
            while len(actions) < total_frames_needed:
                actions.append(np.zeros(2))
            
            # 언어 명령 로드 (H5 파일에서 실제 읽기)
            if 'language_instruction' in f:
                language_bytes = f['language_instruction'][0]
                language_base = language_bytes.decode('utf-8') if isinstance(language_bytes, bytes) else str(language_bytes)
            else:
                # 파일명에서 방향 정보 추출 (Basket Navigation task 등)
                filename = Path(self.episode_files[ep_idx]).name.lower()
                if 'left' in filename:
                    if self.is_training:
                        variations = [
                            "Navigate to the brown pot on the left",
                            "Go to the left brown pot",
                            "Move towards the pot on the left side",
                            "Steer left to the brown pot",
                            "Perform left navigation to the object"
                        ]
                        language_base = np.random.choice(variations)
                    else:
                        language_base = "Navigate to the brown pot on the left"
                elif 'right' in filename:
                    if self.is_training:
                        variations = [
                            "Navigate to the brown pot on the right",
                            "Go to the right brown pot",
                            "Move towards the pot on the right side",
                            "Steer right to the brown pot",
                            "Perform right navigation to the object"
                        ]
                        language_base = np.random.choice(variations)
                    else:
                        language_base = "Navigate to the brown pot on the right"
                else:
                    language_base = "Navigate to the target location"

            language = language_base

        # -------------------------------------------------------------------------
        # Data Augmentation: Mirroring (Left <-> Right)
        # -------------------------------------------------------------------------
        if self.augment and np.random.rand() < 0.5:
            # 1. Flip Images
            images = [torch.flip(img, [2]) for img in images]
            
            # 2. Invert linear_y Action (Left <-> Right)
            actions = [np.array([a[0], -a[1]]) for a in actions]
            
            # 3. Swap Language 'left' <-> 'right' in the entire instruction
            # This handles both base language and our dynamic suffix (sliding left <-> sliding right)
            temp_lang = language.lower()
            temp_lang = temp_lang.replace('left', 'PLACEHOLDER').replace('right', 'left').replace('PLACEHOLDER', 'right')
            # Restore capitalization if needed
            language = temp_lang.capitalize()
        # -------------------------------------------------------------------------

        # 텐서 변환
        images_tensor = torch.stack(images)  # (total_frames_needed, C, H, W)
        
        if self.discrete_action:
            # Action Classification Mapping (Option C)
            # Map (x, y) to class index [0-5]
            # 0: Stop, 1: Forward, 2: Left, 3: Right, 4: Diag FL, 5: Diag FR
            cls_labels = []
            for a in actions:
                x, y = a[0], a[1]
                if abs(x) < 0.5 and abs(y) < 0.5: label = 0
                elif x > 0.5 and abs(y) < 0.1: label = 1
                elif abs(x) < 0.1 and y > 0.5: label = 2
                elif abs(x) < 0.1 and y < -0.5: label = 3
                elif x > 0.5 and y > 0.5: label = 4
                elif x > 0.5 and y < -0.5: label = 5
                else: label = 0 # Default
                cls_labels.append(label)
            actions_tensor = torch.tensor(cls_labels, dtype=torch.long)
        else:
            actions_tensor = torch.from_numpy(np.array(actions)).float()  # (total_frames_needed, 2)
            
            # 방향 제거 옵션: linear_y의 절대값 사용
            if self.abs_action:
                actions_tensor[:, 1] = torch.abs(actions_tensor[:, 1])  # linear_y만 절대값
            
            # 액션 정규화 [-1, 1] (abs_action일 때는 [0, 1]이 됨)
            actions_tensor = torch.clamp(actions_tensor, -1.0, 1.0)
        
        # 언어 토크나이징 (더미 - collate_fn에서 실제 처리)
        input_ids = torch.zeros(256, dtype=torch.long)
        attention_mask = torch.ones(256, dtype=torch.long)
        
        return {
            'rgb': images_tensor,
            'hand_rgb': torch.zeros_like(images_tensor),
            'actions': actions_tensor,
            'action_mask': torch.ones(total_frames_needed),
            'image_mask': torch.ones(total_frames_needed),
            'text': input_ids,
            'text_mask': attention_mask,
            'lang': language,
            'raw_text': language,
            'data_source': 'mobile_vla_action',
            'attention_mask': torch.ones(total_frames_needed),
        }
    
    def collater(self, data):
        """
        배치 데이터를 처리하는 collater 메서드
        DiskCalvinDataset의 collater를 참고하여 구현
        """
        # 액션 텐서 스택 (DiskCalvinDataset과 동일한 구조)
        # s["actions"]는 (window_size + fwd_pred_next_n, 2) 형태
        action_tensors = torch.from_numpy(
            np.array([s["actions"].numpy() for s in data])
        )[:, :-1]  # 마지막 프레임 제거 (DiskCalvinDataset과 동일)
        # Shape: (B, window_size + fwd_pred_next_n - 1, 2)
        
        # 액션 마스크 스택
        action_mask = torch.from_numpy(
            np.array([s["action_mask"].numpy() for s in data])
        )[:, :-1]  # 마지막 프레임 제거
        
        # 이미지 텐서 스택
        image_tensors = torch.stack([s["rgb"] for s in data], dim=0)  # (B, window_size + fwd_pred_next_n, C, H, W)
        
        # 이미지 마스크 스택
        image_mask = torch.from_numpy(
            np.array([s["image_mask"].numpy() for s in data])
        )
        
        # Gripper 이미지 스택 (더미)
        gripper_tensors = torch.stack([s["hand_rgb"] for s in data], dim=0)  # (B, window_size + fwd_pred_next_n, C, H, W)
        
        # 언어 토크나이징 (text_fn 사용)
        stacked_language = [s["lang"] for s in data]
        if self.text_fn is not None:
            text_tensors, attention_mask = self.text_fn(stacked_language)
        else:
            # text_fn이 없으면 더미 사용 (초기화 시점)
            text_tensors = torch.stack([s["text"] for s in data], dim=0)
            attention_mask = torch.stack([s["text_mask"] for s in data], dim=0)
        
        # DiskCalvinDataset과 동일한 방식으로 chunk 생성 (unfold 사용)
        image_chunk = image_tensors.unfold(1, self.fwd_pred_next_n, 1).permute(
            0, 1, 5, 2, 3, 4
        )[:, 1:]  # 첫 번째 제거
        image_tensors = image_tensors[:, : self.window_size]  # window_size만 사용
        
        gripper_chunk = gripper_tensors.unfold(1, self.fwd_pred_next_n, 1).permute(
            0, 1, 5, 2, 3, 4
        )[:, 1:]
        gripper_tensors = gripper_tensors[:, : self.window_size]
        
        fwd_mask = image_mask.unfold(1, self.fwd_pred_next_n, 1)[:, 1:]
        
        # 액션 chunk 생성 (unfold 사용)
        action_chunck = action_tensors.unfold(1, self.fwd_pred_next_n, 1)
        if not self.discrete_action:
            action_chunck = action_chunck.permute(0, 1, 3, 2)
        action_mask = action_mask.unfold(1, self.fwd_pred_next_n, 1)
        
        res = {
            "rgb": image_tensors,  # (B, window_size, C, H, W)
            "hand_rgb": gripper_tensors,  # (B, window_size, C, H, W)
            "action": action_tensors,  # (B, window_size + fwd_pred_next_n - 1, 2)
            "text": text_tensors,  # (B, seq_len)
            "text_mask": attention_mask,  # (B, seq_len)
            "fwd_rgb_chunck": image_chunk,  # (B, window_size + fwd_pred_next_n - 2, fwd_pred_next_n, C, H, W)
            "fwd_hand_rgb_chunck": gripper_chunk,  # (B, window_size + fwd_pred_next_n - 2, fwd_pred_next_n, C, H, W)
            "fwd_mask": fwd_mask,  # (B, window_size + fwd_pred_next_n - 2, fwd_pred_next_n)
            "action_chunck": action_chunck,  # (B, window_size + fwd_pred_next_n - 2, fwd_pred_next_n, 2)
            "chunck_mask": action_mask,  # (B, window_size + fwd_pred_next_n - 2, fwd_pred_next_n)
            "raw_text": stacked_language,  # List[str]
            "data_source": "mobile_vla_action",
        }
        return res


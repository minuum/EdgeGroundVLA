"""
FastAPI Inference Server for Mobile VLA
교수님 서버에서 모델 호출을 위한 API 서버

입력: 이미지 + Language instruction
출력: 2DOF actions [linear_x, linear_y]

보안: API Key 인증
"""

import sys
import os
import time
import io
import base64
import logging
import secrets
import json
from typing import List, Optional, Literal, Any
from pathlib import Path
import gc
import copy

# 1. Add project root and RoboVLMs to sys.path
# __file__ = .../MoNaVLA/robovlm_nav/serve/inference_server.py
# project_root = .../MoNaVLA (two levels up from serve/)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Handle scripts path for inference_logger
scripts_path = os.path.join(project_root, 'scripts')
if os.path.exists(scripts_path) and scripts_path not in sys.path:
    sys.path.append(scripts_path)

try:
    from inference_logger import get_logger
    logger_instance = get_logger()
except ImportError:
    logger_instance = None

# Handle RoboVLMs path (try both RoboVLMs and RoboVLMs_upstream and third_party/RoboVLMs)
for root_name in ['RoboVLMs', 'RoboVLMs_upstream', 'third_party/RoboVLMs']:
    p = os.path.join(project_root, root_name)
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

# Third-party imports
import torch
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.responses import FileResponse, Response
from fastapi.security import APIKeyHeader
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Project imports
try:
    from robovlm_nav.serve.action_buffer import ActionBuffer
    from robovlms.data.data_utils import unnoramalize_action
except ImportError as e:
    print(f"FATAL ERROR: Failed to import project modules. sys.path: {sys.path}")
    raise e

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mobile VLA Inference API", version="1.0.0")
STATIC_DIR = Path(current_dir) / "static"
DEBUG_UI_PATH = STATIC_DIR / "inference_debugger.html"
DEBUG_IMAGE_ROOTS = {
    "dataset_images": Path(project_root) / "ROS_action" / "mobile_vla_dataset_v5(Image)",
    "workspace": Path(project_root),
}
DEFAULT_DEBUG_MODEL_CANDIDATES = [
    (
        Path(project_root) / "runs" / "v5_nav" / "kosmos" / "mobile_vla_v5_exp31" / "2026-04-24" / "v5-exp31-step3-grounding-turnboost-learnedmix-5ep" / "last.ckpt",
        Path(project_root) / "configs" / "mobile_vla_v5_exp31_step3_grounding_turnboost_learnedmix_5ep.json",
    ),
    (
        Path(project_root) / "runs" / "v5_nav" / "kosmos" / "mobile_vla_v5_exp30" / "2026-04-24" / "v5-exp30-step3-grounding-turnboost-bboxcoarse-5ep" / "last.ckpt",
        Path(project_root) / "configs" / "mobile_vla_v5_exp30_step3_grounding_turnboost_coarseonly_5ep.json",
    ),
    (
        Path(project_root) / "runs" / "v5_nav" / "kosmos" / "mobile_vla_v5_exp29" / "2026-04-23" / "v5-exp29-step3-grounding-turnboost-coarseonly-5ep" / "last.ckpt",
        Path(project_root) / "configs" / "mobile_vla_v5_exp29_step3_grounding_turnboost_bboxcoarse_5ep.json",
    ),
    (
        Path(project_root) / "runs" / "v5_nav" / "kosmos" / "mobile_vla_v5_exp28" / "2026-04-23" / "v5-exp28-step3-objective-grounding-turnboost" / "epoch_epoch=epoch=13-val_loss=val_loss=8.708.ckpt",
        Path(project_root) / "configs" / "mobile_vla_v5_exp28_step3_balanced_objective_grounding_turnboost.json",
    ),
    (
        Path(project_root) / "runs" / "v5_nav" / "kosmos" / "mobile_vla_v5_exp27" / "2026-04-23" / "v5-exp27-step3-objective-letterbox224" / "epoch_epoch=epoch=08-val_loss=val_loss=7.932.ckpt",
        Path(project_root) / "configs" / "mobile_vla_v5_exp27_step3_objective_letterbox224.json",
    ),
    (
        Path(project_root) / "runs" / "v5_nav" / "kosmos" / "mobile_vla_v5_exp26" / "2026-04-22" / "v5-exp26-step3-objective-direct224" / "epoch_epoch=epoch=14-val_loss=val_loss=7.036.ckpt",
        Path(project_root) / "configs" / "mobile_vla_v5_exp26_step3_objective_direct224.json",
    ),
    (
        Path(project_root) / "runs" / "v5_nav" / "kosmos" / "mobile_vla_v5_exp25" / "2026-04-22" / "v5-exp25-step3-balanced-objective" / "epoch_epoch=epoch=02-val_loss=val_loss=10.117.ckpt",
        Path(project_root) / "configs" / "mobile_vla_v5_exp25_step3_balanced_objective.json",
    ),
]


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    client_host = request.client.host if request.client else "unknown"
    forwarded_for = request.headers.get("x-forwarded-for")
    real_ip = request.headers.get("x-real-ip")
    source = forwarded_for or real_ip or client_host
    response = await call_next(request)
    duration_ms = (time.time() - start_time) * 1000.0
    logger.info(
        "HTTP %s %s from=%s status=%s duration_ms=%.1f",
        request.method,
        request.url.path,
        source,
        response.status_code,
        duration_ms,
    )
    return response



# API Key 설정
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# 환경 변수에서 API Key 읽기 (없으면 생성)
def get_api_key():
    api_key = os.getenv("VLA_API_KEY")
    if not api_key:
        # API Key 자동 생성 및 출력
        api_key = secrets.token_urlsafe(32)
        logger.warning("="*60)
        logger.warning("⚠️  VLA_API_KEY 환경 변수가 없습니다!")
        logger.warning(f"생성된 API Key: {api_key}")
        logger.warning("다음 명령어로 저장하세요:")
        logger.warning(f'export VLA_API_KEY="{api_key}"')
        logger.warning("="*60)
    return api_key

VALID_API_KEY = get_api_key()

async def verify_api_key(api_key: str = Depends(api_key_header)):
    """API Key 검증"""
    if api_key != VALID_API_KEY:
        logger.warning(f"❌ 인증 실패: {api_key[:10]}...")
        raise HTTPException(
            status_code=403,
            detail="Invalid API Key"
        )
    return api_key

# Global model instance (lazy loading)
model_instance = None
model_override_checkpoint_path = None
model_override_config_path = None

# Exp47 MLP 전역 인스턴스 (lazy loading)
mlp_instance = None

# GoalNav MLP 전역 인스턴스 (lazy loading, exp49/exp54_s2v2/exp55)
goalnav_instance = None
_pure_vision_model = None
_pure_processor = None


class InferenceRequest(BaseModel):
    """추론 요청 스키마"""
    image: str  # base64 encoded image
    instruction: str  # Language instruction
    strategy: Literal["chunk_reuse", "receding_horizon"] = "chunk_reuse"  # Inference strategy
    vlm_model: Optional[str] = None  # "monapi" → π0 Flow Matching, None/기타 → GoalNav (기본)


class InferenceResponse(BaseModel):
    """추론 응답 스키마"""
    action: List[float]  # [linear_x, linear_y]
    latency_ms: float  # Inference latency in milliseconds
    model_name: str
    strategy: str  # Inference strategy used
    source: str  # "inferred" or "reused"
    buffer_status: dict  # Buffer status for chunk_reuse
    predicted_class: Optional[int] = None
    predicted_label: Optional[str] = None
    action_3d: Optional[List[float]] = None
    bbox: Optional[dict[str, Any]] = None
    grounding_caption: Optional[str] = None
    grounding_latency_ms: Optional[float] = None
    goal_near_proxy: Optional[bool] = None
    instruction_used: Optional[bool] = None
    matched_path_type: Optional[str] = None


class ConfigRequest(BaseModel):
    """설정 갱신 요청 스키마"""
    model: Optional[str] = None
    speed_scaling: Optional[bool] = None
    grounding_skip_n: Optional[int] = None
    smooth_enabled: Optional[bool] = None
    smooth_alpha_xy: Optional[float] = None
    smooth_alpha_az: Optional[float] = None
    stop_area_threshold: Optional[float] = None
    stop_cx_tolerance: Optional[float] = None


class InferenceDebugResponse(BaseModel):
    """중간 추론 과정을 포함한 디버그 응답"""
    action: List[float]
    latency_ms: float
    model_name: str
    source: str
    debug: dict[str, Any]


class InferenceDebugPathRequest(BaseModel):
    image_path: str
    instruction: str


class DebugModelReloadRequest(BaseModel):
    checkpoint_path: Optional[str] = None
    config_path: Optional[str] = None
    candidate_name: Optional[str] = None


# ── Exp47 Instruction-Conditioned MLP 스키마 ──────────────────────────────
class MLPInferenceRequest(BaseModel):
    """Exp47 MLP 추론 요청 스키마"""
    instruction: str                     # path_type 키 또는 자연어 instruction
    bbox_cx: float = 0.0                 # bbox 중심 X (정규화 0~1)
    bbox_cy: float = 0.0                 # bbox 중심 Y (정규화 0~1)
    bbox_area: float = 0.0               # bbox 면적 비율 (0~1)
    has_bbox: bool = False               # bbox 탐지 여부
    image: Optional[str] = None          # base64 이미지 (vision cache 갱신 시에만)
    force_vision_update: bool = False    # True면 cache TTL 무시하고 강제 갱신


class MLPInferenceResponse(BaseModel):
    """Exp47 MLP 추론 응답 스키마"""
    action: List[float]                  # [linear_x, linear_y]
    class_idx: int
    class_name: str
    latency_ms: float
    vision_cache_age_ms: float           # 마지막 vision feature 갱신 후 경과 ms
    instruction_matched: str             # 실제 매칭된 path_type


class VisionUpdateRequest(BaseModel):
    image: str                           # base64 이미지


class VisionUpdateResponse(BaseModel):
    status: str
    latency_ms: float
    feature_dim: int


def _get_allowed_debug_roots() -> dict[str, Path]:
    return {name: path.resolve() for name, path in DEBUG_IMAGE_ROOTS.items() if path.exists()}


def _resolve_debug_path(raw_path: str) -> Path:
    if not raw_path or not str(raw_path).strip():
        raise HTTPException(status_code=400, detail="image_path is required")

    candidate = Path(str(raw_path)).expanduser().resolve()
    for root in _get_allowed_debug_roots().values():
        try:
            candidate.relative_to(root)
            return candidate
        except ValueError:
            continue
    raise HTTPException(
        status_code=403,
        detail=f"Path is outside allowed debug roots: {candidate}",
    )


def _list_debug_directory(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")
    if not path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {path}")

    dirs = []
    files = []
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        item = {
            "name": child.name,
            "path": str(child),
            "is_dir": child.is_dir(),
        }
        if child.is_dir():
            dirs.append(item)
        elif child.suffix.lower() in image_exts:
            files.append(item)

    return {
        "path": str(path),
        "parent": str(path.parent) if path.parent != path else None,
        "dirs": dirs,
        "files": files,
    }


def classify_action(vx: float, vy: float, wz: float) -> tuple[int, str]:
    """액션 벡터를 8개 클래스 중 하나로 분류합니다."""
    if abs(vx) < 0.1 and abs(vy) < 0.1 and abs(wz) < 0.1:
        return 0, "STOP"
    if abs(wz) > 0.15 and abs(vx) < 0.2 and abs(vy) < 0.2:
        return (6, "ROT_L") if wz > 0 else (7, "ROT_R")
    if vx > 0.3 and vy > 0.3:  return 4, "FWD+L"
    if vx > 0.3 and vy < -0.3: return 5, "FWD+R"
    if vx > 0.3:                return 1, "FORWARD"
    if vy > 0.3:                return 2, "LEFT"
    if vy < -0.3:               return 3, "RIGHT"
    return 1, "FORWARD"


def snap_monapi_action_to_label(action: np.ndarray, label: str) -> np.ndarray:
    """
    MoNa-pi continuous actions를 안전한 Soft-Decision 기반으로 스냅핑합니다.
    VLA의 연속 조향(Yaw) 성분을 복원하고 하드 임계치로 인한 속도 튐(채터링)을 억제합니다.
    """
    out = np.zeros(3, dtype=np.float32)
    label = (label or "").upper()

    raw_lx = float(action[0]) if action.size > 0 else 0.0
    raw_ly = float(action[1]) if action.size > 1 else 0.0
    raw_az = float(action[2]) if action.size > 2 else 0.0

    # 1. STOP 제어
    if label == "STOP":
        return out

    # 2. Linear X (전진): 최소 0.45 ~ 최대 0.85 범위로 안전 클리핑하되 연속성 보존
    out[0] = np.clip(raw_lx, 0.45, 0.85)

    # 3. Linear Y (횡이동): 하드 스냅핑 대신 연속형 ly 값을 부드럽게 필터링 (최대 ±0.45 제한)
    # 라벨에 횡방향(L/R) 지시가 있는 경우에만 횡이동 허용
    if label in ("FWD+L", "FORWARD_LEFT", "LEFT"):
        out[1] = np.clip(max(raw_ly, 0.15), 0.15, 0.45)
    elif label in ("FWD+R", "FORWARD_RIGHT", "RIGHT"):
        out[1] = np.clip(min(raw_ly, -0.15), -0.45, -0.15)
    else:
        # 직진 또는 제자리 회전 시 불필요한 게걸음 횡이동 억제 (감쇄율 90%)
        out[1] = raw_ly * 0.1

    # 4. Angular Z (조향): 무력화되었던 회전(Yaw) 성분을 VLA 출력값으로부터 복원
    # 제자리 회전 라벨인 경우 강력한 회전 적용, 그 외의 주행 중에는 미세 선조향(50% 게인) 적용
    if label in ("ROT_L", "TURN_L"):
        out[2] = 0.22
    elif label in ("ROT_R", "TURN_R"):
        out[2] = -0.22
    else:
        # 주행(직진/전진) 중 정면 타겟을 부드럽게 지향할 수 있도록 회전 성분 복원
        out[2] = np.clip(raw_az * 0.5, -0.15, 0.15)

    # FWD+L/R 제어 시 과속 방지를 위해 전진 속도 약간 감쇄
    if label in ("FWD+L", "FORWARD_LEFT", "FWD+R", "FORWARD_RIGHT"):
        out[0] = min(out[0], 0.70)

    # 제자리 회전 시 전진/횡이동 속도 0으로 제어
    if label in ("ROT_L", "TURN_L", "ROT_R", "TURN_R"):
        out[0] = 0.0
        out[1] = 0.0

    return out


def _resolve_default_model_paths() -> tuple[str, str]:
    env_ckpt = os.getenv("VLA_CHECKPOINT_PATH")
    env_cfg = os.getenv("VLA_CONFIG_PATH")
    if env_ckpt and env_cfg:
        env_ckpt_path = Path(env_ckpt).expanduser()
        env_cfg_path = Path(env_cfg).expanduser()
        if env_ckpt_path.exists() and env_cfg_path.exists():
            return str(env_ckpt_path.resolve()), str(env_cfg_path.resolve())
        logger.warning(
            "Ignoring stale VLA_CHECKPOINT_PATH/VLA_CONFIG_PATH because one of them does not exist. "
            "ckpt=%s cfg=%s",
            env_ckpt,
            env_cfg,
        )

    for ckpt_path, cfg_path in DEFAULT_DEBUG_MODEL_CANDIDATES:
        if ckpt_path.exists() and cfg_path.exists():
            return str(ckpt_path), str(cfg_path)

    raise FileNotFoundError(
        "No valid default checkpoint/config pair found for inference_server. "
        "Set VLA_CHECKPOINT_PATH and VLA_CONFIG_PATH explicitly."
    )


def _get_model_candidates() -> list[dict[str, str]]:
    items = []
    for ckpt_path, cfg_path in DEFAULT_DEBUG_MODEL_CANDIDATES:
        if ckpt_path.exists() and cfg_path.exists():
            items.append(
                {
                    "name": cfg_path.stem,
                    "checkpoint_path": str(ckpt_path),
                    "config_path": str(cfg_path),
                }
            )
    return items


def _resolve_model_selection(
    candidate_name: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    config_path: Optional[str] = None,
) -> tuple[str, str]:
    if candidate_name:
        for item in _get_model_candidates():
            if item["name"] == candidate_name:
                return item["checkpoint_path"], item["config_path"]
        raise HTTPException(status_code=404, detail=f"Unknown model candidate: {candidate_name}")

    if checkpoint_path and config_path:
        ckpt = Path(checkpoint_path).expanduser().resolve()
        cfg = Path(config_path).expanduser().resolve()
        if not ckpt.exists():
            raise HTTPException(status_code=404, detail=f"Checkpoint not found: {ckpt}")
        if not cfg.exists():
            raise HTTPException(status_code=404, detail=f"Config not found: {cfg}")
        return str(ckpt), str(cfg)

    return _resolve_default_model_paths()


def _build_image_path_context(image_path: Path) -> dict[str, Any]:
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")
    if not image_path.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {image_path}")

    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    siblings = sorted(
        [p for p in image_path.parent.iterdir() if p.is_file() and p.suffix.lower() in image_exts],
        key=lambda p: p.name.lower(),
    )
    names = [str(p) for p in siblings]
    try:
        idx = names.index(str(image_path))
    except ValueError:
        idx = -1

    prev_path = names[idx - 1] if idx > 0 else None
    next_path = names[idx + 1] if idx != -1 and idx + 1 < len(names) else None
    return {
        "current_path": str(image_path),
        "parent_dir": str(image_path.parent),
        "index": idx,
        "count": len(names),
        "prev_path": prev_path,
        "next_path": next_path,
    }

class ModelLoadRequest(BaseModel):
    """모델 로드 요청 스키마"""
    checkpoint_path: str
    config_path: str
    precision: Literal["fp16", "int8"] = "fp16"
    refresh: bool = True
    

class MobileVLAInference:
    """Mobile VLA 추론 파이프라인"""
    
    def _load_config_recursive(self, path: str) -> dict:
        """Load JSON config and recursively merge with parent if specified."""
        # Handle case where path might be hardcoded to old user
        if "/home/billy/25-1kp/vla" in path:
            path = path.replace("/home/billy/25-1kp/vla", project_root)
            
        if not os.path.exists(path):
            # Try relative to configs dir
            alt_path = os.path.join(project_root, "Mobile_VLA", "configs", os.path.basename(path))
            if os.path.exists(alt_path):
                path = alt_path
                
        with open(path, 'r') as f:
            config = json.load(f)
            
        if "parent" in config and config["parent"]:
            parent_path = config["parent"]
            parent_config = self._load_config_recursive(parent_path)
            # Merge: config values override parent_config values
            # For nested dicts like train_setup, we should do a shallow merge or deep merge?
            # Standard RoboVLMs uses deep_update for some things, but shallow merge is usually enough for top level.
            # Let's do a simple recursive merge for dicts.
            def deep_update(d, u):
                for k, v in u.items():
                    if isinstance(v, dict):
                        d[k] = deep_update(d.get(k, {}), v)
                    else:
                        d[k] = v
                return d
            
            merged = copy.deepcopy(parent_config)
            deep_update(merged, config)
            return merged
        return config

    def __init__(self, checkpoint_path: str, config_path: str, device: str = "cuda", use_quant: Optional[bool] = None):
        """
        Args:
            checkpoint_path: LoRA Fine-tuned 모델 checkpoint 경로
            config_path: Config JSON 파일 경로
            device: "cuda" or "cpu"
            use_quant: Explicitly set quantization (True=INT8, False=FP16). If None, use Env.
        """
        import copy
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.model_name = Path(config_path).stem
        
        # Determine quantization mode
        if use_quant is not None:
            self.use_quant = use_quant
        else:
            self.use_quant = os.getenv("VLA_QUANTIZE", "false").lower() == "true"
        
        # --- 1. Core State Initialization ---
        self.action_buffer = ActionBuffer(chunk_size=10)
        self.inference_count = 0
        self.image_history = []
        self.prev_action = np.zeros(3)  # [lx, ly, az] 3DOF for Action Smoothing
        self.smoothing_alpha = 0.45      # XY 평활화 계수 (0.45)
        self.smoothing_alpha_az = 0.80   # AZ 평활화 계수 (0.80)
        self.speed_scaling_enabled = True
        self.grounding_skip_n = 1
        self.smooth_enabled = True
        self.stop_area_threshold = 0.18
        self.stop_cx_tolerance = 0.25
        self.prev_action_3d = [0.0, 0.0, 0.0]
        self._grounding_cache = None
        
        logger.info(f"Loading model from {checkpoint_path}")
        logger.info(f"Using device: {device}")
        
        # Load config recursively
        self.config = self._load_config_recursive(config_path)
        
        # 필수 탑레벨 및 내부 키 기본값 인젝션 (KeyError 방지)
        defaults = {
            "robovlm_name": "RoboKosMos",
            "use_hand_rgb": False,
            "fwd_pred_next_n": 1,
            "seed": 123,
            "norm_action": False,
            "model_path": ".vlms/kosmos-2-patch14-224",
            "model_config": ".vlms/kosmos-2-patch14-224/config.json",
            "fwd_head": None,  # KeyError: 'fwd_head' 방지 및 모듈 임포트 차단
            "act_head": {},  # KeyError: 'act_head' 방지
            "cap_loss_ratio": 0.0,            # KeyError: 'cap_loss_ratio' 방지
            "arm_gripper_loss_ratio": 0.0,    # KeyError: 'arm_gripper_loss_ratio' 방지
            "fwd_loss_ratio": 0.0,            # KeyError: 'fwd_loss_ratio' 방지
            "val_dataset": {},                # KeyError: 'val_dataset' 방지
        }
        for k, v in defaults.items():
            self.config.setdefault(k, v)
            
        # act_head 기본 속성 보완 (KeyError 방지)
        act_head_defaults = {
            "action_space": "continuous",
            "window_size": 8,
            "down_sample": "pooling",
            "latent": 1
        }
        for k, v in act_head_defaults.items():
            self.config["act_head"].setdefault(k, v)
            
        # train_setup 기본값 보완
        train_setup_defaults = {
            "precision": "16-mixed",
            "predict_action": True,
            "predict_forward": False,
            "predict_forward_hand": False,
            "predict_caption": False,
            "train_vision": False,
            "bits": -1,
            "freeze_mm_mlp_adapter": False,
            "freeze_backbone": True,
            "freeze_resampler": True,
            "tune_mm_mlp_adapter": False,
            "tune_mm_projector": True,
            "lora_enable": False
        }
        if "train_setup" not in self.config or not isinstance(self.config["train_setup"], dict):
            self.config["train_setup"] = train_setup_defaults
        else:
            for k, v in train_setup_defaults.items():
                self.config["train_setup"].setdefault(k, v)

        # tokenizer 기본값 보완
        tokenizer_defaults = {
            "type": "AutoProcessor",
            "pretrained_model_name_or_path": ".vlms/kosmos-2-patch14-224",
            "tokenizer_type": "kosmos",
            "max_text_len": 256
        }
        if "tokenizer" not in self.config or not isinstance(self.config["tokenizer"], dict):
            self.config["tokenizer"] = tokenizer_defaults
        else:
            for k, v in tokenizer_defaults.items():
                self.config["tokenizer"].setdefault(k, v)

        # vlm 기본값 보완
        vlm_defaults = {
            "type": "AutoModelForVision2Seq",
            "name": "kosmos",
            "pretrained_model_name_or_path": ".vlms/kosmos-2-patch14-224"
        }
        if "vlm" not in self.config or not isinstance(self.config["vlm"], dict):
            self.config["vlm"] = vlm_defaults
        else:
            for k, v in vlm_defaults.items():
                self.config["vlm"].setdefault(k, v)

        self._normalize_config_paths()

        # Define Window Size with proper fallback priority
        # 1. Root window_size, 2. act_head.window_size, 3. train_dataset.window_size, 4. Default 8
        self.window_size = (
            self.config.get("window_size")
            or self.config.get("act_head", {}).get("window_size")
            or int(self.config.get("train_dataset", {}).get("window_size", 8))
        )
        # base_trainer가 self.configs["window_size"]를 직접 참조하므로 강제 주입
        self.config["window_size"] = self.window_size
        
        # Runtime Intervention parameters
        self.logit_penalty = self.config.get("act_head", {}).get("logit_penalty", {})
        self.temperature = float(self.config.get("act_head", {}).get("temperature", 1.0))
        
        logger.info(f"📊 [INIT] Window: {self.window_size}, T={self.temperature}, Penalty: {self.logit_penalty}")

        self.inference_mode = "classification" if self.config.get("discrete_action", False) else self._resolve_inference_mode()
        self.num_classes = int(self.config.get("train_dataset", {}).get("num_classes", 9))
        self.class_labels, self.class_action_map, self.class_index_action_map = self._build_classification_map()
        
        # V3/V5 Dataset Label Mapping
        speed = float(self.config.get("classification_speed", 1.15))
        angle_speed = 0.5 # Default rotation speed
        diag = speed * 0.707
        
        # [CRITICAL] Sync with nav_h5_dataset_impl.py mapping.
        # 8-class: 0 Stop, 1 F, 2 L, 3 R, 4 FL, 5 FR, 6 turn-L, 7 turn-R.
        # 6-class: same first six classes, omitting turns.
        if self.num_classes == 6:

            self.class_index_action_map = {
                0: [0.0, 0.0, 0.0],   # STOP
                1: [speed, 0.0, 0.0], # FORWARD
                2: [0.0, speed, 0.0],  # LEFT
                3: [0.0, -speed, 0.0], # RIGHT
                4: [diag, diag, 0.0],  # FWD+L
                5: [diag, -diag, 0.0], # FWD+R
            }
            logger.info("🎯 Applied 6-class sync mapping: 2=L, 3=R, 4=FL, 5=FR")
        elif self.num_classes == 8:
            self.class_index_action_map = {
                0: [0.0, 0.0, 0.0],   # STOP
                1: [speed, 0.0, 0.0], # FORWARD
                2: [0.0, speed, 0.0],  # LEFT
                3: [0.0, -speed, 0.0], # RIGHT
                4: [diag, diag, 0.0],  # FWD+L
                5: [diag, -diag, 0.0], # FWD+R
                6: [0.0, 0.0, angle_speed],  # TURN_L
                7: [0.0, 0.0, -angle_speed], # TURN_R
            }
            logger.info("🎯 Applied 8-class V5 sync mapping: 2=L, 3=R, 4=FL, 5=FR, 6=TL, 7=TR")
        else:

            self.class_index_action_map = {
                0: [0.0, 0.0, 0.0],   # STOP
                1: [speed, 0.0, 0.0], # FORWARD (W)
                2: [0.0, 0.0, -angle_speed], # ROTATE LEFT (T key)
                3: [0.0, speed, 0.0],  # STRAFE LEFT (A)
                4: [0.0, -speed, 0.0], # STRAFE RIGHT (D)
                5: [diag, diag, 0.0],  # FWD + STRAFE LEFT
                6: [diag, -diag, 0.0], # FWD + STRAFE RIGHT
                7: [-speed, 0.0, 0.0], # BACKWARD (S)
                8: [0.0, 0.0, angle_speed],  # ROTATE RIGHT (R key)
            }
            logger.info("🎯 Normal 9-class mapping applied")
        
        # Integration fallback for scale_factor
        self.scale_factor = self.config.get("scale_factor", self.config.get("classification_speed", 1.15))
        
        logger.info(f"✅ [INIT] Inference Mode: {self.inference_mode.upper()}")
        
        # --- 2. Build/Load Model Infrastructure ---
        self._load_model()
        
        # Override history window in act_head internal state
        try:
            potential_model = self.model.model if hasattr(self.model, 'model') else self.model
            for name in ['act_head', 'policy_head']:
                if hasattr(potential_model, name):
                    head = getattr(potential_model, name)
                    if hasattr(head, 'history_len'):
                        head.history_len = self.window_size
                        logger.info(f"✅ [INTERVENTION] Applied history window size: {head.history_len}")
                        break
        except Exception as e:
            logger.warning(f"⚠️ [INTERVENTION] Failed to apply window size: {e}")
        
        # Re-normalize tokenizer path after MobileVLATrainer may have overwritten it
        # (MobileVLATrainer.__init__ can derive model_path from model_url which may be a stale path)
        self._normalize_config_paths()
        vlm_model_path = str(Path(project_root) / ".vlms" / "kosmos-2-patch14-224")
        tok_path = self.config.get('tokenizer', {}).get('pretrained_model_name_or_path', vlm_model_path)
        if not Path(tok_path).exists():
            tok_path = vlm_model_path
        from transformers import AutoProcessor
        # 인터넷 연결 시도 없이 로컬 가중치만 사용하도록 local_files_only=True 추가
        self.processor = AutoProcessor.from_pretrained(tok_path, local_files_only=True)
        logger.info("✅ MobileVLAInference initialized successfully")

    def _normalize_config_paths(self) -> None:
        """Normalize relative config paths against project root for stable CWD-independent loading."""
        def _norm_path(raw: Any) -> Any:
            if not isinstance(raw, str) or raw.strip() == "":
                return raw
            val = raw.strip()
            if "://" in val:  # URL / HF repo-like identifiers should remain untouched
                return val
            p = Path(val).expanduser()
            if p.is_absolute():
                # Only return as-is if it actually exists; otherwise fall through to search
                if p.exists():
                    return str(p)
                # Absolute path doesn't exist (e.g. stale /home/soda/... path from ancestor config)
                # Extract the model basename and search under project_root
                basename = p.name
                project_candidate = Path(project_root) / ".vlms" / basename
                if project_candidate.exists():
                    return str(project_candidate.resolve())
                return str(p)  # fallback: return original even if not found

            candidates = [
                Path(project_root) / val,
                Path(self.config_path).resolve().parent / val,
                Path.cwd() / val,
            ]
            for c in candidates:
                if c.exists():
                    return str(c.resolve())
            # Fallback: project-root anchored path
            return str((Path(project_root) / val).resolve())

        for key in ("model_path", "model_config", "model_load_path", "model_url"):
            if key in self.config:
                self.config[key] = _norm_path(self.config.get(key))

        tokenizer_cfg = self.config.get("tokenizer", {})
        if isinstance(tokenizer_cfg, dict) and "pretrained_model_name_or_path" in tokenizer_cfg:
            tokenizer_cfg["pretrained_model_name_or_path"] = _norm_path(tokenizer_cfg.get("pretrained_model_name_or_path"))

        vlm_cfg = self.config.get("vlm", {})
        if isinstance(vlm_cfg, dict) and "pretrained_model_name_or_path" in vlm_cfg:
            vlm_cfg["pretrained_model_name_or_path"] = _norm_path(vlm_cfg.get("pretrained_model_name_or_path"))

    def _resolve_inference_mode(self) -> str:
        """Resolve runtime inference mode from config."""
        mode = str(self.config.get("inference_mode", "")).strip().lower()
        act_head_type = str(self.config.get("act_head", {}).get("type", "")).lower()
        if mode in {"classification", "regression"}:
            return mode
        if "classification" in act_head_type:
            return "classification"
        return "regression"

    def _build_classification_map(self) -> tuple[List[str], dict[str, List[float]], dict[int, List[float]]]:
        """Build class label/action map for classification inference."""
        num_classes = int(self.config.get("act_head", {}).get("num_classes", 9))
        if num_classes == 6:
            default_labels = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD+L", "FWD+R"]
        elif num_classes == 8:
            default_labels = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "TURN_L", "TURN_R"]
        else:
            default_labels = ["STOP", "FORWARD", "BACKWARD", "LEFT", "RIGHT", "FORWARD_LEFT", "FORWARD_RIGHT", "BACKWARD_LEFT", "BACKWARD_RIGHT"]
        labels = self.config.get("class_labels", default_labels[:num_classes])
        if not isinstance(labels, list) or not labels:
            labels = default_labels[:num_classes]
        labels = [str(x).upper() for x in labels]

        speed = float(self.config.get("classification_speed", 1.15))
        diagonal = float(self.config.get("classification_diagonal_speed", speed))
        default_map = {
            "STOP": [0.0, 0.0],
            "FORWARD": [speed, 0.0],
            "BACKWARD": [-speed, 0.0],
            "LEFT": [0.0, speed],
            "RIGHT": [0.0, -speed],
            "FORWARD_LEFT": [diagonal, diagonal],
            "FORWARD_RIGHT": [diagonal, -diagonal],
            "BACKWARD_LEFT": [-diagonal, diagonal],
            "BACKWARD_RIGHT": [-diagonal, -diagonal],
            "FL": [diagonal, diagonal],
            "FR": [diagonal, -diagonal],
            "BL": [-diagonal, diagonal],
            "BR": [-diagonal, -diagonal],
            "F": [speed, 0.0],
            "B": [-speed, 0.0],
            "L": [0.0, speed],
            "R": [0.0, -speed],
        }
        # Index-priority default map (0~8): Stop/F/B/L/R/FL/FR/BL/BR
        default_index_map = {
            0: [0.0, 0.0],
            1: [speed, 0.0],
            2: [-speed, 0.0],
            3: [0.0, speed],
            4: [0.0, -speed],
            5: [diagonal, diagonal],
            6: [diagonal, -diagonal],
            7: [-diagonal, diagonal],
            8: [-diagonal, -diagonal],
        }

        user_map = self.config.get("classification_action_map", {})
        if isinstance(user_map, dict):
            for k, v in user_map.items():
                if isinstance(v, (list, tuple)) and len(v) >= 2:
                    try:
                        idx = int(k)
                        default_index_map[idx] = [float(v[0]), float(v[1])]
                    except (ValueError, TypeError):
                        default_map[str(k).upper()] = [float(v[0]), float(v[1])]

        return labels, default_map, default_index_map

    def _extract_action_tensor(self, action_out: Any) -> Any:
        """Normalize output variants to action/logits tensor."""
        if isinstance(action_out, tuple):
            # Regression heads may return (action, gripper). Classification should not concat.
            if self.inference_mode == "regression":
                if (
                    len(action_out) == 2
                    and isinstance(action_out[0], torch.Tensor)
                    and isinstance(action_out[1], torch.Tensor)
                    and action_out[0].shape[:-1] == action_out[1].shape[:-1]
                ):
                    act_t, grip_t = action_out
                    logger.debug(f"📊 [TENSOR STATS] Actions mean: {act_t.mean():.4f}, Gripper mean: {grip_t.mean():.4f}")
                    return torch.cat(action_out, dim=-1)
            return action_out[0]
        return action_out

    def _get_policy_head(self):
        potential_model = self.model.model if hasattr(self.model, 'model') else self.model
        for name in ['act_head', 'policy_head']:
            if hasattr(potential_model, name):
                return getattr(potential_model, name)
        return None

    def _get_history_length(self, policy_head) -> int:
        try:
            if policy_head is not None and hasattr(policy_head, "history_memory"):
                return len(policy_head.history_memory)
        except Exception:
            pass
        return -1

    def _build_prompt(self, instruction: str) -> str:
        if not instruction.startswith("<grounding>"):
            return f"<grounding>An image of a robot {instruction}"
        return instruction

    def _decode_image_to_pil(self, image_input: str | np.ndarray) -> Image.Image:
        if isinstance(image_input, str):
            image_bytes = base64.b64decode(image_input)
            return Image.open(io.BytesIO(image_bytes)).convert('RGB')
        if isinstance(image_input, np.ndarray):
            return Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        raise ValueError(f"Unsupported image type: {type(image_input)}")

    def _image_to_base64_png(self, image: Image.Image) -> str:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _build_image_debug_payload(self, image_input: str | np.ndarray) -> dict[str, Any]:
        image = self._decode_image_to_pil(image_input)
        resized = image.resize((224, 224), Image.BICUBIC)
        return {
            "original_size": {"width": image.width, "height": image.height},
            "model_input_size": {"width": 224, "height": 224},
            "original_base64_png": self._image_to_base64_png(image),
            "resized_base64_png": self._image_to_base64_png(resized),
        }

    def _tokenize_instruction_with_debug(self, instruction: str) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        lang_x, attention_mask = self._tokenize_instruction(instruction)
        input_ids = lang_x[0].detach().cpu().tolist()
        mask = attention_mask[0].detach().cpu().tolist()
        active_ids = [token_id for token_id, keep in zip(input_ids, mask) if int(keep) == 1]
        tokens_preview = self.processor.tokenizer.convert_ids_to_tokens(active_ids[:64])
        return lang_x, attention_mask, {
            "token_count": int(sum(mask)),
            "active_token_ids_preview": active_ids[:64],
            "tokens_preview": tokens_preview,
            "max_text_len": int(self.config['tokenizer']['max_text_len']),
        }

    def _summarize_classification_chunk(self, full_chunk: np.ndarray) -> dict[str, Any]:
        if full_chunk.size == 0:
            return {
                "raw_logits": [],
                "adjusted_logits": [],
                "top_classes": [],
                "predicted_class_idx": None,
                "predicted_class_name": None,
                "mapped_action": [0.0, 0.0],
            }

        logits = np.array(full_chunk[0], dtype=np.float32).reshape(-1)
        adjusted = logits.copy()
        if adjusted.size > 1 and self.logit_penalty:
            penalty_vec = np.zeros_like(adjusted)
            for k, v in self.logit_penalty.items():
                try:
                    idx = int(k)
                    if idx < len(penalty_vec):
                        penalty_vec[idx] = float(v)
                except Exception:
                    continue
            adjusted = adjusted + penalty_vec

        if adjusted.size > 1 and self.temperature != 1.0:
            adjusted = adjusted / self.temperature

        if adjusted.size == 0:
            class_idx = None
        elif adjusted.size == 1:
            class_idx = int(adjusted[0])
        else:
            class_idx = int(np.argmax(adjusted))

        top_classes = []
        if adjusted.size > 1:
            topk = min(5, adjusted.size)
            order = np.argsort(adjusted)[::-1][:topk]
            for idx in order:
                class_name = self.class_labels[idx] if idx < len(self.class_labels) else f"IDX_{idx}"
                mapped = self.class_index_action_map.get(idx, self.class_action_map.get(class_name, [0.0, 0.0]))
                top_classes.append(
                    {
                        "class_idx": int(idx),
                        "class_name": class_name,
                        "score": float(adjusted[idx]),
                        "mapped_action": [float(x) for x in mapped],
                    }
                )

        predicted_class_name = None
        mapped_action = [0.0, 0.0]
        if class_idx is not None:
            predicted_class_name = self.class_labels[class_idx] if class_idx < len(self.class_labels) else f"IDX_{class_idx}"
            mapped_action = self.class_index_action_map.get(
                class_idx,
                self.class_action_map.get(predicted_class_name, [0.0, 0.0]),
            )

        return {
            "raw_logits": logits.tolist(),
            "adjusted_logits": adjusted.tolist(),
            "top_classes": top_classes,
            "predicted_class_idx": class_idx,
            "predicted_class_name": predicted_class_name,
            "mapped_action": [float(x) for x in mapped_action],
        }

    def _decode_classification_action(self, full_chunk: np.ndarray) -> np.ndarray:
        """Convert class logits/prediction to continuous [linear_x, linear_y]."""
        if full_chunk.size == 0:
            return np.array([0.0, 0.0], dtype=np.float32)

        logits = full_chunk[0]
        # ROT_L(6) / ROT_R(7) 마스킹 — 잘못 선택 시 무한 스핀 유발 방지
        if logits.size > 6:
            logits = logits.copy()
            logits[6] = -np.inf
            logits[7] = -np.inf
            
        if logits.ndim == 0:
            class_idx = int(logits)
            score = float(logits)
        elif logits.ndim == 1 and logits.size == 1:
            # Some models may return class index directly as shape (1,)
            class_idx = int(logits[0])
            score = float(logits[0])
        else:
            # 1. Apply Logit Penalization (Soft-Decision)
            if self.logit_penalty:
                penalty_vec = np.zeros_like(logits)
                for k, v in self.logit_penalty.items():
                    try:
                        idx = int(k)
                        if idx < len(penalty_vec):
                            penalty_vec[idx] = float(v)
                    except: continue
                logits = logits + penalty_vec
                logger.info(f"📤 [INTERVENTION] Penalty Applied: {self.logit_penalty}")
            
            # 2. Apply Temperature Scaling (Confidence Smoothing)
            if self.temperature != 1.0:
                logits = logits / self.temperature
                logger.info(f"📤 [INTERVENTION] Temperature Scaling Applied: T={self.temperature}")

            class_idx = int(np.argmax(logits))
            score = float(logits[class_idx])

        class_name = self.class_labels[class_idx] if class_idx < len(self.class_labels) else f"IDX_{class_idx}"
        mapped = self.class_index_action_map.get(class_idx, self.class_action_map.get(class_name, [0.0, 0.0]))
        action = np.array(mapped, dtype=np.float32)
        logger.info(f"📤 [CLASS ACTION] idx={class_idx}, class={class_name}, score={score:.4f}, action={action.tolist()}")
        return action
        
    def _load_model(self):
        """모델 로딩 (VLA_QUANTIZE=false면 FP16, true면 INT8로 로드)"""
        import sys
        import os
        # Use the module-level project_root (correctly resolved to MoNaVLA dir)
        # Fall back to VLA_ROOT env only if set; do NOT default to /home/soda/vla
        _vla_root = os.getenv("VLA_ROOT", project_root)
        # Ensure RoboVLMs is in path
        if _vla_root not in sys.path:
            sys.path.append(_vla_root)

        def _update_paths(config_dict):
            # kosmos-2 model path normalization: ensure local path is used
            fixed_model_base = os.getenv("VLA_MODEL_PATH", os.path.join(project_root, ".vlms"))
            old_roots = ["/home/billy/25-1kp/vla", "/home/soda/vla"]

            for k, v in config_dict.items():
                if isinstance(v, str):
                    # 1순위: 모델 기반 경로 전용 강제 치환 (가장 확실함)
                    if "kosmos-2-patch14-224" in v:
                        candidate = os.path.join(fixed_model_base, "kosmos-2-patch14-224")
                        if os.path.exists(candidate):
                            config_dict[k] = candidate
                        continue

                    # 2순위: 기타 일반 경로 치환
                    for old_root in old_roots:
                        if old_root in v:
                            config_dict[k] = v.replace(old_root, project_root)
                elif isinstance(v, dict):
                    _update_paths(v)
        _update_paths(self.config)
        
        try:
            from robovlms.train.mobile_vla_trainer import MobileVLATrainer
        except ImportError:
            # Fallback for inference-integration branch structure where it's kept in a different directory
            import sys
            import os
            fallback_path = os.path.join(os.getenv("VLA_ROOT", "/home/soda/vla"), "Robo+", "Mobile_VLA", "core", "train_core")
            if fallback_path not in sys.path:
                sys.path.append(fallback_path)
            from mobile_vla_trainer import MobileVLATrainer

        # Inject model backbone mapping required by newer V3 configs
        try:
            from robovlms.model.backbone.robokosmos import RoboKosMos
            import robovlms.model.backbone as backbone
            backbone.__dict__["RoboVLM-Nav"] = RoboKosMos
            
            import robovlms.model.policy_head as policy_head
            from robovlms.model.policy_head.mobile_vla_policy import MobileVLAClassificationDecoder, MobileVLALSTMDecoder
            policy_head.__dict__["NavPolicy"] = MobileVLAClassificationDecoder
            policy_head.__dict__["NavPolicyRegression"] = MobileVLALSTMDecoder
            
            logger.info("🔧 Injected RoboVLM-Nav & NavPolicy mapping")
        except Exception as base_e:
            logger.warning(f"⚠️ Failed to inject model mapping: {base_e}")
        
        if self.use_quant:
            from transformers import BitsAndBytesConfig
            logger.info("🔧 [MODE] BitsAndBytes INT8 (Standard/Stable)")
            bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
            self.model = MobileVLATrainer(self.config, quantization_config=bnb_config)
        else:
            logger.info("🚀 [MODE] FP16/BF16 (Action Diversity prioritized)")
            self.model = MobileVLATrainer(self.config)
            # FP16 conversion happens after loading params to avoid mismatch during load
            
        # [LEAN LOADING] Load only necessary weights and move them directly to device
        logger.info(f"Loading checkpoint (Lean Mode): {self.checkpoint_path}")
        
        # Load state dict on CPU
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu', weights_only=False)
        full_state_dict = checkpoint.get('model_state_dict', checkpoint.get('state_dict'))
        
        # Filter for Projector and Policy Head only (Backbone is already official)
        # This avoids size mismatch and saves 6GB+ RAM/VRAM
        logger.info("🎯 Filtering: Loading only Projector and Policy Head")
        state_dict = {}
        for k, v in full_state_dict.items():
            if any(x in k for x in ["image_to_text_projection", "act_head", "policy_head", "resampler", "action_token", "lora"]):
                # Handle model. prefix
                new_key = k
                if k.startswith('model.') and not hasattr(self.model, 'model'):
                     new_key = k.replace('model.', '', 1)
                state_dict[new_key] = v
        
        # Cleanup full checkpoint immediately
        del full_state_dict
        del checkpoint
        gc.collect()
        
        # Load into model
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        logger.info(f"✅ Loaded {len(state_dict)} fine-tuned weights")
        
        # [CRITICAL] 1. Finalize Precision
        self.model.to(self.device).eval()
        
        if not self.use_quant:
            # Note: LSTM (act_head) does not support BFloat16 on CUDA; use Float16 for stability
            logger.info("🚀 Converting backbone to Float16 (Standard)")
            self.model.half()
        else:
            logger.info("✅ Kept in INT8 (BitsAndBytes) mode")
             
        # [DEBUG/PATH FIX] Store policy_head for robust First-Frame Safety
        self.policy_head = None
        
        # 실제 모델 구조(RoboKosMos)에서는 'act_head'라는 이름을 사용하기도 합니다.
        potential_model = self.model.model if hasattr(self.model, 'model') else self.model
        
        for name in ['act_head', 'policy_head']:
            if hasattr(potential_model, name):
                self.policy_head = getattr(potential_model, name)
                logger.info(f"✅ Verified: policy_head found at model.{name}")
                break
        
        if self.policy_head is None:
            logger.warning("⚠️ Warning: policy_head (act_head) not found in model structure!")
            
        # Measure actual GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"\n📊 Actual GPU Memory: {gpu_memory:.3f} GB")
            logger.info(f"   Expected: ~1.7 GB (73% reduction from 6.3GB FP32)")
        
        if self.use_quant:
            logger.info("✅ Model loaded with BitsAndBytes INT8")
        else:
            logger.info("✅ Model loaded in FP16/BF16 (High Precision)")
        logger.info("="*60)
        
        # Setup image transforms
        from torchvision import transforms
        # 학습 규격에 맞는 정규화 값 유지
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            # [CRITICAL] 87% 모델(Kosmos기반)은 자체 전처리기에서 정규화를 수행할 가능성이 높습니다.
            # 서버단 정규화가 중복되면 모델이 '장님'이 됩니다. (BGR/RGB 문제 포함)
            # transforms.Normalize(
            #     mean=self.config.get('image_mean', [0.48145466, 0.4578275, 0.40821073]),
            #     std=self.config.get('image_std', [0.26862954, 0.26130258, 0.27577711])
            # )
        ])
        
    def preprocess_image(self, image_input: str | np.ndarray) -> torch.Tensor:
        """
        이미지를 모델 입력 형식으로 변환 (Base64 문자열 또는 NumPy 배열 지원)
        """
        image = self._decode_image_to_pil(image_input)
        
        # Transform
        image_tensor = self.image_transform(image)
        
        # [DEBUG] 이미지 입력 변화 확인 (환각 디버깅용)
        img_mean = image_tensor.mean().item()
        logger.debug(f"🖼️ [INPUT] Image Mean: {img_mean:.6f}")
        
        # Add batch and window dimensions: (3, 224, 224) -> (1, 1, 3, 224, 224)
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        
        return image_tensor.to(self.device)
        
    def _tokenize_instruction(self, instruction: str) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Instruction을 tokenize하여 tensor로 변환
        """
        # [TEST] 접두사 없이 순수 Instruction만 사용 (학습 데이터 규합 확인용)
        # 만약 학습 시 <grounding> 포맷을 썼다면 다시 복구해야 함
        logger.debug(f"📝 [INPUT] Prompt: {instruction}")
            
        # Tokenize instruction using the pre-initialized processor
        encoded = self.processor.tokenizer(
            instruction,
            return_tensors='pt',
            padding='max_length',
            max_length=self.config['tokenizer']['max_text_len'],
            truncation=True
        )
        
        lang_x = encoded['input_ids'].to(self.device)  # (1, seq_len)
        attention_mask = encoded['attention_mask'].to(self.device)  # (1, seq_len)
        
        return lang_x, attention_mask

    
    def extract_vision_feature(self, image_input: str) -> np.ndarray:
        """
        KosMos-2 vision encoder에서 1024-dim global average pooled feature 추출.
        Exp47 MLP의 vision cache 갱신에 사용 (action head 실행 안 함).

        Returns:
            np.ndarray: (1024,) float32 vision feature vector
        """
        try:
            image_tensor = self.preprocess_image(image_input)  # (1, 1, 3, 224, 224)

            potential_model = self.model.model if hasattr(self.model, 'model') else self.model

            # KosMos-2 vision backbone 추출 시도
            vision_encoder = None
            for attr in ['vision_model', 'vision_encoder', 'image_model', 'visual_encoder']:
                if hasattr(potential_model, attr):
                    vision_encoder = getattr(potential_model, attr)
                    break

            if vision_encoder is not None:
                # (1, 1, 3, 224, 224) → (1, 3, 224, 224) 로 squeeze
                img = image_tensor.squeeze(1)
                with torch.no_grad():
                    vis_out = vision_encoder(img)
                # last_hidden_state 또는 직접 tensor
                if hasattr(vis_out, 'last_hidden_state'):
                    feat = vis_out.last_hidden_state  # (1, N, D)
                elif isinstance(vis_out, torch.Tensor):
                    feat = vis_out
                else:
                    feat = vis_out[0]
                # global average pooling → (1024,)
                feat = feat.mean(dim=1).squeeze(0)  # (D,)
                # 1024-dim으로 맞추기
                feat_np = feat.float().cpu().numpy()
                if feat_np.shape[0] != 1024:
                    # 다운샘플 또는 패딩
                    if feat_np.shape[0] > 1024:
                        feat_np = feat_np[:1024]
                    else:
                        feat_np = np.pad(feat_np, (0, 1024 - feat_np.shape[0]))
                return feat_np.astype(np.float32)

            # fallback: 모델 전체 forward에서 hidden state 추출
            logger.warning("⚠️ [extract_vision_feature] vision encoder 직접 접근 실패 → zero feature 반환")
            return np.zeros(1024, dtype=np.float32)

        except Exception as e:
            logger.error(f"Error in extract_vision_feature: {e}")
            return np.zeros(1024, dtype=np.float32)

    def _caption_to_cx(self, caption_lower: str) -> Optional[float]:
        """자연어 설명문에서 방향 관련 키워드를 매칭하여 정규화된 횡방향 cx 값 반환"""
        for phrases, cx in self._CAPTION_DIRECTION_PATTERNS:
            if any(p in caption_lower for p in phrases):
                return cx
        return None

    def _parse_basket_bbox(self, caption: str, entities: list[Any], target_object: str) -> Optional[dict[str, Any]]:
        """Kosmos-2 생성 텍스트의 entity bbox 리스트에서 타겟 오브젝트(기본: basket 등)의 bbox를 추출"""
        target_lower = target_object.lower()
        target_keywords = {target_lower}
        for word in target_lower.split():
            if len(word) > 2:
                target_keywords.add(word)
        if "basket" in target_lower:
            target_keywords.update(["container", "bin", "laundry", "gray box", "pot"])

        candidates = []
        for entity_name, _span, boxes in entities:
            for box in boxes:
                x1, y1, x2, y2 = [float(v) for v in box]
                if max(x1, y1, x2, y2) > 1.5:
                    x1, y1, x2, y2 = x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0
                area = (x2 - x1) * (y2 - y1)
                if area > 0.85: # 전체 화면과 유사한 오검출 방지
                    continue
                candidates.append({
                    "entity": entity_name,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "cx": (x1 + x2) / 2.0, "cy": (y1 + y2) / 2.0,
                    "area": area,
                    "is_target": any(k in entity_name.lower() for k in target_keywords),
                })

        matched = [b for b in candidates if b["is_target"]]
        if matched:
            return matched[0]

        # entity 검출 실패 시 자연어 생성 내용(caption) 기반으로 횡방향 좌표 추정 폴백
        caption_lower = caption.lower()
        cx = self._caption_to_cx(caption_lower)
        if cx is not None:
            if cx < 0.2: label = "caption:far_left"
            elif cx < 0.4: label = "caption:left"
            elif cx > 0.8: label = "caption:far_right"
            elif cx > 0.6: label = "caption:right"
            else: label = "caption:center"
            return {"entity": label, "cx": cx, "cy": 0.6, "area": 0.06}
        return None

    def _run_grounding(self, pil_image: Image.Image, instruction: str) -> dict[str, Any]:
        """서버 내부의 Kosmos-2 backbone 모델을 직접 실행해 bbox 및 caption 획득"""
        target_object = instruction.lower().replace("the", "").replace("a", "").strip()
        prompt = f"<grounding>The {target_object} is at"
        
        # tokenizer로 인코딩
        inputs = self.processor(text=prompt, images=pil_image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        pixel_values = inputs["pixel_values"].to(next(self.model.parameters()).dtype)

        with torch.no_grad():
            potential_model = self.model.model if hasattr(self.model, 'model') else self.model
            generated = potential_model.generate(
                pixel_values=pixel_values,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_embeds=None,
                image_embeds_position_mask=inputs.get("image_embeds_position_mask"),
                use_cache=True,
                max_new_tokens=64,
            )

        new_ids = generated[:, inputs["input_ids"].shape[1] :]
        raw = self.processor.batch_decode(new_ids, skip_special_tokens=False)[0]
        caption, entities = self.processor.post_process_generation(raw)
        bbox = self._parse_basket_bbox(caption, entities, target_object)
        
        return {
            "bbox": bbox,
            "caption": caption,
        }

    # 방향 매칭용 클래스 변수 정의
    _CAPTION_DIRECTION_PATTERNS = [
        (["far left",  "extreme left",  "leftmost",    "bottom left",  "lower left",
          "front left", "left side",    "left corner",  "upper left",   "top left",
          "bottom-left", "lower-left",  "left-hand side"],               0.12),
        (["left"],                                                        0.25),
        (["far right", "extreme right", "rightmost",   "bottom right", "lower right",
          "front right", "right side",  "right corner", "upper right",  "top right",
          "bottom-right", "lower-right", "right-hand side"],             0.88),
        (["right"],                                                       0.75),
        (["center", "middle",  "straight ahead", "in front",
          "directly ahead",    "in the middle",  "front and center"],    0.5),
    ]

    def reset(self) -> None:
        """추론 카운트 및 history memory, 이전 액션 등을 초기화하여 새로운 에피소드를 준비합니다."""
        self.inference_count = 0
        self.image_history = []
        self.prev_action = np.zeros(3)
        self.prev_action_3d = [0.0, 0.0, 0.0]
        self._grounding_cache = None
        
        # 모델 내부 policy head의 history memory 초기화
        try:
            potential_model = self.model.model if hasattr(self.model, 'model') else self.model
            for name in ['act_head', 'policy_head']:
                if hasattr(potential_model, name):
                    head = getattr(potential_model, name)
                    if hasattr(head, 'history_memory'):
                        head.history_memory = []
                        logger.info("🔄 [RESET] act_head history_memory cleared")
                        break
        except Exception as e:
            logger.warning(f"⚠️ Policy head history reset failed: {e}")
            
        logger.info("🔄 [RESET] Inference count and control histories reset successfully")

    def predict(self, image_base64: str, instruction: str) -> tuple[np.ndarray, float, np.ndarray, dict]:
        """
        추론 실행 및 제어 알고리즘 통합 적용
        Returns:
            action: [linear_x, linear_y, angular_z] (EMA 및 speed scaling 적용됨)
            latency_ms: Inference time
            full_chunk: Full action chunk sequence (N, 3)
            debug_data: 모니터링을 위한 풍부한 부가 디버깅 데이터 딕셔너리
        """
        start_time = time.time()
        
        # 1. Grounding 연산 수행 (bbox & caption 획득)
        pil_img = self._decode_image_to_pil(image_base64)
        use_cache = (
            self.grounding_skip_n > 1
            and self.inference_count > 0
            and self.inference_count % self.grounding_skip_n != 0
            and self._grounding_cache is not None
        )
        if use_cache:
            grounding = self._grounding_cache
            g_latency = 0.0
        else:
            g_t0 = time.time()
            try:
                grounding = self._run_grounding(pil_img, instruction)
            except Exception as ge:
                logger.error(f"⚠️ Grounding failed: {ge}")
                grounding = {"bbox": None, "caption": ""}
            g_latency = (time.time() - g_t0) * 1000.0
            self._grounding_cache = grounding
            
        bbox = grounding["bbox"]
        caption = grounding["caption"]
        
        # 2. 정지 판단 로직 (goal_near_proxy)
        goal_near = False
        if bbox is not None:
            area = float(bbox.get("area", 0.0))
            cx = float(bbox.get("cx", 0.5))
            entity = str(bbox.get("entity", ""))
            is_fallback = entity in ("coarse_clf", "center_fallback", "") or entity.startswith("caption:")
            
            if not is_fallback:
                center_ok = abs(cx - 0.5) <= self.stop_cx_tolerance
                area_ok = area >= self.stop_area_threshold
                goal_near = area_ok and center_ok
        
        # 3. First-Frame Zero Enforcement
        potential_model = self.model.model if hasattr(self.model, 'model') else self.model
        current_policy_head = None
        for name in ['act_head', 'policy_head']:
            if hasattr(potential_model, name):
                current_policy_head = getattr(potential_model, name)
                break
        
        is_first_frame = (self.inference_count == 0)
        
        try:
            h_len = len(current_policy_head.history_memory) if current_policy_head and hasattr(current_policy_head, 'history_memory') else -1
            if is_first_frame:
                logger.warning(f"🛡️ First-Frame Zero Enforcement (ID: {id(self)}, InfCount: {self.inference_count}, HistLen: {h_len})")
        except Exception as e:
            logger.error(f"❌ Safety check failed: {e}")
        
        if is_first_frame:
            self.inference_count += 1
            if not instruction.startswith("<grounding>"):
                full_prompt_init = f"<grounding>An image of a robot {instruction}"
            else:
                full_prompt_init = instruction

            if logger_instance:
                logger_instance.update_instruction(instruction)
                
            logger.info(f"🛡️ Applying Enforcement... (Count now: {self.inference_count}, Prompt: {full_prompt_init})")
            with torch.no_grad():
                image_tensor = self.preprocess_image(image_base64)
                lang_x, attention_mask = self._tokenize_instruction(full_prompt_init)
                self.model.model.inference(vision_x=image_tensor, lang_x=lang_x, attention_mask=attention_mask)
            
            self.image_history = [] 
            latency_ms = (time.time() - start_time) * 1000
            
            debug_data = {
                "bbox": bbox,
                "grounding_caption": caption,
                "grounding_latency_ms": g_latency,
                "goal_near_proxy": goal_near,
                "speed_scale": 1.0,
                "grounding_cached": use_cache,
                "action_3d": [0.0, 0.0, 0.0],
                "predicted_class": 0,
                "predicted_label": "STOP",
                "source": "first_frame_enforced"
            }
            return np.array([0.0, 0.0, 0.0]), latency_ms, np.zeros((1, 3)), debug_data

        # 4. Prompt 구성
        if not instruction.startswith("<grounding>"):
            full_prompt = f"<grounding>An image of a robot {instruction}"
        else:
            full_prompt = instruction
            
        try:
            potential_model = self.model.model if hasattr(self.model, 'model') else self.model
            for name in ['act_head', 'policy_head']:
                if hasattr(potential_model, name):
                    head = getattr(potential_model, name)
                    if hasattr(head, 'history_len') and head.history_len != self.window_size:
                         head.history_len = self.window_size
                         logger.debug(f"🔄 [SYNC] History Window re-applied: {head.history_len}")
                         break
        except: pass
            
        with torch.no_grad():
            image_tensor = self.preprocess_image(image_base64)
            lang_x, attention_mask = self._tokenize_instruction(full_prompt)
            target_dtype = next(self.model.parameters()).dtype
            image_tensor = image_tensor.to(dtype=target_dtype, device=self.device)
            
            action = np.array([0.0, 0.0])
            full_chunk = np.zeros((1, 2))
            
            try:
                outputs = self.model.model.inference(
                    vision_x=image_tensor,
                    lang_x=lang_x,
                    attention_mask=attention_mask
                )
                
                if isinstance(outputs, dict) and 'action' in outputs:
                    action_out = outputs['action']
                    action_out = self._extract_action_tensor(action_out)
                    
                    if isinstance(action_out, torch.Tensor):
                        full_chunk = action_out.detach().float().cpu().numpy()
                    else:
                        full_chunk = action_out

                    if full_chunk.ndim >= 2:
                        dim = full_chunk.shape[-1]
                        full_chunk = full_chunk.reshape(-1, dim)
                    else:
                        full_chunk = full_chunk.reshape(1, -1)
                    
                    if full_chunk.size == 0:
                        action = np.array([0.0, 0.0])
                    else:
                        if self.inference_mode == "classification":
                            action = self._decode_classification_action(full_chunk)
                        else:
                            action = full_chunk[0]
                    
                else:
                    logger.warning(f"Unexpected outputs type: {type(outputs)}")
                    action = np.array([0.0, 0.0])
                    full_chunk = np.zeros((1, 2))
                
                hist_len = "N/A"
                try:
                    target_policy = None
                    if hasattr(self.model.model, 'act_head'):
                         target_policy = self.model.model.act_head
                    elif hasattr(self.model, 'act_head'):
                         target_policy = self.model.act_head
                         
                    if target_policy and hasattr(target_policy, 'history_memory'):
                        hist_len = len(target_policy.history_memory)
                except: pass

                logger.info(f"📤 [DETAILED ACTION] InfCount: {self.inference_count}, Hist: {hist_len}/{self.window_size}, Raw: {action}")
                
                if self.inference_mode == "regression":
                    target_min = -1.15
                    target_max = 1.15
                    norm_action = self.config.get('norm_action', False)
                    
                    if not norm_action:
                        if abs(action[0]) <= 1.0 and abs(action[1]) <= 1.0:
                            logger.warning(f"⚠️ Applied auto-scaling to [-1.15, 1.15] (Raw LX: {action[0]:.4f}, LY: {action[1]:.4f})")
                            action = action * 1.15
                    elif norm_action:
                        action = unnoramalize_action(action, action_min=target_min, action_max=target_max)
                        logger.debug("✅ Applied denormalization (norm_action=True)")

                logger.debug(f"📤 [PROCESSED ACTION] After scaling: {action}")

                # 5. Speed Scaling
                speed_scale = 1.0
                if self.speed_scaling_enabled:
                    if bbox is None:
                        speed_scale = 0.7
                    else:
                        area = float(bbox.get("area", 0.0))
                        entity = str(bbox.get("entity", ""))
                        is_fallback = entity in ("coarse_clf", "center_fallback", "") or entity.startswith("caption:")
                        
                        if is_fallback:
                            speed_scale = 1.0
                        elif area > 0.18:
                            speed_scale = 0.25
                        elif area > 0.10:
                            speed_scale = 0.5
                        elif area > 0.05:
                            speed_scale = 0.75
                        else:
                            speed_scale = 1.0
                            
                    action[0] = action[0] * speed_scale
                    if action.size > 1:
                        action[1] = action[1] * speed_scale
                
                # 6. Action Smoothing (EMA)
                action_3d = [action[0], action[1] if action.size > 1 else 0.0, 0.0]
                if action.size > 2:
                    action_3d[2] = action[2]
                
                if self.inference_count > 0:
                    p = self.prev_action_3d
                    action_x = self.smoothing_alpha * action_3d[0] + (1 - self.smoothing_alpha) * p[0]
                    action_y = self.smoothing_alpha * action_3d[1] + (1 - self.smoothing_alpha) * p[1]
                    action_z = self.smoothing_alpha_az * action_3d[2] + (1 - self.smoothing_alpha_az) * p[2]
                    action_3d = [action_x, action_y, action_z]
                
                self.prev_action_3d = list(action_3d)
                
                # Clip to valid range
                action_3d[0] = np.clip(action_3d[0], -1.5, 1.5)
                action_3d[1] = np.clip(action_3d[1], -1.5, 1.5)
                action = np.array([action_3d[0], action_3d[1], action_3d[2]])
                
                logger.info(f"📤 [FINAL ACTION] After clipping: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}]")
                
            except Exception as e:
                logger.error(f"Inference failed: {e}")
                import traceback
                traceback.print_exc()

        latency_ms = (time.time() - start_time) * 1000
        self.inference_count += 1
        
        if logger_instance:
            logger_instance.log_step(self.inference_count, action[:2], latency_ms, full_chunk)
            
        debug_data = {
            "bbox": bbox,
            "grounding_caption": caption,
            "grounding_latency_ms": g_latency,
            "goal_near_proxy": goal_near,
            "speed_scale": speed_scale,
            "grounding_cached": use_cache,
            "action_3d": action_3d,
            "predicted_class": int(np.argmax(outputs['action'][0].cpu().numpy())) if self.inference_mode == "classification" and 'outputs' in locals() and isinstance(outputs, dict) and 'action' in outputs else 0,
            "predicted_label": self.class_labels[int(np.argmax(outputs['action'][0].cpu().numpy()))] if self.inference_mode == "classification" and 'outputs' in locals() and isinstance(outputs, dict) and 'action' in outputs else "REGRESSION",
            "source": "inferred"
        }
            
        return action, latency_ms, full_chunk, debug_data

    def predict_debug(self, image_base64: str, instruction: str) -> dict[str, Any]:
        """
        디버그 추론 실행.
        원본 이미지, 224 입력 이미지, prompt, tokenization, raw chunk, 최종 action을 함께 반환한다.
        """
        start_time = time.time()
        policy_head = self._get_policy_head()
        inference_count_before = int(self.inference_count)
        history_before = int(self._get_history_length(policy_head))
        is_first_frame = inference_count_before == 0
        full_prompt = self._build_prompt(instruction)
        image_debug = self._build_image_debug_payload(image_base64)

        with torch.no_grad():
            image_tensor = self.preprocess_image(image_base64)
            lang_x, attention_mask, token_debug = self._tokenize_instruction_with_debug(full_prompt)
            target_dtype = next(self.model.parameters()).dtype
            image_tensor = image_tensor.to(dtype=target_dtype, device=self.device)

            tensor_cpu = image_tensor.detach().float().cpu()
            image_debug["tensor_stats"] = {
                "shape": list(tensor_cpu.shape),
                "dtype": str(image_tensor.dtype),
                "mean": float(tensor_cpu.mean().item()),
                "std": float(tensor_cpu.std().item()),
                "min": float(tensor_cpu.min().item()),
                "max": float(tensor_cpu.max().item()),
            }

            if is_first_frame:
                self.inference_count += 1
                self.model.model.inference(
                    vision_x=image_tensor,
                    lang_x=lang_x,
                    attention_mask=attention_mask,
                )
                self.image_history = []
                latency_ms = (time.time() - start_time) * 1000.0
                history_after = int(self._get_history_length(self._get_policy_head()))
                return {
                    "action": [0.0, 0.0, 0.0],
                    "latency_ms": latency_ms,
                    "model_name": self.model_name,
                    "source": "first_frame_enforced",
                    "debug": {
                        "instruction": instruction,
                        "full_prompt": full_prompt,
                        "inference_mode": self.inference_mode,
                        "window_size": int(self.window_size),
                        "checkpoint_path": self.checkpoint_path,
                        "config_path": self.config_path,
                        "first_frame_enforced": True,
                        "history": {
                            "inference_count_before": inference_count_before,
                            "inference_count_after": int(self.inference_count),
                            "history_before": history_before,
                            "history_after": history_after,
                        },
                        "image": image_debug,
                        "tokenization": token_debug,
                        "raw_chunk": [[0.0, 0.0, 0.0]],
                        "raw_chunk_shape": [1, 3],
                        "postprocess": {
                            "message": "First frame primed the policy history. Run the same frame once more to inspect the real action output.",
                        },
                    },
                }

            outputs = self.model.model.inference(
                vision_x=image_tensor,
                lang_x=lang_x,
                attention_mask=attention_mask,
            )

            raw_action_type = str(type(outputs))
            raw_chunk = np.zeros((1, 2), dtype=np.float32)
            action = np.array([0.0, 0.0], dtype=np.float32)

            if isinstance(outputs, dict) and 'action' in outputs:
                action_out = outputs['action']
                raw_action_type = str(type(action_out))
                action_out = self._extract_action_tensor(action_out)
                if isinstance(action_out, torch.Tensor):
                    raw_chunk = action_out.detach().float().cpu().numpy()
                else:
                    raw_chunk = np.asarray(action_out, dtype=np.float32)
            else:
                logger.warning(f"Unexpected outputs type in debug mode: {type(outputs)}")

            if raw_chunk.ndim >= 2:
                dim = raw_chunk.shape[-1]
                raw_chunk = raw_chunk.reshape(-1, dim)
            else:
                raw_chunk = raw_chunk.reshape(1, -1)

            postprocess: dict[str, Any] = {
                "raw_output_type": raw_action_type,
                "smoothing_applied": False,
            }

            if raw_chunk.size == 0:
                action = np.array([0.0, 0.0], dtype=np.float32)
            elif self.inference_mode == "classification":
                cls_debug = self._summarize_classification_chunk(raw_chunk)
                postprocess["classification"] = cls_debug
                action = np.array(cls_debug["mapped_action"], dtype=np.float32)
            else:
                action = raw_chunk[0].astype(np.float32)
                postprocess["regression_raw_action"] = action.tolist()

                target_min = -1.15
                target_max = 1.15
                norm_action = self.config.get('norm_action', False)
                if not norm_action:
                    if action.shape[0] >= 2 and abs(action[0]) <= 1.0 and abs(action[1]) <= 1.0:
                        action = action * 1.15
                        postprocess["auto_scaled_to_training_range"] = True
                else:
                    action = unnoramalize_action(
                        action,
                        action_min=target_min,
                        action_max=target_max
                    )
                    postprocess["denormalized"] = True

            action_before_smoothing = action.copy()
            if self.inference_count > 1:
                action = self.smoothing_alpha * action + (1 - self.smoothing_alpha) * self.prev_action
                postprocess["smoothing_applied"] = True
                postprocess["smoothing_alpha"] = float(self.smoothing_alpha)
                postprocess["prev_action"] = self.prev_action.tolist()

            self.prev_action = action.copy()

            if action.shape[0] >= 2:
                action[0] = np.clip(action[0], -1.5, 1.5)
                action[1] = np.clip(action[1], -1.5, 1.5)

            latency_ms = (time.time() - start_time) * 1000.0
            self.inference_count += 1
            history_after = int(self._get_history_length(self._get_policy_head()))

            if logger_instance:
                logger_instance.log_step(self.inference_count, action, latency_ms, raw_chunk)

            postprocess["action_before_smoothing"] = action_before_smoothing.tolist()
            postprocess["final_action"] = action.tolist()

            return {
                "action": action.tolist(),
                "latency_ms": latency_ms,
                "model_name": self.model_name,
                "source": "inferred",
                "debug": {
                    "instruction": instruction,
                    "full_prompt": full_prompt,
                    "inference_mode": self.inference_mode,
                    "window_size": int(self.window_size),
                    "checkpoint_path": self.checkpoint_path,
                    "config_path": self.config_path,
                    "first_frame_enforced": False,
                    "history": {
                        "inference_count_before": inference_count_before,
                        "inference_count_after": int(self.inference_count),
                        "history_before": history_before,
                        "history_after": history_after,
                    },
                    "image": image_debug,
                    "tokenization": token_debug,
                    "raw_chunk": raw_chunk.tolist(),
                    "raw_chunk_shape": list(raw_chunk.shape),
                    "postprocess": postprocess,
                },
            }


# ══════════════════════════════════════════════════════════════════════════════
# Exp47: InstructionMLPInference  (bbox + vision_cache + instr_emb → 8-class)
# ══════════════════════════════════════════════════════════════════════════════

class InstructionMLPInference:
    """
    Exp47 instruction-conditioned MLP 추론기.
    KosMos-2 VLM 없이 bbox 히스토리 + 사전 캐시된 vision feature + instruction embedding으로
    8-class 행동 예측. 평균 추론 레이턴시 < 5ms.

    성능 (offline eval):
      val accuracy : 98.7%
      closed-loop  : 100% success (30/30), PM 99.2%, FPE 0.013m
      sensitivity  : 8/10 (80%)  — instruction 교체 시 action 변화 확인
    """

    # 8-class 액션 매핑 (Exp47 기준 nav_h5_dataset_impl.py 동기화)
    CLASS_NAMES = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "TURN_L", "TURN_R"]
    _SPEED       = 1.15
    _ANGLE       = 0.5
    _DIAG        = _SPEED * 0.707
    CLASS_ACTIONS = {
        0: [0.0,    0.0,    0.0],
        1: [_SPEED, 0.0,    0.0],
        2: [0.0,    _SPEED, 0.0],
        3: [0.0,   -_SPEED, 0.0],
        4: [_DIAG,  _DIAG,  0.0],
        5: [_DIAG, -_DIAG,  0.0],
        6: [0.0,    0.0,    _ANGLE],
        7: [0.0,    0.0,   -_ANGLE],
    }
    NUM_CLASSES = 8
    WINDOW      = 8
    VIS_DIM     = 1024
    INSTR_DIM   = 2048
    D_IN        = WINDOW * 4 + VIS_DIM + INSTR_DIM  # 3104

    def __init__(
        self,
        mlp_weights_path: str,
        instruction_embeddings_path: str,
        device: str = "cuda",
        vision_cache_ttl_sec: float = 1.0,
    ):
        import torch.nn as nn
        from collections import deque

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._weights_path = mlp_weights_path
        self._emb_path = instruction_embeddings_path
        self.vision_cache_ttl_sec = vision_cache_ttl_sec

        # bbox 히스토리 버퍼 (deque, 8프레임 × [cx, cy, area, has_bbox])
        self._bbox_history = deque(
            [[0.0, 0.0, 0.0, 0.0]] * self.WINDOW,
            maxlen=self.WINDOW
        )

        # vision feature 캐시
        self._vision_cache: dict = {
            "feature": np.zeros(self.VIS_DIM, dtype=np.float32),
            "last_update_time": 0.0,
            "initialized": False,
        }

        # instruction → embedding lookup table
        self._instr_embeddings: dict[str, np.ndarray] = {}
        self._instr_map: dict[str, str] = {}  # path_type → instruction text
        self._load_instruction_embeddings()

        # MLP 모델
        self._net = self._build_mlp()
        self._load_weights()
        self._net.eval()
        self._net.to(self.device)

        logger.info(
            "✅ [InstructionMLPInference] loaded — device=%s, cache_ttl=%.1fs, instr_keys=%d",
            self.device, self.vision_cache_ttl_sec, len(self._instr_embeddings)
        )

    # ── 내부 초기화 헬퍼 ────────────────────────────────────────────────────
    def _build_mlp(self):
        import torch.nn as nn
        return nn.Sequential(
            nn.Linear(self.D_IN, 512), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(512, 256),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),  nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64),   nn.ReLU(),
            nn.Linear(64, self.NUM_CLASSES),
        )

    def _load_weights(self):
        """exp47_mlp.pt 로드. 파일 없으면 경고 후 random weights 사용."""
        path = Path(self._weights_path)
        if not path.exists():
            logger.warning(
                "⚠️ [InstructionMLPInference] 가중치 파일 없음: %s — random weights 사용 (학습 후 재시작 필요)",
                path
            )
            return
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        # checkpoint에 d_in이 다르면 재빌드
        ckpt_d_in = ckpt.get("d_in", self.D_IN)
        if ckpt_d_in != self.D_IN:
            logger.warning("⚠️ d_in mismatch: ckpt=%d, expected=%d — rebuilding MLP", ckpt_d_in, self.D_IN)
            import torch.nn as nn
            self._net = self._build_mlp()
        self._net.load_state_dict(ckpt["model_state_dict"])
        # instr_map 동기화
        if "instr_map" in ckpt:
            self._instr_map = ckpt["instr_map"]
        logger.info("✅ [InstructionMLPInference] 가중치 로드 완료: %s", path)

    def _load_instruction_embeddings(self):
        """instruction_embeddings.json → {path_type: np.ndarray(2048)} 로드."""
        emb_path = Path(self._emb_path)
        if not emb_path.exists():
            logger.error("❌ instruction_embeddings.json 없음: %s", emb_path)
            return
        with open(emb_path) as f:
            raw = json.load(f)
        self._instr_embeddings = {
            k: np.array(v, dtype=np.float32) for k, v in raw.items()
        }
        logger.info("✅ instruction embeddings 로드: %d path_types", len(self._instr_embeddings))

    # ── 공개 인터페이스 ─────────────────────────────────────────────────────
    def update_bbox(self, cx: float, cy: float, area: float, has_bbox: bool) -> None:
        """bbox 히스토리 업데이트 (매 프레임 호출)."""
        self._bbox_history.append([float(cx), float(cy), float(area), float(has_bbox)])

    def update_vision_feature(self, feature: np.ndarray) -> None:
        """vision feature 캐시 갱신 (VLM encoder 출력, 1024-dim)."""
        self._vision_cache["feature"] = np.asarray(feature, dtype=np.float32).reshape(self.VIS_DIM)
        self._vision_cache["last_update_time"] = time.time()
        self._vision_cache["initialized"] = True

    def is_vision_cache_stale(self) -> bool:
        """vision feature 캐시가 TTL을 초과했으면 True."""
        if not self._vision_cache["initialized"]:
            return True
        return (time.time() - self._vision_cache["last_update_time"]) > self.vision_cache_ttl_sec

    def vision_cache_age_ms(self) -> float:
        """마지막 vision feature 갱신 후 경과 ms."""
        return (time.time() - self._vision_cache["last_update_time"]) * 1000.0

    def _infer_path_type_from_bbox(self) -> str:
        """bbox 히스토리 cx 기반으로 방향 추론 (has_bbox=True 마지막 프레임 사용)."""
        cx = 0.5
        for frame in reversed(list(self._bbox_history)):
            if frame[3] > 0.5:  # has_bbox
                cx = frame[0]
                break
        if cx > 0.65:
            return "right_right"
        if cx < 0.35:
            return "left_left"
        return "center_straight"

    def _match_instruction(self, instruction_text: str) -> tuple[str, np.ndarray]:
        """
        instruction text → (matched_path_type, embedding 2048-dim).
        우선순위: exact path_type → exact text → substring → bbox cx 자동추론 → word overlap
        """
        text = instruction_text.strip()

        # 1. exact path_type match
        if text in self._instr_embeddings:
            return text, self._instr_embeddings[text]

        # 2. instruction text 역방향 매핑 (instr_map 값과 일치)
        for pt, instr in self._instr_map.items():
            if text.lower() == instr.lower():
                if pt in self._instr_embeddings:
                    return pt, self._instr_embeddings[pt]

        # 3. substring match on path_type key
        for pt, emb in self._instr_embeddings.items():
            if pt in text or text in pt:
                return pt, emb

        # 4. bbox cx 기반 자동 추론 — text 매칭 실패 시 위치로 결정
        auto_pt = self._infer_path_type_from_bbox()
        if auto_pt in self._instr_embeddings:
            logger.info("[MLP] instruction unrecognized → bbox auto: '%s' → '%s'", text, auto_pt)
            return auto_pt, self._instr_embeddings[auto_pt]

        # 5. word overlap (최후 수단)
        best_pt, best_sim, best_emb = "center_straight", -1.0, None
        for pt, emb in self._instr_embeddings.items():
            overlap = sum(w in text.lower() for w in pt.split("_"))
            if overlap > best_sim:
                best_sim = overlap
                best_pt = pt
                best_emb = emb

        if best_emb is None:
            best_emb = self._instr_embeddings.get("center_straight",
                        np.zeros(self.INSTR_DIM, dtype=np.float32))
        logger.info("[MLP] instruction fallback (word overlap): '%s' → '%s'", text, best_pt)
        return best_pt, best_emb

    def predict(self, instruction_text: str) -> dict:
        """
        Exp47 MLP 추론.
        Returns: {action, class_idx, class_name, latency_ms, vision_cache_age_ms, instruction_matched}
        """
        t0 = time.time()

        # bbox 히스토리 feature (8×4 = 32)
        bbox_feat = np.array(list(self._bbox_history), dtype=np.float32).flatten()

        # vision feature (1024)
        vis_feat = self._vision_cache["feature"]

        # instruction embedding (2048)
        matched_pt, instr_emb = self._match_instruction(instruction_text)

        # concat → 3104
        x = np.concatenate([bbox_feat, vis_feat, instr_emb])
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            logits = self._net(x_tensor)  # (1, 8)
            class_idx = int(logits.argmax(1).item())

        class_idx = min(class_idx, self.NUM_CLASSES - 1)
        class_name = self.CLASS_NAMES[class_idx]
        action_3dof = self.CLASS_ACTIONS[class_idx]
        action_2dof = [action_3dof[0], action_3dof[1]]  # [lx, ly] (az 제외)

        latency_ms = (time.time() - t0) * 1000.0
        logger.info(
            "✅ [MLP] cls=%d(%s), action=%s, latency=%.1fms, vis_age=%.0fms, instr=%s",
            class_idx, class_name, action_2dof, latency_ms, self.vision_cache_age_ms(), matched_pt
        )
        return {
            "action": action_2dof,
            "class_idx": class_idx,
            "class_name": class_name,
            "latency_ms": latency_ms,
            "vision_cache_age_ms": self.vision_cache_age_ms(),
            "instruction_matched": matched_pt,
        }

    def reset(self) -> None:
        """bbox 히스토리 및 vision 캐시 초기화."""
        from collections import deque
        self._bbox_history = deque(
            [[0.0, 0.0, 0.0, 0.0]] * self.WINDOW,
            maxlen=self.WINDOW
        )
        self._vision_cache["feature"] = np.zeros(self.VIS_DIM, dtype=np.float32)
        self._vision_cache["last_update_time"] = 0.0
        self._vision_cache["initialized"] = False
        logger.info("🔄 [InstructionMLPInference] reset")

    def reload_weights(self) -> None:
        """가중치 파일 재로드 (학습 완료 후 재시작 없이 갱신)."""
        self._load_weights()
        self._net.eval()
        self._net.to(self.device)
        logger.info("✅ [InstructionMLPInference] 가중치 재로드 완료")


def get_mlp_model() -> InstructionMLPInference:
    """Exp47 MLP 인스턴스 lazy loading (VLM과 독립적)."""
    global mlp_instance
    if mlp_instance is None:
        weights_path = os.getenv(
            "VLA_MLP_WEIGHTS_PATH",
            "docs/v5/bbox_nav_exp47/exp47_mlp.pt"
        )
        emb_path = os.getenv(
            "VLA_MLP_INSTR_EMBEDDINGS_PATH",
            "docs/v5/bbox_nav_exp47/instruction_embeddings.json"
        )
        cache_ttl = float(os.getenv("VLA_MLP_VISION_CACHE_TTL", "1.0"))
        mlp_instance = InstructionMLPInference(
            mlp_weights_path=str(Path(project_root) / weights_path),
            instruction_embeddings_path=str(Path(project_root) / emb_path),
            device="cuda" if torch.cuda.is_available() else "cpu",
            vision_cache_ttl_sec=cache_ttl,
        )
    return mlp_instance


# ══════════════════════════════════════════════════════════════════════════════
# GoalNav MLP (exp49 / exp54_s2v2 / exp55) — Pure Kosmos-2 vision encoder
# ══════════════════════════════════════════════════════════════════════════════

def _get_pure_vision_model(device: str = "cuda"):
    """Pure HF Kosmos-2 vision_model 싱글톤 로더 (Google-robot backbone 아님)."""
    global _pure_vision_model, _pure_processor
    if _pure_vision_model is None:
        from transformers import AutoModelForVision2Seq, AutoProcessor
        vlm_path = str(Path(project_root) / ".vlms" / "kosmos-2-patch14-224")
        logger.info("🔄 [GoalNav] Pure Kosmos-2 vision_model 로드 중: %s", vlm_path)
        full_model = AutoModelForVision2Seq.from_pretrained(
            vlm_path,
            torch_dtype=torch.float16,
            device_map=None,
        )
        _pure_vision_model = full_model.model.vision_model.to(device).eval()
        _pure_processor = AutoProcessor.from_pretrained(vlm_path)
        logger.info("✅ [GoalNav] Pure Kosmos-2 vision_model 로드 완료")
    return _pure_vision_model, _pure_processor


class GoalNavMLPInference:
    """
    Pure Kosmos-2 vision encoder + lightweight MLP action predictor.

    variant:
      "exp49"      → D_IN=1056 (bbox_32 + vis_1024), ckpt key "model_state_dict"
      "exp54_s2v2" → D_IN=288  (bbox_32 + proj_256), needs stage1 image_proj
      "exp55"      → D_IN=288  (same as exp54_s2v2 but separate ckpt)

    Default ckpt paths (override with env vars):
      VLA_GOALNAV_EXP49_CKPT
      VLA_GOALNAV_STAGE1_CKPT  (stage1_v2_projs.pt — shared by exp54_s2v2 and exp55)
      VLA_GOALNAV_STAGE2_CKPT  (stage2_v2_mlp.pt for exp54_s2v2 or exp55_mlp.pt for exp55)
    """

    CLASS_NAMES = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "ROT_L", "ROT_R"]
    CLASS_ACTIONS = {
        0: {"linear_x": 0.0, "linear_y":  0.0, "angular_z":  0.0},
        1: {"linear_x": 0.3, "linear_y":  0.0, "angular_z":  0.0},
        2: {"linear_x": 0.0, "linear_y":  0.3, "angular_z":  0.0},
        3: {"linear_x": 0.0, "linear_y": -0.3, "angular_z":  0.0},
        4: {"linear_x": 0.3, "linear_y":  0.3, "angular_z":  0.0},
        5: {"linear_x": 0.3, "linear_y": -0.3, "angular_z":  0.0},
        6: {"linear_x": 0.0, "linear_y":  0.0, "angular_z":  0.5},
        7: {"linear_x": 0.0, "linear_y":  0.0, "angular_z": -0.5},
    }
    NUM_CLASSES = 8
    WINDOW = 8
    VIS_DIM = 1024
    PROJ_DIM = 256

    _DEFAULT_CKPTS = {
        "exp49": {
            "mlp": "runs/v5_nav/mlp/exp49/exp49_mlp.pt",
        },
        "exp54_s2v2": {
            "stage1": "runs/v5_nav/mlp/exp54/stage1_v2/stage1_v2_projs.pt",
            "mlp":    "runs/v5_nav/mlp/exp54/stage2_v2/stage2_v2_mlp.pt",
        },
        "exp55": {
            "stage1": "runs/v5_nav/mlp/exp54/stage1_v2/stage1_v2_projs.pt",
            "mlp":    "runs/v5_nav/mlp/exp55/exp55_mlp.pt",
        },
    }

    def __init__(self, variant: str = "exp54_s2v2", device: str = "cuda"):
        assert variant in self._DEFAULT_CKPTS, f"Unknown variant: {variant}"
        self.variant = variant
        self.device = device if torch.cuda.is_available() else "cpu"

        self._d_in = self.WINDOW * 4 + (self.VIS_DIM if variant == "exp49" else self.PROJ_DIM)
        self._mlp = self._build_mlp().to(self.device)
        self._image_proj = None  # only for exp54_s2v2 / exp55

        self._load_weights()

        self._bbox_history: list = []
        self._vis_feat_cache: torch.Tensor | None = None

        # ── 도착 STOP 규칙 (plan_20260602_stop_arrival_rule.md) ──
        # area_det 최근 W프레임 평균 > TH_AREA AND |cx-0.5| < TH_CX → STOP(래치).
        # 캘리브레이션 추천값 (243ep PG2): th_area=0.5, th_cx=0.3, W=5, min_steps=0.
        self._stop_enabled   = os.getenv("VLA_GOALNAV_STOP_RULE", "1") != "0"
        self._stop_th_area   = float(os.getenv("VLA_GOALNAV_STOP_TH_AREA", "0.5"))
        self._stop_th_cx     = float(os.getenv("VLA_GOALNAV_STOP_TH_CX", "0.3"))
        self._stop_th_cy     = float(os.getenv("VLA_GOALNAV_STOP_TH_CY", "0.5"))
        self._stop_w         = int(os.getenv("VLA_GOALNAV_STOP_W", "5"))
        self._stop_min_steps = int(os.getenv("VLA_GOALNAV_STOP_MIN_STEPS", "0"))

        # ── 학습 STOP 윈도우 스무딩 규칙 (eval_learned_stop_window.py 결과 반영) ──
        # 모델의 STOP 클래스 Softmax 확률의 W프레임 평균 > TH_PROB → STOP(래치).
        # 최적 추천값: stop_learned_w=3, stop_learned_th=0.8
        self._stop_learned_enabled = os.getenv("VLA_GOALNAV_STOP_LEARNED", "0") != "0"
        self._stop_learned_w       = int(os.getenv("VLA_GOALNAV_STOP_LEARNED_W", "3"))
        self._stop_learned_th      = float(os.getenv("VLA_GOALNAV_STOP_LEARNED_TH", "0.8"))
        self._stop_prob_history: list = [0.0] * self._stop_learned_w

        self._stopped = False
        self._step = 0

        logger.info("✅ [GoalNavMLP] variant=%s D_IN=%d device=%s stop_rule=%s stop_learned=%s",
                    variant, self._d_in, self.device, self._stop_enabled, self._stop_learned_enabled)

    def _build_mlp(self) -> torch.nn.Module:
        d = self._d_in
        if self.variant == "exp49":
            return torch.nn.Sequential(
                torch.nn.Linear(d, 512),  torch.nn.ReLU(), torch.nn.Dropout(0.25),
                torch.nn.Linear(512, 256), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                torch.nn.Linear(128, 64),  torch.nn.ReLU(),
                torch.nn.Linear(64, self.NUM_CLASSES),
            )
        else:  # exp54_s2v2 / exp55
            return torch.nn.Sequential(
                torch.nn.Linear(d, 256),  torch.nn.ReLU(), torch.nn.Dropout(0.25),
                torch.nn.Linear(256, 128), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                torch.nn.Linear(128, 64),  torch.nn.ReLU(), torch.nn.Dropout(0.1),
                torch.nn.Linear(64, self.NUM_CLASSES),
            )

    def _resolve_path(self, env_var: str, default_rel: str) -> str:
        rel = os.getenv(env_var, default_rel)
        return str(Path(project_root) / rel)

    def _load_weights(self):
        defaults = self._DEFAULT_CKPTS[self.variant]

        if self.variant in ("exp54_s2v2", "exp55"):
            stage1_path = self._resolve_path("VLA_GOALNAV_STAGE1_CKPT", defaults["stage1"])
            s1_ckpt = torch.load(stage1_path, map_location="cpu")
            self._image_proj = torch.nn.Linear(self.VIS_DIM, self.PROJ_DIM)
            self._image_proj.load_state_dict(s1_ckpt["image_proj"])
            self._image_proj = self._image_proj.to(self.device).eval()
            logger.info("✅ [GoalNavMLP] stage1 image_proj 로드: %s", stage1_path)

        mlp_path = self._resolve_path("VLA_GOALNAV_STAGE2_CKPT", defaults["mlp"])
        ckpt = torch.load(mlp_path, map_location="cpu")

        if self.variant == "exp49":
            sd = ckpt["model_state_dict"]
        else:
            sd = ckpt["mlp"]
        # 학습 시 self.net = nn.Sequential(...) 래핑 → "net." prefix 제거
        if any(k.startswith("net.") for k in sd):
            sd = {k[len("net."):]: v for k, v in sd.items()}
        self._mlp.load_state_dict(sd)

        self._mlp.eval()
        self._weights_path = mlp_path
        logger.info("✅ [GoalNavMLP] MLP 가중치 로드: %s", mlp_path)

    def update_bbox(self, cx: float, cy: float, area: float, has_bbox: bool):
        self._bbox_history.append([cx, cy, area, float(has_bbox)])
        if len(self._bbox_history) > self.WINDOW:
            self._bbox_history = self._bbox_history[-self.WINDOW:]

    def update_vision_feature(self, image_b64: str):
        """base64 이미지 → Pure Kosmos-2 → vis_feat 캐시 갱신."""
        import base64, io
        from PIL import Image
        img_bytes = base64.b64decode(image_b64)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        vision_model, processor = _get_pure_vision_model(self.device)
        inputs = processor(images=pil_img, return_tensors="pt")
        pv = inputs["pixel_values"].to(self.device, dtype=torch.float16)

        with torch.no_grad():
            out = vision_model(pixel_values=pv)
            feat = out.last_hidden_state.mean(dim=1).float()  # (1, 1024)

        if self._image_proj is not None:
            import torch.nn.functional as F
            feat = F.normalize(self._image_proj(feat), dim=-1)  # (1, 256)

        self._vis_feat_cache = feat

    def _arrival_stop(self) -> bool:
        """도착 area 규칙 (래치). True면 STOP(0) override.
        한 번 트리거되면 에피소드 끝까지 STOP 유지 (reset()으로 해제)."""
        if not self._stop_enabled:
            return False
        if self._stopped:
            return True
        if not self._bbox_history:
            return False
        recent = self._bbox_history[-self._stop_w:]
        area_avg = sum(b[2] for b in recent) / len(recent)
        cy_avg = sum(b[1] for b in recent) / len(recent)
        cx = self._bbox_history[-1][0]
        if (self._step >= self._stop_min_steps
                and area_avg > self._stop_th_area
                and abs(cx - 0.5) < self._stop_th_cx
                and cy_avg > self._stop_th_cy):
            self._stopped = True
            logger.info("🛑 [GoalNavMLP] 도착 STOP 트리거 (area_avg=%.3f cy_avg=%.3f cx=%.3f step=%d)",
                        area_avg, cy_avg, cx, self._step)
            return True
        return False

    def predict(self) -> dict:
        import time
        t0 = time.perf_counter()

        if self._vis_feat_cache is None:
            raise RuntimeError("vision feature가 없습니다. update_vision_feature()를 먼저 호출하세요.")

        if self._stopped:
            cls_idx = 0
        else:
            # bbox window 패딩 (부족하면 0 패딩)
            history = self._bbox_history[-self.WINDOW:]
            while len(history) < self.WINDOW:
                history = [[0.0, 0.0, 0.0, 0.0]] + history
            bbox_feat = torch.tensor(history, dtype=torch.float32).flatten().unsqueeze(0).to(self.device)  # (1,32)

            x = torch.cat([bbox_feat, self._vis_feat_cache], dim=-1)  # (1, D_IN)

            with torch.no_grad():
                logits = self._mlp(x)
                cls_idx = int(logits.argmax(dim=-1).item())
                probs = torch.nn.functional.softmax(logits, dim=-1)[0]
                p_stop = float(probs[0].item())

            # 1. 학습형 STOP 윈도우 스무딩
            if self._stop_learned_enabled:
                self._stop_prob_history.append(p_stop)
                self._stop_prob_history = self._stop_prob_history[-self._stop_learned_w:]
                mean_p_stop = sum(self._stop_prob_history) / self._stop_learned_w
                if mean_p_stop > self._stop_learned_th:
                    self._stopped = True
                    logger.info("🛑 [GoalNavMLP] 학습형 STOP 트리거 (mean_p_stop=%.3f)", mean_p_stop)
                    cls_idx = 0

            # 2. Heuristic Area 규칙 STOP
            if not self._stopped and self._stop_enabled:
                if self._arrival_stop():
                    cls_idx = 0

        self._step += 1

        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "action":      self.CLASS_ACTIONS[cls_idx],
            "class_idx":   cls_idx,
            "class_name":  self.CLASS_NAMES[cls_idx],
            "latency_ms":  round(latency_ms, 2),
            "variant":     self.variant,
        }

    def reset(self):
        self._bbox_history.clear()
        self._vis_feat_cache = None
        self._stopped = False
        self._step = 0
        self._stop_prob_history = [0.0] * self._stop_learned_w


def get_goalnav_model() -> GoalNavMLPInference:
    """GoalNav MLP 인스턴스 lazy loading."""
    global goalnav_instance
    if goalnav_instance is None:
        variant = os.getenv("VLA_GOALNAV_VARIANT", "exp54_s2v2")
        goalnav_instance = GoalNavMLPInference(
            variant=variant,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    return goalnav_instance


def get_model(refresh=False, use_quant=None, checkpoint_path=None, config_path=None):
    """
    모델 인스턴스 가져오기 (lazy loading)
    Args:
        refresh: Force reload model
        use_quant: Override quantization setting
    """
    global model_instance, model_override_checkpoint_path, model_override_config_path

    if checkpoint_path and config_path:
        model_override_checkpoint_path = str(Path(checkpoint_path).expanduser().resolve())
        model_override_config_path = str(Path(config_path).expanduser().resolve())
    
    if refresh:
        if model_instance:
            del model_instance
            torch.cuda.empty_cache()
            model_instance = None
            logger.info("🔄 Model unloaded for refresh")
    
    if model_instance is None:
        if model_override_checkpoint_path and model_override_config_path:
            checkpoint_path = model_override_checkpoint_path
            config_path = model_override_config_path
        else:
            checkpoint_path, config_path = _resolve_default_model_paths()        
        model_instance = MobileVLAInference(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device="cuda" if torch.cuda.is_available() else "cpu",
            use_quant=use_quant
        )
    
    return model_instance


@app.get("/")
async def root():
    """API 정보 (인증 불필요)"""
    return {
        "name": "Mobile VLA Inference API",
        "version": "1.0.0",
        "status": "running",
        "auth": "API Key required (X-API-Key header)"
    }


@app.get("/debug-ui")
async def debug_ui():
    """브라우저에서 중간 추론 과정을 확인하는 디버그 페이지"""
    if not DEBUG_UI_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Debug UI not found: {DEBUG_UI_PATH}")
    return FileResponse(DEBUG_UI_PATH)


@app.get("/debug/roots")
async def debug_roots():
    """디버그 UI용 서버 이미지 루트 목록"""
    roots = _get_allowed_debug_roots()
    return {
        "roots": [
            {"name": name, "path": str(path)}
            for name, path in roots.items()
        ]
    }


@app.get("/debug/files")
async def debug_files(path: Optional[str] = None):
    """디버그 UI용 서버 디렉토리 브라우저"""
    roots = _get_allowed_debug_roots()
    if not roots:
        raise HTTPException(status_code=500, detail="No debug roots are available.")

    target = _resolve_debug_path(path) if path else next(iter(roots.values()))
    return _list_debug_directory(target)


@app.get("/debug/path_context")
async def debug_path_context(path: str):
    """현재 이미지 기준 이전/다음 프레임 문맥"""
    image_path = _resolve_debug_path(path)
    return _build_image_path_context(image_path)


@app.get("/debug/image")
async def debug_image(path: str, resize: Optional[int] = None):
    """디버그 UI용 서버 로컬 이미지 미리보기"""
    image_path = _resolve_debug_path(path)
    if not image_path.exists():
        raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")
    if not image_path.is_file():
        raise HTTPException(status_code=400, detail=f"Not a file: {image_path}")

    if not resize:
        return FileResponse(image_path)

    try:
        image = Image.open(image_path).convert("RGB")
        image = image.resize((int(resize), int(resize)), Image.BICUBIC)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return Response(content=buffer.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to render preview: {e}")


@app.get("/debug/models")
async def debug_models():
    """디버그 UI용 모델 후보 및 현재 로드 상태"""
    current = None
    if model_instance is not None:
        current = {
            "model_name": getattr(model_instance, "model_name", None),
            "checkpoint_path": getattr(model_instance, "checkpoint_path", None),
            "config_path": getattr(model_instance, "config_path", None),
        }
    else:
        try:
            ckpt, cfg = _resolve_default_model_paths()
            current = {
                "model_name": Path(cfg).stem,
                "checkpoint_path": ckpt,
                "config_path": cfg,
            }
        except Exception:
            current = None

    return {
        "current": current,
        "candidates": _get_model_candidates(),
    }


@app.post("/debug/model/reload")
async def debug_model_reload(request: DebugModelReloadRequest):
    """디버그 UI에서 모델 재로드"""
    ckpt, cfg = _resolve_model_selection(
        candidate_name=request.candidate_name,
        checkpoint_path=request.checkpoint_path,
        config_path=request.config_path,
    )
    model = get_model(refresh=True, checkpoint_path=ckpt, config_path=cfg)
    return {
        "status": "success",
        "model_name": model.model_name,
        "checkpoint_path": model.checkpoint_path,
        "config_path": model.config_path,
    }

@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    """로컬 브라우저용 간단한 관리 페이지"""
    api_key = VALID_API_KEY
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>MoNaVLA Server Admin</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #f6f4ee; color: #1f2937; }}
    h1 {{ margin-bottom: 8px; }}
    .row {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 12px 0; }}
    button {{ padding: 10px 14px; border: 0; border-radius: 10px; background: #1f6feb; color: white; cursor: pointer; }}
    button.secondary {{ background: #4b5563; }}
    pre {{ background: white; padding: 16px; border-radius: 12px; overflow: auto; border: 1px solid #d1d5db; }}
    input {{ width: 100%; padding: 10px; border-radius: 10px; border: 1px solid #cbd5e1; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
  </style>
</head>
<body>
  <h1>MoNaVLA Server Admin</h1>
  <p>Local management for the running FastAPI server.</p>
  <div class="row">
    <button onclick="loadHealth()">Refresh Health</button>
    <button onclick="loadInfo()">Refresh Model Info</button>
    <button class="secondary" onclick="resetHistory()">Reset History</button>
  </div>
  <div class="grid">
    <div>
      <p>Checkpoint path</p>
      <input id="ckpt" value="{os.getenv('VLA_CHECKPOINT_PATH', '')}">
    </div>
    <div>
      <p>Config path</p>
      <input id="cfg" value="{os.getenv('VLA_CONFIG_PATH', '')}">
    </div>
  </div>
  <div class="row">
    <button onclick="loadSelected('fp16')">Load FP16</button>
    <button class="secondary" onclick="loadSelected('int8')">Load INT8</button>
  </div>
  <h2>Response</h2>
  <pre id="out">Ready</pre>
  <script>
    const headers = {{
      "Content-Type": "application/json",
      "X-API-Key": {json.dumps(api_key)}
    }};
    async function show(resp) {{
      const text = await resp.text();
      try {{
        document.getElementById("out").textContent = JSON.stringify(JSON.parse(text), null, 2);
      }} catch (_e) {{
        document.getElementById("out").textContent = text;
      }}
    }}
    async function loadHealth() {{
      await show(await fetch('/health'));
    }}
    async function loadInfo() {{
      await show(await fetch('/model/info', {{ headers }}));
    }}
    async function resetHistory() {{
      await show(await fetch('/reset', {{ method: 'POST', headers }}));
    }}
    async function loadSelected(precision) {{
      const body = {{
        checkpoint_path: document.getElementById('ckpt').value,
        config_path: document.getElementById('cfg').value,
        precision,
        refresh: true
      }};
      await show(await fetch('/model/load', {{ method: 'POST', headers, body: JSON.stringify(body) }}));
    }}
    loadHealth();
  </script>
</body>
</html>"""


@app.get("/health")
async def health_check():
    """헬스 체크 (인증 불필요)"""
    gpu_memory = None
    model_name = "Unknown"
    
    if torch.cuda.is_available():
        gpu_memory = {
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "device_name": torch.cuda.get_device_name(0)
        }
        
    if model_instance:
        try:
            config_path = model_instance.config_path
            model_name = os.path.basename(config_path).replace(".json", "")
        except:
            model_name = "Loaded"
    
    return {
        "status": "healthy",
        "model_loaded": model_instance is not None,
        "model_name": model_name,
        "checkpoint_path": getattr(model_instance, "checkpoint_path", None) if model_instance else None,
        "config_path": getattr(model_instance, "config_path", None) if model_instance else None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_memory": gpu_memory
    }


@app.get("/model/info")
async def model_info(api_key: str = Depends(verify_api_key)):
    """현재 로드된 모델 정보 조회"""
    if model_instance is None:
        return {
            "model_loaded": False,
            "model_name": "Unavailable",
            "checkpoint_path": "N/A",
            "config_path": "N/A",
            "precision": "int8" if os.getenv("VLA_QUANTIZE", "false").lower() == "true" else "fp16",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "action_dim": 3,
        }

    return {
        "model_loaded": True,
        "model_name": model_instance.model_name,
        "checkpoint_path": model_instance.checkpoint_path,
        "config_path": model_instance.config_path,
        "precision": "int8" if model_instance.use_quant else "fp16",
        "device": model_instance.device,
        "action_dim": 3,
    }


@app.post("/model/load")
async def load_model(request: ModelLoadRequest, api_key: str = Depends(verify_api_key)):
    """런타임에 모델 로드/교체"""
    try:
        os.environ["VLA_CHECKPOINT_PATH"] = request.checkpoint_path
        os.environ["VLA_CONFIG_PATH"] = request.config_path
        os.environ["VLA_QUANTIZE"] = "true" if request.precision == "int8" else "false"

        model = get_model(
            refresh=request.refresh,
            use_quant=(request.precision == "int8"),
            checkpoint_path=request.checkpoint_path,
            config_path=request.config_path,
        )

        return {
            "status": "success",
            "message": "Model loaded",
            "model_name": model.model_name,
            "checkpoint_path": model.checkpoint_path,
            "config_path": model.config_path,
            "precision": request.precision,
            "device": model.device,
        }
    except Exception as e:
        logger.error(f"Model load failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest, api_key: str = Depends(verify_api_key)):
    """추론 엔드포인트"""
    try:
        model = get_model()
        
        action, latency_ms, chunk, debug_data = model.predict(
            image_base64=request.image,
            instruction=request.instruction
        )
        
        logger.info(f"✅ Prediction: {action}, Latency: {latency_ms:.1f}ms")

        # monapi 모드일 때 continuous actions 스냅핑 및 Yaw 복원 알고리즘 적용
        if request.vlm_model == "monapi":
            vx, vy, wz = float(action[0]), float(action[1]), float(action[2])
            cls_idx, label = classify_action(vx, vy, wz)
            snapped_action_3d = snap_monapi_action_to_label(action, label)
            
            # 2D action은 snapped_action_3d[:2]로 할당
            action_2d = snapped_action_3d[:2]
            
            # debug_data의 action_3d 및 predicted_label, predicted_class를 snapped 값으로 갱신
            debug_data["action_3d"] = snapped_action_3d.tolist()
            debug_data["predicted_class"] = cls_idx
            debug_data["predicted_label"] = label
            
            return InferenceResponse(
                action=action_2d.tolist(),
                latency_ms=latency_ms,
                model_name=model.model_name,
                strategy="receding_horizon",
                source=debug_data.get("source", "inferred"),
                buffer_status={},
                predicted_class=debug_data.get("predicted_class"),
                predicted_label=debug_data.get("predicted_label"),
                action_3d=debug_data.get("action_3d"),
                bbox=debug_data.get("bbox"),
                grounding_caption=debug_data.get("grounding_caption"),
                grounding_latency_ms=debug_data.get("grounding_latency_ms"),
                goal_near_proxy=debug_data.get("goal_near_proxy"),
                instruction_used=True,
                matched_path_type=None
            )
        
        # 일반 모드인 경우
        return InferenceResponse(
            action=action[:2].tolist(),
            latency_ms=latency_ms,
            model_name=model.model_name,
            strategy="receding_horizon",
            source=debug_data.get("source", "inferred"),
            buffer_status={},
            predicted_class=debug_data.get("predicted_class"),
            predicted_label=debug_data.get("predicted_label"),
            action_3d=debug_data.get("action_3d"),
            bbox=debug_data.get("bbox"),
            grounding_caption=debug_data.get("grounding_caption"),
            grounding_latency_ms=debug_data.get("grounding_latency_ms"),
            goal_near_proxy=debug_data.get("goal_near_proxy"),
            instruction_used=True,
            matched_path_type=None
        )
        
    except Exception as e:
        import traceback
        logger.error(f"Prediction failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/config")
async def update_config(request: ConfigRequest, api_key: str = Depends(verify_api_key)):
    """런타임 제어 및 자동 정지 설정을 동적으로 업데이트합니다."""
    model = get_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if request.speed_scaling is not None:
        model.speed_scaling_enabled = request.speed_scaling
    if request.grounding_skip_n is not None:
        model.grounding_skip_n = request.grounding_skip_n
    if request.smooth_enabled is not None:
        model.smooth_enabled = request.smooth_enabled
    if request.smooth_alpha_xy is not None:
        model.smoothing_alpha = request.smooth_alpha_xy
    if request.smooth_alpha_az is not None:
        model.smoothing_alpha_az = request.smooth_alpha_az
    if request.stop_area_threshold is not None:
        model.stop_area_threshold = request.stop_area_threshold
    if request.stop_cx_tolerance is not None:
        model.stop_cx_tolerance = request.stop_cx_tolerance
        
    logger.info(
        f"⚙️ Config updated: speed_scaling={model.speed_scaling_enabled}, "
        f"skip_n={model.grounding_skip_n}, xy_alpha={model.smoothing_alpha}, "
        f"az_alpha={model.smoothing_alpha_az}, stop_area={model.stop_area_threshold}, "
        f"stop_cx={model.stop_cx_tolerance}"
    )
    
    return {
        "status": "success",
        "config": {
            "speed_scaling": model.speed_scaling_enabled,
            "grounding_skip_n": model.grounding_skip_n,
            "smooth_alpha_xy": model.smoothing_alpha,
            "smooth_alpha_az": model.smoothing_alpha_az,
            "stop_area_threshold": model.stop_area_threshold,
            "stop_cx_tolerance": model.stop_cx_tolerance,
        }
    }


@app.post("/predict_debug", response_model=InferenceDebugResponse)
async def predict_debug(request: InferenceRequest, api_key: str = Depends(verify_api_key)):
    """중간 추론 과정을 포함한 디버그 엔드포인트"""
    try:
        model = get_model()
        payload = model.predict_debug(
            image_base64=request.image,
            instruction=request.instruction,
        )
        logger.info(
            "✅ Debug prediction: source=%s, latency=%.1fms",
            payload["source"],
            payload["latency_ms"],
        )
        return InferenceDebugResponse(**payload)
    except Exception as e:
        import traceback
        logger.error(f"Debug prediction failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_debug_path", response_model=InferenceDebugResponse)
async def predict_debug_path(request: InferenceDebugPathRequest):
    """디버그 UI 전용: 서버 로컬 이미지 경로로 추론"""
    try:
        image_path = _resolve_debug_path(request.image_path)
        if not image_path.exists():
            raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")
        if not image_path.is_file():
            raise HTTPException(status_code=400, detail=f"Not a file: {image_path}")

        image_bytes = image_path.read_bytes()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        model = get_model()
        payload = model.predict_debug(
            image_base64=image_base64,
            instruction=request.instruction,
        )
        payload.setdefault("debug", {})
        payload["debug"]["image_path"] = str(image_path)
        logger.info(
            "✅ Debug prediction from path: %s, latency=%.1fms",
            image_path,
            payload["latency_ms"],
        )
        return InferenceDebugResponse(**payload)
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Debug path prediction failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug/reset")
async def debug_reset_history():
    """디버그 UI 전용 무인증 reset"""
    model = get_model()
    try:
        model.reset()
        return {"status": "success", "message": "History, buffer, and logging session reset"}
    except Exception as e:
        logger.error(f"Debug reset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reset")
async def reset_history(api_key: str = Depends(verify_api_key)):
    """추론 히스토리(LSTM Hidden State 등) 초기화 및 세션 종료"""
    model = get_model()
    try:
        model.reset()
        return {"status": "success", "message": "History, buffer, and logging session reset"}
    except Exception as e:
        logger.error(f"Reset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/test")
async def test_endpoint(api_key: str = Depends(verify_api_key)):
    """테스트 엔드포인트"""
    import base64
    from io import BytesIO
    from PIL import Image
    
    dummy_img = Image.new('RGB', (1280, 720), color=(255, 0, 0))
    buffer = BytesIO()
    dummy_img.save(buffer, format='PNG')
    img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    instruction = "Navigate around obstacles and reach the front of the beverage bottle on the left"
    dummy_action = [1.15, 0.319]
    
    return {
        "message": "Test endpoint - using dummy data",
        "instruction": instruction,
        "action": dummy_action,
        "note": "This is a test endpoint. Use POST /predict for real inference."
    }


# ══════════════════════════════════════════════════════════════════════════════
# Exp47 MLP API 엔드포인트
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/predict_mlp", response_model=MLPInferenceResponse)
async def predict_mlp(request: MLPInferenceRequest, api_key: str = Depends(verify_api_key)):
    """
    Exp47 Instruction-Conditioned MLP 추론 엔드포인트.

    기존 /predict(VLM 기반, ~200ms)와 독립적으로 동작.
    bbox + 사전 캐시된 vision feature + instruction embedding → 8-class, <5ms.

    vision feature 갱신 전략:
      - image 필드 제공 + (cache stale OR force_vision_update=True) → VLM encoder 실행 후 캐시 갱신
      - 그 외 → 기존 캐시 재사용 (1Hz 갱신)
    """
    try:
        mlp = get_mlp_model()

        # bbox 업데이트 (매 요청마다)
        mlp.update_bbox(
            cx=request.bbox_cx,
            cy=request.bbox_cy,
            area=request.bbox_area,
            has_bbox=request.has_bbox,
        )

        # vision feature 갱신 필요 여부 판단
        need_vision_update = (
            request.image is not None
            and (request.force_vision_update or mlp.is_vision_cache_stale())
        )

        if need_vision_update:
            # VLM vision encoder로 feature 추출
            try:
                vla_model = get_model()
                vis_feat = vla_model.extract_vision_feature(request.image)
                mlp.update_vision_feature(vis_feat)
                logger.info("🔄 [predict_mlp] vision cache 갱신 완료 (dim=%d)", len(vis_feat))
            except Exception as ve:
                logger.warning("⚠️ [predict_mlp] vision feature 추출 실패 (캐시 재사용): %s", ve)

        # MLP 추론
        result = mlp.predict(request.instruction)

        return MLPInferenceResponse(
            action=result["action"],
            class_idx=result["class_idx"],
            class_name=result["class_name"],
            latency_ms=result["latency_ms"],
            vision_cache_age_ms=result["vision_cache_age_ms"],
            instruction_matched=result["instruction_matched"],
        )

    except Exception as e:
        import traceback
        logger.error(f"MLP prediction failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mlp/update_vision", response_model=VisionUpdateResponse)
async def mlp_update_vision(request: VisionUpdateRequest, api_key: str = Depends(verify_api_key)):
    """
    vision feature 캐시만 별도로 갱신하는 엔드포인트.
    ROS 클라이언트에서 이미지 업데이트와 bbox 업데이트를 분리할 때 사용.
    """
    try:
        t0 = time.time()
        mlp = get_mlp_model()
        vla_model = get_model()
        vis_feat = vla_model.extract_vision_feature(request.image)
        mlp.update_vision_feature(vis_feat)
        latency_ms = (time.time() - t0) * 1000.0
        return VisionUpdateResponse(
            status="ok",
            latency_ms=latency_ms,
            feature_dim=len(vis_feat),
        )
    except Exception as e:
        import traceback
        logger.error(f"Vision update failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mlp/reset")
async def mlp_reset(api_key: str = Depends(verify_api_key)):
    """Exp47 MLP bbox 히스토리 및 vision 캐시 초기화."""
    try:
        mlp = get_mlp_model()
        mlp.reset()
        return {"status": "success", "message": "MLP bbox history and vision cache reset"}
    except Exception as e:
        logger.error(f"MLP reset failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mlp/reload_weights")
async def mlp_reload_weights(api_key: str = Depends(verify_api_key)):
    """exp47_mlp.pt 가중치 재로드 (학습 완료 후 서버 재시작 없이 갱신)."""
    try:
        mlp = get_mlp_model()
        mlp.reload_weights()
        return {"status": "success", "message": "MLP weights reloaded"}
    except Exception as e:
        logger.error(f"MLP weight reload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/mlp/status")
async def mlp_status():
    """Exp47 MLP 모델 상태 확인 (인증 불필요)."""
    global mlp_instance
    if mlp_instance is None:
        return {
            "loaded": False,
            "weights_path": os.getenv("VLA_MLP_WEIGHTS_PATH", "docs/v5/bbox_nav_exp47/exp47_mlp.pt"),
            "message": "MLP not initialized yet. Call /predict_mlp to trigger lazy load.",
        }
    import os.path
    weights_path = mlp_instance._weights_path
    return {
        "loaded": True,
        "weights_path": weights_path,
        "weights_exist": os.path.exists(weights_path),
        "vision_cache_initialized": mlp_instance._vision_cache["initialized"],
        "vision_cache_age_ms": round(mlp_instance.vision_cache_age_ms(), 1),
        "vision_cache_ttl_sec": mlp_instance.vision_cache_ttl_sec,
        "bbox_history_len": len(mlp_instance._bbox_history),
        "instr_keys": list(mlp_instance._instr_embeddings.keys()),
        "num_classes": mlp_instance.NUM_CLASSES,
        "d_in": mlp_instance.D_IN,
    }


# ══════════════════════════════════════════════════════════════════════════════
# GoalNav MLP API 엔드포인트 (exp49 / exp54_s2v2 / exp55)
# ══════════════════════════════════════════════════════════════════════════════

class GoalNavPredictRequest(BaseModel):
    image: str                  # base64 encoded image
    bbox_cx:   float = 0.0
    bbox_cy:   float = 0.0
    bbox_area: float = 0.0
    has_bbox:  bool  = False
    update_vision: bool = True  # True면 매 요청마다 vision feature 갱신


class GoalNavPredictResponse(BaseModel):
    action:     dict
    class_idx:  int
    class_name: str
    latency_ms: float
    variant:    str


@app.post("/goalnav/predict", response_model=GoalNavPredictResponse)
async def goalnav_predict(request: GoalNavPredictRequest, api_key: str = Depends(verify_api_key)):
    """
    GoalNav MLP 추론 (Pure Kosmos-2 기반, exp49 / exp54_s2v2 / exp55).

    매 요청마다 bbox를 히스토리에 추가하고, update_vision=True이면
    Pure Kosmos-2로 vision feature를 갱신한 후 MLP action을 반환한다.
    """
    try:
        m = get_goalnav_model()
        m.update_bbox(
            cx=request.bbox_cx,
            cy=request.bbox_cy,
            area=request.bbox_area,
            has_bbox=request.has_bbox,
        )
        if request.update_vision and request.image:
            m.update_vision_feature(request.image)
        result = m.predict()
        return GoalNavPredictResponse(**result)
    except Exception as e:
        import traceback
        logger.error(f"GoalNav prediction failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/goalnav/reset")
async def goalnav_reset(api_key: str = Depends(verify_api_key)):
    """bbox 히스토리 + vision cache 초기화 (에피소드 시작 시 호출)."""
    global goalnav_instance
    if goalnav_instance is not None:
        goalnav_instance.reset()
    return {"status": "ok", "message": "GoalNav state reset"}


@app.get("/goalnav/status")
async def goalnav_status():
    """GoalNav 모델 상태 확인 (인증 불필요)."""
    global goalnav_instance
    if goalnav_instance is None:
        return {
            "loaded":   False,
            "variant":  os.getenv("VLA_GOALNAV_VARIANT", "exp54_s2v2"),
            "message":  "GoalNav not initialized yet. Call /goalnav/predict to trigger lazy load.",
        }
    return {
        "loaded":               True,
        "variant":              goalnav_instance.variant,
        "d_in":                 goalnav_instance._d_in,
        "weights_path":         goalnav_instance._weights_path,
        "bbox_history_len":     len(goalnav_instance._bbox_history),
        "vision_cache_ready":   goalnav_instance._vis_feat_cache is not None,
        "num_classes":          goalnav_instance.NUM_CLASSES,
        "device":               goalnav_instance.device,
        "stop_rule_enabled":    goalnav_instance._stop_enabled,
        "stop_rule":            {"th_area": goalnav_instance._stop_th_area,
                                 "th_cx": goalnav_instance._stop_th_cx,
                                 "W": goalnav_instance._stop_w,
                                 "min_steps": goalnav_instance._stop_min_steps},
        "stop_learned_enabled": goalnav_instance._stop_learned_enabled,
        "stop_learned_rule":    {"W": goalnav_instance._stop_learned_w,
                                 "th_prob": goalnav_instance._stop_learned_th},
        "stopped":              goalnav_instance._stopped,
        "step":                 goalnav_instance._step,
    }


if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.getenv("VLA_PORT", "8000")))
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    # VLA_GOALNAV_ONLY=1 이면 메인 VLA 모델 스킵, GoalNav MLP만 preload
    if os.getenv("VLA_GOALNAV_ONLY", "0") == "1":
        logger.info("🔄 GoalNav-only mode — pre-loading GoalNav MLP...")
        get_goalnav_model()
        logger.info("✅ GoalNav model pre-loaded. Starting uvicorn...")
    else:
        logger.info("🔄 Pre-loading model before server start...")
        get_model()
        logger.info("✅ Model pre-loaded. Starting uvicorn...")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )

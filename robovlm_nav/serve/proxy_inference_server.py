"""
FastAPI inference server for bbox-based navigation (Exp19 / Exp46 / Exp49 / Exp50 / Exp51).

Two model backends:
  - Exp19 (default, VLA_MODEL unset): bbox history + 16x16 grayscale + proxy feats → ProxyMLP
  - Exp46/49/50/51 (VLA_MODEL=expNN): bbox history + Kosmos-2 vis feat (1024-dim) + grounded goal → GoalNavMLP

Set VLA_MODEL=exp49 (recommended) to use the goal-navigation backend (96.4% val acc, 100% CL).
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cv2
import h5py
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException, Header
from PIL import Image
from pydantic import BaseModel

try:
    from transformers import AutoModelForVision2Seq, AutoProcessor
except ImportError as exc:
    raise RuntimeError("transformers is required for proxy_inference_server.py") from exc


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mobile VLA Proxy API", version="0.2.0")

_recent_predictions: deque[dict] = deque(maxlen=30)

ROOT = Path(project_root)

# 두 서버 호환: billy ↔ minum 데이터셋 자동 resolve. VLA_PROXY_DATA_DIR로 강제 가능.
_DATA_PATH_CANDIDATES = [
    Path("/home/billy/25-1kp/MoNaVLA/ROS_action/mobile_vla_dataset_v5"),
    Path("/home/minum/minum/26CS/MoNa-pi/mobile_vla_dataset_v5"),
    ROOT / "ROS_action" / "mobile_vla_dataset_v5",
]


def _resolve_data_dir() -> Path:
    override = os.getenv("VLA_PROXY_DATA_DIR")
    if override:
        return Path(override)
    for cand in _DATA_PATH_CANDIDATES:
        if cand.exists() and any(cand.glob("episode_*.h5")):
            return cand
    return _DATA_PATH_CANDIDATES[-1]


DATA_DIR = _resolve_data_dir()
DATASET_FILE = ROOT / "docs" / "v5" / "bbox_nav_step1" / "bbox_dataset.json"
DEFAULT_WEIGHTS_PATH = ROOT / "runs" / "v5_nav" / "mlp" / "exp19" / "exp19_proxy_mlp.pt"
DEFAULT_WEIGHTS_PATH_EXP46 = ROOT / "runs" / "v5_nav" / "mlp" / "exp46" / "exp46_mlp.pt"
DEFAULT_WEIGHTS_PATH_EXP47 = ROOT / "runs" / "v5_nav" / "mlp" / "exp47" / "exp47_mlp.pt"
INSTR_EMBS_PATH = ROOT / "docs" / "v5" / "bbox_nav_exp47" / "instruction_embeddings.json"
INSTR_MAP_PATH = ROOT / "docs" / "v5" / "bbox_nav_exp47" / "summary.json"
DEFAULT_GROUNDING_MODEL = ROOT / ".vlms" / "kosmos-2-patch14-224"

_GOAL_NAV_WEIGHTS: dict[str, Path] = {
    "exp46": ROOT / "runs" / "v5_nav" / "mlp" / "exp46" / "exp46_mlp.pt",
    "exp49": ROOT / "runs" / "v5_nav" / "mlp" / "exp49" / "exp49_mlp.pt",
    "exp50": ROOT / "runs" / "v5_nav" / "mlp" / "exp50" / "exp50_mlp.pt",
    "exp51": ROOT / "runs" / "v5_nav" / "mlp" / "exp51" / "exp51_mlp.pt",
    # exp52: lang+vis 2048-dim — weight 로드는 가능하지만 실시간 feature 추출 미지원
    "exp52": ROOT / "runs" / "v5_nav" / "mlp" / "exp52" / "exp52_mlp.pt",
    "exp53": ROOT / "runs" / "v5_nav" / "mlp" / "exp53_clip_lora.pt",
}

# exp53용 CLIP LoRA adapter 경로 (vision_model layers 16-23 q_proj/v_proj)
_GOAL_NAV_LORA_ADAPTERS: dict[str, Path] = {
    "exp53": ROOT / "runs" / "v5_nav" / "mlp" / "clip_lora_adapter",
}

NUM_CLASSES = 8
WINDOW = 3          # Exp19
WINDOW_EXP46 = 8    # Exp46/47
VIS_DIM = 1024      # Kosmos-2 vision encoder output dim
INSTR_DIM = 2048    # Kosmos-2 text encoder output dim
IMG_SIZE = 16
CONSISTENCY_K = 5
CX_TOL = 0.08
AREA_TOL = 0.08
FULLSCREEN_AREA_THRESHOLD = 0.85
GROUNDING_PROMPT = "<grounding>The gray basket is at"
CLASS_NAMES = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "ROT_L", "ROT_R"]
ACTION_2D = {
    0: [0.0, 0.0],
    1: [1.15, 0.0],
    2: [0.0, 1.15],
    3: [0.0, -1.15],
    4: [1.15, 1.15],
    5: [1.15, -1.15],
    6: [0.0, 0.0],
    7: [0.0, 0.0],
}
ACTION_3D = {
    0: [0.0, 0.0, 0.0],
    1: [1.15, 0.0, 0.0],
    2: [0.0, 1.15, 0.0],
    3: [0.0, -1.15, 0.0],
    4: [1.15, 1.15, 0.0],
    5: [1.15, -1.15, 0.0],
    6: [0.0, 0.0, 0.25],
    7: [0.0, 0.0, -0.25],
}

model_instance = None
_goal_nav_cache: dict[str, "GoalNavInferenceModel"] = {}
_active_goal_nav_model: str = os.getenv("VLA_MODEL", "exp49")


class InferenceRequest(BaseModel):
    image: str
    instruction: str
    vlm_model: Optional[str] = "kosmos"


class InferenceResponse(BaseModel):
    action: list[float]
    latency_ms: float
    model_name: str
    strategy: str
    source: str
    buffer_status: dict[str, Any]
    predicted_class: Optional[int] = None
    predicted_label: Optional[str] = None
    action_3d: Optional[list[float]] = None
    bbox: Optional[dict[str, Any]] = None
    grounding_caption: Optional[str] = None
    grounding_latency_ms: Optional[float] = None
    goal_near_proxy: Optional[bool] = None
    instruction_used: bool = False
    matched_path_type: Optional[str] = None
    speed_scale: Optional[float] = None
    grounding_cached: Optional[bool] = None


class ConfigRequest(BaseModel):
    speed_scaling: Optional[bool] = None
    grounding_skip_n: Optional[int] = None
    smooth_enabled: Optional[bool] = None
    smooth_alpha_xy: Optional[float] = None
    smooth_alpha_az: Optional[float] = None
    model: Optional[str] = None  # "exp49" | "exp50" | "exp51" | "exp52"


class ProxyMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        # Exp47 uses a 5-layer MLP for 3104-dim input
        if input_dim == 3104:
            self.net = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes),
            )
        else:
            # Fallback for Exp46/Exp19 (4 layers)
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GoalNavMLP(nn.Module):
    """Exp49/50/51 MLP: 5-layer (512→256→128→64→8), d_in from checkpoint."""

    def __init__(self, d_in: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 512), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(512, 256),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),  nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64),   nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GoalNavMLP(nn.Module):
    """Exp46/49/50/51 MLP: 5-layer, d_in from checkpoint."""

    def __init__(self, d_in: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 512), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(512, 256),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128),  nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64),   nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


_COARSE_CLF_PATH   = ROOT / "runs" / "v5_nav" / "mlp" / "step1" / "coarse_direction_clf.pt"
_GROUNDING_LORA    = ROOT / "docs" / "v5" / "bbox_nav_step1" / "grounding_lora"
_COARSE_LABEL_CX   = {0: 0.25, 1: 0.5, 2: 0.75}

# PaliGemma LoRA adapters
_EXP57_ADAPTER = ROOT / "runs" / "v5_nav" / "grounding" / "exp57"  # PaliGemma-3b-pt LoRA
_EXP59_ADAPTER = ROOT / "runs" / "v5_nav" / "grounding" / "exp59"  # PaliGemma2-3b-mix LoRA
# HF cache paths (downloaded locally)
_HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
_PG_PT_PATH  = _HF_CACHE / "models--google--paligemma-3b-pt-224"
_PG2_MIX_PATH = _HF_CACHE / "models--google--paligemma2-3b-mix-224"


def _parse_paligemma_locs(text: str) -> Optional[dict]:
    # PaliGemma outputs <loc_XXXX> tokens in y1,x1,y2,x2 order (0-1023)
    locs = re.findall(r"<loc(\d{4})>", text)
    if len(locs) < 4:
        return None
    y1, x1, y2, x2 = [int(v) / 1023.0 for v in locs[:4]]
    if x2 <= x1 or y2 <= y1:
        return None
    return {
        "cx":   (x1 + x2) / 2,
        "cy":   (y1 + y2) / 2,
        "area": (x2 - x1) * (y2 - y1),
    }


class GroundingBackend:
    def __init__(self, default_model_path: Path, device: torch.device, vis_lora_adapter_path: Optional[Path] = None):
        self.default_model_path = default_model_path
        self.device = device
        self.vis_lora_adapter_path = vis_lora_adapter_path
        self.current_model_name = None
        self.model = None
        self.processor = None
        self.tokenizer = None

        # Coarse direction classifier (optional, loaded if weights file exists)
        self._coarse_clf: Optional[nn.Linear] = None
        self._coarse_mean: Optional[torch.Tensor] = None
        self._coarse_std: Optional[torch.Tensor] = None
        if _COARSE_CLF_PATH.exists():
            ckpt = torch.load(_COARSE_CLF_PATH, map_location=device, weights_only=False)
            feat_dim = ckpt.get("feature_dim", 1024)
            clf = nn.Linear(feat_dim, 3).to(device).eval()
            clf.load_state_dict(ckpt["model"])
            self._coarse_clf = clf
            self._coarse_mean = ckpt["mean"].float().to(device)
            self._coarse_std = ckpt["std"].float().to(device)
            logger.info("Loaded coarse direction classifier (%d samples)", ckpt.get("n_samples", 0))
        else:
            logger.info("No coarse direction classifier found at %s", _COARSE_CLF_PATH)

    def _switch_model(self, model_name: str) -> None:
        """On-demand switches the active VLM grounding model to save GPU memory."""
        if self.current_model_name == model_name:
            return

        logger.info("Switching VLM grounding model from %s to %s...", self.current_model_name, model_name)

        # Clear existing model to free VRAM
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            self.processor = None
        if self.tokenizer is not None:
            self.tokenizer = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        device = self.device
        dtype = torch.float16 if device.type == "cuda" else torch.float32

        if model_name == "kosmos":
            self.processor = AutoProcessor.from_pretrained(str(self.default_model_path))
            self.model = AutoModelForVision2Seq.from_pretrained(
                str(self.default_model_path),
                torch_dtype=dtype,
            ).to(device).eval()

            # Optional LoRA grounding adapter
            if _GROUNDING_LORA.exists() and os.getenv("VLA_ENABLE_GROUNDING_LORA") == "1":
                try:
                    from peft import PeftModel
                    self.model = PeftModel.from_pretrained(self.model, str(_GROUNDING_LORA))
                    self.model = self.model.merge_and_unload()
                    self.model.eval()
                    logger.info("Loaded grounding LoRA adapter from %s", _GROUNDING_LORA)
                except Exception as e:
                    logger.warning("Failed to load grounding LoRA (%s); using base model", e)

            # Optional vis LoRA adapter (applied for exp53)
            if self.vis_lora_adapter_path is not None and self.vis_lora_adapter_path.exists():
                try:
                    from peft import PeftModel
                    self.model.vision_model = PeftModel.from_pretrained(
                        self.model.vision_model, str(self.vis_lora_adapter_path)
                    ).eval()
                    logger.info("Loaded vis LoRA adapter from %s", self.vis_lora_adapter_path)
                except Exception as e:
                    logger.warning("Failed to load vis LoRA adapter (%s); using base vision model", e)

        elif model_name == "paligemma":
            model_id = _MODEL_IDS.get("paligemma-mix", "google/paligemma-3b-mix-224")
            # PaliGemma prefers bfloat16 on GPU
            pg_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
            from transformers import PaliGemmaForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained(model_id)
            self.model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=pg_dtype
            ).to(device).eval()

        elif model_name == "exp57":
            # PaliGemma-3b-pt + exp57 LoRA (grounding)
            pg_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
            from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
            from peft import PeftModel
            # Resolve local path: prefer local cache, fall back to HF hub ID
            pg_pt_local = next(iter(_PG_PT_PATH.glob("snapshots/*/")), None)
            pg_pt_id = str(pg_pt_local) if pg_pt_local else "google/paligemma-3b-pt-224"
            logger.info("[exp57] Loading PaliGemma-3b-pt from %s", pg_pt_id)
            self.processor = PaliGemmaProcessor.from_pretrained(pg_pt_id)
            base = PaliGemmaForConditionalGeneration.from_pretrained(
                pg_pt_id, torch_dtype=pg_dtype
            ).to(device).eval()
            self.model = PeftModel.from_pretrained(base, str(_EXP57_ADAPTER)).eval()
            logger.info("[exp57] LoRA adapter loaded from %s", _EXP57_ADAPTER)

        elif model_name == "exp59":
            # PaliGemma2-3b-mix + exp59 LoRA (2-class grounding)
            pg_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
            from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
            from peft import PeftModel
            pg2_local = next(iter(_PG2_MIX_PATH.glob("snapshots/*/")), None)
            pg2_id = str(pg2_local) if pg2_local else "google/paligemma2-3b-mix-224"
            logger.info("[exp59] Loading PaliGemma2-3b-mix from %s", pg2_id)
            self.processor = PaliGemmaProcessor.from_pretrained(pg2_id)
            base = PaliGemmaForConditionalGeneration.from_pretrained(
                pg2_id, torch_dtype=pg_dtype
            ).to(device).eval()
            self.model = PeftModel.from_pretrained(base, str(_EXP59_ADAPTER)).eval()
            logger.info("[exp59] LoRA adapter loaded from %s", _EXP59_ADAPTER)

        elif model_name == "moondream":
            model_id = _MODEL_IDS.get("moondream", "vikhyatk/moondream2")
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                revision=_MOONDREAM_REVISION,
                torch_dtype=dtype,
            ).to(device).eval()

        self.current_model_name = model_name
        logger.info("Successfully loaded VLM: %s on %s", model_name, device)

    # caption 방향 패턴: (phrases, cx값) — 긴 구문 먼저 체크
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

    def _caption_to_cx(self, caption_lower: str) -> Optional[float]:
        """caption 텍스트에서 방향 패턴 매칭 → cx 반환. 매칭 없으면 None."""
        for phrases, cx in self._CAPTION_DIRECTION_PATTERNS:
            if any(p in caption_lower for p in phrases):
                return cx
        return None

    def _coarse_direction_cx(self, pixel_values: torch.Tensor) -> Optional[float]:
        """Frozen Kosmos-2 vision features → coarse direction (LEFT/CENTER/RIGHT) → cx."""
        if self._coarse_clf is None:
            return None
        with torch.no_grad():
            vo = self.model.vision_model(pixel_values=pixel_values)
            feats = vo.last_hidden_state.mean(dim=1).float()  # (1, 1024)
            feats_norm = (feats - self._coarse_mean) / (self._coarse_std + 1e-6)
            pred = self._coarse_clf(feats_norm).argmax(dim=-1).item()
        return _COARSE_LABEL_CX[pred]

    def _parse_basket_bbox(self, caption: str, entities: list[Any], target_object: str = "basket") -> Optional[dict[str, Any]]:
        target_lower = target_object.lower()
        target_keywords = {target_lower}
        for word in target_lower.split():
            if len(word) > 2:
                target_keywords.add(word)
        if "basket" in target_lower:
            target_keywords.update(["container", "bin", "laundry", "gray box"])

        candidates = []
        for entity_name, _span, boxes in entities:
            for box in boxes:
                x1, y1, x2, y2 = [float(v) for v in box]
                if max(x1, y1, x2, y2) > 1.5:
                    x1, y1, x2, y2 = x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0
                area = (x2 - x1) * (y2 - y1)
                if area > FULLSCREEN_AREA_THRESHOLD:
                    continue
                candidates.append(
                    {
                        "entity": entity_name,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "cx": (x1 + x2) / 2.0,
                        "cy": (y1 + y2) / 2.0,
                        "area": area,
                        "is_target": any(k in entity_name.lower() for k in target_keywords),
                    }
                )

        matched = [b for b in candidates if b["is_target"]]
        if matched:
            return matched[0]

        # entity 매칭 실패 시 caption 텍스트로 방향 추정
        caption_lower = caption.lower()
        cx = self._caption_to_cx(caption_lower)
        if cx is not None:
            # 방향에 따른 label 결정
            if cx < 0.2:
                label = "caption:far_left"
            elif cx < 0.4:
                label = "caption:left"
            elif cx > 0.8:
                label = "caption:far_right"
            elif cx > 0.6:
                label = "caption:right"
            else:
                label = "caption:center"
            return {"entity": label, "cx": cx, "cy": 0.6, "area": 0.06}

        return None

    def extract_vis_feat(self, image_rgb: np.ndarray) -> np.ndarray:
        """Kosmos-2 vision encoder mean-pool → (1024,) float32. Reuses loaded model."""
        self._switch_model("kosmos")
        pil = Image.fromarray(image_rgb.astype(np.uint8)).convert("RGB")
        inputs = self.processor(text="<grounding>The gray basket is at", images=pil, return_tensors="pt")
        pv = inputs["pixel_values"].to(self.device)
        if self.device.type == "cuda":
            pv = pv.to(torch.float16)
        with torch.no_grad():
            vo = self.model.vision_model(pixel_values=pv)
            feat = vo.last_hidden_state[0].mean(0).float().cpu().numpy()
        return feat  # (1024,)

    def run(self, image_rgb: np.ndarray, instruction: str = "basket", vlm_model: str = "kosmos", extract_vis: bool = False) -> dict[str, Any]:
        self._switch_model(vlm_model)
        target_object = instruction.lower().replace("the", "").replace("a", "").strip()

        pil_image = Image.fromarray(image_rgb.astype(np.uint8)).convert("RGB")
        bbox = None
        caption = ""
        latency_ms = 0.0
        vis_feat = None

        start = time.time()

        if vlm_model == "kosmos":
            prompt = f"<grounding>The {target_object} is at"
            inputs = self.processor(text=prompt, images=pil_image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            pixel_values = inputs["pixel_values"].to(torch.float16 if self.device.type == "cuda" else torch.float32)

            with torch.no_grad():
                generated = self.model.generate(
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

            if extract_vis:
                with torch.no_grad():
                    vo = self.model.vision_model(pixel_values=pixel_values)
                    vis_feat = vo.last_hidden_state[0].mean(0).float().cpu().numpy()

        elif vlm_model in ("paligemma", "exp57", "exp59"):
            prompt = f"detect {target_object}\n"
            inputs = self.processor(text=prompt, images=pil_image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                gen = self.model.generate(**inputs, max_new_tokens=100)
            full_decoded = self.processor.decode(gen[0], skip_special_tokens=False)
            prompt_end = full_decoded.find(prompt.strip())
            decoded = full_decoded[prompt_end + len(prompt):] if prompt_end >= 0 else full_decoded
            caption = decoded.strip()

            locs = _parse_paligemma_locs(caption)
            if locs:
                bbox = {"entity": f"{vlm_model}:{target_object}", "cx": locs["cx"], "cy": locs["cy"], "area": locs["area"]}

        elif vlm_model == "moondream":
            with torch.no_grad():
                try:
                    raw = self.model.detect(pil_image, target_object)
                    objects = raw.get("objects", raw) if isinstance(raw, dict) else raw
                except Exception as e:
                    logger.warning("Moondream detect error: %s", e)
                    objects = []

            caption = f"{len(objects)} object(s) detected"
            if objects:
                obj = objects[0]
                x1 = float(obj.get("x_min", obj.get("xmin", 0)))
                y1 = float(obj.get("y_min", obj.get("ymin", 0)))
                x2 = float(obj.get("x_max", obj.get("xmax", 1)))
                y2 = float(obj.get("y_max", obj.get("ymax", 1)))
                area = abs(x2 - x1) * abs(y2 - y1)
                if area < 0.95:
                    bbox = {"cx": (x1+x2)/2, "cy": (y1+y2)/2, "area": area}

        latency_ms = (time.time() - start) * 1000.0

        # Fallback if detection fails — coarse_clf(0.25/0.75) 대신 center(0.5) 사용
        # coarse_clf는 ROT_R/ROT_L 무한루프 유발: 잘못된 cx 고정 → 목표 방향 고정 → 스핀
        if bbox is None and vlm_model == "kosmos":
            bbox = {"entity": "center_fallback", "cx": 0.5, "cy": 0.6, "area": 0.06}
        elif bbox is None:
            # Caption fallback for other models
            caption_lower = caption.lower()
            cx = self._caption_to_cx(caption_lower)
            if cx is not None:
                if cx < 0.2: label = "caption:far_left"
                elif cx < 0.4: label = "caption:left"
                elif cx > 0.8: label = "caption:far_right"
                elif cx > 0.6: label = "caption:right"
                else: label = "caption:center"
                bbox = {"entity": label, "cx": cx, "cy": 0.6, "area": 0.06}

        result: dict[str, Any] = {
            "caption": caption,
            "bbox": bbox,
            "latency_ms": latency_ms,
            "vis_feat": vis_feat,
        }
        return result


def load_dataset(dataset_path: Path) -> list[dict[str, Any]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    return json.loads(dataset_path.read_text())


def load_episode_frames(stem: str) -> np.ndarray:
    path = next(DATA_DIR.glob(f"{stem}.h5"))
    with h5py.File(path, "r") as handle:
        if "observations" in handle and "images" in handle["observations"]:
            images = handle["observations"]["images"][:]
        else:
            images = handle["images"][:]
    return images


def frame_to_small_feature(frame_rgb: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(frame_rgb.astype(np.uint8)).convert("L").resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return arr.reshape(-1)


def recent_bbox_consistency(frames: list[dict[str, Any]], t: int) -> float:
    start = max(0, t - CONSISTENCY_K + 1)
    tail = frames[start : t + 1]
    valid = [frame for frame in tail if frame["has_bbox"]]
    if not valid:
        return 0.0
    if len(valid) == 1:
        return 1.0

    stable_pairs = 0
    total_pairs = 0
    for prev, cur in zip(valid[:-1], valid[1:]):
        total_pairs += 1
        if (
            abs(float(cur["cx"]) - float(prev["cx"])) <= CX_TOL
            and abs(float(cur["area"]) - float(prev["area"])) <= AREA_TOL
        ):
            stable_pairs += 1
    return stable_pairs / max(total_pairs, 1)


def build_proxy_features(frames: list[dict[str, Any]], t: int) -> list[float]:
    cur = frames[t]
    prev = frames[t - 1] if t > 0 else None
    return [
        float(cur["area"]),
        abs(float(cur["cx"]) - 0.5),
        0.0 if prev is None else abs(float(cur["cx"]) - float(prev["cx"])),
        recent_bbox_consistency(frames, t),
    ]


def make_episode_split(dataset: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(42)
    by_path = defaultdict(list)
    for idx, episode in enumerate(dataset):
        by_path[episode["path_type"]].append(idx)

    train_idx, test_idx = [], []
    for idxs in by_path.values():
        rng.shuffle(idxs)
        k = max(1, int(len(idxs) * 0.2))
        test_idx.extend(idxs[:k])
        train_idx.extend(idxs[k:])

    return [dataset[i] for i in train_idx], [dataset[i] for i in test_idx]


# 정지 판단 기준 (환경변수로 조정 가능)
GOAL_AREA_THRESHOLD  = float(os.getenv("VLA_STOP_AREA",  "0.18"))   # bbox 면적 (0~1)
GOAL_CX_TOLERANCE    = float(os.getenv("VLA_STOP_CX_TOL", "0.25"))  # 중앙에서 허용 편차


def goal_near_proxy(frame: dict[str, Any], area_thr: float = GOAL_AREA_THRESHOLD, cx_tol: float = GOAL_CX_TOLERANCE) -> bool:
    """
    정지 조건:
      1. 실제 그라운딩 bbox (has_bbox=True) → area + center 체크
      2. fallback bbox (coarse_clf/center_fallback) → area=0.06 고정이므로 정지 불가
    """
    area = float(frame.get("area", 0))
    cx   = float(frame.get("cx",   0.5))
    entity = str(frame.get("entity", ""))

    # fallback bbox는 area=0.06 하드코딩 → 실제 거리 정보 없으므로 정지 판단 제외
    is_fallback = entity in ("coarse_clf", "center_fallback", "") or entity.startswith("caption:")

    if is_fallback:
        return False

    center_ok = abs(cx - 0.5) <= cx_tol
    area_ok   = area >= area_thr
    return area_ok and center_ok


class ProxyInferenceModel:
    def __init__(
        self,
        dataset_file: Path,
        weights_path: Path,
        proxy_device: torch.device,
        grounding_device: torch.device,
        grounding_model_path: Path,
        epochs: int,
        force_retrain: bool,
    ):
        self.dataset_file = dataset_file
        self.weights_path = weights_path
        self.proxy_device = proxy_device
        self.grounding_device = grounding_device
        self.grounding_model_path = grounding_model_path
        self.epochs = epochs
        self.force_retrain = force_retrain
        self.model_name = "exp19_proxy_mlp"
        self.model_type = "exp19"     # detected from d_in after load
        self.effective_window = WINDOW
        self.model: Optional[ProxyMLP] = None
        self.model_info: dict[str, Any] = {}
        self.history: list[dict[str, Any]] = []
        self.inference_count = 0
        self._instr_embs: dict[str, np.ndarray] = {}
        self._instr_text_to_path: dict[str, str] = {}

        self.grounder = GroundingBackend(grounding_model_path, grounding_device)
        self._load_or_train()
        self._load_instruction_embeddings()

    def _build_windows(self, dataset: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        features, labels = [], []
        for episode in dataset:
            images = load_episode_frames(episode["episode"])
            frames = episode["frames"]
            img_feats = [frame_to_small_feature(images[f["frame_idx"]]) for f in frames]
            for t in range(len(frames)):
                feat = []
                for k in range(WINDOW):
                    idx = max(0, t - (WINDOW - 1 - k))
                    frame = frames[idx]
                    feat.extend([frame["cx"], frame["cy"], frame["area"], float(frame["has_bbox"])])
                feat.extend(img_feats[t].tolist())
                feat.extend(build_proxy_features(frames, t))
                features.append(feat)
                labels.append(frames[t]["gt_class"])
        return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int64)

    def _train_model(self) -> tuple[ProxyMLP, dict[str, Any]]:
        dataset = load_dataset(self.dataset_file)
        train_ds, test_ds = make_episode_split(dataset)
        x_train, y_train = self._build_windows(train_ds)
        x_test, y_test = self._build_windows(test_ds)

        model = ProxyMLP(input_dim=x_train.shape[1]).to(self.proxy_device)
        class_counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float32)
        class_counts = np.where(class_counts == 0, 1.0, class_counts)
        weights = torch.tensor(1.0 / class_counts, dtype=torch.float32, device=self.proxy_device)
        weights = weights / weights.sum() * NUM_CLASSES
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

        x_train_t = torch.tensor(x_train, dtype=torch.float32, device=self.proxy_device)
        y_train_t = torch.tensor(y_train, dtype=torch.long, device=self.proxy_device)
        x_test_t = torch.tensor(x_test, dtype=torch.float32, device=self.proxy_device)
        y_test_t = torch.tensor(y_test, dtype=torch.long, device=self.proxy_device)

        best_acc = -1.0
        best_state = None
        for epoch in range(self.epochs):
            model.train()
            order = torch.randperm(len(x_train_t), device=self.proxy_device)
            for start in range(0, len(order), 128):
                batch = order[start : start + 128]
                logits = model(x_train_t[batch])
                loss = loss_fn(logits, y_train_t[batch])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                preds = model(x_test_t).argmax(dim=-1)
                acc = (preds == y_test_t).float().mean().item()
            if acc > best_acc:
                best_acc = acc
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            if epoch % 40 == 0 or epoch == self.epochs - 1:
                logger.info("Proxy MLP epoch %03d: test_acc=%.3f best=%.3f", epoch, acc, best_acc)

        assert best_state is not None
        model.load_state_dict(best_state)
        model.eval()

        package = {
            "state_dict": best_state,
            "input_dim": int(x_train.shape[1]),
            "epochs": int(self.epochs),
            "dataset_file": str(self.dataset_file),
            "test_acc": float(best_acc),
            "train_windows": int(len(x_train)),
            "test_windows": int(len(x_test)),
        }
        return model, package

    @staticmethod
    def _detect_model_type(d_in: int) -> str:
        if d_in == 3104:
            return "exp47"
        if d_in == 1056:
            return "exp46"
        return "exp19"

    def _apply_model_type(self, d_in: int) -> None:
        self.model_type = self._detect_model_type(d_in)
        self.effective_window = WINDOW_EXP46 if self.model_type in ("exp46", "exp47") else WINDOW
        if self.model_type != "exp19":
            self.model_name = f"{self.model_type}_mlp"
        logger.info("Model type detected: %s (d_in=%d, window=%d)", self.model_type, d_in, self.effective_window)

    def _load_instruction_embeddings(self) -> None:
        if self.model_type != "exp47":
            return
        if not INSTR_EMBS_PATH.exists():
            logger.warning("Instruction embeddings not found: %s", INSTR_EMBS_PATH)
            return
        data = json.loads(INSTR_EMBS_PATH.read_text())
        self._instr_embs = {k: np.array(v, dtype=np.float32) for k, v in data.items()}
        # Build reverse map: instruction text → path_type
        if INSTR_MAP_PATH.exists():
            summary = json.loads(INSTR_MAP_PATH.read_text())
            instr_map = summary.get("instr_map", {})
            self._instr_text_to_path = {v.lower(): k for k, v in instr_map.items()}
        logger.info("Loaded %d instruction embeddings for Exp47", len(self._instr_embs))

    def _load_or_train(self) -> None:
        if self.weights_path.exists() and not self.force_retrain:
            package = torch.load(self.weights_path, map_location="cpu", weights_only=False)
            input_dim = int(package.get("input_dim", package.get("d_in", 0)))
            model = ProxyMLP(input_dim=input_dim)
            # Support both 'state_dict' and 'model_state_dict'
            sd = package.get("model_state_dict", package.get("state_dict", package))
            
            # Check if keys need 'net.' prefix
            first_key = next(iter(sd.keys()))
            if not first_key.startswith("net."):
                # If keys are like '0.weight', load into self.net
                model.net.load_state_dict(sd)
            else:
                model.load_state_dict(sd)
                
            self.model = model.to(self.proxy_device).eval()
            self._apply_model_type(input_dim)
            self.model_info = {
                "source": "loaded",
                "weights_path": str(self.weights_path),
                "model_type": self.model_type,
                "test_acc": package.get("overall_acc", package.get("test_acc", package.get("best_val_acc"))),
                "train_windows": package.get("train_windows", package.get("n_train")),
                "test_windows": package.get("test_windows", package.get("n_val")),
                "epochs": package.get("epochs"),
                "d_in": input_dim,
            }
            logger.info("Loaded proxy MLP weights from %s", self.weights_path)
            return

        logger.info("Training proxy MLP from cached dataset: %s", self.dataset_file)
        model, package = self._train_model()
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(package, self.weights_path)
        self.model = model
        input_dim = int(package["input_dim"])
        self._apply_model_type(input_dim)
        self.model_info = {
            "source": "trained",
            "weights_path": str(self.weights_path),
            "model_type": self.model_type,
            "test_acc": package.get("test_acc"),
            "train_windows": package.get("train_windows"),
            "test_windows": package.get("test_windows"),
            "epochs": package.get("epochs"),
            "d_in": input_dim,
        }
        logger.info("Saved proxy MLP weights to %s", self.weights_path)

    def reset(self) -> None:
        self.history.clear()
        self.inference_count = 0

    def _decode_image(self, image_base64: str) -> np.ndarray:
        image_bytes = base64.b64decode(image_base64)
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.array(pil)

    def _bbox_frame(self, bbox: Optional[dict[str, Any]]) -> dict[str, Any]:
        if not bbox:
            return {"cx": 0.5, "cy": 0.5, "area": 0.0, "has_bbox": False}
        return {
            "cx": float(bbox["cx"]),
            "cy": float(bbox["cy"]),
            "area": float(bbox["area"]),
            "has_bbox": True,
        }

    def _build_online_feature(self, current_rgb: np.ndarray) -> np.ndarray:
        """Exp19 feature: window=3, 16×16 grayscale + proxy features."""
        frame = frame_to_small_feature(current_rgb)
        feat: list[float] = []
        current_idx = len(self.history) - 1
        for k in range(WINDOW):
            idx = max(0, current_idx - (WINDOW - 1 - k))
            item = self.history[idx]
            feat.extend([item["cx"], item["cy"], item["area"], float(item["has_bbox"])])
        feat.extend(frame.tolist())
        feat.extend(build_proxy_features(self.history, current_idx))
        return np.asarray(feat, dtype=np.float32)

    def _build_exp46_feature(self) -> np.ndarray:
        """Exp46 feature: window=8, bbox history(32) + vision(1024) = 1056-dim."""
        feat: list[float] = []
        current_idx = len(self.history) - 1
        for k in range(WINDOW_EXP46):
            idx = max(0, current_idx - (WINDOW_EXP46 - 1 - k))
            item = self.history[idx]
            feat.extend([item["cx"], item["cy"], item["area"], float(item["has_bbox"])])
        vis_feat = self.history[-1].get("vis_feat")
        if vis_feat is None:
            vis_feat = np.zeros(VIS_DIM, dtype=np.float32)
        feat.extend(vis_feat.tolist())
        return np.asarray(feat, dtype=np.float32)

    def _infer_path_type_from_bbox(self) -> str:
        """bbox cx 기반으로 path_type 자동 추론. instruction 미매칭 시 폴백으로 사용.

        이미지 좌표: cx=0 → 왼쪽, cx=1 → 오른쪽
        basket이 오른쪽(cx > 0.65) → right_right (오른쪽으로 커브)
        basket이 왼쪽(cx < 0.35)  → left_left  (왼쪽으로 커브)
        """
        if not self.history:
            return "center_straight"
        # NO_BBOX 프레임(has_bbox=False)은 cx=0.5 기본값이므로 마지막 유효 cx 사용
        cx = 0.5
        for item in reversed(self.history):
            if item.get("has_bbox", False):
                cx = item.get("cx", 0.5)
                break
        if cx > 0.65:
            return "right_right"
        if cx < 0.35:
            return "left_left"
        return "center_straight"

    def _get_instruction_embedding(self, instruction: str) -> tuple[np.ndarray, str]:
        """instruction → embedding 반환. 매칭된 path_type도 함께 반환."""
        if not self._instr_embs:
            logger.warning("instruction_embeddings not loaded — MLP receives zero vector; check %s", INSTR_EMBS_PATH)
            return np.zeros(INSTR_DIM, dtype=np.float32), "none"
        # Direct path_type key match
        if instruction in self._instr_embs:
            return self._instr_embs[instruction], instruction
        # Match against known instruction text
        path_type = self._instr_text_to_path.get(instruction.lower())
        if path_type and path_type in self._instr_embs:
            return self._instr_embs[path_type], path_type
        # Partial text match
        for text, pt in self._instr_text_to_path.items():
            if text in instruction.lower() or instruction.lower() in text:
                if pt in self._instr_embs:
                    return self._instr_embs[pt], pt
        # Auto-infer from bbox cx (unrecognized instruction)
        auto_pt = self._infer_path_type_from_bbox()
        if auto_pt in self._instr_embs:
            logger.info("Instruction unrecognized → bbox auto path_type: %s (cx=%.3f)",
                        auto_pt, self.history[-1].get("cx", 0.5) if self.history else 0.5)
            return self._instr_embs[auto_pt], f"auto:{auto_pt}"
        return next(iter(self._instr_embs.values())), "fallback"

    def _build_exp47_feature(self, instruction: str) -> tuple[np.ndarray, str]:
        """Exp47 feature: bbox(32) + vision(1024) + instruction_emb(2048) = 3104-dim."""
        exp46_feat = self._build_exp46_feature()
        instr_emb, matched_path = self._get_instruction_embedding(instruction)
        return np.concatenate([exp46_feat, instr_emb]), matched_path

    def predict(self, image_base64: str, instruction: str, vlm_model: str = "kosmos") -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Proxy model is not loaded")

        start = time.time()
        image_rgb = self._decode_image(image_base64)
        extract_vis = self.model_type in ("exp46", "exp47")
        grounding = self.grounder.run(image_rgb, instruction=instruction, vlm_model=vlm_model, extract_vis=extract_vis)
        bbox = grounding["bbox"]
        vis_feat = grounding.get("vis_feat")

        history_item = self._bbox_frame(bbox)
        history_item["vis_feat"] = vis_feat
        self.history.append(history_item)
        keep = max(self.effective_window, CONSISTENCY_K)
        if len(self.history) > keep:
            self.history = self.history[-keep:]

        matched_path = "n/a"
        if self.model_type == "exp47":
            feature, matched_path = self._build_exp47_feature(instruction)
            instruction_used = bool(self._instr_embs)
        elif self.model_type == "exp46":
            feature = self._build_exp46_feature()
            instruction_used = False
        else:
            feature = self._build_online_feature(image_rgb)
            instruction_used = False

        x = torch.tensor(feature[None, :], dtype=torch.float32, device=self.proxy_device)
        with torch.no_grad():
            logits = self.model(x)
            # ROT_L(6) / ROT_R(7) 마스킹 — 학습 데이터 0%, 분포 외 입력에서 잘못 활성화
            if logits.shape[-1] > 6:
                logits = logits.clone()
                logits[:, 6] = float("-inf")
                logits[:, 7] = float("-inf")
            probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()
            pred_class = int(logits.argmax(dim=-1).item())

        bbox_summary = (
            f"cx={bbox['cx']:.3f} area={bbox['area']:.3f} entity={bbox.get('entity','?')}"
            if bbox else "NO_BBOX"
        )
        logger.info(
            "[#%d] %s | bbox: %s | path: %s | caption: %s | probs: %s",
            self.inference_count + 1,
            CLASS_NAMES[pred_class],
            bbox_summary,
            matched_path,
            (grounding["caption"] or "")[:60],
            " ".join(f"{CLASS_NAMES[i]}={probs[i]:.2f}" for i in range(NUM_CLASSES)),
        )

        self.inference_count += 1
        return {
            "action": ACTION_2D[pred_class],   # 2D [lx, ly] — api contract; az preserved in action_3d
            "action_3d": ACTION_3D[pred_class],
            "latency_ms": (time.time() - start) * 1000.0,
            "predicted_class": pred_class,
            "predicted_label": CLASS_NAMES[pred_class],
            "bbox": bbox,
            "grounding_caption": grounding["caption"],
            "grounding_latency_ms": grounding["latency_ms"],
            "goal_near_proxy": goal_near_proxy(self.history[-1], getattr(self, "_stop_area_thr", GOAL_AREA_THRESHOLD), getattr(self, "_stop_cx_tol", GOAL_CX_TOLERANCE)),
            "buffer_status": {
                "history_size": len(self.history),
                "window": self.effective_window,
                "consistency_k": CONSISTENCY_K,
                "model_type": self.model_type,
            },
            "source": self.model_info.get("source", "loaded"),
            "instruction_used": instruction_used,
            "matched_path_type": matched_path,
            "instruction": instruction,
        }

class GoalNavInferenceModel:
    """Exp46/49/50/51: Kosmos-2 vis feat (1024-dim) + bbox history + grounded goal → GoalNavMLP."""

    def __init__(
        self,
        weights_path: Path,
        grounding_model_path: Path,
        grounding_device: torch.device,
        device: torch.device,
        lora_adapter_path: Optional[Path] = None,
    ):
        self.device = device
        self.model: Optional[GoalNavMLP] = None
        self.window: int = 8
        self.goal_dim: int = 0
        self.history: list[dict[str, Any]] = []
        self.goal: Optional[np.ndarray] = None
        self.inference_count = 0
        self.model_info: dict[str, Any] = {}
        self.model_name = weights_path.stem

        self.grounder = GroundingBackend(grounding_model_path, grounding_device, vis_lora_adapter_path=lora_adapter_path)
        self._load(weights_path)

        # grounding 캐시: _grounding_skip_n 스텝마다 1번만 Kosmos-2 실행 (1=캐시 없음)
        self._grounding_skip_n: int = int(os.getenv("VLA_GROUNDING_SKIP_N", "3"))
        self._grounding_cache: Optional[dict] = None
        # 속도 스케일링: bbox area 기반 감속 활성화 여부
        self.speed_scaling_enabled: bool = os.getenv("VLA_SPEED_SCALING", "1") != "0"
        # EMA 스무딩: 연속 스텝 간 액션 블렌딩
        self.smooth_enabled: bool = os.getenv("VLA_SMOOTH", "1") != "0"
        self.smooth_alpha_xy: float = float(os.getenv("VLA_SMOOTH_ALPHA_XY", "0.65"))
        self.smooth_alpha_az: float = float(os.getenv("VLA_SMOOTH_ALPHA_AZ", "0.80"))
        self.prev_action_3d: list[float] = [0.0, 0.0, 0.0]

    def _load(self, weights_path: Path) -> None:
        if not weights_path.exists():
            raise FileNotFoundError(f"GoalNav weights not found: {weights_path}")
        ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)

        if "mlp" in ckpt and "model_state_dict" not in ckpt:
            # exp53 format: {'mlp': full_state_dict, 'val_acc': float}
            state = ckpt["mlp"]
            d_in = state["net.0.weight"].shape[1]
            self.window = 8
            self.goal_dim = d_in - 32 - VIS_DIM  # 1059 - 32 - 1024 = 3
            net = GoalNavMLP(d_in=d_in)
            net.load_state_dict(state)
            overall_acc = ckpt.get("val_acc")
        else:
            # exp49/50/51 format: {'d_in', 'model_state_dict', 'window', ...}
            d_in = int(ckpt["d_in"])
            self.window = int(ckpt.get("window", 8))
            self.goal_dim = int(ckpt.get("goal_dim") or 0)
            net = GoalNavMLP(d_in=d_in)
            net.net.load_state_dict(ckpt["model_state_dict"])
            overall_acc = ckpt.get("overall_acc")

        self.model = net.to(self.device).eval()
        self.model_info = {
            "source": "loaded",
            "weights_path": str(weights_path),
            "d_in": d_in,
            "window": self.window,
            "goal_dim": self.goal_dim,
            "overall_acc": overall_acc,
        }
        logger.info(
            "Loaded GoalNav MLP from %s (d_in=%d, window=%d, goal_dim=%d, acc=%.4f)",
            weights_path, d_in, self.window, self.goal_dim, overall_acc or 0.0,
        )

    def reset(self) -> None:
        self.history.clear()
        self.goal = None
        self.inference_count = 0
        self._grounding_cache = None
        self.prev_action_3d = [0.0, 0.0, 0.0]

    def set_config(
        self,
        speed_scaling: Optional[bool] = None,
        grounding_skip_n: Optional[int] = None,
        smooth_enabled: Optional[bool] = None,
        smooth_alpha_xy: Optional[float] = None,
        smooth_alpha_az: Optional[float] = None,
        stop_area_threshold: Optional[float] = None,
        stop_cx_tolerance: Optional[float] = None,
    ) -> dict:
        if speed_scaling is not None:
            self.speed_scaling_enabled = speed_scaling
        if grounding_skip_n is not None:
            self._grounding_skip_n = max(1, grounding_skip_n)
            self._grounding_cache = None
        if smooth_enabled is not None:
            self.smooth_enabled = smooth_enabled
        if smooth_alpha_xy is not None:
            self.smooth_alpha_xy = max(0.0, min(1.0, smooth_alpha_xy))
        if smooth_alpha_az is not None:
            self.smooth_alpha_az = max(0.0, min(1.0, smooth_alpha_az))
        if stop_area_threshold is not None:
            self._stop_area_thr = max(0.01, min(1.0, stop_area_threshold))
        if stop_cx_tolerance is not None:
            self._stop_cx_tol = max(0.05, min(0.5, stop_cx_tolerance))
        return {
            "speed_scaling_enabled": self.speed_scaling_enabled,
            "grounding_skip_n": self._grounding_skip_n,
            "smooth_enabled": self.smooth_enabled,
            "smooth_alpha_xy": self.smooth_alpha_xy,
            "smooth_alpha_az": self.smooth_alpha_az,
            "stop_area_threshold": getattr(self, "_stop_area_thr", GOAL_AREA_THRESHOLD),
            "stop_cx_tolerance": getattr(self, "_stop_cx_tol", GOAL_CX_TOLERANCE),
        }

    def _build_feature(self, vis_feat: Optional[np.ndarray]) -> np.ndarray:
        feat: list[float] = []
        cur_idx = len(self.history) - 1
        for k in range(self.window):
            idx = max(0, cur_idx - (self.window - 1 - k))
            item = self.history[idx]
            feat.extend([item["cx"], item["cy"], item["area"], float(item["has_bbox"])])
        if vis_feat is not None:
            feat.extend(vis_feat.tolist())
        else:
            # Fallback to zero features if the active VLM does not extract vision features
            feat.extend([0.0] * VIS_DIM)
        if self.goal_dim > 0 and self.goal is not None:
            feat.extend(self.goal.tolist())
        return np.asarray(feat, dtype=np.float32)

    def _decode_image(self, image_base64: str) -> np.ndarray:
        image_bytes = base64.b64decode(image_base64)
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.array(pil)

    def predict(self, image_base64: str, instruction: str, vlm_model: str = "kosmos") -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("GoalNav model not loaded")

        start = time.time()
        image_rgb = self._decode_image(image_base64)

        # Grounding 캐시
        use_cache = (
            self._grounding_skip_n > 1
            and self.inference_count > 0
            and self.inference_count % self._grounding_skip_n != 0
            and self._grounding_cache is not None
            and self._grounding_cache.get("vlm_model") == vlm_model
        )
        if use_cache:
            grounding = self._grounding_cache
            grounding = dict(grounding, latency_ms=0.0)  # 캐시 히트 표시
        else:
            grounding = self.grounder.run(image_rgb, instruction=instruction, vlm_model=vlm_model, extract_vis=True)
            grounding["vlm_model"] = vlm_model
            self._grounding_cache = grounding

        bbox = grounding["bbox"]
        vis_feat: np.ndarray = grounding["vis_feat"]

        # Initialize grounded goal from first frame
        if self.goal_dim > 0 and self.goal is None:
            if bbox is not None:
                self.goal = np.array([bbox["cx"], bbox["cy"], bbox["area"]], dtype=np.float32)
            else:
                self.goal = np.array([0.5, 0.5, 0.0], dtype=np.float32)
            logger.info("GoalNav: goal initialized to [%.3f, %.3f, %.3f]", *self.goal)

        bbox_frame: dict[str, Any] = {
            "cx": float(bbox["cx"]) if bbox else 0.5,
            "cy": float(bbox["cy"]) if bbox else 0.5,
            "area": float(bbox["area"]) if bbox else 0.0,
            "has_bbox": bbox is not None,
        }
        self.history.append(bbox_frame)
        if len(self.history) > self.window:
            self.history = self.history[-self.window:]

        feature = self._build_feature(vis_feat)
        x = torch.tensor(feature[None, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits = self.model(x)
            # ROT_L(6) / ROT_R(7) 마스킹 — 학습 데이터 0%
            if logits.shape[-1] > 6:
                logits = logits.clone()
                logits[:, 6] = float("-inf")
                logits[:, 7] = float("-inf")
            pred_class = int(logits.argmax(dim=-1).item())

        raw_2d = ACTION_2D[pred_class]
        raw_3d = ACTION_3D[pred_class]

        if self.speed_scaling_enabled:
            # area 기반 스케일링은 실제 entity bbox(x1/y1/x2/y2)일 때만 신뢰 가능.
            # caption/coarse 폴백은 area=0.06 하드코딩이라 거리 정보 없음 → 스케일 1.0.
            real_bbox = bbox is not None and "x1" in bbox
            area = bbox_frame["area"]
            if not bbox_frame["has_bbox"]:
                speed_scale = 0.7   # 감지 완전 실패 → 조심
            elif not real_bbox:
                speed_scale = 1.0   # caption/coarse 폴백 → 거리 불명, 풀스피드
            elif area > 0.18:
                speed_scale = 0.25
            elif area > 0.10:
                speed_scale = 0.5
            elif area > 0.05:
                speed_scale = 0.75
            else:
                speed_scale = 1.0
            scaled_2d = [raw_2d[0] * speed_scale, raw_2d[1] * speed_scale]
            scaled_3d = [raw_3d[0] * speed_scale, raw_3d[1] * speed_scale, raw_3d[2] * speed_scale]
        else:
            speed_scale = 1.0
            scaled_2d = list(raw_2d)
            scaled_3d = list(raw_3d)

        # EMA 스무딩: 첫 스텝 이후부터 이전 액션과 블렌딩
        if self.smooth_enabled and self.inference_count > 0:
            p = self.prev_action_3d
            scaled_3d = [
                self.smooth_alpha_xy * scaled_3d[0] + (1 - self.smooth_alpha_xy) * p[0],
                self.smooth_alpha_xy * scaled_3d[1] + (1 - self.smooth_alpha_xy) * p[1],
                self.smooth_alpha_az * scaled_3d[2] + (1 - self.smooth_alpha_az) * p[2],
            ]
            scaled_2d = [scaled_3d[0], scaled_3d[1]]
        self.prev_action_3d = list(scaled_3d)

        self.inference_count += 1
        return {
            "action": scaled_2d,
            "action_3d": scaled_3d,
            "latency_ms": (time.time() - start) * 1000.0,
            "predicted_class": pred_class,
            "predicted_label": CLASS_NAMES[pred_class],
            "bbox": bbox,
            "grounding_caption": grounding["caption"],
            "grounding_latency_ms": grounding["latency_ms"],
            "goal_near_proxy": goal_near_proxy(bbox_frame, getattr(self, "_stop_area_thr", GOAL_AREA_THRESHOLD), getattr(self, "_stop_cx_tol", GOAL_CX_TOLERANCE)),
            "speed_scale": speed_scale,
            "grounding_cached": use_cache,
            "smooth_applied": self.smooth_enabled and self.inference_count > 0,
            "smooth_alpha_xy": self.smooth_alpha_xy,
            "smooth_alpha_az": self.smooth_alpha_az,
            "buffer_status": {
                "history_size": len(self.history),
                "window": self.window,
                "goal": self.goal.tolist() if self.goal is not None else None,
            },
            "source": self.model_info.get("source", "loaded"),
            "instruction_used": self.goal_dim > 0,
            "instruction": instruction,
        }


def resolve_device(raw: str, fallback_cuda: bool = True) -> torch.device:
    raw = raw.strip().lower()
    if raw == "cpu":
        return torch.device("cpu")
    if raw == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if fallback_cuda and torch.cuda.is_available() else "cpu")


def _get_proxy_model(refresh: bool = False) -> ProxyInferenceModel:
    global model_instance
    if refresh:
        model_instance = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if model_instance is None:
        dataset_file = Path(os.getenv("VLA_PROXY_DATASET_FILE", str(DATASET_FILE)))
        weights_path = Path(os.getenv("VLA_PROXY_WEIGHTS_PATH", str(DEFAULT_WEIGHTS_PATH)))
        grounding_model_path = Path(os.getenv("VLA_GROUNDING_MODEL_PATH", str(DEFAULT_GROUNDING_MODEL)))
        proxy_device = resolve_device(os.getenv("VLA_PROXY_DEVICE", "auto"))
        grounding_device = resolve_device(os.getenv("VLA_PROXY_GROUNDING_DEVICE", "auto"))
        epochs = int(os.getenv("VLA_PROXY_TRAIN_EPOCHS", "220"))
        force_retrain = os.getenv("VLA_PROXY_FORCE_RETRAIN", "false").lower() == "true"
        model_instance = ProxyInferenceModel(
            dataset_file=dataset_file,
            weights_path=weights_path,
            proxy_device=proxy_device,
            grounding_device=grounding_device,
            grounding_model_path=grounding_model_path,
            epochs=epochs,
            force_retrain=force_retrain,
        )
    return model_instance


def _get_goal_nav_model(model_name: str, refresh: bool = False) -> GoalNavInferenceModel:
    global _goal_nav_cache, _active_goal_nav_model
    if refresh:
        _goal_nav_cache.pop(model_name, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if model_name not in _goal_nav_cache:
        override = os.getenv("VLA_GOAL_NAV_WEIGHTS_PATH")
        weights_path = Path(override) if override else _GOAL_NAV_WEIGHTS[model_name]
        grounding_model_path = Path(os.getenv("VLA_GROUNDING_MODEL_PATH", str(DEFAULT_GROUNDING_MODEL)))
        device = resolve_device(os.getenv("VLA_GOAL_NAV_DEVICE", "auto"))
        grounding_device = resolve_device(os.getenv("VLA_PROXY_GROUNDING_DEVICE", "auto"))
        lora_path = _GOAL_NAV_LORA_ADAPTERS.get(model_name)
        _goal_nav_cache[model_name] = GoalNavInferenceModel(
            weights_path=weights_path,
            grounding_model_path=grounding_model_path,
            grounding_device=grounding_device,
            device=device,
            lora_adapter_path=lora_path,
        )
    _active_goal_nav_model = model_name
    return _goal_nav_cache[model_name]


def get_model(refresh: bool = False):
    """_active_goal_nav_model → GoalNavInferenceModel. Default → ProxyInferenceModel."""
    if _active_goal_nav_model in _GOAL_NAV_WEIGHTS:
        return _get_goal_nav_model(_active_goal_nav_model, refresh)
    return _get_proxy_model(refresh)


class ModelLoadRequest(BaseModel):
    checkpoint_path: str = str(DEFAULT_WEIGHTS_PATH)
    config_path: str = "N/A"
    precision: str = "fp32"
    refresh: bool = True


@app.get("/")
async def root() -> dict[str, Any]:
    model = _goal_nav_cache.get(_active_goal_nav_model) or model_instance
    model_type = getattr(model, "model_type", "goal_nav") if isinstance(model, GoalNavInferenceModel) else getattr(model, "model_type", "exp19") if model else "exp19"
    return {
        "name": "Mobile VLA Proxy API",
        "version": "0.2.0",
        "status": "running",
        "model_type": model_type,
        "active_model": _active_goal_nav_model,
        "loaded_models": list(_goal_nav_cache.keys()),
        "goal_nav_models": list(_GOAL_NAV_WEIGHTS.keys()),
        "note": "Supports Exp19 (bbox proxy), Exp46/47 (bbox+vision+instruction), Exp49+ (GoalNav). Switch via POST /config {model: expNN}.",
    }


@app.get("/model/info")
async def model_info() -> dict[str, Any]:
    model = get_model()
    if model is None:
        return {
            "model_loaded": False,
            "model_name": "proxy_mlp",
            "checkpoint_path": str(DEFAULT_WEIGHTS_PATH),
            "config_path": "N/A",
            "precision": "fp32",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "action_dim": 3,
        }
    if isinstance(model, GoalNavInferenceModel):
        return {
            "model_loaded": True,
            "model_name": model.model_name,
            "checkpoint_path": model.model_info.get("weights_path", "N/A"),
            "config_path": "N/A",
            "precision": "fp32",
            "device": str(model.device),
            "action_dim": 3,
            "model_type": "goal_nav",
            "strategy": "goal_nav",
            "goal_nav_info": model.model_info,
        }
    return {
        "model_loaded": True,
        "model_name": model.model_name,
        "checkpoint_path": str(model.weights_path),
        "config_path": "N/A",
        "precision": "fp32",
        "device": str(model.proxy_device),
        "action_dim": 3,
        "model_type": model.model_type,
        "effective_window": model.effective_window,
        "proxy_info": model.model_info,
    }


@app.post("/model/load")
async def load_model(request: ModelLoadRequest) -> dict[str, Any]:
    os.environ["VLA_PROXY_WEIGHTS_PATH"] = request.checkpoint_path
    get_model(refresh=request.refresh)
    model = model_instance
    return {
        "status": "success",
        "message": f"Proxy model loaded from {request.checkpoint_path}",
        "model_name": model.model_name if model else "unknown",
        "model_type": getattr(model, "model_type", "exp19") if model else "exp19",
        "checkpoint_path": request.checkpoint_path,
        "precision": request.precision,
    }


@app.get("/health")
async def health_check() -> dict[str, Any]:
    gpu_memory = None
    if torch.cuda.is_available():
        gpu_memory = {
            "allocated_gb": float(torch.cuda.memory_allocated() / 1024**3),
            "reserved_gb": float(torch.cuda.memory_reserved() / 1024**3),
            "device_name": torch.cuda.get_device_name(0),
        }

    active = _goal_nav_cache.get(_active_goal_nav_model) or model_instance
    return {
        "status": "healthy",
        "active_model": _active_goal_nav_model,
        "loaded_models": list(_goal_nav_cache.keys()),
        "model_loaded": active is not None,
        "model_name": None if active is None else getattr(active, "model_name", "unknown"),
        "gpu_memory": gpu_memory,
        "model_info": None if active is None else getattr(active, "model_info", {}),
    }


def _verify_api_key(x_api_key: Optional[str]) -> None:
    expected = os.getenv("VLA_API_KEY", "")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=403, detail="Invalid API Key")


@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest, x_api_key: Optional[str] = Header(default=None)) -> InferenceResponse:
    _verify_api_key(x_api_key)
    try:
        model = get_model()
        result = model.predict(request.image, request.instruction, vlm_model=request.vlm_model)
        strategy = "goal_nav" if isinstance(model, GoalNavInferenceModel) else "proxy_mlp"
        response = InferenceResponse(
            action=result["action"],
            latency_ms=result["latency_ms"],
            model_name=model.model_name,
            strategy=strategy,
            source=result["source"],
            buffer_status=result["buffer_status"],
            predicted_class=result["predicted_class"],
            predicted_label=result["predicted_label"],
            action_3d=result["action_3d"],
            bbox=result["bbox"],
            grounding_caption=result["grounding_caption"],
            grounding_latency_ms=result["grounding_latency_ms"],
            goal_near_proxy=result["goal_near_proxy"],
            instruction_used=result["instruction_used"],
            matched_path_type=result.get("matched_path_type"),
        )
        _recent_predictions.append({
            "ts": time.strftime("%H:%M:%S"),
            "label": result["predicted_label"],
            "cls": result["predicted_class"],
            "latency_ms": round(result["latency_ms"], 1),
            "bbox": result["bbox"],
            "caption": (result["grounding_caption"] or "")[:60],
            "path_type": result.get("matched_path_type"),
            "instruction": result["instruction_used"],
        })
        return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Proxy prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/reset")
async def reset_history(x_api_key: Optional[str] = Header(default=None)) -> dict[str, Any]:
    _verify_api_key(x_api_key)
    model = get_model()
    model.reset()
    return {"status": "success", "message": "Proxy history reset"}


@app.post("/config")
async def set_config(
    request: ConfigRequest,
    x_api_key: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    _verify_api_key(x_api_key)
    # 모델 전환 먼저 처리
    if request.model is not None:
        if request.model not in _GOAL_NAV_WEIGHTS:
            return {"status": "error", "reason": f"Unknown model: {request.model}. Available: {list(_GOAL_NAV_WEIGHTS.keys())}"}
        _get_goal_nav_model(request.model)  # 캐시 확보 + _active_goal_nav_model 갱신
        logger.info("Model switched to %s", request.model)
    model = get_model()
    if isinstance(model, GoalNavInferenceModel):
        cfg = model.set_config(
            speed_scaling=request.speed_scaling,
            grounding_skip_n=request.grounding_skip_n,
            smooth_enabled=request.smooth_enabled,
            smooth_alpha_xy=request.smooth_alpha_xy,
            smooth_alpha_az=request.smooth_alpha_az,
        )
        cfg["active_model"] = _active_goal_nav_model
        return {"status": "success", "config": cfg}
    return {"status": "skipped", "reason": "Not a GoalNavInferenceModel"}


@app.get("/recent")
async def recent_predictions() -> dict[str, Any]:
    return {"count": len(_recent_predictions), "predictions": list(reversed(_recent_predictions))}


_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>VLA Proxy Dashboard</title>
<style>
  body { font-family: monospace; background: #0d1117; color: #c9d1d9; margin: 0; padding: 20px; }
  h1 { color: #58a6ff; margin-bottom: 4px; }
  .sub { color: #8b949e; font-size: 13px; margin-bottom: 20px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 16px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
  .card h3 { margin: 0 0 10px; color: #58a6ff; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; }
  .pill { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; }
  .green { background: #1a4731; color: #3fb950; }
  .red { background: #4b1b1b; color: #f85149; }
  .yellow { background: #3d2c00; color: #e3b341; }
  .kv { display: flex; justify-content: space-between; padding: 3px 0; border-bottom: 1px solid #21262d; font-size: 13px; }
  .kv:last-child { border-bottom: none; }
  .kv .val { color: #e6edf3; }
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th { color: #8b949e; text-align: left; padding: 4px 8px; border-bottom: 1px solid #30363d; }
  td { padding: 4px 8px; border-bottom: 1px solid #21262d; }
  tr:hover td { background: #1c2128; }
  .action-tag { font-weight: bold; color: #58a6ff; }
  .ts { color: #8b949e; }
  #status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #8b949e; margin-right: 6px; }
  .dot-ok { background: #3fb950 !important; }
  .dot-err { background: #f85149 !important; }
  button { background: #21262d; color: #c9d1d9; border: 1px solid #30363d; border-radius: 6px; padding: 6px 14px; cursor: pointer; font-size: 12px; }
  button:hover { background: #30363d; }
</style>
</head>
<body>
<h1>🤖 VLA Proxy Dashboard</h1>
<div class="sub"><span id="status-dot"></span><span id="status-txt">연결 중...</span> &nbsp;|&nbsp; 2초마다 자동 갱신 &nbsp;|&nbsp; <a href="/docs" style="color:#58a6ff">Swagger</a></div>

<div class="grid">
  <div class="card">
    <h3>서버 상태</h3>
    <div id="health-body">로딩...</div>
  </div>
  <div class="card">
    <h3>모델 정보</h3>
    <div id="model-body">로딩...</div>
  </div>
</div>

<div class="card">
  <h3>최근 예측 <span id="pred-count" style="color:#8b949e; font-weight:normal"></span>
    &nbsp;<button onclick="resetHistory()">↺ history 리셋</button>
  </h3>
  <table>
    <thead><tr><th>시각</th><th>액션</th><th>cls</th><th>latency</th><th>path_type</th><th>bbox</th><th>caption</th></tr></thead>
    <tbody id="pred-body"><tr><td colspan="7" style="color:#8b949e">예측 없음</td></tr></tbody>
  </table>
</div>

<script>
const API_KEY = localStorage.getItem("vla_api_key") || "";

function kv(k, v) {
  return `<div class="kv"><span>${k}</span><span class="val">${v ?? "—"}</span></div>`;
}
function pill(ok, t) {
  return `<span class="pill ${ok ? "green" : "red"}">${t}</span>`;
}

async function fetchHealth() {
  try {
    const r = await fetch("/health");
    const d = await r.json();
    const dot = document.getElementById("status-dot");
    dot.className = "dot-ok";
    document.getElementById("status-txt").textContent = "정상 (" + new Date().toLocaleTimeString() + ")";

    const gpu = d.gpu_memory;
    document.getElementById("health-body").innerHTML =
      kv("상태", pill(true, "healthy")) +
      kv("모델 로드", pill(d.model_loaded, d.model_loaded ? "로드됨" : "미로드")) +
      kv("모델명", d.model_name) +
      (gpu ? kv("GPU", `${gpu.device_name}`) : "") +
      (gpu ? kv("GPU 사용", `${gpu.allocated_gb.toFixed(2)} GB / ${gpu.reserved_gb.toFixed(2)} GB`) : kv("GPU", "없음"));
  } catch(e) {
    document.getElementById("status-dot").className = "dot-err";
    document.getElementById("status-txt").textContent = "서버 응답 없음";
    document.getElementById("health-body").innerHTML = `<span class="pill red">연결 실패</span>`;
  }
}

async function fetchModel() {
  try {
    const r = await fetch("/model/info");
    const d = await r.json();
    document.getElementById("model-body").innerHTML =
      kv("타입", d.model_type) +
      kv("디바이스", d.device) +
      kv("체크포인트", (d.checkpoint_path || "").split("/").slice(-1)[0]);
  } catch(e) {}
}

async function fetchRecent() {
  try {
    const r = await fetch("/recent");
    const d = await r.json();
    const preds = d.predictions;
    document.getElementById("pred-count").textContent = `(${d.count}건)`;
    if (!preds.length) return;
    document.getElementById("pred-body").innerHTML = preds.map(p =>
      `<tr>
        <td class="ts">${p.ts}</td>
        <td class="action-tag">${p.label}</td>
        <td>${p.cls}</td>
        <td>${p.latency_ms}ms</td>
        <td>${p.path_type ?? "—"}</td>
        <td>${p.bbox ? JSON.stringify(p.bbox).slice(0,30) : "NO_BBOX"}</td>
        <td style="color:#8b949e">${p.caption}</td>
      </tr>`
    ).join("");
  } catch(e) {}
}

async function resetHistory() {
  await fetch("/reset", { method: "POST", headers: { "X-API-Key": API_KEY } });
  fetchRecent();
}

async function refresh() {
  await Promise.all([fetchHealth(), fetchModel(), fetchRecent()]);
}

refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>"""


@app.get("/dashboard", response_class=None)
async def dashboard():
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=_DASHBOARD_HTML)


@app.get("/test")
async def test_endpoint() -> dict[str, Any]:
    dummy = Image.new("RGB", (224, 224), color=(128, 128, 128))
    buf = io.BytesIO()
    dummy.save(buf, format="PNG")
    return {
        "message": "Test endpoint",
        "image_b64_len": len(base64.b64encode(buf.getvalue()).decode("utf-8")),
        "note": "Use POST /predict for real inference.",
    }


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.getenv("VLA_PORT", "8000")))
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    vla_model = os.getenv("VLA_MODEL", "exp19")
    logger.info("Pre-loading model (VLA_MODEL=%s) ...", vla_model)
    m = get_model()
    logger.info("Model ready (%s). Starting uvicorn...", m.model_name if m else vla_model)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

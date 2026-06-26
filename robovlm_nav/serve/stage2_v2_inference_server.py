"""
Stage2 v2 inference server (Exp54/66/67/68/69/70).

Pipeline: Kosmos-2 vision encoder → image_proj (256-dim, L2-norm) + bbox history → ActionHead
Grounding: PaliGemma2 "detect gray basket" (matches training distribution, Exp65/66)

Architecture: FrozenCLIPV2 (Stage1) + ActionMLP/LSTMHead/FCHead/LinearHead (Stage2)

Environment variables:
  VLA_S2V2_STAGE1           path to stage1_v2_projs.pt
  VLA_S2V2_STAGE2           path to stage2 checkpoint (e.g. stage2_v2_mlp_base_pg2_aug.pt)
  VLA_S2V2_HEAD             head type: mlp | linear | fc | lstm (auto-detected from ckpt)
  VLA_GROUNDING_MODEL_PATH  path to Kosmos-2 model dir (.vlms/kosmos-2-patch14-224)
  VLA_PG2_PATH              path to PaliGemma2 model dir (default: HF cache)
  VLA_PORT                  server port (default: 8001)
  VLA_GROUNDING_SKIP_N      run grounding every N steps, cache in between (default: 3, CH49 확정)
  VLA_STOP_MODE             STOP 결정 방식: proximity (기본) | learned
                              proximity: area+cx threshold로 강제 override
                              learned:   모델이 STOP(class 0) 예측 시 latch (한 번 멈추면 유지)
  VLA_STOP_AREA             bbox area threshold for proximity STOP (default: 0.25)
  VLA_STOP_CX_TOL           cx tolerance from center for proximity STOP (default: 0.35)
  VLA_STOP_CONSEC           consecutive frames required for STOP override (default: 2)
  VLA_API_KEY               optional API key for authentication

Usage:
  # 기본 (proximity override)
  .venv/bin/python3 robovlm_nav/serve/stage2_v2_inference_server.py

  # learned STOP (sw1x ckpt + latch)
  VLA_STOP_MODE=learned \
  VLA_S2V2_STAGE2=runs/v5_nav/mlp/stop_weighted/stop_wt_sw1x.pt \
    .venv/bin/python3 robovlm_nav/serve/stage2_v2_inference_server.py --port 8001
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Header
from PIL import Image
from pydantic import BaseModel

from robovlm_nav.image_preprocess import resize_for_vlm

import re

try:
    from transformers import (
        AutoModelForVision2Seq, AutoProcessor,
        PaliGemmaProcessor, PaliGemmaForConditionalGeneration,
    )
except ImportError as exc:
    raise RuntimeError("transformers is required") from exc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Stage2 v2 VLA API", version="1.0.0")

ROOT = Path(project_root)

# --- defaults ---
DEFAULT_STAGE1 = ROOT / "runs" / "v5_nav" / "mlp" / "shared" / "stage1_v2_projs.pt"
DEFAULT_STAGE2 = ROOT / "runs" / "v5_nav" / "mlp" / "exp66" / "action_mlp.pt"
DEFAULT_VLM    = ROOT / ".vlms" / "kosmos-2-patch14-224"
# PaliGemma2: HF cache path (used for grounding, matches Exp65/66 training distribution)
_PG2_HF_CACHE = (
    Path.home() / ".cache" / "huggingface" / "hub"
    / "models--google--paligemma2-3b-mix-224"
    / "snapshots" / "8e40ab4cc5df93dfb7fd2fff754bcdff8b62ee78"
)
DEFAULT_PG2 = Path(os.getenv("VLA_PG2_PATH", str(_PG2_HF_CACHE)))

NUM_CLASSES = 8
WINDOW_DEFAULT = 8
VIS_DIM  = 1024
PROJ_DIM = 256
SEQ_DIM  = PROJ_DIM + 4   # 260: per-frame LSTM input
HIDDEN_DIM = 2304         # PG2 last-layer hidden state dim (plan_20260622_hidden_state_*)

# CH40 hidden-state head 체크포인트 — 없으면 head_mode="add"/"replace" 요청 시 baseline로 폴백
HIDDEN_ADD_CKPT     = ROOT / "runs" / "v5_nav" / "mlp" / "exp_hidden_state" / "stage2_v2" / "stage2_hidden_add.pt"
HIDDEN_REPLACE_CKPT = ROOT / "runs" / "v5_nav" / "mlp" / "exp_hidden_state" / "stage2_v2" / "stage2_hidden_replace.pt"

CLASS_NAMES = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "ROT_L", "ROT_R"]

ACTION_2D = {
    0: [0.0, 0.0], 1: [1.15, 0.0], 2: [0.0, 1.15], 3: [0.0, -1.15],
    4: [1.15, 1.15], 5: [1.15, -1.15], 6: [0.0, 0.0], 7: [0.0, 0.0],
}
ACTION_3D = {
    0: [0.0, 0.0, 0.0], 1: [1.15, 0.0, 0.0], 2: [0.0, 1.15, 0.0], 3: [0.0, -1.15, 0.0],
    4: [1.15, 1.15, 0.0], 5: [1.15, -1.15, 0.0], 6: [0.0, 0.0, 0.25], 7: [0.0, 0.0, -0.25],
}

FULLSCREEN_AREA_THRESHOLD = 0.85
_LOC_RE = re.compile(r"<loc(\d{4})>")
# Stop-proximity thresholds (tuned from PG2 grounding on last frames: area≈0.25-0.46 vs mid 0.08-0.10)
GOAL_AREA_THRESHOLD = float(os.getenv("VLA_STOP_AREA", "0.25"))
GOAL_CX_TOLERANCE   = float(os.getenv("VLA_STOP_CX_TOL", "0.35"))
# Temporal filter: require this many CONSECUTIVE frames all satisfying near-goal condition.
# history가 N프레임 미만이면 발동 안 함 — 초반 false positive 차단.
GOAL_CONSEC_FRAMES  = int(os.getenv("VLA_STOP_CONSEC", "3"))
# STOP mode: "proximity" (threshold-based override) | "learned" (model prediction + latch)
STOP_MODE = os.getenv("VLA_STOP_MODE", "proximity")

# 객체별 GOAL_AREA 매핑 — instruction(grounding phrase) → 정지 area 임계값.
# GOAL_AREA_THRESHOLD(0.25)는 바스켓 실주행으로 캘리브레이션된 값이라 다른(특히 작은) 객체에
# 그대로 쓰면 너무 늦게 멈춤(충돌 위험). configs/goal_area_map.json이 있으면 로드해 사용,
# 없는 키는 GOAL_AREA_THRESHOLD로 폴백. 캘리브레이션: scripts/calibrate_goal_area.py.
GOAL_AREA_MAP_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "goal_area_map.json"


def _load_goal_area_map() -> dict[str, float]:
    if GOAL_AREA_MAP_PATH.exists():
        try:
            return json.loads(GOAL_AREA_MAP_PATH.read_text())
        except Exception as e:
            logger.warning("goal_area_map.json 로드 실패: %s — 기본값만 사용", e)
    return {}


GOAL_AREA_MAP = _load_goal_area_map()


def get_goal_area(phrase: str) -> float:
    return GOAL_AREA_MAP.get(phrase, GOAL_AREA_THRESHOLD)


# PG2Grounder.run()의 후처리 필터 중 cy_val/area 하한은 바스켓 전용 휴리스틱이라
# (바구니는 항상 화면 하단~중단에 있다는 가정) 다른 객체에선 정상 bbox를 false negative
# 처리할 수 있음(예: 사과 cy=0.344로 기본값 0.35 미달 → 탈락). configs/goal_area_map.json과
# 같은 파일에 phrase별 오버라이드를 둠 — 없는 phrase는 바스켓 기본값(min_cy=0.35, min_area=0.01) 유지.
GROUND_FILTER_DEFAULTS = {"min_cy": 0.35, "min_area": 0.01}
GROUND_FILTER_MAP_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "ground_filter_map.json"


def _load_ground_filter_map() -> dict[str, dict[str, float]]:
    if GROUND_FILTER_MAP_PATH.exists():
        try:
            return json.loads(GROUND_FILTER_MAP_PATH.read_text())
        except Exception as e:
            logger.warning("ground_filter_map.json 로드 실패: %s — 기본값만 사용", e)
    return {}


GROUND_FILTER_MAP = _load_ground_filter_map()


def get_ground_filters(phrase: str) -> dict[str, float]:
    override = GROUND_FILTER_MAP.get(phrase, {})
    return {**GROUND_FILTER_DEFAULTS, **override}


# ---------------------------------------------------------------------------
# Head models (mirror of train_exp54_stage2_v2_action.py)
# ---------------------------------------------------------------------------

class ActionMLP(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 256), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(256, 128),  nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),   nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, NUM_CLASSES),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearHead(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Linear(d_in, NUM_CLASSES)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FCHead(nn.Module):
    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 1024), nn.ReLU(),
            nn.Linear(1024, 512),  nn.ReLU(),
            nn.Linear(512, 256),   nn.ReLU(),
            nn.Linear(256, NUM_CLASSES),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LSTMHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(SEQ_DIM, 256, 2, batch_first=True, dropout=0.1)
        self.classifier = nn.Linear(256, NUM_CLASSES)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1])


HEAD_REGISTRY: dict[str, type] = {
    "mlp": ActionMLP, "linear": LinearHead, "fc": FCHead, "lstm": LSTMHead
}


# ---------------------------------------------------------------------------
# Stage1 encoder
# ---------------------------------------------------------------------------

class Stage1Encoder(nn.Module):
    """Kosmos-2 vision_model + image_proj (256-dim, L2-norm). Frozen."""

    def __init__(self, vlm_path: Path, ckpt_path: Path, device: torch.device):
        super().__init__()
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        logger.info("Stage1 val_acc=%.4f", ckpt["val_acc"])
        self.processor = AutoProcessor.from_pretrained(str(vlm_path))
        base = AutoModelForVision2Seq.from_pretrained(str(vlm_path), torch_dtype=torch.float16)
        self.vision_model = base.vision_model.to(device)
        self.image_proj   = nn.Linear(VIS_DIM, PROJ_DIM).to(device)
        self.image_proj.load_state_dict(ckpt["image_proj"])
        for p in self.vision_model.parameters(): p.requires_grad = False
        for p in self.image_proj.parameters():   p.requires_grad = False
        self._device = device

    @torch.no_grad()
    def encode_image(self, pil_image: Image.Image) -> torch.Tensor:
        """PIL RGB → (256,) float32 L2-normalized tensor on device."""
        pil_image = resize_for_vlm(pil_image)
        inputs = self.processor(images=pil_image, return_tensors="pt")
        pv = inputs["pixel_values"].to(self._device, dtype=torch.float16)
        out = self.vision_model(pixel_values=pv)
        feat = out.last_hidden_state.mean(dim=1).float()         # (1, 1024)
        return F.normalize(self.image_proj(feat), dim=-1)[0]     # (256,)

    @torch.no_grad()
    def extract_vis_feat_raw(self, pil_image: Image.Image) -> torch.Tensor:
        """Kosmos-2 raw 1024-dim feature (needed for grounding in shared model)."""
        inputs = self.processor(text=GROUNDING_PROMPT, images=pil_image, return_tensors="pt")
        pv = inputs["pixel_values"].to(self._device, dtype=torch.float16)
        out = self.vision_model(pixel_values=pv)
        return out.last_hidden_state[0].mean(0).float()          # (1024,)


# ---------------------------------------------------------------------------
# Grounding (Kosmos-2 generate → bbox)
# ---------------------------------------------------------------------------

class Grounder:
    """Wraps Kosmos-2 model for bounding-box extraction. Shares vision model with Stage1."""

    def __init__(self, stage1: Stage1Encoder):
        self._stage1 = stage1
        # Grab the full Kosmos model so we can call generate()
        self._processor = stage1.processor
        # We need the full model (not just vision) for generate() — lazy-load
        self._full_model: Optional[Any] = None
        self._device = stage1._device

    def _ensure_full_model(self, vlm_path: Path) -> None:
        if self._full_model is None:
            base = AutoModelForVision2Seq.from_pretrained(
                str(vlm_path), torch_dtype=torch.float16
            ).to(self._device).eval()
            self._full_model = base
            logger.info("Grounder: full Kosmos-2 model loaded for generate()")

    def run(self, image_rgb: np.ndarray, vlm_path: Optional[Path] = None) -> dict[str, Any]:
        """Run Kosmos-2 grounding → {'cx', 'cy', 'area', 'has_bbox'}."""
        self._ensure_full_model(vlm_path or Path(str(DEFAULT_VLM)))
        pil = Image.fromarray(image_rgb.astype(np.uint8)).convert("RGB")
        pil = resize_for_vlm(pil)
        inputs = self._processor(text=GROUNDING_PROMPT, images=pil, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        pixel_values = inputs["pixel_values"].to(torch.float16)

        with torch.no_grad():
            generated = self._full_model.generate(
                pixel_values=pixel_values,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_embeds=None,
                image_embeds_position_mask=inputs.get("image_embeds_position_mask"),
                use_cache=True,
                max_new_tokens=64,
            )

        new_ids = generated[:, inputs["input_ids"].shape[1]:]
        raw = self._processor.batch_decode(new_ids, skip_special_tokens=False)[0]
        caption, entities = self._processor.post_process_generation(raw)
        bbox = self._parse_bbox(caption, entities)
        if bbox is None:
            bbox = {"cx": 0.5, "cy": 0.6, "area": 0.06, "has_bbox": False}
        else:
            bbox["has_bbox"] = True
        return bbox

    def _parse_bbox(self, caption: str, entities: list) -> Optional[dict[str, Any]]:
        for entity_name, _span, boxes in entities:
            for box in boxes:
                x1, y1, x2, y2 = [float(v) for v in box]
                if max(x1, y1, x2, y2) > 1.5:
                    x1, y1, x2, y2 = x1/1000.0, y1/1000.0, x2/1000.0, y2/1000.0
                area = (x2 - x1) * (y2 - y1)
                if area > FULLSCREEN_AREA_THRESHOLD:
                    continue
                if "basket" in entity_name.lower() or "container" in entity_name.lower():
                    return {"cx": (x1+x2)/2, "cy": (y1+y2)/2, "area": area}
        return None


# ---------------------------------------------------------------------------
# Grounding (PaliGemma2 "detect gray basket" → bbox)
# Matches Exp65/66 training distribution; more reliable than Kosmos-2 for basket detection.
# ---------------------------------------------------------------------------

class PG2Grounder:
    """PaliGemma2-based bbox grounder using 'detect gray basket' prompt."""

    def __init__(self, pg2_path: Path, device: torch.device):
        self._device = device
        self._proc: Optional[Any] = None
        self._model: Optional[Any] = None
        self._pg2_path = pg2_path
        self._dtype = torch.bfloat16

    def _ensure_loaded(self) -> None:
        if self._model is None:
            logger.info("PG2Grounder: loading PaliGemma2 from %s", self._pg2_path)
            self._proc = PaliGemmaProcessor.from_pretrained(str(self._pg2_path))
            self._model = PaliGemmaForConditionalGeneration.from_pretrained(
                str(self._pg2_path), torch_dtype=self._dtype, low_cpu_mem_usage=True
            ).to(self._device).eval()
            logger.info("PG2Grounder: ready")

    def run(self, image_rgb: np.ndarray, _unused_path: Optional[Path] = None,
            return_raw: bool = False, phrase: str = "gray basket",
            return_hidden: bool = False) -> dict[str, Any]:
        """Run PaliGemma2 grounding → {'cx', 'cy', 'area', 'has_bbox', 'raw_output'}.

        return_hidden=True면 같은 generate() 호출의 prefill 단계 hidden state(2304차원,
        마지막 레이어·마지막 입력 위치)도 같이 반환 — 별도 forward 없음
        (plan_20260622_hidden_state_hub_integration.md §1-2).
        """
        self._ensure_loaded()
        pil = Image.fromarray(image_rgb.astype(np.uint8)).convert("RGB")
        pil = resize_for_vlm(pil)
        inp = self._proc(text=f"detect {phrase}", images=pil, return_tensors="pt").to(self._device)
        inp["pixel_values"] = inp["pixel_values"].to(self._dtype)
        hidden_vec = None
        with torch.no_grad():
            if return_hidden:
                out = self._model.generate(
                    **inp, max_new_tokens=48, min_new_tokens=1, do_sample=False,
                    output_hidden_states=True, return_dict_in_generate=True,
                )
                gen = out.sequences
                hidden_vec = out.hidden_states[0][-1][0, -1, :].float().cpu().numpy()
            else:
                gen = self._model.generate(**inp, max_new_tokens=48, min_new_tokens=1, do_sample=False)
        raw = self._proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=False)[0]
        locs = [int(v) / 1023.0 for v in _LOC_RE.findall(raw)]
        if len(locs) >= 4:
            y1, x1, y2, x2 = locs[:4]
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            area = (x2 - x1) * (y2 - y1)
            cx_val = (x1 + x2) / 2
            cy_val = (y1 + y2) / 2
            _fallback = {"cx": 0.5, "cy": 0.6, "area": 0.06, "has_bbox": False,
                         "x1": None, "y1": None, "x2": None, "y2": None}
            # 학습 annotation 필터 4종 — gen_base_pg2_annotation.py 기준 맞춤
            # area>0.9(full-frame 환각)/x-full-width는 객체 무관 보편 규칙으로 유지.
            # min_area/min_cy는 바스켓 전용 휴리스틱(바구니는 항상 화면 하단~중단)이라
            # phrase별로 오버라이드 가능하게 함(configs/ground_filter_map.json).
            filters = get_ground_filters(phrase)
            if area > 0.9:          # full-frame collapse (loc0000~loc1022 전체)
                result = _fallback
            elif area < filters["min_area"]:       # tiny noise detection
                result = _fallback
            elif cy_val < filters["min_cy"]:     # 상단 오탐 (바구니가 프레임 상단에 있을 수 없음)
                result = _fallback
            elif x1 < 0.02 and x2 > 0.98:  # x-full-width collapse (cx≈0.5 항상)
                result = _fallback
            else:
                result = {"cx": cx_val, "cy": cy_val, "area": area, "has_bbox": True,
                          "x1": x1, "y1": y1, "x2": x2, "y2": y2}
        else:
            result = {"cx": 0.5, "cy": 0.6, "area": 0.06, "has_bbox": False,
                      "x1": None, "y1": None, "x2": None, "y2": None}
        if return_raw:
            result["raw_output"] = raw
        if return_hidden:
            result["hidden_state"] = hidden_vec
        return result


# ---------------------------------------------------------------------------
# Online inference model
# ---------------------------------------------------------------------------

class Stage2V2Model:
    """Online inference: maintains bbox history, predicts action per frame."""

    def __init__(
        self,
        stage1_path: Path,
        stage2_path: Path,
        vlm_path: Path,
        head_override: Optional[str],
        device: torch.device,
        pg2_path: Optional[Path] = None,
    ):
        self.device = device
        self.vlm_path = vlm_path
        self.inference_count = 0
        self._grounding_skip_n: int = int(os.getenv("VLA_GROUNDING_SKIP_N", "3"))  # CH49: skip_n=3 SR/FPE 변화 없음 확정
        self._grounding_cache: Optional[dict] = None

        # Stage1 (Kosmos-2 vision encoder — image features only)
        self.enc = Stage1Encoder(vlm_path, stage1_path, device)
        self.enc.eval()

        # Grounder: PG2 if available (matches training), Kosmos-2 fallback
        _pg2 = pg2_path or DEFAULT_PG2
        if _pg2.exists():
            self.grounder: Any = PG2Grounder(_pg2, device)
            logger.info("Grounder: PaliGemma2 (%s)", _pg2)
        else:
            self.grounder = Grounder(self.enc)
            logger.warning("PG2 not found at %s — falling back to Kosmos-2 Grounder (has_bbox=False always)", _pg2)

        # Stage2 head
        ckpt = torch.load(str(stage2_path), map_location=device, weights_only=False)
        self.window: int = int(ckpt.get("window", WINDOW_DEFAULT))
        head_name: str = head_override or ckpt.get("head", "mlp")
        is_lstm = (head_name == "lstm")
        d_in = self.window * 4 + PROJ_DIM
        HeadCls = HEAD_REGISTRY[head_name]
        self.head: nn.Module = (HeadCls() if is_lstm else HeadCls(d_in=d_in)).to(device)
        self.head.load_state_dict(ckpt["mlp"])
        self.head.eval()
        self.is_lstm = is_lstm
        self.head_name = head_name
        self.val_acc: float = float(ckpt.get("val_acc", 0.0))

        logger.info(
            "Stage2V2 ready — head=%s window=%d d_in=%d val_acc=%.4f",
            head_name, self.window, d_in, self.val_acc,
        )
        logger.info("Stage2 ckpt: %s", stage2_path)

        # CH40 hidden-state head 변형(add/replace) — 있으면 같이 로드, 없으면 head_mode 요청 시 baseline 폴백.
        # plan_20260622_hidden_state_hub_integration.md §1-1 — 기본 head_mode="baseline"은 동작 불변.
        self.hidden_heads: dict[str, nn.Module] = {}
        for mode, ckpt_path, mode_d_in in [
            ("add", HIDDEN_ADD_CKPT, self.window * 4 + PROJ_DIM + HIDDEN_DIM),
            ("replace", HIDDEN_REPLACE_CKPT, PROJ_DIM + HIDDEN_DIM),
        ]:
            if ckpt_path.exists():
                h_ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
                h_mlp = ActionMLP(d_in=h_ckpt.get("d_in", mode_d_in)).to(device)
                h_mlp.load_state_dict(h_ckpt["mlp"])
                h_mlp.eval()
                self.hidden_heads[mode] = h_mlp
                logger.info("Hidden-state head[%s] 로드 — val_acc=%.4f", mode, h_ckpt.get("val_acc", 0.0))
            else:
                logger.info("Hidden-state head[%s] 체크포인트 없음(%s) — head_mode='%s' 요청 시 baseline 폴백", mode, ckpt_path, mode)

        # Rolling history: list of {cx, cy, area, has_bbox, vis_feat}
        self.history: list[dict] = []

        # Learned-STOP latch: once model predicts STOP(0), stay stopped until reset()
        self.stop_latched: bool = False

    def reset(self) -> None:
        self.history.clear()
        self.inference_count = 0
        self._grounding_cache = None
        self.stop_latched = False

    def _decode_image(self, image_b64: str) -> np.ndarray:
        image_bytes = base64.b64decode(image_b64)
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return np.array(pil)

    def _bbox_frame(self, bbox: dict) -> dict:
        return {
            "cx":      float(bbox.get("cx",  0.5)),
            "cy":      float(bbox.get("cy",  0.6)),
            "area":    float(bbox.get("area", 0.0)),
            "has_bbox": bool(bbox.get("has_bbox", False)),
        }

    def _build_flat_feature(self, vis_feat: torch.Tensor) -> torch.Tensor:
        """MLP/Linear/FC: [window*4 bbox + PROJ_DIM] flat vector."""
        bbox_parts = []
        for k in range(self.window):
            idx = max(0, len(self.history) - 1 - (self.window - 1 - k))
            idx = min(idx, len(self.history) - 1)
            item = self.history[idx]
            bbox_parts.extend([item["cx"], item["cy"], item["area"], float(item["has_bbox"])])
        bbox_t = torch.tensor(bbox_parts, dtype=torch.float32, device=self.device)
        return torch.cat([bbox_t, vis_feat])  # (d_in,)

    def _build_seq_feature(self) -> torch.Tensor:
        """LSTM: (window, SEQ_DIM) sequence."""
        seq = []
        for k in range(self.window):
            idx = max(0, len(self.history) - 1 - (self.window - 1 - k))
            idx = min(idx, len(self.history) - 1)
            item = self.history[idx]
            vf = item.get("vis_feat")
            if vf is None:
                vf = torch.zeros(PROJ_DIM, device=self.device)
            bbox_t = torch.tensor(
                [item["cx"], item["cy"], item["area"], float(item["has_bbox"])],
                dtype=torch.float32, device=self.device,
            )
            seq.append(torch.cat([vf, bbox_t]))  # (SEQ_DIM,)
        return torch.stack(seq, dim=0)  # (window, SEQ_DIM)

    def predict(self, image_b64: str, instruction: str = "basket",
                head_mode: str = "baseline") -> dict[str, Any]:
        start = time.time()
        image_rgb = self._decode_image(image_b64)
        pil = Image.fromarray(image_rgb.astype(np.uint8)).convert("RGB")

        # CH40 hidden-state head 사용 여부 — 체크포인트 없으면 baseline으로 자동 폴백
        # (plan_20260622_hidden_state_hub_integration.md §1-1).
        use_hidden = head_mode in ("add", "replace") and head_mode in self.hidden_heads
        effective_head_mode = head_mode if use_hidden else "baseline"

        # Grounding (with optional caching) — hidden state 모드는 항상 새로 계산(캐시에 hidden_state가
        # 없을 수 있어 단순화를 위해 skip-cache 비적용, 데모/테스트 용도라 비용 영향 적음).
        use_cache = (
            not use_hidden
            and self._grounding_skip_n > 1
            and self.inference_count > 0
            and self.inference_count % self._grounding_skip_n != 0
            and self._grounding_cache is not None
        )
        # instruction="basket"(기본값)은 기존 하위호환 placeholder — "gray basket"으로 매핑.
        # 그 외 값은 grounding 프롬프트 phrase로 그대로 사용 (예: "red ball", "blue mug").
        phrase = "gray basket" if instruction == "basket" else instruction

        g_start = time.time()
        if use_cache:
            bbox = self._grounding_cache
            grounding_latency_ms = 0.0
        else:
            bbox = self.grounder.run(image_rgb, phrase=phrase, return_hidden=use_hidden)
            self._grounding_cache = bbox
            grounding_latency_ms = (time.time() - g_start) * 1000.0

        # hidden_state(numpy array)는 JSON 응답에 그대로 넣으면 pydantic 직렬화가 깨짐 —
        # feature 계산용으로만 빼두고 응답 bbox에서는 제거 (plan_20260622_hidden_state_hub_integration.md).
        hidden_vec_from_bbox = bbox.pop("hidden_state", None) if use_hidden else None

        # Stage1 encode
        vis_feat = self.enc.encode_image(pil)  # (256,)

        # Update history
        frame = self._bbox_frame(bbox)
        frame["vis_feat"] = vis_feat
        if use_hidden:
            frame["hidden_state"] = hidden_vec_from_bbox
        self.history.append(frame)
        if len(self.history) > max(self.window, 8):
            self.history = self.history[-max(self.window, 8):]

        # Build feature and predict
        with torch.no_grad():
            if use_hidden:
                hv = frame.get("hidden_state")
                h_t = (torch.from_numpy(hv.astype(np.float32)).to(self.device)
                       if hv is not None else torch.zeros(HIDDEN_DIM, device=self.device))
                if effective_head_mode == "add":
                    x = torch.cat([self._build_flat_feature(vis_feat), h_t]).unsqueeze(0)
                else:  # replace
                    x = torch.cat([vis_feat, h_t]).unsqueeze(0)
                logits = self.hidden_heads[effective_head_mode](x)
            elif self.is_lstm:
                x = self._build_seq_feature().unsqueeze(0)  # (1, window, SEQ_DIM)
                logits = self.head(x)
            else:
                x = self._build_flat_feature(vis_feat).unsqueeze(0)  # (1, d_in)
                logits = self.head(x)
            pred_class = int(logits.argmax(dim=-1).item())

        # ── STOP 결정 (STOP_MODE에 따라 분기) ──────────────────────────────
        proximity_override = False
        learned_stop       = False
        is_near_goal       = False

        if STOP_MODE == "learned":
            # 모델이 STOP(0) 예측 → latch. 한 번 멈추면 reset() 전까지 유지.
            # has_bbox=False(미검출 fallback)일 때는 래치 금지 — 진짜 도착 신호가 아님.
            if self.stop_latched:
                pred_class  = 0
                learned_stop = True
            elif pred_class == 0 and frame.get("has_bbox", False):
                self.stop_latched = True
                learned_stop = True
        else:
            # proximity 모드: area + cx threshold로 강제 override
            # 조건: 직전 GOAL_CONSEC_FRAMES 프레임이 모두 근접 조건 만족
            # history가 GOAL_CONSEC_FRAMES 미만이면 발동 안 함 (초반 false positive 방지)
            goal_area = get_goal_area(phrase)
            last_n = self.history[-GOAL_CONSEC_FRAMES:]
            if len(last_n) >= GOAL_CONSEC_FRAMES:
                near_frames = sum(
                    1 for h in last_n
                    if h.get("has_bbox")
                    and h.get("area", 0.0) >= goal_area
                    and abs(h.get("cx", 0.5) - 0.5) <= GOAL_CX_TOLERANCE
                )
                is_near_goal = (near_frames >= GOAL_CONSEC_FRAMES)
            else:
                near_frames = 0
                is_near_goal = False
            if is_near_goal:
                proximity_override = True
                pred_class = 0

        stop_tag = ""
        if proximity_override: stop_tag = " [PROXIMITY STOP]"
        elif learned_stop and self.stop_latched and self.inference_count > 0:
            stop_tag = " [LEARNED STOP — LATCHED]"
        elif learned_stop:
            stop_tag = " [LEARNED STOP]"

        self.inference_count += 1
        total_ms = (time.time() - start) * 1000.0
        temporal_tag = f" [near {near_frames}/{GOAL_CONSEC_FRAMES}]" if STOP_MODE != "learned" else ""
        logger.info(
            "[#%d] %s%s%s | cx=%.3f area=%.3f has=%s | latency=%.0fms",
            self.inference_count, CLASS_NAMES[pred_class], stop_tag, temporal_tag,
            frame["cx"], frame["area"], frame["has_bbox"], total_ms,
        )

        return {
            "action": ACTION_2D[pred_class],
            "action_3d": ACTION_3D[pred_class],
            "predicted_class": pred_class,
            "predicted_label": CLASS_NAMES[pred_class],
            "bbox": bbox,
            "grounding_latency_ms": grounding_latency_ms,
            "latency_ms": total_ms,
            "goal_near_proxy": is_near_goal,
            "proximity_override": proximity_override,
            "learned_stop": learned_stop,
            "stop_latched": self.stop_latched,
            "stop_mode": STOP_MODE,
            "grounding_cached": use_cache,
            "head_mode": effective_head_mode,
            "buffer_status": {
                "history_size": len(self.history),
                "window": self.window,
                "head": self.head_name,
            },
            "source": "stage2_v2",
        }


# ---------------------------------------------------------------------------
# Global model instance
# ---------------------------------------------------------------------------

_model: Optional[Stage2V2Model] = None


def _resolve_device() -> torch.device:
    raw = os.getenv("VLA_DEVICE", "auto")
    if raw == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(reload: bool = False) -> Stage2V2Model:
    global _model
    if _model is None or reload:
        stage1_path = Path(os.getenv("VLA_S2V2_STAGE1", str(DEFAULT_STAGE1)))
        stage2_path = Path(os.getenv("VLA_S2V2_STAGE2", str(DEFAULT_STAGE2)))
        vlm_path    = Path(os.getenv("VLA_GROUNDING_MODEL_PATH", str(DEFAULT_VLM)))
        head_override = os.getenv("VLA_S2V2_HEAD") or None
        device = _resolve_device()
        logger.info("Loading Stage2V2 model on %s ...", device)
        pg2_path = Path(os.getenv("VLA_PG2_PATH", str(DEFAULT_PG2)))
        _model = Stage2V2Model(stage1_path, stage2_path, vlm_path, head_override, device, pg2_path=pg2_path)
    return _model


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

class InferenceRequest(BaseModel):
    image: str
    instruction: str = "basket"
    head_mode: str = "baseline"  # "baseline" | "add" | "replace" (plan_20260622_hidden_state_hub_integration.md)


class InferenceResponse(BaseModel):
    action: list[float]
    action_3d: list[float]
    latency_ms: float
    model_name: str = "stage2_v2"
    strategy: str = "stage2_v2"
    source: str
    buffer_status: dict[str, Any]
    predicted_class: Optional[int] = None
    predicted_label: Optional[str] = None
    bbox: Optional[dict[str, Any]] = None
    grounding_latency_ms: Optional[float] = None
    goal_near_proxy: Optional[bool] = None
    proximity_override: Optional[bool] = None
    learned_stop: Optional[bool] = None
    stop_latched: Optional[bool] = None
    stop_mode: Optional[str] = None
    grounding_cached: Optional[bool] = None
    head_mode: Optional[str] = None


class LoadRequest(BaseModel):
    stage2_path: str
    stage1_path: Optional[str] = None
    head: Optional[str] = None


class ConfigRequest(BaseModel):
    grounding_skip_n: Optional[int] = None
    stop_area_threshold: Optional[float] = None
    stop_cx_tolerance: Optional[float] = None
    stop_consec_frames: Optional[int] = None
    stop_mode: Optional[str] = None          # "proximity" | "learned"
    stop_latched: Optional[bool] = None      # None=그대로, False=latch 해제
    # 하위 호환: 수신은 하되 무시
    model: Optional[str] = None
    speed_scaling: Optional[bool] = None
    smooth_enabled: Optional[bool] = None


def _check_api_key(x_api_key: Optional[str]) -> None:
    expected = os.getenv("VLA_API_KEY", "")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=403, detail="Invalid API Key")


@app.get("/")
async def root() -> dict[str, Any]:
    m = _model
    return {
        "name": "Stage2 v2 VLA API",
        "version": "1.0.0",
        "status": "running",
        "model_loaded": m is not None,
        "head": m.head_name if m else None,
        "window": m.window if m else None,
        "val_acc": m.val_acc if m else None,
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    m = _model
    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "allocated_gb": round(torch.cuda.memory_allocated() / 1e9, 3),
            "device_name": torch.cuda.get_device_name(0),
        }
    return {
        "status": "healthy",
        "model_loaded": m is not None,
        "head": m.head_name if m else None,
        "window": m.window if m else None,
        "stop_mode": STOP_MODE,
        "stop_latched": m.stop_latched if m else False,
        "gpu": gpu,
    }


@app.get("/model/info")
async def model_info() -> dict[str, Any]:
    m = get_model()
    return {
        "model_loaded": True,
        "head": m.head_name,
        "window": m.window,
        "is_lstm": m.is_lstm,
        "val_acc": m.val_acc,
        "device": str(m.device),
        "inference_count": m.inference_count,
    }


@app.post("/predict", response_model=InferenceResponse)
async def predict(
    request: InferenceRequest,
    x_api_key: Optional[str] = Header(default=None),
) -> InferenceResponse:
    _check_api_key(x_api_key)
    try:
        m = get_model()
        result = m.predict(request.image, request.instruction, request.head_mode)
        return InferenceResponse(
            action=result["action"],
            action_3d=result["action_3d"],
            latency_ms=result["latency_ms"],
            source=result["source"],
            buffer_status=result["buffer_status"],
            predicted_class=result["predicted_class"],
            predicted_label=result["predicted_label"],
            bbox=result["bbox"],
            grounding_latency_ms=result["grounding_latency_ms"],
            goal_near_proxy=result["goal_near_proxy"],
            proximity_override=result["proximity_override"],
            grounding_cached=result["grounding_cached"],
            head_mode=result["head_mode"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/reset")
async def reset(x_api_key: Optional[str] = Header(default=None)) -> dict[str, Any]:
    _check_api_key(x_api_key)
    get_model().reset()
    return {"status": "success", "message": "History reset"}


@app.get("/recent")
async def recent_predictions(x_api_key: Optional[str] = Header(default=None)) -> dict[str, Any]:
    """proxy_inference_server 호환 — 최근 예측 기록 반환."""
    m = _model
    if m is None:
        return {"count": 0, "predictions": []}
    history = list(m.history[-30:]) if hasattr(m, "history") else []
    preds = [
        {
            "cx": h.get("cx", 0.5),
            "area": h.get("area", 0.0),
            "has_bbox": h.get("has_bbox", False),
        }
        for h in reversed(history)
    ]
    return {
        "count": len(preds),
        "predictions": preds,
        "inference_count": m.inference_count,
        "stop_latched": m.stop_latched,
        "stop_mode": STOP_MODE,
    }


@app.post("/config")
async def set_config(
    request: ConfigRequest,
    x_api_key: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    _check_api_key(x_api_key)
    applied: dict[str, Any] = {}
    ignored: list[str] = []

    if request.grounding_skip_n is not None:
        m = get_model()
        m._grounding_skip_n = max(1, request.grounding_skip_n)
        applied["grounding_skip_n"] = m._grounding_skip_n

    if request.stop_area_threshold is not None:
        global GOAL_AREA_THRESHOLD
        GOAL_AREA_THRESHOLD = float(request.stop_area_threshold)
        applied["stop_area_threshold"] = GOAL_AREA_THRESHOLD

    if request.stop_cx_tolerance is not None:
        global GOAL_CX_TOLERANCE
        GOAL_CX_TOLERANCE = float(request.stop_cx_tolerance)
        applied["stop_cx_tolerance"] = GOAL_CX_TOLERANCE

    if request.stop_consec_frames is not None:
        global GOAL_CONSEC_FRAMES
        GOAL_CONSEC_FRAMES = max(1, int(request.stop_consec_frames))
        applied["stop_consec_frames"] = GOAL_CONSEC_FRAMES

    if request.stop_mode is not None:
        global STOP_MODE
        if request.stop_mode in ("proximity", "learned"):
            STOP_MODE = request.stop_mode
            applied["stop_mode"] = STOP_MODE
        else:
            ignored.append(f"stop_mode={request.stop_mode} (unknown)")

    if request.stop_latched is not None:
        m = get_model()
        m.stop_latched = bool(request.stop_latched)
        applied["stop_latched"] = m.stop_latched

    for field in ("model", "speed_scaling", "smooth_enabled"):
        if getattr(request, field, None) is not None:
            ignored.append(field)

    return {"status": "ok", "applied": applied, "ignored": ignored}


class GroundRequest(BaseModel):
    image: str  # base64 RGB
    prompt: str = "detect gray basket"


@app.post("/ground")
async def ground(
    request: GroundRequest,
    x_api_key: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    """PG2 grounding only — no action prediction. Returns bbox + raw PG2 output."""
    _check_api_key(x_api_key)
    m = get_model()
    import time as _time
    image_rgb = m._decode_image(request.image)
    t0 = _time.time()
    bbox = m.grounder.run(image_rgb, return_raw=True)
    latency_ms = (_time.time() - t0) * 1000.0
    return {
        "has_bbox": bbox["has_bbox"],
        "cx": bbox["cx"],
        "cy": bbox["cy"],
        "area": bbox["area"],
        "x1": bbox.get("x1"),
        "y1": bbox.get("y1"),
        "x2": bbox.get("x2"),
        "y2": bbox.get("y2"),
        "raw_output": bbox.get("raw_output", ""),
        "latency_ms": round(latency_ms, 1),
        "prompt": request.prompt,
    }


@app.post("/ground/debug")
async def ground_debug(
    request: GroundRequest,
    x_api_key: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    """임시 디버그: generate 파라미터 변화 테스트 + 토큰 길이 반환."""
    _check_api_key(x_api_key)
    m = get_model()
    grounder = m.grounder
    if not hasattr(grounder, "_model") or grounder._model is None:
        grounder._ensure_loaded()
    image_rgb = m._decode_image(request.image)
    pil = Image.fromarray(image_rgb.astype(np.uint8)).convert("RGB")
    pil = resize_for_vlm(pil)
    prompt = request.prompt or "detect gray basket"
    inp = grounder._proc(text=prompt, images=pil, return_tensors="pt").to(grounder._device)
    inp["pixel_values"] = inp["pixel_values"].to(grounder._dtype)
    inp_keys = list(inp.keys())
    input_len = int(inp["input_ids"].shape[1])
    results = {}
    for min_tok in [0, 1, 5]:
        with torch.no_grad():
            gen = grounder._model.generate(
                **inp, max_new_tokens=64, do_sample=False,
                min_new_tokens=min_tok,
            )
        full_len = int(gen.shape[1])
        new_ids = gen[:, input_len:]
        raw = grounder._proc.batch_decode(new_ids, skip_special_tokens=False)[0]
        raw_skip = grounder._proc.batch_decode(new_ids, skip_special_tokens=True)[0]
        results[f"min_new_tokens={min_tok}"] = {
            "raw": raw, "raw_skip_special": raw_skip,
            "full_gen_len": full_len, "new_tokens": full_len - input_len,
        }
    return {"inp_keys": inp_keys, "input_len": input_len,
            "pixel_values_shape": list(inp["pixel_values"].shape),
            "results": results}


@app.post("/model/load")
async def load_model(
    request: LoadRequest,
    x_api_key: Optional[str] = Header(default=None),
) -> dict[str, Any]:
    _check_api_key(x_api_key)
    global _model
    if request.stage1_path:
        os.environ["VLA_S2V2_STAGE1"] = request.stage1_path
    os.environ["VLA_S2V2_STAGE2"] = request.stage2_path
    if request.head:
        os.environ["VLA_S2V2_HEAD"] = request.head
    _model = None
    m = get_model(reload=True)
    return {"status": "success", "head": m.head_name, "window": m.window, "val_acc": m.val_acc}


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.getenv("VLA_PORT", "8001")))
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args_cli = parser.parse_args()

    logger.info("Pre-loading Stage2V2 model ...")
    get_model()
    logger.info("Model ready. Starting uvicorn on %s:%d", args_cli.host, args_cli.port)
    uvicorn.run(app, host=args_cli.host, port=args_cli.port, log_level="info")

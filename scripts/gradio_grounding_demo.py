#!/usr/bin/env python3
# ── ROS workspace 경로 주입 (다른 import보다 먼저) ────────────────────────────
import os, sys as _sys

_ROS_WS = "/home/soda/MoNaVLA/ROS_action/install"
_ros_lib_dirs = [
    f"{_ROS_WS}/camera_interfaces/lib",
    f"{_ROS_WS}/camera_pub/lib",
]
_ros_py_dirs = [
    f"{_ROS_WS}/camera_interfaces/local/lib/python3.10/dist-packages",
    f"{_ROS_WS}/camera_pub/local/lib/python3.10/dist-packages",
]

_ld = os.environ.get("LD_LIBRARY_PATH", "")
_need_restart = any(p not in _ld for p in _ros_lib_dirs if os.path.isdir(p))
if _need_restart:
    _new_ld = ":".join(p for p in _ros_lib_dirs if os.path.isdir(p))
    os.environ["LD_LIBRARY_PATH"] = _new_ld + (":" + _ld if _ld else "")
    _pp = os.environ.get("PYTHONPATH", "")
    _new_pp = ":".join(p for p in _ros_py_dirs if os.path.isdir(p))
    os.environ["PYTHONPATH"] = _new_pp + (":" + _pp if _pp else "")
    os.environ.setdefault("ROS_DOMAIN_ID", "42")
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    os.execv(_sys.executable, [_sys.executable] + _sys.argv)

# Python path (재시작 없이 실행된 경우 대비)
for _p in _ros_py_dirs:
    if os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────────
"""
Object Recognition Demo — Kosmos-2 그라운딩 인터랙티브 테스트
ROS 카메라 자동 피드 / 이미지 업로드 → bbox 오버레이 + VLA 추론 연동

Port: 7863

Usage:
  python3 scripts/gradio_grounding_demo.py
  python3 scripts/gradio_grounding_demo.py --preload
"""
import argparse
import base64
import io
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.utils.camera_proc import camera_control_widget, start_camera, stop_camera, is_camera_running

import cv2
import gradio as gr
import numpy as np
import requests
import torch
from PIL import Image, ImageDraw

# ─── 실시간 터미널 로그 버퍼 (한국어 주석 적용) ───────────────────────────────────
import collections
_log_buffer = collections.deque(maxlen=200)
_log_lock = threading.Lock()

class DualRedirector:
    def __init__(self, stream, callback):
        self.stream = stream
        self.callback = callback

    def write(self, message):
        self.callback(message)
        self.stream.write(message)

    def flush(self):
        self.stream.flush()

    def __getattr__(self, attr):
        return getattr(self.stream, attr)

def _log_callback(msg):
    with _log_lock:
        _log_buffer.append(msg)

sys.stdout = DualRedirector(sys.stdout, _log_callback)
sys.stderr = DualRedirector(sys.stderr, _log_callback)

def get_terminal_logs():
    with _log_lock:
        return "".join(_log_buffer)

from scripts.run_grounding_realtime import (
    load_model,
    ground,
    draw_overlay,
    check_hit,
    DEFAULT_VLM,
    DEFAULT_ADAPTERS,
)

# ─── ROS 환경 설정 ────────────────────────────────────────────────────────────

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("ROS_HOME", "/tmp/ros")
os.environ["ROS_DOMAIN_ID"] = "42"
os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"


def prepend_env_path(key: str, value: str) -> None:
    current = os.environ.get(key, "")
    parts = [p for p in current.split(os.pathsep) if p]
    if value not in parts:
        os.environ[key] = value if not parts else f"{value}{os.pathsep}{current}"


def setup_ros_paths() -> None:
    ros_ws = Path(os.getenv("VLA_ROS_WS", str(ROOT / "ROS_action")))
    install_base = ros_ws / "install"
    if not install_base.exists():
        return
    prepend_env_path("AMENT_PREFIX_PATH", str(install_base))
    prepend_env_path("COLCON_PREFIX_PATH", str(install_base))
    prepend_env_path("CMAKE_PREFIX_PATH", str(install_base))
    for pkg in install_base.iterdir():
        if not pkg.is_dir():
            continue
        lib_path = pkg / "lib"
        if lib_path.exists():
            prepend_env_path("LD_LIBRARY_PATH", str(lib_path))
        for candidate in (
            pkg / "local/lib/python3.10/dist-packages",
            pkg / "lib/python3.10/site-packages",
        ):
            if candidate.exists() and str(candidate) not in sys.path:
                sys.path.append(str(candidate))
                prepend_env_path("PYTHONPATH", str(candidate))


setup_ros_paths()

ROS_AVAILABLE = False
try:
    import rclpy
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.node import Node
    from cv_bridge import CvBridge
    from camera_interfaces.srv import GetImage

    ROS_AVAILABLE = True
    print("[DEMO] ROS2 available ✅")
except ImportError as e:
    print(f"[DEMO] ROS2 unavailable: {e}")

    class Node:
        pass

    class ReentrantCallbackGroup:
        pass


# ─── ROS 카메라 노드 ───────────────────────────────────────────────────────────

class ROSCameraNode(Node):
    def __init__(self):
        super().__init__("grounding_demo_camera_node")
        self.callback_group = ReentrantCallbackGroup()
        self.cv_bridge = CvBridge()
        self.get_image_client = self.create_client(
            GetImage, "get_image_service", callback_group=self.callback_group
        )

    def get_frame(self) -> Image.Image | None:
        if not self.get_image_client.wait_for_service(timeout_sec=1.0):
            return None
        request = GetImage.Request()
        future = self.get_image_client.call_async(request)
        start = time.time()
        while rclpy.ok() and not future.done():
            if time.time() - start > 2.0:
                return None
            time.sleep(0.01)
        if future.done():
            try:
                response = future.result()
                if response and response.image.data:
                    cv_image = self.cv_bridge.imgmsg_to_cv2(response.image, "bgr8")
                    return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            except Exception:
                return None
        return None


ros_node: ROSCameraNode | None = None
if ROS_AVAILABLE:
    try:
        if not rclpy.ok():
            rclpy.init()
        ros_node = ROSCameraNode()
        threading.Thread(target=lambda: rclpy.spin(ros_node), daemon=True).start()
        print("[DEMO] ROSCameraNode started ✅")
    except Exception as e:
        ROS_AVAILABLE = False
        ros_node = None
        print(f"[DEMO] ROSCameraNode disabled: {e}")


# ─── 공유 카메라 상태 ──────────────────────────────────────────────────────────

_last_frame: np.ndarray | None = None
_cam_ok = False


def _fetch_ros_frame() -> np.ndarray | None:
    global _last_frame, _cam_ok
    if not ROS_AVAILABLE or ros_node is None:
        _cam_ok = False
        return None
    img = ros_node.get_frame()
    if img is None:
        _cam_ok = False
        return _last_frame  # 이전 프레임 유지
    arr = np.array(img)
    _last_frame = arr
    _cam_ok = True
    return arr


def timer_tick():
    """gr.Timer tick — 카메라 프레임을 가져와 모든 live feed 업데이트."""
    frame = _fetch_ros_frame()
    status = "📷 Live ✅" if _cam_ok else ("ROS 연결됨, 서비스 대기 중…" if ROS_AVAILABLE else "❌ ROS 없음")
    return frame, frame, frame, frame, status  # (vla_cam, alias_cam, grnd_cam, vqa_cam, status_txt)


def _get_alias_frame():
    return _last_frame


# ─── 정적 설정 및 프리셋 ─────────────────────────────────────────────────────────

PRESET_ALIASES = {
    "🧺 basket": "gray basket\ngray box\ncontainer\nbin\nlaundry basket",
    "🔴 ball":   "red ball\norange ball\nball\nsphere",
    "🚪 door":   "door\nexit\nentrance",
    "🏛 wall":   "white wall\ncorridor wall\nwall",
    "🛣 corridor": "corridor\nhallway\npassage",
}

GOAL_NAV_PRESETS = [
    # basket
    "the gray basket on right",
    "the gray basket on left",
    "the gray basket",
    # chair
    "the chair on right",
    "the chair on left",
    "the chair",
    # door / corridor
    "the door",
    "the open door",
    "the corridor on the left",
    "the corridor on the right",
    "the corridor",
    # wall / exit
    "the wall",
    "the exit",
    # table / box
    "the table",
    "the box on right",
    "the box on left",
    "the box",
]

ACTION_COLORS = {
    "FORWARD":   (46, 204, 113),
    "LEFT":      (52, 152, 219),
    "RIGHT":     (230, 126, 34),
    "FWD+LEFT":  (26, 188, 156),
    "FWD+RIGHT": (243, 156, 18),
    "ROT_L":     (155, 89, 182),
    "ROT_R":     (231, 76, 60),
    "STOP":      (149, 165, 166),
}

API_URL = "http://localhost:8001"
API_KEY = os.getenv("VLA_API_KEY", "vla_devel_key_2026")
_API_HEADERS = {"X-API-Key": API_KEY}


# ─── 모델 싱글톤 ───────────────────────────────────────────────────────────────

_state: dict = {"model": None, "processor": None, "model_name": None, "adapter": None}
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOCAL_VLM_PATHS = {
    "Pure Kosmos-2": ROOT / ".vlms" / "kosmos-2-patch14-224",
    "Exp56 LoRA":    ROOT / ".vlms" / "kosmos-2-patch14-224",
    "PaliGemma-3B":  ROOT / ".vlms" / "paligemma-3b-mix-224",
    "Moondream2":    ROOT / ".vlms" / "moondream2",
}

ONLINE_VLM_IDS = {
    "Pure Kosmos-2": "microsoft/kosmos-2-patch14-224",
    "Exp56 LoRA":    "microsoft/kosmos-2-patch14-224",
    "PaliGemma-3B":  "google/paligemma-3b-mix-224",
    "Moondream2":    "vikhyatk/moondream2",
}

ADAPTER_OPTIONS = {
    "Pure Kosmos-2": None,
    "Exp56 LoRA":    DEFAULT_ADAPTERS.get("exp56"),
    "PaliGemma-3B":  None,
    "Moondream2":    None,
}

_MOONDREAM_REVISION = "2025-01-09"


def _ensure_model(adapter_label: str):
    """Loads and caches the specified local VLM model on-demand to save VRAM."""
    if _state["model"] is not None and _state["model_name"] == adapter_label:
        return _state["model"], _state["processor"]

    print(f"[LOAD LOCAL VLM] Switching from {_state['model_name']} to {adapter_label} ...")

    # Clear existing model
    if _state["model"] is not None:
        del _state["model"]
        _state["model"] = None
    _state["processor"] = None
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = _DEVICE
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    local_path = LOCAL_VLM_PATHS.get(adapter_label)
    online_id = ONLINE_VLM_IDS.get(adapter_label)
    model_path = str(local_path) if (local_path and local_path.exists() and any(local_path.iterdir())) else online_id

    print(f"Loading {adapter_label} from: {model_path}")

    if adapter_label in ("Pure Kosmos-2", "Exp56 LoRA"):
        # Kosmos-2: uint8 image token 텐서 포함 → bitsandbytes 양자화 절대 불가
        # float16(CUDA) / float32(CPU) 로드만 허용 (load_in_4bit/8bit 금지)
        from transformers import AutoModelForVision2Seq, AutoProcessor
        from peft import PeftModel
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        processor = AutoProcessor.from_pretrained(model_path)
        model = AutoModelForVision2Seq.from_pretrained(
            model_path, torch_dtype=dtype, low_cpu_mem_usage=True
        ).to(device).eval()
        adapter_path = ADAPTER_OPTIONS.get(adapter_label)
        if adapter_path and Path(adapter_path).exists():
            model = PeftModel.from_pretrained(model, str(adapter_path))
        print(f"[LOAD] Kosmos-2 완료 ({dtype}, no quantization)")

    elif adapter_label == "PaliGemma-3B":
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig
        processor = AutoProcessor.from_pretrained(model_path)
        if device.type == "cuda":
            # Jetson Orin 등 리소스 제약 환경을 위해 4비트 양자화 로드 적용
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map={"": device}
            ).eval()
        else:
            model = PaliGemmaForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch.float32
            ).to(device).eval()

    elif adapter_label == "Moondream2":
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        processor = AutoTokenizer.from_pretrained(model_path)
        if device.type == "cuda":
            # Jetson Orin 등 리소스 제약 환경을 위해 4비트 양자화 로드 적용
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                revision=_MOONDREAM_REVISION if "moondream2" in model_path else None,
                quantization_config=bnb_config,
                device_map={"": device}
            ).eval()
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                revision=_MOONDREAM_REVISION if "moondream2" in model_path else None,
                torch_dtype=torch.float32,
            ).to(device).eval()
    else:
        raise ValueError(f"Unknown local VLM: {adapter_label}")

    _state.update(model=model, processor=processor, model_name=adapter_label, adapter=adapter_label)
    return model, processor


def _to_numpy(image) -> np.ndarray | None:
    if image is None:
        return None
    if isinstance(image, np.ndarray):
        return image
    return np.array(image.convert("RGB"))


def _img_to_b64(img_np: np.ndarray) -> str:
    pil = Image.fromarray(img_np)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def annotate_image(img_np: np.ndarray, bbox: dict | None = None,
                   label: str = "", draw_grid: bool = True) -> np.ndarray:
    """3×3 격자 + bbox 오버레이 (inference_dashboard 동일 스타일)."""
    arr = img_np.copy()
    h, w = arr.shape[:2]

    if draw_grid:
        color = (100, 255, 100)
        cv2.line(arr, (w // 3, 0), (w // 3, h), color, 1)
        cv2.line(arr, (2 * w // 3, 0), (2 * w // 3, h), color, 1)
        cv2.line(arr, (0, h // 3), (w, h // 3), color, 1)
        cv2.line(arr, (0, 2 * h // 3), (w, 2 * h // 3), color, 1)

    if bbox:
        cx_px = int(bbox.get("cx", 0.5) * w)
        cy_px = int(bbox.get("cy", 0.5) * h)
        ent = str(bbox.get("entity", label or "bbox"))
        color_rgb = ACTION_COLORS.get(label, (255, 80, 80))
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])

        if "x1" in bbox:
            x1, y1 = int(bbox["x1"] * w), int(bbox["y1"] * h)
            x2, y2 = int(bbox["x2"] * w), int(bbox["y2"] * h)
            cv2.rectangle(arr, (x1, y1), (x2, y2), color_bgr, 2)
        else:
            r = 10
            cv2.line(arr, (cx_px - r, cy_px), (cx_px + r, cy_px), color_bgr, 2)
            cv2.line(arr, (cx_px, cy_px - r), (cx_px, cy_px + r), color_bgr, 2)

        cv2.circle(arr, (cx_px, cy_px), 4, color_bgr, -1)
        txt = f"{label}  {ent[:15]}" if label else ent[:20]
        cv2.putText(arr, txt, (max(cx_px - 40, 0), max(cy_px - 10, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_bgr, 1, cv2.LINE_AA)
    return arr


# ─── Grounding 탭 로직 ────────────────────────────────────────────────────────

def run_grounding(image, phrase: str, adapter_label: str):
    if image is None:
        return None, "⚠️ 이미지가 없습니다 (카메라 연결 또는 업로드)", ""
    phrase = phrase.strip()
    if not phrase:
        return None, "⚠️ Phrase를 입력해주세요", ""

    try:
        model, proc = _ensure_model(adapter_label)
    except Exception as e:
        import traceback
        err_msg = f"❌ 모델 로드 오류: {e}\n{traceback.format_exc()}"
        print(err_msg)
        return None, f"⚠️ 모델 로드 실패: {e}", "❌ 에러 로그 확인 필요"

    img_np = _to_numpy(image)
    pil_img = Image.fromarray(img_np).convert("RGB")

    bbox = None
    caption = ""
    entity_lines = []
    hit = False

    if adapter_label in ("Pure Kosmos-2", "Exp56 LoRA"):
        result = ground(model, proc, img_np, phrase, _DEVICE)
        overlay = draw_overlay(img_np, phrase, result)
        hit = check_hit(result, phrase)
        caption = result.get('caption', '')
        for box_info in result.get("boxes", []):
            ent_name = box_info["entity"]
            x1, y1, x2, y2 = box_info["bbox_xyxy"]
            entity_lines.append(
                f"  [{ent_name}]  cx={((x1+x2)/2):.2f}  cy={((y1+y2)/2):.2f}"
                f"  area={((x2-x1)*(y2-y1)):.3f}"
            )

    elif adapter_label == "PaliGemma-3B":
        prompt = f"detect {phrase}\n"
        inputs = proc(text=prompt, images=pil_img, return_tensors="pt").to(_DEVICE)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=100)
        full_decoded = proc.decode(gen[0], skip_special_tokens=False)
        prompt_end = full_decoded.find(prompt.strip())
        caption = full_decoded[prompt_end + len(prompt):] if prompt_end >= 0 else full_decoded
        caption = caption.strip()

        locs = _parse_paligemma_locs(caption)
        hit = locs is not None
        if locs:
            bbox = {"entity": phrase, "cx": locs["cx"], "cy": locs["cy"], "area": locs["area"]}
            x1 = locs["cx"] - (locs["area"]**0.5)/2
            y1 = locs["cy"] - (locs["area"]**0.5)/2
            x2 = locs["cx"] + (locs["area"]**0.5)/2
            y2 = locs["cy"] + (locs["area"]**0.5)/2
            bbox.update(x1=max(x1, 0.0), y1=max(y1, 0.0), x2=min(x2, 1.0), y2=min(y2, 1.0))
            entity_lines.append(f"  [{phrase}]  cx={locs['cx']:.2f}  cy={locs['cy']:.2f}  area={locs['area']:.3f}")

        overlay = annotate_image(img_np, bbox, label="HIT" if hit else "MISS", draw_grid=False)
        overlay = Image.fromarray(overlay)

    elif adapter_label == "Moondream2":
        with torch.no_grad():
            try:
                raw = model.detect(pil_img, phrase)
                objects = raw.get("objects", raw) if isinstance(raw, dict) else raw
            except Exception as e:
                objects = []
        caption = f"{len(objects)} object(s) detected"
        hit = len(objects) > 0
        if objects:
            obj = objects[0]
            x1 = float(obj.get("x_min", obj.get("xmin", 0)))
            y1 = float(obj.get("y_min", obj.get("ymin", 0)))
            x2 = float(obj.get("x_max", obj.get("xmax", 1)))
            y2 = float(obj.get("y_max", obj.get("ymax", 1)))
            area = abs(x2 - x1) * abs(y2 - y1)
            bbox = {"entity": phrase, "cx": (x1+x2)/2, "cy": (y1+y2)/2, "area": area, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
            entity_lines.append(f"  [{phrase}]  cx={(x1+x2)/2:.2f}  cy={(y1+y2)/2:.2f}  area={area:.3f}")

        overlay = annotate_image(img_np, bbox, label="HIT" if hit else "MISS", draw_grid=False)
        overlay = Image.fromarray(overlay)

    status = f"{'HIT' if hit else 'MISS'}   {caption}"
    return overlay, status, "\n".join(entity_lines) or "  (감지 없음)"


def run_alias_test(image, alias_text: str, adapter_label: str):
    if image is None:
        return None, "⚠️ 이미지가 없습니다"
    phrases = [p.strip() for p in alias_text.splitlines() if p.strip()]
    if not phrases:
        return None, "⚠️ Phrase를 한 줄씩 입력해주세요"

    try:
        model, proc = _ensure_model(adapter_label)
    except Exception as e:
        import traceback
        err_msg = f"❌ 모델 로드 오류: {e}\n{traceback.format_exc()}"
        print(err_msg)
        return None, f"⚠️ 모델 로드 실패: {e}\n\n{traceback.format_exc()}"

    img_np = _to_numpy(image)
    pil_img = Image.fromarray(img_np).convert("RGB")

    rows, overlays = [], []
    for phrase in phrases:
        bbox = None
        caption = ""
        hit = False

        if adapter_label in ("Pure Kosmos-2", "Exp56 LoRA"):
            r = ground(model, proc, img_np, phrase, _DEVICE)
            hit = check_hit(r, phrase)
            caption = r.get("caption", "")[:50]
            entities = list({b["entity"] for b in r.get("boxes", [])})
            ov = draw_overlay(img_np, phrase, r)
        elif adapter_label == "PaliGemma-3B":
            prompt = f"detect {phrase}\n"
            inputs = proc(text=prompt, images=pil_img, return_tensors="pt").to(_DEVICE)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=100)
            full_decoded = proc.decode(gen[0], skip_special_tokens=False)
            prompt_end = full_decoded.find(prompt.strip())
            caption = full_decoded[prompt_end + len(prompt):] if prompt_end >= 0 else full_decoded
            caption = caption.strip()
            locs = _parse_paligemma_locs(caption)
            hit = locs is not None
            entities = [phrase] if hit else []
            if locs:
                bbox = {"entity": phrase, "cx": locs["cx"], "cy": locs["cy"], "area": locs["area"]}
                x1 = locs["cx"] - (locs["area"]**0.5)/2
                y1 = locs["cy"] - (locs["area"]**0.5)/2
                x2 = locs["cx"] + (locs["area"]**0.5)/2
                y2 = locs["cy"] + (locs["area"]**0.5)/2
                bbox.update(x1=max(x1, 0.0), y1=max(y1, 0.0), x2=min(x2, 1.0), y2=min(y2, 1.0))
            ov = annotate_image(img_np, bbox, label="HIT" if hit else "MISS", draw_grid=False)
            ov = Image.fromarray(ov)
        elif adapter_label == "Moondream2":
            with torch.no_grad():
                try:
                    raw = model.detect(pil_img, phrase)
                    objects = raw.get("objects", raw) if isinstance(raw, dict) else raw
                except Exception as e:
                    objects = []
            hit = len(objects) > 0
            entities = [phrase] if hit else []
            caption = f"{len(objects)} object(s) detected"
            if objects:
                obj = objects[0]
                x1 = float(obj.get("x_min", obj.get("xmin", 0)))
                y1 = float(obj.get("y_min", obj.get("ymin", 0)))
                x2 = float(obj.get("x_max", obj.get("xmax", 1)))
                y2 = float(obj.get("y_max", obj.get("ymax", 1)))
                area = abs(x2 - x1) * abs(y2 - y1)
                bbox = {"entity": phrase, "cx": (x1+x2)/2, "cy": (y1+y2)/2, "area": area, "x1": x1, "y1": y1, "x2": x2, "y2": y2}
            ov = annotate_image(img_np, bbox, label="HIT" if hit else "MISS", draw_grid=False)
            ov = Image.fromarray(ov)

        rows.append((phrase, hit, entities, caption))
        overlays.append(ov)

    cols = min(len(overlays), 3)
    n_rows = (len(overlays) + cols - 1) // cols
    W, H = overlays[0].size
    LABEL_H = 22
    grid = Image.new("RGB", (W * cols, (H + LABEL_H) * n_rows), (30, 30, 30))

    for i, (ov, (phrase, hit, _, _)) in enumerate(zip(overlays, rows)):
        ri, ci = divmod(i, cols)
        cell = Image.new("RGB", (W, H + LABEL_H), (30, 30, 30))
        d = ImageDraw.Draw(cell)
        lc = (80, 200, 120) if hit else (220, 80, 80)
        d.rectangle([0, 0, W, LABEL_H], fill=(20, 20, 20))
        d.text((6, 4), f"{'HIT' if hit else 'MISS'}  {phrase}", fill=lc)
        cell.paste(ov, (0, LABEL_H))
        grid.paste(cell, (ci * W, ri * (H + LABEL_H)))

    hit_count = sum(1 for _, hit, _, _ in rows if hit)
    lines = [f"Hit: {hit_count} / {len(phrases)}\n"]
    for phrase, hit, entities, _ in rows:
        ent_str = ", ".join(entities) if entities else "—"
        lines.append(f"{'HIT' if hit else 'MISS'}  {phrase:<25}  [{ent_str}]")

    return grid, "\n".join(lines)


def run_vqa(image, prompt: str, adapter_label: str, add_grounding_tag: bool):
    if image is None:
        return "⚠️ 이미지가 없습니다"
    prompt = prompt.strip()
    if not prompt:
        return "⚠️ 질문(Prompt)을 입력해주세요"

    img_np = _to_numpy(image)
    pil_img = Image.fromarray(img_np).convert("RGB")

    device = _DEVICE
    start_time = time.time()

    try:
        model, proc = _ensure_model(adapter_label)
        
        if add_grounding_tag and adapter_label in ("Pure Kosmos-2", "Exp56 LoRA"):
            if not prompt.startswith("<grounding>"):
                prompt = f"<grounding>{prompt}"

        dtype = torch.float16 if device.type == "cuda" else torch.float32

        if adapter_label in ("Pure Kosmos-2", "Exp56 LoRA"):
            inputs = proc(text=prompt, images=pil_img, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            pv = inputs["pixel_values"].to(dtype)
            with torch.no_grad():
                gen = model.generate(
                    pixel_values=pv,
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    image_embeds=None,
                    image_embeds_position_mask=inputs.get("image_embeds_position_mask"),
                    use_cache=True,
                    max_new_tokens=128,
                )
            new_ids = gen[:, inputs["input_ids"].shape[1]:]
            response = proc.batch_decode(new_ids, skip_special_tokens=False)[0]

        elif adapter_label == "PaliGemma-3B":
            if not prompt.endswith("\n"):
                prompt += "\n"
            inputs = proc(text=prompt, images=pil_img, return_tensors="pt").to(device)
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=128)
            full_decoded = proc.decode(gen[0], skip_special_tokens=False)
            prompt_end = full_decoded.find(prompt.strip())
            response = full_decoded[prompt_end + len(prompt):] if prompt_end >= 0 else full_decoded
            response = response.strip()

        elif adapter_label == "Moondream2":
            with torch.no_grad():
                response = model.answer_question(pil_img, prompt, proc)

    except Exception as e:
        response = f"❌ 오류 발생: {e}"

    latency = (time.time() - start_time) * 1000.0
    return f"[{adapter_label} 답변 - {latency:.1f}ms]\n\n{response}"


# ─── VLA 로직 ────────────────────────────────────────────────────────────────

def _vla_call(img_np: np.ndarray, instruction: str, vlm_model: str = "kosmos") -> dict:
    b64 = _img_to_b64(img_np)
    r = requests.post(
        f"{API_URL}/predict",
        json={"image": b64, "instruction": instruction, "vlm_model": vlm_model},
        headers=_API_HEADERS,
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


VLM_NAME_MAP = {
    "Kosmos-2": "kosmos",
    "PaliGemma-3B": "paligemma",
    "Moondream2": "moondream",
}


def run_vla_predict(image, instruction: str, vlm_model_name: str):
    if image is None:
        return None, "⚠️ 카메라 프레임 없음 (카메라 연결 또는 업로드)", ""
    instruction = instruction.strip()
    if not instruction:
        return None, "⚠️ Instruction을 입력해주세요", ""

    img_np = _to_numpy(image)
    vlm_model = VLM_NAME_MAP.get(vlm_model_name, "kosmos")
    try:
        d = _vla_call(img_np, instruction, vlm_model)
    except Exception as e:
        return None, f"❌ API 오류: {e}", ""

    label = d.get("predicted_label", "?")
    out_arr = annotate_image(img_np, d.get("bbox"), label)

    status = (
        f"▶  {label}\n"
        f"latency: {d['latency_ms']:.1f}ms  |  grounding: {d.get('grounding_latency_ms',0):.1f}ms\n"
        f"goal_near: {d.get('goal_near_proxy','?')}  |  model: {d.get('model_name','?')}\n"
        f"caption: {d.get('grounding_caption','')}"
    )

    bbox = d.get("bbox") or {}
    if "cx" in bbox:
        bbox_info = f"entity: {bbox.get('entity','?')}\ncx={bbox['cx']:.3f}  cy={bbox['cy']:.3f}  area={bbox.get('area',0):.3f}"
    elif "x1" in bbox:
        bbox_info = f"entity: {bbox.get('entity','?')}\nx1={bbox['x1']:.3f}  y1={bbox['y1']:.3f}  x2={bbox['x2']:.3f}  y2={bbox['y2']:.3f}"
    else:
        bbox_info = "(bbox 없음)"

    return out_arr, status, bbox_info


def run_vla_alias(image, alias_text: str, vlm_model_name: str):
    if image is None:
        return None, "⚠️ 카메라 프레임 없음"
    phrases = [p.strip() for p in alias_text.splitlines() if p.strip()]
    if not phrases:
        return None, "⚠️ Phrase를 한 줄씩 입력해주세요"

    img_np = _to_numpy(image)
    vlm_model = VLM_NAME_MAP.get(vlm_model_name, "kosmos")
    results = []
    last_d = None

    for phrase in phrases:
        try:
            d = _vla_call(img_np, phrase, vlm_model)
            label = d.get("predicted_label", "?")
            cx = f"{d['bbox']['cx']:.3f}" if d.get("bbox") and "cx" in d["bbox"] else "—"
            results.append((phrase, label, d.get("latency_ms", 0), cx))
            last_d = d
        except Exception as e:
            results.append((phrase, f"ERR:{e}", 0, "—"))

    out_arr = annotate_image(img_np, (last_d or {}).get("bbox"),
                             (last_d or {}).get("predicted_label", ""))

    lines = ["Phrase → Action  (VLA Alias Test)\n"]
    action_counts: dict[str, int] = {}
    for phrase, label, latency, cx in results:
        action_counts[label] = action_counts.get(label, 0) + 1
        lines.append(f"{label:<12}  {phrase:<30}  cx={cx}  ({latency:.0f}ms)")

    lines.append("\n─── 액션 분포 ───")
    for action, cnt in sorted(action_counts.items(), key=lambda x: -x[1]):
        lines.append(f"{action:<12} {'█' * cnt} {cnt}")

    return out_arr, "\n".join(lines)


def reset_api_history():
    try:
        r = requests.post(f"{API_URL}/reset", headers=_API_HEADERS, timeout=5)
        return f"리셋 완료  ({r.json()})"
    except Exception as e:
        return f"오류: {e}"


# ─── UI 빌드 ──────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    cam_status_init = "📷 Live ✅" if ROS_AVAILABLE else "❌ ROS 없음 (업로드 사용)"

    with gr.Blocks(title="Object Recognition Demo") as demo:
        gr.Markdown("# Object Recognition Demo")

        # 카메라 프로세스 제어 (버튼은 타이머 정의 후 연결)
        _cam_proc_st, _cam_start_btn, _cam_stop_btn = camera_control_widget()

        # 상단 상태 바
        with gr.Row():
            cam_status_txt = gr.Textbox(
                value=cam_status_init, label="ROS 카메라 서비스 상태",
                interactive=False, scale=4,
            )
            adapter_dd = gr.Dropdown(
                choices=list(ADAPTER_OPTIONS.keys()),
                value="Pure Kosmos-2",
                label="Local Grounding 모델", scale=2,
            )
            vla_vlm_dd = gr.Dropdown(
                choices=["Kosmos-2", "PaliGemma-3B", "Moondream2"],
                value="Kosmos-2",
                label="VLA API VLM", scale=2,
            )

        with gr.Tabs():

            # ── 탭 1: Grounding ──────────────────────────────────────────────
            with gr.TabItem("Grounding"):
                with gr.Row():
                    with gr.Column():
                        img_g = gr.Image(
                            interactive=False,
                            label="📷 Live Camera (자동 갱신)",
                        )
                        phrase_g = gr.Textbox(
                            value="gray basket", label="Phrase",
                            placeholder="찾을 객체 이름",
                        )
                        with gr.Row():
                            btn_g = gr.Button("▶ Run Grounding", variant="primary", scale=3)
                            btn_up_g = gr.UploadButton("📁 이미지 업로드", file_types=["image"], scale=1)
                    with gr.Column():
                        out_img_g  = gr.Image(label="결과 (bbox 오버레이)")
                        out_status = gr.Textbox(label="Status / Caption", lines=2)
                        out_ents   = gr.Textbox(label="감지된 Entity", lines=6)

                btn_g.click(run_grounding,
                            inputs=[img_g, phrase_g, adapter_dd],
                            outputs=[out_img_g, out_status, out_ents])
                btn_up_g.upload(fn=lambda f=None: np.array(Image.open(f).convert("RGB")) if f else None,
                                outputs=img_g)

            # ── 탭 2: Alias Test ─────────────────────────────────────────────
            with gr.TabItem("Alias Test"):
                gr.Markdown("같은 물체를 다양한 이름으로 — Kosmos-2 인식률 비교")
                with gr.Row():
                    with gr.Column():
                        img_a = gr.Image(interactive=False, label="📷 Live Camera")
                        btn_up_a = gr.UploadButton("📁 이미지 업로드", file_types=["image"])
                        gr.Markdown("**프리셋**")
                        with gr.Row():
                            preset_btns = [gr.Button(label, size="sm") for label in PRESET_ALIASES]
                        alias_txt = gr.Textbox(
                            value=list(PRESET_ALIASES.values())[0],
                            label="Alias 목록 (한 줄 = 1 phrase)", lines=8,
                        )
                        btn_a = gr.Button("▶ Run All", variant="primary")
                    with gr.Column():
                        out_img_a   = gr.Image(label="결과 그리드")
                        out_summary = gr.Textbox(label="Hit / Miss 요약", lines=12)

                btn_up_a.upload(fn=lambda f=None: np.array(Image.open(f).convert("RGB")) if f else None,
                                outputs=img_a)
                for btn, text in zip(preset_btns, PRESET_ALIASES.values()):
                    btn.click(fn=lambda t=text: t, outputs=alias_txt)
                btn_a.click(run_alias_test,
                            inputs=[img_a, alias_txt, adapter_dd],
                            outputs=[out_img_a, out_summary])

            # ── 탭 2.5: VQA & Chat ───────────────────────────────────────────
            with gr.TabItem("VQA & Chat"):
                gr.Markdown(
                    "이미지와 함께 자유 텍스트 질문을 입력하여 VLM의 자연어 답변을 확인합니다.\n"
                    "Kosmos-2 모델의 경우 그라운딩 접두사(`<grounding>`) 추가 여부를 제어할 수 있습니다."
                )
                with gr.Row():
                    with gr.Column():
                        img_q = gr.Image(interactive=False, label="📷 Live Camera")
                        btn_up_q = gr.UploadButton("📁 이미지 업로드 (대체)", file_types=["image"])
                        prompt_q = gr.Textbox(
                            value="Describe this image in detail.",
                            label="질문 (Prompt)",
                            placeholder="VLM에게 보낼 질문을 입력하세요",
                        )
                        add_tag_cb = gr.Checkbox(
                            value=False,
                            label="Kosmos-2 <grounding> 태그 강제 추가"
                        )
                        btn_q = gr.Button("▶ Send Question", variant="primary")
                    with gr.Column():
                        out_vqa_txt = gr.Textbox(label="VLM 자연어 답변", lines=15, interactive=False)

                btn_up_q.upload(fn=lambda f=None: np.array(Image.open(f).convert("RGB")) if f else None,
                                outputs=img_q)
                btn_q.click(run_vqa,
                            inputs=[img_q, prompt_q, adapter_dd, add_tag_cb],
                            outputs=[out_vqa_txt])

            # ── 탭 3: VLA Inference ──────────────────────────────────────────
            with gr.TabItem("VLA Inference"):
                gr.Markdown(
                    "이미지 → GoalNav API (`:8001/predict`) → 액션 예측\n"
                    "카메라 프레임이 자동으로 표시되며, **▶ Run VLA** 클릭 시 현재 프레임으로 추론합니다."
                )
                with gr.Row():
                    with gr.Column():
                        img_v = gr.Image(interactive=False, label="📷 Live Camera")
                        btn_up_v = gr.UploadButton("📁 이미지 업로드 (대체)", file_types=["image"])
                        instr_dd = gr.Dropdown(
                            choices=GOAL_NAV_PRESETS, value=GOAL_NAV_PRESETS[0],
                            label="Instruction 프리셋", allow_custom_value=True,
                        )
                        instr_txt = gr.Textbox(value=GOAL_NAV_PRESETS[0],
                                               label="Instruction (직접 입력)")
                        instr_dd.change(fn=lambda x: x, inputs=instr_dd, outputs=instr_txt)
                        with gr.Row():
                            btn_v     = gr.Button("▶ Run VLA", variant="primary", scale=3)
                            reset_btn = gr.Button("↺ Reset", scale=1)
                    with gr.Column():
                        out_img_v    = gr.Image(label="결과 (3×3 격자 + bbox)")
                        out_status_v = gr.Textbox(label="Action / Status", lines=5)
                        out_bbox_v   = gr.Textbox(label="Bbox 정보", lines=3)

                btn_up_v.upload(fn=lambda f=None: np.array(Image.open(f).convert("RGB")) if f else None,
                                outputs=img_v)
                reset_btn.click(reset_api_history,
                                outputs=gr.Textbox(visible=False))
                btn_v.click(run_vla_predict,
                            inputs=[img_v, instr_txt, vla_vlm_dd],
                            outputs=[out_img_v, out_status_v, out_bbox_v])

            # ── 탭 4: VLA Alias Test ─────────────────────────────────────────
            with gr.TabItem("VLA Alias Test"):
                gr.Markdown(
                    "여러 instruction phrase → VLA API 각각 → **어떤 말 → 어떤 액션** 비교\n"
                    "카메라 프레임이 자동 표시됩니다."
                )
                with gr.Row():
                    with gr.Column():
                        img_va = gr.Image(interactive=False, label="📷 Live Camera")
                        btn_up_va = gr.UploadButton("📁 이미지 업로드 (대체)", file_types=["image"])
                        gr.Markdown("**프리셋**")
                        with gr.Row():
                            vpreset_btns = [gr.Button(label, size="sm") for label in PRESET_ALIASES]
                        alias_txt_v = gr.Textbox(
                            value=list(PRESET_ALIASES.values())[0],
                            label="Instruction 목록 (한 줄 = 1개)", lines=8,
                        )
                        btn_va = gr.Button("▶ Run All VLA", variant="primary")
                    with gr.Column():
                        out_img_va = gr.Image(label="마지막 Bbox (3×3 격자)")
                        out_sum_va = gr.Textbox(label="Phrase → Action 매핑 + 분포", lines=18)

                btn_up_va.upload(fn=lambda f=None: np.array(Image.open(f).convert("RGB")) if f else None,
                                 outputs=img_va)
                for btn, text in zip(vpreset_btns, PRESET_ALIASES.values()):
                    btn.click(fn=lambda t=text: t, outputs=alias_txt_v)
                btn_va.click(run_vla_alias,
                             inputs=[img_va, alias_txt_v, vla_vlm_dd],
                             outputs=[out_img_va, out_sum_va])

        gr.Markdown("---")
        console_log_txt = gr.Textbox(
            label="🖥️ 실시간 터미널 로그 (모델 로드/에러 출력)",
            lines=8,
            max_lines=15,
            interactive=False,
        )

        console_timer = gr.Timer(value=1.0, active=True)
        console_timer.tick(
            fn=get_terminal_logs,
            outputs=[console_log_txt],
        )

        # ── 공유 타이머: 모든 탭의 카메라 이미지 자동 갱신 ─────────────────────
        timer = gr.Timer(value=1.0, active=ROS_AVAILABLE or is_camera_running())
        timer.tick(
            fn=timer_tick,
            outputs=[img_v, img_va, img_g, img_q, cam_status_txt],
        )
        timer.tick(
            fn=_get_alias_frame,
            outputs=[img_a],
        )

        # ── 카메라 위젯 버튼 → 타이머 동기화 ───────────────────────────────────
        def _activate_timer(_=None):
            return gr.update(active=is_camera_running())

        def _deactivate_timer(_=None):
            return gr.update(active=False)

        _cam_start_btn.click(
            fn=start_camera, outputs=_cam_proc_st,
        ).then(fn=_activate_timer, outputs=timer)
        _cam_stop_btn.click(
            fn=stop_camera, outputs=_cam_proc_st,
        ).then(fn=_deactivate_timer, outputs=timer)

    return demo


# ─── 진입점 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7863)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--preload", action="store_true")
    args = parser.parse_args()

    if args.preload:
        print("[PRE-LOAD] Pure Kosmos-2")
        _ensure_model("Pure Kosmos-2")

    demo = build_ui()
    # Tailscale IP 우선, 없으면 로컬 라우팅 IP 사용
    import socket as _sock
    _tailscale_ip = None
    try:
        import subprocess as _sp
        _out = _sp.check_output(["ip", "addr"], text=True)
        for _line in _out.splitlines():
            _line = _line.strip()
            if _line.startswith("inet 100."):
                _tailscale_ip = _line.split()[1].split("/")[0]
                break
    except Exception:
        pass
    if not _tailscale_ip:
        try:
            _s = _sock.socket(_sock.AF_INET, _sock.SOCK_DGRAM)
            _s.connect(("8.8.8.8", 80))
            _tailscale_ip = _s.getsockname()[0]
            _s.close()
        except Exception:
            _tailscale_ip = "localhost"
    _root = f"http://{_tailscale_ip}:{args.port}"
    print(f"[START] {_root}  |  ROS={ROS_AVAILABLE}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
        root_path=_root,
    )


if __name__ == "__main__":
    main()

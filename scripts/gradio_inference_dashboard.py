# ── ROS camera_interfaces LD_LIBRARY_PATH 주입 (다른 import보다 먼저) ──────────
import os, sys as _sys
_global_site = "/home/soda/.local/lib/python3.10/site-packages"
if _global_site in _sys.path:
    _sys.path.remove(_global_site)
_sys.path.insert(0, _global_site)

_ROS_WS = "/home/soda/MoNaVLA/ROS_action/install"
_ros_lib_dirs = [f"{_ROS_WS}/camera_interfaces/lib", f"{_ROS_WS}/camera_pub/lib"]
_ros_py_dirs  = [f"{_ROS_WS}/camera_interfaces/local/lib/python3.10/dist-packages"]
_ld = os.environ.get("LD_LIBRARY_PATH", "")
if any(p not in _ld for p in _ros_lib_dirs if os.path.isdir(p)):
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        p for p in _ros_lib_dirs if os.path.isdir(p)
    ) + (":" + _ld if _ld else "")
    _pp = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = ":".join(
        p for p in _ros_py_dirs if os.path.isdir(p)
    ) + (":" + _pp if _pp else "")
    os.environ.setdefault("ROS_DOMAIN_ID", "42")
    os.environ.setdefault("RMW_IMPLEMENTATION", "rmw_fastrtps_cpp")
    os.execv(_sys.executable, [_sys.executable] + _sys.argv)
for _p in _ros_py_dirs:
    if os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
# ─────────────────────────────────────────────────────────────────────────────
import base64
import gc
import io
import os
import sys
import threading
import time
import warnings
from pathlib import Path
import socket

import cv2
import gradio as gr
import matplotlib
import numpy as np
import requests
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="Unable to import Axes3D")

import subprocess
import signal

def get_vla_processes():
    """VLA 관련 프로세스(PID, Command) 목록을 조회합니다."""
    proc_list = []
    try:
        out = subprocess.check_output(["ps", "-eo", "pid,args"], text=True)
        lines = out.splitlines()
        for line in lines[1:]:
            parts = line.strip().split(maxsplit=1)
            if len(parts) < 2:
                continue
            pid_str, cmd = parts
            pid = int(pid_str)
            
            if pid == os.getpid():
                continue
                
            name = None
            if "inference/server.py" in cmd or "serve/proxy_inference_server.py" in cmd:
                name = "VLA Inference Server"
            elif "robot/camera_node.py" in cmd or "camera_proc.py" in cmd:
                name = "Camera Process"
            elif "ros2_controller.py" in cmd:
                name = "ROS2 Controller"
            elif "keyboard_controller.py" in cmd:
                name = "Keyboard Controller"
            elif "gradio_inference_dashboard.py" in cmd and pid != os.getpid():
                name = "Other Dashboard"
                
            if name:
                proc_list.append({"pid": pid, "name": name, "cmd": cmd[:80] + ("..." if len(cmd) > 80 else "")})
    except Exception as e:
        print(f"Error getting processes: {e}")
    return proc_list

def kill_process(pid: int):
    """지정한 PID의 프로세스를 안전하게 또는 강제 종료합니다."""
    try:
        os.kill(pid, signal.SIGTERM)
        import time
        time.sleep(0.5)
        try:
            os.kill(pid, 0)
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
        return f"✅ PID {pid} 종료 성공"
    except Exception as e:
        return f"❌ PID {pid} 종료 실패: {e}"

def get_system_resources():
    """RAM, Swap, Disk, GPU VRAM의 사용량을 미려한 게이지 Bar(HTML)로 시각화합니다."""
    html = []
    
    def render_bar(label, icon, used, total, unit, threshold_warn=85):
        pct = (used / total * 100) if total > 0 else 0
        pct = min(100, max(0, pct))
        
        # 85% 이상일 때는 경고 그라디언트(주황~레드), 그렇지 않으면 블루~퍼플 그라디언트
        if pct >= threshold_warn:
            bar_color = "linear-gradient(90deg, #f97316 0%, #ef4444 100%)"
        else:
            bar_color = "linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%)"
            
        return f"""
        <div style="margin-bottom: 12px; font-family: sans-serif;">
            <div style="display: flex; justify-content: space-between; font-size: 0.85rem; margin-bottom: 4px; color: #cbd5e1;">
                <span>{icon} <b>{label}</b></span>
                <span style="color: {'#f87171' if pct >= threshold_warn else '#38bdf8'}; font-weight: bold;">
                    {used:.1f}{unit} / {total:.1f}{unit} ({pct:.1f}%)
                </span>
            </div>
            <div style="background: #1e293b; border-radius: 8px; height: 14px; overflow: hidden; border: 1px solid #475569; box-shadow: inset 0 2px 4px rgba(0,0,0,0.6);">
                <div style="background: {bar_color}; width: {pct}%; height: 100%; border-radius: 8px; transition: width 0.5s ease-in-out; box-shadow: 0 0 8px rgba(139, 92, 246, 0.4);"></div>
            </div>
        </div>
        """

    # 1. RAM
    try:
        mem_out = subprocess.check_output(["free", "-m"], text=True)
        for line in mem_out.splitlines():
            if line.startswith("Mem:"):
                parts = line.split()
                total = int(parts[1])
                used = int(parts[2])
                html.append(render_bar("RAM Memory", "🧠", used, total, "MB", 85))
            elif line.startswith("Swap:"):
                parts = line.split()
                total = int(parts[1])
                used = int(parts[2])
                html.append(render_bar("Swap Virtual Memory", "💾", used, total, "MB", 85))
    except Exception:
        pass

    # 2. Disk Space
    try:
        disk_out = subprocess.check_output(["df", "-h", "/"], text=True)
        lines = disk_out.splitlines()
        if len(lines) > 1:
            parts = lines[1].split()
            def to_gb(val_str):
                val_str = val_str.upper()
                if val_str.endswith("G"):
                    return float(val_str[:-1])
                elif val_str.endswith("M"):
                    return float(val_str[:-1]) / 1024.0
                elif val_str.endswith("T"):
                    return float(val_str[:-1]) * 1024.0
                return float(val_str)
                
            total_gb = to_gb(parts[1])
            used_gb = to_gb(parts[2])
            html.append(render_bar("Disk Space (Root)", "💿", used_gb, total_gb, "GB", 90))
    except Exception:
        pass

    # 3. GPU VRAM (Orin Unified)
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            # Orin Nano 가용 램 (15.3GB)을 Unified total로 가정
            system_total_vram = 15656.0 
            html.append(render_bar("GPU VRAM (Orin Unified)", "🎮", allocated, system_total_vram, "MB", 80))
        else:
            html.append("""
            <div style="font-size: 0.85rem; color: #94a3b8; font-family: sans-serif; margin-top: 8px;">
                🎮 <b>GPU VRAM:</b> CUDA Unavailable
            </div>
            """)
    except Exception:
        pass

    return "\n".join(html)

def update_monitor():
    res_str = get_system_resources()
    procs = get_vla_processes()
    choices = []
    for p in procs:
        label = f"[{p['pid']}] {p['name']} ({p['cmd']})"
        choices.append((label, str(p['pid'])))
        
    if not choices:
        choices = [("No active VLA processes found", "")]
        
    return res_str, gr.Dropdown(choices=choices, value=choices[0][1] if choices else "")

def handle_kill_proc(pid_str):
    if not pid_str:
        return "⚠️ 선택된 프로세스가 없습니다.", *update_monitor()
    try:
        pid = int(pid_str)
        kill_msg = kill_process(pid)
        res_str, dropdown_update = update_monitor()
        return f"{kill_msg} (모니터 갱신 완료)", res_str, dropdown_update
    except Exception as e:
        return f"❌ 오류 발생: {e}", *update_monitor()


def handle_start_server():
    import subprocess
    try:
        out = subprocess.check_output(["pgrep", "-f", "inference/server.py"], text=True)
        if out.strip():
            res_str, dropdown_update = update_monitor()
            return "⚠️ VLA Inference Server가 이미 실행 중입니다.", res_str, dropdown_update
    except subprocess.CalledProcessError:
        pass

    try:
        log_file = "/tmp/vla_server_gradio.log"
        cwd_dir = "/home/soda/MoNa-pi"
        
        proc = subprocess.Popen(
            ["python3", "inference/server.py", "--config", "configs/serbot2.yaml", "--ckpt", "checkpoints/best", "--port", "8082"],
            cwd=cwd_dir,
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True
        )
        # 서버 기동 프로세스가 뜨는 시간 대기
        time.sleep(2.0)
        
        res_str, dropdown_update = update_monitor()
        return f"🚀 VLA Inference Server 기동 성공 (PID {proc.pid}) - 로그: {log_file}", res_str, dropdown_update
    except Exception as e:
        res_str, dropdown_update = update_monitor()
        return f"❌ VLA Inference Server 기동 실패: {e}", res_str, dropdown_update


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".vla_env_settings"
# Exp47: path_type 키를 직접 입력하거나 자연어 instruction 사용 가능.
# path_type 키 목록: center_straight, center_left, center_right,
#   left_straight, left_left, left_right,
#   right_straight, right_right, right_left
# 미매칭 시 bbox cx 위치에서 자동 추론 (right_right / left_left / center_straight).
DEFAULT_INSTRUCTION = "[FORWARD] the gray basket on right"
PATH_TYPES = [
    "right_right", "right_left", "right_straight",
    "center_straight", "center_left", "center_right",
    "left_straight", "left_left", "left_right",
]
GOAL_NAV_PRESETS = [
    "[FORWARD] the gray basket on right",
    "[FORWARD] the gray basket on left",
    "[FORWARD] the gray basket",
    "[FORWARD] the door",
    "[FORWARD] the corridor on the left",
    "[FORWARD] the corridor on the right",
]

# 실험 모드: (표시 이름, instruction, backend_instruction_mode, speed_scaling, grounding_skip_n)
EXP_MODES = {
    "GoalNav-fixed (Exp49, 고정속도)": {
        "instruction": GOAL_NAV_PRESETS[0],
        "backend_mode": "GoalNav (exp49)",
        "model": "exp49",
        "speed_scaling": False,
        "grounding_skip_n": 3,
        "smooth_enabled": False,
        "desc": "기본 GoalNav — 96.4% val acc",
        "config": None,
        "checkpoint": "runs/v5_nav/mlp/exp49/exp49_mlp.pt",
    },
    "GoalNav-scaled (Exp49, 거리비례속도)": {
        "instruction": GOAL_NAV_PRESETS[0],
        "backend_mode": "GoalNav (exp49)",
        "model": "exp49",
        "speed_scaling": True,
        "grounding_skip_n": 3,
        "smooth_enabled": False,
        "desc": "기본 GoalNav + 거리비례속도 — 96.4% val acc",
        "config": None,
        "checkpoint": "runs/v5_nav/mlp/exp49/exp49_mlp.pt",
    },
    "GoalNav (Exp50, flip-aug)": {
        "instruction": GOAL_NAV_PRESETS[0],
        "backend_mode": "GoalNav (exp50)",
        "model": "exp50",
        "speed_scaling": False,
        "grounding_skip_n": 3,
        "smooth_enabled": False,
        "desc": "flip augmentation 2x — 92.0% val acc",
    },
    "GoalNav (Exp51, crop-aug)": {
        "instruction": GOAL_NAV_PRESETS[0],
        "backend_mode": "GoalNav (exp51)",
        "model": "exp51",
        "speed_scaling": False,
        "grounding_skip_n": 3,
        "smooth_enabled": False,
        "desc": "crop augmentation 4x — 93.4% val acc",
    },
    "GoalNav (Exp52, lang+vis) ⚠️": {
        "instruction": GOAL_NAV_PRESETS[0],
        "backend_mode": "GoalNav (exp52)",
        "model": "exp52",
        "speed_scaling": False,
        "grounding_skip_n": 3,
        "smooth_enabled": False,
        "desc": "⚠️ lang+vis 2048-dim — 실시간 추출 미지원, 실험적",
    },
    "GoalNav (Exp53, CLIP-LoRA)": {
        "instruction": GOAL_NAV_PRESETS[0],
        "backend_mode": "GoalNav (exp53)",
        "model": "exp53",
        "speed_scaling": False,
        "grounding_skip_n": 3,
        "smooth_enabled": False,
        "desc": "CLIP LoRA fine-tuned vision encoder — 94.7% val acc",
        "config": "configs/bbox_nav_exp53_clip_lora.json",
        "checkpoint": "runs/v5_nav/mlp/exp53_clip_lora.pt",
    },
    "GoalNav (Exp54_s2v2, Best)": {
        "instruction": GOAL_NAV_PRESETS[0],
        "backend_mode": "GoalNav (exp54_s2v2)",
        "model": "exp54_s2v2",
        "speed_scaling": False,
        "grounding_skip_n": 3,
        "smooth_enabled": False,
        "desc": "Stage2 v2 MLP + image projection — 96.7% CL (최고 성능)",
        "config": "configs/exp54_stage2_action.json",
        "checkpoint": "runs/v5_nav/mlp/exp54/stage2_v2/stage2_v2_mlp.pt",
    },
    "PathType-fixed (Exp47, 고정속도)": {
        "instruction": "right_right",
        "backend_mode": "PathType (exp47)",
        "model": None,
        "speed_scaling": False,
        "grounding_skip_n": 1,
        "smooth_enabled": False,
        "desc": "PathType 분류기 — 고정속도",
    },
}
EXP_MODE_NAMES = list(EXP_MODES.keys())
LINEAR_SPEED_VLA = 1.15
ANGULAR_SPEED_VLA = 1.15

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("ROS_HOME", "/tmp/ros")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["ROS_HOME"]).mkdir(parents=True, exist_ok=True)

os.environ["ROS_DOMAIN_ID"] = "42"
os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"
print(f"🔧 Forced ROS_DOMAIN_ID={os.environ['ROS_DOMAIN_ID']}, RMW={os.environ['RMW_IMPLEMENTATION']}")


def load_env() -> None:
    env_path = Path(os.getenv("VLA_ENV_PATH", str(DEFAULT_ENV_PATH)))
    if not env_path.exists():
        fallback = Path("/home/billy/25-1kp/vla/.vla_env_settings")
        if fallback.exists():
            env_path = fallback
    if not env_path.exists():
        return

    with env_path.open("r") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line.startswith("export "):
                continue
            try:
                key, val = line.replace("export ", "", 1).split("=", 1)
            except ValueError:
                continue
            os.environ[key] = val.strip('"').strip("'")
    print(f"✅ Loaded environment from {env_path}")


load_env()

DEFAULT_API_URL = os.getenv("VLA_API_SERVER", "http://localhost:8082")
API_KEY = os.getenv("VLA_API_KEY", "vla_devel_key_2026")
DEFAULT_BACKEND_MODE = os.getenv(
    "VLA_DASHBOARD_BACKEND",
    "API Server" if (os.getenv("VLA_SERVER_ROLE") == "jetson" or os.getenv("VLA_API_SERVER")) else "Local Runtime",
)

sys.path.append(str(PROJECT_ROOT / "scripts"))
try:
    from inference_logger import get_logger

    logger_instance = get_logger()
except ImportError:
    logger_instance = None

sys.path.insert(0, str(PROJECT_ROOT))
from robovlm_nav.serve.inference_server import MobileVLAInference
from robovlm_nav.serve.vla_control_utils import VLAControlManager
from scripts.utils.camera_proc import camera_control_widget, start_camera, stop_camera, is_camera_running


def prepend_env_path(key: str, value: str) -> None:
    current = os.environ.get(key, "")
    parts = [p for p in current.split(os.pathsep) if p]
    if value not in parts:
        os.environ[key] = value if not parts else f"{value}{os.pathsep}{current}"


def setup_ros_paths() -> None:
    ros_humble = Path("/opt/ros/humble")
    if ros_humble.exists():
        prepend_env_path("AMENT_PREFIX_PATH", str(ros_humble))
        prepend_env_path("CMAKE_PREFIX_PATH", str(ros_humble))
        if (ros_humble / "lib").exists(): prepend_env_path("LD_LIBRARY_PATH", str(ros_humble / "lib"))
        if (ros_humble / "opt/rviz_ogre_vendor/lib").exists(): prepend_env_path("LD_LIBRARY_PATH", str(ros_humble / "opt/rviz_ogre_vendor/lib"))
        for candidate in (ros_humble / "local/lib/python3.10/dist-packages", ros_humble / "lib/python3.10/site-packages"):
            if candidate.exists() and str(candidate) not in sys.path:
                sys.path.append(str(candidate))
                prepend_env_path("PYTHONPATH", str(candidate))

    ros_ws = Path(os.getenv("VLA_ROS_WS", str(PROJECT_ROOT / "ROS_action")))
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
        share_path = pkg / "share"
        if share_path.exists():
            prepend_env_path("AMENT_PREFIX_PATH", str(pkg))
        local_path = pkg / "local/lib/python3.10/dist-packages"
        site_path = pkg / "lib/python3.10/site-packages"
        for candidate in (local_path, site_path):
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
    from geometry_msgs.msg import Twist
    from camera_interfaces.srv import GetImage

    ROS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ROS2 environment partially missing: {e}")

    class Node:  # stub so class definitions below don't NameError
        pass

    class ReentrantCallbackGroup:
        pass


CC_PARAMS = {
    "r_gain": 1.0,
    "g_gain": 1.0,
    "b_gain": 1.0,
}


def correct_image(img_pil: Image.Image) -> Image.Image:
    img_rgb = np.array(img_pil).astype(np.float32)
    r, g, b = cv2.split(img_rgb)
    r = r * CC_PARAMS["r_gain"]
    g = g * CC_PARAMS["g_gain"]
    b = b * CC_PARAMS["b_gain"]
    img_corrected = cv2.merge([r, g, b])
    return Image.fromarray(np.clip(img_corrected, 0, 255).astype(np.uint8))


def scan_local_files():
    ckpt_tuples = []
    for root_dir in (PROJECT_ROOT, PROJECT_ROOT / "runs"):
        if not root_dir.exists():
            continue
        pattern = "**/*" if root_dir.name == "runs" else "*"
        for path in root_dir.glob(pattern):
            if path.suffix not in {".ckpt", ".pth", ".pt"} or not path.is_file():
                continue
            try:
                rel = path.relative_to(PROJECT_ROOT)
                display_name = str(rel)
            except ValueError:
                display_name = path.name
            ckpt_tuples.append((display_name, str(path)))

    configs_dir = PROJECT_ROOT / "configs"
    conf_tuples = []
    if configs_dir.exists():
        for path in configs_dir.glob("*.json"):
            conf_tuples.append((path.name, str(path)))

    return sorted(set(ckpt_tuples)), sorted(conf_tuples)


def pick_default_choice(choices, env_key: str):
    preferred = os.getenv(env_key)
    if preferred:
        for _label, value in choices:
            if value == preferred:
                return value
    return choices[0][1] if choices else None


def to_precision(precision_label: str) -> str:
    return "int8" if precision_label == "INT8 (Fast)" else "fp16"


def short_model_name(path_str: str) -> str:
    if not path_str or path_str == "N/A":
        return "N/A"
    path = Path(path_str)
    if len(path.parts) >= 2:
        return f"{path.parent.name}/{path.name}"
    return path.name


def pick_server_port(default_port: int, span: int = 20) -> int:
    try:
        for port in range(default_port, default_port + span):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind(("127.0.0.1", port))
                except OSError:
                    continue
            return port
    except PermissionError:
        return default_port
    return default_port


class LocalSharedRuntime:
    def __init__(self):
        self.model = None
        self.info_cache = {
            "model_loaded": False,
            "model_name": "Unavailable",
            "checkpoint_path": "N/A",
            "config_path": "N/A",
            "precision": "N/A",
            "device": "N/A",
            "action_dim": 3,
        }

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
            gc.collect()
        self.info_cache = {
            "model_loaded": False,
            "model_name": "Unavailable",
            "checkpoint_path": "N/A",
            "config_path": "N/A",
            "precision": "N/A",
            "device": "N/A",
            "action_dim": 3,
        }

    def load_model(self, checkpoint_path: str, config_path: str, precision: str, refresh: bool = False):
        if refresh:
            self.unload()

        use_quant = precision == "int8"
        device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "" else "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        except Exception:
            device = "cpu"

        self.model = MobileVLAInference(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            device=device,
            use_quant=use_quant,
        )
        self.info_cache = {
            "model_loaded": True,
            "model_name": Path(config_path).stem,
            "checkpoint_path": checkpoint_path,
            "config_path": config_path,
            "precision": precision,
            "device": device,
            "action_dim": 3,
        }
        return self.model

    def reset(self, instruction: str = "N/A") -> None:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        self.model.reset(instruction=instruction)

    def predict(self, image_base64: str, instruction: str) -> dict:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        action, latency_ms, chunk = self.model.predict(
            image_base64=image_base64,
            instruction=instruction,
        )
        return {
            "action": action.tolist(),
            "latency_ms": float(latency_ms),
            "chunk": chunk.tolist(),
        }

    def get_model_info(self) -> dict:
        return dict(self.info_cache)


shared_runtime = LocalSharedRuntime()


class LocalInferenceBackend:
    name = "Local Runtime"

    def load_model(self, checkpoint_path: str, config_path: str, precision: str) -> dict:
        model = shared_runtime.load_model(
            checkpoint_path=checkpoint_path,
            config_path=config_path,
            precision=precision,
            refresh=True,
        )
        info = shared_runtime.get_model_info()
        return {
            "status": "success",
            "message": f"✅ Loaded: {short_model_name(model.checkpoint_path)} ({info['precision']})",
            "info": info,
        }

    def reset(self, instruction: str) -> str:
        shared_runtime.reset(instruction=instruction)
        return "✅ Local history cleared"

    def predict(self, image: Image.Image, instruction: str) -> dict:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return shared_runtime.predict(image_base64=img_b64, instruction=instruction)

    def info(self) -> dict:
        return shared_runtime.get_model_info()


class ApiInferenceBackend:
    name = "API Server"

    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip("/")

    def _headers(self) -> dict:
        return {"X-API-Key": API_KEY}

    def _post(self, path: str, payload: dict) -> dict:
        response = requests.post(
            f"{self.api_url}{path}",
            json=payload,
            headers=self._headers(),
            timeout=60,
        )
        response.raise_for_status()
        return response.json()

    def load_model(self, checkpoint_path: str, config_path: str, precision: str) -> dict:
        payload = {
            "checkpoint_path": checkpoint_path,
            "config_path": config_path,
            "precision": precision,
            "refresh": True,
        }
        result = self._post("/model/load", payload)
        info = self.info()
        return {
            "status": result.get("status", "success"),
            "message": f"✅ API loaded: {short_model_name(info['checkpoint_path'])} ({info['precision']})",
            "info": info,
        }

    def reset(self, instruction: str) -> str:
        self._post("/reset", {})
        return f"✅ API history cleared ({instruction})"

    def predict(self, image: Image.Image, instruction: str) -> dict:
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return self._post(
            "/predict",
            {
                "image": img_b64,
                "instruction": instruction,
                "strategy": "receding_horizon",
            },
        )

    def set_config(self, speed_scaling: bool, grounding_skip_n: int, model: str | None = None, smooth_enabled: bool | None = None) -> dict:
        try:
            payload: dict = {"speed_scaling": speed_scaling, "grounding_skip_n": grounding_skip_n}
            if model is not None:
                payload["model"] = model
            if smooth_enabled is not None:
                payload["smooth_enabled"] = smooth_enabled
            return self._post("/config", payload)
        except Exception as e:
            return {"status": "error", "reason": str(e)}

    def info(self) -> dict:
        response = requests.get(
            f"{self.api_url}/model/info",
            headers=self._headers(),
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        
        # MoNa-pi 통합 추론 서버 스키마와의 호환성을 위한 키 정규화
        if "model_loaded" not in data:
            data["model_loaded"] = data.get("engine_ready", True)
        if "checkpoint_path" not in data:
            data["checkpoint_path"] = data.get("checkpoint", "N/A")
        if "precision" not in data:
            data["precision"] = "fp32"
        if "config_path" not in data:
            data["config_path"] = "N/A"
            
        return data


def make_backend(mode: str, api_url: str):
    if mode == "API Server":
        return ApiInferenceBackend(api_url)
    return LocalInferenceBackend()


class ROSDashboardNode(Node):
    def __init__(self):
        import os as _os
        super().__init__(f"gradio_dashboard_{_os.getpid()}")
        self.callback_group = ReentrantCallbackGroup()
        self.cv_bridge = CvBridge()
        self.get_image_client = self.create_client(
            GetImage, "get_image_service", callback_group=self.callback_group
        )
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10, callback_group=self.callback_group)
        self.control = VLAControlManager(self, default_throttle=50, move_duration=0.4)

        # 백그라운드 이미지 수집 관련 초기화
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.capture_active = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

    def _capture_loop(self):
        # 백그라운드에서 주기적으로 이미지를 획득하여 self.latest_frame에 업데이트
        while self.capture_active:
            start_t = time.time()
            try:
                # 백그라운드 스레드이므로 대기는 UI를 방해하지 않음
                if not self.get_image_client.wait_for_service(timeout_sec=0.1):
                    time.sleep(0.05)
                    continue
                request = GetImage.Request()
                future = self.get_image_client.call_async(request)
                deadline = time.time() + 0.15
                while not future.done() and time.time() < deadline and self.capture_active:
                    time.sleep(0.005)
                if future.done():
                    try:
                        response = future.result()
                        if response and response.image.data:
                            cv_image = self.cv_bridge.imgmsg_to_cv2(response.image, "bgr8")
                            img_rgb = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
                            with self.frame_lock:
                                self.latest_frame = img_rgb
                            
                            elapsed = (time.time() - start_t) * 1000
                            if elapsed > 100.0:
                                print(f"[Dashboard Background Capture] ⚠️ Slow frame fetch: {elapsed:.1f} ms")
                    except Exception:
                        pass
            except Exception as e:
                if "context is invalid" in str(e) or "rcl" in str(e).lower():
                    print(f"[Dashboard Background Capture] ROS context 무효 → 재초기화 시도")
                    self.capture_active = False
                    threading.Thread(target=_init_ros_node, daemon=True).start()
                    break
                else:
                    print(f"[Dashboard Background Capture] Error: {e}")
            time.sleep(0.04) # 약 25 fps 속도로 수집하여 카메라 서비스 노드 부하 최소화

    def get_inference_frame(self):
        # UI 스레드는 락만 잠깐 걸어서 최신 이미지 복제본을 즉시 반환 (블로킹 0ms)
        with self.frame_lock:
            return self.latest_frame

    def generate_trajectory_plot(self, full_chunk):
        if full_chunk is None or len(full_chunk) == 0:
            return None

        dt = 0.2
        traj_x, traj_y = [0.0], [0.0]
        curr_x, curr_y = 0.0, 0.0
        for step in full_chunk:
            curr_x += float(step[0]) * dt
            curr_y += float(step[1]) * dt
            traj_x.append(curr_x)
            traj_y.append(curr_y)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(0, 0, "ko", markersize=8, label="Start")
        ax.arrow(0, 0, 0.2, 0, head_width=0.05, head_length=0.05, fc="k", ec="k")
        ax.plot(traj_x, traj_y, "b-", linewidth=3, alpha=0.8)
        ax.plot(traj_x[-1], traj_y[-1], "b*", markersize=10)
        ax.set_title("Predicted Trajectory (2D XY)")
        ax.set_xlabel("Forward (X) [m]")
        ax.set_ylabel("Left/Right (Y) [m]")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.set_aspect("equal")
        all_points = np.column_stack((traj_x, traj_y))
        mins = np.min(all_points, axis=0) - 0.5
        maxs = np.max(all_points, axis=0) + 0.5
        ax.set_xlim(min(mins[0], -0.5), max(maxs[0], 2.0))
        ax.set_ylim(min(mins[1], -1.0), max(maxs[1], 1.0))
        return fig

    def generate_action_history_plot(self, action_history):
        if not action_history or len(action_history) == 0:
            return None

        steps = list(range(1, len(action_history) + 1))
        lx_vals = [act[0] for act in action_history]
        ly_vals = [act[1] for act in action_history]
        az_vals = [act[2] for act in action_history]

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(steps, lx_vals, "r-o", linewidth=2, label="Linear X (Forward)")
        ax.plot(steps, ly_vals, "g-s", linewidth=2, label="Linear Y (Left/Right)")
        ax.plot(steps, az_vals, "b-^", linewidth=2, label="Angular Z (CCW/CW)")
        ax.set_title("Executed Action History (lx, ly, az)")
        ax.set_xlabel("Inference Step")
        ax.set_ylabel("Velocity Value")
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        
        # 스텝이 많을 때 틱 표시 조절
        if len(steps) > 10:
            ax.set_xticks(steps[::2])
        else:
            ax.set_xticks(steps)
            
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper right")
        fig.tight_layout()
        return fig


ros_node = None
_ros_node_lock = threading.Lock()
_ros_spin_active = False

def _spin_forever(node):
    """rclpy.spin 대신 직접 spin_once 루프 — ExternalShutdownException 시 자동 재시도."""
    global _ros_spin_active
    while _ros_spin_active:
        try:
            rclpy.spin_once(node, timeout_sec=0.1)
        except rclpy.executors.ExternalShutdownException:
            break
        except Exception as e:
            print(f"[Dashboard] spin 오류: {e}")
            break

def _init_ros_node():
    global ros_node, _ros_spin_active
    try:
        _ros_spin_active = False
        if ros_node is not None:
            ros_node.capture_active = False
        # 기존 context 종료 시도 (오류 무시)
        try:
            rclpy.shutdown()
        except Exception:
            pass
        # rclpy.init()을 인자 없이 호출하여 글로벌 디폴트 context 초기화
        rclpy.init()
        node = ROSDashboardNode()
        _ros_spin_active = True
        threading.Thread(target=_spin_forever, args=(node,), daemon=True).start()
        ros_node = node
        print("[Dashboard] ROSDashboardNode 초기화 ✅")
        return True
    except Exception as e:
        ros_node = None
        _ros_spin_active = False
        print(f"[Dashboard] ROSDashboardNode 초기화 실패: {e}")
        return False

if ROS_AVAILABLE:
    _init_ros_node()


def annotate_image(img: Image.Image, bbox: dict | None = None, draw_grid: bool = True) -> Image.Image:
    """카메라 이미지에 격자, BBox 및 실시간 VLA 인코더 정보(지시어, 속도, Latency 등) 오버레이를 그려 반환."""
    arr = np.array(img)
    h, w = arr.shape[:2]

    # 1. 3x3 가이드 격자 그리기
    if draw_grid:
        color = (100, 255, 100) # 녹색
        overlay = arr.copy()
        cv2.line(overlay, (w // 3, 0), (w // 3, h), color, 1)
        cv2.line(overlay, (2 * w // 3, 0), (2 * w // 3, h), color, 1)
        cv2.line(overlay, (0, h // 3), (w, h // 3), color, 1)
        cv2.line(overlay, (0, 2 * h // 3), (w, 2 * h // 3), color, 1)
        cv2.addWeighted(overlay, 0.3, arr, 0.7, 0, arr)

    # 2. VLA 추론 상태 및 지시어 오버레이 박스 (상단)
    is_running = state.get("is_running", False)
    
    instr = state.get("current_instruction", "Navigate to the goal")
    model_name = Path(state.get("model_path", "checkpoints/best")).name
    latency = state.get("last_latency", 0.0)
    step = state.get("step_count", 0)

    # 상단 정보 오버레이용 반투명 Rect (높이 76픽셀로 확장하여 2줄 텍스트 수용)
    overlay = arr.copy()
    cv2.rectangle(overlay, (10, 10), (w - 10, 86), (15, 23, 42), -1) # Sleek Dark Blue
    cv2.addWeighted(overlay, 0.75, arr, 0.25, 0, arr)

    # 상단 정보 텍스트 쓰기 (Font scale 및 두께 확장, 1줄: Status, Model, Latency)
    status_str = f"RUNNING (Step {step})" if is_running else "STOPPED"
    status_color = (100, 255, 100) if is_running else (150, 150, 150)
    
    cv2.putText(arr, f"STATUS: {status_str}", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)
    cv2.putText(arr, f"Model: {model_name}", (280, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Latency: 우측 상단 3개 아이콘 영역과의 충돌 방지를 위해 X좌표를 w - 240 으로 정렬
    lat_str = f"Latency: {latency:.1f}ms" if latency > 0 else "Latency: N/A"
    cv2.putText(arr, lat_str, (w - 240, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (100, 200, 255), 2, cv2.LINE_AA)

    # 2줄: Prompt (지시어)를 여유로운 크기로 표시
    cv2.putText(arr, f"Prompt: {instr}", (25, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (220, 255, 255), 2, cv2.LINE_AA)

    # 3. 예측 궤적 및 2D XY 포인트 투영 (하단 반투명 박스)
    last_chunk = state.get("last_chunk")
    if last_chunk is not None and len(last_chunk) > 0 and is_running:
        overlay = arr.copy()
        cv2.rectangle(overlay, (10, h - 65), (w - 10, h - 10), (15, 23, 42), -1)
        cv2.addWeighted(overlay, 0.75, arr, 0.25, 0, arr)

        dt = 0.2
        curr_x, curr_y = 0.0, 0.0
        pts_str = []
        scr_pts = [(w // 2, h - 20)]
        for i, step_act in enumerate(last_chunk[:5]):
            vx, vy = float(step_act[0]), float(step_act[1])
            curr_x += vx * dt
            curr_y += vy * dt
            pts_str.append(f"P{i+1}:({curr_x:.2f},{curr_y:.2f})")
            
            # 스크린 투영 좌표 계산 (왜곡 투영)
            proj_y = int(h - 20 - (curr_x * 80)) 
            proj_x = int(w // 2 - (curr_y * 120 / (curr_x + 0.5)))
            if 0 <= proj_x < w and 0 <= proj_y < h:
                scr_pts.append((proj_x, proj_y))

        # 1) 스크린 상에 예상 주행 궤적(선) 그리기
        for idx in range(len(scr_pts) - 1):
            cv2.line(arr, scr_pts[idx], scr_pts[idx+1], (50, 150, 255), 2, cv2.LINE_AA)
            cv2.circle(arr, scr_pts[idx+1], 3, (100, 200, 255), -1)

        # 2) 하단 텍스트 정보 (글자 크기 및 두께 확장)
        traj_txt = " / ".join(pts_str[:4])
        cv2.putText(arr, f"Traj: {traj_txt}", (25, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (150, 220, 255), 2, cv2.LINE_AA)

    # 4. BBox 오버레이 그리기 (존재할 경우)
    if bbox:
        cx_px = int(bbox["cx"] * w)
        cy_px = int(bbox["cy"] * h)
        label = str(bbox.get("entity", "bbox"))

        if "x1" in bbox:
            x1 = int(bbox["x1"] * w)
            y1 = int(bbox["y1"] * h)
            x2 = int(bbox["x2"] * w)
            y2 = int(bbox["y2"] * h)
            cv2.rectangle(arr, (x1, y1), (x2, y2), (255, 80, 80), 2)
        else:
            r = 10
            cv2.line(arr, (cx_px - r, cy_px), (cx_px + r, cy_px), (255, 80, 80), 2)
            cv2.line(arr, (cx_px, cy_px - r), (cx_px, cy_px + r), (255, 80, 80), 2)

        cv2.circle(arr, (cx_px, cy_px), 4, (255, 80, 80), -1)
        cv2.putText(arr, label[:20], (max(cx_px - 40, 0), max(cy_px - 10, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 80, 80), 2, cv2.LINE_AA)

    return Image.fromarray(arr)


state = {
    "auto_inference": False,
    "is_running": False,
    "is_busy": False,
    "step_count": 0,
    "last_img": None,
    "current_log": "Ready",
    "camera_status": "Unknown",
    "model_status": "Not Loaded",
    "model_path": "N/A",
    "action_history": [],   # [(lx, ly, az), ...] 추론 중 실행된 액션 기록
    "is_returning": False,
    "logs": ["Ready..."],
    
    # 백그라운드 추론 전용 상태 필드
    "last_latency": 0.0,
    "last_chunk": None,
    "current_instruction": "Navigate to the goal",
    "gt_object": "",
    "last_bbox": None,
    "inference_thread": None,
    "inference_run_id": 0,
    "stop_requested": False,
    "logger_active": False,
    "last_report_path": None,
    
    "ui_latency_val": "0 ms",
    "ui_action_val": "0, 0, 0",
    "ui_chunk_val": "Waiting...",
    "ui_run_status_box": "Stopped",
    "ui_fig": None,
    "ui_action_fig": None,
}


def add_console_log(msg: str):
    import datetime
    now_str = datetime.datetime.now().strftime("%H:%M:%S")
    formatted = f"[{now_str}] {msg}"
    logs = state.setdefault("logs", [])
    logs.append(formatted)
    if len(logs) > 20:
        logs.pop(0)


def get_console_logs_str() -> str:
    return "\n".join(state.get("logs", []))


def finish_logger_session(status: str) -> str | None:
    if not logger_instance or not state.get("logger_active"):
        return None
    try:
        report_path = logger_instance.end_session(status=status)
        state["last_report_path"] = report_path
        return report_path
    finally:
        state["logger_active"] = False


def backend_model_info(mode: str, api_url: str) -> dict:
    try:
        return make_backend(mode, api_url).info()
    except Exception:
        return {
            "model_loaded": False,
            "model_name": "Unavailable",
            "checkpoint_path": "N/A",
            "config_path": "N/A",
            "precision": "N/A",
            "device": "N/A",
            "action_dim": 3,
        }


def load_model_wrapper(backend_mode: str, api_url: str, precision_label: str, ckpt_path: str, config_path: str):
    try:
        result = make_backend(backend_mode, api_url).load_model(
            checkpoint_path=ckpt_path,
            config_path=config_path,
            precision=to_precision(precision_label),
        )
        info = result["info"]
        state["model_status"] = result["message"]
        state["model_path"] = info["checkpoint_path"]
        add_console_log(f"📂 Model Loaded: {result['message']}")
        return result["message"], info["checkpoint_path"]
    except Exception as e:
        state["model_status"] = "Load Failed"
        msg = f"❌ Load Failed: {e}"
        add_console_log(msg)
        return msg, state["model_path"]


def _is_active_run(run_id: int) -> bool:
    return (
        state.get("inference_run_id") == run_id
        and state.get("is_running")
        and not state.get("stop_requested")
    )


class StaleInferenceResult(RuntimeError):
    pass


def _background_inference_loop(backend_mode: str, api_url: str, mode: str, run_id: int):
    """자율주행 추론 및 제어를 전담하는 백그라운드 스레드 루프"""
    global ros_node
    add_console_log("🌀 백그라운드 추론 루프 시작")
    
    try:
        # Step 1: 시작 대기 단계
        if not ros_node:
            add_console_log("❌ ROS 노드가 초기화되지 않아 추론을 시작할 수 없습니다.")
            state["is_running"] = False
            return

        state["step_count"] += 1
        current_step = state["step_count"]
        instr = state["current_instruction"]

        if logger_instance:
            logger_instance.start_session(short_model_name(state["model_path"]), instr, instruction_mode=backend_mode)
            state["logger_active"] = True
            state["last_report_path"] = None
            if state.get("gt_object"):
                logger_instance.data["gt_object"] = state["gt_object"]
            # 1단계 프레임 로깅
            img = ros_node.get_inference_frame()
            if img:
                logger_instance.log_step(current_step, [0.0, 0.0, 0.0], 0, image=img)
                
        ros_node.control.robust_stop(source="inference_start")
        try:
            make_backend(backend_mode, api_url).reset(instr)
        except Exception as e:
            add_console_log(f"❌ Reset failed: {e}")
            state["is_running"] = False
            return
            
        state["ui_run_status_box"] = f"Running (step 1)..."
        time.sleep(1.0)  # 시작 웜업 딜레이

        # 주행 루프
        while ROS_AVAILABLE and ros_node and _is_active_run(run_id):
            if state.get("stop_requested"):
                break

            img = ros_node.get_inference_frame()
            if img is None:
                add_console_log("⚠️ [Inference Thread] 카메라 영상을 대기 중...")
                time.sleep(0.2)
                continue
                
            state["step_count"] += 1
            current_step = state["step_count"]

            # 1. 추론 실행
            try:
                result = run_backend_inference(img, instr, backend_mode, api_url, current_step=current_step, run_id=run_id)
            except StaleInferenceResult:
                add_console_log(f"⏹️ Step {current_step} stale result discarded")
                break

            if not _is_active_run(run_id):
                add_console_log(f"⏹️ Step {current_step} result discarded after STOP")
                break
            
            # 2. 상태 업데이트
            state["last_latency"] = result.get("latency_ms", 0.0)
            state["last_chunk"] = result.get("chunk")
            
            state["ui_latency_val"] = result["lat_str"]
            state["ui_action_val"] = result["act_str"]
            state["ui_chunk_val"] = result["chunk_display"]
            state["ui_fig"] = ros_node.generate_trajectory_plot(result["chunk"])
            state["ui_action_fig"] = ros_node.generate_action_history_plot(state["action_history"])
            
            if logger_instance:
                logger_instance.log_step(
                    current_step,
                    result["action"],
                    result.get("latency_ms", 0),
                    result["chunk"],
                    image=img,
                    raw_action=result.get("raw_action"),
                    predicted_label=result.get("predicted_label"),
                    grounding_caption=result.get("grounding_caption"),
                    goal_near=result.get("goal_near"),
                    strategy=result.get("strategy"),
                    bbox=result.get("bbox"),
                    instruction_used=result.get("instruction_used"),
                    matched_path_type=result.get("matched_path_type"),
                    speed_scale=result.get("speed_scale"),
                    grounding_cached=result.get("grounding_cached"),
                )

            log = f"Step {current_step} | {result['log_str']}"
            state["ui_run_status_box"] = f"Running (step {current_step})"

            # 정지 조건 A: 18스텝 도달
            if mode == "Inference (18-step)" and current_step >= 18:
                state["is_running"] = False
                state["step_count"] = 0
                ros_node.control.robust_stop(source="step_limit_reached")
                report_path = finish_logger_session("limit_reached")
                if report_path:
                    log = f"🛑 18-step Limit Reached! | Log: {Path(report_path).name}"
                else:
                    log = "🛑 18-step Limit Reached!"
                add_console_log(log)
                state["ui_run_status_box"] = "Stopped (Limit Reached)"
                break

            # 정지 조건 B: 목적지 도달
            if result.get("goal_near"):
                state["is_running"] = False
                state["step_count"] = 0
                ros_node.control.robust_stop(source="goal_reached")
                report_path = finish_logger_session("goal_reached")
                if report_path:
                    log = f"🎯 Goal Reached! (step {current_step}) | Log: {Path(report_path).name}"
                else:
                    log = f"🎯 Goal Reached! (step {current_step})"
                add_console_log(log)
                state["ui_run_status_box"] = "Stopped (Goal Reached)"
                break

            # 모션 사이클(0.4초) 정밀 주기를 위한 대기시간 보정
            elapsed = state["last_latency"] / 1000.0
            sleep_time = max(0.01, 0.40 - elapsed)
            time.sleep(sleep_time)

    except Exception as e:
        add_console_log(f"❌ [Inference Thread] 오류: {e}")
        if state.get("inference_run_id") == run_id:
            state["is_running"] = False
        if ros_node and state.get("inference_run_id") == run_id:
            ros_node.control.robust_stop(source="inference_thread_exception")
    finally:
        status = "stopped" if state.get("stop_requested") else "completed"
        report_path = finish_logger_session(status)
        if report_path:
            add_console_log(f"📝 Session saved: {Path(report_path).name}")
        if state.get("inference_run_id") == run_id:
            state["is_busy"] = False
            state["is_running"] = False
            state["stop_requested"] = False
            state["inference_thread"] = None
        add_console_log("⏹️ 백그라운드 추론 루프 종료")


def set_running(running: bool, backend_mode: str, api_url: str, instruction: str, gt_object: str = "", mode: str = "Inference (Auto)"):
    state["is_running"] = running
    state["gt_object"] = gt_object
    if running:
        if state["inference_thread"] is not None:
            state["stop_requested"] = True
            state["is_running"] = False
            state["inference_run_id"] += 1
            state["inference_thread"].join(timeout=1.0)

        state["inference_run_id"] += 1
        run_id = state["inference_run_id"]
        state["stop_requested"] = False
        state["step_count"] = 0
        state["action_history"] = []  # 새 에피소드 시작 시 초기화
        state["last_latency"] = 0.0
        state["last_chunk"] = None
        state["current_instruction"] = instruction
        state["ui_run_status_box"] = "Running..."
        state["ui_latency_val"] = "0 ms"
        state["ui_action_val"] = "0, 0, 0"
        state["ui_chunk_val"] = "Waiting..."
        state["ui_fig"] = None
        state["ui_action_fig"] = None
        add_console_log(f"▶️ Inference Started (Prompt: '{instruction}')")

        state["is_running"] = True
        state["is_busy"] = True
        state["inference_thread"] = threading.Thread(
            target=_background_inference_loop,
            args=(backend_mode, api_url, mode, run_id),
            daemon=True
        )
        state["inference_thread"].start()
    else:
        state["stop_requested"] = True
        state["is_running"] = False
        state["inference_run_id"] += 1
        if ROS_AVAILABLE and ros_node:
            ros_node.control.robust_stop(source="inference_stop")
        state["is_busy"] = False
        state["step_count"] = 0
        state["ui_run_status_box"] = "Stopped"
        add_console_log("⏹️ Inference Stopped")
        
    return "Running..." if running else "Stopped"


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


def run_backend_inference(image: Image.Image, instruction: str, backend_mode: str, api_url: str, current_step: int = 1, run_id: int | None = None):
    backend = make_backend(backend_mode, api_url)
    result = backend.predict(image=image, instruction=instruction)
    if run_id is not None and not _is_active_run(run_id):
        raise StaleInferenceResult("stale inference result discarded")
    # action_3d includes az for ROT_L/ROT_R; fall back to 2D action if not present
    action_raw = result.get("action_3d") or result["action"]
    action = np.asarray(action_raw, dtype=np.float32).reshape(-1)
    chunk_raw = result.get("chunk") or result.get("actions") or [action.tolist()]
    chunk = np.asarray(chunk_raw, dtype=np.float32)
    if chunk.ndim == 1:
        chunk = chunk.reshape(1, -1)

    strategy = result.get("strategy", "")
    pred_label = result.get("predicted_label") or ""
    goal_near = result.get("goal_near_proxy")
    state["last_bbox"] = result.get("bbox")
    raw_action = action.copy()

    action_source = str(result.get("source") or result.get("model_name") or "").lower()
    if "monapi" in action_source:
        action = snap_monapi_action_to_label(action, pred_label)
        if chunk.size:
            chunk = chunk.copy()
            chunk[0, : min(chunk.shape[1], action.shape[0])] = action[: min(chunk.shape[1], action.shape[0])]

    if ROS_AVAILABLE and ros_node:
        lx = float(action[0])
        ly = float(action[1])
        az = float(action[2]) if action.size > 2 else 0.0
        if (run_id is not None and not _is_active_run(run_id)) or state.get("stop_requested") or not state.get("is_running"):
            state["current_log"] = "STOP requested; skipped stale inference action"
            log_msg = f"Step {current_step}: Action skipped after STOP"
            add_console_log(log_msg)
        else:
            state["current_log"] = ros_node.control.move_with_watchdog(
                lx, ly, az, source="gradio_inference", stop_after=0.55,
            )
            state["action_history"].append((lx, ly, az))
            log_msg = f"Step {current_step}: Action=[{lx:.2f}, {ly:.2f}, {az:.2f}] ({pred_label}) -> {state['current_log']}"
            add_console_log(log_msg)

    label_prefix = f"[{pred_label}] " if pred_label else ""
    act_str = f"{label_prefix}{action[0]:.4f}, {action[1]:.4f}, {action[2] if action.size > 2 else 0.0:.4f}"

    speed_scale = result.get("speed_scale")
    grounding_cached = result.get("grounding_cached")

    if strategy == "goal_nav":
        near_str = ("✅ NEAR" if goal_near else "⬜ far") if goal_near is not None else "?"
        goal = result.get("goal")
        goal_str = f"[{goal[0]:.2f},{goal[1]:.2f},{goal[2]:.2f}]" if goal else "init"
        caption = result.get("grounding_caption") or ""
        chunk_display = f"[GoalNav] goal={goal_str}  near={near_str}"
        if speed_scale is not None:
            chunk_display += f"  spd={speed_scale:.2f}"
        if grounding_cached is not None:
            chunk_display += f"  cache={'✓' if grounding_cached else '✗'}"
        if caption:
            chunk_display += f"\ngrounding: {caption}"
    else:
        chunk_display = f"Chunk (N={len(chunk)}):\n{np.array2string(chunk, precision=2, separator=', ', suppress_small=True)}"

    return {
        "log_str": f"✅ {backend.name}: {state['current_log']}",
        "lat_str": f"{float(result['latency_ms']):.1f} ms",
        "act_str": act_str,
        "chunk_display": chunk_display,
        "action": action,
        "raw_action": raw_action,
        "chunk": chunk,
        "goal_near": goal_near,
        # logger용 raw 필드
        "latency_ms": result.get("latency_ms"),
        "predicted_label": result.get("predicted_label"),
        "grounding_caption": result.get("grounding_caption"),
        "strategy": result.get("strategy"),
        "bbox": result.get("bbox"),
        "instruction_used": result.get("instruction_used"),
        "matched_path_type": result.get("matched_path_type"),
        "speed_scale": speed_scale,
        "grounding_cached": grounding_cached,
    }


def update_ui(mode, backend_mode, api_url, instr, apply_cc, _run_status):
    # 비동기 모드에서도 상태 변화를 실시간 렌더링하기 위해 is_busy 가드로 차단하지 않고 무조건 뷰 갱신
    state["auto_inference"] = mode in ("Inference (Auto)", "Inference (18-step)")

    if not ROS_AVAILABLE:
        state["camera_status"] = "ROS Not Available"
        return None, "ROS Not Available", "N/A", "N/A", "N/A", gr.update(value="Stopped"), state["camera_status"], state["model_path"], None, get_console_logs_str(), None

    if ros_node is None:
        state["camera_status"] = "ROS 재연결 중..."
        return None, "⏳ ROS 재연결 중...", "N/A", "N/A", "N/A", gr.update(), state["camera_status"], state["model_path"], None, get_console_logs_str(), None

    img = ros_node.get_inference_frame()
    if img is None:
        state["camera_status"] = "Waiting for get_image_service"
        return state["last_img"], "⚠️ Camera Service Waiting...", state["ui_latency_val"], state["ui_action_val"], state["ui_chunk_val"], gr.update(value=state["ui_run_status_box"]), state["camera_status"], state["model_path"], state["ui_fig"], get_console_logs_str(), state["ui_action_fig"]

    if apply_cc:
        img = correct_image(img)

    state["camera_status"] = "OK"
    state["last_img"] = img  # raw image for logging

    display_img = annotate_image(img, bbox=state.get("last_bbox"))

    # API 서버 상태 조회를 50ms마다 동기로 던지는 것은 큰 병목이 되므로, 2초에 한 번만 백그라운드로 갱신
    now = time.time()
    if now - state.get("last_model_info_fetch", 0.0) > 2.0:
        state["last_model_info_fetch"] = now
        def _fetch_backend_info():
            try:
                info = backend_model_info(backend_mode, api_url)
                if info.get("model_loaded", False):
                    state["model_path"] = info.get("checkpoint_path", "N/A")
                    state["model_status"] = f"{backend_mode} ({info.get('precision', 'N/A')})"
            except Exception:
                pass
        threading.Thread(target=_fetch_backend_info, daemon=True).start()
        
    return display_img, f"📡 Live | {state['current_log']}", state["ui_latency_val"], state["ui_action_val"], state["ui_chunk_val"], gr.update(value=state["ui_run_status_box"]), state["camera_status"], state["model_path"], state["ui_fig"], get_console_logs_str(), state["ui_action_fig"]


def handle_control(direction):
    if not ROS_AVAILABLE or not ros_node:
        return "ROS Error"

    mapping = {
        "W": (LINEAR_SPEED_VLA, 0.0, 0.0),
        "S": (-LINEAR_SPEED_VLA, 0.0, 0.0),
        "A": (0.0, LINEAR_SPEED_VLA, 0.0),
        "D": (0.0, -LINEAR_SPEED_VLA, 0.0),
        "Q": (LINEAR_SPEED_VLA, LINEAR_SPEED_VLA, 0.0),
        "E": (LINEAR_SPEED_VLA, -LINEAR_SPEED_VLA, 0.0),
        "R": (0.0, 0.0, ANGULAR_SPEED_VLA),
        "T": (0.0, 0.0, -ANGULAR_SPEED_VLA),
        "STOP": (0.0, 0.0, 0.0),
    }
    lx, ly, az = mapping[direction]

    # 하드웨어 통신 및 ROS 퍼블리싱을 백그라운드 스레드로 격리하여 UI 스레드 블로킹 방지
    def _run_control():
        t0 = time.time()
        if direction == "STOP":
            ros_node.control.robust_stop(source="manual_stop")
            state["current_log"] = "🛑 Force STOP"
        else:
            ros_node.control.move_and_stop_timed(lx, ly, az, source=f"manual_{direction}")
            state["current_log"] = f"🕹️ Moving {direction} (Bang-Bang)"
        elapsed = (time.time() - t0) * 1000
        print(f"[handle_control] Direction={direction} executed in background (HW Time: {elapsed:.1f}ms)")
        add_console_log(state["current_log"])

    threading.Thread(target=_run_control, daemon=True).start()

    if direction == "STOP":
        return "🛑 Force STOP (Processing...)"
    else:
        return f"🕹️ Moving {direction} (Processing...)"


def return_to_start() -> str:
    """추론 중 실행된 액션을 역순/부호반전으로 재생 → 시작 위치 복귀."""
    if state["is_returning"]:
        state["is_returning"] = False
        if ROS_AVAILABLE and ros_node:
            ros_node.control.robust_stop(source="return_cancel")
        msg = "🛑 복귀 취소됨"
        add_console_log(msg)
        return msg

    history = state.get("action_history", [])
    if not history:
        msg = "⚠️ 복귀할 경로 없음 (주행 기록이 없습니다)"
        add_console_log(msg)
        return msg

    def _run():
        state["is_returning"] = True
        add_console_log(f"🔄 복귀 시작 ({len(history)}스텝 역재생)")
        try:
            rev = [(-lx, -ly, -az) for lx, ly, az in reversed(history)]
            for lx, ly, az in rev:
                if not state["is_returning"]:
                    break
                if ROS_AVAILABLE and ros_node:
                    ros_node.control.move_and_stop_ramped(lx, ly, az, source="return")
                    # move_and_stop_ramped는 비동기로 동작하므로, 스텝 주행 완료(0.4초) 및 마진(0.05초) 동안 대기합니다.
                    time.sleep(0.45)
            if ROS_AVAILABLE and ros_node:
                ros_node.control.robust_stop(source="return_done")
            add_console_log("🔄 복귀 완료")
        finally:
            state["is_returning"] = False

    import threading
    threading.Thread(target=_run, daemon=True).start()
    return f"🔄 복귀 중... ({len(history)}스텝 역재생)"


def reset_model_wrapper(backend_mode: str, api_url: str, instruction: str):
    try:
        res = make_backend(backend_mode, api_url).reset(instruction)
        add_console_log(f"🔄 Model Reset: {res}")
        return res
    except Exception as e:
        msg = f"❌ Reset failed: {e}"
        add_console_log(msg)
        return msg


with gr.Blocks(title="VLA PRO Dashboard") as demo:
    gr.Markdown("# 🚀 Mobile VLA Real-time Dashboard & Teleop")
    gr.Markdown(
        """
        <div style="background-color: #1e293b; border-left: 4px solid #3b82f6; padding: 12px; border-radius: 4px; margin-bottom: 15px; color: #e2e8f0;">
            <h4 style="margin: 0 0 6px 0; color: #60a5fa; font-size: 1.05rem;">📊 조이스틱 자유 주행 데이터 수집 현황 (5/15 지시사항)</h4>
            <ul style="margin: 0; padding-left: 20px; font-size: 0.92rem; line-height: 1.5;">
                <li><strong>최종 수집 목표:</strong> 조이스틱 기반 자유 주행 에피소드 <strong>총 30개</strong></li>
                <li><strong>현재 수집 완료:</strong> <strong>21개</strong> (Free Center 7, Free Left 7, Free Right 7 완료)</li>
                <li><strong style="color: #fbbf24;">남은 수집 필요량:</strong> <strong>9개 추가 수집 필요</strong> (각 방향별 균등 보완 권장)</li>
                <li><strong>평가 기록 도구:</strong> 주행 완료 시 즉시 <code>vla-trial-logger</code> (포트 7862)를 통해 기록을 저장하십시오.</li>
            </ul>
        </div>
        """
    )

    with gr.Accordion("🖥️ System Process & Resource Monitor", open=True):
        with gr.Row():
            with gr.Column(scale=3):
                resource_md = gr.HTML(get_system_resources())
            with gr.Column(scale=2):
                proc_dropdown = gr.Dropdown(
                    label="Active VLA Processes (Select to Kill)",
                    choices=[],
                    interactive=True,
                )
                with gr.Row():
                    btn_refresh_mon = gr.Button("↺ Refresh Status", variant="secondary", size="sm")
                    btn_kill_proc = gr.Button("🛑 Kill Selected Process", variant="stop", size="sm")
                    btn_start_server = gr.Button("🚀 Start VLA Server (Port 8082)", variant="primary", size="sm")
                mon_status_box = gr.Textbox(label="Last Action Status", value="Idle", interactive=False)

    _cam_st, _cam_start_btn, _cam_stop_btn = camera_control_widget()

    with gr.Row():
        with gr.Column(scale=2):
            camera_output = gr.Image(label="Live Camera (via Service)", interactive=False, format="png")
            gr.Markdown("🟢 Continuous polling via GetImage service")

            with gr.Group():
                gr.Markdown("### 🕹️ Operation Mode")
                mode_radio = gr.Radio(
                    choices=["Manual Drive", "Inference (Auto)", "Inference (18-step)"],
                    value="Manual Drive",
                    label="Controller Mode",
                )

                with gr.Row(visible=False) as inference_panel:
                    with gr.Column():
                        backend_radio = gr.Radio(
                            choices=["Local Runtime", "API Server"],
                            value=DEFAULT_BACKEND_MODE,
                            label="Inference Backend",
                            visible=False,
                        )
                        
                        ckpts, confs = scan_local_files()
                        
                        # 모델 선택 및 정밀도 설정 드롭다운을 공통 영역으로 이동
                        # API 서버 및 로컬 런타임 양쪽 모두 원하는 모델을 편리하게 로드할 수 있게 함
                        ckpt_dropdown = gr.Dropdown(
                            choices=ckpts,
                            label="🎯 Select Checkpoint (.ckpt/.pth)",
                            value=pick_default_choice(ckpts, "VLA_CHECKPOINT_PATH"),
                        )
                        conf_dropdown = gr.Dropdown(
                            choices=confs,
                            label="⚙️ Select Config (.json)",
                            value=pick_default_choice(confs, "VLA_CONFIG_PATH"),
                        )
                        quant_radio = gr.Radio(
                            choices=["INT8 (Fast)", "FP16 (Accurate)"],
                            value="FP16 (Accurate)",
                            label="Model Precision",
                        )
                        
                        with gr.Tabs(selected="api_tab" if DEFAULT_BACKEND_MODE == "API Server" else "local_tab") as backend_tabs:
                            with gr.Tab("📡 MoNa-pi API Server (Safe Port Mode)", id="api_tab") as api_tab:
                                gr.Markdown("🟢 외부 포트(8082)로 동작 중인 모델을 연동합니다. 온보드 리소스를 중복 점유하지 않아 매우 안전합니다.")
                                api_url_box = gr.Textbox(label="API URL", value=DEFAULT_API_URL)
                                
                            with gr.Tab("🖥️ Local Onboard VLA (Resource Hungry)", id="local_tab") as local_tab:
                                gr.Markdown("⚠️ **주의:** 온보드 메모리에서 모델을 직접 로드하므로 중복 로딩 시 OOM 다운 위험이 있습니다.")
                                
                        btn_load_model = gr.Button("📂 Load Selected Model", variant="primary")
                        
                        load_status = gr.Textbox(
                            label="Model Status",
                            value="API Server 연결됨" if DEFAULT_BACKEND_MODE == "API Server" else "Not Loaded",
                            interactive=False,
                        )
                        model_path = gr.Textbox(label="Active Model / Checkpoint", value="N/A", interactive=False)
                        toggle_cc = gr.Checkbox(label="🎨 RGB Red Gain Boost", value=False)
                        
                        def select_api_backend():
                            return "API Server", "API Server 연결됨"
                            
                        def select_local_backend():
                            return "Local Runtime", "Not Loaded"
                            
                        api_tab.select(
                            fn=select_api_backend,
                            inputs=[],
                            outputs=[backend_radio, load_status]
                        )
                        
                        local_tab.select(
                            fn=select_local_backend,
                            inputs=[],
                            outputs=[backend_radio, load_status]
                        )

                    with gr.Column():
                        gr.Markdown("#### 🏁 Inference Control")
                        with gr.Row():
                            btn_start_inf = gr.Button("▶️ START", variant="primary")
                            btn_stop_inf = gr.Button("⏹️ STOP", variant="stop")
                        
                        with gr.Row():
                            btn_return = gr.Button("🔄 시작 위치 복귀", variant="secondary")
                            btn_reset = gr.Button("🔄 Reset Model History", variant="secondary")
                        
                        with gr.Row():
                            run_status_box = gr.Textbox(label="Run Status", value="Stopped", interactive=False)
                            latency_val = gr.Textbox(label="Latency", value="0 ms", interactive=False)
                            action_val = gr.Textbox(label="Predicted Action [lx, ly, az]", value="0, 0, 0", interactive=False)

                        console_log_box = gr.Textbox(
                            label="실시간 추론 로그 (최근 20줄)",
                            value="Ready...",
                            lines=12,
                            max_lines=15,
                            interactive=False,
                        )
                        
                        with gr.Accordion("📊 상세 예측 정보 (청크 & 궤적 & 액션)", open=False):
                            chunk_val = gr.Textbox(label="Action Chunk Preview", value="N/A", lines=3)
                            with gr.Row():
                                traj_plot = gr.Plot(label="Predicted Trajectory (XY)")
                                action_plot = gr.Plot(label="Executed Action History (lx, ly, az)")

            def on_mode_change(selected_mode):
                state["auto_inference"] = selected_mode in ("Inference (Auto)", "Inference (18-step)")
                state["is_running"] = False
                state["step_count"] = 0
                return gr.Row.update(visible=state["auto_inference"])

            mode_radio.change(fn=on_mode_change, inputs=[mode_radio], outputs=[inference_panel])
            btn_load_model.click(
                fn=load_model_wrapper,
                inputs=[backend_radio, api_url_box, quant_radio, ckpt_dropdown, conf_dropdown],
                outputs=[load_status, model_path],
            )

            with gr.Group():
                gr.Markdown("### 🎮 Manual Controls")
                with gr.Row():
                    btn_q = gr.Button("↖️ Q", scale=1)
                    btn_w = gr.Button("⬆️ W", scale=1)
                    btn_e = gr.Button("↗️ E", scale=1)
                with gr.Row():
                    btn_a = gr.Button("⬅️ A", scale=1)
                    btn_stop = gr.Button("🛑 SPACE (STOP)", variant="danger", scale=1)
                    btn_d = gr.Button("➡️ D", scale=1)
                with gr.Row():
                    btn_r = gr.Button("🔄 CCW (R)", scale=1)
                    btn_s = gr.Button("⬇️ S", scale=1)
                    btn_t = gr.Button("🔄 CW (T)", scale=1)

        with gr.Column(scale=1):
            with gr.Group():
                exp_mode = gr.Dropdown(
                    choices=EXP_MODE_NAMES,
                    value=EXP_MODE_NAMES[0],
                    label="실험 모드",
                )
                exp_config_status = gr.Textbox(label="서버 Config 상태", value="미적용", interactive=False)
                goal_dropdown = gr.Dropdown(
                    choices=["(직접 입력)"] + GOAL_NAV_PRESETS,
                    value=GOAL_NAV_PRESETS[0],
                    label="Goal Object 선택",
                    visible=True,
                )
                path_dropdown = gr.Dropdown(
                    choices=PATH_TYPES,
                    value="right_right",
                    label="Path Type 선택",
                    visible=False,
                )
                instr_box_real = gr.Textbox(
                    label="🤖 Robot Prompt (모델에게 주는 프롬프트 — 틀린 값 테스트 가능)",
                    value=DEFAULT_INSTRUCTION,
                )
                gt_object_box = gr.Textbox(
                    label="🎯 GT Object (실제 있는 물체 — 로깅/평가용, 모델에 전달 안됨)",
                    value="gray basket",
                    placeholder="예: gray basket (wrong prompt 테스트 시 실제 물체 기록)",
                )
            camera_status = gr.Textbox(label="Camera Status", value="Unknown", interactive=False)

            with gr.Accordion("🛑 자동 정지 설정", open=True):
                gr.Markdown(
                    "실제 grounding bbox area가 threshold 이상이면 자동 STOP\n"
                    "_(fallback bbox는 area=0.06 고정 → 정지 안됨)_"
                )
                stop_area_slider = gr.Slider(
                    minimum=0.05, maximum=0.50, step=0.01, value=0.18,
                    label="정지 area threshold (0=항상정지, 0.18=약 0.5m, 0.30=약 0.3m)",
                )
                stop_cx_slider = gr.Slider(
                    minimum=0.10, maximum=0.50, step=0.05, value=0.25,
                    label="중앙 허용 편차 cx ± (0.25 = 화면 중앙 50% 이내)",
                )
                bbox_area_display = gr.Textbox(
                    label="현재 bbox area (실시간)", value="—", interactive=False
                )

                def apply_stop_config(area_thr, cx_tol, api_url):
                    try:
                        import requests as _req
                        r = _req.post(
                            f"{api_url.rstrip('/')}/config",
                            json={"stop_area_threshold": area_thr, "stop_cx_tolerance": cx_tol},
                            headers={"X-API-Key": API_KEY},
                            timeout=5,
                        )
                        return f"✅ 적용: area≥{area_thr:.2f}, cx±{cx_tol:.2f}"
                    except Exception as e:
                        return f"⚠️ 적용 실패: {e}"

                stop_apply_btn = gr.Button("적용", size="sm", variant="secondary")
                stop_config_status = gr.Textbox(label="", value="", interactive=False, lines=1)
                stop_apply_btn.click(
                    fn=apply_stop_config,
                    inputs=[stop_area_slider, stop_cx_slider, api_url_box],
                    outputs=stop_config_status,
                )

            status_log = gr.Textbox(label="Status", value="Ready", visible=False)

    def stop_running_wrapper():
        set_running(False, "", "", "", "")
        return "Stopped"

    btn_start_inf.click(
        fn=lambda mode, url, instr, gt: set_running(True, mode, url, instr, gt),
        inputs=[backend_radio, api_url_box, instr_box_real, gt_object_box],
        outputs=run_status_box,
    )
    btn_stop_inf.click(
        fn=stop_running_wrapper,
        outputs=run_status_box,
    )
    btn_return.click(
        fn=return_to_start,
        outputs=run_status_box,
    )

    directions = {
        btn_w: "W",
        btn_s: "S",
        btn_a: "A",
        btn_d: "D",
        btn_q: "Q",
        btn_e: "E",
        btn_r: "R",
        btn_t: "T",
        btn_stop: "STOP",
    }
    for button, direction in directions.items():
        button.click(fn=handle_control, inputs=[gr.State(direction)], outputs=status_log)

    def _get_bbox_area_display():
        """최근 예측의 bbox area 표시 (정지 판단 기준 시각화)."""
        bbox = state.get("last_bbox")
        if bbox:
            area = bbox.get("area", 0)
            entity = bbox.get("entity", "?")
            cx = bbox.get("cx", 0.5)
            near = "🔴 STOP 조건 충족!" if area >= 0.18 and abs(cx - 0.5) <= 0.25 and entity not in ("coarse_clf", "center_fallback", "") and not entity.startswith("caption:") else ""
            return f"area={area:.3f}  cx={cx:.2f}  [{entity[:20]}]  {near}"
        return "—"

    timer = gr.Timer(0.05, active=ROS_AVAILABLE and is_camera_running())
    timer.tick(
        fn=update_ui,
        inputs=[mode_radio, backend_radio, api_url_box, instr_box_real, toggle_cc, run_status_box],
        outputs=[camera_output, status_log, latency_val, action_val, chunk_val, run_status_box, camera_status, model_path, traj_plot, console_log_box, action_plot],
    )
    timer.tick(fn=_get_bbox_area_display, outputs=bbox_area_display)
    # 페이지 열리자마자 첫 프레임 즉시 표시
    demo.load(
        fn=update_ui,
        inputs=[mode_radio, backend_radio, api_url_box, instr_box_real, toggle_cc, run_status_box],
        outputs=[camera_output, status_log, latency_val, action_val, chunk_val, run_status_box, camera_status, model_path, traj_plot, console_log_box, action_plot],
    )
    
    def _activate_timer(_=None):
        return gr.update(active=ROS_AVAILABLE and is_camera_running())

    def _deactivate_timer(_=None):
        return gr.update(active=False)

    _cam_start_btn.click(
        fn=start_camera, outputs=_cam_st
    ).then(
        fn=update_ui,
        inputs=[mode_radio, backend_radio, api_url_box, instr_box_real, toggle_cc, run_status_box],
        outputs=[camera_output, status_log, latency_val, action_val, chunk_val, run_status_box, camera_status, model_path, traj_plot, console_log_box, action_plot],
    ).then(fn=_activate_timer, outputs=timer)

    _cam_stop_btn.click(
        fn=stop_camera, outputs=_cam_st
    ).then(fn=_deactivate_timer, outputs=timer)

    btn_reset.click(
        fn=reset_model_wrapper,
        inputs=[backend_radio, api_url_box, instr_box_real],
        outputs=status_log,
    )

    def on_exp_mode_change(mode_name, api_url, backend_mode):
        cfg = EXP_MODES.get(mode_name, EXP_MODES[EXP_MODE_NAMES[0]])
        is_goal = "GoalNav" in mode_name
        instr = cfg["instruction"]
        model_key = cfg.get("model")
        desc = cfg.get("desc", "")
 
        auto_conf = cfg.get("config")
        auto_ckpt = cfg.get("checkpoint")
        
        # 상대 경로로 선언된 설정을 로컬 절대 경로로 변환하여 드롭다운 choices에 매칭되도록 조치
        if auto_conf:
            abs_conf = str((PROJECT_ROOT / auto_conf).resolve())
            conf_update = gr.update(value=abs_conf)
        else:
            conf_update = gr.update(value=None)
            
        if auto_ckpt:
            abs_ckpt = str((PROJECT_ROOT / auto_ckpt).resolve())
            ckpt_update = gr.update(value=abs_ckpt)
        else:
            ckpt_update = gr.update(value=None)
 
        cfg_status = ""
        # 탭 상태가 API Server일 때만 외부 서버로 config push 시도
        if backend_mode == "API Server":
            if is_goal:
                try:
                    ApiInferenceBackend(api_url).set_config(
                        speed_scaling=cfg["speed_scaling"],
                        grounding_skip_n=cfg["grounding_skip_n"],
                        model=model_key,
                        smooth_enabled=cfg.get("smooth_enabled"),
                    )
                    parts = []
                    if model_key:
                        parts.append(f"model={model_key}")
                    parts.append(f"skip_n={cfg['grounding_skip_n']}")
                    if cfg["speed_scaling"]:
                        parts.append("속도비례ON")
                    if cfg.get("smooth_enabled") is False:
                        parts.append("smoothingOFF")
                    cfg_status = "✅ API 서버 적용: " + ", ".join(parts)
                except Exception as e:
                    cfg_status = f"⚠️ API 서버 연동 실패: {e}"
            else:
                cfg_status = "ℹ️ API Server 모드 (일반 추론)"
        elif backend_mode == "Local Runtime":
            # Local Runtime 모드일 때는 로컬 config 적용 및 매칭
            if auto_conf:
                cfg_status = f"📋 로컬 자동매칭 완료: {Path(auto_conf).name}"
            else:
                cfg_status = "📋 로컬 모드 적용됨"
        else:
            cfg_status = "미적용"
 
        return (
            gr.update(visible=is_goal),
            gr.update(visible=not is_goal),
            instr,
            cfg_status,
            conf_update,
            ckpt_update,
        )

    exp_mode.change(
        fn=on_exp_mode_change,
        inputs=[exp_mode, api_url_box, backend_radio],
        outputs=[goal_dropdown, path_dropdown, instr_box_real, exp_config_status, conf_dropdown, ckpt_dropdown],
    )

    def on_goal_select(choice):
        if choice == "(직접 입력)":
            return gr.update()
        return choice

    goal_dropdown.change(
        fn=on_goal_select,
        inputs=[goal_dropdown],
        outputs=[instr_box_real],
    )

    path_dropdown.change(
        fn=lambda v: v,
        inputs=[path_dropdown],
        outputs=[instr_box_real],
    )

    demo.load(
        None,
        None,
        None,
        js="""
        () => {
            document.addEventListener('keydown', (e) => {
                const key = e.key.toLowerCase();
                const mapping = {'w': 'W', 's': 'S', 'a': 'A', 'd': 'D', 'q': 'Q', 'e': 'E', 'r': 'R', 't': 'T', ' ': 'STOP'};
                if (!mapping[key]) return;
                const buttons = document.querySelectorAll('button');
                for (let b of buttons) {
                    if (b.innerText.includes(mapping[key]) || (mapping[key] === 'STOP' && b.innerText.includes('SPACE'))) {
                        if (!b.disabled) b.click();
                        break;
                    }
                }
            });
        }
        """,
    )

    btn_refresh_mon.click(
        fn=update_monitor,
        inputs=[],
        outputs=[resource_md, proc_dropdown]
    )

    btn_kill_proc.click(
        fn=handle_kill_proc,
        inputs=[proc_dropdown],
        outputs=[mon_status_box, resource_md, proc_dropdown]
    )

    btn_start_server.click(
        fn=handle_start_server,
        inputs=[],
        outputs=[mon_status_box, resource_md, proc_dropdown]
    )

    demo.load(
        fn=update_monitor,
        inputs=[],
        outputs=[resource_md, proc_dropdown]
    )
    
    demo.load(
        fn=on_exp_mode_change,
        inputs=[exp_mode, api_url_box, backend_radio],
        outputs=[goal_dropdown, path_dropdown, instr_box_real, exp_config_status, conf_dropdown, ckpt_dropdown],
    )


if __name__ == "__main__":
    import socket

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except Exception:
        local_ip = "127.0.0.1"

    requested_port = int(os.getenv("VLA_INFERENCE_PORT", os.getenv("GRADIO_SERVER_PORT", "7865")))
    server_port = pick_server_port(requested_port)
    share_enabled = os.getenv("GRADIO_SHARE", "1").lower() not in {"0", "false", "no"}

    print("=" * 60)
    print("✅ Dashboard starting...")
    print(f"🏠 Local Access: http://{local_ip}:{server_port}")
    print("=" * 60)

    import socket as _sock
    try:
        _s = _sock.socket(_sock.AF_INET, _sock.SOCK_DGRAM)
        _s.connect(("8.8.8.8", 80))
        _server_ip = _s.getsockname()[0]
        _s.close()
    except Exception:
        _server_ip = "localhost"
    # Tailscale IP 우선
    try:
        import subprocess as _sp
        _out = _sp.check_output(["ip", "addr"], text=True)
        for _line in _out.splitlines():
            if _line.strip().startswith("inet 100."):
                _server_ip = _line.strip().split()[1].split("/")[0]
                break
    except Exception:
        pass
    _root = f"http://{_server_ip}:{server_port}"

    demo.launch(
        server_name="0.0.0.0",
        server_port=server_port,
        share=share_enabled,
        theme=gr.themes.Soft(),
        ssl_verify=False,
        root_path=_root,
    )

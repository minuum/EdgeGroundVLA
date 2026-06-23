# ── 중복 인스턴스 kill (같은 스크립트 이름의 이전 PID 제거) ──────────────────
import os as _os, sys as _sys, signal as _signal
def _kill_previous_instances():
    import subprocess, os
    my_pid = os.getpid()
    script_name = os.path.basename(__file__) if '__file__' in dir() else 'gradio_inference_dashboard.py'
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", script_name], text=True
        ).strip().split()
        killed = []
        for pid_str in out:
            pid = int(pid_str)
            if pid != my_pid:
                try:
                    os.kill(pid, _signal.SIGTERM)
                    killed.append(pid)
                except ProcessLookupError:
                    pass
        if killed:
            import time; time.sleep(1)
            for pid in killed:
                try: os.kill(pid, _signal.SIGKILL)
                except ProcessLookupError: pass
            print(f"🔪 이전 대시보드 인스턴스 종료: PID {killed}")
    except Exception:
        pass
_kill_previous_instances()
# ─────────────────────────────────────────────────────────────────────────────

# ── ROS camera_interfaces LD_LIBRARY_PATH 주입 (다른 import보다 먼저) ──────────
import os, sys as _sys
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROS_WS = os.getenv("VLA_ROS_WS", os.path.join(_project_root, "ROS_action"))
if not _ROS_WS.endswith("/install"):
    _ROS_WS = os.path.join(_ROS_WS, "install")
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

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="Unable to import Axes3D")


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = PROJECT_ROOT / ".vla_env_settings"
# Exp47: path_type 키를 직접 입력하거나 자연어 instruction 사용 가능.
# path_type 키 목록: center_straight, center_left, center_right,
#   left_straight, left_left, left_right,
#   right_straight, right_right, right_left
# 미매칭 시 bbox cx 위치에서 자동 추론 (right_right / left_left / center_straight).
DEFAULT_INSTRUCTION = "the gray basket on right"
PATH_TYPES = [
    "right_right", "right_left", "right_straight",
    "center_straight", "center_left", "center_right",
    "left_straight", "left_left", "left_right",
]
GOAL_NAV_PRESETS = [
    "the gray basket on right",
    "the gray basket on left",
    "the gray basket",
    "the door",
    "the corridor on the left",
    "the corridor on the right",
]

# 실험 모드: (표시 이름, instruction, backend_instruction_mode, speed_scaling, grounding_skip_n)
EXP_MODES = {
    # ── SOTA: Stage2 v2 분해 파이프라인 ──────────────────────────────────
    "⭐ Exp66 — Stage2 v2 SOTA (base PG2, L2)": {
        "instruction": GOAL_NAV_PRESETS[0],
        "backend_mode": "GoalNav (exp66)",
        "model": "exp66",
        "speed_scaling": False,
        "grounding_skip_n": 3,
        "desc": "⭐ SOTA — val 93.5%, CL 96.6%, FPE 0.102m  |  ActionMLP w=8, base PG2 cx, L2-norm aug",
        "config": "configs/exp54_stage2_action.json",
        "checkpoint": "runs/v5_nav/mlp/exp66/action_mlp.pt",
    },
    "Exp67 — Stage2 v2 HSV cx": {
        "instruction": GOAL_NAV_PRESETS[0],
        "backend_mode": "GoalNav (exp67)",
        "model": "exp67",
        "speed_scaling": False,
        "grounding_skip_n": 3,
        "desc": "Stage2 v2 HSV cx 소스 (비교용) — val 92.6%, CL 96.6%",
        "config": "configs/exp54_stage2_action.json",
        "checkpoint": "runs/v5_nav/mlp/exp67/action_mlp.pt",
    },
    # ── Legacy (체크포인트 존재, 비교 가능) ──────────────────────────────
    "Exp49 — GoalNav [legacy]": {
        "instruction": GOAL_NAV_PRESETS[0],
        "backend_mode": "GoalNav (exp49)",
        "model": "exp49",
        "speed_scaling": False,
        "grounding_skip_n": 3,
        "desc": "[legacy] 기본 GoalNav MLP — val 96.4%  |  ckpt 존재",
        "config": None,
        "checkpoint": "runs/v5_nav/mlp/exp49/exp49_mlp.pt",
    },
}
EXP_MODE_NAMES = list(EXP_MODES.keys())
LINEAR_SPEED_VLA = 1.15
ANGULAR_SPEED_VLA = 1.15

# 현재 서버의 STOP 모드 (go.sh에서 주입, 없으면 proximity 기본값)
_SERVER_STOP_MODE = os.getenv("VLA_STOP_MODE", "proximity")

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
        fallback = Path(os.path.expanduser("~/26CS/MoNaVLA/.vla_env_settings"))
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

DEFAULT_API_URL = os.getenv("VLA_API_SERVER", "http://localhost:8001")
API_KEY = os.getenv("VLA_API_KEY", "vla_devel_key_2026")
DEFAULT_BACKEND_MODE = os.getenv(
    "VLA_DASHBOARD_BACKEND",
    "API Server" if os.getenv("VLA_SERVER_ROLE") == "jetson" else "Local Runtime",
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
from scripts.utils.camera_proc import camera_control_widget, start_camera, stop_camera


def prepend_env_path(key: str, value: str) -> None:
    current = os.environ.get(key, "")
    parts = [p for p in current.split(os.pathsep) if p]
    if value not in parts:
        os.environ[key] = value if not parts else f"{value}{os.pathsep}{current}"


def setup_ros_paths() -> None:
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
        image.save(buffered, format="PNG")  # lossless — matches H5 numpy training pipeline
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
        image.save(buffered, format="PNG")  # lossless — matches H5 numpy training pipeline
        img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return self._post(
            "/predict",
            {
                "image": img_b64,
                "instruction": instruction,
                "strategy": "receding_horizon",
            },
        )

    def set_config(self, speed_scaling: bool, grounding_skip_n: int, model: str | None = None) -> dict:
        try:
            payload: dict = {"speed_scaling": speed_scaling, "grounding_skip_n": grounding_skip_n}
            if model is not None:
                payload["model"] = model
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
        return response.json()


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
        self.lock = threading.Lock()
        self.latest_ui_frame = None  # BGR numpy array, 10Hz 백그라운드 루프가 업데이트
        threading.Thread(target=self._camera_loop, daemon=True).start()

    def _camera_loop(self):
        """10Hz 백그라운드 폴링 — latest_ui_frame을 항상 최신으로 유지."""
        _consecutive_fail = 0
        while rclpy.ok():
            if not self.get_image_client.service_is_ready():
                # 서비스 미준비 — 짧게 대기 후 재시도 (최대 1s wait)
                self.get_image_client.wait_for_service(timeout_sec=1.0)
                time.sleep(0.05)
                continue
            req = GetImage.Request()
            future = self.get_image_client.call_async(req)
            # 0.3s 대기 — Jetson CompressedImage 서비스 콜 latency 수용
            start = time.time()
            while time.time() - start < 0.30:
                if future.done():
                    break
                time.sleep(0.01)
            if future.done():
                try:
                    res = future.result()
                    if res and res.image.data:
                        cv_img = None
                        # camera_pub는 cv2_to_imgmsg(raw bgr8) 전송
                        # compressed_imgmsg_to_cv2는 예외 없이 None 반환할 수 있음 → None 체크 필수
                        try:
                            cv_img = self.cv_bridge.compressed_imgmsg_to_cv2(res.image, "bgr8")
                        except Exception:
                            pass
                        if cv_img is None:
                            cv_img = self.cv_bridge.imgmsg_to_cv2(res.image, "bgr8")
                        if cv_img is not None:
                            with self.lock:
                                self.latest_ui_frame = cv_img
                            _consecutive_fail = 0
                except Exception:
                    pass
            else:
                _consecutive_fail += 1
                if _consecutive_fail % 20 == 1:
                    print(f"[CamLoop] future 미완료 연속 {_consecutive_fail}회 (>0.3s)")
            time.sleep(0.1)  # 10 Hz

    def get_inference_frame(self):
        """캐시에서 즉시 반환 — 블로킹 없음."""
        try:
            with self.lock:
                frame = self.latest_ui_frame
            if frame is None:
                return None
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        except Exception as e:
            if "context is invalid" in str(e) or "rcl" in str(e).lower():
                print(f"[Dashboard] ROS context 무효 → 재초기화 시도")
                threading.Thread(target=_init_ros_node, daemon=True).start()
        return None

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


ros_node = None
_ros_node_lock = threading.Lock()

def _init_ros_node():
    global ros_node
    try:
        try:
            rclpy.shutdown()
        except Exception:
            pass
        rclpy.init()
        node = ROSDashboardNode()
        threading.Thread(target=lambda: rclpy.spin(node), daemon=True).start()
        ros_node = node
        print("[Dashboard] ROSDashboardNode 초기화 ✅")
        return True
    except Exception as e:
        ros_node = None
        print(f"[Dashboard] ROSDashboardNode 초기화 실패: {e}")
        return False

if ROS_AVAILABLE:
    _init_ros_node()


# ── 조이스틱 (DragonRise) ─────────────────────────────────────────────────────

class DashboardJoystickReader:
    """DragonRise 게임패드로 대시보드 로봇을 직접 제어.
    데이터 수집 그라디오(JoystickReader)와 동일한 SYNC/ASYNC 구조.

    버튼 매핑:
      A (0)     → STOP (robust_stop)
      Start (7) → SYNC ↔ ASYNC 모드 전환 (활성화 토글이 아님)

    SYNC 모드: 0.45s 간격으로 move_and_stop_timed() — V5 bang-bang 호환
    ASYNC 모드: 10Hz 연속 publish_and_move() + 300ms Jitter Hold + 중립 시 robust_stop()
    """

    DEADZONE      = 0.15
    THRESHOLD     = 0.50
    STEP_INTERVAL = 0.45   # SYNC bang-bang 간격 (s)
    ASYNC_INTERVAL = 0.10  # ASYNC 연속 발행 간격 (s) — 10Hz
    JITTER_HOLD   = 0.30   # ASYNC 중립 후 정지 유예 시간 (s)
    DEFAULT_AXES  = {"left_x": 0, "left_y": 1, "right_x": 2}
    BTN_STOP      = 0   # A
    BTN_TOGGLE    = 7   # Start → SYNC/ASYNC 모드 전환

    # 대시보드 속도 상수 재사용
    _VEL_LIN = 1.15
    _VEL_ANG = 1.15

    WASD_TO_VEL = {
        'W': ( 1.15, 0.0,  0.0),
        'S': (-1.15, 0.0,  0.0),
        'Q': ( 1.15, 1.15, 0.0),
        'E': ( 1.15,-1.15, 0.0),
        'A': ( 0.0,  1.15, 0.0),
        'D': ( 0.0, -1.15, 0.0),
        'R': ( 0.0,  0.0,  1.15),
        'T': ( 0.0,  0.0, -1.15),
    }

    def __init__(self):
        self._running  = False
        self._enabled  = True    # 시작 시 기본 활성화
        self._js_mode  = 'async'  # 'sync' | 'async' (Start 버튼으로 전환)
        self._speed    = 1.15    # 속도 슬라이더와 공유
        self._thread   = None
        self._btn_prev = {}
        self._last_step_time = 0.0
        self._prev_key = None
        self._neutral_start_time = 0.0
        self._last_non_neutral_key = None
        self._movement_timer = None
        self._axes = self._load_axes()
        self.status: dict = {
            "connected": False, "name": "—",
            "key": None, "label": "—",
            "enabled": True, "mode": "ASYNC",
        }

    def _load_axes(self):
        cfg = Path(__file__).parent / "joystick_config.json"
        if cfg.exists():
            try:
                import json as _json
                return _json.load(open(cfg)).get("axes", self.DEFAULT_AXES)
            except Exception:
                pass
        return dict(self.DEFAULT_AXES)

    def start(self):
        if not PYGAME_AVAILABLE:
            print("[JS-Dashboard] pygame 없음 — pip install pygame")
            return
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def toggle_enabled(self) -> str:
        self._enabled = not self._enabled
        self.status = {**self.status, "enabled": self._enabled}
        label = "활성화" if self._enabled else "비활성화"
        print(f"[JS-Dashboard] {label}")
        return label

    def toggle_mode(self) -> str:
        self._js_mode = 'async' if self._js_mode == 'sync' else 'sync'
        self.status = {**self.status, "mode": self._js_mode.upper()}
        print(f"[JS-Dashboard] 모드 전환 → {self._js_mode.upper()}")
        return self._js_mode.upper()

    def _axis_to_key(self, lx, ly, az):
        T = self.THRESHOLD
        fwd = lx >=  T; bwd = lx <= -T
        lft = ly >=  T; rgt = ly <= -T
        rl  = az >=  T; rr  = az <= -T
        if fwd and lft: return 'Q'
        if fwd and rgt: return 'E'
        if fwd:         return 'W'
        if bwd:         return 'S'
        if lft:         return 'A'
        if rgt:         return 'D'
        if rl:          return 'R'
        if rr:          return 'T'
        return None

    def _loop(self):
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        try:
            pygame.init()
            pygame.joystick.init()
        except Exception as e:
            print(f"[JS-Dashboard] pygame init 실패: {e}")
            return

        js = None
        LABELS = {
            'W': '▲FWD', 'S': '▼BWD', 'Q': '↖FWD+L', 'E': '↗FWD+R',
            'A': '←LEFT', 'D': '→RIGHT',
            'R': '↺ROT_L', 'T': '↻ROT_R',
        }

        while self._running:
            if js is None:
                if pygame.joystick.get_count() == 0:
                    self.status = {**self.status, "connected": False, "name": "—"}
                    pygame.joystick.quit(); pygame.joystick.init()
                    time.sleep(1.0)
                    continue
                js = pygame.joystick.Joystick(0)
                js.init()
                self.status = {**self.status, "connected": True, "name": js.get_name()}
                self._btn_prev = {i: 0 for i in range(js.get_numbuttons())}
                print(f"[JS-Dashboard] 연결됨: {js.get_name()}")

            try:
                # USB 재연결(다른 로봇에 꽂았다가 같은 포트로 복귀 등) 감지 — 핸들이 죽은 채로
                # 예외 없이 0만 반환하는 걸 막기 위해 핫플러그 이벤트로 강제 재초기화한다.
                # quit()/init()은 호출하지 않는다 — 그 자체가 새 ADDED 이벤트를 만들어
                # 무한 재연결 루프에 빠진다. js=None만 하고 다음 루프의 기존 재탐지 로직에 맡긴다.
                hotplugged = False
                for ev in pygame.event.get():
                    if ev.type in (pygame.JOYDEVICEREMOVED, pygame.JOYDEVICEADDED):
                        hotplugged = True
                if hotplugged:
                    print("[JS-Dashboard] 핫플러그 이벤트 — 재초기화")
                    js = None
                    time.sleep(0.3)
                    continue

                def rd(idx):
                    v = js.get_axis(idx)
                    return v if abs(v) > self.DEADZONE else 0.0

                lx = -rd(self._axes["left_y"])
                ly = -rd(self._axes["left_x"])
                az = -rd(self._axes["right_x"])
                raw_key = self._axis_to_key(lx, ly, az)
                key = raw_key

                # L+R 합성 (gradio_data_collector.py와 동일) — L스틱으로 이동 중일 때
                # R스틱 az를 곡선 회전으로 블렌드. 순수 회전(R/T)에는 적용하지 않음.
                l_moving = abs(lx) > self.DEADZONE or abs(ly) > self.DEADZONE
                az_blend = az if (l_moving and key not in ('R', 'T')) else 0.0

                # ASYNC 모드: 300ms Jitter Hold — 순간 중립 튐 방지
                if self._js_mode == 'async':
                    now_j = time.time()
                    if raw_key is not None:
                        self._last_non_neutral_key = raw_key
                        self._neutral_start_time = 0.0
                    else:
                        if self._neutral_start_time == 0.0:
                            self._neutral_start_time = now_j
                        if now_j - self._neutral_start_time < self.JITTER_HOLD:
                            key = self._last_non_neutral_key
                        else:
                            key = None

                now = time.time()
                if self._enabled and ros_node is not None:
                    ctrl = ros_node.control
                    if key:
                        base = self.WASD_TO_VEL.get(key)
                        if base:
                            # 속도 슬라이더 반영: 방향(부호)은 유지, 크기만 스케일
                            spd = self._speed / 1.15  # 1.15 기준 정규화
                            if az_blend != 0.0:
                                base = (base[0], base[1], az_blend * 0.15)
                            vel = tuple(v * spd for v in base)
                            if self._js_mode == 'sync':
                                if (now - self._last_step_time) >= self.STEP_INTERVAL:
                                    if self._movement_timer:
                                        self._movement_timer.cancel()
                                        self._movement_timer = None
                                    ctrl.move_and_stop_timed(*vel)
                                    self._last_step_time = now
                            else:  # async
                                if (now - self._last_step_time) >= self.ASYNC_INTERVAL:
                                    ctrl.publish_and_move(*vel, source="joystick")
                                    self._last_step_time = now
                    elif self._prev_key:
                        if self._js_mode == 'async':
                            ctrl.publish_and_move(0.0, 0.0, 0.0, source="joystick_neutral")

                self._prev_key = key
                self.status = {
                    "connected": True, "name": js.get_name(),
                    "enabled": self._enabled,
                    "mode": self._js_mode.upper(),
                    "key": key, "label": LABELS.get(key, "●") if key else "○",
                }

                for i in range(js.get_numbuttons()):
                    cur = js.get_button(i)
                    if cur and not self._btn_prev.get(i, 0):
                        if i == self.BTN_STOP:
                            if ros_node is not None:
                                ros_node.control.robust_stop(source="joystick_A")
                        elif i == self.BTN_TOGGLE:
                            self.toggle_mode()
                    self._btn_prev[i] = cur

            except Exception as e:
                print(f"[JS-Dashboard] 루프 오류: {e}")
                js = None
                self.status = {**self.status, "connected": False}

            time.sleep(0.04)  # 25 Hz


_joystick = DashboardJoystickReader()
_joystick.start()


def _gnd_area_html(area: float, has: bool) -> str:
    pct   = min(int(area * 100 / 0.5 * 100), 100)  # 0.5 → 100%
    color = "#22c55e" if has else "#6b7280"
    dist  = "근접" if area >= 0.4 else ("중간" if area >= 0.15 else ("멀리" if area >= 0.05 else "없음"))
    return (
        f'<div style="margin:4px 0">'
        f'<div style="font-size:12px;color:#9ca3af;margin-bottom:3px">'
        f'bbox area — {area:.3f} ({dist})'
        f'<span style="float:right;font-size:11px;color:#6b7280">0=없음 · 0.1=1m · 0.4+=근접</span></div>'
        f'<div style="background:#374151;border-radius:4px;height:16px;width:100%">'
        f'<div style="background:{color};width:{pct}%;height:100%;border-radius:4px;'
        f'transition:width 0.3s"></div></div></div>'
    )


def _gnd_cx_html(cx: float, has: bool) -> str:
    pct   = min(max(int(cx * 100), 0), 100)
    color = "#3b82f6" if has else "#6b7280"
    side  = "왼쪽" if cx < 0.4 else ("오른쪽" if cx > 0.6 else "중앙")
    return (
        f'<div style="margin:4px 0">'
        f'<div style="font-size:12px;color:#9ca3af;margin-bottom:3px">'
        f'cx — {cx:.2f} ({side})'
        f'<span style="float:right;font-size:11px;color:#6b7280">0=왼쪽 · 0.5=중앙 · 1=오른쪽</span></div>'
        f'<div style="background:#374151;border-radius:4px;height:16px;width:100%;position:relative">'
        f'<div style="position:absolute;left:50%;top:0;width:1px;height:100%;background:#4b5563"></div>'
        f'<div style="background:{color};width:8px;height:100%;border-radius:4px;'
        f'margin-left:calc({pct}% - 4px);transition:margin-left 0.3s"></div></div></div>'
    )


def annotate_image(img: Image.Image, bbox: dict | None = None, draw_grid: bool = True) -> Image.Image:
    """카메라 이미지에 3x3 격자 + bbox 오버레이를 그려 반환."""
    arr = np.array(img)
    h, w = arr.shape[:2]

    if draw_grid:
        color = (100, 255, 100)
        cv2.line(arr, (w // 3, 0), (w // 3, h), color, 1)
        cv2.line(arr, (2 * w // 3, 0), (2 * w // 3, h), color, 1)
        cv2.line(arr, (0, h // 3), (w, h // 3), color, 1)
        cv2.line(arr, (0, 2 * h // 3), (w, 2 * h // 3), color, 1)

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
            # cx/cy만 있으면 십자선
            r = 10
            cv2.line(arr, (cx_px - r, cy_px), (cx_px + r, cy_px), (255, 80, 80), 2)
            cv2.line(arr, (cx_px, cy_px - r), (cx_px, cy_px + r), (255, 80, 80), 2)

        cv2.circle(arr, (cx_px, cy_px), 4, (255, 80, 80), -1)
        cv2.putText(arr, label[:20], (max(cx_px - 40, 0), max(cy_px - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 80, 80), 1, cv2.LINE_AA)

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
    # 이동 완료 후 로봇 정지 상태에서 캡처한 프레임 (학습 데이터 분포 일치)
    # None이면 get_inference_frame() 폴백
    "stable_frame": None,
    "stable_frame_cc": False,  # color correction 적용 여부 기록
    # 추론 이동 모드
    # SYNC : 이동 완료 후 150ms settle → stable_frame 캡처 → 다음 스텝 추론 (현재 기본)
    # PRE  : 이동 직전 live frame → 추론 → 이동 (수집 PRE_CACHE와 동일 분포)
    # ASYNC: inference_thread(3Hz) + execution_thread(10Hz) 완전 분리
    "infer_move_mode": "SYNC",
    # ASYNC 모드 공유 상태
    "_async_result": None,   # 최신 추론 결과 (UI 표시용)
    "_async_step": 0,
}


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
        return result["message"], info["checkpoint_path"]
    except Exception as e:
        state["model_status"] = "Load Failed"
        return f"❌ Load Failed: {e}", state["model_path"]


def _flush_session(status: str = "manual_stop"):
    """진행 중인 세션을 저장하고 경로를 반환한다."""
    if logger_instance and logger_instance.data and logger_instance.data.get("history"):
        path = logger_instance.end_session(status)
        if path:
            print(f"💾 세션 저장: {path} ({status})")
        return path
    return None


# ── ASYNC 추론 인프라 ──────────────────────────────────────────────────────
import threading as _threading
from collections import deque as _deque

_async_stop_evt: _threading.Event = _threading.Event()
_async_q: _deque = _deque(maxlen=2)  # 최신 action 2개만 보존


def _async_inference_worker(backend_mode: str, api_url: str, instr: str, apply_cc: bool):
    """3Hz 추론 루프 — /predict latency(~350ms)가 자연스러운 throttle."""
    step = 0
    while not _async_stop_evt.is_set() and state["is_running"]:
        if not (ROS_AVAILABLE and ros_node):
            _threading.Event().wait(0.1)
            continue
        img = ros_node.get_inference_frame()
        if img is None:
            _threading.Event().wait(0.05)
            continue
        if apply_cc:
            img = correct_image(img)
        try:
            result = run_backend_inference(img, instr, backend_mode, api_url, execute_move=False)
        except Exception as e:
            print(f"[ASYNC infer] error: {e}")
            continue
        step += 1
        result["_async_step"] = step
        _async_q.append(result)
        state["_async_result"] = result
        state["_async_step"] = step
        # goal 도달 확인
        if result.get("goal_near"):
            state["is_running"] = False
            if ROS_AVAILABLE and ros_node:
                ros_node.control.robust_stop(source="async_goal_reached")
            _flush_session("goal_reached")
            break


def _async_execution_worker():
    """10Hz 실행 루프 — queue에서 action 꺼내 cmd_vel 연속 발행."""
    lx, ly, az = 0.0, 0.0, 0.0
    last_update = time.time()
    COAST_TIMEOUT = 1.2  # 1.2s 이상 새 action 없으면 정지

    while not _async_stop_evt.is_set() and state["is_running"]:
        if _async_q:
            result = _async_q.popleft()
            action = np.asarray(result.get("action_3d") or result["action"], dtype=np.float32).reshape(-1)
            lx = float(action[0])
            ly = float(action[1])
            az = float(action[2]) if action.size > 2 else 0.0
            last_update = time.time()
            state["action_history"].append((lx, ly, az))

        if time.time() - last_update > COAST_TIMEOUT:
            lx, ly, az = 0.0, 0.0, 0.0

        if ROS_AVAILABLE and ros_node:
            state["current_log"] = ros_node.control.publish_and_move(
                lx, ly, az, source="async_exec",
            )
        time.sleep(0.1)  # 10Hz

    # 루프 종료 시 stop
    if ROS_AVAILABLE and ros_node:
        ros_node.control.robust_stop(source="async_exec_end")


def _start_async_workers(backend_mode: str, api_url: str, instr: str, apply_cc: bool):
    _async_stop_evt.clear()
    _async_q.clear()
    t_infer = _threading.Thread(
        target=_async_inference_worker,
        args=(backend_mode, api_url, instr, apply_cc),
        daemon=True, name="async-infer",
    )
    t_exec = _threading.Thread(
        target=_async_execution_worker,
        daemon=True, name="async-exec",
    )
    t_infer.start()
    t_exec.start()


def _stop_async_workers():
    _async_stop_evt.set()
    _async_q.clear()


# ─────────────────────────────────────────────────────────────────────────────
def set_running(running: bool, backend_mode: str, api_url: str, instruction: str, gt_object: str = "",
                apply_cc: bool = False):
    state["is_running"] = running
    state["gt_object"] = gt_object
    if running:
        state["step_count"] = 0
        state["_async_step"] = 0
        state["_async_result"] = None
        state["action_history"] = []
        state["stable_frame"] = None
        state["stable_frame_cc"] = False
        try:
            make_backend(backend_mode, api_url).reset(instruction)
        except Exception:
            pass
        if state.get("infer_move_mode") == "ASYNC":
            if logger_instance:
                logger_instance.start_session("async", instruction, instruction_mode=backend_mode)
                if gt_object:
                    logger_instance.data["gt_object"] = gt_object
            _start_async_workers(backend_mode, api_url, instruction, apply_cc)
    else:
        if state.get("infer_move_mode") == "ASYNC":
            _stop_async_workers()
        _flush_session("manual_stop")
        state["step_count"] = 0
    return "Running..." if running else "Stopped"


def run_backend_inference(image: Image.Image, instruction: str, backend_mode: str, api_url: str,
                          execute_move: bool = True):
    backend = make_backend(backend_mode, api_url)
    result = backend.predict(image=image, instruction=instruction)
    # action_3d includes az for ROT_L/ROT_R; fall back to 2D action if not present
    action_raw = result.get("action_3d") or result["action"]
    action = np.asarray(action_raw, dtype=np.float32).reshape(-1)
    chunk = np.asarray(result.get("chunk", [action.tolist()]), dtype=np.float32)
    if chunk.ndim == 1:
        chunk = chunk.reshape(1, -1)

    if execute_move and ROS_AVAILABLE and ros_node:
        lx = float(action[0])
        ly = float(action[1])
        az = float(action[2]) if action.size > 2 else 0.0
        state["current_log"] = ros_node.control.move_and_stop_ramped(
            lx, ly, az, source="gradio_inference",
        )
        state["action_history"].append((lx, ly, az))

    strategy = result.get("strategy", "")
    pred_label = result.get("predicted_label") or ""
    goal_near = result.get("goal_near_proxy")

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
        "chunk": chunk,
        "goal_near": goal_near,
        # logger용 raw 필드
        "latency_ms": result.get("latency_ms"),
        "predicted_label": result.get("predicted_label"),
        "grounding_caption": result.get("grounding_caption"),
        "strategy": result.get("strategy"),
        "bbox": result.get("bbox"),
        "grounding_latency_ms": result.get("grounding_latency_ms"),
        "instruction_used": result.get("instruction_used"),
        "matched_path_type": result.get("matched_path_type"),
        "speed_scale": speed_scale,
        "grounding_cached": grounding_cached,
    }


def _append_run_history(step: int, result: dict) -> None:
    """SYNC/PRE 모드 추론 step마다 호출 — run history 표(그라운딩/MLP latency 분리)에 누적."""
    total_ms = result.get("latency_ms") or 0.0
    grounding_ms = result.get("grounding_latency_ms")
    mlp_ms = (total_ms - grounding_ms) if grounding_ms is not None else None
    bbox = result.get("bbox") or {}
    state.setdefault("run_history", []).append([
        step,
        result.get("predicted_label") or result.get("log_str", ""),
        round(total_ms),
        round(grounding_ms) if grounding_ms is not None else "—",
        round(mlp_ms) if mlp_ms is not None else "—",
        round(bbox.get("area", 0.0), 3) if bbox else "—",
    ])
    state["run_history"] = state["run_history"][-30:]


def _run_history_rows():
    rows = state.get("run_history", [])
    return rows[-10:][::-1]


def update_ui(mode=None, backend_mode=None, api_url=None, instr=None, apply_cc=False,
              _run_status=None, infer_move_mode=None):
    # 로드 타이밍에 None 입력이 올 수 있음 — 기본값으로 안전 처리
    mode         = mode         or "Manual Drive"
    backend_mode = backend_mode or "API Server"
    api_url      = api_url      or ""
    instr        = instr        or ""

    if state["is_busy"]:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    state["auto_inference"] = mode == "Inference (Auto)"
    state["infer_move_mode"] = infer_move_mode or state.get("infer_move_mode") or "SYNC"

    if not ROS_AVAILABLE:
        state["camera_status"] = "ROS Not Available"
        return None, "ROS Not Available", "N/A", "N/A", "N/A", gr.update(value="Stopped"), state["camera_status"], state["model_path"], None

    if ros_node is None:
        # 재초기화 시도 중
        state["camera_status"] = "ROS 재연결 중..."
        return None, "⏳ ROS 재연결 중...", "N/A", "N/A", "N/A", gr.update(), state["camera_status"], state["model_path"], None

    img = ros_node.get_inference_frame()
    if img is None:
        state["camera_status"] = "Waiting for get_image_service"
        return state["last_img"], "⚠️ Camera Service Waiting...", "N/A", "N/A", "N/A", gr.update(), state["camera_status"], state["model_path"], None

    if apply_cc:
        img = correct_image(img)

    state["camera_status"] = "OK"
    state["last_img"] = img  # raw image for logging

    # ── ASYNC 모드: background 스레드가 추론+실행 담당, UI는 최신 결과만 표시 ──
    if state["auto_inference"] and state["is_running"] and state.get("infer_move_mode") == "ASYNC":
        result = state.get("_async_result")
        step = state.get("_async_step", 0)
        if result:
            display_img = annotate_image(img, bbox=result.get("bbox"))
            fig = ros_node.generate_trajectory_plot(result["chunk"])
            log = f"[ASYNC] Step {step} | {result.get('log_str', state['current_log'])}"
            if not state["is_running"]:  # goal reached by worker
                log = f"🎯 Goal Reached! (step {step})"
                return display_img, log, result["lat_str"], result["act_str"], result["chunk_display"], gr.update(value="Stopped (Goal Reached)"), state["camera_status"], state["model_path"], fig
            return display_img, log, result["lat_str"], result["act_str"], result["chunk_display"], gr.update(value=f"ASYNC Running (step {step})"), state["camera_status"], state["model_path"], fig
        else:
            return annotate_image(img), f"[ASYNC] 추론 대기 중...", "—", "—", "—", gr.update(value="ASYNC Running (init)"), state["camera_status"], state["model_path"], None

    if state["auto_inference"] and state["is_running"]:
        state["is_busy"] = True
        try:
            state["step_count"] += 1
            current_step = state["step_count"]

            if current_step == 1:
                state["run_history"] = []
                if logger_instance:
                    logger_instance.start_session(short_model_name(state["model_path"]), instr, instruction_mode=backend_mode)
                    if logger_instance and state.get("gt_object"):
                        logger_instance.data["gt_object"] = state["gt_object"]
                    logger_instance.log_step(current_step, [0.0, 0.0, 0.0], 0, image=img)
                ros_node.control.robust_stop(source="inference_start")
                # 로봇이 정지 완료될 때까지 대기 → step 2 추론용 stable frame 캡처
                time.sleep(0.20)
                _sf = ros_node.get_inference_frame()
                if _sf is not None:
                    state["stable_frame"] = correct_image(_sf) if apply_cc else _sf
                    state["stable_frame_cc"] = apply_cc
                try:
                    make_backend(backend_mode, api_url).reset(instr)
                except Exception as e:
                    return annotate_image(img), f"❌ Reset failed: {e}", "0 ms", "STOP", "Waiting...", gr.update(value="Stopped"), state["camera_status"], state["model_path"], None
                return annotate_image(img), "Step 1 (Start/Wait)", "0 ms", "0.0000, 0.0000, 0.0000", "Waiting...", gr.update(value="Running (step 1)..."), state["camera_status"], state["model_path"], None

            if state["infer_move_mode"] == "PRE":
                # PRE 모드: live frame → 추론 → 이동 (수집 PRE_CACHE와 동일 분포)
                infer_img = img
                state["stable_frame"] = None
                result = run_backend_inference(infer_img, instr, backend_mode, api_url)
                # settle 대기 없음 — 다음 스텝은 또 live frame 사용
            else:
                # SYNC 모드 (기본): 이전 이동 완료 후 stable_frame 우선 사용
                infer_img = state.get("stable_frame") or img
                state["stable_frame"] = None  # consume
                result = run_backend_inference(infer_img, instr, backend_mode, api_url)
                # 이동 완료 후 150ms settle → 다음 스텝용 stable_frame 미리 캡처
                time.sleep(0.15)
                _sf = ros_node.get_inference_frame()
                if _sf is not None:
                    state["stable_frame"] = correct_image(_sf) if apply_cc else _sf
                    state["stable_frame_cc"] = apply_cc
            display_img = annotate_image(img, bbox=result.get("bbox"))
            fig = ros_node.generate_trajectory_plot(result["chunk"])
            _append_run_history(current_step, result)
            if logger_instance:
                logger_instance.log_step(
                    current_step,
                    result["action"],
                    result.get("latency_ms", 0),
                    result["chunk"],
                    image=infer_img,  # 실제 추론에 사용한 프레임 로깅
                    predicted_label=result.get("predicted_label"),
                    grounding_caption=result.get("grounding_caption"),
                    goal_near=result.get("goal_near"),
                    strategy=result.get("strategy"),
                    bbox=result.get("bbox"),
                    grounding_latency_ms=result.get("grounding_latency_ms"),
                    instruction_used=result.get("instruction_used"),
                    matched_path_type=result.get("matched_path_type"),
                    speed_scale=result.get("speed_scale"),
                    grounding_cached=result.get("grounding_cached"),
                )
            log = f"Step {current_step} | {result['log_str']}"
            if result.get("goal_near"):
                state["is_running"] = False
                state["step_count"] = 0
                ros_node.control.robust_stop(source="goal_reached")
                if logger_instance:
                    report_path = logger_instance.end_session()
                    log = f"🎯 Goal Reached! (step {current_step}) | Log: {Path(report_path).name}"
                else:
                    log = f"🎯 Goal Reached! (step {current_step})"
                return display_img, log, result["lat_str"], result["act_str"], result["chunk_display"], gr.update(value="Stopped (Goal Reached)"), state["camera_status"], state["model_path"], fig
            return display_img, log, result["lat_str"], result["act_str"], result["chunk_display"], gr.update(value=f"Running (step {current_step})"), state["camera_status"], state["model_path"], fig
        finally:
            state["is_busy"] = False

    info = backend_model_info(backend_mode, api_url)
    if info.get("model_loaded"):
        state["model_path"] = info.get("checkpoint_path", state["model_path"])
        state["model_status"] = f"{backend_mode} ({info.get('precision', 'N/A')})"
    return annotate_image(img), f"📡 Live | {state['current_log']}", "N/A", "N/A", "N/A", gr.update(), state["camera_status"], state["model_path"], None


def handle_control(direction, speed=1.15):
    if not ROS_AVAILABLE or not ros_node:
        return "ROS Error"

    s = float(speed)
    mapping = {
        "W": (s, 0.0, 0.0),
        "S": (-s, 0.0, 0.0),
        "A": (0.0, s, 0.0),
        "D": (0.0, -s, 0.0),
        "Q": (s, s, 0.0),
        "E": (s, -s, 0.0),
        "R": (0.0, 0.0, s),
        "T": (0.0, 0.0, -s),
        "STOP": (0.0, 0.0, 0.0),
    }
    lx, ly, az = mapping[direction]
    if direction == "STOP":
        ros_node.control.robust_stop(source="manual_stop")
        state["current_log"] = "🛑 Force STOP"
    else:
        ros_node.control.move_and_stop_timed(lx, ly, az, source=f"manual_{direction}")
        state["current_log"] = f"🕹️ {direction}  spd={s:.2f}"
    return state["current_log"]


def return_to_start() -> str:
    """추론 중 실행된 액션을 역순/부호반전으로 재생 → 시작 위치 복귀."""
    if state["is_returning"]:
        state["is_returning"] = False
        if ROS_AVAILABLE and ros_node:
            ros_node.control.robust_stop(source="return_cancel")
        return "🛑 복귀 취소됨"

    history = state.get("action_history", [])
    if not history:
        return "⚠️ 복귀할 경로 없음 (주행 기록이 없습니다)"

    def _run():
        state["is_returning"] = True
        try:
            rev = [(-lx, -ly, -az) for lx, ly, az in reversed(history)]
            for lx, ly, az in rev:
                if not state["is_returning"]:
                    break
                if ROS_AVAILABLE and ros_node:
                    ros_node.control.move_and_stop_ramped(lx, ly, az, source="return")
            if ROS_AVAILABLE and ros_node:
                ros_node.control.robust_stop(source="return_done")
        finally:
            state["is_returning"] = False

    import threading
    threading.Thread(target=_run, daemon=True).start()
    return f"🔄 복귀 중... ({len(history)}스텝 역재생)"


def reset_model_wrapper(backend_mode: str, api_url: str, instruction: str):
    try:
        return make_backend(backend_mode, api_url).reset(instruction)
    except Exception as e:
        return f"❌ Reset failed: {e}"


def _make_env_banner() -> str:
    import socket as _sock
    hostname = _sock.gethostname()
    role = os.getenv("VLA_SERVER_ROLE", "unknown")
    model = os.getenv("VLA_MODEL", "exp66")
    api = os.getenv("VLA_API_SERVER", "http://localhost:8001")
    exp_name = EXP_MODE_NAMES[0] if EXP_MODE_NAMES else "—"
    return (
        f'<div style="background:#0f172a;border-left:4px solid #22c55e;padding:10px 14px;'
        f'border-radius:4px;margin-bottom:12px;color:#e2e8f0;font-size:0.88rem;line-height:1.6;">'
        f'<strong style="color:#4ade80;font-size:0.95rem;">MoNaVLA 환경</strong>'
        f'&nbsp;&nbsp;|&nbsp;&nbsp;'
        f'<code style="color:#86efac">{hostname}</code>'
        f'&nbsp;(<span style="color:#fbbf24">{role}</span>)'
        f'&nbsp;&nbsp;|&nbsp;&nbsp;'
        f'모델&nbsp;<code style="color:#67e8f9">{model}</code>'
        f'&nbsp;&nbsp;|&nbsp;&nbsp;'
        f'API&nbsp;<code style="color:#a5b4fc">{api}</code>'
        f'&nbsp;&nbsp;|&nbsp;&nbsp;'
        f'실험&nbsp;<span style="color:#f9a8d4">{exp_name}</span>'
        f'</div>'
    )


with gr.Blocks(title="MoNaVLA Dashboard") as demo:
    gr.Markdown("# MoNaVLA Real-time Dashboard")
    gr.HTML(_make_env_banner())

    _cam_st, _cam_start_btn, _cam_stop_btn = camera_control_widget()
    _cam_start_btn.click(fn=start_camera, outputs=_cam_st)
    _cam_stop_btn.click(fn=stop_camera,   outputs=_cam_st)

    with gr.Tabs():
      with gr.Tab("🤖 Drive / Inference"):
        # ── 기존 메인 탭 내용 시작 ──
        with gr.Row(equal_height=False):
          with gr.Column(scale=1):
            gr.Markdown("## 🎮 제어 (카메라/모델/조작)")
            camera_output = gr.Image(label="Live Camera (via Service)", interactive=False)
            gr.Markdown("🟢 Continuous polling via GetImage service")

            with gr.Group():
                gr.Markdown("### 🕹️ Operation Mode")
                mode_radio = gr.Radio(
                    choices=["Manual Drive", "Inference (Auto)"],
                    value="Manual Drive",
                    label="Controller Mode",
                )

                # Inference Backend — 항상 표시 (Manual Drive에서도 exp_mode config push에 사용)
                with gr.Row():
                    backend_radio = gr.Radio(
                        choices=["Local Runtime", "API Server"],
                        value=DEFAULT_BACKEND_MODE,
                        label="Inference Backend",
                        scale=1,
                    )
                    api_url_box = gr.Textbox(
                        label="API URL",
                        value=DEFAULT_API_URL,
                        scale=2,
                        info="포트 8001 = soda 추론 서버 (proxy_inference_server)",
                    )

                ckpts, confs = scan_local_files()
                _is_api = DEFAULT_BACKEND_MODE == "API Server"

                def _default_from_exp(key: str, choices):
                    """기본 EXP_MODE(Exp66)의 checkpoint/config 절대경로를 초기값으로."""
                    default_cfg = EXP_MODES[EXP_MODE_NAMES[0]]
                    rel = default_cfg.get(key)
                    if rel:
                        abs_path = str(PROJECT_ROOT / rel)
                        for _label, val in choices:
                            if val == abs_path:
                                return abs_path
                    return pick_default_choice(choices, "VLA_CHECKPOINT_PATH" if key == "checkpoint" else "VLA_CONFIG_PATH")

                # Local Runtime 전용 — API Server 선택 시 숨김
                with gr.Column(visible=not _is_api) as local_panel:
                    with gr.Row():
                        ckpt_dropdown = gr.Dropdown(
                            choices=ckpts,
                            label="🎯 Checkpoint (.ckpt/.pth)",
                            value=_default_from_exp("checkpoint", ckpts),
                            scale=2,
                        )
                        conf_dropdown = gr.Dropdown(
                            choices=confs,
                            label="⚙️ Config (.json)",
                            value=_default_from_exp("config", confs),
                            scale=2,
                        )
                        quant_radio = gr.Radio(
                            choices=["INT8 (Fast)", "FP16 (Accurate)"],
                            value="FP16 (Accurate)",
                            label="Precision",
                            scale=1,
                        )
                    btn_load_model = gr.Button("📂 Load Selected Model", variant="primary")

                with gr.Row():
                    load_status = gr.Textbox(
                        label="Model Status",
                        value="API Server 연결됨" if _is_api else "Not Loaded",
                        interactive=False,
                        scale=3,
                    )
                    toggle_cc = gr.Checkbox(label="🎨 Red Gain Boost", value=False, scale=1)
                model_path = gr.Textbox(label="Active Model / Checkpoint", value="N/A", interactive=False)

                # 추론 제어 — Inference (Auto) 선택 시만 표시
                with gr.Column(visible=False) as inference_panel:
                    gr.Markdown("#### 🏁 Inference Control")
                    infer_move_radio = gr.Radio(
                        choices=["SYNC", "PRE", "ASYNC"],
                        value="SYNC",
                        label="이동 모드",
                        info="SYNC: 이동→정착→캡처 (기본) | PRE: 캡처→추론→이동 | ASYNC: 추론(3Hz)+실행(10Hz) 분리스레드",
                    )
                    with gr.Row():
                        btn_start_inf = gr.Button("▶️ START", variant="primary", scale=1)
                        btn_stop_inf = gr.Button("⏹️ STOP", variant="stop", scale=1)
                        btn_return = gr.Button("🔄 복귀", variant="secondary", scale=1)
                    run_status_box = gr.Textbox(label="Run Status", value="Stopped", interactive=False)

                def on_backend_change(backend):
                    is_api = backend == "API Server"
                    status = "API Server 연결됨" if is_api else "Not Loaded"
                    return gr.update(visible=not is_api), gr.update(value=status)

                backend_radio.change(
                    fn=on_backend_change,
                    inputs=[backend_radio],
                    outputs=[local_panel, load_status],
                )

            def on_mode_change(selected_mode):
                state["auto_inference"] = selected_mode == "Inference (Auto)"
                state["is_running"] = False
                state["step_count"] = 0
                return gr.update(visible=state["auto_inference"])

            mode_radio.change(fn=on_mode_change, inputs=[mode_radio], outputs=[inference_panel])
            btn_load_model.click(
                fn=load_model_wrapper,
                inputs=[backend_radio, api_url_box, quant_radio, ckpt_dropdown, conf_dropdown],
                outputs=[load_status, model_path],
            )

            with gr.Group():
                gr.Markdown("### 🎮 Manual Controls")
                manual_speed_slider = gr.Slider(
                    minimum=0.3, maximum=2.0, step=0.05, value=1.15,
                    label="속도 (lx/ly/az 크기)",
                )
                with gr.Row():
                    btn_q = gr.Button("↖ Q", scale=1, size="sm")
                    btn_w = gr.Button("▲ W", scale=1, size="sm")
                    btn_e = gr.Button("↗ E", scale=1, size="sm")
                with gr.Row():
                    btn_a = gr.Button("◀ A", scale=1, size="sm")
                    btn_stop = gr.Button("⏹ STOP", variant="stop", scale=1, size="sm")
                    btn_d = gr.Button("▶ D", scale=1, size="sm")
                with gr.Row():
                    btn_r = gr.Button("↺ R (CCW)", scale=1, size="sm")
                    btn_s = gr.Button("▼ S", scale=1, size="sm")
                    btn_t = gr.Button("↻ T (CW)", scale=1, size="sm")

            with gr.Group():
                gr.Markdown("### 🕹️ Joystick (DragonRise)")
                with gr.Row():
                    js_status = gr.Textbox(
                        label="상태",
                        value="🔌 초기화 중...",
                        interactive=False,
                        scale=4,
                    )
                    btn_js_toggle = gr.Button(
                        "비활성화",
                        variant="primary",
                        scale=1,
                    )
                gr.Markdown(
                    "<small>"
                    "Left Stick → 이동 | Right Stick X → 회전 | "
                    "A → STOP | Start → **SYNC↔ASYNC 모드 전환**<br>"
                    "📸 SYNC: 0.45s bang-bang (V5 호환) | "
                    "🌊 ASYNC: 10Hz 연속 + 300ms Jitter Hold"
                    "</small>",
                )

                def _js_status_text() -> str:
                    s = _joystick.status
                    if not s["connected"]:
                        return "🔌 미연결 (DragonRise 꽂으면 자동 인식)"
                    en    = "🟢 ON" if s["enabled"] else "⚫ OFF"
                    mode  = s.get("mode", "SYNC")
                    badge = "📸 SYNC" if mode == "SYNC" else "🌊 ASYNC"
                    key   = s.get("label", "○")
                    name  = s.get("name", "Controller")
                    # key가 있으면 강조
                    key_str = f"[ {key} ]" if s.get("key") else "○ 중립"
                    return f"{en}  |  {badge}  |  {name}\n▶ {key_str}"

                def _js_toggle() -> tuple:
                    _joystick.toggle_enabled()
                    btn_label = "비활성화" if _joystick._enabled else "활성화"
                    return _js_status_text(), gr.update(
                        value=btn_label,
                        variant="primary" if _joystick._enabled else "secondary",
                    )

                btn_js_toggle.click(fn=_js_toggle, outputs=[js_status, btn_js_toggle])

          with gr.Column(scale=1):
            gr.Markdown("## 📡 모니터링 (실험설정/로그/그래프)")
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

            _is_learned = (_SERVER_STOP_MODE == "learned")
            _stop_acc_label = (
                "🤖 자동 정지 — Learned STOP (N1 모델)"
                if _is_learned else
                "📐 자동 정지 — Proximity Threshold"
            )
            with gr.Accordion(_stop_acc_label, open=True):
                if _is_learned:
                    gr.Markdown(
                        "**현재 서버: `VLA_STOP_MODE=learned`**\n\n"
                        "모델(stop_N1.pt)이 **class 0 (STOP)** 을 직접 예측하면 정지 + latch.\n"
                        "리셋 전까지 추론 루프가 계속 STOP을 유지함.\n\n"
                        "> threshold 슬라이더는 `proximity` 모드에서만 사용. 지금은 비활성."
                    )
                    stop_area_slider = gr.Slider(
                        minimum=0.05, maximum=0.50, step=0.01, value=0.18,
                        label="정지 area threshold (proximity 모드 전용 — 현재 비활성)",
                        interactive=False,
                    )
                    stop_cx_slider = gr.Slider(
                        minimum=0.10, maximum=0.50, step=0.05, value=0.25,
                        label="중앙 허용 편차 cx ± (proximity 모드 전용 — 현재 비활성)",
                        interactive=False,
                    )
                else:
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
                    label="현재 bbox area (실시간 모니터링)", value="—", interactive=False
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

                stop_apply_btn = gr.Button(
                    "적용" if not _is_learned else "적용 (proximity 모드에서만 유효)",
                    size="sm",
                    variant="secondary",
                    interactive=not _is_learned,
                )
                stop_config_status = gr.Textbox(label="", value="", interactive=False, lines=1)
                stop_apply_btn.click(
                    fn=apply_stop_config,
                    inputs=[stop_area_slider, stop_cx_slider, api_url_box],
                    outputs=stop_config_status,
                )

            status_log = gr.Textbox(label="Status", value="Ready")
            latency_val = gr.Textbox(label="Latency", value="0 ms")
            action_val = gr.Textbox(label="Predicted Action [lx, ly, az]", value="0, 0, 0")
            chunk_val = gr.Textbox(label="Action Chunk Preview", value="N/A", lines=3)
            traj_plot = gr.Plot(label="Predicted Trajectory (XY)")
            run_history_table = gr.Dataframe(
                headers=["step", "action", "total(ms)", "grounding(ms)", "mlp(ms)", "bbox_area"],
                datatype=["number", "str", "number", "number", "number", "number"],
                label="Run 히스토리 (최근 10 step, SYNC/PRE 모드만 — ASYNC 미지원)",
                row_count=10,
                col_count=6,
                interactive=False,
            )
            btn_reset = gr.Button("🔄 Reset Model History")
        # ── 기존 메인 탭 내용 끝 ──

      # ────────────────────────────────────────────────────────────────
      with gr.Tab("🔍 Grounding 검증"):
        gr.Markdown(
            "### PG2 실시간 Grounding 검증\n"
            "카메라 프레임을 서버로 전송해 PG2가 바구니를 검출하는지 확인. "
            "추론 시작 전 bbox area가 올라가는지 여기서 먼저 체크."
        )
        with gr.Row():
          with gr.Column(scale=3):
            gnd_image = gr.Image(
                label="Camera + BBox Overlay",
                interactive=False,
            )
            with gr.Row():
                gnd_run_btn  = gr.Button("▶ 단발 검증", variant="primary", scale=2)
                gnd_auto_btn = gr.Button("🔄 자동 (1fps)", variant="secondary", scale=2)
                gnd_stop_btn = gr.Button("⏹ 정지", variant="stop", scale=1)
                gnd_rec_btn  = gr.Button("🔴 녹화", variant="secondary", scale=1)

          with gr.Column(scale=2):
            gnd_clock    = gr.Textbox(label="현재 시각", value="—", interactive=False)
            gnd_has_bbox = gr.Textbox(label="검출 결과", value="—", interactive=False)
            gnd_area_bar = gr.HTML(value=_gnd_area_html(0.06, False), label="")
            gnd_cx_bar   = gr.HTML(value=_gnd_cx_html(0.5, False),   label="")
            gnd_latency  = gr.Textbox(label="Grounding latency", value="—", interactive=False)
            gnd_raw      = gr.Textbox(label="PG2 raw output", value="—", interactive=False, lines=2)
            gnd_server_cmp = gr.Textbox(label="서버 예측", value="—", interactive=False)
            gnd_history   = gr.Dataframe(
                headers=["#", "bbox", "area", "cx", "pred", "lat(ms)"],
                datatype=["number", "str", "number", "number", "str", "number"],
                label="최근 10회 이력",
                row_count=10,
                col_count=6,
                interactive=False,
            )
            with gr.Row():
                gnd_log_display = gr.Textbox(
                    label="JSONL 경로", value="(첫 검증 시 생성됨)",
                    interactive=False, scale=3,
                )
                gnd_new_session_btn = gr.Button("🆕 새 세션", scale=1, size="sm")
            gnd_rec_display = gr.Textbox(label="녹화 경로", value="—", interactive=False)

        # ── Grounding 탭 로직 ─────────────────────────────────────────
        _gnd_auto_state   = gr.State(False)
        _gnd_history_rows = gr.State([])
        _gnd_count        = gr.State(0)

        import datetime as _dt
        _gnd_log_dir = Path("logs/grounding_sessions")
        _gnd_log_dir.mkdir(parents=True, exist_ok=True)
        _gnd_log_file: list = [None]  # mutable ref — session 시작 시 생성
        _gnd_log_last_write: list = [None]  # 마지막 기록 시각 — idle gap 감지용
        _GND_SESSION_IDLE_GAP = _dt.timedelta(minutes=10)  # 이 시간 이상 비면 새 세션으로 간주

        def _gnd_ensure_log():
            """세션 로그 파일이 없거나, idle gap이 길었으면 새로 만들고 경로 반환.

            대시보드 프로세스가 며칠씩 무중단으로 떠 있을 수 있어 버튼 클릭에만
            의존하면 다른 날 세션이 옛 파일에 계속 append되는 문제가 있었음
            (2026-06-18 세션에 2026-06-20 세션이 합쳐진 사례).
            """
            now = _dt.datetime.now()
            last = _gnd_log_last_write[0]
            if _gnd_log_file[0] is None or (last is not None and now - last > _GND_SESSION_IDLE_GAP):
                ts = now.strftime("%Y%m%d_%H%M%S")
                _gnd_log_file[0] = _gnd_log_dir / f"gnd_{ts}.jsonl"
            _gnd_log_last_write[0] = now
            return _gnd_log_file[0]

        # 녹화 mutable refs
        _gnd_recording: list    = [False]
        _gnd_video_writer: list = [None]   # cv2.VideoWriter, 첫 프레임에서 lazy init
        _gnd_video_path: list   = [None]

        def _run_grounding(api_url, backend_mode, instr, history_rows, count):
            """카메라 프레임 → /ground (+ 병렬 /predict) → bbox 오버레이 이미지 + 스탯."""
            import requests as _req, base64, io, threading as _th, re as _re
            from PIL import ImageDraw

            frame = None
            if ROS_AVAILABLE and ros_node:
                frame = ros_node.get_inference_frame()

            log_path_str = str(_gnd_ensure_log())
            now_str = _dt.datetime.now().strftime("%H:%M:%S")
            rec_str = ("🔴 녹화 중..." if _gnd_recording[0]
                       else (str(_gnd_video_path[0]) if _gnd_video_path[0] else "—"))

            if frame is None:
                return (
                    now_str, None, "❌ 카메라 없음",
                    _gnd_area_html(0.0, False), _gnd_cx_html(0.5, False),
                    "—", "카메라 연결 필요", "—", rec_str, history_rows, count, log_path_str,
                )

            # ── /predict 병렬 실행 (execute_move=False) ───────────────────
            pred_container: list = [None]
            def _do_predict():
                try:
                    pred_container[0] = run_backend_inference(
                        frame, instr or "", backend_mode or "", api_url, execute_move=False
                    )
                except Exception:
                    pass
            t_pred = _th.Thread(target=_do_predict, daemon=True)
            t_pred.start()

            # ── /ground 호출 (메인 스레드) ─────────────────────────────────
            buf = io.BytesIO()
            frame.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            try:
                resp = _req.post(
                    f"{api_url.rstrip('/')}/ground",
                    json={"image": b64},
                    headers={"X-API-Key": API_KEY},
                    timeout=10,
                )
                d = resp.json()
            except Exception as e:
                t_pred.join(timeout=0)
                return (
                    now_str, frame, f"❌ 서버 오류: {e}",
                    _gnd_area_html(0.0, False), _gnd_cx_html(0.5, False),
                    "—", str(e), "—", rec_str, history_rows, count, log_path_str,
                )

            # /predict 결과 수집
            t_pred.join(timeout=4)
            pred_r = pred_container[0]
            pred_label = ""
            pred_lat   = 0.0
            pred_near  = None
            if pred_r:
                m = _re.match(r'\[([^\]]+)\]', pred_r.get("act_str", ""))
                pred_label = m.group(1) if m else (pred_r.get("act_str", "")[:12])
                try:
                    pred_lat = float(pred_r.get("lat_str", "0").replace(" ms", ""))
                except Exception:
                    pass
                pred_near = pred_r.get("goal_near_proxy")

            has   = d.get("has_bbox", False)
            area  = float(d.get("area", 0.06))
            cx    = float(d.get("cx", 0.5))
            cy    = float(d.get("cy", 0.6))
            lat   = float(d.get("latency_ms", 0))
            raw   = d.get("raw_output", "—")
            x1, y1, x2, y2 = d.get("x1"), d.get("y1"), d.get("x2"), d.get("y2")

            # ── bbox 오버레이 ──────────────────────────────────────────────
            img_draw = frame.copy()
            draw = ImageDraw.Draw(img_draw)
            W, H = img_draw.size
            if has and x1 is not None:
                bx1, by1 = int(x1 * W), int(y1 * H)
                bx2, by2 = int(x2 * W), int(y2 * H)
                draw.rectangle([bx1, by1, bx2, by2], outline="#00ff88", width=3)
                draw.text((bx1 + 4, by1 + 4), f"area={area:.3f}", fill="#00ff88")
            cx_px = int(cx * W)
            line_color = "#00ff88" if has else "#ef4444"
            draw.line([(cx_px, 0), (cx_px, H)], fill=line_color, width=2)
            draw.line([(W // 2, 0), (W // 2, H)], fill="#4a5568", width=1)
            # 서버 예측 자막
            if pred_label:
                near_tag = "  ✅NEAR" if pred_near else ""
                draw.text((4, 4), f"pred: {pred_label}{near_tag}", fill="#facc15")

            # ── 상태 텍스트 ────────────────────────────────────────────────
            if has:
                status = f"✅ 검출됨  area={area:.3f}  cx={cx:.2f}"
            else:
                status = f"❌ 미검출  (raw: {raw[:30]})"

            # 서버 비교 문자열
            if pred_label:
                near_str = "  ✅NEAR" if pred_near else ("  ⬜far" if pred_near is not None else "")
                server_cmp = f"↳ {pred_label}{near_str}  ({pred_lat:.0f}ms)"
            else:
                server_cmp = "— (서버 비예측)"

            # ── 녹화: 첫 프레임에서 VideoWriter lazy init ─────────────────
            if _gnd_recording[0]:
                import cv2 as _cv2, numpy as _np
                if _gnd_video_writer[0] is None:
                    ts_v = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
                    vpath = _gnd_log_dir / f"gnd_{ts_v}.mp4"
                    _gnd_video_path[0] = vpath
                    _gnd_video_writer[0] = _cv2.VideoWriter(
                        str(vpath), _cv2.VideoWriter_fourcc(*"m", "p", "4", "v"),
                        1.0, (W, H),
                    )
                if _gnd_video_writer[0].isOpened():
                    bgr = _cv2.cvtColor(_np.array(img_draw), _cv2.COLOR_RGB2BGR)
                    subtitle = f"{now_str}  cx={cx:.2f} area={area:.3f} {'BBOX' if has else 'NONE'}  pred:{pred_label}"
                    _cv2.putText(bgr, subtitle, (8, H - 10),
                                 _cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
                                 _cv2.LINE_AA)
                    _gnd_video_writer[0].write(bgr)
                rec_str = f"🔴 녹화 중... {_gnd_video_path[0].name}"

            # ── 이력 업데이트 ──────────────────────────────────────────────
            count += 1
            new_row = [count, "✅" if has else "❌", round(area, 3),
                       round(cx, 3), pred_label or "—", round(lat, 0)]
            rows = ([new_row] + list(history_rows))[:10]

            # ── JSONL 저장 ─────────────────────────────────────────────────
            import json as _json
            log_path = _gnd_ensure_log()
            record = {
                "ts": _dt.datetime.now().isoformat(timespec="milliseconds"),
                "n": count, "has_bbox": has,
                "area": round(area, 4), "cx": round(cx, 4), "cy": round(cy, 4),
                "latency_ms": round(lat, 1), "raw": raw,
                "pred_label": pred_label, "pred_lat_ms": round(pred_lat, 1) if pred_lat else None,
                "pred_goal_near": pred_near,
            }
            try:
                with open(log_path, "a") as _f:
                    _f.write(_json.dumps(record, ensure_ascii=False) + "\n")
            except Exception:
                pass

            return (
                now_str, img_draw, status,
                _gnd_area_html(area, has), _gnd_cx_html(cx, has),
                f"{lat:.0f} ms", raw, server_cmp, rec_str,
                rows, count, str(log_path),
            )

        _gnd_outputs = [gnd_clock, gnd_image, gnd_has_bbox, gnd_area_bar, gnd_cx_bar,
                        gnd_latency, gnd_raw, gnd_server_cmp, gnd_rec_display,
                        _gnd_history_rows, _gnd_count, gnd_log_display]

        # 단발 버튼
        gnd_run_btn.click(
            fn=_run_grounding,
            inputs=[api_url_box, backend_radio, instr_box_real, _gnd_history_rows, _gnd_count],
            outputs=_gnd_outputs,
        )

        # 시각 타이머 — 항상 활성 (auto 모드와 무관하게 시각 표시)
        gnd_clock_timer = gr.Timer(1.0, active=True)
        gnd_clock_timer.tick(
            fn=lambda: _dt.datetime.now().strftime("%H:%M:%S"),
            outputs=gnd_clock,
        )

        # 자동 타이머 (1fps)
        gnd_timer = gr.Timer(1.0, active=False)

        def _toggle_gnd_auto(is_active):
            new_state = not is_active
            label = "⏸ 자동 중지" if new_state else "🔄 자동 (1fps)"
            variant = "primary" if new_state else "secondary"
            return new_state, gr.update(value=label, variant=variant), gr.update(active=new_state)

        gnd_auto_btn.click(
            fn=_toggle_gnd_auto,
            inputs=[_gnd_auto_state],
            outputs=[_gnd_auto_state, gnd_auto_btn, gnd_timer],
        )
        def _gnd_stop_click():
            _gnd_stop_all()
            return False, gr.update(value="🔄 자동 (1fps)", variant="secondary"), gr.update(active=False), \
                   gr.update(value="🔴 녹화", variant="secondary"), "—"

        gnd_stop_btn.click(
            fn=_gnd_stop_click,
            outputs=[_gnd_auto_state, gnd_auto_btn, gnd_timer, gnd_rec_btn, gnd_rec_display],
        )
        gnd_timer.tick(
            fn=_run_grounding,
            inputs=[api_url_box, backend_radio, instr_box_real, _gnd_history_rows, _gnd_count],
            outputs=_gnd_outputs,
        )

        def _gnd_toggle_record():
            if _gnd_recording[0]:
                _gnd_recording[0] = False
                if _gnd_video_writer[0] is not None:
                    _gnd_video_writer[0].release()
                    _gnd_video_writer[0] = None
                saved = str(_gnd_video_path[0]) if _gnd_video_path[0] else "—"
                return gr.update(value="🔴 녹화", variant="secondary"), f"저장됨: {saved}"
            else:
                _gnd_recording[0] = True
                _gnd_video_writer[0] = None  # 첫 프레임에서 lazy init
                return gr.update(value="⏹ 녹화 중지", variant="stop"), "🔴 녹화 시작 대기..."

        gnd_rec_btn.click(
            fn=_gnd_toggle_record,
            outputs=[gnd_rec_btn, gnd_rec_display],
        )

        def _gnd_stop_all():
            """⏹ 정지: 자동+녹화 모두 종료."""
            if _gnd_recording[0]:
                _gnd_recording[0] = False
                if _gnd_video_writer[0] is not None:
                    _gnd_video_writer[0].release()
                    _gnd_video_writer[0] = None

        def _gnd_new_session():
            _gnd_log_file[0] = None
            _gnd_log_last_write[0] = None
            return [], 0, "(새 세션 — 첫 검증 시 생성됨)"

        gnd_new_session_btn.click(
            fn=_gnd_new_session,
            outputs=[_gnd_history_rows, _gnd_count, gnd_log_display],
        )

      # ────────────────────────────────────────────────────────────────
      with gr.Tab("📊 Latency/Drift 진단"):
        gr.Markdown(
            "### \"4초\"는 단발 latency가 아니라 누적 드리프트\n"
            "1fps로 가정하고 자동 호출하지만, 실제 처리시간이 1초보다 길면 누적 차이(drift)가 "
            "계속 커진다. 이 탭은 운영 서버를 실시간으로 호출해 그 드리프트를 직접 재현·기록한다. "
            "(`docs/v5/s6_cl_sim.json` 105프레임 분석: frame 10에서 drift 3.96s ≈ \"4초\")"
        )
        drift_basis_radio = gr.Radio(
            ["1.0s (1fps 운영)", "1.35s (학습 수집 cadence)", "1.92s (SYNC 실측 풀사이클)", "전체 비교"],
            value="1.0s (1fps 운영)", label="가정시간 기준",
            info=(
                "1.35s = auto_play_core() 0.4s 이동타이머+0.15s stop펄스+0.8s rest (학습 데이터 수집 cadence) / "
                "1.92s = move_and_stop_ramped ramp(0.05s)+settle(0.15s)+그라운딩·MLP 추론(실측 ~1.717s) — 실제 SYNC 운영 풀사이클"
            ),
        )
        with gr.Row():
          with gr.Column(scale=3):
            drift_plot = gr.Plot(label="누적 실제시간 vs 가정시간")
            with gr.Row():
                drift_run_btn  = gr.Button("▶ 단발 측정", variant="primary", scale=2)
                drift_auto_btn = gr.Button("🔄 자동 (1fps)", variant="secondary", scale=2)
                drift_stop_btn = gr.Button("⏹ 정지", variant="stop", scale=1)
                drift_diag_btn = gr.Button("🩺 진단 실행 (A+D 체크)", scale=2)

          with gr.Column(scale=2):
            drift_clock     = gr.Textbox(label="현재 시각", value="—", interactive=False)
            drift_latency   = gr.Textbox(label="이번 호출 latency", value="—", interactive=False)
            drift_frame_n   = gr.Textbox(label="누적 프레임 수", value="0", interactive=False)
            drift_cum_real  = gr.Textbox(label="누적 실제시간", value="0.00s", interactive=False)
            drift_cum_nom   = gr.Textbox(label="누적 가정시간", value="0.0s", interactive=False)
            drift_now       = gr.HTML(value="<div style='font-size:1.3rem;font-weight:800;color:#22c55e'>drift: 0.00s</div>")
            drift_dual_panel = gr.Textbox(
                label="기준별 drift 비교 (전체 비교 모드)", value="—", interactive=False,
            )
            drift_history   = gr.Dataframe(
                headers=["#", "latency(ms)", "누적실제(s)", "누적가정(s)", "drift(s)"],
                datatype=["number", "number", "number", "number", "number"],
                label="최근 10회 이력",
                row_count=10,
                col_count=5,
                interactive=False,
            )
            with gr.Row():
                drift_log_display = gr.Textbox(
                    label="JSONL 경로", value="(첫 측정 시 생성됨)",
                    interactive=False, scale=3,
                )
                drift_new_session_btn = gr.Button("🆕 새 세션", scale=1, size="sm")
            drift_diag_result = gr.Textbox(label="진단 결과", value="—", interactive=False, lines=3)

        # ── Drift 탭 로직 ──────────────────────────────────────────────
        _drift_session = gr.State([])  # [{frame, latency_ms}, ...]

        _drift_log_dir = Path("logs/drift_sessions")
        _drift_log_dir.mkdir(parents=True, exist_ok=True)
        _drift_log_file: list = [None]
        _drift_log_last_write: list = [None]

        def _drift_ensure_log():
            now = _dt.datetime.now()
            last = _drift_log_last_write[0]
            if _drift_log_file[0] is None or (last is not None and now - last > _GND_SESSION_IDLE_GAP):
                ts = now.strftime("%Y%m%d_%H%M%S")
                _drift_log_file[0] = _drift_log_dir / f"drift_{ts}.jsonl"
            _drift_log_last_write[0] = now
            return _drift_log_file[0]

        def _drift_basis_values(basis):
            if basis and basis.startswith("1.35"):
                return [1.35]
            if basis and basis.startswith("1.92"):
                return [1.92]
            if basis == "전체 비교":
                return [1.0, 1.35, 1.92]
            return [1.0]

        _DRIFT_BASIS_STYLE = {
            1.0: ("#64748b", "1.0s 가정 (1fps 운영)"),
            1.35: ("#3b82f6", "1.35s 가정 (학습 수집 cadence)"),
            # ramp(0.05s, move_and_stop_ramped 블로킹 구간) + settle(0.15s) + 추론(그라운딩+MLP, 실측 ~1.717s)
            1.92: ("#a855f7", "1.92s 가정 (SYNC 실측 풀사이클)"),
        }

        def _drift_build_plot(session, basis):
            frames = [e["frame"] for e in session]
            cum_real = list(np.cumsum([e["latency_ms"] for e in session]) / 1000.0)
            bases = _drift_basis_values(basis)
            fig, ax = plt.subplots(figsize=(6, 4))
            for b in bases:
                color, label = _DRIFT_BASIS_STYLE[b]
                cum_nom_b = [f * b for f in frames]
                ax.plot(frames, cum_nom_b, "--", color=color, linewidth=2, label=label)
            ax.plot(frames, cum_real, "-", color="#ef4444", linewidth=2.5, label="실제 처리시간")
            primary_nom = [f * bases[0] for f in frames]
            ax.fill_between(frames, primary_nom, cum_real, color="#ef4444", alpha=0.15)
            over4 = [f for f, r, n in zip(frames, cum_real, primary_nom) if (r - n) >= 4.0]
            if over4:
                f0 = over4[0]
                ax.axvline(f0, color="#facc15", linestyle=":", linewidth=1.5)
                ax.annotate("drift ≥ 4.0s", xy=(f0, cum_real[frames.index(f0)]),
                            color="#facc15", fontsize=9, fontweight="bold")
            ax.set_xlabel("frame")
            ax.set_ylabel("누적시간(s)")
            ax.set_title("실제 vs 가정 — 누적시간")
            ax.legend(loc="upper left")
            ax.grid(True, linestyle="--", alpha=0.4)
            return fig

        def _drift_run(api_url, backend_mode, instr, session, basis):
            import json as _json
            bases = _drift_basis_values(basis)
            primary_basis = bases[0]
            frame = None
            if ROS_AVAILABLE and ros_node:
                frame = ros_node.get_inference_frame()
            now_str = _dt.datetime.now().strftime("%H:%M:%S")
            log_path_str = str(_drift_ensure_log())

            if frame is None:
                return (now_str, "❌ 카메라 없음", str(len(session)), "—", "—",
                        "<div style='color:#ef4444'>카메라 연결 필요</div>", "—",
                        [[e["frame"], round(e["latency_ms"]), 0, 0, 0] for e in session[-10:][::-1]],
                        None, log_path_str, session)

            try:
                result = run_backend_inference(frame, instr or "", backend_mode or "", api_url, execute_move=False)
                lat_ms = float(result.get("latency_ms") or 0.0)
            except Exception as e:
                return (now_str, f"❌ 오류: {e}", str(len(session)), "—", "—",
                        "<div style='color:#ef4444'>호출 실패</div>", "—",
                        [[e2["frame"], round(e2["latency_ms"]), 0, 0, 0] for e2 in session[-10:][::-1]],
                        None, log_path_str, session)

            frame_idx = len(session) + 1
            session = session + [{"frame": frame_idx, "latency_ms": lat_ms}]
            cum_real = sum(e["latency_ms"] for e in session) / 1000.0
            cum_nom = frame_idx * primary_basis
            drift = cum_real - cum_nom

            color = "#22c55e" if drift < 1.0 else "#f59e0b" if drift < 4.0 else "#ef4444"
            drift_html = f"<div style='font-size:1.3rem;font-weight:800;color:{color}'>drift: {drift:.2f}s</div>"
            if drift >= 4.0:
                drift_html += "<div style='color:#facc15;font-size:0.8rem'>⚠ 4초 돌파 — 단발 latency 아니라 누적 드리프트</div>"

            if len(bases) > 1:
                parts = []
                for b in bases:
                    d_b = cum_real - frame_idx * b
                    parts.append(f"{b}s 기준: {d_b:+.2f}s")
                dual_panel_str = " | ".join(parts)
            else:
                dual_panel_str = "—"

            rows, cr = [], 0.0
            for e in session:
                cr += e["latency_ms"] / 1000.0
                nom_e = e["frame"] * primary_basis
                rows.append([e["frame"], round(e["latency_ms"]), round(cr, 2), round(nom_e, 2), round(cr - nom_e, 2)])
            history_rows = rows[-10:][::-1]

            record = {
                "ts": _dt.datetime.now().isoformat(timespec="milliseconds"),
                "frame": frame_idx, "latency_ms": round(lat_ms, 1),
                "cum_real_s": round(cum_real, 3), "cum_nominal_s": round(cum_nom, 3), "drift_s": round(drift, 3),
                "nominal_basis_s": primary_basis,
            }
            with open(_drift_log_file[0], "a") as f:
                f.write(_json.dumps(record, ensure_ascii=False) + "\n")

            fig = _drift_build_plot(session, basis)
            status = "PASS" if lat_ms < 1000 else "WARN" if lat_ms < 2000 else "FAIL"
            return (now_str, f"{lat_ms:.0f}ms [{status}]", str(frame_idx), f"{cum_real:.2f}s", f"{cum_nom:.1f}s",
                    drift_html, dual_panel_str, history_rows, fig, log_path_str, session)

        _drift_outputs = [drift_clock, drift_latency, drift_frame_n, drift_cum_real, drift_cum_nom,
                           drift_now, drift_dual_panel, drift_history, drift_plot, drift_log_display, _drift_session]

        drift_run_btn.click(
            fn=_drift_run,
            inputs=[api_url_box, backend_radio, instr_box_real, _drift_session, drift_basis_radio],
            outputs=_drift_outputs,
        )

        _drift_auto_state = gr.State(False)
        drift_timer = gr.Timer(1.0, active=False)

        def _toggle_drift_auto(is_active):
            new_state = not is_active
            label = "⏸ 자동 중지" if new_state else "🔄 자동 (1fps)"
            variant = "primary" if new_state else "secondary"
            return new_state, gr.update(value=label, variant=variant), gr.update(active=new_state)

        drift_auto_btn.click(
            fn=_toggle_drift_auto,
            inputs=[_drift_auto_state],
            outputs=[_drift_auto_state, drift_auto_btn, drift_timer],
        )
        drift_timer.tick(
            fn=_drift_run,
            inputs=[api_url_box, backend_radio, instr_box_real, _drift_session, drift_basis_radio],
            outputs=_drift_outputs,
        )

        def _drift_stop_click():
            return False, gr.update(value="🔄 자동 (1fps)", variant="secondary"), gr.update(active=False)

        drift_stop_btn.click(
            fn=_drift_stop_click,
            outputs=[_drift_auto_state, drift_auto_btn, drift_timer],
        )

        def _drift_new_session():
            _drift_log_file[0] = None
            _drift_log_last_write[0] = None
            return [], None, "(새 세션 — 첫 측정 시 생성됨)"

        drift_new_session_btn.click(
            fn=_drift_new_session,
            outputs=[_drift_session, drift_plot, drift_log_display],
        )

        def _drift_run_diagnostics(api_url):
            try:
                from scripts.eval.diagnose_pipeline_health import check_latency, check_resize
                r_resize = check_resize()
                r_latency = check_latency(api_url, API_KEY, n=5)
                return (f"[D.RESIZE] {r_resize['status']} — output_size={r_resize['output_size']}\n"
                        f"[A.LATENCY] {r_latency['status']} — mean={r_latency['mean_ms']:.0f}ms "
                        f"(n={len(r_latency['samples'])}, 목표<1000ms)")
            except Exception as e:
                return f"❌ 진단 실행 실패: {e}"

        drift_diag_btn.click(
            fn=_drift_run_diagnostics,
            inputs=[api_url_box],
            outputs=drift_diag_result,
        )

        drift_clock_timer = gr.Timer(1.0, active=True)
        drift_clock_timer.tick(
            fn=lambda: _dt.datetime.now().strftime("%H:%M:%S"),
            outputs=drift_clock,
        )

    btn_start_inf.click(
        fn=lambda mode, url, instr, gt, cc: set_running(True, mode, url, instr, gt, apply_cc=cc),
        inputs=[backend_radio, api_url_box, instr_box_real, gt_object_box, toggle_cc],
        outputs=run_status_box,
    )
    btn_stop_inf.click(
        fn=lambda: set_running(False, "", "", ""),
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
        button.click(
            fn=handle_control,
            inputs=[gr.State(direction), manual_speed_slider],
            outputs=status_log,
        )

    # 슬라이더 변경 → 조이스틱 속도 동기화 + 실제 하드웨어 throttle(PWM%) 반영
    # (데이터 수집 그라디오의 throttle_sl → node.throttle 패턴과 동일. 기존엔 lx/ly/az
    # 벡터 크기만 바뀌고 VLAControlManager.throttle은 생성 시 고정값(50)이라 실제
    # PWM 출력엔 영향이 없었음 — 슬라이더 기본값 1.15가 throttle=50과 같아지도록 비례.)
    def _sync_js_speed(spd):
        spd = float(spd)
        _joystick._speed = spd
        if ROS_AVAILABLE and ros_node:
            throttle = int(round(spd / 1.15 * 50))
            ros_node.control.throttle = max(10, min(100, throttle))
        return gr.update()
    manual_speed_slider.change(fn=_sync_js_speed, inputs=manual_speed_slider)

    def _get_bbox_area_display():
        """최근 예측의 bbox area 표시 (정지 판단 기준 시각화)."""
        try:
            import requests as _req
            r = _req.get(f"{DEFAULT_API_URL}/recent", timeout=2)
            preds = r.json().get("predictions", [])
            if preds:
                p = preds[0]
                bbox = p.get("bbox", {})
                area = bbox.get("area", 0)
                entity = bbox.get("entity", "?")
                cx = bbox.get("cx", 0.5)
                near = "🔴 STOP 조건 충족!" if area >= 0.18 and abs(cx - 0.5) <= 0.25 and entity not in ("coarse_clf", "center_fallback", "") and not entity.startswith("caption:") else ""
                return f"area={area:.3f}  cx={cx:.2f}  [{entity[:20]}]  {near}"
        except Exception:
            pass
        return "—"

    _ui_inputs = [mode_radio, backend_radio, api_url_box, instr_box_real, toggle_cc, run_status_box, infer_move_radio]
    _ui_outputs = [camera_output, status_log, latency_val, action_val, chunk_val, run_status_box, camera_status, model_path, traj_plot]

    timer = gr.Timer(0.5, active=True)
    timer.tick(fn=update_ui, inputs=_ui_inputs, outputs=_ui_outputs)
    timer.tick(fn=_get_bbox_area_display, outputs=bbox_area_display)
    timer.tick(fn=_js_status_text, outputs=js_status)
    timer.tick(fn=_run_history_rows, outputs=run_history_table)
    # 페이지 열리자마자 첫 프레임 즉시 표시
    demo.load(fn=update_ui, inputs=_ui_inputs, outputs=_ui_outputs)
    # 카메라 시작 버튼 완료 후 즉시 프레임 가져오기
    _cam_start_btn.click(fn=start_camera, outputs=_cam_st).then(
        fn=update_ui, inputs=_ui_inputs, outputs=_ui_outputs,
    )

    btn_reset.click(
        fn=reset_model_wrapper,
        inputs=[backend_radio, api_url_box, instr_box_real],
        outputs=status_log,
    )

    def on_exp_mode_change(mode_name, api_url, backend_mode):
        cfg = EXP_MODES.get(mode_name, EXP_MODES[EXP_MODE_NAMES[0]])
        instr = cfg["instruction"]
        model_key = cfg.get("model")
        desc = cfg.get("desc", "")
        # model_key가 있으면 모두 API 서버에 동기화 (GoalNav / Stage2v2 등 구분 불필요)
        is_goal = bool(model_key)

        # config/checkpoint 자동 매칭 (상대경로 → 절대경로 변환)
        def _abs(rel):
            if not rel:
                return None
            p = Path(rel)
            return str(p if p.is_absolute() else PROJECT_ROOT / p)

        auto_conf = _abs(cfg.get("config"))
        auto_ckpt = _abs(cfg.get("checkpoint"))
        conf_update = gr.update(value=auto_conf) if auto_conf else gr.update()
        ckpt_update = gr.update(value=auto_ckpt) if auto_ckpt else gr.update()

        # model_key가 있으면 API 서버에 config push
        cfg_status = ""
        if model_key:
            try:
                ApiInferenceBackend(api_url).set_config(
                    speed_scaling=cfg["speed_scaling"],
                    grounding_skip_n=cfg["grounding_skip_n"],
                    model=model_key,
                )
                parts = [f"model={model_key}", f"skip_n={cfg['grounding_skip_n']}"]
                if cfg["speed_scaling"]:
                    parts.append("속도비례ON")
                cfg_status = "✅ 서버 적용: " + ", ".join(parts)
                if auto_conf:
                    cfg_status += f"  |  📋 {Path(auto_conf).name}"
            except Exception as e:
                cfg_status = f"⚠️ 서버 적용 실패: {e}"
                if auto_conf:
                    cfg_status += f"  |  📋 로컬: {Path(auto_conf).name}"
        elif auto_conf:
            cfg_status = f"📋 자동 매칭: {Path(auto_conf).name}"
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

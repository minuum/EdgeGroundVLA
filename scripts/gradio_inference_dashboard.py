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
DEFAULT_INSTRUCTION = "the gray basket"
PATH_TYPES = [
    # ── 경로 검증 (시작위치_목표방향) ────────────────────────────────
    "right_right", "right_left", "right_straight",
    "center_straight", "center_left", "center_right",
    "left_straight", "left_left", "left_right",
    # ── 교수님 프로토콜: 오브젝트 위치별 (각 30회) ───────────────────
    "obj_left", "obj_center", "obj_right",
    # ── 교수님 프로토콜: 박스 거리별 (각 10회) ───────────────────────
    "dist_10cm", "dist_20cm", "dist_30cm",
]
# 경로별 목표 에피소드 수
PATH_TARGETS = {
    # 경로 검증 — 합계 11, 성공 목표 7
    "right_right":    2,
    "right_left":     2,   # ★ 최우선 — 가장 어려운 교차
    "right_straight": 1,
    "center_straight":1,
    "center_left":    1,
    "center_right":   1,
    "left_straight":  1,
    "left_left":      1,
    "left_right":     1,
    # 오브젝트 위치별 — 합계 90
    "obj_left":   30,
    "obj_center": 30,
    "obj_right":  30,
    # 박스 거리별 — 합계 30
    "dist_10cm":  10,
    "dist_20cm":  10,
    "dist_30cm":  10,
}
GOAL_SUCCESS_TARGET = 7  # 논문 기준 (경로 검증)
# 그룹 구분선 — 집계 테이블 렌더링 시 섹션 헤더 삽입
_PATH_GROUPS = [
    ("── 경로 검증 ──────────────",
     ["right_right","right_left","right_straight",
      "center_straight","center_left","center_right",
      "left_straight","left_left","left_right"]),
    ("── 오브젝트 위치별 ──────────",
     ["obj_left","obj_center","obj_right"]),
    ("── 박스 거리별 ──────────────",
     ["dist_10cm","dist_20cm","dist_30cm"]),
]
GOAL_NAV_PRESETS = [
    "the gray basket",
    "the door",
    "the corridor on the left",
    "the corridor on the right",
]

# 실험 모드: (표시 이름, instruction, backend_instruction_mode, speed_scaling, grounding_skip_n)
# grounding_skip_n=3 고정 — 의도적 결정, 버그 아님 (docs/v5/research_story.html CH49 참고).
# skip_n=1(매프레임 그라운딩)은 실측 latency 1.3~1.4s/frame이라 실시간 주행 불가능했고,
# 오히려 skip_n=3의 bbox 캐시 재사용이 잡음 저역통과 필터처럼 작동해 baseline 성능을
# 끌어올림(CL 93.1%→96.6%, FPE 0.145→0.119m). area_delta 기법(CH47, FPE 0.098m)은
# skip_n=1 조건에서만 유효해서 현재는 미배포 — 이 둘을 같이 바꾸려 하지 말 것.
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
    # Local Runtime 요청이지만 모델 미로드 시 → API Server 자동 폴백
    if not shared_runtime.get_model_info().get("model_loaded"):
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
    # 추론 이동 모드 — VLA_ASYNC_MODE=1 env var로 ASYNC 강제 가능 (go.sh 제어)
    # SYNC : 이동 완료 후 150ms settle → stable_frame 캡처 → 다음 스텝 추론 (기본, 데이터 수집용)
    # PRE  : 이동 직전 live frame → 추론 → 이동 (수집 PRE_CACHE와 동일 분포)
    # ASYNC: inference_thread(3Hz) + execution_thread(10Hz) 완전 분리 (데모용)
    "infer_move_mode": "ASYNC" if os.environ.get("VLA_ASYNC_MODE", "0") == "1" else "SYNC",
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
    if not (logger_instance and logger_instance.data and logger_instance.data.get("history")):
        return None
    # 도착/정지 직후 최종 프레임을 한 장 더 수집 (arrival frame)
    try:
        if ROS_AVAILABLE and ros_node:
            import time as _t
            _t.sleep(0.25)
            final_frame = ros_node.get_inference_frame()
            if final_frame is not None:
                n_steps = len(logger_instance.data["history"])
                logger_instance.log_step(
                    n_steps + 1,
                    [0.0, 0.0, 0.0],
                    0,
                    image=final_frame,
                    predicted_label="STOP",
                )
    except Exception:
        pass
    path = logger_instance.end_session(status)
    if path:
        print(f"💾 세션 저장: {path} ({status})")
    return path


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
        # 프레임 로깅 (ASYNC 모드에서도 세션 H5 기록)
        if logger_instance:
            logger_instance.log_step(
                step,
                result.get("action", [0.0, 0.0, 0.0]),
                result.get("latency_ms", 0),
                image=img,
                predicted_label=result.get("predicted_label"),
                bbox=result.get("bbox"),
                grounding_cached=result.get("grounding_cached"),
                grounding_latency_ms=result.get("grounding_latency_ms"),
                goal_near=result.get("goal_near"),
            )
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
        # 그라운딩 캐시 프리워밍 — START 버튼 시점에 현재 프레임으로 /ground 선호출
        # 이를 통해 첫 /predict에서 cached=1이 되어 frame0 STOP 제거
        try:
            import requests as _rq_pw, base64 as _b64_pw, io as _io_pw
            _img_pw = state.get("last_img")
            if _img_pw is not None:
                _pil_pw = Image.fromarray(_img_pw) if isinstance(_img_pw, np.ndarray) else _img_pw
                _buf_pw = _io_pw.BytesIO()
                _pil_pw.save(_buf_pw, format="PNG")
                _b64_pw_str = _b64_pw.b64encode(_buf_pw.getvalue()).decode()
                _phrase_pw = gt_object if gt_object else "gray basket"
                _rq_pw.post(
                    f"{api_url}/ground",
                    json={"image": _b64_pw_str, "prompt": f"detect {_phrase_pw}"},
                    headers={"X-API-Key": API_KEY},
                    timeout=12,
                )
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
    is_preview = bool(result.get("preview_align"))
    preview_attempt = result.get("preview_attempt")

    label_prefix = f"[{pred_label}] " if pred_label else ""
    if is_preview and preview_attempt is not None:
        label_prefix = f"[🔄PREVIEW {preview_attempt}] [{pred_label}] "
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
        if is_preview:
            chunk_display = (f"🔄 PREVIEW 모드 — attempt {preview_attempt}\n"
                             f"bbox 미탐지 → ROT 후 PG2 재검사 중\n") + chunk_display

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

    # is_running=True (Tab4 START 등)이면 mode_radio 값과 무관하게 inference 활성
    state["auto_inference"] = (mode == "Inference (Auto)") or state["is_running"]
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
                except RuntimeError as _re:
                    if "not loaded" in str(_re).lower():
                        # Local 모드이지만 모델 없음 → API 서버로 fallback
                        try:
                            import requests as _rr
                            _rr.post(f"{api_url}/reset", json={}, timeout=3)
                        except Exception:
                            pass  # reset 실패해도 추론은 계속
                except Exception:
                    pass  # reset 실패해도 추론은 계속
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
                    report_path = _flush_session("goal_reached")
                    log = f"🎯 Goal Reached! (step {current_step}) | Log: {Path(report_path).name if report_path else '?'}"
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


def _update_ui_and_cache(*args, **kwargs):
    result = update_ui(*args, **kwargs)
    if isinstance(result, tuple) and len(result) >= 4:
        for i, key in [(1, "_t4_log"), (2, "_t4_lat"), (3, "_t4_act")]:
            if isinstance(result[i], str):
                state[key] = result[i]
    return result


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
                    # move_and_stop_ramped()는 내부에 move_duration 후 자동 STOP을
                    # 쏘는 Timer가 있어서, sleep(move_duration)으로 다음 step을
                    # 부르면 매 step마다 "이동→자동STOP→이동→자동STOP"으로 끊김
                    # (정지 명령이 매 step마다 섞여 들어감). publish_and_move()는
                    # 그 auto-stop Timer가 없어 끊김 없이 연속 재생됨 — STOP은
                    # 루프 끝난 뒤 robust_stop()으로 한 번만.
                    ros_node.control.publish_and_move(lx, ly, az, source="return")
                    time.sleep(ros_node.control.move_duration)
            if ROS_AVAILABLE and ros_node:
                ros_node.control.robust_stop(source="return_done")
        finally:
            state["is_returning"] = False

    import threading
    threading.Thread(target=_run, daemon=True).start()
    return f"🔄 복귀 중... ({len(history)}스텝 역재생)"


def reset_model_wrapper(backend_mode: str, api_url: str, instruction: str):
    try:
        result = make_backend(backend_mode, api_url).reset(instruction)
        return result
    except RuntimeError as e:
        if "not loaded" in str(e).lower():
            # Local 모드이지만 모델이 없는 경우 → API 서버로 fallback
            try:
                import requests as _r
                _r.post(f"{api_url}/reset", json={}, timeout=3)
                return f"✅ API reset ({api_url})"
            except Exception as e2:
                return f"❌ Reset failed (local: {e}, api: {e2})"
        return f"❌ Reset failed: {e}"
    except Exception as e:
        return f"❌ Reset failed: {e}"


def _make_env_banner() -> str:
    import socket as _sock
    import requests as _req
    hostname = _sock.gethostname()
    role = os.getenv("VLA_SERVER_ROLE", "unknown")
    api = os.getenv("VLA_API_SERVER", "http://localhost:8001")
    exp_name = EXP_MODE_NAMES[0] if EXP_MODE_NAMES else "—"

    # API 서버 실시간 상태
    srv_color = "#ef4444"
    srv_label = "❌ 서버 오프라인"
    model_label = os.getenv("VLA_MODEL", "—")
    try:
        info = _req.get(f"{api}/health", timeout=1.5).json()
        if info.get("model_loaded"):
            srv_color = "#22c55e"
            stop_mode = info.get("stop_mode", "?")
            latched = info.get("stop_latched", False)
            latch_tag = " 🔒LATCHED" if latched else ""
            srv_label = f"✅ {info.get('head','?')} w={info.get('window','?')} | {stop_mode}{latch_tag} | GPU {info.get('gpu',{}).get('allocated_gb','?'):.2f}GB"
            ckpt = info.get("checkpoint_path", "")
            if ckpt:
                model_label = Path(ckpt).stem[:28]
        else:
            srv_color = "#f59e0b"
            srv_label = "⚠️ 서버 온라인 (모델 미로드)"
    except Exception:
        pass

    return (
        f'<div style="background:#0f172a;border-left:4px solid {srv_color};padding:10px 14px;'
        f'border-radius:4px;margin-bottom:12px;color:#e2e8f0;font-size:0.88rem;line-height:1.6;">'
        f'<strong style="color:#4ade80;font-size:0.95rem;">MoNaVLA 환경</strong>'
        f'&nbsp;&nbsp;|&nbsp;&nbsp;'
        f'<code style="color:#86efac">{hostname}</code>'
        f'&nbsp;(<span style="color:#fbbf24">{role}</span>)'
        f'&nbsp;&nbsp;|&nbsp;&nbsp;'
        f'모델&nbsp;<code style="color:#67e8f9">{model_label}</code>'
        f'&nbsp;&nbsp;|&nbsp;&nbsp;'
        f'API&nbsp;<code style="color:#a5b4fc">{api}</code>'
        f'&nbsp;&nbsp;|&nbsp;&nbsp;'
        f'<span style="color:{srv_color}">{srv_label}</span>'
        f'&nbsp;&nbsp;|&nbsp;&nbsp;'
        f'실험&nbsp;<span style="color:#f9a8d4">{exp_name}</span>'
        f'</div>'
    )


_FONT_SCALE_CSS = """
.gradio-container { font-size: 120% !important; }

/* ── Tab 4 전용 대형 폰트 ── */
#tab4-root label,
#tab4-root .label-wrap span,
#tab4-root .textbox textarea,
#tab4-root .textbox input,
#tab4-root .prose,
#tab4-root p,
#tab4-root td,
#tab4-root th {
    font-size: 1.15rem !important;
    line-height: 1.5 !important;
}
#tab4-root .gr-button {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    padding: 10px 16px !important;
}
#tab4-root code, #tab4-root pre {
    font-size: 0.95rem !important;
}
#tab4-root .gr-radio label span,
#tab4-root .gr-dropdown label span {
    font-size: 1.15rem !important;
}
/* 진행률 강조 */
#t4-progress textarea {
    font-size: 1.4rem !important;
    font-weight: 800 !important;
    color: #1a6b3c !important;
    background: #e8f5e9 !important;
}
/* 기록 버튼 강조 */
#btn-log { background: #1565c0 !important; font-size: 1.2rem !important; }
#btn-undo { font-size: 1.1rem !important; }
#btn-clear { font-size: 1.1rem !important; }
"""

with gr.Blocks(
    title="MoNaVLA Dashboard",
    css=_FONT_SCALE_CSS,
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.cyan,
        neutral_hue=gr.themes.colors.slate,
        font=[gr.themes.GoogleFont("Noto Sans KR"), "sans-serif"],
    ),
) as demo:
    gr.Markdown("# MoNaVLA Real-time Dashboard")
    _env_banner = gr.HTML(_make_env_banner())
    _banner_timer = gr.Timer(10.0, active=True)

    _cam_st, _cam_start_btn, _cam_stop_btn = camera_control_widget()
    _cam_start_btn.click(fn=start_camera, outputs=_cam_st)
    _cam_stop_btn.click(fn=stop_camera,   outputs=_cam_st)

    with gr.Tabs():
      with gr.Tab("🤖 Drive / Inference"):
        # ── 기존 메인 탭 내용 시작 ──
        with gr.Row(equal_height=False):
          with gr.Column(scale=2):
            gr.Markdown("## 📷 Live Camera")
            camera_output = gr.Image(label="Live Camera (via Service)", interactive=False)
            gr.Markdown("🟢 Continuous polling via GetImage service")

          with gr.Column(scale=1):
            gr.Markdown("## 📡 모니터링 1 (실험설정/상태)")
            with gr.Group():
                with gr.Row(equal_height=True):
                    exp_mode = gr.Dropdown(
                        choices=EXP_MODE_NAMES,
                        value=EXP_MODE_NAMES[0],
                        label="실험 모드",
                        scale=2,
                        min_width=60,
                    )
                    exp_config_status = gr.Textbox(label="서버 Config", value="미적용", interactive=False, scale=1, min_width=60)
                with gr.Row(equal_height=True):
                    goal_dropdown = gr.Dropdown(
                        choices=["(직접 입력)"] + GOAL_NAV_PRESETS,
                        value=GOAL_NAV_PRESETS[0],
                        label="Goal Object 선택",
                        visible=True,
                        scale=1,
                        min_width=60,
                    )
                    path_dropdown = gr.Dropdown(
                        choices=PATH_TYPES,
                        value="right_right",
                        label="Path Type 선택",
                        visible=False,
                        scale=1,
                        min_width=60,
                    )
                instr_box_real = gr.Textbox(
                    label="🤖 Robot Prompt (모델에게 주는 프롬프트 — 틀린 값 테스트 가능)",
                    value=DEFAULT_INSTRUCTION,
                )
                with gr.Row(equal_height=True):
                    gt_object_box = gr.Textbox(
                        label="🎯 GT Object (모델에 전달 안됨, 로깅용)",
                        value="gray basket",
                        placeholder="예: gray basket",
                        scale=1,
                        min_width=60,
                    )
                    camera_status = gr.Textbox(label="Camera Status", value="Unknown", interactive=False, scale=1, min_width=60)

          with gr.Column(scale=1):
            gr.Markdown("## 📟 모니터링 2 (실시간 상태)")
            status_log = gr.Textbox(label="Status", value="Ready")
            latency_val = gr.Textbox(label="Latency", value="0 ms")
            action_val = gr.Textbox(label="Predicted Action [lx, ly, az]", value="0, 0, 0")
            chunk_val = gr.Textbox(label="Action Chunk Preview", value="N/A", lines=3)
            srv_status_t1 = gr.Textbox(
                label="🖥️ 서버 상태", value="—", interactive=False, max_lines=2,
            )
            srv_log_t1 = gr.Textbox(
                label="📋 추론 서버 로그 (최근 4줄)", value="—",
                interactive=False, lines=4, max_lines=4,
            )

          with gr.Column(scale=1):
            gr.Markdown("## 📊 모니터링 3 (그래프/히스토리)")
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

            _is_learned = (_SERVER_STOP_MODE == "learned")
            _stop_acc_label = (
                "🤖 자동 정지 — Learned STOP (N1 모델)"
                if _is_learned else
                "📐 자동 정지 — Proximity Threshold"
            )
            with gr.Accordion(_stop_acc_label, open=False):
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
                # .click() 바인딩은 api_url_box(Operation Mode 블록, 아래에서 정의)가 생긴 뒤로 미룸 — 끝부분 참고


        with gr.Column():
          gr.Markdown("## 🎮 Operation Mode (항상 펼쳐짐, 전체 너비)")
          with gr.Group():
              gr.Markdown("### 🕹️ Operation Mode")
              # Controller Mode + Inference Backend + API URL — 한 행 (항상 표시,
              # Manual Drive에서도 exp_mode config push에 사용)
              with gr.Row():
                  mode_radio = gr.Radio(
                      choices=["Manual Drive", "Inference (Auto)"],
                      value="Manual Drive",
                      label="Controller Mode",
                      scale=1,
                      min_width=120,
                  )
                  backend_radio = gr.Radio(
                      choices=["Local Runtime", "API Server"],
                      value=DEFAULT_BACKEND_MODE,
                      label="Inference Backend",
                      scale=1,
                      min_width=120,
                  )
                  api_url_box = gr.Textbox(
                      label="API URL",
                      value=DEFAULT_API_URL,
                      scale=2,
                      min_width=120,
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
                      scale=2,
                      min_width=100,
                  )
                  toggle_cc = gr.Checkbox(label="🎨 Red Gain Boost", value=False, scale=1, min_width=80)
                  model_path = gr.Textbox(
                      label="Active Model / Checkpoint", value="N/A", interactive=False,
                      scale=2, min_width=120,
                  )

              # 추론 제어 — Inference (Auto) 선택 시만 표시
              with gr.Column(visible=False) as inference_panel:
                  gr.Markdown("#### 🏁 Inference Control")
                  infer_move_radio = gr.Radio(
                      choices=["SYNC", "PRE", "ASYNC"],
                      value="ASYNC" if os.environ.get("VLA_ASYNC_MODE", "0") == "1" else "SYNC",
                      label="이동 모드",
                      info="SYNC: 정지→추론→이동 (데이터수집용) | PRE: 캡처→추론→이동 | ASYNC: 추론(3Hz)+실행(10Hz) 분리 (데모용)",
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

          with gr.Row():
            with gr.Column(scale=1, min_width=200):
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

            with gr.Column(scale=1, min_width=200):
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

        stop_apply_btn.click(
            fn=apply_stop_config,
            inputs=[stop_area_slider, stop_cx_slider, api_url_box],
            outputs=stop_config_status,
        )
        # ── 기존 메인 탭 내용 끝 ──

      # ────────────────────────────────────────────────────────────────
      with gr.Tab("🔍 Grounding 검증"):
        gr.Markdown(
            "### PG2 실시간 Grounding 검증\n"
            "카메라 프레임을 서버로 전송해 PG2가 바구니를 검출하는지 확인. "
            "추론 시작 전 bbox area가 올라가는지 여기서 먼저 체크."
        )
        with gr.Row(equal_height=False):
          with gr.Column(scale=2):
            gnd_image = gr.Image(
                label="Camera + BBox Overlay",
                interactive=False,
            )
            with gr.Row():
                gnd_run_btn  = gr.Button("▶ 단발 검증", variant="primary", scale=2)
                gnd_auto_btn = gr.Button("🔄 자동 (1fps)", variant="secondary", scale=2)
                gnd_stop_btn = gr.Button("⏹ 정지", variant="stop", scale=1)
                gnd_rec_btn  = gr.Button("🔴 녹화", variant="secondary", scale=1)

          with gr.Column(scale=1):
            gr.Markdown("##### 검출 상태")
            gnd_clock    = gr.Textbox(label="현재 시각", value="—", interactive=False)
            gnd_has_bbox = gr.Textbox(label="검출 결과", value="—", interactive=False)
            gnd_area_bar = gr.HTML(value=_gnd_area_html(0.06, False), label="")
            gnd_cx_bar   = gr.HTML(value=_gnd_cx_html(0.5, False),   label="")

          with gr.Column(scale=1):
            gr.Markdown("##### Latency/Raw")
            gnd_latency  = gr.Textbox(label="Grounding latency", value="—", interactive=False)
            gnd_raw      = gr.Textbox(label="PG2 raw output", value="—", interactive=False, lines=2)
            gnd_server_cmp = gr.Textbox(label="서버 예측", value="—", interactive=False)

          with gr.Column(scale=1):
            gr.Markdown("##### 이력/세션")
            gnd_history   = gr.Dataframe(
                headers=["#", "bbox", "area", "cx", "pred", "lat(ms)"],
                datatype=["number", "str", "number", "number", "str", "number"],
                label="최근 10회 이력",
                row_count=10,
                col_count=6,
                interactive=False,
            )
            gnd_log_display = gr.Textbox(
                label="JSONL 경로", value="(첫 검증 시 생성됨)",
                interactive=False,
            )
            gnd_new_session_btn = gr.Button("🆕 새 세션", size="sm")
            gnd_rec_display = gr.Textbox(label="녹화 경로", value="—", interactive=False)

        with gr.Column():
            with gr.Group():
                gr.Markdown("### 🎮 수동 조작 (탭 전환 없이 미세조정)")
                with gr.Row():
                    gnd_btn_q = gr.Button("↖ Q", scale=1, size="sm")
                    gnd_btn_w = gr.Button("▲ W", scale=1, size="sm")
                    gnd_btn_e = gr.Button("↗ E", scale=1, size="sm")
                    gnd_btn_a = gr.Button("◀ A", scale=1, size="sm")
                    gnd_btn_stop = gr.Button("⏹ STOP", variant="stop", scale=1, size="sm")
                    gnd_btn_d = gr.Button("▶ D", scale=1, size="sm")
                gnd_manual_status = gr.Textbox(label="조작 로그", value="—", interactive=False, lines=1)

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

        for _btn, _dir in [
            (gnd_btn_q, "Q"), (gnd_btn_w, "W"), (gnd_btn_e, "E"),
            (gnd_btn_a, "A"), (gnd_btn_stop, "STOP"), (gnd_btn_d, "D"),
        ]:
            _btn.click(fn=handle_control, inputs=[gr.State(_dir), manual_speed_slider], outputs=gnd_manual_status)

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
        with gr.Row(equal_height=False):
          with gr.Column(scale=2):
            drift_plot = gr.Plot(label="누적 실제시간 vs 가정시간")
            with gr.Row():
                drift_run_btn  = gr.Button("▶ 단발 측정", variant="primary", scale=2)
                drift_auto_btn = gr.Button("🔄 자동 (1fps)", variant="secondary", scale=2)
                drift_stop_btn = gr.Button("⏹ 정지", variant="stop", scale=1)
                drift_diag_btn = gr.Button("🩺 진단 실행 (A+D 체크)", scale=2)

          with gr.Column(scale=1):
            gr.Markdown("##### 실시간 수치")
            drift_clock     = gr.Textbox(label="현재 시각", value="—", interactive=False)
            drift_latency   = gr.Textbox(label="이번 호출 latency", value="—", interactive=False)
            drift_frame_n   = gr.Textbox(label="누적 프레임 수", value="0", interactive=False)
            drift_cum_real  = gr.Textbox(label="누적 실제시간", value="0.00s", interactive=False)
            drift_cum_nom   = gr.Textbox(label="누적 가정시간", value="0.0s", interactive=False)
            drift_now       = gr.HTML(value="<div style='font-size:1.3rem;font-weight:800;color:#22c55e'>drift: 0.00s</div>")

          with gr.Column(scale=1):
            gr.Markdown("##### 기준 비교/진단")
            drift_dual_panel = gr.Textbox(
                label="기준별 drift 비교 (전체 비교 모드)", value="—", interactive=False,
            )

          with gr.Column(scale=1):
            gr.Markdown("##### 이력/세션")
            drift_history   = gr.Dataframe(
                headers=["#", "latency(ms)", "누적실제(s)", "누적가정(s)", "drift(s)"],
                datatype=["number", "number", "number", "number", "number"],
                label="최근 10회 이력",
                row_count=10,
                col_count=5,
                interactive=False,
            )
            drift_log_display = gr.Textbox(
                label="JSONL 경로", value="(첫 측정 시 생성됨)",
                interactive=False,
            )
            drift_new_session_btn = gr.Button("🆕 새 세션", size="sm")
            drift_diag_result = gr.Textbox(label="진단 결과", value="—", interactive=False, lines=3)

        with gr.Column():
            with gr.Group():
                gr.Markdown("### 🎮 수동 조작 (탭 전환 없이 미세조정)")
                with gr.Row():
                    drift_btn_q = gr.Button("↖ Q", scale=1, size="sm")
                    drift_btn_w = gr.Button("▲ W", scale=1, size="sm")
                    drift_btn_e = gr.Button("↗ E", scale=1, size="sm")
                    drift_btn_a = gr.Button("◀ A", scale=1, size="sm")
                    drift_btn_stop = gr.Button("⏹ STOP", variant="stop", scale=1, size="sm")
                    drift_btn_d = gr.Button("▶ D", scale=1, size="sm")
                drift_manual_status = gr.Textbox(label="조작 로그", value="—", interactive=False, lines=1)

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

        for _btn, _dir in [
            (drift_btn_q, "Q"), (drift_btn_w, "W"), (drift_btn_e, "E"),
            (drift_btn_a, "A"), (drift_btn_stop, "STOP"), (drift_btn_d, "D"),
        ]:
            _btn.click(fn=handle_control, inputs=[gr.State(_dir), manual_speed_slider], outputs=drift_manual_status)

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

      # ────────────────────────────────────────────────────────────────
      with gr.Tab("🧪 경로 검증 (Path Test)"):
        with gr.Row(equal_height=False, elem_id="tab4-root"):

          # ── Col 1 (scale=2): 카메라 + 모니터링 ───────────────────────
          with gr.Column(scale=2):
            camera_output_test = gr.Image(label="📷 Live Camera", interactive=False)
            with gr.Row():
              status_log_test        = gr.Textbox(label="Status",  value="Ready",   scale=3, max_lines=1)
              latency_val_test       = gr.Textbox(label="Latency", value="0 ms",    scale=1, max_lines=1)
              action_val_test        = gr.Textbox(label="Action",  value="0,0,0",   scale=2, max_lines=1)
              bbox_area_display_test = gr.Textbox(label="area/cx", value="—",       scale=2, max_lines=1, interactive=False)
              run_status_test        = gr.Textbox(label="Run",     value="Stopped", scale=1, max_lines=1, interactive=False)
            # 그라운딩 상세 행
            with gr.Row():
              gnd_entity_test  = gr.Textbox(label="🔍 대상 entity",  value="—", scale=3, max_lines=1, interactive=False)
              gnd_cached_test  = gr.Textbox(label="캐시",             value="—", scale=1, max_lines=1, interactive=False)
              gnd_bbox_test    = gr.Textbox(label="bbox (cx,cy,area)", value="—", scale=3, max_lines=1, interactive=False)
              pred_label_test  = gr.Textbox(label="예측 레이블",       value="—", scale=2, max_lines=1, interactive=False)
            # 서버 로그 행
            with gr.Row():
              srv_status_t4 = gr.Textbox(
                  label="🖥️ 서버 상태", value="—",
                  interactive=False, max_lines=2, scale=2,
              )
              srv_log_t4 = gr.Textbox(
                  label="📋 추론 서버 로그 (스크롤 가능)", value="—",
                  interactive=False, lines=8, max_lines=200, scale=5,
              )

          # ── 에피소드 CSV 프리로드 (UI 빌드 시점) ─────────────────────────
          def _preload_episode_csv_early() -> list:
              import csv as _pcsv2
              _ep_csv2 = PROJECT_ROOT / "logs" / "episode_log.csv"
              if not _ep_csv2.exists():
                  return []
              rows = []
              with open(_ep_csv2, newline="", encoding="utf-8") as _pf2:
                  for row in _pcsv2.reader(_pf2):
                      if not row or row[0] == "#":
                          continue
                      while len(row) < 13:
                          row.append("")
                      try:
                          row[0]  = int(row[0])   if row[0]  else 0
                          row[3]  = int(row[3])   if row[3]  else 0
                          row[4]  = float(row[4]) if row[4]  else 0.0
                          row[6]  = float(row[6]) if row[6]  else 0.0
                          row[7]  = float(row[7]) if row[7]  else 0.0
                          row[8]  = float(row[8]) if row[8]  else 0.0
                          row[10] = float(row[10]) if row[10] else 0.0
                      except Exception:
                          pass
                      rows.append(row[:13])
              return rows

          def _build_summary_early(log_list):
              done_total = {k: 0 for k in PATH_TYPES}
              done_succ  = {k: 0 for k in PATH_TYPES}
              nav_succ = 0
              for r in log_list:
                  pt = str(r[1]).replace(" ★", "").replace("★", "").strip()
                  done_total[pt] = done_total.get(pt, 0) + 1
                  if r[2] == "성공":
                      done_succ[pt] = done_succ.get(pt, 0) + 1
                      if pt in PATH_TARGETS and not pt.startswith(("obj_", "dist_")):
                          nav_succ += 1
              nav_total = sum(PATH_TARGETS[k] for k in PATH_TARGETS
                              if not k.startswith(("obj_", "dist_")))
              obj_done  = sum(done_total.get(k, 0) for k in ("obj_left","obj_center","obj_right"))
              obj_succ  = sum(done_succ.get(k, 0)  for k in ("obj_left","obj_center","obj_right"))
              dist_done = sum(done_total.get(k, 0) for k in ("dist_10cm","dist_20cm","dist_30cm"))
              dist_succ = sum(done_succ.get(k, 0)  for k in ("dist_10cm","dist_20cm","dist_30cm"))
              prog = (f"경로검증 {sum(done_total.get(k,0) for k in PATH_TYPES if not k.startswith(('obj_','dist_')))}/{nav_total}"
                      f"  성공 {nav_succ}/{GOAL_SUCCESS_TARGET}"
                      f"  |  위치별 {obj_done}/90 ({obj_succ}성공)"
                      f"  |  거리별 {dist_done}/30 ({dist_succ}성공)")
              tbl = []
              for header, keys in _PATH_GROUPS:
                  tbl.append([header, "", "", ""])
                  for pt in keys:
                      tbl.append([pt + (" ★" if pt == "right_left" else ""),
                                  PATH_TARGETS.get(pt, 1),
                                  done_total.get(pt, 0),
                                  done_succ.get(pt, 0)])
              return prog, tbl

          _preloaded_rows_early = _preload_episode_csv_early()
          _preloaded_prog, _preloaded_tbl = _build_summary_early(_preloaded_rows_early)

          # ── Col 2 (scale=1): 경로표 + 기록 ───────────────────────────
          with gr.Column(scale=1):
            with gr.Accordion("📋 경로 다이어그램", open=False):
              gr.Markdown(
                  "```\n"
                  " ▦      ▦      ▦\n"
                  "╱│╲    ╱│╲    ╱│╲\n"
                  "L S R  C S L  R S L\n"
                  " 🤖L    🤖C    🤖R\n"
                  "```\n"
                  "L=left, C=center, R=right 시작위치\n"
                  "방향: L(좌)/S(직)/R(우)  ★=right\\_left 우선"
              )
            progress_test = gr.Textbox(
                label="진행률", value=_preloaded_prog,
                interactive=False, max_lines=1, elem_id="t4-progress",
            )
            path_summary_table = gr.Dataframe(
                headers=["경로/테스트", "목표", "완료", "✓"],
                datatype=["str","str","str","str"],
                value=_preloaded_tbl,
                label="집계 (경로검증 | 오브젝트위치별 | 박스거리별)",
                row_count=18, col_count=4, interactive=False,
            )
            with gr.Accordion("📋 프롬프트 전문 보기", open=False):
              with gr.Row():
                btn_show_prompt = gr.Button("🔄 현재 프롬프트 조회", size="sm", scale=1)
              prompt_detail_box = gr.Textbox(
                  label="",
                  value="위 버튼을 눌러 확인",
                  interactive=False,
                  lines=10,
                  max_lines=20,
              )
              gr.Markdown(
                  "<small>💡 경로명(right_left 등)은 **시작위치_목표방향**의 실험 분류 레이블입니다. "
                  "실제 이동경로는 오브젝트 위치·카메라 시점·모델 학습 상태에 따라 달라집니다.</small>"
              )
            with gr.Row():
              path_type_test = gr.Dropdown(choices=PATH_TYPES, value="obj_center", label="테스트 레이블", scale=3)
              success_test   = gr.Radio(choices=["성공", "실패"], value="성공", label="결과", scale=2)
            with gr.Row():
              fpe_test  = gr.Number(minimum=0.0, maximum=9.9, step=0.05, value=0.0, label="FPE (m)", precision=2, scale=2)
              note_test = gr.Textbox(label="메모", value="", scale=4, max_lines=1)
            # FPE 프리셋 빠른 입력
            with gr.Row():
              _fpe_b = [gr.Button(v, size="sm", scale=1) for v in ["0.0","0.3","0.5","0.8","1.0","1.5","2.0","3.0"]]
            btn_log_episode   = gr.Button("📝 에피소드 기록 추가", variant="primary",   size="lg")
            with gr.Row():
              btn_undo_episode  = gr.Button("↩ 마지막 기록 취소 (1건 삭제)",  variant="secondary", scale=1, size="lg")
              btn_clear_episode = gr.Button("🗑 전체 기록 영구 삭제", variant="stop",     scale=1, size="lg")
            with gr.Row():
              btn_refresh_log    = gr.Button("🔄 CSV복원", scale=1, size="sm")
              btn_export_test    = gr.Button("💾 CSV저장", scale=1, size="sm")
              export_status_test = gr.Textbox(label="", value="", interactive=False, scale=2, max_lines=1)
            with gr.Accordion("✏️ 에피소드 수정", open=False):
              edit_ep_dd      = gr.Dropdown(choices=[], label="수정할 에피소드 선택", interactive=True)
              with gr.Row():
                edit_path_dd  = gr.Dropdown(choices=PATH_TYPES, label="경로", scale=3)
                edit_succ_r   = gr.Radio(choices=["성공", "실패"], label="결과", scale=2)
              with gr.Row():
                edit_fpe_n    = gr.Number(minimum=0.0, maximum=9.9, step=0.05, value=0.0,
                                          label="FPE (m)", precision=2, scale=2)
                edit_note_b   = gr.Textbox(label="메모", value="", scale=4, max_lines=1)
              with gr.Row():
                btn_edit_ep   = gr.Button("💾 수정 저장", variant="primary", size="lg", scale=2)
                edit_status   = gr.Textbox(label="", value="", interactive=False, scale=3, max_lines=1)

          # ── Col 3 (scale=1): 제어 + 에피소드 로그 ────────────────────
          with gr.Column(scale=1):
            with gr.Group():
              mode_radio_test = gr.Radio(
                  choices=["Manual Drive", "Inference (Auto)"],
                  value="Manual Drive", label="🎮 Mode",
              )
              with gr.Column(visible=False) as inference_panel_test:
                infer_move_radio_test = gr.Radio(choices=["SYNC", "PRE", "ASYNC"], value="SYNC", label="이동 모드")
                with gr.Row():
                  btn_start_test  = gr.Button("▶️ START", variant="primary",   scale=1)
                  btn_stop_test   = gr.Button("⏹️ STOP",  variant="stop",      scale=1)
                  btn_return_test = gr.Button("🔄 복귀",  variant="secondary", scale=1)

            def on_mode_change_test(selected_mode):
                return gr.update(visible=selected_mode == "Inference (Auto)")
            mode_radio_test.change(fn=on_mode_change_test, inputs=[mode_radio_test], outputs=[inference_panel_test])

            with gr.Row():
              with gr.Column(scale=1):
                gr.Markdown("**🎮 Manual**")
                t4_speed_slider = gr.Slider(minimum=0.3, maximum=2.0, step=0.05, value=1.15, label="속도")
                with gr.Row():
                  t4_btn_q = gr.Button("↖Q", scale=1, size="sm")
                  t4_btn_w = gr.Button("▲W", scale=1, size="sm")
                  t4_btn_e = gr.Button("↗E", scale=1, size="sm")
                with gr.Row():
                  t4_btn_a = gr.Button("◀A", scale=1, size="sm")
                  t4_btn_stop = gr.Button("⏹", variant="stop", scale=1, size="sm")
                  t4_btn_d = gr.Button("▶D", scale=1, size="sm")
                with gr.Row():
                  t4_btn_r = gr.Button("↺R", scale=1, size="sm")
                  t4_btn_s = gr.Button("▼S", scale=1, size="sm")
                  t4_btn_t = gr.Button("↻T", scale=1, size="sm")
              with gr.Column(scale=1):
                gr.Markdown("**🕹️ Joystick**")
                js_status_test = gr.Textbox(label="상태", value="🔌 초기화 중...", interactive=False)
                gr.Markdown("<small>Left → 이동\nRight X → 회전\nA → STOP</small>")

            ep_view_radio = gr.Radio(
                choices=["전체", "정상만", "🚨 이상치만"],
                value="전체", label="필터", interactive=True,
            )
            episode_log_table = gr.Dataframe(
                headers=["#", "경로", "결과", "steps", "lat(ms)", "top액션", "gnd%", "area", "cx", "STOP", "FPE", "메모", "날짜"],
                datatype=["number","str","str","number","number","str","number","number","number","str","number","str","str"],
                label="에피소드 기록 (누적 — 세션 간 유지)",
                value=_preloaded_rows_early if _preloaded_rows_early else None,
                row_count=11, col_count=13, interactive=False,
            )
            outlier_panel = gr.Dataframe(
                headers=["#", "경로", "결과", "lat(ms)", "steps", "이상치 유형", "원인"],
                datatype=["number","str","str","number","number","str","str"],
                label="🚨 이상치 분석",
                value=None, row_count=4, col_count=7, interactive=False,
                visible=False,
            )

        _episode_log_state = gr.State(_preloaded_rows_early)

      # ════════════════════════════════════════════════════════════════════
      # 탭 5: STOP 캘리브레이션
      # ════════════════════════════════════════════════════════════════════
      with gr.Tab("🔧 STOP 캘리브레이션"):
        with gr.Row(equal_height=False, elem_id="tab5-root"):

          # ── 왼쪽: 카메라 + 게이지 ────────────────────────────────────
          with gr.Column(scale=3):
            camera_output_calib = gr.Image(label="📷 Live Camera", interactive=False)
            with gr.Row():
              calib_area_disp = gr.Textbox(label="bbox area",  value="—",      scale=2, max_lines=1, interactive=False)
              calib_cx_disp   = gr.Textbox(label="cx",         value="—",      scale=1, max_lines=1, interactive=False)
              calib_stop_disp = gr.Textbox(label="STOP 상태",  value="—",      scale=2, max_lines=1, interactive=False)
              calib_rec_disp  = gr.Textbox(label="녹화",        value="■ 정지", scale=1, max_lines=1, interactive=False)

            calib_gauge_md = gr.Markdown("_(area 게이지 — 카메라 앞으로 접근하면서 변화 확인)_")

            gr.Markdown("---\n**📸 수동 조작 (Tab 4와 동일 조이스틱 사용)**")
            with gr.Row():
              c5_btn_q = gr.Button("↖Q", scale=1, size="sm")
              c5_btn_w = gr.Button("▲W", scale=1, size="sm")
              c5_btn_e = gr.Button("↗E", scale=1, size="sm")
            with gr.Row():
              c5_btn_a = gr.Button("◀A", scale=1, size="sm")
              c5_btn_stop = gr.Button("⏹", variant="stop", scale=1, size="sm")
              c5_btn_d = gr.Button("▶D", scale=1, size="sm")
            with gr.Row():
              c5_btn_r = gr.Button("↺R", scale=1, size="sm")
              c5_btn_s = gr.Button("▼S", scale=1, size="sm")
              c5_btn_t = gr.Button("↻T", scale=1, size="sm")
            c5_speed_slider = gr.Slider(minimum=0.3, maximum=2.0, step=0.05, value=1.15, label="속도")

          # ── 오른쪽: 임계값 + 세션 + 데이터 ──────────────────────────
          with gr.Column(scale=2):
            gr.Markdown("### 🎯 STOP 임계값 설정")
            with gr.Row():
              calib_thr_slider = gr.Slider(
                  minimum=0.05, maximum=0.50, step=0.005, value=0.18,
                  label="area 임계값 (서버 기본 0.18)", scale=4,
              )
              calib_apply_thr_btn = gr.Button("서버 적용", scale=1)
            calib_thr_status = gr.Textbox(label="적용 결과", value="", interactive=False, max_lines=1)

            gr.Markdown("---\n### 🎬 캘리브레이션 세션 녹화")
            calib_session_name = gr.Textbox(
                label="세션명 (비워두면 자동)", value="", placeholder="calib_YYYYMMDD_HHMMSS",
                max_lines=1,
            )
            with gr.Row():
              calib_start_rec_btn = gr.Button("⏺ 녹화 시작", variant="primary",  scale=2)
              calib_stop_rec_btn  = gr.Button("⏹ 녹화 중지", variant="stop",      scale=2)
              calib_snap_btn      = gr.Button("📸 스냅",      variant="secondary", scale=1)
            with gr.Row():
              calib_clear_btn = gr.Button("🗑 초기화",  scale=1)
              calib_save_btn  = gr.Button("💾 저장",    variant="primary", scale=1)
            calib_rec_status = gr.Textbox(label="", value="준비", interactive=False, max_lines=1)

            calib_data_table = gr.Dataframe(
                headers=["n", "area", "cx", "cy", "lat(ms)", "STOP?", "시각", "메모"],
                datatype=["number","number","number","number","number","str","str","str"],
                label="캡처 데이터 (최근 20개)",
                interactive=False, row_count=8,
            )

            calib_recommend_md = gr.Markdown("_(데이터 캡처 후 추천 임계값 표시)_")
            gr.Markdown(
                "**📂 8083 뷰어에서 세션 확인**\n\n"
                "`logs/calib_sessions/` → 8083 뷰어 「세션 로그」 탭 → Calib 섹션"
            )

      # ════════════════════════════════════════════════════════════════════
      # 탭 6: 세션 히스토리 + 셀프 라벨링 + 이상치 검증
      # ════════════════════════════════════════════════════════════════════
      with gr.Tab("📚 세션 히스토리"):
        _INFER_REPORT_DIR_T6 = PROJECT_ROOT / "docs" / "inference_reports"
        _INFER_H5_DIR_T6     = PROJECT_ROOT / "docs" / "inference_sessions"
        _LABEL_JSON_PATH     = Path("/tmp/mona_preview_labels.json")

        # ── 라벨 파일 I/O ───────────────────────────────────────────────
        def _t6_load_labels() -> dict:
            import json as _jl
            if _LABEL_JSON_PATH.exists():
                try: return _jl.loads(_LABEL_JSON_PATH.read_text())
                except Exception: pass
            return {}

        def _t6_save_labels(labels: dict):
            import json as _jl
            _LABEL_JSON_PATH.write_text(_jl.dumps(labels, indent=2, ensure_ascii=False))

        def _t6_frame_key(sid, idx):
            return f"session_{sid}_f{idx}"

        # ── 이상치 감지 ─────────────────────────────────────────────────
        def _t6_check_anomaly(m: dict, is_old_session: bool, is_prev: bool = False, is_arrival: bool = False) -> list:
            """프레임 메타에서 이상치 판단. 경고 문자열 리스트 반환."""
            warns = []
            ca, lat, has, cx, area = m["cached"], m["latency_ms"], m["has_bbox"], m["cx"], m["area"]
            # 1a. preview ROT인데 latency=0 → 6/30 이전 기록 버그
            if is_prev and lat == 0 and is_old_session:
                warns.append("⚠️ preview latency=0ms (6/30 이전 기록 버그)")
            # 1b. preview ROT인데 latency=0 → 신규 세션에서도 0이면 버그
            if is_prev and lat == 0 and not is_old_session:
                warns.append("⚠️ preview latency=0ms (신규 세션 — 기록 누락)")
            # 2. live/preview PG2인데 latency=0 (arrival/cached/init은 제외)
            if ca == 0.0 and lat == 0 and not is_prev and not is_arrival:
                warns.append("⚠️ live PG2 latency=0ms (기록 버그)")
            # 3. has_bbox=True인데 cx=0.5 (fallback 값 — 실제 탐지 아님)
            if has and abs(cx - 0.5) < 0.001:
                warns.append("⚠️ has_bbox=True지만 cx=0.5 (fallback 의심)")
            # 4. has_bbox=True인데 area=0
            if has and area == 0:
                warns.append("⚠️ has_bbox=True지만 area=0")
            # 5. has_bbox=False인데 cx≠0.5 (모순)
            if not has and abs(cx - 0.5) > 0.01:
                warns.append(f"⚠️ has_bbox=False인데 cx={cx:.3f} (모순)")
            return warns

        def _t6_label_cx_check(user_label: str, cx: float, has_bbox: bool) -> str:
            """사용자 라벨 vs PG2 cx 일관성 검증."""
            if not has_bbox or user_label == "NONE":
                return ""
            if user_label == "L" and cx > 0.55:
                return f"⚠️ 라벨=L이지만 cx={cx:.3f} (오른쪽에 있음)"
            if user_label == "R" and cx < 0.45:
                return f"⚠️ 라벨=R이지만 cx={cx:.3f} (왼쪽에 있음)"
            if user_label == "C" and (cx < 0.35 or cx > 0.65):
                return f"⚠️ 라벨=C이지만 cx={cx:.3f} (중앙 아님)"
            return "✅ 라벨↔cx 일관"

        # ── 세션 목록 ───────────────────────────────────────────────────
        def _t6_list_sessions():
            import glob as _gl, json as _js2, os as _os2
            h5_files  = sorted(_gl.glob(str(_INFER_H5_DIR_T6 / "session_*.h5")), reverse=True)
            json_files = {
                _os2.path.splitext(_os2.path.basename(f))[0].replace("session_", ""): f
                for f in _gl.glob(str(_INFER_REPORT_DIR_T6 / "session_*.json"))
            }
            labels = _t6_load_labels()
            choices = []
            for h5p in h5_files:
                sid = _os2.path.basename(h5p).replace("session_", "").replace(".h5", "")
                n_labeled = sum(1 for k in labels if k.startswith(f"session_{sid}_f"))
                tag = f"[{n_labeled}라벨]" if n_labeled else ""
                label = f"[{sid}] {tag}"
                if sid in json_files:
                    try:
                        d = _js2.load(open(json_files[sid]))
                        steps = d.get("summary", {}).get("total_steps", "?")
                        label += f"  {steps}steps  {d.get('instruction','')[:20]}"
                    except Exception:
                        pass
                choices.append((label, sid))
            return choices

        # ── 프레임 렌더 ─────────────────────────────────────────────────
        def _t6_draw_frame(img_arr, cx, cy, area, has_bbox, frame_type, user_label, warns):
            from PIL import Image as _PIL5, ImageDraw as _IDraw
            import numpy as _np5
            H, W = img_arr.shape[:2]
            pil = _PIL5.fromarray(img_arr.astype(_np5.uint8)).convert("RGB")
            pil = pil.resize((W // 2, H // 2))
            W2, H2 = pil.size
            draw = _IDraw.Draw(pil)

            # bbox 사각형
            if has_bbox and area > 0 and abs(cx - 0.5) > 0.001:
                px, py = cx * W2, cy * H2
                half = (area ** 0.5) * min(W2, H2) * 0.5
                x0, y0 = max(0, px - half), max(0, py - half)
                x1, y1 = min(W2, px + half), min(H2, py + half)
                col = (0, 255, 80) if not warns else (255, 180, 0)
                draw.rectangle([x0, y0, x1, y1], outline=col, width=3)
                draw.ellipse([px-5, py-5, px+5, py+5], fill=col)

            # 상단 배너
            if frame_type == "🔄PREVIEW":
                draw.rectangle([0, 0, W2, 36], fill=(160, 0, 0))
                draw.text((8, 8), "🔄 PREVIEW", fill=(255, 220, 80))
            elif frame_type == "★ARRIVAL":
                draw.rectangle([0, 0, W2, 36], fill=(0, 110, 0))
                draw.text((8, 8), "★ ARRIVAL", fill=(255, 255, 255))
            elif warns:
                draw.rectangle([0, 0, W2, 36], fill=(160, 100, 0))
                draw.text((8, 8), "⚠️ ANOMALY", fill=(255, 240, 80))

            # 라벨 표시 (우상단)
            if user_label:
                lc = {"L": (80,160,255), "R": (80,160,255), "C": (80,255,120), "NONE": (160,160,160)}
                draw.rectangle([W2-60, 0, W2, 36], fill=lc.get(user_label, (100,100,100)))
                draw.text((W2-52, 8), user_label, fill=(0, 0, 0))

            return pil

        # ── H5 로드 ─────────────────────────────────────────────────────
        def _t6_load_h5(sid):
            import h5py as _h5, numpy as _np6
            if not sid:
                return [], [], "_(선택하세요)_", [], {}
            h5p = _INFER_H5_DIR_T6 / f"session_{sid}.h5"
            if not h5p.exists():
                return [], [], f"❌ H5 없음: {h5p}", [], {}

            with _h5.File(h5p) as f:
                imgs   = f["observations/images"][()]
                acts   = f["actions"][()]
                bbox   = f["grounding/bbox"][()]
                cached = f["grounding/cached"][()]
                lats   = f["grounding/latency_ms"][()]
                attrs  = dict(f.attrs)

            n = len(imgs)
            # 구버전 세션 판단 (6/30 이전 = latency 버그 있음)
            is_old = sid < "20260630"

            _amap = {
                (0.0,0.0,0.0):"STOP", (1.15,0.0,0.0):"FWD",
                (0.0,1.15,0.0):"LEFT", (0.0,-1.15,0.0):"RIGHT",
                (1.15,1.15,0.0):"FWD+L", (1.15,-1.15,0.0):"FWD+R",
                (0.0,0.0,0.25):"ROT_L", (0.0,0.0,-0.25):"ROT_R",
            }
            def _lbl(a):
                for k, v in _amap.items():
                    if all(abs(float(a[i])-k[i])<0.05 for i in range(3)): return v
                return f"({a[0]:.1f},{a[1]:.1f})"

            labels = _t6_load_labels()
            frames_pil, meta_list, table_rows = [], [], []
            n_anomaly = 0

            for i in range(n):
                cx, cy, area, has = float(bbox[i,0]), float(bbox[i,1]), float(bbox[i,2]), bool(bbox[i,3])
                ca  = float(cached[i])
                lat = float(lats[i])
                # preview ROT: grounding_cached=False → ca=0.0, action=ROT_L/ROT_R
                # arrival: last frame, ca=-1 (no grounding), action=STOP
                # initial STOP: first frame, ca=-1, action=STOP (before first inference)
                is_arrival = (i == n-1 and ca == -1.0 and _lbl(acts[i]) == "STOP")
                is_prev    = (ca == 0.0 and _lbl(acts[i]) in ("ROT_L", "ROT_R"))
                ftype = ("★ARRIVAL" if is_arrival else
                         "🔄PREVIEW" if is_prev else
                         "📡live" if ca == 0.0 else "💾cache")

                warns  = _t6_check_anomaly({"cached":ca,"latency_ms":lat,"has_bbox":has,"cx":cx,"area":area}, is_old, is_prev, is_arrival)
                if warns: n_anomaly += 1

                ulabel = labels.get(_t6_frame_key(sid, i), "")
                pil    = _t6_draw_frame(imgs[i], cx, cy, area, has, ftype, ulabel, warns)
                frames_pil.append(pil)

                anomaly_tag = "⚠️" if warns else "✅"
                meta_list.append({
                    "idx": i, "sid": sid, "action": _lbl(acts[i]),
                    "cx": cx, "cy": cy, "area": area, "has_bbox": has,
                    "latency_ms": lat, "cached": ca, "type": ftype,
                    "warns": warns, "user_label": ulabel,
                })
                table_rows.append([
                    i, ftype, _lbl(acts[i]),
                    round(lat, 0), round(area, 4), round(cx, 3),
                    "✅" if has else "—", anomaly_tag,
                    ulabel or "—",
                ])

            n_preview = sum(1 for m in meta_list if "PREVIEW" in m["type"])
            n_live    = sum(1 for m in meta_list if m["type"] == "📡live")
            live_lats = [m["latency_ms"] for m in meta_list if m["type"] == "📡live" and m["latency_ms"] > 0]
            bbox_rate = sum(1 for m in meta_list if m["has_bbox"]) / max(1, n)
            dist = {}
            for k in labels:
                if k.startswith(f"session_{sid}_f"):
                    dist[labels[k]] = dist.get(labels[k], 0) + 1
            dist_str = "  ".join(f"{k}:{v}" for k,v in sorted(dist.items())) or "라벨 없음"

            summary_md = (
                f"**{sid}**  |  {n}프레임  |  `{attrs.get('instruction','?')}`  |  `{attrs.get('status','?')}`\n\n"
                f"🔄preview:{n_preview}  📡live:{n_live}  "
                f"{'💾cache:'+str(n-n_preview-n_live)+'  ' if n-n_preview-n_live>0 else ''}"
                f"has_bbox:{bbox_rate:.0%}  "
                + (f"live평균:{sum(live_lats)/len(live_lats):.0f}ms  " if live_lats else "")
                + f"⚠️이상:{n_anomaly}건\n\n"
                f"라벨 분포: {dist_str}"
            )
            return frames_pil, meta_list, summary_md, table_rows, labels

        # ── 프레임 표시 + 이상치 패널 ───────────────────────────────────
        def _t6_show_frame(frames_pil, meta_list, labels, sid, idx):
            if not frames_pil or idx is None:
                return None, "—", "—"
            idx = int(idx)
            m   = meta_list[idx]
            ul  = labels.get(_t6_frame_key(sid, idx), "")
            cx_check = _t6_label_cx_check(ul, m["cx"], m["has_bbox"]) if ul else ""

            info = (
                f"**frame {idx}/{len(frames_pil)-1}**  |  {m['type']}  |  "
                f"**{m['action']}**  |  "
                f"has_bbox:{'✅' if m['has_bbox'] else '❌'}  "
                f"cx:{m['cx']:.3f}  area:{m['area']:.4f}  "
                f"lat:{m['latency_ms']:.0f}ms"
                + (f"  |  라벨: **{ul}**" if ul else "  |  라벨: 미지정")
            )
            warns_md = ""
            if m["warns"]:
                warns_md += "**이상치 경고:**\n" + "\n".join(f"- {w}" for w in m["warns"])
            if cx_check:
                warns_md += ("\n\n" if warns_md else "") + cx_check
            if not warns_md:
                warns_md = "✅ 정상"
            return frames_pil[idx], info, warns_md

        def _t6_on_load(sid):
            frames_pil, meta_list, summary_md, table_rows, labels = _t6_load_h5(sid)
            n = len(frames_pil)
            img, info, warns = _t6_show_frame(frames_pil, meta_list, labels, sid, 0) if n > 0 else (None, "", "")
            slider_up = gr.update(maximum=max(0, n-1), value=0, visible=n > 0)
            return frames_pil, meta_list, labels, sid, summary_md, table_rows, img, info, warns, slider_up

        # ── 라벨 저장 ───────────────────────────────────────────────────
        def _t6_set_label(frames_pil, meta_list, labels, sid, idx, user_lbl):
            if not frames_pil or not sid:
                return frames_pil, meta_list, labels, "저장 실패", "—", "—"
            idx = int(idx)
            key = _t6_frame_key(sid, idx)
            labels = dict(labels)
            labels[key] = user_lbl
            _t6_save_labels(labels)
            # 프레임 다시 렌더 (라벨 표시 반영)
            m = meta_list[idx]
            new_pil = _t6_draw_frame(
                # 원본 이미지는 없으니 기존 pil에서 라벨만 재렌더 → _t6_load_h5 재호출 없이
                # 간단히 현재 pil 위에 라벨 오버레이
                frames_pil[idx], m["cx"], m["cy"], m["area"],
                m["has_bbox"], m["type"], user_lbl, m["warns"]
            )
            # PIL input은 numpy array 필요 → pil을 numpy로 변환 후 재렌더
            import numpy as _npL
            new_pil = _t6_draw_frame(
                _npL.array(frames_pil[idx].convert("RGB")),
                m["cx"], m["cy"], m["area"], m["has_bbox"], m["type"], user_lbl, m["warns"]
            )
            frames_pil = list(frames_pil)
            frames_pil[idx] = new_pil
            meta_list = list(meta_list)
            meta_list[idx] = {**m, "user_label": user_lbl}
            img, info, warns_md = _t6_show_frame(frames_pil, meta_list, labels, sid, idx)

            dist = {}
            for k in labels:
                if k.startswith(f"session_{sid}_f"):
                    dist[labels[k]] = dist.get(labels[k], 0) + 1
            save_msg = f"💾 저장됨: {key}={user_lbl}  |  분포: " + " ".join(f"{k}:{v}" for k,v in sorted(dist.items()))
            return frames_pil, meta_list, labels, save_msg, img, warns_md

        # ── UI ──────────────────────────────────────────────────────────
        with gr.Row():
            t6_session_dd  = gr.Dropdown(choices=_t6_list_sessions(), value=None,
                                         label="세션 선택", scale=7)
            t6_refresh_btn = gr.Button("🔄", scale=1)
            t6_load_btn    = gr.Button("불러오기", scale=1, variant="primary")

        t6_summary_md = gr.Markdown("_(세션을 선택하고 불러오기)_")

        with gr.Row(equal_height=False):
            # 왼쪽: 이미지 + 탐색 + 라벨 버튼
            with gr.Column(scale=3):
                t6_frame_img   = gr.Image(label="프레임", type="pil", height=360)
                t6_frame_info  = gr.Markdown("—")
                t6_warns_md    = gr.Markdown("—")
                with gr.Row():
                    t6_prev_btn     = gr.Button("◀", scale=1)
                    t6_frame_slider = gr.Slider(minimum=0, maximum=0, step=1,
                                                value=0, label="frame", scale=5)
                    t6_next_btn     = gr.Button("▶", scale=1)
                gr.Markdown("**셀프 라벨링** — basket 위치:")
                with gr.Row():
                    t6_lbl_L    = gr.Button("⬅ L (왼쪽)", scale=1)
                    t6_lbl_C    = gr.Button("⬆ C (중앙)", scale=1, variant="primary")
                    t6_lbl_R    = gr.Button("➡ R (오른쪽)", scale=1)
                    t6_lbl_NONE = gr.Button("✕ NONE", scale=1, variant="stop")
                t6_save_status = gr.Markdown("—")

            # 오른쪽: 테이블 (이상치 + 라벨 컬럼 포함)
            with gr.Column(scale=2):
                t6_table = gr.Dataframe(
                    headers=["f","type","action","lat(ms)","area","cx","bbox","⚠️","label"],
                    datatype=["number","str","str","number","number","number","str","str","str"],
                    label="스텝별 (⚠️=이상치  label=셀프라벨)",
                    interactive=False, row_count=12,
                )

        # state
        t6_state_frames = gr.State([])
        t6_state_meta   = gr.State([])
        t6_state_labels = gr.State({})
        t6_state_sid    = gr.State("")

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

    # 현재 추론서버(stage2_v2)의 로그 파일 — 기동 방식에 따라 다를 수 있음
    _SRV_LOG_CANDIDATES = [
        PROJECT_ROOT / "logs" / "s2v2_server.log",
        PROJECT_ROOT / "logs" / "api_server.log",
        PROJECT_ROOT / "logs" / "inference_server_8001.log",
    ]

    def _srv_status_str():
        """서버 /health → 상태 문자열."""
        try:
            import requests as _rq2
            h = _rq2.get(f"{DEFAULT_API_URL}/health", timeout=1.5).json()
            loaded = "✅ 로드됨" if h.get("model_loaded") else "❌ 모델 없음"
            gpu = h.get("gpu", {})
            return (
                f"{loaded}  head={h.get('head','?')}  win={h.get('window','?')}\n"
                f"stop={h.get('stop_mode','?')}  GPU {gpu.get('allocated_gb',0):.2f}GB [{gpu.get('device_name','?')}]"
            )
        except Exception as _e:
            return f"⚠️ 서버 응답 없음 ({_e})"

    def _srv_log_lines(n: int | None = None):
        """서버 로그에서 [#N] 예측 라인 추출. n=None이면 전체."""
        try:
            log_file = next((p for p in _SRV_LOG_CANDIDATES if p.exists()), None)
            if not log_file:
                return "(로그 파일 없음)"
            # 전체 파일 읽기 (최대 300KB)
            with open(log_file, "rb") as _lf:
                _lf.seek(0, 2)
                size = _lf.tell()
                _lf.seek(max(0, size - 300_000))
                tail = _lf.read().decode("utf-8", errors="replace")
            pred_lines = [l.split("INFO:__main__:")[-1]
                          for l in tail.splitlines()
                          if "__main__" in l and "[#" in l]
            if pred_lines:
                return "\n".join(pred_lines if n is None else pred_lines[-n:])
            # 예측 없으면 시작 메시지 전체에서
            with open(log_file, "rb") as _lf2:
                full = _lf2.read().decode("utf-8", errors="replace")
            init_lines = [l.split("INFO:__main__:")[-1]
                          for l in full.splitlines() if "__main__" in l]
            return "\n".join(init_lines) if init_lines else "(추론 로그 없음 — 추론 실행 후 표시)"
        except Exception:
            return "(로그 읽기 실패)"

    def _server_info_tick():
        """Tab 1용: 상태 + 최근 4줄."""
        return _srv_status_str(), _srv_log_lines(4)

    def _server_info_tick_full():
        """Tab 4용: 상태 + 전체 히스토리 (스크롤 가능)."""
        return _srv_status_str(), _srv_log_lines(None)

    _ui_inputs = [mode_radio, backend_radio, api_url_box, instr_box_real, toggle_cc, run_status_box, infer_move_radio]
    _ui_outputs = [camera_output, status_log, latency_val, action_val, chunk_val, run_status_box, camera_status, model_path, traj_plot]

    timer = gr.Timer(0.5, active=True)
    timer.tick(fn=_update_ui_and_cache, inputs=_ui_inputs, outputs=_ui_outputs)
    timer.tick(fn=_get_bbox_area_display, outputs=bbox_area_display)
    timer.tick(fn=_js_status_text, outputs=js_status)
    timer.tick(fn=_run_history_rows, outputs=run_history_table)
    timer.tick(fn=_server_info_tick, outputs=[srv_status_t1, srv_log_t1])
    # 환경 배너 10초마다 갱신 (API 서버 상태 실시간 반영)
    _banner_timer.tick(fn=_make_env_banner, outputs=_env_banner)
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

    # ── 탭4: 경로 검증(Path Test) — 탭1 state/함수 재사용, 탭1 코드 무수정 ──
    def _t4_camera_tick():
        """Tab 4 카메라: 격자 + bbox 오버레이 적용 (Tab 1과 동일)."""
        img = state.get("last_img")
        if img is None:
            return None
        try:
            import requests as _rq_cam
            preds = _rq_cam.get(f"{DEFAULT_API_URL}/recent", timeout=0.8).json().get("predictions", [])
            if preds:
                return annotate_image(img, preds[0].get("bbox"))
        except Exception:
            pass
        return annotate_image(img)  # 격자만 (bbox 없음)

    timer.tick(fn=_t4_camera_tick, outputs=camera_output_test)
    timer.tick(fn=lambda: state.get("_t4_log", "Ready"), outputs=status_log_test)
    timer.tick(fn=lambda: state.get("_t4_lat", "0 ms"), outputs=latency_val_test)
    timer.tick(fn=lambda: state.get("_t4_act", "0,0,0"), outputs=action_val_test)
    timer.tick(fn=_get_bbox_area_display, outputs=bbox_area_display_test)
    timer.tick(fn=_js_status_text, outputs=js_status_test)

    def _gnd_detail_tick():
        """그라운딩 상세 정보 (entity, cached/preview상태, bbox, 예측레이블) 반환."""
        try:
            import requests as _rq
            data = _rq.get(f"{DEFAULT_API_URL}/recent", timeout=1.5).json()
            preds = data.get("predictions", [])
            preview_enabled = data.get("preview_enabled", False)
            preview_attempt = data.get("preview_attempt", 0)
            preview_max     = data.get("preview_max_retry", 5)
            inf_count       = data.get("inference_count", -1)

            # preview 중이면 cached 필드를 preview 상태로 대체
            if preview_enabled and inf_count == 0 and preview_attempt > 0:
                cached_str = f"🔄 PREVIEW {preview_attempt}/{preview_max}"
            elif preview_enabled and inf_count == 0:
                cached_str = "⏳ PREVIEW 대기"
            elif preview_enabled:
                cached_str = "✅ preview 완료"
            else:
                cached_str = "—"

            if preds:
                p = preds[0]
                cx   = p.get("cx",   0.5)
                area = p.get("area", 0.0)
                has  = p.get("has_bbox", False)
                entity = "gray basket"
                label  = "—"
                return entity, cached_str, f"cx={cx:.3f}  area={area:.4f}  {'✓BBOX' if has else '✗NONE'}", label
            return "—", cached_str, "—", "—"
        except Exception:
            pass
        return "—", "—", "—", "—"

    timer.tick(fn=_gnd_detail_tick,
               outputs=[gnd_entity_test, gnd_cached_test, gnd_bbox_test, pred_label_test])
    timer.tick(fn=_server_info_tick_full, outputs=[srv_status_t4, srv_log_t4])

    btn_start_test.click(
        fn=lambda mode, url, instr, gt, cc: set_running(True, mode, url, instr, gt, apply_cc=cc),
        inputs=[backend_radio, api_url_box, instr_box_real, gt_object_box, toggle_cc],
        outputs=run_status_test,
    )
    btn_stop_test.click(fn=lambda: set_running(False, "", "", ""), outputs=run_status_test)
    btn_return_test.click(fn=return_to_start, outputs=run_status_test)

    t4_directions = {
        t4_btn_w: "W", t4_btn_s: "S", t4_btn_a: "A", t4_btn_d: "D",
        t4_btn_q: "Q", t4_btn_e: "E", t4_btn_r: "R", t4_btn_t: "T",
        t4_btn_stop: "STOP",
    }
    for _btn, _dir in t4_directions.items():
        _btn.click(fn=handle_control, inputs=[gr.State(_dir), t4_speed_slider], outputs=status_log_test)

    t4_speed_slider.change(fn=_sync_js_speed, inputs=t4_speed_slider)

    # ── 누적 에피소드 로그 영구 저장 경로 ─────────────────────────────
    _EPISODE_CSV = PROJECT_ROOT / "logs" / "episode_log.csv"
    _EP_HEADERS  = ["#", "경로", "결과", "steps", "lat(ms)", "top액션", "gnd%", "area", "cx", "STOP", "FPE", "메모", "날짜"]

    def _load_episode_csv() -> list:
        import csv as _csv
        if not _EPISODE_CSV.exists():
            return []
        rows = []
        with open(_EPISODE_CSV, newline="", encoding="utf-8") as f:
            reader = _csv.reader(f)
            next(reader, None)  # 헤더 건너뜀
            for row in reader:
                if not row:
                    continue
                # 숫자 컬럼 캐스팅 (호환성: 컬럼 수 부족한 구버전 CSV 패딩)
                while len(row) < 13:
                    row.append("")
                try:
                    row[0]  = int(row[0])   if row[0]  else 0
                    row[3]  = int(row[3])   if row[3]  else 0
                    row[4]  = float(row[4]) if row[4]  else 0.0
                    row[6]  = float(row[6]) if row[6]  else 0.0
                    row[7]  = float(row[7]) if row[7]  else 0.0
                    row[8]  = float(row[8]) if row[8]  else 0.0
                    row[10] = float(row[10]) if row[10] else 0.0
                except Exception:
                    pass
                rows.append(row[:13])
        return rows

    def _append_episode_csv(row: list):
        import csv as _csv
        _EPISODE_CSV.parent.mkdir(parents=True, exist_ok=True)
        write_header = not _EPISODE_CSV.exists()
        with open(_EPISODE_CSV, "a", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            if write_header:
                w.writerow(_EP_HEADERS)
            w.writerow(row)  # row already includes 날짜 as last column

    def _overwrite_episode_csv(rows: list):
        import csv as _csv
        _EPISODE_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(_EPISODE_CSV, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(_EP_HEADERS)
            w.writerows(rows)

    def _build_summary(log_list):
        """log_list → (prog_str, path_summary_rows with group headers). 공통 로직."""
        done_total = {k: 0 for k in PATH_TYPES}
        done_succ  = {k: 0 for k in PATH_TYPES}
        nav_succ = 0
        for r in log_list:
            pt = str(r[1]).replace(" ★", "").replace("★", "").strip()
            done_total[pt] = done_total.get(pt, 0) + 1
            if r[2] == "성공":
                done_succ[pt] = done_succ.get(pt, 0) + 1
                if not pt.startswith(("obj_", "dist_")):
                    nav_succ += 1
        nav_total = sum(PATH_TARGETS[k] for k in PATH_TARGETS
                        if not k.startswith(("obj_", "dist_")))
        obj_done  = sum(done_total.get(k, 0) for k in ("obj_left","obj_center","obj_right"))
        obj_succ  = sum(done_succ.get(k, 0)  for k in ("obj_left","obj_center","obj_right"))
        dist_done = sum(done_total.get(k, 0) for k in ("dist_10cm","dist_20cm","dist_30cm"))
        dist_succ = sum(done_succ.get(k, 0)  for k in ("dist_10cm","dist_20cm","dist_30cm"))
        prog = (f"경로검증 {sum(done_total.get(k,0) for k in PATH_TYPES if not k.startswith(('obj_','dist_')))}/{nav_total}"
                f"  성공 {nav_succ}/{GOAL_SUCCESS_TARGET}"
                f"  |  위치별 {obj_done}/90 ({obj_succ}성공)"
                f"  |  거리별 {dist_done}/30 ({dist_succ}성공)")
        tbl = []
        for header, keys in _PATH_GROUPS:
            tbl.append([header, "", "", ""])
            for pt in keys:
                tbl.append([pt + (" ★" if pt == "right_left" else ""),
                             PATH_TARGETS.get(pt, 1),
                             done_total.get(pt, 0),
                             done_succ.get(pt, 0)])
        return prog, tbl

    def _init_episode_log():
        rows = _load_episode_csv()
        # 번호 재정렬
        for i, r in enumerate(rows):
            r[0] = i + 1
        prog, tbl = _build_summary(rows)
        return rows, rows, prog, tbl

    def log_episode(path_type, success, fpe, note, log_list):
        import requests as _req

        # 1. bbox
        area, cx = 0.0, 0.5
        try:
            r = _req.get(f"{DEFAULT_API_URL}/recent", timeout=2)
            preds = r.json().get("predictions", [])
            if preds:
                bbox = preds[0].get("bbox", {})
                area, cx = bbox.get("area", 0.0), bbox.get("cx", 0.5)
        except Exception:
            pass
        stop_flag = "Y" if area >= 0.18 else "N"

        # 2. inference_logger 세션 통계
        steps, avg_lat, top_action, gnd_pct = 0, 0.0, "—", 0.0
        try:
            if logger_instance and hasattr(logger_instance, "data") and logger_instance.data:
                hist = logger_instance.data.get("history", [])
                steps = len(hist)
                lats = [h["latency_ms"] for h in hist if isinstance(h.get("latency_ms"), (int, float))]
                avg_lat = round(sum(lats) / len(lats), 1) if lats else 0.0
                labels = [h.get("predicted_label") for h in hist if h.get("predicted_label")]
                if labels:
                    from collections import Counter
                    top_action = Counter(labels).most_common(1)[0][0]
                gnd_ok = sum(1 for h in hist if h.get("bbox") and h["bbox"].get("has_bbox"))
                gnd_pct = round(gnd_ok / steps * 100, 1) if steps > 0 else 0.0
        except Exception:
            pass

        import datetime as _dt_ep
        row = [
            len(log_list) + 1, path_type, success,
            steps, avg_lat, top_action, gnd_pct,
            round(area, 3), round(cx, 2), stop_flag, fpe, note,
            _dt_ep.datetime.now().strftime("%Y-%m-%d %H:%M"),
        ]
        _append_episode_csv(row)
        log_list = log_list + [row]
        prog, tbl = _build_summary(log_list)
        return log_list, log_list, prog, tbl

    # FPE 프리셋 버튼 핸들러
    for _fb, _fv in zip(_fpe_b, [0.0, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]):
        _fb.click(fn=lambda v=_fv: v, outputs=fpe_test)

    btn_log_episode.click(
        fn=log_episode,
        inputs=[path_type_test, success_test, fpe_test, note_test, _episode_log_state],
        outputs=[_episode_log_state, episode_log_table, progress_test, path_summary_table],
    )

    def export_episode_log(log_list):
        import csv as _csv2
        import datetime as _dt2
        out_path = PROJECT_ROOT / "logs" / f"realtest_{_dt2.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = _csv2.writer(f)
            w.writerow(_EP_HEADERS)
            w.writerows(log_list)
        return f"✅ 저장: {out_path}"

    btn_export_test.click(fn=export_episode_log, inputs=[_episode_log_state], outputs=export_status_test)

    def _refresh_and_reset_filter():
        rows, rows2, prog, tbl = _init_episode_log()
        return rows, rows2, prog, tbl, gr.update(value="전체"), gr.update(visible=False, value=None)

    btn_refresh_log.click(
        fn=_refresh_and_reset_filter,
        outputs=[_episode_log_state, episode_log_table, progress_test, path_summary_table,
                 ep_view_radio, outlier_panel],
    )

    # ── 에피소드 수정 ─────────────────────────────────────────────────────

    def _ep_choices(log_list):
        return [f"#{r[0]}  {r[1]}  {r[2]}  FPE={r[10]}" for r in log_list]

    # State 변경될 때마다 드롭다운 자동 갱신
    _episode_log_state.change(
        fn=lambda rows: gr.update(choices=_ep_choices(rows), value=None),
        inputs=_episode_log_state,
        outputs=edit_ep_dd,
    )

    def _on_edit_select(choice, log_list):
        """에피소드 선택 → 필드 자동 채움."""
        if not choice or not log_list:
            return gr.update(), gr.update(), gr.update(), gr.update()
        try:
            ep_num = int(choice.split()[0].lstrip("#"))
        except Exception:
            return gr.update(), gr.update(), gr.update(), gr.update()
        for r in log_list:
            if r[0] == ep_num:
                return (
                    gr.update(value=str(r[1])),
                    gr.update(value=str(r[2])),
                    gr.update(value=float(r[10]) if r[10] != "" else 0.0),
                    gr.update(value=str(r[11]) if len(r) > 11 else ""),
                )
        return gr.update(), gr.update(), gr.update(), gr.update()

    edit_ep_dd.change(
        fn=_on_edit_select,
        inputs=[edit_ep_dd, _episode_log_state],
        outputs=[edit_path_dd, edit_succ_r, edit_fpe_n, edit_note_b],
    )

    def _save_edit(choice, new_path, new_succ, new_fpe, new_note, log_list):
        """에피소드 필드 수정 후 CSV 덮어쓰기."""
        if not choice or not log_list:
            return log_list, log_list, *_build_summary(log_list), "❌ 선택 없음"
        try:
            ep_num = int(choice.split()[0].lstrip("#"))
        except Exception:
            return log_list, log_list, *_build_summary(log_list), "❌ 파싱 오류"
        new_list = [r[:] for r in log_list]
        modified = False
        for r in new_list:
            if r[0] == ep_num:
                r[1]  = new_path
                r[2]  = new_succ
                r[10] = float(new_fpe)
                if len(r) > 11:
                    r[11] = new_note
                modified = True
                break
        if not modified:
            return log_list, log_list, *_build_summary(log_list), f"❌ #{ep_num} 없음"
        _overwrite_episode_csv(new_list)
        prog, tbl = _build_summary(new_list)
        return new_list, new_list, prog, tbl, f"✅ #{ep_num} 수정 완료"

    btn_edit_ep.click(
        fn=_save_edit,
        inputs=[edit_ep_dd, edit_path_dd, edit_succ_r, edit_fpe_n, edit_note_b, _episode_log_state],
        outputs=[_episode_log_state, episode_log_table, progress_test, path_summary_table, edit_status],
    )

    def _build_prompt_detail(instr, gt_obj):
        """프롬프트 전문 + 서버 호출 구조 설명 생성."""
        instr = (instr or "").strip()
        gt_obj = (gt_obj or "").strip()
        # 서버 내부 phrase 변환 (stage2_v2_inference_server.py line 564)
        phrase = "gray basket" if instr == "basket" else instr
        mapping_note = '  ⚠ "basket" → "gray basket" 자동 매핑 적용됨\n' if instr == "basket" else ""

        # /recent에서 마지막 예측 정보 가져오기
        recent_note = ""
        try:
            import requests as _rp
            preds = _rp.get(f"{DEFAULT_API_URL}/recent", timeout=1.5).json().get("predictions", [])
            if preds:
                p = preds[0]
                recent_note = (
                    f"\n📡 마지막 추론 결과 (/recent):\n"
                    f"  predicted_label : {p.get('predicted_label', '?')}\n"
                    f"  latency_ms      : {p.get('latency_ms', '?')}\n"
                    f"  grounding_cached: {p.get('grounding_cached', '?')}\n"
                    f"  bbox entity     : {p.get('bbox', {}).get('entity', '?')}\n"
                    f"  bbox cx/cy/area : {p.get('bbox', {}).get('cx', '?'):.3f} / "
                    f"{p.get('bbox', {}).get('cy', '?'):.3f} / "
                    f"{p.get('bbox', {}).get('area', '?'):.4f}"
                )
        except Exception:
            pass

        return (
            f"━━ POST /predict (stage2_v2_inference_server) ━━\n"
            f"{{\n"
            f'  "instruction": "{instr}",\n'
            f'  "image": "<base64 PNG — 현재 카메라 프레임>"\n'
            f"}}\n\n"
            f"━━ 그라운딩 (PaliGemma2 bbox 검출) ━━\n"
            f"  phrase: \"{phrase}\"\n"
            f"{mapping_note}"
            f"  → 모델이 이 객체를 화면에서 찾아 cx/cy/area 추출\n\n"
            f"━━ GT Object (로깅 전용 — 모델 미전달) ━━\n"
            f"  \"{gt_obj}\"\n\n"
            f"━━ 경로 레이블 안내 ━━\n"
            f"  right_left / center_straight 등 경로명은\n"
            f"  [시작위치]_[목표방향] 실험 분류 레이블입니다.\n"
            f"  실제 이동경로는 아래 요소에 따라 달라집니다:\n"
            f"   • 오브젝트 실제 위치 (수동 배치)\n"
            f"   • 카메라 시점 / 조명 조건\n"
            f"   • 모델 학습 수준 (Exp14 Step2 기준)\n"
            f"   • 그라운딩 bbox 정확도"
            f"{recent_note}"
        )

    btn_show_prompt.click(
        fn=_build_prompt_detail,
        inputs=[instr_box_real, gt_object_box],
        outputs=prompt_detail_box,
    )

    def undo_episode(log_list):
        new_list = log_list[:-1] if log_list else []
        # 번호 재정렬 후 CSV 덮어쓰기
        for i, r in enumerate(new_list):
            r[0] = i + 1
        _overwrite_episode_csv(new_list)
        prog, tbl = _build_summary(new_list)
        return new_list, new_list, prog, tbl

    def clear_episodes(_log_list):
        if _EPISODE_CSV.exists():
            _EPISODE_CSV.unlink()
        prog, tbl = _build_summary([])
        return [], [], prog, tbl

    btn_undo_episode.click(
        fn=undo_episode,
        inputs=[_episode_log_state],
        outputs=[_episode_log_state, episode_log_table, progress_test, path_summary_table],
    )
    btn_clear_episode.click(
        fn=clear_episodes,
        inputs=[_episode_log_state],
        outputs=[_episode_log_state, episode_log_table, progress_test, path_summary_table],
        js="""
        (log_list) => {
            if (!confirm("⚠️ 정말로 모든 에피소드 기록(CSV 파일)을 영구 삭제하시겠습니까?\\n이 작업은 되돌릴 수 없으며 복구가 불가능합니다.")) {
                throw new Error("사용자가 초기화 작업을 취소했습니다.");
            }
            return log_list;
        }
        """
    )

    # 페이지 열릴 때 과거 기록 복원
    demo.load(
        fn=_init_episode_log,
        outputs=[_episode_log_state, episode_log_table, progress_test, path_summary_table],
    )

    # ── 이상치 분류 필터 ────────────────────────────────────────────────────
    _LAT_HIGH_MS  = 3000.0   # PG2 cold-start 판정 임계값
    _LAT_ZERO_MS  = 0.0

    def _classify_outlier(row):
        """row[4]=lat(ms), row[3]=steps, row[5]=top액션.
        반환: (is_outlier: bool, 유형: str, 원인: str)"""
        try:
            lat   = float(row[4])
            steps = int(row[3])
            top   = str(row[5])
        except Exception:
            return False, "", ""
        if lat >= _LAT_HIGH_MS:
            return True, "high-latency", f"PG2 cold-start — 서버 첫 추론 시 3B 모델 GPU warmup ({lat:.0f}ms)"
        if lat == _LAT_ZERO_MS and steps <= 1 and top in ("—", "", "None"):
            return True, "zero-latency", "추론 미실행 상태에서 LOG 버튼 누름 (logger 비어있음 → avg_lat=0)"
        return False, "", ""

    def _apply_ep_filter(log_list, view):
        if view == "🚨 이상치만":
            rows = [r for r in log_list if _classify_outlier(r)[0]]
            outlier_rows = [[r[0], r[1], r[2], r[4], r[3],
                             _classify_outlier(r)[1], _classify_outlier(r)[2]] for r in rows]
            return rows, gr.update(visible=True, value=outlier_rows or None)
        elif view == "정상만":
            rows = [r for r in log_list if not _classify_outlier(r)[0]]
            return rows, gr.update(visible=False, value=None)
        else:
            return log_list, gr.update(visible=False, value=None)

    ep_view_radio.change(
        fn=_apply_ep_filter,
        inputs=[_episode_log_state, ep_view_radio],
        outputs=[episode_log_table, outlier_panel],
    )


    # ── 탭5: STOP 캘리브레이션 ────────────────────────────────────────────
    _CALIB_DIR = PROJECT_ROOT / "logs" / "calib_sessions"

    timer.tick(fn=lambda: state.get("last_img"), outputs=camera_output_calib)

    def _calib_tick():
        import requests as _req5
        import datetime as _dt5
        area, cx, cy, latency, has_bbox = 0.0, 0.5, 0.5, 0.0, False
        try:
            r = _req5.get(f"{DEFAULT_API_URL}/recent", timeout=1.5)
            preds = r.json().get("predictions", [])
            if preds:
                p = preds[0]
                bbox = p.get("bbox", {})
                area    = bbox.get("area", 0.0)
                cx      = bbox.get("cx",   0.5)
                cy      = bbox.get("cy",   0.5)
                latency = p.get("latency_ms", 0.0)
                has_bbox = bbox.get("has_bbox", False)
        except Exception:
            pass

        thr = float(state.get("calib_thr", 0.18))
        stop_triggered = has_bbox and area >= thr and abs(cx - 0.5) <= 0.25

        # 자동 녹화: 2틱(~1s)마다 1 샘플
        state["_calib_tick_n"] = state.get("_calib_tick_n", 0) + 1
        if state.get("calib_recording") and state["_calib_tick_n"] % 2 == 0:
            frames = state.get("calib_frames", [])
            n = len(frames) + 1
            frames.append({
                "n": n, "area": round(area, 4), "cx": round(cx, 3),
                "cy": round(cy, 3), "latency_ms": round(latency, 1),
                "stop": "Y" if stop_triggered else "N",
                "ts":   _dt5.datetime.now().strftime("%H:%M:%S"),
                "note": "",
            })
            state["calib_frames"] = frames
            # 카메라 프레임 함께 저장 (최대 120장)
            img = state.get("last_img")
            if img is not None:
                imgs = state.get("calib_imgs", [])
                if len(imgs) < 120:
                    imgs.append(img)
                state["calib_imgs"] = imgs

        # 표시값
        area_s = f"{area:.4f}" if has_bbox else "—"
        cx_s   = f"{cx:.3f}"   if has_bbox else "—"
        stop_s = ("🔴 STOP!" if stop_triggered
                  else ("🟡 근접" if area >= thr * 0.7 and has_bbox else "🟢 이동 중"))
        rec_s  = "⏺ 녹화 중" if state.get("calib_recording") else "■ 정지"

        # 게이지
        MAX_A = 0.40
        bar_n = min(40, int(area / MAX_A * 40))
        thr_n = min(39, int(thr   / MAX_A * 40))
        bar   = ["█" if i < bar_n else " " for i in range(40)]
        # 임계 마커
        bar[thr_n] = "┃"
        gauge = (
            f"```\n0.0 {''.join(bar)} 0.4\n"
            f"    {' '*thr_n}↑{thr:.3f}  현재:{area:.4f}\n```"
        )

        # 데이터 테이블 (최근 20)
        frames = state.get("calib_frames", [])
        table = [
            [f["n"], f["area"], f["cx"], f["cy"], f["latency_ms"], f["stop"], f["ts"], f["note"]]
            for f in frames[-20:]
        ]

        # 추천 임계값
        valid = [f for f in frames if f["area"] > 0.03]
        if len(valid) >= 5:
            areas = sorted(f["area"] for f in valid)
            p85 = areas[int(len(areas) * 0.85)]
            rec = (
                f"**추천 임계값: `{p85:.3f}`**  "
                f"(캡처 {len(valid)}개 기준, 85퍼센타일)\n\n"
                f"현재 설정: `{thr:.3f}`  |  "
                f"min: {min(areas):.4f}  max: {max(areas):.4f}"
            )
        else:
            rec = f"_(5개 이상 필요 — 현재 {len(frames)}개)_"

        return area_s, cx_s, stop_s, rec_s, gauge, table, rec

    timer.tick(
        fn=_calib_tick,
        outputs=[calib_area_disp, calib_cx_disp, calib_stop_disp,
                 calib_rec_disp, calib_gauge_md, calib_data_table, calib_recommend_md],
    )

    def calib_start(session_name):
        import datetime as _dt5
        state["calib_recording"] = True
        state["calib_frames"]    = []
        state["calib_imgs"]      = []
        state["_calib_tick_n"]   = 0
        fname = session_name.strip() or f"calib_{_dt5.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        state["calib_session_name"] = fname
        return f"⏺ 녹화 시작 — {fname}"

    def calib_stop_recording():
        state["calib_recording"] = False
        n = len(state.get("calib_frames", []))
        return f"■ 정지 ({n}개 캡처됨)"

    def calib_snap():
        import requests as _req5
        import datetime as _dt5
        area, cx, cy, latency, has_bbox = 0.0, 0.5, 0.5, 0.0, False
        try:
            r = _req5.get(f"{DEFAULT_API_URL}/recent", timeout=2)
            preds = r.json().get("predictions", [])
            if preds:
                p = preds[0]
                bbox = p.get("bbox", {})
                area    = bbox.get("area", 0.0)
                cx      = bbox.get("cx",   0.5)
                cy      = bbox.get("cy",   0.5)
                latency = p.get("latency_ms", 0.0)
                has_bbox = bbox.get("has_bbox", False)
        except Exception:
            pass
        thr  = float(state.get("calib_thr", 0.18))
        stop = has_bbox and area >= thr and abs(cx - 0.5) <= 0.25
        frames = state.get("calib_frames", [])
        n = len(frames) + 1
        frames.append({
            "n": n, "area": round(area, 4), "cx": round(cx, 3),
            "cy": round(cy, 3), "latency_ms": round(latency, 1),
            "stop": "Y" if stop else "N",
            "ts":   _dt5.datetime.now().strftime("%H:%M:%S"),
            "note": "manual",
        })
        state["calib_frames"] = frames
        # 이미지 캡처 (최대 120장)
        img = state.get("last_img")
        if img is not None:
            imgs = state.get("calib_imgs", [])
            if len(imgs) < 120:
                imgs.append(img)
            state["calib_imgs"] = imgs
        return f"📸 {n}번 스냅 (area={area:.4f})"

    def calib_clear():
        state["calib_frames"]    = []
        state["calib_imgs"]      = []
        state["calib_recording"] = False
        return "🗑 초기화 완료", [], "_(데이터 없음)_"

    def calib_apply_threshold(thr):
        import requests as _req5
        state["calib_thr"] = float(thr)
        try:
            r = _req5.post(
                f"{DEFAULT_API_URL}/set_stop_threshold",
                json={"stop_area_threshold": float(thr), "stop_cx_tolerance": 0.25},
                timeout=3,
            )
            return f"✅ 서버 적용: area≥{thr:.3f}" if r.status_code == 200 else f"⚠️ {r.status_code} (로컬 적용만)"
        except Exception as e:
            return f"⚠️ 서버 오류: {e} (로컬 적용만)"

    def calib_save(session_name):
        import json as _json5
        import datetime as _dt5
        import cv2 as _cv5
        import numpy as _np5
        frames = state.get("calib_frames", [])
        if not frames:
            return "⚠️ 저장할 데이터 없음"
        base = (session_name.strip() or state.get("calib_session_name")
                or f"calib_{_dt5.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        base = base.replace(".jsonl", "")
        _CALIB_DIR.mkdir(parents=True, exist_ok=True)

        # 1. JSONL
        jsonl_path = _CALIB_DIR / (base + ".jsonl")
        with open(jsonl_path, "w") as f:
            for frm in frames:
                _json5.dump({
                    "n":              frm["n"],
                    "ts":             frm["ts"],
                    "has_bbox":       frm["area"] > 0.01,
                    "area":           frm["area"],
                    "cx":             frm["cx"],
                    "cy":             frm["cy"],
                    "latency_ms":     frm["latency_ms"],
                    "pred_label":     frm["note"],
                    "stop_triggered": frm["stop"] == "Y",
                }, f)
                f.write("\n")

        # 2. 캡처 이미지 → MP4
        imgs = state.get("calib_imgs", [])
        mp4_msg = ""
        if imgs:
            try:
                mp4_path = _CALIB_DIR / (base + ".mp4")
                arr0 = _np5.array(imgs[0], dtype=_np5.uint8)
                h, w = arr0.shape[:2]
                writer = _cv5.VideoWriter(
                    str(mp4_path),
                    _cv5.VideoWriter_fourcc(*"mp4v"),
                    2, (w, h),
                )
                for img in imgs:
                    writer.write(_cv5.cvtColor(_np5.array(img, dtype=_np5.uint8), _cv5.COLOR_RGB2BGR))
                writer.release()
                mp4_msg = f" + MP4({len(imgs)}프레임)"
            except Exception as e:
                mp4_msg = f" (MP4 실패: {e})"

        return f"✅ {jsonl_path.name} ({len(frames)}개){mp4_msg}"

    calib_start_rec_btn.click(fn=calib_start, inputs=calib_session_name, outputs=calib_rec_status)
    calib_stop_rec_btn.click(fn=calib_stop_recording, outputs=calib_rec_status)
    calib_snap_btn.click(fn=calib_snap, outputs=calib_rec_status)
    calib_clear_btn.click(fn=calib_clear, outputs=[calib_rec_status, calib_data_table, calib_recommend_md])
    calib_save_btn.click(fn=calib_save, inputs=calib_session_name, outputs=calib_rec_status)
    calib_apply_thr_btn.click(fn=calib_apply_threshold, inputs=calib_thr_slider, outputs=calib_thr_status)

    c5_directions = {
        c5_btn_w: "W", c5_btn_s: "S", c5_btn_a: "A", c5_btn_d: "D",
        c5_btn_q: "Q", c5_btn_e: "E", c5_btn_r: "R", c5_btn_t: "T",
        c5_btn_stop: "STOP",
    }
    for _c5b, _c5d in c5_directions.items():
        _c5b.click(fn=handle_control, inputs=[gr.State(_c5d), c5_speed_slider], outputs=calib_rec_status)

    c5_speed_slider.change(fn=_sync_js_speed, inputs=c5_speed_slider)

    # ── 탭6: 세션 히스토리 + 셀프 라벨링 ────────────────────────────────
    def _t6_refresh_list():
        return gr.update(choices=_t6_list_sessions(), value=None)

    t6_refresh_btn.click(fn=_t6_refresh_list, outputs=t6_session_dd)

    _T6_LOAD_OUT = [t6_state_frames, t6_state_meta, t6_state_labels, t6_state_sid,
                    t6_summary_md, t6_table, t6_frame_img, t6_frame_info,
                    t6_warns_md, t6_frame_slider]

    t6_load_btn.click(fn=_t6_on_load, inputs=t6_session_dd, outputs=_T6_LOAD_OUT)

    def _t6_on_slide(frames, meta, labels, sid, idx):
        img, info, warns = _t6_show_frame(frames, meta, labels, sid, idx)
        return img, info, warns

    t6_frame_slider.change(
        fn=_t6_on_slide,
        inputs=[t6_state_frames, t6_state_meta, t6_state_labels, t6_state_sid, t6_frame_slider],
        outputs=[t6_frame_img, t6_frame_info, t6_warns_md],
    )

    def _t6_nav(frames, meta, labels, sid, idx, delta):
        new_idx = max(0, min(len(frames)-1, int(idx)+delta)) if frames else 0
        img, info, warns = _t6_show_frame(frames, meta, labels, sid, new_idx)
        return img, info, warns, gr.update(value=new_idx)

    t6_prev_btn.click(
        fn=lambda f,m,l,s,i: _t6_nav(f,m,l,s,i,-1),
        inputs=[t6_state_frames, t6_state_meta, t6_state_labels, t6_state_sid, t6_frame_slider],
        outputs=[t6_frame_img, t6_frame_info, t6_warns_md, t6_frame_slider],
    )
    t6_next_btn.click(
        fn=lambda f,m,l,s,i: _t6_nav(f,m,l,s,i,+1),
        inputs=[t6_state_frames, t6_state_meta, t6_state_labels, t6_state_sid, t6_frame_slider],
        outputs=[t6_frame_img, t6_frame_info, t6_warns_md, t6_frame_slider],
    )

    # 라벨 버튼 공통 핸들러
    _T6_LBL_IN  = [t6_state_frames, t6_state_meta, t6_state_labels,
                   t6_state_sid, t6_frame_slider]
    _T6_LBL_OUT = [t6_state_frames, t6_state_meta, t6_state_labels,
                   t6_save_status, t6_frame_img, t6_warns_md]

    for _btn, _lbl in [(t6_lbl_L,"L"),(t6_lbl_C,"C"),(t6_lbl_R,"R"),(t6_lbl_NONE,"NONE")]:
        _btn.click(
            fn=lambda f,m,l,s,i,lbl=_lbl: _t6_set_label(f,m,l,s,i,lbl),
            inputs=_T6_LBL_IN,
            outputs=_T6_LBL_OUT,
        )

    # ── 대시보드 재시작 (맨 아래 고정) ───────────────────────────────────
    gr.Markdown("---\n### 🔄 대시보드 재시작  _(코드 업데이트 후 현재 프로세스 종료 → 새 버전으로 즉시 재기동)_")
    with gr.Row():
        restart_btn    = gr.Button("🔄 지금 재시작", variant="stop", scale=1, min_width=140)
        restart_status = gr.Textbox(label="", value="", interactive=False, scale=4, max_lines=1)

    def do_restart():
        import subprocess as _sp2
        import sys as _sys2
        import threading as _th2
        def _exec():
            import time as _t2
            _t2.sleep(1.5)
            _sp2.Popen(
                [_sys2.executable] + _sys2.argv,
                close_fds=True,
                start_new_session=True,
            )
            _sys2.exit(0)
        _th2.Thread(target=_exec, daemon=True).start()
        return "⏳ 1.5s 후 재시작... 페이지를 새로고침하세요"

    restart_btn.click(fn=do_restart, outputs=restart_status)


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

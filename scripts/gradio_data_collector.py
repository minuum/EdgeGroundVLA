import gradio as gr
import os
import sys
import time
import threading
import numpy as np
import cv2
import h5py
import json
from datetime import datetime
from PIL import Image
from collections import defaultdict
from pathlib import Path
import socket

try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# --- Forced ROS2 Environment Overrides ---
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("ROS_HOME", "/tmp/ros")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["ROS_HOME"]).mkdir(parents=True, exist_ok=True)
os.environ["ROS_DOMAIN_ID"] = "42"
os.environ["RMW_IMPLEMENTATION"] = "rmw_fastrtps_cpp"

# Add ROS Workspace to Path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
ros_ws_path = os.path.join(_PROJECT_ROOT, "ROS_action/install/camera_interfaces/lib/python3.10/site-packages")
if os.path.exists(ros_ws_path) and ros_ws_path not in sys.path:
    sys.path.append(ros_ws_path)

sys.path.insert(0, str(_PROJECT_ROOT))
from scripts.utils.camera_proc import camera_control_widget, start_camera, stop_camera

def load_env():
    env_path = os.path.join(_PROJECT_ROOT, ".vla_env_settings")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip().replace("export ", "", 1)
                if "=" in line:
                    try:
                        k, v = line.split("=", 1)
                        os.environ[k] = v.strip('"').strip("'")
                    except ValueError: continue
load_env()

# --- Hardware Setup (pop.driving) ---
try:
    from pop.driving import Driving
    ROBOT_HW_AVAILABLE = True
except ImportError:
    ROBOT_HW_AVAILABLE = False

# --- ROS2 Setup ---
ROS_IMPORT_ERROR = ""
try:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import Twist
    from cv_bridge import CvBridge
    from camera_interfaces.srv import GetImage
    ROS_AVAILABLE = True
except ImportError as e:
    print(f"CRITICAL: ROS2 IMPORT ERROR -> {e}")
    ROS_IMPORT_ERROR = str(e)
    ROS_AVAILABLE = False
    class Node:
        def __init__(self, *args, **kwargs):
            pass
        def create_client(self, *args, **kwargs):
            class DummyClient:
                def service_is_ready(self): return False
            return DummyClient()
        def create_publisher(self, *args, **kwargs):
            class DummyPublisher:
                def publish(self, msg): pass
            return DummyPublisher()
    class CvBridge:
        def imgmsg_to_cv2(self, imgmsg, desired_encoding='bgr8'):
            return imgmsg
    class Twist:
        def __init__(self):
            class Vector3:
                def __init__(self):
                    self.x, self.y, self.z = 0.0, 0.0, 0.0
            self.linear = Vector3()
            self.angular = Vector3()
    class GetImage:
        class Request:
            pass

# ---------------------------------------------------------------------------
# Capture Mode
# ---------------------------------------------------------------------------
import enum

class CaptureMode(enum.Enum):
    PRE_CACHE  = "pre_cache"   # 주 모드: 액션 직전 캐시 스냅샷  (비블로킹 <1 ms)
    POST_SYNC  = "post_sync"   # 보조 모드: 액션 직후 ROS 서비스 콜 (블로킹 최대 300 ms)


# ---------------------------------------------------------------------------
# Joystick Reader
# ---------------------------------------------------------------------------
class JoystickReader:
    """DragonRise 게임패드를 비동기로 읽어 node.teleop_step()을 호출한다.
    기존 ROS/녹화/H5 로직은 전혀 수정하지 않는다."""

    DEADZONE   = 0.15   # 스틱 노이즈 무시 범위
    THRESHOLD  = 0.50   # bang-bang 판정 임계값
    STEP_INTERVAL = 0.45  # 홀딩 시 반복 발사 간격 (s) — 기존 0.4s 펄스와 맞춤

    # 기본 축 매핑 (calibrate_joystick.py로 확인 후 joystick_config.json 덮어씀)
    DEFAULT_AXES = {"left_x": 0, "left_y": 1, "right_x": 2}

    # 버튼 인덱스 (DragonRise 기본값, 캘리브레이션으로 확정)
    BTN_STOP   = 0   # A  — STOP 명시적 1프레임
    BTN_UNDO   = 1   # B  — 마지막 프레임 취소
    BTN_START  = 7   # Start — teleop_mode 토글
    BTN_SELECT = 6   # Select — 녹화 시작/저장
    BTN_DISCARD = 2  # X  — 에피소드 폐기

    def __init__(self, node):
        self._node = node
        self._running = False
        self._thread = None
        self._btn_prev = {}
        self._last_step_time = 0.0
        self._prev_key = None
        self._axes = self._load_axes()
        self._neutral_start_time = 0.0
        self._last_non_neutral_key = None

        # Gradio 상태 표시용 (lock-free read 허용 — 단순 dict 교체)
        self.status = {
            "connected": False, "name": "—",
            "lx": 0.0, "ly": 0.0, "az": 0.0,
            "key": None, "label": "—",
        }

    # ------------------------------------------------------------------ #
    def _load_axes(self):
        cfg_path = Path(__file__).parent / "joystick_config.json"
        if cfg_path.exists():
            try:
                with open(cfg_path) as f:
                    return json.load(f).get("axes", self.DEFAULT_AXES)
            except Exception:
                pass
        return dict(self.DEFAULT_AXES)

    def start(self):
        if not PYGAME_AVAILABLE:
            print("[Joystick] pygame 없음 — pip install pygame")
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    # ------------------------------------------------------------------ #
    def _axis_to_key(self, lx, ly, az):
        T = self.THRESHOLD
        fwd = lx >=  T
        bwd = lx <= -T
        lft = ly >=  T
        rgt = ly <= -T
        rl  = az >=  T
        rr  = az <= -T

        # 대각선 우선
        if fwd and lft: return 'q'
        if fwd and rgt: return 'e'
        if bwd and lft: return 'z'
        if bwd and rgt: return 'c'
        # 단축
        if fwd: return 'w'
        if bwd: return 'x'
        if lft: return 'a'
        if rgt: return 'd'
        # 회전
        if rl:  return 'r'
        if rr:  return 't'
        return None

    def _on_btn_down(self, btn):
        nd = self._node
        if btn == self.BTN_STOP:
            nd.teleop_step(' ')
        elif btn == self.BTN_UNDO:
            with nd.lock:
                if nd.episode_buffer:
                    nd.episode_buffer.pop()
        elif btn == self.BTN_START:
            nd.js_mode = 'async' if nd.js_mode == 'sync' else 'sync'
            print(f"[Joystick] 모드 전환 → {nd.js_mode.upper()}")
        elif btn == self.BTN_SELECT:
            with nd.lock:
                collecting = nd.collecting
            if collecting:
                nd.stop_rec(save=True)
            # 시나리오가 선택돼 있을 때만 시작
            elif nd.current_scenario_key:
                nd.start_rec(nd.current_scenario_key)
        elif btn == self.BTN_DISCARD:
            nd.stop_rec(save=False)

    def _loop(self):
        try:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
            os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
            pygame.init()
            pygame.joystick.init()
        except Exception as e:
            print(f"[Joystick] pygame init 실패: {e}")
            return

        js = None
        while self._running:
            # 재연결 대기
            if js is None:
                if pygame.joystick.get_count() == 0:
                    self.status = {**self.status, "connected": False, "name": "—"}
                    pygame.joystick.quit(); pygame.joystick.init()
                    time.sleep(1.0)
                    continue
                js = pygame.joystick.Joystick(0)
                js.init()
                self.status = {**self.status, "connected": True, "name": js.get_name()}
                print(f"[Joystick] 연결됨: {js.get_name()}")
                self._btn_prev = {i: 0 for i in range(js.get_numbuttons())}

            try:
                pygame.event.pump()

                # 축 읽기 (deadzone 적용)
                def rd(axis_idx):
                    v = js.get_axis(axis_idx)
                    return v if abs(v) > self.DEADZONE else 0.0

                lx =  -rd(self._axes["left_y"])   # 위 = +lx
                ly =  -rd(self._axes["left_x"])    # 왼쪽 = +ly
                az =  -rd(self._axes["right_x"])

                raw_key = self._axis_to_key(lx, ly, az)
                key = raw_key

                # ASYNC 모드에서만 300ms Jitter Hold 필터를 적용하여 유령 정지(mid-stop) 방지
                if self._node.js_mode == 'async':
                    now = time.time()
                    if raw_key is not None:
                        self._last_non_neutral_key = raw_key
                        self._neutral_start_time = 0.0
                    else:
                        if self._neutral_start_time == 0.0:
                            self._neutral_start_time = now
                        
                        if now - self._neutral_start_time < 0.30:
                            key = self._last_non_neutral_key
                        else:
                            key = None

                # 모드에 따라 분기
                now = time.time()
                if key:
                    if self._node.js_mode == 'sync':
                        # SYNC: V5 스텝 기반 (0.45s 간격, teleop_step 경유)
                        if (now - self._last_step_time) >= self.STEP_INTERVAL:
                            self._node.teleop_step(key)
                            self._last_step_time = now
                    else:
                        # ASYNC: 10Hz 연속 스무스 드라이브
                        if (now - self._last_step_time) >= 0.10:
                            self._node.joystick_drive(key)
                            self._last_step_time = now
                elif self._prev_key:
                    if self._node.js_mode == 'async':
                        self._node.joystick_drive(None)
                self._prev_key = key

                # 상태 갱신
                labels = {'q':'↖FWD+L','w':'▲FWD','e':'↗FWD+R','a':'←LEFT',
                          'd':'→RIGHT','x':'▼BACK','z':'↙','c':'↘',
                          'r':'↺ROT_L','t':'↻ROT_R'}
                raw = [round(js.get_axis(i), 3) for i in range(js.get_numaxes())]
                self.status = {
                    "connected": True, "name": js.get_name(),
                    "lx": round(lx, 2), "ly": round(ly, 2), "az": round(az, 2),
                    "key": key, "label": labels.get(key, "NEUTRAL") if key else "NEUTRAL",
                    "raw": raw,
                }

                # 버튼 엣지 감지 (누르는 순간만)
                for i in range(js.get_numbuttons()):
                    cur = js.get_button(i)
                    if cur and not self._btn_prev.get(i, 0):
                        self._on_btn_down(i)
                    self._btn_prev[i] = cur

            except Exception as e:
                print(f"[Joystick] 루프 오류 ({e}), 재연결 시도")
                js = None
                self.status = {**self.status, "connected": False}

            time.sleep(0.04)  # 25 Hz


joystick_reader: JoystickReader | None = None  # node 생성 후 초기화


# ---------------------------------------------------------------------------
OFFLINE_TELEOP_LABELS = {
    'q': '↖', 'w': '⬆', 'e': '↗',
    'a': '⬅', 's': 'STOP', 'd': '➡',
    'z': '↙', 'x': '⬇', 'c': '↘',
    't': 'L-Angle', 'r': 'R-Angle', 'g': 'RETURN'
}

# --- V5 Scenarios ---
V5_SCENARIOS = {
    "1": {"id": "target_left_left_path",      "name": "좌측 - 왼쪽 곡선",  "target": 15},
    "2": {"id": "target_left_straight_path",  "name": "좌측 - 직선",       "target": 20},
    "3": {"id": "target_left_right_path",     "name": "좌측 - 오른쪽 곡선","target": 15},
    "4": {"id": "target_center_left_path",    "name": "중앙 - 왼쪽 곡선",  "target": 15},
    "5": {"id": "target_center_straight_path","name": "중앙 - 직선",       "target": 20},
    "6": {"id": "target_center_right_path",   "name": "중앙 - 오른쪽 곡선","target": 15},
    "7": {"id": "target_right_left_path",     "name": "우측 - 왼쪽 곡선",  "target": 15},
    "8": {"id": "target_right_straight_path", "name": "우측 - 직선",       "target": 20},
    "9": {"id": "target_right_right_path",    "name": "우측 - 오른쪽 곡선","target": 15},
    "FL": {"id": "free_left",   "name": "🎲 자유-좌측", "target": 7},
    "FC": {"id": "free_center", "name": "🎲 자유-중앙", "target": 7},
    "FR": {"id": "free_right",  "name": "🎲 자유-우측", "target": 7},
}

DIVERSITY_TAGS = {
    "A-바구니좌극단":  "basket_left_extreme",
    "B-바구니우극단":  "basket_right_extreme",
    "C-로봇근접":      "robot_close",
    "D-로봇원거리":    "robot_far",
    "E-사선좌접근":    "diagonal_left",
    "F-사선우접근":    "diagonal_right",
    "G-조명차이":      "lighting_diff",
}

DATASET_ROOT = os.path.join(_PROJECT_ROOT, "ROS_action/mobile_vla_dataset_v5")
os.makedirs(DATASET_ROOT, exist_ok=True)
CORE_DB_PATH = os.path.join(DATASET_ROOT, "core_replay_db.json")

class GradioCollectorNode(Node):
    def __init__(self):
        super().__init__('gradio_vla_collector_v5')
        self.bridge = CvBridge()
        self.latest_ui_frame = None
        self.collecting = False
        self.teleop_mode = False 
        self.episode_buffer = []
        self.current_scenario_key = None
        self.selected_pattern = "core"
        self.selected_distance = "fixed"
        
        self.img_client = self.create_client(GetImage, 'get_image_service')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.throttle = 50
        self.stop_inject_n = 5
        self.js_mode = 'sync'   # 'sync' = V5 스텝 기반 | 'async' = 10Hz 연속
        self.diversity_tag = list(DIVERSITY_TAGS.keys())[0]
        if ROBOT_HW_AVAILABLE:
            try: self.driver = Driving()
            except: self.driver = None
        else: self.driver = None
        
        self.WASD_TO_CONTINUOUS = {
            'q': (1.15, 1.15, 0.0), 'w': (1.15, 0.0, 0.0), 'e': (1.15, -1.15, 0.0),
            'a': (0.0, 1.15, 0.0), 's': (0.0, 0.0, 0.0), 'd': (0.0, -1.15, 0.0),
            'z': (-1.15, 1.15, 0.0), 'x': (-1.15, 0.0, 0.0), 'c': (-1.15, -1.15, 0.0),
            'r': (0.0, 0.0, 0.20), 't': (0.0, 0.0, -0.20),
            'g': (0.0, 0.0, 0.0)
        }
        
        self.TELEOP_LABELS = {
            'q': '↖', 'w': '⬆', 'e': '↗',
            'a': '⬅', 's': 'STOP', 'd': '➡',
            'z': '↙', 'x': '⬇', 'c': '↘',
            't': 'L-Angle', 'r': 'R-Angle', 'g': 'RETURN'
        }
        
        self.stats = defaultdict(int)
        self.core_db = self.load_core_db()
        self.load_all_stats()
        self.lock = threading.Lock()
        self.last_js_log = ""  # 조이스틱 마지막 액션 (UI 폴링용)
        self.capture_mode = CaptureMode.PRE_CACHE  # 기본값: 비블로킹 캐시 스냅샷
        
        self.is_auto_playing = False
        self.is_returning = False
        self.movement_timer = None
        threading.Thread(target=self._camera_loop, daemon=True).start()

    def toggle_teleop(self):
        self.teleop_mode = not self.teleop_mode
        state = "ACTIVE 🟢" if self.teleop_mode else "OFF 🔴"
        btn_update = gr.update(value=f"🕹️ Teleop Mode: {state}", variant="primary" if self.teleop_mode else "secondary")
        return btn_update, f"🕹️ Teleop Mode switched to {state}"

    def publish_cmd_hw(self, _action):
        action = {'linear_x': _action[0], 'linear_y': _action[1], 'angular_z': _action[2]}
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = action['linear_x'], action['linear_y'], action['angular_z']
        self.cmd_pub.publish(msg)
        if ROBOT_HW_AVAILABLE and self.driver:
            try:
                if any(abs(v) > 0.1 for v in action.values()):
                    if abs(action["angular_z"]) > 0.1:
                        self.driver.spin(int(action["angular_z"] * self.throttle))
                    else:
                        angle = np.degrees(np.arctan2(action["linear_y"], action["linear_x"]))
                        if angle < 0: angle += 360
                        self.driver.move(int(angle), self.throttle)
                else: self.driver.stop()
            except: pass

    def joystick_drive(self, key):
        """조이스틱 전용 — stop 타이머 없이 누르는 동안 연속 이동, None이면 즉시 정지."""
        if key is None:
            self.publish_cmd_hw((0.0, 0.0, 0.0))
            self.last_js_log = "[JS] STOP"
            return
        if key not in self.WASD_TO_CONTINUOUS:
            return
        act = self.WASD_TO_CONTINUOUS[key]
        self.last_js_log = f"[JS] {self.TELEOP_LABELS.get(key, key.upper())}  {act}"
        if self.collecting and self.capture_mode == CaptureMode.PRE_CACHE:
            self._capture_pre_cache(act)
        self.publish_cmd_hw(act)

    def teleop_step(self, key):
        if key not in self.WASD_TO_CONTINUOUS: return "Invalid"
        act = self.WASD_TO_CONTINUOUS[key]
        label = self.TELEOP_LABELS.get(key, key.upper())
        self.last_js_log = f"[JS] {label}  act={act}"
        with self.lock:
            if self.movement_timer: self.movement_timer.cancel()

        # PRE_CACHE: 액션 직전 관측 캡처 (s_t → a_t 쌍 보장)
        if self.collecting and self.capture_mode == CaptureMode.PRE_CACHE:
            self._capture_pre_cache(act)

        self.publish_cmd_hw(act)

        if key != ' ':
            def timed_stop():
                for _ in range(3):
                    self.publish_cmd_hw((0.0, 0.0, 0.0))
                    time.sleep(0.05)
            with self.lock:
                self.movement_timer = threading.Timer(0.4, timed_stop)
                self.movement_timer.start()
            # POST_SYNC: 로봇이 움직이기 시작한 후 새 프레임 수신
            if self.collecting and self.capture_mode == CaptureMode.POST_SYNC:
                self._capture_post_sync(act)
        else:
            for _ in range(3):
                self.publish_cmd_hw((0.0, 0.0, 0.0))
                time.sleep(0.05)

        return f"🕹️ {key.upper()} Command Sent"

    def start_auto_return(self):
        if not self.teleop_mode: return "🕹️ Teleop Mode is OFF"
        
        if self.is_returning:
            self.is_returning = False
            # 발송 중지
            for _ in range(3):
                self.publish_cmd_hw((0.0, 0.0, 0.0))
                time.sleep(0.05)
            return "🛑 Returning Cancelled"
            
        if not self.episode_buffer: return "⚠️ No path to reverse"
        
        def run():
            self.is_returning = True
            try:
                rev_actions = [(-a['action'][0], -a['action'][1], -a['action'][2]) for a in reversed(self.episode_buffer)]
                for act in rev_actions:
                    if not self.is_returning: break
                    self.publish_cmd_hw(act); time.sleep(0.4)
                if self.is_returning:
                    for _ in range(3): self.publish_cmd_hw((0.0, 0.0, 0.0)); time.sleep(0.05)
            finally: self.is_returning = False
            
        threading.Thread(target=run, daemon=True).start()
        return "🔄 Returning to Start..."

    def handle_image_click(self, evt: gr.SelectData):
        if not self.teleop_mode: return "🕹️ Teleop Mode is OFF"
        if evt is None: return "No SelectData"
        x, y = evt.index
        with self.lock:
            if self.latest_ui_frame is None: return "No Image"
            h, w = self.latest_ui_frame.shape[:2]
        col, row = int(x / (w / 3.0)), int(y / (h / 3.0))
        grid_map = [['q', 'w', 'e'],['a', ' ', 'd'],['z', 'x', 'c']]
        return self.teleop_step(grid_map[max(0,min(2,row))][max(0,min(2,col))])

    def _capture_pre_cache(self, act):
        """PRE_CACHE 모드: 액션 직전 캐시 스냅샷 복사. 서비스 콜 없음, <1 ms."""
        with self.lock:
            if self.latest_ui_frame is None: return
            self.episode_buffer.append({
                'image': self.latest_ui_frame.copy(),
                'action': list(act),
                'timestamp': time.time(),
            })

    def _capture_post_sync(self, act):
        """POST_SYNC 모드: 액션 직후 ROS 서비스 콜로 최신 프레임 수신. 최대 300 ms 블로킹."""
        if not self.img_client.service_is_ready(): return
        req = GetImage.Request(); future = self.img_client.call_async(req)
        start_t = time.time()
        while time.time() - start_t < 0.3:
            if future.done(): break
            time.sleep(0.01)
        if future.done():
            try:
                res = future.result()
                if res and res.image:
                    cv_img = self.bridge.imgmsg_to_cv2(res.image, desired_encoding='bgr8')
                    with self.lock:
                        self.latest_ui_frame = cv_img
                        self.episode_buffer.append({'image': cv_img.copy(), 'action': list(act), 'timestamp': time.time()})
            except: pass

    def set_capture_mode(self, label: str):
        self.capture_mode = CaptureMode.PRE_CACHE if label == "PRE_CACHE" else CaptureMode.POST_SYNC
        return f"📷 Capture mode → {self.capture_mode.value}"

    def load_core_db(self):
        if os.path.exists(CORE_DB_PATH):
            with open(CORE_DB_PATH, 'r') as f: return json.load(f)
        return {}
    def save_core_db(self):
        with open(CORE_DB_PATH, 'w') as f: json.dump(self.core_db, f, indent=2)
    def _camera_loop(self):
        while ROS_AVAILABLE and rclpy.ok():
            if self.img_client.service_is_ready():
                req = GetImage.Request(); future = self.img_client.call_async(req)
                start = time.time()
                while time.time() - start < 0.15:
                    if future.done(): break
                    time.sleep(0.01)
                if future.done():
                    try:
                        res = future.result()
                        if res and res.image:
                            cv_img = self.bridge.imgmsg_to_cv2(res.image, desired_encoding='bgr8')
                            with self.lock: self.latest_ui_frame = cv_img
                    except: pass
            time.sleep(0.1)  # 10 Hz
    def load_all_stats(self):
        self.stats = defaultdict(int)
        if os.path.exists(DATASET_ROOT):
            for f in os.listdir(DATASET_ROOT):
                for k, v in V5_SCENARIOS.items():
                    if v['id'] in f: self.stats[k] += 1
    def start_rec(self, key):
        with self.lock: self.current_scenario_key, self.episode_buffer, self.collecting = key, [], True
        return f"🔴 Recording: {V5_SCENARIOS[key]['name']}"

    def auto_play_core(self, key):
        if key not in self.core_db or self.is_auto_playing: return "Err"
        def run():
            self.is_auto_playing = True
            try:
                self.start_rec(key)
                for act in self.core_db[key]:
                    if not self.collecting: break

                    # PRE_CACHE: 액션 직전 관측 캡처 (s_t → a_t 쌍)
                    if self.capture_mode == CaptureMode.PRE_CACHE:
                        self._capture_pre_cache(act)

                    # 1) 액션 전송
                    self.publish_cmd_hw(act)

                    # 2) 정확히 0.4초 후 정지하는 타이머
                    def timed_stop():
                        for _ in range(3):
                            self.publish_cmd_hw((0.0, 0.0, 0.0))
                            time.sleep(0.05)
                    timer = threading.Timer(0.4, timed_stop)
                    timer.start()

                    # POST_SYNC: 로봇이 움직이기 시작한 후 새 프레임 수신 (타이머 남은 시간 내)
                    if self.capture_mode == CaptureMode.POST_SYNC:
                        self._capture_post_sync(act)

                    # 3) 0.4초 정지 프로세스 완료 대기
                    timer.join()
                    
                    # 5) 다음 스텝 전 사람처럼 로봇이 완전히 멈추고 쉴 수 있도록 대기
                    time.sleep(0.8)
                    
                for _ in range(3): self.publish_cmd_hw((0.0, 0.0, 0.0)); time.sleep(0.05)
                self.stop_rec(True)
            finally: self.is_auto_playing = False
        threading.Thread(target=run, daemon=True).start()
        return f"🚀 Auto Replay: {V5_SCENARIOS[key]['name']}"

    def analyze_final_frame(self, img_bgr):
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lower_gray = np.array([0, 0, 50])
        upper_gray = np.array([180, 50, 200])
        mask = cv2.inRange(hsv, lower_gray, upper_gray)
        h, w = img_bgr.shape[:2]
        mask[:h//2, :] = 0 # Consider bottom half only
        
        M = cv2.moments(mask)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else w // 2
            
        if cx < w * 0.4:
            pos = "left"
            prompt = "Keep approaching until the gray basket aligns with the left side of the frame and appears large."
        elif cx > w * 0.6:
            pos = "right"
            prompt = "Move forward until the gray basket is positioned in the right half of the view and close to the camera."
        else:
            pos = "center"
            prompt = "Navigate until the gray basket is centered and fills the lower half of the frame."
            
        return pos, prompt

    def stop_rec(self, save=True):
        MIN_FRAMES = 8
        with self.lock:
            if not self.collecting: return "Idle"
            self.collecting = False
            n = len(self.episode_buffer)
            if save and n < MIN_FRAMES:
                self.episode_buffer = []
                return f"⚠️ Too short ({n} frames < {MIN_FRAMES}). Auto-discarded."
            msg = "❌ Discarded or Empty"
            if save and n > 0:
                final_img = self.episode_buffer[-1]['image']
                pos_tag, prompt = self.analyze_final_frame(final_img)
                last_img = self.episode_buffer[-1]['image']
                for _ in range(self.stop_inject_n):
                    self.episode_buffer.append({
                        'image': last_img.copy(),
                        'action': [0.0, 0.0, 0.0],
                        'timestamp': time.time(),
                    })
                fname = self.save_h5(pos_tag, prompt)
                if self.selected_pattern == "core":
                    self.core_db[self.current_scenario_key] = [d['action'] for d in self.episode_buffer]
                    self.save_core_db()
                self.load_all_stats()
                msg = f"✅ Saved [{pos_tag.upper()}]: {os.path.basename(fname)}\n📝 Prompt: {prompt}\n(+{self.stop_inject_n} STOP frames injected)"
            return msg

    def save_h5(self, pos_tag, prompt):
        ts = datetime.now().strftime("%y%m%d_%H%M%S")
        sid = V5_SCENARIOS[self.current_scenario_key]['id']
        div = f"__{DIVERSITY_TAGS.get(self.diversity_tag, 'free')}" if self.current_scenario_key in ('FL','FC','FR') else ""
        fname = f"episode_{ts}_{sid}{div}__{self.selected_pattern}__{self.selected_distance}_{pos_tag}.h5"
        imgs = [cv2.cvtColor(d['image'], cv2.COLOR_BGR2RGB) for d in self.episode_buffer]
        acts = [d['action'] for d in self.episode_buffer]
        timestamps = [d['timestamp'] for d in self.episode_buffer]

        if self.js_mode == 'async':
            # 10Hz 비동기 주행 시 조종자의 반응 지연(Action Lag, 약 100ms)을 보정하기 위해
            # 액션 배열을 1프레임 앞으로 시프트 (s_t 이미지와 a_{t+1} 액션 매핑)
            shifted_acts = []
            for i in range(len(acts) - 1):
                shifted_acts.append(acts[i+1])
            # 마지막 프레임의 액션은 정지 상태([0.0, 0.0, 0.0])로 매핑
            shifted_acts.append([0.0, 0.0, 0.0])
            acts = shifted_acts

        with h5py.File(os.path.join(DATASET_ROOT, fname), 'w') as f:
            f.create_dataset('observations/images', data=np.array(imgs), compression="gzip")
            f.create_dataset('actions', data=np.array(acts))
            f.create_dataset('timestamps', data=np.array(timestamps))
            f.create_dataset('language_instruction', data=[prompt.encode('utf-8')])
            f.attrs.update({'scenario': sid, 'pattern': self.selected_pattern, 'distance': self.selected_distance, 'end_pos': pos_tag})
        return fname

# --- ROS2 Process Setup ---
node = None
NODE_START_ERROR = ""
if ROS_AVAILABLE:
    try:
        if not rclpy.ok(): rclpy.init()
        node = GradioCollectorNode()
        def spin(): rclpy.spin(node)
        threading.Thread(target=spin, daemon=True).start()
    except Exception as e:
        print(f"FAILED TO START ROS NODE: {e}")
        NODE_START_ERROR = str(e)
        node = None

# --- Joystick Setup ---
joystick_reader = None
if node and PYGAME_AVAILABLE:
    joystick_reader = JoystickReader(node)
    joystick_reader.start()
elif not PYGAME_AVAILABLE:
    print("[Joystick] pygame 미설치 — pip install pygame")


def joystick_status_md(_=None):
    if not joystick_reader:
        icon = "⚫"
        msg = "pygame 미설치" if not PYGAME_AVAILABLE else "조이스틱 비활성"
        return f"{icon} **Joystick:** {msg}"
    s = joystick_reader.status
    if not s["connected"]:
        return "🔴 **Joystick:** 미연결 (USB 확인)"
    key_disp = s["label"] if s["key"] else "NEUTRAL"
    return (
        f"🟢 **{s['name']}** &nbsp;|&nbsp; "
        f"lx `{s['lx']:+.2f}` &nbsp; ly `{s['ly']:+.2f}` &nbsp; az `{s['az']:+.2f}` "
        f"&nbsp;→&nbsp; **{key_disp}**"
    )


def joystick_panel_md(_=None):
    if not joystick_reader:
        return "⚫ **Joystick:** 비활성 (pygame 미설치)"
    s = joystick_reader.status
    if not s["connected"]:
        return "🔴 **Joystick 미연결** — USB 확인"

    def axis_bar(v, width=20):
        center = width // 2
        pos = max(0, min(width - 1, int((v + 1) / 2 * width)))
        bar = ["─"] * width
        bar[center] = "┼"
        bar[pos] = "█"
        return "".join(bar)

    action_map = {
        'q': '↖ FWD+LEFT', 'w': '▲ FORWARD', 'e': '↗ FWD+RIGHT',
        'a': '◀ LEFT',     'd': '▶ RIGHT',
        'x': '▼ BACK',     'z': '↙ BWD+LEFT', 'c': '↘ BWD+RIGHT',
        'r': '↺ ROT_L',    't': '↻ ROT_R',
    }
    current = action_map.get(s['key'], '● NEUTRAL') if s['key'] else '● NEUTRAL'
    icon = "🟢" if s['key'] else "⚪"

    raw = s.get("raw", [])
    raw_lines = "  ".join(f"[{i}]{v:+.2f}" for i, v in enumerate(raw))
    last_log = node.last_js_log if node else ""

    js_mode = node.js_mode if node else 'sync'
    rec_state = node.collecting if node else False
    mode_badge = ("📸 **SYNC** (V5 스텝)" if js_mode == 'sync' else "🌊 **ASYNC** (스무스)")
    rec_badge  = " 🔴 **REC**" if rec_state else ""

    return (
        f"🎮 **{s['name']}**  |  {mode_badge}{rec_badge}\n\n"
        f"```\n"
        f"LX {axis_bar(s['lx'])}  {s['lx']:+.2f}  (전/후)\n"
        f"LY {axis_bar(s['ly'])}  {s['ly']:+.2f}  (좌/우)\n"
        f"AZ {axis_bar(s['az'])}  {s['az']:+.2f}  (회전)\n"
        f"\n"
        f"RAW  {raw_lines}\n"
        f"```\n"
        f"{icon} **{current}**"
        + (f"\n\n`{last_log}`" if last_log else "")
    )


def collector_diagnostics(_=None):
    ros_ws = os.getenv("VLA_ROS_WS", os.path.join(_PROJECT_ROOT, "ROS_action"))
    checks = [
        ("ROS import", "OK" if ROS_AVAILABLE else f"FAIL: {ROS_IMPORT_ERROR or 'unknown'}"),
        ("Node ready", "OK" if node else f"OFFLINE: {NODE_START_ERROR or 'node unavailable'}"),
        ("pygame", "OK" if PYGAME_AVAILABLE else "MISSING — pip install pygame"),
        ("Joystick", joystick_reader.status["name"] if joystick_reader and joystick_reader.status["connected"] else "미연결"),
        ("ROS workspace", "OK" if os.path.exists(ros_ws) else f"MISSING: {ros_ws}"),
        ("camera_interfaces", "OK" if os.path.exists(os.path.join(ros_ws, 'install', 'camera_interfaces')) else "MISSING"),
        ("Dataset root", DATASET_ROOT),
    ]
    lines = ["### 🧪 Collector Diagnostics"]
    for key, val in checks:
        lines.append(f"- **{key}**: {val}")
    return "\n".join(lines)


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

def get_feed(_=None):
    if not node: return None
    with node.lock:
        if node.latest_ui_frame is None: return None
        img = cv2.cvtColor(node.latest_ui_frame, cv2.COLOR_BGR2RGB)
        if node.teleop_mode:
            h, w = img.shape[:2]
            cv2.line(img, (w//3, 0), (w//3, h), (100, 255, 100), 1)
            cv2.line(img, (2*w//3, 0), (2*w//3, h), (100, 255, 100), 1)
            cv2.line(img, (0, h//3), (w, h//3), (100, 255, 100), 1)
            cv2.line(img, (0, 2*h//3), (w, 2*h//3), (100, 255, 100), 1)
        return Image.fromarray(img)

def update_ui_state(_=None):
    if not node:
        return "ROS Offline", ""
    node.load_all_stats()
    with node.lock:
        if node.collecting:
            n = len(node.episode_buffer)
            target = V5_SCENARIOS.get(node.current_scenario_key, {}).get('target', 0)
            name = V5_SCENARIOS.get(node.current_scenario_key, {}).get('name', '')
            if target > 0:
                pct = min(100, int(n / target * 100))
                bar = "█" * (pct // 10) + "░" * (10 - pct // 10)
                if n >= target:
                    s = f"✅ TARGET MET [{n}/{target}] {bar} 100% — {name}"
                else:
                    s = f"● REC [{n}/{target}] {bar} {pct}% — {name}"
            else:
                s = f"● REC [{n}] — {name}"
        else:
            s = "IDLE"
        if node.is_auto_playing: s = "🚀 REPLAYING..."
        if node.is_returning: s = "🔄 RETURNING..."
        tbl = "| ID | 시나리오 | 진행률 | 개수/목표 | 자동 |\n|---|---|---|---|---|\n"
        for k, v in V5_SCENARIOS.items():
            c, t = node.stats[k], v['target']
            p = min(100, (c/t*100)) if t > 0 else 0
            tbl += f"| {k} | {v['name']} | {'█'*int(p/10)+'░'*(10-int(p/10))} {p:.1f}% | {c}/{t} | {'✅' if k in node.core_db else '❌'} |\n"
        return s, tbl

CUSTOM_CSS = """
.gradio-container { background-color: #0d1117 !important; color: #c9d1d9 !important; font-family: 'Outfit', sans-serif; }
.main-title { text-align: center; color: #58a6ff; font-weight: 900; letter-spacing: -1px; margin-bottom: 20px; }
.camera-card { border: 2px solid #30363d; border-radius: 16px; background: #010409; padding: 15px; position: relative; }
.status-card { text-align: center; font-family: 'JetBrains Mono'; font-size: 1.1rem; background: #161b22; border-radius: 10px; padding: 12px; margin-bottom: 20px; border-left: 5px solid #58a6ff; }
.scenario-btn { border-radius: 8px !important; text-align: left !important; border: 1px solid #30363d !important; background: #21262d !important; }
.action-btn { font-weight: bold !important; border-radius: 10px !important; }
"""

CUSTOM_JS = """
function() {
    document.addEventListener('keydown', function(event) {
        if (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'TEXTAREA') return;
        const valid = ['w','a','s','d','q','e','z','x','c','r','t','g',' '];
        let key = event.key.toLowerCase();
        if (valid.includes(key)) {
            event.preventDefault();
            let el = document.getElementById('btn_' + (key === ' ' ? 'space' : key));
            if(el) { el.classList.add('active'); el.click(); setTimeout(() => el.classList.remove('active'), 100); }
        }
    });
}
"""

def make_rec_fn(key_val): return lambda: node.start_rec(key_val) if node else "Node Offline"
def make_auto_fn(key_val): return lambda: node.auto_play_core(key_val) if node else "Node Offline"
def make_teleop_fn(k_val): return lambda: node.teleop_step(k_val) if node else "Node Offline"

with gr.Blocks(title="MoNaVLA V5 PRO") as demo:
    gr.Markdown("# 🛸 MoNaVLA V5 Control Hub", elem_classes=["main-title"])

    _cam_st, _cam_start_btn, _cam_stop_btn = camera_control_widget()
    _cam_start_btn.click(fn=start_camera, outputs=_cam_st)
    _cam_stop_btn.click(fn=stop_camera,   outputs=_cam_st)

    with gr.Row():
        with gr.Column(scale=2, elem_classes=["camera-card"]):
            stream = gr.Image(label="Live Target View", interactive=False, elem_id="main_camera")
            status_markdown = gr.Markdown("### IDLE", elem_classes=["status-card"])
            js_status = gr.Markdown(joystick_status_md())
            js_panel = gr.Markdown(joystick_panel_md())
            with gr.Row():
                mode_btn = gr.Button("🕹️ TELEOP MODE: OFF 🔴", variant="secondary", interactive=bool(node))
                stop_save = gr.Button("⏹️ SAVE EPISODE", variant="primary", interactive=bool(node))
                discard = gr.Button("🗑️ DISCARD", variant="stop", interactive=bool(node))
                undo_btn = gr.Button("↩️ Undo", size="sm", interactive=bool(node))
            
            grid_btns = {}
            with gr.Row():
                grid_btns['q'] = gr.Button(f"Q {OFFLINE_TELEOP_LABELS['q']}", elem_id="btn_q", size="sm", interactive=bool(node))
                grid_btns['w'] = gr.Button(f"W {OFFLINE_TELEOP_LABELS['w']}", elem_id="btn_w", size="sm", interactive=bool(node))
                grid_btns['e'] = gr.Button(f"E {OFFLINE_TELEOP_LABELS['e']}", elem_id="btn_e", size="sm", interactive=bool(node))
            with gr.Row():
                grid_btns['a'] = gr.Button(f"A {OFFLINE_TELEOP_LABELS['a']}", elem_id="btn_a", size="sm", interactive=bool(node))
                grid_btns[' '] = gr.Button("STOP 🛑", elem_id="btn_space", variant="stop", size="sm", interactive=bool(node))
                grid_btns['d'] = gr.Button(f"D {OFFLINE_TELEOP_LABELS['d']}", elem_id="btn_d", size="sm", interactive=bool(node))
            with gr.Row():
                grid_btns['z'] = gr.Button(OFFLINE_TELEOP_LABELS['z'], elem_id="btn_z", size="sm", interactive=bool(node))
                grid_btns['x'] = gr.Button(OFFLINE_TELEOP_LABELS['x'], elem_id="btn_x", size="sm", interactive=bool(node))
                grid_btns['c'] = gr.Button(OFFLINE_TELEOP_LABELS['c'], elem_id="btn_c", size="sm", interactive=bool(node))
            with gr.Row():
                grid_btns['t'] = gr.Button(f"{OFFLINE_TELEOP_LABELS['t']} (T)", elem_id="btn_t", size="sm", interactive=bool(node))
                grid_btns['r'] = gr.Button(f"{OFFLINE_TELEOP_LABELS['r']} (R)", elem_id="btn_r", size="sm", interactive=bool(node))
                grid_btns['g'] = gr.Button(f"{OFFLINE_TELEOP_LABELS['g']} (G)", elem_id="btn_g", variant="secondary", size="sm", interactive=bool(node))

        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### ⚙️ Episode Config")
                pattern_sel = gr.Radio(["CORE", "VARIANT"], value="CORE", label="Type")
                dist_sel = gr.Radio(["FIXED", "VAR"], value="FIXED", label="Distance")
                capture_sel = gr.Radio(
                    ["PRE_CACHE", "POST_SYNC"],
                    value="PRE_CACHE",
                    label="Capture Mode",
                    info="PRE_CACHE: 액션 직전 캐시 스냅샷 (<1ms, 권장) | POST_SYNC: 액션 직후 서비스 콜 (최대 300ms 블로킹)"
                )
                throttle_sl = gr.Slider(minimum=10, maximum=100, value=50, step=5, label="Throttle (%)")
                stop_inject_sl = gr.Slider(minimum=0, maximum=10, value=5, step=1, label="STOP Inject N")
                js_mode_sel = gr.Radio(
                    ["SYNC (V5 호환)", "ASYNC (스무스)"],
                    value="SYNC (V5 호환)",
                    label="🕹️ 조이스틱 수집 모드",
                    info="SYNC: 0.45s 스텝, V5 호환 (권장) | ASYNC: 10Hz 연속  /  조이스틱 START 버튼으로도 전환"
                )
                gr.Markdown("#### 🎯 Scenarios")
                scen_click_list = []
                for k, v in V5_SCENARIOS.items():
                    with gr.Row():
                        b_rec = gr.Button(f"[{k}] {v['name']}", elem_classes=["scenario-btn"], scale=4, interactive=bool(node))
                        b_auto = gr.Button("▶️", scale=1, interactive=bool(node))
                        scen_click_list.append((k, b_rec, b_auto))
            with gr.Group():
                gr.Markdown("#### 🎲 자유 수집 (다양성 21개 = 좌/중/우 × 7)")
                diversity_sel = gr.Dropdown(
                    choices=list(DIVERSITY_TAGS.keys()),
                    value=list(DIVERSITY_TAGS.keys())[0],
                    label="다양성 조건 태그",
                )
                with gr.Row():
                    free_left_btn  = gr.Button("🎲 좌측 시작", variant="secondary", interactive=bool(node))
                    free_center_btn = gr.Button("🎲 중앙 시작", variant="secondary", interactive=bool(node))
                    free_right_btn = gr.Button("🎲 우측 시작", variant="secondary", interactive=bool(node))
                free_stats = gr.Markdown("")
            log = gr.Textbox(label="Terminal Log", interactive=False)
            stats_tbl = gr.Markdown("")
            diag_tbl = gr.Markdown(collector_diagnostics())

    if node:
        mode_btn.click(fn=node.toggle_teleop, outputs=[mode_btn, log])
        stream.select(fn=node.handle_image_click, outputs=[log])
        
        for k_char, btn_obj in grid_btns.items():
            if k_char == 'g':
                btn_obj.click(fn=lambda: node.start_auto_return() if node else "Node Offline", outputs=[log])
            else:
                btn_obj.click(fn=make_teleop_fn(k_char), outputs=[log])
        
        for k_val, b_rec, b_auto in scen_click_list:
            b_rec.click(fn=make_rec_fn(k_val), outputs=[log])
            b_auto.click(fn=make_auto_fn(k_val), outputs=[log])
        
        def set_pattern(p): node.selected_pattern = p.lower()
        def set_distance(d): node.selected_distance = d.lower()

        pattern_sel.change(fn=set_pattern, inputs=pattern_sel)
        dist_sel.change(fn=set_distance, inputs=dist_sel)
        capture_sel.change(fn=node.set_capture_mode, inputs=capture_sel, outputs=[log])
        throttle_sl.change(
            fn=lambda v: setattr(node, 'throttle', int(v)) or f"Throttle → {int(v)}%",
            inputs=throttle_sl, outputs=log,
        )
        stop_inject_sl.change(
            fn=lambda v: setattr(node, 'stop_inject_n', int(v)) or f"STOP Inject N → {int(v)}",
            inputs=stop_inject_sl, outputs=log,
        )
        def set_js_mode(v):
            node.js_mode = 'sync' if 'SYNC' in v else 'async'
            return f"조이스틱 모드 → {node.js_mode.upper()}"
        js_mode_sel.change(fn=set_js_mode, inputs=js_mode_sel, outputs=log)

        def set_diversity(tag):
            node.diversity_tag = tag
            return f"다양성 태그 → {tag}"
        diversity_sel.change(fn=set_diversity, inputs=diversity_sel, outputs=log)

        def free_stats_md():
            counts = {}
            for fk, fv in [("FL","free_left"),("FC","free_center"),("FR","free_right")]:
                counts[fk] = len([f for f in os.listdir(DATASET_ROOT) if fv in f and f.endswith('.h5')])
            total = sum(counts.values())
            pct = min(100, int(total/21*100))
            bar = "█"*(pct//10)+"░"*(10-pct//10)
            return (f"좌 {counts['FL']}/7  중 {counts['FC']}/7  우 {counts['FR']}/7  "
                    f"합계 **{total}/21** [{bar}] {pct}%")

        def start_free(key):
            node.current_scenario_key = key
            return node.start_rec(key)

        free_left_btn.click(fn=lambda: start_free("FL"), outputs=[log])
        free_center_btn.click(fn=lambda: start_free("FC"), outputs=[log])
        free_right_btn.click(fn=lambda: start_free("FR"), outputs=[log])
        def undo_frame():
            with node.lock:
                if node.episode_buffer:
                    node.episode_buffer.pop()
                    return f"↩️ Undone — {len(node.episode_buffer)} frames remaining"
                return "⚠️ Nothing to undo"
        undo_btn.click(fn=undo_frame, outputs=[log])
        stop_save.click(fn=lambda: node.stop_rec(True), outputs=[log])
        discard.click(fn=lambda: node.stop_rec(False), outputs=[log])
    
    gr.Timer(1).tick(fn=update_ui_state, outputs=[status_markdown, stats_tbl])
    if node:
        gr.Timer(2).tick(fn=free_stats_md, outputs=[free_stats])
    gr.Timer(1).tick(fn=collector_diagnostics, outputs=[diag_tbl])
    gr.Timer(0.1).tick(fn=get_feed, outputs=stream)
    gr.Timer(0.1).tick(fn=joystick_status_md, outputs=[js_status])
    gr.Timer(0.1).tick(fn=joystick_panel_md, outputs=[js_panel])

if __name__ == "__main__":
    requested_port = int(os.getenv("VLA_COLLECT_PORT", os.getenv("GRADIO_SERVER_PORT", "8081")))
    server_port = pick_server_port(requested_port)
    demo.launch(server_name="0.0.0.0", server_port=server_port, js=CUSTOM_JS, css=CUSTOM_CSS, theme=gr.themes.Soft())

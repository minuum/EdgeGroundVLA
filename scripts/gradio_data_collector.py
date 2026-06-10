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
        def compressed_imgmsg_to_cv2(self, imgmsg, desired_encoding='bgr8'):
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

    # 버튼 인덱스 (DragonRise 기본값. 패드마다 다르면 화면 '버튼 모니터'로 실제 번호 확인 후 수정)
    BTN_STOP   = 0   # A  — STOP 명시적 1프레임
    BTN_UNDO   = 1   # B  — 마지막 프레임 취소
    BTN_DISCARD = 2  # X  — 에피소드 폐기
    BTN_TELEOP = 3   # Y  — teleop 토글 (어느 패드든 버튼 한 번으로)
    BTN_START  = 7   # Start — SYNC/ASYNC 모드 토글
    BTN_SELECT = 6   # Select — 녹화 토글(시작↔저장)
    # ── 어깨 버튼(L1/R1) = 수집 시작/저장 (양 레이아웃 공통) ──
    BTN_REC_START = 4  # L1 — 녹화 시작 (선택 시나리오)
    BTN_REC_SAVE  = 5  # R1 — 녹화 저장 후 종료
    # ── L2/R2: DragonRise=버튼, Controller=트리거축. _detect_layout()가 확정 ──
    BTN_L2 = -1  # DragonRise일 때 버튼 6 (teleop)
    BTN_R2 = -1  # DragonRise일 때 버튼 7 (모드)
    TRIG_L2 = 4  # Controller일 때 축 4 (teleop)
    TRIG_R2 = 5  # Controller일 때 축 5 (모드)
    layout = "controller"

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
        # ROT 전용 임계값 (Issue 2): 낮출수록 미세 회전 보정 입력을 캡처.
        # physical drift(전압/마찰) 보정용 작은 회전을 잡으려면 0.3 정도로.
        self.rot_threshold = 0.5
        self._last_btn = None  # 버튼 모니터: 마지막 눌린 버튼 인덱스
        self._trig_prev = {self.TRIG_L2: -1.0, self.TRIG_R2: -1.0}  # 트리거 엣지 검출용
        self._hat_prev = (0, 0)  # D-pad 엣지 검출용 (시나리오 넘기기)

        # Gradio 상태 표시용 (lock-free read 허용 — 단순 dict 교체)
        self.status = {
            "connected": False, "name": "—",
            "lx": 0.0, "ly": 0.0, "az": 0.0,
            "key": None, "label": "—", "buttons": [], "last_btn": None,
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
        RT = self.rot_threshold  # ROT 전용 임계값 (미세 회전 캡처용, Issue 2)
        fwd = lx >=  T
        bwd = lx <= -T
        lft = ly >=  T
        rgt = ly <= -T
        rl  = az >=  RT
        rr  = az <= -RT

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
        self._last_btn = btn  # 버튼 모니터용 (마지막 누른 인덱스)
        if btn == self.BTN_STOP:
            nd.teleop_step(' ')
        elif btn == self.BTN_UNDO:
            with nd.lock:
                if nd.episode_buffer:
                    nd.episode_buffer.pop()
        elif btn == self.BTN_DISCARD:
            nd.stop_rec(save=False)
        elif btn == self.BTN_TELEOP:
            nd.toggle_teleop()
            nd.last_js_log = f"[Y] teleop → {'ON' if nd.teleop_mode else 'OFF'}"
        elif btn == self.BTN_START:
            nd.js_mode = 'async' if nd.js_mode == 'sync' else 'sync'
            nd.last_js_log = f"[BTN{btn}] 모드 → {nd.js_mode.upper()}"
        # ── DragonRise L2/R2 = 버튼 (Controller는 트리거축 → _loop에서 처리) ──
        elif self.BTN_L2 >= 0 and btn == self.BTN_L2:
            nd.toggle_teleop(); nd.last_js_log = "[L2] teleop 토글"
        elif self.BTN_R2 >= 0 and btn == self.BTN_R2:
            nd.js_mode = 'async' if nd.js_mode == 'sync' else 'sync'
            nd.last_js_log = f"[R2] 모드 → {nd.js_mode.upper()}"
        # ── 수집 전용 (어깨 버튼 L1/R1) ──
        elif btn == self.BTN_REC_START:
            self._rec_start()
        elif btn == self.BTN_REC_SAVE:
            self._rec_save()
        elif btn == self.BTN_SELECT:
            # 토글: 녹화중이면 저장, 아니면 선택 시나리오로 시작
            if nd.collecting:
                self._rec_save()
            else:
                self._rec_start()

    def _rec_start(self):
        nd = self._node
        if nd.collecting:
            nd.last_js_log = "⚠ 이미 녹화 중"
        elif nd.current_scenario_key:
            nd.start_rec(nd.current_scenario_key)
            nd.last_js_log = f"▶ 녹화 시작: {V5_SCENARIOS.get(nd.current_scenario_key,{}).get('name','')}"
        else:
            nd.last_js_log = "⚠ 시나리오 먼저 선택"
        print(f"[Joystick] {nd.last_js_log}")

    def _rec_save(self):
        nd = self._node
        if nd.collecting:
            nd.stop_rec(save=True)
            nd.last_js_log = "⏹ 녹화 저장 완료"
        else:
            nd.last_js_log = "⚠ 녹화 중이 아님"
        print(f"[Joystick] {nd.last_js_log}")

    def _detect_layout(self, js):
        """연결 패드를 보고 버튼 매핑 자동 결정 (DragonRise vs Xbox식 Controller).
        공통: L1=4 녹화시작, R1=5 녹화저장, A=0 STOP, B=1 취소, X=2 폐기.
        차이: L2/R2 — DragonRise=버튼6/7, Controller=트리거축4/5. Select/Start 번호도 다름."""
        name = js.get_name().lower()
        nbtn, nax = js.get_numbuttons(), js.get_numaxes()
        is_dragon = ("dragon" in name or "generic" in name or "usb gamepad" in name
                     or (nbtn <= 12 and nax <= 5))
        self.BTN_STOP, self.BTN_UNDO, self.BTN_DISCARD = 0, 1, 2
        self.BTN_REC_START, self.BTN_REC_SAVE = 4, 5
        if is_dragon:
            self.layout = "dragonrise"
            self.BTN_L2, self.BTN_R2 = 6, 7        # 버튼
            self.BTN_SELECT, self.BTN_START = 8, 9
            self.TRIG_L2, self.TRIG_R2 = -1, -1    # 트리거 없음
        else:
            self.layout = "controller"
            self.BTN_L2, self.BTN_R2 = -1, -1      # 버튼 아님
            self.BTN_SELECT, self.BTN_START = 6, 7
            self.TRIG_L2, self.TRIG_R2 = 4, 5      # 트리거 축

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
                self._detect_layout(js)
                self.status = {**self.status, "connected": True, "name": js.get_name(), "layout": self.layout}
                print(f"[Joystick] 연결됨: {js.get_name()} (layout={self.layout})")
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
                        # SYNC: 모션은 연속 발행(정지 펄스 없음, 버벅임 제거),
                        # 프레임 캡처만 0.45s 스텝 간격 → V5 호환 cadence 유지
                        do_cap = (now - self._last_step_time) >= self.STEP_INTERVAL
                        self._node.joystick_drive_sync(key, do_cap)
                        if do_cap:
                            self._last_step_time = now
                    else:
                        # ASYNC: 10Hz 연속 스무스 드라이브
                        if (now - self._last_step_time) >= 0.10:
                            self._node.joystick_drive(key)
                            self._last_step_time = now
                elif self._prev_key:
                    # 스틱을 놓으면(neutral) 즉시 정지 — SYNC/ASYNC 공통 (안전)
                    self._node.joystick_drive(None)
                self._prev_key = key

                # 상태 갱신
                labels = {'q':'↖FWD+L','w':'▲FWD','e':'↗FWD+R','a':'←LEFT',
                          'd':'→RIGHT','x':'▼BACK','z':'↙','c':'↘',
                          'r':'↺ROT_L','t':'↻ROT_R'}
                raw = [round(js.get_axis(i), 3) for i in range(js.get_numaxes())]
                pressed = [i for i in range(js.get_numbuttons()) if js.get_button(i)]
                self.status = {
                    "connected": True, "name": js.get_name(),
                    "lx": round(lx, 2), "ly": round(ly, 2), "az": round(az, 2),
                    "key": key, "label": labels.get(key, "NEUTRAL") if key else "NEUTRAL",
                    "raw": raw, "buttons": pressed, "last_btn": self._last_btn,
                }

                # 트리거(L2/R2) 엣지 감지 (축 → 눌림 임계 0.3 상승엣지)
                nax = js.get_numaxes()
                for ax, fn in ((self.TRIG_L2, 'teleop'), (self.TRIG_R2, 'mode')):
                    if ax < 0 or ax >= nax:
                        continue
                    tv = js.get_axis(ax)
                    if tv > 0.3 and self._trig_prev.get(ax, -1.0) <= 0.3:
                        if fn == 'teleop':
                            self._node.toggle_teleop()
                            self._node.last_js_log = "[L2] teleop 토글"
                        else:
                            self._node.js_mode = 'async' if self._node.js_mode == 'sync' else 'sync'
                            self._node.last_js_log = f"[R2] 모드 → {self._node.js_mode.upper()}"
                        print(f"[Joystick] {self._node.last_js_log}")
                    self._trig_prev[ax] = tv

                # D-pad(hat) 엣지 → 시나리오 넘기기 (좌=이전 / 우=다음, 상/하도 동일)
                if js.get_numhats() > 0:
                    hx, hy = js.get_hat(0)
                    phx, phy = self._hat_prev
                    if hx != 0 and phx == 0:
                        self._node.select_scenario(1 if hx > 0 else -1)
                    elif hy != 0 and phy == 0:
                        self._node.select_scenario(-1 if hy > 0 else 1)  # 위=이전, 아래=다음
                    self._hat_prev = (hx, hy)

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
    "A-의자좌극단":    "chair_left_extreme",
    "B-의자우극단":    "chair_right_extreme",
    "C-로봇근접":      "robot_close",
    "D-로봇원거리":    "robot_far",
    "E-사선좌접근":    "diagonal_left",
    "F-사선우접근":    "diagonal_right",
    "G-조명차이":      "lighting_diff",
}

# ── V5-2: chair 객체 신규 수집 (기존 V5 = gray basket 과 완전 별개) ──────────
# 폴더/객체를 환경변수로도 덮어쓸 수 있게 함 (기본 = V5-2 / chair)
TARGET_OBJECT = os.getenv("VLA_TARGET_OBJECT", "chair")
DATASET_ROOT = os.path.join(_PROJECT_ROOT, os.getenv("VLA_COLLECT_DATASET", "ROS_action/mobile_vla_dataset_v5_2"))
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
        self.js_mode = 'async'  # V5-2 기본 = ASYNC 10Hz (추론 10Hz와 정합) | 'sync' = V5 스텝
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
        self.last_session_summary = None  # 마지막 세션 타이밍 (Hz 설계용)
        # 도착존(arrival zone) 판정 임계값 — 추론 latch와 동일 스케일로 시작(MonAPI 연동값)
        self.arrival_area_th = 0.18   # bbox넓이/전체프레임 ≥ 이면 근접
        self.arrival_cx_tol  = 0.25   # |cx-0.5| ≤ 이면 중앙 정렬
        self.arrival_cache = None     # 카메라 스레드가 채우는 도착 판정 캐시(UI는 읽기만)
        # 이미지 저장 모드: 'jpeg'(q=90 vlen, 신규 기본) | 'raw'(gzip 원본 배열, 기존 V5 방식)
        self.image_storage = os.getenv("VLA_IMAGE_STORAGE", "jpeg").lower()
        self.jpeg_quality = int(os.getenv("VLA_JPEG_QUALITY", "90"))
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

    def joystick_drive_sync(self, key, do_capture):
        """SYNC 조이스틱 연속 주행 — 정지 펄스(timed_stop) 없이 publish.
        모션은 매 호출 연속 발행(버벅임 제거), 프레임 캡처는 do_capture=True일 때만
        (0.45s 스텝 간격 → V5 호환 기록 cadence 유지). neutral(None)이면 즉시 정지."""
        if key is None:
            self.publish_cmd_hw((0.0, 0.0, 0.0))
            self.last_js_log = "[JS] STOP"
            return
        if key not in self.WASD_TO_CONTINUOUS:
            return
        act = self.WASD_TO_CONTINUOUS[key]
        self.last_js_log = f"[JS] {self.TELEOP_LABELS.get(key, key.upper())}  {act}"
        if do_capture and self.collecting and self.capture_mode == CaptureMode.PRE_CACHE:
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
                    cv_img = self.bridge.compressed_imgmsg_to_cv2(res.image, desired_encoding='bgr8')
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
                            cv_img = self.bridge.compressed_imgmsg_to_cv2(res.image, desired_encoding='bgr8')
                            # 도착 판정을 카메라 스레드에서 1회만 (다운스케일 320×180 → HSV ~10x 저렴).
                            # UI 핫패스(get_feed/arrival_hud)는 이 캐시만 읽음.
                            try:
                                small = cv2.resize(cv_img, (320, 180))
                                area, cx, cy, has, bbox_r = compute_arrival_metric(small)
                                in_zone = bool(has and area >= self.arrival_area_th
                                               and abs(cx - 0.5) <= self.arrival_cx_tol)
                                cache = {"area": area, "cx": cx, "cy": cy, "has": has,
                                         "in_zone": in_zone, "bbox_r": bbox_r}
                            except Exception:
                                cache = None
                            with self.lock:
                                self.latest_ui_frame = cv_img
                                if cache is not None:
                                    self.arrival_cache = cache
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

    def select_scenario(self, step):
        """녹화 시작 없이 current_scenario_key만 순환 (D-pad로 시나리오 넘기기)."""
        if self.collecting:
            self.last_js_log = "⚠ 녹화 중엔 시나리오 변경 불가"
            return self.last_js_log
        keys = list(V5_SCENARIOS.keys())
        i = (keys.index(self.current_scenario_key) + step) % len(keys) if self.current_scenario_key in keys else 0
        self.current_scenario_key = keys[i]
        name = V5_SCENARIOS[self.current_scenario_key]['name']
        self.last_js_log = f"🎯 시나리오: [{self.current_scenario_key}] {name}  (L1=시작)"
        return self.last_js_log

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
        # 저채도(흰/회색) 블롭의 하단부 cx로 도착 위치(left/center/right) 추정.
        # chair(흰 의자 등)도 저채도라 동일 로직 적용 — compute_arrival_metric과 동일 마스크.
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, CHAIR_HSV_LO, CHAIR_HSV_HI)
        h, w = img_bgr.shape[:2]
        mask[:h//2, :] = 0 # Consider bottom half only

        M = cv2.moments(mask)
        cx = int(M["m10"] / M["m00"]) if M["m00"] != 0 else w // 2

        obj = TARGET_OBJECT  # "chair" (V5-2)
        if cx < w * 0.4:
            pos = "left"
            prompt = f"Keep approaching until the {obj} aligns with the left side of the frame and appears large."
        elif cx > w * 0.6:
            pos = "right"
            prompt = f"Move forward until the {obj} is positioned in the right half of the view and close to the camera."
        else:
            pos = "center"
            prompt = f"Navigate until the {obj} is centered and fills the lower half of the frame."

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
                # ── 세션 타이밍 측정 (STOP 주입 전 실제 주행 구간 기준) ──
                ts0 = self.episode_buffer[0]['timestamp']
                ts1 = self.episode_buffer[-1]['timestamp']
                dur = max(1e-6, ts1 - ts0)
                hz = (n - 1) / dur if n > 1 else 0.0
                self.last_session_summary = {
                    "frames": n, "duration_s": round(dur, 2), "hz": round(hz, 1),
                    "mode": self.js_mode, "scenario": self.current_scenario_key,
                }

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
                msg = (f"✅ Saved [{pos_tag.upper()}]: {os.path.basename(fname)}\n"
                       f"⏱️ 세션: {n}프레임 / {dur:.1f}s / 실측 {hz:.1f}Hz ({self.js_mode})\n"
                       f"📝 Prompt: {prompt}\n(+{self.stop_inject_n} STOP frames injected)")
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
            # observations/images 저장 모드 분할 (self.image_storage):
            #  - 'jpeg' (신규 기본): JPEG(q) bytes를 vlen으로 저장 (파일 크기 대폭↓)
            #  - 'raw'  (기존 V5 방식): gzip 압축 원본 배열
            # 로더는 per-frame ndim(1=JPEG bytes / 3=raw)로 자동 분기 → 두 포맷 혼용 호환.
            if self.image_storage == 'jpeg':
                # JPEG는 색순서 무관 → RGB 배열 인코딩→디코딩 시 동일 RGB 복원 (로더 RGB 가정 유지)
                vlen = h5py.vlen_dtype(np.uint8)
                ds = f.create_dataset('observations/images', (len(imgs),), dtype=vlen)
                for i, rgb in enumerate(imgs):
                    buf = cv2.imencode('.jpg', rgb, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])[1]
                    ds[i] = np.frombuffer(buf.tobytes(), dtype=np.uint8)
                ds.attrs['format'] = 'jpeg'
                ds.attrs['quality'] = self.jpeg_quality
            else:  # 'raw' — 기존 V5 방식 그대로
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

    # 버튼 모니터: 현재 눌린 버튼 + 마지막 누른 인덱스 (패드 버튼 번호 확인용)
    pressed = s.get("buttons", [])
    last_btn = s.get("last_btn", None)
    JR = joystick_reader
    btn_map = {
        JR.BTN_REC_START: "▶시작", JR.BTN_REC_SAVE: "⏹저장", JR.BTN_SELECT: "녹화토글",
        JR.BTN_DISCARD: "🗑폐기", JR.BTN_STOP: "STOP", JR.BTN_UNDO: "↩취소",
        JR.BTN_START: "모드",
    }
    layout = s.get("layout", JR.layout)
    pressed = set(pressed)
    teleop = node.teleop_mode if node else False
    sel = node.current_scenario_key if node else None
    sel_name = V5_SCENARIOS.get(sel, {}).get('name', '미선택') if sel else '미선택'

    # ── 버튼 라이트: ●눌림 / ○꺼짐 (index 기준) ──
    def L(idx):  # 버튼 lit
        return "🟢" if idx in pressed else "⚫"
    def T(ax):   # 트리거 lit (축 > 0.3)
        return "🟢" if (ax is not None and ax >= 0 and len(raw) > ax and raw[ax] > 0.3) else "⚫"
    # L2/R2: 패드별 (Controller=트리거 / DragonRise=버튼)
    if layout == "controller":
        l2, r2 = T(JR.TRIG_L2), T(JR.TRIG_R2)
        l2tag, r2tag = f"L2(당김ax{JR.TRIG_L2})", f"R2(당김ax{JR.TRIG_R2})"
    else:
        l2, r2 = L(JR.BTN_L2), L(JR.BTN_R2)
        l2tag, r2tag = f"L2(버튼{JR.BTN_L2})", f"R2(버튼{JR.BTN_R2})"

    teleop_badge = "🟢 ON" if teleop else "⚫ OFF"
    board = (
        f"🎮 **{s['name']}** `[{layout}]`  |  {mode_badge}{rec_badge}  |  TELEOP **{teleop_badge}**\n\n"
        f"```\n"
        f"LX {axis_bar(s['lx'])} {s['lx']:+.2f} 전/후\n"
        f"LY {axis_bar(s['ly'])} {s['ly']:+.2f} 좌/우\n"
        f"AZ {axis_bar(s['az'])} {s['az']:+.2f} 회전\n"
        f"```\n"
        f"**[어깨]**  {l2} {l2tag}=teleop &nbsp; {r2} {r2tag}=모드\n"
        f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{L(JR.BTN_REC_START)} L1(버튼{JR.BTN_REC_START})=▶녹화시작 &nbsp; {L(JR.BTN_REC_SAVE)} R1(버튼{JR.BTN_REC_SAVE})=⏹저장\n\n"
        f"**[방향]**  D-pad ◀▶ = 시나리오 넘기기 &nbsp;→ 현재 **[{sel or '?'}] {sel_name}**\n\n"
        f"**[얼굴]**  {L(JR.BTN_STOP)} A({JR.BTN_STOP})=STOP &nbsp; {L(JR.BTN_UNDO)} B({JR.BTN_UNDO})=취소 &nbsp; "
        f"{L(JR.BTN_DISCARD)} X({JR.BTN_DISCARD})=🗑폐기 &nbsp; {L(JR.BTN_TELEOP)} Y({JR.BTN_TELEOP})=teleop\n\n"
        f"**[시스템]**  {L(JR.BTN_SELECT)} Select({JR.BTN_SELECT})=녹화토글 &nbsp; {L(JR.BTN_START)} Start({JR.BTN_START})=모드"
    )
    return board + (f"\n\n`{last_log}`" if last_log else "")


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

# chair(흰/회색) 검출용 저채도 마스크 — value 상한 255로 밝은 흰 의자까지 포함
CHAIR_HSV_LO = np.array([0, 0, 60])
CHAIR_HSV_HI = np.array([180, 60, 255])


def compute_arrival_metric(frame_bgr):
    """저채도(흰/회색) 블롭의 하단부 bounding box로 도착 근접도 추정 (chair에 적합).
    return (area_frac, cx, cy, has_blob, bbox_ratio). bbox_ratio=(rx,ry,rw,rh) 0~1 (스케일 무관)."""
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, CHAIR_HSV_LO, CHAIR_HSV_HI)
    h, w = frame_bgr.shape[:2]
    mask[:h // 2, :] = 0  # 하단 절반만 (바닥 근처의 타겟)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0, 0.5, 0.5, False, None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < (w * h * 0.002):  # 작은 노이즈 무시
        return 0.0, 0.5, 0.5, False, None
    bx, by, bw, bh = cv2.boundingRect(c)
    area_frac = (bw * bh) / float(w * h)
    cx = (bx + bw / 2) / w
    cy = (by + bh / 2) / h
    return area_frac, cx, cy, True, (bx / w, by / h, bw / w, bh / h)


def get_feed(_=None):
    """라이브 피드 — UI 핫패스. HSV 연산 없음(카메라 스레드 캐시 사용), 640×360 다운스케일 전송."""
    if not node: return None
    with node.lock:
        if node.latest_ui_frame is None: return None
        frame = node.latest_ui_frame
        cache = node.arrival_cache
        teleop = node.teleop_mode
        # BGR→RGB + 다운스케일을 lock 안에서 한 번에 (frame 참조만, copy 불필요)
        disp = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (640, 360))
    DW, DH = 640, 360
    if teleop:
        cv2.line(disp, (DW//3, 0), (DW//3, DH), (100, 255, 100), 1)
        cv2.line(disp, (2*DW//3, 0), (2*DW//3, DH), (100, 255, 100), 1)
        cv2.line(disp, (0, DH//3), (DW, DH//3), (100, 255, 100), 1)
        cv2.line(disp, (0, 2*DH//3), (DW, 2*DH//3), (100, 255, 100), 1)
    # 도착존 오버레이 — 캐시(비율 좌표)에서 읽어 그림 (HSV 재계산 없음)
    if cache and cache["has"] and cache["bbox_r"]:
        rx, ry, rw, rh = cache["bbox_r"]
        x0, y0, x1, y1 = int(rx*DW), int(ry*DH), int((rx+rw)*DW), int((ry+rh)*DH)
        color = (0, 255, 0) if cache["in_zone"] else (255, 200, 0)
        cv2.rectangle(disp, (x0, y0), (x1, y1), color, 2)
        tag = "ARRIVAL - STOP" if cache["in_zone"] else f"area {cache['area']:.2f}"
        cv2.putText(disp, tag, (x0, max(16, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return Image.fromarray(disp)

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
            sel = node.current_scenario_key
            if sel:
                s = f"⏸ IDLE — 선택: [{sel}] {V5_SCENARIOS.get(sel,{}).get('name','')}  ▶ L1(또는 Shift+R)로 녹화 시작"
            else:
                s = "⏸ IDLE — 시나리오 선택 필요 (D-pad 좌/우 또는 화면 버튼 클릭)"
        if node.is_auto_playing: s = "🚀 REPLAYING..."
        if node.is_returning: s = "🔄 RETURNING..."
        tbl = "| ID | 시나리오 | 진행률 | 개수/목표 | 자동 |\n|---|---|---|---|---|\n"
        for k, v in V5_SCENARIOS.items():
            c, t = node.stats[k], v['target']
            p = min(100, (c/t*100)) if t > 0 else 0
            tbl += f"| {k} | {v['name']} | {'█'*int(p/10)+'░'*(10-int(p/10))} {p:.1f}% | {c}/{t} | {'✅' if k in node.core_db else '❌'} |\n"
        return s, tbl

# ───────────────────────── 수집 관측(observability) ─────────────────────────
# 모두 read-only — episode_buffer / DATASET_ROOT 만 읽고 수집 로직은 건드리지 않는다.
# 8-class 분류는 robovlm_nav/datasets/nav_h5_dataset_impl.py 와 동일 임계값 사용:
#   is_x=|lx|>0.3, is_y=|ly|>0.3, is_z=|az|>0.1
CLASS_NAMES_8  = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "ROT_L", "ROT_R"]
CLASS_SYMBOLS  = {0: "●", 1: "▲", 2: "◀", 3: "▶", 4: "↖", 5: "↗", 6: "↺", 7: "↻"}


def classify_8class(action) -> int:
    x  = float(action[0])
    y  = float(action[1])
    az = float(action[2]) if len(action) > 2 else 0.0
    is_x, is_y = abs(x) > 0.3, abs(y) > 0.3
    if not is_x and not is_y:
        if az > 0.1:   return 6
        if az < -0.1:  return 7
        return 0
    if x > 0.3:
        if y > 0.3:    return 4
        if y < -0.3:   return 5
        return 1
    if abs(x) < 0.3:
        if y > 0.3:    return 2
        if y < -0.3:   return 3
        return 0
    return 0  # backward 등은 STOP 취급 (학습 분류와 동일)


def episode_timeline_md(_=None):
    """현재 에피소드 최근 액션을 기호 시퀀스 + 실측 캡처 Hz로 표시."""
    if not node:
        return "### 📊 현재 에피소드\n_(ROS offline)_"
    with node.lock:
        buf = list(node.episode_buffer)
        collecting = node.collecting
    if not buf:
        return "### 📊 현재 에피소드\n_(대기 중 — 시나리오 선택 후 REC/SELECT)_"
    n = len(buf)
    syms = "".join(CLASS_SYMBOLS[classify_8class(d['action'])] for d in buf[-28:])
    ts = [d['timestamp'] for d in buf[-12:]]
    hz = ""
    if len(ts) >= 2:
        dts = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
        mean_dt = sum(dts) / len(dts)
        if mean_dt > 0:
            hz = f" · {1.0 / mean_dt:.1f}Hz"
    badge = "🔴 REC" if collecting else "⏹ STOPPED"
    return (f"### 📊 현재 에피소드  ({badge} · {n} frames{hz})\n"
            f"```\n최근→  {syms}\n```")


def episode_dist_md(_=None):
    """현재 에피소드 버퍼의 8-class 분포 ASCII 막대."""
    if not node:
        return ""
    with node.lock:
        buf = list(node.episode_buffer)
    if not buf:
        return ""
    counts = [0] * 8
    for d in buf:
        counts[classify_8class(d['action'])] += 1
    mx = max(counts) or 1
    lines = ["**이번 에피소드 액션 분포**", "```"]
    for i, c in enumerate(counts):
        if c == 0:
            continue
        lines.append(f"{CLASS_SYMBOLS[i]} {CLASS_NAMES_8[i]:8s} {'█' * int(c / mx * 12):<12} {c}")
    lines.append("```")
    return "\n".join(lines)


def episode_thumbs(_=None):
    """최근 캡처 프레임 6~8장을 액션 라벨과 함께 갤러리로."""
    if not node:
        return []
    with node.lock:
        buf = list(node.episode_buffer)
    if not buf:
        return []
    out = []
    for d in buf[-8:]:
        try:
            # 160×90 썸네일로 축소 → 브라우저 전송량 대폭 절감 (720p 8장 → 작은 8장)
            small = cv2.resize(d['image'], (160, 90))
            img = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            cls = classify_8class(d['action'])
            out.append((Image.fromarray(img), f"{CLASS_SYMBOLS[cls]} {CLASS_NAMES_8[cls]}"))
        except Exception:
            continue
    return out


def arrival_hud_md(_=None):
    """라이브 도착 판정 HUD — 카메라 스레드 캐시만 읽음 (HSV 재계산 없음)."""
    if not node:
        return ""
    c = node.arrival_cache
    if not c:
        return "### 🎯 도착 판정\n_(카메라 대기 중)_"
    if not c["has"]:
        badge = "⚪ 타겟(저채도 블롭) 미검출"
    elif c["in_zone"]:
        badge = "🟢 **도착 가능 — 정지하세요!**"
    else:
        badge = "🟡 접근 중"
    return (
        f"### 🎯 도착 판정  {badge}\n"
        f"```\n"
        f"area {c['area']:.3f}  (th {node.arrival_area_th:.2f})\n"
        f"cx   {c['cx']:.2f}   |Δ| {abs(c['cx'] - 0.5):.2f}  (tol {node.arrival_cx_tol:.2f})\n"
        f"```"
    )


def session_summary_md(_=None):
    """마지막으로 저장된 세션의 소요 초 / 프레임 / 실측 Hz — 수집·추론 Hz 설계용."""
    if not node or not getattr(node, "last_session_summary", None):
        return ("### ⏱️ 마지막 세션\n_(도착 시 정지 → 저장하면 소요 초·Hz가 여기 표시됩니다)_")
    s = node.last_session_summary
    return (
        f"### ⏱️ 마지막 세션 (Hz 설계용)\n"
        f"```\n"
        f"프레임   : {s['frames']}\n"
        f"소요 시간 : {s['duration_s']} s\n"
        f"실측 Hz  : {s['hz']} Hz   (수집 모드: {s['mode'].upper()})\n"
        f"```\n"
        f"→ 추론 가능 상한 13.3Hz (GoalNav exp49 실측). "
        f"이 세션 길이 기준으로 수집/추론 Hz를 맞추면 됩니다."
    )


_dataset_dist_cache = {"n_files": -1, "md": ""}


def dataset_dist_md(_=None):
    """전체 데이터셋 H5의 8-class 누적 분포 (파일 수 변동 시에만 재계산)."""
    files = [f for f in os.listdir(DATASET_ROOT) if f.endswith('.h5')] if os.path.exists(DATASET_ROOT) else []
    if len(files) == _dataset_dist_cache["n_files"]:
        return _dataset_dist_cache["md"]
    counts = [0] * 8
    for fn in files:
        try:
            with h5py.File(os.path.join(DATASET_ROOT, fn), 'r') as f:
                acts = f['actions'][:]
            for a in acts:
                counts[classify_8class(a)] += 1
        except Exception:
            continue
    total = sum(counts) or 1
    lines = [f"**전체 데이터셋 액션 분포**  ({len(files)} ep · {total} frames)", "```"]
    for i, c in enumerate(counts):
        pct = c / total * 100
        lines.append(f"{CLASS_SYMBOLS[i]} {CLASS_NAMES_8[i]:8s} {pct:5.1f}% {'█' * int(pct / 5)}")
    lines.append("```")
    md = "\n".join(lines)
    _dataset_dist_cache.update({"n_files": len(files), "md": md})
    return md


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
    function clickBtn(id) {
        let el = document.getElementById(id);
        if (el) { el.classList.add('active'); el.click(); setTimeout(() => el.classList.remove('active'), 120); }
    }
    document.addEventListener('keydown', function(event) {
        if (document.activeElement.tagName === 'INPUT' || document.activeElement.tagName === 'TEXTAREA') return;
        let key = event.key.toLowerCase();

        // ── 조합 단축키 (수정자 + 키) ──
        // Shift+R: 녹화 토글 (시작 ↔ 저장)  /  Shift+X: 폐기(Discard)
        if (event.shiftKey && key === 'r') { event.preventDefault(); clickBtn('btn_record_toggle'); return; }
        if (event.shiftKey && key === 'x') { event.preventDefault(); clickBtn('btn_discard'); return; }

        // ── 단일 키 텔레옵 (수정자 눌리면 무시 → 조합키와 충돌 방지) ──
        if (event.shiftKey || event.ctrlKey || event.altKey || event.metaKey) return;
        const valid = ['w','a','s','d','q','e','z','x','c','r','t','g',' '];
        if (valid.includes(key)) {
            event.preventDefault();
            clickBtn('btn_' + (key === ' ' ? 'space' : key));
        }
    });
}
"""

def record_toggle():
    """녹화 토글 (Shift+R): 수집 중이면 저장, 아니면 선택된 시나리오로 시작."""
    if not node:
        return "Node Offline"
    if node.collecting:
        return node.stop_rec(True)
    if node.current_scenario_key:
        return node.start_rec(node.current_scenario_key)
    return "⚠️ 먼저 시나리오를 선택하세요 (시나리오 버튼 클릭 후 Shift+R)"


def make_rec_fn(key_val): return lambda: node.start_rec(key_val) if node else "Node Offline"
def make_auto_fn(key_val): return lambda: node.auto_play_core(key_val) if node else "Node Offline"
def make_teleop_fn(k_val): return lambda: node.teleop_step(k_val) if node else "Node Offline"

with gr.Blocks(title="MoNaVLA V5 PRO") as demo:
    gr.Markdown(f"# 🛸 MoNaVLA V5-2 Control Hub &nbsp;·&nbsp; 🎯 target: **{TARGET_OBJECT}**", elem_classes=["main-title"])

    _cam_st, _cam_start_btn, _cam_stop_btn = camera_control_widget()
    _cam_start_btn.click(fn=start_camera, outputs=_cam_st)
    _cam_stop_btn.click(fn=stop_camera,   outputs=_cam_st)

    with gr.Row():
        with gr.Column(scale=2, elem_classes=["camera-card"]):
            stream = gr.Image(label="Live Target View", interactive=False, elem_id="main_camera")
            status_markdown = gr.Markdown("### IDLE", elem_classes=["status-card"])
            js_status = gr.Markdown(joystick_status_md())
            arrival_hud = gr.Markdown("### 🎯 도착 판정", elem_classes=["status-card"])
            js_panel = gr.Markdown(joystick_panel_md())
            with gr.Row():
                mode_btn = gr.Button("🕹️ TELEOP MODE: OFF 🔴", variant="secondary", interactive=bool(node))
                record_toggle_btn = gr.Button("🔴 REC 토글 (Shift+R)", elem_id="btn_record_toggle", variant="primary", interactive=bool(node))
                stop_save = gr.Button("⏹️ SAVE EPISODE", variant="primary", interactive=bool(node))
                discard = gr.Button("🗑️ DISCARD (Shift+X)", elem_id="btn_discard", variant="stop", interactive=bool(node))
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
            # ── 1) 시나리오 (가장 자주 쓰는 액션 — 최상단 고정) ──────────────────
            with gr.Group():
                gr.Markdown("### 🎯 시나리오 — 클릭해서 수집 시작")
                scen_click_list = []
                for k, v in V5_SCENARIOS.items():
                    with gr.Row():
                        b_rec = gr.Button(f"[{k}] {v['name']}", elem_classes=["scenario-btn"], scale=4, interactive=bool(node))
                        b_auto = gr.Button("▶️", scale=1, min_width=44, interactive=bool(node))
                        scen_click_list.append((k, b_rec, b_auto))
            # ── 2) 자유 수집 ─────────────────────────────────────────────────────
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
            # ── 3) 핵심 토글 (수집 모드 / 저장 포맷 — 자주 확인) ──────────────────
            with gr.Group():
                js_mode_sel = gr.Radio(
                    ["SYNC (V5 호환)", "ASYNC (스무스)"],
                    value="ASYNC (스무스)",
                    label="🕹️ 조이스틱 수집 모드",
                    info="V5-2 기본 ASYNC: 10Hz 연속 (추론 10Hz와 정합) | SYNC: 0.45s 스텝  /  조이스틱 START로도 전환"
                )
                storage_sel = gr.Radio(
                    ["JPEG (q90, 신규 기본)", "RAW (gzip, 기존 V5)"],
                    value="JPEG (q90, 신규 기본)",
                    label="💾 이미지 저장 포맷",
                    info="JPEG: 파일 크기 대폭↓ (모델 224 입력이라 품질손실 0) | RAW: 기존 V5. 로더 자동 인식",
                )
            # ── 4) 고급 설정 (자주 안 바꿈 — 접이식) ─────────────────────────────
            with gr.Accordion("⚙️ 고급 설정 (Capture/Throttle/임계값)", open=False):
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
                rot_thresh_sl = gr.Slider(
                    minimum=0.1, maximum=0.7, value=0.5, step=0.05,
                    label="🔄 ROT 회전 민감도 (az 임계값)",
                    info="낮출수록 작은 회전 입력도 ROT_L/R로 캡처 — physical drift 미세보정용 (기본 0.5)",
                )
                gr.Markdown("##### 🎯 도착존(arrival) 판정 임계값")
                arrival_area_sl = gr.Slider(
                    minimum=0.05, maximum=0.6, value=0.18, step=0.01,
                    label="area_th (타겟이 화면 채우는 비율)",
                    info="이 값 이상이면 '근접'. chair로 직접 가까이 가서 area 값 보고 캘리브",
                )
                arrival_cx_sl = gr.Slider(
                    minimum=0.1, maximum=0.5, value=0.25, step=0.05,
                    label="cx_tol (중앙 정렬 허용오차 |cx-0.5|)",
                    info="이 값 이하면 '중앙'. area_th와 동시 충족 시 🟢 도착",
                )
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
        def set_rot_thresh(v):
            if joystick_reader:
                joystick_reader.rot_threshold = float(v)
            return f"🔄 ROT 임계값 → {float(v):.2f} (낮을수록 미세 회전 캡처)"
        rot_thresh_sl.change(fn=set_rot_thresh, inputs=rot_thresh_sl, outputs=log)
        arrival_area_sl.change(
            fn=lambda v: setattr(node, 'arrival_area_th', float(v)) or f"🎯 area_th → {float(v):.2f}",
            inputs=arrival_area_sl, outputs=log,
        )
        arrival_cx_sl.change(
            fn=lambda v: setattr(node, 'arrival_cx_tol', float(v)) or f"🎯 cx_tol → {float(v):.2f}",
            inputs=arrival_cx_sl, outputs=log,
        )
        def set_storage(v):
            node.image_storage = 'raw' if v.startswith("RAW") else 'jpeg'
            return f"💾 이미지 저장 포맷 → {node.image_storage.upper()}"
        storage_sel.change(fn=set_storage, inputs=storage_sel, outputs=log)
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
        record_toggle_btn.click(fn=record_toggle, outputs=[log])
        stop_save.click(fn=lambda: node.stop_rec(True), outputs=[log])
        discard.click(fn=lambda: node.stop_rec(False), outputs=[log])
    
    # ── 수집 관측 패널 (세션에 쌓이는 모습 실시간) ──────────────────────────────
    gr.Markdown("## 📈 수집 관측 (Live)")
    with gr.Row():
        with gr.Column(scale=1):
            ep_timeline_md_box = gr.Markdown("### 📊 현재 에피소드")
            ep_dist_md_box = gr.Markdown("")
            session_summary_box = gr.Markdown("### ⏱️ 마지막 세션")
        with gr.Column(scale=1):
            ds_dist_md_box = gr.Markdown("")
    ep_gallery = gr.Gallery(
        label="🖼️ 최근 캡처 프레임 (액션 라벨)", columns=8, height=150,
        object_fit="cover", show_label=True,
    )

    gr.Timer(1).tick(fn=update_ui_state, outputs=[status_markdown, stats_tbl])
    if node:
        gr.Timer(2).tick(fn=free_stats_md, outputs=[free_stats])
    gr.Timer(1).tick(fn=collector_diagnostics, outputs=[diag_tbl])
    gr.Timer(0.1).tick(fn=get_feed, outputs=stream)              # 경량(캐시 오버레이 + 640×360)
    gr.Timer(0.2).tick(fn=joystick_status_md, outputs=[js_status])   # 텍스트 — 0.1→0.2s
    gr.Timer(0.2).tick(fn=joystick_panel_md, outputs=[js_panel])     # 텍스트 — 0.1→0.2s
    gr.Timer(0.3).tick(fn=arrival_hud_md, outputs=[arrival_hud])     # 캐시 읽기
    # 관측 위젯 폴링 (가벼움 — read-only)
    gr.Timer(0.5).tick(fn=episode_timeline_md, outputs=[ep_timeline_md_box])
    gr.Timer(0.5).tick(fn=episode_dist_md, outputs=[ep_dist_md_box])
    gr.Timer(1.0).tick(fn=episode_thumbs, outputs=[ep_gallery])      # 썸네일 — 0.5→1.0s
    gr.Timer(3).tick(fn=dataset_dist_md, outputs=[ds_dist_md_box])
    gr.Timer(1).tick(fn=session_summary_md, outputs=[session_summary_box])

if __name__ == "__main__":
    requested_port = int(os.getenv("VLA_COLLECT_PORT", os.getenv("GRADIO_SERVER_PORT", "8081")))
    server_port = pick_server_port(requested_port)
    demo.launch(server_name="0.0.0.0", server_port=server_port, js=CUSTOM_JS, css=CUSTOM_CSS, theme=gr.themes.Soft())

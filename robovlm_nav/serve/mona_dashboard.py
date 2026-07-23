#!/usr/bin/env python3
"""
MoNaVLA 드라이브 대시보드 v2 (FastAPI, 포트 7800)
────────────────────────────────────────────────────────────────────
7865 Gradio 대시보드를 완전히 대체하는 프리미엄 FastAPI 싱글페이지 대시보드.
카메라 MJPEG 스트리밍, SYNC/PRE/ASYNC 주행 루프, H5 세션 로깅, 실시간
Grounding bbox 드로잉, Latency/Drift 시뮬레이터, Action Trajectory 궤적 플롯,
STOP 임계값 캘리브레이션, H5 세션 히스토리 검색 & 셀프 라벨링 및 시스템 모니터링 포함.
"""

import argparse
import base64
import collections
import datetime
import glob
import io
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional, List, Dict

import numpy as np
import h5py
from fastapi import FastAPI, Response, Header
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image, ImageOps

# ── 프로젝트 루트 추가 ────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]   # MoNaVLA/
sys.path.insert(0, str(ROOT))

# ── ROS2 (선택적) ─────────────────────────────────────────────────
ROS_AVAILABLE = False
try:
    import rclpy
    from rclpy.callback_groups import ReentrantCallbackGroup
    from rclpy.node import Node
    from cv_bridge import CvBridge
    from geometry_msgs.msg import Twist
    from camera_interfaces.srv import GetImage
    import cv2
    ROS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"ROS2 unavailable: {e}")
    class Node: pass

# ── 조이스틱 (선택적, DragonRise 게임패드) ──────────────────────────
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# ── 환경 변수 ─────────────────────────────────────────────────────
INFER_URL  = os.getenv("VLA_API_SERVER", "http://localhost:8001")
API_KEY    = os.getenv("VLA_API_KEY",    "vla_devel_key_2026")
# exp73 GoalNav 전용 서버(inference_server.py, VLA_GOALNAV_ONLY=1) — stage2_v2(8001)와 별개 프로세스.
GOALNAV_URL = os.getenv("VLA_GOALNAV_SERVER", "http://localhost:8000")
SODA_IP    = os.getenv("SODA_IP",        "100.85.118.58")
DRIVE_PORT = int(os.getenv("DRIVE_PORT", "7800"))

# 세션 디렉토리 정의
INFER_REPORT_DIR = ROOT / "docs" / "inference_reports"
INFER_H5_DIR     = ROOT / "docs" / "inference_sessions"
CALIB_DIR        = ROOT / "logs" / "calib_sessions"
LABEL_JSON_PATH  = Path("/tmp/mona_preview_labels.json")

INFER_REPORT_DIR.mkdir(parents=True, exist_ok=True)
INFER_H5_DIR.mkdir(parents=True, exist_ok=True)
CALIB_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("mona_dash")

# ── 컬러 보정 파라미터 ────────────────────────────────────────────
_cc = {"r_gain": 1.0, "g_gain": 1.0, "b_gain": 1.0}

def _correct(img: Image.Image) -> Image.Image:
    arr = np.array(img).astype(np.float32)
    r, g, b = cv2.split(arr)
    arr = cv2.merge([r * _cc["r_gain"], g * _cc["g_gain"], b * _cc["b_gain"]])
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ═══════════════════════════════════════════════════════════════════
# ROS 제어 노드
# ═══════════════════════════════════════════════════════════════════
class MoNaROSNode(Node):
    """카메라 수신 + cmd_vel 퍼블리셔. VLAControlManager 포함."""

    def __init__(self):
        super().__init__(f"mona_dash_{os.getpid()}")
        cbg = ReentrantCallbackGroup()
        self.cv_bridge = CvBridge()
        self.get_img = self.create_client(
            GetImage, "get_image_service", callback_group=cbg)
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10, callback_group=cbg)

        from robovlm_nav.serve.vla_control_utils import VLAControlManager
        self.ctrl = VLAControlManager(self, default_throttle=50, move_duration=0.4)

        self._lock   = threading.Lock()
        self._frame: Optional[np.ndarray] = None   # BGR
        self._stable: Optional[np.ndarray] = None  # SYNC용
        self.frame_count = 0
        self.last_ts = 0.0

        threading.Thread(target=self._cam_loop, daemon=True, name="cam-loop").start()
        log.info("✅ MoNaROSNode 초기화 완료")

    # ── 카메라 폴링 10Hz ─────────────────────────────────────────
    def _cam_loop(self):
        while rclpy.ok():
            if not self.get_img.service_is_ready():
                self.get_img.wait_for_service(timeout_sec=1.0)
                time.sleep(0.05)
                continue
            fut = self.get_img.call_async(GetImage.Request())
            t0 = time.time()
            while time.time() - t0 < 0.30:
                if fut.done(): break
                time.sleep(0.01)
            if fut.done():
                try:
                    res = fut.result()
                    if res and res.image.data:
                        img = None
                        try:
                            img = self.cv_bridge.compressed_imgmsg_to_cv2(res.image, "bgr8")
                        except Exception: pass
                        if img is None:
                            img = self.cv_bridge.imgmsg_to_cv2(res.image, "bgr8")
                        if img is not None:
                            with self._lock:
                                self._frame = img
                                self.frame_count += 1
                                self.last_ts = time.time()
                except Exception: pass
            time.sleep(0.1)

    # ── 프레임 획득 ───────────────────────────────────────────────
    def latest_bgr(self) -> Optional[np.ndarray]:
        with self._lock: return self._frame

    def latest_rgb(self) -> Optional[np.ndarray]:
        with self._lock: f = self._frame
        return cv2.cvtColor(f, cv2.COLOR_BGR2RGB) if f is not None else None

    def jpeg_bytes(self, quality=75) -> Optional[bytes]:
        rgb = self.latest_rgb()
        if rgb is None: return None
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, "JPEG", quality=quality)
        return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════
# 전역 상태
# ═══════════════════════════════════════════════════════════════════
_ros: Optional[MoNaROSNode] = None


# ═══════════════════════════════════════════════════════════════════
# 데이터수집 (mobile_vla_data_collector.py 이식) — Phase 1+2
# H5 스키마는 원본과 100% 동일 유지 (기존 resync_scenario_progress/분석
# 스크립트 호환). 입력은 VLAControlManager.publish_and_move() 단일 진입점을
# 통과하므로 키보드/조이스틱 어느 쪽으로 명령을 내려도 자동으로 기록됨.
# ═══════════════════════════════════════════════════════════════════

# 원본 mobile_vla_data_collector.py WASD_TO_CONTINUOUS 그대로 (line 42-54)
COLLECT_KEY_TO_VEL = {
    "w": (1.15, 0.0, 0.0),
    "a": (0.0, 1.15, 0.0),
    "s": (-1.15, 0.0, 0.0),
    "d": (0.0, -1.15, 0.0),
    "q": (1.15, 1.15, 0.0),
    "e": (1.15, -1.15, 0.0),
    "z": (-1.15, 1.15, 0.0),
    "c": (-1.15, -1.15, 0.0),
    "r": (0.0, 0.0, 0.25),
    "t": (0.0, 0.0, -0.25),
    " ": (0.0, 0.0, 0.0),
}

# 원본 mode="2" (V5 Phase-1.5) cup_scenarios 그대로 (line 89-99)
COLLECT_SCENARIOS = {
    "target_left_left_path":     {"target": 15, "key": "1", "label": "좌측 위치 · 좌회전 경로"},
    "target_left_straight_path": {"target": 20, "key": "2", "label": "좌측 위치 · 직진 경로"},
    "target_left_right_path":    {"target": 15, "key": "3", "label": "좌측 위치 · 우회전 경로"},
    "target_center_left_path":     {"target": 15, "key": "4", "label": "중앙 위치 · 좌회전 경로"},
    "target_center_straight_path": {"target": 20, "key": "5", "label": "중앙 위치 · 직진 경로"},
    "target_center_right_path":    {"target": 15, "key": "6", "label": "중앙 위치 · 우회전 경로"},
    "target_right_left_path":     {"target": 15, "key": "7", "label": "우측 위치 · 좌회전 경로"},
    "target_right_straight_path": {"target": 20, "key": "8", "label": "우측 위치 · 직진 경로"},
    "target_right_right_path":    {"target": 15, "key": "9", "label": "우측 위치 · 우회전 경로"},
}
COLLECT_PATTERNS = {"core": "핵심 패턴 (Core)", "variant": "변형 패턴 (Variant)"}

# 극단 배치 데이터수집 트랙A (plan_20260707_heterogeneous_instruction_extreme_cx_collection.md)
# — 4포지션 × 접근경로 3종(좌곡선/직진/우곡선, 지시 불필요·오퍼레이터 재량) × 15회 = 180ep
# + 트랙C(오버슈트→재수렴, CH62 근거, 2026-07-16 추가) — 같은 4포지션 재사용,
# 위치당 목표 45(트랙A) + 16(트랙C 2방향×8) = 61
# center는 트랙F(중앙 위치 커버리지, 2026-07-16 추가, plan §1-2) — 트랙A와 동일
# 3경로×15회=45ep만 필요, 트랙C(오버슈트) 대상 아님(오버슈트는 극단 4위치 전용).
# cx 밴드는 plan에 구체 수치가 없어 다른 위치들과 동일한 폭(0.05)으로 화면 정중앙
# 대칭 배치.
COLLECT_CX_POSITIONS = {
    "strong_left":  {"label": "강한좌",   "lo": 0.10, "hi": 0.15, "target": 61},
    "weak_left":    {"label": "준극단좌", "lo": 0.20, "hi": 0.25, "target": 61},
    "center":       {"label": "중앙",     "lo": 0.475, "hi": 0.525, "target": 45},
    "weak_right":   {"label": "준극단우", "lo": 0.75, "hi": 0.80, "target": 61},
    "strong_right": {"label": "강한우",   "lo": 0.85, "hi": 0.90, "target": 61},
}
# 위치당 45개가 "경로 다양하게"로 뭉뚱그려져 있으면 실제로 15/15/15가 지켜졌는지
# 검증이 안 됨 — 위치×경로 조합별로 세분화해서 목표 15씩 따로 추적.
# overshoot_* 2종은 트랙C(CH62 근거, docs/v5/closed_loop_eval/CH62_FORWARD_LOCK_AND_LABEL_CONFOUND.md
# §7, plan_20260707_heterogeneous_instruction_extreme_cx_collection.md §1 "트랙 C") —
# 지시 없이 자유주행이되 "일부러 과하게 꺾었다가 반대로 재보정"하는 동작을 명시적으로 수집.
COLLECT_TRACKA_PATHS = {
    "left_curve":  {"label": "좌곡선", "target": 15},
    "straight":    {"label": "직진",   "target": 15},
    "right_curve": {"label": "우곡선", "target": 15},
    "overshoot_left_recover":  {"label": "오버슈트→우", "target": 8},
    "overshoot_right_recover": {"label": "오버슈트→좌", "target": 8},
}


# Gradio 데이터수집기(scripts/gradio_data_collector.py)의 8-class 분류/기호와
# 동일 정의 — 임계값도 robovlm_nav/datasets/nav_h5_dataset_impl.py와 일치시킴.
COLLECT_CLASS_NAMES_8 = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "ROT_L", "ROT_R"]
COLLECT_CLASS_SYMBOLS = {0: "●", 1: "▲", 2: "◀", 3: "▶", 4: "↖", 5: "↗", 6: "↺", 7: "↻"}


def _collect_classify_8class(action: dict) -> int:
    x, y = float(action["linear_x"]), float(action["linear_y"])
    az = float(action.get("angular_z", 0.0))
    is_x, is_y = abs(x) > 0.3, abs(y) > 0.3
    if not is_x and not is_y:
        if az > 0.1:
            return 6
        if az < -0.1:
            return 7
        return 0
    if x > 0.3:
        if y > 0.3:
            return 4
        if y < -0.3:
            return 5
        return 1
    if abs(x) < 0.3:
        if y > 0.3:
            return 2
        if y < -0.3:
            return 3
    return 0


def _collect_classify_time_period(hour: int) -> str:
    if 5 <= hour < 8:
        return "dawn"
    if 8 <= hour < 18:
        return "morning"
    if 18 <= hour < 21:
        return "evening"
    return "night"


class DataCollectSession:
    """웹 대시보드용 데이터수집 세션 — 원본 스크립트의 상태머신/H5 스키마 이식."""

    ACTION_CHUNK_SIZE = 8
    DEFAULT_LAYOUT_TYPE = "hori"
    STOP_INJECT_N = 5  # 저장 직전 마지막 프레임을 STOP 액션으로 N번 복제 (Gradio stop_inject_n 이식)

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.data_dir / "scenario_progress.json"
        self.time_period_file = self.data_dir / "time_period_stats.json"
        self.core_pattern_file = self.data_dir / "core_patterns.json"

        self._lock = threading.Lock()
        self.active = False
        self.episode_data: List[dict] = []
        self.episode_name: Optional[str] = None
        self.episode_started_at: Optional[float] = None
        self.selected_scenario: Optional[str] = None
        self.selected_pattern: Optional[str] = None
        self.staged_scenario: Optional[str] = None  # 조이스틱 D-pad로 녹화 전 미리 선택해둔 시나리오
        self.staged_cx_position: Optional[str] = None  # 조이스틱 D-pad로 녹화 전 미리 선택해둔 트랙A cx위치
        self.selected_cx_position: Optional[str] = None  # 극단 배치 4포지션(강한좌/준극단좌/준극단우/강한우)
        self.staged_cx_path: Optional[str] = None  # 트랙A 접근경로(좌곡선/직진/우곡선) — 위치와 별개 축
        self.selected_cx_path: Optional[str] = None
        # D-pad 좌/우가 어느 축을 순환시킬지: "trackA"(cx위치, 기본) | "scenario"(9종 시나리오)
        # D-pad 상/하로 전환. 웹 UI에서 해당 카드의 행/◀▶를 쓰면 자동으로 그 축으로 전환됨.
        self.collect_mode = "trackA"

        self.scenario_stats: dict = collections.defaultdict(int)
        self.cx_position_stats: dict = collections.defaultdict(int)
        # 위치×경로 세분화 카운트 — 키: "{position}::{path}" (예: "strong_left::left_curve")
        self.cx_position_path_stats: dict = collections.defaultdict(int)
        self.time_period_stats: dict = collections.defaultdict(int)
        self.core_patterns: dict = {}
        self._load_progress()

        # 복귀(경로 역재생) — Gradio start_auto_return() 이식용. 저장/폐기와 무관하게
        # 직전 에피소드의 액션+타임스탬프만 남겨둠(이미지 없음, 가벼움).
        self._last_episode_actions: List[dict] = []
        self._returning = False
        # 최근 저장/폐기 내역 — "녹화했는지 안했는지" 한눈에 보이도록 (최신이 [0])
        self._recent_saves: List[dict] = []
        # 마지막 저장 세션의 프레임/소요시간/실측Hz — Gradio session_summary_md 이식
        self._last_session_summary: Optional[dict] = None

    # ── VLAControlManager.on_command 훅 — source 무관(키보드/조이스틱) 기록 ──
    # publish_and_move()는 robust_stop()의 5x 중복 정지펄스(0.05s 간격)처럼
    # 의미 없는 반복 호출도 거치므로, 연속된 STOP 프레임은 1개로만 눌러 담아
    # H5가 중복 정지 프레임으로 도배되는 것을 막는다 (원본 collector는 이런
    # 워치독/robust_stop 반복 펄스를 애초에 collect_data=False로 기록 안 함).
    def on_command(self, lx, ly, az, source):
        if not self.active or _ros is None:
            return
        is_stop = abs(lx) < 0.01 and abs(ly) < 0.01 and abs(az) < 0.01
        with self._lock:
            if not self.active:
                return
            if is_stop and self.episode_data:
                last = self.episode_data[-1]["action"]
                if abs(last["linear_x"]) < 0.01 and abs(last["linear_y"]) < 0.01 and abs(last["angular_z"]) < 0.01:
                    return  # 직전도 STOP이면 중복 기록 생략
            frame = _ros.latest_bgr()
            if frame is None:
                return
            self.episode_data.append({
                "image": frame.copy(),
                "action": {"linear_x": lx, "linear_y": ly, "angular_z": az},
                "action_event_type": source,
                "t": time.time(),
            })

    def start_episode(self, episode_name=None, scenario=None, pattern=None, cx_position=None, cx_path=None) -> str:
        with self._lock:
            self.episode_data = []
            self.selected_scenario = scenario
            self.selected_pattern = pattern
            self.selected_cx_position = cx_position
            self.selected_cx_path = cx_path
            if not episode_name:
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                parts = [f"episode_{ts}"]
                if scenario:
                    parts.append(scenario)
                if pattern:
                    parts.append(pattern)
                if cx_position:
                    parts.append(cx_position)
                if cx_path:
                    parts.append(cx_path)
                episode_name = "_".join(parts)
            self.episode_name = episode_name
            self.episode_started_at = time.time()
            self.active = True
        log.info(f"📷 [DataCollect] 에피소드 시작: {episode_name}")
        return episode_name

    def stage_scenario(self, scenario: Optional[str]) -> Optional[str]:
        """웹 UI(진행률 행 클릭)로 시나리오를 직접 지정 — D-pad cycle_scenario와 동일한 staged_scenario를 공유."""
        with self._lock:
            if self.active:
                return self.staged_scenario
            self.staged_scenario = scenario or None
            self.collect_mode = "scenario"
            return self.staged_scenario

    def cycle_scenario(self, step: int) -> Optional[str]:
        """조이스틱 D-pad 좌/우(상/하)로 녹화 시작 전 시나리오를 순환 선택 (Gradio D-pad와 동일 동작)."""
        with self._lock:
            if self.active:
                return self.staged_scenario  # 녹화 중엔 변경 불가
            keys = list(COLLECT_SCENARIOS.keys())
            if not keys:
                return None
            i = (keys.index(self.staged_scenario) + step) % len(keys) if self.staged_scenario in keys else 0
            self.staged_scenario = keys[i]
            self.collect_mode = "scenario"
            return self.staged_scenario

    def stage_cx_position(self, cx_position: Optional[str]) -> Optional[str]:
        """웹 UI(트랙A 막대 행 클릭)로 cx위치를 직접 지정 — D-pad cycle_cx_position과 동일 축 공유."""
        with self._lock:
            if self.active:
                return self.staged_cx_position
            self.staged_cx_position = cx_position or None
            self.collect_mode = "trackA"
            return self.staged_cx_position

    def cycle_cx_position(self, step: int) -> Optional[str]:
        """조이스틱 D-pad 좌/우로 녹화 시작 전 트랙A cx위치를 순환 선택."""
        with self._lock:
            if self.active:
                return self.staged_cx_position
            keys = list(COLLECT_CX_POSITIONS.keys())
            if not keys:
                return None
            i = (keys.index(self.staged_cx_position) + step) % len(keys) if self.staged_cx_position in keys else 0
            self.staged_cx_position = keys[i]
            self.collect_mode = "trackA"
            return self.staged_cx_position

    def stage_cx_path(self, cx_path: Optional[str]) -> Optional[str]:
        """웹 UI(경로 select/행 클릭)로 트랙A 접근경로(좌곡선/직진/우곡선)를 지정 — 위치와 별개 축."""
        with self._lock:
            if self.active:
                return self.staged_cx_path
            self.staged_cx_path = cx_path or None
            self.collect_mode = "trackA"
            return self.staged_cx_path

    def cycle_cx_path(self, step: int) -> Optional[str]:
        with self._lock:
            if self.active:
                return self.staged_cx_path
            keys = list(COLLECT_TRACKA_PATHS.keys())
            if not keys:
                return None
            i = (keys.index(self.staged_cx_path) + step) % len(keys) if self.staged_cx_path in keys else 0
            self.staged_cx_path = keys[i]
            self.collect_mode = "trackA"
            return self.staged_cx_path

    COLLECT_MODES = ["trackA", "cxpath", "scenario"]  # D-pad 상/하로 순환하는 축 3종

    def cycle_current(self, step: int) -> dict:
        """D-pad 좌/우 — 현재 활성 축(collect_mode)에 따라 트랙A cx위치/접근경로/시나리오 중 하나를 순환."""
        if self.collect_mode == "scenario":
            self.cycle_scenario(step)
        elif self.collect_mode == "cxpath":
            self.cycle_cx_path(step)
        else:
            self.cycle_cx_position(step)
        return {"mode": self.collect_mode, "staged_scenario": self.staged_scenario,
                "staged_cx_position": self.staged_cx_position, "staged_cx_path": self.staged_cx_path}

    def toggle_collect_mode(self, step: int = 1) -> str:
        """D-pad 상/하(또는 화면 ▲▼) — 트랙A(cx위치) → 접근경로 → 시나리오(9종) 3축을 순환 전환."""
        with self._lock:
            i = self.COLLECT_MODES.index(self.collect_mode) if self.collect_mode in self.COLLECT_MODES else 0
            self.collect_mode = self.COLLECT_MODES[(i + step) % len(self.COLLECT_MODES)]
            return self.collect_mode

    def undo_last_frame(self) -> dict:
        with self._lock:
            if not self.active or not self.episode_data:
                return {"ok": False, "steps": len(self.episode_data)}
            self.episode_data.pop()
            return {"ok": True, "steps": len(self.episode_data)}

    def start_auto_return(self) -> dict:
        """Gradio start_auto_return() 이식 — 직전 에피소드 경로를 역순+반전 액션으로
        재생해 대략 시작 위치로 되돌아간다. 저장/폐기 여부와 무관하게 동작(토글 가능)."""
        if self.active:
            return {"ok": False, "msg": "녹화 중엔 복귀 불가"}
        if self._returning:
            self._returning = False
            return {"ok": True, "msg": "복귀 중지"}
        if not self._last_episode_actions or _ros is None or _ros.ctrl is None:
            return {"ok": False, "msg": "되돌아갈 경로 없음"}

        def run():
            self._returning = True
            try:
                buf = self._last_episode_actions[:]
                rev_acts = [(-a["action"]["linear_x"], -a["action"]["linear_y"], -a["action"]["angular_z"])
                            for a in reversed(buf)]
                ts = [a["t"] for a in buf]
                dts = [max(0.05, min(ts[i + 1] - ts[i], 0.6)) for i in range(len(ts) - 1)]
                dts.append(dts[-1] if dts else 0.1)
                for act, dt in zip(rev_acts, dts):
                    if not self._returning:
                        break
                    _ros.ctrl.publish_and_move(*act, source="joystick_return")
                    time.sleep(dt)
                if self._returning:
                    _ros.ctrl.robust_stop(source="joystick_return_end")
            finally:
                self._returning = False

        threading.Thread(target=run, daemon=True, name="collect-return").start()
        return {"ok": True, "msg": "🔄 복귀 시작 — 다시 누르면 중지"}

    def stop_episode(self, save=True) -> dict:
        with self._lock:
            self.active = False
            data = self.episode_data
            name = self.episode_name
            duration = time.time() - self.episode_started_at if self.episode_started_at else 0.0
            self.episode_data = []
            if len(data) > 1:
                # 저장/폐기 여부와 무관하게 남겨둠 — Gradio와 동일하게 "복귀"는 저장 안 해도 가능
                self._last_episode_actions = [{"action": d["action"], "t": d["t"]} for d in data]
        if not save:
            self._recent_saves.insert(0, {
                "name": name, "saved": False, "reason": "사용자가 폐기(save=False)",
                "steps": len(data), "duration": duration, "ts": time.time(),
            })
            self._recent_saves = self._recent_saves[:10]
            return {"ok": True, "saved": False, "steps": len(data)}
        if len(data) <= 1:
            self._recent_saves.insert(0, {
                "name": name, "saved": False, "reason": "스텝 부족(<=1) — 저장 안 함",
                "steps": len(data), "duration": duration, "ts": time.time(),
            })
            self._recent_saves = self._recent_saves[:10]
            return {"ok": False, "saved": False, "reason": "스텝 부족(<=1) — 저장 안 함", "steps": len(data)}
        if self.STOP_INJECT_N > 0:
            last_image = data[-1]["image"]
            for _ in range(self.STOP_INJECT_N):
                data.append({
                    "image": last_image,
                    "action": {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0},
                    "action_event_type": "stop_inject",
                })
        path = self._save_episode_data(data, name, duration)
        if self.selected_scenario:
            self.scenario_stats[self.selected_scenario] += 1
        if self.selected_cx_position:
            self.cx_position_stats[self.selected_cx_position] += 1
            if self.selected_cx_path:
                self.cx_position_path_stats[f"{self.selected_cx_position}::{self.selected_cx_path}"] += 1
        self.time_period_stats[_collect_classify_time_period(datetime.datetime.now().hour)] += 1
        self._save_progress()
        log.info(f"✅ [DataCollect] 에피소드 저장: {path} ({len(data)} steps, {duration:.1f}s)")
        self._recent_saves.insert(0, {
            "name": name, "saved": True, "path": str(path),
            "steps": len(data), "duration": duration, "ts": time.time(),
        })
        self._recent_saves = self._recent_saves[:10]
        hz = (len(data) - 1) / duration if duration > 0 else 0.0
        self._last_session_summary = {
            "frames": len(data), "duration_s": round(duration, 2), "hz": round(hz, 2),
        }
        return {"ok": True, "saved": True, "path": str(path), "steps": len(data), "duration": duration}

    def _save_episode_data(self, data: List[dict], name: str, duration: float) -> Path:
        images = np.stack([d["image"] for d in data])
        actions = np.array(
            [[d["action"]["linear_x"], d["action"]["linear_y"], d["action"]["angular_z"]] for d in data],
            dtype=np.float32,
        )
        event_types = np.array([d["action_event_type"] for d in data], dtype=h5py.string_dtype(encoding="utf-8"))
        save_path = self.data_dir / f"{name}.h5"
        now = datetime.datetime.now()
        with h5py.File(save_path, "w") as f:
            f.attrs["episode_name"] = name
            f.attrs["scenario"] = self.selected_scenario or ""
            f.attrs["cx_position"] = self.selected_cx_position or ""
            f.attrs["cx_path"] = self.selected_cx_path or ""
            f.attrs["total_duration"] = duration
            f.attrs["num_frames"] = len(data)
            f.attrs["stop_inject_n"] = self.STOP_INJECT_N
            f.attrs["action_chunk_size"] = self.ACTION_CHUNK_SIZE
            f.attrs["obstacle_layout_type"] = self.DEFAULT_LAYOUT_TYPE
            f.attrs["time_period"] = _collect_classify_time_period(now.hour)
            f.attrs["collection_datetime"] = now.isoformat()
            f.attrs["collection_hour"] = now.hour
            f.attrs["collection_minute"] = now.minute
            f.create_dataset("images", data=images, compression="gzip")
            f.create_dataset("actions", data=actions, compression="gzip")
            f.create_dataset("action_event_types", data=event_types, compression="gzip")
        return save_path

    def _load_progress(self):
        try:
            if self.progress_file.exists():
                d = json.loads(self.progress_file.read_text(encoding="utf-8"))
                for k, v in d.get("scenario_stats", {}).items():
                    self.scenario_stats[k] = v
                for k, v in d.get("cx_position_stats", {}).items():
                    self.cx_position_stats[k] = v
                for k, v in d.get("cx_position_path_stats", {}).items():
                    self.cx_position_path_stats[k] = v
        except Exception as e:
            log.warning(f"[DataCollect] scenario_progress.json 로드 실패: {e}")
        try:
            if self.time_period_file.exists():
                d = json.loads(self.time_period_file.read_text(encoding="utf-8"))
                for k, v in d.get("time_period_stats", {}).items():
                    self.time_period_stats[k] = v
        except Exception as e:
            log.warning(f"[DataCollect] time_period_stats.json 로드 실패: {e}")
        try:
            if self.core_pattern_file.exists():
                self.core_patterns = json.loads(self.core_pattern_file.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning(f"[DataCollect] core_patterns.json 로드 실패: {e}")

    def _save_progress(self):
        try:
            self.progress_file.write_text(json.dumps({
                "last_updated": datetime.datetime.now().isoformat(),
                "scenario_stats": dict(self.scenario_stats),
                "total_completed": sum(self.scenario_stats.values()),
                "total_target": sum(s["target"] for s in COLLECT_SCENARIOS.values()),
                "cx_position_stats": dict(self.cx_position_stats),
                "cx_position_path_stats": dict(self.cx_position_path_stats),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            log.warning(f"[DataCollect] scenario_progress.json 저장 실패: {e}")
        try:
            self.time_period_file.write_text(json.dumps({
                "last_updated": datetime.datetime.now().isoformat(),
                "time_period_stats": dict(self.time_period_stats),
                "total_completed": sum(self.time_period_stats.values()),
                "total_target": 1000,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as e:
            log.warning(f"[DataCollect] time_period_stats.json 저장 실패: {e}")

    def _episode_timeline(self) -> dict:
        """현재 에피소드 버퍼의 최근 액션 기호열 + 실측 Hz + 8-class 분포
        (Gradio episode_timeline_md/episode_dist_md 이식)."""
        buf = self.episode_data
        n = len(buf)
        if n == 0:
            return {"n": 0, "symbols": "", "hz": 0.0, "dist": {}}
        symbols = "".join(COLLECT_CLASS_SYMBOLS[_collect_classify_8class(d["action"])] for d in buf[-28:])
        ts = [d["t"] for d in buf[-12:] if "t" in d]
        hz = 0.0
        if len(ts) >= 2:
            dts = [ts[i + 1] - ts[i] for i in range(len(ts) - 1)]
            mean_dt = sum(dts) / len(dts)
            if mean_dt > 0:
                hz = round(1.0 / mean_dt, 1)
        dist: Dict[str, int] = collections.defaultdict(int)
        for d in buf:
            cls = _collect_classify_8class(d["action"])
            dist[COLLECT_CLASS_NAMES_8[cls]] += 1
        return {"n": n, "symbols": symbols, "hz": hz, "dist": dict(dist)}

    def state(self) -> dict:
        cur_scenario = self.selected_scenario or self.staged_scenario
        scenario_target = COLLECT_SCENARIOS.get(cur_scenario, {}).get("target", 0) if cur_scenario else 0
        return {
            "active": self.active,
            "episode_name": self.episode_name,
            "episode_started_at": self.episode_started_at,
            "steps": len(self.episode_data),
            "recent_saves": self._recent_saves[:10],
            "last_session_summary": self._last_session_summary,
            "episode_timeline": self._episode_timeline(),
            "current_scenario_target": scenario_target,
            "scenario": self.selected_scenario,
            "pattern": self.selected_pattern,
            "staged_scenario": self.staged_scenario,
            "scenarios": COLLECT_SCENARIOS,
            "patterns": COLLECT_PATTERNS,
            "scenario_stats": dict(self.scenario_stats),
            "total_target": sum(s["target"] for s in COLLECT_SCENARIOS.values()),
            "total_completed": sum(self.scenario_stats.values()),
            "cx_position": self.selected_cx_position,
            "cx_positions": COLLECT_CX_POSITIONS,
            "cx_position_stats": dict(self.cx_position_stats),
            "staged_cx_position": self.staged_cx_position,
            "cx_paths": COLLECT_TRACKA_PATHS,
            "cx_path": self.selected_cx_path,
            "staged_cx_path": self.staged_cx_path,
            "cx_position_path_stats": dict(self.cx_position_path_stats),
            "collect_mode": self.collect_mode,
            "returning": self._returning,
            "has_return_path": len(self._last_episode_actions) > 1,
        }


_collect: Optional[DataCollectSession] = None

# 프로세스 시작 시각/PID — 좀비 프로세스 감지 및 UI 신선도 표시용
# (2026-07-02: go.sh 재시작 시 SIGTERM만으론 안 죽는 프로세스가 좀비로 남아
#  옛 MJPEG 스트림을 계속 서빙하던 사고가 있었음 — 재발 시 바로 알아채기 위함)
_PROCESS_PID = os.getpid()
_PROCESS_START_TS = time.time()


# ═══════════════════════════════════════════════════════════════════
# 조이스틱 (DragonRise 게임패드) — Gradio 대시보드 DashboardJoystickReader 이식
# ═══════════════════════════════════════════════════════════════════
class DashboardJoystickReader:
    """DragonRise 게임패드로 대시보드 로봇을 직접 제어.

    버튼 매핑은 연결 시 _detect_layout()이 자동 감지(dragonrise/controller) 후
    scripts/joystick_config.json의 실측값으로 최종 덮어씀 (Gradio
    gradio_data_collector.py JoystickReader와 동일 로직). 하드코딩된 인덱스로
    가정하지 말 것 — 실제 패드에서 버튼이 다르게 잡히면 joystick_config.json만
    고치면 됨.

    SYNC 모드: 0.45s 간격으로 move_and_stop_timed() — V5 bang-bang 호환
    ASYNC 모드: 10Hz 연속 publish_and_move() + 300ms Jitter Hold + 중립 시 robust_stop()
    """

    DEADZONE       = 0.15
    THRESHOLD      = 0.50
    STEP_INTERVAL  = 0.45   # SYNC bang-bang 간격 (s)
    ASYNC_INTERVAL = 0.10   # ASYNC 연속 발행 간격 (s) — 10Hz
    JITTER_HOLD    = 0.30   # ASYNC 중립 후 정지 유예 시간 (s)
    DEFAULT_AXES   = {"left_x": 0, "left_y": 1, "right_x": 2}
    # 클래스 기본값(연결 전 폴백) — 실제 값은 연결 시 _detect_layout()/_apply_button_config()가
    # 인스턴스 속성으로 덮어씀 (Gradio gradio_data_collector.py JoystickReader 이식,
    # joystick_config.json 실측 매핑까지 동일하게 반영).
    BTN_STOP       = 0   # A     — STOP (robust_stop)
    BTN_UNDO       = 1   # B     — 마지막 프레임 취소
    BTN_DISCARD    = 2   # X     — 에피소드 폐기(저장 안 함)
    BTN_TELEOP     = 3   # Y     — 미사용 (대시보드엔 teleop 개념 없음)
    BTN_REC_START  = 4   # L1    — 녹화 시작
    BTN_REC_SAVE   = 5   # R1    — 정지 & 저장
    BTN_SELECT     = 6   # Select — 녹화 토글(시작↔저장)
    BTN_TOGGLE     = 7   # Start  — SYNC/ASYNC 모드 전환
    BTN_L2         = -1  # DragonRise일 때 버튼(레이아웃 감지 전 폴백)
    BTN_R2         = -1  # DragonRise일 때 버튼 — 경로 역재생 복귀
    TRIG_L2        = 4   # Controller일 때 트리거 축
    TRIG_R2        = 5   # Controller일 때 트리거 축 — 경로 역재생 복귀
    layout         = "controller"
    TRIG_THRESHOLD = 0.30

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
        self._speed    = 1.15
        self._thread   = None
        self._btn_prev = {}
        self._last_step_time = 0.0
        self._prev_key = None
        self._neutral_start_time = 0.0
        self._last_non_neutral_key = None
        self._axes = self._load_axes()
        self._last_btn = None
        self._hat_prev = (0, 0)  # D-pad 엣지 검출용 (시나리오 넘기기)
        self._last_hat_dir = None  # 마지막으로 눌린 D-pad 방향 ("left"/"right"/"up"/"down")
        self._trig_r2_prev = -1.0  # R2 트리거 엣지 검출용 (복귀)
        self._trig_l2_prev = -1.0  # L2 트리거 엣지 검출용 (검증 세션 저장)
        self._btn_map: dict = {}
        self.status: dict = {
            "connected": False, "name": "—",
            "key": None, "label": "—",
            "enabled": True, "mode": "ASYNC",
            "buttons": [], "last_btn": None,
            "hat": (0, 0), "last_hat_dir": None,
            "btn_map": {},
        }

    def _load_axes(self):
        cfg = ROOT / "scripts" / "joystick_config.json"
        if cfg.exists():
            try:
                return json.load(open(cfg)).get("axes", self.DEFAULT_AXES)
            except Exception:
                pass
        return dict(self.DEFAULT_AXES)

    def _detect_layout(self, js):
        """연결 패드를 보고 버튼 매핑 자동 결정 (Gradio gradio_data_collector.py
        JoystickReader._detect_layout 이식). 공통: STOP=0, UNDO=1, DISCARD=2,
        REC_START=4, REC_SAVE=5. 차이: L2/R2 — DragonRise=버튼6/7,
        Controller=트리거축4/5. SELECT/START 번호도 다름. 마지막으로
        joystick_config.json의 실측값이 있으면 그걸로 덮어씀."""
        name = js.get_name().lower()
        nbtn, nax = js.get_numbuttons(), js.get_numaxes()
        is_dragon = ("dragon" in name or "generic" in name or "usb gamepad" in name
                     or (nbtn <= 12 and nax <= 5))
        self.BTN_STOP, self.BTN_UNDO, self.BTN_DISCARD = 0, 1, 2
        self.BTN_TELEOP = 3
        self.BTN_REC_START, self.BTN_REC_SAVE = 4, 5
        if is_dragon:
            self.layout = "dragonrise"
            self.BTN_L2, self.BTN_R2 = 6, 7
            self.BTN_SELECT, self.BTN_TOGGLE = 8, 9
            self.TRIG_L2, self.TRIG_R2 = -1, -1
        else:
            self.layout = "controller"
            self.BTN_L2, self.BTN_R2 = -1, -1
            self.BTN_SELECT, self.BTN_TOGGLE = 6, 7
            self.TRIG_L2, self.TRIG_R2 = 4, 5
        self._apply_button_config()
        log.info(f"[Joystick] 레이아웃={self.layout} STOP={self.BTN_STOP} UNDO={self.BTN_UNDO} "
                 f"DISCARD={self.BTN_DISCARD} REC_START={self.BTN_REC_START} REC_SAVE={self.BTN_REC_SAVE} "
                 f"SELECT={self.BTN_SELECT} TOGGLE={self.BTN_TOGGLE} L2={self.BTN_L2} R2={self.BTN_R2} "
                 f"TRIG_R2={self.TRIG_R2}")
        # 실제 감지된 인덱스로 프런트 라벨을 만들어 status에 실어보냄 —
        # 하드코딩 라벨(0=A,1=B...)이 실측 매핑과 어긋나 혼동을 주지 않도록.
        btn_map = {
            self.BTN_STOP: {"name": "STOP", "desc": "STOP (robust_stop)"},
            self.BTN_UNDO: {"name": "UNDO", "desc": "마지막 프레임 취소"},
            self.BTN_DISCARD: {"name": "DISCARD", "desc": "에피소드 폐기"},
            self.BTN_TELEOP: {"name": "Y", "desc": "🔁 조이스틱 모드 전환(수집⇄검증)"},
            self.BTN_REC_START: {"name": "L1", "desc": "녹화 시작"},
            self.BTN_REC_SAVE: {"name": "R1", "desc": "정지 & 저장"},
            self.BTN_SELECT: {"name": "SEL", "desc": "녹화 토글(수집)"},
            self.BTN_TOGGLE: {"name": "START", "desc": "SYNC↔ASYNC 모드"},
        }
        if self.BTN_L2 >= 0:
            btn_map[self.BTN_L2] = {"name": "L2", "desc": "💾 기록 저장(Log Episode)"}
        if self.BTN_R2 >= 0:
            btn_map[self.BTN_R2] = {"name": "R2", "desc": "복귀(경로 역재생)"}
        self._btn_map = {str(k): v for k, v in btn_map.items()}

    def _apply_button_config(self):
        """joystick_config.json의 "buttons" 실측값으로 인덱스 덮어쓰기
        (Gradio JoystickReader._apply_button_config 이식) — 코드 수정 없이
        실측 후 교정 가능하도록."""
        cfg_path = ROOT / "scripts" / "joystick_config.json"
        if not cfg_path.exists():
            return
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
        except Exception:
            return
        if cfg.get("force_layout"):
            self.layout = cfg["force_layout"]
        b = cfg.get("buttons", {})
        keymap = {
            "stop": "BTN_STOP", "undo": "BTN_UNDO", "discard": "BTN_DISCARD",
            "teleop": "BTN_TELEOP", "rec_start": "BTN_REC_START", "rec_save": "BTN_REC_SAVE",
            "select": "BTN_SELECT", "start": "BTN_TOGGLE",
            "l2": "BTN_L2", "r2": "BTN_R2",
            "trig_l2": "TRIG_L2", "trig_r2": "TRIG_R2",
        }
        for k, attr in keymap.items():
            if k in b:
                setattr(self, attr, int(b[k]))

    def start(self):
        if not PYGAME_AVAILABLE:
            log.warning("[Joystick] pygame 없음 — pip install pygame")
            return
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="joystick")
        self._thread.start()

    def toggle_enabled(self) -> bool:
        self._enabled = not self._enabled
        self.status = {**self.status, "enabled": self._enabled}
        log.info(f"[Joystick] {'활성화' if self._enabled else '비활성화'}")
        return self._enabled

    def toggle_mode(self) -> str:
        self._js_mode = 'async' if self._js_mode == 'sync' else 'sync'
        self.status = {**self.status, "mode": self._js_mode.upper()}
        log.info(f"[Joystick] 모드 전환 → {self._js_mode.upper()}")
        return self._js_mode.upper()

    def set_speed(self, spd: float):
        self._speed = float(spd)
        if _ros is not None and _ros.ctrl is not None:
            throttle = int(round(self._speed / 1.15 * 50))
            throttle = max(10, min(100, throttle))
            _ros.ctrl.throttle = throttle
            _ros.ctrl.rot_throttle = throttle * 0.35  # 회전은 항상 직진의 35% 유지 (2026-07-15 추가 30% 축소)

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
            log.warning(f"[Joystick] pygame init 실패: {e}")
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
                self._detect_layout(js)
                self.status = {**self.status, "connected": True, "name": js.get_name()}
                self._btn_prev = {i: 0 for i in range(js.get_numbuttons())}
                log.info(f"[Joystick] 연결됨: {js.get_name()}")

            try:
                # USB 재연결(다른 로봇에 꽂았다가 같은 포트로 복귀 등) 감지 — 핸들이 죽은 채로
                # 예외 없이 0만 반환하는 걸 막기 위해 핫플러그 이벤트로 강제 재초기화한다.
                hotplugged = False
                for ev in pygame.event.get():
                    if ev.type in (pygame.JOYDEVICEREMOVED, pygame.JOYDEVICEADDED):
                        hotplugged = True
                if hotplugged:
                    log.info("[Joystick] 핫플러그 이벤트 — 재초기화")
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

                l_moving = abs(lx) > self.DEADZONE or abs(ly) > self.DEADZONE
                az_blend = az if (l_moving and key not in ('R', 'T')) else 0.0

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
                if self._enabled and _ros is not None and _ros.ctrl is not None:
                    ctrl = _ros.ctrl
                    if key:
                        base = self.WASD_TO_VEL.get(key)
                        if base:
                            spd = self._speed / 1.15
                            if az_blend != 0.0:
                                base = (base[0], base[1], az_blend * 0.15)
                            vel = tuple(v * spd for v in base)
                            if self._js_mode == 'sync':
                                if (now - self._last_step_time) >= self.STEP_INTERVAL:
                                    ctrl.move_and_stop_timed(*vel, source="joystick")
                                    self._last_step_time = now
                            else:  # async
                                if (now - self._last_step_time) >= self.ASYNC_INTERVAL:
                                    ctrl.publish_and_move(*vel, source="joystick")
                                    self._last_step_time = now
                    elif self._prev_key:
                        if self._js_mode == 'async':
                            ctrl.publish_and_move(0.0, 0.0, 0.0, source="joystick_neutral")

                self._prev_key = key
                pressed_buttons = [i for i in range(js.get_numbuttons()) if js.get_button(i)]
                _nax = js.get_numaxes()
                _trig_l2_val = round(js.get_axis(self.TRIG_L2), 2) if 0 <= self.TRIG_L2 < _nax else None
                _trig_r2_val = round(js.get_axis(self.TRIG_R2), 2) if 0 <= self.TRIG_R2 < _nax else None
                self.status = {
                    "connected": True, "name": js.get_name(),
                    "enabled": self._enabled,
                    "mode": self._js_mode.upper(),
                    "key": key, "label": LABELS.get(key, "●") if key else "○",
                    "buttons": pressed_buttons, "last_btn": self._last_btn,
                    "hat": self._hat_prev, "last_hat_dir": self._last_hat_dir,
                    "btn_map": self._btn_map,
                    "num_buttons": js.get_numbuttons(), "num_axes": _nax,
                    "trig_l2_val": _trig_l2_val, "trig_r2_val": _trig_r2_val,
                    "verify_mode": _joystick_verify_mode,
                    "verify_screen_pos": _verify_screen_pos,
                    "verify_pending_result": _verify_pending_result,
                    "verify_save_seq": _verify_save_seq,
                    "verify_experimental": _verify_experimental,
                }

                for i in range(js.get_numbuttons()):
                    cur = js.get_button(i)
                    if cur and not self._btn_prev.get(i, 0):
                        self._last_btn = i
                        if i == self.BTN_TOGGLE:
                            self.toggle_mode()
                        elif i == self.BTN_TELEOP:
                            # Y(3) = 조이스틱 배치모드 전환(collect⇄verify) — 주행/녹화 중엔 무시.
                            _joystick_toggle_verify_mode()
                        elif _joystick_verify_mode:
                            # 🧪 경로검증 모드 배치 (2026-07-23 재설계):
                            # L1=추론시작 · R1=추론정지 · R2=복귀 · X=성공라벨 · A=실패라벨
                            # · L2=최종 세션 저장 · D-pad◀▶=위치 순환
                            # X/A는 즉시기록 아님(라벨만) → L2에서 1건 저장(중복 스팸 방지).
                            # ⚠️ 검증모드엔 별도 비상정지 없음 — R1(추론정지=robust_stop)로 대체.
                            if i == self.BTN_REC_START:      # L1 = 추론 시작
                                _joystick_drive_start()
                            elif i == self.BTN_REC_SAVE:     # R1 = 추론 정지
                                _joystick_drive_stop()
                            elif i == self.BTN_DISCARD:      # X = 성공 라벨
                                _verify_arm_result("성공")
                            elif i == self.BTN_STOP:         # A = 실패 라벨
                                _verify_arm_result("실패")
                            elif self.BTN_L2 >= 0 and i == self.BTN_L2:
                                # 💾 기록 저장 (Log Episode) — L2 버튼(인덱스 8)
                                _verify_save_session()
                            elif self.BTN_R2 >= 0 and i == self.BTN_R2:   # R2 버튼(있는 패드만) = 복귀
                                _joystick_verify_return()
                            elif i == self.BTN_SELECT:       # SEL = 실험용/테스트 모드 토글
                                _joystick_toggle_verify_experimental()
                        else:
                            # 📷 데이터수집 모드: A = 비상정지(모드 전용 유지)
                            if i == self.BTN_STOP:
                                if _ros is not None and _ros.ctrl is not None:
                                    _ros.ctrl.robust_stop(source="joystick_A")
                            # 📷 데이터수집 모드 배치(기존 그대로, 변경 없음)
                            elif i == self.BTN_UNDO:
                                if _collect is not None:
                                    _collect.undo_last_frame()
                            elif i == self.BTN_DISCARD:
                                if _collect is not None and _collect.active:
                                    _collect.stop_episode(save=False)
                            elif i == self.BTN_REC_START:
                                if _collect is not None and not _collect.active:
                                    _collect.start_episode(scenario=_collect.staged_scenario,
                                                            cx_position=_collect.staged_cx_position,
                                                            cx_path=_collect.staged_cx_path)
                            elif i == self.BTN_REC_SAVE:
                                if _collect is not None and _collect.active:
                                    _collect.stop_episode(save=True)
                            elif i == self.BTN_SELECT:
                                if _collect is not None:
                                    if _collect.active:
                                        _collect.stop_episode(save=True)
                                    else:
                                        _collect.start_episode(scenario=_collect.staged_scenario,
                                                            cx_position=_collect.staged_cx_position,
                                                            cx_path=_collect.staged_cx_path)
                            elif self.BTN_R2 >= 0 and i == self.BTN_R2:
                                # DragonRise는 R2가 트리거 축이 아니라 버튼으로 잡힘 (joystick_config.json 실측)
                                if _collect is not None:
                                    _collect.start_auto_return()
                    self._btn_prev[i] = cur

                # D-pad(hat) 엣지 → 좌/우: 현재 활성 축(collect_mode) 순환, 상/하: 트랙A↔시나리오 축 전환
                if js.get_numhats() > 0:
                    hat = js.get_hat(0)
                    if hat != self._hat_prev:
                        hx, hy = hat
                        phx, phy = self._hat_prev
                        if hx != 0 and hx != phx:
                            self._last_hat_dir = "right" if hx > 0 else "left"
                            if _joystick_verify_mode:
                                # 🧪 검증모드: D-pad ◀▶ = 스크리닝 위치 순환(수집 cx 순환과 동일 메커니즘)
                                _verify_cycle_pos(1 if hx > 0 else -1)
                            elif _collect is not None:
                                _collect.cycle_current(1 if hx > 0 else -1)
                        elif hy != 0 and hy != phy:
                            self._last_hat_dir = "up" if hy > 0 else "down"
                            if not _joystick_verify_mode and _collect is not None:
                                # 상/하 축 전환은 수집 모드 전용(검증엔 위치 축 하나뿐)
                                _collect.toggle_collect_mode(1 if hy > 0 else -1)
                        self._hat_prev = hat

                # R2(트리거, 축) 엣지 → 복귀(경로 역재생), Gradio "controller" 레이아웃과 동일 (두 모드 공통)
                nax = js.get_numaxes()
                if 0 <= self.TRIG_R2 < nax:
                    tv = js.get_axis(self.TRIG_R2)
                    if tv > self.TRIG_THRESHOLD and self._trig_r2_prev <= self.TRIG_THRESHOLD:
                        if _joystick_verify_mode:
                            _joystick_verify_return()       # 주행 경로 역재생
                        elif _collect is not None:
                            _collect.start_auto_return()    # 수집 경로 역재생
                    self._trig_r2_prev = tv

                # L2(트리거, 축) 엣지 → 검증모드에서만 '최종 세션 저장'(추론세션 저장).
                # Controller 레이아웃은 L2가 트리거축(TRIG_L2), DragonRise는 버튼(BTN_L2, 위 버튼루프서 처리).
                if 0 <= self.TRIG_L2 < nax:
                    tv = js.get_axis(self.TRIG_L2)
                    if tv > self.TRIG_THRESHOLD and self._trig_l2_prev <= self.TRIG_THRESHOLD:
                        if _joystick_verify_mode:
                            _verify_save_session()
                    self._trig_l2_prev = tv

            except Exception as e:
                log.warning(f"[Joystick] 루프 오류: {e}")
                js = None
                self.status = {**self.status, "connected": False}

            time.sleep(0.04)  # 25 Hz


_joystick = DashboardJoystickReader()
_joystick.start()

_state: dict[str, Any] = {
    "running": False,
    "mode": "ASYNC",
    "step": 0,
    "instruction": "gray basket",
    "gt_object": "",
    "apply_cc": False,
    "last_action": [0.0, 0.0, 0.0],
    "predicted_label": None,
    "latency_ms": 0,
    "goal_near": False,
    "status_log": "대기 중",
    "bbox": None,
    "chunk": None,
    "grounding_cached": None,
    "grounding_caption": None,
    "run_history": [],          # [[step, label, total_ms, gnd_ms, mlp_ms, area], ...]
    "action_history": [],       # [[lx, ly, az], ...] 주행 액션 이력
    "session_id": None,
    "is_returning": False,

    # 캘리브레이션 녹화 데이터
    "calib_recording": False,
    "calib_frames": [],         # [{n, area, cx, cy, latency_ms, stop, ts, note}, ...]
    "calib_imgs": [],           # np.ndarray list
    "calib_session_name": "",

    # 디버그용 드리프트 데이터
    "drift_running": False,
    "drift_frames": 0,
    "drift_cum_real": 0.0,
    "drift_cum_nom": 0.0,
    "drift_history": [],        # [[frame, latency_ms, cum_real, cum_nom, drift], ...]
    "drift_basis": "1.0s (1fps 운영)",
}

_stop_ev   = threading.Event()
_async_q: collections.deque = collections.deque(maxlen=2)
_goalnav_q: collections.deque = collections.deque(maxlen=2)

# ── 조이스틱 버튼 배치 모드(2026-07-22) — 📷 데이터수집 vs 🧪 경로검증 ──
# 기본값 False(=collect) 유지 — 기존 동작 100% 불변. Y(3) 또는 UI 토글로 전환.
_joystick_verify_mode: bool = False
# 조이스틱 X/R1(경로검증 모드에서 성공/실패 즉시기록)이 쓸 path_type —
# Tab4 select onchange가 /verify/path_type으로 동기화해줌.
_verify_current_path_type: str = "trackA_weak_left_left_curve"

# ── 검증 스크리닝 세션 상태(2026-07-23) — D-pad로 위치 순환 + X/A로 결과 라벨,
# L2로 최종 저장하는 방식. X/A는 즉시 기록 안 하고 라벨만 세팅(중복 스팸 방지). ──
# 순서/키는 JS SCREEN_POSITIONS와 일치. center는 trackF_, 나머지는 trackA_로 저장.
VERIFY_SCREEN_POSITIONS = ["strong_left", "weak_left", "center", "weak_right", "strong_right"]
_verify_screen_pos: str = "strong_left"       # D-pad ◀▶로 순환
_verify_pending_result: Optional[str] = None  # "성공" | "실패" | None (X/A로 세팅)
# L2(조이스틱)로 저장할 때마다 +1 — 브라우저가 /joystick/status 폴링 중 이 값이
# 바뀐 걸 보면 loadEpisodeHistory()를 조용히 호출(팝업 없음, 스크리닝 패널 자동 갱신).
_verify_save_seq: int = 0

# 🧪 실험용/테스트 모드(2026-07-23) — 검증모드 SEL(남는 키)로 토글. True인 동안
# L2 저장은 episode_log.csv 대신 episode_log_experimental.csv로 기록되어
# 그라운더 A/B 테스트 등 "정식 스크리닝 집계에 안 셀" 시도를 분리 보관.
_verify_experimental: bool = False


def _verify_pos_to_path_type(pos: str) -> str:
    return "trackF_center" if pos == "center" else f"trackA_{pos}"


def _verify_cycle_pos(step: int):
    """D-pad ◀▶ — 검증 스크리닝 위치 순환(강좌↔약좌↔중앙↔약우↔강우)."""
    global _verify_screen_pos
    if _state.get("running"):
        return  # 주행 중엔 변경 금지
    keys = VERIFY_SCREEN_POSITIONS
    i = (keys.index(_verify_screen_pos) + step) % len(keys) if _verify_screen_pos in keys else 0
    _verify_screen_pos = keys[i]
    log.info(f"[Verify] D-pad → 위치 = {_verify_screen_pos}")


def _verify_arm_result(result: str):
    """X=성공 / A=실패 — 즉시 기록 안 하고 결과 라벨만 세팅(L2에서 저장)."""
    global _verify_pending_result
    _verify_pending_result = result
    log.info(f"[Verify] 결과 라벨 → {result} (L2로 저장 대기)")


_drift_log_file = None


# ═══════════════════════════════════════════════════════════════════
# FastAPI 앱
# ═══════════════════════════════════════════════════════════════════
app = FastAPI(title="MoNaVLA Command Center", version="2.5")

STATIC_DIR = Path(__file__).parent / "static"


@app.on_event("startup")
def _startup():
    global _ros, _collect
    log.info(f"🆔 프로세스 시작 PID={_PROCESS_PID}")
    _warn_if_duplicate_process()

    collect_dir = Path(os.getenv("VLA_DATASET_DIR", str(ROOT / "ROS_action" / "mobile_vla_dataset_v5")))
    _collect = DataCollectSession(collect_dir)
    log.info(f"📷 [DataCollect] 데이터 디렉토리: {collect_dir}")

    if not ROS_AVAILABLE:
        log.warning("ROS 없음 — camera/control 비활성")
        return
    if not rclpy.ok():
        rclpy.init()
    _ros = MoNaROSNode()
    _ros.ctrl.on_command = _collect.on_command
    threading.Thread(target=lambda: rclpy.spin(_ros), daemon=True, name="ros-spin").start()
    log.info("✅ ROS spin 시작")


def _warn_if_duplicate_process():
    """같은 스크립트를 서빙하는 다른 프로세스가 이미 떠 있는지 확인 —
    2026-07-02 좀비 프로세스 사고(옛 프로세스가 안 죽고 MJPEG 스트림을
    계속 물고 있던 문제) 재발 감지용. 자동으로 죽이지는 않음 — 자기 자신을
    잘못 종료할 위험이 있어 경고만 남기고 판단은 사람이 하도록 함."""
    try:
        import subprocess as _sp
        out = _sp.run(["pgrep", "-f", "robovlm_nav/serve/mona_dashboard.py"],
                       capture_output=True, text=True, timeout=3).stdout
        pids = [p for p in out.split() if p and int(p) != _PROCESS_PID]
        if pids:
            log.warning(f"⚠️ 중복 프로세스 감지! PID={pids} — 이전 재시작이 완전히 "
                        f"종료되지 않았을 수 있음. 필요시 수동 확인: kill -9 {' '.join(pids)}")
    except Exception as e:
        log.debug(f"중복 프로세스 확인 실패(무시): {e}")


# ─── 유틸 ─────────────────────────────────────────────────────────
def _infer_post(path: str, payload: dict, timeout=15, base_url: str = INFER_URL) -> dict:
    import requests as rq
    r = rq.post(f"{base_url}{path}", json=payload,
                headers={"X-API-Key": API_KEY}, timeout=timeout)
    r.raise_for_status()
    return r.json()


# ── exp73 GoalNav 전용: PG2(PaliGemma2-448) 그라운더 재사용 ──
# stage2_v2_inference_server.py의 PG2Grounder를 그대로 import — exp73 학습
# 라벨(gen_pg448_annotation.py)과 동일 코드 경로 보장(재구현 금지, 2026-07-22 결정).
# 단, 라이브 PG2Grounder는 학습 annotation에 없던 x-full-width 필터가 추가돼 있어
# (stage2_v2_inference_server.py:602) 학습 분포와 미세하게 다를 수 있음 — 실기에서
# 이상 탐지(has_bbox=False 과다 등)가 잦으면 이 필터 비활성화를 디버깅 옵션으로 고려.
_pg2_grounder = None


def _get_pg2_grounder():
    global _pg2_grounder
    if _pg2_grounder is None:
        import torch
        from robovlm_nav.serve.stage2_v2_inference_server import PG2Grounder, DEFAULT_PG2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _pg2_grounder = PG2Grounder(DEFAULT_PG2, device)
        log.info(f"[GoalNav] PG2Grounder 준비(lazy load): {DEFAULT_PG2}")
    return _pg2_grounder


def _append_history(step, result):
    total = result.get("latency_ms") or 0.0
    gnd   = result.get("grounding_latency_ms")
    mlp   = (total - gnd) if gnd is not None else None
    bbox  = result.get("bbox") or {}
    _state["run_history"].append([
        step,
        result.get("predicted_label") or "—",
        round(total),
        round(gnd) if gnd is not None else "—",
        round(mlp) if mlp is not None else "—",
        round(bbox.get("area", 0.0), 3) if bbox else "—",
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ])
    _state["run_history"] = _state["run_history"][-30:]


def _record_calib_frame(img_pil: Image.Image, res: dict):
    if not _state["calib_recording"]:
        return
    frames = _state["calib_frames"]
    n = len(frames) + 1
    bbox = res.get("bbox") or {}
    area = bbox.get("area", 0.0) if bbox else 0.0
    cx = bbox.get("cx", 0.5) if bbox else 0.5
    cy = bbox.get("cy", 0.5) if bbox else 0.5
    stop_triggered = res.get("goal_near", False)
    
    frames.append({
        "n": n,
        "area": round(area, 4),
        "cx": round(cx, 3),
        "cy": round(cy, 3),
        "latency_ms": round(res.get("latency_ms", 0.0), 1),
        "stop": "Y" if stop_triggered else "N",
        "ts": datetime.datetime.now().strftime("%H:%M:%S"),
        "note": res.get("predicted_label", "") or ""
    })
    _state["calib_frames"] = frames
    
    # 램 OOM 방지를 위해 최대 120장까지만 비디오 캡처 저장
    imgs = _state["calib_imgs"]
    if len(imgs) < 120:
        imgs.append(np.array(img_pil))
        _state["calib_imgs"] = imgs


# ═══════════════════════════════════════════════════════════════════
# 주행 루프 — SYNC / PRE
# ═══════════════════════════════════════════════════════════════════
def _loop_sync(mode: str, instr: str, gt_obj: str, apply_cc: bool):
    from scripts.inference_logger import get_logger
    logger = get_logger()
    logger.start_session("stage2_v2", instr, instruction_mode=mode)
    if gt_obj:
        logger.data["gt_object"] = gt_obj
    logger.data["apply_cc"] = apply_cc
    logger.data["runtime_config"] = _snapshot_runtime_config()
    _state["session_id"] = logger.session_id

    _ros.ctrl.robust_stop(source="start")
    time.sleep(0.20)
    # 이전 세션 마지막 스텝 이후(로그에는 안 남는 "정착 프레임")가 여기 남아있으면
    # 새 세션 첫 스텝이 그 이전 세션 꼬리 프레임을 그대로 읽어버림 — 반드시 초기화.
    _ros._stable = None
    _reset_ok = False
    for _attempt in range(2):
        try:
            _infer_post("/reset", {}, timeout=5)
            _reset_ok = True
            break
        except Exception as _e:
            print(f"⚠️ [SYNC] /reset 실패 (attempt {_attempt+1}/2): {_e}")
    if not _reset_ok:
        _state["status_log"] = "⚠️ /reset 실패 — 이전 세션 grounding 상태가 남아있을 수 있음"

    step = 0
    while not _stop_ev.is_set() and _state["running"]:
        step += 1
        _state["step"] = step

        bgr = (_ros._stable if mode == "SYNC" and _ros._stable is not None
               else _ros.latest_bgr())
        if bgr is None:
            step -= 1; time.sleep(0.05); continue
        _ros._stable = None

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        if apply_cc: img = _correct(img)

        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()

        try:
            res = _infer_post("/predict",
                              {"image": b64, "instruction": instr,
                               "strategy": "receding_horizon"})
        except Exception as e:
            _state["status_log"] = f"추론 에러: {e}"
            time.sleep(0.5); continue

        action = res.get("action_3d") or res.get("action", [0.0, 0.0, 0.0])
        lx, ly = float(action[0]), float(action[1])
        az = float(action[2]) if len(action) > 2 else 0.0

        # 실시간 액션 히스토리에 기록
        _state["action_history"].append([lx, ly, az])

        log_msg = _ros.ctrl.move_and_stop_ramped(lx, ly, az, source=f"{mode.lower()}")
        _state["status_log"]     = log_msg
        _state["last_action"]    = [lx, ly, az]
        _state["predicted_label"]= res.get("predicted_label")
        _state["latency_ms"]     = res.get("latency_ms", 0)
        _state["goal_near"]      = res.get("goal_near", False)
        _state["bbox"]           = res.get("bbox")
        _state["chunk"]          = res.get("chunk")
        _state["grounding_cached"] = res.get("grounding_cached")
        _state["grounding_caption"] = res.get("grounding_caption")
        _append_history(step, res)

        # 캘리브레이션 데이터 기록
        if _state["calib_recording"]:
            _record_calib_frame(img, res)

        logger.log_step(step, np.array(action), res.get("latency_ms", 0),
                        np.array(res.get("chunk", [])), image=img,
                        predicted_label=res.get("predicted_label"),
                        grounding_caption=res.get("grounding_caption"),
                        goal_near=res.get("goal_near"),
                        bbox=res.get("bbox"),
                        grounding_latency_ms=res.get("grounding_latency_ms"),
                        grounding_cached=res.get("grounding_cached"))

        if mode == "SYNC":
            time.sleep(_ros.ctrl.move_duration + 0.15)
            _ros._stable = _ros.latest_bgr()
        else:
            time.sleep(_ros.ctrl.move_duration)

        if res.get("goal_near"):
            _ros.ctrl.robust_stop(source="goal_reached")
            logger.end_session("goal_reached")
            _state["running"]    = False
            _state["status_log"] = f"🎯 목적지 도달! (Step {step})"
            return

    _ros.ctrl.robust_stop(source="manual_stop")
    logger.end_session("manual_stop")
    _state["running"]    = False
    _state["status_log"] = "수동 정지"


# ═══════════════════════════════════════════════════════════════════
# 주행 루프 — ASYNC 추론 / 실행 워커
# ═══════════════════════════════════════════════════════════════════
def _async_infer(instr: str, apply_cc: bool, logger):
    step = 0
    _reset_ok = False
    for _attempt in range(2):
        try:
            _infer_post("/reset", {}, timeout=5)
            _reset_ok = True
            break
        except Exception as _e:
            print(f"⚠️ [ASYNC] /reset 실패 (attempt {_attempt+1}/2): {_e}")
    if not _reset_ok:
        _state["status_log"] = "⚠️ /reset 실패 — 이전 세션 grounding 상태가 남아있을 수 있음"

    while not _stop_ev.is_set() and _state["running"]:
        bgr = _ros.latest_bgr()
        if bgr is None: time.sleep(0.05); continue

        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if apply_cc: img = _correct(img)

        buf = io.BytesIO(); img.save(buf, "JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()

        try:
            res = _infer_post("/predict",
                              {"image": b64, "instruction": instr,
                               "strategy": "receding_horizon"})
        except Exception as e:
            log.warning(f"[ASYNC infer] {e}"); continue

        step += 1
        _state["step"]           = step
        _state["predicted_label"]= res.get("predicted_label")
        _state["latency_ms"]     = res.get("latency_ms", 0)
        _state["goal_near"]      = res.get("goal_near", False)
        _state["bbox"]           = res.get("bbox")
        _state["chunk"]          = res.get("chunk")
        _state["grounding_cached"] = res.get("grounding_cached")
        _state["grounding_caption"] = res.get("grounding_caption")
        _async_q.append(res)
        _append_history(step, res)

        # 캘리브레이션 데이터 기록
        if _state["calib_recording"]:
            _record_calib_frame(img, res)

        action = res.get("action_3d") or res.get("action", [0.0, 0.0, 0.0])
        logger.log_step(step, np.array(action), res.get("latency_ms", 0),
                        image=img, predicted_label=res.get("predicted_label"),
                        bbox=res.get("bbox"),
                        grounding_cached=res.get("grounding_cached"),
                        grounding_latency_ms=res.get("grounding_latency_ms"),
                        goal_near=res.get("goal_near"))

        if res.get("goal_near"):
            _ros.ctrl.robust_stop(source="async_goal")
            logger.end_session("goal_reached")
            _state["running"]    = False
            _state["status_log"] = f"🎯 목적지 도달! (Step {step})"
            return


def _async_exec():
    lx = ly = az = 0.0
    last_upd = time.time()
    # 2026-07-23: grounding_decisions.jsonl 실측 PG2 latency p95=3.27s(grounding_skip_n=3의
    # "신선 그라운딩" 스텝마다 발생) — 기존 COAST=1.2s는 이 정상 지연에도 발동해 로봇이
    # 매 3스텝마다 강제로 velocity=0(정지)됐다가 재개하는 인위적 stop-go를 만들었음
    # (수집 데이터는 ~6Hz 연속 조이스틱이라 이런 정지 구간이 전혀 없음 — train/inference
    # 모션 연속성 불일치의 핵심 원인 중 하나). p95보다 여유있게 4.0s로 상향.
    COAST = 4.0
    while not _stop_ev.is_set() and _state["running"]:
        if _async_q:
            res = _async_q.popleft()
            action = np.asarray(res.get("action_3d") or res["action"],
                                dtype=np.float32).reshape(-1)
            lx, ly = float(action[0]), float(action[1])
            az = float(action[2]) if action.size > 2 else 0.0
            last_upd = time.time()
            _state["last_action"] = [lx, ly, az]
            _state["action_history"].append([lx, ly, az])
        if time.time() - last_upd > COAST:
            lx = ly = az = 0.0
        msg = _ros.ctrl.publish_and_move(lx, ly, az, source="async_exec")
        _state["status_log"] = msg
        time.sleep(0.1)
    _ros.ctrl.robust_stop(source="async_end")


# ═══════════════════════════════════════════════════════════════════
# exp73 GoalNav 전용 드라이빙 루프 (PG2Grounder 재사용 + inference_server.py
# /goalnav/predict, 8000포트 — stage2_v2/8001과는 별개 프로세스/모델)
# ═══════════════════════════════════════════════════════════════════
def _goalnav_infer(phrase: str, apply_cc: bool, logger):
    step = 0
    _reset_ok = False
    for _attempt in range(2):
        try:
            _infer_post("/goalnav/reset", {}, timeout=5, base_url=GOALNAV_URL)
            _reset_ok = True
            break
        except Exception as _e:
            print(f"⚠️ [GOALNAV] /goalnav/reset 실패 (attempt {_attempt+1}/2): {_e}")
    if not _reset_ok:
        _state["status_log"] = "⚠️ /goalnav/reset 실패 — 이전 세션 bbox/vision 캐시가 남아있을 수 있음"

    grounder = _get_pg2_grounder()

    while not _stop_ev.is_set() and _state["running"]:
        bgr = _ros.latest_bgr()
        if bgr is None: time.sleep(0.05); continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        if apply_cc: img = _correct(img); rgb = np.array(img)

        try:
            bbox = grounder.run(rgb, phrase=phrase)
        except Exception as e:
            log.warning(f"[GOALNAV grounding] {e}"); continue

        buf = io.BytesIO(); img.save(buf, "JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()

        try:
            res = _infer_post("/goalnav/predict", {
                "image": b64,
                "bbox_cx": bbox["cx"], "bbox_cy": bbox["cy"],
                "bbox_area": bbox["area"], "has_bbox": bbox["has_bbox"],
                "update_vision": True,
            }, base_url=GOALNAV_URL)
        except Exception as e:
            log.warning(f"[GOALNAV infer] {e}"); continue

        step += 1
        _state["step"]          = step
        _state["predicted_label"] = res.get("class_name")
        _state["latency_ms"]    = res.get("latency_ms", 0)
        _state["bbox"]          = bbox
        _goalnav_q.append(res)
        _append_history(step, {**res, "bbox": bbox})

        if _state["calib_recording"]:
            _record_calib_frame(img, {**res, "bbox": bbox})

        action = res.get("action", {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0})
        action_arr = [action.get("linear_x", 0.0), action.get("linear_y", 0.0), action.get("angular_z", 0.0)]
        logger.log_step(step, np.array(action_arr), res.get("latency_ms", 0),
                        image=img, predicted_label=res.get("class_name"),
                        bbox=bbox)

        if res.get("class_idx") == 0 and step > 1:
            # STOP 예측 — GoalNav 자체 도착규칙(_arrival_stop)이 서버 내부에서 래치하므로
            # 여기선 그냥 계속 진행(다음 스텝도 STOP 유지될 것). 별도 종료 처리 없음
            # (stage2_v2의 goal_near 플래그 같은 명시적 "도달" 신호가 이 엔드포인트엔 없음).
            pass


def _goalnav_exec():
    lx = ly = az = 0.0
    last_upd = time.time()
    # 2026-07-23: grounding_decisions.jsonl 실측 PG2 latency p95=3.27s(grounding_skip_n=3의
    # "신선 그라운딩" 스텝마다 발생) — 기존 COAST=1.2s는 이 정상 지연에도 발동해 로봇이
    # 매 3스텝마다 강제로 velocity=0(정지)됐다가 재개하는 인위적 stop-go를 만들었음
    # (수집 데이터는 ~6Hz 연속 조이스틱이라 이런 정지 구간이 전혀 없음 — train/inference
    # 모션 연속성 불일치의 핵심 원인 중 하나). p95보다 여유있게 4.0s로 상향.
    COAST = 4.0
    while not _stop_ev.is_set() and _state["running"]:
        if _goalnav_q:
            res = _goalnav_q.popleft()
            action = res.get("action", {"linear_x": 0.0, "linear_y": 0.0, "angular_z": 0.0})
            lx, ly, az = float(action.get("linear_x", 0.0)), float(action.get("linear_y", 0.0)), float(action.get("angular_z", 0.0))
            last_upd = time.time()
            _state["last_action"] = [lx, ly, az]
            _state["action_history"].append([lx, ly, az])
        if time.time() - last_upd > COAST:
            lx = ly = az = 0.0
        msg = _ros.ctrl.publish_and_move(lx, ly, az, source="goalnav_exec")
        _state["status_log"] = msg
        time.sleep(0.1)
    _ros.ctrl.robust_stop(source="goalnav_end")


# ═══════════════════════════════════════════════════════════════════
# 시작위치로 복귀 실행 루프
# ═══════════════════════════════════════════════════════════════════
def _return_loop():
    _state["is_returning"] = True
    try:
        history = _state.get("action_history", [])
        if not history:
            _state["status_log"] = "⚠️ 복귀할 경로가 없습니다."
            return
        
        # 반대 방향으로 역재생
        rev = [(-lx, -ly, -az) for lx, ly, az in reversed(history)]
        _state["status_log"] = f"🔄 복귀 시작 ({len(rev)}스텝 역재생)"
        
        for i, (lx, ly, az) in enumerate(rev):
            if not _state["is_returning"]:
                _state["status_log"] = "🛑 복귀 중단됨"
                break
            _state["status_log"] = f"🔄 복귀 중 ({i+1}/{len(rev)}스텝)"
            if _ros and _ros.ctrl:
                _ros.ctrl.publish_and_move(lx, ly, az, source="return")
            time.sleep(_ros.ctrl.move_duration)
            
        _state["status_log"] = "✅ 시작 위치 복귀 완료"
    except Exception as e:
        _state["status_log"] = f"❌ 복귀 실패: {e}"
    finally:
        if _ros and _ros.ctrl:
            _ros.ctrl.robust_stop(source="return_end")
        _state["is_returning"] = False


# ═══════════════════════════════════════════════════════════════════
# API 모델 스키마
# ═══════════════════════════════════════════════════════════════════
class DriveReq(BaseModel):
    mode: str = "ASYNC"         # SYNC | PRE | ASYNC
    instruction: str = "gray basket"
    gt_object: str = ""
    apply_cc: bool = False

class CCReq(BaseModel):
    r_gain: float = 1.0
    g_gain: float = 1.0
    b_gain: float = 1.0

class ManualDriveReq(BaseModel):
    direction: str             # W, S, A, D, Q, E, R, T, STOP
    speed: float = 1.15

class CollectKeyReq(BaseModel):
    key: str                   # w,a,s,d,q,e,z,c,r,t,' '
    event: str = "down"        # "down" | "up"

class CollectEpisodeStartReq(BaseModel):
    episode_name: Optional[str] = None
    scenario: Optional[str] = None
    pattern: Optional[str] = None
    cx_position: Optional[str] = None
    cx_path: Optional[str] = None

class CollectEpisodeStopReq(BaseModel):
    save: bool = True

class CollectScenarioStageReq(BaseModel):
    scenario: Optional[str] = None

class CollectScenarioCycleReq(BaseModel):
    step: int = 1

class CollectCxPosStageReq(BaseModel):
    cx_position: Optional[str] = None

class CollectCxPosCycleReq(BaseModel):
    step: int = 1

class CollectCxPathStageReq(BaseModel):
    cx_path: Optional[str] = None

class CollectCxPathCycleReq(BaseModel):
    step: int = 1

class CollectModeToggleReq(BaseModel):
    step: int = 1

class ConfigToggleReq(BaseModel):
    preview_enabled: Optional[bool] = None
    preview_hint_cx: Optional[bool] = None
    grounding_skip_n: Optional[int] = None
    cx_jump_filter: Optional[bool] = None
    cx_jump_thresh: Optional[float] = None
    stop_area_threshold: Optional[float] = None
    multi_prompt: Optional[bool] = None
    owlv2_thresh: Optional[float] = None
    owlv2_area_scale: Optional[float] = None

class LabelSaveReq(BaseModel):
    session_id: str
    frame_idx: int
    label: str                  # L | C | R | NONE

class EpisodeLogReq(BaseModel):
    path_type: str
    success: str
    fpe: float
    note: str
    experimental: bool = False  # True면 별도 실험용 CSV(episode_log_experimental.csv)에 기록

class EpisodeUpdateReq(BaseModel):
    row: int  # 테이블의 "#" (1-based)
    path_type: str
    success: str
    steps: int
    lat_ms: float
    fpe: float
    note: str


# ═══════════════════════════════════════════════════════════════════
# HTTP 엔드포인트
# ═══════════════════════════════════════════════════════════════════

CAMERA_STALE_S = 3.0  # 이 시간 동안 새 프레임이 없으면 카메라가 멈춘 것으로 판단

@app.get("/health")
def health():
    cam_ok = age = None
    fc = 0
    if _ros:
        fc = _ros.frame_count
        if _ros.last_ts:
            age = round(time.time() - _ros.last_ts, 2)
        # 프레임을 받은 적이 있는지뿐 아니라 "최근에" 받았는지까지 확인 —
        # camera_pub 서비스가 응답은 하지만 동일 프레임만 반복 전달하는
        # (요청은 성공하지만 실제 캡처는 멈춘) 상태를 잡기 위함.
        cam_ok = _ros._frame is not None and age is not None and age < CAMERA_STALE_S
    return {"status": "ok", "ros": ROS_AVAILABLE, "node_up": _ros is not None,
            "camera_ok": cam_ok, "frame_count": fc, "frame_age_s": age,
            "infer_url": INFER_URL,
            "pid": _PROCESS_PID,
            "started_at": datetime.datetime.fromtimestamp(_PROCESS_START_TS).strftime("%Y-%m-%d %H:%M:%S"),
            "uptime_s": round(time.time() - _PROCESS_START_TS, 1)}


@app.get("/camera/frame")
def camera_frame():
    """단일 프레임 — base64 JPEG."""
    if _ros is None:
        return {"ok": False, "error": "ROS 없음"}
    jbytes = _ros.jpeg_bytes()
    if jbytes is None:
        return {"ok": False, "error": "프레임 없음"}
    return {"ok": True, "image": base64.b64encode(jbytes).decode(), "format": "jpeg_base64"}


@app.get("/camera/stream")
def camera_stream():
    """MJPEG 스트림 — <img src=/camera/stream> 직접 사용 가능."""
    def gen():
        while True:
            if _ros is None:
                time.sleep(0.1); continue
            jb = _ros.jpeg_bytes(quality=60)
            if jb:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jb + b"\r\n")
            time.sleep(0.1)
    return StreamingResponse(gen(),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/drive/status")
def drive_status():
    from scripts.inference_logger import get_logger
    logger = get_logger()
    ret = dict(_state)
    
    if logger and hasattr(logger, "data") and logger.data:
        hist = logger.data.get("history", [])
        ret["n_total"] = len(hist)
        ret["n_post"] = sum(1 for h in hist if str(h.get("step","")).endswith("p"))
        ret["n_infer"] = ret["n_total"] - ret["n_post"]
        ret["n_frames"] = len(logger._frames)
        ret["session_id"] = logger.session_id
        
        last_steps = []
        for h in hist[-8:]:
            last_steps.append({
                "step": h.get("step", "?"),
                "predicted_label": h.get("predicted_label") or "—",
                "grounding_cached": h.get("grounding_cached"),
                "latency_ms": h.get("latency_ms", 0),
            })
        ret["last_steps"] = last_steps
        
        gc_vals = [h.get("grounding_cached") for h in hist if h.get("grounding_cached") is not None]
        ret["gnd_live"] = sum(1 for v in gc_vals if v == 0)
        ret["gnd_cache"] = sum(1 for v in gc_vals if v == 1)
        
        lat_l = [h.get("grounding_latency_ms", 0) for h in hist if h.get("grounding_latency_ms")]
        ret["gnd_avg_lat"] = round(sum(lat_l) / len(lat_l)) if lat_l else 0
    else:
        ret["n_total"] = 0
        ret["n_post"] = 0
        ret["n_infer"] = 0
        ret["n_frames"] = 0
        ret["last_steps"] = []
        ret["gnd_live"] = 0
        ret["gnd_cache"] = 0
        ret["gnd_avg_lat"] = 0
        
    return ret

@app.post("/drive/start")
def drive_start(req: DriveReq):
    global _ros
    if _ros is None:
        return {"ok": False, "error": "ROS 노드 없음"}
    if _state["running"]:
        return {"ok": False, "error": "이미 실행 중"}

    _state.update(running=True, mode=req.mode, step=0,
                  instruction=req.instruction, gt_object=req.gt_object,
                  apply_cc=req.apply_cc, goal_near=False,
                  grounding_cached=None, grounding_caption=None,
                  bbox=None, chunk=None,
                  run_history=[], action_history=[], last_action=[0.0,0.0,0.0])
    _stop_ev.clear()
    _async_q.clear()
    _goalnav_q.clear()

    if req.mode in ("SYNC", "PRE"):
        threading.Thread(target=_loop_sync,
                         args=(req.mode, req.instruction, req.gt_object, req.apply_cc),
                         daemon=True, name="drive-sync").start()
    elif req.mode == "GOALNAV":
        from scripts.inference_logger import get_logger
        logger = get_logger()
        logger.start_session("goalnav_exp73", req.instruction, instruction_mode="GOALNAV")
        if req.gt_object: logger.data["gt_object"] = req.gt_object
        logger.data["apply_cc"] = req.apply_cc
        logger.data["runtime_config"] = _snapshot_runtime_config()
        _state["session_id"] = logger.session_id

        phrase = req.gt_object or "gray basket"
        threading.Thread(target=_goalnav_infer,
                         args=(phrase, req.apply_cc, logger),
                         daemon=True, name="goalnav-infer").start()
        threading.Thread(target=_goalnav_exec,
                         daemon=True, name="goalnav-exec").start()
    else:
        from scripts.inference_logger import get_logger
        logger = get_logger()
        logger.start_session("stage2_v2", req.instruction, instruction_mode="ASYNC")
        if req.gt_object: logger.data["gt_object"] = req.gt_object
        logger.data["apply_cc"] = req.apply_cc
        logger.data["runtime_config"] = _snapshot_runtime_config()
        _state["session_id"] = logger.session_id

        threading.Thread(target=_async_infer,
                         args=(req.instruction, req.apply_cc, logger),
                         daemon=True, name="async-infer").start()
        threading.Thread(target=_async_exec,
                         daemon=True, name="async-exec").start()

    return {"ok": True, "mode": req.mode, "message": f"{req.mode} 주행 시작"}


@app.post("/drive/stop")
def drive_stop():
    _state["running"] = False
    _stop_ev.set()
    if _ros and _ros.ctrl:
        _ros.ctrl.robust_stop(source="api_stop")
    try:
        from scripts.inference_logger import get_logger
        get_logger().end_session("manual_stop")
    except Exception: pass
    _state["status_log"] = "정지 완료"
    return {"ok": True}


# ── 🧪 경로검증 모드 조이스틱 버튼용 헬퍼 — HTTP 왕복 없이 직접 호출 ──
def _joystick_drive_start():
    if _state["running"]:
        return
    req = DriveReq(mode=_state.get("mode") or "SYNC",
                   instruction=_state.get("instruction") or "gray basket",
                   gt_object=_state.get("gt_object") or "",
                   apply_cc=bool(_state.get("apply_cc") or False))
    try:
        drive_start(req)
    except Exception as e:
        log.warning(f"[Joystick] drive_start 실패: {e}")


def _joystick_drive_stop():
    if _state["running"]:
        drive_stop()


def _verify_save_session():
    """L2 — 최종 세션 저장(추론세션 저장). D-pad로 고른 위치 + X/A 라벨로 1건만 기록.
    결과 라벨(X/A)이 아직 없으면 저장 안 함(실수 방지). 저장 후 라벨 초기화."""
    global _verify_pending_result, _verify_save_seq
    if not _verify_pending_result:
        _state["status_log"] = "⚠️ 결과 미지정 — X(성공)/A(실패) 먼저 누른 뒤 L2로 저장"
        log.info("[Verify] L2 저장 취소 — 결과 라벨 없음")
        return
    _joystick_drive_stop()
    # 💾 기록 저장 버튼과 동일한 path_type을 쓰도록 폼 드롭다운 값(_verify_current_path_type)을
    # 우선 사용 — A안 미러가 D-pad 위치에 맞춰 드롭다운을 동기화해둠. 없으면 위치 매핑 폴백.
    pt = _verify_current_path_type or _verify_pos_to_path_type(_verify_screen_pos)
    result = _verify_pending_result
    try:
        note = "검증 스크리닝(L2 저장)" + ("· 🧪실험용" if _verify_experimental else "")
        episodes_log(EpisodeLogReq(path_type=pt, success=result, fpe=0.0,
                                   note=note, experimental=_verify_experimental))
        tag = " [실험용]" if _verify_experimental else ""
        log.info(f"[Verify] L2 세션 저장{tag}: {pt} = {result}")
        _state["status_log"] = f"💾 저장{tag}: {pt} = {result}"
        _verify_pending_result = None  # 저장 후 라벨 초기화(다음 세션 대비)
        _verify_save_seq += 1          # 브라우저 폴링이 이 값 변화로 자동 갱신 트리거
    except Exception as e:
        log.warning(f"[Verify] L2 저장 실패: {e}")


def _joystick_verify_return():
    """R2(검증모드) — 주행 경로 역재생 복귀. drive_return()과 동일(수집 return과 별개)."""
    try:
        drive_return()
    except Exception as e:
        log.warning(f"[Verify] 복귀 실패: {e}")


def _joystick_toggle_verify_mode():
    """Y(3) 버튼 — collect⇄verify 토글. 주행/녹화 중엔 무시(안전가드)."""
    global _joystick_verify_mode
    active = bool(_state.get("running")) or bool(_collect is not None and _collect.active)
    if active:
        log.info("[Joystick] Y 모드전환 무시 — 주행/녹화 진행 중")
        return
    _joystick_verify_mode = not _joystick_verify_mode
    log.info(f"[Joystick] Y → 배치모드 = {'verify' if _joystick_verify_mode else 'collect'}")


def _joystick_toggle_verify_experimental():
    """SEL 버튼(검증모드 전용) — 실험용/테스트 기록 모드 토글. 주행 중엔 무시(안전가드).
    켜져 있는 동안 L2 저장은 episode_log_experimental.csv로 분리 기록됨."""
    global _verify_experimental
    if bool(_state.get("running")):
        log.info("[Joystick] SEL 실험모드 전환 무시 — 주행 진행 중")
        return
    _verify_experimental = not _verify_experimental
    _state["status_log"] = f"🧪 실험용 기록 모드: {'ON' if _verify_experimental else 'OFF'}"
    log.info(f"[Joystick] SEL → 실험용 기록 모드 = {_verify_experimental}")


class JoystickModeReq(BaseModel):
    mode: str  # "collect" | "verify"


@app.post("/joystick/mode")
def joystick_set_mode(req: JoystickModeReq):
    global _joystick_verify_mode
    if req.mode not in ("collect", "verify"):
        return {"ok": False, "error": f"알 수 없는 모드: {req.mode}"}
    _joystick_verify_mode = (req.mode == "verify")
    return {"ok": True, "mode": req.mode}


@app.get("/joystick/mode")
def joystick_get_mode():
    return {"mode": "verify" if _joystick_verify_mode else "collect"}


class VerifyPathTypeReq(BaseModel):
    path_type: str


@app.post("/verify/path_type")
def verify_set_path_type(req: VerifyPathTypeReq):
    global _verify_current_path_type
    _verify_current_path_type = req.path_type
    return {"ok": True, "path_type": _verify_current_path_type}


@app.post("/drive/return")
def drive_return():
    if _state["is_returning"]:
        _state["is_returning"] = False
        if _ros and _ros.ctrl:
            _ros.ctrl.robust_stop(source="return_cancel")
        return {"ok": True, "message": "복귀가 취소되었습니다."}
    
    threading.Thread(target=_return_loop, daemon=True, name="return-loop").start()
    return {"ok": True, "message": "시작 위치 복귀를 시작합니다."}


@app.post("/drive/manual")
def drive_manual(req: ManualDriveReq):
    """수동 제어 퍼블리시"""
    if not ROS_AVAILABLE or not _ros:
        return {"ok": False, "error": "ROS 연결 불가"}

    s = float(req.speed)
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

    if req.direction not in mapping:
        return {"ok": False, "error": f"잘못된 방향: {req.direction}"}

    lx, ly, az = mapping[req.direction]
    
    if req.direction == "STOP":
        _ros.ctrl.robust_stop(source="manual_stop")
        _state["status_log"] = "🛑 긴급 정지 (Force STOP)"
    else:
        _ros.ctrl.move_and_stop_timed(lx, ly, az, source=f"manual_{req.direction}")
        _state["status_log"] = f"🕹️ {req.direction} (속도: {s:.2f})"
        _state["last_action"] = [lx, ly, az]
        _state["action_history"].append([lx, ly, az])

    # 수동 운전 중에도 캘리브레이션 캡쳐 지원
    if _state["calib_recording"]:
        rgb = _ros.latest_rgb()
        if rgb is not None:
            _record_calib_frame(Image.fromarray(rgb), {
                "bbox": None,
                "latency_ms": 0.0,
                "goal_near": False,
                "predicted_label": f"MANUAL_{req.direction}"
            })

    return {"ok": True, "log": _state["status_log"]}


@app.get("/collect/ground")
def collect_ground():
    """데이터수집 탭 — 바구니 배치용 실시간 cx 피드백.
    액션예측 없이 그라운딩만 수행하는 추론서버 /ground를 프록시."""
    if not ROS_AVAILABLE or not _ros:
        return {"ok": False, "error": "ROS 연결 불가"}
    frame = _ros.latest_bgr()
    if frame is None:
        return {"ok": False, "error": "카메라 프레임 없음"}
    import requests as rq
    rgb = frame[:, :, ::-1]
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    try:
        r = rq.post(f"{INFER_URL}/ground", json={"image": b64},
                    headers={"X-API-Key": API_KEY}, timeout=3)
        d = r.json()
        d["ok"] = True
        return d
    except rq.exceptions.ConnectionError:
        return {"ok": False, "error": "추론서버(8001) 연결 안 됨"}
    except rq.exceptions.Timeout:
        return {"ok": False, "error": "추론서버 응답 지연(timeout)"}
    except Exception as e:
        return {"ok": False, "error": type(e).__name__}


@app.get("/collect/snapshot")
def collect_snapshot():
    """데이터수집 탭 — 현재 카메라 1프레임을 캡처해 base64 JPEG로 반환 (cx 기준 가이드 이미지용)."""
    if not ROS_AVAILABLE or not _ros:
        return {"ok": False, "error": "ROS 연결 불가"}
    frame = _ros.latest_bgr()
    if frame is None:
        return {"ok": False, "error": "카메라 프레임 없음"}
    rgb = frame[:, :, ::-1]
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return {"ok": True, "image": b64}


@app.post("/collect/scenario/stage")
def collect_scenario_stage(req: CollectScenarioStageReq):
    """진행률 행 클릭으로 시나리오 선택 — 조이스틱 D-pad와 동일한 staged_scenario를 공유(단일 소스)."""
    if _collect is None:
        return {"ok": False, "error": "데이터수집 세션 미초기화"}
    staged = _collect.stage_scenario(req.scenario)
    return {"ok": True, "staged_scenario": staged}


@app.post("/collect/scenario/cycle")
def collect_scenario_cycle(req: CollectScenarioCycleReq):
    """화면의 ◀/▶ 버튼으로 시나리오 순환 — 조이스틱 D-pad 없이도 동일하게 사용 가능."""
    if _collect is None:
        return {"ok": False, "error": "데이터수집 세션 미초기화"}
    staged = _collect.cycle_scenario(req.step)
    return {"ok": True, "staged_scenario": staged}


@app.post("/collect/cxpos/stage")
def collect_cxpos_stage(req: CollectCxPosStageReq):
    """트랙A 막대 행 클릭으로 cx위치 선택 — 조이스틱 D-pad와 동일한 staged_cx_position 공유."""
    if _collect is None:
        return {"ok": False, "error": "데이터수집 세션 미초기화"}
    staged = _collect.stage_cx_position(req.cx_position)
    return {"ok": True, "staged_cx_position": staged}


@app.post("/collect/cxpos/cycle")
def collect_cxpos_cycle(req: CollectCxPosCycleReq):
    """화면의 ◀/▶ 버튼으로 트랙A cx위치 순환."""
    if _collect is None:
        return {"ok": False, "error": "데이터수집 세션 미초기화"}
    staged = _collect.cycle_cx_position(req.step)
    return {"ok": True, "staged_cx_position": staged}


@app.post("/collect/cxpath/stage")
def collect_cxpath_stage(req: CollectCxPathStageReq):
    """트랙A 접근경로(좌곡선/직진/우곡선) 선택 — 위치와 별개 축, 위치×경로 세분화 카운트에 사용."""
    if _collect is None:
        return {"ok": False, "error": "데이터수집 세션 미초기화"}
    staged = _collect.stage_cx_path(req.cx_path)
    return {"ok": True, "staged_cx_path": staged}


@app.post("/collect/cxpath/cycle")
def collect_cxpath_cycle(req: CollectCxPathCycleReq):
    if _collect is None:
        return {"ok": False, "error": "데이터수집 세션 미초기화"}
    staged = _collect.cycle_cx_path(req.step)
    return {"ok": True, "staged_cx_path": staged}


@app.post("/collect/mode/toggle")
def collect_mode_toggle(req: CollectModeToggleReq = CollectModeToggleReq()):
    """D-pad 상/하(또는 화면 ▲▼) — 트랙A(cx위치)/접근경로/시나리오(9종) 3축을 순환 전환."""
    if _collect is None:
        return {"ok": False, "error": "데이터수집 세션 미초기화"}
    mode = _collect.toggle_collect_mode(req.step)
    return {"ok": True, "collect_mode": mode}


@app.post("/collect/return")
def collect_return():
    """조이스틱 R2(트리거) 또는 화면 버튼으로 직전 경로를 역재생해 시작 위치로 복귀 (Gradio 이식)."""
    if _collect is None:
        return {"ok": False, "error": "데이터수집 세션 미초기화"}
    return _collect.start_auto_return()


@app.get("/collect/state")
def collect_state():
    if _collect is None:
        return {"ok": False, "error": "데이터수집 세션 미초기화"}
    return {"ok": True, **_collect.state()}


@app.post("/collect/key")
def collect_key(req: CollectKeyReq):
    """데이터수집 탭 키보드 입력 — keydown마다 원본과 동일한 400ms
    watchdog(move_and_stop_timed)을 재무장. keyup은 즉시 정지로 반응성 확보."""
    if not ROS_AVAILABLE or not _ros:
        return {"ok": False, "error": "ROS 연결 불가"}
    key = req.key.lower()
    if key not in COLLECT_KEY_TO_VEL:
        return {"ok": False, "error": f"알 수 없는 키: {req.key}"}
    lx, ly, az = COLLECT_KEY_TO_VEL[key]
    if req.event == "up":
        _ros.ctrl.robust_stop(source=f"collect_key_up_{key}")
        return {"ok": True}
    _ros.ctrl.move_and_stop_timed(lx, ly, az, source=f"collect_key_{key}")
    return {"ok": True, "action": {"lx": lx, "ly": ly, "az": az}}


@app.post("/collect/episode/start")
def collect_episode_start(req: CollectEpisodeStartReq):
    if _collect is None:
        return {"ok": False, "error": "데이터수집 세션 미초기화"}
    name = _collect.start_episode(req.episode_name, req.scenario, req.pattern, req.cx_position, req.cx_path)
    return {"ok": True, "episode_name": name}


@app.post("/collect/episode/stop")
def collect_episode_stop(req: CollectEpisodeStopReq):
    if _collect is None:
        return {"ok": False, "error": "데이터수집 세션 미초기화"}
    if _ros:
        _ros.ctrl.robust_stop(source="collect_episode_stop")
    return _collect.stop_episode(save=req.save)


# ─── 런타임 설정 프록시 ────────────────────────────────────────────
@app.post("/config")
def config_update(req: ConfigToggleReq):
    """로컬 및 8001 추론 서버에 파라미터 적용"""
    payload = {}
    if req.preview_enabled is not None: payload["preview_enabled"] = req.preview_enabled
    if req.preview_hint_cx is not None: payload["preview_hint_cx"] = req.preview_hint_cx
    if req.grounding_skip_n is not None: payload["grounding_skip_n"] = req.grounding_skip_n
    if req.cx_jump_filter is not None: payload["cx_jump_filter"] = req.cx_jump_filter
    if req.cx_jump_thresh is not None: payload["cx_jump_thresh"] = req.cx_jump_thresh
    if req.stop_area_threshold is not None: payload["stop_area_threshold"] = req.stop_area_threshold
    if req.multi_prompt is not None: payload["multi_prompt"] = req.multi_prompt
    if req.owlv2_thresh is not None: payload["owlv2_thresh"] = req.owlv2_thresh
    if req.owlv2_area_scale is not None: payload["owlv2_area_scale"] = req.owlv2_area_scale

    try:
        res = _infer_post("/config", payload, timeout=3)
        return {"ok": True, "applied": res}
    except Exception as e:
        return {"ok": False, "error": f"추론서버 적용 실패 (로컬 우선): {e}"}


# ─── 컬러 보정 ───────────────────────────────────────────────────
@app.post("/cc/set")
def cc_set(req: CCReq):
    _cc.update(r_gain=req.r_gain, g_gain=req.g_gain, b_gain=req.b_gain)
    return {"ok": True, "cc": dict(_cc)}


@app.get("/cc/get")
def cc_get():
    return dict(_cc)


# ─── 추론 서버 상태 확인 ──────────────────────────────────────────
@app.get("/infer/health")
def infer_health():
    import requests as rq
    try:
        r = rq.get(f"{INFER_URL}/health",
                   headers={"X-API-Key": API_KEY}, timeout=2)
        return r.json()
    except Exception as e:
        return {"status": "error", "detail": str(e)}


# ─── 🔀 모델 전환 (Tab4용) — /model/load 핫스왑 프록시, go.sh 재시작(95s) 불필요 ──
EXP73_MODEL_DIR = ROOT / "runs" / "v5_nav" / "mlp" / "exp73"


@app.get("/verify/model_list")
def verify_model_list():
    """runs/v5_nav/mlp/exp73/*.pt 스캔 + 메타데이터(val_acc/head/held_success/stride)
    가벼운 torch.load(map_location='cpu')로 파싱 — 파일 몇 개뿐이라 부담 적음."""
    import torch
    out = []
    if not EXP73_MODEL_DIR.exists():
        return {"ok": True, "models": []}
    for f in sorted(EXP73_MODEL_DIR.glob("*.pt")):
        meta = {"filename": f.name, "path": str(f.relative_to(ROOT)),
                "size_mb": round(f.stat().st_size / 1e6, 2)}
        try:
            ckpt = torch.load(str(f), map_location="cpu", weights_only=False)
            meta["head"] = ckpt.get("head", "?")
            meta["exp"] = ckpt.get("exp", "")
            meta["window"] = ckpt.get("window", "")
            meta["bbox_scale"] = ckpt.get("bbox_scale", "")
            meta["stride"] = ckpt.get("stride", "")
            va = ckpt.get("val_acc")
            meta["val_acc"] = round(float(va), 4) if va else None
            hs = ckpt.get("held_success")
            meta["held_success"] = round(float(hs), 1) if hs is not None else None
        except Exception as e:
            meta["error"] = str(e)
        out.append(meta)
    return {"ok": True, "models": out}


class ModelSwitchReq(BaseModel):
    path: str  # runs/v5_nav/mlp/exp73/xxx.pt (repo-relative)
    grounder: Optional[str] = None  # "pg2" | "owlv2" — 지정 시 그라운더도 같이 핫스왑


@app.post("/verify/model_switch")
def verify_model_switch(req: ModelSwitchReq):
    """추론서버(8001)의 /model/load 핫스왑 프록시 — 프로세스 재시작(go.sh, ~95s)
    없이 체크포인트(+옵션으로 그라운더)만 즉시 교체. Kosmos-2 vision encoder는 유지."""
    import requests as rq
    full_path = str((ROOT / req.path).resolve())
    if not os.path.exists(full_path):
        return {"ok": False, "error": f"파일 없음: {req.path}"}
    try:
        payload = {"stage2_path": full_path}
        if req.grounder:
            payload["grounder"] = req.grounder
        r = rq.post(f"{INFER_URL}/model/load",
                     json=payload,
                     headers={"X-API-Key": API_KEY}, timeout=30)
        r.raise_for_status()
        result = r.json()
        log.info(f"[ModelSwitch] {req.path} (grounder={req.grounder}) → {result}")
        return {"ok": True, **result}
    except Exception as e:
        log.warning(f"[ModelSwitch] 실패: {e}")
        return {"ok": False, "error": str(e)}


def _snapshot_runtime_config() -> dict:
    """세션 시작 시점의 런타임 설정 스냅샷 — H5 attrs에 박아서 나중에
    로그 없이도 '이 세션 때 뭘 켜놨었는지' 확인 가능하게 함(2026-07-02).

    2026-07-06: OWL-v2 A/B 그라운더 도입 후 이 스냅샷에 grounder 정보가
    빠져있던 걸 발견 — 어느 그라운더로 수집한 세션인지가 오늘 분석의
    핵심인데 누락되고 있었음. grounder/owlv2_thresh/checkpoint/git_commit 추가.
    """
    try:
        import requests as rq
        r = rq.get(f"{INFER_URL}/health", headers={"X-API-Key": API_KEY}, timeout=2)
        h = r.json()
        g = h.get("grounder", {}) or {}
        return {
            "preview_enabled": h.get("preview", {}).get("enabled"),
            "preview_hint_cx": h.get("preview", {}).get("hint_cx"),
            "grounding_skip_n": h.get("grounding_skip_n"),
            "cx_jump_filter": h.get("cx_jump_filter"),
            "cx_jump_thresh": h.get("cx_jump_thresh"),
            "multi_prompt": h.get("multi_prompt"),
            "head": h.get("head"),
            "grounder_model": g.get("model"),
            "grounder_input_px": g.get("input_px"),
            "owlv2_thresh": g.get("owlv2_thresh"),
            "owlv2_area_scale": g.get("owlv2_area_scale"),
            "checkpoint_path": h.get("checkpoint_path"),
            "git_commit": h.get("git_commit"),
        }
    except Exception as e:
        return {"error": str(e)}


# ─── 캘리브레이션 녹화 ─────────────────────────────────────────────
@app.post("/calib/rec/start")
def calib_rec_start():
    _state["calib_recording"] = True
    _state["calib_frames"] = []
    _state["calib_imgs"] = []
    _state["calib_session_name"] = f"calib_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return {"ok": True, "session": _state["calib_session_name"]}

@app.post("/calib/rec/stop")
def calib_rec_stop():
    _state["calib_recording"] = False
    return {"ok": True, "frames_count": len(_state["calib_frames"])}

@app.post("/calib/rec/snap")
def calib_rec_snap():
    """수동 1장 스냅샷 저장"""
    if _ros is None: return {"ok": False, "error": "ROS 카메라 사용불가"}
    rgb = _ros.latest_rgb()
    if rgb is None: return {"ok": False, "error": "카메라 프레임 없음"}

    # 추론 서버로부터 예측 가져오기 시도 (1회용 스냅샷)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    res = {}
    try:
        res = _infer_post("/predict", {"image": b64, "instruction": _state["instruction"]}, timeout=5)
    except Exception:
        pass

    _record_calib_frame(Image.fromarray(rgb), res)
    return {"ok": True, "frames_count": len(_state["calib_frames"])}

@app.post("/calib/rec/clear")
def calib_rec_clear():
    _state["calib_frames"] = []
    _state["calib_imgs"] = []
    _state["calib_session_name"] = ""
    return {"ok": True}

@app.post("/calib/rec/save")
def calib_rec_save(name: Optional[str] = None):
    frames = _state["calib_frames"]
    imgs = _state["calib_imgs"]
    if not frames:
        return {"ok": False, "error": "저장할 프레임이 없습니다."}
    
    base = name.strip() if name else _state["calib_session_name"]
    if not base:
        base = f"calib_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
    CALIB_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = CALIB_DIR / f"{base}.jsonl"
    mp4_path = CALIB_DIR / f"{base}.mp4"
    
    # 1. JSONL 메타 저장
    try:
        with open(jsonl_path, "w") as f:
            for frm in frames:
                json.dump({
                    "n": frm["n"],
                    "ts": frm["ts"],
                    "has_bbox": frm["area"] > 0.01,
                    "area": frm["area"],
                    "cx": frm["cx"],
                    "cy": frm["cy"],
                    "latency_ms": frm["latency_ms"],
                    "pred_label": frm["note"],
                    "stop_triggered": frm["stop"] == "Y",
                }, f)
                f.write("\n")
    except Exception as e:
        return {"ok": False, "error": f"JSONL 저장 오류: {e}"}
        
    # 2. 비디오 인코딩
    mp4_ok = False
    if imgs:
        try:
            arr0 = np.array(imgs[0])
            h, w = arr0.shape[:2]
            writer = cv2.VideoWriter(
                str(mp4_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                5.0, # 5fps
                (w, h)
            )
            for img in imgs:
                # RGB to BGR for cv2 VideoWriter
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
            writer.release()
            mp4_ok = True
        except Exception as e:
            log.warning(f"Calibration MP4 인코딩 실패: {e}")
            
    # 청소
    _state["calib_frames"] = []
    _state["calib_imgs"] = []
    _state["calib_session_name"] = ""
    
    return {"ok": True, "jsonl": str(jsonl_path), "mp4": str(mp4_path) if mp4_ok else None}


# ─── 누적 드리프트 시뮬레이션 ──────────────────────────────────────────
@app.get("/drive/drift/run")
def drive_drift_run(basis: str):
    """현재 카메라 1프레임을 추론하여 가정시간 기준 대비 드리프트 누적 측정"""
    if _ros is None: return {"ok": False, "error": "ROS 카메라 사용불가"}
    bgr = _ros.latest_bgr()
    if bgr is None: return {"ok": False, "error": "프레임이 준비되지 않았습니다."}
    
    global _drift_log_file
    if _drift_log_file is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        _drift_log_file = ROOT / "logs" / "drift_sessions" / f"drift_{ts}.jsonl"
        _drift_log_file.parent.mkdir(parents=True, exist_ok=True)

    img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    # 추론 서버 호출
    t0 = time.time()
    try:
        res = _infer_post("/predict", {"image": b64, "instruction": _state["instruction"]}, timeout=10)
        lat_ms = float(res.get("latency_ms") or 0.0)
    except Exception as e:
        return {"ok": False, "error": f"추론 호출 실패: {e}"}

    # 기준값 분석
    # 1.0s, 1.35s, 1.92s 등
    b_val = 1.0
    if "1.35" in basis: b_val = 1.35
    elif "1.92" in basis: b_val = 1.92

    _state["drift_frames"] += 1
    idx = _state["drift_frames"]
    _state["drift_cum_real"] += (lat_ms / 1000.0)
    _state["drift_cum_nom"]  += b_val
    drift = _state["drift_cum_real"] - _state["drift_cum_nom"]

    row = [idx, round(lat_ms), round(_state["drift_cum_real"], 2), round(_state["drift_cum_nom"], 2), round(drift, 2)]
    _state["drift_history"].append(row)
    _state["drift_history"] = _state["drift_history"][-30:]

    # 로그 파일 기록
    with open(_drift_log_file, "a") as f:
        json.dump({
            "ts": datetime.datetime.now().isoformat(),
            "frame": idx,
            "latency_ms": lat_ms,
            "cum_real_s": _state["drift_cum_real"],
            "cum_nominal_s": _state["drift_cum_nom"],
            "drift_s": drift,
            "basis": b_val
        }, f)
        f.write("\n")

    return {
        "ok": True,
        "frame": idx,
        "latency_ms": lat_ms,
        "cum_real": _state["drift_cum_real"],
        "cum_nom": _state["drift_cum_nom"],
        "drift": drift,
        "history": _state["drift_history"],
        "log_file": str(_drift_log_file)
    }

@app.post("/drive/drift/reset")
def drive_drift_reset():
    global _drift_log_file
    _drift_log_file = None
    _state["drift_frames"] = 0
    _state["drift_cum_real"] = 0.0
    _state["drift_cum_nom"] = 0.0
    _state["drift_history"] = []
    return {"ok": True}


# 액션 레이블 매핑 도우미 — /sessions/load, /overshoot_guide/load 공용
_ACTION_LABEL_MAP = {
    (0.0,0.0,0.0):"STOP", (1.15,0.0,0.0):"FWD",
    (0.0,1.15,0.0):"LEFT", (0.0,-1.15,0.0):"RIGHT",
    (1.15,1.15,0.0):"FWD+L", (1.15,-1.15,0.0):"FWD+R",
    (0.0,0.0,0.25):"ROT_L", (0.0,0.0,-0.25):"ROT_R",
}
def _infer_action_label(a):
    # 구버전 세션은 az 없이 (lx, ly) 2열만 기록된 경우가 있음 — az=0으로 패딩
    a3 = [float(a[0]), float(a[1]), float(a[2]) if len(a) > 2 else 0.0]
    for k, v in _ACTION_LABEL_MAP.items():
        if all(abs(a3[i]-k[i])<0.05 for i in range(3)): return v
    return f"({a3[0]:.1f},{a3[1]:.1f})"


# ─── 세션 히스토리 & 셀프 라벨링 ───────────────────────────────────────
@app.get("/sessions/list")
def sessions_list():
    h5_files = sorted(glob.glob(str(INFER_H5_DIR / "session_*.h5")), reverse=True)
    
    # 저장된 라벨 수 로드
    labels = {}
    if LABEL_JSON_PATH.exists():
        try: labels = json.loads(LABEL_JSON_PATH.read_text())
        except Exception: pass

    session_list = []
    for h5p in h5_files:
        sid = Path(h5p).stem.replace("session_", "")
        n_labeled = sum(1 for k in labels if k.startswith(f"session_{sid}_f"))
        
        # 기본 정보
        steps = "?"
        instruction = "—"
        report_path = INFER_REPORT_DIR / f"session_{sid}.json"
        if report_path.exists():
            try:
                with open(report_path) as rf:
                    d = json.load(rf)
                    steps = d.get("summary", {}).get("total_steps", "?")
                    instruction = d.get("instruction", "—")
            except Exception: pass
            
        session_list.append({
            "sid": sid,
            "steps": steps,
            "instruction": instruction,
            "labeled_count": n_labeled,
            "h5_size_mb": round(os.path.getsize(h5p) / (1024*1024), 2)
        })
    return {"ok": True, "sessions": session_list}


@app.get("/sessions/load")
def sessions_load(sid: str):
    h5p = INFER_H5_DIR / f"session_{sid}.h5"
    if not h5p.exists():
        return JSONResponse(status_code=404, content={"ok": False, "error": f"H5 파일이 없음: {h5p}"})

    labels = {}
    if LABEL_JSON_PATH.exists():
        try: labels = json.loads(LABEL_JSON_PATH.read_text())
        except Exception: pass

    try:
        with h5py.File(h5p, "r") as f:
            acts   = f["actions"][()]
            bbox   = f["grounding/bbox"][()]
            cached = f["grounding/cached"][()]
            lats   = f["grounding/latency_ms"][()]
            attrs  = dict(f.attrs)
            
        # 구버전 여부 판단 (6/30 이전)
        is_old = sid < "20260630"
        n_frames = len(acts)
        _lbl = _infer_action_label

        frames_meta = []
        for i in range(n_frames):
            cx, cy, area, has = float(bbox[i,0]), float(bbox[i,1]), float(bbox[i,2]), bool(bbox[i,3])
            ca  = float(cached[i])
            lat = float(lats[i])
            action_str = _lbl(acts[i])
            
            is_arrival = (i == n_frames-1 and ca == -1.0 and action_str == "STOP")
            is_prev    = (ca == 0.0 and action_str in ("ROT_L", "ROT_R"))
            ftype = ("★ARRIVAL" if is_arrival else
                     "🔄PREVIEW" if is_prev else
                     "📡live" if ca == 0.0 else "💾cache")
            
            # 이상치 검증
            warns = []
            if is_prev and lat == 0:
                warns.append("⚠️ preview latency=0ms")
            if ca == 0.0 and lat == 0 and not is_prev and not is_arrival:
                warns.append("⚠️ live PG2 latency=0ms")
            if has and abs(cx - 0.5) < 0.001:
                warns.append("⚠️ has_bbox=True지만 cx=0.5 (fallback)")
            if has and area == 0:
                warns.append("⚠️ has_bbox=True지만 area=0")
            if not has and abs(cx - 0.5) > 0.01:
                warns.append(f"⚠️ has_bbox=False인데 cx={cx:.3f} (모순)")

            user_label = labels.get(f"session_{sid}_f{i}", "")

            # 라벨 일관성 검사
            if has and user_label and user_label != "NONE":
                if user_label == "L" and cx > 0.55: warns.append("⚠️ L 라벨이지만 cx 우측 치우침")
                if user_label == "R" and cx < 0.45: warns.append("⚠️ R 라벨이지만 cx 좌측 치우침")
                if user_label == "C" and (cx < 0.35 or cx > 0.65): warns.append("⚠️ C 라벨이지만 cx 이탈")

            frames_meta.append({
                "idx": i,
                "action": action_str,
                "cx": cx, "cy": cy, "area": area, "has_bbox": has,
                "latency_ms": lat, "cached": ca, "type": ftype,
                "warns": warns,
                "user_label": user_label
            })
            
        # 경로검증(episode_log.csv) 기록 중 이 세션과 매칭되는 행 — 있으면 동봉
        episode = None
        try:
            ep_rows, _ = _read_episode_csv()
            for r in ep_rows:
                if len(r) > 13 and r[13] == sid:
                    episode = {
                        "row": r[0], "path_type": r[1], "success": r[2],
                        "steps": r[3], "lat_ms": r[4], "top_action": r[5],
                        "gnd_pct": r[6], "area": r[7], "cx": r[8], "stop": r[9],
                        "fpe": r[10], "note": r[11], "date": r[12],
                    }
                    break
        except Exception:
            pass

        return {
            "ok": True,
            "sid": sid,
            "attrs": {k: str(v) for k, v in attrs.items()},
            "frames": frames_meta,
            "episode": episode,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": f"H5 로딩 오류: {e}"})


@app.get("/sessions/frame")
def sessions_frame(sid: str, idx: int):
    """H5 파일에서 특정 인덱스의 이미지 한 프레임을 JPEG 파일로 직접 제공"""
    h5p = INFER_H5_DIR / f"session_{sid}.h5"
    if not h5p.exists():
        return Response(status_code=404, content="H5 file not found")
        
    try:
        with h5py.File(h5p, "r") as f:
            img_arr = f["observations/images"][idx]
            
        # JPEG 변환
        pil = Image.fromarray(img_arr.astype(np.uint8))
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=80)
        return Response(content=buf.getvalue(), media_type="image/jpeg")
    except Exception as e:
        return Response(status_code=500, content=f"Frame load error: {e}")


# ─── 🌀 오버슈트 가이드 — 트랙C(오버슈트→재수렴) 수집 예시 ─────────────────
# CH62(docs/v5/closed_loop_eval/CH62_FORWARD_LOCK_AND_LABEL_CONFOUND.md §2)의
# 실패 반례 세션(205228)에서 "과도한 회전 → 반등 없이 계속 밀림" 실구간을 그대로
# 보여주고, 이후 "이렇게 반대로 재보정했어야 함"을 같은 세션 프레임을
# 좌우반전+역순재생한 합성(실제 센서 데이터 아님) 구간으로 이어붙여 시연한다.
# 실제 재보정 데이터 자체가 없다는 게 CH62/트랙C의 문제의식이라 합성 외엔 방법이 없음.
OVERSHOOT_DEMO_SID = "20260711_205228"
OVERSHOOT_DEMO_REAL_FRAMES = 9  # idx 0..8 — CH62 표에서 cx가 0.73→0.19로 반등없이 하강한 구간


def _overshoot_guide_compose(direction: str, idx: int):
    """합성 타임라인 idx -> (원본 real_idx, phase, mirror 여부).

    idx 0..8: 실제 프레임 그대로(phase=real)
    idx 9..16: 실제 8..1 프레임을 좌우반전한 합성 재보정 예시(phase=synthetic)
    direction=right_recover면 전체를 한 번 더 좌우반전(왼쪽 극단 사례를
    오른쪽 극단 사례처럼 보이게) — 실제로 새 세션을 만들지 않고 기존
    좌측 반례 하나로 양쪽 방향 예시를 다 보여주기 위한 트릭.
    """
    real_n = OVERSHOOT_DEMO_REAL_FRAMES
    if idx < real_n:
        real_idx, phase, local_mirror = idx, "real", False
    else:
        j = idx - real_n
        real_idx, phase, local_mirror = (real_n - 2 - j), "synthetic", True
    global_mirror = (direction == "right_recover")
    return real_idx, phase, (local_mirror != global_mirror)


def _overshoot_guide_len():
    return OVERSHOOT_DEMO_REAL_FRAMES + (OVERSHOOT_DEMO_REAL_FRAMES - 1)


@app.get("/overshoot_guide/load")
def overshoot_guide_load(direction: str = "left_recover"):
    if direction not in ("left_recover", "right_recover"):
        return JSONResponse(status_code=400, content={"ok": False, "error": "direction은 left_recover|right_recover만 허용"})
    h5p = INFER_H5_DIR / f"session_{OVERSHOOT_DEMO_SID}.h5"
    if not h5p.exists():
        return JSONResponse(status_code=404, content={"ok": False, "error": f"예시 원본 세션 없음: {h5p}"})
    try:
        with h5py.File(h5p, "r") as f:
            acts = f["actions"][()]
            bbox = f["grounding/bbox"][()]

        frames = []
        total = _overshoot_guide_len()
        for idx in range(total):
            real_idx, phase, mirror = _overshoot_guide_compose(direction, idx)
            cx = float(bbox[real_idx, 0])
            action = _infer_action_label(acts[real_idx])
            if mirror:
                cx = 1.0 - cx
            note = (
                "실제 데이터 — 과도한 회전 이후 cx가 반등 없이 계속 밀림 (CH62 205228)"
                if phase == "real" else
                "합성 예시 — 같은 세션 프레임 좌우반전, 실제 센서 데이터 아님. "
                "\"여기서부터 반대방향 재보정이 있었어야 함\"을 보여주기 위한 시각 자료"
            )
            frames.append({
                "idx": idx, "source_real_idx": real_idx, "phase": phase,
                "mirrored": mirror, "cx": cx, "action": action, "note": note,
            })
        return {
            "ok": True, "direction": direction, "source_sid": OVERSHOOT_DEMO_SID,
            "real_frame_count": OVERSHOOT_DEMO_REAL_FRAMES, "frames": frames,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": f"로딩 오류: {e}"})


@app.get("/overshoot_guide/frame")
def overshoot_guide_frame(direction: str, idx: int):
    h5p = INFER_H5_DIR / f"session_{OVERSHOOT_DEMO_SID}.h5"
    if not h5p.exists():
        return Response(status_code=404, content="H5 file not found")
    try:
        real_idx, _phase, mirror = _overshoot_guide_compose(direction, idx)
        with h5py.File(h5p, "r") as f:
            img_arr = f["observations/images"][real_idx]
        pil = Image.fromarray(img_arr.astype(np.uint8))
        if mirror:
            pil = ImageOps.mirror(pil)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=80)
        return Response(content=buf.getvalue(), media_type="image/jpeg")
    except Exception as e:
        return Response(status_code=500, content=f"Frame load error: {e}")


# ═══════════════════════════════════════════════════════════════════
# 데이터셋 히스토리 — 데이터수집(collect) 탭이 저장한 원본 학습 H5 브라우징.
# /sessions/* (추론세션, INFER_H5_DIR)와는 완전히 다른 데이터 소스.
# 스키마 2종 혼재: 레거시(observations/images + scenario/pattern/distance/
# end_pos attrs, 289개) vs 신규 대시보드 포맷(images + episode_name/
# cx_position/cx_path/scenario attrs). 아래 헬퍼가 두 스키마를 흡수한다.
# ═══════════════════════════════════════════════════════════════════
_DS_NAME_RE = re.compile(r"^episode_\d+_\d+_(?P<rest>.+)$")


def _dataset_h5_paths():
    return sorted(glob.glob(str(_collect.data_dir / "*.h5")), reverse=True)


def _dataset_parse_scenario_from_name(name: str) -> str:
    """attrs에 scenario가 없는 파일(구버전 또는 커스텀 이름)의 폴백 —
    자동생성 이름(episode_{ts}_{scenario}_{pattern}_{cx_pos}_{cx_path})에서
    scenario로 보이는 토큰을 추출. 매칭 안 되면 빈 문자열."""
    m = _DS_NAME_RE.match(name)
    if not m:
        return ""
    rest = m.group("rest")
    if rest in COLLECT_SCENARIOS:
        return rest
    for key in COLLECT_SCENARIOS:
        if rest.startswith(key + "_") or rest == key:
            return key
    return ""


def _dataset_scan_attrs(h5p: str) -> dict:
    """파일을 열지 않고(attrs만) 목록용 메타데이터 추출 — 290개 스캔이
    빨라야 하므로 이미지 데이터셋은 절대 안 읽음."""
    name = Path(h5p).stem
    with h5py.File(h5p, "r") as f:
        attrs = dict(f.attrs)
        schema = "new" if "images" in f else "legacy"
        if schema == "new":
            n_frames = int(attrs.get("num_frames", f["images"].shape[0] if "images" in f else 0))
        else:
            n_frames = int(f["observations/images"].shape[0]) if "observations/images" in f else 0

    scenario = str(attrs.get("scenario", "") or "") or _dataset_parse_scenario_from_name(name)
    cx_position = str(attrs.get("cx_position", "") or "")
    cx_path = str(attrs.get("cx_path", "") or "")
    pattern = str(attrs.get("pattern", "") or "")
    duration_s = float(attrs.get("total_duration", 0.0) or 0.0)
    collection_dt = attrs.get("collection_datetime", "")

    mtime = os.path.getmtime(h5p)
    dt = (datetime.datetime.fromisoformat(collection_dt) if collection_dt
          else datetime.datetime.fromtimestamp(mtime))

    return {
        "name": name,
        "date": dt.strftime("%Y-%m-%d"),
        "time": dt.strftime("%H:%M:%S"),
        "scenario": scenario,
        "cx_position": cx_position,
        "cx_path": cx_path,
        "pattern": pattern,
        "num_frames": n_frames,
        "duration_s": round(duration_s, 1),
        "size_mb": round(os.path.getsize(h5p) / (1024 * 1024), 2),
        "schema": schema,
    }


@app.get("/dataset/list")
def dataset_list():
    items = []
    for h5p in _dataset_h5_paths():
        try:
            items.append(_dataset_scan_attrs(h5p))
        except Exception as e:
            log.warning(f"[DatasetHistory] {h5p} 스캔 실패: {e}")
    scenarios = sorted({it["scenario"] for it in items if it["scenario"]})
    cx_positions = sorted({it["cx_position"] for it in items if it["cx_position"]})
    return {"ok": True, "items": items, "scenarios": scenarios, "cx_positions": cx_positions,
            "scenario_labels": {k: v["label"] for k, v in COLLECT_SCENARIOS.items()},
            "cx_position_labels": {k: v["label"] for k, v in COLLECT_CX_POSITIONS.items()}}


@app.get("/dataset/load")
def dataset_load(name: str):
    h5p = _collect.data_dir / f"{name}.h5"
    if not h5p.exists():
        return JSONResponse(status_code=404, content={"ok": False, "error": f"H5 파일이 없음: {h5p}"})
    try:
        with h5py.File(h5p, "r") as f:
            attrs = dict(f.attrs)
            schema = "new" if "images" in f else "legacy"
            acts = f["actions"][()]
            n_frames = len(acts)
            event_types = (list(f["action_event_types"][()]) if "action_event_types" in f
                           else [""] * n_frames)

        frames_meta = []
        for i in range(n_frames):
            a = acts[i]
            action = {"linear_x": float(a[0]), "linear_y": float(a[1]),
                      "angular_z": float(a[2]) if len(a) > 2 else 0.0}
            cls = _collect_classify_8class(action)
            et = event_types[i]
            et = et.decode("utf-8") if isinstance(et, bytes) else str(et)
            frames_meta.append({
                "idx": i, "action_class": COLLECT_CLASS_NAMES_8[cls],
                "symbol": COLLECT_CLASS_SYMBOLS[cls], "event_type": et,
                "linear_x": action["linear_x"], "linear_y": action["linear_y"],
                "angular_z": action["angular_z"],
            })

        meta = _dataset_scan_attrs(str(h5p))
        return {"ok": True, "meta": meta, "attrs": {k: (v if not hasattr(v, "item") else v.item())
                                                      for k, v in attrs.items()},
                "frames": frames_meta, "schema": schema}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})


@app.get("/dataset/frame")
def dataset_frame(name: str, idx: int):
    h5p = _collect.data_dir / f"{name}.h5"
    if not h5p.exists():
        return Response(status_code=404, content="H5 file not found")
    try:
        with h5py.File(h5p, "r") as f:
            is_new_schema = "images" in f
            img_key = "images" if is_new_schema else "observations/images"
            img_arr = f[img_key][idx]
        # H5 raw 저장 자체는 건드리지 않음(학습 로더 쪽 결정 사안) — 여기는 사람이 보는
        # 뷰어 표시용 보정만. 실측 결과 신규(mona_dashboard, "images" 루트) 스키마만
        # 실제로 BGR로 저장되어 뒤집혀 보이고, 구(레거시, "observations/images") 스키마는
        # 저장 시점부터 이미 RGB라 변환하면 오히려 깨짐 — 스키마별로 분기.
        if is_new_schema:
            rgb_arr = cv2.cvtColor(img_arr.astype(np.uint8), cv2.COLOR_BGR2RGB)
        else:
            rgb_arr = img_arr.astype(np.uint8)
        pil = Image.fromarray(rgb_arr)
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=80)
        return Response(content=buf.getvalue(), media_type="image/jpeg")
    except Exception as e:
        return Response(status_code=500, content=f"Frame load error: {e}")


@app.post("/dataset/delete")
def dataset_delete(name: str):
    """데이터셋 히스토리 탭에서 선택 삭제 — H5 파일 제거 + 실행 중인 서버의
    in-memory 진행률(scenario_stats/cx_position_stats/time_period_stats)도
    같이 감소시켜 scenario_progress.json/time_period_stats.json에 반영.
    (재시작 없이 즉시 /collect/state에 반영되도록 _collect._save_progress() 재사용)"""
    h5p = _collect.data_dir / f"{name}.h5"
    if not h5p.exists():
        return JSONResponse(status_code=404, content={"ok": False, "error": f"파일 없음: {name}.h5"})

    try:
        with h5py.File(h5p, "r") as f:
            attrs = dict(f.attrs)
    except Exception as e:
        attrs = {}
        log.warning(f"[DatasetHistory] 삭제 전 attrs 읽기 실패({name}): {e}")

    try:
        h5p.unlink()
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": f"삭제 실패: {e}"})

    scenario = str(attrs.get("scenario", "") or "")
    cx_position = str(attrs.get("cx_position", "") or "")
    cx_path = str(attrs.get("cx_path", "") or "")
    time_period = str(attrs.get("time_period", "") or "")

    def _dec(d: dict, key: str):
        if key and key in d:
            d[key] = max(0, d[key] - 1)
            if d[key] == 0:
                del d[key]

    with _collect._lock:
        _dec(_collect.scenario_stats, scenario)
        _dec(_collect.cx_position_stats, cx_position)
        _dec(_collect.cx_position_path_stats, f"{cx_position}::{cx_path}" if cx_position and cx_path else "")
        _dec(_collect.time_period_stats, time_period)
    _collect._save_progress()

    log.info(f"🗑️ [DatasetHistory] 삭제: {name} (scenario={scenario or '-'}, cx={cx_position or '-'}::{cx_path or '-'})")
    return {"ok": True, "deleted": name}


@app.post("/sessions/label")
def sessions_label(req: LabelSaveReq):
    labels = {}
    if LABEL_JSON_PATH.exists():
        try: labels = json.loads(LABEL_JSON_PATH.read_text())
        except Exception: pass
        
    key = f"session_{req.session_id}_f{req.frame_idx}"
    if req.label == "NONE":
        labels.pop(key, None)
    else:
        labels[key] = req.label
        
    try:
        LABEL_JSON_PATH.write_text(json.dumps(labels, indent=2, ensure_ascii=False))
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": f"라벨 파일 쓰기 실패: {e}"}


@app.post("/sessions/delete")
def sessions_delete(sid: str):
    h5p = INFER_H5_DIR / f"session_{sid}.h5"
    jsonp = INFER_REPORT_DIR / f"session_{sid}.json"
    deleted_files = []
    try:
        if h5p.exists():
            h5p.unlink()
            deleted_files.append(h5p.name)
        if jsonp.exists():
            jsonp.unlink()
            deleted_files.append(jsonp.name)
        
        # /tmp/mona_preview_labels.json에서도 해당 세션 라벨 제거
        if LABEL_JSON_PATH.exists():
            try:
                labels = json.loads(LABEL_JSON_PATH.read_text())
                keys_to_pop = [k for k in labels if k.startswith(f"session_{sid}_")]
                for k in keys_to_pop:
                    labels.pop(k, None)
                LABEL_JSON_PATH.write_text(json.dumps(labels, indent=2, ensure_ascii=False))
            except Exception: pass
            
        return {"ok": True, "message": f"성공적으로 삭제됨: {', '.join(deleted_files)}"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": f"파일 삭제 실패: {e}"})


EPISODE_CSV = ROOT / "logs" / "episode_log.csv"
# 🧪 실험용/테스트용 기록 — 정식 스크리닝 집계(episode_log.csv)와 완전히 분리된
# 별도 파일. 조이스틱 검증모드 SEL(2026-07-23, 남는 키)로 토글해서 기록 대상을
# 여기로 돌림 — 그라운더 A/B 테스트처럼 "정식 수치로 안 셀" 시도를 위함.
EXPERIMENTAL_EPISODE_CSV = ROOT / "logs" / "episode_log_experimental.csv"
EP_HEADERS  = ["#", "경로", "결과", "steps", "lat(ms)", "top액션", "gnd%", "area", "cx", "STOP", "FPE", "메모", "날짜", "session_id"]
PATH_TYPES = ["right_right", "right_left", "right_straight",
              "center_straight", "center_left", "center_right",
              "left_straight", "left_left", "left_right",
              "obj_left", "obj_center", "obj_right",
              "dist_10cm", "dist_20cm", "dist_30cm",
              # ── 트랙A 극단배치(V6, docs/DATASET_V6_STATUS.md) 주행검증 라벨 ──
              # 기존 15종(위) 옆에 추가만 함 — 과거 검증기록/분석 스크립트가
              # 위 15종을 참조 중일 수 있어 대체하지 않음. strong_left는
              # 2026-07-16 기준 미수집(target=15, 0건) — UI에서 "미수집" 표시.
              "trackA_weak_left_left_curve", "trackA_weak_left_straight", "trackA_weak_left_right_curve",
              "trackA_weak_right_left_curve", "trackA_weak_right_straight", "trackA_weak_right_right_curve",
              "trackA_strong_right_left_curve", "trackA_strong_right_straight", "trackA_strong_right_right_curve",
              "trackA_strong_left_left_curve", "trackA_strong_left_straight", "trackA_strong_left_right_curve",
              # ── 트랙F 중앙위치(V6, 45ep 2026-07-16/17 수집 완료) ──
              # "center_straight/left/right"(위 15종, 구 9-시나리오 target_center_*)와
              # 이름이 겹쳐 보이지만 완전히 다른 taxonomy(cx_position="center")라
              # trackF_ 접두어로 구분(2026-07-22, 실기 테스트 착수 전 정리).
              "trackF_center_left_curve", "trackF_center_straight", "trackF_center_right_curve"]
PATH_TARGETS = {
    "right_right": 10, "right_left": 10, "right_straight": 10,
    "center_straight": 10, "center_left": 10, "center_right": 10,
    "left_straight": 10, "left_left": 10, "left_right": 10,
    "obj_left": 30, "obj_center": 30, "obj_right": 30,
    "dist_10cm": 10, "dist_20cm": 10, "dist_30cm": 10,
    "trackA_weak_left_left_curve": 15, "trackA_weak_left_straight": 15, "trackA_weak_left_right_curve": 15,
    "trackA_weak_right_left_curve": 15, "trackA_weak_right_straight": 15, "trackA_weak_right_right_curve": 15,
    "trackA_strong_right_left_curve": 15, "trackA_strong_right_straight": 15, "trackA_strong_right_right_curve": 15,
    "trackA_strong_left_left_curve": 15, "trackA_strong_left_straight": 15, "trackA_strong_left_right_curve": 15,
    "trackF_center_left_curve": 15, "trackF_center_straight": 15, "trackF_center_right_curve": 15,
}
# 트랙A 12종 + 트랙F 3종 전체 수집 완료(2026-07-16/17, 225/225) — 미수집 조합 없음
TRACKA_UNCOLLECTED = set()

def _get_episode_summary(rows):
    done_total = {k: 0 for k in PATH_TYPES}
    done_succ  = {k: 0 for k in PATH_TYPES}
    nav_succ = 0
    trackA_succ = 0
    trackF_succ = 0
    for r in rows:
        if len(r) < 3: continue
        pt = str(r[1]).replace(" ★", "").replace("★", "").strip()
        done_total[pt] = done_total.get(pt, 0) + 1
        if r[2] == "성공":
            done_succ[pt] = done_succ.get(pt, 0) + 1
            if pt.startswith("trackA_"):
                trackA_succ += 1
            elif pt.startswith("trackF_"):
                trackF_succ += 1
            elif not pt.startswith(("obj_", "dist_")):
                nav_succ += 1
    # 트랙A/트랙F(V6)는 기존 9종 nav 집계와 별개로 집계 — 서로 다른 목표(10 vs 15)를
    # 섞으면 퍼센트가 왜곡되므로 완전히 분리.
    nav_total = sum(PATH_TARGETS[k] for k in PATH_TARGETS if not k.startswith(("obj_", "dist_", "trackA_", "trackF_")))
    obj_done  = sum(done_total.get(k, 0) for k in ("obj_left","obj_center","obj_right"))
    obj_succ  = sum(done_succ.get(k, 0)  for k in ("obj_left","obj_center","obj_right"))
    dist_done = sum(done_total.get(k, 0) for k in ("dist_10cm","dist_20cm","dist_30cm"))
    dist_succ = sum(done_succ.get(k, 0)  for k in ("dist_10cm","dist_20cm","dist_30cm"))
    trackA_keys = [k for k in PATH_TYPES if k.startswith("trackA_")]
    trackA_total = sum(PATH_TARGETS[k] for k in trackA_keys)
    trackA_done = sum(done_total.get(k, 0) for k in trackA_keys)
    trackF_keys = [k for k in PATH_TYPES if k.startswith("trackF_")]
    trackF_total = sum(PATH_TARGETS[k] for k in trackF_keys)
    trackF_done = sum(done_total.get(k, 0) for k in trackF_keys)
    return (f"경로검증 {sum(done_total.get(k,0) for k in PATH_TYPES if not k.startswith(('obj_','dist_','trackA_','trackF_')))}/{nav_total} "
            f"성공 {nav_succ}/20 (목표) | 위치별 {obj_done}/90 ({obj_succ}성공) | 거리별 {dist_done}/30 ({dist_succ}성공) "
            f"| 트랙A {trackA_done}/{trackA_total} ({trackA_succ}성공) | 트랙F {trackF_done}/{trackF_total} ({trackF_succ}성공)")

def _read_episode_csv(csv_path=None):
    csv_path = csv_path or EPISODE_CSV
    if not csv_path.exists():
        return [], "에피소드 기록 없음"
    import csv
    rows = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None) # skip header
            for r in reader:
                if r: rows.append(r)
    except Exception: pass
    return rows, _get_episode_summary(rows)

_WIKI_FILES = {
    "info": ROOT / "docs" / "DASHBOARD_WIKI.md",
    "status": ROOT / "docs" / "DASHBOARD_LIVE_STATUS.md",
}

@app.get("/wiki/{name}")
def wiki_content(name: str):
    path = _WIKI_FILES.get(name)
    if path is None:
        return {"ok": False, "error": f"알 수 없는 위키: {name}"}
    if not path.exists():
        return {"ok": False, "error": f"파일 없음: {path}"}
    content = path.read_text(encoding="utf-8")
    mtime = datetime.datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return {"ok": True, "content": content, "mtime": mtime}

def _journal_kind(subject: str) -> str:
    """커밋 제목에서 타임라인 색상 분류 (research_story.html .timeline-item 관례 따름)."""
    s = subject.lower()
    if any(k in subject for k in ("재규명", "확정", "pivot", "번복")):
        return "pivot"
    if s.startswith("fix") or "버그" in subject or "bug" in s:
        return "bad"
    if s.startswith("feat") or s.startswith("docs"):
        return "good"
    return "info"

@app.get("/journal")
def research_journal(limit: int = 40):
    """연구일지 — md 파일 버전이 아니라 프로젝트 전체 커밋을 research_story.html
    타임라인 감성으로 훑어보는 용도. 위키/최신현황 탭 공용."""
    import subprocess as _sp_wiki
    try:
        out = _sp_wiki.run(
            ["git", "log", f"-{max(1, min(limit, 100))}",
             "--format=%H%x1f%ad%x1f%s", "--date=format:%Y-%m-%d %H:%M"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=10,
        )
    except Exception as e:
        return {"ok": False, "error": str(e)}
    if out.returncode != 0:
        return {"ok": False, "error": out.stderr.strip() or "git log 실패"}
    entries = []
    for line in out.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("\x1f")
        if len(parts) == 3:
            sha, date, subject = parts
            entries.append({"sha": sha, "date": date, "subject": subject, "kind": _journal_kind(subject)})
    return {"ok": True, "entries": entries}

@app.get("/journal/{sha}")
def research_journal_entry(sha: str):
    """연구일지 엔트리 펼치기 — 커밋 전체 메시지 + 변경 파일 목록."""
    import re as _re_wiki
    import subprocess as _sp_wiki
    if not _re_wiki.fullmatch(r"[0-9a-fA-F]{7,40}", sha):
        return {"ok": False, "error": "잘못된 커밋 해시"}
    try:
        body_out = _sp_wiki.run(["git", "show", "-s", "--format=%B", sha],
                                 cwd=str(ROOT), capture_output=True, text=True, timeout=10)
        files_out = _sp_wiki.run(["git", "show", "--stat", "--format=", sha],
                                  cwd=str(ROOT), capture_output=True, text=True, timeout=10)
    except Exception as e:
        return {"ok": False, "error": str(e)}
    if body_out.returncode != 0:
        return {"ok": False, "error": body_out.stderr.strip() or "git show 실패"}
    return {"ok": True, "sha": sha, "body": body_out.stdout.strip(), "files": files_out.stdout.strip()}

@app.get("/episodes/list")
def episodes_list():
    rows, summary = _read_episode_csv()
    return {"ok": True, "episodes": rows, "summary": summary}


# ── session_id → checkpoint 조회 (검증 스크리닝 체크포인트/시점 필터용, 2026-07-23) ──
_checkpoint_index_cache: dict = {"mtime": 0.0, "map": {}}


@app.get("/verify/checkpoint_index")
def verify_checkpoint_index():
    """docs/inference_reports/session_*.json을 스캔해 {session_id: checkpoint_basename}
    맵을 만든다. A/B 체크포인트 전환 중 스크리닝 패널을 체크포인트별로 필터하는 데 씀.
    폴더 mtime 기준 캐시 — 새 세션 없으면 재스캔 안 함."""
    try:
        cur_mtime = INFER_REPORT_DIR.stat().st_mtime
    except Exception:
        cur_mtime = 0.0
    if _checkpoint_index_cache["mtime"] != cur_mtime:
        idx = {}
        for f in INFER_REPORT_DIR.glob("session_2026*.json"):
            try:
                d = json.loads(f.read_text())
            except Exception:
                continue
            sid = d.get("session_id") or f.stem.replace("session_", "")
            ckpt = (d.get("runtime_config") or {}).get("checkpoint_path", "")
            idx[sid] = os.path.basename(ckpt) if ckpt else "(알수없음)"
        _checkpoint_index_cache["map"] = idx
        _checkpoint_index_cache["mtime"] = cur_mtime
    m = _checkpoint_index_cache["map"]
    checkpoints = sorted(set(m.values()))
    return {"ok": True, "session_checkpoint": m, "checkpoints": checkpoints}

@app.post("/episodes/log")
def episodes_log(req: EpisodeLogReq):
    target_csv = EXPERIMENTAL_EPISODE_CSV if req.experimental else EPISODE_CSV
    rows, _ = _read_episode_csv(target_csv)

    steps = _state["step"]
    avg_lat = 0.0
    top_action = "—"
    gnd_pct = 0.0
    area = 0.0
    cx = 0.5
    
    if _state["run_history"]:
        lats = [r[2] for r in _state["run_history"] if isinstance(r[2], (int, float))]
        avg_lat = round(sum(lats) / len(lats), 1) if lats else 0.0
        
        labels = [r[1] for r in _state["run_history"] if r[1] and r[1] != "—"]
        if labels:
            from collections import Counter
            top_action = Counter(labels).most_common(1)[0][0]
            
        gnd_ok = sum(1 for r in _state["run_history"] if r[3] != "—")
        gnd_pct = round(gnd_ok / len(_state["run_history"]) * 100, 1) if _state["run_history"] else 0.0
        
    if _state["bbox"]:
        area = _state["bbox"].get("area", 0.0)
        cx = _state["bbox"].get("cx", 0.5)
        
    stop_flag = "Y" if area >= 0.18 else "N"
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    
    new_row = [
        len(rows) + 1,
        req.path_type,
        req.success,
        steps,
        avg_lat,
        top_action,
        gnd_pct,
        round(area, 3),
        round(cx, 2),
        stop_flag,
        req.fpe,
        req.note,
        date_str,
        _state.get("session_id") or "",
    ]
    
    import csv
    target_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not target_csv.exists()
    with open(target_csv, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(EP_HEADERS)
        w.writerow(new_row)

    _, summary = _read_episode_csv(target_csv)
    return {"ok": True, "summary": summary}

@app.post("/episodes/undo")
def episodes_undo():
    rows, _ = _read_episode_csv()
    if not rows:
        return {"ok": False, "error": "기록이 없습니다."}
    new_rows = rows[:-1]
    for i, r in enumerate(new_rows):
        r[0] = i + 1
    import csv
    EPISODE_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(EPISODE_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(EP_HEADERS)
        w.writerows(new_rows)
    _, summary = _read_episode_csv()
    return {"ok": True, "episodes": new_rows, "summary": summary}

@app.post("/episodes/clear")
def episodes_clear():
    if EPISODE_CSV.exists():
        EPISODE_CSV.unlink()
    return {"ok": True, "episodes": [], "summary": "에피소드 기록 없음"}


@app.post("/episodes/update")
def episodes_update(req: EpisodeUpdateReq):
    """테이블 행 클릭 → 수정 패널에서 값 편집 후 저장 — 해당 # 행만 in-place 수정."""
    rows, _ = _read_episode_csv()
    idx = next((i for i, r in enumerate(rows) if str(r[0]) == str(req.row)), None)
    if idx is None:
        return {"ok": False, "error": f"#{req.row} 행을 찾을 수 없습니다."}
    r = rows[idx]
    r[1] = req.path_type
    r[2] = req.success
    r[3] = req.steps
    r[4] = req.lat_ms
    r[10] = req.fpe
    r[11] = req.note
    import csv
    with open(EPISODE_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(EP_HEADERS)
        w.writerows(rows)
    _, summary = _read_episode_csv()
    return {"ok": True, "episodes": rows, "summary": summary}


# ─── 대시보드 메인 ────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=_DASHBOARD_HTML)


# ─── 정적 파일 마운트 (존재 시) ──────────────────────────────────
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# 위키/최신현황 탭에서 docs/v5 아래 실험 그래프 이미지를 상대경로로 쓰기 위한 마운트
_DOCS_V5_DIR = ROOT / "docs" / "v5"
if _DOCS_V5_DIR.exists():
    app.mount("/docs-static/v5", StaticFiles(directory=str(_DOCS_V5_DIR)), name="docs-v5-static")


# ═══════════════════════════════════════════════════════════════════
# 내장 대시보드 HTML (Premium Command Center Style)
# ═══════════════════════════════════════════════════════════════════
_DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MoNaVLA Command Center</title>
<!-- Google Fonts & Chart.js -->
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700;900&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  :root {
    --bg-dark: #090d16;
    --panel-dark: #0f1524;
    --border-glow: #1d2b45;
    --text-primary: #f1f5f9;
    --text-muted: #94a3b8;
    --cyan: #06b6d4;
    --cyan-glow: rgba(6, 182, 212, 0.4);
    --emerald: #10b981;
    --emerald-glow: rgba(16, 185, 129, 0.4);
    --amber: #f59e0b;
    --rose: #f43f5e;
    --rose-glow: rgba(244, 63, 94, 0.4);
    --violet: #8b5cf6;
    --font-sans: 'Outfit', sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
  }
  
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background-color: var(--bg-dark);
    color: var(--text-primary);
    font-family: var(--font-sans);
    height: 100vh;
    display: flex;
    overflow: hidden;
  }

  /* ── 사이드바 ── */
  aside {
    width: 280px;
    background-color: var(--panel-dark);
    border-right: 1px solid var(--border-glow);
    display: flex;
    flex-direction: column;
    z-index: 10;
  }
  .brand-panel {
    padding: 24px;
    border-bottom: 1px solid var(--border-glow);
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .brand-pulse {
    width: 10px;
    height: 10px;
    background-color: var(--emerald);
    border-radius: 50%;
    box-shadow: 0 0 10px var(--emerald);
    animation: pulse 1.8s infinite;
  }
  .brand-title {
    font-size: 19px;
    font-weight: 700;
    letter-spacing: 0.5px;
    background: linear-gradient(to right, var(--text-primary), var(--cyan));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  nav {
    flex: 1;
    padding: 20px 12px;
    display: flex;
    flex-direction: column;
    gap: 6px;
  }
  .nav-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    border-radius: 10px;
    color: var(--text-muted);
    text-decoration: none;
    font-weight: 600;
    font-size: 14px;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  .nav-item:hover {
    background-color: rgba(255, 255, 255, 0.03);
    color: var(--text-primary);
  }
  .nav-item.active {
    background: linear-gradient(135deg, rgba(6,182,212,0.1), rgba(6,182,212,0.02));
    border-left: 3px solid var(--cyan);
    color: var(--cyan);
    box-shadow: inset 0 0 10px rgba(6, 182, 212, 0.05);
  }
  .sidebar-footer {
    padding: 20px;
    border-top: 1px solid var(--border-glow);
    font-size: 11px;
    color: var(--text-muted);
    font-family: var(--font-mono);
  }

  /* ── 메인 컨테이너 ── */
  main {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  header {
    background-color: var(--panel-dark);
    height: 70px;
    border-bottom: 1px solid var(--border-glow);
    padding: 0 32px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .srv-status-group {
    display: flex;
    align-items: center;
    gap: 16px;
  }
  .srv-pill {
    background: #151f32;
    border: 1px solid var(--border-glow);
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 12px;
    font-weight: 600;
    color: var(--text-muted);
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .srv-pill span.indicator {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background-color: var(--text-muted);
  }
  .srv-pill.online span.indicator {
    background-color: var(--emerald);
    box-shadow: 0 0 8px var(--emerald);
  }
  .srv-pill.offline span.indicator {
    background-color: var(--rose);
    box-shadow: 0 0 8px var(--rose);
  }

  /* ── 세션 히스토리 카드 리스트 ── */
  .session-date-header {
    position: sticky;
    top: 0;
    z-index: 2;
    background: var(--bg-dark);
    color: var(--text-muted);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    padding: 6px 4px 4px;
  }
  .session-card {
    background: #151f32;
    border: 1px solid var(--border-glow);
    border-radius: 8px;
    padding: 10px 12px;
    cursor: pointer;
    transition: border-color 0.15s ease, background 0.15s ease, transform 0.1s ease;
  }
  .session-card:hover {
    border-color: rgba(6,182,212,0.5);
    background: #1a2540;
  }
  .session-card.active {
    border-color: var(--cyan);
    background: rgba(6,182,212,0.08);
    box-shadow: 0 0 0 1px rgba(6,182,212,0.25);
  }
  .session-card .sc-top {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 8px;
  }
  .session-card .sc-sid {
    font-weight: 700;
    font-size: 12.5px;
    color: var(--cyan);
    font-family: var(--font-mono);
  }
  .session-card .sc-time {
    font-size: 10px;
    color: var(--text-muted);
    white-space: nowrap;
  }
  .session-card .sc-entity {
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 3px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .session-card .sc-bottom {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 6px;
  }
  .session-card .sc-badge {
    font-size: 10px;
    font-weight: 700;
    padding: 2px 7px;
    border-radius: 20px;
    background: rgba(148,163,184,0.12);
    color: var(--text-muted);
  }
  .session-card .sc-badge.labeled {
    background: rgba(16,185,129,0.15);
    color: var(--emerald);
  }

  /* ── 프레임 인스펙터 — 요약/경로검증 기록용 미니 타일 ── */
  .mini-tile-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(88px, 1fr));
    gap: 6px;
  }
  .mini-tile {
    background: #101726;
    border: 1px solid var(--border-glow);
    border-radius: 8px;
    padding: 7px 10px;
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 0;
  }
  .mini-tile .mt-label {
    font-size: 9px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-weight: 700;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .mini-tile .mt-value {
    font-size: 14px;
    font-weight: 700;
    color: #fff;
    font-family: var(--font-mono);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .mini-tile .mt-sub {
    font-size: 9px;
    font-weight: 400;
    color: var(--text-muted);
  }
  .mini-tile.mt-good .mt-value { color: var(--emerald); }
  .mini-tile.mt-bad .mt-value { color: var(--rose); }
  .mini-tile.mt-warn .mt-value { color: var(--amber); }
  .mini-tile.mt-accent .mt-value { color: var(--cyan); }

  /* ── 탭 레이아웃 ── */
  .tab-content {
    display: none;
    flex: 1;
    overflow: hidden;
  }
  .tab-content.active {
    display: flex;
  }
  .scroll-container {
    flex: 1;
    overflow-y: auto;
    padding: 32px;
  }
  #tab-collect .scroll-container {
    zoom: 1.5; /* 데이터수집 탭 전체 글씨/버튼 50% 확대 (사용자 요청) */
  }

  /* ── 카드 및 레이아웃 ── */
  .grid-2 {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
  }
  .grid-main {
    display: grid;
    grid-template-columns: 1.4fr 1fr;
    gap: 24px;
  }
  .card {
    background-color: var(--panel-dark);
    border: 1px solid var(--border-glow);
    border-radius: 16px;
    padding: 24px;
    position: relative;
    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
  }
  .card-title {
    font-size: 15px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--cyan);
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  /* ── 카메라 뷰 포트 ── */
  .viewport-wrapper {
    position: relative;
    border-radius: 12px;
    overflow: hidden;
    background-color: #000;
    aspect-ratio: 16/9;
    border: 1px solid var(--border-glow);
  }
  .viewport-img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
  }
  .viewport-canvas {
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    pointer-events: none;
    z-index: 5;
  }
  .overlay-info {
    position: absolute;
    bottom: 12px;
    left: 12px;
    right: 12px;
    display: flex;
    gap: 8px;
    z-index: 6;
  }
  .overlay-badge {
    background: rgba(15, 21, 36, 0.85);
    backdrop-filter: blur(8px);
    border: 1px solid var(--border-glow);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 600;
    color: var(--text-primary);
  }

  /* ── 폼 컨트롤 ── */
  .form-group {
    margin-bottom: 16px;
  }
  .form-group label {
    display: block;
    font-size: 12px;
    color: var(--text-muted);
    font-weight: 600;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
  }
  input[type=text], select {
    width: 100%;
    background-color: #0c101c;
    border: 1px solid var(--border-glow);
    border-radius: 8px;
    padding: 10px 14px;
    color: var(--text-primary);
    font-size: 14px;
    outline: none;
    font-family: var(--font-sans);
    transition: border-color 0.2s;
  }
  input[type=text]:focus, select:focus {
    border-color: var(--cyan);
  }
  
  /* ── 임계값 게이지 ── */
  .gauge-container {
    background: #090d16;
    border-radius: 8px;
    height: 24px;
    width: 100%;
    position: relative;
    overflow: hidden;
    border: 1px solid var(--border-glow);
  }
  .gauge-fill {
    height: 100%;
    background-color: var(--emerald);
    width: 0%;
    transition: width 0.1s linear, background-color 0.2s;
  }
  .gauge-marker {
    position: absolute;
    top: 0;
    bottom: 0;
    width: 2px;
    background-color: var(--rose);
    box-shadow: 0 0 8px var(--rose);
    z-index: 3;
  }

  /* ── 버튼 ── */
  .btn {
    padding: 11px 18px;
    border-radius: 10px;
    font-size: 14px;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.2s ease;
    border: none;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
  }
  .btn-cyan {
    background-color: var(--cyan);
    color: #000;
    box-shadow: 0 0 12px var(--cyan-glow);
  }
  .btn-cyan:hover:not(:disabled) {
    filter: brightness(1.15);
    box-shadow: 0 0 20px var(--cyan-glow);
  }
  .btn-rose {
    background-color: var(--rose);
    color: #fff;
    box-shadow: 0 0 12px var(--rose-glow);
  }
  .btn-rose:hover:not(:disabled) {
    filter: brightness(1.15);
    box-shadow: 0 0 20px var(--rose-glow);
  }
  .btn-outline {
    background-color: transparent;
    border: 1px solid var(--border-glow);
    color: var(--text-primary);
  }
  .btn-outline:hover:not(:disabled) {
    background-color: rgba(255, 255, 255, 0.03);
    border-color: var(--cyan);
  }
  .btn:active:not(:disabled) {
    transform: scale(0.94);
    filter: brightness(0.85);
  }
  .btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
  }
  /* 주행 START 버튼 — 눌러서 실행 중일 때 명확한 "실행 중" 표시 (페이드아웃 대신 펄스) */
  .btn-cyan.is-running {
    opacity: 1;
    animation: btnRunPulse 1.4s ease-in-out infinite;
  }
  @keyframes btnRunPulse {
    0%, 100% { box-shadow: 0 0 12px var(--cyan-glow); }
    50%      { box-shadow: 0 0 26px var(--cyan-glow); }
  }

  /* ── 데이터수집 상태카드 — Gradio gradio_data_collector.py .status-card 이식 ── */
  .collect-status-card {
    text-align:center; font-family:var(--font-mono); font-size:1.15rem; font-weight:700;
    background:#161b22; border-radius:10px; padding:14px 16px; margin-bottom:16px;
    border-left:5px solid var(--cyan); color:var(--text-primary);
  }
  .collect-status-card.rec { border-left-color:var(--rose); color:var(--rose); }
  .collect-status-card.done { border-left-color:var(--emerald); color:var(--emerald); }
  .collect-status-card.idle { border-left-color:var(--text-muted); color:var(--text-muted); }

  /* ── 조이스틱 요약 칩 — 카메라 아래 소형 박스, 섹션(이동/녹화/버튼/D-pad)별 색 구분 ── */
  .js-chip {
    display:inline-flex; align-items:center; gap:4px; font-size:10px; font-family:var(--font-mono);
    padding:4px 9px; border-radius:14px; white-space:nowrap; border:1px solid transparent;
  }
  .js-chip-move  { background:rgba(56,189,248,0.12); color:var(--cyan); border-color:rgba(56,189,248,0.3); }
  .js-chip-rec   { background:rgba(244,63,94,0.12); color:var(--rose); border-color:rgba(244,63,94,0.3); }
  .js-chip-amber { background:rgba(245,158,11,0.12); color:var(--amber); border-color:rgba(245,158,11,0.3); }
  .js-chip-muted { background:rgba(100,116,139,0.12); color:var(--text-muted); border-color:rgba(100,116,139,0.3); }
  .js-chip-cyan  { background:rgba(163,113,247,0.12); color:#a371f7; border-color:rgba(163,113,247,0.3); }

  /* ── 데이터수집 REC 표시 — 녹화 중임을 놓치지 않도록 점 깜빡임 ── */
  .collect-rec-dot {
    display:inline-block; width:9px; height:9px; border-radius:50%;
    background:var(--rose); margin-right:7px; vertical-align:middle;
    animation: collectRecPulse 1.1s ease-in-out infinite;
  }
  @keyframes collectRecPulse {
    0%, 100% { opacity:1; box-shadow:0 0 6px var(--rose); }
    50%      { opacity:0.25; box-shadow:0 0 0 var(--rose); }
  }

  /* ── 조이스틱 패널 ── */
  .joystick-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 8px;
    max-width: 210px;
    margin: 0 auto;
  }
  .joy-btn {
    aspect-ratio: 1;
    background: #131b2d;
    border: 1px solid var(--border-glow);
    color: var(--text-primary);
    border-radius: 12px;
    font-size: 16px;
    font-weight: 700;
    cursor: pointer;
    transition: all 0.15s;
  }
  .joy-btn:active, .joy-btn.active {
    background-color: var(--cyan);
    color: #000;
    box-shadow: 0 0 12px var(--cyan-glow);
    transform: scale(0.95);
  }
  .joy-btn.stop {
    background-color: var(--rose);
  }

  /* ── 조이스틱 버튼 라이트(숫자 블록) ── */
  .btn-light {
    width: 34px; height: 34px;
    display: flex; align-items: center; justify-content: center;
    background: #131b2d;
    border: 1px solid var(--border-glow);
    border-radius: 8px;
    font-size: 13px; font-weight: 700; font-family: var(--font-mono);
    color: var(--text-muted);
    transition: all 0.1s;
  }
  .btn-light.active {
    background-color: var(--emerald);
    color: #000;
    box-shadow: 0 0 12px rgba(16,185,129,0.6);
    transform: scale(1.08);
  }
  .btn-light.last {
    border-color: var(--cyan);
  }
  .btn-light-wrap {
    display: flex; flex-direction: column; align-items: center; gap: 3px;
  }
  .btn-light-name {
    font-size: 9px; color: var(--text-muted); font-family: var(--font-mono);
  }
  .btn-light-name.active {
    color: var(--emerald); font-weight: 700;
  }
  @keyframes flashHighlight {
    0%   { background: rgba(56,189,248,0.45); box-shadow: 0 0 0 2px var(--cyan); }
    100% { background: transparent; box-shadow: none; }
  }
  .flash-highlight { animation: flashHighlight 1s ease-out; border-radius: 8px; }

  /* ── 테이블 ── */
  .table-wrapper {
    overflow-x: auto;
    border: 1px solid var(--border-glow);
    border-radius: 10px;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    text-align: left;
  }
  th {
    background-color: #121828;
    color: var(--text-muted);
    padding: 10px 14px;
    font-weight: 600;
    border-bottom: 1px solid var(--border-glow);
  }
  td {
    padding: 10px 14px;
    border-bottom: 1px solid rgba(29, 43, 69, 0.3);
    color: var(--text-primary);
  }
  tr:last-child td {
    border-bottom: none;
  }
  .text-cyan { color: var(--cyan); }
  .text-emerald { color: var(--emerald); }
  .text-rose { color: var(--rose); }
  .text-amber { color: var(--amber); }

  /* ── 로딩 스피너 및 유틸 ── */
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }
  .pulse-border {
    animation: border-pulse 2s infinite;
  }
  @keyframes border-pulse {
    0%, 100% { border-color: var(--border-glow); }
    50% { border-color: var(--cyan); }
  }

  .status-console {
    background: #090d16;
    border: 1px solid var(--border-glow);
    border-radius: 10px;
    padding: 14px;
    font-family: var(--font-mono);
    font-size: 12px;
    color: var(--text-muted);
    min-height: 100px;
    max-height: 300px;
    overflow-y: auto;
    white-space: pre-wrap;
  }

  /* 슬라이더 스타일 */
  input[type=range] {
    width: 100%;
    accent-color: var(--cyan);
    margin-top: 8px;
  }

  /* 스위치 스타일 */
  .switch {
    position: relative;
    display: inline-block;
    width: 44px;
    height: 24px;
  }
  .switch input { opacity: 0; width: 0; height: 0; }
  .slider {
    position: absolute;
    cursor: pointer;
    top: 0; left: 0; right: 0; bottom: 0;
    background-color: #212a3e;
    transition: .3s;
    border-radius: 24px;
  }
  .slider:before {
    position: absolute;
    content: "";
    height: 16px; width: 16px;
    left: 4px; bottom: 4px;
    background-color: white;
    transition: .3s;
    border-radius: 50%;
  }
  input:checked + .slider { background-color: var(--cyan); }
  input:checked + .slider:before { transform: translateX(20px); }

  /* 프레임 인스펙터 레이아웃 */
  .frame-inspector {
    display: grid;
    grid-template-columns: 1.5fr 1fr;
    gap: 20px;
  }

  /* ── 위키/최신현황 탭 — docs/v5/research_story.html 스타일 이식 ──
     탭 스코프 한정: 다른 탭들의 Outfit/JetBrains Mono는 그대로 유지 */
  #tab-wikiinfo, #tab-wikistatus {
    --wiki-bg: #0a0f1a; --wiki-surface: #111827; --wiki-surface2: #1e293b;
    --wiki-border: #1e293b; --wiki-text: #f1f5f9; --wiki-muted: #94a3b8;
    --wiki-accent: #38bdf8; --wiki-green: #22c55e; --wiki-red: #ef4444;
    --wiki-yellow: #fbbf24; --wiki-purple: #a78bfa;
    font-family: 'Noto Sans KR', var(--font-sans);
  }
  #tab-wikiinfo .wiki-render, #tab-wikistatus .wiki-render {
    color: var(--wiki-text); line-height: 1.7;
  }
  .wiki-render h1, .wiki-render h2 {
    display:flex; align-items:center; gap:14px; margin:32px 0 18px;
    font-size:1.95rem; font-weight:800; color:var(--wiki-text);
  }
  .wiki-render h1:first-child, .wiki-render h2:first-child { margin-top:0; }
  .wiki-render h3 { font-size:1.43rem; font-weight:700; margin:22px 0 10px; color:var(--wiki-text); }
  .wiki-render p { margin-bottom:12px; color:var(--wiki-muted); font-size:1.22rem; }
  .wiki-render strong { color:var(--wiki-text); }
  .wiki-render ul { margin:0 0 14px 22px; color:var(--wiki-muted); font-size:1.22rem; }
  .wiki-render ul li { margin-bottom:4px; }
  .wiki-render code { font-family:var(--font-mono); background:var(--wiki-surface2);
    padding:1px 6px; border-radius:4px; font-size:0.85em; color:#7dd3fc; }
  .wiki-render pre { background:#0b1220; border:1px solid var(--wiki-border);
    border-radius:8px; padding:12px 14px; overflow:auto; margin-bottom:14px; }
  .wiki-render pre code { background:none; padding:0; color:var(--wiki-muted); }
  .wiki-chapter-num { background:var(--wiki-accent); color:#0a0f1a; font-weight:900;
    font-size:1.01rem; padding:3px 11px; border-radius:20px; white-space:nowrap; }
  .wiki-render table.cl-table { width:100%; border-collapse:collapse; background:var(--wiki-surface);
    border-radius:12px; overflow:hidden; margin-bottom:20px; font-size:1.14rem; }
  .wiki-render table.cl-table th { background:#0b1220; padding:10px 16px; text-align:left;
    color:var(--wiki-muted); font-size:1.01rem; text-transform:uppercase; letter-spacing:0.5px; }
  .wiki-render table.cl-table td { padding:10px 16px; border-bottom:1px solid var(--wiki-border); }
  .wiki-render table.cl-table tr:last-child td { border-bottom:none; }
  .wiki-render .callout { border-radius:10px; padding:14px 18px; margin-bottom:16px;
    line-height:1.7; font-size:1.17rem; }
  .wiki-render .callout.info { background:#0c2340; border-left:4px solid var(--wiki-accent); color:#bae6fd; }
  .wiki-render .callout.warn { background:#2d1b00; border-left:4px solid var(--wiki-yellow); color:#fde68a; }
  .wiki-render .callout.critical { background:#2d0000; border-left:4px solid var(--wiki-red); color:#fca5a5; }
  .wiki-render .callout.success { background:#052e16; border-left:4px solid var(--wiki-green); color:#bbf7d0; }
  .wiki-render .finding-card { background:var(--wiki-surface); border-radius:12px; padding:18px;
    border-left:4px solid var(--wiki-accent); margin-bottom:14px; }
  .wiki-render .img-grid-3 { display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin-bottom:16px; }
  .wiki-render .img-grid-3 img, .wiki-render .fig-card img { width:100%; display:block; border-radius:8px; }
  .wiki-render .fig-card { background:var(--wiki-surface); border-radius:12px; overflow:hidden; margin-bottom:14px; }
  .wiki-render .fig-caption { padding:10px 14px; font-size:1.07rem; color:var(--wiki-muted); }
  @media(max-width:760px){ .wiki-render .img-grid-3 { grid-template-columns:1fr; } }

  /* ── 연구일지(research journal) 타임라인 — research_story.html .timeline 이식 ── */
  /* ── 연구일지 — Tab6 세션 히스토리와 동일한 list+detail 스플릿 UX ── */
  .wj-grid { display:grid; grid-template-columns:260px 1fr; gap:14px; align-items:start; }
  @media(max-width:760px){ .wj-grid { grid-template-columns:1fr; } }
  .wj-list { display:flex; flex-direction:column; gap:6px; max-height:420px; overflow-y:auto; padding-right:4px; }
  .wj-card {
    background:#151f32; border:1px solid var(--border-glow); border-radius:8px;
    padding:9px 11px; cursor:pointer; transition:border-color 0.15s ease, background 0.15s ease;
    border-left:3px solid var(--wiki-accent);
  }
  .wj-card.bad { border-left-color:var(--wiki-red); }
  .wj-card.good { border-left-color:var(--wiki-green); }
  .wj-card.pivot { border-left-color:var(--wiki-purple); }
  .wj-card:hover { border-color:rgba(6,182,212,0.5); background:#1a2540; }
  .wj-card.active { border-color:var(--cyan); background:rgba(6,182,212,0.08);
    box-shadow:0 0 0 1px rgba(6,182,212,0.25); }
  .wj-card .wj-top { display:flex; justify-content:space-between; align-items:baseline; gap:8px; }
  .wj-card .wj-date { font-weight:700; font-size:12px; color:var(--wiki-accent); font-family:var(--font-mono); }
  .wj-card.bad .wj-date { color:var(--wiki-red); }
  .wj-card.good .wj-date { color:var(--wiki-green); }
  .wj-card.pivot .wj-date { color:var(--wiki-purple); }
  .wj-card .wj-rel { font-size:9.5px; color:var(--text-muted); white-space:nowrap; }
  .wj-card .wj-title { font-size:11px; color:var(--text-muted); margin-top:3px; overflow:hidden;
    text-overflow:ellipsis; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; }
  .wj-card .wj-sha { font-size:9.5px; color:var(--text-muted); font-family:var(--font-mono); margin-top:3px; }
  .wj-detail {
    background:#090d16; border:1px solid var(--border-glow); border-radius:8px;
    padding:16px 18px; min-height:200px; max-height:420px; overflow-y:auto;
  }
  .wj-detail .wjd-placeholder { color:var(--wiki-muted); font-size:12px; text-align:center; padding:60px 0; }
  .wj-detail .wjd-head { display:flex; align-items:baseline; gap:10px; margin-bottom:10px;
    padding-bottom:10px; border-bottom:1px solid var(--wiki-border); }
  .wj-detail .wjd-sha { font-family:var(--font-mono); color:var(--cyan); font-weight:700; font-size:13px; }
  .wj-detail .wjd-date { color:var(--wiki-muted); font-size:12px; }
  .wj-detail .wjd-body { color:var(--wiki-text); font-size:0.88rem; line-height:1.7; white-space:pre-wrap; margin-bottom:12px; }
  .wj-detail .wjd-files { color:#7dd3fc; font-size:0.78rem; font-family:var(--font-mono); white-space:pre-wrap;
    background:var(--wiki-surface); border-radius:6px; padding:10px 12px; }
</style>
</head>
<body>

  <!-- 사이드바 -->
  <aside>
    <div class="brand-panel">
      <div class="brand-pulse"></div>
      <div class="brand-title">MoNaVLA v2.5</div>
    </div>
    
    <nav>
      <div class="nav-item active" onclick="switchTab(this, 'drive')">🤖 주행 제어</div>
      <div class="nav-item" onclick="switchTab(this, 'grounding')">🔍 그라운딩 검증</div>
      <div class="nav-item" onclick="switchTab(this, 'latency')">📊 지연 · 드리프트</div>
      <div class="nav-item" onclick="switchTab(this, 'verify')">🧪 경로 검증</div>
      <div class="nav-item" onclick="switchTab(this, 'calib')">🔧 캘리브레이션</div>
      <div class="nav-item" onclick="switchTab(this, 'history')">📚 세션 히스토리</div>
      <div class="nav-item" onclick="switchTab(this, 'overshoot')">🌀 오버슈트 가이드</div>
      <div class="nav-item" onclick="switchTab(this, 'system')">🖥️ 시스템</div>
      <div class="nav-item" onclick="switchTab(this, 'srvcfg')">⚙️ 서버 설정</div>
      <div class="nav-item" onclick="switchTab(this, 'collect')">📷 데이터수집</div>
      <div class="nav-item" onclick="switchTab(this, 'dataset')">🗂 데이터셋 히스토리</div>
      <div class="nav-item" onclick="switchTab(this, 'wikiinfo')">📖 위키</div>
      <div class="nav-item" onclick="switchTab(this, 'wikistatus')">📡 최신현황</div>
    </nav>
    
    <div class="sidebar-footer">
      SODA HOST: 100.85.118.58<br>
      INFER DEV: 8001<br>
      DRIVE API: 7800
    </div>
  </aside>

  <!-- 메인 패널 -->
  <main>
    <header>
      <div style="display:flex;align-items:center;gap:12px;">
        <h2 id="page-title" style="font-size:18px;font-weight:700;">🤖 주행 제어</h2>
      </div>
      <div class="srv-status-group">
        <div class="srv-pill online" id="proc-pill" title="현재 서빙 중인 대시보드 프로세스 PID/가동시간 — 재시작 후 오래된 값이면 좀비 프로세스 의심">
          <span class="indicator"></span> <span id="proc-pill-text">PID —</span>
        </div>
        <div class="srv-pill" id="ros-pill"><span class="indicator"></span> ROS2 Node</div>
        <div class="srv-pill" id="infer-pill"><span class="indicator"></span> Inference Server</div>
        <div class="srv-pill" id="camera-pill"><span class="indicator"></span> Camera Stream</div>
      </div>
    </header>

    <!-- 탭 1: Drive Control -->
    <div id="tab-drive" class="tab-content active">
      <div class="scroll-container">
        <div class="grid-main">
          
          <!-- 왼쪽: 라이브 카메라 스트림 -->
          <div class="card" style="padding:16px;">
            <div class="card-title">📷 실시간 카메라
              <label class="chk-row" style="display:flex;align-items:center;gap:8px;font-size:12px;cursor:pointer;text-transform:none;">
                <input type="checkbox" id="toggle-grid" checked onchange="drawOverlay()" style="accent-color:var(--cyan)"> Grid 표시
              </label>
            </div>
            <div style="display:flex; align-items:center; gap:8px; font-size:11px; margin-bottom:10px; padding:4px 8px; background:#101726; border:1px solid var(--border-glow); border-radius:6px;">
              <span style="color:var(--text-muted);">📹 카메라 프로세스:</span>
              <span id="cam-proc-status-drive" class="cam-proc-status" style="color:var(--cyan); font-family:var(--font-mono); flex:1;">—</span>
              <button class="btn btn-outline" onclick="camProcStart()" style="font-size:10px; padding:2px 8px;">▶ 시작</button>
              <button class="btn btn-outline" onclick="camProcStop()" style="font-size:10px; padding:2px 8px;">■ 정지</button>
              <button class="btn btn-outline" onclick="camProcRefresh()" style="font-size:10px; padding:2px 8px;">↻</button>
            </div>
            <div class="viewport-wrapper" id="drive-viewport">
              <img id="live-stream-img" class="viewport-img" src="/camera/stream" onerror="this.src='https://placehold.co/1280x720/0f1524/94a3b8?text=Camera+Streaming+Offline'">
              <canvas id="live-canvas" class="viewport-canvas" width="640" height="360"></canvas>
              <div class="overlay-info">
                <div class="overlay-badge" id="badge-step-lbl">Step: 0</div>
                <div class="overlay-badge" id="badge-mode-lbl">Mode: ASYNC</div>
                <div class="overlay-badge text-cyan" id="badge-action-lbl">Action: 0.00, 0.00, 0.00</div>
              </div>
            </div>
            
            <div style="margin-top:16px;" class="form-group">
              <label>수행 로그</label>
              <div class="status-console" id="drive-log" style="min-height:70px;">대기 중</div>
            </div>
          </div>
          
          <!-- 오른쪽: 폼 및 조작 -->
          <div class="card" style="display:flex;flex-direction:column;justify-content:space-between;">
            <div>
              <div class="card-title">⚙️ 자율주행 설정</div>
              
              <div class="form-group">
                <label>Instruction</label>
                <input type="text" id="drive-instr" value="gray basket" placeholder="Gray basket을 찾아가세요">
              </div>
              
              <div class="form-group">
                <label>GT Target Object (Optional)</label>
                <input type="text" id="drive-gt" placeholder="예: gray basket">
              </div>
              
              <div class="form-group">
                <label>이동 모드 (Cadence)</label>
                <select id="drive-mode">
                  <option value="ASYNC">ASYNC (연속 비차단)</option>
                  <option value="SYNC" selected>SYNC (안정화 대기 1.92s)</option>
                  <option value="PRE">PRE (격리회전 탐색 루프)</option>
                  <option value="GOALNAV">🎯 GOALNAV (exp73, 포트 8000)</option>
                </select>
              </div>

              <div class="chk-row" style="display:flex;align-items:center;gap:10px;margin:20px 0;">
                <input type="checkbox" id="drive-cc" style="width:18px;height:18px;accent-color:var(--cyan);">
                <label for="drive-cc" style="margin:0;font-size:14px;cursor:pointer;">화이트 밸런스 컬러 보정 적용</label>
              </div>
            </div>
            
            <div>
              <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:12px;">
                <button class="btn btn-cyan" id="drive-start-btn" onclick="startAutopilot()">▶ START DRIVE</button>
                <button class="btn btn-rose" id="drive-stop-btn" onclick="stopAutopilot()" disabled>■ STOP DRIVE</button>
              </div>
              <button class="btn btn-outline" id="drive-return-btn" style="width:100%;" onclick="returnToStart()">🔄 START 위치로 복귀</button>
            </div>

            <!-- 저장된 세션 스택 — STOP(또는 목표 도달로 자동 정지)될 때 갱신되어
                 지금까지 쌓인 세션이 몇 개/어떻게 저장됐는지 바로 보이게 함 -->
            <div id="session-stack-panel" style="margin-top:14px; display:none; background:#101726; border:1px solid var(--border-glow); border-radius:10px; padding:10px 12px;">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                <span style="font-size:11px; color:var(--emerald); font-weight:700;">💾 세션 저장됨 — 최근 스택</span>
                <button class="btn btn-outline" onclick="document.getElementById('session-stack-panel').style.display='none'" style="font-size:10px; padding:2px 8px;">✕</button>
              </div>
              <div id="session-stack-list" style="display:flex; flex-direction:column; gap:4px; font-size:11px; font-family:var(--font-mono);"></div>
            </div>

          </div>

        </div>

        <!-- 하단: 간이 히스토리 -->
        <div class="card" style="margin-top:24px;">
          <div class="card-title">📋 최근 주행 타임라인</div>
          <!-- 수집 세션 요약 (Gradio 수집 모니터 이식) -->
          <div id="collect-summary" style="display:grid; grid-template-columns:1fr 1fr; gap:6px; margin-bottom:10px; font-size:11px;">
            <div style="background:#101726; border:1px solid var(--border-glow); border-radius:6px; padding:6px 10px;">
              <div style="color:var(--cyan); font-weight:700; margin-bottom:3px;">📋 수집 세션</div>
              <div id="cs-session" style="color:var(--text-muted); font-family:var(--font-mono);">—</div>
            </div>
            <div style="background:#101726; border:1px solid var(--border-glow); border-radius:6px; padding:6px 10px;">
              <div style="color:var(--amber); font-weight:700; margin-bottom:3px;">🔍 Grounding</div>
              <div id="cs-grounding" style="color:var(--text-muted); font-family:var(--font-mono);">—</div>
            </div>
          </div>
          <div class="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Step</th>
                  <th>Predicted Label</th>
                  <th>Total Latency</th>
                  <th>Gnd Latency</th>
                  <th>MLP Latency</th>
                  <th>Bbox Area</th>
                  <th>수집 시각</th>
                </tr>
              </thead>
              <tbody id="drive-history-table">
                <tr><td colspan="7" style="text-align:center;color:var(--text-muted);">주행을 시작하면 실시간 분석 데이터가 생성됩니다.</td></tr>
              </tbody>
            </table>
          </div>
        </div>

        <!-- 에피소드 로그 폼은 🧪 경로 검증 탭으로 이관되었습니다. -->
        
      </div>
    </div>

    <!-- 탭 2: Grounding 검증 -->
    <div id="tab-grounding" class="tab-content">
      <div class="scroll-container">
        <div class="grid-main">
          <!-- 왼쪽: 그라운딩 모니터 -->
          <div class="card">
            <div class="card-title">🔍 그라운딩 라이브 모니터</div>
            <div style="display:flex; align-items:center; gap:8px; font-size:11px; margin-bottom:10px; padding:4px 8px; background:#101726; border:1px solid var(--border-glow); border-radius:6px;">
              <span style="color:var(--text-muted);">📹 카메라 프로세스:</span>
              <span id="cam-proc-status-gnd" class="cam-proc-status" style="color:var(--cyan); font-family:var(--font-mono); flex:1;">—</span>
              <button class="btn btn-outline" onclick="camProcStart()" style="font-size:10px; padding:2px 8px;">▶ 시작</button>
              <button class="btn btn-outline" onclick="camProcStop()" style="font-size:10px; padding:2px 8px;">■ 정지</button>
              <button class="btn btn-outline" onclick="camProcRefresh()" style="font-size:10px; padding:2px 8px;">↻</button>
            </div>
            <div class="viewport-wrapper">
              <img id="gnd-stream-img" class="viewport-img" src="/camera/stream">
              <canvas id="gnd-canvas" class="viewport-canvas" width="640" height="360"></canvas>
            </div>
            
            <div class="grid-2" style="margin-top:20px;">
              <div class="form-group">
                <label>BBOX Coordinates</label>
                <input type="text" id="gnd-coords" readonly value="—">
              </div>
              <div class="form-group">
                <label>Cache Hit Status</label>
                <input type="text" id="gnd-cached" readonly value="—">
              </div>
            </div>
          </div>
          
          <!-- 오른쪽: 예측값과 속성 정보 -->
          <div class="card" style="display:flex;flex-direction:column;gap:20px;">
            <div class="card-title">📊 탐지 상세 정보</div>
            
            <div class="kv-grid">
              <div class="form-group">
                <label>Target entity</label>
                <input type="text" id="gnd-entity" readonly value="—">
              </div>
              <div class="form-group">
                <label>Prediction Label</label>
                <input type="text" id="gnd-pred-label" readonly value="—">
              </div>
              <div class="form-group">
                <label>Bbox Area size</label>
                <input type="text" id="gnd-area" readonly value="—">
              </div>
              <div class="form-group">
                <label>Center X (cx)</label>
                <input type="text" id="gnd-cx" readonly value="—">
              </div>
            </div>
            
            <div class="form-group" style="margin-top:auto;">
              <label>임계값 게이지 (STOP 트리거 조건)</label>
              <div class="gauge-container" style="margin-top:8px;">
                <div class="gauge-fill" id="gnd-gauge-fill"></div>
                <div class="gauge-marker" id="gnd-gauge-marker"></div>
              </div>
              <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--text-muted);margin-top:4px;">
                <span>0.0 (Far)</span>
                <span id="gnd-gauge-thr-lbl">임계값: 0.18</span>
                <span>0.4 (Close)</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 탭 3: Latency & Drift -->
    <div id="tab-latency" class="tab-content">
      <div class="scroll-container">
        <div class="grid-main">
          <!-- 왼쪽: 차트 시각화 -->
          <div class="card">
            <div class="card-title">📈 누적 시간 드리프트 시뮬레이터</div>
            <div style="position:relative; height:320px; width:100%">
              <canvas id="drift-chart"></canvas>
            </div>
            
            <div style="display:flex; gap:12px; margin-top:20px;">
              <button class="btn btn-cyan" onclick="runDriftMeasure()">▶ 단발 측정</button>
              <button class="btn btn-outline" id="drift-auto-btn" onclick="toggleDriftAuto()">🔄 자동 (1fps)</button>
              <button class="btn btn-rose" onclick="resetDrift()">🗑 세션 초기화</button>
            </div>
          </div>
          
          <!-- 오른쪽: 상세 수치 -->
          <div class="card" style="display:flex;flex-direction:column;justify-content:space-between;">
            <div>
              <div class="card-title">🩺 드리프트 진단</div>
              
              <div class="form-group">
                <label>가정시간 기준</label>
                <select id="drift-basis-select" onchange="resetDrift()">
                  <option value="1.0s (1fps 운영)" selected>1.0s (1fps 기준 운영)</option>
                  <option value="1.35s (학습 수집 cadence)">1.35s (학습 수집 cadence)</option>
                  <option value="1.92s (SYNC 실측 풀사이클)">1.92s (SYNC 실측 풀사이클)</option>
                </select>
              </div>
              
              <div class="kv-grid" style="margin-top:20px; display:grid; grid-template-columns:1fr 1fr; gap:12px;">
                <div class="srv-pill"><span class="indicator"></span> 프레임 수: <strong id="drift-cnt">0</strong></div>
                <div class="srv-pill"><span class="indicator"></span> 최근 지연: <strong id="drift-lat">0ms</strong></div>
                <div class="srv-pill"><span class="indicator"></span> 누적 실제: <strong id="drift-real">0.00s</strong></div>
                <div class="srv-pill"><span class="indicator"></span> 누적 nominal: <strong id="drift-nom">0.00s</strong></div>
              </div>
              
              <div style="margin-top:24px; text-align:center;">
                <div style="font-size:24px; font-weight:700;" id="drift-val-display" class="text-emerald">drift: 0.00s</div>
                <div style="font-size:12px; color:var(--text-muted); margin-top:4px;">기준 시간 대비 누적 드리프트 격차</div>
              </div>
            </div>
            
            <div class="form-group" style="margin-top:20px;">
              <label>진단 콘솔</label>
              <div class="status-console" id="drift-log-console" style="min-height:90px;">측정을 수행하면 로그 분석 및 파일 저장이 시작됩니다.</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 탭 4: 🧪 경로 검증 (Path Test) -->
    <div id="tab-verify" class="tab-content">
      <div class="scroll-container" style="padding: 20px;">
        <div class="grid-3-verify" style="display: grid; grid-template-columns: 2fr 1.6fr 1.6fr; gap: 20px; align-items: start;">
          
          <!-- Column 1: Live Camera & Telemetry -->
          <div class="card" style="padding:16px; display:flex; flex-direction:column; gap:16px;">
            <div class="card-title">📷 실시간 검증 화면
              <label class="chk-row" style="display:flex;align-items:center;gap:8px;font-size:12px;cursor:pointer;text-transform:none;">
                <input type="checkbox" id="toggle-grid-vfy" checked onchange="drawOverlay()" style="accent-color:var(--cyan)"> Grid 표시
              </label>
              <label class="chk-row" style="display:flex;align-items:center;gap:8px;font-size:12px;cursor:pointer;text-transform:none;">
                <input type="checkbox" id="toggle-cxguide-vfy" onchange="drawOverlay()" style="accent-color:var(--amber)"> 배치가이드 표시
              </label>
              <button class="btn btn-outline joystick-mode-btn" onclick="toggleJoystickMode()" style="font-size:11px; padding:4px 10px; margin-left:auto;" title="🧪 검증: D-pad◀▶=위치 · L1=추론시작 R1=정지 X=성공/A=실패(라벨) L2=세션저장 R2=복귀 · Y=모드전환">🕹️ 조이스틱: 📷 수집</button>
            </div>
            <div style="display:flex; align-items:center; gap:8px; font-size:11px; padding:4px 8px; background:#101726; border:1px solid var(--border-glow); border-radius:6px;">
              <span style="color:var(--text-muted);">📹 카메라 프로세스:</span>
              <span id="cam-proc-status-vfy" class="cam-proc-status" style="color:var(--cyan); font-family:var(--font-mono); flex:1;">—</span>
              <button class="btn btn-outline" onclick="camProcStart()" style="font-size:10px; padding:2px 8px;">▶ 시작</button>
              <button class="btn btn-outline" onclick="camProcStop()" style="font-size:10px; padding:2px 8px;">■ 정지</button>
              <button class="btn btn-outline" onclick="camProcRefresh()" style="font-size:10px; padding:2px 8px;">↻</button>
            </div>
            <div class="viewport-wrapper" style="position:relative; border-radius:12px; overflow:hidden; background-color:#000; aspect-ratio:16/9; border:1px solid var(--border-glow);">
              <img id="verify-stream-img" class="viewport-img" src="/camera/stream" style="width:100%; height:100%; object-fit:contain;">
              <canvas id="verify-canvas" class="viewport-canvas" width="640" height="360" style="position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none; z-index:5;"></canvas>
            </div>
            
            <div class="grid-2" style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
              <div class="form-group" style="margin-bottom:0;">
                <label>Status</label>
                <div id="vfy-status-log" class="status-console" style="padding:8px 12px; min-height:40px; font-size:13px; font-family:var(--font-mono); color:var(--cyan); background:rgba(6,182,212,0.05); border:1px solid rgba(6,182,212,0.15); border-radius:8px;">Ready</div>
              </div>
              <div class="form-group" style="margin-bottom:0;">
                <label>Latency</label>
                <div id="vfy-latency-val" style="padding:8px 12px; font-size:13px; font-family:var(--font-mono); color:var(--amber); background:rgba(245,158,11,0.05); border:1px solid rgba(245,158,11,0.15); border-radius:8px;">0 ms</div>
              </div>
            </div>
            <div class="grid-2" style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
              <div class="form-group" style="margin-bottom:0;">
                <label>Action (lx, ly, az)</label>
                <div id="vfy-action-val" style="padding:8px 12px; font-size:13px; font-family:var(--font-mono); color:var(--emerald); background:rgba(16,185,129,0.05); border:1px solid rgba(16,185,129,0.15); border-radius:8px;">0.0, 0.0, 0.0</div>
              </div>
              <div class="form-group" style="margin-bottom:0;">
                <label>Bbox Size / cx</label>
                <div id="vfy-bbox-val" style="padding:8px 12px; font-size:13px; font-family:var(--font-mono); color:var(--violet); background:rgba(139,92,246,0.05); border:1px solid rgba(139,92,246,0.15); border-radius:8px;">—</div>
              </div>
            </div>

            <!-- 📋 실시간 수집 세션 정보 카드 -->
            <div style="background:#151f32; border:1px solid var(--border-glow); border-radius:10px; padding:12px; display:flex; flex-direction:column; gap:8px;">
              <div style="font-size:11px; color:var(--text-muted); font-weight:600; text-transform:uppercase; display:flex; justify-content:space-between; align-items:center;">
                <span>📋 수집 세션 모니터링</span>
                <span id="vfy-update-time" style="font-size:9px; color:#58a6ff;">갱신 —</span>
              </div>
              <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; font-size:12px; line-height:1.6;">
                <div>상태: <span id="vfy-sess-run-status">—</span></div>
                <div>모드: <span id="vfy-sess-mode" class="text-cyan font-mono">—</span></div>
                <div>스텝: <span id="vfy-sess-step" class="text-emerald font-mono">—</span></div>
                <div>H5 프레임: <span id="vfy-sess-frames" class="text-violet font-mono">—</span></div>
              </div>
              <div style="font-size:12px; border-top:1px solid rgba(255,255,255,0.05); padding-top:6px; display:flex; flex-direction:column; gap:2px;">
                <div>기록: <span id="vfy-sess-record-lbl">—</span></div>
                <div>세션 ID: <span id="vfy-sess-id-lbl" class="font-mono text-cyan" style="font-size:10px;">—</span></div>
              </div>
            </div>

            <!-- 🔍 Grounding & 🔢 최근 스텝 카드 -->
            <div style="background:#151f32; border:1px solid var(--border-glow); border-radius:10px; padding:12px; display:flex; flex-direction:column; gap:8px;">
              <div style="font-size:11px; color:var(--text-muted); font-weight:600; text-transform:uppercase; display:flex; justify-content:space-between; align-items:center;">
                <span>🔍 Grounding 통계</span>
                <span id="vfy-gnd-cached" class="font-mono" style="font-size:10px;">—</span>
              </div>
              <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; font-size:12px; line-height:1.6;">
                <div>live PG2: <span id="vfy-gnd-live-cnt" class="text-amber font-mono">—</span></div>
                <div>캐시 재사용: <span id="vfy-gnd-cache-cnt" class="text-emerald font-mono">—</span></div>
                <div style="grid-column: span 2;">평균 latency: <span id="vfy-gnd-avg-lat" class="text-cyan font-mono">—</span></div>
                <div>대상: <span id="vfy-gnd-entity" class="text-cyan font-mono" style="font-weight:600;">—</span></div>
                <div>레이블: <span id="vfy-pred-label" class="text-emerald font-mono" style="font-weight:600;">—</span></div>
              </div>
              
              <div style="font-size:11px; color:var(--text-muted); font-weight:600; text-transform:uppercase; border-top:1px solid rgba(255,255,255,0.05); padding-top:6px; margin-top:2px;">🔢 최근 스텝</div>
              <div id="vfy-last-steps-container" style="font-size:11px; line-height:1.5; font-family:var(--font-mono); background:#0c1322; border-radius:6px; padding:6px; border:1px solid rgba(255,255,255,0.05); max-height:140px; overflow-y:auto; display:flex; flex-direction:column; gap:2px;">
                <span style="color:var(--text-muted);">스텝 이력 없음</span>
              </div>
            </div>

            <div class="form-group" style="margin-bottom:0;">
              <label>🖥️ Inference Server Status</label>
              <div id="vfy-srv-status" style="padding:8px 12px; font-size:12px; font-family:var(--font-mono); color:var(--text-muted); background:#101726; border:1px solid var(--border-glow); border-radius:8px; min-height:42px;">—</div>
            </div>
          </div>
          
          <!-- Column 2: Progress & Summary Table -->
          <div class="card" style="padding:16px; display:flex; flex-direction:column; gap:16px; overflow-y:auto; max-height:calc(100vh - 120px);">
            <div class="card-title">📊 경로 다이어그램 및 집계</div>
            
            <!-- Progress Bar Placeholder -->
            <div id="vfy-progress-wrapper" style="min-height:92px; background:rgba(255,255,255,0.01); border-radius:8px; padding:4px 0;">
              로딩 중...
            </div>
            
            <div id="vfy-progress-txt" style="font-size:12px; color:var(--text-muted); line-height:1.4; margin-top:-8px;">
              경로검증 계산 중...
            </div>

            <!-- 🔀 모델 전환 — /model/load 핫스왑, go.sh 재시작(95s) 불필요 (2026-07-23) -->
            <details class="card" style="padding:10px; background:#101726; border:1px solid var(--cyan); border-radius:8px; flex-shrink:0;" open>
              <summary style="font-size:12px; font-weight:700; color:var(--cyan); outline:none; cursor:pointer;">🔀 모델 전환
                <span id="vfy-model-current" style="font-size:9px; padding:1px 6px; border-radius:10px; background:rgba(6,182,212,0.15); color:var(--cyan); margin-left:6px;">로딩중...</span>
                <button class="btn btn-outline" onclick="event.preventDefault(); refreshModelList();" style="font-size:9px; padding:2px 6px; float:right;">🔄</button>
              </summary>
              <div style="font-size:9px; color:var(--text-muted); margin:6px 0;">체크포인트 클릭 = 즉시 핫스왑(재시작 없음, Kosmos-2 vision encoder는 유지). 전환 직후 첫 추론에 약간 지연 있을 수 있음.</div>
              <div style="display:flex; gap:6px; align-items:center; margin-bottom:8px; padding:6px; background:#090d16; border-radius:6px; border:1px solid var(--border-glow);">
                <span style="font-size:9px; color:var(--text-muted); white-space:nowrap;">🔭 그라운더:</span>
                <span id="vfy-grounder-current" style="font-size:10px; color:#fff; font-weight:600; flex:1;">—</span>
                <button class="btn btn-outline" onclick="switchGrounder('pg2')" style="font-size:9px; padding:3px 8px;">PG2</button>
                <button class="btn btn-outline" onclick="switchGrounder('owlv2')" style="font-size:9px; padding:3px 8px;">OWLv2</button>
              </div>
              <div id="vfy-model-list" style="display:flex; flex-direction:column; gap:4px;">불러오는 중...</div>
              <div id="vfy-model-status" style="font-size:10px; color:var(--text-muted); text-align:center; margin-top:6px;"></div>
            </details>

            <!-- 🎯 exp73 추론 검증 스크리닝 (데이터셋 목표와 별개) -->
            <div style="background:#101726; border:1px solid var(--amber); border-radius:8px; padding:10px; flex-shrink:0;">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <span style="font-size:12px; font-weight:700; color:var(--amber);">🎯 추론 검증 스크리닝</span>
                <button id="vfy-screen-toggle" class="btn btn-outline" onclick="toggleScreenTarget()" style="font-size:10px; padding:3px 8px;">1차(빠른확인)</button>
              </div>
              <div style="font-size:9px; color:var(--text-muted); margin-bottom:6px;">바구니 위치별 목표 — 데이터셋 수집 목표(트랙 15개)와 별개.<br>🕹️ D-pad◀▶=위치선택 · L1추론시작 · R1정지 · X성공/A실패(라벨) · <b>L2=세션저장</b>(💾버튼과 동일, 여기서만 기록) · R2복귀 · <b>SEL=🧪실험용 토글</b>(그라운더 A/B 등 정식 집계 제외).</div>
              <div id="vfy-screen-current" style="font-size:11px; font-weight:700; text-align:center; padding:6px; margin-bottom:6px; border-radius:6px; background:#090d16; border:1px solid var(--border-glow); color:var(--text-muted);">현재 위치: — · 대기 라벨: —</div>
              <div id="vfy-experimental-badge" style="display:none; font-size:10px; font-weight:700; text-align:center; padding:4px; margin-bottom:6px; border-radius:6px; background:#3a1a1a; border:1px solid #d9534f; color:#ff8080;">🧪 실험용 기록 모드 ON — episode_log_experimental.csv로 저장 (정식 집계 제외)</div>

              <!-- A/B 필터 — 어느 체크포인트/언제부터 기록을 셀지 (2026-07-23) -->
              <div style="display:flex; gap:6px; margin-bottom:6px;">
                <select id="vfy-screen-ckpt" onchange="renderScreenPanel(window._lastVfyRows)" style="flex:1; padding:4px 6px; background:#090d16; border:1px solid var(--border-glow); border-radius:6px; color:#fff; font-size:10px;">
                  <option value="">전체(누적)</option>
                </select>
                <button class="btn btn-outline" onclick="refreshCheckpointOptions()" style="font-size:10px; padding:4px 8px;" title="체크포인트 목록 새로고침(새 세션 반영)">🔄</button>
              </div>
              <div style="display:flex; gap:6px; align-items:center; margin-bottom:6px;">
                <span style="font-size:9px; color:var(--text-muted); white-space:nowrap;">검증 시작:</span>
                <input type="datetime-local" id="vfy-screen-since" onchange="renderScreenPanel(window._lastVfyRows)" style="flex:1; padding:3px 4px; background:#090d16; border:1px solid var(--border-glow); border-radius:6px; color:#fff; font-size:10px;">
                <button class="btn btn-outline" onclick="setScreenSinceNow()" style="font-size:10px; padding:3px 8px; white-space:nowrap;">🔖 지금부터</button>
                <button class="btn btn-outline" onclick="clearScreenSince()" style="font-size:10px; padding:3px 8px;">✕</button>
              </div>

              <div id="vfy-screen-body">—</div>
            </div>

            <div class="table-wrapper" style="min-height:380px; max-height:380px; flex-shrink:0; overflow-y:auto; border:1px solid var(--border-glow); border-radius:8px;">
              <table style="width:100%; border-collapse:collapse; font-size:12px;">
                <thead>
                  <tr style="background:#151f32; border-bottom:1px solid var(--border-glow); text-align:left;">
                    <th style="padding:8px 6px; width:28%; color:var(--cyan); font-weight:700;">경로 / 테스트</th>
                    <th style="padding:8px 4px; width:38px;">목표</th>
                    <th style="padding:8px 4px; width:38px;">완료</th>
                    <th style="padding:8px 4px; width:38px; color:var(--rose);">실패</th>
                    <th style="padding:8px 4px; width:38px; color:var(--emerald);">성공</th>
                    <th style="padding:8px 4px; width:48px; color:var(--cyan);">성공률</th>
                  </tr>
                </thead>
                <tbody id="vfy-summary-table-body">
                  <tr><td colspan="6" style="text-align:center; padding:16px; color:var(--text-muted);">에피소드 데이터를 집계하는 중...</td></tr>
                </tbody>
              </table>
            </div>

            <!-- 경로 다이어그램 Accordion -->
            <details class="card" style="padding:12px; background:#101726; border:1px solid var(--border-glow); border-radius:8px; cursor:pointer; margin-top:8px;">
              <summary style="font-size:12px; font-weight:600; color:var(--cyan); outline:none;">📋 경로 다이어그램 보기</summary>
              <div style="margin-top:10px; cursor:default;">
                <pre style="font-family:var(--font-mono); font-size:11px; line-height:1.4; color:var(--text-muted); background:#090d16; padding:8px; border-radius:6px; border:1px solid var(--border-glow);">
 ▦      ▦      ▦
╱│╲    ╱│╲    ╱│╲
L S R  C S L  R S L
 🤖L    🤖C    🤖R
                </pre>
                <div style="font-size:11px; color:var(--text-muted); margin-top:6px; line-height:1.4;">
                  <strong>L</strong>=left, <strong>C</strong>=center, <strong>R</strong>=right 시작위치<br>
                  방향: L(좌) / S(직) / R(우) · ★=right_left 우선
                </div>
              </div>
            </details>

            <!-- 🕹️ 조이스틱 가이드 (검증모드 기준, 컴팩트) — 데이터수집 패널과 독립 -->
            <details class="card" style="padding:10px; background:#101726; border:1px solid var(--amber); border-radius:8px; margin-top:4px;">
              <summary style="font-size:12px; font-weight:700; color:var(--amber); outline:none;">🕹️ 조이스틱 가이드 (검증모드)
                <span id="vfy-js-badge" style="font-size:9px; padding:1px 6px; border-radius:10px; background:rgba(100,116,139,0.2); color:var(--text-muted); margin-left:6px;">🔌 —</span>
              </summary>
              <div style="margin-top:8px; cursor:default;">
                <div id="vfy-js-btn-lights" style="display:flex; flex-wrap:wrap; gap:6px; margin-bottom:8px;"></div>
                <div id="vfy-js-caption" style="font-size:15px; font-weight:700; font-family:var(--font-mono); text-align:center; margin-bottom:10px; padding:10px; border-radius:8px; background:#090d16; border:1px solid var(--border-glow); color:var(--text-muted); transition:background 0.15s, border-color 0.15s, color 0.15s;">대기 중 — 버튼/D-pad를 누르면 여기 크게 표시</div>
                <table style="width:100%; border-collapse:collapse; font-size:10px; line-height:1.5;">
                  <tr><td style="padding:1px 4px;"><b>D-pad◀▶</b></td><td colspan="3" style="color:var(--cyan);">📍 검증 위치 순환(강좌↔약좌↔중앙↔약우↔강우)</td></tr>
                  <tr><td style="padding:1px 4px;"><b>L1</b>(4)</td><td style="color:#3fb950;">▶ 추론 시작</td>
                      <td style="padding:1px 4px;"><b>R1</b>(5)</td><td>⏹ 추론 정지</td></tr>
                  <tr><td style="padding:1px 4px;"><b>X</b>(2)</td><td style="color:#3fb950;">✅ 성공 라벨</td>
                      <td style="padding:1px 4px;"><b>A</b>(0)</td><td style="color:#f43f5e;">❌ 실패 라벨</td></tr>
                  <tr><td style="padding:1px 4px; color:var(--amber);"><b>L2</b></td><td style="color:var(--amber);">💾 기록 저장</td>
                      <td style="padding:1px 4px; color:var(--amber);"><b>R2</b></td><td style="color:var(--amber);">↩ 복귀</td></tr>
                  <tr><td style="padding:1px 4px;"><b>START</b>(7)</td><td>⚙ SYNC↔ASYNC</td>
                      <td style="padding:1px 4px; color:var(--amber);"><b>Y</b>(3)</td><td style="color:var(--amber);">🔁 모드 전환</td></tr>
                </table>
                <div style="font-size:9px; color:var(--text-muted); margin-top:6px; line-height:1.5; border-top:1px solid var(--border-glow); padding-top:6px;">
                  순서: <b>D-pad로 위치 선택 → L1 시작 → (주행) → R1 정지 → X/A로 성공·실패 라벨 → L2로 💾 기록 저장</b><br>
                  X/A는 <b>라벨만</b> 바꿈(즉시 기록 X) · L2 눌러야 1건 저장 → 💾 기록 저장 버튼과 100% 동일 기록, 중복 방지<br>
                  ⚠️ 별도 비상정지 없음 — <b>R1</b>이 robust_stop. <b>Y</b>=모드전환.
                </div>
              </div>
            </details>
          </div>

          <!-- Column 3: Control & Episode Editor -->
          <div class="card" style="padding:16px; display:flex; flex-direction:column; gap:16px; overflow-y:auto; max-height:calc(100vh - 120px);">
            <div class="card-title">📝 주행 제어 및 기록</div>
            
            <!-- 주행 제어 버튼 연동 -->
            <div style="background:#151f32; border:1px solid var(--border-glow); border-radius:10px; padding:12px; display:flex; flex-direction:column; gap:8px;">
              <div style="font-size:11px; color:var(--text-muted); font-weight:600; text-transform:uppercase;">🎮 Quick Autopilot</div>
              <div style="display:grid; grid-template-columns:1.2fr 1fr 1fr; gap:8px;">
                <button class="btn btn-cyan" id="quick-start-btn" onclick="startAutopilot()" style="padding:8px 0; font-size:12px; font-weight:bold;">▶️ START</button>
                <button class="btn btn-rose" id="quick-stop-btn" onclick="stopAutopilot()" style="padding:8px 0; font-size:12px; font-weight:bold;" disabled>⏹️ STOP</button>
                <button class="btn btn-outline" onclick="returnToStart()" style="padding:8px 0; font-size:12px; font-weight:bold;">🔄 복귀</button>
              </div>
            </div>

            <!-- 런타임 설정 + Autopilot Configuration 통합 패널 (같은 카드, 2열) -->
            <div style="background:#151f32; border:1px solid var(--border-glow); border-radius:10px; padding:12px; display:grid; grid-template-columns:1fr 1fr; gap:14px;">
              <!-- 좌: 런타임 모드 토글 -->
              <div style="display:flex; flex-direction:column; gap:8px;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                  <span style="font-size:11px; color:var(--text-muted); font-weight:600; text-transform:uppercase;">⚙️ 런타임 설정</span>
                  <button class="btn btn-outline" onclick="syncVerifyRuntimeParams()" style="font-size:10px; padding:2px 8px;">🔄</button>
                </div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px;">
                  <button id="vfy-rt-preview" class="btn btn-outline" onclick="toggleVerifyRuntime('preview')" style="font-size:11px; padding:6px; line-height:1.2; text-align:center;">🟢 프리뷰<br><span style="font-size:9px;color:var(--text-muted)">격리회전</span></button>
                  <button id="vfy-rt-hint" class="btn btn-outline" onclick="toggleVerifyRuntime('hint')" style="font-size:11px; padding:6px; line-height:1.2; text-align:center;">⚫ hint_cx<br><span style="font-size:9px;color:var(--text-muted)">방향회전</span></button>
                </div>
                <div style="display:grid; grid-template-columns:1fr 1.2fr; gap:6px;">
                  <button id="vfy-rt-skip" class="btn btn-outline" onclick="toggleVerifyRuntime('skip')" style="font-size:11px; padding:6px; line-height:1.2; text-align:center;">📦 skip 3<br><span style="font-size:9px;color:var(--text-muted)">캐시 사용</span></button>
                  <button id="vfy-rt-jump" class="btn btn-outline" onclick="toggleVerifyRuntime('jump')" style="font-size:11px; padding:6px; line-height:1.2; text-align:center;">⚫ P2필터<br><span style="font-size:9px;color:var(--text-muted)">오탐제거</span></button>
                </div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px;">
                  <button id="vfy-rt-thr" class="btn btn-outline" onclick="toggleVerifyRuntime('thr')" disabled style="font-size:11px; padding:6px; line-height:1.2; text-align:center; opacity:0.5;">🎚 민감도<br><span style="font-size:9px;color:var(--text-muted)">(P2 꺼짐)</span></button>
                  <button id="vfy-rt-multi" class="btn btn-outline" onclick="toggleVerifyRuntime('multi')" style="font-size:11px; padding:6px; line-height:1.2; text-align:center;">🟢 멀티프롬프트<br><span style="font-size:9px;color:var(--text-muted)">fallback</span></button>
                </div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:6px;">
                  <button id="vfy-rt-stopmode" class="btn btn-outline" onclick="toggleVerifyRuntime('stopmode')" style="font-size:11px; padding:6px; line-height:1.2; text-align:center;">🛑 proximity<br><span style="font-size:9px;color:var(--text-muted)">STOP 모드</span></button>
                  <button id="vfy-rt-stopguard" class="btn btn-outline" onclick="toggleVerifyRuntime('stopguard')" style="font-size:11px; padding:6px; line-height:1.2; text-align:center;">🛡 가드 3프레임<br><span style="font-size:9px;color:var(--text-muted)">learned 콜드스타트</span></button>
                </div>
                <div id="vfy-rt-status" style="font-size:10px; color:var(--cyan); text-align:center; font-family:var(--font-mono); margin-top:2px;">—</div>
                <div id="vfy-owl-row" style="display:none; align-items:center; gap:6px; margin-top:2px;">
                  <span style="font-size:9px; color:var(--text-muted); white-space:nowrap;">🔭 OWL 보정계수</span>
                  <input type="number" id="vfy-owl-area-scale" min="0.5" max="10" step="0.1" value="3.0" style="width:52px; padding:3px 4px; background:#090d16; border:1px solid var(--border-glow); border-radius:4px; color:#fff; font-size:11px;">
                  <button class="btn btn-outline" onclick="applyVerifyOwlAreaScale()" style="font-size:10px; padding:3px 8px; flex:1;">적용</button>
                </div>
              </div>

              <!-- 우: Autopilot Configuration (START/STOP은 위 Quick Autopilot 버튼 사용) -->
              <div style="display:flex; flex-direction:column; gap:8px;">
                <span style="font-size:11px; color:var(--text-muted); font-weight:600; text-transform:uppercase;">🎯 Config</span>
                <div class="form-group" style="margin-bottom:0;">
                  <label style="font-size:11px;">Instruction</label>
                  <input type="text" id="drive-instr-vfy" value="gray basket" placeholder="Gray basket을 찾아가세요" style="width:100%; padding:6px 8px; background:#090d16; border:1px solid var(--border-glow); border-radius:6px; color:#fff; font-size:12px;">
                </div>
                <div class="form-group" style="margin-bottom:0;">
                  <label style="font-size:11px;">GT Target Object (Optional)</label>
                  <input type="text" id="drive-gt-vfy" placeholder="예: gray basket" style="width:100%; padding:6px 8px; background:#090d16; border:1px solid var(--border-glow); border-radius:6px; color:#fff; font-size:12px;">
                </div>
                <div class="form-group" style="margin-bottom:0;">
                  <label style="font-size:11px;">이동 모드 (Cadence)</label>
                  <select id="drive-mode-vfy" style="width:100%; padding:6px 8px; background:#090d16; border:1px solid var(--border-glow); border-radius:6px; color:#fff; font-size:12px;">
                    <option value="ASYNC">ASYNC (연속 비차단)</option>
                    <option value="SYNC" selected>SYNC (안정화 대기 1.92s)</option>
                    <option value="PRE">PRE (격리회전 탐색 루프)</option>
                    <option value="GOALNAV">🎯 GOALNAV (exp73, 포트 8000)</option>
                  </select>
                </div>
                <div class="chk-row" style="display:flex;align-items:center;gap:8px;">
                  <input type="checkbox" id="drive-cc-vfy" style="width:15px;height:15px;accent-color:var(--cyan);">
                  <label for="drive-cc-vfy" style="margin:0;font-size:11px;cursor:pointer;">화이트 밸런스 컬러 보정 적용</label>
                </div>
              </div>
            </div>

            <!-- 에피소드 입력 폼 -->
            <div style="background:#151f32; border:1px solid var(--border-glow); border-radius:10px; padding:12px; display:flex; flex-direction:column; gap:10px;">
              <div class="form-group" style="margin-bottom:0;">
                <label>경로 구분 (path_type)</label>
                <select id="ep-path-type" onchange="syncVerifyPathType(this.value); drawOverlay();" style="width:100%; padding:8px; background:#090d16; border:1px solid var(--border-glow); border-radius:6px; color:#fff; font-size:13px;">
                  <option value="right_right">R→R (right_right)</option>
                  <option value="right_left">R→L★ (right_left)</option>
                  <option value="right_straight">R→S (right_straight)</option>
                  <option value="center_straight">C→S (center_straight)</option>
                  <option value="center_left">C→L (center_left)</option>
                  <option value="center_right">C→R (center_right)</option>
                  <option value="left_straight">L→S (left_straight)</option>
                  <option value="left_left">L→L (left_left)</option>
                  <option value="left_right">L→R (left_right)</option>
                  <option value="obj_left">위치:좌 (obj_left)</option>
                  <option value="obj_center">위치:중 (obj_center)</option>
                  <option value="obj_right">위치:우 (obj_right)</option>
                  <option value="dist_10cm">거리:10cm (dist_10cm)</option>
                  <option value="dist_20cm">거리:20cm (dist_20cm)</option>
                  <option value="dist_30cm">거리:30cm (dist_30cm)</option>
                  <optgroup label="🎯 V6 극단/중앙 배치 (위치 기준)">
                    <option value="trackA_strong_left">V6 강극단좌 (strong_left)</option>
                    <option value="trackA_weak_left">V6 약극단좌 (weak_left)</option>
                    <option value="trackF_center" selected>V6 중앙 (center)</option>
                    <option value="trackA_weak_right">V6 약극단우 (weak_right)</option>
                    <option value="trackA_strong_right">V6 강극단우 (strong_right)</option>
                  </optgroup>
                </select>
              </div>

              <div class="form-group" style="margin-bottom:0;">
                <label>주행 결과</label>
                <input type="hidden" id="ep-success" value="성공">
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px;">
                  <button type="button" id="ep-result-succ" class="btn btn-cyan" style="font-weight:bold;" onclick="setEpResult('성공')">✅ 성공 (Success)</button>
                  <button type="button" id="ep-result-fail" class="btn btn-outline" style="font-weight:bold;" onclick="setEpResult('실패')">❌ 실패 (Failure)</button>
                </div>
              </div>

              <div class="form-group" style="margin-bottom:0;">
                <label id="ep-fpe-lbl">False Positive Error (FPE): 0.00</label>
                <input type="range" id="ep-fpe" min="0.0" max="0.5" step="0.01" value="0.0" style="width:100%; accent-color:var(--cyan);" oninput="document.getElementById('ep-fpe-lbl').textContent = 'False Positive Error (FPE): ' + this.value">
                <div style="display:flex; flex-wrap:wrap; gap:4px; margin-top:6px;">
                  <button class="btn btn-outline" onclick="setFpeValue(0.0)" style="font-size:10px; padding:2px 4px;">0.0</button>
                  <button class="btn btn-outline" onclick="setFpeValue(0.01)" style="font-size:10px; padding:2px 4px;">0.01</button>
                  <button class="btn btn-outline" onclick="setFpeValue(0.02)" style="font-size:10px; padding:2px 4px;">0.02</button>
                  <button class="btn btn-outline" onclick="setFpeValue(0.03)" style="font-size:10px; padding:2px 4px;">0.03</button>
                  <button class="btn btn-outline" onclick="setFpeValue(0.05)" style="font-size:10px; padding:2px 4px;">0.05</button>
                  <button class="btn btn-outline" onclick="setFpeValue(0.08)" style="font-size:10px; padding:2px 4px;">0.08</button>
                  <button class="btn btn-outline" onclick="setFpeValue(0.1)" style="font-size:10px; padding:2px 4px;">0.1</button>
                  <button class="btn btn-outline" onclick="setFpeValue(0.15)" style="font-size:10px; padding:2px 4px;">0.15</button>
                  <button class="btn btn-outline" onclick="setFpeValue(0.2)" style="font-size:10px; padding:2px 4px;">0.2</button>
                  <button class="btn btn-outline" onclick="setFpeValue(0.3)" style="font-size:10px; padding:2px 4px;">0.3</button>
                  <button class="btn btn-outline" onclick="setFpeValue(0.5)" style="font-size:10px; padding:2px 4px;">0.5</button>
                </div>
              </div>

              <div class="form-group" style="margin-bottom:0;">
                <label>비고 / 메모 (note)</label>
                <input type="text" id="ep-note" placeholder="메모를 입력하세요" style="width:100%; padding:8px; background:#090d16; border:1px solid var(--border-glow); border-radius:6px; color:#fff; font-size:13px;">
              </div>

              <button class="btn btn-cyan" style="width:100%; font-weight:bold; margin-top:4px;" onclick="commitEpisode()">💾 기록 저장 (Log Episode)</button>
              <button class="btn btn-outline text-rose" style="width:100%; font-size:11px;" onclick="undoEpisode()">↩ 마지막 삭제</button>
            </div>

            <!-- 빠른 경로선택 버튼들 -->
            <div style="display:flex; flex-direction:column; gap:6px;">
              <div style="font-size:11px; color:var(--text-muted); font-weight:600; text-transform:uppercase;">🎯 빠른 레이블 선택</div>
              <div style="display:flex; flex-direction:column; gap:6px; background:#101726; padding:8px; border-radius:8px; border:1px solid var(--border-glow);">
                <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:4px;">
                  <button class="btn btn-outline" onclick="selectPathType('obj_left')" style="font-size:10px; padding:4px 0;">obj_left</button>
                  <button class="btn btn-outline" onclick="selectPathType('obj_center')" style="font-size:10px; padding:4px 0;">obj_center</button>
                  <button class="btn btn-outline" onclick="selectPathType('obj_right')" style="font-size:10px; padding:4px 0;">obj_right</button>
                </div>
                <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:4px;">
                  <button class="btn btn-outline" onclick="selectPathType('left_left')" style="font-size:10px; padding:4px 0;">left_left</button>
                  <button class="btn btn-outline" onclick="selectPathType('left_straight')" style="font-size:10px; padding:4px 0;">left_straight</button>
                  <button class="btn btn-outline" onclick="selectPathType('left_right')" style="font-size:10px; padding:4px 0;">left_right</button>
                </div>
                <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:4px;">
                  <button class="btn btn-outline" onclick="selectPathType('center_left')" style="font-size:10px; padding:4px 0;">center_left</button>
                  <button class="btn btn-outline" onclick="selectPathType('center_straight')" style="font-size:10px; padding:4px 0;">center_straight</button>
                  <button class="btn btn-outline" onclick="selectPathType('center_right')" style="font-size:10px; padding:4px 0;">center_right</button>
                </div>
                <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:4px;">
                  <button class="btn btn-outline" onclick="selectPathType('right_left')" style="font-size:10px; padding:4px 0;">right_left ★</button>
                  <button class="btn btn-outline" onclick="selectPathType('right_straight')" style="font-size:10px; padding:4px 0;">right_straight</button>
                  <button class="btn btn-outline" onclick="selectPathType('right_right')" style="font-size:10px; padding:4px 0;">right_right</button>
                </div>
                <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:4px;">
                  <button class="btn btn-outline" onclick="selectPathType('dist_10cm')" style="font-size:10px; padding:4px 0;">dist_10cm</button>
                  <button class="btn btn-outline" onclick="selectPathType('dist_20cm')" style="font-size:10px; padding:4px 0;">dist_20cm</button>
                  <button class="btn btn-outline" onclick="selectPathType('dist_30cm')" style="font-size:10px; padding:4px 0;">dist_30cm</button>
                </div>
              </div>

              <!-- 트랙A 극단배치(V6) — 기존 15종과 별개 섹션. docs/DATASET_V6_STATUS.md 명명 규정 -->
              <div style="font-size:11px; color:var(--amber); font-weight:600; text-transform:uppercase; margin-top:6px;">🎯 트랙A 극단배치 (V6)</div>
              <div id="verify-tracka-grid" style="display:flex; flex-direction:column; gap:6px; background:#101726; padding:8px; border-radius:8px; border:1px solid var(--border-glow);"></div>
            </div>

            <!-- 에피소드 로그 테이블 -->
            <div style="display:flex; flex-direction:column; gap:8px;">
              <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:11px; color:var(--text-muted); font-weight:600; text-transform:uppercase;">📋 에피소드 누적 기록</span>
                <select id="vfy-filter" onchange="loadEpisodeHistory()" style="background:#151f32; border:1px solid var(--border-glow); border-radius:4px; color:#fff; font-size:10px; padding:2px 4px;">
                  <option value="all">전체</option>
                  <option value="success">성공만</option>
                  <option value="fail">🚨 실패만</option>
                </select>
              </div>
              <div class="table-wrapper" style="max-height:520px; overflow-y:auto; overflow-x:auto; border:1px solid var(--border-glow); border-radius:8px;">
                <table style="width:100%; min-width:820px; border-collapse:collapse; font-size:11px;">
                  <thead style="background:#151f32; border-bottom:1px solid var(--border-glow); text-align:left; position:sticky; top:0;">
                    <tr>
                      <th style="padding:6px 8px;">#</th>
                      <th style="padding:6px 8px;">경로</th>
                      <th style="padding:6px 8px;">결과</th>
                      <th style="padding:6px 8px;">steps</th>
                      <th style="padding:6px 8px;">lat</th>
                      <th style="padding:6px 8px;">top액션</th>
                      <th style="padding:6px 8px;">gnd%</th>
                      <th style="padding:6px 8px;">area</th>
                      <th style="padding:6px 8px;">cx</th>
                      <th style="padding:6px 8px;">STOP</th>
                      <th style="padding:6px 8px;">FPE</th>
                      <th style="padding:6px 8px;">메모</th>
                      <th style="padding:6px 8px;">날짜</th>
                      <th style="padding:6px 8px;">session_id</th>
                    </tr>
                  </thead>
                  <tbody id="episodes-table-body">
                    <tr><td colspan="14" style="text-align:center; padding:12px; color:var(--text-muted);">기록이 없습니다.</td></tr>
                  </tbody>
                </table>
              </div>

              <!-- 행 클릭 시 이 패널이 해당 세션으로 바뀜 — 항상 떠 있음 -->
              <div class="card" style="padding:12px; background:#101726;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                  <span style="font-size:11px; color:var(--text-muted); font-weight:600; text-transform:uppercase;">✏️ 에피소드 수정</span>
                  <span id="ep-edit-target" style="font-size:11px; color:var(--cyan); font-family:var(--font-mono);">— 행을 클릭하세요 —</span>
                </div>
                <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:8px;">
                  <div>
                    <label style="font-size:10px; color:var(--text-muted);">경로</label>
                    <select id="ep-edit-path" style="width:100%; padding:5px 6px; background:#090d16; border:1px solid var(--border-glow); border-radius:6px; color:#fff; font-size:11px;"></select>
                  </div>
                  <div>
                    <label style="font-size:10px; color:var(--text-muted);">결과</label>
                    <select id="ep-edit-success" style="width:100%; padding:5px 6px; background:#090d16; border:1px solid var(--border-glow); border-radius:6px; color:#fff; font-size:11px;">
                      <option value="성공">성공</option>
                      <option value="실패">실패</option>
                    </select>
                  </div>
                  <div>
                    <label style="font-size:10px; color:var(--text-muted);">steps</label>
                    <input id="ep-edit-steps" type="number" style="width:100%; padding:5px 6px; background:#090d16; border:1px solid var(--border-glow); border-radius:6px; color:#fff; font-size:11px;">
                  </div>
                  <div>
                    <label style="font-size:10px; color:var(--text-muted);">lat (ms)</label>
                    <input id="ep-edit-lat" type="number" step="0.1" style="width:100%; padding:5px 6px; background:#090d16; border:1px solid var(--border-glow); border-radius:6px; color:#fff; font-size:11px;">
                  </div>
                  <div>
                    <label style="font-size:10px; color:var(--text-muted);">FPE</label>
                    <input id="ep-edit-fpe" type="number" step="0.1" style="width:100%; padding:5px 6px; background:#090d16; border:1px solid var(--border-glow); border-radius:6px; color:#fff; font-size:11px;">
                  </div>
                  <div>
                    <label style="font-size:10px; color:var(--text-muted);">메모</label>
                    <input id="ep-edit-note" type="text" style="width:100%; padding:5px 6px; background:#090d16; border:1px solid var(--border-glow); border-radius:6px; color:#fff; font-size:11px;">
                  </div>
                </div>
                <div style="display:flex; gap:8px;">
                  <button class="btn btn-cyan" style="flex:1;" onclick="_epEditSave()">💾 저장</button>
                  <button class="btn btn-outline" style="flex:1;" onclick="_epEditClear()">닫기</button>
                </div>
                <div id="ep-edit-status" style="font-size:10px; color:var(--text-muted); margin-top:6px;">—</div>
              </div>
            </div>

          </div>

        </div>

        <!-- 하단: 간이 히스토리 (Drive Control 탭과 동일 데이터, Path Test 탭에도 노출) -->
        <div class="card" style="margin-top:24px;">
          <div class="card-title">📋 최근 주행 타임라인</div>
          <div id="collect-summary-vfy" style="display:grid; grid-template-columns:1fr 1fr; gap:6px; margin-bottom:10px; font-size:11px;">
            <div style="background:#101726; border:1px solid var(--border-glow); border-radius:6px; padding:6px 10px;">
              <div style="color:var(--cyan); font-weight:700; margin-bottom:3px;">📋 수집 세션</div>
              <div id="cs-session-vfy" style="color:var(--text-muted); font-family:var(--font-mono);">—</div>
            </div>
            <div style="background:#101726; border:1px solid var(--border-glow); border-radius:6px; padding:6px 10px;">
              <div style="color:var(--amber); font-weight:700; margin-bottom:3px;">🔍 Grounding</div>
              <div id="cs-grounding-vfy" style="color:var(--text-muted); font-family:var(--font-mono);">—</div>
            </div>
          </div>
          <div class="table-wrapper">
            <table>
              <thead>
                <tr>
                  <th>Step</th>
                  <th>Predicted Label</th>
                  <th>Total Latency</th>
                  <th>Gnd Latency</th>
                  <th>MLP Latency</th>
                  <th>Bbox Area</th>
                </tr>
              </thead>
              <tbody id="drive-history-table-vfy">
                <tr><td colspan="6" style="text-align:center;color:var(--text-muted);">주행을 시작하면 실시간 분석 데이터가 생성됩니다.</td></tr>
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>

    <!-- 탭 5: STOP & Calibration -->
    <div id="tab-calib" class="tab-content">
      <div class="scroll-container">
        <div class="grid-main">
          <!-- 왼쪽: 뷰어 및 수동 조작 -->
          <div class="card">
            <div class="card-title">📷 캘리브레이션 카메라</div>
            <div style="display:flex; align-items:center; gap:8px; font-size:11px; margin-bottom:10px; padding:4px 8px; background:#101726; border:1px solid var(--border-glow); border-radius:6px;">
              <span style="color:var(--text-muted);">📹 카메라 프로세스:</span>
              <span id="cam-proc-status-calib" class="cam-proc-status" style="color:var(--cyan); font-family:var(--font-mono); flex:1;">—</span>
              <button class="btn btn-outline" onclick="camProcStart()" style="font-size:10px; padding:2px 8px;">▶ 시작</button>
              <button class="btn btn-outline" onclick="camProcStop()" style="font-size:10px; padding:2px 8px;">■ 정지</button>
              <button class="btn btn-outline" onclick="camProcRefresh()" style="font-size:10px; padding:2px 8px;">↻</button>
            </div>
            <div class="viewport-wrapper">
              <img id="calib-stream-img" class="viewport-img" src="/camera/stream">
              <canvas id="calib-canvas" class="viewport-canvas" width="640" height="360"></canvas>
            </div>
            
            <!-- 조이스틱 컨트롤 -->
            <div style="margin-top:24px; display:flex; justify-content:space-around; align-items:center;">
              <div class="joystick-panel">
                <div class="joystick-grid">
                  <button class="joy-btn" onmousedown="sendJoy('Q')" onmouseup="sendJoy('STOP')" id="joy-Q">↖Q</button>
                  <button class="joy-btn" onmousedown="sendJoy('W')" onmouseup="sendJoy('STOP')" id="joy-W">▲W</button>
                  <button class="joy-btn" onmousedown="sendJoy('E')" onmouseup="sendJoy('STOP')" id="joy-E">↗E</button>
                  <button class="joy-btn" onmousedown="sendJoy('A')" onmouseup="sendJoy('STOP')" id="joy-A">◀A</button>
                  <button class="joy-btn stop" onclick="sendJoy('STOP')" id="joy-STOP">⏹</button>
                  <button class="joy-btn" onmousedown="sendJoy('D')" onmouseup="sendJoy('STOP')" id="joy-D">▶D</button>
                  <button class="joy-btn" onmousedown="sendJoy('R')" onmouseup="sendJoy('STOP')" id="joy-R">↺R</button>
                  <button class="joy-btn" onmousedown="sendJoy('S')" onmouseup="sendJoy('STOP')" id="joy-S">▼S</button>
                  <button class="joy-btn" onmousedown="sendJoy('T')" onmouseup="sendJoy('STOP')" id="joy-T">↻T</button>
                </div>
              </div>
              
              <div style="flex:1; margin-left:24px;">
                <div class="form-group">
                  <label id="joy-speed-lbl">이동 수동 속도: 1.15</label>
                  <input type="range" id="joy-speed" min="0.3" max="2.0" step="0.05" value="1.15" oninput="document.getElementById('joy-speed-lbl').textContent = '이동 수동 속도: ' + this.value" onchange="syncJoystickSpeed(this.value)">
                </div>
                <div style="font-size:11px;color:var(--text-muted);">
                  키보드 단축키 지원: 키패드 WASD 및 Q/E/R/T/SpaceBar 정지 대응
                </div>
              </div>
            </div>

            <!-- 🕹️ 게임패드 (DragonRise) -->
            <div style="margin-top:16px; background:#151f32; border:1px solid var(--border-glow); border-radius:10px; padding:12px; display:flex; flex-direction:column; gap:8px;">
              <div style="display:flex; align-items:center; justify-content:space-between;">
                <span style="font-size:11px; color:var(--text-muted); font-weight:600; text-transform:uppercase;">🕹️ 게임패드 (DragonRise)</span>
                <div style="display:flex; gap:6px;">
                  <button id="js-toggle-btn" class="btn btn-cyan" style="font-size:11px; padding:4px 10px;" onclick="joystickToggle()">비활성화</button>
                  <button id="js-mode-btn" class="btn btn-outline" style="font-size:11px; padding:4px 10px;" onclick="joystickMode()">SYNC↔ASYNC</button>
                </div>
              </div>
              <div id="js-status" style="font-size:12px; font-family:var(--font-mono); color:var(--cyan); white-space:pre-line;">🔌 초기화 중...</div>
              <div style="font-size:10px; color:var(--text-muted); line-height:1.4;">
                Left Stick → 이동 | Right Stick X → 회전 | A → STOP | Start → SYNC↔ASYNC 전환<br>
                📸 SYNC: 0.45s bang-bang | 🌊 ASYNC: 10Hz 연속 + 300ms Jitter Hold
              </div>
            </div>
          </div>

          <!-- 오른쪽: 캘리브레이션 세팅 -->
          <div class="card" style="display:flex; flex-direction:column; justify-content:space-between;">
            <div>
              <div class="card-title"> 임계값 및 녹화 관리</div>
              
              <div class="form-group">
                <label id="calib-thr-lbl">Area 임계값: 0.18</label>
                <input type="range" id="calib-thr" min="0.05" max="0.50" step="0.005" value="0.18" oninput="document.getElementById('calib-thr-lbl').textContent = 'Area 임계값: ' + this.value">
                <button class="btn btn-cyan" style="width:100%; margin-top:12px;" onclick="applyThreshold()">서버 임계값 적용</button>
              </div>
              
              <hr style="border:0; border-top:1px solid var(--border-glow); margin:20px 0;">
              
              <div class="form-group">
                <label>세션 녹화 이름 (비워두면 자동)</label>
                <input type="text" id="calib-rec-name" placeholder="calib_YYYYMMDD_HHMMSS">
              </div>
              
              <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:12px;">
                <button class="btn btn-rose" id="calib-start-rec-btn" onclick="startCalibRec()">⏺ 녹화 시작</button>
                <button class="btn btn-outline" id="calib-stop-rec-btn" onclick="stopCalibRec()" disabled>⏹ 녹화 중지</button>
              </div>
              
              <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:10px;">
                <button class="btn btn-outline" onclick="snapCalibFrame()">📸 스냅샷 추가</button>
                <button class="btn btn-outline" onclick="clearCalibRec()">🗑 초기화</button>
              </div>
              
              <button class="btn btn-cyan" style="width:100%; margin-top:10px;" onclick="saveCalibRec()">💾 캘리브레이션 데이터 저장</button>
            </div>
            
            <div class="form-group" style="margin-top:20px;">
              <label>캘리브레이션 세션 로그</label>
              <div class="status-console" id="calib-status-log" style="min-height:90px;">준비</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 탭 6: Session History & Labeling -->
    <div id="tab-history" class="tab-content">
      <div class="scroll-container">
        <div class="grid-main" style="grid-template-columns: 280px 1fr;">
          
          <!-- H5 세션 리스트 -->
          <div class="card" style="padding:16px; overflow-y:auto; max-height:calc(100vh - 150px);">
            <div class="card-title">📂 저장된 세션</div>
            <button class="btn btn-outline" style="width:100%; margin-bottom:12px; font-size:12px;" onclick="loadSessionList()">🔄 리스트 새로고침</button>
            <div id="session-list-group" style="display:flex; flex-direction:column; gap:8px;">
              <!-- 동적 로드 -->
            </div>
          </div>
          
          <!-- H5 프레임 인스펙터 -->
          <div class="card">
            <div class="card-title">📚 프레임 인스펙터 <span id="inspect-sid-lbl" class="text-cyan" style="font-size:13px; text-transform:none;"></span></div>
            
            <div id="inspector-placeholder" style="text-align:center; padding:80px 0; color:var(--text-muted);">
              왼쪽 목록에서 세션을 선택하면 상세 프레임 분석 패널이 활성화됩니다.
            </div>
            
            <div id="inspector-body" class="frame-inspector" style="display:none;">

              <!-- 프레임 비디오/이미지 뷰어 -->
              <div>
                <div class="viewport-wrapper" style="background:#000;">
                  <img id="inspect-frame-img" class="viewport-img" src="">
                  <canvas id="inspect-canvas" class="viewport-canvas" width="640" height="360"></canvas>
                </div>

                <!-- 플레이어 컨트롤 슬라이더 -->
                <div style="margin-top:16px;">
                  <input type="range" id="inspect-slider" min="0" max="0" value="0" oninput="showInspectFrame(this.value)">
                  <div style="display:flex; justify-content:space-between; font-size:12px; color:var(--text-muted); margin-top:4px;">
                    <span id="inspect-frame-idx-lbl">Frame: 0 / 0</span>
                    <span id="inspect-frame-type-lbl">📡 live</span>
                  </div>
                </div>

                <!-- 프레임 타임라인 스트립 — 이상치/타입을 색으로, 클릭하면 바로 점프 -->
                <div id="inspect-timeline" style="display:flex; gap:1px; margin-top:10px; height:20px; border-radius:4px; overflow:hidden; border:1px solid var(--border-glow);"></div>
                <div style="display:flex; gap:12px; font-size:9px; color:var(--text-muted); margin-top:4px;">
                  <span><span style="display:inline-block;width:8px;height:8px;background:#10b981;border-radius:2px;margin-right:3px;"></span>정상</span>
                  <span><span style="display:inline-block;width:8px;height:8px;background:#f43f5e;border-radius:2px;margin-right:3px;"></span>이상치</span>
                  <span><span style="display:inline-block;width:8px;height:8px;background:#f59e0b;border-radius:2px;margin-right:3px;"></span>ARRIVAL</span>
                  <span><span style="display:inline-block;width:8px;height:8px;background:#8b5cf6;border-radius:2px;margin-right:3px;"></span>PREVIEW</span>
                  <span><span style="display:inline-block;width:8px;height:8px;background:#06b6d4;border-radius:2px;margin-right:3px;"></span>현재</span>
                </div>

                <div style="display:flex; gap:8px; margin-top:12px; justify-content:center;">
                  <button class="btn btn-outline" onclick="prevInspectFrame()">◀ 이전</button>
                  <button class="btn btn-outline" id="btn-inspect-play" onclick="toggleInspectPlay()">▶ PLAY</button>
                  <button class="btn btn-outline" onclick="nextInspectFrame()">다음 ▶</button>
                  <button class="btn btn-outline" onclick="jumpToNextAnomaly()">⚠️ 다음 이상치로</button>
                </div>

              <!-- 세션 요약 + 경로검증 기록 — 1:1 두 컬럼으로 나란히, 딱 2행만 차지 -->
              <div style="margin-top:20px; display:grid; grid-template-columns:1fr 1fr; gap:12px;">

              <!-- 세션 요약 — 스크럽하기 전에 한눈에 보는 통계, 타일 블록 형태 -->
              <div style="background:#151f32; border:1px solid var(--border-glow); border-radius:10px; padding:12px 14px;">
                <div style="font-size:11px; color:var(--text-muted); font-weight:700; text-transform:uppercase; margin-bottom:8px;">📊 세션 요약</div>
                <div class="mini-tile-grid">
                  <div class="mini-tile"><span class="mt-label">프레임</span><span class="mt-value" id="inspect-sum-frames">—</span></div>
                  <div class="mini-tile"><span class="mt-label">평균 지연</span><span class="mt-value" id="inspect-sum-lat">—</span><span class="mt-sub" id="inspect-sum-lat-sub">—</span></div>
                  <div class="mini-tile"><span class="mt-label">총 소요</span><span class="mt-value" id="inspect-sum-total">—</span></div>
                  <div class="mini-tile"><span class="mt-label">Live / Cache</span><span class="mt-value" id="inspect-sum-cache">—</span></div>
                  <div class="mini-tile"><span class="mt-label">라벨링</span><span class="mt-value" id="inspect-sum-labeled">—</span></div>
                  <div class="mini-tile" id="inspect-sum-warns-tile"><span class="mt-label">이상치</span><span class="mt-value" id="inspect-sum-warns">—</span></div>
                </div>
                <div style="margin-top:10px;">
                  <div class="mt-label" style="font-size:9px; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.04em; font-weight:700; margin-bottom:5px;">액션 분포</div>
                  <div id="inspect-sum-actions" class="mini-tile-grid"></div>
                </div>
              </div>

              <!-- 경로검증(Tab 4 episode_log) 매칭 기록 — 도착 위치(cx/area/STOP) 등 실제 검증 결과 + 수정 -->
              <div id="inspect-episode-box" style="background:#151f32; border:1px solid var(--border-glow); border-radius:10px; padding:12px 14px; display:none;">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                  <span style="font-size:11px; color:var(--text-muted); font-weight:700; text-transform:uppercase;">🧪 경로검증 기록 (episode_log)</span>
                  <div style="display:flex; align-items:center; gap:8px;">
                    <span id="inspect-ep-status" style="font-size:10px; color:var(--text-muted);">—</span>
                    <button class="btn btn-outline" style="font-size:10px; padding:3px 8px;" onclick="toggleInspectEpEdit()">✏️ 수정</button>
                  </div>
                </div>
                <div id="inspect-ep-view" style="display:none;">
                  <div class="mini-tile-grid">
                    <div class="mini-tile"><span class="mt-label">경로</span><span class="mt-value" id="iep-path">—</span></div>
                    <div class="mini-tile" id="iep-success-tile"><span class="mt-label">결과</span><span class="mt-value" id="iep-success">—</span></div>
                    <div class="mini-tile"><span class="mt-label">steps</span><span class="mt-value" id="iep-steps">—</span></div>
                    <div class="mini-tile"><span class="mt-label">평균 지연</span><span class="mt-value" id="iep-lat">—</span></div>
                    <div class="mini-tile"><span class="mt-label">top액션</span><span class="mt-value" id="iep-topaction">—</span></div>
                    <div class="mini-tile"><span class="mt-label">grounding</span><span class="mt-value" id="iep-gnd">—</span></div>
                    <div class="mini-tile" title="도착 시점 bbox 면적 — 클수록 근접"><span class="mt-label">도착 area</span><span class="mt-value" id="iep-area">—</span></div>
                    <div class="mini-tile" title="도착 시점 bbox 중심 x좌표 — 0.5가 중앙"><span class="mt-label">도착 cx</span><span class="mt-value" id="iep-cx">—</span></div>
                    <div class="mini-tile" id="iep-stop-tile"><span class="mt-label">STOP</span><span class="mt-value" id="iep-stop">—</span></div>
                    <div class="mini-tile"><span class="mt-label">FPE</span><span class="mt-value" id="iep-fpe">—</span></div>
                    <div class="mini-tile"><span class="mt-label">날짜</span><span class="mt-value" id="iep-date" style="font-size:11px;">—</span></div>
                  </div>
                  <div class="mini-tile" style="margin-top:6px;">
                    <span class="mt-label">메모</span>
                    <span id="iep-note" style="font-size:12px; font-weight:400; color:#fff; font-family:var(--font-sans, inherit); white-space:pre-wrap; word-break:break-word;">—</span>
                  </div>
                </div>
                <div id="inspect-ep-empty" style="font-size:11px; color:var(--text-muted); display:none;">
                  이 세션에 매칭되는 경로검증 기록이 없습니다 — Tab 4(🧪 경로 검증)에서 기록을 저장하면 session_id로 자동 연결됩니다.
                </div>
                <div id="inspect-ep-edit" style="display:none; margin-top:10px; padding-top:10px; border-top:1px solid var(--border-glow);">

                  <!-- 경로 — Tab 4 빠른 레이블 선택과 동일한 버튼 그리드 -->
                  <div style="margin-bottom:10px;">
                    <label style="font-size:10px; color:var(--text-muted);">경로 (path_type)</label>
                    <div style="display:flex; flex-direction:column; gap:4px; background:#101726; padding:8px; border-radius:8px; border:1px solid var(--border-glow); margin-top:4px;">
                      <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:4px;">
                        <button type="button" class="btn btn-outline iep-path-btn" data-path="obj_left" onclick="selectInspectPathType('obj_left')" style="font-size:10px; padding:4px 0;">obj_left</button>
                        <button type="button" class="btn btn-outline iep-path-btn" data-path="obj_center" onclick="selectInspectPathType('obj_center')" style="font-size:10px; padding:4px 0;">obj_center</button>
                        <button type="button" class="btn btn-outline iep-path-btn" data-path="obj_right" onclick="selectInspectPathType('obj_right')" style="font-size:10px; padding:4px 0;">obj_right</button>
                      </div>
                      <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:4px;">
                        <button type="button" class="btn btn-outline iep-path-btn" data-path="left_left" onclick="selectInspectPathType('left_left')" style="font-size:10px; padding:4px 0;">left_left</button>
                        <button type="button" class="btn btn-outline iep-path-btn" data-path="left_straight" onclick="selectInspectPathType('left_straight')" style="font-size:10px; padding:4px 0;">left_straight</button>
                        <button type="button" class="btn btn-outline iep-path-btn" data-path="left_right" onclick="selectInspectPathType('left_right')" style="font-size:10px; padding:4px 0;">left_right</button>
                      </div>
                      <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:4px;">
                        <button type="button" class="btn btn-outline iep-path-btn" data-path="center_left" onclick="selectInspectPathType('center_left')" style="font-size:10px; padding:4px 0;">center_left</button>
                        <button type="button" class="btn btn-outline iep-path-btn" data-path="center_straight" onclick="selectInspectPathType('center_straight')" style="font-size:10px; padding:4px 0;">center_straight</button>
                        <button type="button" class="btn btn-outline iep-path-btn" data-path="center_right" onclick="selectInspectPathType('center_right')" style="font-size:10px; padding:4px 0;">center_right</button>
                      </div>
                      <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:4px;">
                        <button type="button" class="btn btn-outline iep-path-btn" data-path="right_left" onclick="selectInspectPathType('right_left')" style="font-size:10px; padding:4px 0;">right_left ★</button>
                        <button type="button" class="btn btn-outline iep-path-btn" data-path="right_straight" onclick="selectInspectPathType('right_straight')" style="font-size:10px; padding:4px 0;">right_straight</button>
                        <button type="button" class="btn btn-outline iep-path-btn" data-path="right_right" onclick="selectInspectPathType('right_right')" style="font-size:10px; padding:4px 0;">right_right</button>
                      </div>
                      <div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:4px;">
                        <button type="button" class="btn btn-outline iep-path-btn" data-path="dist_10cm" onclick="selectInspectPathType('dist_10cm')" style="font-size:10px; padding:4px 0;">dist_10cm</button>
                        <button type="button" class="btn btn-outline iep-path-btn" data-path="dist_20cm" onclick="selectInspectPathType('dist_20cm')" style="font-size:10px; padding:4px 0;">dist_20cm</button>
                        <button type="button" class="btn btn-outline iep-path-btn" data-path="dist_30cm" onclick="selectInspectPathType('dist_30cm')" style="font-size:10px; padding:4px 0;">dist_30cm</button>
                      </div>

                      <!-- 트랙A 극단배치(V6) — Tab 4와 동일 섹션 -->
                      <div style="font-size:10px; color:var(--amber); font-weight:600; text-transform:uppercase; margin-top:6px;">🎯 트랙A 극단배치 (V6)</div>
                      <div id="inspect-tracka-grid" style="display:flex; flex-direction:column; gap:4px; margin-top:4px;"></div>
                    </div>
                  </div>

                  <!-- 결과 — Tab 4 성공/실패 큰 버튼과 동일 -->
                  <div style="margin-bottom:10px;">
                    <label style="font-size:10px; color:var(--text-muted);">주행 결과</label>
                    <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:4px;">
                      <button type="button" id="iep-edit-succ-btn" class="btn btn-outline" style="font-weight:bold; font-size:11px;" onclick="setInspectEpResult('성공')">✅ 성공</button>
                      <button type="button" id="iep-edit-fail-btn" class="btn btn-outline" style="font-weight:bold; font-size:11px;" onclick="setInspectEpResult('실패')">❌ 실패</button>
                    </div>
                  </div>

                  <!-- FPE — Tab 4와 동일한 슬라이더 + 프리셋 버튼 -->
                  <div style="margin-bottom:10px;">
                    <label id="iep-edit-fpe-lbl" style="font-size:10px; color:var(--text-muted);">FPE: 0.00</label>
                    <input type="range" id="iep-edit-fpe" min="0.0" max="0.5" step="0.01" value="0.0" style="width:100%; accent-color:var(--cyan);" oninput="document.getElementById('iep-edit-fpe-lbl').textContent = 'FPE: ' + this.value">
                    <div style="display:flex; flex-wrap:wrap; gap:4px; margin-top:4px;">
                      <button type="button" class="btn btn-outline" onclick="setInspectFpeValue(0.0)" style="font-size:9px; padding:2px 4px;">0.0</button>
                      <button type="button" class="btn btn-outline" onclick="setInspectFpeValue(0.01)" style="font-size:9px; padding:2px 4px;">0.01</button>
                      <button type="button" class="btn btn-outline" onclick="setInspectFpeValue(0.02)" style="font-size:9px; padding:2px 4px;">0.02</button>
                      <button type="button" class="btn btn-outline" onclick="setInspectFpeValue(0.03)" style="font-size:9px; padding:2px 4px;">0.03</button>
                      <button type="button" class="btn btn-outline" onclick="setInspectFpeValue(0.05)" style="font-size:9px; padding:2px 4px;">0.05</button>
                      <button type="button" class="btn btn-outline" onclick="setInspectFpeValue(0.08)" style="font-size:9px; padding:2px 4px;">0.08</button>
                      <button type="button" class="btn btn-outline" onclick="setInspectFpeValue(0.1)" style="font-size:9px; padding:2px 4px;">0.1</button>
                      <button type="button" class="btn btn-outline" onclick="setInspectFpeValue(0.15)" style="font-size:9px; padding:2px 4px;">0.15</button>
                      <button type="button" class="btn btn-outline" onclick="setInspectFpeValue(0.2)" style="font-size:9px; padding:2px 4px;">0.2</button>
                      <button type="button" class="btn btn-outline" onclick="setInspectFpeValue(0.3)" style="font-size:9px; padding:2px 4px;">0.3</button>
                      <button type="button" class="btn btn-outline" onclick="setInspectFpeValue(0.5)" style="font-size:9px; padding:2px 4px;">0.5</button>
                    </div>
                  </div>

                  <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:8px;">
                    <div>
                      <label style="font-size:10px; color:var(--text-muted);">steps</label>
                      <input id="iep-edit-steps" type="number" style="width:100%; padding:5px 6px; background:#090d16; border:1px solid var(--border-glow); border-radius:6px; color:#fff; font-size:11px;">
                    </div>
                    <div>
                      <label style="font-size:10px; color:var(--text-muted);">lat (ms)</label>
                      <input id="iep-edit-lat" type="number" step="0.1" style="width:100%; padding:5px 6px; background:#090d16; border:1px solid var(--border-glow); border-radius:6px; color:#fff; font-size:11px;">
                    </div>
                  </div>
                  <div style="margin-bottom:8px;">
                    <label style="font-size:10px; color:var(--text-muted);">메모</label>
                    <input id="iep-edit-note" type="text" style="width:100%; padding:5px 6px; background:#090d16; border:1px solid var(--border-glow); border-radius:6px; color:#fff; font-size:11px;">
                  </div>

                  <div style="display:flex; gap:8px;">
                    <button class="btn btn-cyan" style="flex:1;" onclick="saveInspectEpisode()">💾 저장</button>
                    <button class="btn btn-outline" style="flex:1;" onclick="toggleInspectEpEdit()">닫기</button>
                  </div>
                </div>
              </div>

              </div><!-- /세션 요약 + 경로검증 기록 1:1 컬럼 -->
              </div>

              <!-- 프레임 메타데이터 & 셀프 라벨링 -->
              <div style="display:flex; flex-direction:column; justify-content:space-between;">
                <div>
                  <div class="form-group">
                    <label>Action & Latency</label>
                    <div class="kv-grid" style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:4px;">
                      <div class="srv-pill" style="padding:4px 8px;">액션: <strong id="inspect-action-lbl">—</strong></div>
                      <div class="srv-pill" style="padding:4px 8px; flex-direction:column; align-items:flex-start; gap:1px;">
                        <span>지연: <strong id="inspect-lat-lbl">—ms</strong></span>
                        <span id="inspect-lat-sec-lbl" style="font-size:10px; font-weight:400; color:var(--text-muted);">—</span>
                      </div>
                    </div>
                  </div>

                  <div class="form-group" style="margin-top:14px;">
                    <label>Grounding (bbox)</label>
                    <div class="kv-grid" style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:4px;">
                      <div class="srv-pill" style="padding:4px 8px;">cx: <strong id="inspect-cx-lbl">—</strong></div>
                      <div class="srv-pill" style="padding:4px 8px;">cy: <strong id="inspect-cy-lbl">—</strong></div>
                      <div class="srv-pill" style="padding:4px 8px;">area: <strong id="inspect-area-lbl">—</strong></div>
                      <div class="srv-pill" style="padding:4px 8px;">has_bbox: <strong id="inspect-hasbbox-lbl">—</strong></div>
                    </div>
                  </div>

                  <div class="form-group" style="margin-top:16px;">
                    <label>이상치 감지 알림 (Anomaly warnings)</label>
                    <div id="inspect-anomaly-box" style="padding:10px; border-radius:8px; background:#1e1b1b; border:1px solid #3d2323; font-size:12px; min-height:60px;">
                      ✅ 감지된 이상치 경고 없음
                    </div>
                  </div>

                  <div class="form-group" style="margin-top:20px;">
                    <label>셀프 라벨링 (Self-Labeling)</label>
                    <div style="display:grid; grid-template-columns:repeat(4, 1fr); gap:8px; margin-top:6px;">
                      <button class="btn btn-outline" id="lbl-btn-L" onclick="saveFrameLabel('L')">L (좌)</button>
                      <button class="btn btn-outline" id="lbl-btn-C" onclick="saveFrameLabel('C')">C (중)</button>
                      <button class="btn btn-outline" id="lbl-btn-R" onclick="saveFrameLabel('R')">R (우)</button>
                      <button class="btn btn-outline text-rose" id="lbl-btn-NONE" onclick="saveFrameLabel('NONE')">삭제</button>
                    </div>
                    <div style="font-size:11px; color:var(--text-muted); margin-top:8px;">
                      라벨을 선택하면 서버 `/tmp/mona_preview_labels.json`에 즉시 자동 기록됩니다.
                    </div>
                  </div>
                </div>

                <div class="form-group" id="inspect-runtime-cfg-box" style="margin-top:20px; display:none;">
                  <label>⚙️ 당시 런타임 설정 (경로검증 Config + 서버 스냅샷)</label>
                  <div id="inspect-runtime-cfg-grid" class="mini-tile-grid" style="margin-top:6px;"></div>
                </div>

                <div class="srv-pill" style="font-size:11px; margin-top:16px; display:block;">
                  H5 파일 속성:<br>
                  <span id="inspect-attrs-lbl" style="font-family:var(--font-mono); color:var(--text-muted); word-break:break-all;">—</span>
                </div>

                <div style="margin-top:16px;">
                  <button class="btn btn-rose" style="width:100%; font-size:12px; padding:8px 12px;" onclick="deleteActiveSession()">🗑️ 세션 파일 영구 삭제</button>
                </div>
              </div>
            </div>

          </div>
        </div>
      </div>
    </div>

    <!-- 탭 6.5: 🌀 오버슈트 가이드 (트랙C 수집 예시) -->
    <div id="tab-overshoot" class="tab-content">
      <div class="scroll-container" style="padding:20px;">
        <div class="card" style="padding:16px; margin-bottom:16px; background:rgba(245,158,11,0.08); border:1px solid var(--amber);">
          <div class="card-title" style="color:var(--amber);">⚠️ 이 탭은 실제 학습 데이터가 아닙니다</div>
          <div style="font-size:12px; color:var(--text-muted); line-height:1.6;">
            트랙C(오버슈트→재수렴, CH62 근거)가 필요한 이유는 <b>"과하게 꺾은 뒤 반대로
            재보정하는" 궤적 자체가 기존 데이터에 없기 때문</b>입니다. 아래는 실패
            반례 세션(<code>session_20260711_205228</code>)의 <b>실제 프레임</b>으로
            "과도한 회전 → 반등 없이 계속 밀림" 구간을 보여주고, 이어서 같은 프레임을
            <b>좌우반전+역순재생한 합성 구간</b>으로 "여기서부터 반대방향 재보정이
            있었어야 함"을 시각적으로 예시할 뿐입니다. 실제 물리 수집은
            📷 데이터수집 탭에서 <code>오버슈트→우</code> / <code>오버슈트→좌</code>
            라벨로 직접 진행해야 합니다.
          </div>
        </div>

        <div class="grid-main">
          <div class="card" style="padding:16px;">
            <div class="card-title">🎬 예시 재생
              <div style="display:flex; gap:6px; margin-left:auto;">
                <button class="btn btn-outline" id="osg-btn-left" onclick="osgSetDirection('left_recover')">오버슈트→우 재보정</button>
                <button class="btn btn-outline" id="osg-btn-right" onclick="osgSetDirection('right_recover')">오버슈트→좌 재보정</button>
              </div>
            </div>
            <div class="viewport-wrapper" style="position:relative; border-radius:12px; overflow:hidden; background:#000; aspect-ratio:16/9; border:1px solid var(--border-glow);">
              <img id="osg-frame-img" style="width:100%; height:100%; object-fit:contain;">
              <div class="overlay-info">
                <div class="overlay-badge" id="osg-phase-badge">—</div>
                <div class="overlay-badge" id="osg-idx-badge">Frame: 0/0</div>
              </div>
            </div>
            <div style="display:flex; align-items:center; gap:10px; margin-top:14px;">
              <button class="btn btn-outline" onclick="osgStep(-1)">◀ 이전</button>
              <button class="btn btn-cyan" id="osg-play-btn" onclick="osgTogglePlay()">▶ 재생</button>
              <button class="btn btn-outline" onclick="osgStep(1)">다음 ▶</button>
              <input type="range" id="osg-scrub" min="0" max="0" value="0" style="flex:1;" oninput="osgSeek(this.value)">
            </div>
            <div id="osg-timeline" style="display:flex; gap:2px; margin-top:10px; height:14px;"></div>
            <div style="display:flex; justify-content:space-between; font-size:10px; color:var(--text-muted); margin-top:4px;">
              <span>◀ 실제 프레임(과도한 회전)</span>
              <span>합성 예시(재보정 시연) ▶</span>
            </div>
          </div>

          <div class="card" style="display:flex; flex-direction:column; gap:14px;">
            <div class="card-title">📊 현재 프레임 정보</div>
            <div class="kv-grid">
              <div class="form-group">
                <label>구간</label>
                <input type="text" id="osg-info-phase" readonly value="—">
              </div>
              <div class="form-group">
                <label>원본 프레임 idx</label>
                <input type="text" id="osg-info-realidx" readonly value="—">
              </div>
              <div class="form-group">
                <label>cx (표시용, 합성 구간은 1-cx)</label>
                <input type="text" id="osg-info-cx" readonly value="—">
              </div>
              <div class="form-group">
                <label>원본 액션 라벨</label>
                <input type="text" id="osg-info-action" readonly value="—">
              </div>
            </div>
            <div class="form-group">
              <label>설명</label>
              <div id="osg-info-note" class="status-console" style="min-height:70px; font-size:12px;">—</div>
            </div>
            <div class="srv-pill" style="font-size:11px;">
              원본 세션: <code id="osg-source-sid">—</code> · CH62 참고:
              <code>docs/v5/closed_loop_eval/CH62_FORWARD_LOCK_AND_LABEL_CONFOUND.md</code>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 탭 7: System Manage -->
    <div id="tab-system" class="tab-content">
      <div class="scroll-container">
        <div class="grid-main">
          <!-- 왼쪽: 스위치 제어 -->
          <div class="card" style="display:flex; flex-direction:column; gap:20px;">
            <div class="card-title">⚙️ 서버 런타임 파라미터</div>
            
            <div style="display:flex; flex-direction:column; gap:16px;">
              <div style="display:flex; align-items:center; justify-content:space-between;">
                <div>
                  <h4 style="font-size:14px; font-weight:600;">프리뷰 격리회전 (preview_enabled)</h4>
                  <p style="font-size:11px; color:var(--text-muted);">그라운딩 실패 시 좁은범위 회전 탐색 활성화</p>
                </div>
                <label class="switch">
                  <input type="checkbox" id="sys-preview" onchange="applySysParams()">
                  <span class="slider"></span>
                </label>
              </div>
              
              <div style="display:flex; align-items:center; justify-content:space-between;">
                <div>
                  <h4 style="font-size:14px; font-weight:600;">회전방향 힌트 (preview_hint_cx)</h4>
                  <p style="font-size:11px; color:var(--text-muted);">직전 타겟 cx 치우침 방향으로 우선 회전</p>
                </div>
                <label class="switch">
                  <input type="checkbox" id="sys-hint" onchange="applySysParams()">
                  <span class="slider"></span>
                </label>
              </div>
              
              <div style="display:flex; align-items:center; justify-content:space-between;">
                <div>
                  <h4 style="font-size:14px; font-weight:600;">급변 오탐 필터 (cx_jump_filter)</h4>
                  <p style="font-size:11px; color:var(--text-muted);">바운딩박스의 비정상 좌우 점프 현상 차단</p>
                </div>
                <label class="switch">
                  <input type="checkbox" id="sys-jump" onchange="applySysParams()">
                  <span class="slider"></span>
                </label>
              </div>
              
              <div style="display:flex; align-items:center; justify-content:space-between;">
                <div>
                  <h4 style="font-size:14px; font-weight:600;">민감도 캐시 skip N (grounding_skip_n)</h4>
                  <p style="font-size:11px; color:var(--text-muted);">활성화 시 매프레임 대신 3프레임 주기로 캐싱 추론</p>
                </div>
                <label class="switch">
                  <input type="checkbox" id="sys-skip" onchange="applySysParams()">
                  <span class="slider"></span>
                </label>
              </div>
            </div>
            
            <div style="margin-top:20px;">
              <div class="form-group">
                <label id="sys-jump-thresh-lbl">오탐 필터 임계값: 0.30</label>
                <input type="range" id="sys-jump-thresh" min="0.10" max="0.60" step="0.05" value="0.30" onchange="applySysParams()" oninput="document.getElementById('sys-jump-thresh-lbl').textContent = '오탐 필터 임계값: ' + this.value">
              </div>
            </div>
          </div>
          
          <!-- 오른쪽: 추론서버 정보 & 리셋 -->
          <div class="card" style="display:flex; flex-direction:column; justify-content:space-between;">
            <div>
              <div class="card-title">🖥️ 추론 서버 상태</div>
              <div class="kv-grid" id="sys-info-panel">
                <!-- 동적 채움 -->
              </div>
            </div>
            
            <div>
              <button class="btn btn-rose" style="width:100%; margin-bottom:12px;" onclick="resetServerModel()">🧹 CLEAR / RESET SERVER CACHE</button>
              <div style="font-size:11px; color:var(--text-muted); text-align:center;">
                추론 서버가 예기치 않게 멈추거나 메모리 정리가 필요할 때 캐시를 초기화합니다.
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 탭 8: ⚙️ 서버 설정 (모델/그라운더/체크포인트 관리) -->
    <div id="tab-srvcfg" class="tab-content">
      <div class="scroll-container" style="padding:20px;">
        <div style="display:grid; grid-template-columns:1.2fr 1fr; gap:20px; align-items:start;">

          <!-- Column 1: 서버 상태 + 체크포인트 목록 -->
          <div style="display:flex; flex-direction:column; gap:20px;">
            <div class="card" style="padding:16px;">
              <div class="card-title">🖥️ 추론 서버(8001) 상태
                <button class="btn btn-outline" onclick="loadSrvCfg()" style="font-size:11px; padding:4px 10px;">↻ 새로고침</button>
              </div>
              <div id="srvcfg-handshake" style="display:none; margin-bottom:10px; padding:8px 12px; border-radius:8px; background:rgba(244,63,94,0.12); border:1px solid var(--rose); color:var(--rose); font-size:12px; font-weight:600;"></div>
              <div class="table-wrapper">
                <table style="width:100%; font-size:12px;">
                  <tbody id="srvcfg-status-body">
                    <tr><td colspan="2" style="text-align:center; color:var(--text-muted); padding:12px;">로딩 중...</td></tr>
                  </tbody>
                </table>
              </div>
            </div>

            <div class="card" style="padding:16px;">
              <div class="card-title">📦 체크포인트 목록 (runs/*.pt)</div>
              <div style="font-size:11px; color:var(--text-muted); margin-bottom:8px;">
                행 클릭 → 선택. 전환은 우측 "서버 재시작" 필요 (~120s).
              </div>
              <div class="table-wrapper" style="max-height:340px; overflow-y:auto;">
                <table style="width:100%; font-size:11px;">
                  <thead><tr style="text-align:left; background:#151f32;">
                    <th style="padding:6px 8px;">경로</th><th style="padding:6px 8px; width:70px;">크기</th><th style="padding:6px 8px; width:110px;">수정</th>
                  </tr></thead>
                  <tbody id="srvcfg-ckpt-body">
                    <tr><td colspan="3" style="text-align:center; color:var(--text-muted); padding:12px;">로딩 중...</td></tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <!-- Column 2: 그라운더/설정 변경 + 재시작 + 로그 -->
          <div style="display:flex; flex-direction:column; gap:20px;">
            <div class="card" style="padding:16px;">
              <div class="card-title">🔭 그라운더 (A/B)</div>
              <div style="display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:12px;">
                <button id="srvcfg-gr-pg2"   class="btn btn-outline" onclick="selGrounder('pg2')">PG2-448<br><span style="font-size:9px;color:var(--text-muted)">3B · 가변 latency</span></button>
                <button id="srvcfg-gr-owlv2" class="btn btn-outline" onclick="selGrounder('owlv2')">OWL-v2<br><span style="font-size:9px;color:var(--text-muted)">경량 · ~2s 고정</span></button>
              </div>
              <div class="form-group" style="margin-bottom:8px;">
                <label style="font-size:11px;">OWL-v2 threshold (기본 0.25 — 실측 확정값)</label>
                <input type="number" id="srvcfg-owl-thr" min="0.05" max="0.9" step="0.05" value="0.25" style="width:100%; padding:6px 8px; background:#090d16; border:1px solid var(--border-glow); border-radius:6px; color:#fff; font-size:12px;">
              </div>
              <button class="btn btn-cyan" style="width:100%; font-size:12px;" onclick="applyOwlThresh()">threshold 즉시 적용 (재시작 불필요)</button>
              <div class="form-group" style="margin:10px 0 8px;">
                <label style="font-size:11px;">OWL-v2 bbox 보정계수 (기본 3.0 — PG2 대비 area 축소 보정)</label>
                <input type="number" id="srvcfg-owl-area-scale" min="0.5" max="10" step="0.1" value="3.0" style="width:100%; padding:6px 8px; background:#090d16; border:1px solid var(--border-glow); border-radius:6px; color:#fff; font-size:12px;">
              </div>
              <button class="btn btn-cyan" style="width:100%; font-size:12px;" onclick="applyOwlAreaScale()">보정계수 즉시 적용 (재시작 불필요)</button>
              <div style="font-size:10px; color:var(--text-muted); margin-top:6px; line-height:1.4;">
                그라운더 전환(PG2↔OWL)은 모델 로딩이 필요해 아래 재시작으로만 적용됨.
                threshold는 런타임 즉시 반영.
              </div>
            </div>

            <div class="card" style="padding:16px;">
              <div class="card-title">🔁 서버 재시작 (선택 설정 적용)</div>
              <div id="srvcfg-restart-preview" style="font-size:11px; font-family:var(--font-mono); color:var(--amber); background:#101726; border:1px solid var(--border-glow); border-radius:6px; padding:8px 10px; margin-bottom:10px;">변경 없음 — 현재 설정으로 재시작</div>
              <button id="srvcfg-restart-btn" class="btn btn-rose" style="width:100%; font-weight:bold;" onclick="restartInferServer()">🔁 추론 서버 재시작</button>
              <div id="srvcfg-restart-status" style="font-size:11px; color:var(--cyan); text-align:center; font-family:var(--font-mono); margin-top:8px;">—</div>
            </div>

            <div class="card" style="padding:16px;">
              <div class="card-title">📜 서버 로그 (tail)
                <button class="btn btn-outline" onclick="loadSrvLog()" style="font-size:11px; padding:4px 10px;">↻</button>
              </div>
              <pre id="srvcfg-log" style="font-size:10px; font-family:var(--font-mono); background:#090d16; border:1px solid var(--border-glow); border-radius:8px; padding:10px; max-height:260px; overflow:auto; white-space:pre-wrap; color:var(--text-muted);">—</pre>
            </div>
          </div>

        </div>
      </div>
    </div>

    <!-- 탭 9: 📷 데이터수집 (mobile_vla_data_collector.py 웹 이식) -->
    <div id="tab-collect" class="tab-content">
      <div class="scroll-container" style="padding:20px;">

        <!-- Row1 (3:2) — 카메라 | 실시간 상태(녹화현황+입력현황+cx+가이드) -->
        <div style="display:grid; grid-template-columns:3fr 2fr; gap:20px; align-items:start; margin-bottom:20px;">

          <!-- 좌 (3fr): 카메라 + 조이스틱 요약 -->
          <div style="display:flex; flex-direction:column; gap:12px;">
            <div class="card" style="padding:16px;">
              <div class="card-title">📹 실시간 카메라
                <label style="font-size:11px; font-weight:400; color:var(--text-muted); float:right; cursor:pointer; margin-left:10px;">
                  <input type="checkbox" id="toggle-grid-collect" checked onchange="_collectCxDrawOverlay(_collectLastCx, _collectLastColor)" style="accent-color:var(--cyan);"> Grid 표시
                </label>
                <label style="font-size:11px; font-weight:400; color:var(--text-muted); float:right; cursor:pointer;">
                  <input type="checkbox" id="toggle-cxguide-collect" onchange="_collectCxDrawOverlay(_collectLastCx, _collectLastColor)" style="accent-color:var(--amber);"> 배치가이드 표시
                </label>
                <button class="btn btn-outline joystick-mode-btn" onclick="toggleJoystickMode()" style="font-size:10px; padding:3px 8px; float:right; margin-right:8px;" title="🧪 검증: D-pad◀▶=위치 · L1=추론시작 R1=정지 X=성공/A=실패(라벨) L2=기록저장 R2=복귀 · Y=모드전환">🕹️ 조이스틱: 📷 수집</button>
              </div>
              <div style="font-size:10px; color:var(--text-muted); margin-bottom:8px;">cx 오버레이는 실시간 cx 켜면 표시 · 배치가이드는 위 체크박스로 카메라 위에 바로 그려짐</div>
              <div style="display:flex; align-items:center; gap:8px; font-size:11px; margin-bottom:10px; padding:4px 8px; background:#101726; border:1px solid var(--border-glow); border-radius:6px;">
                <span style="color:var(--text-muted);">📹 카메라 프로세스:</span>
                <span id="cam-proc-status-collect" class="cam-proc-status" style="color:var(--cyan); font-family:var(--font-mono); flex:1;">—</span>
                <button class="btn btn-outline" onclick="camProcStart()" style="font-size:10px; padding:2px 8px;">▶ 시작</button>
                <button class="btn btn-outline" onclick="camProcStop()" style="font-size:10px; padding:2px 8px;">■ 정지</button>
                <button class="btn btn-outline" onclick="camProcRefresh()" style="font-size:10px; padding:2px 8px;">↻</button>
              </div>
              <div class="viewport-wrapper">
                <img id="collect-stream-img" src="/camera/stream" class="viewport-img"
                     onerror="this.src='https://placehold.co/1280x720/0f1524/94a3b8?text=Camera+Streaming+Offline'">
                <canvas id="collect-cx-canvas" class="viewport-canvas" width="1280" height="720"></canvas>
              </div>
            </div>

            <!-- 조이스틱 조작 요약 (전체 설명서는 Row2 조이스틱 카드에 있음) — 아이콘 칩 그리드로 섹션 구분 -->
            <div class="card" style="padding:12px 14px;">
              <div class="card-title" style="font-size:12px; margin-bottom:8px;">🕹️ 조이스틱 요약</div>
              <div style="display:flex; flex-wrap:wrap; gap:6px; margin-bottom:8px;">
                <span class="js-chip js-chip-move">🕹️ 왼쪽 스틱 → 이동</span>
                <span class="js-chip js-chip-move">🔄 오른쪽 스틱X → 회전</span>
              </div>
              <div style="display:flex; flex-wrap:wrap; gap:6px; margin-bottom:8px;">
                <span class="js-chip js-chip-rec"><b>L1</b> 녹화시작</span>
                <span class="js-chip js-chip-rec"><b>R1</b> 정지&저장</span>
                <span class="js-chip js-chip-rec"><b>SEL</b> 녹화토글</span>
                <span class="js-chip js-chip-amber"><b>R2</b> 🔄복귀주행</span>
              </div>
              <div style="display:flex; flex-wrap:wrap; gap:6px; margin-bottom:8px;">
                <span class="js-chip js-chip-muted"><b>A</b> STOP</span>
                <span class="js-chip js-chip-muted"><b>B</b> 마지막프레임취소</span>
                <span class="js-chip js-chip-muted"><b>X</b> 에피소드폐기</span>
                <span class="js-chip js-chip-muted"><b>START</b> SYNC↔ASYNC</span>
              </div>
              <div style="display:flex; flex-wrap:wrap; gap:6px;">
                <span class="js-chip js-chip-cyan">◀▶ D-pad 좌우 → 값 순환</span>
                <span class="js-chip js-chip-cyan">▲▼ D-pad 상하 → 축 전환(트랙A/경로/시나리오)</span>
              </div>
            </div>
          </div>

          <!-- 우 (2fr): 실시간 상태 — Gradio status-card 이식(선택현황 통합) + 입력현황(키보드/조이스틱 공용) + cx + 가이드 -->
          <div style="display:flex; flex-direction:column; gap:16px;">
            <div id="collect-status-card" class="collect-status-card">
              <div id="collect-status-main">⏸ IDLE</div>
              <div style="margin-top:8px;">
                <span id="collect-mode-badge-rt" style="font-size:0.55em; padding:3px 10px; border-radius:10px; background:rgba(56,189,248,0.15); color:var(--cyan); font-weight:700;">🎮 D-pad 대상: 트랙A(위치)</span>
              </div>
              <div id="collect-status-sub" style="font-size:0.7em; font-weight:600; opacity:0.85; margin-top:6px;">시나리오: 미지정 · 트랙A: 미지정 + 미지정</div>
            </div>

            <div class="card" style="padding:16px;">
              <div class="card-title">🎮 입력 현황 (키보드 · 조이스틱 공용)</div>
              <div id="collect-last-action" style="font-size:20px; font-family:var(--font-mono); color:var(--emerald); font-weight:700; text-align:center; padding:6px;">STOP</div>
              <div style="display:flex; gap:8px; margin-top:6px; align-items:center; justify-content:center;">
                <span id="collect-active-badge" style="font-size:13px; font-weight:700; padding:5px 14px; border-radius:20px; background:rgba(100,116,139,0.2); color:var(--text-muted);">⏸ 대기중</span>
                <span id="collect-steps-badge" style="font-size:12px; padding:4px 10px; border-radius:20px; background:rgba(100,116,139,0.2); color:var(--text-muted); font-family:var(--font-mono);">0 steps</span>
                <span id="collect-timer-badge" style="font-size:12px; padding:4px 10px; border-radius:20px; background:rgba(100,116,139,0.2); color:var(--text-muted); font-family:var(--font-mono);"></span>
              </div>
              <div id="collect-js-btn-caption" style="font-size:14px; font-weight:700; font-family:var(--font-mono); text-align:center; margin-top:10px; padding:8px; border-radius:8px; background:#090d16; border:1px solid var(--border-glow); color:var(--text-muted); transition:background 0.15s, border-color 0.15s, color 0.15s;">대기 중 — 버튼/D-pad를 누르면 여기 표시</div>
              <div id="collect-episode-status" style="font-size:12px; font-weight:600; color:var(--text-muted); margin-top:10px; padding:8px 10px; border-radius:6px; background:#090d16; border:1px solid var(--border-glow); min-height:16px;">—</div>
            </div>

            <div class="card" style="padding:16px;">
              <div class="card-title">🎯 실시간 cx (바구니 배치용)
                <span id="collect-cx-toggle-badge" style="font-size:10px; padding:2px 8px; border-radius:10px; background:rgba(100,116,139,0.2); color:var(--text-muted); cursor:pointer;" onclick="collectToggleCxFeed()">⏸ 꺼짐 — 클릭해서 시작</span>
              </div>
              <div id="collect-cx-value" style="font-size:32px; font-family:var(--font-mono); font-weight:700; text-align:center; padding:8px; min-height:44px; max-height:44px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;">—</div>
              <div id="collect-cx-band" style="font-size:11px; text-align:center; color:var(--text-muted);">극단 배치 기준: 0.10~0.15 강한좌 · 0.20~0.25 준극단좌 · 0.75~0.80 준극단우 · 0.85~0.90 강한우</div>
              <div style="font-size:10px; color:var(--text-muted); text-align:center; margin-top:4px;">📍 배치가이드(밴드+라벨)는 왼쪽 카메라 박스에서 "배치가이드 표시" 체크박스로 확인</div>
            </div>
          </div>

        </div>

        <!-- Row2 (1:1) — 트랙A 진행률 | 조이스틱 -->
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:20px; align-items:start; margin-bottom:20px;">

          <div class="card" style="padding:16px;">
            <div class="card-title">📏 트랙A 극단배치 진행률 (cx축 막대그래프)
              <span id="collect-mode-badge" style="font-size:10px; padding:2px 8px; border-radius:10px; background:rgba(56,189,248,0.15); color:var(--cyan);">🎮 D-pad 대상: 트랙A</span>
            </div>
            <div id="collect-cxpos-current" style="font-size:12px; font-weight:700; color:var(--emerald); margin-bottom:8px;">현재 선택: 미지정 + 미지정</div>
            <div style="font-size:10px; color:var(--text-muted); margin-bottom:8px;">
              위치×경로(좌곡선/직진/우곡선) 조합별로 각 15개씩 — 위치당 3×15=45개, 총 180개.
              <b>경로도 아래서 직접 선택</b>해야 실제로 15/15/15가 채워졌는지 추적됨 (안 고르면 "미지정"으로만 카운트).<br>
              D-pad ◀▶(또는 아래 ◀▶)= 현재 축 순환 · D-pad ▲▼(또는 ▲▼ 버튼)= 트랙A위치↔접근경로↔시나리오 전환
            </div>
            <div style="display:flex; gap:6px; margin-bottom:8px;">
              <button class="btn btn-outline" style="padding:6px 10px; font-size:13px;" onclick="_collectCycleCxPos(-1)" title="이전 cx위치 (D-pad 좌와 동일)">◀</button>
              <select id="collect-cxpos-select" style="flex:1; padding:6px 8px; background:#090d16; border:1px solid var(--border-glow); border-radius:6px; color:#fff; font-size:12px;" onchange="_collectSyncCxPosHighlight()">
                <option value="">— cx 위치 미지정 —</option>
              </select>
              <button class="btn btn-outline" style="padding:6px 10px; font-size:13px;" onclick="_collectCycleCxPos(1)" title="다음 cx위치 (D-pad 우와 동일)">▶</button>
              <button class="btn btn-outline" style="padding:6px 8px; font-size:13px;" onclick="_collectToggleMode(1)" title="다음 축 (D-pad 상과 동일)">▲</button>
              <button class="btn btn-outline" style="padding:6px 8px; font-size:13px;" onclick="_collectToggleMode(-1)" title="이전 축 (D-pad 하와 동일)">▼</button>
            </div>
            <div style="display:flex; gap:6px; margin-bottom:10px;">
              <button class="btn btn-outline" style="padding:6px 10px; font-size:13px;" onclick="_collectCycleCxPath(-1)" title="이전 접근경로">◀</button>
              <select id="collect-cxpath-select" style="flex:1; padding:6px 8px; background:#090d16; border:1px solid var(--border-glow); border-radius:6px; color:#fff; font-size:12px;" onchange="_collectSyncCxPathHighlight()">
                <option value="">— 접근경로 미지정 —</option>
              </select>
              <button class="btn btn-outline" style="padding:6px 10px; font-size:13px;" onclick="_collectCycleCxPath(1)" title="다음 접근경로">▶</button>
            </div>
            <div id="collect-cxpos-chart" style="display:flex; flex-direction:column; gap:10px; font-family:var(--font-mono); font-size:11px;">로딩 중...</div>
          </div>

          <div class="card" style="padding:16px;">
            <div class="card-title">🕹️ 조이스틱 (DragonRise) — 자동 기록됨
              <span id="collect-js-badge" style="font-size:10px; padding:2px 8px; border-radius:10px; background:rgba(100,116,139,0.2); color:var(--text-muted);">🔌 —</span>
            </div>
            <div style="font-size:10px; color:var(--text-muted); margin-bottom:6px;">버튼 라이트 (번호 + 물리 버튼 이름, DragonRise 기준)</div>
            <div id="collect-js-btn-lights" style="display:flex; flex-wrap:wrap; gap:10px; margin-bottom:10px;"></div>
            <div style="font-size:10px; color:var(--text-muted); margin-bottom:6px;">D-pad (시나리오 순환 전용)</div>
            <div style="display:flex; justify-content:center; gap:8px; margin-bottom:12px;">
              <div id="collect-dpad-left" class="btn-light">◀</div>
              <div id="collect-dpad-up" class="btn-light">▲</div>
              <div id="collect-dpad-down" class="btn-light">▼</div>
              <div id="collect-dpad-right" class="btn-light">▶</div>
            </div>
            <div style="font-size:10px; color:var(--text-muted); line-height:1.6; border-top:1px solid var(--border-glow); padding-top:8px;">
              <b>조작 설명서</b> (Gradio 대시보드와 동일 매핑)<br>
              왼쪽 스틱 → 이동(전/후/좌/우) &nbsp;|&nbsp; 오른쪽 스틱 X축 → 회전 (왼쪽 버튼패드에 라이트업)<br>
              <b>D-pad 좌/우</b> → 현재 활성 축(트랙A cx위치 / 접근경로 / 시나리오) 순환 선택 &nbsp;|&nbsp; <b>D-pad 상/하</b> → 3축 순환 전환 (녹화 중엔 변경 불가, L1/SEL로 시작 시 선택된 값으로 태깅됨)<br>
              <table style="width:100%; border-collapse:collapse; margin-top:6px;">
                <tr><td style="padding:1px 4px;"><b>A</b>(0)</td><td>STOP</td>
                    <td style="padding:1px 4px;"><b>B</b>(1)</td><td>마지막 프레임 취소</td></tr>
                <tr><td style="padding:1px 4px;"><b>X</b>(2)</td><td>에피소드 폐기</td>
                    <td style="padding:1px 4px;"><b>Y</b>(3)</td><td style="color:var(--amber);">🔁 모드 전환(수집⇄검증)</td></tr>
                <tr><td style="padding:1px 4px;"><b>L1</b>(4)</td><td>녹화 시작</td>
                    <td style="padding:1px 4px;"><b>R1</b>(5)</td><td>정지 & 저장</td></tr>
                <tr><td style="padding:1px 4px;"><b>SEL</b>(6)</td><td>녹화 토글</td>
                    <td style="padding:1px 4px;"><b>START</b>(7)</td><td>SYNC↔ASYNC 모드</td></tr>
                <tr><td style="padding:1px 4px; color:var(--amber);"><b>R2</b>(트리거)</td><td colspan="3" style="color:var(--amber);">🔄 복귀 — 직전 경로 역주행 (Gradio 이식, 다시 당기면 중지)</td></tr>
              </table>
              ⚠️ 대각선 후진(Z/C)은 조이스틱 축으로는 안 나옴 — 버튼패드/키보드로만 가능
            </div>
          </div>

        </div>

        <!-- Row3 (1:1:1) — 키보드 조작 | 시나리오 & 진행률 | 데이터수집(에피소드제어+타임라인+세션요약+최근저장) -->
        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:20px; align-items:start;">

          <div class="card" style="padding:16px;">
            <div class="card-title">🕹️ 조작 (탭 클릭 후 키보드 W A S D Q E Z C R T Space)</div>
            <div id="collect-key-surface" tabindex="0"
                 style="outline:none; border:2px dashed var(--border-glow); border-radius:10px; padding:24px; text-align:center; background:#090d16; cursor:pointer;">
              <div style="font-size:13px; color:var(--text-muted); margin-bottom:8px;">여기 클릭해서 포커스 → 키보드로 조작 (조이스틱은 그대로 사용 가능, 자동 기록됨)</div>
            </div>
            <div style="margin-top:16px; display:flex; justify-content:center;">
              <div class="joystick-grid">
                <button id="collect-pad-q" class="joy-btn" onpointerdown="_collectPadDown('q')" onpointerup="_collectPadUp('q')" onpointerleave="_collectPadUp('q')">↖Q</button>
                <button id="collect-pad-w" class="joy-btn" onpointerdown="_collectPadDown('w')" onpointerup="_collectPadUp('w')" onpointerleave="_collectPadUp('w')">▲W</button>
                <button id="collect-pad-e" class="joy-btn" onpointerdown="_collectPadDown('e')" onpointerup="_collectPadUp('e')" onpointerleave="_collectPadUp('e')">↗E</button>
                <button id="collect-pad-a" class="joy-btn" onpointerdown="_collectPadDown('a')" onpointerup="_collectPadUp('a')" onpointerleave="_collectPadUp('a')">◀A</button>
                <button class="joy-btn stop" onclick="_collectPadStop()">⏹</button>
                <button id="collect-pad-d" class="joy-btn" onpointerdown="_collectPadDown('d')" onpointerup="_collectPadUp('d')" onpointerleave="_collectPadUp('d')">▶D</button>
                <button id="collect-pad-z" class="joy-btn" onpointerdown="_collectPadDown('z')" onpointerup="_collectPadUp('z')" onpointerleave="_collectPadUp('z')">↙Z</button>
                <button id="collect-pad-s" class="joy-btn" onpointerdown="_collectPadDown('s')" onpointerup="_collectPadUp('s')" onpointerleave="_collectPadUp('s')">▼S</button>
                <button id="collect-pad-c" class="joy-btn" onpointerdown="_collectPadDown('c')" onpointerup="_collectPadUp('c')" onpointerleave="_collectPadUp('c')">↘C</button>
                <button id="collect-pad-r" class="joy-btn" onpointerdown="_collectPadDown('r')" onpointerup="_collectPadUp('r')" onpointerleave="_collectPadUp('r')">↺R</button>
                <div></div>
                <button id="collect-pad-t" class="joy-btn" onpointerdown="_collectPadDown('t')" onpointerup="_collectPadUp('t')" onpointerleave="_collectPadUp('t')">↻T</button>
              </div>
            </div>
            <div id="collect-pad-caption" style="font-size:13px; font-family:var(--font-mono); text-align:center; margin-top:10px; padding:6px; border-radius:6px; background:#090d16; border:1px solid var(--border-glow); color:var(--text-muted);">대기 중 — 버튼/키보드를 누르면 여기 표시</div>
            <div style="font-size:10px; color:var(--text-muted); text-align:center; margin-top:6px;">버튼을 누르고 있으면 계속 이동, 떼면 정지 (키보드와 동일하게 기록됨)</div>
          </div>

          <div class="card" style="padding:16px;">
            <div class="card-title">🎯 시나리오 & 진행률 (행 클릭 또는 조이스틱 D-pad로 선택)
              <span id="collect-scenario-dpad-badge" style="display:none; font-size:10px; padding:2px 8px; border-radius:10px; background:rgba(56,189,248,0.15); color:var(--cyan);">🕹️ D-pad 선택됨</span>
            </div>
            <div id="collect-scenario-row" style="display:flex; gap:6px; margin-bottom:8px; padding:4px;">
              <button class="btn btn-outline" style="padding:6px 10px; font-size:13px;" onclick="_collectCycleScenario(-1)" title="이전 시나리오 (D-pad 좌와 동일)">◀</button>
              <select id="collect-scenario-select" style="flex:1; padding:6px 8px; background:#090d16; border:1px solid var(--border-glow); border-radius:6px; color:#fff; font-size:12px;" onchange="_collectSyncScenarioHighlight()">
                <option value="">— 미지정 (episode_name 수동) —</option>
              </select>
              <button class="btn btn-outline" style="padding:6px 10px; font-size:13px;" onclick="_collectCycleScenario(1)" title="다음 시나리오 (D-pad 우와 동일)">▶</button>
            </div>
            <select id="collect-pattern-select" style="width:100%; padding:6px 8px; background:#090d16; border:1px solid var(--border-glow); border-radius:6px; color:#fff; font-size:12px; margin-bottom:10px;">
              <option value="">— 패턴 미지정 —</option>
              <option value="core">핵심 패턴 (Core)</option>
              <option value="variant">변형 패턴 (Variant)</option>
            </select>
            <div id="collect-progress-list" style="display:flex; flex-direction:column; gap:4px; font-size:11px; font-family:var(--font-mono);">로딩 중...</div>
          </div>

          <div style="display:flex; flex-direction:column; gap:16px;">
            <div class="card" style="padding:16px;">
              <div class="card-title">📼 에피소드 제어</div>
              <input type="text" id="collect-episode-name" placeholder="episode_name (비우면 자동 생성)"
                     style="width:100%; padding:6px 8px; background:#090d16; border:1px solid var(--border-glow); border-radius:6px; color:#fff; font-size:12px; margin-bottom:10px;">
              <div style="display:flex; gap:8px;">
                <button class="btn btn-cyan" style="flex:1;" onclick="collectStartEpisode()">▶ 시작</button>
                <button class="btn btn-outline" style="flex:1; border-color:var(--rose); color:var(--rose);" onclick="collectStopEpisode()">⏹ 정지 & 저장</button>
              </div>
              <button id="collect-return-btn" class="btn btn-outline" style="width:100%; margin-top:8px; border-color:var(--amber); color:var(--amber);" onclick="collectAutoReturn()">🔄 복귀 (직전 경로 역주행)</button>
              <div style="font-size:10px; color:var(--text-muted); text-align:center; margin-top:8px;">진행 상태는 상단 "🎮 입력 현황"에 표시됩니다</div>
            </div>

            <div class="card" style="padding:16px;">
              <div class="card-title">📊 현재 에피소드 타임라인 (Gradio 이식 — 기호 시퀀스 + 실측 Hz)</div>
              <div id="collect-timeline-box" style="font-family:var(--font-mono); font-size:11px; color:var(--text-muted);">대기 중 — 시나리오 선택 후 녹화 시작</div>
              <div id="collect-dist-box" style="font-family:var(--font-mono); font-size:11px; color:var(--text-muted); margin-top:8px; white-space:pre;"></div>
            </div>

            <div class="card" style="padding:16px;">
              <div class="card-title">⏱️ 마지막 세션 (Hz 설계용)</div>
              <div id="collect-session-summary" style="font-family:var(--font-mono); font-size:11px; color:var(--text-muted); white-space:pre;">도착 시 정지 → 저장하면 소요 초·Hz가 여기 표시됩니다</div>
            </div>

            <div class="card" style="padding:16px;">
              <div class="card-title">🗂️ 최근 저장 내역 (녹화 성공/폐기 이력)</div>
              <div id="collect-recent-saves" style="display:flex; flex-direction:column; gap:6px; max-height:220px; overflow-y:auto;">
                <div style="font-size:11px; color:var(--text-muted); padding:6px 0;">아직 없음</div>
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>

    <!-- 탭: 🗂 데이터셋 히스토리 — 데이터수집 탭이 저장한 원본 학습 H5 브라우징
         (세션 히스토리 tab-history와 같은 목록+프레임인스펙터 UX, 다른 데이터 소스) -->
    <div id="tab-dataset" class="tab-content">
      <div class="scroll-container">
        <div class="grid-main" style="grid-template-columns: 300px 1fr;">

          <!-- 목록 + 필터 -->
          <div class="card" style="padding:16px; overflow-y:auto; max-height:calc(100vh - 150px);">
            <div class="card-title">🗂 저장된 에피소드
              <span id="ds-count-badge" style="font-size:10px; font-weight:400; color:var(--text-muted); float:right;">0개</span>
            </div>
            <button class="btn btn-outline" style="width:100%; margin-bottom:10px; font-size:12px;" onclick="loadDatasetList()">🔄 리스트 새로고침</button>

            <input type="text" id="ds-search" placeholder="이름 검색..." oninput="renderDatasetList()"
                   style="width:100%; padding:6px 8px; background:#090d16; border:1px solid var(--border-glow); border-radius:6px; color:#fff; font-size:12px; margin-bottom:8px;">

            <!-- 스키마 구분(레거시 vs 신규) — 버튼클릭 필터 -->
            <div style="font-size:9px; color:var(--text-muted); text-transform:uppercase; font-weight:700; margin-bottom:4px;">스키마</div>
            <div id="ds-filter-schema" style="display:flex; gap:4px; margin-bottom:10px;">
              <button type="button" class="btn btn-outline ds-schema-btn active" data-schema="" onclick="_dsSetSchemaFilter('')" style="flex:1; font-size:14px; font-weight:700; padding:6px 0;">전체</button>
              <button type="button" class="btn btn-outline ds-schema-btn" data-schema="legacy" onclick="_dsSetSchemaFilter('legacy')" style="flex:1; font-size:14px; font-weight:700; padding:6px 0;">V5</button>
              <button type="button" class="btn btn-outline ds-schema-btn" data-schema="new" onclick="_dsSetSchemaFilter('new')" style="flex:1; font-size:14px; font-weight:700; padding:6px 0;">V6</button>
            </div>

            <!-- 시나리오 — 버튼클릭 필터 -->
            <div style="font-size:9px; color:var(--text-muted); text-transform:uppercase; font-weight:700; margin-bottom:4px;">시나리오</div>
            <div id="ds-filter-scenario" style="display:flex; flex-wrap:wrap; gap:4px; margin-bottom:10px;"></div>

            <!-- 트랙A cx위치 — 버튼클릭 필터 -->
            <div style="font-size:9px; color:var(--text-muted); text-transform:uppercase; font-weight:700; margin-bottom:4px;">트랙A 위치</div>
            <div id="ds-filter-cxpos" style="display:flex; flex-wrap:wrap; gap:4px; margin-bottom:10px;"></div>

            <button type="button" id="ds-compare-toggle-btn" class="btn btn-outline" style="width:100%; margin-bottom:10px; font-size:12px;" onclick="_dsToggleCompareMode()">☑️ 다중 선택(비교 모드) 켜기 — 최대 6개</button>

            <div id="ds-list-group" style="display:flex; flex-direction:column; gap:8px;"></div>
          </div>

          <!-- 상세 — 단일 프레임인스펙터 또는 다중 비교 -->
          <div class="card">
            <div class="card-title">🗂 데이터셋 상세 <span id="ds-detail-lbl" class="text-cyan" style="font-size:13px; text-transform:none;"></span></div>

            <div id="ds-placeholder" style="text-align:center; padding:80px 0; color:var(--text-muted);">
              왼쪽 목록에서 에피소드를 선택하면 상세 정보가 표시됩니다. 다중 선택(비교 모드)을
              켜고 2개 이상 체크하면 요약 비교 카드로 전환됩니다.
            </div>

            <!-- 비교 모드: 선택된 항목들의 요약 카드 나열 -->
            <div id="ds-compare-body" style="display:none; padding:16px; flex-direction:column; gap:10px;"></div>

            <!-- 단일 상세: 프레임 인스펙터 -->
            <div id="ds-inspector-body" class="frame-inspector" style="display:none;">
              <div>
                <div class="viewport-wrapper" style="background:#000;">
                  <img id="ds-frame-img" class="viewport-img" src="">
                </div>
                <div style="margin-top:16px;">
                  <input type="range" id="ds-slider" min="0" max="0" value="0" oninput="showDsFrame(this.value)">
                  <div style="display:flex; justify-content:space-between; font-size:12px; color:var(--text-muted); margin-top:4px;">
                    <span id="ds-frame-idx-lbl">Frame: 0 / 0</span>
                    <span id="ds-frame-action-lbl">—</span>
                  </div>
                </div>
                <div id="ds-timeline" style="display:flex; gap:1px; margin-top:10px; height:20px; border-radius:4px; overflow:hidden; border:1px solid var(--border-glow);"></div>
                <div style="display:flex; gap:8px; margin-top:12px; justify-content:center;">
                  <button class="btn btn-outline" onclick="dsPrevFrame()">◀ 이전</button>
                  <button class="btn btn-outline" id="btn-ds-play" onclick="toggleDsPlay()">▶ PLAY</button>
                  <button class="btn btn-outline" onclick="dsNextFrame()">다음 ▶</button>
                  <button class="btn btn-outline" onclick="if (_dsSelected.size >= 2) { renderDatasetCompare(); } else { setDatasetCompareMode(false); document.getElementById('ds-placeholder').style.display='block'; }">← 목록으로</button>
                  <button class="btn btn-outline" style="border-color:var(--rose); color:var(--rose);" onclick="if (_dsDetail) _dsDeleteOne(_dsDetail.meta.name)">🗑️ 이 에피소드 삭제</button>
                </div>

                <div style="background:#151f32; border:1px solid var(--border-glow); border-radius:10px; padding:12px 14px; margin-top:16px;">
                  <div style="font-size:11px; color:var(--text-muted); font-weight:700; text-transform:uppercase; margin-bottom:8px;">📊 에피소드 요약</div>
                  <div class="mini-tile-grid">
                    <div class="mini-tile"><span class="mt-label">프레임</span><span class="mt-value" id="ds-sum-frames">—</span></div>
                    <div class="mini-tile"><span class="mt-label">소요시간</span><span class="mt-value" id="ds-sum-duration">—</span></div>
                    <div class="mini-tile"><span class="mt-label">시나리오</span><span class="mt-value" id="ds-sum-scenario" style="font-size:15px;">—</span></div>
                    <div class="mini-tile"><span class="mt-label">트랙A</span><span class="mt-value" id="ds-sum-cxpos" style="font-size:13px;">—</span></div>
                    <div class="mini-tile"><span class="mt-label">스키마</span><span class="mt-value" id="ds-sum-schema" style="font-size:15px; font-weight:700;">—</span></div>
                    <div class="mini-tile"><span class="mt-label">날짜</span><span class="mt-value" id="ds-sum-date" style="font-size:11px;">—</span></div>
                  </div>

                  <!-- 수집 관련 상세 정보 — 데이터수집 탭 attrs 그대로 노출 -->
                  <div class="mt-label" style="font-size:9px; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.04em; font-weight:700; margin:12px 0 5px;">🎛️ 수집 설정</div>
                  <div class="mini-tile-grid">
                    <div class="mini-tile"><span class="mt-label">패턴</span><span class="mt-value" id="ds-sum-pattern" style="font-size:11px;">—</span></div>
                    <div class="mini-tile"><span class="mt-label">장애물배치</span><span class="mt-value" id="ds-sum-obstacle" style="font-size:11px;">—</span></div>
                    <div class="mini-tile"><span class="mt-label">시간대</span><span class="mt-value" id="ds-sum-timeperiod" style="font-size:11px;">—</span></div>
                    <div class="mini-tile"><span class="mt-label">STOP 주입</span><span class="mt-value" id="ds-sum-stopinject">—</span></div>
                    <div class="mini-tile"><span class="mt-label">액션청크</span><span class="mt-value" id="ds-sum-chunk">—</span></div>
                    <div class="mini-tile"><span class="mt-label">파일크기</span><span class="mt-value" id="ds-sum-size">—</span></div>
                  </div>

                  <div style="margin-top:10px;">
                    <div class="mt-label" style="font-size:9px; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.04em; font-weight:700; margin-bottom:5px;">🕹️ 입력 소스</div>
                    <div id="ds-sum-sources" class="mini-tile-grid"></div>
                  </div>

                  <div style="margin-top:10px;">
                    <div class="mt-label" style="font-size:9px; color:var(--text-muted); text-transform:uppercase; letter-spacing:0.04em; font-weight:700; margin-bottom:5px;">액션 분포</div>
                    <div id="ds-sum-actions" class="mini-tile-grid"></div>
                  </div>
                </div>
              </div>
            </div>
          </div>

        </div>
      </div>
    </div>

    <!-- 탭 10: 📖 위키 (정적 참조 정보) -->
    <div id="tab-wikiinfo" class="tab-content">
      <div class="scroll-container" style="padding:20px;">
        <div class="card" style="padding:16px;">
          <div class="card-title">📖 프로젝트 위키 (핵심 요약)</div>
          <div style="font-size:11px; color:var(--text-muted); margin-bottom:8px;">
            docs/DASHBOARD_WIKI.md — 거의 안 바뀌는 참조 정보. CLAUDE.md 바뀔 때 수동 갱신.
          </div>
          <div id="wiki-info-content" class="wiki-render" style="background:#090d16; border:1px solid var(--border-glow); border-radius:8px; padding:20px 24px;">로딩 중...</div>
        </div>
        <div class="card" style="padding:16px; margin-top:16px;">
          <div class="card-title">🗓️ 연구일지 (git 커밋 기준 — 클릭해서 상세보기)</div>
          <div class="wj-grid">
            <div id="wiki-info-journal-list" class="wj-list">로딩 중...</div>
            <div id="wiki-info-journal-detail" class="wj-detail"><div class="wjd-placeholder">← 항목을 클릭하면 커밋 상세가 여기 표시됩니다</div></div>
          </div>
        </div>
      </div>
    </div>

    <!-- 탭 11: 📡 최신현황 (스킬로 갱신) -->
    <div id="tab-wikistatus" class="tab-content">
      <div class="scroll-container" style="padding:20px;">
        <div class="card" style="padding:16px;">
          <div class="card-title">📡 최신현황
            <button class="btn btn-outline" onclick="loadWikiContent('status')" style="font-size:11px; padding:4px 10px;">↻ 새로고침</button>
          </div>
          <div style="font-size:11px; color:var(--text-muted); margin-bottom:8px;">
            docs/DASHBOARD_LIVE_STATUS.md — <span id="wiki-status-mtime" style="color:var(--amber); font-weight:700; font-family:var(--font-mono);">-</span>.
            실시간 자동 갱신 아님 — "대시보드 최신현황 갱신해줘" 요청 시 스킬이 이 파일을 다시 씀.
          </div>
          <div id="wiki-status-content" class="wiki-render" style="background:#090d16; border:1px solid var(--border-glow); border-radius:8px; padding:20px 24px;">로딩 중...</div>
        </div>
        <div class="card" style="padding:16px; margin-top:16px;">
          <div class="card-title">🗓️ 연구일지 (git 커밋 기준 — 클릭해서 상세보기)</div>
          <div class="wj-grid">
            <div id="wiki-status-journal-list" class="wj-list">로딩 중...</div>
            <div id="wiki-status-journal-detail" class="wj-detail"><div class="wjd-placeholder">← 항목을 클릭하면 커밋 상세가 여기 표시됩니다</div></div>
          </div>
        </div>
      </div>
    </div>

  </main>

  <!-- JS 컨트롤 스크립트 -->
  <script>
    const API = "";
    let activeTab = "drive";
    let state = {};
    let _prevRunning = false;
    let driftChartInstance = null;
    let trajChartInstance = null;

    // H5 세션 검사 상태
    let inspectSession = null;
    let inspectPlayTimer = null;

    // 키보드 단축키 매핑
    const joyKeys = {
      "ArrowUp": "W", "w": "W", "W": "W",
      "ArrowDown": "S", "s": "S", "S": "S",
      "ArrowLeft": "A", "a": "A", "A": "A",
      "ArrowRight": "D", "d": "D", "D": "D",
      "q": "Q", "Q": "Q",
      "e": "E", "E": "E",
      "r": "R", "R": "R",
      "t": "T", "T": "T",
      " ": "STOP", "Escape": "STOP"
    };

    window.addEventListener("keydown", (e) => {
      // 입력 폼에 포커스가 가있으면 단축키 동작 방지
      if (document.activeElement.tagName === "INPUT" || document.activeElement.tagName === "TEXTAREA") return;

      // 세션 히스토리(history) 탭일 때 프레임 인스펙터 재생 단축키 바인딩
      if (activeTab === "history") {
        if (e.key === "ArrowLeft") {
          e.preventDefault();
          prevInspectFrame();
        } else if (e.key === "ArrowRight") {
          e.preventDefault();
          nextInspectFrame();
        } else if (e.key === " ") {
          e.preventDefault();
          toggleInspectPlay();
        }
      }

      // 데이터셋 히스토리(dataset) 탭 — 세션 히스토리와 동일한 좌우/스페이스 단축키
      if (activeTab === "dataset" && _dsDetail) {
        if (e.key === "ArrowLeft") {
          e.preventDefault();
          dsPrevFrame();
        } else if (e.key === "ArrowRight") {
          e.preventDefault();
          dsNextFrame();
        } else if (e.key === " ") {
          e.preventDefault();
          toggleDsPlay();
        }
      }

      if (activeTab === "calib" && joyKeys[e.key]) {
        e.preventDefault();
        const dir = joyKeys[e.key];
        const btn = document.getElementById("joy-" + dir);
        if (btn) btn.classList.add("active");
        sendJoy(dir);
      }
    });

    window.addEventListener("keyup", (e) => {
      if (activeTab === "calib" && joyKeys[e.key]) {
        const dir = joyKeys[e.key];
        const btn = document.getElementById("joy-" + dir);
        if (btn) btn.classList.remove("active");
        if (dir !== "STOP") {
          sendJoy("STOP");
        }
      }
    });

    function setSafeText(id, text) {
      const el = document.getElementById(id);
      if (el) el.textContent = text;
    }
    function setSafeHtml(id, html) {
      const el = document.getElementById(id);
      if (el) el.innerHTML = html;
    }
    function setSafeValue(id, val) {
      const el = document.getElementById(id);
      if (el) el.value = val;
    }

    async function api(path, opts={}) {
      const r = await fetch(API + path, opts);
      return r.json();
    }

    // ── 카메라 프로세스(usb_camera_service_server) 제어 — 모든 탭 공용 ──
    function _setCamProcText(text) {
      document.querySelectorAll(".cam-proc-status").forEach(el => el.textContent = text);
    }

    async function camProcRefresh() {
      _setCamProcText("확인 중...");
      const res = await api("/camera_proc/status");
      _setCamProcText(res.text || "—");
    }

    async function camProcStart() {
      _setCamProcText("⏳ 시작 중...");
      const res = await api("/camera_proc/start", { method: "POST" });
      _setCamProcText(res.text || "—");
      setTimeout(camProcRefresh, 3000);
    }

    async function camProcStop() {
      if (!confirm("카메라 프로세스를 정지하시겠습니까? (스트림이 끊깁니다)")) return;
      _setCamProcText("⏳ 정지 중...");
      const res = await api("/camera_proc/stop", { method: "POST" });
      _setCamProcText(res.text || "—");
    }

    // ── 🕹️ 게임패드(DragonRise) 제어 — Gradio 대시보드에서 이식 ──
    const _collectPadIds = ["q","w","e","a","d","z","s","c","r","t"];
    async function joystickRefresh() {
      const s = await api("/joystick/status");

      const el = document.getElementById("js-status");
      const btn = document.getElementById("js-toggle-btn");
      if (el && btn) {
        if (!s.pygame_available) {
          el.textContent = "⚠️ pygame 미설치 — 게임패드 사용 불가";
        } else if (!s.connected) {
          el.textContent = "🔌 미연결 (DragonRise 꽂으면 자동 인식)";
        } else {
          const en = s.enabled ? "🟢 ON" : "⚫ OFF";
          const badge = s.mode === "SYNC" ? "📸 SYNC" : "🌊 ASYNC";
          const keyStr = s.key ? `[ ${s.label} ]` : "○ 중립";
          el.textContent = `${en}  |  ${badge}  |  ${s.name}\n▶ ${keyStr}`;
        }
        btn.textContent = s.enabled ? "비활성화" : "활성화";
        btn.className = s.enabled ? "btn btn-cyan" : "btn btn-outline";
      }

      const cjBadge = document.getElementById("collect-js-badge");
      if (cjBadge) {
        if (!s.pygame_available) {
          cjBadge.textContent = "⚠️ pygame 없음";
        } else if (!s.connected) {
          cjBadge.textContent = "🔌 미연결";
        } else {
          const badge = s.mode === "SYNC" ? "📸 SYNC" : "🌊 ASYNC";
          cjBadge.textContent = `🟢 ${s.name} · ${badge}`;
          cjBadge.style.background = "rgba(16,185,129,0.15)";
          cjBadge.style.color = "var(--emerald)";
        }
      }
      // Y 버튼(조이스틱)으로 서버측 모드가 바뀌면 UI 토글 버튼도 동기화
      if (s.verify_mode !== undefined && s.verify_mode !== joystickVerifyMode) {
        joystickVerifyMode = s.verify_mode;
        document.querySelectorAll(".joystick-mode-btn").forEach(btn => {
          btn.textContent = joystickVerifyMode ? "🕹️ 조이스틱: 🧪 검증" : "🕹️ 조이스틱: 📷 수집";
          btn.style.borderColor = joystickVerifyMode ? "var(--amber)" : "";
          btn.style.color = joystickVerifyMode ? "var(--amber)" : "";
        });
      }

      // 검증 스크리닝 현재 선택 위치 + 대기 라벨 표시
      const vfyCur = document.getElementById("vfy-screen-current");
      if (vfyCur) {
        const POS_LABEL = {strong_left:"강좌◀◀", weak_left:"약좌◀", center:"중앙●", weak_right:"약우▶", strong_right:"강우▶▶"};
        const pos = POS_LABEL[s.verify_screen_pos] || (s.verify_screen_pos || "—");
        const res = s.verify_pending_result;
        const resHtml = res
          ? `<span style="color:${res === '성공' ? 'var(--emerald)' : 'var(--rose)'}">${res}</span> (L2로 저장)`
          : `<span style="color:var(--text-muted)">— (X성공/A실패)</span>`;
        vfyCur.innerHTML = `현재 위치: <span style="color:var(--amber)">${pos}</span> · 대기 라벨: ${resHtml}`;
        vfyCur.style.borderColor = res ? "var(--amber)" : "var(--border-glow)";
        const expBadge = document.getElementById("vfy-experimental-badge");
        if (expBadge) expBadge.style.display = s.verify_experimental ? "block" : "none";
        // A안 — 검증모드면 수동 폼(드롭다운/성공·실패 버튼)이 조이스틱 상태를 미러링
        if (s.verify_mode) {
          // D-pad 위치 → 드롭다운 동기화(같은 위치의 첫 옵션 선택) + 오버레이 강조 갱신
          if (s.verify_screen_pos && s.verify_screen_pos !== window._verifyScreenPos) {
            window._verifyScreenPos = s.verify_screen_pos;
            const sel = document.getElementById("ep-path-type");
            if (sel) {
              const targetVal = (s.verify_screen_pos === "center") ? "trackF_center" : ("trackA_" + s.verify_screen_pos);
              sel.value = targetVal;
              syncVerifyPathType(sel.value);
            }
            if (typeof drawOverlay === "function") drawOverlay();
          }
          // X/A 라벨 → ✅/❌ 버튼 하이라이트 미러
          if (s.verify_pending_result && s.verify_pending_result !== window._verifyPendingShown) {
            window._verifyPendingShown = s.verify_pending_result;
            if (typeof setEpResult === "function") setEpResult(s.verify_pending_result);
          } else if (!s.verify_pending_result) {
            window._verifyPendingShown = null;
          }
        }
      }

      // L2(조이스틱)로 저장될 때마다 서버 verify_save_seq가 +1 됨 — 값이 바뀌면
      // 팝업/알림 없이 조용히 loadEpisodeHistory() 호출해 스크리닝 패널 즉시 갱신.
      if (s.verify_save_seq !== undefined) {
        if (window._verifySaveSeq === undefined) window._verifySaveSeq = s.verify_save_seq;
        else if (s.verify_save_seq !== window._verifySaveSeq) {
          window._verifySaveSeq = s.verify_save_seq;
          if (typeof loadEpisodeHistory === "function") loadEpisodeHistory();
        }
      }

      // 🕹️ 검증 탭 조이스틱 가이드(컴팩트) — 같은 /joystick/status 재사용, 수집 패널과 독립(ID 다름)
      const vfyBadge = document.getElementById("vfy-js-badge");
      if (vfyBadge) {
        if (!s.pygame_available) vfyBadge.textContent = "⚠️ pygame 없음";
        else if (!s.connected) vfyBadge.textContent = "🔌 미연결";
        else {
          const md = s.verify_mode ? "🧪검증" : "📷수집";
          vfyBadge.textContent = `🟢 ${s.name} · ${md}`;
          vfyBadge.style.background = "rgba(16,185,129,0.15)";
          vfyBadge.style.color = "var(--emerald)";
        }
      }
      const vfyLights = document.getElementById("vfy-js-btn-lights");
      if (vfyLights) {
        const btnInfo2 = (i) => (s.btn_map && s.btn_map[i]) || JOYSTICK_BTN_INFO[i] || {name: "#" + i};
        const pressed2 = new Set(s.buttons || []);
        const n2 = Math.max(10, ...(s.buttons || []).map(i => i + 1));
        if (vfyLights.children.length !== n2) {
          vfyLights.innerHTML = "";
          for (let i = 0; i < n2; i++) {
            const wrap = document.createElement("div"); wrap.className = "btn-light-wrap";
            const d = document.createElement("div"); d.className = "btn-light"; d.id = "vfy-js-light-" + i; d.textContent = i;
            const nm = document.createElement("div"); nm.className = "btn-light-name"; nm.id = "vfy-js-name-" + i; nm.textContent = btnInfo2(i).name;
            wrap.appendChild(d); wrap.appendChild(nm); vfyLights.appendChild(wrap);
          }
        }
        let vfyActive = null;
        for (let i = 0; i < n2; i++) {
          const d = document.getElementById("vfy-js-light-" + i);
          const nm = document.getElementById("vfy-js-name-" + i);
          if (!d) continue;
          const isP = pressed2.has(i);
          d.classList.toggle("active", isP);
          d.classList.toggle("last", s.last_btn === i);
          if (nm) { nm.classList.toggle("active", isP); nm.textContent = btnInfo2(i).name; }
          if (isP) vfyActive = btnInfo2(i);
        }
        // 지금 눌린 버튼을 크게 표시 — 현재 모드(검증/수집)에 맞는 의미로
        const vfyCap = document.getElementById("vfy-js-caption");
        if (vfyCap) {
          if (vfyActive) {
            const meaning = s.verify_mode
              ? (VERIFY_BTN_MEANING[vfyActive.name] || vfyActive.desc || "—")
              : (vfyActive.desc || "—");
            vfyCap.textContent = `${vfyActive.name} — ${meaning}`;
            vfyCap.style.color = "var(--emerald)";
            vfyCap.style.borderColor = "var(--emerald)";
          } else {
            vfyCap.textContent = "대기 중 — 버튼/D-pad를 누르면 여기 크게 표시";
            vfyCap.style.color = "var(--text-muted)";
            vfyCap.style.borderColor = "var(--border-glow)";
          }
        }
      }

      const lights = document.getElementById("collect-js-btn-lights");
      if (lights) {
        // 실제 감지된 매핑(btn_map)이 있으면 우선 사용 — 하드코딩 라벨은 연결 전 폴백일 뿐,
        // 실측 매핑과 어긋나면 (예: 물리 R1이 START로 표시) 혼동을 주므로 우선순위를 둠.
        const btnInfo = (i) => (s.btn_map && s.btn_map[i]) || JOYSTICK_BTN_INFO[i] || {name: "#" + i, desc: "미사용"};
        const pressed = new Set(s.buttons || []);
        const n = Math.max(10, ...(s.buttons || []).map(i => i + 1));
        if (lights.children.length !== n) {
          lights.innerHTML = "";
          for (let i = 0; i < n; i++) {
            const wrap = document.createElement("div");
            wrap.className = "btn-light-wrap";
            const d = document.createElement("div");
            d.className = "btn-light";
            d.id = "collect-js-light-" + i;
            d.textContent = i;
            const nameEl = document.createElement("div");
            nameEl.className = "btn-light-name";
            nameEl.id = "collect-js-name-" + i;
            nameEl.textContent = btnInfo(i).name;
            wrap.appendChild(d);
            wrap.appendChild(nameEl);
            lights.appendChild(wrap);
          }
        }
        const cap = document.getElementById("collect-js-btn-caption");
        let activeInfo = null;
        for (let i = 0; i < n; i++) {
          const d = document.getElementById("collect-js-light-" + i);
          const nameEl = document.getElementById("collect-js-name-" + i);
          if (!d) continue;
          const isPressed = pressed.has(i);
          d.classList.toggle("active", isPressed);
          d.classList.toggle("last", s.last_btn === i);
          if (nameEl) {
            nameEl.classList.toggle("active", isPressed);
            nameEl.textContent = btnInfo(i).name;  // 매핑이 나중에 도착해도 라벨 갱신
          }
          if (isPressed) activeInfo = btnInfo(i);
        }

        // D-pad 방향 라이트 + 캡션 — 버튼과 동일한 자리에 크게 표시
        const hat = s.hat || [0, 0];
        const dpadState = {
          left:  hat[0] < 0, right: hat[0] > 0,
          up:    hat[1] > 0, down:  hat[1] < 0,
        };
        const DPAD_LABEL = {left: "◀ D-pad LEFT", right: "▶ D-pad RIGHT", up: "▲ D-pad UP", down: "▼ D-pad DOWN"};
        let dpadInfo = null;
        for (const dir of ["left", "up", "down", "right"]) {
          const el = document.getElementById("collect-dpad-" + dir);
          if (!el) continue;
          const isHeld = dpadState[dir];
          el.classList.toggle("active", isHeld);
          el.classList.toggle("last", s.last_hat_dir === dir && !isHeld);
          if (isHeld) dpadInfo = { name: DPAD_LABEL[dir], desc: "시나리오 순환 선택 중" };
        }
        // 방금 뗀 D-pad 방향도 짧게 표시(순간 탭이라 held로 못 잡을 때 대비)
        if (!dpadInfo && !activeInfo && s.last_hat_dir && s.last_hat_dir !== _collectLastHatDirShown) {
          dpadInfo = { name: DPAD_LABEL[s.last_hat_dir], desc: "시나리오 순환 선택됨" };
        }
        if (s.last_hat_dir !== _collectLastHatDirShown) _collectLastHatDirShown = s.last_hat_dir;

        const shown = activeInfo || dpadInfo;
        if (cap) {
          if (shown) {
            cap.textContent = `${shown.name} — ${shown.desc}`;
            cap.style.color = "var(--emerald)";
            cap.style.borderColor = "var(--emerald)";
          } else {
            cap.textContent = "대기 중 — 버튼/D-pad를 누르면 여기 크게 표시";
            cap.style.color = "var(--text-muted)";
            cap.style.borderColor = "var(--border-glow)";
          }
        }
      }

      // 조이스틱으로 실제 이동 중인 방향키를 버튼패드에도 라이트업 (수동/키보드 입력 중이 아닐 때만)
      const activeKey = (s.connected && s.key) ? s.key.toLowerCase() : null;
      if (!_collectPressedKey) {
        for (const k of _collectPadIds) {
          const padBtn = document.getElementById("collect-pad-" + k);
          if (padBtn) padBtn.classList.toggle("active", k === activeKey);
        }
        const cap = document.getElementById("collect-pad-caption");
        if (cap) {
          if (activeKey) {
            cap.textContent = "🕹️ " + (COLLECT_DESC[activeKey] || activeKey);
            cap.style.color = "var(--emerald)";
            cap.style.borderColor = "var(--emerald)";
          } else {
            cap.textContent = "대기 중 — 버튼/키보드를 누르면 여기 표시";
            cap.style.color = "var(--text-muted)";
            cap.style.borderColor = "var(--border-glow)";
          }
        }
      }
    }

    async function joystickToggle() {
      await api("/joystick/toggle", { method: "POST" });
      joystickRefresh();
    }

    async function joystickMode() {
      await api("/joystick/mode", { method: "POST" });
      joystickRefresh();
    }

    async function syncJoystickSpeed(v) {
      await api("/joystick/speed", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ speed: parseFloat(v) })
      });
    }

    function switchTab(el, tab) {
      activeTab = tab;
      document.querySelectorAll(".nav-item").forEach(item => item.classList.remove("active"));
      if (el) el.classList.add("active");
      
      document.querySelectorAll(".tab-content").forEach(content => content.classList.remove("active"));
      const targetTab = document.getElementById("tab-" + tab);
      if (targetTab) {
        targetTab.classList.add("active");
      }
      
      const titleMap = {
        drive: "🤖 주행 제어",
        grounding: "🔍 그라운딩 검증",
        latency: "📊 지연 · 드리프트",
        verify: "🧪 경로 검증",
        calib: "🔧 캘리브레이션 (STOP·수동조작)",
        history: "📚 세션 히스토리",
        system: "🖥️ 시스템",
        srvcfg: "⚙️ 서버 설정 (모델·그라운더)",
        collect: "📷 데이터수집",
        dataset: "🗂 데이터셋 히스토리",
        wikiinfo: "📖 위키",
        wikistatus: "📡 최신현황",
        overshoot: "🌀 오버슈트 가이드"
      };
      document.getElementById("page-title").textContent = titleMap[tab] || (tab.toUpperCase() + " Panel");

      if (tab === "history") {
        loadSessionList();
      }
      if (tab === "latency") {
        initDriftChart();
      }
      if (tab === "verify") {
        syncVerifyRuntimeParams();
        loadEpisodeHistory();
      }
      if (tab === "srvcfg") {
        loadSrvCfg();
        loadSrvLog();
      }
      if (tab === "wikiinfo") {
        loadWikiContent("info");
      }
      if (tab === "wikistatus") {
        loadWikiContent("status");
      }
      if (tab === "overshoot" && !osgState.frames.length) {
        osgSetDirection(osgState.direction);
      }
      if (tab === "collect") {
        collectRefreshState();
        collectStartKeyPolling();
        document.getElementById("collect-key-surface")?.focus();
      } else {
        collectStopKeyPolling();
      }
      if (tab === "dataset") {
        loadDatasetList();
      }
    }

    // ── 📷 데이터수집 탭 ─────────────────────────────────────────
    const COLLECT_KEYS = new Set(["w","a","s","d","q","e","z","c","r","t"," "]);
    const COLLECT_LABELS = {w:"FORWARD", s:"BACKWARD", a:"LEFT", d:"RIGHT", q:"FWD+LEFT", e:"FWD+RIGHT",
                             z:"BACK+LEFT", c:"BACK+RIGHT", r:"ROT_L", t:"ROT_R", " ":"STOP"};
    const COLLECT_DESC = {
      q: "↖ Q — 전진+좌 (FWD+LEFT)", w: "▲ W — 전진 (FORWARD)", e: "↗ E — 전진+우 (FWD+RIGHT)",
      a: "◀ A — 좌 (LEFT)", d: "▶ D — 우 (RIGHT)",
      z: "↙ Z — 후진+좌 (BACK+LEFT)", s: "▼ S — 후진 (BACKWARD)", c: "↘ C — 후진+우 (BACK+RIGHT)",
      r: "↺ R — 좌회전 (ROT_L)", t: "↻ T — 우회전 (ROT_R)", " ": "⏹ SPACE — 정지 (STOP)",
    };
    const JOYSTICK_BTN_INFO = {
      0: {name:"A",     desc:"STOP"},
      1: {name:"B",     desc:"마지막 프레임 취소"},
      2: {name:"X",     desc:"에피소드 폐기"},
      3: {name:"Y",     desc:"미사용"},
      4: {name:"L1",    desc:"녹화 시작"},
      5: {name:"R1",    desc:"정지 & 저장"},
      6: {name:"SEL",   desc:"녹화 토글"},
      7: {name:"START", desc:"SYNC↔ASYNC 모드"},
    };
    let _collectPressedKey = null;
    let _collectRepeatTimer = null;
    let _collectPollTimer = null;
    let _collectScenariosLoaded = false;
    let _collectCxPosLoaded = false;
    let _collectCxPathsLoaded = false;
    let _collectPrevStagedScenario = undefined;
    let _collectLastHatDirShown = null;
    let _collectEpisodeStartedAt = null;  // 서버 episode_started_at(epoch초) — REC 타이머용
    let _collectTimerTick = null;

    function _collectSendKey(key, event) {
      api("/collect/key", { method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ key, event }) });
    }

    function _collectPadLight(key, on) {
      const padBtn = document.getElementById("collect-pad-" + key);
      if (padBtn) padBtn.classList.toggle("active", on);
      const cap = document.getElementById("collect-pad-caption");
      if (cap) {
        if (on) {
          cap.textContent = COLLECT_DESC[key] || key;
          cap.style.color = "var(--emerald)";
          cap.style.borderColor = "var(--emerald)";
        } else {
          cap.textContent = "대기 중 — 버튼/키보드를 누르면 여기 표시";
          cap.style.color = "var(--text-muted)";
          cap.style.borderColor = "var(--border-glow)";
        }
      }
    }

    function collectKeyDown(e) {
      const key = e.key.length === 1 ? e.key.toLowerCase() : (e.key === " " ? " " : null);
      if (key === null || !COLLECT_KEYS.has(key)) return;
      e.preventDefault();
      if (_collectPressedKey === key) return;  // 이미 눌려있음(브라우저 autorepeat) — 무시
      _collectPressedKey = key;
      document.getElementById("collect-last-action").textContent = COLLECT_LABELS[key] || key;
      _collectPadLight(key, true);
      _collectSendKey(key, "down");
      if (_collectRepeatTimer) clearInterval(_collectRepeatTimer);
      _collectRepeatTimer = setInterval(() => _collectSendKey(key, "down"), 150);
    }

    function collectKeyUp(e) {
      const key = e.key.length === 1 ? e.key.toLowerCase() : (e.key === " " ? " " : null);
      if (key === null || key !== _collectPressedKey) return;
      _collectPressedKey = null;
      if (_collectRepeatTimer) { clearInterval(_collectRepeatTimer); _collectRepeatTimer = null; }
      document.getElementById("collect-last-action").textContent = "STOP";
      _collectPadLight(key, false);
      _collectSendKey(key, "up");
    }

    // 버튼 패드 — 키보드와 동일한 press/repeat 상태(_collectPressedKey) 공유
    function _collectPadDown(key) {
      if (_collectPressedKey === key) return;
      _collectPressedKey = key;
      document.getElementById("collect-last-action").textContent = COLLECT_LABELS[key] || key;
      _collectPadLight(key, true);
      _collectSendKey(key, "down");
      if (_collectRepeatTimer) clearInterval(_collectRepeatTimer);
      _collectRepeatTimer = setInterval(() => _collectSendKey(key, "down"), 150);
    }

    function _collectPadUp(key) {
      if (_collectPressedKey !== key) return;
      _collectPressedKey = null;
      if (_collectRepeatTimer) { clearInterval(_collectRepeatTimer); _collectRepeatTimer = null; }
      document.getElementById("collect-last-action").textContent = "STOP";
      _collectPadLight(key, false);
      _collectSendKey(key, "up");
    }

    function _collectPadStop() {
      if (_collectRepeatTimer) { clearInterval(_collectRepeatTimer); _collectRepeatTimer = null; }
      _collectPressedKey = null;
      document.getElementById("collect-last-action").textContent = "STOP";
      const cap = document.getElementById("collect-pad-caption");
      if (cap) { cap.textContent = COLLECT_DESC[" "]; cap.style.color = "var(--rose)"; cap.style.borderColor = "var(--rose)"; }
      _collectSendKey(" ", "down");
    }

    let _collectCxTimer = null;

    function _collectCxBand(cx) {
      if (cx >= 0.10 && cx <= 0.15) return {label: "강한좌", color: "var(--rose)"};
      if (cx >= 0.20 && cx <= 0.25) return {label: "준극단좌", color: "var(--amber)"};
      if (cx >= 0.75 && cx <= 0.80) return {label: "준극단우", color: "var(--amber)"};
      if (cx >= 0.85 && cx <= 0.90) return {label: "강한우", color: "var(--rose)"};
      return {label: "목표 구간 밖", color: "var(--text-muted)"};
    }

    let _collectLastCx = null;
    let _collectLastColor = null;

    function _collectCxDrawOverlay(cx, color) {
      _collectLastCx = cx;
      _collectLastColor = color;
      const cv = document.getElementById("collect-cx-canvas");
      if (!cv) return;
      const ctx = cv.getContext("2d");
      ctx.clearRect(0, 0, cv.width, cv.height);
      const guideChk = document.getElementById("toggle-cxguide-collect");
      if (guideChk && guideChk.checked) {
        _collectDrawCxGuideBands(ctx, cv.width, cv.height);
      }
      const gridChk = document.getElementById("toggle-grid-collect");
      if (!gridChk || gridChk.checked) {
        drawGridLines(ctx, cv.width, cv.height);
      }
      if (cx == null) return;
      const x = cx * cv.width;
      // 목표 밴드 배경(참고용, 반투명)
      ctx.fillStyle = "rgba(244,63,94,0.08)";
      ctx.fillRect(0.10 * cv.width, 0, 0.05 * cv.width, cv.height);
      ctx.fillRect(0.85 * cv.width, 0, 0.05 * cv.width, cv.height);
      ctx.fillStyle = "rgba(245,158,11,0.08)";
      ctx.fillRect(0.20 * cv.width, 0, 0.05 * cv.width, cv.height);
      ctx.fillRect(0.75 * cv.width, 0, 0.05 * cv.width, cv.height);
      // 현재 cx 세로선
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, cv.height);
      ctx.stroke();
    }

    async function _collectCxPoll() {
      const res = await api("/collect/ground");
      const valEl = document.getElementById("collect-cx-value");
      const bandEl = document.getElementById("collect-cx-band");
      if (!res.ok) { valEl.textContent = "⚠️ " + (res.error || "실패"); _collectCxDrawOverlay(null); return; }
      if (!res.has_bbox) { valEl.textContent = "검출 안 됨"; valEl.style.color = "var(--text-muted)"; _collectCxDrawOverlay(null); return; }
      const band = _collectCxBand(res.cx);
      valEl.textContent = "cx = " + res.cx.toFixed(3);
      valEl.style.color = band.color;
      bandEl.innerHTML = `<span style="color:${band.color}; font-weight:700;">${band.label}</span> · area=${res.area.toFixed(3)} · ${res.latency_ms}ms`;
      _collectCxDrawOverlay(res.cx, band.color);
    }

    function _collectDrawCxGuideBands(ctx, W, H) {
      const bands = [
        {lo: 0.10, hi: 0.15, label: "강한좌", color: "rgba(244,63,94,0.95)", fill: "rgba(244,63,94,0.22)"},
        {lo: 0.20, hi: 0.25, label: "준극단좌", color: "rgba(245,158,11,0.95)", fill: "rgba(245,158,11,0.22)"},
        {lo: 0.75, hi: 0.80, label: "준극단우", color: "rgba(245,158,11,0.95)", fill: "rgba(245,158,11,0.22)"},
        {lo: 0.85, hi: 0.90, label: "강한우", color: "rgba(244,63,94,0.95)", fill: "rgba(244,63,94,0.22)"},
      ];
      bands.forEach(b => {
        const x0 = b.lo * W, x1 = b.hi * W;
        ctx.fillStyle = b.fill;
        ctx.fillRect(x0, 0, x1 - x0, H);
        ctx.strokeStyle = b.color;
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(x0, 0); ctx.lineTo(x0, H); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(x1, 0); ctx.lineTo(x1, H); ctx.stroke();
        ctx.fillStyle = b.color;
        ctx.textAlign = "center";
        ctx.font = "bold 22px monospace";
        const midX = (x0 + x1) / 2;
        ctx.fillText(b.label, midX, 30);
        ctx.font = "16px monospace";
        ctx.fillText(b.lo.toFixed(2) + "~" + b.hi.toFixed(2), midX, 54);
      });
      ctx.setLineDash([8, 6]);
      ctx.strokeStyle = "rgba(148,163,184,0.8)";
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(W / 2, 0); ctx.lineTo(W / 2, H); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(148,163,184,0.95)";
      ctx.textAlign = "center";
      ctx.font = "16px monospace";
      ctx.fillText("center 0.50", W / 2, H - 12);
    }

    function collectToggleCxFeed() {
      const badge = document.getElementById("collect-cx-toggle-badge");
      if (_collectCxTimer) {
        clearInterval(_collectCxTimer);
        _collectCxTimer = null;
        badge.textContent = "⏸ 꺼짐 — 클릭해서 시작";
        badge.style.background = "rgba(100,116,139,0.2)";
        badge.style.color = "var(--text-muted)";
        _collectCxDrawOverlay(null);
      } else {
        _collectCxPoll();
        _collectCxTimer = setInterval(_collectCxPoll, 600);
        badge.textContent = "🔴 실시간 중 — 클릭해서 정지";
        badge.style.background = "rgba(244,63,94,0.15)";
        badge.style.color = "var(--rose)";
      }
    }

    function _collectUpdateTimerBadge() {
      const timerBadge = document.getElementById("collect-timer-badge");
      if (!timerBadge || !_collectEpisodeStartedAt) return;
      const elapsed = Math.max(0, Date.now() / 1000 - _collectEpisodeStartedAt);
      const m = Math.floor(elapsed / 60);
      const s = Math.floor(elapsed % 60);
      timerBadge.textContent = `⏱ ${m}:${String(s).padStart(2, "0")}`;
    }

    function _collectRenderRecentSaves(saves) {
      const el = document.getElementById("collect-recent-saves");
      if (!el) return;
      if (!saves.length) {
        el.innerHTML = `<div style="font-size:11px; color:var(--text-muted); padding:6px 0;">아직 없음</div>`;
        return;
      }
      el.innerHTML = saves.map(s => {
        const t = new Date(s.ts * 1000);
        const timeStr = t.toLocaleTimeString("ko-KR", { hour12: false });
        const ok = s.saved;
        const color = ok ? "var(--emerald)" : "var(--amber)";
        const icon = ok ? "✅" : "⚠️";
        const detail = ok
          ? `${s.steps} steps · ${s.duration.toFixed(1)}s`
          : (s.reason || "저장 안 됨");
        return `
          <div style="border-left:3px solid ${color}; background:#090d16; border-radius:4px; padding:6px 10px;">
            <div style="display:flex; justify-content:space-between; gap:8px;">
              <span style="font-size:11px; color:${color}; font-weight:700;">${icon} ${_wikiEscapeHtml(s.name || "")}</span>
              <span style="font-size:10px; color:var(--text-muted); font-family:var(--font-mono); white-space:nowrap;">${timeStr}</span>
            </div>
            <div style="font-size:10.5px; color:var(--text-muted); margin-top:2px;">${_wikiEscapeHtml(detail)}</div>
          </div>`;
      }).join("");
    }

    // Gradio update_ui_state() 이식 — IDLE/REC/TARGET MET/RETURNING 큰 상태카드
    // + 선택현황(시나리오/트랙A cx위치+경로) 통합, 값 바뀌면 flash로 눈에 띄게 표시
    let _collectPrevStatusSub = null;
    function _collectRenderStatusCard(res) {
      const el = document.getElementById("collect-status-card");
      const mainEl = document.getElementById("collect-status-main");
      const subEl = document.getElementById("collect-status-sub");
      if (!el || !mainEl) return;
      const scenarioKey = res.scenario || res.staged_scenario;
      const scenarioLabel = (res.scenarios && scenarioKey && res.scenarios[scenarioKey])
        ? res.scenarios[scenarioKey].label : "";
      let text, cls;
      if (res.returning) {
        text = "🔄 RETURNING..."; cls = "rec";
      } else if (res.active) {
        const n = res.steps;
        const target = res.current_scenario_target || 0;
        if (target > 0) {
          const pct = Math.min(100, Math.floor(n / target * 100));
          const bar = "█".repeat(Math.floor(pct / 10)) + "░".repeat(10 - Math.floor(pct / 10));
          if (n >= target) { text = `✅ TARGET MET [${n}/${target}] ${bar} 100% — ${scenarioLabel}`; cls = "done"; }
          else { text = `● REC [${n}/${target}] ${bar} ${pct}% — ${scenarioLabel}`; cls = "rec"; }
        } else {
          text = `● REC [${n}] — ${scenarioLabel}`; cls = "rec";
        }
      } else if (scenarioKey) {
        text = `⏸ IDLE — 선택: [${scenarioKey}] ${scenarioLabel}  ▶ 아래 시작 버튼으로 녹화 시작`; cls = "idle";
      } else {
        text = "⏸ IDLE — 시나리오 선택 필요 (D-pad 좌/우 또는 화면 드롭다운)"; cls = "idle";
      }
      mainEl.textContent = text;
      el.className = "collect-status-card " + cls;

      if (subEl) {
        const posLabel = (res.cx_positions && res.staged_cx_position)
          ? res.cx_positions[res.staged_cx_position].label : "미지정";
        const pathLabel = (res.cx_paths && res.staged_cx_path)
          ? res.cx_paths[res.staged_cx_path].label : "미지정";
        const subText = `시나리오: ${scenarioLabel || "미지정"} · 트랙A: ${posLabel} + ${pathLabel}`;
        subEl.textContent = subText;
        if (_collectPrevStatusSub !== null && subText !== _collectPrevStatusSub) {
          subEl.classList.remove("flash-highlight");
          void subEl.offsetWidth;
          subEl.classList.add("flash-highlight");
        }
        _collectPrevStatusSub = subText;
      }
    }

    // Gradio episode_timeline_md/episode_dist_md 이식 — 최근 액션 기호열 + Hz + 8-class 분포
    function _collectRenderTimeline(t) {
      const box = document.getElementById("collect-timeline-box");
      const distBox = document.getElementById("collect-dist-box");
      if (!box) return;
      if (!t.n) {
        box.textContent = "대기 중 — 시나리오 선택 후 녹화 시작";
        if (distBox) distBox.textContent = "";
        return;
      }
      const hzStr = t.hz ? ` · ${t.hz}Hz` : "";
      box.textContent = `${t.n} frames${hzStr}\\n최근→  ${t.symbols}`;
      if (distBox && t.dist) {
        const entries = Object.entries(t.dist);
        const mx = Math.max(...entries.map(([, c]) => c), 1);
        distBox.textContent = entries.map(([name, c]) =>
          `${name.padEnd(8)} ${"█".repeat(Math.round(c / mx * 12)).padEnd(12)} ${c}`
        ).join("\\n");
      }
    }

    // Gradio session_summary_md 이식 — 마지막 저장 세션의 프레임/소요시간/실측Hz
    function _collectRenderSessionSummary(s) {
      const el = document.getElementById("collect-session-summary");
      if (!el) return;
      if (!s) {
        el.textContent = "도착 시 정지 → 저장하면 소요 초·Hz가 여기 표시됩니다";
        return;
      }
      el.textContent = `프레임    : ${s.frames}\\n소요 시간 : ${s.duration_s} s\\n실측 Hz   : ${s.hz} Hz`;
    }

    async function collectRefreshState() {
      const res = await api("/collect/state");
      if (!res.ok) return;
      const activeBadge = document.getElementById("collect-active-badge");
      const stepsBadge = document.getElementById("collect-steps-badge");
      if (activeBadge) {
        activeBadge.innerHTML = res.active
          ? `<span class="collect-rec-dot"></span>REC: ${res.episode_name || ""}`
          : "⏸ 대기중 (녹화 안 됨)";
        activeBadge.style.background = res.active ? "rgba(244,63,94,0.15)" : "rgba(100,116,139,0.2)";
        activeBadge.style.color = res.active ? "var(--rose)" : "var(--text-muted)";
      }
      if (stepsBadge) stepsBadge.textContent = res.steps + " steps";

      const timerBadge = document.getElementById("collect-timer-badge");
      if (res.active && res.episode_started_at) {
        _collectEpisodeStartedAt = res.episode_started_at;
        if (!_collectTimerTick) {
          _collectTimerTick = setInterval(_collectUpdateTimerBadge, 500);
          _collectUpdateTimerBadge();
        }
      } else {
        _collectEpisodeStartedAt = null;
        if (_collectTimerTick) { clearInterval(_collectTimerTick); _collectTimerTick = null; }
        if (timerBadge) timerBadge.textContent = "";
      }

      _collectRenderRecentSaves(res.recent_saves || []);
      _collectRenderStatusCard(res);
      _collectRenderTimeline(res.episode_timeline || {});
      _collectRenderSessionSummary(res.last_session_summary);

      const returnBtn = document.getElementById("collect-return-btn");
      if (returnBtn) {
        returnBtn.disabled = res.active || !res.has_return_path;
        returnBtn.style.opacity = returnBtn.disabled ? "0.4" : "1";
        if (res.returning) {
          returnBtn.textContent = "🛑 복귀 중지";
          returnBtn.style.background = "rgba(245,158,11,0.15)";
        } else {
          returnBtn.textContent = "🔄 복귀 (직전 경로 역주행)";
          returnBtn.style.background = "";
        }
      }

      if (!_collectScenariosLoaded && res.scenarios) {
        const sel = document.getElementById("collect-scenario-select");
        Object.entries(res.scenarios).forEach(([id, info]) => {
          const opt = document.createElement("option");
          opt.value = id;
          opt.textContent = `${info.label} (목표 ${info.target})`;
          sel.appendChild(opt);
        });
        _collectScenariosLoaded = true;
      }

      const selEl = document.getElementById("collect-scenario-select");
      if (selEl && !res.active && selEl.value !== (res.staged_scenario || "")) {
        selEl.value = res.staged_scenario || "";
      }
      const dpadBadge = document.getElementById("collect-scenario-dpad-badge");
      if (dpadBadge) dpadBadge.style.display = res.staged_scenario ? "inline" : "none";

      // 시나리오가 바뀔 때마다(D-pad/◀▶/클릭 무관) 눈에 잘 띄게 카드 행을 잠깐 강조
      if (res.staged_scenario !== _collectPrevStagedScenario) {
        _collectPrevStagedScenario = res.staged_scenario;
        const row = document.getElementById("collect-scenario-row");
        if (row) {
          row.classList.remove("flash-highlight");
          void row.offsetWidth; // 리플로우 강제 — 연속 변경 시에도 애니메이션 재시작
          row.classList.add("flash-highlight");
        }
      }

      const cxSelEl = document.getElementById("collect-cxpos-select");
      if (cxSelEl && !res.active && cxSelEl.value !== (res.staged_cx_position || "")) {
        cxSelEl.value = res.staged_cx_position || "";
      }
      {
        const MODE_LABELS = {trackA: "트랙A(위치)", cxpath: "접근경로", scenario: "시나리오"};
        const MODE_COLORS = {trackA: ["rgba(56,189,248,0.15)", "var(--cyan)"],
                              cxpath: ["rgba(245,158,11,0.15)", "var(--amber)"],
                              scenario: ["rgba(163,113,247,0.15)", "#a371f7"]};
        const m = res.collect_mode || "trackA";
        const [bg, fg] = MODE_COLORS[m] || MODE_COLORS.trackA;
        const modeText = "🎮 D-pad 대상: " + (MODE_LABELS[m] || m);
        ["collect-mode-badge", "collect-mode-badge-rt"].forEach(id => {
          const badge = document.getElementById(id);
          if (!badge) return;
          if (badge.textContent !== modeText && badge.dataset.inited === "1") {
            badge.classList.remove("flash-highlight");
            void badge.offsetWidth;
            badge.classList.add("flash-highlight");
          }
          badge.dataset.inited = "1";
          badge.textContent = modeText;
          badge.style.background = bg;
          badge.style.color = fg;
        });
      }
      const selBadge = document.getElementById("collect-cxpos-current");
      const posLabel = (res.cx_positions && res.staged_cx_position)
        ? res.cx_positions[res.staged_cx_position].label : "미지정";
      const pathLabel = (res.cx_paths && res.staged_cx_path)
        ? res.cx_paths[res.staged_cx_path].label : "미지정";
      if (selBadge) selBadge.textContent = `현재 선택: ${posLabel} + ${pathLabel}`;

      const progList = document.getElementById("collect-progress-list");
      const curSel = document.getElementById("collect-scenario-select")?.value || "";
      if (progList && res.scenarios) {
        progList.innerHTML = Object.entries(res.scenarios).map(([id, info]) => {
          const done = (res.scenario_stats || {})[id] || 0;
          const pct = Math.min(100, Math.round(done / info.target * 100));
          const isSel = id === curSel;
          return `<div onclick="_collectClickScenario('${id}')"
                       style="display:flex; justify-content:space-between; gap:8px; padding:4px 6px; cursor:pointer; border-radius:4px;
                              background:${isSel ? "rgba(56,189,248,0.15)" : "transparent"};
                              border:1px solid ${isSel ? "var(--cyan)" : "transparent"};">
                    <span style="color:${isSel ? "var(--cyan)" : "var(--text-muted)"}; font-weight:${isSel ? "700" : "400"};">${isSel ? "▶ " : ""}${info.label}</span>
                    <span>${done}/${info.target} (${pct}%)</span>
                  </div>`;
        }).join("") + `<div style="margin-top:6px; padding-top:6px; border-top:1px solid var(--border-glow); display:flex; justify-content:space-between;">
                    <span style="color:var(--emerald); font-weight:700;">전체</span>
                    <span style="color:var(--emerald); font-weight:700;">${res.total_completed}/${res.total_target}</span>
                  </div>`;
      }

      if (!_collectCxPosLoaded && res.cx_positions) {
        const cxSel = document.getElementById("collect-cxpos-select");
        Object.entries(res.cx_positions).forEach(([id, info]) => {
          const opt = document.createElement("option");
          opt.value = id;
          opt.textContent = `${info.label} (${info.lo.toFixed(2)}~${info.hi.toFixed(2)}, 목표 ${info.target})`;
          cxSel.appendChild(opt);
        });
        _collectCxPosLoaded = true;
      }
      if (!_collectCxPathsLoaded && res.cx_paths) {
        const pathSel = document.getElementById("collect-cxpath-select");
        Object.entries(res.cx_paths).forEach(([id, info]) => {
          const opt = document.createElement("option");
          opt.value = id;
          opt.textContent = `${info.label} (목표 ${info.target})`;
          pathSel.appendChild(opt);
        });
        _collectCxPathsLoaded = true;
      }
      const cxPathSelEl = document.getElementById("collect-cxpath-select");
      if (cxPathSelEl && !res.active && cxPathSelEl.value !== (res.staged_cx_path || "")) {
        cxPathSelEl.value = res.staged_cx_path || "";
      }
      _collectRenderCxPosChart(res);

      const nameInput = document.getElementById("collect-episode-name");
      if (nameInput && res.active && res.episode_name) nameInput.value = res.episode_name;
    }

    // 극단 배치 4포지션을 cx축(0~1) 위에 사각형 막대로 표시 — 각 막대 채움 = 수집 진행률
    function _collectRenderCxPosChart(res) {
      const chart = document.getElementById("collect-cxpos-chart");
      if (!chart || !res.cx_positions) return;
      const curCx = document.getElementById("collect-cxpos-select")?.value || "";
      const curPath = document.getElementById("collect-cxpath-select")?.value || "";
      const order = ["strong_left", "weak_left", "center", "weak_right", "strong_right"];
      const pathOrder = ["left_curve", "straight", "right_curve", "overshoot_left_recover", "overshoot_right_recover"];
      const posStats = res.cx_position_stats || {};
      const pathStats = res.cx_position_path_stats || {};
      const paths = res.cx_paths || {};
      let totalDone = 0, totalTarget = 0;
      const blocks = order.filter(id => res.cx_positions[id]).map((id, i) => {
        const info = res.cx_positions[id];
        const done = posStats[id] || 0;
        totalDone += done; totalTarget += info.target;
        const pct = Math.min(100, Math.round(done / info.target * 100));
        const isSel = id === curCx;

        // center(트랙F)는 오버슈트(트랙C, 극단 4위치 전용) 대상이 아니라 해당 경로는 숨김
        const pathRows = pathOrder.filter(p => paths[p] && !(id === "center" && p.startsWith("overshoot"))).map(p => {
          const pinfo = paths[p];
          const pdone = pathStats[`${id}::${p}`] || 0;
          const ppct = Math.min(100, Math.round(pdone / pinfo.target * 100));
          const isPathSel = isSel && p === curPath;
          return `
            <div onclick="_collectClickCxPath('${id}','${p}')"
                 style="display:flex; align-items:center; gap:6px; padding-left:14px; cursor:pointer; ${isPathSel ? "outline:1px solid var(--amber); border-radius:4px;" : ""}">
              <span style="width:60px; font-size:10px; color:${isPathSel ? "var(--amber)" : "var(--text-muted)"};">${pinfo.label}</span>
              <div style="flex:1; height:10px; background:#090d16; border:1px solid var(--border-glow); border-radius:3px; overflow:hidden;">
                <div style="width:${ppct}%; height:100%; background:${ppct >= 100 ? "var(--emerald)" : "var(--amber)"};"></div>
              </div>
              <span style="width:34px; text-align:right; font-size:10px;">${pdone}/${pinfo.target}</span>
            </div>`;
        }).join("");

        return `
          <div>
            <div onclick="_collectClickCxPos('${id}')"
                 style="display:flex; align-items:center; gap:8px; cursor:pointer; ${isSel ? "outline:1px solid var(--cyan); border-radius:6px; padding:2px 4px;" : ""}">
              <span style="width:52px; color:${isSel ? "var(--cyan)" : "var(--text-muted)"}; font-weight:${isSel ? "700" : "400"};">${info.label}</span>
              <div style="flex:1; height:16px; background:#090d16; border:1px solid var(--border-glow); border-radius:3px; overflow:hidden;">
                <div style="width:${pct}%; height:100%; background:${pct >= 100 ? "var(--emerald)" : "var(--cyan)"};"></div>
              </div>
              <span style="width:52px; text-align:right;">${done}/${info.target}</span>
              <span style="width:78px; text-align:right; color:var(--text-muted);">${info.lo.toFixed(2)}~${info.hi.toFixed(2)}</span>
            </div>
            <div style="display:flex; flex-direction:column; gap:2px; margin-top:3px;">${pathRows}</div>
          </div>`;
      }).join("");
      chart.innerHTML = blocks + `<div style="margin-top:4px; padding-top:6px; border-top:1px solid var(--border-glow); display:flex; justify-content:space-between;">
                  <span style="color:var(--emerald); font-weight:700;">전체</span>
                  <span style="color:var(--emerald); font-weight:700;">${totalDone}/${totalTarget}</span>
                </div>`;
    }

    // 진행률 목록의 행을 클릭해서 시나리오 선택 — 서버 staged_scenario에 반영(조이스틱 D-pad와 동일 소스 공유)
    async function _collectClickScenario(id) {
      await api("/collect/scenario/stage", { method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ scenario: id }) });
      collectRefreshState();
    }
    // 화면 ◀/▶ 버튼으로 시나리오 순환 — 조이스틱 D-pad와 동일한 서버 로직(cycle_scenario) 재사용
    async function _collectCycleScenario(step) {
      await api("/collect/scenario/cycle", { method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ step }) });
      collectRefreshState();
    }
    async function _collectSyncScenarioHighlight() {
      const sel = document.getElementById("collect-scenario-select");
      await api("/collect/scenario/stage", { method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ scenario: sel?.value || null }) });
      collectRefreshState();
    }

    // 트랙A 막대 행 클릭 → 서버 staged_cx_position에 반영(조이스틱 D-pad와 동일 소스 공유), 모드도 트랙A로 전환
    async function _collectClickCxPos(id) {
      await api("/collect/cxpos/stage", { method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ cx_position: id }) });
      collectRefreshState();
    }
    async function _collectSyncCxPosHighlight() {
      const sel = document.getElementById("collect-cxpos-select");
      await api("/collect/cxpos/stage", { method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ cx_position: sel?.value || null }) });
      collectRefreshState();
    }
    async function _collectCycleCxPos(step) {
      await api("/collect/cxpos/cycle", { method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ step }) });
      collectRefreshState();
    }
    async function _collectToggleMode(step) {
      await api("/collect/mode/toggle", { method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ step: step || 1 }) });
      collectRefreshState();
    }
    async function _collectCycleCxPath(step) {
      await api("/collect/cxpath/cycle", { method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ step }) });
      collectRefreshState();
    }
    async function _collectSyncCxPathHighlight() {
      const sel = document.getElementById("collect-cxpath-select");
      await api("/collect/cxpath/stage", { method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ cx_path: sel?.value || null }) });
      collectRefreshState();
    }
    async function _collectClickCxPath(posId, pathId) {
      await api("/collect/cxpos/stage", { method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ cx_position: posId }) });
      await api("/collect/cxpath/stage", { method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ cx_path: pathId }) });
      collectRefreshState();
    }

    function collectStartKeyPolling() {
      if (_collectPollTimer) return;
      _collectPollTimer = setInterval(collectRefreshState, 2000);
    }

    function collectStopKeyPolling() {
      if (_collectRepeatTimer) { clearInterval(_collectRepeatTimer); _collectRepeatTimer = null; }
      if (_collectPollTimer) { clearInterval(_collectPollTimer); _collectPollTimer = null; }
      if (_collectPressedKey) { _collectSendKey(_collectPressedKey, "up"); _collectPressedKey = null; }
      if (_collectCxTimer) {
        clearInterval(_collectCxTimer);
        _collectCxTimer = null;
        const badge = document.getElementById("collect-cx-toggle-badge");
        if (badge) { badge.textContent = "⏸ 꺼짐 — 클릭해서 시작"; badge.style.background = "rgba(100,116,139,0.2)"; badge.style.color = "var(--text-muted)"; }
      }
    }

    async function collectStartEpisode() {
      const episode_name = document.getElementById("collect-episode-name").value.trim() || null;
      const scenario = document.getElementById("collect-scenario-select").value || null;
      const pattern = document.getElementById("collect-pattern-select").value || null;
      const cx_position = document.getElementById("collect-cxpos-select").value || null;
      const cx_path = document.getElementById("collect-cxpath-select").value || null;
      const res = await api("/collect/episode/start", { method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ episode_name, scenario, pattern, cx_position, cx_path }) });
      const statusEl = document.getElementById("collect-episode-status");
      statusEl.textContent = res.ok ? "▶ 시작됨: " + res.episode_name : "⚠️ " + res.error;
      statusEl.style.color = res.ok ? "var(--rose)" : "var(--amber)";
      collectRefreshState();
    }

    async function collectStopEpisode() {
      const res = await api("/collect/episode/stop", { method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ save: true }) });
      const statusEl = document.getElementById("collect-episode-status");
      statusEl.textContent = res.ok
        ? (res.saved ? `✅ 저장됨: ${res.path} (${res.steps} steps, ${res.duration.toFixed(1)}s, STOP프레임 +5 포함)` : "⚠️ 저장 안 함: " + (res.reason || ""))
        : "⚠️ " + res.error;
      statusEl.style.color = (res.ok && res.saved) ? "var(--emerald)" : "var(--amber)";
      document.getElementById("collect-episode-name").value = "";
      collectRefreshState();
    }

    // 직전 경로를 역재생해 시작 위치로 복귀 — 조이스틱 R2 트리거와 동일 기능(화면 버튼)
    async function collectAutoReturn() {
      const res = await api("/collect/return", { method: "POST" });
      const statusEl = document.getElementById("collect-episode-status");
      if (statusEl) statusEl.textContent = res.ok ? res.msg : "⚠️ " + (res.error || res.msg);
      collectRefreshState();
    }

    (function collectInit() {
      const surface = document.getElementById("collect-key-surface");
      if (!surface) return;
      surface.addEventListener("keydown", collectKeyDown);
      surface.addEventListener("keyup", collectKeyUp);
      surface.addEventListener("blur", () => { if (_collectPressedKey) collectKeyUp({key: _collectPressedKey}); });
      collectStartKeyPolling();
      _collectCxDrawOverlay(null, null);
    })();

    async function loadWikiContent(name) {
      const contentEl = document.getElementById(name === "info" ? "wiki-info-content" : "wiki-status-content");
      if (!contentEl) return;
      try {
        const res = await api("/wiki/" + name);
        if (res.ok) {
          contentEl.innerHTML = renderWikiMarkdown(res.content);
          if (name === "status") {
            const mtimeEl = document.getElementById("wiki-status-mtime");
            if (mtimeEl) mtimeEl.textContent = `최근 갱신: ${res.mtime} (${_wikiRelTime(res.mtime)})`;
          }
        } else {
          contentEl.textContent = "⚠️ 로드 실패: " + res.error;
        }
      } catch (e) {
        contentEl.textContent = "⚠️ 서버 오류: " + e;
      }
      loadResearchJournal(name);
    }

    // ── 🗓️ 연구일지 — Tab6 세션 히스토리(list + Frame Inspector)와 동일한
    // list+detail 스플릿 UX로 git 커밋을 훑어보기. 같은 md 파일의 과거 버전을
    // 보여주는 게 아니라, 프로젝트 전체 진행 기록을 날짜순으로 브라우징하는 용도.
    let _wjActiveSha = { info: null, status: null };

    async function loadResearchJournal(name) {
      const listEl = document.getElementById(`wiki-${name}-journal-list`);
      if (!listEl) return;
      try {
        const res = await api("/journal");
        if (!res.ok || !res.entries || res.entries.length === 0) {
          listEl.innerHTML = `<div style="font-size:11px; color:var(--wiki-muted); padding:6px 0;">연구일지 없음</div>`;
          return;
        }
        listEl.innerHTML = res.entries.map(e => `
          <div class="wj-card ${e.kind}" data-sha="${e.sha}" onclick="selectJournalEntry('${name}', '${e.sha}', '${e.date}')">
            <div class="wj-top">
              <span class="wj-date">${e.date}</span>
              <span class="wj-rel">${_wikiRelTime(e.date)}</span>
            </div>
            <div class="wj-title" title="${_wikiEscapeHtml(e.subject)}">${_wikiEscapeHtml(e.subject)}</div>
            <div class="wj-sha">${e.sha.slice(0, 7)}</div>
          </div>
        `).join("");
      } catch (e) {
        listEl.innerHTML = `<div style="font-size:11px; color:var(--wiki-red);">연구일지 로드 실패</div>`;
      }
    }

    async function selectJournalEntry(name, sha, dateStr) {
      _wjActiveSha[name] = sha;
      document.querySelectorAll(`#wiki-${name}-journal-list .wj-card`).forEach(el => {
        el.classList.toggle("active", el.dataset.sha === sha);
      });
      const detailEl = document.getElementById(`wiki-${name}-journal-detail`);
      if (!detailEl) return;
      detailEl.innerHTML = `<div class="wjd-placeholder">로딩 중...</div>`;
      try {
        const res = await api(`/journal/${sha}`);
        if (res.ok) {
          detailEl.innerHTML = `
            <div class="wjd-head">
              <span class="wjd-sha">${res.sha.slice(0, 7)}</span>
              <span class="wjd-date">${dateStr} (${_wikiRelTime(dateStr)})</span>
            </div>
            <div class="wjd-body">${_wikiEscapeHtml(res.body)}</div>
            ${res.files ? `<div class="wjd-files">${_wikiEscapeHtml(res.files)}</div>` : ""}
          `;
        } else {
          detailEl.innerHTML = `<div class="wjd-placeholder">⚠️ 로드 실패: ${_wikiEscapeHtml(res.error || "")}</div>`;
        }
      } catch (e) {
        detailEl.innerHTML = `<div class="wjd-placeholder">⚠️ 서버 오류: ${e}</div>`;
      }
    }

    // ── 위키/최신현황 탭 전용 경량 markdown → HTML 렌더러 ──────────────
    // docs/v5/research_story.html 양식(챕터/콜아웃/표/이미지그리드)에 맞춘
    // 최소 문법만 지원 (헤더 #~###, 표, **굵게**, `코드`, ``` 블록,
    // > [!info|warn|critical|success] 콜아웃, ![](경로) 이미지)
    function _wikiRelTime(dateStr) {
      // "YYYY-MM-DD HH:MM" (KST, 로컬 표시와 동일 타임존 가정) → "N분/시간/일 전"
      const t = new Date(dateStr.replace(" ", "T"));
      if (isNaN(t.getTime())) return "";
      const diffMin = Math.round((Date.now() - t.getTime()) / 60000);
      if (diffMin < 1) return "방금 전";
      if (diffMin < 60) return `${diffMin}분 전`;
      const diffHr = Math.round(diffMin / 60);
      if (diffHr < 24) return `${diffHr}시간 전`;
      const diffDay = Math.round(diffHr / 24);
      return `${diffDay}일 전`;
    }
    function _wikiEscapeHtml(s) {
      return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }
    function _wikiInline(text) {
      let t = _wikiEscapeHtml(text);
      t = t.replace(/`([^`]+)`/g, "<code>$1</code>");
      t = t.replace(/\\*\\*([^*]+)\\*\\*/g, "<strong>$1</strong>");
      return t;
    }
    function _wikiSplitRow(line) {
      return line.trim().replace(/^\\||\\|$/g, "").split("|").map(s => s.trim());
    }
    function _wikiResolveImgSrc(path) {
      if (/^https?:\\/\\//.test(path) || path.startsWith("/")) return path;
      return "/docs-static/v5/" + path.replace(/^docs\\/v5\\//, "").replace(/^\\.\\//, "");
    }
    function renderWikiMarkdown(md) {
      const lines = (md || "").split(/\\r?\\n/);
      let html = "";
      let i = 0;
      while (i < lines.length) {
        const line = lines[i];

        // 코드 블록
        if (line.trim().startsWith("```")) {
          const buf = [];
          i++;
          while (i < lines.length && !lines[i].trim().startsWith("```")) { buf.push(lines[i]); i++; }
          i++;
          html += `<pre><code>${_wikiEscapeHtml(buf.join("\\n"))}</code></pre>`;
          continue;
        }

        // 콜아웃: > [!info|warn|critical|success] ...
        const calloutMatch = line.match(/^>\\s*\\[!(info|warn|critical|success)\\]\\s*(.*)$/);
        if (calloutMatch) {
          const type = calloutMatch[1];
          const buf = [calloutMatch[2]];
          i++;
          while (i < lines.length && lines[i].startsWith(">")) {
            buf.push(lines[i].replace(/^>\\s?/, ""));
            i++;
          }
          html += `<div class="callout ${type}">${buf.map(_wikiInline).join("<br>")}</div>`;
          continue;
        }

        // 표
        if (line.includes("|") && lines[i + 1] && /^\\s*\\|?\\s*[-:]+\\s*(\\|\\s*[-:]+\\s*)+\\|?\\s*$/.test(lines[i + 1])) {
          const headerCells = _wikiSplitRow(line);
          i += 2;
          const rows = [];
          while (i < lines.length && lines[i].includes("|")) { rows.push(_wikiSplitRow(lines[i])); i++; }
          html += '<table class="cl-table"><thead><tr>' +
            headerCells.map(c => `<th>${_wikiInline(c)}</th>`).join("") +
            "</tr></thead><tbody>" +
            rows.map(r => "<tr>" + r.map(c => `<td>${_wikiInline(c)}</td>`).join("") + "</tr>").join("") +
            "</tbody></table>";
          continue;
        }

        // 이미지 (연속되면 3열 그리드, 단독이면 카드)
        const imgMatch = line.match(/^!\\[(.*?)\\]\\((.*?)\\)\\s*$/);
        if (imgMatch) {
          const imgs = [];
          while (i < lines.length) {
            const m = lines[i].match(/^!\\[(.*?)\\]\\((.*?)\\)\\s*$/);
            if (!m) break;
            imgs.push(m);
            i++;
          }
          const fig = m => `<img src="${_wikiResolveImgSrc(m[2])}" alt="${_wikiEscapeHtml(m[1])}">` +
            (m[1] ? `<div class="fig-caption">${_wikiEscapeHtml(m[1])}</div>` : "");
          if (imgs.length >= 2) {
            html += '<div class="img-grid-3">' + imgs.map(m => `<div class="fig-card">${fig(m)}</div>`).join("") + "</div>";
          } else {
            html += `<div class="fig-card">${fig(imgs[0])}</div>`;
          }
          continue;
        }

        // 헤더
        const h = line.match(/^(#{1,3})\\s+(.*)$/);
        if (h) {
          const level = h[1].length;
          const text = h[2];
          const numMatch = text.match(/^(\\d+[.)])\\s*(.*)$/);
          const inner = (level <= 2 && numMatch)
            ? `<span class="wiki-chapter-num">${_wikiEscapeHtml(numMatch[1])}</span> ${_wikiInline(numMatch[2])}`
            : _wikiInline(text);
          const tag = level === 1 ? "h1" : level === 2 ? "h2" : "h3";
          html += `<${tag}>${inner}</${tag}>`;
          i++;
          continue;
        }

        // 빈 줄
        if (line.trim() === "") { i++; continue; }

        // 불릿 리스트
        if (/^\\s*[-*]\\s+/.test(line)) {
          const items = [];
          while (i < lines.length && /^\\s*[-*]\\s+/.test(lines[i])) {
            items.push(lines[i].replace(/^\\s*[-*]\\s+/, ""));
            i++;
          }
          html += "<ul>" + items.map(it => `<li>${_wikiInline(it)}</li>`).join("") + "</ul>";
          continue;
        }

        // 문단 (다음 특수 블록/빈 줄까지 누적)
        const buf = [line];
        i++;
        while (i < lines.length && lines[i].trim() !== "" &&
               !/^(#{1,3}\\s|```|>\\s*\\[!|!\\[|\\s*[-*]\\s)/.test(lines[i]) && !lines[i].includes("|")) {
          buf.push(lines[i]);
          i++;
        }
        html += `<p>${_wikiInline(buf.join(" "))}</p>`;
      }
      return html;
    }

    // ── ⚙️ 서버 설정 탭 ─────────────────────────────────────────────
    let srvcfgSel = { grounder: null, ckpt: null, activeGrounder: null, activeCkpt: null };

    function _srvcfgUpdatePreview() {
      const parts = [];
      if (srvcfgSel.grounder && srvcfgSel.grounder !== srvcfgSel.activeGrounder)
        parts.push(`VLA_GROUNDER=${srvcfgSel.grounder}`);
      if (srvcfgSel.ckpt && srvcfgSel.ckpt !== srvcfgSel.activeCkpt)
        parts.push(`VLA_S2V2_STAGE2=${srvcfgSel.ckpt}`);
      document.getElementById("srvcfg-restart-preview").textContent =
        parts.length ? parts.join("\\n") : "변경 없음 — 현재 설정으로 재시작";
    }

    function selGrounder(g) {
      srvcfgSel.grounder = g;
      document.getElementById("srvcfg-gr-pg2").className   = g === "pg2"   ? "btn btn-cyan" : "btn btn-outline";
      document.getElementById("srvcfg-gr-owlv2").className = g === "owlv2" ? "btn btn-cyan" : "btn btn-outline";
      _srvcfgUpdatePreview();
    }

    async function loadSrvCfg() {
      const h = await api("/infer/health");
      const body = document.getElementById("srvcfg-status-body");
      if (h.status === "error") {
        body.innerHTML = `<tr><td colspan="2" style="color:var(--rose); padding:12px;">서버 응답 없음: ${h.detail}</td></tr>`;
        return;
      }
      const g = h.grounder || {};
      const rows = [
        ["상태", h.status],
        ["git_commit", h.git_commit],
        ["프로세스 시작", h.process_started_at ? new Date(h.process_started_at*1000).toLocaleString() : "—"],
        ["그라운더", `${g.model} (${g.input_px}px, phrase="${g.phrase}")`],
        ["OWL threshold", g.owlv2_thresh !== undefined ? g.owlv2_thresh : "—"],
        ["헤드", `${h.head} (window=${h.window}, val_acc=${h.val_acc ? (h.val_acc*100).toFixed(1)+'%' : '—'})`],
        ["체크포인트", h.checkpoint_path],
        ["STOP 모드", `${h.stop_mode} (latched=${h.stop_latched})`],
        ["GPU", h.gpu ? `${h.gpu.device_name} · ${h.gpu.allocated_gb}GB 할당` : "—"],
        ["skip_n / multi_prompt", `${h.grounding_skip_n} / ${h.multi_prompt}`],
      ];
      body.innerHTML = rows.map(([k,v]) => `
        <tr style="border-bottom:1px solid rgba(29,43,69,0.5);">
          <td style="padding:6px 10px; color:var(--text-muted); width:140px;">${k}</td>
          <td style="padding:6px 10px; font-family:var(--font-mono); word-break:break-all;">${v ?? "—"}</td>
        </tr>`).join("");

      // Fix3 핸드셰이크 경고: 코드가 프로세스 기동 이후 수정됐으면 표시
      const hs = document.getElementById("srvcfg-handshake");
      if (h.code_mtime && h.process_started_at && h.code_mtime > h.process_started_at) {
        hs.style.display = "block";
        hs.textContent = "⚠️ 코드 파일이 프로세스 기동 이후 수정됨 — 서버가 구버전 코드로 실행 중일 수 있음. 재시작 권장.";
      } else {
        hs.style.display = "none";
      }

      // 활성 상태 반영
      srvcfgSel.activeGrounder = (g.model || "").toLowerCase().includes("owl") ? "owlv2" : "pg2";
      srvcfgSel.activeCkpt = h.checkpoint_path || null;
      if (!srvcfgSel.grounder) selGrounder(srvcfgSel.activeGrounder);
      if (g.owlv2_thresh !== undefined)
        document.getElementById("srvcfg-owl-thr").value = g.owlv2_thresh;
      if (g.owlv2_area_scale !== undefined)
        document.getElementById("srvcfg-owl-area-scale").value = g.owlv2_area_scale;

      // 체크포인트 목록 — kind별 섹션 그룹핑, 액션 헤드만 선택 가능
      const c = await api("/server_proc/checkpoints");
      const cb = document.getElementById("srvcfg-ckpt-body");
      if (c.ok) {
        const order = ["action", "stop", "stage1", "other"];
        const kindIcons = { action: "🎯", stop: "🛑", stage1: "🧠", other: "📎" };
        let html = "";
        for (const kind of order) {
          const group = c.checkpoints.filter(it => it.kind === kind);
          if (!group.length) continue;
          const note = kind === "action" ? "행 클릭으로 선택 → 재시작 시 적용"
                                          : "Stage2 교체 대상 아님 — 참고용";
          html += `<tr style="background:#1c2638;"><td colspan="3" style="padding:5px 8px; font-size:10px; font-weight:700; color:var(--text-muted);">${kindIcons[kind]} ${group[0].kind_label} (${group.length}) — ${note}</td></tr>`;
          html += group.map(it => {
            const isActive = c.active && it.path === c.active;
            const isSel = srvcfgSel.ckpt === it.path;
            const sel = it.selectable;
            return `<tr ${sel ? `onclick="selCkpt('${it.path}')"` : ""} title="${it.path}"
              style="${sel ? 'cursor:pointer;' : 'opacity:0.45; cursor:not-allowed;'} border-bottom:1px solid rgba(29,43,69,0.5); ${isActive ? 'background:rgba(16,185,129,0.10);' : isSel ? 'background:rgba(6,182,212,0.10);' : ''}">
              <td style="padding:5px 8px;">
                <div style="font-weight:600; font-size:11px;">${isActive ? '🟢 ' : isSel ? '🔵 ' : ''}${it.label}</div>
                <div style="font-family:var(--font-mono); font-size:9px; color:var(--text-muted); word-break:break-all;">${it.path}</div>
              </td>
              <td style="padding:5px 8px;">${it.size_mb}MB</td>
              <td style="padding:5px 8px; color:var(--text-muted);">${it.mtime}</td>
            </tr>`;
          }).join("");
        }
        cb.innerHTML = html;
      }
      _srvcfgUpdatePreview();
    }

    function selCkpt(p) {
      srvcfgSel.ckpt = (srvcfgSel.ckpt === p) ? null : p;  // 재클릭 시 해제
      loadSrvCfg();
    }

    async function applyOwlThresh() {
      const v = parseFloat(document.getElementById("srvcfg-owl-thr").value);
      const res = await api("/config", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ owlv2_thresh: v })
      });
      const inner = (res.applied && res.applied.applied) || {};
      document.getElementById("srvcfg-restart-status").textContent =
        inner.owlv2_thresh !== undefined ? `✅ threshold=${inner.owlv2_thresh} 즉시 적용됨` : "⚠️ 적용 실패 — 서버 로그 확인";
      setTimeout(loadSrvCfg, 500);
    }

    async function applyOwlAreaScale() {
      const v = parseFloat(document.getElementById("srvcfg-owl-area-scale").value);
      const res = await api("/config", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ owlv2_area_scale: v })
      });
      const inner = (res.applied && res.applied.applied) || {};
      document.getElementById("srvcfg-restart-status").textContent =
        inner.owlv2_area_scale !== undefined ? `✅ 보정계수=${inner.owlv2_area_scale} 즉시 적용됨` : "⚠️ 적용 실패 — 서버 로그 확인";
      setTimeout(loadSrvCfg, 500);
    }

    async function restartInferServer() {
      const body = {};
      if (srvcfgSel.grounder && srvcfgSel.grounder !== srvcfgSel.activeGrounder) body.grounder = srvcfgSel.grounder;
      if (srvcfgSel.ckpt && srvcfgSel.ckpt !== srvcfgSel.activeCkpt) body.ckpt = srvcfgSel.ckpt;
      const thr = parseFloat(document.getElementById("srvcfg-owl-thr").value);
      if (!isNaN(thr)) body.owlv2_thresh = thr;
      const areaScale = parseFloat(document.getElementById("srvcfg-owl-area-scale").value);
      if (!isNaN(areaScale)) body.owlv2_area_scale = areaScale;
      const desc = Object.keys(body).length ? JSON.stringify(body) : "현재 설정 유지";
      if (!confirm(`추론 서버(8001)를 재시작합니다 (~120s, 주행 불가).\\n적용: ${desc}\\n계속할까요?`)) return;

      const btn = document.getElementById("srvcfg-restart-btn");
      const st  = document.getElementById("srvcfg-restart-status");
      btn.disabled = true;
      const res = await api("/server_proc/restart", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body)
      });
      if (!res.ok) { st.textContent = "⚠️ " + res.error; btn.disabled = false; return; }

      let sec = 0;
      const iv = setInterval(async () => {
        sec += 5;
        st.textContent = `⏳ 재시작 중... ${sec}s (PG2 포함 최대 ~120s)`;
        try {
          const h = await api("/infer/health");
          if (h.status === "healthy" && h.process_started_at &&
              (Date.now()/1000 - h.process_started_at) < sec + 30) {
            clearInterval(iv);
            st.textContent = "✅ 재시작 완료";
            btn.disabled = false;
            srvcfgSel.grounder = null; srvcfgSel.ckpt = null;
            loadSrvCfg(); loadSrvLog();
          }
        } catch(e) {}
        if (sec >= 180) { clearInterval(iv); st.textContent = "⚠️ 타임아웃 — 서버 로그 확인 필요"; btn.disabled = false; }
      }, 5000);
    }

    async function loadSrvLog() {
      const res = await api("/server_proc/log?n=40");
      document.getElementById("srvcfg-log").textContent = (res.lines || []).join("\\n");
    }

    // ── Autopilot 시작 / 정지 / 복귀 ─────────────────────────────────
    // START/STOP 버튼 눌린 순간 즉시 피드백 (폴링 500ms 기다리지 않고) +
    // 실행 중일 때는 펄스 애니메이션으로 "실행 중" 명확히 표시
    function _syncStartStopBtn(startId, stopId, startLabel, running) {
      const startBtn = document.getElementById(startId);
      const stopBtn = document.getElementById(stopId);
      if (!startBtn || !stopBtn) return;
      startBtn.disabled = running;
      stopBtn.disabled = !running;
      if (running) {
        startBtn.textContent = "🟢 실행 중...";
        startBtn.classList.add("is-running");
      } else {
        startBtn.textContent = startLabel;
        startBtn.classList.remove("is-running");
      }
    }

    async function startAutopilot() {
      // Path Test 탭에도 동일한 설정 입력창이 있음 — 활성 탭 기준으로 읽음
      const suf = (activeTab === "verify") ? "-vfy" : "";
      const body = {
        mode: document.getElementById("drive-mode" + suf).value,
        instruction: document.getElementById("drive-instr" + suf).value,
        gt_object: document.getElementById("drive-gt" + suf).value,
        apply_cc: document.getElementById("drive-cc" + suf).checked
      };
      // 눌린 즉시 피드백 — 서버 응답(폴링) 기다리지 않음
      ["drive-start-btn", "quick-start-btn"].forEach(id => {
        const b = document.getElementById(id);
        if (b) { b.disabled = true; b.textContent = "⏳ 시작 중..."; }
      });
      const res = await api("/drive/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      if (!res.ok) {
        alert("Autopilot 시작 실패: " + res.error);
        _syncStartStopBtn("drive-start-btn", "drive-stop-btn", "▶ START DRIVE", false);
        _syncStartStopBtn("quick-start-btn", "quick-stop-btn", "▶️ START", false);
        return;
      }
      pollStatus();
    }

    async function stopAutopilot() {
      ["drive-stop-btn", "quick-stop-btn"].forEach(id => {
        const b = document.getElementById(id);
        if (b) b.disabled = true;
      });
      await api("/drive/stop", { method: "POST" });
      pollStatus();
      showSessionStack();
    }

    async function returnToStart() {
      const res = await api("/drive/return", { method: "POST" });
      alert(res.message);
    }

    // STOP(수동) 또는 목표 도달로 인한 자동 정지 직후 호출 — 지금까지
    // 저장된 세션들을 최근순으로 스택처럼 보여줘서 몇 개가 쌓였는지,
    // 방금 세션이 제대로 저장됐는지 바로 확인 가능하게 함.
    async function showSessionStack(n = 5) {
      try {
        const res = await api("/sessions/list");
        if (!res.ok || !res.sessions || !res.sessions.length) return;
        const panel = document.getElementById("session-stack-panel");
        const list  = document.getElementById("session-stack-list");
        if (!panel || !list) return;
        const top = res.sessions.slice(0, n);
        list.innerHTML = top.map((s, i) => {
          const isNewest = i === 0;
          return `<div style="display:flex; justify-content:space-between; gap:8px; padding:4px 6px; border-radius:4px;
                       background:${isNewest ? 'rgba(16,185,129,0.12)' : 'transparent'};
                       border-left:2px solid ${isNewest ? 'var(--emerald)' : 'var(--border-glow)'};">
                    <span style="color:${isNewest ? 'var(--emerald)' : 'var(--text-muted)'};">${isNewest ? '🆕 ' : ''}${s.sid}</span>
                    <span style="color:var(--text-muted);">${s.steps}steps · ${s.h5_size_mb}MB</span>
                  </div>`;
        }).join("");
        panel.style.display = "block";
      } catch(e) { /* 조용히 무시 — 상태 확인용 부가 UI */ }
    }

    // ── 수동 제어 / 캘리브레이션 ─────────────────────────────────────
    async function sendJoy(dir) {
      const speed = parseFloat(document.getElementById("joy-speed").value);
      try {
        const res = await api("/drive/manual", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ direction: dir, speed })
        });
        document.getElementById("calib-status-log").textContent = res.log || "보냄";
      } catch(e) {}
    }

    async function applyThreshold() {
      const val = parseFloat(document.getElementById("calib-thr").value);
      const res = await api("/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ stop_area_threshold: val })
      });
      if (res.ok) {
        document.getElementById("calib-status-log").textContent = "✅ 임계값 " + val + " 적용 성공";
      } else {
        alert("적용 실패: " + res.error);
      }
    }

    async function startCalibRec() {
      const res = await api("/calib/rec/start", { method: "POST" });
      document.getElementById("calib-start-rec-btn").disabled = true;
      document.getElementById("calib-stop-rec-btn").disabled = false;
      document.getElementById("calib-status-log").textContent = "⏺ 캘리브레이션 녹화 시작 (" + res.session + ")";
    }

    async function stopCalibRec() {
      const res = await api("/calib/rec/stop", { method: "POST" });
      document.getElementById("calib-start-rec-btn").disabled = false;
      document.getElementById("calib-stop-rec-btn").disabled = true;
      document.getElementById("calib-status-log").textContent = "⏹ 녹화 완료 (프레임 수: " + res.frames_count + ")";
    }

    async function snapCalibFrame() {
      const res = await api("/calib/rec/snap", { method: "POST" });
      document.getElementById("calib-status-log").textContent = "📸 스냅샷 추가됨 (총: " + res.frames_count + ")";
    }

    async function clearCalibRec() {
      await api("/calib/rec/clear", { method: "POST" });
      document.getElementById("calib-status-log").textContent = "🗑 캘리브레이션 임시 버퍼 초기화됨";
    }

    async function saveCalibRec() {
      const name = document.getElementById("calib-rec-name").value;
      const res = await api("/calib/rec/save?name=" + encodeURIComponent(name), { method: "POST" });
      if (res.ok) {
        document.getElementById("calib-status-log").textContent = "💾 저장 성공:\\nJSONL: " + res.jsonl + (res.mp4 ? "\\nMP4: " + res.mp4 : "");
        alert("성공적으로 저장되었습니다.");
      } else {
        alert("저장 실패: " + res.error);
      }
    }

    // ── 누적 드리프트 측정 ──────────────────────────────────────────
    async function runDriftMeasure() {
      const basis = document.getElementById("drift-basis-select").value;
      const res = await api("/drive/drift/run?basis=" + encodeURIComponent(basis));
      if (!res.ok) {
        document.getElementById("drift-log-console").textContent = "오류: " + res.error;
        return;
      }
      
      document.getElementById("drift-cnt").textContent = res.frame;
      document.getElementById("drift-lat").textContent = res.latency_ms.toFixed(0) + "ms";
      document.getElementById("drift-real").textContent = res.cum_real.toFixed(2) + "s";
      document.getElementById("drift-nom").textContent = res.cum_nom.toFixed(2) + "s";
      
      const elVal = document.getElementById("drift-val-display");
      elVal.textContent = "drift: " + res.drift.toFixed(2) + "s";
      if (res.drift >= 4.0) {
        elVal.className = "text-rose";
      } else if (res.drift >= 1.0) {
        elVal.className = "text-amber";
      } else {
        elVal.className = "text-emerald";
      }
      
      document.getElementById("drift-log-console").textContent = "저장 로그 경로: " + res.log_file + "\\n최근 측정값 기록 완료";
      updateDriftChart(res.history);
    }

    let driftAutoTimer = null;
    function toggleDriftAuto() {
      const btn = document.getElementById("drift-auto-btn");
      if (driftAutoTimer) {
        clearInterval(driftAutoTimer);
        driftAutoTimer = null;
        btn.textContent = "🔄 자동 (1fps)";
        btn.className = "btn btn-outline";
      } else {
        driftAutoTimer = setInterval(runDriftMeasure, 1000);
        btn.textContent = "⏸ 자동 정지";
        btn.className = "btn btn-rose";
      }
    }

    async function resetDrift() {
      await api("/drive/drift/reset", { method: "POST" });
      document.getElementById("drift-cnt").textContent = "0";
      document.getElementById("drift-lat").textContent = "0ms";
      document.getElementById("drift-real").textContent = "0.00s";
      document.getElementById("drift-nom").textContent = "0.00s";
      document.getElementById("drift-val-display").textContent = "drift: 0.00s";
      document.getElementById("drift-val-display").className = "text-emerald";
      initDriftChart();
    }

    function initDriftChart() {
      const ctx = document.getElementById("drift-chart").getContext("2d");
      if (driftChartInstance) driftChartInstance.destroy();
      
      driftChartInstance = new Chart(ctx, {
        type: 'line',
        data: {
          labels: [],
          datasets: [
            { label: '실제 누적시간', data: [], borderColor: '#f43f5e', tension: 0.1, fill: false },
            { label: '가정 누적시간', data: [], borderColor: '#64748b', borderDash: [5,5], tension: 0.1, fill: false }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            y: { grid: { color: '#1d2b45' }, ticks: { color: '#94a3b8' } },
            x: { grid: { color: '#1d2b45' }, ticks: { color: '#94a3b8' } }
          },
          plugins: { legend: { labels: { color: '#f1f5f9' } } }
        }
      });
    }

    function updateDriftChart(history) {
      if (!driftChartInstance) return;
      const labels = history.map(h => h[0]);
      const real = history.map(h => h[2]);
      const nom = history.map(h => h[3]);
      
      driftChartInstance.data.labels = labels;
      driftChartInstance.data.datasets[0].data = real;
      driftChartInstance.data.datasets[1].data = nom;
      driftChartInstance.update();
    }

    // Action Trajectory 차트는 경로 검증 탭으로 대체되어 제거되었습니다.
    // ── 🧪 경로 검증 (Path Test) 관련 전역 설정 ──
    const PATH_TYPES = [
      "right_right", "right_left", "right_straight",
      "center_straight", "center_left", "center_right",
      "left_straight", "left_left", "left_right",
      "obj_left", "obj_center", "obj_right",
      "dist_10cm", "dist_20cm", "dist_30cm",
      "trackA_weak_left_left_curve", "trackA_weak_left_straight", "trackA_weak_left_right_curve",
      "trackA_weak_right_left_curve", "trackA_weak_right_straight", "trackA_weak_right_right_curve",
      "trackA_strong_right_left_curve", "trackA_strong_right_straight", "trackA_strong_right_right_curve",
      "trackA_strong_left_left_curve", "trackA_strong_left_straight", "trackA_strong_left_right_curve",
      "trackF_center_left_curve", "trackF_center_straight", "trackF_center_right_curve"
    ];

    const PATH_TARGETS = {
      "right_right": 10, "right_left": 10, "right_straight": 10,
      "center_straight": 10, "center_left": 10, "center_right": 10,
      "left_straight": 10, "left_left": 10, "left_right": 10,
      "obj_left": 30, "obj_center": 30, "obj_right": 30,
      "dist_10cm": 10, "dist_20cm": 10, "dist_30cm": 10,
      "trackA_weak_left_left_curve": 15, "trackA_weak_left_straight": 15, "trackA_weak_left_right_curve": 15,
      "trackA_weak_right_left_curve": 15, "trackA_weak_right_straight": 15, "trackA_weak_right_right_curve": 15,
      "trackA_strong_right_left_curve": 15, "trackA_strong_right_straight": 15, "trackA_strong_right_right_curve": 15,
      "trackA_strong_left_left_curve": 15, "trackA_strong_left_straight": 15, "trackA_strong_left_right_curve": 15,
      "trackF_center_left_curve": 15, "trackF_center_straight": 15, "trackF_center_right_curve": 15,
    };

    // 트랙A 12종 + 트랙F 3종 전체 수집 완료(2026-07-16/17, 225/225) — 미수집 조합 없음
    const TRACKA_UNCOLLECTED = new Set([]);

    const PATH_GROUPS = [
      ["── 오브젝트 위치별 ──────────", ["obj_left","obj_center","obj_right"]],
      ["── 경로 검증 ──────────────", ["right_right","right_left","right_straight","center_straight","center_left","center_right","left_straight","left_left","left_right"]],
      ["── 박스 거리별 ──────────────", ["dist_10cm","dist_20cm","dist_30cm"]],
      ["── 🎯 트랙A 극단배치(V6) ──────", [
        "trackA_weak_left_left_curve", "trackA_weak_left_straight", "trackA_weak_left_right_curve",
        "trackA_weak_right_left_curve", "trackA_weak_right_straight", "trackA_weak_right_right_curve",
        "trackA_strong_right_left_curve", "trackA_strong_right_straight", "trackA_strong_right_right_curve",
        "trackA_strong_left_left_curve", "trackA_strong_left_straight", "trackA_strong_left_right_curve"
      ]],
      ["── 🎯 트랙F 중앙(V6) ──────", [
        "trackF_center_left_curve", "trackF_center_straight", "trackF_center_right_curve"
      ]]
    ];

    let runtimeState = {
      preview_enabled: true,
      preview_hint_cx: false,
      grounding_skip_n: 3,
      cx_jump_filter: false,
      cx_jump_thresh: 0.30,
      multi_prompt: true,
      stop_mode: "proximity",
      stop_learned_min_steps: 3
    };

    function selectPathType(type) {
      const select = document.getElementById("ep-path-type");
      if (select) {
        select.value = type;
      }
      syncVerifyPathType(type);
      if (typeof drawOverlay === "function") drawOverlay();
    }

    // 조이스틱 경로검증 모드(X/R1 즉시기록)가 쓸 path_type을 서버에 동기화.
    // select onchange + selectPathType(퀵라벨 버튼) 양쪽에서 호출됨.
    async function syncVerifyPathType(type) {
      try {
        await api("/verify/path_type", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ path_type: type })
        });
      } catch(e) { /* 조용히 무시 — 다음 선택 때 재시도됨 */ }
    }

    // 🕹️ 조이스틱 버튼 배치 모드 전환 — 📷 데이터수집 ⇄ 🧪 경로검증
    let joystickVerifyMode = false;
    async function toggleJoystickMode() {
      joystickVerifyMode = !joystickVerifyMode;
      const mode = joystickVerifyMode ? "verify" : "collect";
      try {
        await api("/joystick/mode", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ mode })
        });
      } catch(e) { /* 무시 — 버튼 텍스트는 로컬 상태로도 갱신됨 */ }
      document.querySelectorAll(".joystick-mode-btn").forEach(btn => {
        btn.textContent = joystickVerifyMode ? "🕹️ 조이스틱: 🧪 검증" : "🕹️ 조이스틱: 📷 수집";
        btn.style.borderColor = joystickVerifyMode ? "var(--amber)" : "";
        btn.style.color = joystickVerifyMode ? "var(--amber)" : "";
      });
    }

    // 검증모드 버튼 의미(2026-07-23 재설계) — 서버 btn_map name 기준
    const VERIFY_BTN_MEANING = {
      "L1": "▶ 추론 시작", "R1": "⏹ 추론 정지", "DISCARD": "✅ 성공 라벨",
      "STOP": "❌ 실패 라벨", "L2": "💾 기록 저장", "R2": "↩ 복귀",
      "START": "⚙ SYNC↔ASYNC", "Y": "🔁 모드 전환(수집⇄검증)",
    };

    // ── 🎯 추론 검증 스크리닝 패널 (데이터셋 목표와 별개) ──
    // 바구니 위치별 목표: 1차(빠른확인) vs 확정(논문용). episode_log의 trackA_/trackF_
    // path_type을 "위치"로 버킷팅해서 집계 — 경로(곡선방향)는 무시하고 cx 위치만.
    const SCREEN_POSITIONS = [
      {pos: "strong_left",  label: "강좌", icon: "◀◀", fg: "var(--cyan)"},
      {pos: "weak_left",    label: "약좌", icon: "◀",   fg: "var(--cyan)"},
      {pos: "center",       label: "중앙", icon: "●",   fg: "#3fb950"},
      {pos: "weak_right",   label: "약우", icon: "▶",   fg: "var(--amber)"},
      {pos: "strong_right", label: "강우", icon: "▶▶", fg: "var(--amber)"},
    ];
    const SCREEN_TARGETS = {
      "1차":  {strong_left:5, weak_left:3, center:3, weak_right:3, strong_right:5},
      "확정": {strong_left:15, weak_left:10, center:5, weak_right:10, strong_right:15},
    };
    let screenTargetMode = "1차";

    function verifyPosOf(pt) {
      // "trackA_strong_left_left_curve" → "strong_left", "trackF_center_straight" → "center"
      if (!pt) return null;
      let s = String(pt);
      if (!s.startsWith("trackA_") && !s.startsWith("trackF_")) return null;
      s = s.replace(/^track[AF]_/, "").replace(/_(left_curve|straight|right_curve)$/, "");
      return s;
    }

    function toggleScreenTarget() {
      screenTargetMode = (screenTargetMode === "1차") ? "확정" : "1차";
      const btn = document.getElementById("vfy-screen-toggle");
      if (btn) btn.textContent = screenTargetMode === "1차" ? "1차(빠른확인)" : "확정(논문용)";
      if (window._lastVfyRows) renderScreenPanel(window._lastVfyRows);
    }

    // ── 🔀 모델 전환 패널 — /verify/model_list + /verify/model_switch (2026-07-23) ──
    async function refreshModelList() {
      const listEl = document.getElementById("vfy-model-list");
      const statusEl = document.getElementById("vfy-model-status");
      if (listEl) listEl.innerHTML = "불러오는 중...";
      let current = "";
      try {
        const h = await api("/infer/health");
        current = (h.checkpoint_path || "").split("/").pop();
        window._currentCheckpointPath = h.checkpoint_path || "";
        const curEl = document.getElementById("vfy-model-current");
        if (curEl) curEl.textContent = "현재: " + (current || "—");
        const gEl = document.getElementById("vfy-grounder-current");
        if (gEl) gEl.textContent = (h.grounder && h.grounder.model) || "—";
      } catch (e) { /* 무시 */ }

      try {
        const res = await api("/verify/model_list");
        if (!res.ok || !listEl) return;
        listEl.innerHTML = (res.models || []).map(m => {
          const isCur = m.filename === current;
          const bits = [];
          if (m.head) bits.push(`head=${m.head}`);
          if (m.window) bits.push(`w=${m.window}`);
          if (m.bbox_scale) bits.push(`scale=${m.bbox_scale}`);
          if (m.val_acc != null) bits.push(`val_acc=${m.val_acc}`);
          if (m.held_success != null) bits.push(`held=${m.held_success}%`);
          if (m.stride) bits.push(`stride=${m.stride}`);
          bits.push(`${m.size_mb}MB`);
          const info = bits.join(" · ") + (m.error ? ` ⚠️${m.error}` : "");
          return `<div style="display:grid; grid-template-columns:auto 1fr; gap:8px; align-items:center; padding:4px 6px; border-radius:6px; background:${isCur ? "rgba(6,182,212,0.12)" : "#090d16"}; border:1px solid ${isCur ? "var(--cyan)" : "var(--border-glow)"};">
            <button class="btn ${isCur ? "btn-cyan" : "btn-outline"}" style="font-size:10px; padding:4px 8px; white-space:nowrap;" ${isCur ? "disabled" : ""} onclick="switchModel('${m.path}', '${m.filename}')">${isCur ? "✓ 로드됨" : "전환"}</button>
            <div style="font-size:9px; color:var(--text-muted); line-height:1.4;">
              <span style="color:#fff; font-weight:600;">${m.filename}</span><br>${info}
            </div>
          </div>`;
        }).join("");
      } catch (e) {
        if (listEl) listEl.innerHTML = "⚠️ 목록 조회 실패: " + e;
      }
    }

    async function switchModel(path, filename) {
      const statusEl = document.getElementById("vfy-model-status");
      if (statusEl) statusEl.textContent = `🔄 ${filename}로 전환 중...`;
      try {
        const res = await api("/verify/model_switch", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ path })
        });
        if (res.ok) {
          if (statusEl) statusEl.textContent = `✅ 전환 완료: ${filename} (head=${res.head}, val_acc=${res.val_acc})`;
        } else {
          if (statusEl) statusEl.textContent = "⚠️ 전환 실패: " + res.error;
        }
      } catch (e) {
        if (statusEl) statusEl.textContent = "⚠️ 오류: " + e;
      }
      refreshModelList();
    }

    async function switchGrounder(kind) {
      const statusEl = document.getElementById("vfy-model-status");
      const path = window._currentCheckpointPath;
      if (!path) { if (statusEl) statusEl.textContent = "⚠️ 현재 체크포인트 경로를 아직 모름 — 🔄 눌러서 새로고침 후 재시도"; return; }
      if (statusEl) statusEl.textContent = `🔄 그라운더 → ${kind} 전환 중... (첫 호출은 모델 로딩으로 느릴 수 있음)`;
      try {
        const res = await api("/verify/model_switch", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ path, grounder: kind })
        });
        if (res.ok) {
          if (statusEl) statusEl.textContent = `✅ 그라운더 전환 완료: ${res.grounder}`;
        } else {
          if (statusEl) statusEl.textContent = "⚠️ 전환 실패: " + res.error;
        }
      } catch (e) {
        if (statusEl) statusEl.textContent = "⚠️ 오류: " + e;
      }
      refreshModelList();
    }

    // A/B 필터 — 체크포인트별/시점별로 스크리닝 집계를 나눠보기 위함(2026-07-23).
    // session_id → checkpoint 매핑을 서버에서 받아와 episode_log 행(마지막 컬럼이
    // session_id)과 조인해서 필터링.
    window._checkpointIndex = window._checkpointIndex || {};

    async function refreshCheckpointOptions() {
      try {
        const res = await api("/verify/checkpoint_index");
        if (!res.ok) return;
        window._checkpointIndex = res.session_checkpoint || {};
        const sel = document.getElementById("vfy-screen-ckpt");
        if (!sel) return;
        const prevVal = sel.value;
        sel.innerHTML = '<option value="">전체(누적)</option>' +
          (res.checkpoints || []).map(c => `<option value="${c}">${c}</option>`).join("");
        if ([...sel.options].some(o => o.value === prevVal)) sel.value = prevVal;
      } catch (e) { /* 무시 */ }
      if (window._lastVfyRows) renderScreenPanel(window._lastVfyRows);
    }

    function _toDatetimeLocalValue(d) {
      const pad = n => String(n).padStart(2, "0");
      return `${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
    }

    function setScreenSinceNow() {
      const el = document.getElementById("vfy-screen-since");
      if (el) el.value = _toDatetimeLocalValue(new Date());
      if (window._lastVfyRows) renderScreenPanel(window._lastVfyRows);
    }

    function clearScreenSince() {
      const el = document.getElementById("vfy-screen-since");
      if (el) el.value = "";
      if (window._lastVfyRows) renderScreenPanel(window._lastVfyRows);
    }

    function renderScreenPanel(rows) {
      window._lastVfyRows = rows;
      const tgt = SCREEN_TARGETS[screenTargetMode];

      // A/B 필터 적용: 체크포인트 선택 + 검증 시작 시점
      const ckptSel = document.getElementById("vfy-screen-ckpt");
      const wantCkpt = ckptSel ? ckptSel.value : "";
      const sinceEl = document.getElementById("vfy-screen-since");
      const sinceDate = (sinceEl && sinceEl.value) ? new Date(sinceEl.value) : null;
      const idx = window._checkpointIndex || {};
      const filtered = (rows || []).filter(r => {
        if (r.length < 3) return false;
        if (wantCkpt) {
          const sid = r[r.length - 1];
          if ((idx[sid] || "") !== wantCkpt) return false;
        }
        if (sinceDate) {
          const dateStr = r[12]; // "YYYY-MM-DD HH:MM"
          const rowDate = dateStr ? new Date(String(dateStr).replace(" ", "T")) : null;
          if (!rowDate || isNaN(rowDate) || rowDate < sinceDate) return false;
        }
        return true;
      });

      const done = {}, succ = {};
      SCREEN_POSITIONS.forEach(p => { done[p.pos] = 0; succ[p.pos] = 0; });
      filtered.forEach(r => {
        const pos = verifyPosOf(String(r[1]).replace(/ ★/g, "").replace(/★/g, "").trim());
        if (pos && done[pos] !== undefined) {
          done[pos] += 1;
          if (r[2] === "성공") succ[pos] += 1;
        }
      });
      let totDone = 0, totTgt = 0, totSucc = 0;
      const rowsHtml = SCREEN_POSITIONS.map(p => {
        const d = done[p.pos], s = succ[p.pos], t = tgt[p.pos];
        totDone += d; totTgt += t; totSucc += s;
        const pct = Math.min(100, (d / t) * 100);
        const doneColor = d >= t ? "#3fb950" : "var(--text-muted)";
        return `<div style="display:grid; grid-template-columns:52px 1fr 62px; align-items:center; gap:6px; font-size:11px; padding:2px 0;">
          <span><span style="color:${p.fg}">${p.icon}</span> ${p.label}</span>
          <div style="background:#21262d; height:6px; border-radius:3px; overflow:hidden;">
            <div style="width:${pct.toFixed(0)}%; height:100%; background:${p.fg}; border-radius:3px;"></div>
          </div>
          <span style="text-align:right; font-family:var(--font-mono); color:${doneColor}">${d}/${t} <span style="color:#3fb950">✓${s}</span></span>
        </div>`;
      }).join("");
      const body = document.getElementById("vfy-screen-body");
      if (body) body.innerHTML = rowsHtml +
        `<div style="border-top:1px solid rgba(255,255,255,0.08); margin-top:6px; padding-top:4px; font-size:10px; color:var(--text-muted); text-align:right;">합계 ${totDone}/${totTgt} · 방향성공 ${totSucc}</div>`;
    }

    // ── 트랙A/트랙F 극단배치+중앙(V6) 퀵라벨 버튼 — Tab4/Tab6 공용 렌더러.
    // 데이터셋 히스토리 탭의 아이콘/색상 규칙(◀/▶/▶▶, ↰/↑/↱, ●) 재사용.
    // prefix로 trackA(극단 4위치)/trackF(중앙, 2026-07-22 추가) 구분 — 서로
    // 다른 target(15는 동일하나 별도 집계)이라 값 문자열이 겹치면 안 됨.
    const TRACKA_POS_ROWS = [
      {pos: "weak_left",    icon: "◀",  fg: "var(--cyan)",  prefix: "trackA"},
      {pos: "weak_right",   icon: "▶",  fg: "var(--amber)", prefix: "trackA"},
      {pos: "strong_right", icon: "▶▶", fg: "var(--amber)", prefix: "trackA"},
      {pos: "strong_left",  icon: "◀◀", fg: "var(--cyan)",  prefix: "trackA"},
      {pos: "center",       icon: "●",  fg: "var(--emerald, #3fb950)", prefix: "trackF"},
    ];
    const TRACKA_PATH_COLS = [
      {path: "left_curve", icon: "↰"},
      {path: "straight",   icon: "↑"},
      {path: "right_curve", icon: "↱"},
    ];

    function _renderTrackAPathButtons(containerId, btnClass, onclickFn) {
      const el = document.getElementById(containerId);
      if (!el) return;
      el.innerHTML = TRACKA_POS_ROWS.map(row => {
        const btns = TRACKA_PATH_COLS.map(col => {
          const pt = `${row.prefix}_${row.pos}_${col.path}`;
          const uncollected = TRACKA_UNCOLLECTED.has(pt);
          const disabledAttr = uncollected ? "disabled" : "";
          const title = uncollected ? "미수집 — 아직 이 조합의 데이터가 없음" : "";
          return `<button type="button" class="${btnClass}" data-path="${pt}" ${disabledAttr} title="${title}"
                    onclick="${onclickFn}('${pt}')"
                    style="font-size:10px; padding:4px 2px; opacity:${uncollected ? 0.4 : 1};">
                    <span style="color:${row.fg};">${row.icon}</span> ${col.icon}${uncollected ? " 🚫" : ""}
                  </button>`;
        }).join("");
        return `<div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:4px;">${btns}</div>`;
      }).join("");
    }

    // ── 🌀 오버슈트 가이드 (트랙C 수집 예시) ──────────────────────────────
    let osgState = { direction: "left_recover", frames: [], idx: 0, playing: false, playTimer: null };

    async function osgSetDirection(direction) {
      osgState.direction = direction;
      const leftBtn = document.getElementById("osg-btn-left");
      const rightBtn = document.getElementById("osg-btn-right");
      if (leftBtn) leftBtn.className = direction === "left_recover" ? "btn btn-cyan" : "btn btn-outline";
      if (rightBtn) rightBtn.className = direction === "right_recover" ? "btn btn-cyan" : "btn btn-outline";
      const res = await api(`/overshoot_guide/load?direction=${direction}`);
      if (!res.ok) {
        document.getElementById("osg-info-note").textContent = "로딩 실패: " + (res.error || "");
        return;
      }
      osgState.frames = res.frames;
      osgState.idx = 0;
      document.getElementById("osg-source-sid").textContent = res.source_sid;
      const scrub = document.getElementById("osg-scrub");
      scrub.max = res.frames.length - 1;
      osgRenderFrame();
    }

    function osgRenderTimeline() {
      const el = document.getElementById("osg-timeline");
      if (!el) return;
      el.innerHTML = osgState.frames.map((f, i) => {
        const bg = f.phase === "real" ? "var(--cyan)" : "var(--amber)";
        const outline = i === osgState.idx ? "outline:2px solid #fff;" : "";
        return `<div onclick="osgSeek(${i})" style="flex:1; background:${bg}; opacity:${i === osgState.idx ? 1 : 0.5}; cursor:pointer; border-radius:2px; ${outline}"></div>`;
      }).join("");
    }

    function osgRenderFrame() {
      const f = osgState.frames[osgState.idx];
      if (!f) return;
      document.getElementById("osg-frame-img").src = `/overshoot_guide/frame?direction=${osgState.direction}&idx=${f.idx}`;
      document.getElementById("osg-phase-badge").textContent = f.phase === "real" ? "🟦 실제" : "🟧 합성(좌우반전)";
      document.getElementById("osg-idx-badge").textContent = `Frame: ${osgState.idx + 1}/${osgState.frames.length}`;
      document.getElementById("osg-info-phase").value = f.phase === "real" ? "실제 데이터" : "합성 예시(실제 아님)";
      document.getElementById("osg-info-realidx").value = f.source_real_idx;
      document.getElementById("osg-info-cx").value = f.cx.toFixed(3);
      document.getElementById("osg-info-action").value = f.action;
      document.getElementById("osg-info-note").textContent = f.note;
      document.getElementById("osg-scrub").value = osgState.idx;
      osgRenderTimeline();
    }

    function osgStep(delta) {
      if (!osgState.frames.length) return;
      osgState.idx = Math.max(0, Math.min(osgState.frames.length - 1, osgState.idx + delta));
      osgRenderFrame();
    }

    function osgSeek(v) {
      osgState.idx = parseInt(v);
      osgRenderFrame();
    }

    function osgTogglePlay() {
      const btn = document.getElementById("osg-play-btn");
      if (osgState.playing) {
        clearInterval(osgState.playTimer);
        osgState.playing = false;
        btn.textContent = "▶ 재생";
        return;
      }
      osgState.playing = true;
      btn.textContent = "⏸ 정지";
      osgState.playTimer = setInterval(() => {
        if (osgState.idx >= osgState.frames.length - 1) { osgTogglePlay(); return; }
        osgStep(1);
      }, 400);
    }

    function setFpeValue(val) {
      const fpeRange = document.getElementById("ep-fpe");
      const fpeLbl = document.getElementById("ep-fpe-lbl");
      if (fpeRange) {
        fpeRange.value = val;
        if (fpeLbl) {
          fpeLbl.textContent = 'False Positive Error (FPE): ' + val;
        }
      }
    }

    function setEpResult(val) {
      document.getElementById("ep-success").value = val;
      const succBtn = document.getElementById("ep-result-succ");
      const failBtn = document.getElementById("ep-result-fail");
      succBtn.className = val === "성공" ? "btn btn-cyan" : "btn btn-outline";
      failBtn.className = val === "실패" ? "btn btn-rose" : "btn btn-outline";
    }

    async function syncVerifyRuntimeParams() {
      try {
        const statusEl = document.getElementById("vfy-rt-status");
        statusEl.textContent = "동기화 중...";
        
        const res = await api("/infer/health");
        if (res.status === "error") {
          statusEl.textContent = "⚠️ 동기화 실패: " + res.detail;
          return;
        }
        
        // /health의 preview 상태는 최상위가 아니라 preview.{enabled,hint_cx}에 중첩됨 —
        // 예전엔 res.preview_enabled(항상 undefined)를 읽어서 프리뷰=항상 ON,
        // hint=항상 OFF로 표시되던 버그가 있었음 (2026-07-02 B4)
        const prev = res.preview || {};
        runtimeState.preview_enabled = prev.enabled !== false;
        runtimeState.preview_hint_cx = prev.hint_cx === true;
        runtimeState.grounding_skip_n = res.grounding_skip_n !== undefined ? parseInt(res.grounding_skip_n) : 3;
        runtimeState.cx_jump_filter = res.cx_jump_filter === true;
        runtimeState.cx_jump_thresh = res.cx_jump_thresh !== undefined ? parseFloat(res.cx_jump_thresh) : 0.30;
        runtimeState.multi_prompt = res.multi_prompt !== false;
        runtimeState.stop_mode = res.stop_mode || "proximity";
        runtimeState.stop_learned_min_steps = res.stop_learned_min_steps !== undefined ? parseInt(res.stop_learned_min_steps) : 3;

        const g = res.grounder || {};
        const owlRow = document.getElementById("vfy-owl-row");
        if ((g.model || "").toLowerCase().includes("owl")) {
          owlRow.style.display = "flex";
          document.getElementById("vfy-owl-area-scale").value = g.owlv2_area_scale ?? 3.0;
        } else {
          owlRow.style.display = "none";
        }

        updateVerifyRuntimeUI();
        statusEl.textContent = "✅ 동기화 완료";
      } catch(e) {
        document.getElementById("vfy-rt-status").textContent = "⚠️ 동기화 오류: " + e;
      }
    }

    function updateVerifyRuntimeUI() {
      // preview
      const previewBtn = document.getElementById("vfy-rt-preview");
      if (runtimeState.preview_enabled) {
        previewBtn.className = "btn btn-cyan";
        previewBtn.innerHTML = "🟢 프리뷰<br><span style='font-size:9px;color:#000;'>격리회전 ON</span>";
      } else {
        previewBtn.className = "btn btn-outline";
        previewBtn.innerHTML = "⚫ 프리뷰<br><span style='font-size:9px;color:var(--text-muted);'>격리회전 OFF</span>";
      }
      
      // hint
      const hintBtn = document.getElementById("vfy-rt-hint");
      if (runtimeState.preview_hint_cx) {
        hintBtn.className = "btn btn-cyan";
        hintBtn.innerHTML = "🟢 hint_cx<br><span style='font-size:9px;color:#000;'>방향회전 ON</span>";
      } else {
        hintBtn.className = "btn btn-outline";
        hintBtn.innerHTML = "⚫ hint_cx<br><span style='font-size:9px;color:var(--text-muted);'>방향회전 OFF</span>";
      }
      
      // skip
      const skipBtn = document.getElementById("vfy-rt-skip");
      if (runtimeState.grounding_skip_n === 1) {
        skipBtn.className = "btn btn-cyan";
        skipBtn.innerHTML = "📦 skip 1<br><span style='font-size:9px;color:#000;'>매프레임</span>";
      } else {
        skipBtn.className = "btn btn-outline";
        skipBtn.innerHTML = "📦 skip 3<br><span style='font-size:9px;color:var(--text-muted);'>캐시 사용</span>";
      }
      
      // jump
      const jumpBtn = document.getElementById("vfy-rt-jump");
      const thrBtn = document.getElementById("vfy-rt-thr");
      if (runtimeState.cx_jump_filter) {
        jumpBtn.className = "btn btn-cyan";
        jumpBtn.innerHTML = "🟢 P2필터<br><span style='font-size:9px;color:#000;'>오탐제거 ON</span>";
        
        thrBtn.disabled = false;
        thrBtn.style.opacity = "1";
        thrBtn.className = "btn btn-outline";
        thrBtn.innerHTML = `🎚 민감도<br><span style='font-size:9px;'>${runtimeState.cx_jump_thresh.toFixed(2)}</span>`;
      } else {
        jumpBtn.className = "btn btn-outline";
        jumpBtn.innerHTML = "⚫ P2필터<br><span style='font-size:9px;color:var(--text-muted);'>오탐제거 OFF</span>";
        
        thrBtn.disabled = true;
        thrBtn.style.opacity = "0.5";
        thrBtn.className = "btn btn-outline";
        thrBtn.innerHTML = "🎚 민감도<br><span style='font-size:9px;color:var(--text-muted);'>(P2 꺼짐)</span>";
      }

      // multi-prompt
      const multiBtn = document.getElementById("vfy-rt-multi");
      if (runtimeState.multi_prompt) {
        multiBtn.className = "btn btn-cyan";
        multiBtn.innerHTML = "🟢 멀티프롬프트<br><span style='font-size:9px;color:#000;'>fallback ON</span>";
      } else {
        multiBtn.className = "btn btn-outline";
        multiBtn.innerHTML = "⚫ 멀티프롬프트<br><span style='font-size:9px;color:var(--text-muted);'>fallback OFF</span>";
      }

      // stop mode — proximity(안전, 3프레임 연속 임계값) vs learned(모델 예측+래치)
      const stopModeBtn = document.getElementById("vfy-rt-stopmode");
      if (stopModeBtn) {
        if (runtimeState.stop_mode === "learned") {
          stopModeBtn.className = "btn btn-outline";
          stopModeBtn.style.borderColor = "var(--amber)";
          stopModeBtn.style.color = "var(--amber)";
          stopModeBtn.innerHTML = "🛑 learned<br><span style='font-size:9px;color:var(--amber);'>STOP 모드 (래치, 주의)</span>";
        } else {
          stopModeBtn.style.borderColor = "";
          stopModeBtn.style.color = "";
          stopModeBtn.className = "btn btn-cyan";
          stopModeBtn.innerHTML = "🛑 proximity<br><span style='font-size:9px;color:#000;'>STOP 모드 (안전)</span>";
        }
      }

      // stop guard — learned 모드 콜드스타트 최소 프레임(0=가드 없음, 예전 동작과 동일)
      const stopGuardBtn = document.getElementById("vfy-rt-stopguard");
      if (stopGuardBtn) {
        const n = runtimeState.stop_learned_min_steps;
        stopGuardBtn.className = n > 0 ? "btn btn-outline" : "btn btn-outline";
        stopGuardBtn.innerHTML = n > 0
          ? `🛡 가드 ${n}프레임<br><span style='font-size:9px;color:var(--text-muted)'>learned 콜드스타트</span>`
          : `🚫 가드 없음<br><span style='font-size:9px;color:var(--text-muted)'>learned 콜드스타트</span>`;
      }
    }

    async function toggleVerifyRuntime(param) {
      if (param === 'preview') {
        runtimeState.preview_enabled = !runtimeState.preview_enabled;
      } else if (param === 'hint') {
        runtimeState.preview_hint_cx = !runtimeState.preview_hint_cx;
      } else if (param === 'skip') {
        runtimeState.grounding_skip_n = runtimeState.grounding_skip_n === 3 ? 1 : 3;
      } else if (param === 'jump') {
        runtimeState.cx_jump_filter = !runtimeState.cx_jump_filter;
      } else if (param === 'thr') {
        if (!runtimeState.cx_jump_filter) return;
        const thresholds = [0.20, 0.30, 0.40, 0.50];
        let idx = thresholds.indexOf(runtimeState.cx_jump_thresh);
        if (idx === -1) idx = 1;
        runtimeState.cx_jump_thresh = thresholds[(idx + 1) % thresholds.length];
      } else if (param === 'multi') {
        runtimeState.multi_prompt = !runtimeState.multi_prompt;
      } else if (param === 'stopmode') {
        runtimeState.stop_mode = runtimeState.stop_mode === "proximity" ? "learned" : "proximity";
      } else if (param === 'stopguard') {
        const steps = [0, 1, 3, 5, 10];
        let idx = steps.indexOf(runtimeState.stop_learned_min_steps);
        if (idx === -1) idx = 2;
        runtimeState.stop_learned_min_steps = steps[(idx + 1) % steps.length];
      }

      updateVerifyRuntimeUI();

      try {
        const res = await api("/config", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            preview_enabled: runtimeState.preview_enabled,
            preview_hint_cx: runtimeState.preview_hint_cx,
            grounding_skip_n: runtimeState.grounding_skip_n,
            cx_jump_filter: runtimeState.cx_jump_filter,
            cx_jump_thresh: runtimeState.cx_jump_thresh,
            multi_prompt: runtimeState.multi_prompt,
            stop_mode: runtimeState.stop_mode,
            stop_learned_min_steps: runtimeState.stop_learned_min_steps
          })
        });
        if (res.ok) {
          // "적용 완료"라는 낙관 문구 대신 추론 서버가 실제로 적용했다고
          // 응답한 값을 그대로 표시 — 토글이 조용히 무시된(applied 비어있음)
          // 케이스를 눈으로 바로 잡기 위함 (2026-07-02 B4)
          const inner = (res.applied && res.applied.applied) || {};
          const parts = Object.entries(inner).map(([k, v]) => `${k}=${v}`);
          if (parts.length > 0) {
            document.getElementById("vfy-rt-status").textContent = "✅ 서버 적용: " + parts.join(", ");
          } else {
            document.getElementById("vfy-rt-status").textContent = "⚠️ 서버가 아무것도 적용 안 함 (applied 비어있음) — 필드명/프록시 확인";
          }
          // 서버 실제 상태로 UI 재동기화 (로컬 추측값과의 불일치 방지)
          setTimeout(syncVerifyRuntimeParams, 800);
        } else {
          document.getElementById("vfy-rt-status").textContent = "⚠️ 적용 실패: " + res.error;
        }
      } catch(e) {
        document.getElementById("vfy-rt-status").textContent = "⚠️ 서버 오류: " + e;
      }
    }

    async function applyVerifyOwlAreaScale() {
      const statusEl = document.getElementById("vfy-rt-status");
      const v = parseFloat(document.getElementById("vfy-owl-area-scale").value);
      if (isNaN(v) || v <= 0) { statusEl.textContent = "⚠️ 보정계수 값이 올바르지 않음"; return; }
      statusEl.textContent = "OWL 보정계수 적용 중...";
      try {
        const res = await api("/config", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ owlv2_area_scale: v })
        });
        if (res.ok) {
          const inner = (res.applied && res.applied.applied) || {};
          const parts = Object.entries(inner).map(([k, v2]) => `${k}=${v2}`);
          statusEl.textContent = parts.length > 0
            ? "✅ 서버 적용: " + parts.join(", ")
            : "⚠️ 서버가 아무것도 적용 안 함 (applied 비어있음)";
        } else {
          statusEl.textContent = "⚠️ 적용 실패: " + res.error;
        }
      } catch(e) {
        statusEl.textContent = "⚠️ 서버 오류: " + e;
      }
    }

    function updatePathSummary(rows) {
      const done_total = {};
      const done_succ = {};
      PATH_TYPES.forEach(k => {
        done_total[k] = 0;
        done_succ[k] = 0;
      });
      
      let nav_done = 0;
      let nav_succ = 0;

      rows.forEach(r => {
        if (r.length < 3) return;
        const pt = String(r[1]).replace(/ ★/g, "").replace(/★/g, "").trim();
        if (done_total[pt] !== undefined) {
          done_total[pt] += 1;
        }
        if (r[2] === "성공") {
          if (done_succ[pt] !== undefined) {
            done_succ[pt] += 1;
          }
          if (!pt.startsWith("obj_") && !pt.startsWith("dist_") && !pt.startsWith("trackA_") && !pt.startsWith("trackF_")) {
            nav_succ += 1;
          }
        }
        if (PATH_TARGETS[pt] !== undefined && !pt.startsWith("obj_") && !pt.startsWith("dist_") && !pt.startsWith("trackA_") && !pt.startsWith("trackF_")) {
          nav_done += 1;
        }
      });

      const nav_total = 90; // 9 routes * 10 targets = 90
      const obj_done = (done_total["obj_left"] || 0) + (done_total["obj_center"] || 0) + (done_total["obj_right"] || 0);
      const obj_succ = (done_succ["obj_left"] || 0) + (done_succ["obj_center"] || 0) + (done_succ["obj_right"] || 0);
      const dist_done = (done_total["dist_10cm"] || 0) + (done_total["dist_20cm"] || 0) + (done_total["dist_30cm"] || 0);
      const dist_succ = (done_succ["dist_10cm"] || 0) + (done_succ["dist_20cm"] || 0) + (done_succ["dist_30cm"] || 0);
      // 트랙A/트랙F(V6)는 기존 집계와 목표수(15 vs 10)가 달라 완전히 분리 집계
      const trackAKeys = PATH_TYPES.filter(k => k.startsWith("trackA_"));
      const trackA_total = trackAKeys.reduce((s, k) => s + (PATH_TARGETS[k] || 0), 0);
      const trackA_done = trackAKeys.reduce((s, k) => s + (done_total[k] || 0), 0);
      const trackA_succ = trackAKeys.reduce((s, k) => s + (done_succ[k] || 0), 0);
      const trackFKeys = PATH_TYPES.filter(k => k.startsWith("trackF_"));
      const trackF_total = trackFKeys.reduce((s, k) => s + (PATH_TARGETS[k] || 0), 0);
      const trackF_done = trackFKeys.reduce((s, k) => s + (done_total[k] || 0), 0);
      const trackF_succ = trackFKeys.reduce((s, k) => s + (done_succ[k] || 0), 0);

      const total_done = rows.length;
      const total_target = 210; // 90 nav + 90 obj + 30 dist = 210 (트랙A/F 별개 집계, 미포함)

      const pct_total = Math.min(100.0, Math.max(0.0, (total_done / total_target) * 100));
      const pct_nav   = Math.min(100.0, Math.max(0.0, (nav_done / nav_total) * 100));
      const pct_obj   = Math.min(100.0, Math.max(0.0, (obj_done / 90) * 100));
      const pct_dist  = Math.min(100.0, Math.max(0.0, (dist_done / 30) * 100));
      const pct_trackA = Math.min(100.0, Math.max(0.0, (trackA_done / trackA_total) * 100));
      const pct_trackF = Math.min(100.0, Math.max(0.0, (trackF_done / trackF_total) * 100));
      
      const progressHtml = `
      <div style="width: 100%; box-sizing: border-box; padding: 2px 0;">
        <div style="margin-bottom: 10px; background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 8px;">
          <div style="display: flex; justify-content: space-between; font-size: 11px; color: #c9d1d9; margin-bottom: 4px; font-weight: bold;">
            <span>🌐 전체 수집 완료도 (목표 ${total_target} ep)</span>
            <span style="color: var(--cyan);">${pct_total.toFixed(1)}% (${total_done}/${total_target} ep)</span>
          </div>
          <div style="width: 100%; background-color: #21262d; height: 10px; border-radius: 5px; overflow: hidden; box-shadow: inset 0 1px 3px rgba(0,0,0,0.5);">
            <div style="width: ${pct_total.toFixed(1)}%; height: 100%; background: linear-gradient(90deg, #1f6feb 0%, #388bfd 100%); border-radius: 5px; transition: width 0.3s ease;"></div>
          </div>
        </div>
        
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 6px;">
          <div style="background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 6px;">
            <div style="font-size: 9px; margin-bottom: 2px; display: flex; justify-content: space-between; flex-wrap: wrap;">
              <span style="color: #58a6ff; font-weight: 600;">🛣️ 경로 검증</span>
              <span style="color: #8b949e;">${nav_done}/${nav_total} (${nav_succ}✓)</span>
            </div>
            <div style="width: 100%; background-color: #21262d; height: 6px; border-radius: 3px; overflow: hidden;">
              <div style="width: ${pct_nav.toFixed(1)}%; height: 100%; background: linear-gradient(90deg, #1f6feb 0%, #58a6ff 100%); border-radius: 3px; transition: width 0.3s ease;"></div>
            </div>
          </div>

          <div style="background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 6px;">
            <div style="font-size: 9px; margin-bottom: 2px; display: flex; justify-content: space-between; flex-wrap: wrap;">
              <span style="color: #3fb950; font-weight: 600;">🎯 위치별</span>
              <span style="color: #8b949e;">${obj_done}/90 (${obj_succ}✓)</span>
            </div>
            <div style="width: 100%; background-color: #21262d; height: 6px; border-radius: 3px; overflow: hidden;">
              <div style="width: ${pct_obj.toFixed(1)}%; height: 100%; background: linear-gradient(90deg, #238636 0%, #3fb950 100%); border-radius: 3px; transition: width 0.3s ease;"></div>
            </div>
          </div>

          <div style="background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 6px;">
            <div style="font-size: 9px; margin-bottom: 2px; display: flex; justify-content: space-between; flex-wrap: wrap;">
              <span style="color: #a371f7; font-weight: 600;">📦 거리별</span>
              <span style="color: #8b949e;">${dist_done}/30 (${dist_succ}✓)</span>
            </div>
            <div style="width: 100%; background-color: #21262d; height: 6px; border-radius: 3px; overflow: hidden;">
              <div style="width: ${pct_dist.toFixed(1)}%; height: 100%; background: linear-gradient(90deg, #8957e5 0%, #a371f7 100%); border-radius: 3px; transition: width 0.3s ease;"></div>
            </div>
          </div>

          <div style="background: #161b22; border: 1px solid var(--amber); border-radius: 6px; padding: 6px;">
            <div style="font-size: 9px; margin-bottom: 2px; display: flex; justify-content: space-between; flex-wrap: wrap;">
              <span style="color: var(--amber); font-weight: 600;">🎯 트랙A(V6)</span>
              <span style="color: #8b949e;">${trackA_done}/${trackA_total} (${trackA_succ}✓)</span>
            </div>
            <div style="width: 100%; background-color: #21262d; height: 6px; border-radius: 3px; overflow: hidden;">
              <div style="width: ${pct_trackA.toFixed(1)}%; height: 100%; background: linear-gradient(90deg, #b45309 0%, #f59e0b 100%); border-radius: 3px; transition: width 0.3s ease;"></div>
            </div>
          </div>

          <div style="background: #161b22; border: 1px solid #3fb950; border-radius: 6px; padding: 6px;">
            <div style="font-size: 9px; margin-bottom: 2px; display: flex; justify-content: space-between; flex-wrap: wrap;">
              <span style="color: #3fb950; font-weight: 600;">● 트랙F(V6)</span>
              <span style="color: #8b949e;">${trackF_done}/${trackF_total} (${trackF_succ}✓)</span>
            </div>
            <div style="width: 100%; background-color: #21262d; height: 6px; border-radius: 3px; overflow: hidden;">
              <div style="width: ${pct_trackF.toFixed(1)}%; height: 100%; background: linear-gradient(90deg, #238636 0%, #3fb950 100%); border-radius: 3px; transition: width 0.3s ease;"></div>
            </div>
          </div>
        </div>
      </div>
      `;
      document.getElementById("vfy-progress-wrapper").innerHTML = progressHtml;

      document.getElementById("vfy-progress-txt").innerHTML = `
        경로검증 ${nav_done}/${nav_total} ep 성공 ${nav_succ}/20 (목표)<br>
        위치별 ${obj_done}/90 (${obj_succ} 성공) | 거리별 ${dist_done}/30 (${dist_succ} 성공) |
        트랙A ${trackA_done}/${trackA_total} (${trackA_succ} 성공) | 트랙F ${trackF_done}/${trackF_total} (${trackF_succ} 성공)
      `;

      renderScreenPanel(rows);

      let tblHtml = "";
      PATH_GROUPS.forEach(group => {
        const header = group[0];
        const keys = group[1];
        tblHtml += `<tr style="background:#1c2638; font-weight:bold;"><td colspan="6" style="padding:4px 6px; color:var(--text-muted); font-size:11px;">${header}</td></tr>`;
        keys.forEach(pt => {
          const pt_display = pt + (pt === "right_left" ? " ★" : "");
          const target = PATH_TARGETS[pt];
          const total = done_total[pt] || 0;
          const succ = done_succ[pt] || 0;
          const fail = total - succ;
          const rate = total > 0 ? Math.round((succ / total) * 100) + "%" : "—";
          const is_done = total >= target;
          const rowStyle = is_done ? "color:var(--text-muted); opacity:0.75;" : "";

          tblHtml += `
          <tr style="border-bottom:1px solid rgba(29,43,69,0.5); ${rowStyle}">
            <td style="padding:6px; font-family:var(--font-mono); color:#58a6ff; cursor:pointer;" onclick="selectPathType('${pt}')">${pt_display}</td>
            <td style="padding:6px 4px;">${target}</td>
            <td style="padding:6px 4px;">${total}</td>
            <td style="padding:6px 4px; color:var(--rose); font-weight:bold;">${fail}</td>
            <td style="padding:6px 4px; color:var(--emerald); font-weight:bold;">${succ}</td>
            <td style="padding:6px 4px; color:var(--cyan); font-weight:bold;">${rate}</td>
          </tr>
          `;
        });
      });
      document.getElementById("vfy-summary-table-body").innerHTML = tblHtml;
    }

    // ── H5 세션 리스트 & 인스펙터 ─────────────────────────────────────
    let _activeSid = null;

    function _sidToDateLabel(sid) {
      // sid: YYYYMMDD_HHMMSS
      const m = sid.match(/^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})$/);
      if (!m) return { date: sid, time: "" };
      return { date: `${m[1]}-${m[2]}-${m[3]}`, time: `${m[4]}:${m[5]}` };
    }

    async function loadSessionList() {
      const res = await api("/sessions/list");
      const listEl = document.getElementById("session-list-group");
      if (res.sessions.length === 0) {
        listEl.innerHTML = "<div style='text-align:center;color:var(--text-muted);font-size:13px;padding:20px 0;'>H5 세션 파일 없음</div>";
        return;
      }

      let html = "";
      let lastDate = null;
      res.sessions.forEach(s => {
        const { date, time } = _sidToDateLabel(s.sid);
        if (date !== lastDate) {
          html += `<div class="session-date-header">${date}</div>`;
          lastDate = date;
        }
        const labeledCls = s.labeled_count > 0 ? "labeled" : "";
        const activeCls = s.sid === _activeSid ? "active" : "";
        html += `
          <div class="session-card ${activeCls}" data-sid="${s.sid}" onclick="loadSessionDetail('${s.sid}')">
            <div class="sc-top">
              <span class="sc-sid">${s.sid}</span>
              <span class="sc-time">${time}</span>
            </div>
            <div class="sc-entity" title="${s.instruction}">${s.instruction}</div>
            <div class="sc-bottom">
              <span class="sc-badge">Steps ${s.steps}</span>
              <span class="sc-badge ${labeledCls}">${s.labeled_count} Labeled</span>
            </div>
          </div>
        `;
      });
      listEl.innerHTML = html;
    }

    async function loadSessionDetail(sid) {
      _activeSid = sid;
      document.querySelectorAll("#session-list-group .session-card").forEach(el => {
        el.classList.toggle("active", el.dataset.sid === sid);
      });
      document.getElementById("inspect-sid-lbl").textContent = "[" + sid + "]";
      document.getElementById("inspector-placeholder").style.display = "none";
      document.getElementById("inspector-body").style.display = "grid";
      
      const res = await api("/sessions/load?sid=" + sid);
      if (!res.ok) {
        alert("세션 로드 실패: " + res.error);
        return;
      }
      
      inspectSession = res;

      // 슬라이더 초기화
      const slider = document.getElementById("inspect-slider");
      slider.max = res.frames.length - 1;
      slider.value = 0;

      // 속성 노출
      let attrText = "";
      for (const k in res.attrs) {
        attrText += `${k}: ${res.attrs[k]}\n`;
      }
      document.getElementById("inspect-attrs-lbl").textContent = attrText || "속성 없음";

      renderInspectRuntimeConfig(res.attrs);
      renderInspectSummary(res.frames);
      renderInspectEpisode(res.episode);
      renderInspectTimeline(res.frames, 0);
      showInspectFrame(0);
    }

    // ── 경로검증(episode_log) 기록 표시/수정 — session_id로 Tab 4 기록과 연결 ──
    let _inspEpisode = null;

    function renderInspectEpisode(episode) {
      _inspEpisode = episode;
      document.getElementById("inspect-episode-box").style.display = "block";
      document.getElementById("inspect-ep-edit").style.display = "none";
      document.getElementById("inspect-ep-status").textContent = "—";
      const view = document.getElementById("inspect-ep-view");
      const empty = document.getElementById("inspect-ep-empty");
      if (!episode) {
        view.style.display = "none";
        empty.style.display = "block";
        return;
      }
      view.style.display = "block";
      empty.style.display = "none";
      document.getElementById("iep-path").textContent = episode.path_type;
      document.getElementById("iep-success").textContent = episode.success;
      document.getElementById("iep-steps").textContent = episode.steps;
      document.getElementById("iep-lat").textContent = episode.lat_ms + "ms";
      document.getElementById("iep-topaction").textContent = episode.top_action;
      document.getElementById("iep-gnd").textContent = episode.gnd_pct + "%";
      document.getElementById("iep-area").textContent = episode.area;
      document.getElementById("iep-cx").textContent = episode.cx;
      document.getElementById("iep-stop").textContent = episode.stop;
      document.getElementById("iep-fpe").textContent = episode.fpe;
      document.getElementById("iep-date").textContent = episode.date;
      document.getElementById("iep-note").textContent = episode.note || "—";

      document.getElementById("iep-success-tile").className = "mini-tile " + (episode.success === "성공" ? "mt-good" : "mt-bad");
      document.getElementById("iep-stop-tile").className = "mini-tile " + (episode.stop === "Y" ? "mt-accent" : "");
    }

    let _inspEditPath = null;
    let _inspEditSuccess = null;

    function selectInspectPathType(pt) {
      _inspEditPath = pt;
      document.querySelectorAll(".iep-path-btn").forEach(btn => {
        btn.className = btn.dataset.path === pt ? "btn btn-cyan iep-path-btn" : "btn btn-outline iep-path-btn";
      });
    }

    function setInspectEpResult(val) {
      _inspEditSuccess = val;
      document.getElementById("iep-edit-succ-btn").className = val === "성공" ? "btn btn-cyan" : "btn btn-outline";
      document.getElementById("iep-edit-fail-btn").className = val === "실패" ? "btn btn-rose" : "btn btn-outline";
    }

    function setInspectFpeValue(val) {
      document.getElementById("iep-edit-fpe").value = val;
      document.getElementById("iep-edit-fpe-lbl").textContent = "FPE: " + val;
    }

    function toggleInspectEpEdit() {
      if (!_inspEpisode) return;
      const editBox = document.getElementById("inspect-ep-edit");
      if (editBox.style.display !== "none") { editBox.style.display = "none"; return; }
      selectInspectPathType(_inspEpisode.path_type);
      setInspectEpResult(_inspEpisode.success);
      setInspectFpeValue(_inspEpisode.fpe);
      document.getElementById("iep-edit-steps").value = _inspEpisode.steps;
      document.getElementById("iep-edit-lat").value = _inspEpisode.lat_ms;
      document.getElementById("iep-edit-note").value = _inspEpisode.note || "";
      editBox.style.display = "block";
    }

    async function saveInspectEpisode() {
      if (!_inspEpisode) return;
      const body = {
        row: parseInt(_inspEpisode.row),
        path_type: _inspEditPath || _inspEpisode.path_type,
        success: _inspEditSuccess || _inspEpisode.success,
        steps: parseInt(document.getElementById("iep-edit-steps").value) || 0,
        lat_ms: parseFloat(document.getElementById("iep-edit-lat").value) || 0,
        fpe: parseFloat(document.getElementById("iep-edit-fpe").value) || 0,
        note: document.getElementById("iep-edit-note").value,
      };
      const res = await api("/episodes/update", { method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify(body) });
      const statusEl = document.getElementById("inspect-ep-status");
      if (res.ok) {
        Object.assign(_inspEpisode, body);
        renderInspectEpisode(_inspEpisode);
        document.getElementById("inspect-ep-status").textContent = `✅ 저장됨 (${new Date().toLocaleTimeString()})`;
      } else {
        statusEl.textContent = "⚠️ " + (res.error || "저장 실패");
      }
    }

    function renderInspectRuntimeConfig(attrs) {
      const box = document.getElementById("inspect-runtime-cfg-box");
      const grid = document.getElementById("inspect-runtime-cfg-grid");
      let cfg = {};
      const raw = attrs && attrs.runtime_config;
      if (raw) {
        try { cfg = JSON.parse(raw); } catch (e) { cfg = {}; }
      }

      // 경로검증(Tab 4) 당시 Config 패널 값 — H5 attrs에 직접 기록됨(instruction_mode/gt_object/apply_cc).
      // 서버 스냅샷(cfg)과 달리 이건 그 세션에서 "실제로 눌렀던" 값.
      const VFY_KEYS = [
        ["instruction", "instruction"], ["instruction_mode", "이동 모드"],
        ["gt_object", "GT object"], ["apply_cc", "색보정(CC)"],
      ];
      const CFG_KEYS = [
        ["checkpoint_path", "checkpoint"], ["head", "head"], ["git_commit", "git"],
        ["grounder_model", "grounder"], ["grounder_input_px", "grounder px"],
        ["owlv2_thresh", "owlv2_thresh"], ["owlv2_area_scale", "area_scale"],
        ["grounding_skip_n", "skip_n"], ["preview_enabled", "preview"],
        ["preview_hint_cx", "hint_cx"], ["multi_prompt", "multi_prompt"],
        ["cx_jump_filter", "cx_jump_filter"], ["cx_jump_thresh", "cx_jump_thresh"],
      ];

      const vfyTiles = VFY_KEYS.filter(([k]) => attrs && attrs[k] !== undefined && attrs[k] !== "")
        .map(([k, label]) => `<div class="mini-tile mt-accent"><span class="mt-label">${label}</span><span class="mt-value" style="font-size:12px;">${attrs[k]}</span></div>`);
      const cfgTiles = CFG_KEYS.filter(([k]) => cfg[k] !== undefined)
        .map(([k, label]) => `<div class="mini-tile"><span class="mt-label">${label}</span><span class="mt-value" style="font-size:12px;">${cfg[k]}</span></div>`);

      if (vfyTiles.length === 0 && cfgTiles.length === 0) {
        box.style.display = "none";
        return;
      }
      grid.innerHTML = vfyTiles.concat(cfgTiles).join("");
      box.style.display = "block";
    }

    function renderInspectSummary(frames) {
      const n = frames.length;
      const avgLat = n ? (frames.reduce((s, f) => s + (f.latency_ms || 0), 0) / n) : 0;
      const liveN = frames.filter(f => f.cached === 0.0).length;
      const cacheN = frames.filter(f => f.cached === 1.0).length;
      const labeledN = frames.filter(f => f.user_label).length;
      const warnN = frames.filter(f => f.warns && f.warns.length > 0).length;
      const counts = {};
      frames.forEach(f => { counts[f.action] = (counts[f.action] || 0) + 1; });
      const sortedActions = Object.entries(counts).sort((a,b) => b[1]-a[1]);
      const maxCount = sortedActions.length ? sortedActions[0][1] : 1;

      const totalLatSec = frames.reduce((s, f) => s + (f.latency_ms || 0), 0) / 1000;

      document.getElementById("inspect-sum-frames").textContent = n;
      document.getElementById("inspect-sum-lat").textContent = `${avgLat.toFixed(0)}ms`;
      document.getElementById("inspect-sum-lat-sub").textContent = `약 ${(avgLat/1000).toFixed(1)}초`;
      document.getElementById("inspect-sum-total").textContent = `${totalLatSec.toFixed(1)}초`;
      document.getElementById("inspect-sum-cache").textContent = `${liveN} / ${cacheN}`;
      document.getElementById("inspect-sum-labeled").textContent = `${labeledN}/${n}`;
      document.getElementById("inspect-sum-warns").textContent = warnN;
      document.getElementById("inspect-sum-warns-tile").className = "mini-tile " + (warnN > 0 ? "mt-bad" : "mt-good");

      document.getElementById("inspect-sum-actions").innerHTML = sortedActions.map(([k, v]) => `
        <div class="mini-tile">
          <span class="mt-label">${k}</span>
          <span class="mt-value">${v}</span>
          <div style="height:4px; background:#1d2b45; border-radius:2px; margin-top:2px; overflow:hidden;">
            <div style="height:100%; width:${(v / maxCount * 100).toFixed(0)}%; background:var(--cyan);"></div>
          </div>
        </div>
      `).join("");
    }

    function _timelineColor(f) {
      if (f.type === "★ARRIVAL") return "#f59e0b";
      if (f.type === "🔄PREVIEW") return "#8b5cf6";
      if (f.warns && f.warns.length > 0) return "#f43f5e";
      return "#10b981";
    }

    function renderInspectTimeline(frames, activeIdx) {
      const el = document.getElementById("inspect-timeline");
      el.innerHTML = frames.map((f, i) => `
        <div title="Frame ${i+1}: ${f.action}${f.warns.length ? ' — ' + f.warns.length + '건 경고' : ''}"
             onclick="document.getElementById('inspect-slider').value=${i}; showInspectFrame(${i});"
             style="flex:1; min-width:2px; cursor:pointer; background:${_timelineColor(f)};
                    ${i === activeIdx ? 'outline:2px solid #06b6d4; outline-offset:-2px;' : 'opacity:0.75;'}">
        </div>
      `).join("");
    }

    function jumpToNextAnomaly() {
      if (!inspectSession) return;
      const cur = parseInt(document.getElementById("inspect-slider").value);
      const frames = inspectSession.frames;
      for (let i = cur + 1; i < frames.length; i++) {
        if (frames[i].warns && frames[i].warns.length > 0) {
          document.getElementById("inspect-slider").value = i;
          showInspectFrame(i);
          return;
        }
      }
      alert("이후 프레임에 이상치가 없습니다.");
    }

    function showInspectFrame(idx) {
      if (!inspectSession) return;
      idx = parseInt(idx);
      const frame = inspectSession.frames[idx];
      
      document.getElementById("inspect-frame-idx-lbl").textContent = `Frame: ${idx + 1} / ${inspectSession.frames.length}`;
      document.getElementById("inspect-frame-type-lbl").textContent = frame.type;
      document.getElementById("inspect-action-lbl").textContent = frame.action;
      document.getElementById("inspect-lat-lbl").textContent = frame.latency_ms.toFixed(0) + "ms";
      document.getElementById("inspect-lat-sec-lbl").textContent = `약 ${(frame.latency_ms/1000).toFixed(1)}초`;
      document.getElementById("inspect-cx-lbl").textContent = frame.cx.toFixed(3);
      document.getElementById("inspect-cy-lbl").textContent = frame.cy.toFixed(3);
      document.getElementById("inspect-area-lbl").textContent = frame.area.toFixed(3);
      document.getElementById("inspect-hasbbox-lbl").textContent = frame.has_bbox ? "true" : "false";

      // 이미지 소스 설정
      document.getElementById("inspect-frame-img").src = `/sessions/frame?sid=${inspectSession.sid}&idx=${idx}`;
      renderInspectTimeline(inspectSession.frames, idx);

      // 아노말리 출력
      const abox = document.getElementById("inspect-anomaly-box");
      if (frame.warns.length > 0) {
        abox.innerHTML = frame.warns.map(w => `<div class="text-rose" style="margin-bottom:2px;">${w}</div>`).join("");
        abox.style.borderColor = "var(--rose)";
      } else {
        abox.innerHTML = "<div class='text-emerald'>✅ 감지된 이상치 경고 없음</div>";
        abox.style.borderColor = "var(--border-glow)";
      }

      // 라벨 선택 버튼 동기화
      document.querySelectorAll("[id^='lbl-btn-']").forEach(btn => btn.className = "btn btn-outline");
      const activeBtn = document.getElementById("lbl-btn-" + (frame.user_label || "NONE"));
      if (activeBtn) activeBtn.className = "btn btn-cyan";

      // 캔버스 박스 렌더
      drawInspectBbox(frame);
    }

    function drawInspectBbox(frame) {
      const cv = document.getElementById("inspect-canvas");
      const ctx = cv.getContext("2d");
      ctx.clearRect(0, 0, cv.width, cv.height);
      
      if (frame.has_bbox && frame.area > 0) {
        const W = cv.width;
        const H = cv.height;
        const cx = frame.cx * W;
        const cy = frame.cy * H;
        
        // 면적 비례 바운딩 박스 크기 추측 계산
        const half = Math.sqrt(frame.area) * Math.min(W, H) * 0.5;
        const x0 = cx - half;
        const y0 = cy - half;
        
        ctx.strokeStyle = frame.warns.length > 0 ? "#f59e0b" : "#10b981";
        ctx.lineWidth = 3;
        ctx.strokeRect(x0, y0, half*2, half*2);
        
        ctx.fillStyle = frame.warns.length > 0 ? "#f59e0b" : "#10b981";
        ctx.beginPath();
        ctx.arc(cx, cy, 5, 0, 2*Math.PI);
        ctx.fill();
      }
    }

    async function saveFrameLabel(label) {
      if (!inspectSession) return;
      const idx = parseInt(document.getElementById("inspect-slider").value);
      const res = await api("/sessions/label", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: inspectSession.sid,
          frame_idx: idx,
          label: label
        })
      });
      if (res.ok) {
        inspectSession.frames[idx].user_label = label === "NONE" ? "" : label;
        showInspectFrame(idx);
        loadSessionList(); // 리스트 카운트 배지 갱신
      }
    }

    async function deleteActiveSession() {
      if (!inspectSession) return;
      if (!confirm("⚠️ 정말로 이 세션 파일(" + inspectSession.sid + ")을 영구 삭제하시겠습니까?\\n이 작업은 복구가 불가능합니다.")) {
        return;
      }
      
      if (inspectPlayTimer) {
        clearInterval(inspectPlayTimer);
        inspectPlayTimer = null;
        document.getElementById("btn-inspect-play").textContent = "▶ PLAY";
      }

      const res = await api("/sessions/delete?sid=" + inspectSession.sid, { method: "POST" });
      if (res.ok) {
        alert(res.message);
        inspectSession = null;
        document.getElementById("inspector-placeholder").style.display = "block";
        document.getElementById("inspector-body").style.display = "none";
        loadSessionList();
      } else {
        alert("삭제 실패: " + res.error);
      }
    }


    // ── Canvas Overlay 드로잉 ─────────────────────────────────────────
    function drawOverlay() {
      const showGrid = document.getElementById("toggle-grid").checked;
      
      // 1. Drive Control 탭 오버레이
      const cvDrive = document.getElementById("live-canvas");
      const ctxDrive = cvDrive.getContext("2d");
      ctxDrive.clearRect(0, 0, cvDrive.width, cvDrive.height);
      
      if (showGrid) {
        drawGridLines(ctxDrive, cvDrive.width, cvDrive.height);
      }
      if (state.bbox && state.bbox.area > 0) {
        drawBbox(ctxDrive, state.bbox, cvDrive.width, cvDrive.height);
      }
      
      // 2. Grounding 탭 오버레이
      const cvGnd = document.getElementById("gnd-canvas");
      const ctxGnd = cvGnd.getContext("2d");
      ctxGnd.clearRect(0, 0, cvGnd.width, cvGnd.height);
      if (state.bbox && state.bbox.area > 0) {
        drawBbox(ctxGnd, state.bbox, cvGnd.width, cvGnd.height);
      }
      
      // 3. Calibration 탭 오버레이
      const cvCal = document.getElementById("calib-canvas");
      const ctxCal = cvCal.getContext("2d");
      ctxCal.clearRect(0, 0, cvCal.width, cvCal.height);
      if (state.bbox && state.bbox.area > 0) {
        drawBbox(ctxCal, state.bbox, cvCal.width, cvCal.height);
      }
      
      // 4. Verification 탭 오버레이
      const cvVfy = document.getElementById("verify-canvas");
      if (cvVfy) {
        const ctxVfy = cvVfy.getContext("2d");
        ctxVfy.clearRect(0, 0, cvVfy.width, cvVfy.height);
        const guideVfy = document.getElementById("toggle-cxguide-vfy");
        if (guideVfy && guideVfy.checked) {
          // 데이터수집과 동일한 cx 배치가이드 밴드 재사용 + 현재 선택 위치 강조
          _collectDrawCxGuideBands(ctxVfy, cvVfy.width, cvVfy.height);
          _drawVerifyTargetHighlight(ctxVfy, cvVfy.width, cvVfy.height);
        }
        const showGridVfy = document.getElementById("toggle-grid-vfy").checked;
        if (showGridVfy) {
          drawGridLines(ctxVfy, cvVfy.width, cvVfy.height);
        }
        if (state.bbox && state.bbox.area > 0) {
          drawBbox(ctxVfy, state.bbox, cvVfy.width, cvVfy.height);
        }
      }
    }

    // 현재 Tab4에서 선택된 path_type의 위치를 카메라 위에 강하게 강조 —
    // "지금 이 검증에서 바구니를 여기 놓아라"를 시각적으로 표시.
    const VERIFY_TARGET_BANDS = {
      strong_left:  {lo: 0.10, hi: 0.15, label: "강한좌 여기", color: "#f43f5e"},
      weak_left:    {lo: 0.20, hi: 0.25, label: "준극단좌 여기", color: "#f59e0b"},
      center:       {lo: 0.475, hi: 0.525, label: "중앙 여기", color: "#3fb950"},
      weak_right:   {lo: 0.75, hi: 0.80, label: "준극단우 여기", color: "#f59e0b"},
      strong_right: {lo: 0.85, hi: 0.90, label: "강한우 여기", color: "#f43f5e"},
    };
    function _drawVerifyTargetHighlight(ctx, W, H) {
      // 검증모드면 조이스틱 D-pad로 고른 위치(window._verifyScreenPos)를 우선,
      // 아니면 Tab4 select의 path_type에서 위치 추출.
      let pos = (joystickVerifyMode && window._verifyScreenPos) ? window._verifyScreenPos : null;
      if (!pos) {
        const sel = document.getElementById("ep-path-type");
        if (!sel) return;
        pos = verifyPosOf(sel.value);
      }
      const b = pos && VERIFY_TARGET_BANDS[pos];
      if (!b) return;
      const x0 = b.lo * W, x1 = b.hi * W;
      ctx.save();
      ctx.globalAlpha = 0.35;
      ctx.fillStyle = b.color;
      ctx.fillRect(x0, 0, x1 - x0, H);
      ctx.restore();
      ctx.strokeStyle = b.color;
      ctx.lineWidth = 4;
      ctx.strokeRect(x0, 0, x1 - x0, H);
      ctx.fillStyle = b.color;
      ctx.textAlign = "center";
      ctx.font = "bold 20px monospace";
      ctx.fillText("🎯 " + b.label, (x0 + x1) / 2, H / 2);
    }

    function drawGridLines(ctx, W, H) {
      ctx.strokeStyle = "rgba(132, 204, 22, 0.55)"; // 연두색(lime), 기존보다 진하게
      ctx.lineWidth = 1.5;
      
      // 수직 3분할 그리드
      ctx.beginPath();
      ctx.moveTo(W / 3, 0); ctx.lineTo(W / 3, H);
      ctx.moveTo(2 * W / 3, 0); ctx.lineTo(2 * W / 3, H);
      
      // 수평 3분할 그리드
      ctx.moveTo(0, H / 3); ctx.lineTo(W, H / 3);
      ctx.moveTo(0, 2 * H / 3); ctx.lineTo(W, 2 * H / 3);
      ctx.stroke();
    }

    function drawBbox(ctx, bbox, W, H) {
      // bbox: {x1, y1, x2, y2, cx, cy, area, has_bbox} — 서버 필드명은 x1/y1/x2/y2 (xmin/ymin 아님)
      if (bbox.x1 != null && bbox.y1 != null && bbox.x2 != null && bbox.y2 != null) {
        const x0 = bbox.x1 * W;
        const y0 = bbox.y1 * H;
        const w = (bbox.x2 - bbox.x1) * W;
        const h = (bbox.y2 - bbox.y1) * H;
        ctx.strokeStyle = "#10b981";
        ctx.lineWidth = 3;
        ctx.strokeRect(x0, y0, w, h);
      }

      // 중심점 (PG2 grounding이 실제로 내놓은 cx, cy)
      ctx.fillStyle = "#10b981";
      ctx.beginPath();
      ctx.arc(bbox.cx * W, bbox.cy * H, 6, 0, 2 * Math.PI);
      ctx.fill();
    }

    // ── 에피소드 저장 / 실행 취소 / 누적 기록 로드 ──────────────────────────
    let _epEditingRow = null;
    let _epByRow = {};

    function _epEditLoadByRow(rowKey) {
      const ep = _epByRow[rowKey];
      if (ep) _epEditLoad(ep);
    }

    function _epEditLoad(ep) {
      _epEditingRow = ep[0];
      const pathSel = document.getElementById("ep-edit-path");
      if (pathSel && pathSel.options.length === 0) {
        pathSel.innerHTML = document.getElementById("ep-path-type").innerHTML;
      }
      document.getElementById("ep-edit-target").textContent = `#${ep[0]} 수정 중`;
      pathSel.value = ep[1];
      document.getElementById("ep-edit-success").value = ep[2];
      document.getElementById("ep-edit-steps").value = ep[3];
      document.getElementById("ep-edit-lat").value = ep[4];
      document.getElementById("ep-edit-fpe").value = ep[10];
      document.getElementById("ep-edit-note").value = ep[11] || "";
      document.getElementById("ep-edit-status").textContent = "—";
      loadEpisodeHistory(); // 클릭한 행 하이라이트 갱신
    }

    function _epEditClear() {
      _epEditingRow = null;
      document.getElementById("ep-edit-target").textContent = "— 행을 클릭하세요 —";
      document.getElementById("ep-edit-status").textContent = "—";
      loadEpisodeHistory();
    }

    async function _epEditSave() {
      if (_epEditingRow == null) {
        document.getElementById("ep-edit-status").textContent = "⚠️ 먼저 행을 클릭하세요";
        return;
      }
      const body = {
        row: parseInt(_epEditingRow),
        path_type: document.getElementById("ep-edit-path").value,
        success: document.getElementById("ep-edit-success").value,
        steps: parseInt(document.getElementById("ep-edit-steps").value) || 0,
        lat_ms: parseFloat(document.getElementById("ep-edit-lat").value) || 0,
        fpe: parseFloat(document.getElementById("ep-edit-fpe").value) || 0,
        note: document.getElementById("ep-edit-note").value,
      };
      const res = await api("/episodes/update", { method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify(body) });
      const statusEl = document.getElementById("ep-edit-status");
      statusEl.textContent = res.ok ? `✅ #${body.row} 저장됨` : "⚠️ " + (res.error || "저장 실패");
      if (res.ok) loadEpisodeHistory();
    }

    async function loadEpisodeHistory() {
      try {
        const res = await api("/episodes/list");
        if (res.ok) {
          // 경로 집계 패널(진행바 및 요약 표)을 갱신합니다.
          updatePathSummary(res.episodes || []);
          const tbody = document.getElementById("episodes-table-body");
          if (!tbody) return;
          
          let episodes = res.episodes || [];
          
          // 필터 선택 값에 따라 에피소드 성공/실패 여부를 걸러냅니다.
          const filterVal = document.getElementById("vfy-filter") ? document.getElementById("vfy-filter").value : "all";
          if (filterVal === "success") {
            episodes = episodes.filter(ep => ep[2] === "성공");
          } else if (filterVal === "fail") {
            episodes = episodes.filter(ep => ep[2] === "실패");
          }

          if (episodes.length === 0) {
            tbody.innerHTML = `<tr><td colspan="14" style="text-align:center; color:var(--text-muted);">기록이 없습니다.</td></tr>`;
            return;
          }
          _epByRow = {};
          episodes.forEach(ep => { _epByRow[ep[0]] = ep; });
          tbody.innerHTML = episodes.map(ep => {
            const pathAbbr = {
              right_right: "R→R", right_left: "R→L★", right_straight: "R→S",
              center_straight: "C→S", center_left: "C→L", center_right: "C→R",
              left_straight: "L→S", left_left: "L→L", left_right: "L→R",
              obj_left: "위치:좌", obj_center: "위치:중", obj_right: "위치:우",
              dist_10cm: "10cm", dist_20cm: "20cm", dist_30cm: "30cm",
            }[ep[1]] || ep[1];

            const resColor = ep[2] === "성공" ? "text-emerald" : "text-rose";
            const isEditing = String(ep[0]) === String(_epEditingRow);
            const stopColor = ep[9] === "Y" ? "text-emerald" : "text-muted";
            return `
              <tr onclick='_epEditLoadByRow("${String(ep[0]).replace(/"/g, "&quot;")}")' style="cursor:pointer; ${isEditing ? "background:rgba(56,189,248,0.12);" : ""}">
                <td>${ep[0]}</td>
                <td><strong class="text-cyan">${pathAbbr}</strong></td>
                <td><strong class="${resColor}">${ep[2]}</strong></td>
                <td>${ep[3]}</td>
                <td>${ep[4]} ms</td>
                <td style="font-size:10px; color:var(--text-muted);">${ep[5] || "—"}</td>
                <td class="font-mono" style="font-size:10px;">${ep[6] ?? "—"}</td>
                <td class="font-mono" style="font-size:10px;">${ep[7] ?? "—"}</td>
                <td class="font-mono" style="font-size:10px;">${ep[8] ?? "—"}</td>
                <td class="${stopColor}" style="font-size:10px;">${ep[9] || "—"}</td>
                <td class="font-mono text-cyan" style="font-size:10px;">${ep[10] || "—"}</td>
                <td style="font-size:11px; color:var(--text-muted);">${ep[11] || "—"}</td>
                <td style="font-family:var(--font-sans); color:var(--text-muted); font-size:10px; white-space:nowrap;">${ep[12] || "—"}</td>
                <td class="font-mono" style="font-size:9px; color:var(--text-muted); white-space:nowrap;">${ep[13] || "—"}</td>
              </tr>
            `;
          }).reverse().join("");
        }
      } catch (e) {
        console.error("loadEpisodeHistory error:", e);
      }
    }

    async function commitEpisode() {
      const pathType = document.getElementById("ep-path-type").value;
      const success = document.getElementById("ep-success").value;
      const fpe = parseFloat(document.getElementById("ep-fpe").value);
      const note = document.getElementById("ep-note").value;
      
      const res = await api("/episodes/log", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          path_type: pathType,
          success: success,
          fpe: fpe,
          note: note
        })
      });
      const sl = document.getElementById("vfy-status-log");
      if (res.ok) {
        document.getElementById("ep-note").value = "";
        if (sl) sl.textContent = `💾 저장됨: ${pathType} = ${success}`;
        loadEpisodeHistory();
      } else {
        if (sl) sl.textContent = "⚠️ 기록 실패: " + res.error;
      }
    }

    async function undoEpisode() {
      if (!confirm("⚠️ 정말로 마지막 에피소드 기록 1건을 삭제하시겠습니까?")) return;
      const res = await api("/episodes/undo", { method: "POST" });
      if (res.ok) {
        alert("마지막 기록이 삭제되었습니다.");
        loadEpisodeHistory();
      } else {
        alert("삭제 실패: " + res.error);
      }
    }

    // ── Frame-by-Frame Inspector 프레임 재생 및 탐색 제어 ───────────────────
    function nextInspectFrame() {
      const slider = document.getElementById("inspect-slider");
      if (!slider) return;
      let val = parseInt(slider.value) + 1;
      if (val <= parseInt(slider.max)) {
        slider.value = val;
        showInspectFrame(val);
      }
    }

    function prevInspectFrame() {
      const slider = document.getElementById("inspect-slider");
      if (!slider) return;
      let val = parseInt(slider.value) - 1;
      if (val >= 0) {
        slider.value = val;
        showInspectFrame(val);
      }
    }

    function toggleInspectPlay() {
      const btn = document.getElementById("btn-inspect-play");
      if (!btn) return;
      if (inspectPlayTimer) {
        clearInterval(inspectPlayTimer);
        inspectPlayTimer = null;
        btn.textContent = "▶ PLAY";
      } else {
        btn.textContent = "⏸ PAUSE";
        inspectPlayTimer = setInterval(() => {
          const slider = document.getElementById("inspect-slider");
          if (!slider) return;
          let val = parseInt(slider.value) + 1;
          if (val > parseInt(slider.max)) val = 0;
          slider.value = val;
          showInspectFrame(val);
        }, 300);
      }
    }

    // ── 주기적 폴링 루프 (500ms) ──────────────────────────────────────
    async function pollStatus() {
      try {
        state = await api("/drive/status");
        
        // 1. 상태 배지 & 정보 바인딩 (안전 헬퍼 적용)
        setSafeText("badge-step-lbl", "Step: " + state.step);
        setSafeText("badge-mode-lbl", "Mode: " + state.mode);
        
        const actStr = state.last_action ? state.last_action.map(v => v.toFixed(2)).join(", ") : "0.00, 0.00, 0.00";
        setSafeText("badge-action-lbl", "Action: " + actStr);
        
        setSafeText("drive-log", state.status_log || "");
        
        // 2. 버튼 활성화 동기화
        _syncStartStopBtn("drive-start-btn", "drive-stop-btn", "▶ START DRIVE", state.running);
        _syncStartStopBtn("quick-start-btn", "quick-stop-btn", "▶️ START", state.running);

        // running: true→false 전환 감지 — 목표 도달로 인한 자동 정지처럼
        // STOP 버튼을 안 눌러도 세션이 끝난 경우까지 세션 스택을 갱신
        if (_prevRunning && !state.running) showSessionStack();
        _prevRunning = state.running;

        const returnBtn = document.getElementById("drive-return-btn");
        if (returnBtn) {
          returnBtn.textContent = state.is_returning ? "⏹️ 복귀 중단" : "🔄 START 위치로 복귀";
        }
        
        // 3. Grounding 정보 노출
        if (state.bbox) {
          const area = state.bbox.area || 0;
          const cx = state.bbox.cx || 0.5;
          const cy = state.bbox.cy || 0.5;
          setSafeValue("gnd-coords", `cx:${cx.toFixed(3)}, cy:${cy.toFixed(3)}, area:${area.toFixed(4)}`);
          setSafeValue("gnd-area", area.toFixed(4));
          setSafeValue("gnd-cx", cx.toFixed(3));
          
          // 게이지 업데이트
          const MAX_A = 0.40;
          const pct = Math.min(100, (area / MAX_A) * 100);
          const fill = document.getElementById("gnd-gauge-fill");
          if (fill) {
            fill.style.width = pct + "%";
            if (state.goal_near) {
              fill.style.backgroundColor = "var(--rose)";
            } else if (area >= 0.18 * 0.7) {
              fill.style.backgroundColor = "var(--amber)";
            } else {
              fill.style.backgroundColor = "var(--emerald)";
            }
          }
        } else {
          setSafeValue("gnd-coords", "—");
          setSafeValue("gnd-area", "—");
          setSafeValue("gnd-cx", "—");
          const fill = document.getElementById("gnd-gauge-fill");
          if (fill) fill.style.width = "0%";
        }
        
        setSafeValue("gnd-entity", state.instruction || "—");
        setSafeValue("gnd-pred-label", state.predicted_label || "—");
        
        // 4-0. 수집 세션 요약
        {
          const rh = state.run_history || [];
          const nInfer = rh.length;
          const runBadge = state.running ? "🟢 실행중" : "⚫ 정지";
          const sessionHtml = `${runBadge} · ${state.mode || "—"} · 스텝 ${state.step || 0}<br>`
            + `기록 ${nInfer}개 · ID ${state.session_id || "—"}`;
          setSafeHtml("cs-session", sessionHtml);
          setSafeHtml("cs-session-vfy", sessionHtml);
          const liveN  = rh.filter(r => r[3] !== "—" && r[3] !== null && r[3] !== undefined).length;
          const cacheN = nInfer - liveN;
          const lats   = rh.map(r => r[3]).filter(v => typeof v === "number");
          const avgGnd = lats.length ? Math.round(lats.reduce((a,b)=>a+b,0)/lats.length) : 0;
          const groundingHtml = `live ${liveN} · 캐시 ${cacheN}<br>평균 gnd ${avgGnd}ms`;
          setSafeHtml("cs-grounding", groundingHtml);
          setSafeHtml("cs-grounding-vfy", groundingHtml);
        }
 
        // 4. 타임라인 히스토리 업데이트
        if (state.run_history && state.run_history.length > 0) {
          const rows = state.run_history.slice(-10).reverse();
          const historyHtml = rows.map(r => `
            <tr>
              <td>${r[0]}</td>
              <td><span class="text-cyan">${r[1]}</span></td>
              <td>${r[2]} ms</td>
              <td>${r[3]} ms</td>
              <td>${r[4]} ms</td>
              <td><strong class="text-emerald">${r[5]}</strong></td>
              <td style="font-size:11px; color:var(--text-muted); white-space:nowrap;">${r[6] || "—"}</td>
            </tr>
          `).join("");
          setSafeHtml("drive-history-table", historyHtml);
          setSafeHtml("drive-history-table-vfy", historyHtml);
        }
 
        // 5. 🧪 경로 검증(Verify) 탭 실시간 업데이트
        if (activeTab === "verify") {
          setSafeText("vfy-status-log", state.status_log || "Ready");
          
          let latMs = 0;
          if (state.run_history && state.run_history.length > 0) {
            const latestRun = state.run_history[state.run_history.length - 1];
            latMs = latestRun[2] || 0;
          }
          setSafeText("vfy-latency-val", latMs + " ms");
          
          setSafeText("vfy-action-val", actStr);
          
          if (state.bbox) {
            setSafeText("vfy-bbox-val", `${state.bbox.area.toFixed(4)} / ${state.bbox.cx.toFixed(3)}`);
          } else {
            setSafeText("vfy-bbox-val", "—");
          }
          
          setSafeText("vfy-gnd-entity", state.instruction || "—");
          setSafeText("vfy-pred-label", state.predicted_label || "—");
          
          // 그라운딩 캐시 상태
          let cacheStatus = "—";
          if (state.grounding_cached !== undefined && state.grounding_cached !== null) {
            cacheStatus = state.grounding_cached === 1 ? "Yes (Cached)" : (state.grounding_cached === 0 ? "No (Computed)" : "—");
          }
          setSafeText("vfy-gnd-cached", cacheStatus);

          // 📋 신규 추가: 수집 세션 상태
          const nowStr = new Date().toLocaleTimeString("ko-KR", {hour12:false});
          setSafeHtml("vfy-update-time", "갱신 " + nowStr);
          setSafeText("vfy-sess-run-status", state.running ? "🟢 실행중" : "⚫ 정지");
          setSafeText("vfy-sess-mode", state.mode || "SYNC");
          setSafeText("vfy-sess-step", state.step || "0");
          setSafeText("vfy-sess-frames", state.n_frames !== undefined ? state.n_frames + "장" : "0장");
          setSafeText("vfy-sess-record-lbl", `추론 ${state.n_infer || 0} + post ${state.n_post || 0} = ${state.n_total || 0}`);
          setSafeText("vfy-sess-id-lbl", state.session_id || "—");

          // 🔍 신규 추가: 그라운딩 통계
          setSafeText("vfy-gnd-live-cnt", state.gnd_live !== undefined ? state.gnd_live + "회" : "0회");
          setSafeText("vfy-gnd-cache-cnt", state.gnd_cache !== undefined ? state.gnd_cache + "회" : "0회");
          setSafeText("vfy-gnd-avg-lat", state.gnd_avg_lat !== undefined ? state.gnd_avg_lat + "ms" : "0ms");

          // 🔢 신규 추가: 최근 스텝 타임라인 이력
          let stepsHtml = "";
          if (state.last_steps && state.last_steps.length > 0) {
            stepsHtml = state.last_steps.map(s => {
              const isCached = s.grounding_cached === 1;
              const gndType = isCached ? '<span style="color:var(--emerald);">[캐시]</span>' : (s.grounding_cached === 0 ? '<span style="color:var(--amber);">[live]</span>' : '<span style="color:var(--text-muted);">[-]</span>');
              const isPost = String(s.step).endsWith("p");
              const stepColor = isPost ? "color:var(--violet);" : "color:var(--cyan);";
              return `<div style="display:flex; justify-content:space-between; gap:8px;">`
                   + `<span style="${stepColor}">Step ${s.step}</span>`
                   + `<span style="color:var(--text-muted);">${s.predicted_label}</span>`
                   + `<span>${gndType} ${s.latency_ms}ms</span>`
                   + `</div>`;
            }).reverse().join("");
          } else {
            stepsHtml = `<span style="color:var(--text-muted);">스텝 이력 없음</span>`;
          }
          setSafeHtml("vfy-last-steps-container", stepsHtml);
        }
 
        // 7. 오버레이 드로잉 호출
        drawOverlay();
      } catch(e) {
        console.error("Polling status error:", e);
      }
    }

    // ── 헬스 체크 루프 (3초) ──────────────────────────────────────────
    function _fmtUptime(s) {
      if (s < 60) return `${Math.floor(s)}s`;
      if (s < 3600) return `${Math.floor(s/60)}m`;
      const h = Math.floor(s/3600), m = Math.floor((s%3600)/60);
      return `${h}h ${m}m`;
    }

    async function pollHealth() {
      try {
        const h = await api("/health");

        // 프로세스 PID/가동시간 — 재시작 후에도 옛날 PID/긴 가동시간이 보이면
        // 좀비 프로세스에 붙어있는 것일 수 있음 (2026-07-02 사고 참고)
        const procEl = document.getElementById("proc-pill-text");
        if (procEl && h.pid) {
          procEl.textContent = `PID ${h.pid} · ${_fmtUptime(h.uptime_s)} (${h.started_at})`;
        }

        // ROS Node status
        const rosPill = document.getElementById("ros-pill");
        if (h.node_up) {
          rosPill.className = "srv-pill online";
        } else {
          rosPill.className = "srv-pill offline";
        }
        
        // Camera status
        const camPill = document.getElementById("camera-pill");
        if (h.camera_ok) {
          camPill.className = "srv-pill online";
          camPill.innerHTML = `<span class="indicator"></span> Live Camera (${h.frame_count}f)`;
        } else {
          camPill.className = "srv-pill offline";
          camPill.innerHTML = `<span class="indicator"></span> Camera Offline`;
        }
      } catch(e) {
        document.getElementById("ros-pill").className = "srv-pill offline";
        document.getElementById("camera-pill").className = "srv-pill offline";
      }

      try {
        const inf = await api("/infer/health");
        const infPill = document.getElementById("infer-pill");
        const sysPanel = document.getElementById("sys-info-panel");
        
        if (inf.model_loaded) {
          infPill.className = "srv-pill online";
          
          sysPanel.innerHTML = `
            <div class="form-group"><label>Loaded Checkpoint</label><input type="text" readonly value="${inf.checkpoint_path}"></div>
            <div class="form-group"><label>Precision</label><input type="text" readonly value="${inf.precision}"></div>
            <div class="form-group"><label>Model Head</label><input type="text" readonly value="${inf.head}"></div>
            <div class="form-group">
              <label>Stop Mode</label>
              <div style="display:flex; gap:6px; align-items:center;">
                <input type="text" readonly value="${inf.stop_mode} (${inf.stop_latched ? 'Latched' : 'Unlatched'}) · guard=${inf.stop_learned_min_steps ?? 3}f" style="flex:1;">
                <button class="btn btn-outline" onclick="toggleVerifyRuntime('stopmode')" style="font-size:11px; padding:6px 10px; white-space:nowrap;">🔁 전환</button>
                <button class="btn btn-outline" onclick="toggleVerifyRuntime('stopguard')" style="font-size:11px; padding:6px 10px; white-space:nowrap;">🛡 가드</button>
              </div>
            </div>
          `;
          
          // verify 탭 서버 상태 갱신 — 메인 모델 + 그라운딩 모델을 서버 변경 즉시 반영
          const vfySrvStatus = document.getElementById("vfy-srv-status");
          if (vfySrvStatus) {
            const vg = inf.grounder || {};
            const grIsOwl = (vg.model || "").toLowerCase().includes("owl");
            const grBadge = grIsOwl
              ? `<span style="color:var(--emerald); font-weight:700;">🔭 ${vg.model}</span> th=${vg.owlv2_thresh ?? '—'} scale=${vg.owlv2_area_scale ?? '—'}`
              : `<span style="color:var(--amber); font-weight:700;">🔭 ${vg.model || '—'}</span>`;
            vfySrvStatus.innerHTML =
              `<span style="color:var(--cyan); font-weight:700;">🧠 ${inf.checkpoint_path ? inf.checkpoint_path.split('/').pop() : '—'}</span> (${inf.head} W${inf.window})<br>`
              + `${grBadge} · phrase="${vg.phrase || '—'}"<br>`
              + `STOP: ${inf.stop_mode} (${inf.stop_latched ? 'Latched' : 'Unlatched'}) · git ${inf.git_commit || '—'}`;
          }

          // 시스템 토글 스위치 동기화
          document.getElementById("sys-preview").checked = inf.preview_enabled !== false;
          document.getElementById("sys-hint").checked = inf.preview_hint_cx === true;
          document.getElementById("sys-skip").checked = inf.grounding_skip_n === 3;
          document.getElementById("sys-jump").checked = inf.cx_jump_filter === true;
          if (inf.cx_jump_thresh) {
            document.getElementById("sys-jump-thresh").value = inf.cx_jump_thresh;
            document.getElementById("sys-jump-thresh-lbl").textContent = '오탐 필터 임계값: ' + inf.cx_jump_thresh.toFixed(2);
          }
          if (inf.stop_area_threshold) {
            document.getElementById("calib-thr").value = inf.stop_area_threshold;
            document.getElementById("calib-thr-lbl").textContent = 'Area 임계값: ' + inf.stop_area_threshold.toFixed(3);
            document.getElementById("gnd-gauge-thr-lbl").textContent = '임계값: ' + inf.stop_area_threshold.toFixed(3);
            
            // 게이지 마커 위치 지정
            const MAX_A = 0.40;
            const markerPos = Math.min(100, (inf.stop_area_threshold / MAX_A) * 100);
            document.getElementById("gnd-gauge-marker").style.left = markerPos + "%";
          }
        } else {
          infPill.className = "srv-pill offline";
          sysPanel.innerHTML = "<div style='grid-column: 1/-1; text-align:center; color:var(--rose); padding:20px 0;'>추론 서버 모델이 적재되지 않았습니다.</div>";
        }
      } catch(e) {
        document.getElementById("infer-pill").className = "srv-pill offline";
      }
    }

    // ── 🗂 데이터셋 히스토리 탭 — 데이터수집이 저장한 원본 H5 브라우징 ──
    let _dsItems = [];
    let _dsScenarioLabels = {};
    let _dsCxPosLabels = {};
    let _dsSelected = new Set();     // 비교 모드에서 체크된 name들
    let _dsCxPosFilter = new Set();  // 클릭형 칩 필터 (다중 선택, OR)
    let _dsSchemaFilter = "";        // "" | "legacy" | "new" — 버튼 단일선택
    let _dsScenarioFilter = "";      // "" 또는 scenario 키 — 버튼 단일선택
    let _dsCompareMode = false;
    let _dsDetail = null;            // 현재 단일 상세로 열린 dataset_load() 응답
    let _dsPlayTimer = null;
    const DS_MAX_COMPARE = 6;
    const DS_SCHEMA_VERSION = {legacy: "V5", new: "V6"};  // docs/DATASET_V6_STATUS.md 명명 규정

    // cx_position/cx_path/scenario/schema를 아이콘+색깔 칩으로 파싱 — 리스트 카드에서
    // 파일명 전체를 안 보여줘도(제목=날짜시간, 파일명은 title 툴팁) 한눈에 구분되도록.
    const DS_CXPOS_STYLE = {
      strong_left:  {icon: "◀◀", bg: "rgba(56,189,248,0.15)", fg: "var(--cyan)"},
      weak_left:    {icon: "◀",  bg: "rgba(56,189,248,0.15)", fg: "var(--cyan)"},
      center:       {icon: "●",  bg: "rgba(148,163,184,0.15)", fg: "var(--text-muted)"},
      weak_right:   {icon: "▶",  bg: "rgba(245,158,11,0.15)", fg: "var(--amber)"},
      strong_right: {icon: "▶▶", bg: "rgba(245,158,11,0.15)", fg: "var(--amber)"},
    };
    const DS_CXPATH_ICON = {left_curve: "↰", straight: "↑", right_curve: "↱",
      overshoot_left_recover: "⟲", overshoot_right_recover: "⟳"};

    function _dsInfoChips(it) {
      const chip = (text, bg, fg) =>
        `<span style="font-size:11px; font-weight:700; padding:2px 8px; border-radius:10px; background:${bg}; color:${fg};">${text}</span>`;
      const chips = [];
      if (it.cx_position) {
        const st = DS_CXPOS_STYLE[it.cx_position] || {icon: "●", bg: "rgba(148,163,184,0.15)", fg: "var(--text-muted)"};
        const label = _dsCxPosLabels[it.cx_position] || it.cx_position;
        chips.push(chip(`${st.icon} ${label}`, st.bg, st.fg));
      }
      if (it.cx_path) {
        chips.push(chip(`${DS_CXPATH_ICON[it.cx_path] || "•"} ${it.cx_path}`, "rgba(163,113,247,0.15)", "#a371f7"));
      }
      if (it.scenario) {
        chips.push(chip(`🎯 ${_dsScenarioLabels[it.scenario] || it.scenario}`, "rgba(16,185,129,0.15)", "var(--emerald)"));
      }
      chips.push(chip(DS_SCHEMA_VERSION[it.schema] || it.schema, "rgba(148,163,184,0.15)", "var(--text-muted)"));
      return chips.join("");
    }

    async function _dsDeleteOne(name) {
      if (!confirm(`삭제하시겠습니까?\\n${name}`)) return;
      const res = await api("/dataset/delete?name=" + encodeURIComponent(name), { method: "POST" });
      if (!res.ok) { alert("삭제 실패: " + res.error); return; }
      _dsSelected.delete(name);
      if (_dsDetail && _dsDetail.meta.name === name) {
        _dsDetail = null;
        setDatasetCompareMode(false);
        document.getElementById("ds-placeholder").style.display = "block";
      }
      await loadDatasetList();
    }

    async function _dsDeleteSelected() {
      if (_dsSelected.size === 0) return;
      const names = [..._dsSelected];
      if (!confirm(`선택한 ${names.length}개를 삭제하시겠습니까?\\n\\n${names.join("\\n")}`)) return;
      for (const name of names) {
        const res = await api("/dataset/delete?name=" + encodeURIComponent(name), { method: "POST" });
        if (!res.ok) console.error("삭제 실패:", name, res.error);
      }
      _dsSelected.clear();
      setDatasetCompareMode(false);
      document.getElementById("ds-placeholder").style.display = "block";
      await loadDatasetList();
    }

    async function loadDatasetList() {
      const res = await api("/dataset/list");
      if (!res.ok) return;
      _dsItems = res.items;
      _dsScenarioLabels = res.scenario_labels || {};
      _dsCxPosLabels = res.cx_position_labels || {};

      const scWrap = document.getElementById("ds-filter-scenario");
      const scOptions = [["", "전체"]].concat(res.scenarios.map(s => [s, _dsScenarioLabels[s] || s]));
      scWrap.innerHTML = scOptions.map(([sc, label]) =>
        `<button type="button" class="btn btn-outline ds-scenario-btn" data-sc="${sc}" onclick="_dsSetScenarioFilter(this.dataset.sc)" style="font-size:13px; font-weight:600; padding:4px 10px;">${label}</button>`
      ).join("");
      _dsSyncScenarioButtons();

      const cxWrap = document.getElementById("ds-filter-cxpos");
      cxWrap.innerHTML = res.cx_positions.map(cx =>
        `<button type="button" class="btn btn-outline ds-cxpos-btn" data-cx="${cx}" onclick="_dsToggleCxFilter('${cx}')" style="font-size:10px; padding:3px 8px;">${_dsCxPosLabels[cx] || cx}</button>`
      ).join("");
      _dsSyncCxposButtons();

      renderDatasetList();
    }

    function _dsSyncScenarioButtons() {
      document.querySelectorAll(".ds-scenario-btn").forEach(b => {
        const active = b.dataset.sc === _dsScenarioFilter;
        b.classList.toggle("active", active);
        b.style.background = active ? "rgba(56,189,248,0.2)" : "";
        b.style.borderColor = active ? "var(--cyan)" : "";
        b.style.color = active ? "var(--cyan)" : "";
      });
    }
    function _dsSetScenarioFilter(sc) {
      _dsScenarioFilter = sc;
      _dsSyncScenarioButtons();
      renderDatasetList();
    }

    function _dsSyncCxposButtons() {
      document.querySelectorAll(".ds-cxpos-btn").forEach(b => {
        const active = _dsCxPosFilter.has(b.dataset.cx);
        b.classList.toggle("active", active);
        b.style.background = active ? "rgba(56,189,248,0.2)" : "";
        b.style.borderColor = active ? "var(--cyan)" : "";
        b.style.color = active ? "var(--cyan)" : "";
      });
    }
    function _dsToggleCxFilter(cx) {
      if (_dsCxPosFilter.has(cx)) _dsCxPosFilter.delete(cx); else _dsCxPosFilter.add(cx);
      _dsSyncCxposButtons();
      renderDatasetList();
    }

    function _dsSetSchemaFilter(schema) {
      _dsSchemaFilter = schema;
      document.querySelectorAll(".ds-schema-btn").forEach(b => {
        const active = b.dataset.schema === schema;
        b.classList.toggle("active", active);
        b.style.background = active ? "rgba(56,189,248,0.2)" : "";
        b.style.borderColor = active ? "var(--cyan)" : "";
        b.style.color = active ? "var(--cyan)" : "";
      });
      renderDatasetList();
    }

    function _dsToggleCompareMode() {
      _dsCompareMode = !_dsCompareMode;
      const btn = document.getElementById("ds-compare-toggle-btn");
      btn.textContent = _dsCompareMode ? "✅ 다중 선택(비교 모드) 끄기" : "☑️ 다중 선택(비교 모드) 켜기 — 최대 6개";
      btn.classList.toggle("btn-cyan", _dsCompareMode);
      if (!_dsCompareMode) {
        _dsSelected.clear();
        setDatasetCompareMode(false);
        document.getElementById("ds-placeholder").style.display = "block";
      }
      renderDatasetList();
    }

    function renderDatasetList() {
      const compareMode = _dsCompareMode;
      const search = (document.getElementById("ds-search").value || "").toLowerCase();
      const scenarioFilter = _dsScenarioFilter;

      let items = _dsItems.filter(it => {
        if (search && !it.name.toLowerCase().includes(search)) return false;
        if (scenarioFilter && it.scenario !== scenarioFilter) return false;
        if (_dsCxPosFilter.size > 0 && !_dsCxPosFilter.has(it.cx_position)) return false;
        if (_dsSchemaFilter && it.schema !== _dsSchemaFilter) return false;
        return true;
      });

      document.getElementById("ds-count-badge").textContent = `${items.length}/${_dsItems.length}개`;

      if (items.length === 0) {
        document.getElementById("ds-list-group").innerHTML =
          "<div style='text-align:center;color:var(--text-muted);font-size:13px;padding:20px 0;'>조건에 맞는 에피소드 없음</div>";
        return;
      }

      let html = "";
      let lastDate = null;
      items.forEach(it => {
        if (it.date !== lastDate) {
          html += `<div class="session-date-header">${it.date}</div>`;
          lastDate = it.date;
        }
        const checked = _dsSelected.has(it.name) ? "checked" : "";
        const checkbox = compareMode
          ? `<input type="checkbox" ${checked} onclick="event.stopPropagation(); _dsToggleSelect('${it.name}')" style="accent-color:var(--cyan); margin-right:6px;">`
          : "";
        html += `
          <div class="session-card" data-name="${it.name}" title="${it.name}" onclick="_dsCardClick('${it.name}')">
            <div class="sc-top">
              ${checkbox}<span style="font-weight:700; font-size:13px; color:var(--cyan); font-family:var(--font-mono);">${it.date} ${it.time}</span>
              <button class="btn btn-outline" style="font-size:10px; padding:2px 7px; margin-left:auto; border-color:var(--rose); color:var(--rose);" onclick="event.stopPropagation(); _dsDeleteOne('${it.name}')">🗑️</button>
            </div>
            <div style="display:flex; flex-wrap:wrap; gap:4px; margin-top:5px;">${_dsInfoChips(it)}</div>
            <div class="sc-bottom">
              <span class="sc-badge">${it.num_frames}f</span>
              <span class="sc-badge">${it.duration_s}s</span>
            </div>
          </div>
        `;
      });
      document.getElementById("ds-list-group").innerHTML = html;

      document.querySelectorAll("#ds-list-group .session-card").forEach(el => {
        el.classList.toggle("active", _dsDetail && el.dataset.name === _dsDetail.meta.name);
      });
    }

    function _dsToggleSelect(name) {
      if (_dsSelected.has(name)) {
        _dsSelected.delete(name);
      } else {
        if (_dsSelected.size >= DS_MAX_COMPARE) {
          alert(`비교는 최대 ${DS_MAX_COMPARE}개까지 가능합니다.`);
          return;
        }
        _dsSelected.add(name);
      }
      if (_dsSelected.size >= 2) {
        renderDatasetCompare();
      } else if (_dsSelected.size === 1) {
        loadDatasetDetail([..._dsSelected][0]);
      } else {
        setDatasetCompareMode(false);
        document.getElementById("ds-placeholder").style.display = "block";
      }
      renderDatasetList();
    }

    function _dsCardClick(name) {
      if (_dsCompareMode) {
        _dsToggleSelect(name);
      } else {
        _dsSelected = new Set([name]);
        loadDatasetDetail(name);
        renderDatasetList();
      }
    }

    function setDatasetCompareMode(on) {
      document.getElementById("ds-placeholder").style.display = on ? "none" : "block";
      document.getElementById("ds-compare-body").style.display = on ? "flex" : "none";
      document.getElementById("ds-inspector-body").style.display = "none";
      if (_dsPlayTimer) { clearInterval(_dsPlayTimer); _dsPlayTimer = null; }
    }

    function renderDatasetCompare() {
      setDatasetCompareMode(true);
      document.getElementById("ds-detail-lbl").textContent = `[비교 ${_dsSelected.size}개]`;
      const items = _dsItems.filter(it => _dsSelected.has(it.name));
      const header = `
        <div style="display:flex; justify-content:flex-end;">
          <button class="btn btn-outline" style="font-size:12px; padding:5px 12px; border-color:var(--rose); color:var(--rose);" onclick="_dsDeleteSelected()">🗑️ 선택 ${items.length}개 전부 삭제</button>
        </div>`;
      const cards = items.map(it => {
        const scLabel = it.scenario ? (_dsScenarioLabels[it.scenario] || it.scenario) : "미지정";
        const cxLabel = it.cx_position ? (_dsCxPosLabels[it.cx_position] || it.cx_position) : "미지정";
        return `
          <div style="background:#151f32; border:1px solid var(--border-glow); border-radius:10px; padding:12px 14px; display:flex; align-items:center; gap:14px;">
            <div style="flex:1; min-width:0;">
              <div style="font-family:var(--font-mono); font-size:12px; color:var(--cyan); overflow:hidden; text-overflow:ellipsis; white-space:nowrap;" title="${it.name}">${it.date} ${it.time}</div>
              <div style="font-size:11px; color:var(--text-muted); margin-top:4px;">${scLabel} · ${cxLabel}${it.cx_path ? " + " + (it.cx_path) : ""}</div>
            </div>
            <div class="mini-tile-grid" style="flex:0 0 auto; grid-template-columns:repeat(3,1fr); gap:6px;">
              <div class="mini-tile"><span class="mt-label">프레임</span><span class="mt-value">${it.num_frames}</span></div>
              <div class="mini-tile"><span class="mt-label">시간</span><span class="mt-value">${it.duration_s}s</span></div>
              <div class="mini-tile"><span class="mt-label">크기</span><span class="mt-value">${it.size_mb}MB</span></div>
            </div>
            <button class="btn btn-outline" style="font-size:11px; padding:4px 10px; flex:0 0 auto;" onclick="_dsOpenDetailFromCompare('${it.name}')">🔍 자세히</button>
            <button class="btn btn-outline" style="font-size:11px; padding:4px 10px; flex:0 0 auto; border-color:var(--rose); color:var(--rose);" onclick="_dsDeleteOne('${it.name}')">🗑️</button>
          </div>
        `;
      }).join("");
      document.getElementById("ds-compare-body").innerHTML = header + cards;
    }

    function _dsOpenDetailFromCompare(name) {
      loadDatasetDetail(name);
    }

    async function loadDatasetDetail(name) {
      document.getElementById("ds-detail-lbl").textContent = "[" + name + "]";
      document.getElementById("ds-placeholder").style.display = "none";
      document.getElementById("ds-compare-body").style.display = "none";
      document.getElementById("ds-inspector-body").style.display = "grid";
      if (_dsPlayTimer) { clearInterval(_dsPlayTimer); _dsPlayTimer = null; }

      const res = await api("/dataset/load?name=" + encodeURIComponent(name));
      if (!res.ok) {
        alert("에피소드 로드 실패: " + res.error);
        return;
      }
      _dsDetail = res;

      const slider = document.getElementById("ds-slider");
      slider.max = res.frames.length - 1;
      slider.value = 0;

      renderDsSummary(res);
      renderDsTimeline(res.frames, 0);
      showDsFrame(0);
      renderDatasetList();
    }

    function renderDsSummary(res) {
      const m = res.meta;
      const a = res.attrs || {};
      document.getElementById("ds-sum-frames").textContent = m.num_frames;
      document.getElementById("ds-sum-duration").textContent = m.duration_s + "s";
      document.getElementById("ds-sum-scenario").textContent = m.scenario ? (_dsScenarioLabels[m.scenario] || m.scenario) : "미지정";
      document.getElementById("ds-sum-cxpos").textContent = m.cx_position
        ? `${_dsCxPosLabels[m.cx_position] || m.cx_position}${m.cx_path ? " + " + m.cx_path : ""}` : "미지정";
      document.getElementById("ds-sum-schema").textContent = DS_SCHEMA_VERSION[m.schema] || m.schema;
      document.getElementById("ds-sum-date").textContent = m.date + " " + m.time;

      // 수집 설정 — 레거시/신규 스키마에 따라 존재하는 attrs가 다름
      document.getElementById("ds-sum-pattern").textContent = a.pattern || "—";
      document.getElementById("ds-sum-obstacle").textContent = a.obstacle_layout_type || a.end_pos || "—";
      document.getElementById("ds-sum-timeperiod").textContent = a.time_period || "—";
      document.getElementById("ds-sum-stopinject").textContent = (a.stop_inject_n !== undefined ? a.stop_inject_n : "—");
      document.getElementById("ds-sum-chunk").textContent = (a.action_chunk_size !== undefined ? a.action_chunk_size : "—");
      document.getElementById("ds-sum-size").textContent = m.size_mb + "MB";

      // 입력 소스 분포 — action_event_types(keyboard/joystick/stop_inject 등)
      const srcDist = {};
      res.frames.forEach(f => {
        const src = f.event_type || "(레거시-미기록)";
        srcDist[src] = (srcDist[src] || 0) + 1;
      });
      document.getElementById("ds-sum-sources").innerHTML = Object.entries(srcDist).map(([k, v]) =>
        `<div class="mini-tile"><span class="mt-label">${k}</span><span class="mt-value">${v}</span></div>`
      ).join("");

      const dist = {};
      res.frames.forEach(f => { dist[f.action_class] = (dist[f.action_class] || 0) + 1; });
      const total = res.frames.length || 1;
      document.getElementById("ds-sum-actions").innerHTML = Object.entries(dist).map(([k, v]) =>
        `<div class="mini-tile"><span class="mt-label">${k}</span><span class="mt-value">${v} (${Math.round(v/total*100)}%)</span></div>`
      ).join("");
    }

    const DS_CLASS_COLORS = {
      STOP: "#64748b", FORWARD: "#10b981", LEFT: "#38bdf8", RIGHT: "#38bdf8",
      "FWD+L": "#a371f7", "FWD+R": "#a371f7", ROT_L: "#f59e0b", ROT_R: "#f59e0b",
    };

    function renderDsTimeline(frames, curIdx) {
      const el = document.getElementById("ds-timeline");
      el.innerHTML = frames.map((f, i) => {
        const color = i === curIdx ? "#06b6d4" : (DS_CLASS_COLORS[f.action_class] || "#64748b");
        return `<div title="${i}: ${f.action_class}" onclick="document.getElementById('ds-slider').value=${i}; showDsFrame(${i});" style="flex:1; background:${color}; cursor:pointer;"></div>`;
      }).join("");
    }

    function showDsFrame(idx) {
      idx = parseInt(idx);
      if (!_dsDetail || !_dsDetail.frames[idx]) return;
      const f = _dsDetail.frames[idx];
      document.getElementById("ds-frame-img").src = `/dataset/frame?name=${encodeURIComponent(_dsDetail.meta.name)}&idx=${idx}`;
      document.getElementById("ds-frame-idx-lbl").textContent = `Frame: ${idx} / ${_dsDetail.frames.length - 1}`;
      document.getElementById("ds-frame-action-lbl").textContent = `${f.symbol} ${f.action_class}`;
      renderDsTimeline(_dsDetail.frames, idx);
    }

    function dsPrevFrame() {
      const slider = document.getElementById("ds-slider");
      const v = Math.max(0, parseInt(slider.value) - 1);
      slider.value = v; showDsFrame(v);
    }
    function dsNextFrame() {
      const slider = document.getElementById("ds-slider");
      const v = Math.min(parseInt(slider.max), parseInt(slider.value) + 1);
      slider.value = v; showDsFrame(v);
    }
    function toggleDsPlay() {
      const btn = document.getElementById("btn-ds-play");
      if (_dsPlayTimer) {
        clearInterval(_dsPlayTimer); _dsPlayTimer = null;
        btn.textContent = "▶ PLAY";
        return;
      }
      btn.textContent = "⏸ PAUSE";
      _dsPlayTimer = setInterval(() => {
        const slider = document.getElementById("ds-slider");
        const v = parseInt(slider.value);
        if (v >= parseInt(slider.max)) { toggleDsPlay(); return; }
        slider.value = v + 1; showDsFrame(v + 1);
      }, 120);
    }

    // 초기 스타트
    setInterval(pollStatus, 500);
    setInterval(pollHealth, 3000);
    setInterval(camProcRefresh, 10000);
    setInterval(joystickRefresh, 500);
    // D-pad(하드웨어)로 바뀐 collect_mode/staged_scenario/staged_cx_position 등은
    // 키 입력 없이도 즉시 반영되어야 하므로 별도로 상시 폴링 — 이전엔 키보드를
    // 누르고 있을 때(2000ms)만 갱신돼 D-pad 단독 조작 시 반응이 느려 보였음.
    setInterval(collectRefreshState, 500);
    pollStatus();
    pollHealth();
    loadEpisodeHistory();
    camProcRefresh();
    joystickRefresh();
    collectRefreshState();
    _renderTrackAPathButtons("verify-tracka-grid", "btn btn-outline", "selectPathType");
    _renderTrackAPathButtons("inspect-tracka-grid", "btn btn-outline iep-path-btn", "selectInspectPathType");
    refreshCheckpointOptions();
    refreshModelList();

  </script>
</body>
</html>"""


# ═══════════════════════════════════════════════════════════════════
# 카메라 프로세스 제어 (usb_camera_service_server) — Gradio 대시보드 이식
# ═══════════════════════════════════════════════════════════════════
import subprocess as _subprocess

_CAM_KILL_PATTERN = "usb_camera_service_server"
_CAM_ROS_DIST = "/opt/ros/humble"
_CAM_ROS_WS = str(ROOT / "ROS_action" / "install")
_CAM_START_CMD = (
    f"export PATH={_CAM_ROS_DIST}/bin:$PATH; "
    f"export PYTHONPATH={_CAM_ROS_DIST}/local/lib/python3.10/dist-packages"
    f":{_CAM_ROS_DIST}/lib/python3.10/site-packages"
    f":{_CAM_ROS_WS}/camera_interfaces/local/lib/python3.10/dist-packages"
    f":{_CAM_ROS_WS}/camera_pub/local/lib/python3.10/dist-packages:$PYTHONPATH; "
    f"export LD_LIBRARY_PATH={_CAM_ROS_DIST}/lib"
    f":{_CAM_ROS_WS}/camera_interfaces/lib:{_CAM_ROS_WS}/camera_pub/lib:$LD_LIBRARY_PATH; "
    f"source {_CAM_ROS_DIST}/setup.bash 2>/dev/null; "
    f"cd {_CAM_ROS_WS} && source setup.bash 2>/dev/null; "
    "export ROS_DOMAIN_ID=42; "
    "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp; "
    "nohup ros2 run camera_pub usb_camera_service_server "
    f"> {ROOT}/logs/camera_pub.log 2>&1 &"
)


def _cam_pids() -> list[str]:
    r = _subprocess.run(["pgrep", "-f", _CAM_KILL_PATTERN], capture_output=True, text=True)
    return [p for p in r.stdout.strip().split() if p]


@app.get("/camera_proc/status")
def camera_proc_status():
    pids = _cam_pids()
    return {"running": bool(pids), "pids": pids,
            "text": (f"🟢 실행 중 pid={','.join(pids)}" if pids else "🔴 정지됨")}


@app.post("/camera_proc/start")
def camera_proc_start():
    global _cam_user_stopped
    _cam_user_stopped = False  # 사용자가 다시 켰으면 워치독 재개
    if _cam_pids():
        return {"ok": True, "already_running": True, "text": f"🟢 이미 실행 중 pid={','.join(_cam_pids())}"}
    _subprocess.Popen(["bash", "-c", _CAM_START_CMD])
    return {"ok": True, "already_running": False, "text": "⏳ 시작 중... (몇 초 후 새로고침)"}


@app.post("/camera_proc/stop")
def camera_proc_stop():
    global _cam_user_stopped
    _cam_user_stopped = True  # 의도적 정지 — 워치독이 되살리지 않도록
    _subprocess.run(["pkill", "-9", "-f", _CAM_KILL_PATTERN])
    time.sleep(0.4)
    pids = _cam_pids()
    return {"ok": True, "text": (f"🟢 실행 중 pid={','.join(pids)}" if pids else "🔴 정지됨")}


# ═══════════════════════════════════════════════════════════════════
# 추론 서버(8001) 프로세스 관리 — ⚙️ 서버 설정 탭용
# ═══════════════════════════════════════════════════════════════════
class ServerRestartReq(BaseModel):
    grounder: Optional[str] = None       # "pg2" | "owlv2"
    owlv2_thresh: Optional[float] = None
    owlv2_area_scale: Optional[float] = None
    ckpt: Optional[str] = None           # runs/ 상대경로 (VLA_S2V2_STAGE2)


def _classify_ckpt(rel_path: str) -> dict:
    """체크포인트 경로를 종류/실험/헤드로 파싱해 사람이 읽을 라벨 생성.

    kind 분류가 중요한 이유: VLA_S2V2_STAGE2로 교체 가능한 건 Stage2 액션
    헤드뿐 — stop_*(STOP 헤드)나 stage1_*(인코더 projection)을 넣으면 서버가
    로드 단계에서 깨진다. UI에서 액션 헤드만 선택 가능하게 막는 근거.
    """
    import re as _re
    p = Path(rel_path)
    name = p.stem            # e.g. action_transformer, stop_N1, mlp_w16
    parent = p.parent.name   # e.g. exp71_window6, stop_lastN, ablation_window
    full = f"{parent}/{name}".lower()

    # kind
    if "stop" in full:
        kind, kind_label = "stop", "STOP 헤드"
    elif "stage1" in full or "projs" in full:
        kind, kind_label = "stage1", "Stage1 인코더"
    elif "clip" in full or "lora" in full:
        kind, kind_label = "other", "기타 (CLIP/LoRA)"
    else:
        kind, kind_label = "action", "액션 헤드"

    # head 아키텍처
    head = next((h for h in ("transformer", "lstm", "cx_geom", "linear", "mlp", "fc")
                 if h in full), None)

    # 실험 그룹
    m = _re.search(r"(exp\d+)", full)
    if m:
        group = m.group(1)
    elif "ablation" in full:
        group = "ablation"
    elif "data_exp" in full:
        group = "data_exp"
    elif parent in ("mlp", "runs"):
        group = "루트"
    else:
        group = parent

    # window 크기
    w = _re.search(r"w(?:indow)?[_]?(\d+)", full)
    label_parts = [group]
    if head: label_parts.append(head.upper() if head in ("mlp", "fc", "lstm") else head.capitalize())
    if w: label_parts.append(f"W{w.group(1)}")
    extra = _re.sub(r"^(action|mlp|lstm|fc|linear|stop|stage1)[_]?", "", name)
    extra = _re.sub(r"w(?:indow)?\d+|transformer|lstm|cx_geom|linear|mlp|fc", "", extra).strip("_")
    if extra and extra not in group: label_parts.append(extra)

    return {"kind": kind, "kind_label": kind_label,
            "label": " · ".join(label_parts), "selectable": kind == "action"}


@app.get("/server_proc/checkpoints")
def server_proc_checkpoints():
    """runs/ 아래 .pt 체크포인트 목록 — 종류별 분류 + 라벨 포함."""
    items = []
    for p in sorted((ROOT / "runs").rglob("*.pt"), key=lambda x: -x.stat().st_mtime)[:80]:
        st = p.stat()
        rel = str(p.relative_to(ROOT))
        items.append({
            "path": rel,
            "size_mb": round(st.st_size / 1e6, 1),
            "mtime": datetime.datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M"),
            **_classify_ckpt(rel),
        })
    active = None
    try:
        import requests as rq
        h = rq.get(f"{INFER_URL}/health", headers={"X-API-Key": API_KEY}, timeout=2).json()
        active = h.get("checkpoint_path")
    except Exception:
        pass
    return {"ok": True, "checkpoints": items, "active": active}


@app.post("/server_proc/restart")
def server_proc_restart(req: ServerRestartReq):
    """추론 서버(8001)를 지정 설정으로 재시작 — go.sh --server를 백그라운드 실행.

    go.sh가 pkill→로그 보관→기동→헬스대기까지 처리하므로 여기선 env만 조립해
    던진다. PG2 로딩 포함 최대 ~120s — UI가 /infer/health를 폴링해 완료 감지.

    req의 필드가 비어있으면(=UI에서 명시적으로 안 건드림) go.sh 자체의 하드코딩된
    기본값(VLA_GROUNDER=pg2 등)으로 조용히 되돌아가버리는 문제가 있었음 — "변경
    없음 — 현재 설정 유지" 버튼을 눌러도 실제로는 유지가 안 되고 grounder/ckpt가
    리셋됐음(2026-07-07 실측). 재시작 전 현재 /health를 조회해서 명시 안 된 필드는
    "지금 떠 있는 값"으로 채워 넣어 진짜로 유지되게 함.
    """
    current = {}
    try:
        import requests as rq
        h = rq.get(f"{INFER_URL}/health", headers={"X-API-Key": API_KEY}, timeout=2).json()
        g = h.get("grounder", {}) or {}
        current = {
            "grounder": "owlv2" if "owl" in (g.get("model") or "").lower() else "pg2",
            "owlv2_thresh": g.get("owlv2_thresh"),
            "owlv2_area_scale": g.get("owlv2_area_scale"),
            "ckpt": h.get("checkpoint_path"),
        }
    except Exception:
        pass  # 서버가 이미 내려가 있으면 go.sh 기본값으로 진행

    grounder = req.grounder if req.grounder in ("pg2", "owlv2") else current.get("grounder")
    owlv2_thresh = req.owlv2_thresh if req.owlv2_thresh is not None else current.get("owlv2_thresh")
    owlv2_area_scale = req.owlv2_area_scale if req.owlv2_area_scale is not None else current.get("owlv2_area_scale")
    ckpt = req.ckpt if req.ckpt else current.get("ckpt")

    env_parts = []
    if grounder in ("pg2", "owlv2"):
        env_parts.append(f"VLA_GROUNDER={grounder}")
    if owlv2_thresh is not None:
        env_parts.append(f"VLA_OWLV2_THRESH={float(owlv2_thresh)}")
    if owlv2_area_scale is not None:
        env_parts.append(f"VLA_OWLV2_AREA_SCALE={float(owlv2_area_scale)}")
    if ckpt:
        ckpt_path = (ROOT / ckpt).resolve()
        if not str(ckpt_path).startswith(str(ROOT)) or not ckpt_path.exists():
            return {"ok": False, "error": f"체크포인트 없음/경로 이탈: {ckpt}"}
        env_parts.append(f"VLA_S2V2_STAGE2={ckpt}")
    cmd = f"cd {ROOT} && {' '.join(env_parts)} nohup bash scripts/run/go.sh --server > logs/server_restart_from_dash.log 2>&1 &"
    _subprocess.Popen(["bash", "-c", cmd])
    log.warning(f"🔁 [ServerRestart] 대시보드에서 추론서버 재시작 요청: {env_parts or '(현재 설정 유지)'}")
    return {"ok": True, "message": "재시작 시작 — PG2 포함 시 최대 ~120s. 아래 상태가 갱신될 때까지 대기.",
            "env": env_parts}


@app.get("/server_proc/log")
def server_proc_log(n: int = 40):
    p = ROOT / "logs" / "s2v2_server.log"
    if not p.exists():
        return {"ok": False, "lines": ["(로그 없음)"]}
    lines = p.read_text(errors="replace").splitlines()[-max(5, min(n, 200)):]
    return {"ok": True, "lines": lines}


# ── 카메라 워치독 (B3) ───────────────────────────────────────────────
# usb_camera_service_server가 살아있는 척하며 같은 프레임만 반복 전달하는
# 행(hang)이 2026-07-02 하루 3회 발생 — frame 정체가 지속되면 자동 재시작.
CAM_WATCHDOG_STALE_S    = 10.0   # 이 시간 이상 새 프레임 없으면 행으로 판정
CAM_WATCHDOG_COOLDOWN_S = 120.0  # 자동 재시작 간 최소 간격 (재시작 루프 방지)
_cam_user_stopped = False        # 사용자가 ■정지 누른 상태면 워치독 개입 금지
_cam_auto_restart_ts = 0.0


def _camera_watchdog_loop():
    global _cam_auto_restart_ts
    while True:
        time.sleep(5.0)
        try:
            if _ros is None or _cam_user_stopped:
                continue
            if not _ros.last_ts:          # 아직 첫 프레임 전 (기동 직후)
                continue
            age = time.time() - _ros.last_ts
            if age < CAM_WATCHDOG_STALE_S:
                continue
            if time.time() - _cam_auto_restart_ts < CAM_WATCHDOG_COOLDOWN_S:
                continue
            _cam_auto_restart_ts = time.time()
            log.warning(f"🐶 [CamWatchdog] 프레임 정체 {age:.1f}s — 카메라 서비스 자동 재시작")
            _subprocess.run(["pkill", "-9", "-f", _CAM_KILL_PATTERN])
            time.sleep(1.0)
            _subprocess.Popen(["bash", "-c", _CAM_START_CMD])
        except Exception as e:
            log.warning(f"[CamWatchdog] 오류(무시): {e}")


threading.Thread(target=_camera_watchdog_loop, daemon=True, name="cam-watchdog").start()


# ═══════════════════════════════════════════════════════════════════
# 조이스틱(DragonRise 게임패드) 제어 — Gradio 대시보드 기능 이식
# ═══════════════════════════════════════════════════════════════════
@app.get("/joystick/status")
def joystick_status():
    s = _joystick.status
    return {"pygame_available": PYGAME_AVAILABLE, **s}


@app.post("/joystick/toggle")
def joystick_toggle():
    enabled = _joystick.toggle_enabled()
    return {"ok": True, "enabled": enabled}


@app.post("/joystick/mode")
def joystick_mode():
    mode = _joystick.toggle_mode()
    return {"ok": True, "mode": mode}


class JoystickSpeedReq(BaseModel):
    speed: float = 1.15


@app.post("/joystick/speed")
def joystick_speed(req: JoystickSpeedReq):
    _joystick.set_speed(req.speed)
    return {"ok": True, "speed": _joystick._speed}


# ═══════════════════════════════════════════════════════════════════
# 강제 클리어 추가 엔드포인트
# ═══════════════════════════════════════════════════════════════════
@app.post("/system/reset")
def system_reset():
    """모든 프로세스 세션 강제 초기화 및 OOM 완화 조치"""
    if _ros and _ros.ctrl:
        _ros.ctrl.robust_stop(source="system_reset")
    try:
        # 8001 리셋
        _infer_post("/reset", {}, timeout=3)
    except Exception:
        pass
    _state.update(running=False, step=0, goal_near=False,
                  grounding_cached=None, grounding_caption=None,
                  run_history=[], action_history=[], last_action=[0.0,0.0,0.0])
    return {"ok": True, "message": "성공적으로 시스템 상태 및 추론 캐시가 리셋되었습니다."}


# ═══════════════════════════════════════════════════════════════════
# 진입점
# ═══════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="MoNaVLA Command Center v2")
    ap.add_argument("--port", type=int,
                    default=int(os.getenv("DRIVE_PORT", "7800")))
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    import uvicorn
    log.info(f"🚀 MoNaVLA Command Center  http://{SODA_IP}:{args.port}")
    log.info(f"   Inference URL: {INFER_URL}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

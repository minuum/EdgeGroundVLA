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
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional, List

import numpy as np
import h5py
from fastapi import FastAPI, Response, Header
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image

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
COLLECT_CX_POSITIONS = {
    "strong_left":  {"label": "강한좌",   "lo": 0.10, "hi": 0.15, "target": 45},
    "weak_left":    {"label": "준극단좌", "lo": 0.20, "hi": 0.25, "target": 45},
    "weak_right":   {"label": "준극단우", "lo": 0.75, "hi": 0.80, "target": 45},
    "strong_right": {"label": "강한우",   "lo": 0.85, "hi": 0.90, "target": 45},
}
# 위치당 45개가 "경로 다양하게"로 뭉뚱그려져 있으면 실제로 15/15/15가 지켜졌는지
# 검증이 안 됨 — 위치×경로 조합별로 세분화해서 목표 15씩 따로 추적.
COLLECT_TRACKA_PATHS = {
    "left_curve":  {"label": "좌곡선", "target": 15},
    "straight":    {"label": "직진",   "target": 15},
    "right_curve": {"label": "우곡선", "target": 15},
}


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
            return {"ok": True, "saved": False, "steps": len(data)}
        if len(data) <= 1:
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

    def state(self) -> dict:
        return {
            "active": self.active,
            "episode_name": self.episode_name,
            "steps": len(self.episode_data),
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

    버튼 매핑:
      A (0)     → STOP (robust_stop)
      Start (7) → SYNC ↔ ASYNC 모드 전환

    SYNC 모드: 0.45s 간격으로 move_and_stop_timed() — V5 bang-bang 호환
    ASYNC 모드: 10Hz 연속 publish_and_move() + 300ms Jitter Hold + 중립 시 robust_stop()
    """

    DEADZONE       = 0.15
    THRESHOLD      = 0.50
    STEP_INTERVAL  = 0.45   # SYNC bang-bang 간격 (s)
    ASYNC_INTERVAL = 0.10   # ASYNC 연속 발행 간격 (s) — 10Hz
    JITTER_HOLD    = 0.30   # ASYNC 중립 후 정지 유예 시간 (s)
    DEFAULT_AXES   = {"left_x": 0, "left_y": 1, "right_x": 2}
    # DragonRise 기본 버튼 인덱스 (Gradio gradio_data_collector.py와 동일 매핑)
    BTN_STOP       = 0   # A     — STOP (robust_stop)
    BTN_UNDO       = 1   # B     — 마지막 프레임 취소
    BTN_DISCARD    = 2   # X     — 에피소드 폐기(저장 안 함)
    BTN_TELEOP     = 3   # Y     — 미사용 (대시보드엔 teleop 개념 없음)
    BTN_REC_START  = 4   # L1    — 녹화 시작
    BTN_REC_SAVE   = 5   # R1    — 정지 & 저장
    BTN_SELECT     = 6   # Select — 녹화 토글(시작↔저장)
    BTN_TOGGLE     = 7   # Start  — SYNC/ASYNC 모드 전환
    TRIG_R2        = 5   # R2(트리거, 축) — Gradio "controller" 레이아웃과 동일: 경로 역재생 복귀
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
        self.status: dict = {
            "connected": False, "name": "—",
            "key": None, "label": "—",
            "enabled": True, "mode": "ASYNC",
            "buttons": [], "last_btn": None,
            "hat": (0, 0), "last_hat_dir": None,
        }

    def _load_axes(self):
        cfg = ROOT / "scripts" / "joystick_config.json"
        if cfg.exists():
            try:
                return json.load(open(cfg)).get("axes", self.DEFAULT_AXES)
            except Exception:
                pass
        return dict(self.DEFAULT_AXES)

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
            _ros.ctrl.throttle = max(10, min(100, throttle))

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
                self.status = {
                    "connected": True, "name": js.get_name(),
                    "enabled": self._enabled,
                    "mode": self._js_mode.upper(),
                    "key": key, "label": LABELS.get(key, "●") if key else "○",
                    "buttons": pressed_buttons, "last_btn": self._last_btn,
                    "hat": self._hat_prev, "last_hat_dir": self._last_hat_dir,
                }

                for i in range(js.get_numbuttons()):
                    cur = js.get_button(i)
                    if cur and not self._btn_prev.get(i, 0):
                        self._last_btn = i
                        if i == self.BTN_STOP:
                            if _ros is not None and _ros.ctrl is not None:
                                _ros.ctrl.robust_stop(source="joystick_A")
                        elif i == self.BTN_TOGGLE:
                            self.toggle_mode()
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
                    self._btn_prev[i] = cur

                # D-pad(hat) 엣지 → 좌/우: 현재 활성 축(collect_mode) 순환, 상/하: 트랙A↔시나리오 축 전환
                if js.get_numhats() > 0:
                    hat = js.get_hat(0)
                    if hat != self._hat_prev:
                        hx, hy = hat
                        phx, phy = self._hat_prev
                        if hx != 0 and hx != phx:
                            self._last_hat_dir = "right" if hx > 0 else "left"
                            if _collect is not None:
                                _collect.cycle_current(1 if hx > 0 else -1)
                        elif hy != 0 and hy != phy:
                            self._last_hat_dir = "up" if hy > 0 else "down"
                            if _collect is not None:
                                _collect.toggle_collect_mode(1 if hy > 0 else -1)
                        self._hat_prev = hat

                # R2(트리거, 축) 엣지 → 복귀(경로 역재생), Gradio "controller" 레이아웃과 동일
                nax = js.get_numaxes()
                if 0 <= self.TRIG_R2 < nax:
                    tv = js.get_axis(self.TRIG_R2)
                    if tv > self.TRIG_THRESHOLD and self._trig_r2_prev <= self.TRIG_THRESHOLD:
                        if _collect is not None:
                            _collect.start_auto_return()
                    self._trig_r2_prev = tv

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
def _infer_post(path: str, payload: dict, timeout=15) -> dict:
    import requests as rq
    r = rq.post(f"{INFER_URL}{path}", json=payload,
                headers={"X-API-Key": API_KEY}, timeout=timeout)
    r.raise_for_status()
    return r.json()


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
    logger.data["runtime_config"] = _snapshot_runtime_config()
    _state["session_id"] = logger.session_id

    _ros.ctrl.robust_stop(source="start")
    time.sleep(0.20)
    try: _infer_post("/reset", {}, timeout=5)
    except Exception: pass

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
    try: _infer_post("/reset", {}, timeout=5)
    except Exception: pass

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
    COAST = 1.2
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
                  run_history=[], action_history=[], last_action=[0.0,0.0,0.0])
    _stop_ev.clear()
    _async_q.clear()

    if req.mode in ("SYNC", "PRE"):
        threading.Thread(target=_loop_sync,
                         args=(req.mode, req.instruction, req.gt_object, req.apply_cc),
                         daemon=True, name="drive-sync").start()
    else:
        from scripts.inference_logger import get_logger
        logger = get_logger()
        logger.start_session("stage2_v2", req.instruction, instruction_mode="ASYNC")
        if req.gt_object: logger.data["gt_object"] = req.gt_object
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
    except Exception as e:
        return {"ok": False, "error": str(e)}


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
        
        # 액션 레이블 매핑 도우미
        _amap = {
            (0.0,0.0,0.0):"STOP", (1.15,0.0,0.0):"FWD",
            (0.0,1.15,0.0):"LEFT", (0.0,-1.15,0.0):"RIGHT",
            (1.15,1.15,0.0):"FWD+L", (1.15,-1.15,0.0):"FWD+R",
            (0.0,0.0,0.25):"ROT_L", (0.0,0.0,-0.25):"ROT_R",
        }
        def _lbl(a):
            # 구버전 세션은 az 없이 (lx, ly) 2열만 기록된 경우가 있음 — az=0으로 패딩
            a3 = [float(a[0]), float(a[1]), float(a[2]) if len(a) > 2 else 0.0]
            for k, v in _amap.items():
                if all(abs(a3[i]-k[i])<0.05 for i in range(3)): return v
            return f"({a3[0]:.1f},{a3[1]:.1f})"

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
            
        return {
            "ok": True,
            "sid": sid,
            "attrs": {k: str(v) for k, v in attrs.items()},
            "frames": frames_meta
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
EP_HEADERS  = ["#", "경로", "결과", "steps", "lat(ms)", "top액션", "gnd%", "area", "cx", "STOP", "FPE", "메모", "날짜", "session_id"]
PATH_TYPES = ["right_right", "right_left", "right_straight",
              "center_straight", "center_left", "center_right",
              "left_straight", "left_left", "left_right",
              "obj_left", "obj_center", "obj_right",
              "dist_10cm", "dist_20cm", "dist_30cm"]
PATH_TARGETS = {
    "right_right": 10, "right_left": 10, "right_straight": 10,
    "center_straight": 10, "center_left": 10, "center_right": 10,
    "left_straight": 10, "left_left": 10, "left_right": 10,
    "obj_left": 30, "obj_center": 30, "obj_right": 30,
    "dist_10cm": 10, "dist_20cm": 10, "dist_30cm": 10,
}

def _get_episode_summary(rows):
    done_total = {k: 0 for k in PATH_TYPES}
    done_succ  = {k: 0 for k in PATH_TYPES}
    nav_succ = 0
    for r in rows:
        if len(r) < 3: continue
        pt = str(r[1]).replace(" ★", "").replace("★", "").strip()
        done_total[pt] = done_total.get(pt, 0) + 1
        if r[2] == "성공":
            done_succ[pt] = done_succ.get(pt, 0) + 1
            if not pt.startswith(("obj_", "dist_")):
                nav_succ += 1
    nav_total = sum(PATH_TARGETS[k] for k in PATH_TARGETS if not k.startswith(("obj_", "dist_")))
    obj_done  = sum(done_total.get(k, 0) for k in ("obj_left","obj_center","obj_right"))
    obj_succ  = sum(done_succ.get(k, 0)  for k in ("obj_left","obj_center","obj_right"))
    dist_done = sum(done_total.get(k, 0) for k in ("dist_10cm","dist_20cm","dist_30cm"))
    dist_succ = sum(done_succ.get(k, 0)  for k in ("dist_10cm","dist_20cm","dist_30cm"))
    return (f"경로검증 {sum(done_total.get(k,0) for k in PATH_TYPES if not k.startswith(('obj_','dist_')))}/{nav_total} "
            f"성공 {nav_succ}/20 (목표) | 위치별 {obj_done}/90 ({obj_succ}성공) | 거리별 {dist_done}/30 ({dist_succ}성공)")

def _read_episode_csv():
    if not EPISODE_CSV.exists():
        return [], "에피소드 기록 없음"
    import csv
    rows = []
    try:
        with open(EPISODE_CSV, newline="", encoding="utf-8") as f:
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

@app.get("/episodes/list")
def episodes_list():
    rows, summary = _read_episode_csv()
    return {"ok": True, "episodes": rows, "summary": summary}

@app.post("/episodes/log")
def episodes_log(req: EpisodeLogReq):
    rows, _ = _read_episode_csv()
    
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
    EPISODE_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not EPISODE_CSV.exists()
    with open(EPISODE_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(EP_HEADERS)
        w.writerow(new_row)
        
    _, summary = _read_episode_csv()
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
      <div class="nav-item" onclick="switchTab(this, 'system')">🖥️ 시스템</div>
      <div class="nav-item" onclick="switchTab(this, 'srvcfg')">⚙️ 서버 설정</div>
      <div class="nav-item" onclick="switchTab(this, 'collect')">📷 데이터수집</div>
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

            <div class="table-wrapper" style="max-height:380px; overflow-y:auto; border:1px solid var(--border-glow); border-radius:8px;">
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
                <select id="ep-path-type" style="width:100%; padding:8px; background:#090d16; border:1px solid var(--border-glow); border-radius:6px; color:#fff; font-size:13px;">
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
              <div class="table-wrapper" style="max-height:240px; overflow-y:auto; border:1px solid var(--border-glow); border-radius:8px;">
                <table style="width:100%; border-collapse:collapse; font-size:11px;">
                  <thead style="background:#151f32; border-bottom:1px solid var(--border-glow); text-align:left; position:sticky; top:0;">
                    <tr>
                      <th style="padding:6px 8px;">#</th>
                      <th style="padding:6px 8px;">경로</th>
                      <th style="padding:6px 8px;">결과</th>
                      <th style="padding:6px 8px;">steps</th>
                      <th style="padding:6px 8px;">lat</th>
                      <th style="padding:6px 8px;">FPE</th>
                      <th style="padding:6px 8px;">메모</th>
                      <th style="padding:6px 8px;">날짜</th>
                    </tr>
                  </thead>
                  <tbody id="episodes-table-body">
                    <tr><td colspan="8" style="text-align:center; padding:12px; color:var(--text-muted);">기록이 없습니다.</td></tr>
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
                
                <div style="display:flex; gap:8px; margin-top:12px; justify-content:center;">
                  <button class="btn btn-outline" onclick="prevInspectFrame()">◀ 이전</button>
                  <button class="btn btn-outline" id="btn-inspect-play" onclick="toggleInspectPlay()">▶ PLAY</button>
                  <button class="btn btn-outline" onclick="nextInspectFrame()">다음 ▶</button>
                </div>
              </div>
              
              <!-- 프레임 메타데이터 & 셀프 라벨링 -->
              <div style="display:flex; flex-direction:column; justify-content:space-between;">
                <div>
                  <div class="form-group">
                    <label>Action & Latency</label>
                    <div class="kv-grid" style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:4px;">
                      <div class="srv-pill" style="padding:4px 8px;">액션: <strong id="inspect-action-lbl">—</strong></div>
                      <div class="srv-pill" style="padding:4px 8px;">지연: <strong id="inspect-lat-lbl">—ms</strong></div>
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
                
                <div class="srv-pill" style="font-size:11px; margin-top:20px; display:block;">
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

        <!-- 상단 줄: 카메라 : 조이스틱 : 시나리오&진행률 = 2:1:1, 한눈에 쭉 보이도록 -->
        <div style="display:grid; grid-template-columns:2fr 1fr 1fr; gap:20px; align-items:start; margin-bottom:20px;">
          <div style="display:flex; flex-direction:column; gap:16px;">
            <div class="card" style="padding:16px;">
              <div class="card-title">📹 실시간 카메라 (cx 오버레이는 실시간 cx 켜면 표시)
                <label style="font-size:11px; font-weight:400; color:var(--text-muted); float:right; cursor:pointer;">
                  <input type="checkbox" id="toggle-grid-collect" checked onchange="_collectCxDrawOverlay(_collectLastCx, _collectLastColor)" style="accent-color:var(--cyan);"> Grid 표시
                </label>
              </div>
              <div class="viewport-wrapper">
                <img id="collect-stream-img" src="/camera/stream" class="viewport-img"
                     onerror="this.src='https://placehold.co/1280x720/0f1524/94a3b8?text=Camera+Streaming+Offline'">
                <canvas id="collect-cx-canvas" class="viewport-canvas" width="1280" height="720"></canvas>
              </div>
            </div>

            <div class="card" style="padding:16px;">
              <div class="card-title">🎯 실시간 cx (바구니 배치용)
                <span id="collect-cx-toggle-badge" style="font-size:10px; padding:2px 8px; border-radius:10px; background:rgba(100,116,139,0.2); color:var(--text-muted); cursor:pointer;" onclick="collectToggleCxFeed()">⏸ 꺼짐 — 클릭해서 시작</span>
              </div>
              <div id="collect-cx-value" style="font-size:32px; font-family:var(--font-mono); font-weight:700; text-align:center; padding:8px;">—</div>
              <div id="collect-cx-band" style="font-size:12px; text-align:center; color:var(--text-muted);">극단 배치 기준: 0.10~0.15 강한좌 · 0.20~0.25 준극단좌 · 0.75~0.80 준극단우 · 0.85~0.90 강한우</div>
            </div>

            <div class="card" style="padding:16px;">
              <div class="card-title">📍 시작 프레임 기준 cx 배치 가이드
                <button class="btn btn-outline" style="font-size:11px; padding:3px 10px; float:right;" onclick="_collectCaptureGuide()">📸 캡처</button>
              </div>
              <div style="font-size:11px; color:var(--text-muted); margin-bottom:8px;">지금 화면을 캡처해서 극단 cx 구간을 선/밴드+수치로 표시 — 바구니를 어디에 놓아야 하는지 참고용 스냅샷</div>
              <div class="viewport-wrapper">
                <canvas id="collect-guide-canvas" class="viewport-img" width="1280" height="720"></canvas>
              </div>
            </div>
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
            <div id="collect-js-btn-caption" style="font-size:16px; font-weight:700; font-family:var(--font-mono); text-align:center; margin-bottom:12px; padding:10px; border-radius:8px; background:#090d16; border:1px solid var(--border-glow); color:var(--text-muted); transition:background 0.15s, border-color 0.15s, color 0.15s;">대기 중 — 버튼/D-pad를 누르면 여기 크게 표시</div>
            <div style="font-size:10px; color:var(--text-muted); line-height:1.6; border-top:1px solid var(--border-glow); padding-top:8px;">
              <b>조작 설명서</b> (Gradio 대시보드와 동일 매핑)<br>
              왼쪽 스틱 → 이동(전/후/좌/우) &nbsp;|&nbsp; 오른쪽 스틱 X축 → 회전 (왼쪽 버튼패드에 라이트업)<br>
              <b>D-pad 좌/우</b> → 현재 활성 축(트랙A cx위치 / 접근경로 / 시나리오) 순환 선택 &nbsp;|&nbsp; <b>D-pad 상/하</b> → 3축 순환 전환 (녹화 중엔 변경 불가, L1/SEL로 시작 시 선택된 값으로 태깅됨)<br>
              <table style="width:100%; border-collapse:collapse; margin-top:6px;">
                <tr><td style="padding:1px 4px;"><b>A</b>(0)</td><td>STOP</td>
                    <td style="padding:1px 4px;"><b>B</b>(1)</td><td>마지막 프레임 취소</td></tr>
                <tr><td style="padding:1px 4px;"><b>X</b>(2)</td><td>에피소드 폐기</td>
                    <td style="padding:1px 4px;"><b>Y</b>(3)</td><td style="color:#555;">미사용</td></tr>
                <tr><td style="padding:1px 4px;"><b>L1</b>(4)</td><td>녹화 시작</td>
                    <td style="padding:1px 4px;"><b>R1</b>(5)</td><td>정지 & 저장</td></tr>
                <tr><td style="padding:1px 4px;"><b>SEL</b>(6)</td><td>녹화 토글</td>
                    <td style="padding:1px 4px;"><b>START</b>(7)</td><td>SYNC↔ASYNC 모드</td></tr>
                <tr><td style="padding:1px 4px; color:var(--amber);"><b>R2</b>(트리거)</td><td colspan="3" style="color:var(--amber);">🔄 복귀 — 직전 경로 역주행 (Gradio 이식, 다시 당기면 중지)</td></tr>
              </table>
              ⚠️ 대각선 후진(Z/C)은 조이스틱 축으로는 안 나옴 — 버튼패드/키보드로만 가능
            </div>
          </div>

          <div style="display:flex; flex-direction:column; gap:16px;">
            <div class="card" style="padding:16px;">
              <div class="card-title">📏 트랙A 극단배치 진행률 (cx축 막대그래프)
                <span id="collect-mode-badge" style="font-size:10px; padding:2px 8px; border-radius:10px; background:rgba(56,189,248,0.15); color:var(--cyan);">🎮 D-pad 대상: 트랙A</span>
              </div>
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
          </div>
        </div>

        <div style="display:grid; grid-template-columns:1fr; gap:20px; align-items:start;">

          <div style="display:flex; flex-direction:column; gap:16px;">
            <div class="card" style="padding:16px;">
              <div class="card-title">🕹️ 조작 (탭 클릭 후 키보드 W A S D Q E Z C R T Space)</div>
              <div id="collect-key-surface" tabindex="0"
                   style="outline:none; border:2px dashed var(--border-glow); border-radius:10px; padding:24px; text-align:center; background:#090d16; cursor:pointer;">
                <div style="font-size:13px; color:var(--text-muted); margin-bottom:8px;">여기 클릭해서 포커스 → 키보드로 조작 (조이스틱은 그대로 사용 가능, 자동 기록됨)</div>
                <div id="collect-last-action" style="font-size:20px; font-family:var(--font-mono); color:var(--emerald); font-weight:700;">STOP</div>
              </div>
              <div style="display:flex; gap:8px; margin-top:10px;">
                <span id="collect-active-badge" style="font-size:11px; padding:3px 10px; border-radius:20px; background:rgba(100,116,139,0.2); color:var(--text-muted);">⏸ 대기중</span>
                <span id="collect-steps-badge" style="font-size:11px; padding:3px 10px; border-radius:20px; background:rgba(100,116,139,0.2); color:var(--text-muted);">0 steps</span>
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
              <div class="card-title">📼 에피소드 제어</div>
              <input type="text" id="collect-episode-name" placeholder="episode_name (비우면 자동 생성)"
                     style="width:100%; padding:6px 8px; background:#090d16; border:1px solid var(--border-glow); border-radius:6px; color:#fff; font-size:12px; margin-bottom:10px;">
              <div style="display:flex; gap:8px;">
                <button class="btn btn-cyan" style="flex:1;" onclick="collectStartEpisode()">▶ 시작</button>
                <button class="btn btn-outline" style="flex:1; border-color:var(--rose); color:var(--rose);" onclick="collectStopEpisode()">⏹ 정지 & 저장</button>
              </div>
              <button id="collect-return-btn" class="btn btn-outline" style="width:100%; margin-top:8px; border-color:var(--amber); color:var(--amber);" onclick="collectAutoReturn()">🔄 복귀 (직전 경로 역주행)</button>
              <div id="collect-episode-status" style="font-size:11px; color:var(--text-muted); margin-top:8px;">—</div>
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
          <pre id="wiki-info-content" style="font-size:12px; font-family:var(--font-mono); background:#090d16; border:1px solid var(--border-glow); border-radius:8px; padding:14px; max-height:70vh; overflow:auto; white-space:pre-wrap; color:#e2e8f0;">로딩 중...</pre>
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
            docs/DASHBOARD_LIVE_STATUS.md — <span id="wiki-status-mtime">-</span>.
            실시간 자동 갱신 아님 — "대시보드 최신현황 갱신해줘" 요청 시 스킬이 이 파일을 다시 씀.
          </div>
          <pre id="wiki-status-content" style="font-size:12px; font-family:var(--font-mono); background:#090d16; border:1px solid var(--border-glow); border-radius:8px; padding:14px; max-height:70vh; overflow:auto; white-space:pre-wrap; color:#e2e8f0;">로딩 중...</pre>
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

      // 경로검증(verify) 탭일 때 프레임 인스펙터 재생 단축키 바인딩
      if (activeTab === "verify") {
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
      const lights = document.getElementById("collect-js-btn-lights");
      if (lights) {
        const pressed = new Set(s.buttons || []);
        const n = Math.max(10, ...(s.buttons || []).map(i => i + 1));
        if (lights.children.length !== n) {
          lights.innerHTML = "";
          for (let i = 0; i < n; i++) {
            const info = JOYSTICK_BTN_INFO[i] || {name: "#" + i, desc: "미사용"};
            const wrap = document.createElement("div");
            wrap.className = "btn-light-wrap";
            const d = document.createElement("div");
            d.className = "btn-light";
            d.id = "collect-js-light-" + i;
            d.textContent = i;
            const nameEl = document.createElement("div");
            nameEl.className = "btn-light-name";
            nameEl.id = "collect-js-name-" + i;
            nameEl.textContent = info.name;
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
          if (nameEl) nameEl.classList.toggle("active", isPressed);
          if (isPressed) activeInfo = JOYSTICK_BTN_INFO[i] || {name: "#" + i, desc: "미사용"};
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
        wikiinfo: "📖 위키",
        wikistatus: "📡 최신현황"
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
      if (tab === "collect") {
        collectRefreshState();
        collectStartKeyPolling();
        document.getElementById("collect-key-surface")?.focus();
      } else {
        collectStopKeyPolling();
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

    // 시작 프레임 캡처 → 극단 cx 구간을 밴드+선+수치로 정적 표시 (에피소드 시작 전 배치 참고용)
    async function _collectCaptureGuide() {
      const res = await api("/collect/snapshot");
      const cv = document.getElementById("collect-guide-canvas");
      if (!cv) return;
      const ctx = cv.getContext("2d");
      if (!res.ok) {
        ctx.clearRect(0, 0, cv.width, cv.height);
        ctx.fillStyle = "#94a3b8";
        ctx.font = "24px sans-serif";
        ctx.fillText("⚠️ " + (res.error || "캡처 실패"), 20, 40);
        return;
      }
      const img = new Image();
      img.onload = () => {
        ctx.clearRect(0, 0, cv.width, cv.height);
        ctx.drawImage(img, 0, 0, cv.width, cv.height);
        _collectDrawCxGuideBands(ctx, cv.width, cv.height);
      };
      img.src = "data:image/jpeg;base64," + res.image;
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

    async function collectRefreshState() {
      const res = await api("/collect/state");
      if (!res.ok) return;
      const activeBadge = document.getElementById("collect-active-badge");
      const stepsBadge = document.getElementById("collect-steps-badge");
      if (activeBadge) {
        activeBadge.textContent = res.active ? "🔴 수집중: " + (res.episode_name || "") : "⏸ 대기중";
        activeBadge.style.background = res.active ? "rgba(244,63,94,0.15)" : "rgba(100,116,139,0.2)";
        activeBadge.style.color = res.active ? "var(--rose)" : "var(--text-muted)";
      }
      if (stepsBadge) stepsBadge.textContent = res.steps + " steps";

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
      const modeBadge = document.getElementById("collect-mode-badge");
      if (modeBadge) {
        const MODE_LABELS = {trackA: "트랙A(위치)", cxpath: "접근경로", scenario: "시나리오"};
        const MODE_COLORS = {trackA: ["rgba(56,189,248,0.15)", "var(--cyan)"],
                              cxpath: ["rgba(245,158,11,0.15)", "var(--amber)"],
                              scenario: ["rgba(163,113,247,0.15)", "#a371f7"]};
        const m = res.collect_mode || "trackA";
        const [bg, fg] = MODE_COLORS[m] || MODE_COLORS.trackA;
        modeBadge.textContent = "🎮 D-pad 대상: " + (MODE_LABELS[m] || m);
        modeBadge.style.background = bg;
        modeBadge.style.color = fg;
      }

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
      const order = ["strong_left", "weak_left", "weak_right", "strong_right"];
      const pathOrder = ["left_curve", "straight", "right_curve"];
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
        const centerGap = (i === 2) ? `<div style="font-size:10px; color:var(--text-muted); text-align:center; padding:2px 0;">(중앙 미해당 구간)</div>` : "";

        const pathRows = pathOrder.filter(p => paths[p]).map(p => {
          const pinfo = paths[p];
          const pdone = pathStats[`${id}::${p}`] || 0;
          const ppct = Math.min(100, Math.round(pdone / pinfo.target * 100));
          const isPathSel = isSel && p === curPath;
          return `
            <div onclick="_collectClickCxPath('${id}','${p}')"
                 style="display:flex; align-items:center; gap:6px; padding-left:14px; cursor:pointer; ${isPathSel ? "outline:1px solid var(--amber); border-radius:4px;" : ""}">
              <span style="width:44px; font-size:10px; color:${isPathSel ? "var(--amber)" : "var(--text-muted)"};">${pinfo.label}</span>
              <div style="flex:1; height:10px; background:#090d16; border:1px solid var(--border-glow); border-radius:3px; overflow:hidden;">
                <div style="width:${ppct}%; height:100%; background:${ppct >= 100 ? "var(--emerald)" : "var(--amber)"};"></div>
              </div>
              <span style="width:34px; text-align:right; font-size:10px;">${pdone}/${pinfo.target}</span>
            </div>`;
        }).join("");

        return centerGap + `
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
      collectRefreshState();
    }

    async function collectStopEpisode() {
      const res = await api("/collect/episode/stop", { method: "POST", headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ save: true }) });
      const statusEl = document.getElementById("collect-episode-status");
      statusEl.textContent = res.ok
        ? (res.saved ? `✅ 저장됨: ${res.path} (${res.steps} steps, ${res.duration.toFixed(1)}s, STOP프레임 +5 포함)` : "⚠️ 저장 안 함: " + (res.reason || ""))
        : "⚠️ " + res.error;
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
          contentEl.textContent = res.content;
          if (name === "status") {
            const mtimeEl = document.getElementById("wiki-status-mtime");
            if (mtimeEl) mtimeEl.textContent = "최근 갱신: " + res.mtime;
          }
        } else {
          contentEl.textContent = "⚠️ 로드 실패: " + res.error;
        }
      } catch (e) {
        contentEl.textContent = "⚠️ 서버 오류: " + e;
      }
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
      "dist_10cm", "dist_20cm", "dist_30cm"
    ];

    const PATH_TARGETS = {
      "right_right": 10, "right_left": 10, "right_straight": 10,
      "center_straight": 10, "center_left": 10, "center_right": 10,
      "left_straight": 10, "left_left": 10, "left_right": 10,
      "obj_left": 30, "obj_center": 30, "obj_right": 30,
      "dist_10cm": 10, "dist_20cm": 10, "dist_30cm": 10,
    };

    const PATH_GROUPS = [
      ["── 오브젝트 위치별 ──────────", ["obj_left","obj_center","obj_right"]],
      ["── 경로 검증 ──────────────", ["right_right","right_left","right_straight","center_straight","center_left","center_right","left_straight","left_left","left_right"]],
      ["── 박스 거리별 ──────────────", ["dist_10cm","dist_20cm","dist_30cm"]]
    ];

    let runtimeState = {
      preview_enabled: true,
      preview_hint_cx: false,
      grounding_skip_n: 3,
      cx_jump_filter: false,
      cx_jump_thresh: 0.30,
      multi_prompt: true
    };

    function selectPathType(type) {
      const select = document.getElementById("ep-path-type");
      if (select) {
        select.value = type;
      }
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
            multi_prompt: runtimeState.multi_prompt
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
          if (!pt.startsWith("obj_") && !pt.startsWith("dist_")) {
            nav_succ += 1;
          }
        }
        if (PATH_TARGETS[pt] !== undefined && !pt.startsWith("obj_") && !pt.startsWith("dist_")) {
          nav_done += 1;
        }
      });
      
      const nav_total = 90; // 9 routes * 10 targets = 90
      const obj_done = (done_total["obj_left"] || 0) + (done_total["obj_center"] || 0) + (done_total["obj_right"] || 0);
      const obj_succ = (done_succ["obj_left"] || 0) + (done_succ["obj_center"] || 0) + (done_succ["obj_right"] || 0);
      const dist_done = (done_total["dist_10cm"] || 0) + (done_total["dist_20cm"] || 0) + (done_total["dist_30cm"] || 0);
      const dist_succ = (done_succ["dist_10cm"] || 0) + (done_succ["dist_20cm"] || 0) + (done_succ["dist_30cm"] || 0);
      
      const total_done = rows.length;
      const total_target = 210; // 90 nav + 90 obj + 30 dist = 210
      
      const pct_total = Math.min(100.0, Math.max(0.0, (total_done / total_target) * 100));
      const pct_nav   = Math.min(100.0, Math.max(0.0, (nav_done / nav_total) * 100));
      const pct_obj   = Math.min(100.0, Math.max(0.0, (obj_done / 90) * 100));
      const pct_dist  = Math.min(100.0, Math.max(0.0, (dist_done / 30) * 100));
      
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
        
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px;">
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
        </div>
      </div>
      `;
      document.getElementById("vfy-progress-wrapper").innerHTML = progressHtml;
      
      document.getElementById("vfy-progress-txt").innerHTML = `
        경로검증 ${nav_done}/${nav_total} ep 성공 ${nav_succ}/20 (목표)<br>
        위치별 ${obj_done}/90 (${obj_succ} 성공) | 거리별 ${dist_done}/30 (${dist_succ} 성공)
      `;
      
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
    async function loadSessionList() {
      const res = await api("/sessions/list");
      const listEl = document.getElementById("session-list-group");
      if (res.sessions.length === 0) {
        listEl.innerHTML = "<div style='text-align:center;color:var(--text-muted);font-size:13px;padding:20px 0;'>H5 세션 파일 없음</div>";
        return;
      }
      
      listEl.innerHTML = res.sessions.map(s => `
        <div class="srv-pill" style="cursor:pointer; display:flex; flex-direction:column; align-items:flex-start; padding:12px; gap:4px; hover:border-color:var(--cyan);" onclick="loadSessionDetail('${s.sid}')">
          <div style="font-weight:700; font-size:13px; color:var(--cyan);">${s.sid}</div>
          <div style="font-size:11px; color:var(--text-muted); word-break:break-all;">Entity: ${s.instruction}</div>
          <div style="display:flex; justify-content:space-between; width:100%; font-size:11px; margin-top:2px;">
            <span>Steps: ${s.steps}</span>
            <span class="text-emerald">[${s.labeled_count} Labeled]</span>
          </div>
        </div>
      `).join("");
    }

    async function loadSessionDetail(sid) {
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
      
      showInspectFrame(0);
    }

    function showInspectFrame(idx) {
      if (!inspectSession) return;
      idx = parseInt(idx);
      const frame = inspectSession.frames[idx];
      
      document.getElementById("inspect-frame-idx-lbl").textContent = `Frame: ${idx + 1} / ${inspectSession.frames.length}`;
      document.getElementById("inspect-frame-type-lbl").textContent = frame.type;
      document.getElementById("inspect-action-lbl").textContent = frame.action;
      document.getElementById("inspect-lat-lbl").textContent = frame.latency_ms.toFixed(0) + "ms";
      
      // 이미지 소스 설정
      document.getElementById("inspect-frame-img").src = `/sessions/frame?sid=${inspectSession.sid}&idx=${idx}`;
      
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
        const showGridVfy = document.getElementById("toggle-grid-vfy").checked;
        if (showGridVfy) {
          drawGridLines(ctxVfy, cvVfy.width, cvVfy.height);
        }
        if (state.bbox && state.bbox.area > 0) {
          drawBbox(ctxVfy, state.bbox, cvVfy.width, cvVfy.height);
        }
      }
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
            tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; color:var(--text-muted);">기록이 없습니다.</td></tr>`;
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
            return `
              <tr onclick="_epEditLoadByRow(${JSON.stringify(String(ep[0]))})" style="cursor:pointer; ${isEditing ? "background:rgba(56,189,248,0.12);" : ""}">
                <td>${ep[0]}</td>
                <td><strong class="text-cyan">${pathAbbr}</strong></td>
                <td><strong class="${resColor}">${ep[2]}</strong></td>
                <td>${ep[3]}</td>
                <td>${ep[4]} ms</td>
                <td class="font-mono text-cyan" style="font-size:10px;">${ep[10] || "—"}</td>
                <td style="font-size:11px; color:var(--text-muted);">${ep[11] || "—"}</td>
                <td style="font-family:var(--font-sans); color:var(--text-muted); font-size:10px; white-space:nowrap;">${ep[12] || "—"}</td>
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
      if (res.ok) {
        document.getElementById("ep-note").value = "";
        alert("에피소드 기록이 정상적으로 추가되었습니다.");
        loadEpisodeHistory();
      } else {
        alert("기록 실패: " + res.error);
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
            <div class="form-group"><label>Stop Mode</label><input type="text" readonly value="${inf.stop_mode} (${inf.stop_latched ? 'Latched' : 'Unlatched'})"></div>
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

    // 초기 스타트
    setInterval(pollStatus, 500);
    setInterval(pollHealth, 3000);
    setInterval(camProcRefresh, 10000);
    setInterval(joystickRefresh, 500);
    pollStatus();
    pollHealth();
    loadEpisodeHistory();
    camProcRefresh();
    joystickRefresh();

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

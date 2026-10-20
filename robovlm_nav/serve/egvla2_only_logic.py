"""EGVLA2 단독 서버(단계 B)의 구동 상태기계 — 순수 로직(모델·torch import 없음).

plan: docs/plans/plan_20261019_egvla2_inference_server.md §1 단계 B, §2 결정 1~3(승인 기본값).
입력: 프레임마다 (action_cont, has_bbox, bbox_area[, bbox_cx]). action_cont는 continuous_speed.to_drive_command 출력 형태
      {stop, dir_lx, dir_ly, speed_scale, ...}.
상태: DRIVE(모델 방향으로 구동) → STOPPED(래치, reset 전까지 정지) / SEARCH(타깃 소실 시 제자리 회전 탐색).
규칙(전부 생성자 인자로 조정, 기본값은 승인안):
  (a) 정지: action_cont.stop 이 stop_debounce(3)프레임 연속이면 STOPPED 래치. 도달 전에는 이동 0(stop_pending).
  (b) 근접 오버라이드: has_bbox 이고 bbox_area >= near_area(0.25, None이면 끔)가 near_frames(2)프레임 연속이면 STOPPED.
  (c) 타깃 소실: has_bbox=False 가 lost_frames(5)프레임 연속이면 SEARCH(제자리 회전 az=±search_az, 마지막 본 cx<0.5면 좌(+), 아니면 우(-), 없으면 우).
      has_bbox 가 다시 True 이면 DRIVE 복귀. SEARCH 가 search_max_frames(40)프레임을 넘으면 STOPPED.
  (d) DRIVE: lx,ly = 단위 방향 × gain(1.15), az=0(첫 적용), speed_scale = action_cont.speed_scale.
az 부호: +az = 좌회전(기존 ACTION_3D ROT_L = +0.25).
"""
from __future__ import annotations

from typing import Any

DRIVE, SEARCH, STOPPED = "DRIVE", "SEARCH", "STOPPED"


class EgVla2Controller:
    def __init__(self, *, stop_debounce: int = 3, near_area: float | None = 0.25, near_frames: int = 2,
                 lost_frames: int = 5, search_az: float = 0.25, search_max_frames: int = 40,
                 search_speed_scale: float = 0.7, gain: float = 1.15):
        self.stop_debounce, self.near_area, self.near_frames = int(stop_debounce), near_area, int(near_frames)
        self.lost_frames, self.search_az, self.search_max_frames = int(lost_frames), float(search_az), int(search_max_frames)
        self.search_speed_scale, self.gain = float(search_speed_scale), float(gain)
        self.reset()

    def reset(self) -> None:
        self.state = DRIVE
        self.stop_run = self.near_run = self.lost_run = self.search_frames = 0
        self.last_cx: float | None = None
        self.search_sign = -1.0
        self.stop_reason = ""

    # ── 출력 조립 ──
    def _out(self, lx: float, ly: float, az: float, speed: float, reason: str) -> dict[str, Any]:
        return dict(lx=float(lx), ly=float(ly), az=float(az), speed_scale=float(speed), state=self.state, reason=reason,
                    stopped=self.state == STOPPED, stop_reason=self.stop_reason)

    def _stop(self, reason: str) -> dict[str, Any]:
        self.state, self.stop_reason = STOPPED, reason
        return self._out(0.0, 0.0, 0.0, 0.0, reason)

    def step(self, action_cont: dict[str, Any] | None, has_bbox: bool, bbox_area: float | None = None,
             bbox_cx: float | None = None) -> dict[str, Any]:
        if self.state == STOPPED:
            return self._out(0.0, 0.0, 0.0, 0.0, "latched:" + self.stop_reason)

        has_bbox = bool(has_bbox)
        if has_bbox and bbox_cx is not None:
            self.last_cx = float(bbox_cx)
        self.lost_run = 0 if has_bbox else self.lost_run + 1
        self.near_run = self.near_run + 1 if (has_bbox and self.near_area is not None and bbox_area is not None
                                              and float(bbox_area) >= self.near_area) else 0

        # (b) 근접 오버라이드 — 어떤 상태에서든 우선
        if self.near_area is not None and self.near_run >= self.near_frames:
            return self._stop("near_goal")

        # (c) 탐색 상태
        if self.state == SEARCH:
            if has_bbox:
                self.state, self.search_frames = DRIVE, 0
                reacquired = True
            else:
                self.search_frames += 1
                if self.search_frames > self.search_max_frames:
                    return self._stop("search_timeout")
                return self._out(0.0, 0.0, self.search_sign * self.search_az, self.search_speed_scale,
                                 f"search({self.search_frames}/{self.search_max_frames})")
        else:
            reacquired = False
            if self.lost_run >= self.lost_frames:
                self.state, self.search_frames = SEARCH, 1
                self.search_sign = 1.0 if (self.last_cx is not None and self.last_cx < 0.5) else -1.0
                return self._out(0.0, 0.0, self.search_sign * self.search_az, self.search_speed_scale,
                                 f"search_start(lost {self.lost_run})")

        # 정책 출력 확인
        if not isinstance(action_cont, dict) or action_cont.get("error"):
            return self._out(0.0, 0.0, 0.0, 0.0, "policy_error")

        # (a) 정지 디바운스·래치
        if action_cont.get("stop", True):
            self.stop_run += 1
            if self.stop_run >= self.stop_debounce:
                return self._stop("stop_debounce")
            return self._out(0.0, 0.0, 0.0, 0.0, f"stop_pending({self.stop_run}/{self.stop_debounce})")
        self.stop_run = 0

        # (d) 구동
        reason = "target_reacquired" if reacquired else ("bbox_missing(%d)" % self.lost_run if self.lost_run else "drive")
        return self._out(float(action_cont["dir_lx"]) * self.gain, float(action_cont["dir_ly"]) * self.gain, 0.0,
                         float(action_cont["speed_scale"]), reason)

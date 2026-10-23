"""EGVLA2 단독 서버(단계 B)의 구동 상태기계 — 순수 로직(모델·torch import 없음).

plan: docs/plans/plan_20261019_egvla2_inference_server.md §1 단계 B, §2 결정 1~3(승인 기본값).
입력: 프레임마다 (action_cont, has_bbox, bbox_area[, bbox_cx]). action_cont는 continuous_speed.to_drive_command 출력 형태
      {stop, dir_lx, dir_ly, speed_scale, ...}.
상태: DRIVE(모델 방향으로 구동) → STOPPED(래치, reset 전까지 정지) / SEARCH(타깃 소실 시 제자리 회전 탐색).
규칙(전부 생성자 인자로 조정, 기본값은 승인안):
  (a0) 회전: 정책 출력 크기가 작아도(stop) |az|>=rotate_az_thresh(0.2)이면 정지가 아니라 제자리 회전. **회전 방향은 타깃 위치(bbox cx)로 정한다**: 타깃이 오른쪽(cx>0.5)이면 +az(우회전), 왼쪽이면 -az(좌회전). 학습 데이터의 기록 az 부호와 로봇 실행 부호는 같다(+ = 우회전; 데이터에서 az>0이면 회전 후 cx가 줄어드는 비율 99%, 실기에서 az<0이면 cx가 늘어남). cx가 중앙(|cx-0.5|<rotate_cx_dead)이면 회전하지 않고, cx를 모르면 모델 az 부호를 그대로 쓴다. 학습 데이터의 53.7%가
      제자리 회전으로 시작하므로(plans/finding_20261022_start_rotation_mismatch.md) 이를 버리면 출발하지 못한다. 연속 rotate_max_frames(12) 초과 시 회전 중단.
  (a) 정지: 크기 기반 정지 래치는 기본 끔(stop_debounce=0). 켜면(>0) action_cont.stop 이 그 프레임 수만큼 연속일 때 STOPPED 래치.
      끈 상태에서 정책이 계속 0을 내면 idle(이동 0)로 두고, idle_max_frames(20) 연속이면 안전망으로 STOPPED(idle_timeout). 에피소드 종료는 (b) 근접·(c) 소실 규칙이 담당.
  (b) 근접 오버라이드: has_bbox 이고 bbox_area >= near_area(0.25, None이면 끔)가 near_frames(2)프레임 연속이면 STOPPED.
  (c) 타깃 소실: has_bbox=False 가 lost_frames(5)프레임 연속이면 SEARCH(제자리 회전 az=±search_az, 마지막 본 cx<0.5면 좌(-az), 아니면 우(+az), 없으면 우).
      has_bbox 가 다시 True 이면 DRIVE 복귀. SEARCH 가 search_max_frames(40)프레임을 넘으면 STOPPED.
  (d) DRIVE: lx,ly = 단위 방향 × gain(1.15), az=0(첫 적용), speed_scale = action_cont.speed_scale.
az 부호(실기 확인 2026-10-22): **이 로봇에서 +az = 우회전, -az = 좌회전**(학습 데이터의 기록 규약과 같음. 타깃 cx=0.758일 때 az=-0.25를 보냈더니 타깃이 오른쪽으로 밀려남을 관측).
"""
from __future__ import annotations

from typing import Any

DRIVE, SEARCH, STOPPED = "DRIVE", "SEARCH", "STOPPED"


class EgVla2Controller:
    def __init__(self, *, stop_debounce: int = 0, near_area: float | None = 0.25, near_frames: int = 2,
                 lost_frames: int = 5, search_az: float = 0.25, search_max_frames: int = 40,
                 search_speed_scale: float = 0.7, gain: float = 1.15, rotate_az_thresh: float = 0.2, rotate_az: float = 0.25,
                 rotate_max_frames: int = 12, rotate_speed_scale: float = 0.7, idle_max_frames: int = 20, rotate_cx_dead: float = 0.08,
                 servo_blend: float = 0.0, servo_k: float = 2.0, servo_max: float = 0.7):
        self.stop_debounce, self.near_area, self.near_frames = int(stop_debounce), near_area, int(near_frames)
        self.lost_frames, self.search_az, self.search_max_frames = int(lost_frames), float(search_az), int(search_max_frames)
        self.search_speed_scale, self.gain = float(search_speed_scale), float(gain)
        self.rotate_az_thresh, self.rotate_az, self.rotate_max_frames = float(rotate_az_thresh), float(rotate_az), int(rotate_max_frames)
        self.rotate_speed_scale, self.idle_max_frames = float(rotate_speed_scale), int(idle_max_frames)
        self.rotate_cx_dead = float(rotate_cx_dead)
        self.servo_blend, self.servo_k, self.servo_max = float(servo_blend), float(servo_k), float(servo_max)
        self.reset()

    def reset(self) -> None:
        self.state = DRIVE
        self.stop_run = self.near_run = self.lost_run = self.search_frames = 0
        self.rotate_run = self.idle_run = 0
        self.last_cx: float | None = None
        self.search_sign = 1.0
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
                self.search_sign = -1.0 if (self.last_cx is not None and self.last_cx < 0.5) else 1.0
                return self._out(0.0, 0.0, self.search_sign * self.search_az, self.search_speed_scale,
                                 f"search_start(lost {self.lost_run})")

        # 정책 출력 확인
        if not isinstance(action_cont, dict) or action_cont.get("error"):
            return self._out(0.0, 0.0, 0.0, 0.0, "policy_error")

        small = bool(action_cont.get("stop", True))
        # (a0) 제자리 회전: 크기는 작고 az가 크면 정지가 아니라 회전
        az_raw = float(action_cont.get("az", 0.0) or 0.0)
        if small and abs(az_raw) >= self.rotate_az_thresh and self.rotate_run < self.rotate_max_frames:
            cx = bbox_cx if bbox_cx is not None else self.last_cx
            if cx is None:
                direction = 1.0 if az_raw > 0 else -1.0           # 기록 규약 = 로봇 규약(+ = 우회전)
            elif abs(float(cx) - 0.5) >= self.rotate_cx_dead:
                direction = -1.0 if float(cx) < 0.5 else 1.0      # 타깃이 왼쪽이면 좌회전(-az), 오른쪽이면 우회전(+az)
            else:
                direction = 0.0                                    # 이미 중앙: 회전 불필요
            if direction != 0.0:
                self.rotate_run += 1
                self.stop_run = self.idle_run = 0
                return self._out(0.0, 0.0, direction * self.rotate_az, self.rotate_speed_scale,
                                 f"rotate({self.rotate_run}/{self.rotate_max_frames})")
        # (a) 정지: 크기 기반 래치는 stop_debounce>0일 때만. 끈 상태는 idle(이동 0) + 안전망 타임아웃
        if small:
            if self.stop_debounce > 0:
                self.stop_run += 1
                if self.stop_run >= self.stop_debounce:
                    return self._stop("stop_debounce")
                return self._out(0.0, 0.0, 0.0, 0.0, f"stop_pending({self.stop_run}/{self.stop_debounce})")
            self.idle_run += 1
            if self.idle_run >= self.idle_max_frames:
                return self._stop("idle_timeout")
            return self._out(0.0, 0.0, 0.0, 0.0, f"idle({self.idle_run}/{self.idle_max_frames})")
        self.stop_run = self.idle_run = self.rotate_run = 0

        # (d) 구동
        reason = "target_reacquired" if reacquired else ("bbox_missing(%d)" % self.lost_run if self.lost_run else "drive")
        ux, uy = float(action_cont["dir_lx"]), float(action_cont["dir_ly"])
        cx = bbox_cx if bbox_cx is not None else self.last_cx
        if self.servo_blend > 0.0 and has_bbox and cx is not None:
            # 타깃 서보(모델 밖 규칙, 기본 끔): 하드웨어는 방향각·부호·throttle만 쓰므로(크기 무시) 방향 벡터만 바꾼다.
            # 데이터·실기 공통 관측: ly<0이면 타깃 cx가 늘어남 → cx>0.5이면 ly>0(cx 감소) 쪽으로 보정.
            sy = max(-self.servo_max, min(self.servo_max, self.servo_k * (float(cx) - 0.5)))
            sx = (1.0 - sy * sy) ** 0.5
            w = self.servo_blend
            bx, by = (1.0 - w) * ux + w * sx, (1.0 - w) * uy + w * sy
            n = (bx * bx + by * by) ** 0.5
            if n > 1e-6:
                ux, uy = bx / n, by / n
                reason = reason + f"+servo(w={w:g})"
        return self._out(ux * self.gain, uy * self.gain, 0.0, float(action_cont["speed_scale"]), reason)

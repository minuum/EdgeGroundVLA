"""Phase 4 이종 로봇 이식 어댑터: Serbot(전방향 [lx, ly, az]) → YahBoom 카(차동/아커만 [linear_x, angular_z]).

핵심 설계:
  - VLA 모델(MOD-A3-AG / VLM-E2E-SoftLoc)의 재학습 없이, 구동계 변환과 1프레임 카메라 면적 보정만 교체한다.
  - Serbot 좌표 규약(실기 확인): lx=전진(+), ly=좌(+)/우(-), az=우회전(+)/좌회전(-).
  - 표준 ROS2 Twist 규약(YahBoom): linear.x=전진(+), angular.z=좌회전(+)/우회전(-) → 부호 반전 주의!
"""
from __future__ import annotations

import math
from typing import Any


def serbot_to_yahboom_twist(
    lx: float,
    ly: float,
    az: float,
    speed_scale: float = 1.0,
    *,
    max_linear_v: float = 0.25,
    max_angular_w: float = 0.80,
    steer_gain: float = 1.2,
    invert_ros_az: bool = True,
) -> dict[str, Any]:
    """Serbot 구동 명령(lx, ly, az, speed_scale)을 YahBoom ROS2 Twist(linear_x, angular_z)로 변환."""
    sp = max(0.0, min(1.0, float(speed_scale)))
    # 제자리 탐색/회전 모드 (lx, ly가 거의 0이고 az가 존재할 때)
    mag = math.hypot(lx, ly)
    if mag < 1e-4:
        if abs(az) < 1e-4 or sp <= 1e-4:
            return dict(linear_x=0.0, angular_z=0.0, mode="stop")
        # Serbot +az=우회전 → 표준 ROS2 Twist에서는 좌회전이 +angular_z이므로 기본 반전
        sign = -1.0 if invert_ros_az else 1.0
        w = max(-max_angular_w, min(max_angular_w, sign * float(az) * max_angular_w * sp))
        return dict(linear_x=0.0, angular_z=round(w, 4), mode="rotate")

    # 전진 + 조향 결합: 방향각 theta = atan2(ly, lx) (+theta = 좌측 목표 → ROS2 +angular_z)
    theta = math.atan2(float(ly), max(1e-4, float(lx)))
    # 회전각이 클수록 전진 속도를 자연스럽게 감속 (cos(theta))
    v = max(0.0, math.cos(theta)) * max_linear_v * sp
    w = max(-max_angular_w, min(max_angular_w, steer_gain * theta * sp))
    return dict(linear_x=round(v, 4), angular_z=round(w, 4), mode="drive", heading_deg=round(math.degrees(theta), 1))


def calibrate_near_area(
    serbot_ref_area: float,
    yahboom_ref_area: float,
    *,
    serbot_near_area: float = 0.25,
    serbot_decel_start: float = 0.14,
) -> dict[str, float]:
    """동일 목표물·동일 정지 거리에서 촬영한 1프레임 BBox 면적 비율로 YahBoom 정지/감속 임계치를 자동 보정."""
    if serbot_ref_area <= 1e-5 or yahboom_ref_area <= 1e-5:
        raise ValueError("Reference bbox areas must be positive")
    ratio = float(yahboom_ref_area) / float(serbot_ref_area)
    return dict(
        area_scale_ratio=round(ratio, 3),
        yahboom_near_area=round(min(0.65, max(0.05, serbot_near_area * ratio)), 4),
        yahboom_decel_start_area=round(min(0.50, max(0.03, serbot_decel_start * ratio)), 4),
    )

"""연속 액션 → 구동 명령(방향 단위벡터 + throttle_scale) 변환 규칙.

plan: docs/plans/plan_20261014_inference_continuous_speed.md §1~2.1
- 모델 출력 (lx, ly, az)의 크기 m=hypot(lx,ly)를 속력 스케일 s로 보간한다(단계 3개로 양자화하지 않음).
- speed_source: model(모델 크기 사용) | rule(고정 규칙 속력) | shadow(규칙으로 구동, 모델 속력은 기록만, 기본).
- 첫 적용은 az=0(이동만), s<=S_MAX(기본 0.7).
"""
from __future__ import annotations

import math
import os
from typing import Any

STOP_M = 0.2            # m < STOP_M → 정지
S_MIN = 0.4             # 보정된 최저 속력(0.4 미만은 검증 없음)
S_MAX_CAL = 1.0
RULE_S = 0.7            # rule/shadow 구동 속력(기본값, VLA_RULE_SPEED로 재정의 가능)
BASE_THROTTLE = 50
V_INTERCEPT, V_SLOPE = 6.59, 19.85   # 속도 보정 선형식 v=a+b·s (cm/s), logs/speed_calibration.json 30건
SPEED_SOURCES = ("model", "rule", "shadow")


def expected_cmps(s: float) -> float:
    return V_INTERCEPT + V_SLOPE * s


def to_drive_command(lx: float, ly: float, az: float = 0.0, *, speed_source: str = "shadow",
                     s_max: float | None = None, use_az: bool = False) -> dict[str, Any]:
    if speed_source not in SPEED_SOURCES:
        raise ValueError(f"speed_source must be one of {SPEED_SOURCES}")
    s_cap = float(os.environ.get("VLA_SPEED_MAX", 0.7)) if s_max is None else s_max
    s_cap = max(S_MIN, min(S_MAX_CAL, s_cap))
    m = math.hypot(lx, ly)
    out: dict[str, Any] = dict(model_magnitude=m, speed_source=speed_source, az=float(az) if use_az else 0.0)
    if m < STOP_M:
        out.update(stop=True, dir_lx=0.0, dir_ly=0.0, speed_scale=0.0, throttle=0, expected_cmps=0.0,
                   model_speed_scale=0.0)
        return out
    # 2026-10-23 minum 분석(docs/plans/analysis_20261023_controller_physical_mapping.md §3):
    # 수집 데이터는 throttle=50(speed_scale=1.0 상당) 고정이었는데 추론 기본값(RULE_S=0.7)은
    # 그보다 30% 낮아 학습 물리와 어긋남 — 재기동 없이 A/B 비교할 수 있도록 env로 노출.
    rule_s = float(os.environ.get("VLA_RULE_SPEED", RULE_S))
    model_s = max(S_MIN, min(s_cap, m))
    s = model_s if speed_source == "model" else min(rule_s, s_cap)
    out.update(stop=False, dir_lx=lx / m, dir_ly=ly / m, speed_scale=s, model_speed_scale=model_s,
               throttle=max(1, round(BASE_THROTTLE * s)), expected_cmps=expected_cmps(s))
    return out


MOVE_CLASSES = (1, 2, 3, 4, 5)   # FORWARD LEFT RIGHT FWD+L FWD+R. STOP(0)·ROT(6,7)은 기존 판단 유지


def hybrid_drive(pred_class: int, blocked: bool, action_cont: Any, gain: float = 1.15) -> dict[str, Any] | None:
    """VLA_DRIVE_SOURCE=egvla2_hybrid: 기존 파이프라인이 '이동'이라고 판단한 프레임에서만 구동 방향·속력을 연속값으로 대체한다.
    STOP·ROT·막힘은 기존 판단 유지(None), EGVLA2가 정지(|m|<STOP_M)를 내도 보수적으로 기존 클래스 구동(로그만).
    lx/ly = 단위 방향 × gain(기존 FORWARD의 1.15와 같은 크기: 속도 보정은 직진 기준), az=0(첫 적용), throttle_scale = speed_scale."""
    if blocked or pred_class not in MOVE_CLASSES:
        return None
    if not isinstance(action_cont, dict) or action_cont.get("error") or action_cont.get("stop", True):
        return None
    return dict(lx=float(action_cont["dir_lx"]) * gain, ly=float(action_cont["dir_ly"]) * gain, az=0.0,
                speed_scale=float(action_cont["speed_scale"]), source="egvla2_hybrid")

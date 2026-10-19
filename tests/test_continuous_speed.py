import math
import pytest
from robovlm_nav.serve.continuous_speed import to_drive_command, expected_cmps


def test_stop_below_threshold():
    r = to_drive_command(0.1, 0.1)
    assert r["stop"] and r["throttle"] == 0


def test_model_interpolates_and_clamps():
    r = to_drive_command(0.5, 0.0, speed_source="model", s_max=1.0)
    assert r["speed_scale"] == 0.5 and r["throttle"] == 25
    assert to_drive_command(0.25, 0.0, speed_source="model")["speed_scale"] == 0.4   # 최저 0.4로 올림
    assert to_drive_command(2.0, 0.0, speed_source="model", s_max=1.0)["speed_scale"] == 1.0
    assert to_drive_command(1.0, 0.0, speed_source="model")["speed_scale"] == 0.7    # 기본 s_max 0.7


def test_shadow_uses_rule_logs_model():
    r = to_drive_command(1.0, 0.0, speed_source="shadow", s_max=1.0)
    assert r["speed_scale"] == 0.7 and r["model_speed_scale"] == 1.0


def test_direction_unit_vector_and_az_zero():
    r = to_drive_command(0.6, 0.8, 0.9)
    assert math.isclose(math.hypot(r["dir_lx"], r["dir_ly"]), 1.0) and r["az"] == 0.0


def test_expected_speed_matches_calibration():
    assert abs(expected_cmps(0.4) - 14.53) < 0.1 and abs(expected_cmps(1.0) - 26.44) < 0.1


def test_bad_source():
    with pytest.raises(ValueError):
        to_drive_command(1, 0, speed_source="x")


def test_hybrid_drive():
    from robovlm_nav.serve.continuous_speed import hybrid_drive, to_drive_command
    ac = to_drive_command(0.3, 0.4, speed_source="shadow")          # m=0.5 → 이동, 방향 (0.6, 0.8), s=0.7
    d = hybrid_drive(1, False, ac)
    assert d and abs(d["lx"] - 0.6 * 1.15) < 1e-9 and abs(d["ly"] - 0.8 * 1.15) < 1e-9 and d["az"] == 0.0 and abs(d["speed_scale"] - 0.7) < 1e-9
    assert hybrid_drive(0, False, ac) is None and hybrid_drive(6, False, ac) is None and hybrid_drive(7, False, ac) is None   # STOP·ROT는 기존 판단
    assert hybrid_drive(1, True, ac) is None                                                                                # 막힘
    assert hybrid_drive(1, False, to_drive_command(0.05, 0.05)) is None                                                     # EGVLA2 정지 → 기존 클래스 구동
    assert hybrid_drive(1, False, {"error": "x"}) is None and hybrid_drive(1, False, None) is None

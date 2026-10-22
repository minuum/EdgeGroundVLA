"""EGVLA2 단독 서버 구동 상태기계(EgVla2Controller) — 합성 입력 단위 테스트. 모델·GPU 불필요."""
import math

from robovlm_nav.serve.continuous_speed import to_drive_command
from robovlm_nav.serve.egvla2_only_logic import DRIVE, SEARCH, STOPPED, EgVla2Controller

GO = to_drive_command(0.3, 0.4, speed_source="shadow")      # m=0.5 → 이동, 방향 (0.6, 0.8), s=0.7
STOP = to_drive_command(0.05, 0.05)                         # m<0.2 → stop


def test_drive_unit_direction_gain_speed():
    c = EgVla2Controller()
    o = c.step(GO, True, 0.05, 0.5)
    assert o["state"] == DRIVE and not o["stopped"] and o["reason"] == "drive"
    assert math.isclose(o["lx"], 0.6 * 1.15) and math.isclose(o["ly"], 0.8 * 1.15) and o["az"] == 0.0
    assert math.isclose(o["speed_scale"], 0.7)


def test_stop_debounce_boundary_and_latch():
    c = EgVla2Controller(stop_debounce=3)
    for i in (1, 2):                                          # 디바운스 미달: 이동 0, 래치 아님
        o = c.step(STOP, True, 0.05, 0.5)
        assert o["state"] == DRIVE and o["lx"] == o["ly"] == 0.0 and o["reason"] == f"stop_pending({i}/3)"
    o = c.step(STOP, True, 0.05, 0.5)                         # 3프레임째 → STOPPED
    assert o["state"] == STOPPED and o["stopped"] and o["reason"] == "stop_debounce"
    o = c.step(GO, True, 0.05, 0.5)                           # 래치: 이후 이동 신호가 와도 정지
    assert o["state"] == STOPPED and o["lx"] == o["ly"] == o["az"] == 0.0 and o["reason"].startswith("latched")


def test_stop_debounce_resets_on_move():
    c = EgVla2Controller(stop_debounce=3)
    c.step(STOP, True); c.step(STOP, True)
    assert c.step(GO, True)["state"] == DRIVE                 # 연속이 끊기면 카운터 초기화
    assert c.step(STOP, True)["reason"] == "stop_pending(1/3)"


def test_near_goal_override():
    c = EgVla2Controller(near_area=0.25, near_frames=2)
    assert c.step(GO, True, 0.30, 0.5)["state"] == DRIVE      # 1프레임: 아직
    o = c.step(GO, True, 0.30, 0.5)                           # 2프레임 연속 → STOPPED
    assert o["state"] == STOPPED and o["reason"] == "near_goal"
    c.reset()
    c.step(GO, True, 0.30, 0.5); c.step(GO, True, 0.10, 0.5)  # 끊기면 초기화
    assert c.step(GO, True, 0.30, 0.5)["state"] == DRIVE


def test_near_goal_off_when_none():
    c = EgVla2Controller(near_area=None)
    for _ in range(5):
        assert c.step(GO, True, 0.9, 0.5)["state"] == DRIVE


def test_lost_search_direction_left_when_last_cx_left():
    c = EgVla2Controller(lost_frames=5)
    c.step(GO, True, 0.05, 0.3)                               # 마지막으로 본 cx=0.3 (<0.5 → 좌)
    for i in range(1, 5):
        o = c.step(GO, False)
        assert o["state"] == DRIVE and o["reason"] == f"bbox_missing({i})"   # 5프레임 미만은 계속 구동
    o = c.step(GO, False)                                     # 5프레임째 → SEARCH
    assert o["state"] == SEARCH and o["az"] == -0.25 and o["lx"] == o["ly"] == 0.0 and o["reason"].startswith("search_start")


def test_lost_search_direction_right_when_last_cx_right_or_unknown():
    c = EgVla2Controller(lost_frames=2)
    c.step(GO, True, 0.05, 0.8)
    c.step(GO, False)
    assert c.step(GO, False)["az"] == 0.25                    # cx>=0.5 → 우(+az)
    c2 = EgVla2Controller(lost_frames=2)
    c2.step(GO, False)
    assert c2.step(GO, False)["az"] == 0.25                   # 본 적 없음 → 우(+az)


def test_search_reacquire_returns_to_drive():
    c = EgVla2Controller(lost_frames=2)
    c.step(GO, False); c.step(GO, False)
    assert c.state == SEARCH
    c.step(GO, False)
    o = c.step(GO, True, 0.05, 0.5)
    assert o["state"] == DRIVE and o["reason"] == "target_reacquired" and o["az"] == 0.0 and o["lx"] > 0
    assert c.step(GO, True, 0.05, 0.5)["reason"] == "drive"


def test_search_timeout_stops():
    c = EgVla2Controller(lost_frames=2, search_max_frames=3)
    c.step(GO, False); c.step(GO, False)                      # search_start (search_frames=1)
    assert c.step(GO, False)["state"] == SEARCH               # 2
    assert c.step(GO, False)["state"] == SEARCH               # 3
    o = c.step(GO, False)                                     # 4 > 3 → STOPPED
    assert o["state"] == STOPPED and o["reason"] == "search_timeout"
    assert c.step(GO, True, 0.05, 0.5)["state"] == STOPPED    # 래치


def test_policy_error_gives_zero_and_keeps_state():
    c = EgVla2Controller()
    for bad in ({"error": "x"}, None):
        o = c.step(bad, True, 0.05, 0.5)
        assert o["state"] == DRIVE and o["lx"] == o["ly"] == 0.0 and o["reason"] == "policy_error"


def test_reset_clears_latch_and_counters():
    c = EgVla2Controller(stop_debounce=1)
    assert c.step(STOP, True)["state"] == STOPPED
    c.reset()
    o = c.step(GO, True, 0.05, 0.5)
    assert o["state"] == DRIVE and o["lx"] > 0 and c.stop_run == 0 and c.last_cx == 0.5


def _small(az):   # 크기 작음(stop) + az
    return {"stop": True, "dir_lx": 0.0, "dir_ly": 0.0, "speed_scale": 0.0, "model_magnitude": 0.01, "az": az}


def test_rotate_instead_of_stop_when_az_large():
    c = EgVla2Controller()
    o = c.step(_small(0.438), True, 0.03, 0.76)
    assert o["state"] == DRIVE and not o["stopped"] and o["lx"] == 0.0 and o["ly"] == 0.0
    assert o["az"] == 0.25 and o["reason"].startswith("rotate(")          # 타깃이 오른쪽(cx 0.76) → 우회전(+az)
    o2 = c.step(_small(-0.4), True, 0.03, 0.2)
    assert o2["az"] == -0.25                                              # 타깃이 왼쪽 → 좌회전(-az)
    o3 = EgVla2Controller().step(_small(0.4), True, 0.03, None)
    assert o3["az"] == 0.25                                               # cx를 모르면 모델 az 부호 그대로(+ = 우회전)


def test_no_rotate_when_target_centered():
    c = EgVla2Controller()
    o = c.step(_small(0.4), True, 0.03, 0.52)
    assert o["reason"].startswith("idle(") and o["az"] == 0.0


def test_rotate_stops_after_max_frames_then_idle_not_latched():
    c = EgVla2Controller(rotate_max_frames=3, idle_max_frames=5)
    for i in range(3):
        assert c.step(_small(0.4), True, 0.03, 0.9)["reason"].startswith("rotate(")
    o = c.step(_small(0.4), True, 0.03, 0.9)
    assert o["reason"].startswith("idle(") and not o["stopped"]


def test_rotate_resets_on_move_and_default_has_no_latch():
    c = EgVla2Controller(rotate_max_frames=2, idle_max_frames=50)
    assert c.stop_debounce == 0
    c.step(_small(0.4), True, 0.03, 0.9); c.step(_small(0.4), True, 0.03, 0.9)
    c.step({"stop": False, "dir_lx": 1.0, "dir_ly": 0.0, "speed_scale": 0.7, "az": 0.0}, True, 0.03, 0.5)
    assert c.step(_small(0.4), True, 0.03, 0.9)["reason"].startswith("rotate(1/")
    for _ in range(10):                                                   # az 작은 stop이 반복돼도 래치 없음
        o = c.step(_small(0.0), True, 0.03, 0.5)
    assert not o["stopped"] and o["reason"].startswith("idle(")


def test_idle_timeout_safety_net():
    c = EgVla2Controller(idle_max_frames=3)
    for _ in range(2):
        assert c.step(_small(0.0), True, 0.03, 0.5)["reason"].startswith("idle(")
    o = c.step(_small(0.0), True, 0.03, 0.5)
    assert o["stopped"] and o["reason"] == "idle_timeout"

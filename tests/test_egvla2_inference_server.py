"""EGVLA2 단독 서버 — 모의 추출기·정책을 주입해 응답 조립·컨트롤러 연동을 검증한다(모델·GPU·실서버 불필요)."""
import base64
import io

import numpy as np

from robovlm_nav.serve.continuous_speed import to_drive_command
from robovlm_nav.serve.egvla2_inference_server import EgVla2Service, build_response, decode_image
from robovlm_nav.serve.egvla2_only_logic import EgVla2Controller

REQ_KEYS = {"action", "action_3d", "action_speed_scale", "drive", "goal_near", "stop_latched", "bbox", "predicted_label", "latency_ms",
            "grounding_latency_ms", "grounding_cached", "action_cont", "controller_state", "reason", "subgoal_missing", "ckpt", "ckpt_sha256"}


class MockPolicy:
    window = 3

    def __init__(self, seq):
        self.seq, self.i, self._hist = list(seq), 0, []

    def reset(self):
        self.i, self._hist = 0, []

    def step(self, bbox, feat):
        lx, ly = self.seq[min(self.i, len(self.seq) - 1)]; self.i += 1
        self._hist.append(bbox)
        return dict(first=(lx, ly, 0.0), action_cont=to_drive_command(lx, ly), subgoal_missing=bool(bbox[3] < 0.5), spread=0.1, arch="mock")


def mock_extractor(bboxes):
    it = iter(bboxes)
    return lambda img: (np.array(next(it), np.float32), np.zeros(768, np.float32))


def svc(seq, bboxes, **kw):
    return EgVla2Service(mock_extractor(bboxes), MockPolicy(seq), EgVla2Controller(**kw), meta=dict(ckpt="mock.pt", sha256="abc", arch="mock", dtype="fp32"))


IMG = np.zeros((8, 8, 3), np.uint8)
SEE = [0.5, 0.6, 0.05, 1.0]


def test_decode_image_roundtrip():
    from PIL import Image
    b = io.BytesIO(); Image.fromarray(np.full((4, 4, 3), (200, 10, 30), np.uint8)).save(b, "PNG")
    out = decode_image(base64.b64encode(b.getvalue()).decode())
    assert out.shape == (4, 4, 3) and tuple(out[0, 0]) == (200, 10, 30)             # RGB 순서 유지


def test_response_keys_and_drive_values():
    s = svc([(0.3, 0.4)], [SEE])
    r = s.predict_rgb(IMG)
    assert REQ_KEYS <= set(r)
    assert r["action_3d"][2] == 0.0 and abs(r["action"][0] - 0.6 * 1.15) < 1e-9 and abs(r["action_speed_scale"] - 0.7) < 1e-9
    assert r["goal_near"] is False and r["controller_state"] == "DRIVE" and r["drive"]["source"] == "egvla2_only"
    b = r["bbox"]
    assert abs(b["cx"] - 0.5) < 1e-6 and abs(b["cy"] - 0.6) < 1e-6 and abs(b["area"] - 0.05) < 1e-6 and b["has_bbox"] is True
    assert r["ckpt"] == "mock.pt" and r["ckpt_sha256"] == "abc" and r["buffer_status"]["window"] == 3


def test_stop_debounce_sets_goal_near_and_latch():
    s = svc([(0.05, 0.05)] * 4, [SEE] * 4, stop_debounce=3)
    outs = [s.predict_rgb(IMG) for _ in range(4)]
    assert [o["goal_near"] for o in outs] == [False, False, True, True]
    assert outs[2]["reason"] == "stop_debounce" and outs[3]["action_3d"] == [0.0, 0.0, 0.0] and outs[3]["stop_latched"] is True


def test_lost_search_rotation_in_response():
    miss = [0.5, 0.6, 0.0, 0.0]
    s = svc([(0.3, 0.4)] * 6, [SEE] + [miss] * 5, lost_frames=5)
    outs = [s.predict_rgb(IMG) for _ in range(6)]
    assert outs[-2]["controller_state"] == "DRIVE"                                   # 4프레임 소실까지는 구동 유지
    assert outs[-1]["controller_state"] == "SEARCH" and outs[-1]["action_3d"] == [0.0, 0.0, -0.25]   # 마지막 cx=0.5 → 우회전
    assert outs[-1]["subgoal_missing"] is True


def test_near_goal_proximity_flag():
    s = svc([(0.3, 0.4)] * 2, [[0.5, 0.6, 0.4, 1.0]] * 2, near_area=0.25, near_frames=2)
    o = [s.predict_rgb(IMG) for _ in range(2)][-1]
    assert o["goal_near"] and o["proximity_override"] and o["reason"] == "near_goal"


def test_reset_clears_service_state():
    s = svc([(0.05, 0.05)] * 2 + [(0.3, 0.4)], [SEE] * 3, stop_debounce=1)
    assert s.predict_rgb(IMG)["goal_near"] is True
    s.reset()
    assert s.n == 0 and s.controller.state == "DRIVE" and s.policy.i == 0


def test_build_response_standalone():
    out = dict(lx=0.0, ly=0.0, az=0.25, speed_scale=0.7, state="SEARCH", reason="search(2/40)", stopped=False, stop_reason="")
    r = build_response(out=out, action_cont=None, bbox=np.array([0.5, 0.6, 0.0, 0.0], np.float32), latency_ms=1.0, grounding_ms=0.5,
                       meta={}, subgoal_missing=True, window=3, history_size=2)
    assert r["action_3d"] == [0.0, 0.0, 0.25] and r["bbox"]["has_bbox"] is False and r["goal_near"] is False

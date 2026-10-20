"""EGVLA2 단독 추론 서버(단계 B 스켈레톤) — Florence-2 + FM 헤드만으로 구동 명령을 낸다. 기존 stage2_v2_inference_server.py 는 건드리지 않는다.

plan: docs/plans/plan_20261019_egvla2_inference_server.md §1 단계 B, §7.
파이프라인: base64 JPEG → PIL RGB → FusedF2Extractor → EgVla2FMPolicy(K 평균 + EMA) → EgVla2Controller(정지·근접·소실탐색) → 응답.
대시보드(mona_dashboard.py)가 읽는 키와 호환: action, action_3d, action_speed_scale(throttle), goal_near(정지 신호), bbox, predicted_label,
latency_ms, grounding_latency_ms, grounding_cached, drive. 응답은 dict 그대로 반환(response_model로 키가 잘리지 않게).
환경변수(기본 꺼짐이 아니라 이 서버를 따로 띄울 때만 사용):
  VLA_EGVLA2_CKPT(필수) VLA_SPEED_SOURCE(shadow) VLA_EGVLA2_EMA(0) VLA_EGVLA2_DTYPE(fp32|fp16, 기본 fp32) VLA_EGVLA2_K(8)
  VLA_EGVLA2_STOP_DEBOUNCE(3) VLA_EGVLA2_NEAR_AREA(0.25, off=끔) VLA_EGVLA2_NEAR_FRAMES(2) VLA_EGVLA2_LOST_FRAMES(5)
  VLA_EGVLA2_SEARCH_MAX(40) VLA_EGVLA2_GAIN(1.15) VLA_API_KEY(있으면 X-API-Key 검사) VLA_PORT(8001)
모델은 import 시 로드하지 않고 시작/첫 요청에서 지연 로드한다. JPEG 품질 q95 이상·색 보정 끔 등 입력 조건은 핸드오프 §15.2.
"""
from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from robovlm_nav.serve.egvla2_only_logic import EgVla2Controller

logger = logging.getLogger("egvla2_server")
MODEL_NAME = "egvla2_only"


def decode_image(image_b64: str) -> np.ndarray:
    from PIL import Image
    return np.array(Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB"))


def build_response(*, out: dict[str, Any], action_cont: Optional[dict[str, Any]], bbox: np.ndarray, latency_ms: float,
                   grounding_ms: float, meta: dict[str, Any], subgoal_missing: bool, window: int, history_size: int,
                   chunk_first: Optional[list[float]] = None, spread: Optional[float] = None) -> dict[str, Any]:
    """컨트롤러 출력과 정책 출력을 대시보드가 읽는 응답 dict로 조립한다(모델 불필요)."""
    lx, ly, az, sp = out["lx"], out["ly"], out["az"], out["speed_scale"]
    stopped = bool(out["stopped"])
    cx, cy, area, has = (float(bbox[0]), float(bbox[1]), float(bbox[2]), bool(bbox[3] >= 0.5))
    return {
        "action": [lx, ly], "action_3d": [lx, ly, az],
        "action_speed_scale": sp,                                   # 대시보드가 throttle_scale로 사용
        "drive": dict(lx=lx, ly=ly, az=az, speed_scale=sp, source=MODEL_NAME),
        "goal_near": stopped,                                       # 대시보드는 goal_near=True 이면 주행을 끝낸다(정지·도달·탐색 초과 모두)
        "stop_latched": stopped, "stop_mode": MODEL_NAME, "proximity_override": out["reason"] == "near_goal" or out.get("stop_reason") == "near_goal",
        "learned_stop": False, "goal_near_proxy": stopped,
        "predicted_class": None, "predicted_label": f"{out['state']}:{out['reason']}",
        "bbox": {"cx": cx, "cy": cy, "area": area, "has_bbox": has},
        "latency_ms": latency_ms, "grounding_latency_ms": grounding_ms, "grounding_cached": False, "head_mode": "egvla2",
        "buffer_status": {"history_size": history_size, "window": window, "head": meta.get("arch")},
        "action_cont": action_cont, "controller_state": out["state"], "reason": out["reason"], "subgoal_missing": bool(subgoal_missing),
        "chunk_first": chunk_first, "spread": spread,
        "model_name": MODEL_NAME, "strategy": MODEL_NAME, "source": MODEL_NAME,
        "ckpt": meta.get("ckpt"), "ckpt_sha256": meta.get("sha256"), "dtype": meta.get("dtype"),
    }


class EgVla2Service:
    """extractor(image_rgb)->(bbox4, feat768), policy.step(bbox, feat)->dict, controller 를 주입받는다(테스트는 모의 객체 주입)."""

    def __init__(self, extractor: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]], policy: Any, controller: EgVla2Controller,
                 meta: Optional[dict[str, Any]] = None):
        self.extractor, self.policy, self.controller, self.meta = extractor, policy, controller, dict(meta or {})
        self.n = 0
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self.policy.reset()
            self.controller.reset()
            self.n = 0

    def predict_rgb(self, image_rgb: np.ndarray) -> dict[str, Any]:
        t0 = time.time()
        with self._lock:
            bbox, feat = self.extractor(image_rgb)
            g_ms = (time.time() - t0) * 1000
            r = self.policy.step(bbox, feat)
            has = bool(bbox[3] >= 0.5)
            out = self.controller.step(r["action_cont"], has, float(bbox[2]), float(bbox[0]))
            self.n += 1
            hist = getattr(self.policy, "_hist", [])
            resp = build_response(out=out, action_cont=r["action_cont"], bbox=bbox, latency_ms=(time.time() - t0) * 1000, grounding_ms=g_ms,
                                  meta=self.meta, subgoal_missing=bool(r.get("subgoal_missing", False)), window=int(getattr(self.policy, "window", 0)),
                                  history_size=len(hist), chunk_first=list(r["first"]), spread=r.get("spread"))
        logger.info("[#%d] %s lat=%.0fms bbox(cx=%.2f area=%.3f has=%s)", self.n, resp["predicted_label"], resp["latency_ms"],
                    resp["bbox"]["cx"], resp["bbox"]["area"], resp["bbox"]["has_bbox"])
        return resp

    def predict(self, image_b64: str) -> dict[str, Any]:
        return self.predict_rgb(decode_image(image_b64))


# ── 환경변수 → 서비스(지연 로드) ──
def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


def load_service_from_env() -> EgVla2Service:
    import torch
    from robovlm_nav.serve.egvla2_fm_policy import EgVla2FMPolicy, FusedF2Extractor
    ckpt = Path(os.environ["VLA_EGVLA2_CKPT"])
    dt_name = os.getenv("VLA_EGVLA2_DTYPE", "fp32")
    if dt_name not in ("fp32", "fp16"):
        raise ValueError("VLA_EGVLA2_DTYPE must be fp32|fp16")
    dtype = torch.float32 if dt_name == "fp32" else torch.float16
    near = os.getenv("VLA_EGVLA2_NEAR_AREA", "0.25")
    ctrl = EgVla2Controller(stop_debounce=int(os.getenv("VLA_EGVLA2_STOP_DEBOUNCE", 3)), near_area=None if near == "off" else float(near),
                            near_frames=int(os.getenv("VLA_EGVLA2_NEAR_FRAMES", 2)), lost_frames=int(os.getenv("VLA_EGVLA2_LOST_FRAMES", 5)),
                            search_max_frames=int(os.getenv("VLA_EGVLA2_SEARCH_MAX", 40)), gain=_env_float("VLA_EGVLA2_GAIN", 1.15))
    policy = EgVla2FMPolicy(ckpt, k_samples=int(os.getenv("VLA_EGVLA2_K", 8)), speed_source=os.getenv("VLA_SPEED_SOURCE", "shadow"),
                            ema_alpha=_env_float("VLA_EGVLA2_EMA", 0.0))
    sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()
    meta = dict(ckpt=ckpt.name, sha256=sha, arch=policy.meta.get("arch"), dtype=dt_name, subgoal_dropout=policy.meta.get("subgoal_dropout", 0))
    logger.info("EGVLA2 service: ckpt=%s sha256=%s dtype=%s", ckpt.name, sha[:8], dt_name)
    return EgVla2Service(FusedF2Extractor(dtype=dtype), policy, ctrl, meta)


_service: Optional[EgVla2Service] = None
_service_lock = threading.Lock()


def get_service() -> EgVla2Service:
    global _service
    with _service_lock:
        if _service is None:
            _service = load_service_from_env()
        return _service


# ── FastAPI (기존 stage2 서버와 같은 엔드포인트 규약: /health, /predict, /reset, /model/info) ──
try:
    from pydantic import BaseModel

    class InferenceRequest(BaseModel):          # 모듈 수준 정의(from __future__ annotations 하에서 FastAPI가 타입을 해석하도록)
        image: str
        instruction: str = "basket"             # 호환용(무시: 프롬프트는 'gray basket' 고정)
except ImportError:                              # pragma: no cover
    InferenceRequest = None  # type: ignore[assignment,misc]


def create_app(service_getter: Callable[[], EgVla2Service] = get_service):
    from fastapi import FastAPI, Header, HTTPException

    app = FastAPI(title="EGVLA2 only inference API", version="0.1.0")

    def check_key(k: Optional[str]) -> None:
        expected = os.getenv("VLA_API_KEY", "")
        if expected and k != expected:
            raise HTTPException(status_code=403, detail="Invalid API Key")

    @app.get("/health")
    async def health() -> dict[str, Any]:
        s = _service
        d: dict[str, Any] = {"status": "ok", "model": MODEL_NAME, "loaded": s is not None, "ckpt_env": os.getenv("VLA_EGVLA2_CKPT")}
        if s is not None:
            d.update(meta=s.meta, controller_state=s.controller.state, frames=s.n)
        return d

    @app.get("/model/info")
    async def info(x_api_key: Optional[str] = Header(default=None)) -> dict[str, Any]:
        check_key(x_api_key)
        return dict(service_getter().meta, controller=vars(service_getter().controller))

    @app.post("/predict")
    async def predict(request: InferenceRequest, x_api_key: Optional[str] = Header(default=None)) -> dict[str, Any]:
        check_key(x_api_key)
        try:
            return service_getter().predict(request.image)
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Prediction failed")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.post("/reset")
    async def reset(x_api_key: Optional[str] = Header(default=None)) -> dict[str, Any]:
        check_key(x_api_key)
        service_getter().reset()
        return {"status": "success", "message": "EGVLA2 policy/controller reset"}

    return app


try:                      # fastapi가 없는 환경(테스트)에서도 import 가능하게
    app = create_app()
except ImportError:
    app = None


if __name__ == "__main__":
    import argparse

    import uvicorn
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.getenv("VLA_PORT", "8001")))
    ap.add_argument("--host", type=str, default="0.0.0.0")
    a = ap.parse_args()
    get_service()                       # 시작 시 로드(체크포인트 없으면 여기서 실패)
    uvicorn.run(app, host=a.host, port=a.port, log_level="info")

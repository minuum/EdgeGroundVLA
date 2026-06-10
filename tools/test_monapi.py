#!/usr/bin/env python3
"""
MoNa-pi inference server 테스트 도구 (MoNaVLA tools에서 실행)

MoNaVLA의 test_quick.py / test_2models.py 패턴으로
MoNa-pi 서버(8080)의 API 호환성과 응답을 검증한다.

실행:
    # MoNa-pi 서버가 먼저 기동되어 있어야 함
    python inference/server.py --mock --port 8080  # 테스트용
    python tools/test_monapi.py

환경변수:
    MONAPI_URL      MoNa-pi 서버 주소 (기본: http://localhost:8080)
    MONAPI_API_KEY  API 키 (기본: 없음)
"""

import base64
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # MoNaVLA 루트

MONAPI_URL = os.getenv("MONAPI_URL", "http://localhost:8080")
API_KEY    = os.getenv("MONAPI_API_KEY", "")
HEADERS    = {"x-api-key": API_KEY, "Content-Type": "application/json"} if API_KEY \
             else {"Content-Type": "application/json"}


# ─────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────

def _dummy_image_b64(w: int = 224, h: int = 224) -> str:
    try:
        from PIL import Image
        img = Image.new("RGB", (w, h), color=(120, 140, 160))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()
    except ImportError:
        # PIL 없으면 최소 JPEG 헤더 (mock 서버는 통과)
        import struct
        fake = b"\xff\xd8\xff\xe0" + b"\x00" * 100 + b"\xff\xd9"
        return base64.b64encode(fake).decode()


def _post(path: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{MONAPI_URL}{path}", data=data, headers=HEADERS, method="POST"
    )
    resp = urllib.request.urlopen(req, timeout=10)
    return json.loads(resp.read())


def _get(path: str) -> dict:
    req = urllib.request.Request(f"{MONAPI_URL}{path}", headers=HEADERS)
    resp = urllib.request.urlopen(req, timeout=5)
    return json.loads(resp.read())


def _section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("="*60)


def _ok(msg: str):
    print(f"  ✅ {msg}")


def _fail(msg: str):
    print(f"  ❌ {msg}")


# ─────────────────────────────────────────────
# 테스트
# ─────────────────────────────────────────────

results: list[tuple[str, bool]] = []


def run(name: str, fn):
    try:
        fn()
        results.append((name, True))
    except Exception as e:
        _fail(f"{name}: {e}")
        results.append((name, False))


def test_root():
    _section("1. GET / (루트)")
    data = _get("/")
    assert "service" in data, f"'service' 없음: {data}"
    _ok(f"service = {data['service']}")


def test_health():
    _section("2. GET /health")
    data = _get("/health")
    assert data["status"] in ("ok", "healthy"), f"status={data['status']}"
    _ok(f"engine_ready={data.get('engine_ready')}")


def test_predict_image_field():
    """MoNaVLA 포맷: 'image' 필드"""
    _section("3. POST /predict — MoNaVLA 포맷 (image 필드)")
    t0 = time.time()
    data = _post("/predict", {
        "image": _dummy_image_b64(),
        "instruction": "gray basket",
    })
    elapsed_ms = (time.time() - t0) * 1000
    assert "action" in data,  f"'action' 필드 없음"
    assert "actions" in data, f"'actions' 필드 없음"
    assert len(data["action"]) == 2, f"action 길이 {len(data['action'])} (기대: 2)"
    assert len(data["actions"]) == 10, f"chunk 길이 {len(data['actions'])} (기대: 10)"
    _ok(f"action={[round(v,3) for v in data['action']]}  latency={elapsed_ms:.0f}ms")


def test_predict_image_b64_field():
    """MoNa-pi 원래 포맷: 'image_b64' 필드"""
    _section("4. POST /predict — MoNa-pi 포맷 (image_b64 필드)")
    data = _post("/predict", {
        "image_b64": _dummy_image_b64(),
        "instruction": "Navigate to goal",
    })
    assert data["chunk_size"] == 10, f"chunk_size={data['chunk_size']}"
    assert data["action"] == data["actions"][0][:2], "action != actions[0][:2]"
    _ok(f"chunk_size={data['chunk_size']}, model_name={data.get('model_name')}")


def test_snap_to_grid_applied():
    """snap_to_grid 파라미터 적용 및 소프트 스냅핑 동작 검증"""
    _section("5. POST /predict — snap_to_grid 소프트 스냅핑 검증")
    
    # 1. snap_to_grid=False 일 때 (기본값)
    data_raw = _post("/predict", {
        "image": _dummy_image_b64(),
        "snap_to_grid": False,
    })
    
    # 2. snap_to_grid=True 일 때
    data_snap = _post("/predict", {
        "image": _dummy_image_b64(),
        "snap_to_grid": True,
    })
    
    assert data_snap["action"] is not None
    assert data_snap["action_3d"] is not None
    
    # mock 서버가 켜져있는지 검증
    # 만약 mock 서버라면, snap_to_grid=True 일 때 raw [0.6, -0.35, 0.1] -> 소프트 스냅핑되어 [0.6, -0.35, 0.05] 여야 함
    # snap_to_grid=False 일 때는 mock 서버 특성상 [0.0, 0.0, 0.0] 임
    if data_raw["action_3d"] == [0.0, 0.0, 0.0]:
        _ok("Mock 서버 환경 감지됨 - 소프트 스냅핑 수학적 연산 결과 정밀 검증")
        
        # snap 적용 전 raw_az = 0.1 이었으나 snap 후에는 raw_az * 0.5 = 0.05 여야 함.
        # 또한 label은 FWD+R
        assert data_snap["predicted_label"] == "FWD+R", f"Label mismatch: {data_snap['predicted_label']}"
        assert abs(data_snap["action_3d"][0] - 0.6) < 1e-5, f"Linear X mismatch: {data_snap['action_3d'][0]}"
        assert abs(data_snap["action_3d"][1] - (-0.35)) < 1e-5, f"Linear Y mismatch: {data_snap['action_3d'][1]}"
        assert abs(data_snap["action_3d"][2] - 0.05) < 1e-5, f"Angular Z mismatch (yaw 복원 실패): {data_snap['action_3d'][2]}"
        
        _ok(f"Mock 스냅 결과: {data_snap['action_3d']} (예상대로 raw_az 0.1 -> 0.05 로 부드럽게 감쇄 및 Yaw 복원)")
    else:
        # 실제 모델 서버라면 동작 검증만 수행
        _ok(f"실제 모델 서버 환경: snap 적용 전={data_raw['action_3d']}, snap 적용 후={data_snap['action_3d']}")



def test_model_switch():
    """MoNaVLA test_quick.py 패턴: /model/switch"""
    _section("6. POST /model/switch (MoNaVLA 호환)")
    data = _post("/model/switch", {"model_name": "exp49"})
    assert data["status"] == "ok", f"status={data['status']}"
    _ok(f"switch ok → model={data['model']}")


def test_metrics():
    _section("7. GET /metrics")
    data = _get("/metrics")
    assert "request_count" in data
    _ok(
        f"requests={data['request_count']}  "
        f"avg={data.get('avg_latency_ms')}ms  "
        f"p95={data.get('p95_latency_ms')}ms"
    )


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print(f"\n🤖 MoNa-pi Server Test  [{MONAPI_URL}]")
    if API_KEY:
        print(f"   API Key: {API_KEY[:6]}...")

    # 서버 응답 확인
    try:
        urllib.request.urlopen(f"{MONAPI_URL}/health", timeout=3)
    except Exception:
        print(f"\n❌ 서버에 연결할 수 없습니다: {MONAPI_URL}")
        print("   먼저 서버를 시작하세요:")
        print("   python inference/server.py --mock --port 8080")
        sys.exit(1)

    run("root",             test_root)
    run("health",           test_health)
    run("predict/image",    test_predict_image_field)
    run("predict/image_b64", test_predict_image_b64_field)
    run("snap_to_grid",     test_snap_to_grid_applied)
    run("model_switch",     test_model_switch)
    run("metrics",          test_metrics)

    # 결과 요약
    print(f"\n{'='*60}")
    passed = sum(1 for _, ok in results if ok)
    total  = len(results)
    for name, ok in results:
        icon = "✅" if ok else "❌"
        print(f"  {icon} {name}")
    print(f"\n  {passed}/{total} passed")
    print("="*60)

    sys.exit(0 if passed == total else 1)

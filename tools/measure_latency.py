#!/usr/bin/env python3
"""추론 latency 실측 — 8082 통합 서버 대상.

VLA /predict 와 GoalNav /goalnav/predict 의 1회 추론 latency를 N회 측정해
평균/p50/p95/최대 Hz를 보고한다. 합성 이미지(640x480) 사용.
"""
import base64
import io
import statistics
import time
import sys

import numpy as np
import requests
from PIL import Image

BASE = "http://localhost:8082"
N = 30
INSTRUCTION = "Move to the chair"


def make_image_b64() -> str:
    arr = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def measure(endpoint: str, payload_extra: dict) -> None:
    lat_server = []   # 서버가 보고한 latency_ms (순수 추론)
    lat_wall = []     # 클라이언트 왕복 wall-clock
    img = make_image_b64()
    # warmup 3
    for _ in range(3):
        try:
            requests.post(f"{BASE}{endpoint}", json={"image": img, "instruction": INSTRUCTION, **payload_extra}, timeout=60)
        except Exception as e:
            print(f"  warmup 실패: {e}")
            return
    for i in range(N):
        img = make_image_b64()
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{BASE}{endpoint}", json={"image": img, "instruction": INSTRUCTION, **payload_extra}, timeout=60)
            dt = (time.perf_counter() - t0) * 1000
        except Exception as e:
            print(f"  요청 {i} 실패: {e}")
            continue
        if r.status_code != 200:
            print(f"  요청 {i} status={r.status_code}: {r.text[:200]}")
            continue
        body = r.json()
        lat_wall.append(dt)
        if "latency_ms" in body:
            lat_server.append(float(body["latency_ms"]))
    if not lat_wall:
        print(f"  [{endpoint}] 유효 응답 없음")
        return
    def stats(name, xs):
        xs = sorted(xs)
        p50 = statistics.median(xs)
        p95 = xs[int(len(xs) * 0.95) - 1] if len(xs) > 1 else xs[0]
        mx = max(xs)
        print(f"    {name}: 평균 {statistics.mean(xs):6.1f}ms  p50 {p50:6.1f}ms  p95 {p95:6.1f}ms  max {mx:6.1f}ms  → {1000/statistics.mean(xs):.1f}Hz(평균)")
    print(f"  [{endpoint}] n={len(lat_wall)}")
    if lat_server:
        stats("서버추론", lat_server)
    stats("왕복wall", lat_wall)


if __name__ == "__main__":
    print(f"=== latency 실측 (N={N}) ===")
    # 1. VLA /predict (monapi 모드)
    print("\n[1] VLA /predict (vlm_model=monapi)")
    measure("/predict", {"vlm_model": "monapi"})
    # 2. GoalNav /goalnav/predict
    print("\n[2] GoalNav /goalnav/predict")
    measure("/goalnav/predict", {})

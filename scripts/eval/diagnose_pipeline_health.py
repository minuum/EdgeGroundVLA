#!/usr/bin/env python3
"""
파이프라인 건강진단 체크리스트 — 이번 세션에 실제로 겪은 문제들을 재현 가능하게
감지하는 스크립트. CLAUDE.md 체크리스트 문서(checklists/pipeline_health.md)와
1:1로 대응한다.

체크 항목:
  A. latency      — /predict 단발 호출 latency(웜업 후), 목표 1초 이내 확인
  B. drift        — 세션 JSON(예: s6_cl_sim.json)에서 "1fps 가정 vs 실제 처리속도"
                     누적 드리프트 계산 — "N초 차이"가 단발 latency 문제인지 드리프트인지 구분
  C. continuity   — 세션 JSON의 frame 번호가 연속인지(누락 프레임 감지)
  D. resize       — resize_for_vlm()이 실제로 224x224를 만드는지 자체 점검
  E. grounding    — 세션 JSON의 has_bbox/area 분포로 그라운딩 신뢰도 요약(CH41 패턴)

Usage:
  .venv/bin/python3 scripts/eval/diagnose_pipeline_health.py --server http://localhost:8001 --api-key $VLA_API_KEY
  .venv/bin/python3 scripts/eval/diagnose_pipeline_health.py --session docs/v5/s6_cl_sim.json --fps 1.0
  .venv/bin/python3 scripts/eval/diagnose_pipeline_health.py --server ... --api-key ... --session ... --fps 1.0
"""
import argparse
import base64
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def check_latency(server: str, api_key: str, n: int = 5) -> dict:
    import requests
    img_path = ROOT / "docs/object_test_images/test_apple_floor_1768456959811.png"
    b64 = base64.b64encode(img_path.read_bytes()).decode()
    headers = {"x-api-key": api_key} if api_key else {}

    requests.post(f"{server}/reset", headers=headers, timeout=10)
    lats = []
    for _ in range(n):
        r = requests.post(f"{server}/predict", json={"image": b64, "instruction": "basket"},
                           headers=headers, timeout=30)
        r.raise_for_status()
        lats.append(r.json()["latency_ms"])

    mean_ms = float(np.mean(lats))
    status = "PASS" if mean_ms < 1000 else "WARN" if mean_ms < 2000 else "FAIL"
    print(f"\n[A. LATENCY] n={n}  mean={mean_ms:.0f}ms  max={max(lats):.0f}ms  "
          f"목표<1000ms  →  {status}")
    return {"check": "latency", "status": status, "mean_ms": mean_ms, "samples": lats}


def check_drift(session_path: str, fps: float) -> dict:
    data = json.loads(Path(session_path).read_text())
    frames = data["frames"]
    total = [f["total_latency_ms"] for f in frames]
    cum_real_s = np.cumsum(total) / 1000.0
    nominal_s = np.arange(1, len(frames) + 1) / fps
    drift = cum_real_s - nominal_s

    mean_latency_s = np.mean(total) / 1000.0
    nominal_interval_s = 1.0 / fps
    will_diverge = mean_latency_s > nominal_interval_s

    status = "FAIL(드리프트 발산)" if will_diverge else "PASS"
    print(f"\n[B. DRIFT] session={data.get('session')}  frames={len(frames)}  fps={fps}")
    print(f"  평균 처리시간={mean_latency_s:.2f}s/frame  vs  1fps 가정 간격={nominal_interval_s:.2f}s/frame")
    print(f"  마지막 프레임 드리프트={drift[-1]:.1f}s  →  {status}")
    if will_diverge:
        print(f"  ⚠ 처리속도가 가정한 fps보다 느려서 드리프트가 시간이 갈수록 계속 커진다 "
              f"(\"N초 차이\"로 보이는 건 단발 latency가 아니라 이 누적값일 가능성 높음)")
    return {"check": "drift", "status": status, "final_drift_s": float(drift[-1]),
            "mean_latency_s": mean_latency_s, "nominal_interval_s": nominal_interval_s}


def check_continuity(session_path: str) -> dict:
    data = json.loads(Path(session_path).read_text())
    frames = data["frames"]
    nums = [f["frame"] for f in frames]
    expected = list(range(1, len(frames) + 1))
    missing = sorted(set(expected) - set(nums))
    status = "PASS" if not missing else "FAIL"
    print(f"\n[C. CONTINUITY] {len(nums)}개 프레임, 누락={len(missing)}개  →  {status}")
    if missing:
        print(f"  누락된 frame 번호: {missing[:20]}{'...' if len(missing) > 20 else ''}")
    return {"check": "continuity", "status": status, "missing": missing}


def check_resize() -> dict:
    from robovlm_nav.image_preprocess import resize_for_vlm, VLM_INPUT_SIZE
    from PIL import Image
    test_img = Image.new("RGB", (1280, 720), (128, 128, 128))
    out = resize_for_vlm(test_img)
    ok = out.size == (VLM_INPUT_SIZE, VLM_INPUT_SIZE)
    status = "PASS" if ok else "FAIL"
    print(f"\n[D. RESIZE] 1280x720 입력 → {out.size} 출력 (목표 {VLM_INPUT_SIZE}x{VLM_INPUT_SIZE})  →  {status}")
    return {"check": "resize", "status": status, "output_size": list(out.size)}


def check_grounding(session_path: str) -> dict:
    data = json.loads(Path(session_path).read_text())
    frames = data["frames"]
    has_bbox = [f["bbox"]["has_bbox"] for f in frames]
    no_bbox_rate = 1 - sum(has_bbox) / len(has_bbox)
    areas = [f["bbox"]["area"] for f in frames if f["bbox"]["has_bbox"]]
    status = "WARN" if no_bbox_rate > 0.3 else "PASS"
    print(f"\n[E. GROUNDING] has_bbox=False 비율={no_bbox_rate*100:.1f}%  "
          f"area 평균(검출된 것만)={np.mean(areas):.3f}  →  {status}")
    return {"check": "grounding", "status": status, "no_bbox_rate": no_bbox_rate}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", default=None, help="추론 서버 URL (예: http://localhost:8001)")
    p.add_argument("--api-key", default="", help="VLA_API_KEY")
    p.add_argument("--session", default=None, help="세션 JSON 경로(예: docs/v5/s6_cl_sim.json)")
    p.add_argument("--fps", type=float, default=1.0, help="세션 영상 추출 fps(드리프트 계산용)")
    p.add_argument("--n", type=int, default=5, help="latency 측정 반복 횟수")
    args = p.parse_args()

    results = []
    results.append(check_resize())

    if args.server:
        results.append(check_latency(args.server, args.api_key, args.n))

    if args.session:
        results.append(check_continuity(args.session))
        results.append(check_drift(args.session, args.fps))
        results.append(check_grounding(args.session))

    print(f"\n{'='*50}")
    fails = [r for r in results if "FAIL" in r["status"]]
    warns = [r for r in results if "WARN" in r["status"]]
    print(f"[종합] {len(results)}개 체크 — FAIL={len(fails)} WARN={len(warns)} PASS={len(results)-len(fails)-len(warns)}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()

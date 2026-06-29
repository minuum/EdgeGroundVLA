#!/usr/bin/env python3
"""
세션 CL 시뮬레이션 스크립트 (범용)
soda에서 직접 실행: python3 run_s6_cl_sim.py <session_name> [frame_dir] [out_path]

/reset → 순차 /predict (N프레임) → temporal N=3 포함 실제 CL 파이프라인 재현
"""

import os, json, base64, time, subprocess, sys
from pathlib import Path

API_KEY   = os.getenv("VLA_API_KEY", "vla_devel_key_2026")
BASE_URL  = "http://localhost:8001"
SESSION_NAME = sys.argv[1] if len(sys.argv) > 1 else "gnd_20260618_172621"
FRAME_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(os.path.expanduser(f"~/tmp_gnd_eval/results/{SESSION_NAME}"))
OUT_PATH  = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(os.path.expanduser(f"~/tmp_gnd_eval/results/{SESSION_NAME}_cl_sim.json"))

HEADERS = ["-H", "Content-Type: application/json", "-H", f"x-api-key: {API_KEY}"]

def curl_post(endpoint: str, payload: bytes, timeout: int = 30) -> dict:
    result = subprocess.run(
        ["curl", "-sf", "-X", "POST", f"{BASE_URL}{endpoint}"] + HEADERS + ["--data-binary", "@-"],
        input=payload, capture_output=True, timeout=timeout
    )
    if result.returncode != 0:
        raise RuntimeError(f"curl failed ({result.returncode}): {result.stderr.decode()[:200]}")
    return json.loads(result.stdout.decode())

# ── 1. 서버 상태 확인 ──────────────────────────────────────────────────────
r = subprocess.run(
    ["curl", "-sf", f"{BASE_URL}/model/info"] + HEADERS,
    capture_output=True, timeout=10
)
info = json.loads(r.stdout.decode())
print(f"서버: head={info['head']} window={info['window']} loaded={info['model_loaded']}")

# ── 2. /reset — history 초기화 ──────────────────────────────────────────────
reset_res = curl_post("/reset", b"{}")
print(f"Reset: {reset_res}")

# ── 3. 순차 /predict — 105프레임 ───────────────────────────────────────────
frames = sorted(FRAME_DIR.glob("frame_*.jpg"))
print(f"\n총 {len(frames)}프레임 순차 CL 시뮬레이션 시작...")
print("=" * 70)

results = []
consec_counter = 0  # 서버 내부 로직 미러링용 (로컬 검증)

GOAL_AREA  = 0.25
GOAL_CX    = 0.35
GOAL_N     = 3

for i, frm in enumerate(frames):
    with open(frm, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = json.dumps({"image": b64, "instruction": "basket"}).encode()

    t0 = time.time()
    try:
        res = curl_post("/predict", payload, timeout=30)
        lat = (time.time() - t0) * 1000
    except Exception as e:
        print(f"  f{i+1:03d}: ERROR — {e}")
        results.append({"frame": i+1, "error": str(e)})
        continue

    # 응답 파싱
    bbox      = res.get("bbox") or {}
    has_bbox  = bbox.get("has_bbox", False)
    area      = bbox.get("area", 0.0) or 0.0
    cx        = bbox.get("cx", 0.5)
    label     = res.get("predicted_label", "?")
    cls_id    = res.get("predicted_class")
    prox      = res.get("proximity_override", False)
    near_prox = res.get("goal_near_proxy", False)
    g_lat     = res.get("grounding_latency_ms", 0)

    # 로컬 temporal 카운터 (검증용)
    meets = has_bbox and area >= GOAL_AREA and abs(cx - 0.5) <= GOAL_CX
    if meets:
        consec_counter += 1
    else:
        consec_counter = 0

    # 카테고리 분류 (그라운딩 기준)
    if not has_bbox:
        cat = "NO_BBOX"
    elif area < 0.01:
        cat = "TINY"
    elif area > 0.9:
        cat = "FULL"
    elif bbox.get("cy", 0.5) < 0.35:
        cat = "TOP"
    else:
        if area >= GOAL_AREA:
            cat = "NEAR"   # area≥0.25 → GOAL 후보
        else:
            cat = "OK"

    stop_flag = "🛑STOP" if prox else ""
    near_tag  = f"[{min(consec_counter,GOAL_N)}/{GOAL_N}]" if consec_counter > 0 else ""
    print(f"  f{i+1:03d}: {label:10s} {stop_flag:8s} | "
          f"{cat:7s} a={area:.3f} cx={cx:.3f} {near_tag:6s} | "
          f"gnd={g_lat:.0f}ms tot={lat:.0f}ms")

    records = {
        "frame": i + 1,
        "label": label,
        "cls_id": cls_id,
        "proximity_override": prox,
        "goal_near_proxy": near_prox,
        "bbox": {"has_bbox": has_bbox, "area": area, "cx": cx,
                 "cy": bbox.get("cy", 0.5), "cat": cat},
        "consec_local": consec_counter,
        "grounding_latency_ms": g_lat,
        "total_latency_ms": lat,
    }
    results.append(records)

# ── 4. 통계 ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
total = len(results)
errors = sum(1 for r in results if "error" in r)
stops  = sum(1 for r in results if r.get("proximity_override"))
by_cls = {}
for r in results:
    if "error" not in r:
        lbl = r.get("label", "?")
        by_cls[lbl] = by_cls.get(lbl, 0) + 1

near_frames = sum(1 for r in results if r.get("bbox", {}).get("cat") == "NEAR")
ok_frames   = sum(1 for r in results if r.get("bbox", {}).get("cat") == "OK")
no_bbox     = sum(1 for r in results if r.get("bbox", {}).get("cat") == "NO_BBOX")

print(f"[CL 시뮬레이션 결과] 총 {total}프레임")
print(f"  PROXIMITY STOP 발동: {stops}회")
print(f"  NO_BBOX: {no_bbox} | OK: {ok_frames} | NEAR(≥0.25): {near_frames}")
print(f"  액션 분포: {dict(sorted(by_cls.items(), key=lambda x: -x[1]))}")

# 최대 연속 near 구간
max_consec = 0
cur = 0
for r in results:
    if r.get("bbox", {}).get("cat") == "NEAR":
        cur += 1
        max_consec = max(max_consec, cur)
    else:
        cur = 0
print(f"  최대 연속 NEAR 구간: {max_consec}프레임 (GOAL 발동 임계값: {GOAL_N})")

# ── 5. 저장 ────────────────────────────────────────────────────────────────
out = {
    "session": SESSION_NAME,
    "total_frames": total,
    "goal_params": {"area": GOAL_AREA, "cx_tol": GOAL_CX, "n": GOAL_N},
    "summary": {
        "proximity_stops": stops,
        "no_bbox": no_bbox,
        "ok": ok_frames,
        "near": near_frames,
        "max_consec_near": max_consec,
        "action_dist": by_cls,
    },
    "frames": results,
}
with open(OUT_PATH, "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\n[완료] {OUT_PATH}")

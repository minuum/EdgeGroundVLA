#!/usr/bin/env python3
"""
PG2 Grounding 오프라인 평가 스크립트
soda에서 직접 실행: python3 run_pg2_ground_eval.py

1. S4 24 JPEG 프레임 → /ground API → JSONL 기록과 비교
2. MP4 비디오 → 1fps 추출 → /ground API → JSONL 기록과 비교
3. 결과를 JSON으로 저장
"""

import os, json, base64, time, subprocess, sys
from pathlib import Path

GROUND_URL = "http://localhost:8001/ground"
WORK_DIR = Path(os.path.expanduser("~/tmp_gnd_eval"))
OUT_DIR = Path(os.path.expanduser("~/tmp_gnd_eval/results"))
OUT_DIR.mkdir(exist_ok=True)

# ── JSONL 기준 데이터 ────────────────────────────────────────────────
S4_JSONL_REF = [
    # 첫 24건 (gnd_20260618_152628.jsonl에서 추출)
    # n, has_bbox, area, cx, cy, cat (filmstrip 분류)
    (1,  True,  0.4815, 0.4976, 0.7581, "XFULL"),
    (2,  True,  0.0011, 0.4985, 0.5357, "TINY"),
    (3,  True,  0.4514, 0.4976, 0.7732, "XFULL"),
    (4,  True,  0.4514, 0.4976, 0.7732, "XFULL"),
    (5,  True,  0.0014, 0.4707, 0.5191, "TINY"),
    (6,  True,  0.0014, 0.4707, 0.5191, "TINY"),
    (7,  True,  0.0013, 0.4985, 0.5191, "TINY"),
    (8,  True,  0.0013, 0.5015, 0.5191, "TINY"),
    (9,  True,  0.2581, 0.6026, 0.4115, "OK"),
    (10, True,  0.9873, 0.4995, 0.4941, "FULL"),
    (11, True,  0.9980, 0.4995, 0.4995, "FULL"),
    (12, True,  0.9980, 0.4995, 0.4995, "FULL"),
    (13, True,  0.9863, 0.4995, 0.4936, "FULL"),
    (14, True,  0.9863, 0.4995, 0.4936, "FULL"),
    (15, True,  0.0564, 0.5049, 0.6026, "OK"),
    (16, True,  0.5506, 0.4976, 0.7234, "XFULL"),
    (17, True,  0.0741, 0.3309, 0.8842, "OK"),
    (18, True,  0.1051, 0.5181, 0.5924, "OK"),
    (19, True,  0.0779, 0.4902, 0.5767, "OK"),
    (20, True,  0.0779, 0.4902, 0.5767, "OK"),
    (21, True,  0.5054, 0.5000, 0.7473, "XFULL"),
    (22, True,  0.5054, 0.5000, 0.7473, "XFULL"),
    (23, True,  0.9863, 0.4995, 0.4936, "FULL"),
    (24, True,  0.2335, 0.5518, 0.4164, "OK"),
]

def ground_image(img_path: str) -> dict:
    """이미지 파일을 base64로 인코딩해 /ground에 POST (stdin 방식)."""
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    payload = json.dumps({"image": b64, "prompt": "detect gray basket"}).encode()
    t0 = time.time()
    api_key = os.getenv("VLA_API_KEY", "vla_devel_key_2026")
    result = subprocess.run(
        ["curl", "-sf", "-X", "POST", GROUND_URL,
         "-H", "Content-Type: application/json",
         "-H", f"x-api-key: {api_key}",
         "--data-binary", "@-"],
        input=payload,
        capture_output=True, timeout=30
    )
    latency_ms = (time.time() - t0) * 1000
    if result.returncode != 0:
        return {"error": result.stderr.decode()[:200], "latency_ms": latency_ms}
    try:
        d = json.loads(result.stdout.decode())
        d["latency_ms"] = latency_ms
        return d
    except Exception as e:
        return {"error": str(e), "raw": result.stdout.decode()[:200], "latency_ms": latency_ms}

def compute_iou(a1, b1):
    """두 (area, cx, cy) bbox 쌍의 IoU 근사치 (정사각 bbox 가정)."""
    # bbox: cx, cy, w, h (h≈w≈sqrt(area) 정사각 가정)
    def bbox(a, cx, cy):
        s = a**0.5
        return cx - s/2, cy - s/2, cx + s/2, cy + s/2
    if not (a1 and b1 and a1.get("area") and b1.get("area")):
        return None
    x1a, y1a, x2a, y2a = bbox(a1["area"], a1["cx"], a1["cy"])
    x1b, y1b, x2b, y2b = bbox(b1["area"], b1["cx"], b1["cy"])
    ix1, iy1 = max(x1a, x1b), max(y1a, y1b)
    ix2, iy2 = min(x2a, x2b), min(y2a, y2b)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = a1["area"] + b1["area"] - inter
    return inter / union if union > 0 else 0.0

# ── 1. S4 24프레임 평가 ────────────────────────────────────────────────
print("=" * 60)
print("S4 24 JPEG FRAMES → /ground 비교")
print("=" * 60)

s4_results = []
for n, ref_has_bbox, ref_area, ref_cx, ref_cy, ref_cat in S4_JSONL_REF:
    img_file = WORK_DIR / f"s4_n{n:02d}_{ref_cat.lower()}.jpg"
    if not img_file.exists():
        print(f"  n={n:02d}: FILE NOT FOUND ({img_file.name})")
        continue

    res = ground_image(str(img_file))
    server_has_bbox = res.get("has_bbox", False)
    server_area = res.get("area", 0.0) or 0.0
    server_cx = res.get("cx", 0.5) or 0.5
    server_cy = res.get("cy", 0.6) or 0.6
    lat = res.get("latency_ms", 0)

    # IoU (근사)
    iou = compute_iou(
        {"area": ref_area, "cx": ref_cx, "cy": ref_cy} if ref_has_bbox else None,
        {"area": server_area, "cx": server_cx, "cy": server_cy} if server_has_bbox else None
    )
    area_diff = abs(server_area - ref_area) if server_has_bbox else None
    cx_diff = abs(server_cx - ref_cx) if server_has_bbox else None

    match = (server_has_bbox == ref_has_bbox)
    status = "✅" if match and (iou is None or iou > 0.5) else "⚠️" if match else "❌"

    iou_str = f"{iou:.2f}" if iou is not None else "N/A"
    raw_out = res.get("raw_output", "")[:40]
    print(f"  n={n:02d} [{ref_cat:5s}] {status} | "
          f"ref: has={ref_has_bbox} a={ref_area:.3f} cx={ref_cx:.3f} | "
          f"srv: has={server_has_bbox} a={server_area:.3f} cx={server_cx:.3f} | "
          f"IoU={iou_str} lat={lat:.0f}ms raw='{raw_out}'")

    s4_results.append({
        "n": n, "cat": ref_cat,
        "ref": {"has_bbox": ref_has_bbox, "area": ref_area, "cx": ref_cx, "cy": ref_cy},
        "server": {"has_bbox": server_has_bbox, "area": server_area, "cx": server_cx, "cy": server_cy,
                   "raw": res.get("raw_output", ""), "latency_ms": lat},
        "iou": iou, "area_diff": area_diff, "cx_diff": cx_diff, "match": match
    })

# 요약
total = len(s4_results)
has_bbox_match = sum(1 for r in s4_results if r["match"])
iou_vals = [r["iou"] for r in s4_results if r["iou"] is not None]
area_diffs = [r["area_diff"] for r in s4_results if r["area_diff"] is not None]
cx_diffs = [r["cx_diff"] for r in s4_results if r["cx_diff"] is not None]

print(f"\n[S4 요약]")
print(f"  has_bbox 일치: {has_bbox_match}/{total} ({has_bbox_match/total*100:.1f}%)")
if iou_vals:
    print(f"  IoU 평균: {sum(iou_vals)/len(iou_vals):.3f} (min={min(iou_vals):.3f} max={max(iou_vals):.3f})")
if area_diffs:
    print(f"  area 오차 평균: {sum(area_diffs)/len(area_diffs):.4f}")
if cx_diffs:
    print(f"  cx 오차 평균: {sum(cx_diffs)/len(cx_diffs):.4f}")

# ── 2. MP4 → 1fps 추출 → 비교 ─────────────────────────────────────────
print("\n" + "=" * 60)
print("MP4 → 1fps 추출 → /ground")
print("=" * 60)

import glob
mp4_files = sorted(glob.glob(str(WORK_DIR / "gnd_*.mp4")))

mp4_results = {}
for mp4 in mp4_files:
    name = Path(mp4).stem
    frame_dir = OUT_DIR / name
    frame_dir.mkdir(exist_ok=True)
    print(f"\n  [{name}] 프레임 추출 중...")

    # ffmpeg 1fps 추출
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", mp4, "-vf", "fps=1",
         str(frame_dir / "frame_%04d.jpg"), "-loglevel", "error"],
        capture_output=True
    )
    frames = sorted(frame_dir.glob("frame_*.jpg"))
    print(f"  → {len(frames)} 프레임 추출됨")

    session_res = []
    for i, frm in enumerate(frames):
        res = ground_image(str(frm))
        has_b = res.get("has_bbox", False)
        area = res.get("area", 0.0) or 0.0
        cx = res.get("cx", 0.5) or 0.5
        lat = res.get("latency_ms", 0)
        raw = res.get("raw_output", "")
        cat = "OK"
        if has_b:
            if area > 0.9: cat = "FULL"
            elif area < 0.01: cat = "TINY"
            elif res.get("cy", 0.6) < 0.35: cat = "TOP"
        else:
            cat = "NO_BBOX"
        session_res.append({
            "frame": i+1, "has_bbox": has_b, "area": area, "cx": cx,
            "cat": cat, "raw": raw, "latency_ms": lat
        })
        print(f"    f{i+1:03d}: {cat:7s} a={area:.3f} cx={cx:.3f} lat={lat:.0f}ms")

    # 세션 통계
    ok = [r for r in session_res if r["cat"] == "OK"]
    no_bbox = [r for r in session_res if r["cat"] == "NO_BBOX"]
    noisy = [r for r in session_res if r["cat"] in ("FULL","TINY","TOP","XFULL")]
    print(f"  → OK={len(ok)} ({len(ok)/len(session_res)*100:.1f}%) "
          f"NO_BBOX={len(no_bbox)} NOISY={len(noisy)}")
    mp4_results[name] = session_res

# ── 결과 저장 ─────────────────────────────────────────────────────────
out = {
    "s4_frames": s4_results,
    "mp4_sessions": {k: v for k, v in mp4_results.items()}
}
out_path = OUT_DIR / "ground_eval_results.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\n[완료] 결과 저장: {out_path}")

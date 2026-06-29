#!/usr/bin/env python3
"""
객체별 GOAL_AREA 캘리브레이션 — soda에서 물리적으로 실행.

방법: 캘리브레이션하려는 객체를 "로봇이 멈춰야 하는 바로 그 거리"에 실제로 놓고,
이 스크립트로 현재 카메라 프레임을 잡아 /ground에 보내 area를 측정한다.
바스켓의 GOAL_AREA=0.25도 이런 식으로(실주행 세션 S6~S8) 경험적으로 나온 값이라,
사진 한 장으로 수학적으로 환산하지 않고 동일한 방식으로 실측한다.

n회 반복(기본 3) 측정 후 median을 그 객체의 GOAL_AREA로 configs/goal_area_map.json에 저장.
서버는 재시작 없이 다음 /predict 호출부터 즉시 새 값을 읅음 — 단, 현재 구현은 모듈 로드
시점에 1회 읆으므로(GOAL_AREA_MAP 전역), 갱신 반영을 위해서는 서버 재시작이 필요하다
(이번 버전 한계 — 핫리로드는 추후 필요시 추가).

사용 (soda에서, 카메라가 보이는 상태로 물체를 직접 들고 거리 맞춘 뒤):
  python3 scripts/calibrate_goal_area.py --instruction "green apple" --camera /dev/video0
  # 또는 이미 캡처한 이미지 파일로:
  python3 scripts/calibrate_goal_area.py --instruction "green apple" --image-glob "/tmp/calib_apple_*.jpg"
"""
import argparse
import base64
import glob
import json
import os
import statistics
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
MAP_PATH = ROOT / "configs" / "goal_area_map.json"
GROUND_URL = "http://localhost:8001/ground"


def ground_image_path(img_path: str, instruction: str) -> dict:
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    api_key = os.getenv("VLA_API_KEY", "vla_devel_key_2026")
    resp = requests.post(
        GROUND_URL,
        json={"image": b64, "prompt": f"detect {instruction}"},
        headers={"x-api-key": api_key},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def capture_from_camera(camera_dev: str, out_path: str) -> bool:
    """v4l2 카메라에서 1프레임 캡처 (fswebcam 또는 ffmpeg 필요)."""
    import subprocess
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "v4l2", "-i", camera_dev, "-frames:v", "1", out_path],
            capture_output=True, timeout=10, check=True,
        )
        return True
    except Exception as e:
        print(f"  [캡처 실패] {e}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruction", required=True, help='예: "green apple" — grounding phrase로 그대로 쓰임')
    ap.add_argument("--n", type=int, default=3, help="반복 측정 횟수")
    ap.add_argument("--camera", default=None, help="v4l2 디바이스(예: /dev/video0) — 지정 시 매 반복마다 새로 캡처")
    ap.add_argument("--image-glob", default=None, help="이미 캡처된 이미지 glob 패턴 — camera 대신 사용")
    ap.add_argument("--dry-run", action="store_true", help="goal_area_map.json에 저장하지 않고 측정값만 출력")
    args = ap.parse_args()

    areas = []
    if args.image_glob:
        files = sorted(glob.glob(args.image_glob))
        if not files:
            print(f"[오류] {args.image_glob} 에 해당하는 파일 없음")
            return
        for f in files[: args.n]:
            res = ground_image_path(f, args.instruction)
            area = res.get("area")
            has_bbox = res.get("has_bbox")
            print(f"  {f}: has_bbox={has_bbox} area={area}")
            if has_bbox and area:
                areas.append(area)
    else:
        if not args.camera:
            print("[오류] --camera 또는 --image-glob 중 하나는 필요")
            return
        tmp_dir = Path("/tmp/goal_area_calib")
        tmp_dir.mkdir(exist_ok=True)
        for i in range(args.n):
            out = str(tmp_dir / f"{args.instruction.replace(' ', '_')}_{i}.jpg")
            print(f"  [{i+1}/{args.n}] 캡처 중... (물체를 정지거리에 고정해두세요)")
            if not capture_from_camera(args.camera, out):
                continue
            res = ground_image_path(out, args.instruction)
            area = res.get("area")
            has_bbox = res.get("has_bbox")
            print(f"    has_bbox={has_bbox} area={area}")
            if has_bbox and area:
                areas.append(area)
            time.sleep(0.5)

    if not areas:
        print("\n[실패] 유효한 측정값이 없음 — grounding이 안 됐거나 객체가 화면에 없음")
        return

    median_area = statistics.median(areas)
    print(f"\n[측정 결과] n={len(areas)} median_area={median_area:.4f} (raw={[round(a,4) for a in areas]})")

    if args.dry_run:
        print("[dry-run] goal_area_map.json에 저장하지 않음")
        return

    cur = json.loads(MAP_PATH.read_text()) if MAP_PATH.exists() else {}
    cur[args.instruction] = round(median_area, 4)
    MAP_PATH.parent.mkdir(exist_ok=True)
    MAP_PATH.write_text(json.dumps(cur, indent=2, ensure_ascii=False))
    print(f"[저장] {MAP_PATH} ← {args.instruction}: {round(median_area, 4)}")
    print("⚠️ 운영 서버는 모듈 로드 시점에 1회만 읆음 — 반영하려면 서버 재시작 필요.")


if __name__ == "__main__":
    main()

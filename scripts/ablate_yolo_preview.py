"""
Stage 2 YOLO 오프라인 어블레이션 (CH54)

목적: 6/22~6/26 세션에서 YOLO가 탐지한 최대 객체의 cx가
      PG2가 그라운딩한 basket cx와 얼마나 일치하는지 측정.

측정값:
  - cx 일치율 (|yolo_cx - pg2_cx| < 0.2)
  - 방향 일치율 (left/center/right 3-bucket)
  - YOLO 탐지 성공률 (at least one box found)
  - 모델별 (yolov8n / yolov8s / yolov8m) 비교

실행:
  python3 scripts/ablate_yolo_preview.py [--models n s m] [--conf 0.3] [--seeds 3]
"""
import argparse, io, json, random, time
from pathlib import Path
from collections import defaultdict

import h5py
import numpy as np
from PIL import Image

SESSIONS_DIR = Path("docs/inference_sessions")
RESULTS_DIR  = Path("docs/v5/ch53_session_analysis.json").parent


def cx_bucket(cx: float) -> str:
    if cx < 0.4:  return "left"
    if cx > 0.6:  return "right"
    return "center"


def load_frames_with_pg2(session_path: Path):
    """세션에서 (image_rgb, pg2_cx, pg2_area, has_bbox) 튜플 리스트 반환"""
    frames = []
    with h5py.File(session_path, "r") as f:
        if "grounding/bbox" not in f:
            return frames
        bbox   = f["grounding/bbox"][:]   # (N, 4): cx, cy, area, has_bbox
        images = f["observations"]["images"][:] if "observations" in f else None
        if images is None and "images" in f:
            images = f["images"][:]
        if images is None:
            return frames
        n = min(len(bbox), len(images))
        for t in range(n):
            raw = images[t]
            try:
                if raw.ndim == 1:  # JPEG bytes
                    img = Image.open(io.BytesIO(bytes(raw))).convert("RGB")
                else:
                    img = Image.fromarray(raw.astype(np.uint8)).convert("RGB")
            except Exception:
                continue
            cx_v, cy_v, area_v, has_v = float(bbox[t,0]), float(bbox[t,1]), float(bbox[t,2]), float(bbox[t,3])
            frames.append((np.array(img), cx_v, area_v, has_v > 0.5))
    return frames


def run_yolo_ablation(model_size: str, conf: float, frames_all: list):
    from ultralytics import YOLO
    print(f"\n  [YOLO yolov8{model_size}] conf={conf} 로딩...")
    t0 = time.time()
    model = YOLO(f"yolov8{model_size}.pt")
    print(f"  로딩 완료 {time.time()-t0:.1f}s")

    results_by_frame = []
    for img_rgb, pg2_cx, pg2_area, pg2_has in frames_all:
        res = model(img_rgb, conf=conf, verbose=False)[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            results_by_frame.append({
                "yolo_detected": False, "yolo_cx": None,
                "pg2_cx": pg2_cx, "pg2_has": pg2_has, "pg2_area": pg2_area
            })
            continue
        # 최대 면적 박스 선택
        xyxy_list = [b.xyxy[0].cpu().numpy() for b in boxes]
        areas = [(x[2]-x[0])*(x[3]-x[1]) for x in xyxy_list]
        best = xyxy_list[int(np.argmax(areas))]
        W = img_rgb.shape[1]
        yolo_cx = float((best[0] + best[2]) / 2) / W
        results_by_frame.append({
            "yolo_detected": True, "yolo_cx": yolo_cx,
            "pg2_cx": pg2_cx, "pg2_has": pg2_has, "pg2_area": pg2_area
        })
    return results_by_frame


def compute_metrics(results, cx_tol=0.2):
    """PG2 has_bbox=True 인 프레임에서만 비교"""
    valid = [r for r in results if r["pg2_has"]]
    if not valid:
        return {}
    detected   = [r for r in valid if r["yolo_detected"]]
    matched_cx = [r for r in detected if abs(r["yolo_cx"] - r["pg2_cx"]) < cx_tol]
    bucket_match = [r for r in detected
                    if cx_bucket(r["yolo_cx"]) == cx_bucket(r["pg2_cx"])]
    return {
        "n_valid_pg2":   len(valid),
        "yolo_det_rate": len(detected)   / len(valid),
        "cx_match_rate": len(matched_cx) / len(valid),
        "dir_match_rate":len(bucket_match)/ len(valid),
        "cx_match_rate_cond": len(matched_cx)/len(detected) if detected else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["n", "s"], help="yolov8 sizes: n s m")
    parser.add_argument("--conf",   type=float, default=0.3)
    parser.add_argument("--seeds",  type=int,   default=3,  help="랜덤 시드 반복 횟수")
    parser.add_argument("--max-frames", type=int, default=None, help="세션당 최대 프레임 수 (속도 테스트용)")
    args = parser.parse_args()

    # 사용할 날짜 필터
    date_filters = ["20260622", "20260624", "20260626"]
    sessions = []
    for d in date_filters:
        sessions.extend(sorted(SESSIONS_DIR.glob(f"session_{d}*.h5")))
    print(f"세션 {len(sessions)}개 로드 ({', '.join(date_filters)})")

    # 전체 프레임 수집
    frames_all = []
    frame_meta = []  # (date, session_name)
    for sp in sessions:
        date = sp.stem.split("_")[1]
        frs  = load_frames_with_pg2(sp)
        if args.max_frames:
            frs = frs[:args.max_frames]
        for fr in frs:
            frames_all.append(fr)
            frame_meta.append((date, sp.stem))
    print(f"총 프레임 {len(frames_all)}개  (PG2 has_bbox>0: {sum(1 for f in frames_all if f[3])}개)")

    # 시드별로 프레임 샘플 섞어서 결과 안정성 확인
    all_results = {}
    for model_size in args.models:
        seed_metrics = []
        for seed in range(args.seeds):
            random.seed(seed)
            shuffled = list(zip(frames_all, frame_meta))
            random.shuffle(shuffled)
            frames_s, meta_s = zip(*shuffled)
            res = run_yolo_ablation(model_size, args.conf, list(frames_s))
            m   = compute_metrics(res)
            seed_metrics.append(m)
            print(f"  seed={seed} | det={m.get('yolo_det_rate',0):.1%}  cx={m.get('cx_match_rate',0):.1%}  dir={m.get('dir_match_rate',0):.1%}")

        # 평균 / 표준편차
        for key in ["yolo_det_rate", "cx_match_rate", "dir_match_rate", "cx_match_rate_cond"]:
            vals = [m.get(key, 0) for m in seed_metrics]
            print(f"  [{model_size}] {key}: {np.mean(vals):.1%} ± {np.std(vals):.1%}")
        all_results[f"yolov8{model_size}"] = {
            "conf": args.conf,
            "seeds": args.seeds,
            "metrics": seed_metrics,
        }

    # 날짜별 분석 (frame 0만 추출)
    print("\n=== 날짜별 frame 0 분포 ===")
    frame0_by_date = defaultdict(list)
    for sp in sessions:
        date = sp.stem.split("_")[1]
        with h5py.File(sp, "r") as f:
            if "grounding/bbox" not in f: continue
            b = f["grounding/bbox"][0]
            frame0_by_date[date].append(float(b[3]))
    for d, vals in sorted(frame0_by_date.items()):
        print(f"  {d}: frame0_has={np.mean(vals):.0%}  n={len(vals)}")

    # 결과 저장
    out = RESULTS_DIR / "ablate_yolo_preview.json"
    with open(out, "w") as fp:
        json.dump(all_results, fp, indent=2)
    print(f"\n결과 저장: {out}")


if __name__ == "__main__":
    main()

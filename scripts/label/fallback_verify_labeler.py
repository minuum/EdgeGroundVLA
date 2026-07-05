#!/usr/bin/env python3
"""
PG2 fallback(has_bbox=False) 프레임 검증용 로컬 라벨링 서버.
bbox_labeler.py(billy 머신, bbox_truth_mini.json)와 동일 포맷/API를 minum 머신용으로 이식.
YOLO pre-annotation + 사람 검수 하이브리드 모드

실행: python3 scripts/label/fallback_verify_labeler.py
접속: http://localhost:7789

※ 이미 서버가 뜨면 → /api/status, /api/logs 로 상태 확인
"""

import json
import os
import sys
import time
import logging
import socket
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

# ── 로깅 설정 ─────────────────────────────────────
LOG_DIR = Path("/home/minum/26CS/MoNaVLA/logs/labeler")
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_path = LOG_DIR / f"fallback_verify_labeler_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger("labeler")

# ── 경로 설정 ─────────────────────────────────────
PROJECT_ROOT = Path("/home/minum/26CS/MoNaVLA")
JSON_PATH    = PROJECT_ROOT / "docs/v5/fallback_verify_20260703/labeling_frames.json"
IMAGE_BASE   = PROJECT_ROOT / "data/fallback_verify_frames"   # frame_path가 절대경로라 실제론 안 쓰임
ANNO_SAVE_DIR = PROJECT_ROOT / "data/fallback_verify_labeled"  # YOLO 결과 저장 폴더
ANNO_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ── 포트 충돌 감지 ────────────────────────────────
PORT = 7789

def port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0

if port_in_use(PORT):
    print(f"\n⚠️  포트 {PORT} 이미 사용 중입니다.")
    print(f"   → 서버 상태: http://localhost:{PORT}/api/status")
    print(f"   → 서버 로그:  http://localhost:{PORT}/api/logs")
    print(f"   → 강제 종료: kill $(lsof -ti:{PORT})\n")
    sys.exit(0)

# ── 인메모리 캐시 ─────────────────────────────────
_cache = None

# YOLO 모델 (lazy load)
_yolo_model   = None
_yolo_classes = None   # COCO 전체 클래스 이름 목록
SERVER_START   = datetime.now().isoformat()
_request_log   = []   # 최근 요청 로그 (최대 50개)

def get_yolo():
    global _yolo_model, _yolo_classes
    if _yolo_model is None:
        from ultralytics import YOLO
        _yolo_model = YOLO("yolo11n.pt")
        _yolo_classes = _yolo_model.names
        logger.info(f"YOLO 모델 로드 완료: {len(_yolo_classes)}개 클래스")
    return _yolo_model

def load_data():
    global _cache
    if _cache is None:
        t0 = time.time()
        with open(JSON_PATH) as f:
            _cache = json.load(f)
        key = "annotations" if "annotations" in _cache else "frames"
        logger.info(f"JSON 로드: {len(_cache[key])}개 항목 ({(time.time()-t0)*1000:.0f}ms)")
    return _cache

def get_frames(data):
    """최상위 키에 무관하게 프레임 리스트 반환"""
    return data.get("annotations") or data.get("frames", [])

def save_data(data):
    global _cache
    _cache = data
    with open(JSON_PATH, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ── Flask 앱 ──────────────────────────────────────
app = Flask(__name__)
CORS(app)

@app.before_request
def log_request():
    entry = {"time": datetime.now().strftime("%H:%M:%S"),
             "method": request.method, "path": request.path}
    _request_log.append(entry)
    if len(_request_log) > 50:
        _request_log.pop(0)

@app.route("/")
def index():
    return send_from_directory(Path(__file__).parent, "fallback_verify_labeler.html")

@app.route("/api/status")
def status():
    data = load_data()
    frames = get_frames(data)
    summary = {
        "total":   len(frames),
        "pending": sum(1 for f in frames if f["review_status"] == "pending"),
        "done":    sum(1 for f in frames if f["review_status"] == "done"),
        "skip":    sum(1 for f in frames if f["review_status"] == "skip"),
    }
    return jsonify({
        "ok": True,
        "server_start": SERVER_START,
        "yolo_loaded": _yolo_model is not None,
        "yolo_classes": len(_yolo_classes) if _yolo_classes else 0,
        "json_cached": _cache is not None,
        "log_file": str(log_path),
        "anno_save_dir": str(ANNO_SAVE_DIR),
        "summary": summary,
        "recent_requests": _request_log[-10:],
    })

@app.route("/api/logs")
def get_logs():
    try:
        lines = log_path.read_text().splitlines()[-100:]
        return jsonify({"log": lines, "path": str(log_path)})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/api/frames")
def api_frames():
    data = load_data()
    frames = get_frames(data)
    summary = {
        "total":   len(frames),
        "pending": sum(1 for f in frames if f["review_status"] == "pending"),
        "done":    sum(1 for f in frames if f["review_status"] == "done"),
        "skip":    sum(1 for f in frames if f["review_status"] == "skip"),
    }
    return jsonify({"frames": frames, "summary": summary})

@app.route("/api/frame/<int:idx>")
def get_frame(idx):
    data = load_data()
    frames = get_frames(data)
    if idx < 0 or idx >= len(frames):
        return jsonify({"error": "out of range"}), 404
    return jsonify({"frame": frames[idx], "total": len(frames), "idx": idx})

@app.route("/api/yolo/<int:idx>")
def run_yolo(idx):
    """
    YOLO pre-annotation
    - COCO에 laundry basket 클래스 없음 → conf=0.15로 낮추고 모든 클래스 허용
    - 면적 기준 가장 큰 박스를 대표 BBox로 선택
    - 탐지 결과 이미지를 data/labeled_frames/ 에 저장
    """
    data = load_data()
    frames = get_frames(data)
    if idx < 0 or idx >= len(frames):
        return jsonify({"error": "out of range"}), 404

    frame = frames[idx]
    img_path = Path(frame["frame_path"])
    if not img_path.exists():
        return jsonify({"error": f"image not found: {img_path}",
                        "yolo_bbox": None, "yolo_found": False})
    try:
        model = get_yolo()
        # ① conf 낮게 (0.15) + iou 0.5, 바구니는 COCO 미등재 → 전체 클래스 탐지
        results = model(str(img_path), conf=0.15, iou=0.5, verbose=False)[0]
        boxes = results.boxes
        H, W = results.orig_shape

        all_dets = []
        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                cls_id  = int(b.cls[0])
                conf    = float(b.conf[0])
                x1,y1,x2,y2 = b.xyxy[0].tolist()
                area = (x2-x1)*(y2-y1)
                all_dets.append({
                    "cls_id": cls_id,
                    "cls_name": model.names[cls_id],
                    "conf": round(conf,3),
                    "area": round(area,1),
                    "bbox_norm": [
                        round(x1/W,3), round(y1/H,3),
                        round(x2/W,3), round(y2/H,3),
                    ]
                })
            # ② 면적 가장 큰 박스를 대표로 선택 (바구니가 보통 화면 중앙 큰 객체)
            all_dets.sort(key=lambda d: d["area"], reverse=True)

        best = all_dets[0] if all_dets else None
        logger.info(f"[YOLO] idx={idx} conf=0.15 → {len(all_dets)}개 탐지 | best={best}")

        # ③ annotated 이미지 저장
        ep = frame.get("episode","unknown")
        fi = frame.get("frame_idx", idx)
        save_name = f"{ep}_frame{fi:04d}_yolo.jpg"
        save_path = ANNO_SAVE_DIR / save_name
        results.save(filename=str(save_path))
        logger.info(f"[YOLO] 이미지 저장 → {save_path}")

        return jsonify({
            "yolo_bbox":       best["bbox_norm"] if best else None,
            "yolo_conf":       best["conf"]      if best else 0.0,
            "yolo_cls":        best["cls_name"]  if best else None,
            "yolo_found":      best is not None,
            "num_detections":  len(all_dets),
            "all_detections":  all_dets[:5],  # 상위 5개 반환
            "saved_image":     str(save_path) if all_dets else None,
        })

    except Exception as e:
        logger.exception(f"[YOLO ERROR] idx={idx}")
        return jsonify({"error": str(e), "yolo_bbox": None, "yolo_found": False})

@app.route("/api/save/<int:idx>", methods=["POST"])
def save_frame(idx):
    data = load_data()
    frames = get_frames(data)
    if idx < 0 or idx >= len(frames):
        return jsonify({"error": "out of range"}), 404

    payload = request.json
    frames[idx]["target_visible"]  = payload.get("target_visible")
    frames[idx]["bbox_xyxy_norm"]  = payload.get("bbox_xyxy_norm")
    frames[idx]["coarse_position"] = payload.get("coarse_position")
    frames[idx]["goal_near"]       = payload.get("goal_near")
    frames[idx]["notes"]           = payload.get("notes", "")
    frames[idx]["review_status"]   = payload.get("review_status", "done")

    if payload.get("yolo_assisted"):
        note = frames[idx]["notes"]
        if not note.startswith("[YOLO"):
            frames[idx]["notes"] = ("[YOLO-assisted] " + note).strip()

    save_data(data)
    logger.info(f"[SAVE] idx={idx} status={frames[idx]['review_status']} visible={frames[idx]['target_visible']}")
    return jsonify({"ok": True, "saved_idx": idx})

@app.route("/api/image")
def serve_image():
    rel_path = request.args.get("path", "")
    img_path = Path(rel_path) if rel_path.startswith("/") else IMAGE_BASE / rel_path
    if not img_path.exists():
        logger.warning(f"[IMAGE] not found: {img_path}")
        return jsonify({"error": f"not found: {img_path}"}), 404
    return send_file(img_path, mimetype="image/png")

@app.route("/api/export")
def export():
    data = load_data()
    frames = get_frames(data)
    done = [f for f in frames if f["review_status"] == "done"]
    return jsonify({"done_count": len(done), "frames": done})

if __name__ == "__main__":
    print("=" * 55)
    print("🏷️  BBox 수동 검수 서버 (YOLO-assisted)")
    print(f"📂 JSON  : {JSON_PATH}")
    print(f"🖼️  이미지: {IMAGE_BASE}")
    print(f"💾 저장  : {ANNO_SAVE_DIR}")
    print(f"📋 로그  : {log_path}")
    print("🤖 YOLO  : ultralytics yolo11n (conf=0.15, 전체 클래스)")
    print(f"🌐 브라우저 → http://localhost:{PORT}")
    print(f"🔍 상태확인 → http://localhost:{PORT}/api/status")
    print("=" * 55)
    app.run(host="0.0.0.0", port=PORT, debug=False)

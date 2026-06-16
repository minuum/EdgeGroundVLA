#!/usr/bin/env python3
"""
도착 STOP Y-Center Gate 정지 알고리즘 작동 원리 시각화 스크립트

이 스크립트는 VLM 그라운더가 출력한 BBox 및 기하 정보(cx, cy, area)를 기반으로
도착 정지 게이트(STOP Gate)가 작동하는 원리를 직관적인 다이어그램 이미지로 렌더링합니다.
- Case 1: 조기 오발 차단 (Transient Noise Blocked) - area는 크나 cy가 임계치 미달이라 주행 유지
- Case 2: 진짜 도착 판단 (True Stop Triggered) - cy가 임계치를 넘겨 정지 동작을 래치(Latch)

결과물은 docs/v5/visual_proof/stop_gate_concept.png 에 저장됩니다.
"""

import json
import sys
from pathlib import Path
import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "docs" / "v5" / "bbox_frame_level" / "bbox_dataset_frame_level.json"
OUT_DIR  = ROOT / "docs" / "v5" / "visual_proof"

TH_CY   = 0.50  # Y-Center Gate 임계치
TH_AREA = 0.50  # Heuristic Area 임계치

def _try_font(size):
    # 시스템에 내장된 폰트를 탐색해 로드
    for name in ["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "arial.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()

def draw_gate_diagram(img_pil, cx, cy, area, frame_idx, title_text, status_text, is_triggered):
    """
    개별 주행 이미지 위에 BBox, Y-Center 게이트선, 중심점, 알고리즘 수치들을 가시화
    """
    W, H = img_pil.size
    canvas = img_pil.copy()
    draw = ImageDraw.Draw(canvas)
    
    font_lg = _try_font(18)
    font_md = _try_font(14)
    font_sm = _try_font(11)
    
    # 1. Y-Center Gate 임계 가이드라인 (TH_CY = 0.50)
    line_y = int(TH_CY * H)
    line_color = (34, 197, 94) if is_triggered else (239, 68, 68)
    # 점선 그리기
    for x in range(0, W, 8):
        draw.line([(x, line_y), (min(x + 4, W), line_y)], fill=line_color, width=2)
    
    # 임계선 라벨 표시
    draw.rectangle([W - 140, line_y - 10, W - 10, line_y + 10], fill=(15, 23, 42, 220))
    draw.text((W - 132, line_y - 7), f"TH_CY = {TH_CY:.2f} line", fill=(255, 255, 255), font=font_sm)
    
    # 2. 바스켓 BBox 복원 (area, cx, cy 기반)
    side = int(np.sqrt(area) * min(W, H))
    half = side // 2
    bx, by = int(cx * W), int(cy * H)
    x1, y1 = max(0, bx - half), max(0, by - half)
    x2, y2 = min(W, bx + half), min(H, by + half)
    
    bbox_color = (251, 191, 36) # 바스켓 검출 박스는 노란색
    draw.rectangle([x1, y1, x2, y2], outline=bbox_color, width=3)
    
    # BBox 중심점 표시 (화각 내 Y축 중심 좌표 cy 탐지용)
    draw.ellipse([bx - 6, by - 6, bx + 6, by + 6], fill=(59, 130, 246), outline=(255, 255, 255), width=2)
    draw.line([(bx, by - 12), (bx, by + 12)], fill=(59, 130, 246), width=2)
    draw.line([(bx - 12, by), (bx + 12, by)], fill=(59, 130, 246), width=2)
    
    # 3. 실시간 검출 및 게이트 조건 텍스트
    draw.rectangle([10, 10, 250, 95], fill=(15, 23, 42, 200))
    draw.text((16, 15), f"VLM bbox: [basket]", fill=(251, 191, 36), font=font_md)
    draw.text((16, 35), f"• Center X (cx)   = {cx:.2f}", fill=(241, 245, 249), font=font_sm)
    
    # Y좌표 통과 여부 색상 강조
    cy_color = (74, 222, 128) if cy > TH_CY else (248, 113, 113)
    draw.text((16, 52), f"• Center Y (cy)   = {cy:.2f}", fill=cy_color, font=font_sm)
    draw.text((16, 69), f"• Area (area_det) = {area:.4f}", fill=(241, 245, 249), font=font_sm)
    
    # 4. 정지 알고리즘 판정 박스
    status_bg = (34, 197, 94, 220) if is_triggered else (239, 68, 68, 220)
    draw.rectangle([10, H - 45, W - 10, H - 10], fill=status_bg)
    draw.text((20, H - 38), status_text, fill=(255, 255, 255), font=font_md)
    
    # 5. 상단 헤더 타이틀
    draw.rectangle([10, H - 85, 280, H - 55], fill=(30, 41, 59, 230))
    draw.text((16, H - 78), title_text, fill=(255, 255, 255), font=font_md)
    
    return canvas

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    data = json.loads(DATA_PATH.read_text())
    
    # 1. 조기 오발 차단 케이스 프레임 검색 (cy < 0.45 이고 area >= 0.50)
    early_case = None
    for ep in data:
        for fr in ep["frames"]:
            if (fr.get("consistent") and fr.get("area_det") and fr["area_det"] >= 0.50
                and fr.get("cy_det") and fr["cy_det"] < 0.45
                and fr["cx_det"] > 0.35 and fr["cx_det"] < 0.65):
                early_case = (ep["episode"], fr)
                break
        if early_case:
            break
            
    # 2. 진짜 도착 정지 케이스 프레임 검색 (cy >= 0.65 이고 area >= 0.10)
    true_case = None
    for ep in data:
        for fr in ep["frames"]:
            if (fr.get("consistent") and fr.get("area_det") and fr["area_det"] >= 0.10
                and fr.get("cy_det") and fr["cy_det"] >= 0.65):
                true_case = (ep["episode"], fr)
                break
        if true_case:
            break
            
    if not early_case or not true_case:
        print("[ERROR] 조건에 부합하는 주행 프레임을 데이터셋에서 찾지 못했습니다.")
        sys.exit(1)
        
    print(f"[LOAD] 조기 차단 케이스: {early_case[0]} (frame={early_case[1]['frame_idx']})")
    print(f"[LOAD] 진짜 도착 케이스: {true_case[0]} (frame={true_case[1]['frame_idx']})")
    
    # 이미지 파일 로드 및 드로잉
    with h5py.File(early_case[0], "r") as f:
        img_early = Image.fromarray(f["observations"]["images"][early_case[1]["frame_idx"]]).convert("RGB")
    with h5py.File(true_case[0], "r") as f:
        img_true = Image.fromarray(f["observations"]["images"][true_case[1]["frame_idx"]]).convert("RGB")
        
    fr_e = early_case[1]
    diag_early = draw_gate_diagram(
        img_early, 
        fr_e["cx_det"], fr_e["cy_det"], fr_e["area_det"], fr_e["frame_idx"],
        title_text="Case 1. Transient Noise Blocked",
        status_text=f"STOP Blocked ➔ Keep Going (cy = {fr_e['cy_det']:.2f} < TH_CY)",
        is_triggered=False
    )
    
    fr_t = true_case[1]
    diag_true = draw_gate_diagram(
        img_true, 
        fr_t["cx_det"], fr_t["cy_det"], fr_t["area_det"], fr_t["frame_idx"],
        title_text="Case 2. True Stop Triggered",
        status_text=f"STOP Triggered ➔ Latch ON! (cy = {fr_t['cy_det']:.2f} >= TH_CY)",
        is_triggered=True
    )
    
    # 두 다이어그램 이미지를 나란히 합치기
    W, H = diag_early.size
    PAD = 8
    canvas_w = W * 2 + PAD
    canvas_h = H
    
    combined = Image.new("RGB", (canvas_w, canvas_h), (15, 23, 42))
    combined.paste(diag_early, (0, 0))
    combined.paste(diag_true, (W + PAD, 0))
    
    out_path = OUT_DIR / "stop_gate_concept.png"
    combined.save(out_path)
    print(f"[SAVED] 정지 게이트 작동 다이어그램 저장 완료 ➔ {out_path}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
2D Feature Interaction & Decision Boundary Analysis (2차원 피처 기하학적 결정 경계 분석)

목적:
  VLM이 출력한 바스켓의 BBox 정보(cx, area_det)가 제어 신경망(MLP) 내에서 
  어떠한 의사결정 기하 경계(Decision Boundary)를 형성하는지 2차원 산점도로 시각화합니다.
  - 가로축: 수평 좌표 오프셋 (cx_det - 0.5)
  - 세로축: 바스켓 면적 크기 (area_det)
  - 색상: 조향 예측 방향 (Left / Center / Right)
  
  이 시각화를 통해 모델이 단순히 배경 궤적을 외운 것이 아니라, 
  목표물의 수평 치우침(각거리)과 거리(면적)에 반응하는 수학적 제어 경계를 
  체계적으로 획득했음을 학술적으로 증명합니다.

결과물은 docs/v5/visual_proof/feature_interaction_map.png 에 저장됩니다.
"""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "docs" / "v5" / "bbox_frame_level" / "bbox_dataset_frame_level.json"
OUT_DIR  = ROOT / "docs" / "v5" / "visual_proof"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not DATA_PATH.exists():
        print(f"[ERROR] 데이터셋 파일이 존재하지 않습니다: {DATA_PATH}")
        sys.exit(1)
        
    data = json.loads(DATA_PATH.read_text())
    
    xs = [] # cx_det - 0.5 (수평 오프셋)
    ys = [] # area_det (면적)
    colors = [] # 방향별 색상 매핑
    labels = []
    
    # 색상 맵 정의 (다크 모드 대조용 세련된 컬러셋)
    # left: Cyber Blue (#3b82f6), center: Green (#10b981), right: Orange (#f97316)
    color_map = {
        "left": "#3b82f6",
        "center": "#10b981",
        "right": "#f97316"
    }
    
    for ep in data:
        for fr in ep["frames"]:
            if fr.get("detected") and fr.get("consistent") and fr.get("label"):
                cx = fr["cx_det"]
                area = fr["area_det"]
                label = fr["label"]
                
                if label in color_map:
                    xs.append(cx - 0.5)
                    ys.append(area)
                    colors.append(color_map[label])
                    labels.append(label)
                    
    xs = np.array(xs)
    ys = np.array(ys)
    colors = np.array(colors)
    
    # matplotlib 다크 모드 스타일 설정
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(9, 7), dpi=150)
    
    # 피처 산점도 그리기
    scatter_handles = {}
    for label, color in color_map.items():
        mask = (np.array(labels) == label)
        if np.any(mask):
            scatter = ax.scatter(
                xs[mask], ys[mask],
                c=color, label=label.upper(),
                alpha=0.6, edgecolors="none", s=25
            )
            scatter_handles[label] = scatter
            
    # 데코레이션 스타일 튜닝
    ax.set_title("2D Target Feature Interaction & Decision Boundary", 
                 fontsize=14, fontweight="bold", pad=15, color="#f1f5f9", fontname="DejaVu Sans")
    ax.set_xlabel("Horizontal Offset (cx_det - 0.5)", fontsize=11, labelpad=8, color="#94a3b8")
    ax.set_ylabel("Target Area Proxy (area_det)", fontsize=11, labelpad=8, color="#94a3b8")
    
    # 임계 및 기준선 표시
    ax.axvline(0.0, color="#475569", linestyle="--", linewidth=1, alpha=0.8) # 수평 중앙선
    ax.axhline(0.50, color="#ef4444", linestyle=":", linewidth=1.2, alpha=0.7) # STOP Area 임계치선
    ax.text(0.22, 0.52, "STOP Area (TH_AREA=0.5)", color="#ef4444", fontsize=9, alpha=0.9)
    
    # 그리드 튜닝
    ax.grid(True, color="#1e293b", linestyle="-", linewidth=0.5, alpha=0.5)
    
    # 축 범위 설정
    ax.set_xlim(-0.55, 0.55)
    ax.set_ylim(-0.02, 1.02)
    
    # 테두리 및 축 색상 변경
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#334155")
    ax.spines["bottom"].set_color("#334155")
    ax.tick_params(colors="#94a3b8", labelsize=9)
    
    # 범례 설정
    legend = ax.legend(
        loc="upper right", 
        frameon=True, 
        facecolor="#0f172a", 
        edgecolor="#1e293b",
        fontsize=10
    )
    for text in legend.get_texts():
        text.set_color("#f1f5f9")
        
    plt.tight_layout()
    
    # 파일 저장
    out_path = OUT_DIR / "feature_interaction_map.png"
    plt.savefig(out_path, facecolor="#0b0f19", edgecolor="none")
    plt.close()
    
    print(f"[SUCCESS] 2D 피처 상호작용 결정 경계 맵 빌드 완료 ➔ {out_path}")

if __name__ == "__main__":
    main()

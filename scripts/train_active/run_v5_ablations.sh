#!/bin/bash
# v5 LoRA Ablation Study 자동화 순차 구동 스크립트
set -e
mkdir -p logs/v5_ablation

# 이미 완료된 실험은 중복 실행을 막기 위해 주석 처리합니다.
# echo "=================================================="
# echo "시작: v5_ablation_top2_proj_frozen"
# echo "=================================================="
# .venv/bin/python3 third_party/RoboVLMs/main.py configs/v5_ablation/v5_ablation_top2_proj_frozen.json > logs/v5_ablation/v5_ablation_top2_proj_frozen.log 2>&1
# echo "완료: v5_ablation_top2_proj_frozen"

# echo "=================================================="
# echo "시작: v5_ablation_top2_proj_tuned"
# echo "=================================================="
# .venv/bin/python3 third_party/RoboVLMs/main.py configs/v5_ablation/v5_ablation_top2_proj_tuned.json > logs/v5_ablation/v5_ablation_top2_proj_tuned.log 2>&1
# echo "완료: v5_ablation_top2_proj_tuned"

# echo "=================================================="
# echo "시작: v5_ablation_top4_proj_frozen"
# echo "=================================================="
# .venv/bin/python3 third_party/RoboVLMs/main.py configs/v5_ablation/v5_ablation_top4_proj_frozen.json > logs/v5_ablation/v5_ablation_top4_proj_frozen.log 2>&1
# echo "완료: v5_ablation_top4_proj_frozen"

# echo "=================================================="
# echo "시작: v5_ablation_top4_proj_tuned"
# echo "=================================================="
# .venv/bin/python3 third_party/RoboVLMs/main.py configs/v5_ablation/v5_ablation_top4_proj_tuned.json > logs/v5_ablation/v5_ablation_top4_proj_tuned.log 2>&1
# echo "완료: v5_ablation_top4_proj_tuned"

# echo "=================================================="
# echo "시작: v5_ablation_top6_proj_frozen"
# echo "=================================================="
# .venv/bin/python3 third_party/RoboVLMs/main.py configs/v5_ablation/v5_ablation_top6_proj_frozen.json > logs/v5_ablation/v5_ablation_top6_proj_frozen.log 2>&1
# echo "완료: v5_ablation_top6_proj_frozen"

# echo "=================================================="
# echo "시작: v5_ablation_top6_proj_tuned"
# echo "=================================================="
# .venv/bin/python3 third_party/RoboVLMs/main.py configs/v5_ablation/v5_ablation_top6_proj_tuned.json > logs/v5_ablation/v5_ablation_top6_proj_tuned.log 2>&1
# echo "완료: v5_ablation_top6_proj_tuned"

# echo "=================================================="
# echo "시작: v5_ablation_top8_proj_frozen"
# echo "=================================================="
# .venv/bin/python3 third_party/RoboVLMs/main.py configs/v5_ablation/v5_ablation_top8_proj_frozen.json > logs/v5_ablation/v5_ablation_top8_proj_frozen.log 2>&1
# echo "완료: v5_ablation_top8_proj_frozen"

echo "=================================================="
echo "시작: v5_ablation_top8_proj_tuned (resume from last.ckpt)"
echo "=================================================="
.venv/bin/python3 third_party/RoboVLMs/main.py configs/v5_ablation/v5_ablation_top8_proj_tuned.json > logs/v5_ablation/v5_ablation_top8_proj_tuned.log 2>&1
echo "완료: v5_ablation_top8_proj_tuned"


#!/bin/bash
# v5 LoRA Ablation Study 자동화 순차 구동 스크립트
set -e
mkdir -p logs/v5_ablation

echo "=================================================="
echo "시작: v5_ablation_top2_proj_frozen"
echo "=================================================="
.venv/bin/python3 third_party/RoboVLMs/main.py configs/v5_ablation/v5_ablation_top2_proj_frozen.json > logs/v5_ablation/v5_ablation_top2_proj_frozen.log 2>&1
echo "완료: v5_ablation_top2_proj_frozen"

echo "=================================================="
echo "시작: v5_ablation_top2_proj_tuned"
echo "=================================================="
.venv/bin/python3 third_party/RoboVLMs/main.py configs/v5_ablation/v5_ablation_top2_proj_tuned.json > logs/v5_ablation/v5_ablation_top2_proj_tuned.log 2>&1
echo "완료: v5_ablation_top2_proj_tuned"

echo "=================================================="
echo "시작: v5_ablation_top4_proj_frozen"
echo "=================================================="
.venv/bin/python3 third_party/RoboVLMs/main.py configs/v5_ablation/v5_ablation_top4_proj_frozen.json > logs/v5_ablation/v5_ablation_top4_proj_frozen.log 2>&1
echo "완료: v5_ablation_top4_proj_frozen"

echo "=================================================="
echo "시작: v5_ablation_top4_proj_tuned"
echo "=================================================="
.venv/bin/python3 third_party/RoboVLMs/main.py configs/v5_ablation/v5_ablation_top4_proj_tuned.json > logs/v5_ablation/v5_ablation_top4_proj_tuned.log 2>&1
echo "완료: v5_ablation_top4_proj_tuned"

echo "=================================================="
echo "시작: v5_ablation_top6_proj_frozen"
echo "=================================================="
.venv/bin/python3 third_party/RoboVLMs/main.py configs/v5_ablation/v5_ablation_top6_proj_frozen.json > logs/v5_ablation/v5_ablation_top6_proj_frozen.log 2>&1
echo "완료: v5_ablation_top6_proj_frozen"

echo "=================================================="
echo "시작: v5_ablation_top6_proj_tuned"
echo "=================================================="
.venv/bin/python3 third_party/RoboVLMs/main.py configs/v5_ablation/v5_ablation_top6_proj_tuned.json > logs/v5_ablation/v5_ablation_top6_proj_tuned.log 2>&1
echo "완료: v5_ablation_top6_proj_tuned"

echo "=================================================="
echo "시작: v5_ablation_top8_proj_frozen"
echo "=================================================="
.venv/bin/python3 third_party/RoboVLMs/main.py configs/v5_ablation/v5_ablation_top8_proj_frozen.json > logs/v5_ablation/v5_ablation_top8_proj_frozen.log 2>&1
echo "완료: v5_ablation_top8_proj_frozen"

echo "=================================================="
echo "시작: v5_ablation_top8_proj_tuned"
echo "=================================================="
.venv/bin/python3 third_party/RoboVLMs/main.py configs/v5_ablation/v5_ablation_top8_proj_tuned.json > logs/v5_ablation/v5_ablation_top8_proj_tuned.log 2>&1
echo "완료: v5_ablation_top8_proj_tuned"


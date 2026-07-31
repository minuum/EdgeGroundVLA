#!/bin/bash
# exp73 파이프라인: PG448 주석 대기 → OWL 주석 → 학습(pg448 2-arm × 3-head × 3-seed) → 학습(owl v6 arm)
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python3

echo "[1/4] PG448 주석 실행..."
if pgrep -f "gen_v6_pg448_annotation.py --src" > /dev/null; then
    echo "  이미 실행 중인 프로세스 대기..."
    while pgrep -f "gen_v6_pg448_annotation.py --src" > /dev/null; do sleep 30; done
else
    $PY scripts/gen_v6_pg448_annotation.py \
        --src docs/v5/bbox_frame_level/bbox_dataset_v6_frame_level.json \
        --out docs/v5/bbox_frame_level/bbox_dataset_v6_pg448_cx.json \
        --stride 3 --batch 64 --resume 2>&1 | tee logs/gen_v6_pg448.log
fi
if ! grep -q "완료: LIVE" logs/gen_v6_pg448.log; then
    echo "FATAL: PG448 주석이 정상 종료되지 않음"; exit 1
fi
echo "  PG448 주석 완료"

echo "[2/4] OWL-v2 주석 생성..."
$PY scripts/gen_v6_owl_annotation.py \
    --src docs/v5/bbox_frame_level/bbox_dataset_v6_frame_level.json \
    --out docs/v5/bbox_nav_owl/bbox_dataset_v6_owl.json \
    --stride 3 --batch 16

echo "[3/4] 학습: pg448 (v6, v6v5 × transformer,mlp,cxgeom × seed 0,1,2)..."
$PY scripts/train_exp73_trackA_heads.py --tag pg448

echo "[4/4] 학습: owl (v6 arm — 그라운더 ablation)..."
$PY scripts/train_exp73_trackA_heads.py --tag owl \
    --ann-v6 docs/v5/bbox_nav_owl/bbox_dataset_v6_owl.json --arms v6

echo "PIPELINE_DONE"

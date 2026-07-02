#!/usr/bin/env bash
# exp71 Transformer WINDOW ablation: 4 / 6 / 8 / 12 / 16
# WINDOW=8은 이미 seed0 결과 있음 → 나머지만 학습
# 학습 후 인라인 CL eval (eval_multiseed_cl.py와 같은 로직)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv/bin/python3"
LOG_DIR="$ROOT/logs/window_ablation"
mkdir -p "$LOG_DIR"

WINDOWS=(4 6 12 16)
SEED=42

echo "=============================================="
echo " exp71 Window Ablation: ${WINDOWS[*]}  (+ 기존 WINDOW=8)"
echo "=============================================="

# exp71 학습 스크립트에 --window 파라미터 추가 여부 확인 후 학습
for W in "${WINDOWS[@]}"; do
  OUT="$ROOT/runs/v5_nav/mlp/exp71_window${W}"
  LOG="$LOG_DIR/train_window${W}.log"
  echo ""
  echo "[TRAIN] exp71 Transformer  WINDOW=$W  seed=$SEED  → $OUT"
  $VENV scripts/train_exp71_stage2_transformer.py \
    --window  "$W" \
    --seed    "$SEED" \
    --out-dir "$OUT" \
    2>&1 | tee "$LOG" | grep -E "BEST|val_acc|epoch 300|결과"
done

echo ""
echo "=============================================="
echo " Window Ablation CL Evaluation"
echo "=============================================="

$VENV scripts/eval_window_cl.py 2>&1 | tee "$LOG_DIR/cl_eval.log"

echo ""
echo "[DONE] 로그: $LOG_DIR"

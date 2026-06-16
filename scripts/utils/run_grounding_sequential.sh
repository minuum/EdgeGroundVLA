#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
VENV="/home/minum/26CS/MoNa-pi/.venv/bin/python"
LOG_DIR="docs/v5/grounding_comparison"
mkdir -p $LOG_DIR

models=("moondream" "paligemma-mix" "paligemma2-mix")

for model in "${models[@]}"; do
    echo "=== Starting Analysis: $model ==="
    $VENV scripts/test_pretrained_vlm_grounding_compat.py --model "$model" --num_samples 20 > "$LOG_DIR/log_$model.txt" 2>&1
    echo "=== Finished Analysis: $model ==="
done
echo "🎉 All Sequential Grounding Analysis Completed!"

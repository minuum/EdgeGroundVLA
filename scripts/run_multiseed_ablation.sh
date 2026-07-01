#!/usr/bin/env bash
# exp67(MLP) / exp71(Transformer) / exp72(cx-Geom) × 5 seeds 순차 학습
# 결과: runs/v5_nav/mlp/exp{67,71,72}_seed{0,1,2,3,4}/
# 완료 후: scripts/collect_multiseed_results.py 로 mean±std 집계

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SEEDS=(0 1 2 3 4)
LOG_DIR="$ROOT/logs/multiseed_ablation"
mkdir -p "$LOG_DIR"

VENV="$ROOT/.venv/bin/python3"
TOTAL=$(( ${#SEEDS[@]} * 3 ))
DONE=0

echo "======================================================"
echo " Multi-seed ablation: exp67 / exp71 / exp72 × 5 seeds"
echo " Total runs: $TOTAL"
echo " Logs: $LOG_DIR"
echo "======================================================"

for SEED in "${SEEDS[@]}"; do

  # ── exp67 MLP ─────────────────────────────────────────────
  OUT="$ROOT/runs/v5_nav/mlp/exp67_seed${SEED}"
  LOG="$LOG_DIR/exp67_seed${SEED}.log"
  DONE=$((DONE+1))
  echo ""
  echo "[$DONE/$TOTAL] exp67 MLP  seed=$SEED  → $OUT"
  $VENV scripts/train_exp67_stage2_pg448.py \
    --seed "$SEED" \
    --out  "$OUT" \
    2>&1 | tee "$LOG" | grep -E "BEST|val_acc|epoch 300|결과"

  # ── exp71 Transformer ─────────────────────────────────────
  OUT="$ROOT/runs/v5_nav/mlp/exp71_seed${SEED}"
  LOG="$LOG_DIR/exp71_seed${SEED}.log"
  DONE=$((DONE+1))
  echo ""
  echo "[$DONE/$TOTAL] exp71 Transformer  seed=$SEED  → $OUT"
  $VENV scripts/train_exp71_stage2_transformer.py \
    --seed    "$SEED" \
    --out-dir "$OUT" \
    2>&1 | tee "$LOG" | grep -E "BEST|val_acc|epoch 300|결과"

  # ── exp72 cx-Geom ─────────────────────────────────────────
  OUT="$ROOT/runs/v5_nav/mlp/exp72_seed${SEED}"
  LOG="$LOG_DIR/exp72_seed${SEED}.log"
  DONE=$((DONE+1))
  echo ""
  echo "[$DONE/$TOTAL] exp72 cx-Geom  seed=$SEED  → $OUT"
  $VENV scripts/train_exp72_stage2_cxgeom.py \
    --seed    "$SEED" \
    --out-dir "$OUT" \
    2>&1 | tee "$LOG" | grep -E "BEST|val_acc|epoch 300|결과"

done

echo ""
echo "======================================================"
echo " 전체 완료! 결과 집계:"
echo "======================================================"

# 인라인 집계
$VENV - <<'PYEOF'
import torch, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent if '__file__' in dir() else Path('.')
ROOT = Path('.')

results = {}
for exp, fname in [("exp67","action_mlp.pt"), ("exp71","action_transformer.pt"), ("exp72","action_cxgeom.pt")]:
    accs = []
    for seed in range(5):
        pt = ROOT / f"runs/v5_nav/mlp/{exp}_seed{seed}" / fname
        if pt.exists():
            d = torch.load(str(pt), map_location="cpu", weights_only=False)
            accs.append(d.get("val_acc", 0) * 100)
    if accs:
        import statistics
        mean = sum(accs)/len(accs)
        std  = statistics.stdev(accs) if len(accs)>1 else 0
        results[exp] = {"seeds": len(accs), "mean": mean, "std": std, "all": accs}
        print(f"  {exp}: {mean:.1f}±{std:.1f}%  ({len(accs)} seeds)  vals={[f'{a:.1f}' for a in accs]}")

out_path = ROOT / "logs/multiseed_ablation/summary.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  → 요약 저장: {out_path}")
PYEOF

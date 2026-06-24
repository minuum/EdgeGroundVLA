#!/bin/bash
# CH50 줌 재그라운딩(zoomsmall) 결과의 5-seed 노이즈 검증.
# train_hidden_state_lstm.py에 명시적 seed 인자가 없으므로(원본 CH43-2d 관행과 동일하게)
# 매 실행마다 torch/np 기본 RNG 상태가 달라지는 자연스러운 stochasticity를 5회 반복해 평균±표준편차로 본다.
set -e
cd "$(dirname "$0")/../.."

CKPT_DIR=runs/v5_nav/mlp/exp_hidden_state/stage2_v2_lstm
DATA=docs/v5/bbox_nav_exp46/bbox_dataset_full_pg2_zoomsmall.json
OUT_DIR=docs/v5/closed_loop_eval/zoomsmall_5seed
mkdir -p "$OUT_DIR"

for i in 1 2 3 4 5; do
  echo "=== seed $i ==="
  .venv/bin/python3 scripts/train_hidden_state_lstm.py --use_hidden_state none --data "$DATA"
  mv "$CKPT_DIR/stage2_lstm_none.pt" "$CKPT_DIR/stage2_lstm_none_zoomsmall_seed${i}.pt"
  .venv/bin/python3 scripts/eval/closed_loop_eval_zoomsmall.py \
    --ckpt "$CKPT_DIR/stage2_lstm_none_zoomsmall_seed${i}.pt" \
    --out "$OUT_DIR/seed${i}.json"
done

echo "=== 5-seed 결과 요약 ==="
.venv/bin/python3 - <<'PYEOF'
import json
from pathlib import Path
srs, fpes = [], []
for i in range(1, 6):
    d = json.loads(Path(f"docs/v5/closed_loop_eval/zoomsmall_5seed/seed{i}.json").read_text())
    srs.append(d["none"]["sr"])
    fpes.append(d["none"]["fpe_mean"])
import statistics as st
print(f"SR:  mean={st.mean(srs)*100:.1f}%  std={st.pstdev(srs)*100:.1f}%p  raw={[f'{x*100:.1f}' for x in srs]}")
print(f"FPE: mean={st.mean(fpes):.3f}m  std={st.pstdev(fpes):.3f}m  raw={[f'{x:.3f}' for x in fpes]}")
PYEOF

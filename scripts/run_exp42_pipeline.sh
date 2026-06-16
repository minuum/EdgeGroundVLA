#!/usr/bin/env bash
# Exp42 Phase A 자동 파이프라인 (counterfactual_steer/stop)
# 학습 완료 감지 → attention → PM eval → prompt sensitivity → pass/fail 출력
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

EXP_DIR="runs/v5_nav/kosmos/mobile_vla_v5_exp42"
CONFIG="configs/mobile_vla_v5_exp42_counterfactual_pta.json"
VENV=".venv/bin/activate"
OUT_DIR="docs/v5/exp41_prompt_lockin"
mkdir -p "$OUT_DIR"

echo "[pipeline] Exp42 Phase A pipeline started: $(date)"

# ── 1. 학습 완료 대기 — train.py 프로세스가 죽을 때까지 ───
# Lightning은 매 epoch마다 last.ckpt를 갱신해서 size 안정성 체크는 false positive
# 발생 (epoch 0 직후 평가됨, 5/6 22:59 incident). 진짜 완료는 train.py PID 부재로 판단.
TRAIN_CONFIG_NAME="$(basename "$CONFIG")"
echo "[pipeline] Waiting for train.py with config $TRAIN_CONFIG_NAME to exit ..."
while true; do
    if ! pgrep -af "robovlm_nav/train.py.*$TRAIN_CONFIG_NAME" >/dev/null 2>&1; then
        # 프로세스가 없으면 학습 종료 (또는 시작 안 함)
        echo "[pipeline] train.py process gone."
        break
    fi
    sleep 120
done

# 학습 종료 후 last.ckpt 안정화 추가 대기 (마지막 flush)
sleep 30

CKPT=$(find "$EXP_DIR" -name "last.ckpt" 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    echo "[pipeline] ERROR: No last.ckpt found in $EXP_DIR" >&2
    exit 1
fi
echo "[pipeline] Training done. ckpt=$CKPT"

# best epoch ckpt (val_loss 기준 최솟값)
BEST_CKPT=$(find "$EXP_DIR" -name "epoch*.ckpt" 2>/dev/null \
    | sort -t= -k4 -n | head -1)
[ -z "$BEST_CKPT" ] && BEST_CKPT="$CKPT"
echo "[pipeline] Best ckpt: $BEST_CKPT"

source "$VENV"

# ── 2. Attention measurement ───────────────────────────────
echo "[pipeline] === Step 2: measure_attention ==="
python3 scripts/measure_attention.py
echo "[pipeline] Attention done."

# ── 3. PM eval ─────────────────────────────────────────────
echo "[pipeline] === Step 3: PM eval ==="
mkdir -p docs/v5/pm_eval
python3 scripts/test_v5_pm_dm.py \
    --ckpt "$BEST_CKPT" \
    --config "$CONFIG" \
    --instruction_preset path_type_aware \
    --eval_split val --eval_t 0 \
    --output_json docs/v5/pm_eval/exp42_results.json
echo "[pipeline] PM eval done."

# ── 4. Prompt sensitivity ──────────────────────────────────
echo "[pipeline] === Step 4: prompt sensitivity ==="
python3 scripts/eval_prompt_sensitivity.py \
    --ckpt "$BEST_CKPT" \
    --config "$CONFIG" \
    --n-frames 30 \
    --output_json "$OUT_DIR/exp42_sensitivity.json"
echo "[pipeline] Sensitivity done."

# ── 5. Pass/Fail 판정 ──────────────────────────────────────
echo "[pipeline] === Step 5: Phase A verdict ==="
python3 - <<'PY'
import json, sys

attn_file  = "docs/v5/attention_analysis/summary.json"
pm_file    = "docs/v5/pm_eval/exp42_results.json"
sens_file  = "docs/v5/exp41_prompt_lockin/exp42_sensitivity.json"

data_attn = json.load(open(attn_file))
key = "exp42_counterfactual_pta"
if key not in data_attn:
    print(f"[verdict] WARNING: {key} not in attention summary — skipping attention check")
    text_pct = None
else:
    by_p = data_attn[key]
    layers = by_p.get("forward", {}).get("per_layer", [])
    txt = [l["text_sum_mean"] for l in layers]
    text_pct = sum(txt) / len(txt) * 100 if txt else 0
    print(f"[verdict] text attention: {text_pct:.4f}% (goal ≥5%)")

pm_data = json.load(open(pm_file))
pm = pm_data.get("pm_rate", 0) * 100
print(f"[verdict] PM: {pm:.2f}% (goal ≥50%)")

sens_data = json.load(open(sens_file))
l1_lr = sens_data.get("mean_l1_left_vs_right", 0)
l1_avg = sens_data.get("mean_softmax_l1", 0)
pred_changes = sens_data.get("frames_with_pred_change", 0)
print(f"[verdict] action L1 (L↔R): {l1_lr:.5f} | mean softmax L1: {l1_avg:.5f} | pred changes: {pred_changes}/30")

criteria = {
    "text_attention_ge5pct": (text_pct or 0) >= 5.0,
    "action_l1_ge_1e2": l1_lr >= 1e-2,
    "pm_ge50": pm >= 50.0,
    "pred_changes_gt0": pred_changes > 0,
}
passed = sum(criteria.values())
total  = len(criteria)

print("\n=== Exp42 Phase A Result ===")
for k, v in criteria.items():
    print(f"  {'✅' if v else '❌'} {k}")

if passed == total:
    print(f"\nPHASE A: PASS ({passed}/{total}) → counterfactual succeeded; Exp42 deployment candidate")
elif passed >= 2:
    print(f"\nPHASE A: PARTIAL ({passed}/{total}) → adjust counterfactual ratios and retry")
else:
    print(f"\nPHASE A: FAIL ({passed}/{total}) → Track 3 (Phase D structural change)")

# JSON 저장
result = {
    "exp": "exp42_counterfactual_pta",
    "text_attention_pct": text_pct,
    "pm_rate": pm,
    "action_l1_lr": l1_lr,
    "action_l1_avg": l1_avg,
    "pred_changes": pred_changes,
    "criteria": criteria,
    "verdict": "PASS" if passed == total else ("PARTIAL" if passed >= 2 else "FAIL"),
}
out = "docs/v5/exp41_prompt_lockin/exp42_phase_a_verdict.json"
json.dump(result, open(out, "w"), indent=2)
print(f"\nSaved verdict: {out}")
PY

echo "[pipeline] Pipeline complete: $(date)"

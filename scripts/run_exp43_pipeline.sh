#!/usr/bin/env bash
# Exp43 Phase D 자동 파이프라인 (cross-attention text head)
# 학습 완료 감지 → attention → text_gate → PM eval → prompt sensitivity → pass/fail 출력
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

EXP_DIR="runs/v5_nav/kosmos/mobile_vla_v5_exp43"
CONFIG="configs/mobile_vla_v5_exp43_cross_attn_text.json"
VENV=".venv/bin/activate"
OUT_DIR="docs/v5/exp43_phase_d"
mkdir -p "$OUT_DIR"
mkdir -p docs/v5/pm_eval

echo "[pipeline] Exp43 Phase D pipeline started: $(date)"

# ── 1. 학습 완료 대기 ───────────────────────────────────────
TRAIN_CONFIG_NAME="$(basename "$CONFIG")"
echo "[pipeline] Waiting for train.py with config $TRAIN_CONFIG_NAME to exit ..."
while true; do
    if ! pgrep -af "robovlm_nav/train.py.*$TRAIN_CONFIG_NAME" >/dev/null 2>&1; then
        echo "[pipeline] train.py process gone."
        break
    fi
    sleep 120
done
sleep 30

CKPT=$(find "$EXP_DIR" -name "last.ckpt" 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    echo "[pipeline] ERROR: No last.ckpt found in $EXP_DIR" >&2
    exit 1
fi
echo "[pipeline] Training done. ckpt=$CKPT"

BEST_CKPT=$(find "$EXP_DIR" -name "epoch*.ckpt" 2>/dev/null \
    | sort -t= -k4 -n | head -1)
[ -z "$BEST_CKPT" ] && BEST_CKPT="$CKPT"
echo "[pipeline] Best ckpt: $BEST_CKPT"

source "$VENV"

# ── 2. text_gate 값 확인 (Phase D 핵심 지표) ───────────────
echo "[pipeline] === Step 2: text_gate magnitude ==="
python3 - <<PY
import torch, json
ckpt = torch.load("$BEST_CKPT", map_location="cpu", weights_only=False)
sd = ckpt.get("model_state_dict", ckpt.get("state_dict", {}))
gate_keys = [k for k in sd if "text_gate" in k]
if gate_keys:
    for k in gate_keys:
        v = sd[k].item()
        print(f"  {k} = {v:.6f}  (init=0.1, useful if |gate| >= 0.05)")
    gate_val = abs(sd[gate_keys[0]].item())
    print(f"  gate_useful: {gate_val >= 0.05}")
    json.dump({"gate_keys": gate_keys, "gate_val": sd[gate_keys[0]].item(), "gate_useful": gate_val >= 0.05},
              open("$OUT_DIR/text_gate.json", "w"), indent=2)
else:
    print("  WARNING: text_gate not found in checkpoint — cross-attn not loaded?")
PY

# ── 3. Attention measurement ───────────────────────────────
echo "[pipeline] === Step 3: measure_attention ==="
python3 scripts/measure_attention.py
echo "[pipeline] Attention done."

# ── 4. PM eval ─────────────────────────────────────────────
echo "[pipeline] === Step 4: PM eval ==="
python3 scripts/test_v5_pm_dm.py \
    --ckpt "$BEST_CKPT" \
    --config "$CONFIG" \
    --instruction_preset path_type_aware \
    --eval_split val --eval_t 0 \
    --output_json docs/v5/pm_eval/exp43_results.json
echo "[pipeline] PM eval done."

# ── 5. Prompt sensitivity ──────────────────────────────────
echo "[pipeline] === Step 5: prompt sensitivity ==="
python3 scripts/eval_prompt_sensitivity.py \
    --ckpt "$BEST_CKPT" \
    --config "$CONFIG" \
    --n-frames 30 \
    --output_json "$OUT_DIR/exp43_sensitivity.json"
echo "[pipeline] Sensitivity done."

# ── 6. Phase D Pass/Fail 판정 ──────────────────────────────
echo "[pipeline] === Step 6: Phase D verdict ==="
python3 - <<'PY'
import json, sys

attn_file  = "docs/v5/attention_analysis/summary.json"
pm_file    = "docs/v5/pm_eval/exp43_results.json"
sens_file  = "docs/v5/exp43_phase_d/exp43_sensitivity.json"
gate_file  = "docs/v5/exp43_phase_d/text_gate.json"

data_attn = json.load(open(attn_file))
key = "exp43_cross_attn_text"
if key not in data_attn:
    print(f"[verdict] WARNING: {key} not in attention summary — skipping attention check")
    text_pct = None
else:
    by_p = data_attn[key]
    layers = by_p.get("forward", {}).get("per_layer", [])
    txt = [l["text_sum_mean"] for l in layers]
    text_pct = sum(txt) / len(txt) * 100 if txt else 0
    print(f"[verdict] text attention: {text_pct:.4f}% (goal >= 5%)")

pm_data = json.load(open(pm_file))
pm = pm_data.get("pm_rate", 0) * 100
print(f"[verdict] PM: {pm:.2f}% (goal >= 50%)")

sens_data = json.load(open(sens_file))
l1_lr       = sens_data.get("mean_l1_left_vs_right", 0)
l1_avg      = sens_data.get("mean_softmax_l1", 0)
pred_changes = sens_data.get("frames_with_pred_change", 0)
print(f"[verdict] action L1 (L<->R): {l1_lr:.5f} | mean softmax L1: {l1_avg:.5f} | pred changes: {pred_changes}/30")

gate_data = json.load(open(gate_file)) if __import__("os").path.exists(gate_file) else {}
gate_val = abs(gate_data.get("gate_val", 0))
print(f"[verdict] text_gate magnitude: {gate_val:.6f} (goal >= 0.05)")

criteria = {
    "text_attention_ge5pct":  (text_pct or 0) >= 5.0,
    "action_l1_ge_1e2":       l1_lr >= 1e-2,
    "pred_changes_gt0":       pred_changes > 0,
    "pm_ge50":                pm >= 50.0,
    "gate_magnitude_ge_0_05": gate_val >= 0.05,
}
passed = sum(criteria.values())
total  = len(criteria)

print("\n=== Exp43 Phase D Result ===")
for k, v in criteria.items():
    print(f"  {'OK' if v else 'NG'} {k}")

if passed == total:
    verdict = "PASS"
    msg = "cross-attention text conditioning succeeded — deployment candidate"
elif passed >= 3:
    verdict = "PARTIAL"
    msg = "partial — tune gate init or num_heads (Exp43b/c)"
else:
    verdict = "FAIL"
    msg = "Phase E: backbone replacement (TICVLA / MobilityVLA)"

print(f"\nPHASE D: {verdict} ({passed}/{total}) — {msg}")

result = {
    "exp": "exp43_cross_attn_text",
    "text_attention_pct": text_pct,
    "pm_rate": pm,
    "action_l1_lr": l1_lr,
    "action_l1_avg": l1_avg,
    "pred_changes": pred_changes,
    "gate_val": gate_val,
    "criteria": criteria,
    "verdict": verdict,
}
out = "docs/v5/exp43_phase_d/exp43_phase_d_verdict.json"
json.dump(result, open(out, "w"), indent=2)
print(f"Saved verdict: {out}")
PY

echo "[pipeline] Pipeline complete: $(date)"

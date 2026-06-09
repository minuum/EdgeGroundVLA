#!/bin/bash
cd /home/minum/26CS/MoNaVLA
LOG=logs/v5_ablation/vlora_overall.log
mkdir -p logs/v5_ablation
CFGS=(top2_proj_tuned_vlora top4_proj_tuned_vlora top6_proj_tuned_vlora top8_proj_tuned_vlora)

check_lorab () {  # $1=exp → echo max|abs of vision lora_B from epoch0 ckpt
  .venv/bin/python3 - "$1" <<'PY'
import sys,glob,torch
exp=sys.argv[1]
cks=glob.glob(f"runs/mobile_vla_paligemma/v5_ablation_{exp}/**/epoch_*00*.ckpt",recursive=True)
if not cks: print("NOCKPT"); sys.exit()
sd=torch.load(cks[0],map_location="cpu",weights_only=False,mmap=True); sd=sd.get("state_dict",sd)
B=[v for k,v in sd.items() if "vision_tower" in k and "lora_B" in k]
print(f"{max(float(t.abs().max()) for t in B):.6f}" if B else "NOLORA")
PY
}

echo "[vlora] $(date) GATE: ${CFGS[0]} 먼저 학습 → epoch0 후 lora_B 검증" >> $LOG
nohup .venv/bin/python3 scripts/train_active/train_one_visionlora.py configs/v5_ablation/v5_ablation_${CFGS[0]}.json > logs/v5_ablation/vlora_${CFGS[0]}.log 2>&1 &
PID=$!
echo "[vlora] config1 PID $PID" >> $LOG

# epoch0 ckpt 대기 (최대 3시간)
for i in $(seq 1 180); do
  ck=$(check_lorab ${CFGS[0]})
  if [ "$ck" != "NOCKPT" ] && [ "$ck" != "NOLORA" ]; then
    echo "[vlora] $(date) epoch0 lora_B_max=$ck" >> $LOG
    awk "BEGIN{exit !($ck>0.000001)}" && PASS=1 || PASS=0
    break
  fi
  sleep 60
done

if [ "$PASS" != "1" ]; then
  echo "[vlora] $(date) ❌ GATE 실패 (lora_B=$ck) — vision LoRA 여전히 미학습. 중단." >> $LOG
  kill $PID 2>/dev/null
  exit 1
fi
echo "[vlora] ✅ GATE 통과 — config1 완주 대기 후 나머지 3개 진행" >> $LOG
wait $PID
echo "[vlora] 완료: ${CFGS[0]}" >> $LOG
for c in "${CFGS[@]:1}"; do
  echo "[vlora] $(date) 시작: $c" >> $LOG
  .venv/bin/python3 scripts/train_active/train_one_visionlora.py configs/v5_ablation/v5_ablation_${c}.json > logs/v5_ablation/vlora_${c}.log 2>&1
  echo "[vlora] 완료: $c" >> $LOG
done
echo "[vlora] $(date) ALL DONE" >> $LOG

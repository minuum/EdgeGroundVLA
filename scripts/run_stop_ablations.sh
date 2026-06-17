#!/usr/bin/env bash
# ============================================================
# STOP Ablation Runner — L1 + L2 (eval) + L3 (train+eval)
# 병렬 비동기 실행, 로그 → logs/stop_ablation/
#
# 사용법:
#   bash scripts/run_stop_ablations.sh [--seeds N] [--dry-run]
# ============================================================
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${REPO}/.venv/bin/python3"
LOGDIR="${REPO}/logs/stop_ablation"
SEEDS=5
DRY_RUN=0
START_TIME=$(date +%s)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds)  SEEDS="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

mkdir -p "${LOGDIR}"
echo "=========================================================="
echo "  STOP Ablation Runner"
echo "  Seeds     : ${SEEDS}"
echo "  Python    : ${PYTHON}"
echo "  Logs      : ${LOGDIR}"
echo "  Started   : $(date)"
echo "=========================================================="

run_bg() {
  local NAME="$1"
  local SCRIPT="$2"
  local EXTRA_ARGS="${3:-}"
  local LOGFILE="${LOGDIR}/${NAME}.log"

  local CMD="${PYTHON} -u ${REPO}/scripts/${SCRIPT} --seeds ${SEEDS} ${EXTRA_ARGS}"

  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "[DRY-RUN] would run: ${CMD}"
    echo "[DRY-RUN] log: ${LOGFILE}"
    return
  fi

  echo ""
  echo "▶ [${NAME}] 시작"
  echo "  cmd : ${CMD}"
  echo "  log : ${LOGFILE}"

  nohup bash -c "
    echo '=== ${NAME} START ===' >> '${LOGFILE}'
    date >> '${LOGFILE}'
    ${CMD} >> '${LOGFILE}' 2>&1
    echo '=== ${NAME} DONE ===' >> '${LOGFILE}'
    date >> '${LOGFILE}'
  " > /dev/null 2>&1 &

  local PID=$!
  echo "  PID : ${PID}"
  echo "${PID}" > "${LOGDIR}/${NAME}.pid"
}

# ── L1: STOP Parameter Sweep ────────────────────────────────
run_bg "L1_stop_param" \
       "ablate_stop_proximity.py" \
       "--tag L1_param_sweep"

# ── L2: CLIP Cosine-Sim STOP ────────────────────────────────
run_bg "L2_stop_clipsim" \
       "ablate_stop_clip_sim.py" \
       "--tag L2_clipsim_stop"

# ── L3: STOP-Weighted Training (합성 STOP 주입 + 재학습) ────
# L1/L2와 독립 — GPU 공유 안 되도록 L1/L2 시작 후 딜레이 없이 병렬 실행
# (L3는 feature 캐싱 후 VLM 해제하므로 L1/L2와 GPU 충돌 최소)
if [[ "${DRY_RUN}" -eq 1 ]]; then
  echo "[DRY-RUN] would run: L3_stop_weighted"
else
  L3_LOG="${LOGDIR}/L3_stop_weighted.log"
  L3_CMD="${PYTHON} -u ${REPO}/scripts/ablate_stop_weighted_train.py --tag L3_stop_weighted"
  echo ""
  echo "▶ [L3_stop_weighted] 시작 (합성 STOP ×4 weight variants + CL eval)"
  echo "  cmd : ${L3_CMD}"
  echo "  log : ${L3_LOG}"
  nohup bash -c "
    echo '=== L3_stop_weighted START ===' >> '${L3_LOG}'
    date >> '${L3_LOG}'
    ${L3_CMD} >> '${L3_LOG}' 2>&1
    echo '=== L3_stop_weighted DONE ===' >> '${L3_LOG}'
    date >> '${L3_LOG}'
  " > /dev/null 2>&1 &
  L3_PID=$!
  echo "  PID : ${L3_PID}"
  echo "${L3_PID}" > "${LOGDIR}/L3_stop_weighted.pid"
fi

# ── 완료 대기 & 결과 요약 ───────────────────────────────────
if [[ "${DRY_RUN}" -ne 1 ]]; then
  echo ""
  echo "모든 job 비동기 실행 완료. 진행 상황 확인:"
  echo "  tail -f ${LOGDIR}/L1_stop_param.log"
  echo "  tail -f ${LOGDIR}/L2_stop_clipsim.log"
  echo "  tail -f ${LOGDIR}/L3_stop_weighted.log"
  echo ""
  echo "결과 확인:"
  echo "  cat docs/v5/closed_loop_eval/stop_ablation_results.json"
  echo "  cat docs/v5/closed_loop_eval/stop_clipsim_results.json"
  echo "  cat docs/v5/closed_loop_eval/stop_weighted_results.json"
  echo ""

  MONITOR_SCRIPT="${LOGDIR}/monitor.sh"
  cat > "${MONITOR_SCRIPT}" << 'MONITOR'
#!/usr/bin/env bash
LOGDIR="$(dirname "$0")"
check_done() {
  local PID_FILE="${LOGDIR}/${1}.pid"
  [[ ! -f "${PID_FILE}" ]] && return 1
  ! kill -0 "$(cat "${PID_FILE}")" 2>/dev/null && return 0 || return 1
}
while true; do
  ALL_DONE=1
  for JOB in L1_stop_param L2_stop_clipsim L3_stop_weighted; do
    if check_done "${JOB}"; then
      echo "  [DONE] ${JOB}"
    else
      echo "  [RUN ] ${JOB} (PID $(cat "${LOGDIR}/${JOB}.pid" 2>/dev/null || echo '?'))"
      ALL_DONE=0
    fi
  done
  [[ "${ALL_DONE}" -eq 1 ]] && echo "모든 job 완료!" && break
  echo "  ($(date '+%H:%M:%S')) 30초 후 재확인..."
  sleep 30
done
MONITOR
  chmod +x "${MONITOR_SCRIPT}"
  echo "  모니터링: bash ${MONITOR_SCRIPT}"
fi

echo ""
echo "=========================================================="
echo "  Runner 완료  ($(date))"
echo "=========================================================="

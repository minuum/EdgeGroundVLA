#!/usr/bin/env bash
# MoNaVLA 전체 서비스 시작 스크립트
#
# Usage:
#   bash scripts/start_all.sh           # 전체 시작
#   bash scripts/start_all.sh hub       # hub만
#   bash scripts/start_all.sh inference # inference 관련만
#   bash scripts/start_all.sh status    # 상태 확인만

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS="${ROOT}/logs"
mkdir -p "$LOGS"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; RESET='\033[0m'; BOLD='\033[1m'

# ─── ROS 환경 소스 (카메라 연동 서비스에 필수) ────────────────────────────────
ROS_SETUP="/opt/ros/humble/setup.bash"
ROS_WS_SETUP="${ROOT}/ROS_action/install/setup.bash"

# ROS가 있으면 현재 셸에도 소스 (LD_LIBRARY_PATH 등 설정)
# set -u 와 충돌하는 ROS setup.bash 변수 때문에 -u 임시 해제
set +u
if [[ -f "$ROS_SETUP" ]]; then
  # shellcheck source=/dev/null
  source "$ROS_SETUP"
fi
if [[ -f "$ROS_WS_SETUP" ]]; then
  # shellcheck source=/dev/null
  source "$ROS_WS_SETUP"
  # colcon setup.bash는 install/*/lib 를 누락 → 명시적으로 추가
  for _pkg_lib in "${ROOT}/ROS_action/install/"*/lib; do
    [[ -d "$_pkg_lib" ]] && export LD_LIBRARY_PATH="${_pkg_lib}:${LD_LIBRARY_PATH:-}"
  done
  printf "  ${GREEN}✓${RESET} ROS2 sourced + camera_interfaces/lib 경로 추가\n"
else
  printf "  ${YELLOW}⚠${RESET} ROS2 workspace not found: %s\n" "$ROS_WS_SETUP"
fi
set -u

# ROS 소스된 환경을 그대로 물려주는 래퍼 명령어 (nohup 자식 프로세스에도 전달)
ROS_CMD_PREFIX=""
if [[ -f "$ROS_SETUP" && -f "$ROS_WS_SETUP" ]]; then
  ROS_CMD_PREFIX="source '${ROS_SETUP}' && source '${ROS_WS_SETUP}' && \
for _p in '${ROOT}/ROS_action/install/'*/lib; do export LD_LIBRARY_PATH=\"\${_p}:\${LD_LIBRARY_PATH:-}\"; done && "
fi

# ─── 서비스 정의: "이름|포트|명령어" ───────────────────────────────────────────
declare -a SERVICES=(
  "hub|7860|python3 scripts/gradio_hub.py"
  "grounding_demo|7863|${ROS_CMD_PREFIX}python3 scripts/gradio_grounding_demo.py"
  "inference_dashboard|7865|${ROS_CMD_PREFIX}python3 scripts/gradio_inference_dashboard.py"
  "trial_logger|7862|python3 scripts/real_robot_trial_logger.py"
  "data_collector|8081|python3 scripts/gradio_data_collector.py"
  "session_eval|7861|python3 scripts/gradio_session_eval.py"
  "h5_analyzer|7866|python3 scripts/gradio_offline_h5_analyzer.py"
  "monitor|8080|python3 scripts/monitor_dashboard.py"
  "goalnav_api|8001|${ROS_CMD_PREFIX}python3 robovlm_nav/serve/proxy_inference_server.py --port 8001"
  "inference_server|8000|VLA_GOALNAV_VARIANT=exp54_s2v2 VLA_GOALNAV_ONLY=1 python3 robovlm_nav/serve/inference_server.py"
)

# ─── 유틸 함수 ─────────────────────────────────────────────────────────────────

is_port_up() {
  local port="$1"
  ss -tlnp 2>/dev/null | grep -q ":${port} " && return 0 || return 1
}

get_pid() {
  local port="$1"
  ss -tlnp 2>/dev/null | grep ":${port} " | grep -oP 'pid=\K[0-9]+' | head -1
}

start_svc() {
  local name="$1" port="$2" cmd="$3"
  local log="${LOGS}/${name}.log"

  if is_port_up "$port"; then
    local pid; pid=$(get_pid "$port")
    printf "  ${GREEN}●${RESET} %-22s already up  ${CYAN}pid=%-6s${RESET} :${port}\n" "$name" "$pid"
    return
  fi

  printf "  ${YELLOW}○${RESET} %-22s starting..." "$name"
  # shellcheck disable=SC2086
  nohup bash -c "cd '${ROOT}' && ${cmd}" > "$log" 2>&1 &
  local bg_pid=$!

  local waited=0
  while ! is_port_up "$port" && [[ $waited -lt 90 ]]; do
    sleep 1; ((waited++))
    printf "."
  done

  if is_port_up "$port"; then
    local pid; pid=$(get_pid "$port")
    printf "\r  ${GREEN}●${RESET} %-22s started     ${CYAN}pid=%-6s${RESET} :${port}\n" "$name" "$pid"
  else
    printf "\r  ${RED}✗${RESET} %-22s timeout      log: %s\n" "$name" "$log"
  fi
}

stop_svc() {
  local name="$1" port="$2"
  if ! is_port_up "$port"; then
    printf "  ${CYAN}–${RESET} %-22s not running  :${port}\n" "$name"
    return
  fi
  local pid; pid=$(get_pid "$port")
  kill "$pid" 2>/dev/null && \
    printf "  ${RED}●${RESET} %-22s stopped      pid=%s\n" "$name" "$pid" || \
    printf "  ${RED}✗${RESET} %-22s failed to stop\n" "$name"
}

print_status() {
  local ip; ip=$(hostname -I | awk '{print $1}')
  echo ""
  printf "${BOLD}%-22s  %-6s  %-8s  %s${RESET}\n" "SERVICE" "PORT" "STATUS" "URL"
  printf '%0.s─' {1..70}; echo ""
  for entry in "${SERVICES[@]}"; do
    IFS='|' read -r name port _ <<< "$entry"
    if is_port_up "$port"; then
      local pid; pid=$(get_pid "$port")
      printf "${GREEN}●${RESET} %-22s  :%-5s  ${GREEN}UP%-5s${RESET}  http://%s:%s\n" \
        "$name" "$port" " pid=$pid" "$ip" "$port"
    else
      printf "${RED}●${RESET} %-22s  :%-5s  ${RED}DOWN${RESET}\n" "$name" "$port"
    fi
  done
  echo ""
}

# ─── 실행 필터 ─────────────────────────────────────────────────────────────────

FILTER="${1:-all}"

case "$FILTER" in
  status)
    print_status
    exit 0
    ;;
  stop)
    echo -e "${BOLD}Stopping all services...${RESET}"
    for entry in "${SERVICES[@]}"; do
      IFS='|' read -r name port _ <<< "$entry"
      stop_svc "$name" "$port"
    done
    exit 0
    ;;
esac

# 시작할 서비스 필터링
declare -a TO_START=()
for entry in "${SERVICES[@]}"; do
  IFS='|' read -r name port cmd <<< "$entry"
  case "$FILTER" in
    all)       TO_START+=("$entry") ;;
    hub)       [[ "$name" == "hub" ]] && TO_START+=("$entry") ;;
    inference) [[ "$name" =~ inference|goalnav ]] && TO_START+=("$entry") ;;
    demo)      [[ "$name" =~ grounding|hub ]] && TO_START+=("$entry") ;;
    robot)     [[ "$name" =~ inference|trial|goalnav ]] && TO_START+=("$entry") ;;
    data)      [[ "$name" =~ collector|session|h5 ]] && TO_START+=("$entry") ;;
    "$name")   TO_START+=("$entry") ;;
  esac
done

if [[ ${#TO_START[@]} -eq 0 ]]; then
  echo "Unknown filter: $FILTER"
  echo "Usage: $0 [all|hub|inference|demo|robot|data|stop|status|<service_name>]"
  exit 1
fi

echo ""
echo -e "${BOLD}MoNaVLA — Starting Services${RESET}  (filter: $FILTER)"
printf '%0.s─' {1..50}; echo ""

cd "$ROOT"
for entry in "${TO_START[@]}"; do
  IFS='|' read -r name port cmd <<< "$entry"
  start_svc "$name" "$port" "$cmd"
done

echo ""
print_status

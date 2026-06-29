#!/bin/bash
# MoNaVLA API 서버 관리 스크립트
# Usage: ./manage_api_server.sh [start|stop|restart|status|logs]

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${VLA_PROJECT_DIR:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PROFILE_SCRIPT="$PROJECT_DIR/scripts/vla_profile.py"
E2E_SERVER_SCRIPT="$PROJECT_DIR/robovlm_nav/serve/inference_server.py"
PROXY_SERVER_SCRIPT="$PROJECT_DIR/robovlm_nav/serve/proxy_inference_server.py"
STAGE2V2_SERVER_SCRIPT="$PROJECT_DIR/robovlm_nav/serve/stage2_v2_inference_server.py"
SERVER_SCRIPT="$E2E_SERVER_SCRIPT"  # overridden by resolve_profile_env for proxy_server/stage2_v2 runtime
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/api_server.log"
PID_FILE="$LOG_DIR/api_server.pid"
HOST="${VLA_API_HOST:-0.0.0.0}"
PORT="${VLA_PORT:-8000}"
PROFILE_NAME="${2:-${VLA_PROFILE:-end_to_end_default}}"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

check_process() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            return 0
        fi
        rm -f "$PID_FILE"
    fi

    if pgrep -f "$SERVER_SCRIPT" > /dev/null 2>&1; then
        return 0
    fi
    return 1
}

resolve_profile_env() {
    if [ ! -f "$PROFILE_SCRIPT" ]; then
        echo -e "${RED}Profile resolver not found: $PROFILE_SCRIPT${NC}"
        return 1
    fi

    PROFILE_EXPORTS=$(python3 "$PROFILE_SCRIPT" env --profile "$PROFILE_NAME") || return 1
    eval "$PROFILE_EXPORTS"

    if [ "${VLA_MODEL_RUNTIME:-}" = "mlp_step2" ]; then
        echo -e "${YELLOW}Profile ${PROFILE_NAME} resolves to runtime ${VLA_MODEL_RUNTIME}, which is not served by inference_server.py${NC}"
        return 1
    elif [ "${VLA_MODEL_RUNTIME:-}" = "proxy_server" ]; then
        SERVER_SCRIPT="$PROXY_SERVER_SCRIPT"
        export VLA_PROXY_WEIGHTS_PATH="${VLA_PROXY_WEIGHTS_PATH:-${VLA_CHECKPOINT_PATH:-}}"
    elif [ "${VLA_MODEL_RUNTIME:-}" = "stage2_v2" ]; then
        SERVER_SCRIPT="$STAGE2V2_SERVER_SCRIPT"
        PORT="${VLA_PORT:-8001}"
    else
        SERVER_SCRIPT="$E2E_SERVER_SCRIPT"
    fi
    return 0
}

start_server() {
    echo -e "${BLUE}Starting API server...${NC}"

    if check_process; then
        echo -e "${YELLOW}API server is already running${NC}"
        return 1
    fi

    mkdir -p "$LOG_DIR"
    cd "$PROJECT_DIR" || exit 1
    resolve_profile_env || return 1
    echo -e "Profile: ${YELLOW}${VLA_PROFILE:-unknown}${NC} | Model: ${YELLOW}${VLA_MODEL_LABEL:-unknown}${NC}"

    nohup python3 "$SERVER_SCRIPT" --host "$HOST" --port "$PORT" > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "$PID" > "$PID_FILE"

    sleep 2

    if check_process; then
        echo -e "${GREEN}✓ API server started (PID: $PID)${NC}"
        echo -e "Log file: ${YELLOW}$LOG_FILE${NC}"
        HEALTH_OK=0
        for _ in $(seq 1 30); do
            if curl -s "http://127.0.0.1:$PORT/health" > /dev/null; then
                HEALTH_OK=1
                break
            fi
            sleep 1
        done
        if [ "$HEALTH_OK" -eq 1 ]; then
            echo -e "${GREEN}✓ Server is healthy${NC}"
        else
            echo -e "${YELLOW}⚠ Server started but health check timed out${NC}"
        fi
    else
        echo -e "${RED}✗ Failed to start API server${NC}"
        rm -f "$PID_FILE"
        return 1
    fi
}

stop_server() {
    echo -e "${BLUE}Stopping API server...${NC}"

    if ! check_process; then
        echo -e "${YELLOW}API server is not running${NC}"
        return 1
    fi

    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        kill "$PID" 2>/dev/null
        rm -f "$PID_FILE"
    fi

    pkill -f "$SERVER_SCRIPT" 2>/dev/null || true

    sleep 1
    if ! check_process; then
        echo -e "${GREEN}✓ API server stopped${NC}"
    else
        echo -e "${RED}✗ Failed to stop API server${NC}"
        return 1
    fi
}

restart_server() {
    stop_server
    sleep 1
    start_server
}

status_server() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}API Server Status${NC}"
    echo -e "${BLUE}========================================${NC}"

    if check_process; then
        resolve_profile_env >/dev/null 2>&1 || true
        PID=$(pgrep -f "$SERVER_SCRIPT" | head -n 1)
        echo -e "Status: ${GREEN}Running ✓${NC}"
        echo -e "PID: ${YELLOW}$PID${NC}"
        echo -e "Profile: ${YELLOW}${VLA_PROFILE:-unknown}${NC}"
        echo -e "Model: ${YELLOW}${VLA_MODEL_LABEL:-unknown}${NC}"
        ps -p "$PID" -o %cpu,%mem,cmd 2>/dev/null | tail -n 1
        echo ""
        echo "Health Check:"
        curl -s "http://127.0.0.1:$PORT/health" | python3 -m json.tool 2>/dev/null || echo "Failed"
    else
        echo -e "Status: ${RED}Not Running ✗${NC}"
        echo -e "${YELLOW}Note: log tail below may be from a previous run and does not mean the server is currently alive.${NC}"
    fi

    echo ""
    echo -e "Log file: ${YELLOW}$LOG_FILE${NC}"
    if [ -f "$LOG_FILE" ]; then
        echo "Last 5 lines:"
        tail -5 "$LOG_FILE"
    fi
}

view_logs() {
    if [ ! -f "$LOG_FILE" ]; then
        echo -e "${RED}Log file not found: $LOG_FILE${NC}"
        return 1
    fi

    echo -e "${BLUE}API Server Logs (tail -f)${NC}"
    echo -e "${YELLOW}Press Ctrl+C to exit${NC}"
    tail -f "$LOG_FILE"
}

case "${1:-status}" in
    start)
        start_server
        ;;
    stop)
        stop_server
        ;;
    restart)
        restart_server
        ;;
    status)
        status_server
        ;;
    logs)
        view_logs
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        exit 1
        ;;
esac

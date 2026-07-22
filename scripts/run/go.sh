#!/usr/bin/env bash
# mona-up / vla-go — soda에서 실행. 추론 서버 + 대시보드 + 허브 한 번에 시작
#
# 사용법:
#   go.sh                # 전부 (서버 + Gradio대시보드 + 허브 + 뷰어)
#   go.sh --server       # 추론 서버만 (포트 8001)
#   go.sh --mona-dash    # FastAPI 대시보드만 (포트 7800) ★ 새 권장 UI
#   go.sh --dashboard    # Gradio 대시보드 (포트 7865) 구버전
#   go.sh --hub          # 허브만 (포트 7860)
#   go.sh --viewer       # 데이터셋 뷰어만 (포트 8083)
#   go.sh --status       # 실행 중 서비스 확인
#   go.sh --stop         # 모두 종료

set -euo pipefail
cd "${VLA_PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

# API 키 단일 소스: .vla_api_key 파일이 있으면 셸 환경변수보다 우선한다.
# 서버/대시보드를 서로 다른 셸에서 재시작하면 키가 어긋나 403 연쇄→STOP만
# 발행되는 사고가 2026-07-02에 두 번 발생 — 어느 셸에서 실행해도 같은 키를
# 쓰도록 파일 하나로 고정 (plan_20260702_healthcheck_routine.md B2).
if [[ -f .vla_api_key ]]; then
    VLA_API_KEY="$(head -1 .vla_api_key | tr -d '[:space:]')"
    export VLA_API_KEY
fi

# Tailscale IP 우선 사용 (원격 접속용), 없으면 로컬 IP fallback
SODA_IP=$(ip addr show tailscale0 2>/dev/null | awk '/inet /{print $2}' | cut -d/ -f1)
[[ -z "$SODA_IP" ]] && SODA_IP=$(hostname -I | awk '{print $1}')
SERVER_PORT=8001
DRIVE_PORT=7800      # FastAPI 대시보드 (mona_dashboard.py)
DASH_PORT=7865       # Gradio 대시보드 (구버전)
HUB_PORT=7860
S1_PT="runs/v5_nav/mlp/shared/stage1_v2_projs.pt"
S2_PT="runs/v5_nav/mlp/stop_lastN/stop_N1.pt"
# VLA_S2V2_STAGE2 env로 오버라이드 가능 (대시보드 ⚙️ 서버 설정 탭의 체크포인트 전환)
ACTION_PT="${VLA_S2V2_STAGE2:-runs/v5_nav/mlp/exp71_window6/action_transformer.pt}"

# .venv 있으면 우선, 없으면 시스템 python3
if [[ -x ".venv/bin/python3" ]]; then
    PY=".venv/bin/python3"
else
    PY="$(which python3)"
fi

MODE="${1:---all}"

# ── 포트 헬스체크 헬퍼 ────────────────────────────────────────────────────
wait_port() {
    local label="$1" port="$2" timeout="${3:-40}" logfile="${4:-}"
    printf "  %-18s 대기 " "$label"
    for i in $(seq 1 "$timeout"); do
        sleep 1
        if curl -sf "http://localhost:$port" > /dev/null 2>&1; then
            echo " ✓ (${i}s)"
            return 0
        fi
        printf "."
        if [ "$i" -eq "$timeout" ]; then
            echo " ✗ (${timeout}s 타임아웃)"
            [[ -n "$logfile" && -f "$logfile" ]] && tail -10 "$logfile"
            return 1
        fi
    done
}

# ── 상태 확인 ─────────────────────────────────────────────────────────────
status_check() {
    echo ""
    echo "━━ 서비스 상태 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    if curl -sf "http://localhost:$SERVER_PORT/health" > /dev/null 2>&1; then
        echo "  ✓ 추론 서버     http://$SODA_IP:$SERVER_PORT"
        curl -s "http://localhost:$SERVER_PORT/health" | python3 -m json.tool 2>/dev/null \
            | grep -E '"status|"model_loaded|"head|"window' | sed 's/^/      /' || true
    else
        echo "  ✗ 추론 서버     (포트 $SERVER_PORT 응답 없음)"
    fi
    if curl -sf "http://localhost:$DRIVE_PORT/health" > /dev/null 2>&1; then
        echo "  ✓ FastAPI 대시보드 http://$SODA_IP:$DRIVE_PORT  ★ 신규 UI"
    else
        echo "  ✗ FastAPI 대시보드 (포트 $DRIVE_PORT 응답 없음)"
    fi
    if curl -sf "http://localhost:$DASH_PORT" > /dev/null 2>&1; then
        echo "  ✓ Gradio 대시보드  http://$SODA_IP:$DASH_PORT  (구버전)"
    else
        echo "  - Gradio 대시보드  (포트 $DASH_PORT 미실행)"
    fi
    if curl -sf "http://localhost:$HUB_PORT" > /dev/null 2>&1; then
        echo "  ✓ 허브 UI          http://$SODA_IP:$HUB_PORT"
    else
        echo "  - 허브 UI          (포트 $HUB_PORT 미실행)"
    fi
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ── 종료 ──────────────────────────────────────────────────────────────────
if [[ "$MODE" == "--stop" ]]; then
    pkill -f "stage2_v2_inference_server" 2>/dev/null && echo "  추론 서버 종료"        || echo "  추론 서버: 실행 중 아님"
    pkill -f "mona_dashboard"             2>/dev/null && echo "  FastAPI 대시보드 종료" || echo "  FastAPI 대시보드: 실행 중 아님"
    pkill -f "gradio_inference_dashboard" 2>/dev/null && echo "  Gradio 대시보드 종료" || echo "  Gradio 대시보드: 실행 중 아님"
    pkill -f "gradio_hub"                 2>/dev/null && echo "  허브 종료"             || echo "  허브: 실행 중 아님"
    pkill -f "gradio_dataset_viewer"      2>/dev/null && echo "  데이터셋 뷰어 종료"   || echo "  데이터셋 뷰어: 실행 중 아님"
    # MJPEG 스트림 등 오래 붙잡는 커넥션이 있으면 SIGTERM만으론 안 죽는 경우가
    # 있어 잠깐 대기 후 강제종료로 마무리 (좀비가 옛 스트림을 계속 서빙하는 것 방지)
    sleep 0.5
    for pat in "stage2_v2_inference_server" "mona_dashboard" "gradio_inference_dashboard" "gradio_hub" "gradio_dataset_viewer"; do
        pkill -9 -f "$pat" 2>/dev/null || true
    done
    exit 0
fi

if [[ "$MODE" == "--status" ]]; then
    status_check; exit 0
fi

# ── 체크포인트 확인 ───────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  mona-up  (exp71 Transformer W=6 val_acc=99.2% — Stage2 v2 + learned STOP)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
for f in "$S1_PT" "$S2_PT" "$ACTION_PT"; do
    if [[ -f "$f" ]]; then
        echo "  ✓ $(du -sh "$f" | cut -f1)  $f"
    else
        echo "  ✗ 없음: $f  ←  mona-ship 먼저 실행 필요"
        [[ "$MODE" == "--hub" || "$MODE" == "--dashboard" || "$MODE" == "--viewer" ]] || exit 1
    fi
done
mkdir -p logs

# ── 1. 추론 서버 (FastAPI, 포트 8001) ────────────────────────────────────

if [[ "$MODE" == "--all" || "$MODE" == "--server" ]]; then
    echo ""
    echo "▶ 1/3  추론 서버 (포트 $SERVER_PORT)"
    pkill -f "stage2_v2_inference_server" 2>/dev/null && sleep 1 || true

    # 재시작마다 로그를 덮어써서 이전 세션 분석에 필요한 로그가 통째로 사라지는
    # 문제가 있었음(2026-07-02) — 덮어쓰기 전에 타임스탬프 붙여 보관
    [[ -f logs/s2v2_server.log ]] && mv logs/s2v2_server.log "logs/s2v2_server.$(date +%Y%m%d_%H%M%S).log"
    nohup env VLA_S2V2_STAGE2="$ACTION_PT" VLA_STOP_MODE="${VLA_STOP_MODE:-proximity}" \
        VLA_PREVIEW_ENABLED="${VLA_PREVIEW_ENABLED:-0}" \
        VLA_PREVIEW_MAX_RETRY="${VLA_PREVIEW_MAX_RETRY:-5}" \
        VLA_PREVIEW_ROT_DIR="${VLA_PREVIEW_ROT_DIR:-R}" \
        VLA_PREVIEW_HINT_CX="${VLA_PREVIEW_HINT_CX:-1}" \
        VLA_MULTI_PROMPT="${VLA_MULTI_PROMPT:-0}" \
        VLA_API_KEY="${VLA_API_KEY:-vla_devel_key_2026}" \
        VLA_GROUNDER="${VLA_GROUNDER:-pg2}" \
        VLA_OWLV2_THRESH="${VLA_OWLV2_THRESH:-0.25}" \
        VLA_OWLV2_AREA_SCALE="${VLA_OWLV2_AREA_SCALE:-3.0}" \
        VLA_PREVIEW_GROUNDER="${VLA_PREVIEW_GROUNDER:-pg2}" \
        "$PY" robovlm_nav/serve/stage2_v2_inference_server.py \
        --port "$SERVER_PORT" > logs/s2v2_server.log 2>&1 &
    disown $!
    echo "  PID=$!  logs/s2v2_server.log"

    echo "  헬스체크 (최대 120s — PG2 3B 로딩 포함)"
    for i in $(seq 1 120); do
        sleep 1
        if curl -sf "http://localhost:$SERVER_PORT/health" > /dev/null 2>&1; then
            echo "  ✓ 준비 완료 (${i}s)"; break
        fi
        if (( i % 10 == 0 )); then
            LAST=$(tail -1 logs/s2v2_server.log 2>/dev/null \
                | sed 's/.*INFO:__main__://' | cut -c1-60)
            printf "  [%3ds] %s\n" "$i" "$LAST"
        fi
        if [ "$i" -eq 120 ]; then
            echo "  ✗ 타임아웃"
            tail -20 logs/s2v2_server.log 2>/dev/null
            exit 1
        fi
    done
fi

# ── 1.5 FastAPI 대시보드 (mona_dashboard.py, 포트 7800) ★ 신규 권장 UI ──
if [[ "$MODE" == "--all" || "$MODE" == "--mona-dash" ]]; then
    echo ""
    echo "▶ FastAPI 대시보드 (포트 $DRIVE_PORT)  ★ 신규 권장 UI"
    # SIGTERM만 보내고 안 기다리면, MJPEG 스트림처럼 오래 붙잡고 있는 커넥션이
    # 있는 프로세스가 완전히 안 죽은 채 좀비로 남아 옛 스트림을 계속 서빙하는
    # 문제가 있었음(2026-07-02 실측) — 죽었는지 확인 후 안 죽으면 강제종료.
    if pgrep -f "mona_dashboard" > /dev/null 2>&1; then
        pkill -f "mona_dashboard" 2>/dev/null || true
        for i in $(seq 1 10); do
            pgrep -f "mona_dashboard" > /dev/null 2>&1 || break
            sleep 0.3
        done
        pkill -9 -f "mona_dashboard" 2>/dev/null || true
        sleep 0.5
    fi
    # 재시작마다 로그를 덮어써서 이전 세션 분석에 필요한 로그가 통째로 사라지는
    # 문제가 있었음(2026-07-02) — 덮어쓰기 전에 타임스탬프 붙여 보관
    [[ -f logs/mona_dashboard.log ]] && mv logs/mona_dashboard.log "logs/mona_dashboard.$(date +%Y%m%d_%H%M%S).log"

    # ROS2 환경 주입
    ROS_DIST="/opt/ros/humble"
    ROS_WS="$PWD/ROS_action/install"
    ROS2_PY="${ROS_DIST}/local/lib/python3.10/dist-packages:${ROS_DIST}/lib/python3.10/site-packages"
    WS_PY_PATHS=""
    for pkg in camera_interfaces ros_action_msgs; do
        p="$ROS_WS/$pkg/local/lib/python3.10/dist-packages"
        [[ -d "$p" ]] && WS_PY_PATHS="$WS_PY_PATHS:$p"
    done
    FULL_PY="${ROS2_PY}${WS_PY_PATHS}${PYTHONPATH:+:$PYTHONPATH}"
    ROS2_LIB="${ROS_DIST}/lib"
    WS_LIB_PATHS=""
    for pkg in camera_interfaces camera_pub; do
        p="$ROS_WS/$pkg/lib"
        [[ -d "$p" ]] && WS_LIB_PATHS="$WS_LIB_PATHS:$p"
    done
    FULL_LIB="${ROS2_LIB}${WS_LIB_PATHS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    SODA_IP="$SODA_IP" \
    VLA_API_SERVER="http://localhost:$SERVER_PORT" \
    VLA_API_KEY="${VLA_API_KEY:-vla_devel_key_2026}" \
    DRIVE_PORT="$DRIVE_PORT" \
    ROS_DOMAIN_ID=42 \
    RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    PYTHONPATH="$FULL_PY" \
    LD_LIBRARY_PATH="$FULL_LIB" \
    PATH="${ROS_DIST}/bin:$PATH" \
    nohup "$PY" robovlm_nav/serve/mona_dashboard.py --port "$DRIVE_PORT" \
        > logs/mona_dashboard.log 2>&1 &
    disown $!
    echo "  PID=$!  logs/mona_dashboard.log"
    wait_port "FastAPI 대시보드" "$DRIVE_PORT" 30 "logs/mona_dashboard.log"
    echo "  접속: http://$SODA_IP:$DRIVE_PORT"
fi

# ── 2. Gradio 대시보드 (포트 7865) — 구버전 (명시적 --dashboard 시만) ───
if [[ "$MODE" == "--dashboard" ]]; then
    echo ""
    echo "▶ 2/3  추론 대시보드 (포트 $DASH_PORT)  ★ 주 제어 UI"
    pkill -f "gradio_inference_dashboard" 2>/dev/null && sleep 1 || true

    # ROS2 전체 환경 주입 (source 없이 경로 직접 구성)
    ROS_DIST="/opt/ros/humble"
    ROS_WS="$PWD/ROS_action/install"

    # PYTHONPATH: rclpy + geometry_msgs + camera_interfaces + ros_action_msgs
    ROS2_PY="${ROS_DIST}/local/lib/python3.10/dist-packages:${ROS_DIST}/lib/python3.10/site-packages"
    WS_PY_PATHS=""
    for pkg in camera_interfaces ros_action_msgs; do
        p="$ROS_WS/$pkg/local/lib/python3.10/dist-packages"
        [[ -d "$p" ]] && WS_PY_PATHS="$WS_PY_PATHS:$p"
    done
    FULL_PY="${ROS2_PY}${WS_PY_PATHS}${PYTHONPATH:+:$PYTHONPATH}"

    # LD_LIBRARY_PATH: ROS2 + workspace .so
    ROS2_LIB="${ROS_DIST}/lib"
    WS_LIB_PATHS=""
    for pkg in camera_interfaces camera_pub; do
        p="$ROS_WS/$pkg/lib"
        [[ -d "$p" ]] && WS_LIB_PATHS="$WS_LIB_PATHS:$p"
    done
    FULL_LIB="${ROS2_LIB}${WS_LIB_PATHS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

    # PATH: ros2 binary
    FULL_PATH="${ROS_DIST}/bin:$PATH"

    VLA_API_SERVER="http://localhost:$SERVER_PORT" \
    VLA_SERVER_ROLE=jetson \
    VLA_INFERENCE_PORT="$DASH_PORT" \
    VLA_ROS_WS="$ROS_WS" \
    ROS_DOMAIN_ID=42 \
    RMW_IMPLEMENTATION=rmw_fastrtps_cpp \
    GRADIO_SHARE=0 \
    VLA_ASYNC_MODE=0 \
    VLA_INFERENCE_HZ=3.0 \
    VLA_EXECUTION_HZ=10.0 \
    PYTHONPATH="$FULL_PY" \
    LD_LIBRARY_PATH="$FULL_LIB" \
    PATH="$FULL_PATH" \
    nohup "$PY" scripts/gradio_inference_dashboard.py \
        > logs/inference_dashboard.log 2>&1 &
    disown $!
    echo "  PID=$!  logs/inference_dashboard.log"

    wait_port "추론 대시보드" "$DASH_PORT" 50 "logs/inference_dashboard.log"
fi

# ── 3. 허브 (Gradio, 포트 7860) ──────────────────────────────────────────
if [[ "$MODE" == "--all" || "$MODE" == "--hub" ]]; then
    echo ""
    echo "▶ 3/3  허브 UI (포트 $HUB_PORT)"
    pkill -f "gradio_hub" 2>/dev/null && sleep 1 || true

    nohup "$PY" scripts/gradio_hub.py \
        > logs/hub.log 2>&1 &
    disown $!
    echo "  PID=$!  logs/hub.log"

    wait_port "허브" "$HUB_PORT" 40 "logs/hub.log"
fi

# ── 4. 데이터셋 뷰어 (Gradio, 포트 8083) ────────────────────────────────
VIEWER_PORT=8083
if [[ "$MODE" == "--all" || "$MODE" == "--viewer" ]]; then
    echo ""
    echo "▶ 데이터셋 뷰어 (포트 $VIEWER_PORT)"
    pkill -f "gradio_dataset_viewer" 2>/dev/null && sleep 1 || true

    VLA_API_KEY="${VLA_API_KEY:-}" \
    "$PY" scripts/gradio_dataset_viewer.py \
        >> logs/dataset_viewer.log 2>&1 &
    _viewer_pid=$!
    disown $_viewer_pid 2>/dev/null || true
    echo "  PID=$_viewer_pid  logs/dataset_viewer.log  (시작 ~20s)"
fi

status_check

echo ""
echo "  ★ FastAPI 대시보드: http://$SODA_IP:$DRIVE_PORT   (신규 권장)"
echo "  Gradio 대시보드:    http://$SODA_IP:$DASH_PORT   (구버전)"
echo "  추론 서버:          http://$SODA_IP:$SERVER_PORT"
echo "  허브:               http://$SODA_IP:$HUB_PORT"
echo "  서버 로그:          tail -f ~/MoNaVLA/logs/mona_dashboard.log"

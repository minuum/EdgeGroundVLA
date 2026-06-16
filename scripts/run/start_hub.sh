#!/usr/bin/env bash
# MoNaVLA Hub 영구 실행 — SSH 세션 독립 (setsid + nohup)
# 사용: bash scripts/run/start_hub.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${VLA_PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

LOG=logs/hub.log
mkdir -p logs

# 기존 허브 프로세스 종료
EXISTING=$(pgrep -f "gradio_hub.py" 2>/dev/null)
if [ -n "$EXISTING" ]; then
  echo "[start_hub] 기존 PID $EXISTING 종료 중..."
  kill $EXISTING 2>/dev/null
  sleep 2
fi

echo "[start_hub] Hub 시작 → $LOG"
setsid nohup python3 scripts/gradio_hub.py > "$LOG" 2>&1 &
HUB_PID=$!
disown $HUB_PID

sleep 3
if kill -0 $HUB_PID 2>/dev/null; then
  echo "[start_hub] ✅ PID=$HUB_PID  http://100.85.118.58:7860"
else
  echo "[start_hub] ❌ 시작 실패 — $LOG 확인"
fi

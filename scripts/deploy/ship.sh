#!/usr/bin/env bash
# vla-ship — minum → soda 전체 배포 파이프라인
#
# 코드 수정 후 한 번에:
#   git push → rsync .pt → soda git pull → 서버 재시작 → 헬스체크
#
# 사용법:
#   vla-ship                  # 코드 + .pt 동시 배포 + 서버 재시작
#   vla-ship --code-only      # .pt 전송 없이 코드만
#   vla-ship --ckpt-only      # git push 없이 .pt + 서버 재시작만
#   vla-ship --no-restart     # 서버 재시작 없이 파일만 전송
#   vla-ship --dry-run        # 실제 전송 없이 할 일만 출력

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

SODA="soda@100.85.118.58"
SODA_ROOT="~/MoNaVLA"

CODE_ONLY=0; CKPT_ONLY=0; NO_RESTART=0; DRY_RUN=0
for arg in "$@"; do
    case "$arg" in
        --code-only)  CODE_ONLY=1  ;;
        --ckpt-only)  CKPT_ONLY=1  ;;
        --no-restart) NO_RESTART=1 ;;
        --dry-run)    DRY_RUN=1    ;;
    esac
done

run() { [[ $DRY_RUN -eq 1 ]] && echo "  [dry] $*" || eval "$@"; }

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  vla-ship  →  $SODA:$SODA_ROOT"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ── 1. git push (코드) ───────────────────────────────────────────────────
if [[ $CKPT_ONLY -eq 0 ]]; then
    echo ""
    echo "▶ 1/4  git push (inference-integration → monavla-driving)"
    BRANCH=$(git rev-parse --abbrev-ref HEAD)
    if [[ $DRY_RUN -eq 0 ]]; then
        git push origin "$BRANCH" --quiet && echo "  ✓ pushed $BRANCH"
        # monavla-driving 도 동기화 (로봇 pull 브랜치)
        git push origin "$BRANCH":monavla-driving --quiet && echo "  ✓ synced monavla-driving"
    else
        echo "  [dry] git push origin $BRANCH"
        echo "  [dry] git push origin $BRANCH:monavla-driving"
    fi
fi

# ── 2. .pt rsync ─────────────────────────────────────────────────────────
if [[ $CODE_ONLY -eq 0 ]]; then
    echo ""
    echo "▶ 2/4  rsync .pt checkpoints"
    CKPTS=(
        "runs/v5_nav/mlp/shared/stage1_v2_projs.pt"
        "runs/v5_nav/mlp/exp66/action_mlp.pt"
    )
    for f in "${CKPTS[@]}"; do
        if [[ -f "$f" ]]; then
            SIZE=$(du -sh "$f" | cut -f1)
            echo "  $(printf '%-10s' $SIZE) $f"
        else
            echo "  [없음] $f  ← 확인 필요"
        fi
    done
    run "rsync -avz --relative ${CKPTS[*]} $SODA:$SODA_ROOT/"
fi

# ── 3. 서버 코드 rsync (gradio + serve) ─────────────────────────────────
echo ""
echo "▶ 3/4  서버 코드 sync"
SCRIPTS=(
    "robovlm_nav/serve/stage2_v2_inference_server.py"
    "scripts/gradio_hub.py"
    "scripts/gradio_inference_dashboard.py"
)
run "rsync -avz --relative ${SCRIPTS[*]} $SODA:$SODA_ROOT/"

# ── 4. soda: git pull + 서버 재시작 ─────────────────────────────────────
if [[ $NO_RESTART -eq 0 ]]; then
    echo ""
    echo "▶ 4/4  soda 원격 재시작"
    REMOTE_CMD=$(cat <<'REMOTE'
set -e
cd ~/MoNaVLA

# git pull (monavla-driving 브랜치)
BRANCH=$(git rev-parse --abbrev-ref HEAD)
if git pull --quiet origin "$BRANCH" 2>/dev/null; then
    echo "  git pulled: $BRANCH"
fi

# 기존 서버 종료
pkill -f "stage2_v2_inference_server" 2>/dev/null && echo "  기존 서버 종료" || true
sleep 1

# 서버 재시작 (백그라운드)
mkdir -p logs
if [ -x ".venv/bin/python3" ]; then PY=".venv/bin/python3"; else PY="$(which python3)"; fi
nohup "$PY" robovlm_nav/serve/stage2_v2_inference_server.py \
    --port 8001 > logs/s2v2_server.log 2>&1 &
SERVER_PID=$!
disown $SERVER_PID

# 헬스체크 (최대 120초 — PG2 3B 로딩 포함)
echo "  서버 시작 중 (PID=$SERVER_PID, 최대 120s)..."
for i in $(seq 1 120); do
    sleep 1
    if curl -sf http://localhost:8001/health > /dev/null 2>&1; then
        echo "  ✓ 서버 준비 완료 (${i}s)"
        curl -s http://localhost:8001/health | python3 -m json.tool 2>/dev/null | grep -E '"model|"head|"window|"stage' || true
        break
    fi
    if (( i % 15 == 0 )); then
        LAST=$(tail -1 logs/s2v2_server.log 2>/dev/null | sed 's/.*INFO:__main__://' | cut -c1-60)
        printf "  [%3ds] %s\n" "$i" "$LAST"
    fi
    if [ $i -eq 120 ]; then
        echo "  ✗ 헬스체크 실패 (120s 타임아웃)"
        echo "  ── 서버 로그 ──────────────────────────────────────"
        tail -15 logs/s2v2_server.log 2>/dev/null || echo "  (로그 없음)"
        echo "  ──────────────────────────────────────────────────"
    fi
done
REMOTE
)
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  [dry] ssh $SODA << 'REMOTE_CMD'"
        echo "$REMOTE_CMD" | sed 's/^/    /'
    else
        ssh "$SODA" "$REMOTE_CMD"
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  배포 완료"
echo "  추론 서버:  http://100.85.118.58:8001/health"
echo "  허브 UI:    http://100.85.118.58:7860  (vla-go 실행 필요)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

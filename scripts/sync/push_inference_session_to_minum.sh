#!/usr/bin/env bash
# MoNaVLA 추론 세션 + 시스템 설정 + 모델 정보를 minum 서버로 전송
#
# 이번 대화(2026-07-02)에서 수동으로 하던 절차를 스킬화:
#   방금 수집한 세션 H5 + 서버 /health 설정 스냅샷 + 활성 모델 ckpt + episode_log
#   → minum:~/MoNaVLA/inference_sessions_recv/<date>/ 로 rsync + 매니페스트 생성
#
# 사용법:
#   bash scripts/sync/push_inference_session_to_minum.sh              # 가장 최근 세션 1개
#   bash scripts/sync/push_inference_session_to_minum.sh -n 5         # 최근 5개
#   bash scripts/sync/push_inference_session_to_minum.sh --last-24h   # 최근 24시간
#   bash scripts/sync/push_inference_session_to_minum.sh --all        # 전체 세션
#   bash scripts/sync/push_inference_session_to_minum.sh 20260702_100143  # 특정 세션 ID
#
# 환경변수:
#   MINUM_HOST (기본 minum) · API (기본 http://localhost:8001) · VLA_API_KEY

set -euo pipefail

MINUM_HOST="${MINUM_HOST:-minum}"
API="${API:-http://localhost:8001}"
APIKEY="${VLA_API_KEY:-vla_devel_key_2026}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SESS_DIR="${LOCAL_ROOT}/docs/inference_sessions"
DATE_TAG="$(date +%Y%m%d)"
REMOTE_DIR="~/MoNaVLA/inference_sessions_recv/${DATE_TAG}"

cd "$LOCAL_ROOT"

# ── 1. 전송할 세션 선정 ─────────────────────────────────────────────────
N=1; MODE="recent"; SPECIFIC=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n)          N="$2"; shift 2 ;;
    --last-24h)  MODE="24h"; shift ;;
    --all)       MODE="all"; shift ;;
    *)           SPECIFIC+=("$1"); MODE="specific"; shift ;;
  esac
done

declare -a SESSIONS
case "$MODE" in
  specific) for id in "${SPECIFIC[@]}"; do
              f="${SESS_DIR}/session_${id}.h5"; [[ -f "$f" ]] || f="${SESS_DIR}/${id}"
              [[ -f "$f" ]] && SESSIONS+=("$f") || echo "  ⚠️ 없음: $id"
            done ;;
  all)      while IFS= read -r f; do SESSIONS+=("$f"); done < <(ls -1 "${SESS_DIR}"/session_*.h5 2>/dev/null) ;;
  24h)      while IFS= read -r f; do SESSIONS+=("$f"); done < <(find "${SESS_DIR}" -name "session_*.h5" -mtime -1 | sort) ;;
  recent)   while IFS= read -r f; do SESSIONS+=("$f"); done < <(ls -1t "${SESS_DIR}"/session_*.h5 2>/dev/null | head -n "$N") ;;
esac

[[ ${#SESSIONS[@]} -eq 0 ]] && { echo "❌ 전송할 세션 없음"; exit 1; }

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  추론 세션 → minum 전송 (${DATE_TAG})"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  세션 ${#SESSIONS[@]}개:"
for s in "${SESSIONS[@]}"; do echo "    $(du -h "$s"|cut -f1)  $(basename "$s")"; done

# ── 2. 서버 설정/모델 스냅샷 (/health) ──────────────────────────────────
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
HEALTH_JSON="${STAGE}/server_health.json"
if curl -sf "${API}/health" -H "X-API-Key: ${APIKEY}" -o "$HEALTH_JSON" 2>/dev/null; then
  echo "  ✓ 서버 설정 스냅샷 (/health)"
else
  echo "  ⚠️ 서버 /health 응답 없음 — 설정 스냅샷 생략" ; echo '{"error":"server offline"}' > "$HEALTH_JSON"
fi

# 활성 모델 ckpt 경로 추출
CKPT_PATH="$(python3 -c "import json;print(json.load(open('$HEALTH_JSON')).get('checkpoint_path') or '')" 2>/dev/null || true)"

# ── 3. 매니페스트 작성 ──────────────────────────────────────────────────
MANIFEST="${STAGE}/README.txt"
{
  echo "=== MoNaVLA 추론 세션 패키지 ($(date '+%Y-%m-%d %H:%M')) ==="
  echo "출처: $(whoami)@$(hostname) ($(ip addr show tailscale0 2>/dev/null | awk '/inet /{print $2}' | cut -d/ -f1))"
  echo ""
  echo "── 세션 (${#SESSIONS[@]}) ──"
  for s in "${SESSIONS[@]}"; do echo "  $(basename "$s")  ($(du -h "$s"|cut -f1))"; done
  echo ""
  echo "── 서버 설정/모델 스냅샷 (server_health.json 전문) ──"
  python3 -c "
import json
d=json.load(open('$HEALTH_JSON'))
for k in ['head','window','val_acc','checkpoint_path','stop_mode','grounding_skip_n','multi_prompt','fallback_prompts','cx_jump_filter','cx_jump_thresh']:
    print(f'  {k}: {d.get(k)}')
p=d.get('preview',{}); g=d.get('grounder',{})
print(f'  preview: enabled={p.get(\"enabled\")} hint_cx={p.get(\"hint_cx\")} max_retry={p.get(\"max_retry\")}')
print(f'  grounder: {g.get(\"model\")} {g.get(\"input_px\")}px phrase=\"{g.get(\"phrase\")}\"')
" 2>/dev/null || echo "  (health 파싱 실패)"
  echo ""
  echo "── H5 구조 ──"
  echo "  observations/images (N,720,1280,3) / actions (N,3)"
  echo "  grounding/bbox (N,4)=[cx,cy,area,has_bbox] / cached (N,) -1=없음 0=live 1=캐시 / latency_ms (N,)"
  echo ""
  echo "── 활성 모델 ckpt ── ${CKPT_PATH:-(불명)}"
  echo "── episode_log.csv ── 실주행 에피소드 기록 포함"
} > "$MANIFEST"
echo "  ✓ 매니페스트 작성"

# ── 4. 원격 디렉토리 생성 + 전송 ────────────────────────────────────────
ssh "${MINUM_HOST}" "mkdir -p ${REMOTE_DIR}/models" >/dev/null 2>&1 \
  || { echo "❌ minum 접속 실패 (${MINUM_HOST})"; exit 1; }

echo ""
echo "▶ 전송 → ${MINUM_HOST}:${REMOTE_DIR}/"
rsync -avh "${SESSIONS[@]}" "${MINUM_HOST}:${REMOTE_DIR}/" 2>&1 | tail -2
rsync -avh "$HEALTH_JSON" "$MANIFEST" "${MINUM_HOST}:${REMOTE_DIR}/" 2>&1 | tail -1
[[ -f "${LOCAL_ROOT}/logs/episode_log.csv" ]] && \
  rsync -avh "${LOCAL_ROOT}/logs/episode_log.csv" "${MINUM_HOST}:${REMOTE_DIR}/" 2>&1 | tail -1
if [[ -n "$CKPT_PATH" && -f "${LOCAL_ROOT}/${CKPT_PATH}" ]]; then
  rsync -avh "${LOCAL_ROOT}/${CKPT_PATH}" "${MINUM_HOST}:${REMOTE_DIR}/models/" 2>&1 | tail -1
  echo "  ✓ 모델 ckpt 전송: $(basename "$CKPT_PATH")"
fi

echo ""
echo "✅ 완료 → ${MINUM_HOST}:${REMOTE_DIR}/"
echo "  확인: ssh ${MINUM_HOST} 'ls -lh ${REMOTE_DIR}/'"

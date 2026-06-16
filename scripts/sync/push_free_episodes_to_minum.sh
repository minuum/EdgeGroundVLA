#!/usr/bin/env bash
# Free 에피소드 H5를 minum 서버로 rsync 전송
#
# 사용법:
#   bash scripts/sync/push_free_episodes_to_minum.sh            # 전체 (FL+FC+FR)
#   bash scripts/sync/push_free_episodes_to_minum.sh --dry-run  # 목록만 확인
#   bash scripts/sync/push_free_episodes_to_minum.sh --bg       # 백그라운드 실행 (세션 끊겨도 유지)
#   bash scripts/sync/push_free_episodes_to_minum.sh fc         # FC만
#   bash scripts/sync/push_free_episodes_to_minum.sh fl fr      # FL+FR만

set -euo pipefail

MINUM_HOST="${MINUM_HOST:-minum}"
MINUM_PATH="${MINUM_PATH:-/home/minum/26CS/MoNaVLA}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOCAL_DATASET="${LOCAL_ROOT}/ROS_action/mobile_vla_dataset_v5"
LOG_FILE="${LOCAL_ROOT}/logs/push_free_minum.log"

DRY_RUN=false
BG_MODE=false
TARGETS=()

for arg in "$@"; do
  case "$arg" in
    --dry-run|-n) DRY_RUN=true ;;
    --bg)         BG_MODE=true ;;
    fc|FC) TARGETS+=("free_center") ;;
    fl|FL) TARGETS+=("free_left") ;;
    fr|FR) TARGETS+=("free_right") ;;
  esac
done

# 백그라운드 모드: 자기 자신을 nohup으로 재실행 (--bg 빼고)
if $BG_MODE; then
  mkdir -p "$(dirname "$LOG_FILE")"
  RERUN_ARGS=()
  for arg in "$@"; do [[ "$arg" != "--bg" ]] && RERUN_ARGS+=("$arg"); done
  nohup bash "${BASH_SOURCE[0]}" "${RERUN_ARGS[@]+"${RERUN_ARGS[@]}"}" \
    > "$LOG_FILE" 2>&1 &
  BG_PID=$!
  echo "백그라운드 전송 시작 (PID: ${BG_PID})"
  echo "로그: tail -f ${LOG_FILE}"
  echo "중단: kill ${BG_PID}"
  disown $BG_PID
  exit 0
fi

# 전송 파일 수집
FILES=()
for f in "${LOCAL_DATASET}"/episode_*free_*.h5; do
  [[ -f "$f" ]] || continue
  if [[ ${#TARGETS[@]} -eq 0 ]]; then
    FILES+=("$f")
  else
    for t in "${TARGETS[@]}"; do
      [[ "$f" == *"${t}"* ]] && FILES+=("$f") && break
    done
  fi
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  soda → minum: Free 에피소드 전송"
echo "  목적지: ${MINUM_HOST}:${MINUM_PATH}/ROS_action/mobile_vla_dataset_v5/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "전송할 free 에피소드가 없습니다."
  exit 1
fi

echo "전송 파일 (${#FILES[@]}개):"
for f in "${FILES[@]}"; do
  size=$(du -sh "$f" | cut -f1)
  tag=$(basename "$f" | sed 's/episode_[0-9_]*_//' | sed 's/__core.*$//')
  echo "  ${size}  ${tag}"
done
echo ""
TOTAL=$(du -ch "${FILES[@]}" | tail -1 | cut -f1)
echo "총 크기: ${TOTAL}"
echo ""

if $DRY_RUN; then
  echo "[DRY RUN] 실제 전송하지 않음."
  exit 0
fi

rsync -avz --progress --partial \
  "${FILES[@]}" \
  "${MINUM_HOST}:${MINUM_PATH}/ROS_action/mobile_vla_dataset_v5/"

echo ""
echo "✅ 전송 완료 ($(date '+%Y-%m-%d %H:%M:%S'))"
echo ""
echo "minum 확인:"
ssh "${MINUM_HOST}" "ls ${MINUM_PATH}/ROS_action/mobile_vla_dataset_v5/ | grep free_ | sort"

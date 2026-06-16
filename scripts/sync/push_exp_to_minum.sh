#!/usr/bin/env bash
# MoNaVLA 실험 체크포인트를 minum 서버로 전송
#
# 사용법:
#   bash scripts/sync/push_exp_to_minum.sh exp38
#   bash scripts/sync/push_exp_to_minum.sh v5_exp38          # v5_ 접두사 허용
#   bash scripts/sync/push_exp_to_minum.sh exp35 --all       # 모든 ckpt 전송 (best만 기본)
#   bash scripts/sync/push_exp_to_minum.sh exp35 exp36 exp37 exp38  # 복수 실험

set -euo pipefail

# ── 설정 ──────────────────────────────────────────────────────────────
MINUM_HOST="${MINUM_HOST:-minum}"
MINUM_PATH="${MINUM_PATH:-/home/billy/25-1kp/MoNaVLA}"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNS_ROOT="${LOCAL_ROOT}/runs/v5_nav/kosmos"
CONFIGS_ROOT="${LOCAL_ROOT}/configs"
# ──────────────────────────────────────────────────────────────────────

SEND_ALL=false
EXPS=()

for arg in "$@"; do
  if [[ "$arg" == "--all" ]]; then
    SEND_ALL=true
  else
    # v5_exp38 → exp38, exp38 → exp38
    exp="${arg#v5_}"
    EXPS+=("$exp")
  fi
done

if [[ ${#EXPS[@]} -eq 0 ]]; then
  echo "사용법: bash $0 exp38 [exp35 exp36 ...] [--all]"
  echo "  --all  : best ckpt만이 아닌 전체 run 디렉토리 전송"
  exit 1
fi

# ── best ckpt 선택 (val_loss 가장 낮은 ckpt 파일명 기준) ──────────────
best_ckpt_in_dir() {
  local dir="$1"
  # 파일명에서 val_loss=X.XXX 파싱해서 최솟값 선택
  local best
  best=$(find "$dir" -maxdepth 1 -name "epoch_epoch=*val_loss=*.ckpt" \
    | grep -oP "val_loss=\K[\d.]+" \
    | paste - <(find "$dir" -maxdepth 1 -name "epoch_epoch=*val_loss=*.ckpt") \
    | sort -n | head -1 | awk '{print $2}')
  # epoch_ 파일이 없으면 last.ckpt fallback
  if [[ -z "$best" ]]; then
    best="${dir}/last.ckpt"
  fi
  echo "$best"
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Billy → minum: Checkpoint 전송"
echo "  대상 서버: ${MINUM_HOST}:${MINUM_PATH}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

RSYNC_SOURCES=()

for exp in "${EXPS[@]}"; do
  full_name="mobile_vla_v5_${exp}"
  run_dir="${RUNS_ROOT}/${full_name}"

  if [[ ! -d "$run_dir" ]]; then
    echo "❌ run 디렉토리 없음: $run_dir"
    continue
  fi

  echo "── ${exp} ─────────────────────────────────────"

  # config 파일 찾기 (정확 매칭 우선, 없으면 glob)
  config_exact="${CONFIGS_ROOT}/mobile_vla_v5_${exp}.json"
  config_file=""
  if [[ -f "$config_exact" ]]; then
    config_file="$config_exact"
  else
    config_file=$(find "$CONFIGS_ROOT" -maxdepth 1 -name "mobile_vla_v5_${exp}_*.json" | sort | head -1)
  fi

  if [[ -n "$config_file" ]]; then
    echo "  config : $(basename "$config_file")"
    RSYNC_SOURCES+=("$config_file")
  else
    echo "  config : (없음)"
  fi

  if $SEND_ALL; then
    echo "  ckpts  : 전체 run 디렉토리"
    RSYNC_SOURCES+=("$run_dir")
  else
    # 날짜 디렉토리 하위에서 best ckpt 탐색
    while IFS= read -r dated_dir; do
      while IFS= read -r subdir; do
        ckpt="$(best_ckpt_in_dir "$subdir")"
        if [[ -f "$ckpt" ]]; then
          size=$(du -sh "$ckpt" | cut -f1)
          echo "  best   : $(basename "$ckpt") (${size})"
          RSYNC_SOURCES+=("$ckpt")
        fi
      done < <(find "$dated_dir" -mindepth 1 -maxdepth 1 -type d)
    done < <(find "$run_dir" -mindepth 1 -maxdepth 1 -type d)
  fi
  echo ""
done

if [[ ${#RSYNC_SOURCES[@]} -eq 0 ]]; then
  echo "전송할 파일이 없습니다."
  exit 1
fi

echo "전송 파일 수: ${#RSYNC_SOURCES[@]}"
echo ""

rsync -avz --progress --relative \
  "${RSYNC_SOURCES[@]}" \
  "${MINUM_HOST}:${MINUM_PATH}/"

echo ""
echo "✅ 전송 완료"
echo "minum에서 확인:"
for exp in "${EXPS[@]}"; do
  echo "  ls ${MINUM_PATH}/runs/v5_nav/kosmos/mobile_vla_v5_${exp}/"
done

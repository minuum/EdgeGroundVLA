#!/bin/bash
# Stage2 v2 핵심 파일을 soda 서버로 rsync
# 같은 레포 구조 → --relative로 경로 그대로 유지
#
# 사용법:
#   bash scripts/deploy/rsync_stage2_v2.sh          # best 모델만 (기본)
#   bash scripts/deploy/rsync_stage2_v2.sh --all    # ablation ckpt 전부

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

REMOTE="soda@100.85.118.58:~/MoNaVLA"
INCLUDE_ALL=0
[[ "${1:-}" == "--all" ]] && INCLUDE_ALL=1

# 전송 파일 목록
FILES=(
    # Stage1 encoder (3.1MB)
    "runs/v5_nav/mlp/exp54/stage1_v2/stage1_v2_projs.pt"

    # Stage2 best model (456KB, exp66 base PG2, 96.6% CL)
    "runs/v5_nav/mlp/exp54/stage2_v2/stage2_v2_mlp_base_pg2_aug.pt"

    # 추론 서버 + 학습/평가 스크립트
    "robovlm_nav/serve/stage2_v2_inference_server.py"
    "scripts/train_exp54_stage2_v2_action.py"
    "scripts/eval_exp54_stage2_v2_closedloop.py"

    # 모델 파라미터 요약
    "docs/v5/DEPLOY_MANIFEST.json"
)

if [[ $INCLUDE_ALL -eq 1 ]]; then
    # ablation ckpt 전부 (head 4종 + window 변형)
    while IFS= read -r pt; do
        FILES+=("$pt")
    done < <(find runs/v5_nav/mlp/exp54/stage2_v2 -name "*.pt" | sort)
fi

echo "==> $REMOTE"
echo ""
for f in "${FILES[@]}"; do
    [[ -f "$f" ]] && echo "  $(du -sh "$f" | cut -f1)  $f" || echo "  [없음]   $f"
done
echo ""

rsync -avz --relative "${FILES[@]}" "$REMOTE/"

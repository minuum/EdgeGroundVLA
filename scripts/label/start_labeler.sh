#!/usr/bin/env bash
# CH54 프리뷰 후보 O/X 라벨러 (포트 7793) 재시작 스크립트.
# 이미 떠있으면 죽이고 새로 띄운다. alias: labeler
set -u
ROOT="/home/minum/26CS/MoNaVLA"
pkill -f serve_hsv_owlv2_labeler.py 2>/dev/null
sleep 0.5
nohup "$ROOT/.venv/bin/python3" "$ROOT/scripts/label/serve_hsv_owlv2_labeler.py" \
  > /tmp/labeler_7793.log 2>&1 &
disown
sleep 1
if curl -sf -o /dev/null http://localhost:7793/; then
  echo "라벨러 실행 중 → http://localhost:7793  (로그: /tmp/labeler_7793.log)"
else
  echo "시작 실패 — /tmp/labeler_7793.log 확인" >&2
  exit 1
fi

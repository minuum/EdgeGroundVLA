#!/usr/bin/env bash
# Step 2: 커널 raw 읽기 — 스틱/버튼 누르면 hex 나와야 함. Ctrl+C 종료
echo "=== /dev/input/js0 raw (Ctrl+C 종료) ==="
echo "아무것도 안 나오면 → MODE/ANALOG 버튼 눌러보세요"
echo ""
cat /dev/input/js0 | xxd

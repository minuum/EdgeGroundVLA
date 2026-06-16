#!/usr/bin/env bash
# Step 1: 장치 존재 & 권한 확인
echo "=== /dev/input/js* ==="
ls -la /dev/input/js* 2>/dev/null || echo "❌ js0 없음"

echo ""
echo "=== 내 그룹 ==="
groups

echo ""
echo "=== js0 읽기 권한 ==="
if [ -r /dev/input/js0 ]; then
    echo "✅ js0 읽기 가능"
else
    echo "❌ js0 읽기 불가 — sudo usermod -aG input soda 후 재로그인 필요"
fi

echo ""
echo "=== 커널이 인식한 컨트롤러 ==="
cat /proc/bus/input/devices | grep -A 5 "js0"

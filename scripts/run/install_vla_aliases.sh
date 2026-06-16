#!/bin/bash
# VLA Aliases 설치 스크립트

echo "╔════════════════════════════════════════╗"
echo "║   VLA Aliases 설치 스크립트            ║"
echo "╚════════════════════════════════════════╝"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLA_DIR="${VLA_PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
BASHRC="$HOME/.bashrc"
ALIAS_FILE="$VLA_DIR/.vla_aliases"

# 1. Alias 파일 존재 확인
if [ ! -f "$ALIAS_FILE" ]; then
    echo "❌ Error: $ALIAS_FILE not found"
    exit 1
fi

echo "✅ Found alias file: $ALIAS_FILE"

# 2. .bashrc에 이미 추가되었는지 확인
if grep -q "source.*\.vla_aliases" "$BASHRC"; then
    echo "⚠️  VLA aliases already installed in ~/.bashrc"
    echo ""
    read -p "다시 추가하시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Installation cancelled"
        exit 0
    fi
fi

# 3. .bashrc에 추가
echo "" >> "$BASHRC"
echo "# ==================== VLA Project Aliases ====================" >> "$BASHRC"
echo "# Auto-generated on $(date)" >> "$BASHRC"
echo "if [ -f $ALIAS_FILE ]; then" >> "$BASHRC"
echo "    source $ALIAS_FILE" >> "$BASHRC"
echo "fi" >> "$BASHRC"
echo "" >> "$BASHRC"

echo "✅ VLA aliases added to ~/.bashrc"

# 4. 즉시 적용
source "$ALIAS_FILE"
echo "✅ Aliases loaded in current session"

# 5. 설치 완료 메시지
echo ""
echo "╔════════════════════════════════════════╗"
echo "║        설치 완료! 🎉                   ║"
echo "╚════════════════════════════════════════╝"
echo ""
echo "📝 사용 방법:"
echo "   - 현재 터미널: 바로 사용 가능"
echo "   - 새 터미널: 자동으로 로드됨"
echo ""
echo "💡 빠른 시작:"
echo "   vla-help      - 전체 명령어 목록"
echo "   vla-overview  - 프로젝트 상태 확인"
echo "   vla           - 프로젝트 디렉토리로 이동"
echo ""
echo "📖 상세 가이드: docs/VLA_ALIASES_GUIDE.md"
echo ""

# 6. 도움말 표시
vla-help

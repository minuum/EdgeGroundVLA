#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

show_path() {
  local label="$1"
  local path="$2"

  if [ -e "$path" ]; then
    printf "%-24s %s\n" "$label" "$path"
  else
    printf "%-24s MISSING %s\n" "$label" "$path"
  fi
}

echo "MoNaVLA memory context"
show_path "workspace" "$ROOT_DIR"
show_path "entrypoint" "$ROOT_DIR/docs/AGENT_ENTRYPOINT.md"
show_path "sync_map" "$ROOT_DIR/docs/MEMORY_SYNC_MAP.md"
show_path "agent_skill" "$ROOT_DIR/.agent/skills/memory-sync-hub/SKILL.md"
show_path "menemory_core" "$ROOT_DIR/.menemory/core/master_memory.md"
show_path "menemory_session" "$ROOT_DIR/.menemory/sessions/active_session.json"
show_path "claude_memory" "/home/soda/.claude/projects/-home-soda-MoNaVLA/memory/MEMORY.md"
show_path "codex_history" "/home/soda/.codex/history.jsonl"
show_path "codex_memories" "/home/soda/.codex/memories"
show_path "antigravity_brain" "/home/soda/.gemini/antigravity-ide/brain"
show_path "antigravity_index" "/home/soda/.gemini/antigravity-ide/conversations"
show_path "antigravity_srv" "/home/soda/.antigravity-server"

if command -v menemory >/dev/null 2>&1; then
  echo
  echo "menemory_status"
  menemory status
fi

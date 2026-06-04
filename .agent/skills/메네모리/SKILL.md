---
name: 메네모리
description: Project-local memory sync alias. Use when the user just says '메네모리' and wants prior MoNaVLA context recovered across Menemory, Claude, Codex, and Antigravity.
---

# 메네모리

사용자가 `메네모리` 라고만 입력해도 이 스킬을 사용한다.

## Required Order

1. `docs/MEMORY_SYNC_MAP.md`
2. `.agent/skills/memory-sync-hub/SKILL.md`
3. `scripts/utils/collect_memory_context.sh`
4. `.menemory/core/master_memory.md`

## Rules

- Antigravity 본문은 `~/.gemini/antigravity-ide/brain/<uuid>/` 에서 읽는다.
- `.pb` 파일은 인덱스일 뿐이다.
- 이번 세션 진행은 Menemory session에 요약 저장한다.
- Claude memory / Menemory / Codex / Antigravity 역할을 혼동하지 않는다.

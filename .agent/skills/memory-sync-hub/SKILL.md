---
name: memory-sync-hub
description: Recover and reconcile MoNaVLA context across Menemory, Claude memory, Codex logs, and Antigravity recovery artifacts. Use at session start, handoff, resume, or when prior decisions are unclear.
---

# Memory Sync Hub

이 스킬은 MoNaVLA의 여러 AI 메모리 시스템을 하나의 조회 절차로 묶는다.
새 세션, 세션 복구, handoff, 오래된 의사결정 추적 시 항상 우선 사용한다.

## Read First

1. `docs/AGENT_ENTRYPOINT.md`
2. `docs/MEMORY_SYNC_MAP.md`
3. `.menemory/core/master_memory.md`

## Quick Start

먼저 아래를 실행해 현재 접근 가능한 메모리 경로를 확인한다.

```bash
scripts/utils/collect_memory_context.sh
menemory status
```

## Source Priority

1. **현재 규칙/진입점**
   - `CLAUDE.md`
   - `docs/AGENT_ENTRYPOINT.md`
2. **장기 메모리**
   - `.menemory/core/master_memory.md`
   - `.menemory/core/memory_systems_integration.md`
3. **현재 세션 상태**
   - `.menemory/sessions/active_session.json`
   - `menemory status`
4. **프로젝트별 사용자 규칙**
   - `~/.claude/projects/-home-soda-MoNaVLA/memory/MEMORY.md`
   - `user_profile.md`, `feedback_workflow.md`, `project_monavla.md`
5. **과거 대화/명령 흔적**
   - `~/.codex/history.jsonl`
   - 필요 시 `~/.codex/memories/`
6. **복구 전용 원문**
   - `docs/antigravity_recovery_20260318.md`
   - `~/.gemini/antigravity-ide/brain/<uuid>/`

## Recovery Rules

- Antigravity의 `.pb` 파일은 세션 인덱스다. 본문은 `brain/<uuid>/` 에서 읽는다.
- Claude memory와 Menemory core에 같은 내용을 중복 저장하지 않는다.
- 대화 로그를 시스템 간에 통째로 복사하지 않는다. 요약만 남긴다.
- 문서가 충돌하면 최신 워크스페이스 상태와 실행 로그를 우선한다.

## Save Rules

- 장기적이고 재사용될 프로젝트 원칙:
  - `.menemory/core/`
- 현재 작업 handoff / 진행 요약:
  - Menemory session (`menemory add`)
- 사용자 취향, 협업 규칙:
  - Claude project memory
- 실험 결과/보고서:
  - repo 문서(`docs/`, `docs/v5/`)

워크스페이스 밖 메모리(예: `~/.claude_MINU/...`)를 수정해야 하는데 권한이 없으면,
즉시 follow-up 항목으로 남기고 워크스페이스 쪽 계약 문서부터 갱신한다.

## Expected Output

이 스킬을 쓴 뒤에는 최소한 아래를 분리해서 보고한다.

1. 어떤 시스템에서 확인한 사실인지
2. 최신성 위험이 있는 부분이 무엇인지
3. 지금 세션에 저장한 내용과 아직 남은 동기화 작업이 무엇인지

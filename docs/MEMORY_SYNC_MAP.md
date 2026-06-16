# Memory Sync Map
작성일: 2026-04-26

## 목적

MoNaVLA에서 `Claude`, `Codex`, `Antigravity`, `Menemory`를 한 시스템처럼
조회하기 위한 공용 계약 문서다. 어떤 AI를 켜더라도 이 문서와
`.agent/skills/memory-sync-hub/SKILL.md`를 읽으면 같은 절차로 과거 맥락을
복구하고, 어디에 무엇을 저장할지 판단할 수 있어야 한다.

## 공용 읽기 순서

1. [docs/AGENT_ENTRYPOINT.md](/home/soda/MoNaVLA/docs/AGENT_ENTRYPOINT.md:1)
2. 이 문서
3. `.agent/skills/memory-sync-hub/SKILL.md`
4. `.menemory/core/master_memory.md`
5. 필요 시 개별 메모리 시스템 원본

## 시스템별 실제 위치

| 시스템 | 실제 위치 | 역할 | 비고 |
|---|---|---|---|
| Menemory core | `.menemory/core/master_memory.md` | 장기 목표, 아키텍처 원칙, 금지 규칙 | 워크스페이스 기준 1차 참조 |
| Menemory session | `.menemory/sessions/active_session.json` | 최근 세션 요약과 turn 기록 | `menemory status`, `menemory show` 사용 |
| Claude project memory | `~/.claude/projects/-home-soda-MoNaVLA/memory/` | 사용자 선호, 작업 규칙, 프로젝트 상태 | 프로젝트 격리 |
| Codex history | `~/.codex/history.jsonl` | 과거 대화/명령 흔적 | 대화 로그 검색용 |
| Codex local memory | `~/.codex/memories/` | Codex 로컬 메모 | 도구/환경 종속 |
| Antigravity recovery | `~/.gemini/antigravity/brain/<uuid>/` | 복구 가능한 원문 아티팩트 | 실제 본문은 여기 |
| Antigravity conversation index | `~/.gemini/antigravity/conversations/*.pb` | 세션 인덱스 | 본문 원천 아님 |
| Antigravity runtime | `~/.antigravity-server/` | 서버 런타임/로그 | 시스템 상태 확인용 |

## 저장 정책

| 정보 종류 | 저장 위치 | 규칙 |
|---|---|---|
| 장기 프로젝트 원칙 | `.menemory/core/master_memory.md` | 장기적으로 계속 유지될 내용만 |
| 현재 세션 진행 상황 | Menemory session | 요약형으로만 저장 |
| 사용자 성향/피드백 규칙 | Claude project memory | 프로젝트별 작업 방식만 |
| 명령/대화 흔적 | Codex history/logs | 자동 생성 로그로 취급 |
| 과거 복구용 문서 | `docs/antigravity_recovery_*.md`, `docs/recovered_antigravity/` | Antigravity 원문을 선별 반영 |

## 강제 규칙

- 네 시스템 사이에 대화 로그를 통째로 복사하지 않는다.
- `~/.gemini/antigravity/conversations/*.pb` 는 인덱스일 뿐, 본문 원천으로 쓰지 않는다.
- Claude memory와 Menemory core에 같은 내용을 중복 저장하지 않는다.
- 최신 실험 상태는 단일 문서 하나를 맹신하지 말고 `git status`, Menemory 세션,
  관련 보고서를 교차 확인한다.
- 워크스페이스 밖 경로를 수정해야 하면 권한/승인 가능 여부를 먼저 확인한다.

## 빠른 조회 명령

```bash
scripts/utils/collect_memory_context.sh
menemory status
rg -n "memory-sync|menemory|codex|claude|antigravity" docs .menemory .agent
rg -n "MoNaVLA|inference-integration|exp17|exp28|grounding" ~/.codex/history.jsonl
```

## 복구 우선순위

1. 현재 작업 규칙: `CLAUDE.md`, `docs/AGENT_ENTRYPOINT.md`
2. 장기 메모리: `.menemory/core/master_memory.md`
3. 현재 세션: `.menemory/sessions/active_session.json`, `menemory status`
4. 사용자/프로젝트 로컬 규칙: Claude project memory
5. 과거 작업 흔적: Codex history
6. 누락 문서 복구: Antigravity `brain/<uuid>/`

## 현재 알려진 주의사항

- `docs/AGENT_ENTRYPOINT.md` 의 프로젝트 상태 표는 과거 스냅샷일 수 있다.
- Claude project memory의 `project_monavla.md` 는 현재 브랜치 진행보다 오래된
  상태일 수 있다.
- Antigravity의 실제 복구 소스는 `~/.gemini/antigravity-ide/` 이며,
  `~/.antigravity/` 는 현재 기준 핵심 저장소가 아니다.

## AI handoff 체크리스트

- [ ] `docs/AGENT_ENTRYPOINT.md` 읽음
- [ ] `docs/MEMORY_SYNC_MAP.md` 읽음
- [ ] `.agent/skills/memory-sync-hub/SKILL.md` 읽음
- [ ] `scripts/utils/collect_memory_context.sh` 실행
- [ ] 필요한 경우 Menemory / Claude / Codex / Antigravity 순서로 추가 조회
- [ ] 장기 결정과 세션 요약을 어디에 남길지 구분 완료

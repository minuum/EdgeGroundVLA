---
name: dashboard-status-sync
description: Use when the user asks to refresh/update the dashboard's "최신현황" (live status) tab, or after a major event (commit, checkpoint deploy, session sent to minum, experiment finished) that should be reflected there. Regenerates docs/DASHBOARD_LIVE_STATUS.md, which the mona_dashboard.py (port 7800) "📡 최신현황" tab reads and displays.
---

# Dashboard Status Sync Skill

## When to Use
- User says "대시보드 최신현황 갱신해줘" / "현황 탭 업데이트해줘"
- Right after: a commit that changes serving behavior, a checkpoint swap on the
  inference server, a session bundle sent to minum, a new finding/analysis doc
  written to `docs/v5/grounding_analysis/`

## What This Is NOT
- Not a cron job, not triggered automatically by the FastAPI server itself.
  The dashboard only *displays* whatever `docs/DASHBOARD_LIVE_STATUS.md`
  currently contains + its mtime ("최근 갱신: N분 전"). This skill is how
  that file gets rewritten.

## Steps

1. Gather current state:
   ```bash
   git log --oneline -8
   git rev-parse HEAD
   API_KEY=$(cat .vla_api_key); curl -s http://localhost:8001/health -H "X-API-Key: $API_KEY" | python3 -m json.tool
   tail -8 logs/episode_log.csv
   ls -t docs/v5/grounding_analysis/*.md docs/v5/closed_loop_eval/*.md 2>/dev/null | head -5
   git log --all --oneline --since="7 days ago" -- docs/v5  # 다른 브랜치(minum)에서 온 미병합 분석 문서도 확인
   ```
2. Rewrite `docs/DASHBOARD_LIVE_STATUS.md` (same structure as existing file —
   keep section headers: 현재 서빙 중인 체크포인트 / 할 일 / 대기 중 / 진행 중 A/B /
   최근 발견·이슈 / 최근 커밋) with fresh values. Update the "마지막 갱신" line
   at the top to today's date.
3. `docs/DASHBOARD_WIKI.md`는 "거의 안 바뀌는 참조 정보"라 평소엔 손대지 않음 —
   단, 체크포인트가 실제로 교체됐거나 새로운 상시성 gotcha(예: 특정 입력 구간에서
   재현되는 고장 패턴)가 발견되면 이 스킬을 실행할 때 같이 갱신해도 됨.
4. No restart or redeploy needed — the dashboard's `/wiki/status` (and `/journal`)
   endpoints read live on each tab open / refresh-button click.

## Source of Truth: git-default, conversation-sourced needs approval

**기본 원칙: 이 문서에 적는 사실은 git(커밋 로그/diff/`git show`)이나 위 1단계의
실측 명령(health, episode_log, grounding_analysis 문서) 등 검증 가능한 아티팩트에서
가져온다.** 이게 기본값이고, 별도 지시 없이 이 스킬을 실행할 때는 이 원천만 쓴다.

대화(현재 세션) 중에 이 스킬을 실행하면서, git으로 추적되지 않는 정보 — 예:
사용자가 방금 구두로 알려준 결정, 아직 커밋 안 된 다음 계획, 대화에서만 언급된
추측성 원인분석 — 을 `DASHBOARD_LIVE_STATUS.md`에 넣고 싶다고 판단되면, **먼저
사용자에게 넣어도 되는지 확인받는다.** git 근거가 없는 내용을 조용히 끼워넣지
말 것 — 나중에 "이거 어디서 나온 얘기야"를 git으로 추적할 수 없게 되면 이 문서의
신뢰성이 깨진다. 확인 없이 써도 되는 예외: 이미 승인된 위 워크플로(각 단계 완료
표시, plan.md 갱신)에서 자연스럽게 따라오는 요약.

## Markdown Syntax (rendered by `renderWikiMarkdown()` in mona_dashboard.py)

Both `docs/DASHBOARD_WIKI.md` and `docs/DASHBOARD_LIVE_STATUS.md` are rendered
client-side into `docs/v5/research_story.html`-styled HTML (chapters/callouts/
tables/image grids), not shown as raw text. Only this subset is supported —
stick to it so the tab renders correctly:

- `#`/`##`/`###` → headers. A leading `N.`/`N)` right after `##` or `#`
  (e.g. `## 1. 제목`) renders as a pill-badge chapter number — use for
  numbered sections only.
- `| a | b |` + `|---|---|` separator row → styled table (`.cl-table`)
- `- item` / `* item` (consecutive lines) → bullet list
- `` `code` `` inline, ` ```code block``` ` fenced → monospace
- `**bold**` → emphasis
- `> [!info|warn|critical|success]` followed by `>`-prefixed lines →
  color-coded callout box. Use `critical`/`warn` for blockers or gotchas,
  `success` for confirmed wins, `info` for neutral context.
- `![caption](relative/path.png)` → image card. Path resolves against
  `docs/v5/` (served at `/docs-static/v5/...`) — e.g. a file at
  `docs/v5/portfolio/exp_progression.png` is referenced as
  `![...](portfolio/exp_progression.png)`. **Only reference images that
  already exist** — do not invent paths; check with `ls docs/v5/**/*.png`
  first. Two or more consecutive image lines auto-lay out as a 3-column grid.

No other markdown (nested lists, links `[text](url)`, HTML passthrough) is
supported — it will render as literal text, not break the page, but won't
look right.

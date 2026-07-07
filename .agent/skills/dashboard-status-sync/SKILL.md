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
   git log --oneline -5
   git rev-parse HEAD
   API_KEY=$(cat .vla_api_key); curl -s http://localhost:8001/health -H "X-API-Key: $API_KEY" | python3 -m json.tool
   tail -5 logs/episode_log.csv
   ls -t docs/v5/grounding_analysis/*.md 2>/dev/null | head -3
   ```
2. Rewrite `docs/DASHBOARD_LIVE_STATUS.md` (same structure as existing file —
   keep section headers: 현재 서빙 중인 체크포인트 / 진행 중인 A/B 테스트 /
   최근 발견·이슈 / 최근 커밋 / 대기 중) with fresh values. Update the
   "마지막 갱신" line at the top to today's date.
3. Do not touch `docs/DASHBOARD_WIKI.md` — that's the static/rarely-changing
   info tab, edited manually only when CLAUDE.md's project context changes.
4. No restart or redeploy needed — the dashboard's `/wiki/status` endpoint
   reads the file live on each tab open / refresh-button click.

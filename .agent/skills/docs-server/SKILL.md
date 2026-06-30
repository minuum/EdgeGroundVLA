---
name: docs-server
description: MoNaVLA docs/ 로컬 웹서버 시작/확인/중지. "로컬 서버 띄워줘", "docs 서버", "localhost로 보고싶어" 요청에 사용. 포트 8765.
---

# Docs Local Server

`docs/` 디렉토리를 포트 8765로 서빙. GitHub Pages 대신 이미지 포함 전체 확인 가능.

## 서버 상태 확인

```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:8765/ 2>/dev/null || echo "down"
```

## 서버 시작 (이미 떠있으면 스킵)

```bash
# minum 터미널에서 직접 실행 (Claude가 0.0.0.0 바인딩 권한 없음)
cd ~/26CS/MoNaVLA/docs && python3 -m http.server 8765 &
```

> ⚠️ Claude Code 자동실행 제한으로 `0.0.0.0` 바인딩은 사용자가 직접 실행해야 함.
> 사용자에게 아래 명령어 안내:

```
! cd ~/26CS/MoNaVLA/docs && python3 -m http.server 8765
```

## 서버 중지

```bash
kill $(lsof -ti:8765) 2>/dev/null && echo "중지됨"
```

## 주요 페이지 로컬 주소

| 페이지 | 로컬 URL |
|--------|----------|
| 메인 | http://localhost:8765/ |
| Research Story | http://localhost:8765/v5/research_story.html |
| Grounding Hub | http://localhost:8765/v5/grounding_hub.html |
| CH51 | http://localhost:8765/v5/research_story.html#ch51 |
| CH57 | http://localhost:8765/v5/research_story.html#ch57 |
| CH58 | http://localhost:8765/v5/research_story.html#ch58 |

## Claude가 링크 공유 시 규칙

이미지가 포함된 분석 결과 공유 시 항상 두 주소 같이 제공:
- **로컬 (이미지 포함):** `http://localhost:8765/v5/...`
- **GitHub Pages (이미지 제외):** `https://minuum.github.io/MoNaVLA/v5/...`

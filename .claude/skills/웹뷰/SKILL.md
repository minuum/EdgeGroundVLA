---
name: 웹뷰
description: docs/*.html 페이지(revision_briefing.html, model_architecture_brief.html, research_story.html, 위키 등)를 언급하거나 결과를 보고할 때, 로컬호스트 프리뷰 서버(포트 8001)가 항상 떠 있도록 확인/기동하고 실제 접속 가능한 localhost URL을 항상 함께 제시한다. 사용자가 "웹뷰"라고만 쳐도 즉시 실행.
---

# 웹뷰 — docs/*.html 로컬 프리뷰 항상 띄우기

`docs/` 아래 HTML 문서(연구 브리핑, 위키, research_story.html 등)를 언급할 때마다
말로만 설명하지 않고 **로컬호스트 프리뷰 서버가 살아있는지 확인 → 없으면 기동 → 실제
URL을 응답에 포함**하는 걸 표준 동작으로 삼는다.

## 트리거
- 사용자가 `웹뷰`라고만 입력해도 즉시 아래 절차 실행
- `docs/*.html` 파일을 새로 만들거나 수정한 직후
- revision_briefing.html / model_architecture_brief.html / research_story.html /
  docs/wiki/* / index.html 등 결과를 사용자에게 보고할 때 — "이런 내용 반영했다"고
  말로만 끝내지 말고 항상 로컬 URL을 같이 제시

## 절차 (매번 이 순서)

1. **서버 생존 확인** (매번 새로 띄우지 말 것 — 포트 충돌/좀비 프로세스 방지):
   ```bash
   curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8001/index.html
   ```
   200이면 이미 떠 있는 것 — 재기동 불필요, 바로 3번으로.

2. **없으면(000/connection refused) 백그라운드로 기동**:
   ```bash
   cd /home/minum/26CS/MoNaVLA/docs && PYTHONUTF8=1 python3 -m http.server 8001 --bind 127.0.0.1 &
   ```
   (foreground로 실행하면 응답 못 하고 멈추므로 반드시 `run_in_background: true` 또는 `&`)

3. **방금 언급/수정한 파일이 실제로 200인지 확인**:
   ```bash
   curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:8001/<상대경로>"
   ```
   (예: `01.paper/revision_briefing.html`, `v5/research_story.html#ch72`,
   `wiki/paper-peer-review.html`)

4. **응답에 URL을 명시적으로 포함**해서 사용자에게 보고:
   - 형식: `http://127.0.0.1:8001/<상대경로>` (앵커 `#ch72` 등도 그대로 붙여서)
   - "반영했습니다"로 끝내지 말고 반드시 클릭 가능한 로컬 URL 한 줄을 덧붙일 것.
   - 여러 파일을 동시에 언급했다면 파일별로 URL을 각각 나열.

## 자주 쓰는 경로 매핑

| 문서 | 로컬 URL |
|---|---|
| 메인 랜딩 페이지 | http://127.0.0.1:8001/index.html |
| 논문 수정사항 브리핑 | http://127.0.0.1:8001/01.paper/revision_briefing.html |
| 모델 구조 브리핑 | http://127.0.0.1:8001/v5/model_architecture_brief.html |
| OWLv2 그라운더 브리핑 | http://127.0.0.1:8001/v5/owlv2_grounder_brief.html |
| 연구 스토리(CH별) | http://127.0.0.1:8001/v5/research_story.html |
| 연구 대시보드(구 아카이브) | http://127.0.0.1:8001/research_dashboard.html |
| 위키 인덱스 | http://127.0.0.1:8001/wiki/index.html |
| 위키 특정 주제 | http://127.0.0.1:8001/wiki/<slug>.html (예: paper-peer-review.html) |

## 하지 말 것
- 서버가 이미 떠 있는데 또 띄우지 말 것 (포트 충돌 에러만 남기고 기존 서버는 그대로 살아있음 — 확인만 하고 넘어갈 것)
- URL 없이 "브리핑에 반영했습니다"로만 끝내지 말 것 — 이 스킬의 존재 이유가 URL을 빠뜨리지 않는 것
- 200 확인 없이 URL만 추측해서 던지지 말 것 — curl로 실제 확인 후 제시

---
name: wiki-sync
description: research_story.html에 새 챕터(예 CH71)가 추가된 뒤 docs/wiki/(주제별 LLM wiki)를 안전하게 재동기화한다. "위키 갱신해줘", "CH71 위키에 반영해줘", "위키 동기화" 같은 요청에 사용.
---

# wiki-sync — 새 챕터 추가 후 위키 재동기화

`docs/wiki/`는 `docs/v5/research_story.html`(시간순 연구일지, 81+챕터)을 8개
주제로 재구성한 Karpathy 방식 LLM wiki다(`docs/plans/plan_20260828_llm_wiki.md`).
새 챕터를 research_story.html에 추가할 때마다 이 스킬로 위키를 갱신한다.

## 언제 쓰나
- `docs/v5/research_story.html`에 새 CH를 추가한 직후
- 사용자가 "위키에도 반영해줘", "위키 갱신", "CH71 위키 동기화" 등을 요청할 때
- 기존 챕터의 내용을 크게 수정했을 때(원문 발췌가 낡음)

## 절차

### 1. 오케스트레이터 실행 (기계적 부분 — 스크립트가 전부 처리)

```bash
.venv/bin/python3 scripts/wiki/sync_wiki.py
```

이 한 줄이 순서대로 다 한다:
1. `parse_research_story.py` — research_story.html 재파싱 → `chapters.json` 갱신
2. `build_wiki_pages.py` — 주제별 `.md` 재생성. **기존 "## 압축 요약" 섹션은
   절대 안 지워짐** — 새 챕터가 추가된 주제만 요약 위에 "⚠️ 새 챕터 추가됨"
   배지가 자동으로 붙음
3. `build_archive_index.py` — `docs/*.md` 스냅샷 카탈로그 재생성(357개+새 파일)
4. `render_wiki_html.py` — 정적 HTML 재렌더링(`.venv`의 markdown 패키지 필요 —
   없으면 `pip install -r scripts/wiki/requirements.txt`)

마지막에 스크립트가 "다음 확인 필요" 목록을 출력한다 — 이건 **스크립트가 절대
자동으로 하지 않는, 판단이 필요한 일**이다:
- 재압축 필요 주제 목록(새 챕터가 걸린 곳)
- 어떤 주제에도 안 걸린 새 챕터 목록

### 2. 새 챕터를 주제에 배정 (판단 필요 — 안 걸린 챕터가 있을 때만)

출력에 "안 걸린 챕터"가 있으면, 그 챕터의 제목/내용을 확인하고
`docs/wiki/data/topic_index.json`의 해당 주제 `chapter_ids` 배열에 id를 추가한다.
기존 8개 주제(그라운딩/액션헤드/언어조건화/flow-matching/데이터셋/실기테스트/
백본/미팅) 중 어디에도 안 맞으면 새 주제 키를 추가해도 된다(`title`/`summary`/
`chapter_ids` 3개 필드 필수 — 기존 항목 형식 그대로 따를 것).

배정 후 `sync_wiki.py`를 한 번 더 돌린다(이제 "안 걸린 챕터"가 사라져야 함).

### 3. 낡은 요약 재압축 (판단 필요 — "재압축 필요" 목록이 있을 때만)

목록에 있는 각 주제 파일(`docs/wiki/{slug}.md`)에 대해:
1. 파일을 **전체** Read(챕터 수가 많으면 offset 나눠서라도 끝까지)
2. `## 압축 요약` 섹션(⚠️ 배지 포함)을 다음 형식으로 교체:
   ```
   ## 압축 요약

   (5~12문장, Karpathy 방식 — "지금 이 주제에 대해 확정적으로 아는 것"을 압축.
   시간순 서사가 아니라 최종 결론 위주. 핵심 수치는 실제 값 인용.
   미해결/논쟁 중인 부분은 마지막 한 문장으로.)
   ```
3. "챕터별 원문 발췌" 섹션은 절대 손대지 않는다(스크립트가 관리하는 영역)

**파일이 크면(수만 자) `general-purpose` 서브에이전트에 위임 권장** — 이전
위키 최초 구축 세션에서 8개 주제를 병렬 서브에이전트로 압축했던 패턴을 그대로
재사용(각 에이전트에 파일 경로 + 위 지침을 그대로 전달). 새 챕터 1~2개만
추가된 경우처럼 변경 폭이 작으면 직접 읽고 기존 요약에 새 내용만 짧게
추가하는 것으로 충분 — 매번 전체를 새로 쓸 필요는 없다.

### 4. 재렌더링 + 커밋

```bash
.venv/bin/python3 scripts/wiki/render_wiki_html.py   # 요약 수정했으면 HTML도 재생성
git add docs/wiki/ scripts/wiki/
git commit -m "docs(wiki): sync CHxx into topic wiki"
git push origin inference-integration
```

## 원칙 (왜 이렇게 나뉘어 있나)

- **기계적 재구성(파싱/재배열/백링크)은 스크립트가 100% 결정론적으로 처리** —
  매번 똑같이 나와야 하고 사람 판단이 필요 없다.
- **"무엇을 아는가"의 압축은 반드시 LLM이 원문을 읽고 판단** — 스크립트가
  임의로 요약을 지어내면 환각 위험이 크므로 절대 자동화하지 않는다.
- 이 경계 때문에 `sync_wiki.py`는 "무엇을 해야 하는지 알려주는 것"까지만 하고,
  실제 압축/배정은 항상 이 스킬을 실행하는 세션(사람 또는 LLM)이 한다.

# 계획 — research_story.html(70챕터) 기반 주제별 LLM Wiki

## 확정된 스코프 (사용자 답변)

- **원천**: `docs/v5/research_story.html`(70챕터, `cl-*` 보조챕터 포함 81개 `chapter` div) +
  `docs/RESEARCH_STATUS.md`. `docs/*.md` 357개 스냅샷 파일은 압축 편입 없이
  **카탈로그만** 생성(2026-08-28 후속 결정, 아래 참고).

## 후속 결정 — 357개 스냅샷 파일 (2026-08-28)

조사 결과 이 파일들은 research_story.html이 다루는 시기(2026-05~)보다 **훨씬
이전(2024-08~2026-04)의 별개 프로젝트 단계** 기록(양자화/서빙 인프라, manipulation
vs navigation 액션 공간 비교, "환각 제거" 반복 수정 이력 등)으로 확인됨 — 지금
EdgeGround-VLA 네비게이션 연구와 직접 연관이 약함. 사용자 판단: "대부분 죽은
기록(폐기된 방향) — 아카이브로만 충분". 그래서 원문을 읽어 압축하는 대신
**`scripts/wiki/build_archive_index.py`로 제목/날짜/한줄요약만 뽑은 색인**
(`docs/wiki/archive-index.md`)만 생성 — 원본 파일은 이동/삭제 없이 그대로 두고,
필요시 직접 열어보는 찾아가기용으로만 사용.
- **용도**: ① 미래 Claude 세션/에이전트의 빠른 문맥 파악, ② 사용자 본인이 웹에서
  주제별 탐색, ③ RAG/검색 인덱스.
- **형식**: 주제별 HTML 페이지(GitHub Pages, research_story.html과 통일된 스타일).
  단, RAG/검색 요구를 만족시키려면 HTML과 별개로 **청크 단위 구조화 JSON**을
  중간 산출물로 둔다(HTML은 사람이 보는 최종 형태, JSON은 검색/재생성용 데이터 계층).

## 아키텍처

```
research_story.html (70챕터, 원본 그대로 유지)
        │  ① 파싱
        ▼
docs/wiki/data/chapters.json   ← 챕터별 {id, num, title, subtitle, date, finding_cards[]}
        │  ② 주제 태깅 (LLM이 각 챕터 요약 읽고 태그 부여, 반자동)
        ▼
docs/wiki/data/topic_index.json  ← {topic_slug: {title, chapter_ids[], summary}}
        │  ③ 페이지 생성 (주제별로 관련 챕터의 finding_card들을 모아 재구성)
        ▼
docs/wiki/<topic-slug>.html   ← 주제별 위키 페이지(각 항목은 원본 챕터 앵커로 백링크)
docs/wiki/index.html          ← 위키 진입점(주제 목록 + 검색창)
```

- **원본은 절대 안 건드림** — research_story.html은 지금처럼 시간순 연구일지로
  계속 성장(CH71, CH72...). 위키는 그 위에 얹는 읽기 전용 재구성 레이어.
- **재생성 가능해야 함** — 새 챕터가 추가되면(예: 오늘 한 것처럼 CH70을 CH69 뒤에
  추가) 파서를 다시 돌려 `chapters.json`을 갱신하고, 영향받는 주제 페이지만
  다시 생성. 전체를 매번 새로 안 만들어도 되게 증분 처리.

## 주제 태그 체계 (1차 안, 사용자 검토 필요)

챕터 제목/요약을 훑어본 결과로 제안하는 1차 분류(챕터 수는 대략치, 실제 파싱 후 확정):

| 주제 slug | 제목 | 관련 챕터(예시) |
|---|---|---|
| `grounding-detector` | 그라운딩/검출기(OWL-v2, Florence-2, phrase grounding) | CH64, 67, 69 |
| `action-head-architecture` | 액션헤드 구조·손실함수 실험 | CH70, CH68 일부 |
| `text-attention-dead-path` | 텍스트 어텐션 구조적 사망 | CH65~68 일부(project_text_path_dead 메모리 연계) |
| `dataset-v5-v6` | V5/V6 데이터셋, 라벨링, 수집 프로토콜 | CH61~63 대 |
| `real-robot-testing` | 실기 100건 테스트, closed-loop 검증 | CH64, 66, robot_tests |
| `backbone-model` | Kosmos-2/Google-robot/CLIP 백본 비교 | CH54~60 대 |
| `paper-direction` | 논문 방향(경량화), 미팅 결정사항 | meeting-* 챕터들 |

이 표는 챕터를 실제로 파싱해서 제목/요약을 전부 뽑아본 뒤 정확히 확정한다 —
지금은 이 세션의 기억에 의존한 추정치라 챕터 번호가 부정확할 수 있음.

## 구현 단계

### 1단계 — 파서 스크립트
`scripts/wiki/parse_research_story.py`: `research_story.html`을 정규식/BeautifulSoup으로
파싱해 각 `<div class="chapter" id="...">` 블록에서:
- 챕터 번호/제목/부제(날짜 포함)
- 그 안의 `<div class="finding-card" ...>` 각각의 제목·색상(심각도)·본문 텍스트
- 이미지(`<img src="...">`) 경로만 남기고 base64는 건너뜀(파일 크기 문제)

출력: `docs/wiki/data/chapters.json`

### 2단계 — 주제 태깅 (LLM 보조, 반자동)
파싱된 chapters.json을 놓고 Claude가 각 챕터를 읽어 위 태그 체계에 맞게
`topic_ids: []`를 부여(챕터 하나가 여러 주제에 속할 수 있음 — 예: CH69는
grounding-detector와 action-head 둘 다). 이 단계는 자동 키워드 매칭이 아니라
챕터 요약을 실제로 읽고 판단해야 정확함 — 스크립트가 아니라 대화형으로 진행.

출력: `docs/wiki/data/topic_index.json`

### 3단계 — 위키 페이지 생성
`scripts/wiki/build_wiki_pages.py`: topic_index.json 기준으로 관련 챕터의
finding_card들을 시간순으로 모아 주제별 HTML 페이지 생성. 각 항목은
`research_story.html#chXX`로 백링크(원문 확인용). research_story.html과
동일한 다크 테마/카드 스타일 재사용(공용 CSS 분리: `docs/wiki/wiki.css`).

출력: `docs/wiki/<topic-slug>.html` × 7개(1차 안 기준) + `docs/wiki/index.html`

### 4단계 — GitHub Pages 연결 + 검색
- `docs/index.html` 히어로 버튼에 위키 진입 링크 추가(CLAUDE.md 규칙)
- `docs/wiki/index.html`에 클라이언트 사이드 간단 검색(주제 제목/요약 대상,
  fuse.js 등 CDN 라이브러리 없이 순수 JS 문자열 매칭으로 시작 — 필요시 고도화)
- RAG용 산출물은 `topic_index.json` + `chapters.json` 그대로 사용 가능(별도
  벡터 인덱싱은 이번 스코프 밖, 필요해지면 후속 작업)

## 유지보수 방침

- 새 챕터 추가 시(CH71+) `parse_research_story.py`를 다시 돌려 chapters.json
  갱신 → 영향받는 주제만 2~3단계 재실행. **CLAUDE.md에 한 줄 규칙 추가 제안**:
  "새 챕터를 research_story.html에 추가하면 관련 주제 위키 페이지도 함께 갱신한다"
  (사용자 승인 시 CLAUDE.md 수정은 별도 확인받고 진행)

## 형식 결정 (2026-08-28, 사용자 승인)

WebSearch로 2026-08 기준 트렌드 확인 — Karpathy가 2026-04-03 공개한 "LLM wiki"
(raw/wiki/index.md 3폴더, 벡터DB 없음, "지식이 유한·안정적일 때" RAG보다 효율적)가
우리 규모(70챕터, ~19k줄, 컨텍스트에 넉넉히 들어감)에 정확히 맞는 사례. **원천
진실은 markdown 위키**(`docs/wiki/*.md` + `index.md`)로 만들고, GitHub Pages
브라우징용으로는 이 markdown을 `.venv`의 `markdown` 패키지로 빌드 시점에
정적 HTML로 사전 렌더링(클라이언트 JS 파서 없음, 이 프로젝트 다른 docs/*.html과
동일한 정적 파일 방식)해서 이중 유지보수 없이 사람/에이전트 양쪽 다 만족시킴.

## 완료 기준

- [x] 파서 스크립트로 81개 챕터 전부 파싱, `chapters.json` 생성 확인
      (`scripts/wiki/parse_research_story.py` — div 정규식/트리 추적 둘 다 실패해서
      "마커 간 텍스트 구간" 방식으로 재설계, CH6→CH66 통삼킴 버그 해결)
- [x] 주제 태그 1차 안을 실제 81개 챕터 제목 확인 후 확정 → `topic_index.json`
      (8개 주제: grounding-detector·action-head-architecture·language-conditioning·
      flow-matching-vla·dataset-collection·real-robot-closed-loop·backbone-model·
      meetings-and-direction, 79/81챕터 커버 — `vis`는 이미지 전용이라 제외)
- [x] markdown 위키 페이지 생성(`scripts/wiki/build_wiki_pages.py`) + 정적 HTML
      렌더링(`scripts/wiki/render_wiki_html.py`), 로컬 확인(http.server, 전부 200)
- [x] `docs/index.html` docs-grid에 위키 링크 추가
- [x] 각 주제 페이지 맨 위 "압축 요약" 섹션 채움 — 8개 주제 병렬 서브에이전트가
      각 주제 파일을 전체 정독하고 Karpathy 방식대로 "지금 확정적으로 아는 것"을
      5~12문장으로 압축(예: grounding-detector — "실주행 성패는 액션헤드가 아니라
      그라운딩 가용성"이 핵심 결론; action-head-architecture — "헤드 구조 축은
      한계, 손실함수(ordinal soft label)만 실질 개선"). 이제 특정 주제를 CH 번호
      몰라도 위키 페이지 하나만 읽으면 파악 가능 — 이번 세션에서 겪은 grep 반복
      비효율이 실제로 해소됨.

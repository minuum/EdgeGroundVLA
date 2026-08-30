# 연구 위키 & 문서 인프라

<p class="tagline">research_story.html을 주제별로 재구성한 LLM wiki 구축(Karpathy 방식), 새 챕터 추가 시 안전 재동기화(wiki-sync 스킬), 아카이브 색인 등 문서화 도구 자체에 대한 기록.</p>

<div class="summary-box" markdown="1">

**압축 요약**

`docs/v5/research_story.html`(81+챕터, 시간순 연구일지)은 특정 주제를 찾으려면
전체를 grep해야 하는 비효율이 있어, **주제별로 재구성한 markdown 위키**
(`docs/wiki/`)를 구축했다. 형식은 2026-04 공개된 Karpathy의 "LLM wiki"
방식(raw/wiki/index.md 3계층, 벡터DB·임베딩 없음)을 채택했다 — 우리 규모
(81챕터, 컨텍스트 한 번에 들어감)에서는 RAG보다 이쪽이 효율적이라는 판단.

파싱은 두 번 실패했다(정규식 div 깊이 추적, `html.parser` 트리 추적) — 원인은
19,123줄짜리 파일 자체에 실제 `<div>`/`</div>` 짝이 안 맞는 부분이 있어서였다
(전체 합계는 우연히 맞아떨어져 단순 카운트 체크로는 안 잡힘). **"챕터/카드는
다음 마커 전까지의 텍스트 구간"**으로 재정의해 해결 — 정확한 HTML 트리 없이도
81챕터·301개 카드 전부 안전하게 추출된다.

주제는 챕터 실제 제목을 다 확인한 뒤 8개(그라운딩/액션헤드/언어조건화/
flow-matching/데이터셋/실기테스트/백본/미팅)로 확정했고, 각 페이지 상단
압축 요약은 **general-purpose 서브에이전트 8개를 병렬로 launch**해서 각자
자기 주제 파일을 전체 정독하고 작성하게 했다(직접 순차로 읽는 것보다 훨씬
빠름). `docs/*.md` 357개 스냅샷(2024-08~2026-04, 별개 프로젝트 단계로 확인)은
압축 없이 제목/날짜/한줄요약만 카탈로그화했다.

새 챕터가 계속 추가되는 프로젝트 특성상 **`wiki-sync` 스킬**로 재동기화
절차를 고정했다 — 핵심은 재실행해도 기존 압축 요약을 절대 안 지우고,
새 챕터가 걸린 주제만 배지로 표시해 재압축 대상을 알려주는 것(기계적
재구성/LLM 판단의 경계를 명확히 분리). CH71(이 챕터 자체)을 실제로 추가해
전체 파이프라인(파싱→배정→재압축→렌더링)을 end-to-end로 검증했다.

</div>

## 챕터별 원문 발췌 (시간순)

<div class="chapter-block accent-a" markdown="1">

<div class="chapter-block-head"><span class="chapter-badge">CH 71</span> LLM Wiki 구축 — 81챕터를 주제별로 재구성한 Karpathy 방식 위키</div>

<p class="chapter-subtitle-line">2026-08-28~29 ·
research_story.html(시간순 연구일지)을 주제별로 조회 가능하게 재구성한 docs/wiki/
구축. 2026-04 공개된 Karpathy LLM-wiki 방식(raw/wiki/index, 벡터DB 없음)을 채택하고,
새 챕터 추가 시 안전하게 재동기화하는 wiki-sync 스킬까지 만들었다.</p>

<div class="card" markdown="1">

🟢 3줄 요약
① 이 세션에서 CH69/70을 찾으려고 grep을 반복했던 비효율을 계기로, research_story.html
81챕터를 8개 주제(그라운딩/액션헤드/언어조건화/flow-matching/데이터셋/실기테스트/백본/미팅)로
재구성한 markdown 위키를 만들었다.
② 형식은 2026-04 공개된 Karpathy의 "LLM wiki"(raw/wiki/index.md 3계층, 벡터DB 없음)를
채택 — 우리 규모(81챕터, 컨텍스트에 들어감)에 RAG보다 적합하다고 판단.
③ 각 주제 페이지 상단에 병렬 서브에이전트 8개가 전체 정독 후 쓴 압축 요약을 채웠고,
새 챕터가 추가돼도 기존 요약을 안 지우고 안전하게 재동기화하는 wiki-sync
스킬까지 구축·검증했다.

</div>

<div class="card" markdown="1">

**📐 71-1. 파싱 — div 트리 추적 2번 실패 후 "마커 간 텍스트 구간" 방식으로 해결**

81개 `chapter` div와 그 안의 finding-card/callout을 뽑아내려고 두 가지를 시도했다가
둘 다 실패했다: ① 정규식 기반 div 중첩 깊이 추적, ② 표준 `html.parser.HTMLParser`
트리 추적. 둘 다 CH7이 CH66까지 통째로 삼켜버리는
버그가 재현됐다 — 원인은 파일 자체(19,123줄, 여러 세션에 걸친 수작업 편집
히스토리)에 실제 ``/`
` 짝이 안 맞는 부분이 있어서였다(파일 전체 합계는
우연히 맞아떨어짐, `open div == close div` 체크로는 안 잡힘).
해결: 정확한 HTML 트리가 필요 없다는 데 착안해 "챕터/카드는
다음 같은 종류 마커가 나오기 전까지의 텍스트 구간"으로 재정의(`scripts/wiki/parse_research_story.py`).
div 짝이 깨져 있어도 마커 자체(챕터 시작 태그, finding-card/callout 시작 태그)의
선형 순서만 있으면 안전하게 파싱된다 — 81개 챕터, 301개 finding-card/callout 전부
정확히 추출됨.

</div>

<div class="card" markdown="1">

**🗂 71-2. 주제 태깅 + 압축 요약 — 8개 주제, 서브에이전트 병렬 정독**

81개 챕터를 실제 제목을 다 확인한 뒤 8개 주제로 분류(`docs/wiki/data/topic_index.json`,
79/81챕터 커버 — `vis`는 이미지 전용이라 의도적 제외). 각 주제 markdown 페이지는
챕터별 원문을 시간순으로 재배열한 뒤, 일반 목적(general-purpose)
서브에이전트 8개를 병렬로 launch해서 각자 자기 주제 파일을 전체 정독하고
"지금 이 주제에 대해 확정적으로 아는 것"을 5~12문장으로 압축하게 했다(Karpathy 방식
핵심 — 시간순 서사가 아니라 최종 결론 위주).
예: grounding-detector 요약은 "실주행 성패는 액션헤드가 아니라 그라운딩 가용성이
가른다(gnd%≥80 → 98.8% 성공)"로 시작하고, action-head-architecture 요약은
"헤드 구조 축은 한계, ordinal soft label(손실함수)만 실질 개선"(CH70)으로 시작한다 —
두 문장만 읽어도 그 챕터 번호를 몰라도 핵심을 파악할 수 있다.

</div>

<div class="card" markdown="1">

**📦 71-3. 아카이브 색인 — docs/*.md 357개 스냅샷은 압축 없이 카탈로그만**

`docs/*.md` 최상위 357개 파일도 편입을 검토했으나, 조사 결과 이들은
research_story.html이 다루는 시기(2026-05~)보다 훨씬 이전(2024-08~2026-04)의
별개 프로젝트 단계(양자화/서빙 인프라, manipulation
vs navigation 액션 공간 비교, "환각 제거" 반복 수정 이력 등) 기록으로 확인됐다 —
지금 EdgeGround-VLA 네비게이션 연구와 직접 연관이 약함. 사용자 판단으로
"대부분 죽은 기록 — 아카이브로만 충분"이라 결정, 원문을 읽어 압축하는 대신
제목/날짜/한줄요약만 뽑은 카탈로그(`docs/wiki/archive-index.md`, 357행)만 생성했다.
원본 파일은 이동·삭제 없이 그대로 둠.

</div>

<div class="card" markdown="1">

**🔁 71-4. wiki-sync 스킬 — 새 챕터 추가 시 압축 요약을 절대 안 지우는 안전한 재동기화**

새 챕터(CH71 같은)가 추가될 때마다 위키를 손으로 다시 만들 순 없으므로,
.claude/skills/wiki-sync/SKILL.md + scripts/wiki/sync_wiki.py로
절차를 스킬화했다. 핵심 안전장치: `build_wiki_pages.py`가 몇 번을 다시 돌아도
기존 "## 압축 요약" 섹션은 파일에서 읽어와 그대로 보존하고, "챕터별 원문 발췌"
섹션만 최신 `chapters.json` 기준으로 재생성한다. 새 챕터가 어떤 주제에 추가되면
그 요약 위에 "⚠️ 새 챕터 추가됨 — 재압축 필요" 배지를 자동으로 붙여 낡았음을 표시하고,
배지 자체는 다음 실행 때 요약 텍스트에서 다시 벗겨내(영구히 눌어붙지 않게) 매번
새로 판단한다.
기계적 재구성(파싱/재배열/백링크, 스크립트가 결정론적으로 처리)과
"무엇을 아는가"의 압축(반드시 LLM이 원문을 읽고 판단, 스크립트가 임의로 안 함)의
경계를 명확히 나눈 게 이 스킬의 설계 원칙 — `sync_wiki.py`는 "무엇을 해야 하는지"만
알려주고 실제 판단은 항상 스킬을 실행하는 세션이 한다.
검증: 가짜 챕터를 임시로 한 주제에 추가했다가 제거하는 실험으로 stale 배지
생성/제거가 정확히 작동함을 확인했고, 그 과정에서 배지 텍스트가 요약에 영구히
눌어붙는 실제 버그 하나를 잡아 수정했다.
스크립트
scripts/wiki/parse_research_story.py ·
scripts/wiki/build_wiki_pages.py ·
scripts/wiki/build_archive_index.py ·
scripts/wiki/render_wiki_html.py ·
scripts/wiki/sync_wiki.py · 결과
docs/wiki/index.html · 계획
docs/plans/plan_20260828_llm_wiki.md

</div>

<a class="src-link" href="../v5/research_story.html#ch71">→ 원문 전체 보기 (research_story.html#ch71)</a>

</div>

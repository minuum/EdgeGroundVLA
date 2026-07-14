# Plan: 대시보드 위키/최신현황 탭을 리서치 히스토리 스타일로 재단장 (2026-07-12)

> 상태: **확정 (2026-07-12) — B안 + 체크포인트/실험 그래프 포함, 구현 대기**
> 확정: 옵션 B(경량 렌더러+CSS 이식) / 이미지=체크포인트·실험 그래프 포함 /
> 폰트 스코프·콜아웃 문법은 아래 "구현 시 세부 결정" 참고

---

## 배경

`plan_20260707_dashboard_wiki_status_tabs.md`에서 위키(📖)/최신현황(📡) 탭 2개를
만들 때, 마크다운 원문을 `<pre>` 모노스페이스로 그대로 찍는 방식을 "구현 리스크
없음"이라는 이유로 선택했음(`mona_dashboard.py:4420-4448`). 그 결과 지금 두 탭은:

- 폰트: 대시보드 전역 `--font-mono: 'JetBrains Mono'` — 코드처럼 보임
- 구조: 헤더/표/굵게 구분 없이 markdown 원문 그대로 (렌더링 안 함)
- 이미지: 전혀 없음
- 색상: 단색 텍스트(`#e2e8f0`), 카드도 `.card` 하나뿐, 색상 구분 없음

반면 `docs/v5/research_story.html`(GitHub Pages "리서치 히스토리", `docs/index.html`
Hero에서 링크됨)은:

- 폰트: **Noto Sans KR**(400/500/700/900) — 한글 가독성 좋음
- 색상: `--bg:#0a0f1a` `--accent:#38bdf8` `--green:#22c55e` `--red:#ef4444`
  `--yellow:#fbbf24` `--purple:#a78bfa` 등 의미별 색상 구분
- 구조: hero(그라디언트 타이틀) → KPI 카드 → 챕터(`.chapter`, 번호 배지) →
  타임라인(`.timeline-item.good/bad/pivot`) → 파인딩 카드(좌측 색띠,
  `.finding-card.warn/bad/good/info`) → 콜아웃 박스(`.callout.info/warn/critical/success`)
  → 표(`.cl-table`) → 이미지 그리드(`.fig-grid`, `.img-grid-3`, `.frame-strip`)
- 이미지: base64 인라인 또는 `docs/v5/` 하위 상대경로 PNG/JPG, 카드에 담겨 표시

즉 지금 위키/최신현황 탭은 "markdown 파일을 그대로 텍스트로 보여주는 뷰어"이고,
리서치 히스토리는 "손으로 짠 구조화 HTML 리포트"라 근본 구조가 다름. 스타일만
색/폰트 입히는 걸로는 카드/타임라인/이미지 그리드까지 못 따라감 — 콘텐츠
자체를 구조화된 형태로 다시 짜야 함.

## 옵션 (택1 필요)

### A. 색/폰트만 이식 (최소 변경)
- `<pre>` 유지, `font-family`를 Noto Sans KR로 바꾸고 배경/텍스트 색상만
  research_story.html 팔레트로 맞춤
- 장점: 구현 20분 내, 콘텐츠(md 파일) 안 건드림
- 단점: 카드/타임라인/이미지 그리드 같은 "느낌"은 전혀 안 남 — 그냥 색 바뀐
  모노스페이스 텍스트. 사용자가 원하는 "양식"까지는 안 따라감

### B. 경량 렌더러 + CSS 이식 (권장)
- `docs/DASHBOARD_WIKI.md` / `docs/DASHBOARD_LIVE_STATUS.md`를 구조화된
  마크다운 관례로 재작성 (예: `## 챕터`, `> [!warn]` 콜아웃 블록, `| 표 |`,
  `![](경로)` 이미지)
- 클라이언트 JS에 경량 MD→HTML 파서 추가 (헤더/표/굵게/코드블록/콜아웃
  블록/이미지 정도만 지원, ~80줄 내외)
- research_story.html의 CSS 블록(`.chapter`, `.finding-card`, `.callout`,
  `.kpi`, `.fig-card` 등)을 위키 탭 스코프(`#tab-wikiinfo`, `#tab-wikistatus`
  하위 셀렉터)로 이식 + `<link>`로 Noto Sans KR 로드
- 이미지: 새로 캡처/생성할 스크린샷이 없다면 위키 탭엔 이미지가 안 들어감 —
  최신현황 탭에 체크포인트 비교표/실험 그래프 등을 넣고 싶으면 별도로 준비 필요
- 장점: 실제로 "챕터/타임라인/카드/콜아웃" 양식이 재현됨
- 단점: md 파일 재작성 + 파서 작성 필요, 작업량 A보다 큼 (체감 1~2시간)

### C. 위키/최신현황도 정적 HTML로 직접 작성 (research_story.html과 동일 방식)
- markdown 대신 아예 손으로 짠 HTML 조각을 서버가 그대로 서빙
- 장점: 시각적으로 가장 정확히 일치
- 단점: `plan_20260707`에서 의도했던 "md 파일만 고치면 코드 안 건드리고 갱신"
  이라는 이점이 사라짐 — 특히 "최신현황"은 스킬로 자주 갱신하는 용도라 HTML
  손으로 유지보수하기엔 부담

**추천**: B. A는 사용자가 말한 "양식을 따랐으면"에 못 미치고, C는 최신현황
탭의 운영 편의성(스킬 자동 갱신)을 해침.

## 확정된 결정

1. **방향: B안** — md 파일을 구조화 문법으로 재작성 + 경량 JS 렌더러 +
   research_story.html CSS를 위키 탭 스코프로 이식
2. **이미지: 체크포인트/실험 그래프 포함** — 최신현황 탭에 현재 서빙 체크포인트
   관련 그래프(예: Exp11/Exp14/Exp16 PM·val_loss 비교, closed-loop 성공률)를
   `.fig-card`/`.img-grid-3` 형태로 삽입. 소스는 `docs/v5/` 하위 기존 그래프
   이미지를 상대경로로 재사용(가능한 것 우선) — 없는 조합은 이번 범위에서
   새로 생성하지 않고, 있는 것부터 반영 (신규 그래프 생성은 별도 요청 시 진행)
3. **최신현황 md는 스킬이 자동 갱신** — `.agent/skills/dashboard-status-sync/`도
   새 문법(콜아웃/이미지 삽입 규칙)에 맞춰 출력하도록 같이 업데이트 필요
4. **재사용 가능한 기존 그래프 확인 완료** — `docs/v5/portfolio/exp_progression.png`,
   `docs/v5/bbox_nav_exp51/report_figs/fig1_exp_progression.png`,
   `docs/v5/closed_loop_eval/trajectory_examples/*.png` 등 실험 진행·궤적
   그래프가 이미 존재 — 최신현황 탭에서 상대경로로 바로 참조 (신규 생성 없음)

## B안 상세 설계

### 파일 변경 목록
- `robovlm_nav/serve/mona_dashboard.py`
  - `<head>`에 Noto Sans KR `<link>` 추가 (위키 탭 스코프 한정, 전역 폰트는 안 바꿈)
  - `.chapter`/`.finding-card`/`.callout`/`.kpi`/`.timeline`/`.fig-card` 등
    CSS를 `#tab-wikiinfo`/`#tab-wikistatus` 하위 셀렉터로 이식
  - `renderWikiMarkdown(md)` JS 함수 신규 — 헤더(`#`~`###`)→`.chapter-header`,
    표→`.cl-table`, `> [!info/warn/...]`→`.callout`, 이미지→`.fig-card`로 변환
  - 기존 `loadWikiContent()`가 `<pre>`에 원문 넣던 걸 `renderWikiMarkdown()`
    결과로 교체
- `docs/DASHBOARD_WIKI.md`, `docs/DASHBOARD_LIVE_STATUS.md`
  - 챕터 구분(`##`)과 콜아웃 블록 문법을 반영해 재구성 (내용 자체는 거의
    유지, 구조 마크업만 추가)

### 트레이드오프 / 확인 필요한 점
1. **이미지**: 지금 위키/최신현황 md에 이미지가 없습니다. research_story.html
   처럼 이미지 그리드까지 넣으려면 어떤 이미지를(체크포인트 비교 그래프?
   실험 스크린샷?) 넣을지 알려주셔야 합니다 — 없으면 카드/타임라인/콜아웃
   구조만 이식하고 이미지 그리드는 스킵.
2. **폰트 전역 적용 여부**: Noto Sans KR을 위키/최신현황 탭에만 스코프할지,
   대시보드 전체(다른 8개 탭 포함)에 적용할지 — 전체 적용 시 코드/버튼
   monospace(JetBrains Mono)는 유지, 일반 텍스트만 바뀌는 형태가 자연스러움.
3. **콜아웃/파인딩카드 문법**: `docs/DASHBOARD_LIVE_STATUS.md`는 스킬이 자동
   갱신하는 파일이라, 스킬 쪽(`.agent/skills/dashboard-status-sync/`)도 새
   문법(콜아웃 블록 등)에 맞춰 출력하도록 같이 업데이트해야 함.

---

## 추가 범위 (2026-07-12, 사용자 요청으로 확장) — 수정 히스토리 브라우징

Tab6 세션 히스토리(`.session-card` 블록 리스트 → 클릭 시 상세 로드)와 동일한
UX로, 위키/최신현황 md 파일의 **git 커밋 히스토리**를 블록/버튼 형태로 나열하고
클릭하면 그 시점 버전을 렌더링해서 볼 수 있게 함.

- 백엔드: `GET /wiki/{name}/history` — `git log --follow --format=... -- <path>`
  로 해당 md 파일을 건드린 커밋 목록(sha/날짜/제목) 반환
  `GET /wiki/{name}/at/{sha}` — `git show <sha>:<path>` 로 특정 시점 콘텐츠 반환
- 프론트: 위키/최신현황 탭 상단에 `.wiki-hist-card` 블록 가로 리스트 추가
  (세션카드와 같은 시각 언어: 배경/보더/hover/active). 클릭 시 해당 커밋 시점
  콘텐츠를 `renderWikiMarkdown()`로 렌더링 + "과거 버전 보는 중" 배너 표시,
  "최신으로" 버튼으로 라이브 파일로 복귀
- 이미지(`![]()`)는 과거 커밋 시점엔 원본 이미지가 그때 경로에 없을 수 있음 —
  현재 워킹트리의 `/docs-static/v5/...`로 그대로 resolve (히스토리 이미지 자체를
  git show로 복원하진 않음, 범위 밖)

**구현 완료 v1** (2026-07-12): `/wiki/{name}/history`, `/wiki/{name}/at/{sha}` 백엔드
+ `.wiki-hist-card` 블록 리스트 + `loadWikiHistory`/`loadWikiAtCommit` JS +
"과거 버전 보는 중" 배너/최신복귀 버튼. 날짜/시간 가독성 추가 개선(사용자 요청):
히스토리 카드에 절대날짜(앰버, monospace) + 상대시간("N일 전") 병기, 최신현황
mtime도 절대+상대시간 병기.

**피벗 v2** (2026-07-12, 사용자 피드백 — "같은 md 여러버전 보는 게 중요한 게
아니라 연구일지 보는 느낌으로"): v1의 파일-버전-브라우징 방식을 폐기하고,
research_story.html의 `.timeline` 감성을 이식한 **🗓️ 연구일지** 타임라인으로
교체. `/journal`(레포 전체 git log, 커밋 제목 키워드로 good/bad/pivot 색상 분류)
+ `/journal/{sha}`(클릭 시 커밋 본문+변경파일 펼치기) 엔드포인트로 변경.
위키/최신현황 두 탭 모두 이 컴포넌트를 공유(같은 프로젝트 진행 기록이므로).

**콘텐츠 최신화**: `DASHBOARD_LIVE_STATUS.md`를 실제 최신 상태(git log, health,
episode_log, CH61 §17/18, 미병합 CH62 발견)로 전면 갱신. `DASHBOARD_WIKI.md`에
극단cx FWD 고착 gotcha 콜아웃 추가. `dashboard-status-sync` SKILL.md에
"git이 기본 출처, 대화에서만 나온 비-git 정보는 사용자 승인 필요" 원칙 명문화.

검증: node --check/ast.parse/div밸런스 통과, 재기동 후 `/`, `/journal`,
`/journal/{sha}`, `/wiki/info`, `/wiki/status` 200 확인, 8001 미영향(전 과정 반복 확인).

**재설계 v3** (2026-07-12, 사용자 피드백 — "세션 히스토리의 UI UX와 유사하게"):
v2의 단일 컬럼 타임라인+인라인 아코디언 방식을 폐기하고, Tab6 세션 히스토리와
동일한 **리스트+상세패널 분리** 구조로 교체. `.wj-grid`(260px `.wj-list` +
1fr `.wj-detail`, 760px 이하 반응형 1컬럼) — 왼쪽 `.wj-card` 블록 리스트
(세션카드와 동일한 시각 언어, kind별 좌측 색띠로 good/bad/pivot 구분, 클릭 시
active 하이라이트), 오른쪽 별도 `.wj-detail` 패널이 `loadSessionDetail` 패턴처럼
클릭한 커밋의 상세(sha/날짜/본문/변경파일)를 렌더링. 백엔드(`/journal`,
`/journal/{sha}`)는 변경 없이 재사용.

**폰트 확대 + 전체스크롤 전환** (2026-07-12, 사용자 요청 — "글자 크기 30%씩
키우고 md 덤프보다 리서치 히스토리처럼 스크롤 전체 내리는 형태로"):
`.wiki-render` 내 모든 폰트 크기(`h1/h2` 1.5→1.95rem, `h3` 1.1→1.43rem,
`p`/`ul` 0.94→1.22rem, 표 0.88/0.78→1.14/1.01rem, 콜아웃 0.9→1.17rem,
챕터번호뱃지 0.78→1.01rem, 이미지캡션 0.82→1.07rem) 약 30% 확대. 위키/최신현황
콘텐츠 박스의 `max-height:70vh; overflow:auto`를 제거해 내부 스크롤박스가
아닌 탭 전체(`.scroll-container`) 스크롤로 흐르도록 변경 — research_story.html처럼
긴 문서를 쭉 내려보는 형태.

검증(v3+폰트): node --check/ast.parse 통과, 재기동 후 `/`, `/wiki/status` 200 확인,
8001 미영향.

## 확인 완료

- ✅ B안 확정, 이미지(체크포인트/실험 그래프) 포함 확정
- Noto Sans KR 폰트 스코프는 별도 이견 없어 기본안(위키 탭 스코프 한정,
  다른 8개 탭의 Outfit/JetBrains Mono는 그대로 유지)으로 진행 — 전역 적용을
  원하시면 이후 요청 시 확장

## 구현 순서 (완료)

- [x] research_story.html CSS 발췌(`.chapter`/`.finding-card`/`.callout`/
      `.kpi`/`.timeline`/`.fig-card` 등) → `#tab-wikiinfo`/`#tab-wikistatus`
      스코프 셀렉터로 `mona_dashboard.py` `<style>`에 이식 + Noto Sans KR
      `<link>` 추가
- [x] `renderWikiMarkdown(md)` JS 파서 신규 작성 (헤더/표/굵게/코드블록/
      불릿리스트/`> [!info|warn|critical|success]` 콜아웃/`![](경로)` 이미지)
- [x] `loadWikiContent()`가 `<pre>` 대신 `renderWikiMarkdown()` 결과를 쓰도록 교체
- [x] `docs/DASHBOARD_WIKI.md` 콜아웃 문법으로 재구성 (금지사항/함정 섹션)
- [x] `docs/DASHBOARD_LIVE_STATUS.md` 콜아웃 문법 + 기존 그래프 이미지
      (`portfolio/exp_progression.png`, `bbox_nav_exp51/report_figs/fig1_exp_progression.png`)
      참조 추가. `/docs-static/v5` 마운트 신설(`mona_dashboard.py`)
- [x] `.agent/skills/dashboard-status-sync/SKILL.md`에 새 markdown 문법 가이드 절 추가
- [x] 검증: node --check, ast.parse, div 밸런스, 대시보드 재기동 후 `/`, `/wiki/info`,
      `/wiki/status`, `/docs-static/v5/...` 응답 확인, 8001 서버 미영향 확인 — 전부 통과

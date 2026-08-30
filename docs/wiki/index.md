# MoNaVLA 연구 위키

<p class="tagline">`docs/v5/research_story.html`(시간순 연구일지, 80+챕터)을 주제별로 재구성한 위키. Karpathy LLM-wiki 방식(raw/wiki/index 3계층, 벡터DB 없음) — 이 페이지만 먼저 읽고 필요한 주제 카드만 열어보면 된다. 원본은 이 위키가 대체하지 않는다 — 각 항목은 원문 챕터로 백링크되어 있다.</p>

## 주제

<div class="topic-grid" markdown="1">

<a class="topic-card accent-a" href="wiki-infrastructure.html">
<span class="topic-card-count">1개 챕터</span>
<span class="topic-card-title">연구 위키 & 문서 인프라</span>
<span class="topic-card-summary">research_story.html을 주제별로 재구성한 LLM wiki 구축(Karpathy 방식), 새 챕터 추가 시 안전 재동기화(wiki-sync 스킬), 아카이브 색인 등 문서화 도구 자체에 대한 기록.</span>
</a>

<a class="topic-card accent-b" href="grounding-detector.html">
<span class="topic-card-count">19개 챕터</span>
<span class="topic-card-title">그라운딩/검출기 (OWL-v2, Florence-2, PaliGemma, phrase grounding)</span>
<span class="topic-card-summary">타겟 물체를 이미지에서 찾아내는 검출기 계열 전체 이력 — PaliGemma2 → Kosmos-2/OWL-v2 → Florence-2로 이어지는 그라운딩 방식 전환과 재현율 개선.</span>
</a>

<a class="topic-card accent-c" href="action-head-architecture.html">
<span class="topic-card-count">12개 챕터</span>
<span class="topic-card-title">액션헤드 아키텍처 & 손실함수 실험</span>
<span class="topic-card-summary">bbox/vis 특징을 받아 액션을 뽑는 헤드 구조(MLP/LSTM/Transformer/FiLM/cross-attention)와 손실함수(hard CE vs ordinal soft label) 실험 전체.</span>
</a>

<a class="topic-card accent-d" href="language-conditioning.html">
<span class="topic-card-count">6개 챕터</span>
<span class="topic-card-title">언어 조건화 & 텍스트 어텐션 구조적 사망</span>
<span class="topic-card-summary">지시문(텍스트)이 액션에 영향을 미치는 진짜 언어조건화 VLA로 가려는 시도들 — 그리고 반복적으로 발견된 'text attention = 0%' 구조적 문제.</span>
</a>

<a class="topic-card accent-e" href="flow-matching-vla.html">
<span class="topic-card-count">1개 챕터</span>
<span class="topic-card-title">Flow Matching / π0 스타일 연속 액션</span>
<span class="topic-card-summary">이산 8-class 분류가 아니라 연속 action chunk를 flow matching으로 예측하는 MoNa-Pi 설계 사상과 실측(부진).</span>
</a>

<a class="topic-card accent-a" href="dataset-collection.html">
<span class="topic-card-count">8개 챕터</span>
<span class="topic-card-title">데이터셋 & 수집 프로토콜 (V3~V6)</span>
<span class="topic-card-summary">에피소드 경로 설계, 동기/비동기 수집, STOP 라벨링 규칙, 이미지 파이프라인 통일, 프레임 크롭/줌 등 데이터 자체를 다루는 실험들.</span>
</a>

<a class="topic-card accent-b" href="real-robot-closed-loop.html">
<span class="topic-card-count">13개 챕터</span>
<span class="topic-card-title">실로봇 테스트 & Closed-Loop 검증</span>
<span class="topic-card-summary">val_acc가 실기 성능을 예측하지 못한다는 반복 확인, closed-loop/궤적재생 근사, 100건 실기 테스트, 좌우 비대칭 원인 추적.</span>
</a>

<a class="topic-card accent-c" href="backbone-model.html">
<span class="topic-card-count">6개 챕터</span>
<span class="topic-card-title">VLM 백본 비교 (PaliGemma/Kosmos-2/CLIP/Florence-2/Google-robot)</span>
<span class="topic-card-summary">어떤 VLM을 백본으로 쓸지, LoRA가 실제로 무엇을 개선/손상시키는지, RoboVLMs 프레임워크 해부.</span>
</a>

<a class="topic-card accent-d" href="meetings-and-direction.html">
<span class="topic-card-count">19개 챕터</span>
<span class="topic-card-title">미팅 기록 & 연구 방향 결정</span>
<span class="topic-card-summary">교수님 미팅별 질문·피드백·결정사항, 논문 기여점, 연구 전체 요약과 로드맵.</span>
</a>

</div>

## 부록

<div class="topic-grid" markdown="1">

<a class="topic-card accent-archive" href="archive-index.html">
<span class="topic-card-count">357개 파일</span>
<span class="topic-card-title">아카이브 색인</span>
<span class="topic-card-summary">docs/*.md 스냅샷(2024-08~2026-04, 별개 프로젝트 단계) — 압축 없이 제목/날짜/한줄요약만 모은 찾아가기용 색인.</span>
</a>

</div>

<div class="summary-box" markdown="1">

**메타**

- 원본 챕터 수: 82개 (`docs/v5/research_story.html`)
- 위키 주제 수: 9개
- 생성 스크립트: `scripts/wiki/parse_research_story.py`, `scripts/wiki/build_wiki_pages.py`, `scripts/wiki/build_archive_index.py`, `scripts/wiki/render_wiki_html.py`
- **새 챕터 추가 시 자동 갱신**: `wiki-sync` 스킬(`.claude/skills/wiki-sync/SKILL.md`) 또는 `scripts/wiki/sync_wiki.py` 직접 실행 — 기존 압축 요약은 보존되고, 새로 추가된 챕터가 걸린 주제만 재압축 대상으로 표시됨
- 위키 재생성 의존성: HTML 렌더링(`render_wiki_html.py`)만 `pip install -r scripts/wiki/requirements.txt` 필요(Markdown 패키지) — 나머지 스크립트와 위키 페이지 열람 자체는 의존성 없음
- 최신 상태 요약(시간순, 별도 문서): `docs/RESEARCH_STATUS.md`

</div>
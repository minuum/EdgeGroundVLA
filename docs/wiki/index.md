# MoNaVLA 연구 위키 — 주제별 색인

`docs/v5/research_story.html`(시간순 연구일지, 70+챕터)을 주제별로
재구성한 위키다. Karpathy LLM-wiki 방식(raw/wiki/index 3계층, 벡터DB 없음) —
미래 세션은 이 index.md만 먼저 읽고, 필요한 주제 파일만 열어보면 된다.

원본(research_story.html)은 이 위키가 절대 대체하지 않는다 — 각 항목은
원문 챕터로 백링크되어 있으니, 세부 근거·수치·이미지가 필요하면 원문을 본다.

## 주제 목록

- **[그라운딩/검출기 (OWL-v2, Florence-2, PaliGemma, phrase grounding)](grounding-detector.md)** (19개 챕터) — 타겟 물체를 이미지에서 찾아내는 검출기 계열 전체 이력 — PaliGemma2 → Kosmos-2/OWL-v2 → Florence-2로 이어지는 그라운딩 방식 전환과 재현율 개선.
- **[액션헤드 아키텍처 & 손실함수 실험](action-head-architecture.md)** (12개 챕터) — bbox/vis 특징을 받아 액션을 뽑는 헤드 구조(MLP/LSTM/Transformer/FiLM/cross-attention)와 손실함수(hard CE vs ordinal soft label) 실험 전체.
- **[언어 조건화 & 텍스트 어텐션 구조적 사망](language-conditioning.md)** (6개 챕터) — 지시문(텍스트)이 액션에 영향을 미치는 진짜 언어조건화 VLA로 가려는 시도들 — 그리고 반복적으로 발견된 'text attention = 0%' 구조적 문제.
- **[Flow Matching / π0 스타일 연속 액션](flow-matching-vla.md)** (1개 챕터) — 이산 8-class 분류가 아니라 연속 action chunk를 flow matching으로 예측하는 MoNa-Pi 설계 사상과 실측(부진).
- **[데이터셋 & 수집 프로토콜 (V3~V6)](dataset-collection.md)** (8개 챕터) — 에피소드 경로 설계, 동기/비동기 수집, STOP 라벨링 규칙, 이미지 파이프라인 통일, 프레임 크롭/줌 등 데이터 자체를 다루는 실험들.
- **[실로봇 테스트 & Closed-Loop 검증](real-robot-closed-loop.md)** (13개 챕터) — val_acc가 실기 성능을 예측하지 못한다는 반복 확인, closed-loop/궤적재생 근사, 100건 실기 테스트, 좌우 비대칭 원인 추적.
- **[VLM 백본 비교 (PaliGemma/Kosmos-2/CLIP/Florence-2/Google-robot)](backbone-model.md)** (6개 챕터) — 어떤 VLM을 백본으로 쓸지, LoRA가 실제로 무엇을 개선/손상시키는지, RoboVLMs 프레임워크 해부.
- **[미팅 기록 & 연구 방향 결정](meetings-and-direction.md)** (18개 챕터) — 교수님 미팅별 질문·피드백·결정사항, 논문 기여점, 연구 전체 요약과 로드맵.

## 부록

- **[아카이브 색인(archive-index.md)](archive-index.md)** — `docs/*.md` 357개 스냅샷 파일
  (2025-12~2026-04, research_story.html 이전 시기의 별개 프로젝트 단계, 대부분 폐기된
  방향의 죽은 기록). 압축 없이 제목/날짜/한줄요약만 모은 찾아가기용 색인.

## 메타

- 원본 챕터 수: 81개 (`docs/v5/research_story.html`)
- 위키 주제 수: 8개
- 생성 스크립트: `scripts/wiki/parse_research_story.py`, `scripts/wiki/build_wiki_pages.py`,
  `scripts/wiki/build_archive_index.py`, `scripts/wiki/render_wiki_html.py`
- **새 챕터 추가 시 자동 갱신**: `wiki-sync` 스킬(`.claude/skills/wiki-sync/SKILL.md`)
  또는 `scripts/wiki/sync_wiki.py` 직접 실행 — 기존 압축 요약은 보존되고, 새로
  추가된 챕터가 걸린 주제만 재압축 대상으로 표시됨
- 위키 재생성 의존성: HTML 렌더링(`render_wiki_html.py`)만 `pip install -r scripts/wiki/requirements.txt` 필요(Markdown 패키지) — 나머지 스크립트와 위키 페이지 열람 자체는 의존성 없음
- 최신 상태 요약(시간순, 별도 문서): `docs/RESEARCH_STATUS.md`
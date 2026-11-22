---
name: research-history
description: 메인 리서치 히스토리 웹서버(docs/v5/research_story.html 및 docs/v5/research_story_egvla2.html)의 주소와 최신 챕터(#ch82 등) 링크를 확인하고 출력한다. 대화 마무리, 분석 리포트 전달 후, 또는 사용자가 "리서치 히스토리 링크", "연구 문서 링크", "히스토리 서버" 등을 요청할 때 사용.
---

# Research History Server Links & Publishing Architecture

MoNaVLA 연구의 메인 기록 문서인 `docs/v5/research_story.html`(CH1~73) 및 `docs/v5/research_story_egvla2.html`(CH74~82, EGVLA2 이후) 관련 발행 규칙과 서버 링크를 안내하는 스킬입니다.

## 발행 아키텍처 (커밋 `79c36d12` 기준)

1. **원본(Source of Truth)**: `docs/v5/research_story_src.html` (4.4MB)
   - 챕터를 끼워 넣거나 수정할 때는 반드시 `research_story_src.html`을 대상으로 합니다.
   - `scripts/gen_ch76_viz.py` 및 `scripts/gen_ch77.py` ~ `scripts/gen_ch82.py`는 원본을 수정한 뒤 자동으로 `scripts/story_publish.py`를 호출합니다.
   - 수동 발행 명령: `python3 scripts/story_publish.py`
   - **주의**: `research_story.html`이나 `research_story_egvla2.html`을 직접 수정하는 일회성 스크립트는 다음 발행 때 덮어써지므로 금지합니다.
2. **발행 페이지 2종**:
   - `docs/v5/research_story.html` (2.1MB): CH1~73 + 요약 섹션 (옛 `#ch74+` 해시 접근 시 JS가 자동으로 `research_story_egvla2.html`로 넘김)
   - `docs/v5/research_story_egvla2.html` (2.3MB): CH74~82 (Florence-2 단일화 EGVLA2 이후 기록)
3. **이미지 외부화 (`docs/v5/figures/inline/`)**:
   - 기존 base64 이미지들을 SHA-1 16자리 파일명(`figures/inline/<hash>.<ext>`)으로 외부화하고 `<img loading="lazy" decoding="async">`를 적용합니다.

## 빠른 실행 (CLI)

```bash
# 기본: 현재 서버 상태 및 최신 챕터 링크 출력
.agent/skills/research-history/scripts/show_history_links.py

# 특정 챕터 지정 (예: ch82)
.agent/skills/research-history/scripts/show_history_links.py --ch ch82
```

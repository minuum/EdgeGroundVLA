---
trigger: always_on
---

# Research & Documentation
- 논문 비교 분석 시, 항상 표 형식으로 정리하고 출처(논문 제목, 섹션)를 명확히 표기
- 기술적 주장을 할 때는 코드나 논문의 구체적 근거를 함께 제시
- 분석 리포트는 Markdown 형식으로 작성하되, 섹션 구조를 명확히 (Background, Analysis, Findings, Conclusion)
- 실험 결과나 모델 비교 시 정량적 메트릭(episodes, tasks, objects 수 등)을 우선 제시

# Code & Experiments
- Python 분석 스크립트 작성 시 주석을 한국어로 작성
- Git 작업 전 항상 현재 상태 확인 후 진행
- 대용량 파일(50MB 이상)은 자동으로 .gitignore 또는 Git LFS 처리 제안

# Communication Style
- 기본 언어는 한국어 사용
- 기술 용어는 영어 원문 유지 (예: LoRA, VLM, fine-tuning)
- 복잡한 개념 설명 시 구체적 예시나 다이어그램 활용

# Project Context
- 작업 디렉토리: /home/billy/25-1kp/vla
- 주요 관심 분야: Vision-Language-Action models, LoRA fine-tuning, VLM architectures
- 문서는 docs/ 디렉토리에 체계적으로 정리

# Workflow Preferences
- 논문 분석 → 표 작성 → 코드 검증 → 리포트 작성 순서 선호
- 교수님 미팅 준비 시 증거 기반 논증 구조 (주장 → 근거 → 데이터) 사용
- 불확실한 내용은 추측하지 않고 논문/코드에서 확인 후 답변

# Research History Server Link & Publishing Workflow
- 연구, 실험, 분석 보고 또는 세션 대화 마무리 시, 항상 리서치 히스토리 서버 링크(로컬 포트 8001/8766 및 GitHub Pages 원격 링크)를 함께 출력
- `research_story` 구조(커밋 `79c36d12`):
  - **원본(Source of Truth)**: `docs/v5/research_story_src.html` (챕터 추가·수정 시 반드시 이 원본을 대상으로 하며, `gen_ch76_viz.py` 및 `gen_ch77.py`~`gen_ch82.py`가 원본 수정 후 `scripts/story_publish.py`를 자동 호출함. 수동 발행은 `python3 scripts/story_publish.py`)
  - **발행본 1 (CH1~73 + 요약)**: `docs/v5/research_story.html` (2.1MB, 직접 수정 금지 — 다음 발행 시 덮어써짐)
  - **발행본 2 (CH74~ EGVLA2 이후)**: `docs/v5/research_story_egvla2.html` (2.3MB, `#ch74`~`#ch82` 링크는 이 페이지가 정식 주소이며 옛 `research_story.html#ch82` 주소도 JS로 자동 리다이렉트됨)
  - **외부화 이미지**: `docs/v5/figures/inline/<sha1_16>.<ext>` (`loading="lazy" decoding="async"` 적용, base64 직접 내장 지양)

# 인수인계 — image_proj 5-class 재학습 + 논문 최신화 (2026-08-07)

## 지금 상태 (한 줄 요약)
image_proj Stage 1 재학습(5-class, 225ep, OWL-v2 라벨, BGR→RGB 버그 수정판)이 **val_acc
94.09%로 완료**됨. 다음은 plan Step 3(Stage 2 액션 헤드 A/B 실험) → Step 4(체크포인트 배포)
→ Step 5(soda 실기 재검증, 좌측만) → 논문/브리핑 반영.

## 필수 참고 파일
- **plan.md**: `docs/plans/plan_20260806_stage1_image_proj_5cls_retrain.md` — 전체 계획,
  완료 표시 체크박스, 발견된 BGR→RGB 버그 기록. **가장 먼저 읽을 파일.**
- **재학습 스크립트**: `scripts/train_stage1_v3_5cls_owl.py` (원본
  `scripts/train_exp54_stage1_v2_frame_level.py`는 보존, 손대지 않음)
- **재학습 로그**: `logs/train_stage1_v3_5cls_owl.log` — 최종 val_acc 94.09%, 혼동행렬
  (강좌 99.0%·약좌 88.0%·중앙 96.6%·약우 96.7%·강우 90.3%, weak_left→strong_left 오분류 32건)
- **새 체크포인트**: `runs/v5_nav/mlp/stage1_v3_5cls/stage1_v3_5cls_owl_projs.pt`
  (기존 배포 체크포인트 `runs/v5_nav/mlp/shared/stage1_v2_projs.pt`는 그대로 유지 — 아직 전환 안 함)
- **논문 문서 4개** (전부 로컬서버 8081로 서빙 중, 아래 "서버" 참조):
  - `docs/01.paper/redfill_draft_0805.md`/`.html` — 절별 붙여넣기 초고 (원본은 .md, .html은
    스크립트로 재생성 — 아래 "HTML 재생성 방법" 참조)
  - `docs/01.paper/revision_briefing.html` — 메인 갱신 허브. 상단에 "⚠️ 재학습 후 교체 필요"
    배너와 "📋 docx 노란부분 최신화" 배너 있음. 변경이력이 여기 전부 기록됨.
  - `docs/v5/paper_redfill_worklist.html` — 25+34건 문제/대응/실답변
  - `docs/v5/model_architecture_brief.html` — 미팅 발표자료, 머메이드 다이어그램 포함
  - `docs/01.paper/terminology_glossary.html` — 용어 대체 근거(hold-aware→cadence-aligned,
    arm→condition 등)
- **최신화된 docx**: `docs/01.paper/EdgeGround-VLA_..._초안_0806_최신화.docx` (원본은 안 건드림,
  노란 하이라이트 5곳 텍스트만 교체한 새 파일)

## 로컬 서버
- 포트 **8081**에서 `python3 /tmp/.../scratchpad/utf8_server.py` 실행 중 (UTF-8 헤더 + `/__save_svg`
  POST 엔드포인트로 SVG 에디터가 실제 파일에 저장/PNG·PDF 재생성 가능)
- 안 떠 있으면: 새 세션에서 같은 방식으로 `python3 -m http.server` 대체 실행하거나, 이 세션의
  scratchpad에 있는 `utf8_server.py`를 찾아서 재실행 요청

## HTML 재생성 방법 (draft만 — 나머지는 직접 HTML 편집)
`redfill_draft_0805.md`를 고친 뒤에는 markdown → HTML 재생성 필요:
```python
import markdown
from pathlib import Path
md_path = Path("docs/01.paper/redfill_draft_0805.md")
html_path = Path("docs/01.paper/redfill_draft_0805.html")
md_text = md_path.read_text()
body_html = markdown.markdown(md_text, extensions=['tables','fenced_code','toc','sane_lists'])
lines = html_path.read_text().split("\n")
head_idx = next(i for i,l in enumerate(lines) if l.strip() == '<div class="wrap">')
script_idx = next(i for i,l in enumerate(lines) if l.strip() == '<script>')
tail_idx = script_idx - 1
while lines[tail_idx].strip() != '</div>': tail_idx -= 1
new_lines = lines[:head_idx+1] + body_html.split("\n") + lines[tail_idx:]
html_path.write_text("\n".join(new_lines))
```
(`.venv/bin/python3`로 실행 — markdown 패키지가 venv에만 있음)

## 편집 후 항상 할 것
- div 균형 체크(정규식으로 `<div>`/`</div>` 개수 비교) — 이 세션에서 여러 번 `.replace()`가
  닫는 태그를 잘못 삼켜서 깨진 적 있음
- `curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8081/<path>`로 200 확인

## 다음 할 일 (plan Step 3부터)
1. **Step 3**: 새 image_proj(`stage1_v3_5cls_owl_projs.pt`) 위에서
   - 옵션 A: 기존 exp73 헤드(`exp73_owl_trackF_v6_mlp_holdaware_seed0.pt`) 그대로 재사용해서
     val_acc 확인
   - 옵션 B: 헤드도 새로 학습(`scripts/train_exp73_trackA_heads.py` 계열 참고, "colorfixed"
     BGR→RGB 반드시 유지)
   - 둘 다 진행 후 비교(사용자가 "시간 차이 안 나면 둘 다"라고 지시함)
2. **Step 4**: 배포 경로(`scripts/run/go.sh`의 `S1_PT`,
   `stage2_v2_inference_server.py`의 `DEFAULT_STAGE1`)는 **soda 검증 통과 전까지 변경 금지**
3. **Step 5**: soda에 새 체크포인트 전달 → **약좌·강좌 2곳만** 실기 재검증(중앙/약우/강우는
   기존 결과 재사용, 하나의 모델 결과로 합쳐서 공식 보고)
4. 완료되면 `revision_briefing.html`의 "⚠️ 재학습 후 교체 필요" 배너에 나열된 항목들
   (F6 혼동행렬, F4 CH64 64-12 표, 89/100 실기 수치, F1 캡션, 배포이력 표) 재생성

## 이번 세션에서 확정된 핵심 결정/사실 (재확인 불필요)
- `stop_mode`: 코드 기본값은 `proximity`지만 **실제 실행(`go.sh`)은 `learned`로 오버라이드** —
  둘 다 정확히 구분해서 서술해야 함
- image_proj는 **Stage 1에서 실제로 학습**됨(0.262M) — "학습되는 건 헤드(0.866M)뿐"은 틀린 서술,
  전체는 1.128M
- V6 raw H5(`f["images"]`)는 **BGR 저장** — 반드시 `[:, :, ::-1]` 반전 필요
  (`gen_v6_owl_annotation.py`, `train_exp73_trackA_heads.py`의 "colorfixed 규격" 참고)
- 용어: `hold-aware`→`cadence-aligned`, `실험 arm`→`실험 조건(condition)` (코드/체크포인트
  내부 명칭은 안 바꿈, 논문 노출 텍스트만 교체)
- threshold 0.25 정탐률은 **94.9%**가 정답(95.3%는 초고 자체 오타였음, 이미 전부 수정)
- optimizer는 실제로 `AdamW`(원문 요구가 `Adam`이라고 한 건 원문 자체의 오류)

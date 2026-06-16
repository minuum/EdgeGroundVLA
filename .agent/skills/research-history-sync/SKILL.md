# research-history-sync

**트리거**: `/sync-research` 또는 사용자가 "싱크", "업데이트", "결과 반영" 같은 말을 할 때

## 목적

`docs/RESEARCH_STATUS.md` 한 파일만 수정하면 README / GitHub Pages / menemory 등 전체가 자동 동기화됨.

```
docs/RESEARCH_STATUS.md  ←  유일한 진실 (편집 대상)
         ↓  sync_research.py
  ├── README.md              (핵심 결과 표 + 체크포인트)
  ├── docs/index.html        (Hero 수치 + Metrics Bar + Results 표)
  └── [제안] .menemory/core/master_memory.md
```

---

## 주요 커맨드

| 명령 | 설명 |
|------|------|
| `mona-sync` | 실제 동기화 (파일 3개 갱신) |
| `mona-sync --diff` | 변경사항 미리보기 (diff 출력) |
| `mona-sync --dry-run` | 파일 수정 없이 로그만 출력 |
| `mona-sync --validate` | 앵커 존재 여부 검사 |
| `mona-sync --status` | 현재 SOTA 요약 출력 |
| `mona-sync --add-exp` | 새 실험 행 대화형 추가 |
| `mona-sync --propose-menemory` | menemory 업데이트 텍스트 출력 |

---

## RESEARCH_STATUS.md 편집 가이드

### 새 실험 추가 (권장: 대화형)
```bash
mona-sync --add-exp
# → Exp 이름, 날짜, CL%, FPE, 특이사항 입력 → experiment_history 섹션에 자동 append
mona-sync   # 동기화
```

### 직접 편집
1. `docs/RESEARCH_STATUS.md` 열기
2. YAML front-matter의 `sota_*` 값 수정 (SOTA 모델 변경 시)
3. `<!-- BEGIN:results_table -->` ~ `<!-- END:results_table -->` 사이 표 수정
4. `<!-- BEGIN:experiment_history -->` 끝에 새 행 append
5. `mona-sync` 실행

### YAML front-matter 핵심 키
```yaml
sota_exp: "Exp66"
sota_cl: "96.6%"
sota_fpe: "0.094 m"
sota_ckpt_s1: "runs/v5_nav/mlp/shared/stage1_v2_projs.pt"
sota_ckpt_s2: "runs/v5_nav/mlp/exp66/action_mlp.pt"
updated: "2026-06-16"
hero_metric1_value: "96.6%"    # index.html Metrics Bar 첫 번째 칸
```

---

## Agent 사용 시 절차

1. `docs/RESEARCH_STATUS.md` 읽어 현재 SOTA 파악
2. 변경할 내용 확인 (새 Exp 결과 / SOTA 교체 / 날짜 갱신 등)
3. RESEARCH_STATUS.md 수정 (YAML + 해당 섹션)
4. `python3 scripts/utils/sync_research.py --diff` 로 검증
5. `python3 scripts/utils/sync_research.py` 실행
6. `git add docs/RESEARCH_STATUS.md README.md docs/index.html && git commit`

---

## sync 대상 앵커 목록

### README.md
- `<!-- SYNC:results_table:start/end -->` — 핵심 결과 표
- `<!-- SYNC:checkpoints:start/end -->` — 체크포인트 경로
- `<!-- SYNC:updated -->...<!-- /SYNC:updated -->` — 날짜

### docs/index.html
- `<!-- SYNC:hero_tagline:start/end -->` — Hero 설명 텍스트
- `<!-- SYNC:metrics_bar:start/end -->` — 4개 KPI 수치
- `<!-- SYNC:results_table_body:start/end -->` — Main Results tbody

---

## 동기화 안 되는 것 (수동 관리)

- `docs/v5/research_story.html` — 챕터 단위 서술, 수동 편집
- `docs/v5/grounding_hub.html` — Grounding 전문 페이지
- `.menemory/core/master_memory.md` — CLAUDE.md 정책상 제안만 출력, 직접 수정 안 함

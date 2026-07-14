# 최신 현황 (dashboard-status-sync 스킬로 갱신됨)

> 마지막 갱신: 2026-07-12 (soda) — 갱신 방법: `.agent/skills/dashboard-status-sync/SKILL.md` 참고

## 현재 서빙 중인 체크포인트
- `runs/v5_nav/mlp/exp71_window3_bboxscale3/action_transformer.pt`
  (window=3, bbox_scale=3.0, 5-seed val_acc 80.7%±4.3%, 최고 84.4%)
- grounder: OWL-v2 (thresh=0.25, area_scale=3.0)
- 8001 서버 마지막 재시작 시점 코드: `62079476` (재시작 이후 대시보드 코드는
  `806ae27e`까지 진행됐지만 8001 프로세스 자체는 안 건드림 — 정상)

## 실험 진행 그래프

![Exp 진행 추이 (portfolio)](portfolio/exp_progression.png)
![Exp1 진행 추이 (bbox_nav_exp51 리포트)](bbox_nav_exp51/report_figs/fig1_exp_progression.png)

## ✅ 할 일 (다음 세션 픽업용)

### 지금 당장 — FWD 고착 원인 추적 (진행 중, 최우선)
- [ ] 2026-07-11 obj_right 12회 배치(#109~121, `20260711_205228`~`220439`) 재현성
      확인 — grounding이 정상 검출(cx 0.76~0.9대, area 0.24~0.41)돼도 FWD만
      뱉는 케이스 다수 발생. `docs/v5/closed_loop_eval/CH61_OWL_LIVE_FAILURE_AND_FIX.md`
      §17/§18 참고 (grounding 성공/실패가 성공률을 가르지만, 극단cx 구간
      1.4~3.2% 학습데이터 희소성이 근본 원인으로 추정)
- [ ] minum 쪽 CH62(`inference-integration` 브랜치, 커밋 `21c2c27e`/`5188fbf3`,
      2026-07-12 새벽)에서 "그라운딩 실패가 아니라 액션-라벨 confound"로
      원인을 재규명함(cx-액션 방향 일치율 VSC 35.5%, 우연 이하) — **이 브랜치는
      아직 monavla-driving에 merge 안 됨**, 다음 세션에서 병합 여부 검토 필요

### 지금 당장 — 극단 배치 데이터수집 (확정안, 실행 가능)
- [ ] 1일차 목표 120ep: 4포지션 × 지시 2종(①바구니로 가 / ②반대쪽으로 가) × 15회
      (③정지 지시는 여유되면 추가)
- [ ] 극단 배치 기준: cx 0.10~0.15(강한좌) / 0.20~0.25(준극단좌) /
      0.75~0.80(준극단우) / 0.85~0.90(강한우) — 📷 데이터수집 탭 실시간 cx로 확인
- [ ] 📷 데이터수집 탭에서 시나리오/패턴 선택 후 키보드 또는 조이스틱으로 주행,
      ⏹ 정지 & 저장으로 H5 기록

### 대기 중 (minum 응답 필요)

> [!warn]
> - CH62(액션-라벨 confound 재규명)에 대한 soda 쪽 의견/merge 여부
> - FINDING_20260706 flicker 분석에 대한 의견
> - cx-bin/action 상관관계 분석 (그라운딩은 방향 맞게 주는데 액션 헤드가 특정
>   방향을 거의 안 뱉는 문제)

### 진행 중 A/B
- [ ] window3+bboxscale3 vs 기존 window6 — 실로봇 비교 테스트 (체크포인트는
      window3+bboxscale3로 전환된 채 유지, 비교 주행/평가는 아직)

## 최근 발견/이슈

> [!critical]
> **2026-07-11 obj_right 배치(12회 중 1회만 성공)** — grounding이 정상 검출된
> 케이스(cx 0.76~0.9대)에서도 FWD 고착 재현. 세션 `213650`은 CH61 §17("grounding
> 성공 시 88.9% 성공")의 잔여 실패(11.1%) 구간으로 보이며, cx>0.75 구간
> 학습데이터가 전체의 1.4%뿐이라는 게 서버 코드 자체 정의(`CX_RULE_THRESHOLDS`)로
> 확인됨. minum의 CH62 재규명(액션-라벨 confound, VSC 35.5%)과 층을 이루는 문제.

- **🕓 위키/최신현황 탭 리서치 히스토리 스타일 재단장** (`806ae27e` 직전 작업,
  아직 커밋 전) — `<pre>` 텍스트 덤프였던 위키/최신현황 탭을
  `docs/v5/research_story.html` 양식(Noto Sans KR, 챕터/콜아웃/이미지그리드)으로
  재구성. 경량 markdown 렌더러(`renderWikiMarkdown`) 신규 작성, `/docs-static/v5`
  마운트로 실험 그래프 이미지 서빙. **🗓️ 연구일지** 타임라인도 추가 —
  `/journal`, `/journal/{sha}` 엔드포인트로 git 커밋을 research_story.html
  타임라인 감성으로 훑어보고 클릭하면 커밋 메시지 전문이 펼쳐짐(같은 md
  파일의 과거 버전을 보여주는 방식이 아님, 프로젝트 진행 기록 브라우징 목적).
- **Tab6 세션 히스토리에 경로검증 기록 결합 + 그라운딩 잔상버그 수정** (`806ae27e`)
  — Frame Inspector에서 episode_log 매칭 행을 타일로 표시/수정 가능. `/drive/start`가
  `_state["bbox"]`/`"chunk"`를 초기화 안 해서 새 세션 시작 직후 이전 세션 grounding
  값이 화면에 잔상처럼 남던 버그 수정, `/reset` 실패 시 재시도+경고 표시로 변경.
- **OWL-v2 detection flicker 의심** (`docs/v5/grounding_analysis/FINDING_20260706_action_collapse_across_conditions.md`)
  — has_bbox가 세션당 40~60% 프레임에서 꺼지고 고정 fallback(cx=0.5, area=0.06)으로
  대체되는 현상. 2026-07-11 배치에서도 fallback 케이스 다수 재확인.
- **2026-07-11 세션 13개(전체) minum 서버로 전송 완료** —
  `~/MoNaVLA/inference_sessions_recv/20260712/`. grounding_decisions 112줄,
  episode_log(#109~121), 활성 checkpoint 동봉.

## 최근 커밋 (HEAD 기준 6개)
```
806ae27e feat(dashboard): Tab6 세션 히스토리에 경로검증 기록 결합 + fix(dashboard): 세션 전환 시 그라운딩 잔상 제거
62079476 fix(dashboard): reset _ros._stable on session start — stops next-session tail-frame reuse
ae2478a4 fix(dashboard): episode row click did nothing — onclick attribute quote collision
13c7364e feat(dashboard): click episode row to edit (Tab4 log) + fix lat/FPE column swap bug
db7361d0 feat(dashboard): D-pad up/down now cycles 3 axes (trackA position -> path -> scenario)
78e49d02 docs(plan): flag trackA missing distance/depth control (SODA finding)
```

---
*이 파일은 수동 스킬 실행으로 갱신됩니다 (실시간 자동 아님). "대시보드 최신현황 갱신해줘"라고 요청하면 됩니다.*

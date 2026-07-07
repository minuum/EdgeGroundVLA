# 최신 현황 (dashboard-status-sync 스킬로 갱신됨)

> 마지막 갱신: 2026-07-08 (soda) — 갱신 방법: `.agent/skills/dashboard-status-sync/SKILL.md` 참고

## 현재 서빙 중인 체크포인트
- `runs/v5_nav/mlp/exp71_window3_bboxscale3/action_transformer.pt`
  (window=3, bbox_scale=3.0, 5-seed val_acc 80.7%±4.3%, 최고 84.4%)
- grounder: OWL-v2 (thresh=0.25, area_scale=3.0)
- 코드 커밋: `37449f27`

## ✅ 할 일 (다음 세션 픽업용)

### 지금 당장 — 극단 배치 데이터수집 (확정안, 실행 가능)
- [ ] 1일차 목표 120ep: 4포지션 × 지시 2종(①바구니로 가 / ②반대쪽으로 가) × 15회
      (③정지 지시는 여유되면 추가)
- [ ] 극단 배치 기준: cx 0.10~0.15(강한좌) / 0.20~0.25(준극단좌) /
      0.75~0.80(준극단우) / 0.85~0.90(강한우) — 📷 데이터수집 탭 실시간 cx로 확인
- [ ] 📷 데이터수집 탭에서 시나리오/패턴 선택 후 키보드 또는 조이스틱으로 주행,
      ⏹ 정지 & 저장으로 H5 기록
- [ ] 수집 후 `push_inference_session_to_minum.sh`류 스크립트로 전송할지, 아니면
      H5 파일 그대로 동기화할지 결정 필요 (아직 미정)

### 대기 중 (minum 응답 필요)
- [ ] FINDING_20260706 flicker 분석에 대한 의견
- [ ] cx-bin/action 상관관계 분석 (그라운딩은 방향 맞게 주는데 액션 헤드가 특정
      방향을 거의 안 뱉는 문제) — 아직 minum에게 미전송, 필요시 요청

### 진행 중 A/B
- [ ] window3+bboxscale3 vs 기존 window6 — 실로봇 비교 테스트 (체크포인트는 이미
      window3+bboxscale3로 전환됨, 비교 주행/평가는 아직)

### Phase 3 (미착수, 별도 논의 필요)
- [ ] 데이터수집 탭(웹) vs 레거시 `mobile_vla_data_collector.py`(터미널) 동시 실행
      방지 — 락파일 방식 제안됨, 구현은 안 함
      (`docs/plans/plan_20260707_dashboard_data_collector_tab.md` 참고)

## 최근 발견/이슈
- **OWL-v2 detection flicker 의심** (`docs/v5/grounding_analysis/FINDING_20260706_action_collapse_across_conditions.md`)
  — 조건이 달라도 액션 패턴(RIGHT↔FWD+L 왕복)이 수렴하는 현상. has_bbox가
  세션당 40~60% 프레임에서 꺼지고 고정 fallback(cx=0.5, area=0.06)으로 대체되는 게
  원인으로 의심됨. minum에게 전송, 의견 대기 중.
- **미적용 완화안 (의견 대기)**: `STOP_MODE=proximity` 전환 / "sticky bbox"(마지막
  실제 검출값 N프레임 유지)
- **📷 데이터수집 탭 신설** (`5a55448e`, `37449f27`) — `mobile_vla_data_collector.py`
  웹 이식(Phase 1+2). 키보드/조이스틱 둘 다 자동 기록(VLAControlManager
  `publish_and_move()` 단일 훅). 실시간 cx 피드백(`/collect/ground`, 액션예측 없이
  그라운딩만) + 카메라 프리뷰 포함. 라이브테스트 중 `robust_stop()` 5x 중복
  정지펄스가 H5에 개별 프레임으로 기록되던 버그 발견·수정.
- **📖 위키 / 📡 최신현황 탭 신설** (`5a55448e`) — 이 파일이 그 최신현황 탭에
  표시되는 파일.
- **대시보드 서버 재시작 "변경없음" 버그 수정 완료** (`f44e5f2d`) — 이제 빈 body로
  재시작해도 현재 grounder/ckpt 유지됨

## 최근 커밋 (HEAD 기준 6개)
```
37449f27 feat(dashboard): live cx readout in 데이터수집 tab for basket placement
02606740 docs: update session_analysis_report + add exp71_window3_bboxscale3 eval json
5a55448e feat(dashboard): add 위키/최신현황/데이터수집 tabs to 7800 dashboard
604f2661 feat(exp71): bbox_scale 체크포인트 메타데이터 지원 (window3+bboxscale3 배포 대비)
51adce65 feat(preview): 옵션D — attempt별 cx/outcome을 logs/preview_decisions.jsonl에 영구 기록
f44e5f2d fix(dashboard): server restart "no changes" silently reset grounder/ckpt
```

---
*이 파일은 수동 스킬 실행으로 갱신됩니다 (실시간 자동 아님). "대시보드 최신현황 갱신해줘"라고 요청하면 됩니다.*

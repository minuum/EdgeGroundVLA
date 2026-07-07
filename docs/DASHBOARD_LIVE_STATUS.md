# 최신 현황 (dashboard-status-sync 스킬로 갱신됨)

> 마지막 갱신: 2026-07-07 (soda) — 갱신 방법: `.agent/skills/dashboard-status-sync/SKILL.md` 참고

## 현재 서빙 중인 체크포인트
- `runs/v5_nav/mlp/exp71_window3_bboxscale3/action_transformer.pt`
  (window=3, bbox_scale=3.0, 5-seed val_acc 80.7%±4.3%, 최고 84.4%)
- grounder: OWL-v2 (thresh=0.25, area_scale=3.0)
- 코드 커밋: `604f2661`

## 진행 중인 A/B 테스트
- window3+bboxscale3 vs 기존 window6 — 실로봇 비교 테스트 진행 예정

## 최근 발견/이슈
- **OWL-v2 detection flicker 의심** (`docs/v5/grounding_analysis/FINDING_20260706_action_collapse_across_conditions.md`)
  — 조건이 달라도 액션 패턴(RIGHT↔FWD+L 왕복)이 수렴하는 현상. has_bbox가
  세션당 40~60% 프레임에서 꺼지고 고정 fallback(cx=0.5, area=0.06)으로 대체되는 게
  원인으로 의심됨. minum에게 전송, 의견 대기 중.
- **미적용 완화안 (의견 대기)**: `STOP_MODE=proximity` 전환 / "sticky bbox"(마지막
  실제 검출값 N프레임 유지)
- **대시보드 서버 재시작 "변경없음" 버그 수정 완료** (`f44e5f2d`) — 이제 빈 body로
  재시작해도 현재 grounder/ckpt 유지됨

## 최근 커밋 (HEAD 기준 5개)
```
f44e5f2d fix(dashboard): server restart "no changes" silently reset grounder/ckpt
10a2d402 docs(grounding): 조건 달라도 액션 패턴 수렴하는 문제 분석 — OWL-v2 detection flicker 의심
063ba77f feat(dashboard): show recent-session stack on drive stop / auto-stop
fe3e6417 feat(owlv2): add bbox area calibration coefficient, adjustable from Tab4
08bb6f5f docs(skill): sync-inference-session — OWL-v2 A/B 시대에 맞게 갱신
```

## 대기 중 (minum 응답 필요)
- FINDING_20260706 flicker 분석에 대한 의견
- cx-bin/action 상관관계 분석 (그라운딩은 방향 맞게 주는데 액션 헤드가 특정
  방향을 거의 안 뱉는 문제) — 아직 minum에게 미전송, 필요시 요청

---
*이 파일은 수동 스킬 실행으로 갱신됩니다 (실시간 자동 아님). "대시보드 최신현황 갱신해줘"라고 요청하면 됩니다.*

# 로봇 세션 체크리스트 — 폐루프 시작 (2026-10-22, minum 작성)

> 결정(사용자 승인): **단계 A(하이브리드) 켠다 → 비교군(MLP 회귀) 먼저 → 짧은 드라이런 후 본 구동.** 서버는 항상 한 개만.

## 0. 순서
1. 서버 구동(주 모델 C) → 20프레임 점검 → **짧은 드라이런(로봇 정지, 사람이 조작, 목표 위치 5종 각 1개)**
2. 로그 확인(minum) → OK면 `VLA_DRIVE_SOURCE=egvla2_hybrid` 켬
3. **비교군 먼저**: 체크포인트 `MLP_regression_drop30_e60.pt`, pos1 × 목표 위치 5종 × 4회 = 20회
4. 서버 재시작(체크포인트 교체, 동시 2개 금지) → 주 모델 `C_pi0_adaln_drop30_e120.pt` 본 시험(pos1 우선, 그다음 pos2·pos3)

## 1. 환경(공통)
`VLA_ACTION_MODE=continuous VLA_SPEED_SOURCE=shadow VLA_EGVLA2_EMA=0.5 VLA_JPEG_QUALITY=95`, 색 보정 `apply_cc` 끔, fp32, `HF_HUB_OFFLINE=1`, 단계 A: `VLA_DRIVE_SOURCE=egvla2_hybrid`. 단계 B(단독 서버)는 켜지 않는다.

## 2. 매 전환마다 점검
- sha256이 MANIFEST와 일치, `/reset` 후 GPU에 discrete 모델이 같이 올라가 있지 않은지(메모리 확인)
- `robot_check_server_vs_local.py` 20프레임: 주 모델 `pass: true`, 회귀는 방향 차 ≤1.0°
- 지연(`latency_ms`, `grounding_latency_ms`) 평균·p95 기록(`/reset` 수정 이후 값)

## 3. 기록(분석에 필수)
- 세션마다 메타데이터: 모델 이름, 체크포인트 sha, EMA, JPEG 품질, dtype, 구동 모드, 시작 위치(pos1/2/3), 목표 위치(5종)
- 프레임마다: `action_cont`, `drive`, `chunk_first`, `spread`, `subgoal_missing`, `bbox`, `controller_state`, `reason`, 시각
- 에피소드 결과: 성공/실패, 충돌, 개입 여부, 소요 시간

## 4. 안전
- 비상 정지가 서버 응답과 무관하게 동작하는지 사람이 구동 전 1회 확인
- 확인 안 된 값: gain 1.15 단위, 대각선 속도 가정 → 드라이런 로그에서 이상하면 즉시 중단·회신
- 비교군 20회 중 연속 충돌 2회 이상이면 중단하고 회신

## 5. 회신
드라이런 로그 경로, 점검 JSON, 지연 요약, 막힌 점. 구동 결과는 세션 H5와 결과 표를 `ROS_action/`·커밋으로 전달.

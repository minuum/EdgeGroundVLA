# 로봇 세션 전달 — 폐루프 비교 팔별 체크포인트·설정 (2026-10-27, minum 작성)

> E2E VLA는 아직 같은 조건 비교가 끝나지 않아 **이번 전달에서 제외**한다(진행 중). 아래는 학습 서버에서 준비·점검이 끝난 모델과 서빙 설정이다. 이전 문서(`HANDOFF_20261022_ROBOT_CHECKLIST_closedloop.md`, `..._LOGGING_and_followup.md`, `HANDOFF_20261023_ROBOT_PHYSICAL_CALIBRATION.md`)의 조건(JPEG q95, EMA 0.5, fp32, apply_cc 끔, 서버 1개)은 그대로 적용.

## 1. 코드: `inference-integration` 최신(≥ `f280b33f`) pull
- 회전 상태·부호 정정(`eb7d8436`, 이미 포팅했다고 회신됨) + **타깃 서보 환경변수**(`f280b33f`: `VLA_EGVLA2_SERVO_BLEND`, `SERVO_K`, `SERVO_MAX`, 기본 0=끔) 반영 여부를 확인.
- 크기 기반 stop 래치는 기본 끔(`VLA_EGVLA2_STOP_DEBOUNCE=0`), 근접 정지(면적 ≥ 0.25)·소실 탐색은 유지.

## 2. 체크포인트 (rsync, `--delete` 금지) 와 sha256 (전부 MANIFEST.json과 학습 서버에서 대조 완료)
`rsync -ah --progress minum:/home/minum/26CS/MoNaVLA/runs/egvla2_fm/deploy/ <로봇측>/runs/egvla2_fm/deploy/`
| 파일 | 크기 | sha256(앞 10) | 용도 |
|---|---|---|---|
| `C_pi0_adaln_drop30_e120.pt` | 2.9MB | f88b944fa6 | **기준(현재 실기 투입 모델)** |
| `C_pi0_servolabel0.5_drop30_e120.pt` | 2.9MB | 30c78b945c | 타깃 서보 라벨 λ=0.5 |
| `C_pi0_servolabel1.0_drop30_e120.pt` | 2.9MB | 5248ea1564 | 타깃 서보 라벨 λ=1.0 |
| `MLP_regression_drop30_e60.pt` | 0.85MB | 38ac424bc9 | MLP 회귀 비교군 |
| `C_pi0_e1b_tuned_drop30_e120.pt` | 2.9MB | c34b0746e7 | (선택) E1b_tuned 후보 — 실기 미검증, held-out 사람 방향 오차 22~26°로 기준보다 큼 |
`sha256sum`이 위와 다르면 사용하지 말고 회신.

## 3. 비교 팔 (제안, 사용자 승인 후) — 팔당 pos1 × 목표 위치 5종 × 4회 = 20회
| 팔 | 체크포인트 | 추가 환경변수 | 보는 것 |
|---|---|---|---|
| A0 | 기준 | (없음) | 기준선 |
| A1 | 기준 | `VLA_EGVLA2_SERVO_BLEND=0.5` | 외부 서보 혼합 효과 |
| A2 | servolabel0.5 | (없음) | 학습 서보 약 |
| A3 | servolabel1.0 | (없음) | 학습 서보 강(오프라인 사람 방향 오차 큼) |
| A4 | MLP 회귀 | (없음) | 회귀 비교군 |
| A5(선택) | E1b_tuned | (없음) | 후보 |
- **속도·구동 모드는 모든 팔에서 같게 고정**(속도 정렬은 별도 축). 한 번에 서버 하나, 체크포인트만 바꿔 순서대로(`VLA_EGVLA2_CKPT`).
- 오프라인 점검: 위 5개 체크포인트가 첫 프레임(weak_right)에서 의미 있는 출력을 냄(기준·servolabel·회귀는 우회전 후 구동, E1b_tuned는 곧바로 구동). `robot_check_server_vs_local.py` 20프레임 점검(주 모델 `pass:true`, 회귀 ≤1.0°)은 팔 교체 시마다.

## 4. 기록(필수) — 팔 비교를 가능하게 하는 것
세션 메타: 체크포인트 이름·sha, `SERVO_BLEND`, EMA, 구동 모드(SYNC/ASYNC), throttle·rot_throttle, 시작·목표 위치. 에피소드 결과: 성공/실패, **타깃 소실 발생 여부와 스텝**, 회전 오버슈트, 개입, 소요 시간, 정지 위치와 목표까지 거리(cm).

## 5. 주의·한계
- 서보 계열(A1~A3)은 오프라인에서 타깃 반응(반사실 민감도)이 생겼지만 **폐루프 이탈이 줄어드는지는 미검증**이다. 이탈이 줄면 소실률(현재 10/17 세션)이 먼저 내려가야 한다.
- 안전: 비상 정지 구동 전 확인, 비교군에서 연속 충돌 2회면 중단·회신(이전 기준 유지).
- E2E VLA(Gemini 세션 산출물)는 비교가 끝나기 전까지 사용 금지.

## 6. 회신
(1) pull 커밋과 sha256 대조 결과, (2) A0~A4 중 먼저 돌릴 순서 합의, (3) 샘플 세션 H5의 스냅샷 키(`ckpt_sha256`, `SERVO_BLEND`).

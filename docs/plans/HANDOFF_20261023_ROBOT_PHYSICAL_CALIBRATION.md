# 로봇 세션 전달 — 컨트롤러 물리 환산 점검·보정 실험 요청 (2026-10-23, minum 작성)

> 근거: `docs/plans/analysis_20261023_controller_physical_mapping.md`. 요지: 하드웨어는 명령 크기를 쓰지 않고(방향각·회전 부호·throttle만), 학습 데이터의 물리 속도는 throttle 50 고정이었으며, 서빙은 speed_scale 0.7(throttle 35)로 약 30% 느리다. **아래 2번(속도 변경)은 안전 속도를 바꾸므로 사용자 승인 후에만.**

## 1. 로그 보강(동작 변화 없음)
스냅샷·스텝 로그에 추가: `ctrl.throttle`, `ctrl.rot_throttle`, 실제 호출된 `driver.spin(...)`/`driver.move(angle, throttle)` 인자, `drive_loop_mode`(SYNC/ASYNC), 추론 시 `throttle_scale`(speed_scale).

## 2. 속도 정렬 비교군(승인 후)
수집 물리와 맞추기: `VLA_SPEED_MAX`·규칙 속력 1.0(throttle 50, 직진 약 26.4cm/s) 대 0.7(현재). 드라이런(로봇 정지 로그) → 안전 확인 → pos1 × 5 위치 × 4회 비교. 한 번에 한 가지만 변경.

## 3. 회전 보정 실험(정지 상태, 목표 고정)
| 변수 | 값 |
|---|---|
| rot throttle 스케일 | 0.5, 0.7, 1.0 |
| 버스트 시간 | 0.2, 0.4, 0.6초 |
각 조합 5회, 부호 +/− 각각. 회전 전·후 타깃 cx 또는 각도기(가능하면 바닥 눈금) 기록, **명령 후 즉시 프레임과 1초 뒤 프레임 둘 다**(정지 후 관성 분리). 결과: 조합별 Δcx(또는 도). 이 표로 비례 회전을 throttle_scale 또는 버스트 시간으로 확정.

## 4. 코드 수정 제안
순수 회전(lx=ly=0)은 `move_and_stop_ramped`의 램프를 건너뛴다(첫 램프 단계 az×0.33<0.1이면 하드웨어가 전진 move로 해석해 약 17ms 전진 펄스).

## 5. SYNC vs ASYNC 회전 비교
같은 회전 명령(az=±0.25)을 SYNC(0.4초 버스트)와 ASYNC(다음 응답까지 유지)로 각 5회 → 회전 시간 구조와 관성 분리.

## 6. 회신
(1)의 로그 샘플 H5, (3)의 조합별 표, 구동기 `spin/move` 인자 값, 막힌 점.

# 계획: 추론 서버의 연속 속력 출력 (속력 보간 규칙 포함)

> 작성일: 2026-10-14 | 상태: **계획 — 코드 미작성, 사용자 승인 대기**
> 관련: CH76(76-8), 핸드오프 `HANDOFF_20261013_continuous_collection_for_robot_session.md` §10~11, `plan_20261013_egvla2_fm_action_head.md` §8

## 0. 결정 (사용자, 2026-10-14)
- **속력 단계를 연속값으로 취급**한다(3-class 분류 아님). 중간 속력의 선형성 추가 검증은 하지 않는다 — 로봇은 한 스텝 동안 균일 속도로 움직이므로 **속력을 임의 보간해서 지정**한다.

## 1. 속력 보간 규칙 (확정안)
모델 출력 (lx, ly, az) → 방향·크기 분해 → throttle 지정.

| 단계 | 규칙 |
|---|---|
| 크기 | `m = hypot(lx, ly)` (az가 지배적이면 회전 경로, 아래 §3) |
| STOP | `m < 0.2` → 정지 (`robust_stop`) |
| 속력 스케일 | `s = clamp(m, 0.4, 1.0)` — 0.2≤m<0.4는 최저 이동 속도 0.4로 올림(0.4 미만은 보정·검증이 없음), m>1은 1.0 |
| 구동 | `publish_and_move(dir_lx, dir_ly, az, throttle_scale=s)`, `dir = (lx,ly)/m` (방향 단위벡터) |
| throttle | `effective_throttle = max(1, round(base_throttle × s))`, base 50 → 20(0.4) / 35(0.7) / 50(1.0), 사이 값은 연속 |
| 예상 속도 | 보정 선형식 `v ≈ 6.59 + 19.85·s` cm/s (측정점 s=0.4/0.7/1.0 → 14.45/20.64/26.36, 사이는 보간 가정, 미검증) |

근거: 학습 타깃 `actions_exec`의 크기가 곧 throttle scale(0.4/0.7/1.0)이므로 모델 출력 크기를 그대로 `throttle_scale`로 쓰면 학습·구동이 같은 단위다. 이미 `VLAControlManager.publish_and_move`가 `throttle_scale`을 지원한다(vla_control_utils.py:44~60).

## 2. 서버 변경 (`robovlm_nav/serve/stage2_v2_inference_server.py`, 승인 후)
1. 응답에 `action_cont = {dir_lx, dir_ly, az, speed_scale, mode}` 추가. 기존 `action`(8-class 표)은 호환용으로 유지, 요청 파라미터/설정 `action_mode=class|continuous`로 선택.
2. 연속 모드: FM 헤드(`scripts/train_egvla2_fm_head.py`의 A 또는 C) 체크포인트 로딩, Euler 10스텝, 샘플 K개의 **평균**을 첫 스텝 액션으로 사용(다봉성이 약한 현 데이터 기준; 데이터 확대 후 재검토).
3. 출력 후처리: §1 규칙(STOP 0.2, clamp 0.4~1.0).
4. 대시보드(`mona_dashboard.py` `_async_infer` 등)에서 `action_cont`가 있으면 `throttle_scale`을 함께 전달.
5. 로그: 예측 크기 m, 적용 s, 예상 속도(cm/s)를 세션 기록에 저장(사후 검증용).

## 3. 열린 항목 / 리스크
- **중간 속력 미검증**: 사용자 결정으로 선형 보간 가정을 그대로 쓴다. 이상 동작(저속 정지, 불균일)이 보이면 §1의 하한 0.4를 올리거나 단계 양자화로 되돌린다.
- **회전(az)**: 현재 구동은 `|az|>0.1`이면 spin만 한다. 모델 az 예측은 비영 28프레임이라 신뢰 불가(CH76-7) → 연속 모드 첫 적용 시 az는 0 처리(이동만)로 두고 회전은 기존 STOP-회전 규칙 유지 제안.
- **지연**: 추론 180~817ms 동안 최대 속력에서 5~21cm 이동. 속력을 크게 낼수록 반응이 늦다.
- **STOP 거리**: 속력별로 달라 현 임계(고정 속도 기준)가 안 맞을 수 있다. 첫 적용은 s≤0.7로 제한해 보는 안.
- **수집**: 현재 수집 구동은 3단계(방식 C). 연속 구동(방식 A)으로 바꾸는 건 이 계획 밖(별도 결정).
- 학습 데이터는 오픈 루프 15ep라 이 모드의 폐루프 성능은 확인된 바 없다.

## 4. 검증 계획
- 오프라인: 저장된 파일럿 프레임에 서버 연속 경로를 돌려 `action_cont` 크기·방향을 정답(exec)과 비교(축별 MSE, 단계값 근처 분포).
- 로봇 드라이런: 바퀴 없이/저속으로 `throttle_scale` 변화(0.4~1.0)가 명령에 반영되는지 로그 확인.
- 폐루프 실주행은 별도 승인 후.

## 5. 체크리스트
- [ ] 사용자 승인(§1 규칙·§2 변경 범위)
- [ ] 서버 `action_cont` 구현 + 8-class 호환 유지
- [ ] FM 체크포인트 저장 스크립트(학습 스크립트에 `--save`) 
- [ ] 대시보드 `throttle_scale` 전달
- [ ] 오프라인 검증 → 드라이런 로그
- [ ] CH76/SUMMARY에 결과 반영

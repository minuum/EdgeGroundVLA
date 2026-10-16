# 계획: EGVLA2 추론 서빙 경로 (Florence-2 fused → FM 헤드 → 속력 규칙)

> 작성일: 2026-10-15 | 상태: **사전 구현 완료(2026-10-15, 사용자 지시). 헤드 선택·폐루프만 남음**
> 관련: `plan_20261014_inference_continuous_speed.md`(속력 규칙·섀도), `plan_20261015_egvla2_300ep_pipeline.md`, CH76-10
> 배경: 현 `stage2_v2_inference_server.py`는 Kosmos/PG2·OWLv2 기반 구형 스택이라 Florence-2 fused 특징도 FM 헤드도 없다. 연속 속력 plan §2의 `action_cont`는 이 경로가 있어야 실제로 나온다.

## 1. 구성 (학습과 같은 입력을 서빙에서 만든다)
```
카메라 프레임 ─▶ Florence-2 fused(1회) ─▶ bbox 4-d [cx,cy,area,has_bbox] + 풀링 특징 768-d
                                         │  (precompute_egvla2_continuous_features.py와 동일 함수)
최근 3프레임 bbox 창(12-d) + 현재 768-d ─▶ FM 헤드(A/B2/C) Euler 10스텝, K개 샘플 평균 ─▶ H=8 청크
첫 스텝 (lx,ly,az) ─▶ continuous_speed.to_drive_command(섀도·az=0) ─▶ action_cont + throttle_scale
```
- 학습-서빙 일치가 핵심: 캐시 생성 코드(`run_batch`, `florence2_output_to_bbox`)를 **그대로 import**해 쓴다(재구현 금지). 입력 창 규칙(앞이 부족하면 첫 프레임 반복, `WINDOW=3`)도 학습 `load_episodes`와 동일하게.
- 지연: fp32 젯슨 fused 817ms / 서버 180ms(CH74). FM 헤드(Euler 10스텝×K=8)는 작은 MLP라 수 ms 수준으로 추정(미측정, 측정 항목에 포함).

## 2. 변경 파일 (승인 후)
1. `robovlm_nav/serve/egvla2_fm_policy.py`(신규): 헤드 클래스 로딩(`train_egvla2_fm_head`의 A/B2/C 정의를 공유 모듈로 분리), 체크포인트 로드, 창 버퍼, `predict(frame_feat, bbox)` → 청크·`action_cont`. 헤드 정의를 서빙에서 import하려면 학습 스크립트에서 모델 클래스를 `robovlm_nav/models/egvla2_fm_heads.py`로 옮기고 학습 스크립트가 거기서 import하도록 수정(동작 불변, 학습 결과 재현 확인 필수).
2. `scripts/train_egvla2_s1.py`: `SAVE=1` 경로 보강 — 전체 218ep로 학습한 최종 체크포인트에 `epoch`, 세트 해시, 입력 규격(window, bdim, fdim) 저장. 지금은 조기 종료 기준이 fold0 참고용이라 **별도 hold-out 없이 S1에서 얻은 최적 epoch 수로 고정 학습**하도록 바꾼다.
3. `robovlm_nav/serve/stage2_v2_inference_server.py`: `VLA_ACTION_MODE=class|continuous`(기본 class), continuous일 때만 `egvla2_fm_policy` 호출. 응답에 `action_cont`·`speed_shadow`(모델 속력)를 추가하고 기존 `action`은 유지. 기본 동작 불변.
4. `mona_dashboard.py`: `action_cont`가 있으면 `throttle_scale`을 전달. 섀도 모드에서는 규칙 속력으로 구동하고 모델 속력은 로그에만 남긴다.
5. 세션 로그에 예측 크기 m, 적용 s, 예상 속도(cm/s), 모델 청크 요약 저장.

## 3. 검증 (순서)
1. **동등성**: 같은 프레임에서 서빙 경로가 만든 bbox·특징이 캐시 값과 허용 오차 안에서 같은가(`verify_fused_bbox_equivalence.py` 방식).
2. **오프라인 재생**: 218ep를 서빙 경로로 재생해 예측 청크 vs 정답 exec의 축별 MSE·각도 오차가 S1 hold-out 수치와 같은 범위인가(학습 에피소드 재생은 낙관적이라 hold-out fold 체크포인트로 따로 확인).
3. **지연 측정**: 젯슨·서버 fp32에서 전체 파이프라인 ms, 이전 구형 스택과 같은 조건으로 비교.
4. **드라이런**: 로봇 정지, 출력·`throttle_scale` 로그만 확인.
5. **폐루프**: 별도 승인 후 15칸×20회=300회(plan_20261015 §6). 첫 적용은 섀도·az=0·s≤0.7.

## 4. 트레이드오프·리스크
- 헤드 클래스 분리는 학습 스크립트 수정이 필요해 재현성 위험이 있다 → 분리 전후 1-fold 결과가 같은지 확인 후 진행.
- 체크포인트 선택: 지금 S1에서는 A/B2/C가 거의 같다(MSE 차 ≤0.003). 방향 오차는 C가 낮고(11.3°) 지연은 A가 가장 작을 것이므로 **epoch 확대 재시험 결과(진행 중)로 정한다**. 기본 후보 C, 비교 후보 B2.
- az는 학습되지 않으므로(R²≈0.05) 첫 적용에서 쓰지 않는다. 회전은 기존 8-class 규칙 유지.
- 속력(모델)은 가이드 이전 데이터라 거리와 무관(ρ=-0.02) → 섀도 기록만.
- 한 장소 수집이라 위치 일반화 미확인.

## 5. 체크리스트
- [x] 사용자 승인("미리 구현할 수 있는 것은 미리", 2026-10-15)
- [x] 헤드 클래스 공유 모듈 `robovlm_nav/models/egvla2_fm_heads.py` 분리 — 분리 전후 4개 구조의 초기 가중치·샘플 출력 비트 단위 동일 확인
- [x] 체크포인트 규격 보강: `SAVE_ONLY=1 FINAL_EPOCHS=N`(전체 학습, 파일 목록·epochs·azmask 포함). 최종 학습은 300ep 파이프라인 6단계가 수행
- [x] `egvla2_fm_policy.py`(정책·F2 추출기) + 서버 `VLA_ACTION_MODE=continuous`(기본 class, 실패해도 class 경로 영향 없음) + 대시보드 `VLA_APPLY_CONT_SPEED=1`일 때만 throttle 적용
- [x] 동등성(점검용 체크포인트 + 앞 6프레임): bbox 최대 차이 0.0, 특징 최대 차이 0.002(fp16 반올림 수준)
- [x] 오프라인 재생 스크립트 `scripts/replay_egvla2_policy.py`(동작 확인만, 점검 체크포인트는 3 epoch라 성능 수치는 의미 없음)
- [ ] 최종 체크포인트로 재생·hold-out 비교, 지연 측정(젯슨·서버 fp32)
- [ ] 드라이런 → 폐루프 승인 요청

## 5.1 켜는 법 (서버)
`VLA_ACTION_MODE=continuous VLA_EGVLA2_CKPT=runs/egvla2_fm/C_pi0_adaln.pt VLA_SPEED_SOURCE=shadow` 로 서버 실행 → 응답에 `action_cont`(dir_lx, dir_ly, speed_scale, model_speed_scale, throttle, expected_cmps, stop) 추가. 대시보드에서 실제 구동 속력에 반영하려면 `VLA_APPLY_CONT_SPEED=1`(기본 0). 추출기는 transformers가 있는 `.venv`에서 동작(`.venv_lerobot`에는 없음).

## 6. 결정 필요
1. 서빙 헤드 선택 기준(방향 오차 vs 지연) — epoch 확대 결과 후 확정.
2. 헤드 클래스를 공유 모듈로 옮기는 학습 스크립트 수정 허용 여부.

## 최종 체크포인트(299건 학습) 서빙 경로 재생 점검 (2026-10-16)
`replay_egvla2_policy.py --ckpt runs/egvla2_fm_299/{A,B2,C}.pt --equiv 6` (학습 에피소드 5개·405프레임 재생이라 **낙관적, 동작 확인용**):
A 방향 5.1°·GO/STOP F1 0.967, B2 3.1°·0.960, C 2.8°·0.978. 서빙 추출기 vs 캐시: bbox 차 0.0, 특징 차 0.002(세 체크포인트 동일), 섀도 적용 속력 s=0.70. 결과: `docs/v5/egvla2_fm/replay_*_299.json`. hold-out 성능은 S1/LOPO 값을 볼 것. 300건 재학습 체크포인트로 완료 후 같은 점검을 반복한다.

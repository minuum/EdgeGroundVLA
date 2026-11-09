# T7 기록 세션 재생 비교 (A3 vs V4) — 2026-11-09, 사용자 승인("진행 ㄱㄱ해바")
목적: 실기 100% 성공한 A3(`C_pi0_servolabel1.0_drop30_e120.pt`, EG2-09)와 텍스트 조건화 V4(+V0 기준)가 같은 실기 프레임에서 어떻게 다르게 내는지 본다.
데이터: docs/inference_sessions A3 세션(runtime_config ckpt=servolabel1.0, servo_blend=0.0) 452개. 입력은 세션 H5의 이미지·서버 박스(`grounding/bbox`)·실행 액션(`actions`).
단계: (1) `scripts/t7_extract_session_feats.py` 이미지 풀링 특징 + 문구별 텍스트 특징(생성 없음) → runs/egvla2_fm/t7_session_cache.npz
(2) `scripts/t7_train_save.py` V0·V4를 300에피소드 전체로 120ep 학습, 체크포인트 저장(PN 부여) (3) `scripts/t7_compare.py` 비교 → docs/v5/egvla2_fm/t7_replay.json
한계: 기준은 "A3가 낸 명령"이라 유사도만 잰다(품질 아님). 실기 성공 예측 불가. V4 체크포인트는 실기 미검증.

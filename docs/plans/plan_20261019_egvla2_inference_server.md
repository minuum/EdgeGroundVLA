# 계획: EGVLA2 실제 추론 서버 — 연속 방향으로 로봇을 구동하기 위한 구현안

> 작성일: 2026-10-19 | 상태: **계획 — 구현 승인 대기** | 관련: plan_20261016_egvla2_serving_path.md, 핸드오프 §15, CH77

## 0. 현재 상태 (코드 확인)
- 서버 `stage2_v2_inference_server.py`: `VLA_ACTION_MODE=continuous`이면 기존 8-class 파이프라인(OWLv2·Kosmos-2 계열 Stage1/2, 프리뷰 ROT 재시도, STOP 래치, 근접 오버라이드)을 **그대로 돌린 뒤** 응답에 `action_cont`만 추가한다(Florence-2 + FM은 병렬이 아니라 추가 연산).
- 대시보드 `mona_dashboard.py`는 구동에 `res["action_3d"] or res["action"]`(8-class 경로)를 쓰고, `VLA_APPLY_CONT_SPEED=1`이면 `action_cont.speed_scale`만 적용한다. **연속 방향(dir_lx, dir_ly)은 구동에 쓰이지 않는다.**
- 따라서 지금 상태로는 "EGVLA2로 로봇을 움직인다"가 성립하지 않는다. 지연도 EGVLA1 경로 + Florence-2 합산이라 EGVLA2 단독 지연(817ms fp32)을 대표하지 못한다.

## 1. 구현 단계
### 단계 A — 하이브리드 구동(저위험, 먼저)
- 새 환경변수 `VLA_DRIVE_SOURCE=egvla2_hybrid`(기본 `class`). 기존 서버가 STOP·ROT·프리뷰를 결정하고, **이동 프레임에서만** 구동 방향·속력을 `action_cont`(dir_lx, dir_ly, speed_scale)로 대체. az는 0(첫 적용).
- 서버 응답에 `drive`(적용된 구동 명령)와 `drive_source` 필드 추가. 대시보드는 `drive`가 있으면 그것을 `new_lx/new_ly/throttle_scale`로 사용. 비교용으로 `action_3d`(class 값)도 응답에 유지·로그.
- 목적: 구동 안전 장치를 그대로 두고 연속 방향의 실기 거동을 먼저 본다. 단점: 지연이 EGVLA1+Florence-2 합산이라 지연 비교는 안 됨.

### 단계 B — EGVLA2 단독 서버(본 평가용)
- 새 모드 `VLA_ACTION_MODE=egvla2_only`(또는 별도 파일 `egvla2_inference_server.py`): 이미지 디코드 → `FusedF2Extractor`(fp32) → `EgVla2FMPolicy` → 평활 → 정지·탐색 규칙 → 기존 응답 계약(`action`, `action_3d`, `bbox`, `latency_ms`, `stop_latched` 등)으로 변환.
- 기존 서버에서 EGVLA2가 스스로 처리하지 못하는 동작을 어떻게 대체할지 결정 필요(아래 §2).
- 지연은 이 모드로만 대표값이 나온다.

## 2. 사용자 결정이 필요한 항목
1. **정지 규칙**: 헤드의 |m|<0.2(GO/STOP F1 0.95)에 디바운스 N프레임 + 한 번 정지하면 래치(기존 STOP 래치와 동일)? 근접 오버라이드(바구니 면적 임계)를 유지?
2. **타깃 소실/초기 탐색**: 첫 프레임에 그라운딩 실패 시 기존처럼 ROT 후 재시도(CH54 프리뷰)를 이식할지, `has_bbox=0`이 N프레임 지속되면 제자리 회전 탐색을 할지. 서브골 드롭아웃 체크포인트는 결측에서 직진 편향 없이 방향을 낼 수 있으나 안전 규칙은 별도.
3. **회전(az)**: 첫 적용은 az=0 확정이므로 ROT 클래스 동작은 기존 규칙 유지(단계 A) 또는 탐색 규칙만(단계 B)?
4. **평활**: K=8 평균 + EMA(α=0.5?) + 최대 변화율 제한. 저크가 사람의 2~3배라 필요(CH77 ⑥-3). 파라미터는 드라이런 로그로 정한다.
5. **관측 입력**: JPEG q95 이상(핸드오프 §15.2) 대시보드 설정 변경 승인, 기록 항목(충돌·개입·시간) 추가.

## 3. 선행 점검 (minum 측정 완료)
- fp16 vs fp32 특징 차이: 방향 0.03° (무시 가능). 지연 비교는 fp32.
- 대시보드 JPEG q80 경로: 방향 평균 4.1°(최대 17°) 어긋남 → q95 이상 필요(0.2°). `docs/v5/egvla2_fm/serving_equiv*.json`.
- 서브골 결측 처리: 정책이 `has_bbox<0.5`이면 bbox 창 전체를 0으로(드롭아웃 체크포인트 기본). 단위 점검 통과.

## 4. 변경될 파일(승인 후)
- `robovlm_nav/serve/stage2_v2_inference_server.py`(`VLA_DRIVE_SOURCE`, `drive` 응답, 단계 B 모드), `robovlm_nav/serve/mona_dashboard.py`(`drive` 사용, JPEG 품질 옵션), `robovlm_nav/serve/egvla2_fm_policy.py`(EMA·정지 디바운스), `robovlm_nav/serve/continuous_speed.py`(정지 래치 옵션), 단위 테스트, 핸드오프/plan 갱신.

## 5. 리스크
- 연속 방향 첫 구동은 오프라인으로 검증된 적이 없다(안 본 위치 방향 오차 약 17°, 안 본 시작 위치는 미측정). 단계 A로 하이브리드 구동을 먼저 하고, 드라이런 로그로 평활·정지 규칙을 정한 뒤 단계 B로 간다.
- 로봇 세션과 코드 동기화(로봇 쪽 repo가 최신인지, `mona_dashboard.py` 로컬 수정 여부) 확인 필요.

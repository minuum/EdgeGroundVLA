# 계획: EGVLA2 실제 추론 서버 — 연속 방향으로 로봇을 구동하기 위한 구현안

> 작성일: 2026-10-19 | 상태: **승인(2026-10-19) — 단계 A 구현 완료(기본 꺼짐), 단계 B 미착수** | 관련: plan_20261016_egvla2_serving_path.md, 핸드오프 §15, CH77

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

## 6. 구현 현황 (2026-10-19, 단계 A)
- `robovlm_nav/serve/continuous_speed.py`: `hybrid_drive()` 순수 함수(이동 클래스 1~5이고 막힘 없음, EGVLA2가 이동일 때만 연속 방향×1.15·speed_scale·az=0 반환. STOP·ROT·막힘·EGVLA2 정지·오류는 None=기존 판단). 테스트 7개 통과.
- 서버: `VLA_DRIVE_SOURCE=egvla2_hybrid`(기본 `class`)일 때 응답에 `drive`·`drive_source` 추가. `VLA_EGVLA2_EMA=<alpha>`(기본 0=끔)로 정책의 EMA 평활.
- 정책: `ema_alpha`(첫 스텝 지수이동평균, reset 시 초기화), 서브골 결측 처리(`subgoal_dropout>0` 체크포인트 기본).
- 대시보드: `VLA_DRIVE_SOURCE=egvla2_hybrid`이고 응답에 `drive`가 있으면 방향·throttle_scale을 대체(az=0). 서버 전송 JPEG 품질을 `VLA_JPEG_QUALITY`(기본 80=기존)로 조절, EGVLA2 경로는 95 이상.
- 기본값은 모두 기존 동작이며, 켜는 법: 서버 `VLA_ACTION_MODE=continuous VLA_EGVLA2_CKPT=runs/egvla2_fm/deploy/C_pi0_adaln_drop30_e120.pt VLA_SPEED_SOURCE=shadow VLA_DRIVE_SOURCE=egvla2_hybrid VLA_EGVLA2_EMA=0.5`, 대시보드 `VLA_DRIVE_SOURCE=egvla2_hybrid VLA_JPEG_QUALITY=95`.
- **미확인(드라이런에서 확인)**: 연속 방향에 곱한 gain 1.15가 컨트롤러의 lx/ly 단위와 맞는지(대각선 속도는 직진과 같다고 가정), EMA α 값, EGVLA2 정지 판단과 기존 STOP의 불일치 빈도.
- 결정(승인 시 적용한 기본값): 단계 A에서는 STOP·ROT·프리뷰를 기존 규칙이 담당(결정 1~3은 단계 B에서 확정), 평활은 EMA 옵션(기본 끔), JPEG 품질은 환경변수로.

## 7. 단계 B 스켈레톤 구현 현황 (2026-10-20, 구현만·미커밋·실서버 미구동)
**구현 파일**
- `robovlm_nav/serve/egvla2_only_logic.py` — 순수 상태기계 `EgVla2Controller`(모델 import 없음). 상태 DRIVE / SEARCH / STOPPED(래치).
- `robovlm_nav/serve/egvla2_inference_server.py` — 별도 경량 FastAPI 서버(기존 `stage2_v2_inference_server.py` 무수정). `/health`, `/predict`, `/reset`, `/model/info`. 이미지 b64→PIL RGB→`FusedF2Extractor`→`EgVla2FMPolicy`→컨트롤러→응답 dict(response_model로 키가 잘리지 않게 dict 그대로 반환). 모델은 시작/첫 요청에서 지연 로드, 서비스는 추출기·정책·컨트롤러를 주입받아 모의 객체로 테스트 가능.
- `robovlm_nav/serve/egvla2_fm_policy.py` — `FusedF2Extractor(dtype=torch.float16)` 인자 추가(기본 동작 동일).
- `scripts/precompute_v6_florence2_fused.py` — `run_batch(..., dtype=torch.float16)` 인자 추가(기본 동일; fp32 입력 캐스팅에 필요).
- 테스트: `tests/test_egvla2_only_logic.py`(11), `tests/test_egvla2_inference_server.py`(7), 기존 `tests/test_continuous_speed.py`(7) 모두 pytest shim 직접 실행으로 통과(.venv, 모델·GPU 불필요). fp32/fp16 추출기가 학습 캐시 대비 특징 코사인 1.000000으로 로드·동작함을 1프레임으로 확인.

**규칙 기본값(승인안, 생성자·환경변수로 조정)**: `stop_debounce=3`(|m|<0.2 연속 3프레임 → STOPPED 래치, 도달 전 이동 0), `near_area=0.25`·`near_frames=2`(`off`로 끔), `lost_frames=5`(has_bbox=False 연속 → SEARCH 제자리 회전 az=±0.25, 마지막 cx<0.5 좌(+)/그 외·미관측 우(-)), `search_max_frames=40`(초과 시 STOPPED), DRIVE는 단위 방향×`gain=1.15`·az=0·`speed_scale`=정책 값. 각 전이는 `reason`에 기록(예: `stop_pending(2/3)`, `near_goal`, `search_start`, `target_reacquired`, `search_timeout`, `latched:*`).

**켜는 법(이 서버를 stage2 서버 대신 띄울 때만)**: `VLA_EGVLA2_CKPT=runs/egvla2_fm/deploy/C_pi0_adaln_drop30_e120.pt VLA_EGVLA2_DTYPE=fp32 VLA_EGVLA2_EMA=0.5 VLA_SPEED_SOURCE=shadow .venv/bin/python3 -m robovlm_nav.serve.egvla2_inference_server --port 8001`. 대시보드는 그대로(`/predict` 응답의 `action_3d`·`action_speed_scale`·`goal_near`를 사용). 기존 stage2 서버·대시보드 기본 동작은 변경 없음.

**대시보드 호환 키**: `action`, `action_3d`(lx,ly,az), `action_speed_scale`(throttle), `goal_near`(STOPPED이면 True → 대시보드가 주행 종료), `stop_latched`, `bbox{cx,cy,area,has_bbox}`, `predicted_label`("STATE:reason"), `latency_ms`, `grounding_latency_ms`, `grounding_cached`, `drive`, + `action_cont`, `controller_state`, `reason`, `subgoal_missing`, `ckpt`, `ckpt_sha256`, `dtype`.

**미확인·주의**
- 실서버·로봇·대시보드 연동은 하지 않았다(드라이런 전). gain 1.15와 컨트롤러 lx/ly 단위, SEARCH 회전 중 대시보드의 `move_and_stop_ramped`(move_duration) 동작, 탐색 az=0.25 크기·부호(+=좌)는 실기 확인 필요.
- `goal_near=True`는 도달뿐 아니라 STOPPED 전부(탐색 초과·정지 디바운스)에 쓴다 → 대시보드가 주행을 종료한다. `reason`으로 구분.
- **기존 stage2 서버의 `/predict`는 `response_model=InferenceResponse`로 직렬화하는데 이 모델에 `action_cont`·`drive` 필드가 없어, 단계 A의 `drive`·`action_cont`가 HTTP 응답에서 잘린다**(`stage2_v2_inference_server.py` 1552행 `InferenceResponse`, 1778행 `/predict`). 단계 A를 쓰려면 이 모델에 두 필드를 추가해야 한다(이 작업 범위 밖이라 수정하지 않음).
- 정지 래치는 `/reset`에서만 풀린다. 타깃 소실 탐색은 첫 프레임 프리뷰(CH54)를 대체하지 않는다. 탐색 중 정책 EMA 상태는 유지된다.

## 8. 버그 수정과 권장 설정 (2026-10-20)
- **버그**: `/predict`의 `response_model=InferenceResponse`에 `action_cont`·`drive`·`drive_source`가 없어 FastAPI가 응답에서 잘라 냈다(포크 에이전트가 발견, 코드로 확인). 단계 A의 `drive`와 기존 `action_cont`가 대시보드까지 도달하지 못했다. `InferenceResponse`에 세 필드(Optional)를 추가해 수정, pydantic 직렬화 확인.
- **권장 평활**: K=8, `VLA_EGVLA2_EMA=0.5`(새 값 가중, 오프라인 스윕 `smoothing_sweep.json`: 저크 0.056 ≈ 사람 수준, 지연 ≤2프레임). 드라이런은 이 값으로 시작.
- 서버 dtype: 단계 A 서버의 추출기는 기본 fp16(지연 짧음), 단계 B 서버는 fp32 기본(`VLA_EGVLA2_DTYPE`). 학습 캐시(fp16) 대비 fp32 차이는 방향 0.03°.

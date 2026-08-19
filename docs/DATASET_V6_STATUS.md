# 데이터셋 V6 — 명명 규정 (2026-07-15)

> 로봇서버(soda, `monavla-driving`)/학습서버(minum, `monavla-train`) 공유 문서.
> H5 데이터 자체는 `.gitignore`로 제외되어 git에 올라가지 않음 — 이 문서는
> "지금부터 새로 수집되는 데이터를 V6로 부른다"는 **명명 규정과 근거**만 공유.

## 규정

**2026-07-15부터 `📷 데이터수집` 대시보드 탭(포트 7800, `mona_dashboard.py`
`DataCollectSession`)으로 수집하는 데이터를 V6로 명명한다.**

- 저장 위치는 기존과 동일: `ROS_action/mobile_vla_dataset_v5/` (디렉터리명은
  안 바꿈 — 이미 여러 스크립트/경로가 `v5` 경로를 참조하고 있어 이름만으로
  버전을 구분하면 하위 호환이 깨짐. **버전 구분은 H5 파일 스키마로 한다**,
  아래 참조)
- `VLA_DATASET_DIR` 환경변수로 오버라이드 가능(변경 없음)

## V5 vs V6 스키마 차이 (자동 판별 가능)

`🗂 데이터셋 히스토리` 탭(`/dataset/list`)이 파일별로 자동 분류하는 것과
동일 기준:

| | V5 이하(레거시, `schema: "legacy"`) | **V6(신규, `schema: "new"`)** |
|---|---|---|
| H5 키 | `observations/images`, `language_instruction`, `actions` | `images`, `actions`, `action_event_types` |
| attrs | `scenario`, `pattern`, `distance`, `end_pos` | `episode_name`, **`scenario`**(2026-07-15부터 추가), `cx_position`, `cx_path`, `total_duration`, `num_frames`, `stop_inject_n`, `action_chunk_size`, `obstacle_layout_type`, `time_period`, `collection_datetime` |
| 수집 방식 | `mobile_vla_data_collector.py` 원본(터미널 raw keyboard) | 대시보드 웹 UI — 키보드/조이스틱 공용 단일 진입점(`VLAControlManager.publish_and_move`) |
| 수집 대상 | 9-시나리오(`target_*`) + 18-key core/variant 패턴 | 트랙A 극단배치(`cx_position` × `cx_path`, 위치당 3×15=45개, 총 180개) + 기존 9-시나리오 병행 가능 |

## 참고 — 테스트 데이터 오염 케이스 (처리 완료)

2026-07-15 데이터수집 탭 기능검증 중 실제로 1개 에피소드가 저장되어
(`episode_20260715_173157_weak_left_left_curve.h5`) `scenario_progress.json`/
`time_period_stats.json`을 건드렸음 — 실데이터가 아닌 검증용 산출물로 판단,
삭제 후 progress json을 `git checkout`으로 원상복구함(과거에도 동일 이슈
전례 있음, `docs/plans/plan_20260707_dashboard_data_collector_tab.md` 참조).
**현재 V6 스키마 실데이터는 0건 — 아직 본 수집 시작 전.**

## 2026-07-15 세션 변경사항 요약 (대시보드 코드, `mona_dashboard.py`)

- **🗂 데이터셋 히스토리 탭 신설** (📷 데이터수집 바로 아래) — V5/V6 두 스키마를
  자동 판별해 목록+프레임인스펙터로 브라우징. 신규 엔드포인트
  `/dataset/list`, `/dataset/load`, `/dataset/frame` (`/sessions/*`와는
  별개 — 그쪽은 추론세션 전용, 이쪽은 원본 학습데이터 전용)
- 목록 필터: 스키마(V5/V6) 버튼, 시나리오 버튼, 트랙A cx위치 클릭형 칩,
  이름 검색 — 전부 버튼 클릭형으로 구성(드롭다운 최소화)
- 다중 선택(체크박스, 최대 6개) → 비교 카드 모드, 개별 "🔍 자세히"로 단일
  상세 전환
- 상세 패널: 세션 히스토리와 동일하게 좌/우 방향키로 프레임 넘기기, Space로
  재생/정지. 수집 설정(패턴/장애물배치/시간대/STOP주입/액션청크/파일크기)과
  입력 소스(키보드/조이스틱/stop_inject 등 event_type) 분포 추가 노출
- `_save_episode_data()`에 `scenario` attr 저장 추가 — 이제부터 저장되는
  V6 파일은 시나리오 필터가 정상 동작함(과거엔 cx_position/cx_path만 있고
  scenario가 비어 있었음)

## 현재 상태 — V6 첫 실데이터 1건 수집됨 (2026-07-15 19:23)

세션 후반부에 "방금 수집한 데이터셋이 하나 있을 것"이라는 문의가 있었고,
처음 재스캔 시점엔 실제로 없었으나(당시엔 테스트 파일 삭제 직후였음),
이후 실제로 1건이 새로 수집됨을 확인:

- `episode_20260715_192339_weak_left_left_curve.h5`
- `cx_position=weak_left`, `cx_path=left_curve` (트랙A 극단배치 — 준극단좌 + 좌곡선)
- 72 frames, 15.0s, `time_period=evening`, 93.7MB
- `scenario` attr은 빈 문자열 — 트랙A 수집이라 9-시나리오 축은 안 씀(정상)
- `🗂 데이터셋 히스토리` 탭에서 V6/오늘날짜/weak_left+left_curve로 정상 노출 확인

`scenario_progress.json`/`time_period_stats.json`에도 반영됨
(`cx_position_stats.weak_left=1`, `time_period_stats.evening=1`).

## 최신 현황 업데이트 (soda → inference-integration 동기화, 2026-07-17)

> 이 섹션은 soda `monavla-driving`에서 이후 진행된 물리 수집 결과를
> 요약 반영한 것 — 위 "V6 첫 실데이터 1건" 이후 실제로는 트랙A+트랙F
> 전량 수집이 끝났음. `monavla-driving`은 대시보드 코드(V6/트랙C/트랙F
> UI, LeRobotDataset 변환 스크립트 등) 커밋이 20개 이상 더 있어 전체
> 브랜치를 merge하지 않고 이 요약만 옮김 — 상세 커밋 이력이 필요하면
> `monavla-driving` 참조.

- **트랙A(극단배치 4위치×3경로) 180/180 완료** — weak_right/strong_right/
  weak_left/strong_left 각 45(15×3), soda→minum rsync 전송 완료
  (`/home/minum/26CS/MoNaVLA/ROS_action/mobile_vla_dataset_v5/`)
- **트랙F(center 위치, plan §1-2) 45/45 완료 — 2026-07-16/17.**
  `mona_dashboard.py`의 `cx_position`에 `center`(cx 0.475~0.525) 추가,
  동일 3경로(left_curve/straight/right_curve)×15회. soda→minum 전송
  완료, minum 쪽 개수 45건 일치 확인.
- **→ V6가 5위치(weak_right/strong_right/weak_left/strong_left/center)
  × 3경로 = 225ep로 완결.** V5 레거시 혼합 없이 v6 단독 arm 학습 가능.
- 손상 파일 2건(`strong_left::straight`, 디스크 압박 추정) 발견·soda
  로컬에서 삭제 처리 — 정상 파일만으로 목표 충족이라 재수집 불필요.
- **남은 것: 트랙C(오버슈트→재수렴, 64ep)만.** soda `mona_dashboard.py`에
  `overshoot_left_recover`/`overshoot_right_recover` cx_path 라벨 추가
  완료(수집 UI 준비됨), 실제 물리 수집은 미착수 — exp73(트랙A 단독 학습)
  결과 확인 후 우선순위 판단 예정.
- 참고용 데모 탭도 추가됨: 🌀 오버슈트 가이드(soda 7800 대시보드) — CH62
  반례 세션(`session_20260711_205228`)의 실제 프레임 + 좌우반전 합성으로
  "왜 트랙C가 필요한지" 시각적으로 보여주는 탭(실제 학습데이터 아님, 예시용).

## 🚧 [2026-07-21] 실기 테스트 착수 전 확인 필요 — soda에 exp73 자산이 없음 (soda → minum)

`monavla-driving` 최신 요청(`81a1ec3d` 실기 테스트 요청)에 대한 확인 결과입니다.
착수 전 soda 쪽 상태를 실제로 확인해보니 아래 두 가지가 **soda에는 물리적으로
존재하지 않음**을 확인했습니다.

1. **체크포인트 파일 3개 부재**: soda `runs/v5_nav/mlp/`엔 `exp71`까지만 있고,
   `exp73_pg448_v6_mlp.pt` / `exp73_pg448_trackF_v6_mlp.pt` /
   `exp73_pg448_trackF_v6_hybrid.pt` 전부 없음 (minum 학습 산출물이라 당연히
   minum→soda 전송이 아직 안 된 상태로 보임)
2. **서버 통합 코드도 soda 브랜치(`monavla-driving`)엔 없음**: `inference_server.py`
   exp73 variant 통합(`GoalNavMLPInference`의 `_DEFAULT_CKPTS["exp73_hybrid"]`,
   `_HYBRID_VARIANTS`, HybridHead forward 로직 등)은 `inference-integration`
   브랜치(이 파일 있는 브랜치)에만 있고, soda가 실제 로봇을 구동하는
   `monavla-driving`으로는 아직 안 건너온 상태입니다.

**요청**:
1. 체크포인트 3개를 soda의 `runs/v5_nav/mlp/exp73/`로 전송(또는 전송 스크립트/경로
   알려주시면 soda에서 pull)
2. `inference_server.py`의 exp73 관련 diff를 `monavla-driving`에 반영 가능한
   형태로 공유 요청 — 두 브랜치가 많이 갈라져 있어 전체 merge보다 exp73 관련
   부분만 별도 커밋/패치로 받는 걸 선호

위 2개 확인되는 대로 soda에서 3-variant 반복 실기 테스트(요청하신 mlp 우선,
trackF-mlp 비교, hybrid 후순위) 바로 착수하겠습니다. (`monavla-driving`
`49501ad9`에 동일 내용 기록됨)

## 관련 문서

## ✅ [2026-07-22] 요청 2건 전달 완료 (minum → soda)

앞선 "🚧 실기 테스트 착수 전 확인" 요청에 대한 응답입니다. `runs/`는 기본
`.gitignore` 대상이지만 이 3개 체크포인트는 각 ~3.4MB로 작아 **git에 force-add로
직접 커밋**했습니다 — 별도 전송 스크립트 없이 두 브랜치 모두 `git pull`만 하면
받아집니다.

1. **체크포인트 3개** — `runs/v5_nav/mlp/exp73/`에 커밋됨:
   - `exp73_pg448_v6_mlp.pt` (정정된 진짜 1위, 우선 테스트)
   - `exp73_pg448_trackF_v6_mlp.pt` (비교용)
   - `exp73_pg448_trackF_v6_hybrid.pt` (참고용, 서버엔 이미 variant로 있음)
   - `inference-integration` 커밋 `2c81ddf4`, `monavla-driving`에도 동일 반영(아래)
2. **`inference_server.py` diff** — 전체 merge 대신 요청하신 대로 별도 패치 파일로:
   `docs/patches/0001-exp73-hybrid-variant-inference_server.patch`
   (커밋 `9f3fcc8c` 단독, `git apply`로 `monavla-driving`에 깨끗하게 적용 확인 완료 —
   `git apply docs/patches/0001-exp73-hybrid-variant-inference_server.patch` 또는
   `git am`으로 커밋까지 그대로 가져올 수 있음)

이 2개 반영되면 요청하신 3-variant 반복 실기 테스트(mlp 우선, trackF-mlp 비교,
hybrid 후순위) 바로 착수 가능합니다.

## 🚨 [2026-07-23] 수집 대비 실기 프레임/액션 밀도 불일치 — 실측 확정 (soda → minum)

exp73 실기 검증(트랙A/F 위치별 스크리닝, 21회) 중 발견: **수집 시 프레임/액션 빈도와
실제 추론 시 로봇이 움직이는 빈도가 크게 다름.** 학습 요소 없이는 근본 해결이
어려워 보여, 실측 수치 + soda 쪽 즉시조치(완료) + minum 검토 요청(대기)을 정리합니다.

### 실측 — 수집 vs 추론 cadence

| | 수집(H5, joystick) | 추론(exp73_mlp, PG2-448) |
|---|---|---|
| 실측 fps | **~6.0Hz** (5개 에피소드 실측 5.2~6.6, `attrs.num_frames/total_duration`) | **~1.3Hz 평균, 비균일** |
| 스텝 간 실제 간격 패턴 | 연속(모든 프레임 균일 간격) | **버스트형**: 0.08s → 0.08s → **2.2s** 반복(3스텝 주기) |
| window=6이 담는 실시간 폭 | 6/6.0 ≈ **1.0초** | 6/1.3 ≈ **4.6초** (약 4.6배) |

버스트 패턴의 원인: `grounding_skip_n=3` — 3스텝마다 1번 "신선 그라운딩"(PG2 median
2.13s, p95 3.27s, `logs/grounding_decisions.jsonl` n=273 실측)을 부르고 나머지
2스텝은 캐시(∼65ms)를 씀. **OWLv2로 바꿔도 median 1.93s로 비슷**(n=8070) — CH64
"그라운더 무죄"가 여기서도 재현, 그라운더 문제가 아니라 구조적 지연.

### 추가 발견 — 물리적 정지-재개(stop-go) 스터터

`_async_exec`/`_goalnav_exec`의 `COAST=1.2s`(정해진 시간 내 새 액션 없으면
velocity=0) < 실측 그라운딩 지연(2.2s) → **3스텝마다 로봇이 실제로 약 1초간
멈췄다가 재개**하는 인위적 스터터가 있었음(세션 `20260723_101821` 히스토리로
`[0.08, 0.08, 2.2, 0.09, 0.08, 2.21, ...]` 확인). 수집 데이터는 6Hz 연속
조이스틱이라 이런 정지 구간이 원천적으로 없음 — **모션 연속성 자체가 train/inference
간 다른 분포**라는 뜻.

### soda 즉시조치(완료, `monavla-driving` 반영됨)

- `COAST` 1.2s → **4.0s**(p95=3.27s보다 여유) 상향 — 정상적인 그라운딩 지연에
  로봇이 인위적으로 정지하는 걸 막음. 재학습 없이 바로 적용 가능한 부분만 우선 처리.

### minum 검토 요청 — 학습 쪽 요소

COAST 상향으로 "강제 정지"는 없앴지만, **window=6이 담는 실시간 폭(1.0s vs 4.6s)
자체는 여전히 다름** — 이건 추론 속도를 아무리 최적화해도(그라운더 교체 무효였듯)
근본적으로는 안 좁혀질 가능성이 높습니다. 검토 요청:

1. 학습 시 window 구성을 원본 6Hz 프레임 그대로 쓰지 말고, **실제 서빙 cadence(~1.3Hz,
   버스트형)에 맞춰 리샘플링**해서 만드는 것도 고려 가능한지(윈도우 1칸이 담는 실제
   이동거리/시간을 학습·서빙 간 맞추는 방향)
2. 혹은 반대로 **서빙 cadence를 수집 cadence(6Hz)에 최대한 가깝게 만드는 게 더
   합리적인지**(soda 쪽에서 grounding 비동기화/파이프라이닝으로 시도 가능하나
   구조 변경이 커서 minum 의견 먼저 필요)
3. 참고 세션: `session_20260723_101821.json`(60스텝, 버스트패턴 확인용),
   `logs/grounding_decisions.jsonl`(PG2/OWLv2 latency 분포 n=273/8070) — 필요하면
   soda→minum 전송하겠습니다.

이 불일치가 이번 검증에서 관찰된 스퓨리어스 STOP/방향 오류의 배경 요인 중
하나일 가능성이 있습니다.

## ✅ [2026-07-23] 액션 실행 메커니즘 답변 — 코드 직접 확인 (soda → minum, `monavla-driving` 질문 응답)

minum이 `monavla-driving`에 남긴 3개 질문(수집 cmd_vel 매핑 / 서빙 시 액션 유지
방식 / 속도값 물리적 동일성)에 대한 답변입니다. 세 질문 전부 코드 실측으로
답 가능했습니다. **결론부터: 보간/감속은 어디에도 없고(둘 다 순수 bang-bang),
물리 속도는 lx/ly/az 수치값과 무관하게 완전히 고정값(throttle)이라 수집/서빙
간 속도값 차이는 물리적으로 영향이 없습니다.** 단, 오늘 보낸 세션 25개는
전부 COAST 수정 **이전** 데이터라 그 부분만 정정합니다.

### 1) 수집 시 조이스틱→cmd_vel 매핑 (`mona_dashboard.py`, `DashboardJoystickReader._loop`)

- 기본 모드는 **ASYNC**(`_js_mode='async'`, :746) — SYNC(bang-bang 0.45s 주기) 아님.
  독트 "6.2Hz 연속 조이스틱 수집"과 일치.
- 스틱 값 → 8방향 키(`_axis_to_key`) → `WASD_TO_VEL` 고정 테이블 조회(:732, 예:
  `W:(1.15,0,0)`) → `speed/1.15` 배율 → `ctrl.publish_and_move(*vel)`를 **매 루프
  (ASYNC_INTERVAL=0.10s 목표, 오버헤드로 실측 ~6Hz) 그대로 재호출**(:980-983).
- **보간/감속 전혀 없음** — 스틱 놓으면(neutral) 다음 루프에 즉시
  `publish_and_move(0,0,0)`(:986), 관성/램프 없이 순간 정지. 유지 시간 개념
  자체가 없고 "누르는 동안 매 루프 그 값 그대로 반복 발행"이 전부.

### 2) 서빙 시 액션 적용 (`_async_exec`, mona_dashboard.py :1499-1518)

- 10Hz 루프(`sleep(0.1)`)가 **매 iteration마다 마지막으로 받은 lx/ly/az를
  그대로 재발행**(`ctrl.publish_and_move`) — 수집과 **동일 함수, 동일 무보간
  방식**. 새 예측이 없어도 이전 값을 계속 재발행한다는 점에서 근본적으로는 (a).
- 단 `COAST` 타임아웃을 넘기면 그 시점부터 (0,0,0) 재발행으로 전환.
- **⚠️ 정정**: 오늘 전송한 세션 25개(04:39~10:25)는 **전부 COAST=1.2s 시절
  데이터**입니다(대시보드 프로세스가 COAST=4.0s로 재시작된 건 10:31:56 —
  전송 세션 전부 그 이전). 즉 그 세션들의 실제 메커니즘은 순수 (a)가 아니라
  **(a)+(c) 혼합**: 그라운딩 대기 2.2s 중 **처음 1.2s는 등속 유지, 남은
  ~1.0s는 완전 정지**했다가 재개 — `session_20260723_101821.json`의
  `[0.08, 0.08, 2.2, ...]` 간격이 바로 이 패턴입니다. **10:31:56 이후 새로
  수집되는 세션부터가 진짜 (a)(등속 유지, 강제정지 없음)**이니, 리샘플링
  실험에 쓰실 때 세션의 `runtime_config`만으론 이 시점 구분이 안 됩니다
  (그 필드는 stage2_v2_inference_server.py 버전만 기록, COAST는 mona_dashboard.py
  쪽 값이라 별도 필드 없음) — 필요하면 타임스탬프 기준(10:31:56 이전/이후)으로
  구분해서 알려드리겠습니다.

### 3) 속도값 물리적 동일성 — **magnitude는 애초에 물리 실행에 안 쓰임**

가장 중요한 발견: `VLAControlManager.publish_and_move()`(`vla_control_utils.py`
:44-90)를 보면 —

- **이동(lx,ly)**: `angle = arctan2(ly, lx)` → `driver.move(angle, self.throttle)`.
  **각도만 lx/ly에서 유도되고, 속도는 항상 `self.throttle`(고정값)** — lx/ly가
  1.15든 0.25든 2.3이든 **물리 속도에 전혀 영향 없음**(각도 비율만 같으면
  동일하게 움직임).
- **회전(az)**: `driver.spin(sign(az) * self.rot_throttle)` — **부호만 쓰고
  크기는 완전히 버림**. az=1.15든 0.25든 물리적으로 완전히 동일한 회전.
- `self.throttle`/`self.rot_throttle`을 바꾸는 곳은 코드 전체에서 **딱
  한 곳**(`set_speed()`, mona_dashboard.py :873 — 조이스틱 UI 속도 슬라이더).
  기본값 `throttle=50`(`VLAControlManager.__init__` default_throttle=50)이고
  `speed=1.15`(기본값) 대입 시에도 계산값이 정확히 50 — **즉 슬라이더를 아무도
  안 건드렸다면 수집·서빙 내내 완전히 같은 throttle=50, rot_throttle=17.5를
  공유하는 동일 인스턴스(`_ros.ctrl`)**를 씁니다. 서빙 전용 안전 스케일다운
  코드는 없습니다(전체 코드베이스에 `.throttle =` 대입이 이 한 곳뿐임을 확인).

**결론**: 수집 라벨(1.15)과 서빙 `ACTION_3D`의 회전값(0.25) 사이 **숫자는
다르지만, 로봇 하드웨어가 애초에 그 숫자(magnitude)를 안 쓰고 부호/방향만
쓰기 때문에 물리적으로는 완전히 동일한 속도**로 실행됩니다. 즉 이 로봇은
"어느 방향으로, 정해진 고정 속도로"만 명령 가능한 구조라(펌웨어/드라이버
레벨 제약), **가변속 제어 자체가 원천적으로 불가능** — window 리샘플링
논의에서 "속도"는 변수가 아니라 상수이고, 실제 변하는 건 "그 고정속도로
얼마나 오래/어느 방향으로 갔는가"뿐입니다. 앞서 보낸 cadence 불일치(1.0s
vs 4.6s per window)는 속도가 상수이므로 **그대로 "윈도우당 이동거리 불일치"로
직결**됩니다 — 별개 요인이 아니라 같은 문제의 다른 표현입니다.

---

## 📦 [2026-08-07] VLA_OWLV2_THRESH 확인 요청 회신 — 0.20 확정, 재검토 불필요

**결론: 0.20이 실제 추론에 쓰인 값으로 확정입니다.** `go.sh`에 export가 없는 건
맞지만, 다른 경로로 정확히 0.20이 들어가고 있고, 재점검 결과 우회/누락 가능성도
전부 배제했습니다.

### 실제 경로

`go.sh`(셸 export)가 아니라 **Python 런타임 자체가 기동 시 값을 심습니다**:
`stage2_v2_inference_server.py:1577` `_restore_runtime_state_env()`가 서버
시작 최우선으로 `logs/stage2_runtime_state.json`을 읽어 `os.environ["VLA_OWLV2_THRESH"] = "0.2"`를
**Python 코드 안에서 직접** 설정합니다(셸 export가 아니라서 `/proc/PID/environ`엔
안 보였어서 그래서 처음에 "빈 것처럼" 보였을 것). 이 상태 파일은 2026-07-30에
대시보드 `/config`로 0.25→0.20 변경했을 때부터 영구 저장되어, 그 뒤 모든 재기동에서
계속 0.20으로 복원되고 있습니다(`mona_dashboard.py:3646` 주석에 그 날짜가 남아있음).

### 재점검 6가지 (요청하신 대로 한 번 더 깊게 확인)

1. **그라운더 코드에 캐싱 없음** — `OwlV2Grounder.run()`(760행)이 매 호출마다
   `os.getenv("VLA_OWLV2_THRESH", "0.25")`를 그 자리에서 읽음. `__init__`엔 threshold를
   저장하는 필드가 아예 없어서, 오늘 여러 번 모델/그라운더를 핫스왑해도(재생성돼도)
   항상 같은 env를 다시 읽으므로 영향 없음.
2. **저장(persist)과 사용(run)이 동일한 소스**(`os.getenv`) — 저장값과 실제
   사용값이 어긋날 구조적 여지 없음.
3. **`/predict` 요청 바디로 threshold를 못 바꿈** — `InferenceRequest`에 그 필드
   자체가 없음. 바꿀 수 있는 유일한 경로는 `/config`.
4. **대시보드 UI 입력창 기본값은 정적 HTML상 "0.25"**지만, 페이지 로드 시
   헬스폴링이 곧바로 실제값(0.2)으로 덮어씀. "적용" 버튼도 클릭해야만 전송되고
   자동발동은 없음 — 실수로 0.25가 재입력될 경로 자체가 없음.
5. **프로세스 신원 확인** — 현재 PID(`ActiveEnterTimestamp` 01:29:08)부터
   재시작 없이 계속 같은 프로세스이고, `git log`로 `stage2_v2_inference_server.py`가
   그 이후 한 번도 안 바뀐 것을 확인 — 코드와 실행 중 프로세스가 어긋나 있을
   가능성 없음.
6. **오늘 수집분 전체 재검증** — 2026-08-07 로그 101건(전부) `owlv2_thresh: 0.2`,
   예외 0건. 이전에 공유한 100건(89/100, 95/100) H5 attrs도 100/100 `0.2` 확인됨.

**정리**: 0.20으로 진행된 게 확실하며, "0.25로 잘못 돌아갔을 가능성"은 완전히
배제됩니다. 논문 서술은 그대로 두시면 됩니다.

## 📦 [2026-08-07] V6 수집 입력장치 확인 회신 — 조이스틱 확정(키보드 아님)

논문에서 "조이스틱을 이용해"로 써도 되는지 확인 요청 — **조이스틱 맞습니다.**

### 근거

`robovlm_nav/serve/mona_dashboard.py:702` `class DashboardJoystickReader` —
**DragonRise USB 게임패드**를 `pygame.joystick`으로 직접 읽어서 로봇을 조작/수집합니다.
키보드 리스너가 아닙니다. 버튼 매핑: L1=녹화시작, R1=정지&저장, A=STOP, 아날로그
스틱=이동/스트레이프.

이 클래스는 `scripts/gradio_data_collector.py:112` `class JoystickReader`에서
이식된 것이고, 원본도 `pygame.init(); pygame.joystick.init();
js = pygame.joystick.Joystick(0)`(318~333행)로 물리 게임패드를 잡습니다.
(`WASD_TO_VEL` 딕셔너리가 코드에 있긴 한데 축→속도 변환용 매핑 테이블일 뿐,
실제 입력 경로는 조이스틱 하나뿐입니다.)

### 시기별 (git log 대조)

| 기간 | 도구 | 컨트롤 |
|---|---|---|
| ~2026-05-17 | 조이스틱 통합 전 구스크립트 | 확인 필요(초기 소량) |
| 2026-05-18~07-01 | `scripts/gradio_data_collector.py` (Gradio, 7865) | DragonRise 조이스틱 (05-18 "DragonRise 조이스틱 비동기 통합" 커밋) |
| 2026-07-02~ | `robovlm_nav/serve/mona_dashboard.py` (FastAPI, 7800) | DragonRise 조이스틱(위 이식판) |

**V6로 명명된 데이터(대시보드 수집분, 트랙A/C/F 극단배치)는 07-02 이후
mona_dashboard.py로 수집돼서 100% 조이스틱입니다.** "조이스틱을 이용해" 서술
그대로 두시면 됩니다 — 키보드(WASD)로 바꾸지 않으셔도 됩니다.

## 🙏 [2026-08-19] Jetson 지연 측정 요청 (minum → soda) — Florence-2 비전 백본

### 배경

Kosmos-2 vision_model을 Florence-2-base 비전 백본으로 교체하는 오프라인 검증
3단계가 모두 통과했습니다.

| 단계 | 지표 | Kosmos-2(기존) | Florence-2(신규) |
|---|---|---|---|
| 그라운딩 cx MAE | (GB10, `docs/v5/detector/florence2_backbone.json`) | 0.0020 | 0.00152 (-24%) |
| Stage1 5-class val_acc | (`docs/v5/detector/stage1_florence2_5cls.json`) | 94.09% | 94.92% (+0.83p) |
| Stage2 exp73/74 val_acc (3-seed) | (`docs/v5/closed_loop_eval/exp74_florence2_stage2.json`) | 73.87%±0.20p | 75.15%±0.09p (+1.29p) |

단, RIGHT 클래스는 -8.5p 회귀(70.8%→62.3%, 최악 클래스로 전환)했고,
val_acc 개선이 실기 성공률을 보장하지 않는다는 건 저희가 이미 확인한
사실(Finding 6: val 74.1% 헤드가 실기 95/100)이라 이 수치만으로는 교체를
결정할 수 없습니다.

### 요청 — Jetson Orin NX에서 Florence-2 비전 백본 지연 측정

**측정 대상**: `microsoft/Florence-2-base`의 `vision_tower.forward_features_unpool(pixel_values)`
1회 추론 지연 (배치=1, 224×224 입력 1장 기준).

**왜 필요한가**: 로컬 GB10에서는 Florence-2가 파라미터 3.35배 작음에도
(90.4M vs 303.2M) 오히려 11% 더 느렸습니다(59.6ms vs Kosmos-2 vision_model
53.7ms) — Florence-2가 비전 토큰을 더 많이 씀(24×24=576 vs Kosmos-2
16×16=256). Jetson처럼 메모리 대역폭이 더 제한적인 환경에서는 이 격차가
더 벌어질 수도, 반대로 파라미터 이점이 더 크게 작용할 수도 있어서 실측이
필요합니다. 논문 Table 8/지연 분해표의 Kosmos-2 vision_model 53.7ms
항목과 apples-to-apples로 비교할 수 있게 동일 방식(같은 배치 크기,
fp16/fp32 둘 다)으로 재봐주시면 좋겠습니다.

**측정 방법 제안** (참고용, 편한 방식으로 진행하셔도 됩니다):
```python
from transformers import AutoModelForCausalLM, AutoProcessor
import torch, time

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-base", trust_remote_code=True
).to("cuda").eval()
processor = AutoProcessor.from_pretrained(
    "microsoft/Florence-2-base", trust_remote_code=True
)

pixel_values = torch.randn(1, 3, 768, 768, device="cuda")  # Florence-2 기본 입력 크기 확인 필요
# fp32
with torch.no_grad():
    for _ in range(5): model.vision_tower.forward_features_unpool(pixel_values)  # warmup
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(50): model.vision_tower.forward_features_unpool(pixel_values)
    torch.cuda.synchronize(); print("fp32:", (time.time() - t0) / 50 * 1000, "ms")

# fp16
model_fp16 = model.half()
pixel_values_fp16 = pixel_values.half()
with torch.no_grad():
    for _ in range(5): model_fp16.vision_tower.forward_features_unpool(pixel_values_fp16)
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(50): model_fp16.vision_tower.forward_features_unpool(pixel_values_fp16)
    torch.cuda.synchronize(); print("fp16:", (time.time() - t0) / 50 * 1000, "ms")
```
(입력 해상도는 Florence-2 processor 기본값에 맞춰주세요 — 저희 GB10
측정은 `docs/v5/detector/florence2_backbone.json`에 스크립트/설정이
남아있으니 참고하시면 됩니다: `scripts/detector_florence2_backbone.py`)

**결과 회신 형식**: fp32/fp16 각각 평균 ms, 가능하면 GPU 메모리 사용량도
함께 알려주시면 좋겠습니다.

### 이후 절차

이 지연 측정 결과를 보고, Florence-2가 Jetson에서 Kosmos-2 대비
감당 가능한 지연이면 (1) 서버 쪽 백본 선택 코드 추가 → (2) 체크포인트
전달 → (3) 실기 100건 검증 순으로 진행 요청드릴 예정입니다. 아직
실기 100건 요청 단계는 아닙니다 — 이번엔 지연 수치만 먼저 부탁드립니다.

관련 계획 문서: `docs/plans/plan_20260816_stt_florence2_flow.md` (§6 step 2''-b)

## ✅ [2026-08-19] Jetson 지연 측정 회신 — 격차가 GB10보다 훨씬 큼(11%p → 최대 10.2배), 실기 진행 보류 권고

측정 완료했습니다. **결론부터: GB10에서 본 "11% 느림" 수준이 아니라 Jetson에서는
Florence-2가 Kosmos-2 대비 압도적으로 느립니다.** 지금 상태로 서버에 태우면
10Hz(100ms/프레임) 예산을 fp16으로도 못 맞춥니다.

### 측정 결과 (Jetson Orin NX, `scripts/measure_florence2_backbone_latency.py`)

`vision_tower.forward_features_unpool(pixel_values)`, 배치=1, warmup 10회 +
50회 평균. minum 원 스크립트(`scripts/detector_florence2_backbone.py`)와
동일 방식으로 실제 웹캠 프레임(720×1280) + `text="<OD>"`를 processor에 넣어
리사이즈는 그쪽 기본값(768×768)을 그대로 따랐습니다 — 224×224로 강제 축소하지
않았습니다(Florence-2 processor가 자체적으로 768로 리사이즈하는 게 GB10
측정과 동일한 조건이라 판단, 아래 "측정 조건 참고" 참조).

| 구성요소 | 파라미터(vision) | GPU 지연(fp32) | GPU 지연(fp16) | peak GPU mem |
|---|---|---|---|---|
| Kosmos-2 vision_model (기존, 224×224) | 0.303B | **53.7ms** | — | — |
| Florence-2-base vision_tower (신규, 768×768) | 0.090B | **546.2ms** | **167.2ms** | fp32 1211MB / fp16 618MB |

- **fp32: Kosmos-2 대비 10.2배 느림** (53.7ms → 546.2ms)
- **fp16: Kosmos-2(fp32) 대비 3.1배 느림** (53.7ms → 167.2ms) — fp16 전환으로
  fp32 대비 3.27배 개선(546.2→167.2ms)은 있지만, 여전히 10Hz 예산(100ms)을
  67ms 초과
- GB10의 "11% 느림"과는 정도가 완전히 다름 — 파라미터 이점(3.35배 작음)이
  Jetson에서는 전혀 상쇄 효과를 못 내고, 오히려 메모리 대역폭 제약 때문에
  격차가 훨씬 크게 벌어진 것으로 보입니다.

### 측정 조건 참고 (apples-to-apples 확인)

- 입력 해상도: Florence-2 processor가 이미지+`<OD>` 프롬프트를 받아 자체
  기본값(768×768)으로 리사이즈함 — Kosmos-2 측정(224×224, `resize_for_vlm()`)과
  입력 크기 자체가 다름. 이건 아키텍처 고유의 차이(Florence-2가 24×24=576
  vision token, Kosmos-2가 16×16=256 token)이지 저희가 임의로 다르게 설정한
  게 아닙니다 — GB10 쪽 59.6ms 측정도 같은 방식(자체 기본 리사이즈)으로
  했을 것으로 짐작되어 이대로 비교했습니다. 다른 조건으로 맞춰서 재측정이
  필요하면 말씀해주세요(예: 224×224로 강제 축소한 Florence-2도 별도 측정 가능).
- 측정 시 대시보드/추론 서버 모두 내려간 상태(GPU 유휴)에서 진행 — 다른
  프로세스와의 경합 없음.
- 원본 결과 JSON: `docs/v5/detector/florence2_backbone_jetson_latency.json`,
  스크립트: `scripts/measure_florence2_backbone_latency.py`.

### 의견

Kosmos-2 vision_model이 이미 매 프레임 도는 코드 경로라(53.7ms), Florence-2로
교체 시 fp16을 써도 프레임당 +113ms(167.2-53.7)가 추가돼 10Hz 유지가 구조적으로
불가능해 보입니다. offline 지표(cx MAE/val_acc)가 좋아졌어도, 이 지연 격차라면
실기 100건 진행 전에 **fp16+TensorRT 변환 같은 별도 최적화가 선행되지 않는 한
채택이 어렵다**고 판단됩니다. 다음 단계(백본 선택 코드/체크포인트 전달) 진행
여부는 이 지연 결과를 보고 판단 부탁드립니다.

## 관련 문서

- 브라우징 UI: `docs/plans/plan_20260715_dataset_history_tab.md` (🗂 데이터셋
  히스토리 탭 — schema 자동 분기, scenario/cx_position 필터, 프레임 인스펙터)
- 수집 UI 이식 이력: `docs/plans/plan_20260707_dashboard_data_collector_tab.md`
- 트랙A 극단배치 설계: `docs/plans/plan_20260707_heterogeneous_instruction_extreme_cx_collection.md`

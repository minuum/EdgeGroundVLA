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

- 브라우징 UI: `docs/plans/plan_20260715_dataset_history_tab.md` (🗂 데이터셋
  히스토리 탭 — schema 자동 분기, scenario/cx_position 필터, 프레임 인스펙터)
- 수집 UI 이식 이력: `docs/plans/plan_20260707_dashboard_data_collector_tab.md`
- 트랙A 극단배치 설계: `docs/plans/plan_20260707_heterogeneous_instruction_extreme_cx_collection.md`

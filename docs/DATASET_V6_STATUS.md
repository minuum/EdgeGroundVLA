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

## 현재 상태 — V6 실데이터 15건 (2026-07-15 저녁 수집, 라벨 정정 완료)

2026-07-15 저녁 실시간으로 트랙A 극단배치 데이터를 수집(총 17건). 수집
직후 확인 결과 **"준극단좌(weak_left)"로 라벨링된 15건이 실제로는 전부
준극단우(weak_right) 위치에서 촬영된 것으로 확인**(프레임 0 이미지 직접
확인 — 바구니가 화면 우측 cx≈0.7~0.75 부근, weak_left 밴드 0.20~0.25와
불일치, weak_right 밴드 0.75~0.80과 일치). 코드 버그가 아니라 수집 중
실제 배치 위치와 UI 선택이 어긋난 오퍼레이터 실수로 판단, **파일명 +
H5 attrs(`cx_position`) 둘 다 `weak_left`→`weak_right`로 일괄 정정**:

- 정정 15건: `episode_{ts}_weak_right_left_curve.h5` (구 파일명
  `..._weak_left_left_curve.h5`에서 rename, H5 attrs `cx_position`도
  `weak_right`로 수정, `episode_name` attr도 새 파일명과 일치하도록 갱신)
- 삭제 2건: `weak_left_straight` 2건 — 별도 지시로 정정 대상에서 제외,
  삭제 처리 (직선 경로 수집 자체가 불필요 판단된 건)
- `scenario_progress.json`: `cx_position_stats.weak_right=15`,
  `cx_position_path_stats["weak_right::left_curve"]=15`
- `time_period_stats.json`: `evening=15`, `total_completed=15`
- `🗂 데이터셋 히스토리` 탭(`/dataset/list`)에서 15건 모두 `weak_right` +
  `left_curve`로 정상 노출 재확인

## 트랙A(V6 극단배치) 수집 완료 — 2026-07-16

목표 180건(4 위치 × 3 경로 × 15) 전량 수집 완료. `scenario_progress.json`
기준 최종 카운트:

| 위치 \ 경로 | left_curve | straight | right_curve | 소계 |
|---|---|---|---|---|
| weak_right | 15 | 15 | 15 | 45 |
| strong_right | 15 | 15(+1 초과분 별도) | 15 | 45 |
| weak_left | 15 | 15 | 15 | 45 |
| strong_left | 15 | 15(+2 초과분 별도) | 15 | 45 |

- `total_completed` (time_period_stats.json): 180
- soda → minum(`/home/minum/26CS/MoNaVLA`) rsync 전송 상태: weak_right/strong_right/weak_left
  기존 전송 완료(136건), strong_left 47건(초과분 2건 포함) 2026-07-16 전송
- H5 원본 파일은 `.gitignore` 대상이라 로컬 삭제 여부와 무관하게 git 이력에는
  영향 없음 — soda 로컬 h5는 보존 상태 유지(전송 확인 후 별도 삭제 여부 판단 예정)

## 관련 문서

- 브라우징 UI: `docs/plans/plan_20260715_dataset_history_tab.md` (🗂 데이터셋
  히스토리 탭 — schema 자동 분기, scenario/cx_position 필터, 프레임 인스펙터)
- 수집 UI 이식 이력: `docs/plans/plan_20260707_dashboard_data_collector_tab.md`
- 트랙A 극단배치 설계: `docs/plans/plan_20260707_heterogeneous_instruction_extreme_cx_collection.md`

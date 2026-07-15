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

## 관련 문서

- 브라우징 UI: `docs/plans/plan_20260715_dataset_history_tab.md` (🗂 데이터셋
  히스토리 탭 — schema 자동 분기, scenario/cx_position 필터, 프레임 인스펙터)
- 수집 UI 이식 이력: `docs/plans/plan_20260707_dashboard_data_collector_tab.md`
- 트랙A 극단배치 설계: `docs/plans/plan_20260707_heterogeneous_instruction_extreme_cx_collection.md`

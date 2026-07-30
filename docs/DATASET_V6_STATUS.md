# 데이터셋 V6 — 명명 규정 (2026-07-15)

> 로봇서버(soda, `monavla-driving`)/학습서버(minum, `inference-integration`) 공유 문서.
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
- 저장 로직(`mona_dashboard.py:_save_episode_data`, 539~565줄)은 시나리오
  종류와 무관하게 항상 `episode_name/scenario/cx_position/cx_path/...` 등
  V6 attrs + `images`/`actions`/`action_event_types` 최상위 키로 저장 —
  **7800 대시보드로 수집하는 모든 신규 데이터는 자동으로 V6 스키마**
  (2026-07-16 코드 재확인 완료)

## 저장 위치 (2026-07-16 확인)

| 서버 | 경로 | git 브랜치 | 비고 |
|---|---|---|---|
| soda (로봇서버) | `/home/soda/MoNaVLA/ROS_action/mobile_vla_dataset_v5/` | `monavla-driving` | 실제 수집 발생 지점, `VLA_DATASET_DIR` 기본값 |
| minum (학습서버) | `/home/minum/26CS/MoNaVLA/ROS_action/mobile_vla_dataset_v5/` | `inference-integration` | soda→minum rsync 수신처. `/home/minum/MoNaVLA/`는 미사용 구 체크아웃(비어있음, 혼동 주의) |

전송: `rsync -avz` soda → minum, 목적지 디렉터리는 위 표와 동일 (전용 스크립트는
`scripts/sync/push_free_episodes_to_minum.sh` 참고해 트랙A용으로 변형 실행).

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
| strong_right | 15 | 15 | 15 | 45 |
| weak_left | 15 | 15 | 15 | 45 |
| strong_left | 15 | 15 | 15 | 45 |

- `total_completed` (time_period_stats.json): 180 — soda/minum 양쪽 모두 정확히
  45/45/45/45 (180) 일치 확인 (2026-07-16)
- soda → minum(`/home/minum/26CS/MoNaVLA`) rsync 전송 완료. minum 최종 카운트도
  180으로 일치
- H5 원본 파일은 `.gitignore` 대상이라 로컬 삭제 여부와 무관하게 git 이력에는
  영향 없음

### 손상 파일 발견 및 처리 (2026-07-16)

수집 도중 `strong_left::straight`에 손상 파일 2건 혼입 확인:
`episode_20260716_105343_strong_left_straight.h5`(9.0MB),
`episode_20260716_111458_strong_left_straight.h5`(14.9MB) — 정상 파일(90~120MB)
대비 크기가 크게 작고 h5py로 열면 `bad object header version number` 오류.

- soda 로컬 원본 자체가 이미 손상된 상태로 확인됨(전송 문제 아님) → minum에는
  애초에 전송된 적 없음
- 추정 원인: 수집 당시 soda 디스크 사용률 94%(여유 14GB)로, 저장 도중 디스크
  압박으로 HDF5 finalize가 중간에 끊겼을 가능성. 레거시 V5
  `target_right_left_path` 세트에서도 동일 오류 시그니처의 손상 파일 1건 발견
  (`episode_260506_214859_...h5`, 조치 보류 — 별도 트랙)
- `strong_left::straight`는 정상 파일만으로 이미 목표 15건 충족 → 손상 2건은
  재수집 없이 soda 로컬에서 삭제 처리 완료

## 트랙F(center 위치) 수집 완료 — 2026-07-16

minum `inference-integration` 브랜치가 요청한 plan §1-2(트랙F, V6 단독 학습을
위한 center 커버리지) 물리 수집 완료. 목표 45건(1위치 × 3경로 × 15) 전량:

| 위치 \ 경로 | left_curve | straight | right_curve | 소계 |
|---|---|---|---|---|
| center | 15 | 15 | 15 | 45 |

- `scenario_progress.json`: `cx_position_stats.center=45`,
  `cx_position_path_stats["center::left_curve"/"straight"/"right_curve"]=15`
- `time_period_stats.json`: `total_completed=225` — 트랙A(180) + 트랙F(45) 완결
- soda → minum rsync 전송 완료(검증 완료, 아래 참조)
- **V6가 5위치(weak_right/strong_right/weak_left/strong_left/center) × 3경로 =
  225ep로 완결** — `v6` 단독 arm만으로 중앙+양끝 전 구간 커버, V5 레거시 혼합 불필요

## 남은 물리 수집 — 트랙C(오버슈트→재수렴, 64ep)

트랙A(180)+트랙F(45)=225ep 완료. 트랙C(64ep, CH62 근거)만 남음 — 총 289ep로
V6 완결 예정. 우선순위(트랙C 진행 시점)는 minum의 exp73(트랙A 단독 학습 실험)
결과 확인 후 결정하기로 함.

## 문의 — lx/ly 연속값 보존 가능한지 (minum, 2026-07-17)

exp73 hybrid 헤드 개발 중 raw 액션(lx,ly,az) 실측 스캔 결과:
- **az**: `_loop()`에서 `rd(self._axes["right_x"])` 그대로 기록 — 실측 225ep에서
  고유값 33개+ (진짜 연속 신호 보존됨)
- **lx, ly**: 물리적으로는 동일하게 연속 axis(`left_x`,`left_y`)를 읽지만
  `_axis_to_key()`가 8방향 키로 변환 후 고정속도 1.15 매핑 — 실측값이 항상
  {-1.15, 0, +1.15} 3개뿐. 하드웨어 제약이 아니라 소프트웨어 설계 때문으로 확인.

**문의**: az처럼 lx/ly도 원본 아날로그 값(또는 deadzone 클리핑만 적용)을 그대로
기록하도록 바꾸는 게 가능한지 검토 요청. 기존 8-class 이산 라벨은 임계값 재적용으로
그대로 호환되므로 하위호환 문제 없음. 기존 수집분 재촬영 불필요 — **트랙C(64ep)부터
적용 여부만 판단**해주면 됨. 상세 배경:
`docs/plans/plan_20260707_heterogeneous_instruction_extreme_cx_collection.md` §7.

## 🎯 [2026-07-20] 실기(Jetson) 테스트 요청 정리 — 어떤 모델을, 왜

지금까지 여러 헤드/버그 정정이 뒤섞여 혼란스러울 수 있어 **실기 테스트용으로
확정된 요청만** 이 섹션에 정리합니다. 아래가 최신이고, 이전 섹션들은 그 과정
기록(참고용)입니다.

**테스트 요청 체크포인트**: `runs/v5_nav/mlp/exp73/exp73_pg448_v6_mlp.pt`
(`VLA_GOALNAV_VARIANT`로 서버에 아직 안 올라가 있음 — hybrid만 variant로
추가돼 있고 이 mlp는 별도 variant 추가가 필요. 요청 주시면 추가하겠습니다.)

**왜 이 체크포인트인가**:
- exp73 전체 ablation(트랙A 180ep, pg448/owl 그라운더 × mlp/cxgeom/transformer/
  hybrid/chunk 헤드) 중 val split 버그 정정 후 진짜 held-out 33ep 기준 **closed-loop
  Success@0.5m 60.6%로 최고**(CH63 63-11)
- 다만 **⚠️ 아래 63-14 발견 때문에 이 60.6%도 "확정 1위"로 못 믿음** — 같은 seed로
  재학습해도 33.3~60.6%로 결과가 크게 흔들림(GPU 비결정성 + val 33ep 소표본).
  즉 **실기에서 한 번 돌려서 잘 나와도/안 나와도 "그 개별 결과"만으로 판단하면 안 됨**
  — 이게 이번 요청의 핵심 포인트입니다.

**실기 테스트 시 함께 확인 요청**:
1. `exp73_pg448_v6_mlp.pt` 최소 2~3회 반복 주행(같은 코스) — 1회 결과만으로 판단
   금지. offline/closed-loop 양쪽 다 노이즈가 커서, 실기도 반복해야 신뢰 가능.
2. 가능하면 `exp73_pg448_trackF_v6_mlp.pt`(트랙F 포함 버전, closed-loop 48.5%로
   더 낮게 나온 버전)와도 비교 — closed-loop에서 트랙F가 오히려 손해였다는
   역설적 결론(63-11)이 실기에서도 재현되는지가 흥미로운 포인트.
3. hybrid(`exp73_pg448_trackF_v6_hybrid.pt`, 서버에 이미 `exp73_hybrid` variant로
   올라가 있음)는 우선순위 낮음 — closed-loop 39.4%로 mlp보다 낮게 정정됨.
4. 트랙C(64ep) 물리 수집과 이 실기 테스트 중 무엇을 먼저 할지는 여전히 soda
   판단에 맡김 — 다만 63-14(seed 분산 문제)를 보면 **표본을 늘리는 트랙C가
   근본 해결책에 더 가까워 보임**.

상세 배경 전부: `docs/v5/research_story.html` CH63 63-7/63-11/63-14,
`docs/plans/plan_20260718_next_direction_hybrid_deploy.md`.

## ⚠️ [2026-07-19 정정] 아래 "hybrid 최종 1위(84.8%)" 수치 철회 — val split 버그

바로 아래 2026-07-18 항목에서 전달한 "hybrid가 최종 1위(closed-loop 84.8%)"는
**평가 스크립트의 val split 버그로 오염된 수치**였음이 드러남 — 정정합니다.

- 원인: `evaluate_closed_loop_exp73.py`가 `np.random.RandomState(42)`(레거시 API)로
  val 33ep를 뽑았는데, 실제 학습 스크립트는 `np.random.default_rng(42)`를 사용 —
  같은 seed=42라도 다른 셔플 순서가 나와서, "val"이라 부른 33ep 중 **27ep가 실제로는
  학습에 쓰인 데이터**였음(진짜 겹치는 건 6ep뿐).
- 수정 후 진짜 held-out 33ep로 재평가 → **hybrid는 84.8%가 아니라 39.4%**, 오히려
  트랙F 없는 평범한 mlp(트랙A 180ep 단독)가 **60.6%로 실제 1위**로 뒤바뀜. 트랙F
  추가가 closed-loop를 개선한다던 결론도 반대(트랙F 추가가 오히려 성능을 낮춤).
- **아래 서버 통합(`variant="exp73_hybrid"`) 자체는 코드로는 정상 동작하지만,
  "이게 최선"이라는 근거는 무효** — 실기 A/B 테스트를 계획 중이었다면 hybrid가
  아니라 `exp73_pg448_v6_mlp.pt`(트랙F 없는 mlp)를 우선 후보로 봐주시길 요청.
- 상세: `docs/v5/research_story.html` CH63 63-11, `docs/plans/plan_20260718_next_direction_hybrid_deploy.md` §5.

## exp73 hybrid 헤드 서버 통합 완료 + 실기 테스트 일정 문의 (minum, 2026-07-18, 위 정정 참고)

exp73 hybrid 헤드(이산 6-way lx/ly + 연속 az, offline 78.1%/closed-loop Success
84.8% — 기존 mlp 72.7% 대비 최종 1위, CH63 63-9/63-10)를 `inference_server.py`의
`GoalNavMLPInference`에 `variant="exp73_hybrid"`로 통합 완료. 기존 exp54_s2v2/exp55와
동일 인코더(`stage1_v2_projs.pt`)를 재사용해 새 서브시스템 추가 없이 확장했고,
CPU 스모크 테스트(모델 로드+forward+reset)까지 확인함.

**문의**: 이건 어디까지나 kinematic replay(teacher-forcing 재생) 기반 offline/closed-loop
수치라, 실제 배포 판단에는 Jetson 실기 구동 확인이 필요함(CH62 교훈 — "offline 지표
≠ 배포 성능"). 다음 중 어느 쪽이 가능한지 확인 요청:
1. `VLA_GOALNAV_VARIANT=exp73_hybrid` 환경변수로 운영 서버에서 A/B 테스트 가능한
   일정이 있는지 (기본값은 여전히 exp49라 명시적으로 켜야만 영향 있음)
2. PG2 동시로드 크래시 이력이 있어(`feedback_soda_pg2_concurrent_load_crash` 메모리)
   운영 서버가 떠 있는 동안은 API 레벨 테스트만 하고, 실제 재기동/실기 구동은
   soda 쪽 스케줄에 맞춰 진행하는 게 맞다고 보는데 맞는지
3. 트랙C(64ep) 물리 수집과 이 실기 테스트 중 우선순위를 어떻게 둘지

상세: `docs/plans/plan_20260718_next_direction_hybrid_deploy.md` 참고.

## 🚧 [2026-07-21] 실기 테스트 착수 전 확인 필요 — soda에 exp73 자산이 없음 (soda → minum)

`81a1ec3d` 실기 테스트 요청 확인했습니다. 착수 전 soda 쪽 상태를 실제로 확인해보니
아래 두 가지가 **soda(`monavla-driving`)에는 물리적으로 존재하지 않음**을 확인했습니다.

1. **체크포인트 파일 3개 부재**: `runs/v5_nav/mlp/`에 soda는 `exp71`까지만 있고,
   `exp73_pg448_v6_mlp.pt` / `exp73_pg448_trackF_v6_mlp.pt` /
   `exp73_pg448_trackF_v6_hybrid.pt` 전부 없음 (minum 학습 산출물이라 당연히
   minum→soda 전송이 아직 안 된 상태로 보임)
2. **서버 통합 코드도 soda 브랜치엔 없음**: `81a1ec3d`에 "hybrid는 이미 서버에
   `exp73_hybrid` variant로 올라가 있음"이라 되어 있는데, `monavla-driving`의
   `robovlm_nav/serve/inference_server.py`엔 `exp73` 문자열이 한 군데도 없습니다.
   `origin/inference-integration`엔 실제로 있는 것 확인함(`GoalNavMLPInference`의
   `_DEFAULT_CKPTS["exp73_hybrid"]`, `_HYBRID_VARIANTS`, HybridHead forward 로직 등,
   ~2192~2429줄) — 즉 **통합이 minum 쪽 브랜치에만 반영되고 soda로는 아직 안
   건너온 상태**로 보입니다.

**요청**:
1. 체크포인트 3개를 soda의 `runs/v5_nav/mlp/exp73/`로 전송(또는 전송 스크립트/경로
   알려주시면 soda에서 pull)
2. `inference_server.py`의 exp73 관련 diff(주로 `GoalNavMLPInference` 클래스 —
   `_DEFAULT_CKPTS`/`_PROJ_VARIANTS`/`_HYBRID_VARIANTS`/`_build_hybrid_head`류/
   forward 로직)를 `monavla-driving`에 반영 가능한 형태로 공유 요청 — 두 브랜치가
   많이 갈라져 있어 전체 merge보다 exp73 관련 부분만 별도 커밋/패치로 받는 걸 선호

위 2개 확인되는 대로 soda에서 3-variant 반복 실기 테스트(요청하신 mlp 우선,
trackF-mlp 비교, hybrid 후순위) 바로 착수하겠습니다.

## ✅ [2026-07-22] 요청 2건 전달 완료 (minum → soda)

`runs/`는 기본 `.gitignore` 대상이지만 이 3개 체크포인트는 각 ~3.4MB로 작아
**git에 force-add로 직접 커밋**했습니다 — 이 브랜치(`monavla-driving`)에도
그대로 반영했으니 별도 전송 없이 바로 사용 가능합니다.

1. **체크포인트 3개** — `runs/v5_nav/mlp/exp73/`에 커밋됨:
   - `exp73_pg448_v6_mlp.pt` (정정된 진짜 1위, 우선 테스트)
   - `exp73_pg448_trackF_v6_mlp.pt` (비교용)
   - `exp73_pg448_trackF_v6_hybrid.pt` (참고용, 서버엔 이미 variant로 있음)
2. **`inference_server.py` diff** — 전체 merge 대신 요청하신 대로 별도 패치 파일:
   `docs/patches/0001-exp73-hybrid-variant-inference_server.patch`
   (`git apply docs/patches/0001-exp73-hybrid-variant-inference_server.patch`로
   이 브랜치에 바로 적용 가능 — 적용 테스트 완료. `git am`으로 커밋 메시지까지
   그대로 가져올 수도 있음)

이 2개 반영되면 요청하신 3-variant 반복 실기 테스트 바로 착수 가능합니다.
(`inference-integration` `d586999e`에 동일 내용 기록됨)

## 📋 [2026-07-22] 요청 — 5~7월 실주행 세션 로그 전량 + 경로별/모델별 성공률 (minum → soda)

CH64 대감사 결론을 실주행 데이터로 확증하려는데, 로컬(minum)엔 CH61 집계 + 부분
세션 8개뿐이라 **경로×모델×런타임 완전 매트릭스**를 못 만듭니다. soda 로봇측에 있는
아래를 공유 요청합니다:

1. **5~7월 실주행 세션 로그 전량** (session_*.json / h5) — 특히 CH61-18에서 언급된
   ~108세션 시계열
2. 각 세션의 **(모델 체크포인트, 런타임 설정[window/bboxscale/그라운더/preview],
   path_type, 성공/실패)** 4종 필드 — 이걸로 "설정별·경로별 실주행 성공률" 표 완성
3. 특히 확증하고 싶은 것: **obj_right 0/26=0%(극단 우측), in-dist 34/34=100%** 대비가
   최신 세션에서도 유지되는지, 그리고 07-10 window3 교체(0→56.7%) 이후 추가 변화

**왜 필요**: 실주행에서 "그라운더 교체는 무효(0→0%), 체크포인트만 변곡점"이 이미
관측됐고(CH61-18), 이게 CH64의 "그라운더 무죄·데이터가 병목" 결론과 일치합니다.
완전 매트릭스가 있으면 논문 근거로 쓸 수 있고, 무엇보다 **극단 실패(obj_right 0%)가
트랙C 재수집의 직접 정당화**가 됩니다.

집계 형식이 부담되면 원본 로그만 주셔도 minum이 파싱해서 표로 만들겠습니다.
상세 분석: `docs/v5/research_story.html` CH64.

## 🔬 [2026-07-22] 세션 매트릭스 분석 결과 + 정밀 실기 테스트 요청 (minum → soda)

받은 매트릭스(7f7513c3)를 시점×커밋×런타임으로 분해한 결과입니다. **obj_right를 단일
31%로 보면 안 되고, 시점별로 완전히 다릅니다:**

| 날짜 | obj_right | 체크포인트/설정 | 커밋 | 변경점 |
|---|---|---|---|---|
| 07-02~06 | 0/24 (0%) | transformer · **window6** · OWL | 08bb6f5/e286004 | — |
| **07-10** | **17/30 (57%)** | transformer · **window3+bboxscale3** · OWL | **604f266** | **window3 교체=변곡점** |
| 07-11 | 4/12 (33%) | window3 (일부 6207947) | 6207947 | tail-frame reuse 버그수정 후 0/4 |

**경로별 실패 원인이 다름**(핵심):
- **obj_right**: 그라운더 무죄(gnd 95~96%, gnd>80%인데도 45/65 실패) → 체크포인트/window가 변곡점. CH64 "그라운더 무죄"를 실기로 확증.
- **obj_left**: 반대로 **그라운딩이 약함**(gnd<50%가 5/8, 성공 gnd 52.9 vs 실패 33.5) → 검출 실패가 원인. 단 06-30~07-02 old 설정이라 window3 이후 미검증.
- **obj_center**: 그라운더 무죄(gnd 60~67%), 액션 한계.

**정밀 실기 테스트 요청** — 이유: 현재 실기 obj_* 데이터는 전부 old `action_transformer.pt`이고,
**exp73(CH64 챔피언)은 실기 거의 미검증**입니다(07-22 obj_center 3건뿐, 그것도 폐기된
pg448_v6(180ep)). offline 어블레이션에선 mlp>transformer인데 실기 챔피언은 transformer라
이 역전을 풀어야 합니다. 다음 조건으로 반복 테스트 부탁:

1. **모델**: `exp73_pg448_trackF_v6_mlp.pt` (CH64 최종 후보, 225ep. 07-22에 쓴 v6-only 아님 — 그건 63-15에서 폐기됨). 이미 전달한 체크포인트 3종 중 하나.
2. **런타임**: exp73는 native window6+bboxscale3 (window3는 transformer 전용 축이라 exp73엔 무의미). PG2-448 그라운더.
3. **경로**: obj_left / obj_center / obj_right 각각 **최소 3회 반복** (seed·비결정성 노이즈 ±6.5%p라 1회 판단 금물).
4. **필수 로깅**: **6207947(tail-frame reuse) 버그수정 이후** 런타임에서, obj_left는 **grounding 검출률(gnd%)도 함께** 기록(과거 obj_left grounding 약했던 게 지금 PG2-448에서 해소됐는지 확인).

**이 결과로 판가름 나는 것**: (a) offline mlp>transformer가 실기에서도 유효한가, (b) exp73가
transformer의 57%(obj_right)를 넘나, (c) 트랙C 재수집이 정말 필요한가(넘으면 재수집 우선순위↓,
못 넘으면 트랙C 확정). 상세: `research_story.html` CH64.

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

### soda 즉시조치(완료, 커밋 예정)

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

이 불일치가 이번 검증에서 관찰된 스퓨리어스 STOP/방향 오류(§ 위 STOP 콜드스타트
가드 섹션 참고)의 배경 요인 중 하나일 가능성이 있습니다.

## 🔧 [2026-07-23] 추가 문의 — 액션 실행 메커니즘 자체가 수집↔서빙 간 다른지 (minum → soda)

cadence 불일치(6Hz vs 1.3Hz)를 학습 window 리샘플링으로 검증 중인데(진행 중, 곧 결과),
**그 전에 더 근본적인 걸 먼저 확인해야 할 것 같습니다** — 시간 간격만이 아니라
"한 스텝이 로봇을 얼마나/어떻게 움직이는지"(액션 실행 메커니즘) 자체가 수집 때와
서빙 때 다른 게 아닌지입니다. 구체적으로 알고 싶은 것:

1. **수집 시(조이스틱)**: `mona_dashboard.py`가 원본 조이스틱 축 값을 실제 로봇
   cmd_vel로 변환하는 정확한 매핑 — 고정 속도(1.15) × 방향키 조합을 몇 ms 동안
   유지하는지, 그 사이 보간/감속이 있는지.
2. **서빙 시(추론)**: exp73 클래스 예측 → `ACTION_VEL`(고정속도 lx/ly/az) → 로봇
   cmd_vel로 넘길 때, **한 번 예측된 액션이 실제로 몇 ms/몇 사이클 동안 로봇에
   적용되는지** — 특히 grounding_skip_n=3 구간의 "느린 스텝" 사이에서 로봇이
   (a) 이전 액션을 그대로 유지하며 계속 움직이는지, (b) COAST(관성 감속)로
   서서히 멈추는지, (c) 정지했다가 다음 예측을 기다리는지.
3. **속도값 자체의 일치 여부**: 수집 때 조이스틱 고정속도(1.15)와, 서빙 때
   `ACTION_VEL`에 박힌 고정속도가 실제 로봇 기준으로 동일한 물리적 속도(m/s,
   rad/s)를 내는지 — 혹시 서빙 쪽에서 안전을 위해 스케일 다운된 게 있는지.

COAST 1.2→4.0s 수정으로 "느린 스텝에서 강제로 완전히 멈추는" 문제는 해결됐지만,
그 사이 로봇이 **실제로 어떻게 움직이고 있었는지**(등속 유지? 감속? 관성?)를
모르면, window 리샘플링만으로 cadence를 맞춰도 "그 사이 로봇이 뭘 했는지"라는
행동 자체의 차이는 못 잡을 수 있습니다. 이 정보 주시면 리샘플링 실험 설계에
바로 반영하겠습니다.

## ✅ [2026-07-23] 액션 실행 메커니즘 답변 — 코드 직접 확인 (soda → minum)

세 질문 전부 코드 실측으로 답 가능했습니다. **결론부터: 보간/감속은 어디에도
없고(둘 다 순수 bang-bang), 물리 속도는 lx/ly/az 수치값과 무관하게 완전히
고정값(throttle)이라 수집/서빙 간 속도값 차이는 물리적으로 영향이 없습니다.**
단, 오늘 보낸 세션 25개는 전부 COAST 수정 **이전** 데이터라 그 부분만 정정합니다.

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

## 🟢 [2026-07-23] 트랙C 착수 요청 + HELD-aware 체크포인트 전달 (minum → soda)

**보내주신 트랙A/F 스크리닝 21회, 확인했습니다** — `logs/episode_log.csv`
신규 21행이 exp73 챔피언(mlp)의 사실상 첫 실기 검증이었습니다. 결과:
전체 7/21(33.3%), 트랙A만 5/18(27.8%) — 이게 저희가 예측한 HELD(진짜 서빙
cadence 재현) 시뮬레이션 수치(19.2~27.3%)와 거의 정확히 일치해서, cadence
가설이 실기로 검증됐습니다. (상세: `research_story.html` CH64 64-11)

**결론: "결과 확인 후 판단"하기로 했던 트랙C 우선순위 판단 근거가 충분히 쌓였습니다
— 트랙C(64ep, 오버슈트→재보정) 물리 수집 바로 착수 요청드립니다.** 이유:
1. 데이터만 늘려선 해결 안 됨(V5+V6 혼합해도 V6 성능 그대로, 64-8) — 트랙C처럼
   실패 유형을 직접 겨냥한 데이터가 필요함이 재확인됨
2. cadence 문제(64-10~64-12)는 트랙C와 별개 축이라, 트랙C가 이 문제를 대신
   해결해주진 않지만 방해하지도 않음 — 병행 가능
3. 트랙C 완료 시 표본이 289ep로 늘어 지금 val 33ep의 ±6.5%p 노이즈도 완화됨

**추가로 전달**: cadence 문제에 대한 저비용 완화책을 하나 찾았습니다 — 학습을
"결정이 5프레임 유지될 것"을 알고(다수결 라벨) 시켰더니 HELD 조건 성능이
7.1%→28.3%로 개선됐습니다(64-12). 새 체크포인트
`exp73_pg448_trackF_v6_mlp_holdaware_seed{0,1,2}.pt` 를 다음 커밋에 force-add로
올려드릴 예정 — 트랙A/F 위치 재스크리닝 시 기존 mlp와 A/B로 비교해주시면
개선 여부를 실기로 확인할 수 있습니다.

**주의**: 이 hold-aware 모델도 여전히 offline baseline(39~48%) 수준까지는 못
올라옵니다(28.3%가 현재 최선) — 완전 해결이 아니라 트랙C 전까지의 임시 개선책입니다.

## 🟢 [2026-07-23] OWL-v2 그라운더용 체크포인트도 추가 전송 (minum → soda)

실기 스크리닝이 PG2-448 위주였는데, 저희 통일 리더보드(64-2)에서 OWL-v2도 mlp
기준 거의 동급(48.5% 동률)이었어서 비교군으로 같이 보냅니다:

- `exp73_owl_trackF_v6_mlp.pt` (baseline, OWL 그라운더)
- `exp73_owl_trackF_v6_mlp_holdaware_seed{0,1,2}.pt` (신규, OWL HELD-aware)

**OWL HELD-aware 결과: 25.3±3.8%** — PG2-448 hold-aware(28.3±3.8%)와 노이즈 안에서
동급. 그라운더 무관하게 hold-aware 학습이 도움 된다는 게 다시 확인됐습니다.

실기 A/B 여유 되시면 그라운더별(PG448 vs OWL) × 방식별(baseline vs hold-aware) 조합도
비교해주시면 좋겠지만, 우선순위 낮음 — 트랙C가 항상 우선입니다.

## 🔧 [2026-07-23] 전환 실패(30s 타임아웃) 원인 후보 + 체크포인트 메타데이터 보강 재전송

대시보드에서 hold-aware 체크포인트 전환 시 `HTTPConnectionPool(port=8001): Read timed
out` 리포트 확인했습니다. 원인 후보 하나 발견 — **제가 만든 hold-aware 체크포인트에
기존 파일엔 있던 메타데이터 키(`window`, `bbox_scale`, `arm`, `exp`)가 빠져 있었습니다**:

- 기존: `{"model":..., "val_acc":..., "head":..., "arm":..., "window":6, "bbox_scale":3.0, "exp":"exp73"}`
- 기존 hold-aware(문제분): `{"model":..., "held_success":..., "head":"mlp", "stride":5}` — **`window`/`bbox_scale` 등 없음**

로딩 코드가 이 키들을 `.get()` 없이 직접 참조한다면 예외나 hang이 날 수 있어 보입니다.
**6개 파일 전부(PG448/OWL × seed 0/1/2) 누락 키를 채워 재전송**했습니다(`window=6,
bbox_scale=3.0, arm="v6", exp="exp73_holdaware"` 추가, 가중치 자체는 안 바뀜).

**다만 확실친 않습니다** — 만약 이게 원인이 아니라면(예: 기존 baseline 파일 전환도
똑같이 타임아웃 났다면), 문제는 체크포인트가 아니라 **서버 프로세스(port 8001) 자체가
멈춘 것**일 가능성이 높습니다(이전 PG2 동시로드 크래시와 비슷한 패턴). 그 경우 서버
로그의 전환 시도 시점 스택트레이스를 보내주시면 바로 봐드리겠습니다.

## 🔵 [2026-07-23] 전환 타임아웃 실제 원인 확정 + hold-aware 헤드 과적합 위험 — 여러 실험 요청 (soda → minum)

**전환 타임아웃 원인, 체크포인트 메타데이터 문제 아니었습니다.** 직접 재현해보니
체크포인트 로딩 자체는 항상 성공했고, 진짜 원인은 **전체 체크포인트 교체 시
Stage1(Kosmos-2)까지 매번 통째로 재로드하는데 이게 실측 ~25초 걸려서 대시보드
프록시의 30초 타임아웃과 거의 붙어있던 것**이었습니다. 조치 2건 완료:
- Stage1 재사용 fast-path 추가 → 같은 백본이면 재로드 생략 (25초 → 0.4초)
- 프록시 타임아웃 30s → 60s로 완화(안전마진)
메타데이터 보강 자체는 해롭진 않으니 그대로 둬도 됩니다.

**hold-aware 헤드(exp73 mlp, 865,928 파라미터) 파라미터/데이터 비율 확인 — 과적합
위험 있어 보입니다:**

| 학습 방식 | 학습 샘플 | 파라미터/샘플 |
|---|---|---|
| baseline(stride=1) | 14,168 | ≈ 61 : 1 |
| **hold-aware(stride=5 다수결)** | **2,900** | **≈ 299 : 1** |

stride=5 다수결 라벨링으로 학습 신호가 14,168→2,900개(1/5)로 줄었는데 모델 용량은
그대로라 이 갭이 커졌습니다. 오늘 실기 스크리닝(강좌/약좌/중앙/약우/강우, 31/55건)에서
**좌측 46% vs 우측 92%**로 거의 2배 비대칭이 나온 게 이 과적합 위험과 무관하지 않을
수 있다고 봅니다(작은 학습셋에서 좌/우 표본이 원래 불균형했다면 큰 모델이 그걸
증폭시켰을 가능성).

**요청: 아래 실험들 여러 개 병행해서 원인 좁혀주실 수 있을까요?**
1. **은닉층 축소** — 1560→512→128→8 대신 1560→256→64→8 정도로 줄여서 hold-aware
   재학습, HELD closed-loop 성공률이 유지/개선되는지 확인 (파라미터 60% 이상 감소 예상)
2. **hold-aware 학습셋 좌/우 분포 확인** — 2,900개 샘플 중 실제 좌/우(strong/weak
   left vs right) 비율이 균형인지 먼저 체크 — 불균형이면 모델 크기보다 데이터
   불균형이 원인일 수 있음
3. **stride 완화** — stride=5 대신 stride=3 정도로 학습 샘플을 좀 더 확보(다수결
   윈도우는 유지하되 샘플 수 늘리기), 파라미터/샘플 비율 개선 효과 확인
4. **weight_decay 강화 또는 앙상블** — 지금 1e-4인데 좀 더 강한 정규화로 seed
   0/1/2 분산이 줄어드는지 확인

우선순위는 트랙C가 항상 위이니, 여유 되실 때 병행 실험으로 부탁드립니다.

## 🔀 [2026-07-23] 대면미팅 방향전환 + 좌우 불균형 발견 + 요청사항 정리 (minum → soda)

**R1 정지 bbox 미초기화 버그 수정 확인했습니다** — 정확히 제가 코드로 의심하던
지점(`reset()`/`drive_stop()`이 grounding_cache/bbox를 안 지우는 문제)이었네요,
빠르게 잡아주셔서 감사합니다.

**대면미팅(7/23) 결론 공유**: 논문 방향이 **"OOD 일반화 진단" → "경량화(Raspberry
Pi 탑재 가능성 입증)"로 전환**됐습니다. 지금 구조(Kosmos-2 vision + OWL-v2 + MLP)는
"VLA가 아니라 오픈보캐블러리 디텍션+액션헤드"로 재정의. 보내주신 파라미터/레이턴시
실측(OWL-v2가 파라미터 절반인데 레이턴시 32배)이 이 논문의 핵심 근거표가 됩니다 —
`research_story.html` CH64 64-13에 반영해뒀습니다.

**좌우 데이터 불균형 확인 결과** (미팅 액션아이템 대응):
- 목표 위치(에피소드 수 90:90)·평균 길이(76.8:76.3프레임) — **완벽 균형**
- **액션 클래스 프레임 수(LEFT+FWD+L+ROT_L=3384 vs RIGHT+FWD+R+ROT_R=4122) —
  22% 불균형** (우측계열이 더 많음)
- 즉 수집 설계는 균형이었는데, 실제 주행 중 발생한 액션 자체가 비대칭 —
  실기 좌측 약세(64-11: strong_left 0% vs strong_right 60%)의 원인 후보입니다.
  **트랙C 재수집 시 위치·경로 개수뿐 아니라 액션 클래스 비율도 맞춰주시면
  좋겠습니다** (예: 좌측 회전/보정 동작을 의도적으로 조금 더 반복).

**요청**: 미팅 결정대로 **Tab4 스크리닝 목표치를 5가짓수×20개=100개로 갱신**
부탁드립니다. 이번에 보낸 21회 스크리닝은 큰 도움이 됐고(64-11), 표본을 100개로
늘리면 좌우 비대칭이 진짜인지(우연이 아닌지)도 더 확실해집니다.

## 🟢 [2026-07-29] 100개 스크리닝 — Tab4 갱신 기다리지 말고 기존 체크포인트로 바로 시작 요청

**Tab4 목표치 갱신(100개) 응답을 기다릴 필요 없습니다** — 이미 전달드린 체크포인트로
**지금 바로 20개씩×5가짓수 수집 진행해주세요.** 대상 체크포인트는 이미 보내드린:
- `exp73_pg448_v6_mlp.pt` (baseline)
- `exp73_pg448_trackF_v6_mlp_holdaware_seed{0,1,2}.pt` (hold-aware, 트랙A/F용)
- OWL 계열도 동일하게 있음

**배포용 최종 체크포인트도 추가 전달**: val 33ep 셀프검증(64-14, 라벨/성공판정
100% 정확 확인 완료) 끝난 뒤, hold-aware mlp를 **225ep 전체(train+val 다 합쳐서)로
재학습**한 최종 배포 버전입니다:
- `exp73_pg448_trackF_v6_mlp_holdaware_DEPLOY_seed{0,1,2}.pt`

**주의**: 이건 held-out 없이 전체 데이터로 학습한 거라 저희 쪽 offline 지표는
못 냅니다(비교용 아님) — **오직 실기 스크리닝(100개) 대상으로만 써주세요.**
기존 hold-aware(seed0/1/2, val 분리된 버전)와 이 DEPLOY 버전을 실기로 A/B
비교해주시면, "val 합쳐서 더 학습하니 실제로 나아지는지"도 확인 가능합니다.

## 🔧 [2026-07-29] DEPLOY 체크포인트 과적합 버그 수정 — v3로 교체 부탁드립니다

방금 드린 `exp73_pg448_trackF_v6_mlp_holdaware_DEPLOY_seed{0,1,2}.pt`에 **에폭 선택
과적합 버그**가 있었습니다 — 체크포인트 선택 기준을 실수로 "학습 데이터 자체 정확도"로
줘서, train-acc 94.5~94.7%(다른 정상 학습은 75~78%대)로 비정상적으로 높게 나왔습니다.
과적합된 늦은 에폭이 뽑혔을 가능성이 높습니다.

**수정**: 225ep 중 22ep(10%)를 순수 "에폭 선택용 감시 세트"로 진짜 분리(학습에서
완전히 제외)해서 재학습 → **74.5~75.1%로 정상 범위 복귀**(다른 held-out 학습들과
일치). 최종 학습은 203ep 사용(나머지 22ep은 조기종료 감시 전용, 보고엔 미사용).

**요청**: 기존 `..._DEPLOY_seed{0,1,2}.pt` 3개는 **삭제하고**,
`exp73_pg448_trackF_v6_mlp_holdaware_DEPLOY_v3_seed{0,1,2}.pt`로 교체 부탁드립니다.
스크리닝 아직 안 하셨으면 이 v3로 바로 시작해주시고, 혹시 이미 구버전으로 몇 번
돌리셨으면 그 결과는 참고만 하고 v3로 다시 해주시면 됩니다.

## 🟢 [2026-07-29] 진행 중 스크리닝(31개 시점) 확인 + 다음 모델(DEPLOY_v3) 병행 요청

지금 진행 중인 5위치×20=100개 스크리닝, **31개째 시점 스냅샷 확인했습니다**
(강좌5/10, 약좌1/3, 중앙4/5, 약우2/3, 강우10/10 — 71%). 이게 최종이 아니라
**중간 스냅샷**이라는 점, 그리고 매번 보고 주실 때 **어느 체크포인트로 돌린
결과인지**도 같이 표시해주시면 제가 정확히 추적하겠습니다(baseline인지
hold-aware인지 헷갈리기 쉬워서요).

**요청 2가지**:
1. 지금 진행 중인 체크포인트(어느 파일인지)를 계속 진행해서 100개까지 완료
2. **다음 순서로 `exp73_pg448_trackF_v6_mlp_holdaware_DEPLOY_v3_seed0.pt`
   (방금 과적합 수정한 배포용)도 같은 5위치×20개=100개로 스크리닝 병행/이어서
   부탁드립니다** — 지금 것과 비교하면 "최대 데이터로 학습한 게 실기에서
   실제로 나은지"를 바로 확인할 수 있습니다.

`episode_log.csv`를 중간중간 push해주시면 그때그때 스냅샷으로 표 갱신해서
CH64에 반영하겠습니다(최종 결론으로는 100개 다 채워진 뒤에 씀).

## 🟢 [2026-07-29] 31/33개 시점 배치의 체크포인트 확정 + 경량화 실측 결과 + Tab4 100개 갱신 완료

**체크포인트 확정** — episode_log.csv를 직접 대조해서 확인했습니다. 문의하신
17:24~18:55 배치는 `#148`~`#180` 33건(git 동기화 완료)이고, 이 구간 동안
`/model/load`를 호출한 적이 없어(다음 전환은 오늘(7/29) 파라미터 측정 작업
때뿐, 측정 후 정확히 원복함) **처음부터 끝까지 단일 체크포인트로 진행**됐습니다:

- **체크포인트: `exp73_owl_trackF_v6_mlp_holdaware_seed0.pt`**
- **그라운더: OWL-v2**
- **stop_mode: learned** (`stop_learned_min_steps=3`)
- 근거: 배치 시작 직후(#148, session `20260723_172253`) `/health`를 직접 조회해
  `inference_count=47`(그 세션의 steps 값과 정확히 일치), `stop_latched=True`,
  위 체크포인트/그라운더 값을 실측 확인함. `docs/inference_reports/session_*.json`
  파일이 이 구간엔 생성 안 돼 있어(원인 미확인 — 다음 조사 필요) 사후 인덱스로는
  못 찾고, 실시간 확인 기록으로 대신 확정한 것입니다.

**git 동기화 최신 기준 재계산** (episode_log.csv #148~180, 33건):

| 위치 | 성공률 | 시도 |
|---|---|---|
| 강좌(strong_left) | 50% | 6/12 |
| 약좌(weak_left) | 33% | 1/3 |
| 중앙(center) | 80% | 4/5 |
| 약우(weak_right) | 67% | 2/3 |
| 강우(strong_right) | 100% | 10/10 |
| **합계** | **69.7%** | **23/33** |

말씀하신 22/31(71%)과 거의 같은데, 강좌가 10→12건으로 늘어난 차이입니다.
**데이터 품질 이슈 하나 발견**: `#148`과 `#149`가 완전히 동일한 session_id
(`20260723_172253`)·steps·area·cx인데 결과 라벨만 다릅니다(실패/성공) — 같은
물리 에피소드가 라벨만 바꿔 두 번 저장된 것으로 보입니다. 100개 최종 집계 때
이런 중복은 걸러내는 게 좋을 것 같은데, 확인 후 필요하면 정리하겠습니다
(에피소드 수정 패널에 개별 삭제 버튼 추가해뒀습니다, 아래 참조).

**경량화(라즈베리파이) 실측 결과 — 요청하신 Cosmos2/OWL-v2 단독 측정 완료**:

| 구성요소 | 파라미터 | 로딩 | GPU 지연(fp32) | GPU 지연(fp16) |
|---|---|---|---|---|
| Kosmos-2 vision_model만 | 0.303B | 35.0s | 58.6ms | — |
| OWL-v2 단독 | 0.155B | 3.9s | 1901.7ms | 962.1ms |

- fp16 전환만으로 OWL-v2 지연시간 2배 개선(1901.7→962.1ms), 메모리 절반 —
  코드 한 줄로 바로 적용 가능
- GPU 없는 조건(CPU만, 라즈베리파이 유사)에서 OWL-v2 fp32는 **21.5초/프레임** —
  현재로선 실사용 불가능한 수준
- int8 동적 양자화는 **이 Jetson PyTorch 빌드에 양자화 백엔드 자체가 없어서
  시도 자체가 불가능**(`torch.backends.quantized.supported_engines == ['none']`)
  — qnnpack 지원 빌드 교체 또는 ONNX/TensorRT 변환이 선행돼야 하는 별도 과제
- 상세는 `docs/DASHBOARD_WIKI.md`("경량화 실측" 섹션)에 정리해뒀습니다

**Tab4 스크리닝 목표치 100개(5×20) 갱신 완료** — 대시보드에 "100개(미팅확정)"
모드 추가, 기본값으로 설정해뒀습니다(1차/확정 모드도 그대로 유지, 버튼으로 순환).

**추가 수정 (오늘)**:
- R1 정지 시 bbox 잔류 버그 재확인·수정 완료(캐시 초기화 문제, 미팅 액션아이템
  대응) — 이 위 배치(17:24~18:55) 전체가 이 버그가 있던 상태에서 진행됐다는 점
  참고 부탁드립니다(그라운딩 자체보다 화면 표시 문제였어서 실제 정책 판단에는
  영향 없었을 것으로 보이지만, 완전히 배제는 못 함)
- 에피소드 수정 패널에 개별 행 삭제 버튼 추가(#148/#149 같은 중복 정리용)
- 지난 스크리닝 배치를 체크포인트+날짜 기준으로 한 번에 다시 불러오는 드롭다운 추가

## 🔵 [2026-07-30] "그라운딩 실패 시 회복 불가" 원인 — 그라운더 문제인지 모델(학습데이터) 문제인지 확인 요청

오늘 강한좌(trackA_strong_left) 스크리닝 세션 3개를 프레임 단위로 분석했습니다
(soda, 로그 경로 버그도 같이 발견/수정 — session_*.json이 7/23 18:53부터
`/home/soda/docs/...`(프로젝트 밖)에 저장되고 있었음, 원인은 `inference_logger.py`가
`VLA_ROOT` 환경변수에 의존했는데 최근 launch 방식 변경으로 안 채워졌던 것.
코드는 `__file__` 기준 경로로 고쳐서 환경변수 의존 제거, 기존 45개 파일도 복구함).

**발견**: 3개 중 2개(073121, 073353)는 세션 시작부터 끝까지 OWL-v2가 단 한 번도
타겟을 못 찾았고(`has_bbox=False` 전 프레임), 정책 헤드는 이 상태에서 `FWD+L`을
8초 넘게 그대로 유지하다가(로봇이 10Hz로 계속 그 방향 이동) 뒤늦게 `FWD+R`→`ROT_L`로
전환. 그라운딩 재시도(3프레임마다, skip_n=3)도 매번 동일하게 실패해서 회복이 전혀
없었습니다.

**확인 요청**: `_build_flat_feature()`(`stage2_v2_inference_server.py:1019-1028`)를
보면 `has_bbox`가 실제로 입력 피처 4번째 값으로 명시적으로 들어갑니다
(`[cx, cy, area, has_bbox] × bbox_scale`) — 즉 모델이 "탐지 실패"라는 신호 자체는
받고 있습니다. 문제는 **학습 데이터에 `has_bbox=False` 프레임이 실제로 얼마나
있었고, 그때 사람(조이스틱) 액션이 뭐였는지**입니다:
- 만약 이런 프레임이 학습셋에 거의 없었다면(잘 정렬된 프레임 위주로 수집됐다면),
  모델이 이 입력 조합에 대해 사실상 배운 게 없어서 지금처럼 랜덤하게 방향을
  유지/전환하는 것으로 보입니다.
- **225ep 학습 데이터에서 `has_bbox=False`(또는 area가 그라운딩 실패 폴백값인)
  프레임 비율과, 그때 페어링된 액션 라벨 분포를 확인해주실 수 있을까요?**
  적다면 "모델이 이 상황을 학습 못 한 것"이 확정되는 거고, 그러면 그라운더
  앙상블/임계값 조정보다는 **`has_bbox=False`일 때 명시적으로 정지/탐색하는
  라벨을 데이터에 추가해서 재학습**하는 쪽이 근본 해법일 것 같습니다.

## 관련 문서

- 브라우징 UI: `docs/plans/plan_20260715_dataset_history_tab.md` (🗂 데이터셋
  히스토리 탭 — schema 자동 분기, scenario/cx_position 필터, 프레임 인스펙터)
- 수집 UI 이식 이력: `docs/plans/plan_20260707_dashboard_data_collector_tab.md`
- 트랙A/C/F 극단배치+오버슈트+center 설계: `docs/plans/plan_20260707_heterogeneous_instruction_extreme_cx_collection.md`
- exp73 hybrid 서버 통합 + 다음 방향 플랜: `docs/plans/plan_20260718_next_direction_hybrid_deploy.md`

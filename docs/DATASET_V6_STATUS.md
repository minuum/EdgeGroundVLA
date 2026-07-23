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

## 관련 문서

- 브라우징 UI: `docs/plans/plan_20260715_dataset_history_tab.md` (🗂 데이터셋
  히스토리 탭 — schema 자동 분기, scenario/cx_position 필터, 프레임 인스펙터)
- 수집 UI 이식 이력: `docs/plans/plan_20260707_dashboard_data_collector_tab.md`
- 트랙A/C/F 극단배치+오버슈트+center 설계: `docs/plans/plan_20260707_heterogeneous_instruction_extreme_cx_collection.md`
- exp73 hybrid 서버 통합 + 다음 방향 플랜: `docs/plans/plan_20260718_next_direction_hybrid_deploy.md`

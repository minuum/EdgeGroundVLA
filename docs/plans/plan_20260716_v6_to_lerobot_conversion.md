# Plan: V6 H5 → LeRobotDataset 변환 스크립트 (상호운용용) (2026-07-16)

> 상태: **리서치 완료, 계획 검토 대기 — 코드 미작성**

---

## 배경

기존 학습 파이프라인(RoboVLMs, `robovlm_nav/`)은 그대로 두고, 다른 VLA
프레임워크(ACT, Diffusion Policy, openpi, OpenVLA 계열 등)와의 상호운용을
위해 V6 데이터셋을 HuggingFace `LeRobotDataset` 포맷으로 **export하는
변환 스크립트만 추가**한다. 기존 수집/학습 경로는 무변경.

## 리서치 결과

### 1) LeRobot 최신 스펙 확인 (웹 검색, 2026-07-16 기준)

- 최신 안정 릴리스: `lerobot` 0.5.x (2026-07-06 PyPI 릴리스)
- 현재 포맷은 **v3.0**이 최신(v2.1에서 전환): 에피소드당 개별
  parquet/mp4 파일(v2.1) → 여러 에피소드를 묶은 샤드 parquet/mp4
  파일(v3.0)로 구조 변경. v3.0은 "수백만 에피소드" 규모를 위한 설계.
- 생성 API: `LeRobotDataset.create(repo_id, fps, features, root=...)` →
  루프 안에서 `add_frame(dict)` → 에피소드 끝나면
  `save_episode(task="...")` → 전부 끝나면 반드시 `finalize()` 호출
  (안 하면 parquet 파일 깨짐 — 공식 문서에 명시된 흔한 실수)
- `pip install lerobot` 요구사항: **Python >= 3.12**
- 별도로 `lerobot-dataset`이라는 경량 패키지도 존재(2026-03-30 출시,
  의존성 완화 버전) — PyPI 페이지가 렌더링 안 돼 세부 API는 못 봤음,
  구현 단계에서 직접 설치해 확인 필요

참고 문서(전부 2026-07-16 조회):
- [LeRobotDataset v3.0 — 공식 문서](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)
- [LeRobotDataset v3.0 블로그 발표글](https://huggingface.co/blog/lerobot-datasets-v3)
- [Porting Large Datasets to v3.0 (DROID 예제, feature 스키마 네이밍 참고용)](https://huggingface.co/docs/lerobot/en/porting_datasets_v3)
- [huggingface/lerobot GitHub](https://github.com/huggingface/lerobot)

### 2) 환경 제약 — soda는 Python 3.10, lerobot 설치 불가

| 서버 | Python | lerobot 설치 가능? |
|---|---|---|
| soda (로봇서버) | 3.10.12 | ❌ (3.12 미만) |
| minum (학습서버) | 3.12.3 | ✅ |

→ **변환 스크립트는 minum에서 실행**. minum엔 이미 soda→minum rsync로
V6 전체(180/180) + 레거시 V5 데이터가 `/home/minum/26CS/MoNaVLA/
ROS_action/mobile_vla_dataset_v5/`에 동기화돼 있어 별도 전송 불필요.

### 3) 우리 H5 스키마 → LeRobotDataset feature 매핑 초안

실제 V6 H5 하나(`strong_left_left_curve`, 89프레임) 구조:

```
attrs: episode_name, scenario(빈값 가능), cx_position, cx_path,
       total_duration, num_frames, stop_inject_n, action_chunk_size,
       obstacle_layout_type, time_period, collection_datetime,
       collection_hour, collection_minute
datasets:
  images            (N, 720, 1280, 3) uint8
  actions           (N, 3) float32   — [linear_x, linear_y, angular_z]
  action_event_types (N,) string     — 'keyboard' | 'joystick' | 'stop_inject'
```

제안 매핑 (DROID_FEATURES 네이밍 컨벤션 참고):

| V6 H5 | LeRobotDataset feature | dtype |
|---|---|---|
| `images` | `observation.images.cam_front` | `"video"` (mp4 인코딩) |
| `actions[:,0:3]` | `action` | `float32`, shape `(3,)`, `names=["linear_x","linear_y","angular_z"]` |
| `action_event_types` | `action.event_type` (비표준 커스텀 컬럼) | `"string"` |
| `cx_position` + `cx_path` | `task` (자연어 instruction로 조합, 예: `"strong_left 위치에서 left_curve로 접근"`) → `meta/tasks.jsonl` | — |
| `scenario`, `obstacle_layout_type`, `stop_inject_n`, `action_chunk_size`, `time_period`, `collection_datetime`, `episode_name` | 에피소드 레벨 커스텀 메타(별도 사이드카 parquet 또는 `meta/episodes` 확장 컬럼) | 각각 |
| — (없음) | `observation.state` | **없음 — 우리는 odometry/proprioception을 저장 안 함.** feature 자체를 생략(스펙상 필수 아님, 우리가 features dict를 직접 정의하므로 선택 사항) |

### 4) 열린 질문 (승인 시 정해주세요)

1. **v2.1 vs v3.0 타겟**: v3.0은 "수백만 에피소드" 대규모 샤딩용이라
   우리 규모(~600 에피소드)엔 과함 — 반대로 v2.1(에피소드당 파일 1개)이
   구조는 단순하지만 공식 문서는 "v2.1 → v3.0 마이그레이션"만 다루고
   v2.1을 새로 만드는 공식 경로는 비중이 줄어드는 추세. **추천: v3.0
   그대로 사용**(`LeRobotDataset.create()` API 자체가 이미 v3.0 저장소
   포맷으로 씀 — 에피소드 몇 개든 API는 동일, 그냥 파일이 자동으로
   여러 에피소드씩 묶여 샤딩됨). 다르게 생각하는 부분 있으면 알려주세요.
2. **변환 대상 범위**: V6(트랙A, 180개)만 할지, 레거시 V5(약 269개)까지
   같이 할지? V5는 스키마가 달라(`observations/images` 등) 매핑 규칙을
   따로 하나 더 만들어야 함.
3. **HF Hub에 push할지**: 이 문서 3)의 매핑은 로컬 저장까지만 가정.
   `push_to_hub()`까지 할지, 로컬 `root=` 경로에만 저장하고 끝낼지?
   (Hub push는 공개 저장소가 되므로 별도 신중한 결정 필요 — 사설/비공개
   repo_id 사용 여부도 같이 정해야 함)
4. **fps 처리**: LeRobotDataset은 `meta/info.json`에 전역 `fps`가 필요한데
   우리 에피소드는 실측 결과 프레임레이트가 에피소드마다 다름(예:
   89프레임/14.47초 ≈ 6.15fps). 대표 fps 하나로 고정할지(예: 6fps
   반올림), 에피소드별 `timestamp` 컬럼으로 실제 시간 간격을 그대로
   기록해 fps는 표시용 근사치로만 둘지?
5. **`action.event_type`처럼 표준 스키마에 없는 필드**: 그냥 버릴지,
   비표준 커스텀 컬럼으로 유지할지? (버리면 정보 손실, 유지하면 다른
   LeRobot 도구가 이 컬럼을 무시하고 넘어가는지 확인 필요)

## 파일별 변경사항 (승인 후)

- 신규 스크립트: `scripts/convert/h5_v6_to_lerobot.py` (minum에서 실행,
  soda 쪽 `robovlm_nav`/`RoboVLMs` 코드는 무변경)
- 신규 문서: `docs/LEROBOT_EXPORT.md` — 실행 방법, 매핑 규칙, 알려진 제약
  (odometry 없음, fps 근사치 등) 기록

## 스코프 밖

- soda 쪽 수집 파이프라인 변경 없음 (H5 저장 방식 그대로 유지)
- RoboVLMs 학습 코드 변경 없음 — export는 별도 산출물

# Plan: 데이터셋 히스토리 탭 — 수집된 H5 학습데이터 브라우저 (2026-07-15)

> 상태: **리서치 완료, 계획 검토 대기 — 코드 미작성**

---

## 배경 / 요청

사용자 요청(원문 취지): "세션 히스토리(📚)와 같은 포맷으로, 데이터수집(📷) 탭
아래에 데이터셋 히스토리 탭을 만들어달라. 세션 히스토리처럼 프레임 단위로
확인 가능해야 하고, 한 번에 여러 개를 확인할 수 있어야 하고, 저장 목적
(시나리오 등) 별로 필터링(드롭다운 또는 클릭)할 수 있어야 한다."

즉 지금 있는 "📚 세션 히스토리"(닫힌 루프 추론 세션 H5를 프레임 단위로
검사하는 탭)와 같은 UX를, **데이터수집(📷) 탭에서 실제로 저장한 학습용
원본 H5 에피소드**에 대해서도 만들어달라는 요청. 둘은 완전히 다른 데이터
소스임 — 세션 히스토리는 추론 결과(`docs/inference_reports/`,
`grounding/bbox` 등), 데이터셋 히스토리는 로봇을 직접 조종해 모은 학습
원본(`ROS_action/mobile_vla_dataset_v5/*.h5`).

## 리서치 결과

### 1) 기존 "세션 히스토리" 탭 구조 (그대로 참고할 UX 패턴)

- 나비 아이템: `switchTab(this, 'history')`, `tab-history` (mona_dashboard.py:4174)
- 레이아웃: 좌측 `280px` 리스트 패널(날짜별 그룹 헤더 + 카드) + 우측 프레임
  인스펙터(이미지뷰어 + 슬라이더 + 타임라인 스트립 + 요약타일 + 재생컨트롤)
- 백엔드: `GET /sessions/list`(목록, 파일명에서 sid 추출 + report json 부가정보),
  `GET /sessions/load?sid=`(프레임 메타데이터 전체를 JSON으로, 이미지는 미포함),
  `GET /sessions/frame?sid=&idx=`(프레임 1장을 JPEG로 지연 로딩)
- 리스트 카드 클릭 → `loadSessionDetail(sid)` → 슬라이더/타임라인 초기화 →
  `showInspectFrame(0)`이 `/sessions/frame`을 호출해 이미지 표시
- 이 패턴(목록 지연로딩 + 프레임 지연로딩 + 슬라이더)은 **그대로 재사용 가능** —
  같은 방식으로 데이터셋 탭도 만들면 됨.

### 2) 데이터수집 원본 H5의 실제 위치와 스키마 — **스키마가 2종류 혼재**

`collect_dir = ROS_action/mobile_vla_dataset_v5/` (총 290개 `.h5`, env var
`VLA_DATASET_DIR`로 오버라이드 가능).

**레거시 스키마 (289개, `mobile_vla_data_collector.py` 원본이 쓰던 구버전)**:
```
keys: observations/images, language_instruction, actions
attrs: scenario, pattern, distance, end_pos
```

**신규 스키마 (대시보드 `DataCollectSession._save_episode_data`, 현재 1개 —
7/15 테스트 중 생성됨, 앞으로 이 탭에서 실제로 모으는 데이터는 전부 이 형식)**:
```
keys: images, actions, action_event_types
attrs: episode_name, cx_position, cx_path, total_duration, num_frames,
       stop_inject_n, action_chunk_size, obstacle_layout_type, time_period,
       collection_datetime, collection_hour, collection_minute
       ⚠️ scenario 속성이 없음 — start_episode(scenario=...)로 받았지만
       progress json 통계용으로만 쓰고 H5 attrs에는 안 씀
```

**"저장 목적(시나리오)별 필터링" 요구사항에 필요한 정보가 신규 스키마
H5에는 없다** — `cx_position`/`cx_path`는 있지만 `scenario`(예:
`target_left_left_path`)는 없음. `episode_name`이 자동생성된 경우
`episode_{ts}_{scenario}_{pattern}_{cx_position}_{cx_path}` 형태로 이름에
포함되긴 하지만, 사용자가 커스텀 이름을 입력하면 유실됨.

→ **이번 계획에 포함**: `_save_episode_data()`에 `f.attrs["scenario"] =
self.selected_scenario or ""` 한 줄 추가 (기존 스키마 필드 추가일 뿐,
기존 필드 삭제/변경 없음 — 하위 호환 안전). 과거 데이터(레거시 289개는
이미 scenario 있음, 신규 1개는 파일명 파싱으로 커버)는 새로 안 건드림.

### 3) 프레임 이미지 서빙 — 스키마별 키가 다름

`/sessions/frame`은 `f["observations/images"][idx]`를 고정으로 읽음(추론
세션 전용). 데이터셋 탭에서는 파일마다 `images` 또는 `observations/images`
중 있는 쪽을 읽어야 함 — 새 엔드포인트에서 `"images" if "images" in f else
"observations/images"`로 분기.

### 4) episode_log.csv / 라벨링(labels.json) — 데이터셋 탭에는 해당 없음

세션 히스토리의 "라벨링"(`sessions/label`)과 "경로검증 기록 매칭"은 닫힌
루프 추론 결과 검증용 개념이라 원본 학습데이터에는 적용 대상이 아님 —
포팅하지 않음(범위 초과).

---

## 설계

### 신규 탭

- 나비 아이템 위치: **"📷 데이터수집" 바로 아래** (요청대로) —
  `switchTab(this, 'collect')` 다음 줄에 `switchTab(this, 'dataset')` 추가
  (mona_dashboard.py:3377 부근)
- 탭 라벨: `🗂 데이터셋 히스토리`
- 레이아웃: 세션 히스토리와 동일한 `280px + 1fr` 그리드, 단 리스트 패널
  상단에 **필터 바** 추가(아래 참조)

### 백엔드 (신규 엔드포인트 3개, 기존 `/sessions/*`는 건드리지 않음)

```python
GET /dataset/list
  → 290개 파일을 스캔, 각 파일 attrs만 읽어(빠름 — 이미지는 안 엶) 다음을 반환:
    { name, date, time, scenario, cx_position, cx_path, pattern,
      num_frames, duration_s, size_mb, schema: "legacy"|"new" }
  scenario 없는 신규 파일은 attrs 우선, 없으면 파일명에서 정규식 파싱 폴백.

GET /dataset/load?name=<episode_name>
  → 세션 히스토리 /sessions/load와 동일한 형태로 프레임 메타(액션 라벨,
    타입) 배열 반환. 이미지 미포함(지연 로딩).

GET /dataset/frame?name=<episode_name>&idx=<i>
  → 프레임 1장 JPEG. images/observations-images 키 자동 분기.
```

세 엔드포인트 모두 `INFER_H5_DIR`이 아니라 `_collect.data_dir`
(`ROS_action/mobile_vla_dataset_v5/`)을 읽음 — 완전히 독립된 데이터 소스.

### 필터링 UX (요청: "드롭다운이나 눌러서")

리스트 패널 상단에 두 단계:
1. **시나리오 드롭다운** — `/dataset/list` 응답에서 등장하는 고유
   scenario 값들을 모아 자동 채움(9종 + "미지정"). 선택 시 리스트 필터.
2. **cx_position 칩(클릭형)** — 강한좌/준극단좌/준극단우/강한우/(레거시는
   없음) 버튼 그리드, 클릭해서 토글 필터(다중 선택 가능, 데이터수집 탭의
   기존 필터 버튼 스타일 재사용).
드롭다운(시나리오) + 칩(cx_position)을 AND 조건으로 결합. 검색창(episode_name
부분일치)도 추가.

### 다중 선택 ("한 번에 여러 개 확인")

리스트 카드에 체크박스 추가. 1개 선택 시 기존과 동일하게 우측에 풀
프레임인스펙터. **2개 이상 선택 시** 우측 패널이 "비교 모드"로 전환:
- 선택된 각 항목을 가로 카드로 나열 — 요약 타일(프레임수/길이/시나리오/
  cx위치+경로/날짜)만 표시, 프레임 인스펙터는 안 엶(N개 동시 이미지 로드
  방지)
- 각 비교 카드에 "🔍 자세히" 버튼 → 클릭하면 그 항목만 단일 상세보기로
  전환(체크는 유지, 다시 목록으로 갈 필요 없이 비교↔상세 토글)

### 프레임 인스펙터 (단일 선택 시)

세션 히스토리의 `showInspectFrame`/타임라인 로직을 `ds-` 프리픽스로
분리 구현(동일 함수를 공유하면 `inspectSession` 전역상태가 세션탭과
꼬일 위험 → 별도 네임스페이스 권장). 세션탭에 있던 "이상치 검증"
(`warns`, latency=0 등)은 추론 세션 전용 개념이라 데이터셋 탭에는
불필요 — 대신 액션 분포(8-class, 데이터수집 탭에서 이미 만든
`_collect_classify_8class`/`COLLECT_CLASS_SYMBOLS` 재사용) 요약을
프레임 타임라인 색상으로 사용.

---

## 파일별 변경사항

- `robovlm_nav/serve/mona_dashboard.py`
  - `_save_episode_data()`: `f.attrs["scenario"] = self.selected_scenario or ""` 1줄 추가
  - 신규 함수 3개: `_dataset_scan_attrs(h5p)`(공용 attrs 파서, legacy/new 스키마 분기),
    `dataset_list()`, `dataset_load()`, `dataset_frame()` — `/sessions/*`
    엔드포인트들 근처(2130~2300줄대)에 나란히 추가
  - HTML: nav item 1줄 + `tab-dataset` 새 탭 블록(세션히스토리 `tab-history`
    블록을 뼈대로 복제 후 필터바/체크박스/비교모드 추가)
  - JS: `loadDatasetList()`, `loadDatasetDetail()`, `renderDatasetFilter()`,
    `toggleDatasetCompare()`, `showDsFrame()` 등 — 세션탭 함수명과 겹치지
    않게 `ds`/`Dataset` 프리픽스로 전부 분리

## 스코프 밖 (이번엔 안 함)

- 라벨링/episode_log 매칭 (세션탭 전용 개념)
- H5 파일 삭제/이름변경 UI (조회 전용)
- 레거시 289개 파일의 스키마 마이그레이션(그대로 둠, 읽기 시 분기 처리만)

## 열린 질문 (승인 시 같이 정해주세요)

1. **비교 모드 최대 선택 개수** — 몇 개까지 동시 비교 허용할지 (제안: 6개,
   그 이상은 카드가 너무 작아짐)
2. **scenario attrs 백필 여부** — 기존 1개 신규스키마 파일(테스트로 만든
   `episode_20260715_173157_...`)에 scenario를 소급 기록할지, 아니면
   파일명 파싱 폴백으로만 처리하고 그대로 둘지
3. **리스트 정렬 기본값** — 세션히스토리처럼 최신순 고정이면 되는지, 아니면
   시나리오별 그룹핑을 기본으로 할지

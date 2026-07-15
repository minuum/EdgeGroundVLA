# Plan: 주행검증/추론 설정을 V6 트랙A 데이터에 맞춰 최신화 (2026-07-16)

> 상태: **리서치 완료, 계획 검토 대기 — 코드 미작성**

---

## 배경

V6 트랙A(cx_position × cx_path) 데이터 수집이 완료됨: `weak_left` /
`weak_right` / `strong_right` × `left_curve` / `straight` / `right_curve`,
각 15개씩 총 135개 (`docs/DATASET_V6_STATUS.md`). 하지만 대시보드의
"주행검증"(🧪 경로 검증, Tab 4/6)과 "추론 모델 설정"(⚙️ 서버 설정) 쪽은
**이 taxonomy를 전혀 모른다** — 여전히 예전 시스템을 쓰고 있음. 실제
추론 모델로 트랙A 극단배치 상황을 검증하려 해도, 결과를 기록할 라벨
버튼 자체가 없는 상태.

## 리서치 결과

### 1) 경로검증(Tab 4/6)이 쓰는 taxonomy는 트랙A와 다른, 별개의 15종 체계

`PATH_TYPES`가 서버(Python, `mona_dashboard.py:2538-2548`)와 클라이언트
(JS, `:6817-6835` + 라벨맵 `:7726-7728`) **양쪽에 중복 정의**되어 있음:

- 위치×경로 9종: `left_left/left_straight/left_right, center_left/
  center_straight/center_right, right_left/right_straight/right_right`
- 위치 단독 3종: `obj_left/obj_center/obj_right` (목표 30개씩)
- 거리 단독 3종: `dist_10cm/dist_20cm/dist_30cm` (목표 10개씩)

버튼 UI는 두 군데 존재:
- Tab 4(`id="tab-verify"`) 퀵라벨 버튼 `:4121-4143` + 동일 옵션의
  `<select>` `:4060-4076`
- Tab 6 프레임 인스펙터(경로검증 기록 수정) 버튼 `:4483-4506`
  (`selectInspectPathType`, 주석에 "Tab 4와 동일한 버튼 그리드"라고 명시)

이 15종 중 **트랙A(weak_left/weak_right/strong_right ×
left_curve/straight/right_curve) 어디에도 대응되는 게 없음** — 트랙A로
주행검증을 하면 오퍼레이터가 결과를 억지로 기존 라벨에 끼워 맞춰야 함
(예: strong_right 데이터를 `right_right`로 잘못 기록하는 식).

### 2) `COLLECT_CX_POSITIONS`는 4종인데 실제 수집은 3종만 완료

`COLLECT_CX_POSITIONS`(`:208-213`)엔 `strong_left`도 정의돼 있음(목표
45개) — 하지만 이번 V6 수집은 `weak_left/weak_right/strong_right` 3종만
됐고 **`strong_left`는 0개**. 검증 설정을 트랙A 기준으로 맞출 때 이
gap도 같이 보여야 함(수집 자체는 이번 플랜 범위 밖, UI에 반영만).

### 3) 추론 서버의 CX_RULE_THRESHOLDS가 트랙A 밴드와 우연히 겹침 — 근거 없이 하드코딩

`stage2_v2_inference_server.py`의 `CX_RULE_THRESHOLDS`(`:228-233`):
`rot_l=0.25, fwd_l=0.40, fwd_r=0.60, rot_r=0.75` (`VLA_CX_RULE` env var로
on/off, 기본 꺼짐). **`rot_l=0.25`가 weak_left 밴드 상한(0.20~0.25)과,
`rot_r=0.75`가 weak_right 밴드 하한(0.75~0.80)과 정확히 겹침** — 하지만
코드/주석 어디에도 `COLLECT_CX_POSITIONS`와의 연결이 명시돼 있지 않고,
srvcfg 탭에서 이 4개 임계값을 편집할 UI도 없음(env var로만 조정 가능).
`scripts/sim/evaluate_closed_loop_v5.py`도 동일 구조의 임계값을
독자적으로 갖고 있음(`:371-380`, `--cx-rule` CLI 플래그).

### 4) `docs/DASHBOARD_LIVE_STATUS.md`는 아직 V6 관련 언급이 전혀 없음

현재 서빙 체크포인트(`exp71_window3_bboxscale3`)와 FWD 고착 이슈가
문서화돼 있지만(2026-07-12 기준), 트랙A/V6 데이터가 등장한 건 이후라
연결이 안 돼 있음.

---

## 설계 — 단계적 접근

### Phase 1 (이번에 승인 요청) — 경로검증에 트랙A 라벨 추가 (기존 15종 유지, 대체 아님)

기존 `PATH_TYPES`/버튼을 **삭제하지 않고** 트랙A 9종을 추가:
`trackA_weak_left_left_curve` … `trackA_strong_right_right_curve` (3×3).
이유: 기존 15종이 과거 검증 기록(episode_log.csv)이나 분석 스크립트가
참조 중일 수 있어 삭제는 위험 — 기존 검증 라인은 그대로 두고 새 옵션만
얹는 게 안전.

- `PATH_TYPES` 딕셔너리에 트랙A 9종 추가(서버 Python + 클라이언트 JS
  양쪽, 지금처럼 중복 정의된 구조 그대로 따라감 — 이 기회에 단일
  소스화까지 하면 범위가 커지므로 이번 플랜에서는 안 함, 3)에서 별도
  제안)
- Tab 4/Tab 6 버튼 그리드에 트랙A 9종을 데이터셋 히스토리 탭에서 이미
  만든 아이콘/색상 규칙 재사용(◀weak_left/▶weak_right/▶▶strong_right,
  ↰left_curve/↑straight/↱right_curve) — 시각적으로 기존 15종과 구분되게
  별도 섹션으로 묶음
- `strong_left`는 수집 안 됐으므로 버튼은 만들되 비활성화 + "미수집"
  표시 (나중에 수집되면 바로 쓸 수 있게 구조는 미리 준비)

### Phase 2 (승인 시 별도 진행) — CX_RULE_THRESHOLDS를 srvcfg에 노출 + 트랙A 밴드와 명시적 연결

- ⚙️ 서버 설정 탭에 `CX_RULE_THRESHOLDS` 4개 값 편집 UI 추가(현재
  env var로만 가능) — `applyOwlThresh()`류와 동일 패턴으로 즉시반영
- 코드 주석/문서에 "rot_l=0.25는 weak_left 상한과 동일, rot_r=0.75는
  weak_right 하한과 동일 — 트랙A 밴드 기준으로 튜닝됨"이라고 명시해
  다음 사람이 우연의 일치로 오해 안 하게 함
- `evaluate_closed_loop_v5.py`의 `--cx-rule` 기본값도 트랙A 밴드 기준과
  일치하는지 재검토(현재도 같은 값이라 아마 문제 없음, 확인만)

### Phase 3 (제안만, 별도 승인 필요 — 범위 큼) — PATH_TYPES 서버/클라 중복 정의 단일화

지금 `PATH_TYPES`가 Python(`mona_dashboard.py`)과 JS(같은 파일 내
`<script>`) 양쪽에 손으로 동기화된 채 중복 정의돼 있음. Phase 1에서
트랙A 9종을 추가하면 이 중복이 더 커짐 — 장기적으로는 서버가
`/verify/config` 같은 엔드포인트로 `PATH_TYPES`를 내려주고 JS가 그걸
받아쓰는 구조로 단일화하는 게 맞음. **이번엔 안 함**, 필요성만 기록.

### Phase 4 (제안만) — `docs/DASHBOARD_LIVE_STATUS.md`에 V6 트랙A 연결 언급 추가

`dashboard-status-sync` 스킬 규칙상 이 문서는 git/health/episode_log
근거 기반으로만 갱신 — 대화에서만 나온 내용은 사용자 승인 필요. 승인되면
"V6 트랙A 데이터 135개 수집 완료, 경로검증 탭에 대응 라벨 추가됨" 한 줄
추가 제안.

---

## 파일별 변경사항 (Phase 1 기준)

- `robovlm_nav/serve/mona_dashboard.py`
  - Python `PATH_TYPES`/`PATH_TYPE_TARGETS`(`:2538-2548` 근처)에 트랙A
    9종 추가
  - JS `PATH_TYPES` 미러(`:6817-6835`) + 라벨맵(`:7726-7728`) 동일 추가
  - Tab 4 버튼 그리드(`:4121-4143`) + `<select>`(`:4060-4076`)에 트랙A
    섹션 추가
  - Tab 6 버튼 그리드(`:4483-4506`, `selectInspectPathType`)에 동일 추가
  - `COLLECT_CX_POSITIONS`의 `strong_left`를 참조해 "미수집" 배지 로직

## 스코프 밖 (이번엔 안 함)

- 실제 `strong_left` 데이터 수집(별도 데이터수집 작업)
- Phase 2/3/4는 승인 후 별도 플랜/커밋으로 분리 진행

## 열린 질문 (승인 시 같이 정해주세요)

1. **Phase 1만 먼저 할지, 2까지 같이 할지** — 2(CX_RULE 노출)는 실제
   추론 서버 동작에 영향 주는 값이라 좀 더 신중해야 함
2. **트랙A 9종 버튼을 기존 15종과 같은 그리드에 섞을지, 완전히 분리된
   섹션(예: "🎯 트랙A 극단배치" 헤더로 구분)으로 둘지**
3. **`strong_left` 미수집 상태를 버튼에 어떻게 표시할지** — 그냥
   회색으로 비활성화 vs "수집 필요" 뱃지로 눈에 띄게

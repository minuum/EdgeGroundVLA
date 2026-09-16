# Plan: 신규조건 실기(pos2/pos3 + OWLv2/Kosmos-2 ablation) path_type 라벨 추가

**작성일**: 2026-09-16
**배경**: minum 요청(이번 주 실기 280건 + 여유 50건) — 로봇 출발위치를 기존
중앙 옆 좌·우로 새로 잡고(pos2/pos3), 각 위치에서 5방향×20(200건) + 좌우
방향에서 OWLv2/Kosmos-2 단독 ablation 20×2×2(80건) 수집. minum은 결과를
`episode_log.csv`로 받아서 `scripts/analyze_newcond_experiment.py`(minum 쪽
스크립트)로 분석할 계획.

## 문제

`episode_log.csv`는 🧪 경로검증 탭의 `path_type` 필드로만 채워지고, 이 필드는
UI(드롭다운/조이스틱 X·A)상 `PATH_TYPES`(mona_dashboard.py:3609) 고정 목록에만
있는 값으로 제한된다. `pos2_강좌`/`owl_only_강우` 같은 새 라벨은 이 목록에
없어서, 지금 상태로는 이 실기 결과가 `episode_log.csv`에 올바르게 안 남는다.

(별개 시스템인 "🎯 추론 검증 스크리닝" — D-pad 위치순환 + 체크포인트별 100건
집계 — 은 `trackA_strong_left` 같은 5-포지션 flat 라벨을 쓰는데, 이건 "타겟
객체가 화면에서 어디 있는지"를 뜻하고 minum이 원하는 "로봇 물리 출발위치
pos2/pos3"와는 다른 축이라 그대로 재사용 불가.)

## 접근 방식

`pos2_*`, `pos3_*`, `owl_only_*`, `kosmos_only_*` 14개를 minum이 준 이름
그대로 `PATH_TYPES`/`PATH_TARGETS`에 새 항목으로 추가하고(target=20 each),
"📊 경로 다이어그램 및 집계" 탭에 기존 트랙A/트랙F와 같은 패턴으로 별도 그룹/
진행률 칩을 만든다. 기존 28개 항목·트랙A/트랙F 집계 로직은 그대로 두고 순수
추가만 한다(대체 없음 — 기존 분석 스크립트가 참조 중일 수 있어서 위 트랙A/F
추가 때와 동일한 원칙 유지).

**추가할 14개 라벨**:
```
pos2_강좌, pos2_약좌, pos2_중앙, pos2_약우, pos2_강우   (target 20 × 5 = 100)
pos3_강좌, pos3_약좌, pos3_중앙, pos3_약우, pos3_강우   (target 20 × 5 = 100)
owl_only_강좌, owl_only_강우                              (target 20 × 2 = 40)
kosmos_only_강좌, kosmos_only_강우                        (target 20 × 2 = 40)
```
합계 280 (minum이 말한 "1순위 280회"와 정확히 일치).

**중요한 운영상 주의(코드로 막을 수 없음, 운영자가 인지해야 함)**: 🧪 경로검증
탭의 D-pad ◀▶(위치순환)은 여전히 기존 5개 위치(강좌/약좌/중앙/약우/강좌)만
순환하고 드롭다운을 그 값으로 되돌려버린다(`_verify_cycle_pos`,
mona_dashboard.py:1245). 이번 실기 동안은 **D-pad ◀▶를 쓰지 말고, 새로 추가되는
드롭다운 옵션 또는 빠른 선택 버튼으로 직접 라벨을 고른 뒤 X(성공)/A(실패)→L2(저장)
하는 방식**으로 진행해야 한다 — D-pad를 한 번이라도 누르면 라벨이 되돌아간다.

## 변경 파일

### 1. `robovlm_nav/serve/mona_dashboard.py` — Python (서버 집계)

**PATH_TYPES**(line 3609~3626) 끝에 추가:
```python
              "trackF_center_left_curve", "trackF_center_straight", "trackF_center_right_curve",
              # ── 신규조건 실기(2026-09-16, minum 요청) — 로봇 물리 출발위치
              # pos2(중앙 옆 좌)/pos3(중앙 옆 우) 신규 + OWLv2/Kosmos-2 단독 ablation.
              # 기존 trackA_/trackF_(화면상 타겟 위치)와는 다른 축(로봇 출발위치)이라
              # 별도 접두어로 구분.
              "pos2_강좌", "pos2_약좌", "pos2_중앙", "pos2_약우", "pos2_강우",
              "pos3_강좌", "pos3_약좌", "pos3_중앙", "pos3_약우", "pos3_강우",
              "owl_only_강좌", "owl_only_강우",
              "kosmos_only_강좌", "kosmos_only_강우"]
```

**PATH_TARGETS**(line 3627~3638) 끝에 추가:
```python
    "trackF_center_left_curve": 15, "trackF_center_straight": 15, "trackF_center_right_curve": 15,
    "pos2_강좌": 20, "pos2_약좌": 20, "pos2_중앙": 20, "pos2_약우": 20, "pos2_강우": 20,
    "pos3_강좌": 20, "pos3_약좌": 20, "pos3_중앙": 20, "pos3_약우": 20, "pos3_강우": 20,
    "owl_only_강좌": 20, "owl_only_강우": 20,
    "kosmos_only_강좌": 20, "kosmos_only_강우": 20,
}
```

**`_get_episode_summary()`**(line 3642~3675)에 newcond 집계 추가:
```python
    newcond_succ = 0
    ...
    for r in rows:
        pt, ok = r[1], (r[2] == "성공")
        if ok:
            if pt.startswith("trackA_"): trackA_succ += 1
            elif pt.startswith("trackF_"): trackF_succ += 1
            elif pt.startswith(("pos2_", "pos3_", "owl_only_", "kosmos_only_")): newcond_succ += 1
            ...
    newcond_keys = [k for k in PATH_TYPES if k.startswith(("pos2_", "pos3_", "owl_only_", "kosmos_only_"))]
    newcond_total = sum(PATH_TARGETS[k] for k in newcond_keys)
    newcond_done = sum(done_total.get(k, 0) for k in newcond_keys)
    # return 문에 "| 신규조건 {newcond_done}/{newcond_total} ({newcond_succ}성공)" 추가
```

### 2. `robovlm_nav/serve/mona_dashboard.py` — JS (클라이언트 미러 + UI)

- **`PATH_TYPES`**(line 8865~8876), **`PATH_TARGETS`**(line 8878~8889): Python과
  동일하게 14개 추가.
- **`PATH_GROUPS`**(line 8894~8907) 끝에 새 그룹 추가:
  ```js
  ["── 🆕 신규조건(pos2/pos3+ablation, 2026-09-16) ──", [
    "pos2_강좌","pos2_약좌","pos2_중앙","pos2_약우","pos2_강우",
    "pos3_강좌","pos3_약좌","pos3_중앙","pos3_약우","pos3_강우",
    "owl_only_강좌","owl_only_강우","kosmos_only_강좌","kosmos_only_강우"
  ]]
  ```
- **`updatePathSummary()`**(line 10303~10429): `trackAKeys`/`trackFKeys` 패턴과
  동일하게 `newcondKeys`/`newcond_total`/`newcond_done`/`newcond_succ`/`pct_newcond`
  계산 추가, `progressHtml`에 6번째 진행률 칩(🆕 신규조건, 목표 280) 추가 —
  기존 `grid-template-columns: repeat(5, 1fr)`(line 10370)을 `repeat(6, 1fr)`로
  변경(또는 좁은 화면 대비 `repeat(3, 1fr)` 2행 — 구현 시 기존 5개 칩 스타일과
  일관되게 결정). `vfy-progress-txt`(line 10425~10429)에도 신규조건 텍스트 한 줄 추가.

### 3. `robovlm_nav/serve/mona_dashboard.py` — HTML (수동 선택 UI)

- **`#ep-path-type` 드롭다운**(line 5758~5781) 끝에 새 optgroup 추가:
  ```html
  <optgroup label="🆕 신규조건(pos2/pos3+ablation)">
    <option value="pos2_강좌">pos2 강좌</option>
    <option value="pos2_약좌">pos2 약좌</option>
    <option value="pos2_중앙">pos2 중앙</option>
    <option value="pos2_약우">pos2 약우</option>
    <option value="pos2_강우">pos2 강우</option>
    <option value="pos3_강좌">pos3 강좌</option>
    <option value="pos3_약좌">pos3 약좌</option>
    <option value="pos3_중앙">pos3 중앙</option>
    <option value="pos3_약우">pos3 약우</option>
    <option value="pos3_강우">pos3 강우</option>
    <option value="owl_only_강좌">OWLv2단독 강좌</option>
    <option value="owl_only_강우">OWLv2단독 강우</option>
    <option value="kosmos_only_강좌">Kosmos-2단독 강좌</option>
    <option value="kosmos_only_강우">Kosmos-2단독 강우</option>
  </optgroup>
  ```
- **빠른 레이블 선택 버튼**(line 5820~ 부근, obj_left 3-버튼 그리드 패턴 참고)에
  14개 버튼 추가 — `onclick="selectPathType('pos2_강좌')"` 등, 기존
  `grid-template-columns:repeat(3, 1fr)` 그리드 재사용(5줄 정도 추가).

### 변경 없음 (참고용, 손 안 댐)
- `_verify_cycle_pos`/`VERIFY_SCREEN_POSITIONS`/D-pad 로직 — 위 "운영상 주의"
  참조, 코드 변경 없이 운영으로 회피.
- `ablation_mode`(`VLA_ABLATION_MODE`) 관련 코드 — 이미 존재, 이번 변경과 무관.

## 트레이드오프

- **장점**: minum이 준 이름 그대로 써서 `analyze_newcond_experiment.py`와의
  문자열 매칭 리스크 없음. 기존 28개 타입/트랙A·F 집계 로직 무변경(순수 추가).
- **단점**: 진행률 칩 6개로 늘어나면서 그리드 레이아웃을 살짝 조정해야 함
  (5→6칸, 화면 좁으면 줄바꿈 필요할 수 있음 — 구현 시 실제로 확인).
- **대안 검토**: `note` 필드에 라벨을 넣는 방법도 가능했지만, `note`는 자유
  텍스트라 오타/형식 불일치 위험이 크고 기존 PATH_TARGETS 진행률 UI(20/20 카운터,
  성공률 등)를 전혀 못 씀 — 드롭다운 방식이 오타 방지·진행률 가시성 면에서 더 나음.

## 완료 기준
- [x] Python `PATH_TYPES`/`PATH_TARGETS`/`_get_episode_summary` 14개 반영
- [x] JS `PATH_TYPES`/`PATH_TARGETS`/`PATH_GROUPS`/`updatePathSummary` 미러링
- [x] 드롭다운 optgroup 14개 옵션 추가
- [x] 빠른 선택 버튼 14개 추가
- [x] `py_compile` 통과, 대시보드 재시작 후 `/episodes/log`에 `pos2_강좌`로 직접
      curl 테스트 → `episode_log.csv`에 정상 기록 확인(`신규조건 1/280 (1성공)`),
      테스트 행은 `/episodes/delete`로 원복(`신규조건 0/280`)
- [ ] 진행률 칩(🆕 신규조건 0/280)이 UI에 정상 렌더링되는지 브라우저로 확인 —
      API 레벨 검증만 완료, 실제 브라우저 렌더링은 미확인(soda가 실기 시작 시
      확인 예정)

## 참고: 무관한 기존 이슈 발견 (이번 변경과 무관, 수정 안 함)

테스트 중 `트랙A 0/180 (176성공)`처럼 done/succ 불일치가 있는 걸 발견했는데,
이건 이번 변경과 무관한 기존 상태입니다 — 🎯 추론 검증 스크리닝(D-pad) 흐름이
`trackA_strong_left` 같은 5-position flat 라벨로 기록하는 반면, `PATH_TYPES`의
트랙A 12종은 커브 접미사가 붙은 `trackA_strong_left_left_curve` 형태라 서로
다른 문자열이라서 `done_total` 딕셔너리에는 안 잡히고 prefix 매칭(`trackA_succ`)
에만 잡히는 것으로 보입니다. 이번 pos2_*/pos3_*/owl_only_*/kosmos_only_*는
스크리닝(D-pad) 흐름과 접두어가 전혀 겹치지 않아 이 문제에 영향받지 않습니다.

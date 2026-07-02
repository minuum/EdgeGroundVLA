---
name: episode-analysis
description: 경로검증 탭 에피소드 기록(logs/episode_log.csv)과 추론 세션 H5(docs/inference_sessions/)를 분석해 오브젝트/경로별 성공률·FPE·실패 유형을 정리한다. "에피소드 분석해줘", "오브젝트별 성공률", "방금 세션 왜 실패했는지 확인", "obj_right만 왜 이렇게 안 되지" 같은 요청에 사용.
---

# Episode / Session Analysis

실로봇 경로검증 탭에서 쌓인 에피소드 기록과 추론 세션 H5를 분석하는 스킬.
2026-07-02 대화에서 수동으로 하던 절차(집계표 + FPE 기준 성공/실패 분리 +
실패 메모 키워드 분류 + 특정 세션 프레임별 grounding 덤프)를 스크립트로 코드화.

## 언제 쓰나
- "에피소드 분석해줘", "오브젝트별/경로별 성공률", "왜 실패했는지 확인"
- "방금 세션 분석해줘" → `--recent`
- "특정 세션 프레임별로 봐줘" → `--session <id>`
- Tab4(경로 검증) 집계표에 이상한 수치가 보일 때 원인 추적

## 실제 스크립트
```
scripts/analysis/analyze_episode_log.py
```

## 사용법
```bash
# 전체 집계 (obj_/dist_/right_/center_/left_ 그룹별 성공률·FPE·실패분류)
python3 scripts/analysis/analyze_episode_log.py

# 특정 그룹만 (접두사 필터)
python3 scripts/analysis/analyze_episode_log.py --group obj_

# 가장 최근 수집 세션을 프레임별로 덤프 (action/cx/cy/area/has_bbox/cached/latency)
python3 scripts/analysis/analyze_episode_log.py --recent

# 특정 세션 ID 지정
python3 scripts/analysis/analyze_episode_log.py --session 20260702_200616
```

## 무엇을 계산하나
1. **경로/위치별 집계**: 건수, 성공률, 평균 FPE, 평균 steps, 평균 latency
2. **FPE 기준 성공/실패 분리 검증**: 성공군 평균 FPE vs 실패군 평균 FPE 비교.
   2026-07-02 기준 **FPE ≈0.2가 성공/실패 분기점**으로 관찰됨 (성공 평균 0.03,
   실패 평균 1.04) — 데이터 늘어나면 스크립트 상단 `FPE_SUCCESS_THRESHOLD` 재검증할 것.
3. **실패 메모 키워드 분류** (`FAILURE_PATTERNS`, 스크립트 상단에서 직접 수정 가능):
   - **프리뷰/회전 반복** ("도리도리", "빙글뱅글", "반복되는") — 로봇이 제자리에서
     맴도는 증상. 2026-07-02에 SYNC/PRE 모드가 az(회전)를 실제로 못 내던 버그
     (`mona_dashboard.py` `_loop_sync`/`_async_infer`가 2D action만 읽던 문제,
     커밋 `45fcff07`로 수정됨)와 연관된 것으로 추정 — 버그 수정 후 재발 여부 확인 필요.
   - **그라운딩 오탐/미탐** ("다른 객체", "인식", "엇나") — PG2 자체 한계.
   - **경로 이탈** ("오른쪽/왼쪽으로 가버림", "너무 돌아버림") — 방향 확정 후 발산.
   - **각도/정밀도 부족** ("각도", "살짝") — 방향은 맞는데 정밀도 부족, 성공/실패
     경계 케이스가 많음.
4. **세션 프레임 덤프 시 자동 탐지**: grounding이 특정 프레임부터 세션 끝까지
   계속 실패(`has_bbox=False`)하면 경고 출력 — "그 시점부터 fallback bbox로
   방향 근거 없이 주행했을 가능성".

## 알려진 구조적 발견 (재사용 가능한 컨텍스트)
- **obj_right 성공률 0%** (2026-07-02 기준 3건 중 0건) — minum이 찾은 PG2 학습분포
  편향(cx>0.7 케이스가 학습데이터의 7.8%뿐)과 직결. 오른쪽 목표는 카메라 상
  cx가 높은 쪽으로 치우쳐야 탐지되는데 그 구간 데이터가 적음.
- **첫 프레임 탐지 후 곧바로 놓치는 패턴**: 0번 프레임에서 실제 탐지 성공 →
  로봇이 움직이자마자(다음 2~3프레임) 그라운딩이 놓치고 세션 끝까지 fallback으로
  방황 — `docs/v5/grounding_analysis/`의 first-frame failure 분석과 같은 계열 문제.
- 관련 스킬: `grounding-session-pipeline`(jsonl/mp4 그라운딩 검증 세션 — 이건 H5
  추론 세션과 다른 데이터), `sync-inference-session`(분석 후 minum으로 전송).

## 주의
- `episode_log.csv`는 Tab4(경로 검증)에서 "💾 기록 저장" 버튼을 눌러야 쌓임 —
  버튼 안 누르고 주행만 하면 이 스크립트로는 안 잡히고 H5 세션(`--recent`/`--session`)으로만 확인 가능.
- 표본이 작을 때(건당 3~8건) 성공률/FPE 수치는 참고용 — 확정적 결론 내리기 전
  건수 언급할 것.

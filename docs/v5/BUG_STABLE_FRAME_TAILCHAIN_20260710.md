# 버그: 세션 간 첫 프레임 꼬리물림 (`_ros._stable` 미초기화)

> 2026-07-10, SODA 발견/수정. `robovlm_nav/serve/mona_dashboard.py`

## 증상

오늘 `obj_right` 폐루프 테스트 중, 새 세션 시작 직후 "왜 그렇게 갔는지 모르겠다"는
방향 이상 실패가 반복 관찰됨 (`아예 반대 방향으로 감`, `아예방향을틈 왜인지도
모르겠고` 등).

## 원인

`_loop_sync()`(SYNC/PRE 모드, obj_right 테스트 기본값)에서:

```python
bgr = (_ros._stable if mode == "SYNC" and _ros._stable is not None
       else _ros.latest_bgr())
...
if mode == "SYNC":
    time.sleep(_ros.ctrl.move_duration + 0.15)
    _ros._stable = _ros.latest_bgr()   # 매 스텝 종료 후 "정착 프레임" 캐시
```

`_ros._stable`은 세션이 끝나도 초기화되지 않는 전역 상태. 세션 종료 시 마지막
스텝의 "정착 프레임"이 여기 남아있고, **다음 세션이 시작되면 그 값을 그대로
재사용** — 즉 새 세션의 1번째 추론(`action[0]`)이 실제로는 이전 세션이 끝나던
순간의 낡은 이미지를 입력으로 8001 서버에 전달되어 계산됨. 이후 스텝(2번째부터)은
정상.

## 수정

`mona_dashboard.py:1044`, `_loop_sync()` 시작부에 한 줄 추가:

```python
_ros._stable = None
```

세션 시작 시 캐시를 비워 첫 스텝부터 항상 `_ros.latest_bgr()`(실시간 프레임)를
쓰도록 함. 배포/검증 완료 (대시보드 재시작, `/health`·`/camera/stream` 정상,
8001 추론 서버 프로세스는 미영향).

## 오늘자 데이터 영향 범위 및 조치

- 오늘 obj_right 세션 30개 중 **28개(대시보드 재시작 직후인 2개 제외 전부)**가
  이 패턴에 해당. 사실상 하루치 연속 체인 전체.
- H5(`docs/inference_sessions/session_*.h5`)와 JSON 리포트
  (`docs/inference_reports/session_*.json`), `logs/episode_log.csv`
  (전부 gitignore 대상이라 git으로는 안 감 — 로컬/SODA 서버에만 존재)를
  **실물로 보정**: 오염된 `frame[0]+action[0]+bbox`를, 그게 실제로 속했던
  이전 세션 파일 끝으로 이동. 프레임 총합은 이동 전/후 509개로 동일(유실 없음).
  `episode_log.csv`의 `steps`도 ±1 반영, 메모에
  `[frame0 이전세션으로 이동보정됨]` 태그.
- 수정 전 원본은 백업 보관(SODA 로컬 `/tmp/.../scratchpad/session_backup_20260710/`).

## 실제 성공/실패에 영향을 줬는가 — 정량 분석

이미 기록된 실제 추론 결과(오염된 `action[0]` vs 그 직후 진짜 결정)를 비교:

| | 발산(라벨 다름) | 일치(라벨 같음) |
|---|---|---|
| 세션 수 | 10/29 | 19/29 |
| 성공률 | 6/10 = 60.0% | 10/18 = 55.6% |

통계적으로 유의미한 차이 없음(표본 작음). 오염된 첫 액션 29개 중 27개가
`FORWARD`(데이터셋 최빈 클래스)라 "틀린 이미지를 봐도 결과적으로 같은 액션을
뱉는" 경우가 대부분. 다만 개별 사례로 `#98`(155545, 실패, "아예방향을틈"),
`#99`(155802, 실패, "아예반대방향으로감")는 오염된 첫 액션=FORWARD인데 직후
진짜 결정은 ROT_L/ROT_R로 크게 갈려 — 이 두 건은 버그가 실제 원인이었을
가능성이 높음.

**결론**: 오늘 obj_right 성공률(56.7%, 17/30)을 이 버그 때문에 통계적으로
재계산할 근거는 약함(모델이 FORWARD를 워낙 선호해서 첫 프레임 오류가 결과에
잘 안 묻어남). 다만 구조적으로 하루 데이터 대부분이 1스텝 노이즈를 안고 있었던
건 사실이며, **내일부터의 세션이 진짜 클린 베이스라인**.

## 관련

[[plan_20260707_heterogeneous_instruction_extreme_cx_collection]] — 별개 계획
문서지만 같은 세션에서 발견됨.

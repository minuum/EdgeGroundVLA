# Finding: flicker의 실제 메커니즘 — PG2 다중 중복 탐지 (n_locs 이상 패턴)

**보낸 쪽:** soda
**관련:** `REQUEST_20260702_FLICKER_LOGGING.md` (Fix3+Fix4 구현 요청)에 대한 응답 + 신규 발견
**데이터:** `inference_sessions_recv/20260703/` (세션 5개 + `grounding_decisions_20260703.jsonl` 68줄)

---

## 추가 (같은 날 재확인): stopping criteria 실측 결과 — 대부분 해결, 엣지케이스 1건

minum이 이미 세미콜론 `StoppingCriteria`(`bd65c96a`)를 구현해놓은 걸 확인하고 soda에서
재시작 후 실측했습니다. **대부분 케이스는 잘 잘림** — 재시작 직후 8회 호출 중 7회가
`n_locs=4`, ~2.1~2.2s로 정상화됨.

**근데 1건은 여전히 새어나감**: `n_locs=23`, `latency=7906ms`. raw_output 확인 결과
`phrase="gray plastic bin"`(멀티프롬프트 fallback 프롬프트)로 호출된 케이스였고,
`;`가 포함된 긴 출력이 그대로 나왔습니다.

**추정 원인**: `self._proc.tokenizer.encode(";", add_special_tokens=False)`이 **독립된
";"의 토큰 ID**(235289)를 구하는데, 실제 생성 컨텍스트에서는 `" ;"`(공백+세미콜론)이
BPE 규칙상 **다른 토큰 ID로 병합**될 수 있습니다. `phrase`가 바뀌면 앞뒤 문맥이
달라지니 이 병합 결과도 달라질 수 있어서, 특정 프롬프트/문맥에서만 새어나가는
것으로 보입니다.

**개선 방향 제안**: 토큰 ID 단일 매칭 대신, `" ;"`/`";"` 등 공백 유무 두 variant를
모두 인코딩해서 stop 토큰 집합에 넣거나, `StoppingCriteria`에서 디코딩된 텍스트가
`";"`로 끝나는지 문자열 기준으로 검사하는 방식이 더 견고할 것 같습니다.

---

## 상태: Fix3 + Fix4 구현·검증·적용 완료

- Fix3(버전 핸드셰이크): `/health`에 `git_commit`/`process_started_at`/`code_mtime` 추가,
  재시작 후 실측 확인 (`git_commit=6501a600`, `code_mtime < process_started_at` 정상)
- Fix4(PG2 판정 영구 로그): `logs/grounding_decisions.jsonl`에 `filter_reason`/raw
  locs/호출당 latency 기록 확인
- 커밋: `ec032a93` (monavla-driving에 푸시됨, minum 요청 문서 `6501a600` 이후)
- 재주행: `grounding_skip_n=1`, `multi_prompt=OFF`, SYNC 위주로 obj_right 5세션
  재수집 (H5 attrs에 세션별 `runtime_config` 스냅샷 포함됨 — 이번에 같이 추가한 기능)

## 신규 발견: filter_reason만으로는 안 보이던 것 — n_locs

jsonl의 `n_locs` 필드(원래는 latency 분해용으로 추가했던 것)를 보다가 발견:
**정상 케이스는 `n_locs=4`(bbox 1개), flicker 실패 케이스 중 상당수가 `n_locs=24`(bbox 6개
분량)** 로 명확히 두 그룹으로 갈립니다. 그리고 이 두 그룹의 latency도 정확히 갈립니다
(2.1s vs 8.3~8.6s — 토큰 수 6배 차이가 그대로 생성 시간 6배로 반영됨).

### raw_output 실측 (n_locs=24 케이스)

```
<loc0568><loc0635><loc0610><loc0658> gray basket ;
<loc0577><loc0698><loc0596><loc0717> gray basket ;
<loc0577><loc0726><loc0591><loc0741> gray basket ;
<loc0577><loc0726><loc0585><loc0741> gray baske...
```

PG2가 `;` 구분자로 **"gray basket"을 4회 연속 검출**했습니다. 좌표를 계산해보면 4개
전부 `cx≈0.55~0.58, cy≈0.62~0.74`의 **좁은 한 구역**에 몰려 있고, 그 구역의 실제
면적은 `(0.6432-0.6207)×(0.5963-0.5552) ≈ 0.0009`로 진짜 티끌 수준입니다.

**즉 필터 로직 자체는 정상 작동**하고 있습니다(진짜 tiny라서 걸러진 것). 문제는
**PG2가 이 자잘한 텍스처를 "gray basket"으로 4번이나 중복 확신**하고 있다는 점입니다.

### 이 좌표가 여러 시각에 걸쳐 반복됨 — 랜덤이 아니라 고정된 함정

5개의 `n_locs=24` 케이스(23:42~23:53, 10분 이상 간격)를 비교하면 좌표가 거의
동일합니다:

| 시각 | 대표 좌표(x1,y1) | phrase |
|---|---|---|
| 23:42:01 | (0.555, 0.621) | gray basket |
| 23:42:14 | (0.555, 0.621) | gray bin (fallback) |
| 23:52:41 | (0.555, 0.621) | gray basket |
| 23:53:00 | (0.548, 0.621) | gray basket |
| 23:53:29 | (0.537, 0.621) | gray basket |

**씬 안의 특정 물체/텍스처(바닥 패턴, 케이블, 그림자 등으로 추정)를 PG2가 계속
"gray basket 조각"으로 오인**하는 것으로 보입니다. Jetson 환경에서만 재현되고
minum 로컬(GB10)에서는 안 됐다는 기존 관찰과 맞춰보면, **이 씬 자체를 minum 쪽에서도
정확히 같은 카메라 각도/거리로 재현하지 않으면 로컬에서 이 함정이 안 걸릴 수 있음**
— torch/transformers 버전 차이보다 **입력 이미지(씬) 자체의 차이**일 가능성도
열어둘 필요가 있어 보입니다. (H5의 해당 프레임 이미지를 직접 비교해보시는 걸 추천)

## 소소하지만 바로 적용 가능한 개선안

`generate(max_new_tokens=48, ...)` — 단일 박스는 9토큰(`<loc>×4+phrase+eos`)이면
충분한데 48로 열어놔서 다중 검출 시 8초 가까이 낭비됩니다. `;` 토큰이나 첫 `<eos>`에서
멈추는 stopping criteria를 걸면 이 낭비를 없앨 수 있을 것 같은데, filter_reason
분포 분석이 끝난 뒤 논의하는 게 나을 것 같아 일단 적용은 안 했습니다.

---

*관련: FIX3_SERVER_VERSION_HANDSHAKE.md, FIX4_PG2_DECISION_LOG.md, REQUEST_20260702_FLICKER_LOGGING.md*

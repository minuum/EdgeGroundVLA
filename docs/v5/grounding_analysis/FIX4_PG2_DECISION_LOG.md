# Fix 4: PG2 판정 영구 로그 (grounding_decisions.jsonl)

**우선순위:** ★★ (사후 분석 불가 문제 재발 방지)
**작업 위치:** `soda:~/MoNaVLA/robovlm_nav/serve/stage2_v2_inference_server.py`
**작업 난이도:** PG2Grounder.run() 끝에 JSONL append 5줄
**배경:** 2026-07-02 21:26~21:27 세션(212648)에서 t9/t15가 왜 has_bbox=False가
됐는지 조사하려 했으나, 해당 시각 `s2v2_server.log`가 이미 로테이션되어 사라져
원인 규명이 불가능했음.

---

## 문제

`[PG2] FILTER full-frame/tiny/top/x-full` 로그는 `logger.info()`로만 남고,
일반 서버 로그(`logs/s2v2_server.log`)에 다른 로그와 섞여 쌓인다.
서버 재시작마다 로그가 로테이션(`s2v2_server.YYYYMMDD_HHMMSS.log`로 백업)되고,
용량 관리를 위해 오래된 로그가 삭제되거나 덮어써질 수 있어
**며칠 뒤에는 특정 프레임의 판정 근거를 재구성할 방법이 없다.**

실제로 로컬에서 동일 이미지로 PG2를 재실행했을 때 has_bbox=True가 나왔는데
soda 실측은 False였음 — 이 불일치가 "필터 때문"인지 "raw 생성 자체가 달라서"인지
구분하려면 **그 순간의 raw_output과 필터 판정 결과가 영구 보존**돼 있어야 한다.

---

## 수정 내용

### `PG2Grounder.run()` 끝부분에 JSONL 기록 추가

`robovlm_nav/serve/stage2_v2_inference_server.py`의 `run()` 메서드,
`return result` 직전(현재 라인 498 부근)에 추가:

```python
        if return_raw:
            result["raw_output"] = raw
        if return_hidden:
            result["hidden_state"] = hidden_vec

        # Fix 4: 판정 근거를 영구 JSONL로 기록 (일반 로그와 분리, 로테이션 영향 없음)
        _log_pg2_decision(phrase=phrase, raw=raw, locs=locs, result=result)

        return result
```

### 로깅 함수 정의 (파일 상단, `PG2Grounder` 클래스 정의 이전)

```python
import json as _json_pg2
from datetime import datetime as _dt_pg2

_PG2_DECISION_LOG = ROOT / "logs" / "grounding_decisions.jsonl"
_PG2_DECISION_LOG.parent.mkdir(parents=True, exist_ok=True)


def _log_pg2_decision(phrase: str, raw: str, locs: list[float], result: dict) -> None:
    """PG2 grounding 판정 근거를 영구 JSONL에 append. 로그 로테이션과 무관하게 보존.
    2026-07-02 사고(해당 시각 로그 소실로 원인 규명 불가) 재발 방지."""
    try:
        entry = {
            "ts": _dt_pg2.now().isoformat(),
            "phrase": phrase,
            "raw_output": raw[:200],
            "n_locs": len(locs),
            "has_bbox": result.get("has_bbox", False),
            "cx": result.get("cx"),
            "cy": result.get("cy"),
            "area": result.get("area"),
        }
        with open(_PG2_DECISION_LOG, "a") as f:
            f.write(_json_pg2.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass  # 로깅 실패가 추론을 막아서는 안 됨
```

---

## 로그 파일 관리

- 경로: `~/MoNaVLA/logs/grounding_decisions.jsonl`
- **일반 서버 로그와 별도** — 로테이션/재시작에 영향받지 않고 계속 append됨
- 무한정 커지는 걸 막으려면 주기적으로 `logs/grounding_decisions_archive/`로 월별 이동 권장
  (자동 로테이션은 이번 스코프에서 제외 — 필요해지면 별도 논의)

### 분석 예시

```python
import json
entries = [json.loads(l) for l in open("logs/grounding_decisions.jsonl")]
false_entries = [e for e in entries if not e["has_bbox"]]
print(f"has_bbox=False 비율: {len(false_entries)/len(entries)*100:.1f}%")
# raw_output이 비어있으면(<loc> 토큰 미생성) 진짜 미탐지
# raw_output에 <loc>가 있는데 has_bbox=False면 필터에 걸린 것 (area/cy 값으로 확인)
```

---

## sync-inference-session 반영

soda `sync-inference-session` 스킬이 `server_health.json`과 함께
`logs/grounding_decisions.jsonl`의 최근 세션 구간(전송 시각 -N분)만 잘라
같이 전송하도록 확장하면, minum 쪽에서 특정 프레임의 판정 근거를 즉시 재구성 가능.
(이번 스코프에는 미포함 — 필요 시 `sync-inference-session` SKILL.md에 Step 추가)

---

## 부수 효과 없음 확인

- 로깅 실패해도 `except: pass`로 추론 흐름 막지 않음
- 파일 append만 — 기존 grounding 로직/필터 값 변경 없음
- 매 grounding 호출마다 파일 open/append 1회 — I/O 오버헤드 미미 (< 1ms, skip_n=3이라 실제 호출 빈도 낮음)

---

## 관련 문서

- `docs/v5/grounding_analysis/FIX3_SERVER_VERSION_HANDSHAKE.md` — 코드-프로세스 버전 불일치 방지
- `docs/v5/grounding_analysis/FIX1_RESIZE_BUG.md` — 이중 리사이즈 버그

---

*작성: 2026-07-02 | 근거: session_20260702_212648.h5 t9/t15 has_bbox flicker 원인 조사 중 로그 소실 확인*

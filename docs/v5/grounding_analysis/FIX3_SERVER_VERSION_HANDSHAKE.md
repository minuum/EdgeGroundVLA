# Fix 3: 서버 버전 핸드셰이크 (코드-프로세스 불일치 방지)

**우선순위:** ★★★ (데이터 오염 재발 방지 — 2026-07-02 Fix1 사고의 재발 방지책)
**작업 위치:** `soda:~/MoNaVLA/robovlm_nav/serve/stage2_v2_inference_server.py`
**작업 난이도:** 헬스 응답에 필드 3개 추가 + 수신측 검증 로직
**검증:** `/health` 응답에 `git_commit`, `process_started_at`, `code_mtime` 확인

---

## 사고 경위 (왜 필요한가)

2026-07-02, PG2Grounder의 `resize_for_vlm` 이중 리사이즈 버그(Fix1)를 커밋(13:46)했지만
**서버 프로세스 재시작은 15:49**에야 이뤄졌다.

그 사이(13:47~13:55) 수집된 `obj_left`/`obj_right`/`obj_center` 실주행 세션 5개(ep50~54)는
**Fix1이 반영되지 않은 구 프로세스**로 그라운딩됐다 — 코드는 고쳐졌지만 실행 중인 프로세스는
여전히 버그 있는 코드였던 것.

문제는 이걸 **아무도 즉시 알아챌 수 없었다는 점**이다. `git log`로 커밋 시각을 확인하고,
`ps aux`로 프로세스 시작 시각을 대조해야만 발견 가능했다. 세션을 수신하는 쪽(minum)은
`server_health.json`만 보고는 "이 세션이 어떤 코드로 수집됐는지" 전혀 알 수 없었다.

**해결 방향:** 서버가 자신이 실행 중인 코드의 버전과 기동 시각을 스스로 노출하게 만들고,
수신 측 스킬이 "코드 파일 수정 시각 > 프로세스 시작 시각"이면 자동 경고를 띄우게 한다.

---

## 수정 대상 파일

`robovlm_nav/serve/stage2_v2_inference_server.py`

### 1) 모듈 상단에 프로세스 기동 시각 기록

파일 최상단 import 구역 근처에 추가:

```python
import subprocess
import time as _time

_PROCESS_START_TS = _time.time()  # 모듈이 로드된 시각 = 프로세스(uvicorn worker) 기동 시각


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],  # repo root
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


_GIT_COMMIT = _get_git_commit()  # 프로세스 기동 시점에 1회만 계산 (매 요청마다 subprocess 호출 방지)
```

### 2) `/health` 엔드포인트 응답에 필드 추가

기존 health 응답 딕셔너리(예: `{"status": "healthy", "model_loaded": ..., ...}`)를 구성하는
지점에 아래 3개 필드를 추가한다:

```python
health_response = {
    "status": "healthy",
    "model_loaded": model_loaded,
    # ... 기존 필드들 ...
    "git_commit": _GIT_COMMIT,
    "process_started_at": _PROCESS_START_TS,
    "code_mtime": os.path.getmtime(__file__),
}
```

- `git_commit` — 프로세스가 기동될 때 체크아웃돼 있던 커밋 해시. "Fix1이 포함된 커밋을 돌리고 있는가"를 즉시 확인 가능.
- `process_started_at` — 유닉스 타임스탬프. 프로세스가 언제 뜬 서버인지 명시.
- `code_mtime` — 현재 이 파일(`stage2_v2_inference_server.py`)의 마지막 수정 시각. 코드가 프로세스 기동 이후에 다시 수정됐는지 판별하는 기준.

> `code_mtime > process_started_at` 이면 **"파일은 고쳐졌는데 프로세스는 옛날 상태로 떠있다"**는 뜻 — 바로 이번 사고 패턴.

---

## soda 반영 방법

```bash
cd ~/MoNaVLA
# 1) 파일 수정 (위 내용 참고: import 구역 + health 응답 딕셔너리 2곳)
nano robovlm_nav/serve/stage2_v2_inference_server.py

# 2) 서버 재시작 (필수 — 재시작해야 새 process_started_at이 기록됨)
bash scripts/run/stop.sh
bash scripts/run/go.sh

# 3) 검증
curl -s http://localhost:8001/health -H "X-API-Key: vla-secret-key-2025" | python3 -m json.tool
# git_commit, process_started_at, code_mtime 3개 필드가 응답에 있는지 확인
```

---

## sync-inference-session 스킬 반영 (soda 측, 전송 시점)

`server_health.json`을 생성/전송하는 스킬(soda `sync-inference-session`)이 있다면,
`/health` 응답을 그대로 스냅샷하므로 위 수정만으로 자동 반영된다.
별도 수정 불필요 — health 엔드포인트 확장이 곧 전송 필드 확장이다.

---

## minum 측 검증 로직 (이미 반영됨)

`.agent/skills/receive-inference-session/SKILL.md` Step 1에 아래 체크가 추가되어,
세션 수신 즉시 코드-프로세스 불일치를 자동 경고한다:

```python
h = json.load(open(f"{RECV}/{DATE}/server_health.json"))
code_mtime = h.get("code_mtime")
started_at = h.get("process_started_at")
if code_mtime and started_at and code_mtime > started_at:
    print(f"⚠️ 경고: 코드 파일이 프로세스 시작({started_at}) 이후 수정됨(mtime={code_mtime})")
    print("   → 이 세션들은 구 코드(버그 미수정 상태)로 수집됐을 가능성 있음. 서버 재시작 필요.")
print(f"git_commit: {h.get('git_commit', 'unknown')}  (Fix1 포함 커밋: 2324c42f 이후인지 확인)")
```

---

## 부수 효과 없음 확인

- 기존 health 응답 필드 변경 없음 — 필드 3개 추가만
- `_get_git_commit()`은 프로세스 기동 시 1회만 호출 (요청마다 subprocess 호출 없음 → 레이턴시 영향 없음)
- 그라운딩/추론 로직 변경 없음

---

## 관련 문서

- `docs/v5/grounding_analysis/FIX1_RESIZE_BUG.md` — 이번에 재발한 이중 리사이즈 버그 (Fix1)
- `.agent/skills/receive-inference-session/SKILL.md` — minum 수신측 분석 스킬
- `.agent/skills/sync-inference-session/SKILL.md` — soda 송신측 스킬 (soda 저장소 참조)

---

*작성: 2026-07-02 | 근거: Fix1 커밋(13:46) vs 서버 프로세스 기동(15:49) 시각 불일치로 obj_* 세션 5개 오염 확인*

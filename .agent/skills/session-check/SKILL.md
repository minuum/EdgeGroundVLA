---
name: session-check
description: Use when verifying that inference sessions are being saved correctly after robot runs. Checks inference_sessions/ H5 files and inference_reports/ JSON files, reports latest session status, frame counts, and action label distribution.
---

# Session Check Skill

## When to Use
- After a robot inference run, to verify the session was actually saved
- When debugging missing session data (manual stop not saving)
- When adding new sessions to robot_tests.html
- When the user asks "세션 저장됐어?", "최근 세션 확인해줘", "세션 수집됐어?"

## Key Directories
- **H5 (images + actions):** `docs/inference_sessions/session_YYYYMMDD_HHMMSS.h5`
- **JSON (metadata + history):** `docs/inference_reports/session_YYYYMMDD_HHMMSS.json`
- **Web display:** `docs/v5/robot_tests.html`

## How to Check

### 1. 최근 세션 목록 확인
```bash
ls -lt docs/inference_sessions/*.h5 | head -5
ls -lt docs/inference_reports/*.json | head -5
```

### 2. 세션 내용 확인 (Python)
```python
import json, h5py

# JSON 확인
with open("docs/inference_reports/session_YYYYMMDD_HHMMSS.json") as f:
    d = json.load(f)
print(d["model_name"], d["instruction"])
print(f"steps={len(d['history'])}, summary={d.get('summary')}")

# H5 확인
with h5py.File("docs/inference_sessions/session_YYYYMMDD_HHMMSS.h5", "r") as f:
    print(f["observations/images"].shape)  # (N, H, W, 3)
    print(f["actions"].shape)              # (N, 3)
```

### 3. 짝 맞는지 확인
모든 `.h5`에 대응하는 `.json`이 있어야 함. 없으면 `end_session()`이 불린 JSON만 있고 H5는 이미지가 없는 경우.

## Known Save Triggers

| 종료 방식 | 저장 여부 | 코드 위치 |
|-----------|-----------|-----------|
| goal_near 자동 종료 | ✅ | `update_ui()` line ~1079 |
| ⏹️ STOP 버튼 | ✅ (수정 후) | `btn_stop_inf.click` → `set_running(False)` → `_flush_session()` |
| 수동 set_running(False) | ✅ (수정 후) | `set_running()` 내 `_flush_session("manual_stop")` |

## Session Save Bug History
- **Before 2026-06-17**: `btn_stop_inf.click`이 `state["is_running"] = False`만 하고 `end_session()` 미호출 → 수동 stop 세션 전부 유실
- **After 2026-06-17**: `_flush_session()` 헬퍼 추가, `set_running(False)` 경로 통일

## Checking Whether a Session Has Images

H5 파일이 작으면 (< 2MB) 이미지가 거의 없는 것 (step 1만 있는 빈 세션):
```bash
ls -lh docs/inference_sessions/*.h5
```
1-2MB → 1프레임 (empty run)
20-150MB → 정상 세션 (15~100 프레임)

## Adding Sessions to robot_tests.html
세션을 `docs/v5/robot_tests.html`에 추가하려면:
1. H5에서 이미지 추출 → `docs/v5/robot_sessions/session_YYYYMMDD_HHMMSS_images/`
2. JSON에서 action history 추출 → chip 타임라인 HTML 생성
3. 두 브랜치에 push: `monavla-driving` + `inference-integration`

참고: `docs/inference_reports/` 와 `docs/inference_reports/*_images/` 는 gitignore됨.
`docs/v5/robot_sessions/`는 gitignore 제외 → 이미지는 반드시 이 경로에 저장.

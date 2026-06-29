# Plan — exp66/stage2_v2 프로필 등록 + "모델 프로모션" 스킬화

> 작성: 2026-06-22 · 상태: **검토 대기**
> 동기: `vla-inference-gradio`(=`mona-inference-gradio`) alias가 "옵션 동기화" 런처로
> 기대됐는데, 실제로는 `vla_profile.py` 레지스트리에 exp66/stage2_v2가 전혀 등록 안 돼
> 있어서 기본값이 exp49(`proxy_default`)로 빠지는 게 확인됨. 사용자 요청: 이런 "새 모델을
> 운영에 올릴 때" 흐름을 형식 정해서 스킬로 규격화하고, 양쪽 서버(soda/minum)에 동일하게
> 적용되도록.

---

## 0. 리서치 결과 — 격차의 정확한 위치

| 구성요소 | 현재 상태 |
|---|---|
| `configs/model_registry.json` | `models`에 exp11/17/10/14/46/47/49/local_v4만 있음. **exp66 없음.** `profiles`도 exp66을 가리키는 항목 없음(`proxy_default`가 exp49). |
| `scripts/manage_api_server.sh` | `SERVER_SCRIPT`를 `api_server` 또는 `proxy_server` 런타임으로만 분기(`E2E_SERVER_SCRIPT` / `PROXY_SERVER_SCRIPT`). **`stage2_v2_inference_server.py`를 launch하는 분기가 없음.** `mlp_step2` 런타임은 명시적으로 "not served"로 막혀 있음. |
| `scripts/vla_profile.py` `doctor()` | `mlp_step2`에 대해 "Dedicated decomposition runtime: NOT IMPLEMENTED IN LAUNCHER" 경고를 이미 출력 중 — 이 gap을 스크립트 자신도 알고 있었음. |
| `.vla_aliases`의 `vla-inference-gradio()` | `export VLA_DASHBOARD_BACKEND="API Server"`까지는 맞게 하지만, profile 기본값이 위 레지스트리에 의존하므로 결국 exp49로 흐름. |

→ exp66을 "진짜 기본값"으로 만들려면 **레지스트리(JSON) + 런처(bash) 양쪽**에 새 런타임 종류(`stage2_v2`)를 추가해야 함. 코드 변경 범위가 있어 plan으로 먼저 정리.

---

## 1. 변경 — exp66/stage2_v2를 vla_profile 시스템에 등록

### 1-1. `configs/model_registry.json`
- `models`에 `exp66_stage2v2` 추가:
  ```json
  "exp66_stage2v2": {
    "label": "Exp66 Stage2 v2 SOTA (base PG2, L2)",
    "kind": "decomposition",
    "runtime": "stage2_v2",
    "status": "validated",
    "checkpoint": "runs/v5_nav/mlp/exp66/action_mlp.pt",
    "config": "",
    "evidence": ["runs/v5_nav/mlp/shared/stage1_v2_projs.pt", "runs/v5_nav/mlp/exp66/action_mlp.pt"]
  }
  ```
- `profiles`에 `stage2v2_default` 추가(`default_model: exp66_stage2v2`, `allow_runtimes: ["stage2_v2"]`)
- **`proxy_default`는 그대로 둠** — exp49도 여전히 유효한 프로필이라 기존 동작 깨지 않음. 다만 `vla-inference-gradio`를 인자 없이 호출했을 때 기본값이 헷갈리지 않도록 ②에서 안내.

### 1-2. `scripts/vla_profile.py`
- `shell_exports()`에 `runtime == "stage2_v2"` 분기 추가:
  ```python
  if resolved["runtime"] == "stage2_v2":
      lines.append('export VLA_SERVER_SCRIPT="robovlm_nav/serve/stage2_v2_inference_server.py"')
      lines.append(f'export VLA_S2V2_STAGE1="runs/v5_nav/mlp/shared/stage1_v2_projs.pt"')
  ```
- `resolve_profile()`에서 `runtime == "mlp_step2"`처럼 launchable=False로 막아둔 특수 분기와 별개로, `stage2_v2`는 정상적으로 `launchable=True` 경로를 타도록(이미 일반 분기로 충분 — `mlp_step2`만 특별 처리돼 있었음)

### 1-3. `scripts/manage_api_server.sh`
- `STAGE2V2_SERVER_SCRIPT="$PROJECT_DIR/robovlm_nav/serve/stage2_v2_inference_server.py"` 추가
- `resolve_profile_env()`에 분기 추가:
  ```bash
  elif [ "${VLA_MODEL_RUNTIME:-}" = "stage2_v2" ]; then
      SERVER_SCRIPT="$STAGE2V2_SERVER_SCRIPT"
  ```
- 기본 포트가 `VLA_PORT:-8000`인데 exp66은 8001 운영 중 — `stage2_v2` 런타임일 때 `PORT="${VLA_PORT:-8001}"`로 기본값 분기 필요(기존 운영 포트와 일치시켜야 함)

### 1-4. `.vla_aliases`
- 변경 없음 — 이미 `VLA_DASHBOARD_BACKEND="API Server"` export하고 있어서, 위 1-1~1-3만 추가되면 `vla-inference-gradio stage2v2_default`가 정확히 exp66/8001을 가리키게 됨.

---

## 2. 신규 스킬 — `.agent/skills/promote-model-profile/SKILL.md`

기존 스킬 포맷(`grounding-session-pipeline` 등 frontmatter `name`/`description` + 번호 절차)을 그대로 따름.

### 내용 초안
```markdown
---
name: promote-model-profile
description: 새 실험(체크포인트)을 운영 기본 프로필로 승격할 때 vla_profile.py 레지스트리 +
  manage_api_server.sh 런처 + 양쪽 서버(soda/minum) git 동기화까지 표준 절차로 처리.
  "이 모델을 운영에 올려줘", "새 프로필 등록", "vla-inference-gradio 동기화 깨짐" 같은
  요청에 사용.
---

# Promote Model Profile

## 0. 트리거
- 새 실험(exp**)이 "현재 최선" 모델로 확정됐을 때
- `vla-inference-gradio`/`vla-doctor` 결과가 기대한 모델과 다를 때

## 1. 런타임 종류 확인
`scripts/manage_api_server.sh`가 아는 런타임: api_server / proxy_server / (이번에 추가될) stage2_v2.
새 실험이 이 중 어디에도 안 맞으면(새 추론 서버 스크립트라면) §4 먼저.

## 2. model_registry.json에 모델 추가
configs/model_registry.json의 "models"에 항목 추가 — label/kind/runtime/status/checkpoint/config/evidence.
체크포인트 경로는 PROJECT_ROOT 기준 상대경로로.

## 3. 프로필 추가/갱신
"profiles"에 default_model이 새 모델을 가리키는 프로필 추가 (또는 기존 프로필 갱신).
allow_runtimes에 해당 런타임 포함 확인.

## 4. (새 런타임이면) 런처 확장
manage_api_server.sh의 resolve_profile_env()에 SERVER_SCRIPT 분기 추가.
vla_profile.py의 shell_exports()에도 필요한 env var 분기 추가.

## 5. 검증
  vla-doctor --profile <새 프로필>      # CKPT/Config OK, Launchable: yes 확인
  vla-inference-gradio <새 프로필>      # 실제 대시보드가 올바른 포트/모델로 뜨는지 확인
  curl <api_url>/health, /model/info    # 모델 체크포인트/val_acc 일치 확인

## 6. 양쪽 서버 동기화 — git이 sync 메커니즘
이 레포에서 "soda/minum 양쪽에 저장"은 **rsync가 아니라 git push(양 브랜치)**로 한다
(grounding-session-pipeline 스킬과 동일 원칙). model_registry.json / manage_api_server.sh /
vla_profile.py / .vla_aliases / SKILL.md 전부 git 추적 대상이므로:
  git add configs/model_registry.json scripts/manage_api_server.sh scripts/vla_profile.py
  git commit -m "..."
  git push origin monavla-driving
  git checkout inference-integration && git cherry-pick <commit> && git push origin inference-integration
체크포인트 파일 자체(.pt)는 git에 안 올리므로 sync-ckpt-to-minum 스킬로 별도 전송.
```

---

## 3. 변경 파일 정리

| 파일 | 작업 |
|---|---|
| `configs/model_registry.json` | exp66_stage2v2 모델 + stage2v2_default 프로필 추가 |
| `scripts/vla_profile.py` | `stage2_v2` 런타임 shell_exports 분기 추가 |
| `scripts/manage_api_server.sh` | `stage2_v2` 런타임 SERVER_SCRIPT/PORT 분기 추가 |
| `.agent/skills/promote-model-profile/SKILL.md` (신규) | 위 초안 그대로 |

## 4. 트레이드오프
- 기존 `proxy_default`/exp49 동작은 전혀 안 건드림 — 새 런타임 분기 추가만이라 회귀 위험 낮음
- `manage_api_server.sh`로 exp66을 직접 start/stop하게 만들면, 지금처럼 수동 `nohup ... VLA_S2V2_STAGE2=...`으로 띄운 프로세스와 관리 방식이 이원화될 수 있음 — 다음부터는 `vla-server start stage2v2_default` 한 줄로 통일하는 게 목표
- 포트 기본값 분기(8000 vs 8001)를 놓치면 launcher가 엉뚱한 포트에서 헬스체크해서 "안 떠 있다"고 오판하고 중복 기동 시도할 수 있음 — §1-3에서 명시적으로 처리

## 5. 완료 기준
- [ ] `configs/model_registry.json`에 exp66 등록
- [ ] `vla_profile.py`/`manage_api_server.sh`에 `stage2_v2` 런타임 분기 추가
- [ ] `vla-doctor --profile stage2v2_default` → Launchable: yes 확인
- [ ] `.agent/skills/promote-model-profile/SKILL.md` 작성
- [ ] git commit + push (monavla-driving, inference-integration 양 브랜치) — 이게 "양쪽 서버 저장" 규칙의 실체
- [ ] minum에서 `git pull` 후 동일하게 `vla-doctor --profile stage2v2_default` 통과하는지 확인(가능하면)

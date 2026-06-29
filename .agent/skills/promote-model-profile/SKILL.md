---
name: promote-model-profile
description: 새 실험(체크포인트)을 운영 기본 프로필로 승격할 때 vla_profile.py 레지스트리 + manage_api_server.sh 런처 + 양쪽 서버(soda/minum) git 동기화까지 표준 절차로 처리. "이 모델을 운영에 올려줘", "새 프로필 등록", "vla-inference-gradio 동기화 깨짐" 같은 요청에 사용.
---

# Promote Model Profile

새 실험이 "현재 최선" 모델로 확정되면, `vla-inference-gradio`/`vla-server`/`vla-doctor`
같은 alias가 자동으로 그 모델을 가리키도록 레지스트리·런처를 같이 갱신하고, git push로
soda/minum 양쪽에 동일하게 반영하는 절차.

## 0. 트리거

- 새 실험(exp**)이 "현재 최선" 모델로 확정됐을 때
- `vla-doctor`/`vla-inference-gradio` 결과가 기대한 모델과 다를 때 ("동기화가 깨졌다"는 느낌이 들 때)

## 1. 런타임 종류 확인

`scripts/manage_api_server.sh`가 현재 아는 런타임:

| runtime 값 | 서버 스크립트 |
|---|---|
| `api_server` | `robovlm_nav/serve/inference_server.py` |
| `proxy_server` | `robovlm_nav/serve/proxy_inference_server.py` |
| `stage2_v2` | `robovlm_nav/serve/stage2_v2_inference_server.py` |
| `mlp_step2` | (런처 미구현 — `vla_profile.py doctor`가 경고 출력) |

새 실험이 위 4종 중 어디에도 안 맞으면(완전히 새로운 추론 서버 스크립트라면) §4 먼저 진행.

## 2. `configs/model_registry.json`에 모델 추가

`"models"`에 항목 추가:

```json
"exp{N}_{name}": {
  "label": "사람이 읽을 이름",
  "kind": "decomposition|end_to_end|grounding",
  "runtime": "stage2_v2",
  "checkpoint": "runs/v5_nav/mlp/exp{N}/action_mlp.pt",
  "config": "",
  "status": "validated",
  "pinned": true,
  "evidence": ["근거 문서/파일 경로들"],
  "notes": "핵심 지표 한 줄 (val_acc, CL, FPE 등)"
}
```

- 체크포인트/config 경로는 `PROJECT_ROOT` 기준 **상대경로**로 적는다(`expand_path()`가 자동 절대화).
- `stage2_v2`/`proxy_server` runtime은 `config`가 없어도 된다(`resolve_profile()`에서 면제됨).

## 3. 프로필 추가/갱신

`"profiles"`에 새 모델을 가리키는 프로필 추가(또는 기존 프로필의 `default_model` 갱신):

```json
"{runtime}_default": {
  "label": "...",
  "default_model": "exp{N}_{name}",
  "fallback_models": [],
  "allow_runtimes": ["stage2_v2"]
}
```

기존 프로필(`proxy_default` 등)은 그대로 둔다 — 다른 실험을 가리키는 프로필을 지우지 않는다.

## 4. (새 런타임이면) 런처 확장

`scripts/vla_profile.py`:
- `resolve_profile()`의 `config_required = runtime not in (...)` 튜플에 새 runtime 추가(config 불필요한 경우)
- `shell_exports()`에 `if resolved["runtime"] == "...":` 분기 추가 — `VLA_SERVER_SCRIPT`, 필요한 모델별 env var export

`scripts/manage_api_server.sh`:
- 새 `*_SERVER_SCRIPT` 변수 선언
- `resolve_profile_env()`의 `elif` 체인에 분기 추가 — `SERVER_SCRIPT`와 (필요시) `PORT` 기본값 설정

## 5. 검증

```bash
python3 scripts/vla_profile.py doctor --profile <새 프로필>   # Launchable: yes 확인
python3 scripts/vla_profile.py env --profile <새 프로필>      # export 값 확인
vla-inference-gradio <새 프로필>                               # 대시보드가 올바른 포트/모델로 뜨는지
curl <api_url>/health
curl <api_url>/model/info                                      # val_acc 등 기대 체크포인트 일치 확인
```

## 6. 양쪽 서버 동기화 — git이 sync 메커니즘

이 프로젝트에서 "soda/minum 양쪽에 저장"은 **rsync가 아니라 git push(양 브랜치)**로 한다
([[grounding-session-pipeline]]과 동일 원칙). `model_registry.json`/`manage_api_server.sh`/
`vla_profile.py`/`.vla_aliases`/이 SKILL.md 전부 git 추적 대상이므로:

```bash
git add configs/model_registry.json scripts/manage_api_server.sh scripts/vla_profile.py \
        .agent/skills/promote-model-profile/SKILL.md
git commit -m "feat(profile): exp{N} 운영 프로필 등록"
git push origin monavla-driving

git checkout inference-integration
git pull origin inference-integration
git cherry-pick <commit-hash>
git push origin inference-integration
git checkout monavla-driving
```

체크포인트 파일 자체(`.pt`)는 git에 안 올라가므로 별도로 [[sync-ckpt-to-minum]] 스킬로 전송한다.
minum 쪽에서는 `git pull` 후 동일하게 `vla_profile.py doctor --profile <프로필>`이 통과하는지 확인.

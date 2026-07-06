---
name: sync-inference-session
description: 방금 수집한 추론 세션(docs/inference_sessions/*.h5) + 서버 설정 스냅샷(/health) + 활성 모델 ckpt + episode_log를 한 번에 minum 서버로 전송한다. "방금 세션 서버로 보내줘", "이번 주행 minum에 전송", "세션이랑 설정 다 보내줘" 같은 요청에 사용.
---

# Sync Inference Session to Minum

실로봇 추론 세션 하나(또는 여럿)를 **당시 시스템 설정 + 모델 정보와 함께** minum 서버로
전송해 분석을 넘기는 스킬. 2026-07-02 대화에서 수동으로 하던 절차를 코드화.

## 언제 쓰나
- "방금 수집한 세션 서버로 보내자", "이번 주행 minum으로", "세션+설정 다 전송" 등.
- grounding jsonl/mp4가 아니라 **추론 세션 H5**(docs/inference_sessions/)가 대상일 때.
  (jsonl/mp4 그라운딩 검증은 `grounding-session-pipeline`, 체크포인트만은 `sync-ckpt-to-minum`.)

## 실제 스크립트
```
scripts/sync/push_inference_session_to_minum.sh
```

## 사용법
```bash
bash scripts/sync/push_inference_session_to_minum.sh              # 가장 최근 세션 1개 (기본)
bash scripts/sync/push_inference_session_to_minum.sh -n 5         # 최근 5개
bash scripts/sync/push_inference_session_to_minum.sh --last-24h   # 최근 24시간
bash scripts/sync/push_inference_session_to_minum.sh --all        # 전체
bash scripts/sync/push_inference_session_to_minum.sh 20260702_100143  # 특정 세션 ID
```

## 무엇을 보내나 (한 세트)
1. **세션 H5** — 선택된 `docs/inference_sessions/session_*.h5`
   - `attrs.runtime_config`(JSON 문자열)에 **그 세션 수집 시점**의 그라운더/설정
     스냅샷이 세션마다 개별 저장돼 있음(2026-07-02+, 2026-07-06부터 grounder_model/
     owlv2_thresh/checkpoint_path/git_commit 포함) — 여러 세션이 다른 설정으로
     수집됐어도(예: PG2 vs OWL-v2 A/B) 세션별로 정확히 구분 가능
2. **서버 설정 스냅샷** — `curl /health` → `server_health.json`
   (head/window/val_acc/checkpoint · preview · grounder(PG2-448 또는 OWL-v2, A/B) ·
   owlv2_thresh · skip_n · multi_prompt · cx_jump
   + Fix3 버전 핸드셰이크: git_commit / process_started_at / code_mtime)
   ⚠️ 이건 **전송 시점의 현재 설정**만 찍음 — 세션 수집 당시와 다를 수 있으니
   각 세션의 실제 설정은 1번의 runtime_config를 봐야 함(매니페스트에 자동 포함됨)
3. **활성 모델 ckpt** — `/health`의 checkpoint_path를 자동 추출해 `models/`로
4. **episode_log.csv** — 실주행 에피소드 기록
5. **grounding_decisions_YYYYMMDD.jsonl** — **필수 동봉 (Fix4)**: 전송 세션 날짜의
   그라운딩 판정 영구 로그 — PG2는 filter_reason(no-locs/tiny/top/full-frame/x-full/
   None) + 필터 전 raw locs, OWL-v2는 `"model":"owlv2"` 태그로 구분(no-locs=미검출).
   호출 1회당 latency 포함. minum receive-inference-session이 H5 LIVE 프레임과
   ts로 매칭해 원인 분석에 사용.
6. **README.txt** — 전송 시점 서버 설정 + **세션별 실제 수집 설정(runtime_config)** +
   H5 구조 매니페스트

## 전송 위치
```
minum:~/MoNaVLA/inference_sessions_recv/<YYYYMMDD>/
   ├── session_*.h5
   ├── server_health.json
   ├── grounding_decisions_<YYYYMMDD>.jsonl
   ├── README.txt
   ├── episode_log.csv
   └── models/<ckpt>.pt
```

## 환경변수
| 변수 | 기본 | 용도 |
|------|------|------|
| `MINUM_HOST` | `minum` | ssh 대상 (ssh config alias) |
| `API` | `http://localhost:8001` | 설정 스냅샷 뽑을 추론 서버 |
| `VLA_API_KEY` | `vla_devel_key_2026` | /health 인증 |

## 주의
- `server_health.json`은 **전송 시점의 서버 상태**만 찍는다 — 세션 수집 당시와
  서버 설정(그라운더 A/B, preview, skip_n 등)이 바뀌었을 수 있으니, 최종 근거는
  항상 각 세션의 `attrs.runtime_config`를 봐야 함(매니페스트가 자동으로 보여줌).
  가능하면 **수집 직후** 전송하는 게 여전히 안전.
- 2026-07-06 이전 세션은 `runtime_config`에 그라운더 필드가 없음(구버전) —
  이 경우 서버 프로세스 재시작 이력(`ps -ef`로 기동 시각 확인)으로 그 시점
  활성 그라운더를 역추정해야 함. 근거 없이 추정하지 말 것.
- 서버가 내려가 있으면 설정 스냅샷은 `{"error":"server offline"}`으로 남고 세션만 전송된다.
- 전송 후 확인: `ssh minum 'ls -lh ~/MoNaVLA/inference_sessions_recv/<date>/'`

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
2. **서버 설정 스냅샷** — `curl /health` → `server_health.json`
   (head/window/val_acc/checkpoint · preview · grounder(PG2-448) · skip_n · multi_prompt · cx_jump
   + Fix3 버전 핸드셰이크: git_commit / process_started_at / code_mtime)
3. **활성 모델 ckpt** — `/health`의 checkpoint_path를 자동 추출해 `models/`로
4. **episode_log.csv** — 실주행 에피소드 기록
5. **grounding_decisions_YYYYMMDD.jsonl** — **필수 동봉 (Fix4)**: 전송 세션 날짜의
   PG2 판정 영구 로그 (filter_reason: no-locs/tiny/top/full-frame/x-full/None,
   필터 전 raw locs, 호출 1회당 latency). minum receive-inference-session이 H5
   LIVE 프레임과 ts로 매칭해 flicker 원인을 확정하는 데 사용.
6. **README.txt** — 위 설정 요약 + H5 구조 매니페스트

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
- 설정 스냅샷은 **전송 시점의 서버 상태**를 찍는다. 세션 수집 당시와 서버 설정이 바뀌었으면
  런타임 토글(preview/skip_n/multi_prompt 등)이 달라졌을 수 있으니, 가능하면 **수집 직후** 전송.
- 서버가 내려가 있으면 설정 스냅샷은 `{"error":"server offline"}`으로 남고 세션만 전송된다.
- 전송 후 확인: `ssh minum 'ls -lh ~/MoNaVLA/inference_sessions_recv/<date>/'`

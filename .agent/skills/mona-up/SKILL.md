---
name: mona-up
description: MoNaVLA 추론 서버(8001) + FastAPI 대시보드(7800, 신규 권장) / Gradio 대시보드(7865, 구버전) 시작/중지/상태 확인. "서버 올려줘", "대시보드 띄워줘", "mona 상태 확인", "서버 내려줘" 등의 요청에 사용.
---

# mona-up 스킬

## 역할
MoNaVLA 실로봇 주행 파이프라인을 시작/중지/확인한다.
모든 서비스 제어는 `scripts/run/go.sh` 단일 진입점을 통해 수행한다.

## 서비스 구성

| 서비스 | 포트 | 역할 |
|---|---|---|
| 추론 서버 (stage2_v2) | 8001 | PG2 그라운딩 + VLA MLP 추론 |
| **FastAPI 대시보드** (`mona_dashboard.py`) | **7800** | **★ 신규 권장 UI** — ROS 노드 자체 소유, 주행/그라운딩/드리프트/캘리브레이션/세션 히스토리 전 탭 이식 완료 (`docs/dashboard_architecture_comparison.md` 참조) |
| Gradio 대시보드 (`gradio_inference_dashboard.py`) | 7865 | 구버전 — 유지보수 중단 예정, 참고용으로만 |

접속 IP: `100.85.118.58` (Tailscale)

## 기본 명령

```bash
# 전체 시작 (서버 + FastAPI 대시보드 + 허브 + 뷰어; Gradio는 --all에 포함 안 됨)
bash scripts/run/go.sh --all

# 추론 서버만
bash scripts/run/go.sh --server

# FastAPI 대시보드만 (★ 신규 권장)
bash scripts/run/go.sh --mona-dash

# Gradio 대시보드 (구버전, 명시적 지정 시만)
bash scripts/run/go.sh --dashboard

# 상태 확인
bash scripts/run/go.sh --status
curl -s http://localhost:8001/health | python3 -m json.tool
curl -s http://localhost:7800/health | python3 -m json.tool

# 전체 중지 (서버 + FastAPI 대시보드 + Gradio 대시보드 + 허브 + 뷰어 모두)
bash scripts/run/go.sh --stop

# 로그 실시간 확인
tail -f logs/s2v2_server.log | grep -E "PG2|CH54|GND|ACTION"
tail -f logs/mona_dashboard.log
```

## 환경변수 (go.sh 기본값)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `VLA_PREVIEW_ENABLED` | `1` | CH54 preview 루프 활성 |
| `VLA_PREVIEW_HINT_CX` | `1` | PG2 필터 결과의 cx 방향 힌트 활용 |
| `VLA_PREVIEW_MAX_RETRY` | `5` | 최대 ROT 재시도 횟수 |
| `VLA_PREVIEW_ROT_DIR` | `R` | 기본 탐색 회전 방향 |
| `VLA_STOP_MODE` | `learned` | STOP 결정 방식 |

## 주의사항

- **PG2 OOM 방지**: 서버 실행 중에 PG2를 직접 로드하지 않는다 (Jetson OOM). 테스트 전 반드시 서버 내리기.
- `.venv` 없음: `python3` = `/usr/bin/python3` 사용
- 모델 로딩 시간: PG2 3B 로딩에 2~3분 소요 (health 응답 대기)
- `go.sh --server`로 시작하면 PG2 워밍업까지 자동 수행

## 워크플로우

### 1. 서버 상태 확인
```bash
bash scripts/run/go.sh --status
```

### 2. 서버가 꺼져있으면 시작
```bash
bash scripts/run/go.sh --server
# health 응답 올 때까지 대기 (최대 3분)
```

### 3. 대시보드 시작
```bash
nohup python3 scripts/gradio_inference_dashboard.py > logs/inference_dashboard.log 2>&1 &
```

### 4. 접속
- 대시보드: http://100.85.118.58:7865
- 추론 서버 health: http://100.85.118.58:8001/health

### 5. 종료
```bash
bash scripts/run/go.sh --stop
pkill -f gradio_inference_dashboard
```

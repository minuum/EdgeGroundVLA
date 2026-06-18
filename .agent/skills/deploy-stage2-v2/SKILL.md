---
name: deploy-stage2-v2
description: Stage2 v2 추론 서버 코드를 soda@100.85.118.58:~/MoNaVLA 로 rsync 전송하고 서버 재시작까지 자동 처리. "soda에 배포해줘", "go.sh 재시작", "서버 업데이트" 등의 요청에 사용.
---

# Deploy Stage2 v2 to soda

minum → soda 직접 rsync 가능 (Tailscale, 100.85.118.58).  
모델 체크포인트는 `rsync_stage2_v2.sh` 로, 코드 파일은 직접 rsync.

## SSH 접근

```bash
ssh soda@100.85.118.58 'echo ok'   # 연결 확인
```

## 코드만 배포 (서버 재시작 포함)

```bash
cd ~/26CS/MoNaVLA

# 1. 코드 전송
rsync -avz --relative \
  robovlm_nav/serve/stage2_v2_inference_server.py \
  scripts/gradio_inference_dashboard.py \
  scripts/run/go.sh \
  soda@100.85.118.58:~/MoNaVLA/

# 2. 재시작
ssh soda@100.85.118.58 'cd ~/MoNaVLA && bash scripts/run/go.sh --stop && nohup bash scripts/run/go.sh > /tmp/go_startup.log 2>&1 &'

# 3. 서버 준비 대기
ssh soda@100.85.118.58 'until curl -sf http://localhost:8001/health > /dev/null; do sleep 3; done && curl -s http://localhost:8001/health'
```

## 모델 체크포인트 포함 전체 배포

```bash
bash scripts/deploy/rsync_stage2_v2.sh        # best 모델만
bash scripts/deploy/rsync_stage2_v2.sh --all  # ablation ckpt 전부
```

## 상태 확인

```bash
ssh soda@100.85.118.58 'cd ~/MoNaVLA && bash scripts/run/go.sh --status'
ssh soda@100.85.118.58 'curl -s http://localhost:8001/health'
ssh soda@100.85.118.58 'tail -20 ~/MoNaVLA/logs/s2v2_server.log'
```

## 전송 코드 파일 목록

| 파일 | 용도 |
|---|---|
| `robovlm_nav/serve/stage2_v2_inference_server.py` | FastAPI 추론 서버 (PG2 grounding + MLP) |
| `scripts/gradio_inference_dashboard.py` | 로봇 제어 대시보드 (async 2-thread) |
| `scripts/run/go.sh` | 서버+대시보드 통합 시작 스크립트 |

## 헬스 응답 예시

```json
{"status":"healthy","model_loaded":true,"head":"mlp","window":8,"gpu":{"allocated_gb":0.609,"device_name":"Orin"}}
```

## 에이전트 사용 예시

"soda 서버 재시작해줘", "배포해줘", "코드 업데이트 반영해줘":
→ 위 "코드만 배포" 절차 실행

"체크포인트까지 새로 올려줘":
→ `rsync_stage2_v2.sh` 실행 후 재시작

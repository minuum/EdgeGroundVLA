# STOP Mode 배포 가이드

> 작성: 2026-06-18 · 브랜치: inference-integration

## 배경

기존 `proximity` STOP은 `area ≥ 0.25 AND |cx-0.5| ≤ 0.35 AND consec=2` 조건으로 강제 override.  
오프라인 ablation(L1)에서 40%의 에피소드가 **첫 프레임부터 area ≥ 0.25** → TLD ≈ 0.33 → 실패.

**`learned` STOP**: 학습 시 마지막 프레임을 STOP(gt_class=0)으로 합성 주입한 모델이  
바스켓이 충분히 클 때 자연스럽게 STOP을 예측 → latch로 유지.

### 실험 결과 (stop_aware 메트릭, val N=29)

| 모델 | CL ↑ | Good STOP (last 5f) ↑ | Premature STOP ↓ | Never STOP ↓ |
|------|------|----------------------|-----------------|-------------|
| Exp66 (no STOP) | 79.3%* | 0% | 0% | 100% |
| sw1x (last 1f, wt×1) | **96.6%** | **86.2%** | **0%** | 13.8% |
| sw5x (last 1f, wt×5) | **96.6%** | 79.3% | **0%** | 20.7% |

*Exp66는 bbox_dataset_full(150ep) val split 기준. README의 96.6%는 다른 split.

**sw1x 선택 이유**: CL 유지 + Good STOP 최고 + Premature 0%.

---

## 체크포인트

| 파일 | 크기 | 설명 |
|------|------|------|
| `runs/v5_nav/mlp/stop_weighted/stop_wt_sw1x.pt` | 456KB | learned STOP 모델 |
| `runs/v5_nav/mlp/exp66/action_mlp.pt` | 456KB | 기존 best (no STOP) |
| `runs/v5_nav/mlp/shared/stage1_v2_projs.pt` | 3.1MB | Stage1 encoder (공유) |

---

## 실행 방법

### 기존 proximity 모드 (변경 없음)
```bash
VLA_S2V2_STAGE2=runs/v5_nav/mlp/exp66/action_mlp.pt \
  .venv/bin/python3 robovlm_nav/serve/stage2_v2_inference_server.py --port 8001
```

### learned STOP 모드
```bash
VLA_STOP_MODE=learned \
VLA_S2V2_STAGE2=runs/v5_nav/mlp/stop_weighted/stop_wt_sw1x.pt \
  .venv/bin/python3 robovlm_nav/serve/stage2_v2_inference_server.py --port 8001
```

또는 `go.sh`에 환경변수 추가:
```bash
export VLA_STOP_MODE=learned
export VLA_S2V2_STAGE2=runs/v5_nav/mlp/stop_weighted/stop_wt_sw1x.pt
```

---

## 동작 원리

```
predict() 호출
    ↓
[STOP_MODE == "learned"]?
    YES → stop_latched?
              YES  → STOP 반환 (latch 유지)
              NO   → 모델 추론 → pred == STOP?
                         YES → stop_latched = True, STOP 반환 [LEARNED STOP]
                         NO  → 정상 액션 반환
    NO  → proximity 체크 (기존 로직)
              area ≥ 0.25 AND |cx-0.5| ≤ 0.35 AND consec=2
              → 조건 만족 시 STOP override [PROXIMITY STOP]
```

**Latch**: 한 번 STOP이 발동되면 `/reset` 호출 전까지 모든 추론이 STOP을 반환.  
응답의 `stop_latched: true` 필드로 상태 확인 가능.

---

## API

### STOP latch 해제 (에피소드 리셋)
```bash
curl -X POST http://localhost:8001/reset
# → {"status": "success", "message": "History reset"}
# stop_latched = False로 초기화됨
```

### 런타임 모드 전환 (재시작 없이)
```bash
# learned → proximity 전환
curl -X POST http://localhost:8001/config \
  -H "Content-Type: application/json" \
  -d '{"stop_mode": "proximity"}'

# latch 수동 해제 (reset 없이)
curl -X POST http://localhost:8001/config \
  -H "Content-Type: application/json" \
  -d '{"stop_latched": false}'
```

### 응답에서 STOP 상태 확인
```json
{
  "predicted_class": 0,
  "predicted_label": "STOP",
  "stop_mode": "learned",
  "learned_stop": true,
  "stop_latched": true,
  ...
}
```

---

## 파일 전송 (soda 서버)

```bash
# sw1x 체크포인트 + 서버 코드 + 가이드 한 번에
bash scripts/deploy/rsync_stage2_v2.sh

# 전송 대상:
#   runs/v5_nav/mlp/stop_weighted/stop_wt_sw1x.pt  (456KB)
#   robovlm_nav/serve/stage2_v2_inference_server.py
#   docs/v5/STOP_MODE_GUIDE.md
```

---

## ROS 노드 연동

`vla_inference_node.py`는 `/predict` 응답의 `action` 필드를 그대로 사용.  
STOP(class 0) = `action: [0.0, 0.0]` → 로봇 정지.  
응답의 `stop_latched` 필드를 ROS topic으로 퍼블리시하면 상위 FSM에서 활용 가능.

```python
# vla_inference_node.py에서 추가 가능
if result.get("stop_latched"):
    self.get_logger().info("STOP latched — awaiting reset")
    self.publish_stop_event()   # 필요 시
```

에피소드 종료 후 새 목표 시작 전 반드시:
```python
requests.post("http://localhost:8001/reset")
```

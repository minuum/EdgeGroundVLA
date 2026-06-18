# Plan: 비동기 추론 루프 (Async Inference Loop)

**작성일**: 2026-06-18  
**브랜치**: inference-integration  
**상태**: 🔍 구현 전 확인 필요

---

## 문제 핵심: Train-Deploy 갭

| | 수집 | 배포 (현재) |
|---|---|---|
| 이동 방식 | 조이스틱 **누른 채 연속 이동** 중 캡처 (`PRE_CACHE`) | `stop → move 0.4s → stop` 반복 (bang-bang) |
| 이미지 타이밍 | 이동 중 live frame | 이동 완료 후 settle 150ms 뒤 stable_frame |
| 사이클 | ~10Hz 연속 | ~0.85s (infer 350ms + move 400ms + stop) |

**결과**: 모델이 학습에서 본 적 없는 "정지 직후 정착 프레임" 으로 추론 → 분포 불일치 → 방향 오차 누적.

---

## 제안 아키텍처: 비동기 2-스레드

```
기존 (동기):
  infer(350ms) → stop → move(400ms) → stop → infer ...
  사이클 ~0.85s, 이미지는 정지 후 캡처

제안 (비동기):
  ┌─ inference_thread (3Hz):  /predict → action_queue
  └─ execution_thread (10Hz): queue.get() → cmd_vel 발행 (없으면 last 유지)
```

### 핵심 변화

- **이미지 타이밍**: 이동 중 10Hz live frame → `/predict` 호출 (수집과 동일)
- **이동**: `cmd_vel` 연속 발행, stop 없음
- **추론 결과 반영**: 3Hz 결과가 queue에 들어오면 execution_thread가 즉시 반영
- **안전**: queue empty → 직전 액션 유지 (coasting), timeout → stop

---

## 구현 대상 파일

현재 soda 운용 경로는 아래 **확인 필요 #1** 참조.

| 파일 | 역할 | 비고 |
|---|---|---|
| `ROS_action/src/mobile_vla_package/mobile_vla_package/api_client_node.py` | ROS 노드 독립 운용 | 현재 `inference_loop`이 10Hz로 `/predict` 호출 후 즉시 publish (stop 없음) |
| `scripts/gradio_inference_dashboard.py` | Gradio UI + 추론 | `update_ui` 0.5s 타이머 — SYNC/PRE 모드 이미 추가됨 (2026-06-18) |

### api_client_node.py 현재 상태

```python
def inference_loop(self):
    while rclpy.ok():
        if self.inference_mode:
            image = self.get_latest_image_via_service()
            self.run_inference(resized)   # 내부에서 publish_cmd_vel 즉시 호출
        time.sleep(0.1)  # 10Hz — 하지만 /predict latency 350ms면 실제 ~3Hz
```

`api_client_node.py`는 이미 **stop 없이 연속 publish** 구조다.  
문제는 `get_latest_image_via_service()` — 서비스 콜이 블로킹이라 10Hz가 실제로는 ~2-3Hz.

---

## 확인이 필요한 것 (구현 전)

### ✅ 확인 필요 #1: 실제 운용 경로

**질문**: soda에서 실제 추론 시 `api_client_node.py`를 쓰는가, `gradio_inference_dashboard.py`를 쓰는가?

```bash
# soda에서 실행
ps aux | grep -E "api_client|gradio_inference"
cat ~/MoNaVLA/scripts/run/go.sh | grep -A3 "추론"
```

- `api_client_node.py` 사용 중이면 → 거기서 비동기화
- `gradio_inference_dashboard.py` 사용 중이면 → `update_ui`의 SYNC/PRE 모드 개선으로 충분할 수 있음

### ✅ 확인 필요 #2: 수집 시 이동 시간

**질문**: 조이스틱 버튼 한 번 누름 = 이동 시간이 얼마인가?

데이터 수집 코드 기준:
```python
# gradio_data_collector.py line 118
STEP_INTERVAL = 0.45  # 홀딩 시 반복 발사 간격 (s)
```

```python
# teleop_step (버튼 1회) — line 622
self.movement_timer = threading.Timer(0.4, timed_stop)  # 0.4s 후 stop
```

즉 **버튼 탭 1회 = 0.4s 이동**.  
연속 홀딩 모드(async joystick)에서는 `STEP_INTERVAL=0.45s`마다 캡처.

**확인할 것**: V5 에피소드의 실제 프레임 간격.

```bash
# minum에서 실행
python3 - <<'EOF'
import h5py, numpy as np, glob
files = glob.glob("ROS_action/mobile_vla_dataset_v5/*.h5")[:5]
for f in files:
    with h5py.File(f) as h:
        ts = h['timestamps'][:]
        diffs = np.diff(ts)
        print(f"{f.split('/')[-1]}: mean={diffs.mean():.3f}s  min={diffs.min():.3f}s  max={diffs.max():.3f}s  n={len(ts)}")
EOF
```

---

## 구현 계획 (확인 완료 후)

### Option A: `api_client_node.py` 개선 (ROS 독립 운용 경로)

```python
# 현재 구조 문제: get_latest_image_via_service()가 블로킹 → 실질 3Hz
# 개선: 카메라 서비스 콜을 별도 스레드로 분리

class MobileVLAAPIClient(Node):
    def __init__(self):
        self._latest_image = None
        self._image_lock = threading.Lock()
        # 카메라 폴링 스레드 (10Hz)
        threading.Thread(target=self._camera_poll_loop, daemon=True).start()
        # action_queue 추가
        self._action_queue = deque(maxlen=3)
        # execution_thread (10Hz)
        threading.Thread(target=self._execution_loop, daemon=True).start()

    def _camera_poll_loop(self):
        """10Hz로 카메라 서비스 콜 — 블로킹이지만 추론과 분리"""
        while rclpy.ok():
            img = self.get_latest_image_via_service()
            with self._image_lock:
                self._latest_image = img
            time.sleep(0.1)

    def inference_loop(self):
        """3Hz로 /predict → action_queue"""
        while rclpy.ok():
            if self.inference_mode:
                with self._image_lock:
                    img = self._latest_image
                if img is not None:
                    action = self._call_predict(img)   # 블로킹 ~350ms
                    if action:
                        self._action_queue.append(action)
            # sleep 없음 — /predict latency가 자연스럽게 3Hz 만듦

    def _execution_loop(self):
        """10Hz로 queue에서 액션 꺼내 publish"""
        last_action = self.STOP_ACTION
        while rclpy.ok():
            if self._action_queue:
                last_action = self._action_queue.popleft()
            if self.inference_mode:
                self.publish_cmd_vel(last_action, "async_exec")
            time.sleep(0.1)
```

### Option B: `gradio_inference_dashboard.py` PRE 모드 활용

이미 `infer_move_mode = "PRE"` 로 전환하면:
- live frame → 추론 → 이동 (settle 없음)
- bang-bang 제거

단, Gradio 0.5s 타이머 제약 = 2Hz 상한. `api_client_node.py` 대비 느림.

---

## 우선순위

1. **확인 #1, #2 실행** → 운용 경로 + 프레임 간격 파악
2. **Option A** 우선 (더 빠름, ROS 레벨에서 동작)
3. **Option B** 는 Gradio UI에서 시각 확인용으로 병행 유지

---

## 관련 파일

- `ROS_action/src/mobile_vla_package/mobile_vla_package/api_client_node.py` — 361줄
- `scripts/gradio_inference_dashboard.py` — SYNC/PRE 모드 (2026-06-18 추가)
- `scripts/gradio_data_collector.py` — `CaptureMode.PRE_CACHE` / `teleop_step` 0.4s
- `docs/v5/robot_tests.html` — June 18 세션 분석 (FWD+R 편향, STOP 미발동)

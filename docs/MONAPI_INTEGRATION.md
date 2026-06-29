# MoNaVLA ↔ MonAPI 연동 가이드

> **대상:** monapi 레포에서 MoNaVLA 추론 서버를 호출하는 쪽  
> **최종 업데이트:** 2026-06-01  
> **기준 브랜치:** `inference-integration`

---

## 1. 서버 구성 (SODA 기준)

MoNaVLA는 두 서버로 구성됩니다. monapi는 **8001만 사용**하면 됩니다.

| 서버 | 포트 | 역할 | monapi 사용 여부 |
|------|------|------|-----------------|
| `inference_server.py` | **8000** | VLA backbone (Kosmos-2) | ❌ 직접 호출 불필요 |
| `proxy_inference_server.py` | **8001** | GoalNav MLP + Grounding | ✅ **여기만 호출** |

> 8001이 내부적으로 8000과 통신하는 구조. 8000은 노출 불필요.

---

## 2. 인증

모든 POST 요청에 API 키 헤더 필요.

```
X-Api-Key: vla_devel_key_2026
```

> 환경변수 `VLA_API_KEY`로 설정. GET /health는 인증 불필요.

---

## 3. 핵심 엔드포인트

### 3-1. 헬스 체크

```
GET http://soda:8001/health
```

**응답 예시:**
```json
{
  "status": "healthy",
  "active_model": "exp49",
  "model_loaded": true,
  "gpu_memory": { "allocated_gb": 3.1, "device_name": "Orin" }
}
```

---

### 3-2. 추론 (메인)

```
POST http://soda:8001/predict
X-Api-Key: vla_devel_key_2026
Content-Type: application/json
```

**요청:**
```json
{
  "image": "<base64 JPEG string>",
  "instruction": "gray basket",
  "vlm_model": "kosmos"
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `image` | string | **JPEG base64 인코딩** 이미지 |
| `instruction` | string | 목표 물체 이름 (예: `"gray basket"`) |
| `vlm_model` | string | grounding 모델. `"kosmos"` (기본) / `"exp59"` (PG2 LoRA) |

**응답:**
```json
{
  "action": [1.15, 0.0],
  "action_3d": [1.15, 0.0, 0.0],
  "predicted_class": 1,
  "predicted_label": "FORWARD",
  "bbox": { "cx": 0.51, "cy": 0.62, "area": 0.14 },
  "grounding_caption": "...",
  "grounding_latency_ms": 85.2,
  "latency_ms": 92.1,
  "goal_near_proxy": false
}
```

| 필드 | 설명 |
|------|------|
| `action` | `[linear_x, linear_y]` — 로봇에 그대로 전달 |
| `action_3d` | `[linear_x, linear_y, angular_z]` — ROT 포함 전체 |
| `predicted_label` | 8개 클래스 중 하나 (아래 참조) |
| `bbox` | 탐지된 목표물 위치 (cx/cy는 0~1 정규화) |
| `goal_near_proxy` | `true`이면 목표 도달 → 정지 신호 |

---

### 3-3. 상태 초기화

새 에피소드 시작 전 반드시 호출.

```
POST http://soda:8001/reset
X-Api-Key: vla_devel_key_2026
```

---

### 3-4. 설정 변경 (런타임)

```
POST http://soda:8001/config
X-Api-Key: vla_devel_key_2026
Content-Type: application/json
```

```json
{
  "model": "exp49",
  "speed_scaling": true,
  "grounding_skip_n": 3,
  "smooth_enabled": true,
  "smooth_alpha_xy": 0.65,
  "smooth_alpha_az": 0.80,
  "stop_area_threshold": 0.18,
  "stop_cx_tolerance": 0.25
}
```

| 필드 | 기본값 | 설명 |
|------|--------|------|
| `model` | `"exp49"` | `"exp49"` / `"exp54_s2v2"` / `"exp55"` |
| `grounding_skip_n` | `1` | N프레임마다 grounding 갱신 (3이면 3배 빠름) |
| `speed_scaling` | `true` | bbox 거리 기반 속도 자동 조절 |
| `smooth_enabled` | `true` | action 평활화 (방향 전환 충격 완화) |
| `smooth_alpha_xy` | `0.65` | XY 평활 계수 (낮을수록 부드러움) |
| `smooth_alpha_az` | `0.80` | 회전 평활 계수 |
| `stop_area_threshold` | `0.18` | bbox area ≥ 이 값이면 goal_near_proxy=true (≈0.5m) |
| `stop_cx_tolerance` | `0.25` | 중앙 허용 편차: cx가 0.5±tol 범위여야 정지 |

> **대시보드에서 슬라이더로 조정 가능:** http://100.85.118.58:7865  
> → **"🛑 자동 정지 설정"** 아코디언  
> area 0.18 ≈ 0.5m 거리, 0.30 ≈ 0.3m 거리. 실로봇 테스트마다 조정 권장.

---

## 4. 액션 클래스

| idx | 이름 | linear_x | linear_y | angular_z | 비고 |
|-----|------|----------|----------|-----------|------|
| 0 | STOP | 0.0 | 0.0 | 0.0 | |
| 1 | FORWARD | 1.15 | 0.0 | 0.0 | |
| 2 | LEFT | 0.0 | 1.15 | 0.0 | strafe 좌 |
| 3 | RIGHT | 0.0 | -1.15 | 0.0 | strafe 우 |
| 4 | FWD+L | 1.15 | 1.15 | 0.0 | 대각선 |
| 5 | FWD+R | 1.15 | -1.15 | 0.0 | 대각선 |
| 6 | ROT_L | 0.0 | 0.0 | +0.25 | 제자리 좌회전 |
| 7 | ROT_R | 0.0 | 0.0 | -0.25 | 제자리 우회전 |

> **⚠️ 8001 현재 ROT_L/R 마스킹 적용 중**  
> `proxy_inference_server.py` 내 ROT_L(6)/ROT_R(7)는 소프트웨어 마스킹 상태.  
> 이유: 분포 외 입력에서 잘못 활성화 시 무한 스핀 유발.  
> V5_add_free 데이터셋 재학습 후 마스킹 해제 예정.

> **정지 판단:** `goal_near_proxy: true` 또는 `predicted_label: "STOP"` 중 하나로 처리.  
> 조건: bbox_area ≥ stop_area_threshold **AND** |cx - 0.5| ≤ stop_cx_tolerance

---

## 5. 이미지 포맷 규격

```python
import base64, cv2

frame = camera.get_frame()                    # numpy array (H, W, 3) BGR
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
_, buf = cv2.imencode('.jpg', frame_rgb, [cv2.IMWRITE_JPEG_QUALITY, 85])
image_b64 = base64.b64encode(buf).decode('utf-8')
```

- 입력 해상도: **제한 없음** (내부에서 리사이즈)
- 권장: 640×480 이상, JPEG quality 80~90
- 채널: RGB 순서로 인코딩 권장 (BGR도 동작하나 grounding 정확도 차이)

---

## 6. 호출 예시 (Python)

```python
import requests, base64, cv2

SODA_URL = "http://100.85.118.58:8001"
API_KEY   = "vla_devel_key_2026"
HEADERS   = {"x-api-key": API_KEY}

def get_action(frame_bgr: "np.ndarray", instruction: str = "gray basket") -> dict:
    _, buf = cv2.imencode('.jpg', frame_bgr)
    b64 = base64.b64encode(buf).decode()

    r = requests.post(
        f"{SODA_URL}/predict",
        json={"image": b64, "instruction": instruction, "vlm_model": "kosmos"},
        headers=HEADERS,
        timeout=5.0,
    )
    r.raise_for_status()
    return r.json()

# 에피소드 시작 전
requests.post(f"{SODA_URL}/reset", headers=HEADERS)

# 루프
while running:
    frame = camera.read()
    result = get_action(frame)

    linear_x  = result["action_3d"][0]
    linear_y  = result["action_3d"][1]
    angular_z = result["action_3d"][2]

    robot.move(linear_x, linear_y, angular_z)

    if result.get("goal_near_proxy"):
        robot.stop()
        break
```

---

## 7. ROS 연동 (선택)

MoNaVLA에는 기존 ROS 브릿지 노드가 있습니다. 직접 사용 가능합니다.

```bash
# SODA에서 실행
ros2 run mobile_vla_package api_client_node \
  --ros-args \
  -p api_server_url:=http://localhost:8001 \
  -p default_instruction:="gray basket" \
  -p vlm_model:=kosmos
```

**환경변수로도 설정:**
```bash
export VLA_API_SERVER=http://localhost:8001
export VLA_API_KEY=vla_devel_key_2026
export VLA_INSTRUCTION="gray basket"
export VLA_VLM_MODEL=kosmos
```

**발행 토픽:** `/cmd_vel` (`geometry_msgs/Twist`)  
**서비스 호출:** `/get_image_service` (`camera_interfaces/GetImage`)

---

## 8. 현재 운영 모델 (2026-06-01)

| 서버 | 포트 | 모델 | 성능 |
|------|------|------|------|
| proxy_inference_server | 8001 | exp49 GoalNavMLP | CL 96.4% |
| inference_server (GoalNav-only) | 8000 | exp54_s2v2 | CL ~100% |

> exp59 (PG2 LoRA grounding) 배포 준비 중. 완료 시 `vlm_model: "exp59"` 사용 가능.

---

## 9. 주의사항

1. **에피소드마다 `/reset` 필수** — bbox 히스토리 누적으로 오동작 방지
2. **타임아웃 5초 이상** 권장 — grounding 첫 프레임은 모델 로드 포함
3. **`vlm_model: "exp59"` 사용 시** — PG2 모델 메모리 ~6GB 추가 점유, 첫 요청 ~30초
4. **`grounding_skip_n: 3`** — 실시간성이 중요하면 grounding을 3프레임에 1번만 갱신
5. **STOP 판단** — `goal_near_proxy: true` 또는 `predicted_label: "STOP"` 중 하나로 처리

---

## 10. 브랜치 / 파일 위치

```
MoNaVLA/
├── robovlm_nav/serve/
│   ├── proxy_inference_server.py   ← 8001 서버 (메인)
│   └── inference_server.py         ← 8000 서버 (내부용)
├── ROS_action/src/mobile_vla_package/
│   └── api_client_node.py          ← ROS↔API 브릿지
└── docs/
    └── MONAPI_INTEGRATION.md       ← 이 문서
```

**기준 브랜치:** `inference-integration`  
**서버 기동 스크립트:** `scripts/start_all.sh inference`

# V5 데이터 수집 시스템 — 설계 문서

> 최종 업데이트: 2026-05-18  
> 대상 파일: `scripts/gradio_data_collector.py`

---

## 1. ROS 생태계 구조

```
하드웨어 레이어
┌─────────────────────┐        ┌──────────────────────────────┐
│  CSI 카메라 (Jetson) │        │  옴니휠 로봇 모터            │
│  nvarguscamerasrc   │        │  (pop.Driving 직접 제어)      │
└──────────┬──────────┘        └──────────────┬───────────────┘
           │ GStreamer                          │
           ▼                                   │
[camera_service_server]                        │
  camera_publisher_continuous.py               │
  · buffer_lock (내부 스레드 안전)              │
  · 요청마다 버퍼 3프레임 플러시 후 최신 읽기  │
                                               │
  서비스 제공                                  │
  ├─ get_image_service (GetImage)              │
  └─ reset_camera_service (Empty)              │
           │                                   │
           │ ROS2 Service                      │ pop.Driving
           ▼                                   ▼
[gradio_vla_collector_v5]  ←──────────────────┘
  gradio_data_collector.py

  내부 스레드
  ├─ rclpy.spin()          ROS 이벤트 루프
  ├─ _camera_loop()        10 Hz, 서비스 콜 전담
  └─ JoystickReader._loop() 25 Hz, 조이스틱 읽기

  공유 상태 (threading.Lock 보호)
  ├─ latest_ui_frame       _camera_loop이 채움
  └─ episode_buffer        capture 함수가 채움

  발행 토픽
  └─ /cmd_vel (Twist)      수집 모드에서 구독자 없음
                           실제 이동은 pop.Driving 직접 호출

Gradio UI (브라우저)
  gr.Timer(0.1) → get_feed() → latest_ui_frame 읽기
  버튼/키보드/조이스틱 → teleop_step() / start_rec() / stop_rec()
```

### cmd_vel과 카메라 서비스 간 간섭 여부

| | cmd_vel 발행 | 카메라 서비스 콜 |
|---|---|---|
| ROS2 통신 방식 | Topic (fire-and-forget) | Service (req/res) |
| 실제 경로 | pop.Driving (ROS 외부) | camera_service_server |
| 공유 자원 | 없음 | buffer_lock (서버 내부) |

**결론: 두 경로는 물리적으로 분리돼 있어 간섭 없음.**

---

## 2. 카메라 루프 Hz 선택 근거

| Hz | 간격 | 스텝(0.45 s) 대비 | 판단 |
|---|---|---|---|
| 5 Hz | 200 ms | 스텝보다 느림 | 부족 — 스텝 사이 프레임 보장 안 됨 |
| **10 Hz** | **100 ms** | **스텝의 4.5배** | **권장** |
| 15 Hz | 67 ms | 스텝의 6.7배 | 여유 있음 |
| 20 Hz | 50 ms | 스텝의 9배 | 과함 (기존 ui_poll_loop 방식) |

10 Hz 선택 이유:
- 스텝(0.45 s) 사이 최소 4프레임 갱신 → 항상 최신 이미지 보장
- 서비스 콜 overhead가 카메라 주기의 30 % 이하

---

## 3. 캡처 모드 비교

### 타임라인 (한 스텝, teleop_step 호출 t=0 기준)

```
시간축  0ms      100ms    200ms    300ms    400ms   450ms
        │                                    │       │
        ├────────────────────────────────────┤       │
        │←────────── 로봇 이동 구간 ─────────→│ STOP  │다음스텝


── 기존 (capture_frame_sync) ──────────────────────────────────
  t=0      t=1ms    t≈1~301ms (블로킹)          t=400ms
  publish  timer    [서비스 콜 후 episode_buffer] stop
  ⚠️ 로봇이 이미 이동 중인 프레임이 저장됨 (s_{t+δ})

── PRE_CACHE (주 모드) ────────────────────────────────────────
  t=0      t<1ms    t=1ms                       t=400ms
  cache    buffer   publish                     stop
  복사     추가
  ✅ 정지 상태 관측 저장 (s_t) — VLA 학습 정합

── POST_SYNC (보조 모드) ──────────────────────────────────────
  t=0      t=1ms    t≈1~301ms (블로킹)          t=400ms
  publish  timer    [서비스 콜 후 episode_buffer] stop
  기존과 동일 구조, camera_loop 10 Hz로 충돌 빈도 낮춤
```

### 구현 함수

```python
class CaptureMode(enum.Enum):
    PRE_CACHE = "pre_cache"   # 주 모드: 액션 직전 캐시 스냅샷 (<1 ms)
    POST_SYNC = "post_sync"   # 보조 모드: 액션 직후 서비스 콜 (최대 300 ms)

# PRE_CACHE: lock + memcopy, 비블로킹
def _capture_pre_cache(self, act): ...

# POST_SYNC: ROS 서비스 콜, 블로킹
def _capture_post_sync(self, act): ...
```

- `self.capture_mode = CaptureMode.PRE_CACHE` (기본값)
- Gradio UI의 **Capture Mode** 라디오 버튼으로 런타임 전환 가능

---

## 4. 리소스 비교

```
── ROS 서비스 콜 (초당) ────────────────────────────────────────
기존        22회/s  ████████████████████  100%
            (폴링 20 Hz + 캡처 ~2.2회)
POST_SYNC   12회/s  ██████████░░░░░░░░░░   55%
PRE_CACHE   10회/s  █████████░░░░░░░░░░░   45%

── 메인 스레드 블로킹 (teleop_step 호출당) ────────────────────
기존        최대 300 ms  ████████████████████  100%
POST_SYNC   최대 300 ms  ████████████████████  100%
PRE_CACHE      <1 ms  █░░░░░░░░░░░░░░░░░░░    0.3%

── 스텝 응답 지연 ─────────────────────────────────────────────
기존        최대 301 ms  ████████████████████  100%
POST_SYNC   최대 301 ms  ████████████████████  100%
PRE_CACHE       ~2 ms  █░░░░░░░░░░░░░░░░░░░    0.7%

── 스텝 여유 시간 (0.45 s 기준) ───────────────────────────────
기존        최소 149 ms  ████████░░░░░░░░░░░░   33%  ⚠️
POST_SYNC   최소 149 ms  ████████░░░░░░░░░░░░   33%  ⚠️
PRE_CACHE   최소 448 ms  ████████████████████   99%  ✅

── VLA 학습 데이터 정합성 ─────────────────────────────────────
기존          낮음  ████░░░░░░░░░░░░░░░░░░   20%  s_{t+δ}
POST_SYNC     낮음  ████░░░░░░░░░░░░░░░░░░   20%  s_{t+δ}
PRE_CACHE     높음  ████████████████████  100%  s_t ✅

── 기존 V5 데이터 (150 ep) 호환성 ────────────────────────────
기존        완전 호환  ████████████████████  100%
POST_SYNC   완전 호환  ████████████████████  100%
PRE_CACHE   분포 불일치  ░░░░░░░░░░░░░░░░░░░░    0%  ⚠️
```

---

## 5. VLA 학습 정합성

VLA 학습 데이터 목표: `(관측 s_t, 액션 a_t)` 쌍

| 수집 방식 | 저장 이미지 | 정합성 |
|---|---|---|
| 기존 / POST_SYNC | `s_{t+δ}` — 로봇 이동 도중 프레임 | ⚠️ 낮음 |
| PRE_CACHE | `s_t` — 정지 상태 관측 | ✅ 정확 |

---

## 6. 기존 V5 데이터와 호환성 주의

기존 150개 에피소드는 **POST 방식**으로 수집됨 (publish → capture).

PRE_CACHE 데이터와 혼합 학습 시 입력 이미지 분포가 달라져
val_loss 상승 또는 예측 불안정 가능성 있음.

**권장:**
- 기존 데이터와 혼합 (Exp16 등): `POST_SYNC` 선택
- 새 데이터셋으로 처음부터 학습: `PRE_CACHE` 선택

---

## 7. 동시 서비스 콜 안전성

| 모드 | 동시 콜 발생 가능성 | 처리 |
|---|---|---|
| PRE_CACHE | 없음 (_camera_loop 단독) | 원천 차단 ✅ |
| POST_SYNC | _camera_loop + _capture_post_sync 겹칠 수 있음 | camera_server buffer_lock이 직렬화 |
| 기존 | ui_poll + capture_frame_sync 자주 겹침 | 동일하게 직렬화, 충돌 빈도 최다 |

끊김은 발생하지 않으나 POST_SYNC에서 두 번째 콜이 ~50 ms 추가 대기할 수 있음.

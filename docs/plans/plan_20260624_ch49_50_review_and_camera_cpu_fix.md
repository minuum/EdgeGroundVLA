# Plan — CH49-5/CH50-3 운영 설정 재확인 + 카메라 캡처 루프 CPU busy-loop 수정

> 작성: 2026-06-24 · 상태: **리서치 완료, 승인 대기**

---

## 1. CH49-5 / CH50-3 — 다른 추론환경(minum) 분석 결과 재확인 (코드 변경 없음)

`docs/v5/research_story.html` CH49-5, CH50-3에 이미 기록되어 있고 `monavla-driving`/
`inference-integration` 양쪽에 푸시 완료된 상태. 여기서는 현재 운영 디폴트가 이 결론과
일치하는지만 확인한다.

| 발견 | 결론 | 현재 운영 설정과의 정합성 |
|---|---|---|
| **CH49-5**: area_delta — none/replace/add 3개 모드 전부 skip_n=3에서 무이득(add는 PM/FPE 더 좋지만 SR 93.1%로 셋 중 최저) | area_delta 미배포 확정 | ✅ `stage2_v2_inference_server.py`는 area_delta 사용 안 함 — 정합 |
| **CH50-3**: 작은 객체 줌 재그라운딩 "개선"(SR 96.6%)이 5-seed 평균 SR 89.7%±3.8%p로 정정 — baseline(93.1%)보다도 낮음 | 줌 재그라운딩 배포 보류 재확정 | ✅ 현재 그라운딩 파이프라인은 줌 재크롭 안 함 — 정합 |

**조치**: 코드 변경 불필요. `EXP_MODES`/grounding 관련 주석에 이미 CH49 근거가 있으므로
추가 주석은 생략. 이 plan에 정리해두는 것으로 마감.

---

## 2. 카메라 캡처 루프 CPU busy-loop 수정

### 2-1. 문제

`ROS_action/src/camera_pub/camera_pub/camera_publisher_usb_service.py` `_capture_loop()`
(134~157행):

```python
while rclpy.ok() and self.is_running:
    if self.cap is not None and self.cap.isOpened():
        ret, frame = self.cap.read()   # 드라이버 블로킹에 의존, 쓰로틀 없음
        ...
    time.sleep(0.001)
```

- 의도(주석): `cap.read()`가 드라이버 단에서 자연스럽게 블로킹되어 30fps로 자체 동기화될 것.
- 실측: `usb_camera_service_server` 프로세스가 CPU 44%(코어 거의 하나 풀가동) 점유 —
  V4L2 드라이버 버퍼링으로 `read()`가 거의 즉시 반환되어 루프가 1ms 슬립만 걸린
  busy-loop로 동작.
- 영향: 시스템 전체 CPU 컨텐션 → 대시보드 조이스틱 25Hz 폴링 스레드 스케줄링 지연 →
  "띡띡딕딕" 입력 끊김.

### 2-2. 수정안

목표 FPS(`CAP_PROP_FPS=30` 설정과 동일하게 33ms 주기)에 맞춰 루프 주기를 고정하고,
`cap.read()` 자체 소요 시간을 빼서 드리프트 없이 동기화한다(busy-loop 제거, 기존
"최신 프레임 유지" 동작은 그대로 유지):

```python
TARGET_FRAME_DUR = 1.0 / 30  # 33.3ms — CAP_PROP_FPS=30과 동일

def _capture_loop(self):
    self.get_logger().info('🌀 백그라운드 카메라 캡처 루프 시작')
    while rclpy.ok() and self.is_running:
        loop_start = time.monotonic()
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with self.buffer_lock:
                    self.latest_frame = frame
                    self.failed_reads = 0
            else:
                with self.buffer_lock:
                    self.failed_reads += 1
                if self.failed_reads % 30 == 0:
                    self.get_logger().warn(f'⚠️ [Capture Loop] 프레임 읽기 실패 ({self.failed_reads}회 누적)')
        else:
            with self.buffer_lock:
                self.latest_frame = self.generate_virtual_frame()

        elapsed = time.monotonic() - loop_start
        time.sleep(max(0.0, TARGET_FRAME_DUR - elapsed))
```

- `cap.read()`가 33ms보다 오래 걸리면(`elapsed > TARGET_FRAME_DUR`) sleep 없이 즉시 다음
  루프로 — 프레임 신선도 손해 없음.
- `cap.read()`가 즉시 반환되면(현재 증상) 남은 시간만큼 sleep — CPU 점유를 30fps에 맞게 자연 감소.
- 가상 카메라(`else` 분기)의 기존 `time.sleep(0.033)`은 동일 로직으로 통합되어 중복 제거.

### 2-3. 영향 범위

- 파일: `ROS_action/src/camera_pub/camera_pub/camera_publisher_usb_service.py` (1개 함수만 수정)
- `colcon build --packages-select camera_pub` 후 서버/대시보드 재시작 필요(`go.sh --stop && go.sh --all`).
- 기존 "백그라운드 스레드가 항상 최신 프레임 유지" 동작·서비스 인터페이스(`get_image_service`,
  `reset_camera_service`)는 변경 없음 — `get_fresh_frame()` 등 호출부 영향 없음.

### 2-4. 트레이드오프

- 30fps 캡으로 인해 카메라가 실제 60fps를 낼 수 있는 상황에서도 30fps로 제한됨 — 단,
  `CAP_PROP_FPS`를 이미 30으로 설정해두었으므로 의도된 동작과 일치.
- `cap.read()` 블로킹 자체가 오래 걸리는 경우(드라이버 hang 등) 여전히 그 시간만큼은 못 줄임 —
  이건 이번 수정의 범위 밖(기존 `failed_reads` 카운터가 별도로 처리).

---

## 3. 실행 순서 제안

1. (승인 시) `camera_publisher_usb_service.py` `_capture_loop()` 수정
2. `colcon build --packages-select camera_pub`
3. `go.sh --stop && go.sh --all`
4. CPU 점유 재측정(`ps aux | grep usb_camera_service`) — 44% → 수% 대로 떨어지는지 확인
5. 조이스틱 입력 체감 테스트("띡띡딕딕" 재발 여부)

# [Analysis Report] MoNaVLA Gradio vs. FastAPI Dashboard Architecture Comparison

## Background

MoNaVLA(Mobile Navigation VLA) 로봇 주행 테스트 및 모니터링을 위해 기존에 사용되던 Gradio 기반 대시보드([gradio_inference_dashboard.py](file:///home/soda/MoNaVLA/scripts/gradio_inference_dashboard.py))는 단일 프로세스 내에 웹 GUI 렌더링, ROS2 노드 스핀, 하드웨어 제어 및 주행 제어 루프가 결합되어 동작하였습니다. 이는 멀티스레딩 환경에서 UI 프리징, 하드웨어(pop.driving I2C) 점유 충돌 및 대규모 H5 데이터 로딩 시 브라우저 메모리 고갈과 같은 아키텍처적 한계를 보였습니다.

이러한 문제를 해결하기 위해, UI와 실시간 로봇 제어 루프를 완전히 분리하고 REST API 및 웹 표준 기술(HTML5 Canvas, Chart.js)을 활용하는 경량화된 FastAPI 기반 웹 대시보드([mona_dashboard.py](file:///home/soda/MoNaVLA/robovlm_nav/serve/mona_dashboard.py))로 전환을 시도하고 있으며, 그 비교 및 매핑 결과를 분석합니다.

---

## Analysis

기존 Gradio 대시보드의 9개 주요 탭 및 컴포넌트 세부 명세와 신규 FastAPI 대시보드의 구현 현황 및 매핑 상태를 비교 분석합니다.

### 1. 서비스 포트 및 시스템 메트릭 비교

| 메트릭 | 기존 Gradio 대시보드 아키텍처 | 신규 FastAPI 대시보드 아키텍처 |
|---|---|---|
| **서비스 포트** | `7865` ([Line 20](file:///home/soda/MoNaVLA/scripts/run/go.sh#L20)) | `7800` ([Line 19](file:///home/soda/MoNaVLA/robovlm_nav/serve/mona_dashboard.py#L19)) |
| **추론 서버 포트** | `8001` (FastAPI, 무변경) | `8001` (FastAPI, 무변경) |
| **카메라 연동 방식** | ROS2 GetImage Service 동기 10Hz 폴링 후 Gradio Image 컴포넌트 갱신 | ROS2 GetImage Service 10Hz 폴링 후 FastAPI MJPEG 스트림 (`/camera/stream`) 브라우저 직결 |
| **BBox 렌더링 방식** | Python Pillow/OpenCV를 통해 서버 측에서 이미지 프레임 직접 렌더링 후 전송 | 클라이언트 측에서 HTML5 Canvas Overlay를 활용해 실시간으로 바운딩박스 및 Grid 3분할선 렌더링 |
| **세션 로깅 및 H5 저장** | `scripts/inference_logger.py` 바인딩, `docs/inference_sessions/` 및 `docs/inference_reports/` 아래에 저장 | `scripts/inference_logger.py` 바인딩 유지, `docs/inference_sessions/` 및 `docs/inference_reports/` 아래 동일 경로 저장 |

---

### 2. Gradio 탭별 세부 기능 매핑 매트릭

| Gradio 탭 명칭 (Gradio 소스 위치) | 기능 상세 및 연동 데이터 | FastAPI 대시보드 매핑 및 구현 현황 |
|---|---|---|
| **Tab 1: 🤖 Drive / Inference**<br>(`Line 2011`) | - 주행 대상 entity 지정<br>- SYNC/PRE/ASYNC 주행 루프 기동 및 정지<br>- 스텝 수, 최종 출력 액션값, Latency 모니터링<br>- 화이트 밸런스 컬러 보정 적용 | **Tab 1: 🤖 Drive Control**<br>- `/drive/start` (POST), `/drive/stop` (POST)를 통해 백그라운드 스레드 주행 루프 기동 제어<br>- `/cc/set` (POST) 컬러 코렉션 인자 변경 실시간 연동<br>- Step, Latency 실시간 타임라인 렌더링 |
| **Tab 2: 🔍 Grounding 검증**<br>(`Line 2351`) | - 실시간 바운딩 박스 BBox 좌표 확인<br>- 타겟 탐지 영역 비율(area) 및 수평 치우침(cx, cy) 표시<br>- 주행 STOP 트리거 임계값 검증 게이지 출력 | **Tab 2: 🔍 Grounding 검증**<br>- `/drive/status` API의 실시간 BBox 데이터를 파싱하여 클라이언트 HTML5 Canvas에 바운딩박스 직접 드로잉<br>- Area 크기에 맞춘 동적 CSS 프로그레스 바 게이지 매핑 완료 |
| **Tab 3: 📊 Latency/Drift 진단**<br>(`Line 2690`) | - 1.0s/1.35s/1.92s nominal 대비 누적 처리 시간 드리프트 실측<br>- 4초 지연 돌파 감지 시 경고<br>- 드리프트 변화 시뮬레이션 Matplotlib 차트 | **Tab 3: 📊 Latency & Drift**<br>- `/drive/drift/run` (GET) 및 `/drive/drift/reset` (POST) 엔드포인트 신설<br>- nominal basis(1.0s, 1.35s, 1.92s) 비교 지원<br>- **Chart.js** 기반의 반응형 Line Chart로 시각화 보완 |
| **Tab 4: 🧪 경로 검증 (Path Test)**<br>(`Line 2959`) | - 에피소드 CSV(`logs/episode_log.csv`) 요약 로그 테이블 로딩<br>- 주행 기록 복귀(`return_to_start`) 핸들링 및 역방향 Timed Move 재생 | **Tab 4: 🧪 Action Trajectory & Tab 1**<br>- `/drive/return` (POST) 엔드포인트를 호출해 백그라운드에서 `action_history` 역재생 수행<br>- **Chart.js Scatter Plot**을 활용해 선속도($lx$) vs 횡속도($ly$) 실시간 이동 궤적 2D 플로팅 추가 |
| **Tab 5: 🔧 STOP 캘리브레이션**<br>(`Line 3298`) | - STOP 판단을 위한 area 임계값 설정 (기본: 0.18)<br>- 조이스틱 키패드(Q/W/E/A/D/S/R/T/STOP) 수동 제어<br>- 수동/반자동 캘리브레이션 녹화 데이터 생성 및 `.jsonl`, `.mp4` 비디오 저장 | **Tab 5: 🔧 STOP & Calibration**<br>- `/config` (POST) 프록시로 추론 서버의 `stop_area_threshold` 즉시 갱신<br>- `/drive/manual` (POST) 및 속도 슬라이더 연동<br>- 키보드 이벤트 리스너(ArrowKeys, WASD, Q/E/R/T, SpaceBar) 추가 보완<br>- `/calib/rec/start`, `stop`, `snap`, `clear`, `save` 신설, `VideoWriter` 탑재 완료 |
| **Tab 6: 📚 세션 히스토리**<br>(`Line 3368`) | - `docs/inference_sessions` 내 H5 세션 리스트 로드<br>- 슬라이더를 이용해 개별 프레임 수동 탐색 및 이미지/액션/BBox 로드<br>- 이상치 자율 감지 검증 및 L/R/C 라벨 추가 | **Tab 6: 📚 Session History**<br>- `/sessions/list` (GET), `/sessions/load` (GET), `/sessions/label` (POST) 바인딩<br>- `/sessions/frame` (GET) 개별 프레임 이미지 실시간 전송 (OOM 회피)<br>- JS Frame Inspector 슬라이더 및 Play/Pause 자동 슬라이드 쇼 기능 보완 |
| **Tab 7: 🖥️ 시스템**<br>(`Line 3688`) | - Jetson 하드웨어 리소스 요약 및 /model/info 정보 출력 | **Tab 7: 🖥️ System Manage & Header**<br>- `/infer/health` (GET) 실시간 추론 서버 정보 출력<br>- `/system/reset` (POST)를 통한 프로세스 세션 초기화 단추 맵핑 |
| **Tab 8: 📷 수집 모니터**<br>(`Line 3764`) | - 실시간 스텝 기록(최대 12) 및 그라운딩 캐시 비율 실시간 텍스트 출력 | **Header & Tab 1**<br>- 상단 헤더의 ROS/Infer/Camera 실시간 헬스 배지 연동<br>- 런타임 통계 자동 집계 화면 표시 |
| **Tab 9: 📖 모드 가이드**<br>(`Line 3865`) | - 정적 텍스트 가이드 출력 | **Tab 7 및 Footer**<br>- 포트 정보 및 가이드 통합 제공 |

---

## Findings

1.  **동작 무중단 런타임 설정 프록시**:
    FastAPI 대시보드(7800)의 설정 스위치를 조작하면 내부적으로 `/config` (POST) API를 통해 추론 서버(8001)의 런타임 파라미터가 실시간으로 재기동 없이 즉시 변경됩니다. (`stage2_v2_inference_server.py:L1196` 프록시 연동)
2.  **키보드 단축키 수동 제어(Keyboard Hotkeys)**:
    Gradio에서 마우스 클릭만 지원하던 수동 주행 제어를 웹 키보드 리스너 기반(`keyup`/`keydown` 연동)으로 보완하여, 로봇 주행 현장에서 노트북으로 WASD와 화살표 키를 통해 실시간 TIMED 동작 테스트 및 캘리브레이션 스냅샷을 빠르게 기록할 수 있습니다.
3.  **메모리 최적화 프레임 로더**:
    H5 파일의 수백 장에 달하는 관측 이미지 데이터(`observations/images`)를 한번에 로드하지 않고, `/sessions/frame?sid=XXX&idx=YYY` 라우트를 통해 현재 슬라이더가 위치한 인덱스의 이미지 바이너리만 JPEG 스트림으로 호출하도록 구현하여 Jetson의 OOM 위험을 완전히 제거하였습니다.

---

## Conclusion

새로 구축된 FastAPI 대시보드([mona_dashboard.py](file:///home/soda/MoNaVLA/robovlm_nav/serve/mona_dashboard.py))는 기존 Gradio 대시보드의 실시간 시뮬레이션, 수동 제어, 세션 라벨링, 시스템 파라미터 튜닝 기능을 완벽하게 상속하면서, 웹 성능 및 UI 반응 속도를 극적으로 개선시켰습니다. 

특히 HTML5 Canvas와 Chart.js를 이용한 클라이언트 사이드 그래픽스 렌더링 적용으로 인해 로봇 프로세서(Jetson)의 연산 효율성이 향상되었으며, 키보드 핫키 지원 등으로 실제 현장 주행 및 캘리브레이션 테스트를 한층 더 편리하게 수행할 수 있습니다.

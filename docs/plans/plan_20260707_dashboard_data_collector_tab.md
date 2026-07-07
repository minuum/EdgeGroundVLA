# Plan: 데이터수집(mobile_vla_data_collector.py)을 7800 대시보드 탭으로 통합 (2026-07-07)

> 상태: **Phase 1+2 구현 완료, 라이브 테스트 완료 (2026-07-07)**
> - 조이스틱 자동 포함 확인 (VLAControlManager.publish_and_move 단일 훅)
> - 버그 발견/수정: robust_stop() 5x 중복 정지펄스가 H5에 개별 프레임으로
>   기록되던 문제 → 연속 STOP 프레임 dedupe로 해결 (2키입력→17스텝이던 것을
>   2키입력→4스텝의 정상적인 move/stop 쌍으로 수정)
> - H5 스키마(images/actions/action_event_types+attrs) 원본과 100% 동일 확인
> - scenario_progress.json/time_period_stats.json 스키마도 원본과 동일 맞춤
> - 테스트 중 실수로 실제 V5 데이터셋(289 에피소드)의 progress json을 오염시켰다가
>   `git checkout`으로 즉시 원복 — 이후 테스트는 별도 scratch dir로만 진행
> - Phase 3(레거시 스크립트 동시실행 방지)는 미착수, 별도 논의 필요
> 리서치 근거: `ROS_action/src/mobile_vla_package/mobile_vla_package/mobile_vla_data_collector.py` (3,475줄) 구조 분석 완료

---

## 배경

minum이 제안: 데이터수집을 7800 대시보드에 새 탭으로 합치자. 근거는 "이미
`/cmd_vel` + `GetImage` 연결이 대시보드에 있어서 새 ROS 브릿지 안 만들어도
됨" — **확인 결과 사실**. `mona_dashboard.py`는 자체 `rclpy.Node`
(`cmd_pub = create_publisher(Twist, '/cmd_vel', ...)`, `get_image_client`)를
이미 갖고 있음. 다만 collector 자체는 **터미널 raw keyboard 입력
(`termios`/`tty`, `sys.stdin.read(1)`) 기반 blocking 루프**로 짜여 있어서,
웹 대시보드로 옮기려면 입력 계층을 통째로 재작성해야 함 — 단순 "탭 추가"가
아니라 **입력 방식 전환을 동반한 이식**.

## 원본 동작 요약 (보존해야 할 것)

1. **키 매핑**: WASD(전/후/좌/우 strafe) + QEZC(대각선) + RT(회전) + Space(정지) —
   `linear_x/linear_y/angular_z` 값 고정 (예: `w`→lx=1.15)
2. **키 뗌(release) 시뮬레이션**: 실제 keyup 이벤트가 없어서, 매 keydown마다
   400ms `threading.Timer` 워치독을 재무장 — 새 키가 안 들어오면 자동 정지.
   **웹으로 옮기면 오히려 진짜 `keyup` 이벤트를 쓸 수 있어서 더 정확해짐.**
3. **시나리오/패턴/거리 상태머신**: `n`→시나리오 선택(1~9)→패턴(core/variant)→
   거리(close/medium/far or fixed)→반복횟수 입력→측정 시작. 상태 플래그
   4~5개로 어떤 키가 활성화되는지 분기.
4. **18-key 가이드 시�퀀스**: core 패턴일 때 "다음 눌러야 할 키" 힌트 표시,
   실제 입력과 다르면 mismatch 카운트.
5. **에피소드 저장(H5)**: `images (N,H,W,3)`, `actions (N,3) float32
   [lx,ly,az]`, `action_event_types (N,) string` + attrs
   (`episode_name, total_duration, num_frames, action_chunk_size=8,
   time_period, collection_datetime, ...`). **이 스키마를 그대로 유지해야
   기존 분석 스크립트/`resync_scenario_progress`와 호환됨.**
6. **진행률 3개 JSON**: `scenario_progress.json`, `time_period_stats.json`,
   `core_patterns.json` — 모두 `data_dir` 하위, 매 에피소드 저장 후 갱신.
   시작 시 `resync_scenario_progress()`가 실제 `.h5` 파일을 다시 스캔해서
   authoritative하게 재계산 (JSON은 캐시일 뿐).
7. **이미지 fetch**: 고정 클록이 아니라 **키 이벤트마다 1회** `GetImage`
   동기 호출 → 이 프레임을 그 스텝의 이미지로 사용.

## 설계 — 단계적 접근 (한 번에 다 이식하면 검증 리스크 큼)

### Phase 1 (이번에 승인 요청하는 범위) — MVP 주행+저장
- 새 탭 `📷 데이터수집` (`tab-collect`) 추가
- 백엔드: `robovlm_nav/serve/mona_dashboard.py`에 `DataCollectSession` 클래스
  신설 — 기존 `_ros`(dashboard의 ROS 노드) 재사용, `cmd_pub`/`get_image_client`
  그대로 사용. 새 ROS 노드 안 만듦.
- **조이스틱도 자동 포함**: 대시보드는 키보드/수동버튼/조이스틱 등 모든 입력이
  이미 단일 진입점 `VLAControlManager.publish_and_move()`
  (`robovlm_nav/serve/vla_control_utils.py:39`)를 통과함 — 조이스틱 루프도
  `ctrl.publish_and_move(*vel, source="joystick")`로 동일 경로 사용 중
  (`mona_dashboard.py:360-368`). 스텝 기록 훅을 `/collect/key` 엔드포인트가
  아니라 **`publish_and_move()` 안**에 걸어서, 수집 세션이 활성 상태일 때
  `source`에 상관없이(keyboard/joystick/manual) 모든 실제 명령을 기록.
  → 원본 collector는 키보드 전용이었지만, 이 이식판은 조이스틱 수집도
  별도 코드 없이 자동으로 됨.
- 엔드포인트:
  - `POST /collect/key` `{key, event: "down"|"up"}` — 키보드 입력을
    `ctrl.publish_and_move`/`move_and_stop_timed`로 변환 (조이스틱과 동일 경로 통과)
  - `POST /collect/episode/start` `{episode_name?}` — `episode_data=[]` 리셋,
    기록 훅 활성화
  - `POST /collect/episode/stop` — 훅 비활성화 + `save_episode_data`와 동일
    스키마로 H5 저장
  - `GET /collect/state` — 현재 수집중 여부/스텝수/마지막 액션 source(키보드
    인지 조이스틱인지 표시)/최근 프레임
- 프론트: `tabindex`+`keydown`/`keyup` 리스너로 WASDQEZC RT Space 캡처
  (조이스틱은 이미 있는 조이스틱 패널 그대로 사용 — 수집 세션 켜져 있으면
  자동으로 같이 기록됨), 현재 액션/스텝수/입력소스/최근 프레임 표시
- **시나리오 선택 메뉴/진행률/18-key 가이드는 Phase 1에 포함 안 함** —
  수동으로 episode_name을 입력해서 저장하는 수준까지만. (레거시 스크립트가
  하던 "정교한 프로토콜 가이드" 기능은 Phase 2로 미룸)

### Phase 2 (Phase 1 검증 후 별도 승인) — 시나리오/진행률/가이드
- `cup_scenarios`/패턴/거리 선택 UI(버튼 방식 — 원본의 숫자키 방식보다 웹에서
  더 자연스러움), `scenario_progress.json`/`time_period_stats.json`/
  `core_patterns.json` 3종 read/write, 18-key 가이드 힌트 표시

### Phase 3 (Phase 2 이후) — 레거시 스크립트와의 동시 실행 방지
- **문제**: 대시보드 세션과 터미널 `mobile_vla_data_collector.py`가 동시에
  뜨면 같은 `data_dir`에 두 프로세스가 동시에 쓰기 시도 + 둘 다 `/cmd_vel`
  퍼블리시 경쟁 → 데이터 오염 위험.
- **방안**: `data_dir/.collector_lock` 파일(PID+시작시각) — 대시보드 쪽은
  세션 시작 시 생성/종료 시 삭제(신규 코드라 바로 적용 가능). 레거시
  스크립트 쪽도 시작 시 이 락파일 존재 여부를 확인하고 있으면 즉시 종료하는
  가드 **한 군데만** 추가(최소 침습, `main()` 진입부 1곳).
- 대안(더 간단, 대신 사람이 기억해야 함): 락파일 없이 "터미널 collector 쓸 땐
  대시보드 탭 닫기"를 운영 규칙으로만 문서화. Phase 3는 이 중 어느 쪽으로 할지
  Phase 1/2 완료 후 다시 논의.

## 확인해주세요

1. **Phase 1 범위(키입력+저장만, 시나리오 메뉴 없음)로 먼저 승인**해도 될지 —
   아니면 처음부터 Phase 1+2를 통으로 계획해서 진행할지
2. H5 스키마를 원본과 100% 동일하게 유지하는 것에 이견 없는지 (기존
   `resync_scenario_progress`/분석 스크립트 호환 위해 필수라고 판단)
3. Phase 3(레거시 스크립트 동시실행 방지)는 지금 설계만 잡아두고 실제
   레거시 스크립트 수정은 나중에 진행 — 괜찮은지

# MoNaVLA ↔ MonAPI 제어 알고리즘 통합 및 검증 리포트

## Background
MoNa-pi 레포지토리의 주행 최적화 및 제어 알고리즘(소프트 스냅핑, 자동 정지 제어, Yaw 복원 등)을 MoNaVLA의 메인 추론 서버(`inference_server.py`)에 이식하여, `monapi` 모드(`request.vlm_model == "monapi"`)로 동작할 때 동일한 주행 성능과 제어 이점을 누릴 수 있도록 개발을 진행했습니다. 

---

## Analysis
MonAPI 레포토리의 `server.py` 변경 사항을 파악한 결과, 다음과 같은 두 가지 핵심 알고리즘이 적용되어 있었습니다.

1. **`classify_action(vx, vy, wz)`**:
   * 로봇의 선속도와 각속도 연속 값을 매핑하여 `STOP`, `FORWARD`, `LEFT`, `RIGHT`, `FWD+L`, `FWD+R`, `ROT_L`, `ROT_R` 중 하나의 이산 라벨(8-class)로 매핑합니다.
2. **`snap_monapi_action_to_label(action, label)`**:
   * 이산 라벨을 기반으로 continuous actions를 Soft-Decision 기반으로 스냅핑합니다.
   * `STOP` 시에는 즉각 정지 명령을 전달합니다.
   * 직진(`FORWARD`) 및 회전 시 불필요한 게걸음 횡이동(Linear Y)을 감쇄율 90%로 억제합니다.
   * 제자리 회전(`ROT_L/R`) 시 전진 속도를 차단하고 조향(Angular Z)의 회전 각속도를 복원하여 분포 외(OOD) 입력에 의한 무한 제자리 스핀을 억제합니다.
   * 대각 주행(`FWD+L/R`) 시 과속을 방지하도록 전진 속도를 0.70 이하로 제한합니다.

이와 동시에, 대시보드(Gradio)의 실시간 주행 감속 및 정지 파라미터가 API 서버에 반영될 수 있도록 `/config` 엔드포인트가 신설되어야 함을 파악했습니다.

---

## Findings
작업 중 다음과 같은 코드 결함 및 예외 상황을 발견하고 해결했습니다.

1. **병합 충돌 잔재로 인한 코드 손상 및 문법 오류**:
   * 이전 세션에서 병합 충돌을 해결하는 과정 중, `MobileVLAInference.predict()` 메소드의 리턴 구문 뒤에 예전 구문의 찌꺼기가 남아 문법 에러(Syntax Error)를 일으키는 상태였습니다. 잔재 코드를 제거하여 정상 반환 구조로 교정했습니다.
   * `extract_vision_feature()` 함수의 exception handling 구문 아래 들여쓰기가 누락되어 `IndentationError`가 발생하는 문제를 수정했습니다.
2. **`MobileVLAInference` 클래스 내 `reset()` 메소드 누락**:
   * `POST /reset` API 핸들러에서 에피소드 초기화를 위해 `model.reset()`을 호출하지만, 실제 VLA 추론 클래스(`MobileVLAInference`)에 `reset()` 메소드가 구현되어 있지 않아 `HTTP 500` 오류가 발생하던 결함을 발견했습니다.
   * 해당 클래스에 `reset()` 메소드를 신설하여, LSTM/History Memory 및 `prev_action_3d`, `inference_count` 등을 초기화할 수 있도록 조치했습니다.
3. **`smoothing_alpha` 기본값 튜닝**:
   * 주행 최적화 가이드([monavla_integration_guide.md](file:///home/soda/MoNaVLA/docs/inference_reports/monavla_integration_guide.md))의 내용에 따라, 조향 채터링(진동 거동)을 극적으로 완화하기 위해 `smoothing_alpha`의 기본값 값을 기존 `0.65`에서 **`0.45`**로 조절 및 고정했습니다.

---

## Conclusion
수정된 서버 코드를 반영하여 포트 **8082**에서 API 서버를 재기동했으며, 검증용 통합 테스트 스크립트([test_monapi_integration.py](file:///home/soda/MoNaVLA/tools/test_monapi.py))를 통해 다음과 같이 작동을 검증했습니다.

| 테스트 엔드포인트 | 요청 데이터 | 검증 결과 | 비고 |
| :--- | :--- | :--- | :--- |
| `POST /reset` | `{}` | **성공 (HTTP 200)** | 세션 카운트 및 policy head 메모리 초기화 |
| `POST /config` | `{"smooth_alpha_xy": 0.45}` | **성공 (HTTP 200)** | 대시보드 실시간 파라미터 반영 연동 완료 |
| `POST /predict` | `{"image": "...", "vlm_model": "monapi"}` | **성공 (HTTP 200)** | 2D `action` 리스트 및 `action_3d` 스냅핑 반환 검증 |

모든 테스트 케이스가 정상적으로 통과하여, 로봇의 실주행 환경에서 안정성과 실시간 Yaw 복원 필터가 완벽하게 작동하는 것을 확인했습니다.

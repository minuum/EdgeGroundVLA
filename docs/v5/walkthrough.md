# Walkthrough - PaliGemma 대체 객체 가이드 수립 및 Gradio 수집기 비동기 10Hz 최적화 완료

LoRA 재학습 실패에 대비하여 PaliGemma 사전학습 강도가 높은 기본 객체 교체 가이드라인을 수립하고, 비동기 주행 시의 정합성 문제를 해결하기 위해 Gradio 수집기(`scripts/gradio_data_collector.py`) 코드를 최적화 및 검증하였습니다.

## 1. 주요 변경 및 신설 파일

* **[PRETRAINED_OBJECT_REPLACEMENT_PLAN.md](file:///home/minum/26CS/MoNaVLA/docs/v5/PRETRAINED_OBJECT_REPLACEMENT_PLAN.md)** [NEW]
  * **Background**: 6/4 회의의 224개 에피소드 정량 메트릭과 오프라인 Closed-Loop 정확도(96.2%)를 배경으로 제시하며, OOD 상황 및 텍스트 프레이즈 편차에 따른 궤적 붕괴 문제를 해결하기 위한 백업 플랜의 필요성 정의.
  * **Analysis**: 사전학습 인지 강도, 획득 비용, 로봇 조작 및 탐지 적합성(30cm 로봇 카메라 시점 및 224px 해상도에서의 탐지성)의 3대 축을 기준으로 실내 99개 클래스 분석.
  * **Findings (우선순위 매트릭스)**:
    * **1순위**: `Chair / Stool` 및 `Waste container` (우수한 탐지성, $15 이내 저렴한 실물 획득 비용, 밀기 및 파지 조작성 충족)
    * **2순위**: `Laptop` 및 `Flowerpot`
    * **보류/제외**: 충돌 위험이 크거나(`Lamp`), 크기가 너무 작아(`Mobile phone`) 탐지가 불가능한 객체 선별.
  * **Conclusion (탐지 기준 및 수집 가이드)**:
    * Confidence Score $\ge 0.85$ 필터링 및 픽셀 폭 25px 확보 최소 탐지 기준 설정.
    * 200~350ep 규모의 데이터 신규 수집을 위한 다양성(조명 20%, 진입각 60%, 장애물 회피 20%) 표준화 프로토콜 마련.

* **[gradio_data_collector.py](file:///home/minum/26CS/MoNaVLA/scripts/gradio_data_collector.py)** [MODIFY]
  * **Action Lag 100ms 보정 시프트**: `save_h5` 내부에서 `js_mode == 'async'`(10Hz 비동기 주행 모드)일 때 조종자의 반응 지연 속도(약 100ms)를 상쇄시키기 위해 액션 배열을 1프레임 앞으로 시프트(`s_t`에 `a_{t+1}`을 매핑)하여 정합성을 정밀 보정하고, 마지막 프레임은 정지 상태(`[0.0, 0.0, 0.0]`)로 패딩.
  * **H5 내 `timestamps` 데이터셋 신설**: H5 파일에 주행 타임스탬프(`timestamps`) 데이터셋을 추가 저장하여 추후 비동기 분석 및 재생 정합성을 추적할 수 있도록 개선.
  * **300ms Jitter Hold 필터**: `JoystickReader` 루프에서 입력 감지 튐(Jitter)으로 인해 일시적으로 중립(Neutral)으로 떨어지는 현상을 300ms 동안 방지하도록 홀딩 처리하여, 주행 도중 불필요하게 정지 라벨이 인젝션되는 유령 정지(mid-stop) 문제 사전 차단.
  * **ROS2 Fallback Dummy Class 보강**: ROS2 환경이 소싱되지 않은 오프라인/일반 로컬 개발 환경에서도 수집기 모듈이 문제없이 import 및 인스턴스화가 가능하도록 더미 `Node`, `CvBridge`, `Twist`, `GetImage` 클래스를 정의하고 `_camera_loop`에 `ROS_AVAILABLE` 예외 분기 추가.

## 2. 검증 완료

* **Syntax & Compile 검증**: `python3 -m py_compile scripts/gradio_data_collector.py` 실행 결과 구문 오류 없이 성공적으로 컴파일 완료.
* **H5 무결성 & Action Shift 검증**:
  * [verify_data_collector.py](file:///home/minum/.gemini/antigravity-ide/brain/98acbc56-6443-4d86-b117-8834b02da56c/scratch/verify_data_collector.py)를 작성하여 `.venv` 가상환경 내에서 직접 시뮬레이션 H5 데이터 저장을 수행.
  * `async` 모드에서는 원본 액션 대비 1프레임 시프트(Action Lag 보정) 및 `[0.0, 0.0, 0.0]` 최종 패딩이 정밀하게 일치함을 검증.
  * `sync` 모드에서는 액션 변경 없이 원본 그대로 저장됨을 검증.
  * `timestamps` 데이터셋이 100ms 주기로 올바르게 H5 내에 쓰여졌음을 확인 완료.

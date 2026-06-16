# Gradio Data Collector 비동기 10Hz 데이터 수집 최적화 계획

비동기식 주행 데이터 수집 루프(`self.js_mode == 'async'`)를 적용할 때 발생할 수 있는 시간축 정합성(Action Lag) 불일치 및 중간 유령 정지(mid-stop) 라벨 오염을 방지하기 위해, 데이터 수집기(`scripts/gradio_data_collector.py`) 코드를 최적화합니다.

## User Review Required

> [!IMPORTANT]
> 1. **Action Lag 100ms(1프레임) 시간축 시프트 적용**: 조종자의 반응 속도를 역보정하여 이미지 $s_t$에 1프레임 뒤의 액션 $a_{t+1}$을 매핑합니다.
> 2. **유령 정지(Mid-stop) 필터링**: 주행 중 조이스틱 제어 입력의 미세한 튐으로 인해 속도가 0에 가깝게 잠깐 떨어지는 구간(유령 정지)을 직전 속도로 복원하여 라벨 오염을 방지합니다.
> 3. **H5 내 timestamps 저장 신설**: 비동기 타임라인 정합성을 위해 캡처 시각 타임스탬프를 H5 파일의 신규 데이터셋으로 저장합니다.

## Open Questions

> [!NOTE]
> * **1프레임(100ms) 시프트의 타당성**: 실기 테스트 도중 조종 피드백 속도에 따라 시프트 윈도우를 1~2프레임으로 미세 조정해야 할 수 있습니다. 우선 1프레임(100ms)을 기본값으로 적용하고 검증하겠습니다.

## Proposed Changes

### scripts/ (수집기 소스코드 수정)

#### [MODIFY] [gradio_data_collector.py](file:///home/minum/26CS/MoNaVLA/scripts/gradio_data_collector.py)
* **`save_h5` 함수 수정**: 
  - `timestamps` 데이터셋 신설 및 저장 (`f.create_dataset('timestamps', ...)`).
  - 저장 직전 Action Lag 보정을 위해 액션 리스트를 1프레임 시간축으로 시프팅 정렬.
* **`joystick_drive` 및 `_capture_pre_cache` 비동기 수집 루프 수정**:
  - `key`가 일시적으로 누락되는 300ms 이하의 튐(Jitter)에 대해서는 `STOP` 액션을 인젝션하지 않고 직전 액션을 일시적으로 홀딩(Hold)하여 `mid-stop` 현상 사전 차단.
  - 최종 정지(Stop Rec) 시점에만 도착 정지 라벨(Plateau STOP)이 자동으로 주입되도록 최적화.

---

## Verification Plan

### Automated Tests
- `scripts/gradio_data_collector.py` 실행을 통한 런타임 syntax 에러 여부 확인 (`python -m py_compile scripts/gradio_data_collector.py`).
- 새롭게 수집하여 저장한 임시 H5 파일의 HDF5 키 및 차원 정밀 검증 스크립트 실행.

### Manual Verification
- 조이스틱 비동기(Async) 모드로 주행 수집을 1회 진행한 뒤 생성된 H5 파일의 `actions`, `observations/images`, `timestamps`가 10Hz 스펙에 맞게 올바르게 정렬되었는지 h5py 모듈을 통해 검증.

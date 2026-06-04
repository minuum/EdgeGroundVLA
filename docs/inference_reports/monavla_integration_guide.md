# MoNaVLA 로봇 서버 최종 제어 및 주행 최적화 가이드

본 문서는 $2\text{m} \times 2\text{m}$ 캡스톤 주행 시험 환경에서 MoNa-pi 모델이 오버슈트(과도 주행)나 채터링(진동 거동) 없이 정교하게 목적지에 수렴할 수 있도록 로봇 구동 세팅을 일치시키는 연동 가이드입니다.

---

## 1. 물리 거리-속도 비율 동기화 (0.25x 선속도 스케일)
* **배경**: VLA 학습 데이터셋은 10Hz($0.1\text{초}$ 간격)로 수집되어 최고 속도($1.15\text{ m/s}$) 기준 1스텝에 $\approx 11.5\text{ cm}$ 이동을 상정합니다. 반면 실로봇 루프는 $0.4\text{초}$ 주기로 제어되어, 스케일 보정 없이 구동 시 1스텝당 **46~63cm를 폭주**해 버립니다.
* **해결**: 선속도 성분에 **`0.25`**의 스케일 비율을 적용하여, 로봇의 실주행 속도를 최고 $0.28\text{ m/s}$로 제한하고 1스텝당 실제 이동 거리를 **`11.2 cm`**로 동기화합니다.

### 🛠️ 반영 방법
[vla_control_utils.py](file:///home/minum/26CS/MoNaVLA/robovlm_nav/serve/vla_control_utils.py) 파일을 다음과 같이 수정합니다. (로컬은 수정 완료)

```python
# 1. VLAControlManager.__init__ 내에 스케일 속성 추가
self.scale_factor = 0.25  # 2m x 2m 맵 맞춤형 감속 스케일

# 2. publish_and_move 메서드 초입에 선속도 스케일링 계산 추가
scaled_lx = lx * self.scale_factor if not is_stop else 0.0
scaled_ly = ly * self.scale_factor if not is_stop else 0.0

# 3. ROS /cmd_vel Twist 및 하드웨어 구동에 scaled_lx, scaled_ly 대입
twist.linear.x = float(scaled_lx)
twist.linear.y = float(scaled_ly)
```

---

## 2. 추론 속도 극대화 (ODE Steps = 2 또는 3 단축)
* **배경**: 기존 기본값인 ODE 5스텝 구동 시, 내부적으로 VLM 신경망 포워드가 10회 반복되어 평균 **`411 ms`** 수준의 제어 지연이 발생합니다.
* **해결**: 오프라인 Closed-Loop 검증 결과, ODE 단계를 2~3단계로 대폭 줄여도 정형 성공률은 **100%**로 완벽히 수렴하며 위치 오차(FPE) 하락은 단 `2~3 mm` 수준으로 무시할 수 있는 수준이었습니다. 
* **설정**: 로봇 API 구동 또는 대시보드 기동 시, Heun Solver ODE Steps 매개변수를 **`3` 또는 `2`**로 설정합니다.
  * **2스텝 적용 시**: VLM 포워드가 4회로 급감하여 **레이턴시가 180~200ms 대(약 5Hz)로 즉시 60% 단축**됩니다.

---

## 3. 조향 채터링(진동 거동) 완화 (Smoothing Alpha 보정)
* **배경**: VLA 모델 출력 연속값이 이산 임계치 근처에 걸쳐 있을 때 0.4초 만에 직진 $\leftrightarrow$ 횡이동 명령이 급변하는 진동 거동이 발생할 수 있습니다.
* **해결**: 지수 이동 평균(EMA) 필터 가중치(`smoothing_alpha`)를 강화하여 명령의 충격을 감쇄시킵니다.

### 🛠️ 반영 방법
[inference_server.py](file:///home/minum/26CS/MoNaVLA/robovlm_nav/serve/inference_server.py) 내부 `self.smoothing_alpha` 값을 다음과 같이 튜닝합니다.

```python
# 기존 0.6 (현재 값 60% 반영) -> 0.45 (이전 관성 55%, 현재 값 45% 반영)
self.smoothing_alpha = 0.45
```

---

## 📈 4. 학습 서버 검증 수치 자료 (참고용)

학습 서버 오프라인 Closed-Loop 시뮬레이션으로 입증된 검증 데이터셋 수치입니다. (2스텝 Heun만으로도 완벽한 성공률을 띰)

* **ODE 5스텝 (기본)**: 정형 성공률 **100%** | FPE **0.0697 m (6.9 cm)**
* **ODE 3스텝 (권장)**: 정형 성공률 **100%** | FPE **0.0707 m (7.0 cm)**
* **ODE 2스텝 (최대속도)**: 정형 성공률 **100%** | FPE **0.0722 m (7.2 cm)**
* **ODE 1스텝 (Euler)**: 정형 성공률 68.4% | FPE 0.1345 m (13.4 cm)

# Plan: 조이스틱 기반 데이터 수집기 통합

**날짜:** 2026-05-18  
**브랜치:** monavla-driving  
**상태:** 구현 중

---

## 목표

DragonRise USB 게임패드를 기존 `gradio_data_collector.py`의 비동기 입력 소스로 추가.  
ROS/녹화/H5 저장 로직은 전혀 건드리지 않고, `teleop_step(key)` 호출만 대체.

---

## 현재 시스템 분석

### 기존 이동 흐름

```
키보드 keydown
  → teleop_step(key)
      → publish_cmd_hw(act)   # ROS Twist + HW driver
      → Timer(0.4s) → stop
      → capture_frame_sync(act)  # ROS 이미지 서비스 → 버퍼 추가
```

- 키 1번 = 0.4s 펄스 = 프레임 1개
- 계속 누르려면 수동 반복 필요 → 에피소드 타이밍 불규칙

---

## DragonRise 입력 타입별 특성

| 입력 소스 | 타입 | 값 범위 | 노이즈 | 데드존 필요 |
|-----------|------|---------|--------|------------|
| Left Stick X/Y | Analog axis | −1.0 ~ +1.0 | 있음 (±0.05) | 필수 |
| Right Stick X/Y | Analog axis | −1.0 ~ +1.0 | 있음 | 필수 |
| D-pad (Hat) | Digital tuple | (−1/0/1, −1/0/1) | 없음 | 불필요 |
| A/B/X/Y 버튼 | Digital | 0 or 1 | 없음 | 불필요 |
| L2/R2 트리거 | Analog or Digital | 0.0 ~ 1.0 | 거의 없음 | 상황에 따라 |

---

## 이동 방식 비교

| 항목 | 현재 (키보드) | Left Stick 홀딩 | D-pad 홀딩 | **채택 방식** |
|------|-------------|----------------|-----------|--------------|
| 트리거 | 키 1번 = 1회 | 홀딩 → 0.45s마다 반복 | 홀딩 → 0.45s마다 반복 | **홀딩 반복** |
| 로봇 동작 | 전진→정지→전진 | 동일 (bang-bang 유지) | 동일 | **bang-bang 유지** |
| 대각선 수집 | Q/E 키 (의식적) | 스틱 대각 → 자연스러움 | 어려움 | **Left Stick** |
| STOP 수집 | space 수동 (부자연) | 스틱 release → neutral | release → neutral | **A버튼 명시적 1프레임** |
| 에피소드 타이밍 | 불규칙 | 0.45s 고정 | 0.45s 고정 | **0.45s 고정** |
| ROT_L/R | R/T 키 | Right Stick X | 없음 | **Right Stick X** |
| 조작 난이도 | 높음 | 낮음 | 중간 | **낮음** |

---

## 채택 매핑

### 축 → 액션 (threshold = 0.5)

```
lx =  -js.get_axis(1)   # Left Y 반전 (위 = +lx = FORWARD)
ly =  -js.get_axis(0)   # Left X 반전 (왼쪽 = +ly = LEFT strafe)
az =  -js.get_axis(2)   # Right X 반전

우선순위: 대각 > 직진/스트레이프 > 회전
```

| lx | ly | az | → key | 액션 |
|----|----|----|-------|------|
| ≥0.5 | ≥0.5 |  | `q` | FWD+LEFT |
| ≥0.5 | ≤-0.5 |  | `e` | FWD+RIGHT |
| ≥0.5 |  |  | `w` | FORWARD |
|  | ≥0.5 |  | `a` | LEFT strafe |
|  | ≤-0.5 |  | `d` | RIGHT strafe |
| ≤-0.5 |  |  | `x` | BACKWARD |
|  |  | ≥0.5 | `r` | ROT_L |
|  |  | ≤-0.5 | `t` | ROT_R |
| ≈0 | ≈0 | ≈0 | None | NEUTRAL (발동 안 함) |

### 버튼 매핑

| 버튼 | 기능 |
|------|------|
| A (0) | STOP 명시적 1프레임 저장 (space 대체) |
| B (1) | 마지막 프레임 취소 (undo) |
| Start (7) | teleop_mode 토글 |
| Select (6) | 녹화 시작 / 저장 토글 |
| X (2) | 에피소드 폐기 |

> 실제 버튼 번호는 `calibrate_joystick.py` 실행 후 확정

---

## 구현 구조

### 추가되는 컴포넌트

```
JoystickReader (background thread, 25Hz)
  ├── pygame.event.pump() + get_axis() polling
  ├── _axis_to_key() → threshold snap → key string
  ├── 0.45s 간격으로 node.teleop_step(key) 호출  (홀딩 반복)
  ├── _handle_buttons() → start_rec / stop_rec / toggle_teleop
  └── _status dict → Gradio UI 상태 표시용

joystick_status (shared dict, lock-free read)
  ├── connected: bool
  ├── name: str
  ├── lx, ly, az: float
  └── current_key: str | None
```

### 변경 파일

```
scripts/gradio_data_collector.py   ← JoystickReader 클래스 추가 (~90줄)
                                      UI에 조이스틱 상태 Markdown 1줄 추가
scripts/calibrate_joystick.py      ← 신규 (축 번호 확인 유틸, ~40줄)
```

**기존 코드 무변경:**
- `GradioCollectorNode` 전체
- `teleop_step()`, `capture_frame_sync()`, `publish_cmd_hw()`
- H5 저장 로직, ROS 인프라

---

## 데이터셋 품질 개선 기대효과

| 문제 | 현재 | 개선 후 |
|------|------|---------|
| FORWARD bias 71~74% | space 누르기 번거로워 STOP 안 찍음 | A버튼으로 자연스럽게 STOP 수집 |
| 대각선 FWD+L/R 부족 | Q/E 키 의식적으로 눌러야 | 스틱 대각 = 자연스럽게 발생 |
| 에피소드 타이밍 불규칙 | 사람이 누르는 속도에 의존 | 0.45s 고정 반복 |
| 조작 피로도 | 계속 키 입력 필요 | 홀딩으로 해결 |

---

## 구현 체크리스트

- [ ] `calibrate_joystick.py` 작성
- [ ] `JoystickReader` 클래스 구현
- [ ] `gradio_data_collector.py`에 통합
- [ ] Gradio UI 조이스틱 상태 표시 추가
- [ ] 실기기 테스트 (DragonRise 연결 확인)

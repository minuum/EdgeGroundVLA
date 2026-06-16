# MoNaVLA Gradio Inference Dashboard 가이드

> **명령어:** `vla-inference-gradio` (별칭: `mona-inference-gradio`)
> **스크립트:** `scripts/gradio_inference_dashboard.py`
> **공식 실행 구조:** `vla-camera` + `vla-server` + `vla-inference-gradio`

---

## 목차

1. [시스템 구성](#시스템-구성)
2. [사전 준비](#사전-준비)
3. [실행 방법](#실행-방법)
4. [인터페이스 설명](#인터페이스-설명)
5. [자주 발생하는 오류](#자주-발생하는-오류)
6. [V4 모델 체크포인트 경로](#v4-모델-체크포인트-경로)

---

## 시스템 구성

```
[Jetson Nano]  ←── ROS2 토픽 ───→  [Billy GPU 서버]
  카메라 발행                          API 서버 (FastAPI)
  로봇 제어                            Gradio 대시보드
                                       Kosmos-2 + LoRA
```

- **Jetson IP (Tailscale):** `$JETSON_TAILSCALE_IP`
- **Billy IP (Tailscale):** `$BILLY_TAILSCALE_IP`
- **API 서버 포트:** `8000`
- **Gradio 포트:** `7860`

---

## 사전 준비

### 1. 환경 변수 확인

```bash
# Billy 서버에서
vla-env   # 또는 mona-env

# 필수 환경 변수:
# VLA_PROJECT_DIR=/home/billy/25-1kp/MoNaVLA
# VLA_API_SERVER=http://localhost:8000
# VLA_API_KEY=...
# JETSON_TAILSCALE_IP=100.x.x.x
```

### 2. API 서버 시작 (Gradio 실행 전 권장)

```bash
vla-server   # API 서버 시작
vla-status   # 서버 상태 확인
vla-health   # Health check
```

`vla-inference-gradio`는 API 런타임 프로파일일 때 `VLA_API_SERVER`가 로컬 주소면
서버가 꺼져 있을 경우 `vla-server`를 먼저 자동으로 실행합니다.

### 3. 의존성 확인

```bash
pip install gradio requests opencv-python pillow
```

---

## 실행 방법

### 빠른 실행 (alias 사용)

```bash
# 권장 3터미널 구성
# 터미널 1
vla-camera

# 터미널 2
vla-server

# 터미널 3
vla-inference-gradio

# 또는
mona-inference-gradio
```

> **참고:** alias가 없으면 `.vla_aliases`를 `source`하거나 아래 직접 실행.

### 직접 실행

```bash
cd /home/soda/MoNaVLA
python3 scripts/gradio_inference_dashboard.py
```

### 브라우저 접속

```
http://localhost:7865
# 또는 외부에서: http://<BILLY_TAILSCALE_IP>:7865
```

---

## 인터페이스 설명

| 항목 | 설명 |
|----|------|
| **Inference Backend** | `Local Runtime` 또는 `API Server` 선택 |
| **Model Loader** | checkpoint/config/precision 선택 후 즉시 로드 |
| **Live Inference** | 실시간 카메라 이미지 + 18-step 추론 실행 |
| **Manual Controls** | 3DOF 수동 조작 (`lx`, `ly`, `az`) |
| **Trajectory Plot** | 표준화된 `(N, 3)` chunk 기준 XY 궤적 표시 |

### 액션 인터페이스

- 표준 출력은 항상 **`[lx, ly, az]` 3DOF** 입니다.
- 2DOF/V4 계열 체크포인트는 runtime shim으로 `az=0.0`이 붙습니다.
- `Local Runtime`과 `API Server` 모두 같은 응답 shape를 사용합니다.

---

## 자주 발생하는 오류

### `Connection refused` (API 서버 미실행)

```bash
vla-start    # API 서버 시작
vla-status   # 서버 상태 확인
vla-health   # 헬스체크 확인
```

### `CUDA out of memory`

```bash
# 다른 GPU 프로세스 확인
nvidia-smi
# 학습 중인 경우 학습 일시 중지 후 실행
```

### `Model not loaded` (체크포인트/설정 경로 오류)

API 서버 로그 확인:
```bash
vla-logs   # 또는 mona-logs
```

대시보드에서는 `Load Selected Model` 버튼으로 현재 선택된 checkpoint/config를 직접 API 또는 Local Runtime에 로드할 수 있습니다.

---

## 운영 메모

- `vla-server`는 `robovlm_nav/serve/inference_server.py`를 기준으로 FastAPI 서버를 띄웁니다.
- `vla-inference-gradio`는 `Local Runtime`과 `API Server`를 모두 지원합니다.
- API 런타임 프로파일에서는 `vla-inference-gradio`가 기본 backend를 `API Server`로 사용합니다.
- 공식 alias는 `vla-inference-gradio`, `vla-collect-gradio`, `vla-server` 입니다.
- 기존 `vla-dashboard`, `vla-collect`, `vla-start`는 호환 명령으로 남아 있습니다.

---

## Alias 설치

`.vla_aliases`에 아래 내용이 추가되어 있어야 합니다:

```bash
# Gradio 공식 명령
alias vla-inference-gradio='python3 /home/soda/MoNaVLA/scripts/gradio_inference_dashboard.py'
alias vla-collect-gradio='python3 /home/soda/MoNaVLA/scripts/gradio_data_collector.py'
alias mona-inference-gradio='vla-inference-gradio'
```

설치 확인:
```bash
grep -i "gradio" ~/.bashrc $VLA_PROJECT_DIR/.vla_aliases 2>/dev/null
```

재설치:
```bash
cd $VLA_PROJECT_DIR && bash scripts/install_vla_aliases.sh
```

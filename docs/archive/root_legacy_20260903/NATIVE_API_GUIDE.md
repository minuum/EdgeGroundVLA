# Native API Server 배포 가이드

**A5000 (Billy 서버)**: FastAPI 추론 서버  
**로봇 서버**: ROS2 클라이언트 (API 호출)

---

## 🚀 A5000 서버 설정 (한 번만)

### 1. 자동 설정 스크립트

```bash
cd /home/billy/25-1kp/vla
bash scripts/setup_api_server.sh
```

**실행 내용**:
- Python 환경 확인
- 필수 패키지 설치 (fastapi, uvicorn)
- CUDA/GPU 확인
- 체크포인트 확인
- 방화벽 포트 8000 열기
- IP 주소 확인

### 2. API 서버 시작

```bash
# 포그라운드 실행 (테스트용 - 터미널에서 로그 확인)
python3 Mobile_VLA/inference_server.py

# 백그라운드 실행 (운영용 - 터미널 닫아도 계속 실행)
nohup python3 Mobile_VLA/inference_server.py > logs/api_server.log 2>&1 &

# 프로세스 확인
ps aux | grep inference_server

# 종료 (필요시)
pkill -f inference_server.py
```

### 3. Health Check

```bash
# 로컬 테스트
curl http://localhost:8000/health

# 응답 예시:
# {
#   "status": "healthy",
#   "model_loaded": true,
#   "device": "cuda",
#   "gpu_memory": {
#     "allocated_gb": 8.2,
#     "device_name": "NVIDIA RTX A5000"
#   }
# }
```

### 4. IP 주소 확인

```bash
# 내부 IP (로봇 서버가 같은 네트워크에 있을 때)
hostname -I

# 외부 IP (외부 네트워크에서 접근할 때)
curl ifconfig.me
```

**이 IP를 로봇 서버에 알려주세요!**

---

## 🤖 로봇 서버 설정

### 1. 클라이언트 파일 복사

```bash
# A5000에서 로봇 서버로 복사
scp /home/billy/25-1kp/vla/ros2_client/vla_api_client.py robot@robot_ip:~/
```

또는 Git으로 동기화:

```bash
# 로봇 서버에서
git fetch origin
git checkout feature/deployment-prep
git pull
```

### 2. 환경 변수 설정

```bash
# A5000 서버 IP 지정
export VLA_API_SERVER="http://<a5000_ip>:8000"

# 예시:
export VLA_API_SERVER="http://192.168.1.100:8000"

# 영구 설정 (선택)
echo 'export VLA_API_SERVER="http://192.168.1.100:8000"' >> ~/.bashrc
```

### 3. 네트워크 테스트

```bash
# 테스트 스크립트 복사 (A5000에서)
scp /home/billy/25-1kp/vla/scripts/test_network.sh robot@robot_ip:~/

# 로봇 서버에서 실행
bash test_network.sh <a5000_ip>

# 예시:
bash test_network.sh 192.168.1.100
```

**테스트 항목**:
- Ping 연결
- 포트 8000 열림
- API Health check
- Latency 측정 (< 100ms 권장)

### 4. 클라이언트 테스트

```bash
# Python 클라이언트 테스트
python3 vla_api_client.py --test

# 또는 서버 지정
python3 vla_api_client.py --server http://192.168.1.100:8000 --test
```

**예상 출력**:
```
✅ 서버 연결 성공
   Model loaded: True
   Device: cuda
   GPU: NVIDIA RTX A5000

Left instruction:
  Navigate around obstacles and reach the front of the beverage bottle on the left
  
  결과:
    Action: [1.15 0.319]
    linear_x: 1.150 m/s
    linear_y: 0.319 m/s
    Latency: 45.2 ms
    ✅ Success
```

---

## 📡 API 엔드포인트

### GET /

API 정보

```bash
curl http://<a5000_ip>:8000/
```

### GET /health

서버 상태 확인

```bash
curl http://<a5000_ip>:8000/health
```

### GET /test

더미 데이터로 테스트

```bash
curl http://<a5000_ip>:8000/test
```

### POST /predict

실제 추론 (JSON)

```bash
curl -X POST http://<a5000_ip>:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "image": "<base64_encoded_image>",
    "instruction": "Navigate around obstacles and reach the front of the beverage bottle on the left"
  }'
```

---

## 🧪 통합 테스트

### A5000 서버

```bash
# 1. 서버 실행
python3 Mobile_VLA/inference_server.py

# 2. 자동 테스트 (전체 엔드포인트)
python3 scripts/test_inference_api.py

# 3. Control Center로 모니터링
python3 scripts/control_center.py
```

### 로봇 서버

```bash
# 1. 환경 변수 설정
export VLA_API_SERVER="http://192.168.1.100:8000"

# 2. 네트워크 테스트
bash test_network.sh 192.168.1.100

# 3. 클라이언트 테스트
python3 vla_api_client.py --test

# 4. ROS2와 통합 (실제 로봇)
ros2 run mobile_vla_package vla_demo_client
```

---

## 🔧 문제 해결

### 연결 실패

**증상**: `Connection refused`

**해결**:
1. A5000에서 API 서버 실행 중인지 확인
   ```bash
   ps aux | grep inference_server
   ```

2. 방화벽 포트 8000 열림 확인
   ```bash
   sudo ufw status | grep 8000
   sudo ufw allow 8000/tcp
   ```

3. IP 주소 확인
   ```bash
   hostname -I
   ```

### Latency 높음 (> 100ms)

**원인**:
- 네트워크 거리
- GPU 부하
- 대용량 이미지

**해결**:
1. 이미지 크기 줄이기 (1280x720 → 640x480)
2. A5000 GPU 사용률 확인 (`nvidia-smi`)
3. 네트워크 대역폭 확인

### Model not loaded

**증상**: `"model_loaded": false`

**해결**:
1. 체크포인트 경로 확인
   ```bash
   ls runs/mobile_vla_no_chunk_20251209/checkpoints/
   ```

2. `inference_server.py`에서 경로 수정
   ```python
   DEFAULT_CHECKPOINT = "runs/.../last.ckpt"
   ```

---

## 📊 성능 모니터링

### A5000 서버

```bash
# GPU 사용량
nvidia-smi

# 실시간 모니터링
watch -n 1 nvidia-smi

# API 서버 로그
tail -f logs/api_server.log

# Control Center
python3 scripts/control_center.py
```

### Latency 측정

```bash
# 로봇 서버에서
for i in {1..10}; do
  curl -w "Latency: %{time_total}s\n" -o /dev/null -s http://<a5000_ip>:8000/health
done
```

---

## 🎯 다음 단계

**Day 2 (12/17)**: A5000 서버 설정
- [x] setup_api_server.sh 실행
- [ ] API 서버 시작
- [ ] Health check 확인

**Day 3 (12/18)**: 로봇 서버 통합
- [ ] 클라이언트 배포
- [ ] 네트워크 테스트
- [ ] ROS2 통합 테스트

**Day 4 (12/19)**: 데모
- [ ] 실제 로봇 동작
- [ ] 16시 교수님 데모

---

**작성**: 2025-12-16  
**서버**: A5000 (Billy) Native FastAPI

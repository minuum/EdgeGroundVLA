# Billy 서버 API 실행 요청 (안티그래비티용)

---

## 📋 요청 사항

Billy 서버(A5000)에서 Mobile VLA API 서버를 실행해주세요.

---

## 🎯 Billy 서버에서 실행할 명령어

```bash
# 1. 디렉토리 이동
cd /home/billy/25-1kp/vla

# 2. 현재 브랜치 확인 (이미 feature/inference-integration이어야 함)
git branch

# 3. 필수 파일 확인
ls -lh Mobile_VLA/inference_server.py  # API 서버 코드
ls -lh runs/mobile_vla_no_chunk_20251209/checkpoints/last.ckpt  # 체크포인트

# 4. 방화벽 포트 열기 (최초 1회만)
sudo ufw allow 8000/tcp

# 5. API 서버 실행
mkdir -p logs
nohup python3 Mobile_VLA/inference_server.py > logs/api_server.log 2>&1 &

# 6. 확인
sleep 10
curl http://localhost:8000/health
```

---

## ✅ 예상 결과

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

---

## 📡 Jetson 연결 정보

- **Billy IP**: 223.194.115.11
- **포트**: 8000
- **Jetson에서 테스트**: `curl http://223.194.115.11:8000/health`

---

## 🔧 문제 발생 시

### 로그 확인
```bash
tail -100 logs/api_server.log
```

### 프로세스 확인
```bash
ps aux | grep inference_server
```

### 재시작
```bash
pkill -f inference_server.py
nohup python3 Mobile_VLA/inference_server.py > logs/api_server.log 2>&1 &
```

---

**작성**: 2025-12-16 23:52  
**목적**: Jetson ROS2 클라이언트가 Billy API 서버와 통신  
**Git 작업**: 불필요 (이미 feature/inference-integration 브랜치 사용 중)

# VLA 배포 준비 - Quick Start Guide

**목표**: 목요일 16시 데모를 위한 모든 도구와 인프라 준비 완료

---

## 🎮 Control Center - 메인 대시보드

**모든 것을 한눈에!** 학습 상태, 추론 서버, 데이터셋 검증을 실시간 모니터링

```bash
# 자동 새로고침 대시보드 (5초마다)
python3 scripts/control_center.py

# 한 번만 실행
python3 scripts/control_center.py --once

# 새로고침 간격 변경 (10초)
python3 scripts/control_center.py --refresh 10
```

**표시되는 정보**:
- 📚 학습 프로세스 상태 (PID, CPU, Memory, Progress)
- 🚀 추론 서버 상태 (Running/Not Running, API Endpoints)
- 📊 데이터셋 검증 결과 (Valid/Invalid episodes)
- ⚡ Quick Commands (자주 쓰는 명령어 모음)

---

## 🚀 FastAPI 추론 서버

### 서버 시작

```bash
# 서버 실행
python3 Mobile_VLA/inference_server.py

# 백그라운드 실행
nohup python3 Mobile_VLA/inference_server.py > logs/inference_server.log 2>&1 &
```

### API 엔드포인트

```bash
# 1. Root (API 정보)
curl http://localhost:8000/

# 2. Health Check
curl http://localhost:8000/health

# 3. Test (더미 데이터로 테스트)
curl http://localhost:8000/test

# 4. Predict (실제 추론)
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "image": "base64_encoded_image",
    "instruction": "Navigate around obstacles and reach the front of the beverage bottle on the left"
  }'
```

### 자동 테스트

```bash
# 모든 엔드포인트 자동 테스트
python3 scripts/test_inference_api.py

# 다른 URL 테스트
python3 scripts/test_inference_api.py --url http://192.168.1.100:8000
```

**테스트 항목**:
- ✅ GET / (Root)
- ✅ GET /health (Health check)
- ✅ GET /test (Test endpoint)
- ✅ POST /predict (Left instruction)
- ✅ POST /predict (Right instruction)

---

## 📚 학습 관리

### 학습 시작

```bash
# Frozen VLM 학습 (LoRA 비교용)
bash scripts/train_active/train_frozen_vlm.sh

# LoRA Fine-tuning (Best model)
bash scripts/train_active/train_lora.sh
```

### 학습 모니터링

```bash
# 실시간 모니터링 (progress, loss, ETA)
python3 scripts/monitor_training.py \
  --log logs/train_frozen_vlm_20251216_204511.log

# Loss 히스토리만 보기
python3 scripts/monitor_training.py \
  --log logs/train_frozen_vlm_20251216_204511.log \
  --history

# 로그 직접 확인
tail -f logs/train_frozen_vlm_20251216_204511.log

# GPU 모니터링
watch -n 1 nvidia-smi
```

### 학습 프로세스 관리

```bash
# 프로세스 확인
ps aux | grep main.py

# 프로세스 종료
kill -9 <PID>

# 전체 종료
pkill -9 -f "main.py"
```

---

## 📊 데이터셋 검증

### 검증 실행

```bash
# 전체 데이터셋 검증
python3 scripts/validate_dataset.py

# 특정 패턴만 검증
python3 scripts/validate_dataset.py \
  --pattern "episode_20251207*.h5"

# 보고서 경로 지정
python3 scripts/validate_dataset.py \
  --output docs/my_validation_report.md
```

**검증 항목**:
- 프레임 손상 감지 (그레이스케일 std < 5.0)
- 지지직 프레임 감지 (프레임 간 차이 > 100.0)
- Action 범위 (linear_x: [0.5, 1.5], linear_y: [-1.5, 1.5])
- Episode 길이 (최소 70 frames)

**출력**:
- `docs/dataset_validation_report.md` (Markdown)
- `docs/dataset_validation_report.json` (JSON)

---

## 🎬 에피소드 데모

### 궤적 시각화

```bash
# 기본 (4 episodes: 2 Left + 2 Right)
python3 scripts/demo_episodes.py

# 특정 에피소드 지정
python3 scripts/demo_episodes.py \
  --episodes \
    episode_20251207_061643_1box_hori_left_core_medium.h5 \
    episode_20251207_061651_1box_hori_right_core_medium.h5

# 출력 디렉토리 변경
python3 scripts/demo_episodes.py --output docs/my_demo
```

**출력**:
- `docs/episode_demo/trajectories.png` (XY 궤적 + 속도 플롯)
- 콘솔에 통계 출력 (distance, duration, mean/std)

---

## 🛠️ 전체 워크플로우

### 1. 학습 시작 → 모니터링

```bash
# Terminal 1: 학습 시작
bash scripts/train_active/train_frozen_vlm.sh

# Terminal 2: Control Center 실행
python3 scripts/control_center.py

# Terminal 3: 학습 모니터링
python3 scripts/monitor_training.py --log logs/train_frozen_vlm_*.log
```

### 2. 추론 서버 시작 → 테스트

```bash
# Terminal 1: 서버 시작
python3 Mobile_VLA/inference_server.py

# Terminal 2: 자동 테스트
python3 scripts/test_inference_api.py

# Terminal 3: Control Center로 확인
python3 scripts/control_center.py
```

### 3. 데이터셋 검증 → 에피소드 데모

```bash
# 1. 데이터셋 검증
python3 scripts/validate_dataset.py

# 2. Valid 에피소드 확인 (docs/dataset_validation_report.md)

# 3. 에피소드 데모 실행 (valid episodes만 사용)
python3 scripts/demo_episodes.py \
  --episodes <valid_episode_names>
```

---

## 📁 파일 구조

```
vla/
├── Mobile_VLA/
│   ├── configs/
│   │   ├── mobile_vla_no_chunk_20251209.json       # Best LoRA model
│   │   └── mobile_vla_frozen_vlm_20251216.json     # Frozen VLM (비교용)
│   ├── inference_server.py                         # FastAPI 서버
│   └── inference_pipeline.py                       # 추론 파이프라인
│
├── scripts/
│   ├── control_center.py                           # 🎮 통합 대시보드
│   ├── test_inference_api.py                       # 🧪 API 자동 테스트
│   ├── monitor_training.py                         # 📊 학습 모니터링
│   ├── validate_dataset.py                         # 📋 데이터셋 검증
│   ├── demo_episodes.py                            # 🎬 에피소드 데모
│   └── train_active/
│       └── train_frozen_vlm.sh                     # Frozen VLM 학습
│
├── logs/                                            # 모든 로그
│   ├── train_frozen_vlm_*.log
│   └── inference_server.log
│
└── docs/
    ├── dataset_validation_report.md                # 검증 보고서
    └── episode_demo/                                # 데모 결과
        └── trajectories.png
```

---

## ⚡ Quick Commands Cheat Sheet

```bash
# 🎮 Control Center (가장 중요!)
python3 scripts/control_center.py

# 🚀 서버 시작
python3 Mobile_VLA/inference_server.py

# 🧪 서버 테스트
python3 scripts/test_inference_api.py
curl http://localhost:8000/test

# 📚 학습 시작
bash scripts/train_active/train_frozen_vlm.sh

# 📊 학습 모니터링
python3 scripts/monitor_training.py --log logs/train_*.log

# 📋 데이터셋 검증
python3 scripts/validate_dataset.py

# 🎬 에피소드 데모
python3 scripts/demo_episodes.py

# 🔍GPU 확인
nvidia-smi
watch -n 1 nvidia-smi

# 🛑 프로세스 종료
ps aux | grep main.py
kill -9 <PID>
```

---

## 🎯 목요일 데모 준비 체크리스트

### Day 1 (오늘, 12/16) ✅
- [x] Frozen VLM 학습 시작
- [x] Control Center 구축
- [x] FastAPI 서버 구현
- [x] 자동 테스트 스크립트

### Day 2 (12/17)
- [ ] Frozen VLM 학습 완료 확인
- [ ] LoRA vs Frozen 성능 비교
- [ ] 데이터셋 검증 보고서 리뷰

### Day 3 (12/18)
- [ ] 추론 서버 실제 테스트
- [ ] Latency 측정 (목표: < 50ms)
- [ ] 에피소드 데모 실행

### Day 4 (12/19, 목요일)
- [ ] 최종 검증
- [ ] 16시 데모 준비
- [ ] 🎉 교수님 데모!

---

**작성**: 2025-12-16 21:55  
**주요 도구**: Control Center, FastAPI Server, 자동 테스트

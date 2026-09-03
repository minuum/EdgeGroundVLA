# 서버 간 파일 동기화 가이드

**목표**: Git으로 코드만, rsync로 대용량 파일만 선택적 동기화

---

## 📁 파일 구조

### Billy 서버 (학습/추론)
```
vla/
├── Mobile_VLA/          # 코드 (Git)
├── scripts/             # 코드 (Git)
├── docs/                # 문서 (Git)
├── runs/                # 체크포인트 (rsync) ⚠️
│   └── */checkpoints/*.ckpt
├── ROS_action/          # 소량 데이터
└── .vlms/               # VLM 모델 (한 번만)
```

### Jetson 서버 (로봇/데이터수집)
```
vla/
├── Mobile_VLA/          # 코드 (Git)
├── scripts/             # 코드 (Git)
├── ros2_client/         # 코드 (Git)
├── ROS_action/          # 데이터셋 (rsync) ⚠️
│   ├── last.ckpt        # Billy에서 받은 체크포인트
│   └── mobile_vla_dataset/*.h5  # 수집한 데이터
└── .vlms/               # VLM 모델 (한 번만)
```

---

## 🔄 동기화 워크플로우

### 1. 코드 동기화 (Git)

**Billy 서버**:
```bash
cd /home/billy/25-1kp/vla

# 자동 동기화
bash scripts/sync/sync_code.sh

# 또는 수동
git add -A
git commit -m "feat: update something"
git push origin feature/inference-integration
```

**Jetson 서버**:
```bash
cd ~/vla

# 자동 동기화
bash scripts/sync/sync_code.sh

# 또는 수동
git pull origin feature/inference-integration
```

### 2. 체크포인트 전송 (Billy → Jetson)

**Billy 서버에서 실행**:
```bash
# 자동 선택
bash scripts/sync/push_checkpoint_to_jetson.sh

# 특정 파일 지정
bash scripts/sync/push_checkpoint_to_jetson.sh \
  runs/mobile_vla_no_chunk_20251209/checkpoints/epoch=04-val_loss=0.001.ckpt

# 환경 변수로 Jetson 주소 변경
export JETSON_HOST="soda@jetson_ip"
bash scripts/sync/push_checkpoint_to_jetson.sh
```

**Jetson에서 확인**:
```bash
ls -lh ~/vla/ROS_action/last.ckpt
```

### 3. 데이터셋 전송 (Jetson → Billy)

**Billy 서버에서 실행** (pull 방식):
```bash
# 대화형 선택
bash scripts/sync/pull_dataset_from_jetson.sh

# 옵션:
# 1) 전체 전송
# 2) 특정 날짜 (예: 20251216)
# 3) 최근 N개만
```

**또는 직접 rsync**:
```bash
# 특정 날짜만
rsync -avz --progress \
  soda@100.99.189.94:~/vla/ROS_action/mobile_vla_dataset/episode_20251216*.h5 \
  ROS_action/mobile_vla_dataset/

# 최근 10개만
ssh soda@100.99.189.94 "ls -t ~/vla/ROS_action/mobile_vla_dataset/*.h5 | head -10" | \
  xargs -I {} rsync -avz --progress soda@100.99.189.94:{} ROS_action/mobile_vla_dataset/
```

---

## 🎯 일반적인 시나리오

### 시나리오 1: Billy에서 새 코드 작성 → Jetson에 배포

**Billy**:
```bash
# 1. 코드 수정
vim Mobile_VLA/inference_server.py

# 2. Git 동기화
bash scripts/sync/sync_code.sh

# 3. 새 체크포인트 전송 (필요시)
bash scripts/sync/push_checkpoint_to_jetson.sh
```

**Jetson**:
```bash
# 1. 코드 받기
bash scripts/sync/sync_code.sh

# 2. 새 API 클라이언트 테스트
python3 ros2_client/vla_api_client.py --test
```

### 시나리오 2: Jetson에서 데이터 수집 → Billy에서 학습

**Jetson**:
```bash
# 1. 데이터 수집
ros2 run camera_pub camera_publisher_node

# 2. 데이터 확인
ls -lh ~/vla/ROS_action/mobile_vla_dataset/

# 3. Billy 서버에 알림 (또는 자동화)
```

**Billy**:
```bash
# 1. 데이터 가져오기
bash scripts/sync/pull_dataset_from_jetson.sh
# -> 옵션 2 선택, 날짜: 20251216

# 2. 학습 시작
bash scripts/train_active/train_lora.sh

# 3. 학습 완료 후 체크포인트 전송
bash scripts/sync/push_checkpoint_to_jetson.sh
```

### 시나리오 3: 초기 설정 (VLM 모델 동기화)

**한 번만 실행 (Billy → Jetson)**:
```bash
# VLM 모델은 용량이 크므로 한 번만 전송
rsync -avz --progress \
  /home/billy/25-1kp/vla/.vlms/ \
  soda@100.99.189.94:~/vla/.vlms/
```

---

## 📋 크기 확인

### Billy 서버
```bash
# 체크포인트 크기
du -sh runs/*/checkpoints/

# 전체 runs 디렉토리
du -sh runs/
```

### Jetson 서버
```bash
# 데이터셋 크기
du -sh ~/vla/ROS_action/mobile_vla_dataset/

# 파일 개수
ls ~/vla/ROS_action/mobile_vla_dataset/*.h5 | wc -l
```

---

## 🔐 환경 변수 설정

**~/.bashrc에 추가** (각 서버):

**Billy**:
```bash
# Jetson 접속 정보
export JETSON_HOST="soda@100.99.189.94"
export JETSON_PATH="~/vla"
```

**Jetson**:
```bash
# Billy 접속 정보
export BILLY_HOST="billy@100.99.189.94"
export BILLY_PATH="/home/billy/25-1kp/vla"

# API 서버
export VLA_API_SERVER="http://100.99.189.94:8000"
export VLA_API_KEY="your-api-key"
```

---

## ✅ 체크리스트

### 초기 설정 (한 번만)
- [x] .gitignore 설정 (대용량 파일 제외)
- [ ] 양쪽 서버에 Tailscale 설치
- [ ] SSH 키 교환 (비밀번호 없이 접속)
- [ ] 환경 변수 설정
- [ ] sync 스크립트 실행 권한

### 일상 작업
- [ ] 코드 수정 후 `sync_code.sh` 실행
- [ ] 새 체크포인트 → Jetson
- [ ] 새 데이터셋 → Billy
- [ ] Git 충돌 주의

---

## 🚀 자동화 (고급)

### Cron으로 자동 동기화

**Billy (매일 자정에 데이터셋 가져오기)**:
```bash
crontab -e

# 추가
0 0 * * * cd /home/billy/25-1kp/vla && bash scripts/sync/pull_dataset_from_jetson.sh auto
```

**Jetson (매시간 코드 업데이트)**:
```bash
crontab -e

# 추가
0 * * * * cd ~/vla && git pull origin feature/inference-integration
```

---

**작성**: 2025-12-16  
**업데이트**: 필요 시

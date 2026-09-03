# VLA Aliases Quick Start

## 🚀 빠른 설치 (추천)

```bash
cd ~/25-1kp/vla
bash scripts/install_vla_aliases.sh
```

이 스크립트가 자동으로:
- `.bashrc`에 alias 추가
- 현재 세션에 즉시 적용
- 도움말 표시

---

## 📋 수동 설치

```bash
# ~/.bashrc 파일 끝에 추가
echo '' >> ~/.bashrc
echo '# VLA Project Aliases' >> ~/.bashrc
echo 'source ~/25-1kp/vla/.vla_aliases' >> ~/.bashrc

# 적용
source ~/.bashrc
```

---

## 💡 주요 명령어

### 즉시 사용 가능
```bash
vla-help              # 전체 도움말
vla-overview          # 프로젝트 상태 한눈에
vla-server-start      # 추론 서버 시작
vla-gpu-quick         # GPU 상태 확인
vla-tb                # Tensorboard 시작
```

### 학습 관찰
```bash
vla-log               # 학습 로그 실시간
vla-ckpt              # 최근 체크포인트
vla-train-status      # 학습 진행 상황
```

### 서버 관리
```bash
vla-server-start      # 서버 시작
vla-server-stop       # 서버 종료  
vla-server-status     # 상태 확인
vla-server-health     # 헬스 체크
```

---

## 📖 상세 가이드

전체 alias 목록 및 사용법:
- `vla-help` 명령어
- `docs/VLA_ALIASES_GUIDE.md` 문서

---

## 🎯 자주 사용하는 조합

### 미팅 준비
```bash
vla && vla-meeting-ready
```

### 전체 모니터링 (3개 터미널)
```bash
# 터미널 1
vla-log

# 터미널 2  
vla-gpu

# 터미널 3
vla-tb
```

### 서버 관리
```bash
vla-server-status     # 상태 확인
vla-server-restart    # 재시작
vla-api-test          # 테스트
```

---

**설치 완료 후 `vla-help`로 모든 명령어를 확인하세요!**

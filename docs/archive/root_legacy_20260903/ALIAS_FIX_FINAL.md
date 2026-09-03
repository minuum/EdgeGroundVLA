# VLA Alias 완전 수정 완료 ✅

## 📅 작성: 2025-12-17 18:55

## 🔧 최종 수정 사항

### 1. **ROS 환경 경로 수정** (`~/.zshrc`)

#### Before:
```bash
alias vla-ros-env="cd /home/soda/vla/ROS_action && source /opt/ros/humble/setup.bash && source install/setup.bash && echo \"VLA 환경 설정 완료!\""
```

#### After:
```bash
alias vla-ros-env="cd /home/soda/vla/ROS_action/install && source /opt/ros/humble/setup.bash && source local_setup.bash && echo \"VLA 환경 설정 완료!\""
```

**변경 이유**: 
- `local_setup.bash`는 `ROS_action/install/` 디렉토리에 있음
- 실행 전에 반드시 해당 디렉토리로 이동해야 함

### 2. **Alias 충돌 방지** (`.vla_aliases`)

함수 정의 전에 기존 alias 제거:
```bash
# vla-env 함수 정의 전
unalias vla-env 2>/dev/null || true
vla-env() {
    ...
}

# vla-help 함수 정의 전  
unalias vla-help 2>/dev/null || true
vla-help() {
    ...
}
```

## ✅ 테스트 결과

```bash
source ~/.zshrc
# ✓ VLA API Server aliases loaded
#   Run 'vla-help' for available commands

vla-env
# VLA Environment Variables:
#   VLA_PROJECT_DIR  = /home/soda/vla
#   VLA_API_SERVER   = http://localhost:8000
#   VLA_API_KEY      = jFLQzbwEch8_S2lpioP6...

vla-help
# ==========================================
# VLA API Server Commands
# ==========================================
# ...
```

## 📋 최종 명령어 정리

### API 서버 관리 (`.vla_aliases`)
| 명령어 | 설명 | 타입 |
|--------|------|------|
| `vla-env` | API 환경 변수 표시 | Function |
| `vla-help` | 도움말 표시 | Function |
| `vla-start` | 서버 시작 | Alias |
| `vla-stop` | 서버 중지 | Alias |
| `vla-restart` | 서버 재시작 | Alias |
| `vla-status` | 서버 상태 | Alias |
| `vla-logs` | 로그 보기 | Alias |
| `vla-test` | 전체 테스트 | Alias |
| `vla-health` | Health check | Alias |
| `vla-curl-health` | 빠른 health check | Alias |
| `vla-curl-test` | 빠른 test | Alias |
| `vla-ps` | 프로세스 확인 | Alias |
| `vla-cd` | 프로젝트로 이동 | Alias |
| `vla-gpu` | GPU 상태 | Alias |

### ROS2 관련 (`~/.zshrc`)
| 명령어 | 설명 | 타입 |
|--------|------|------|
| `vla-ros-env` | ROS2 환경 설정 | Alias |
| `vla-collect` | 데이터 수집 | Alias |
| `robot-move` | 로봇 이동 | Alias |
| `vla-system` | VLA 시스템 | Alias |

## 🚀 사용 방법

### 새 터미널에서
```bash
# Shell 재로드 (자동으로 .vla_aliases 로드됨)
source ~/.zshrc

# API 환경 확인
vla-env

# 도움말 보기
vla-help

# 서버 시작
vla-start

# 서버 상태
vla-status

# ROS2 환경 설정
vla-ros-env  # 자동으로 ROS_action/install로 이동하고 setup
```

## 📝 주요 변경 사항 요약

### ✅ 해결된 문제
1. ~~`vla-env` alias/function 충돌~~ → `unalias`로 해결
2. ~~ROS 경로 오류~~ → `ROS_action/install`로 수정
3. ~~`vla-help` 작동 안 함~~ → `unalias`로 해결

### ✅ 적용된 수정
1. `~/.zshrc`: `vla-env` → `vla-ros-env` 변경
2. `~/.zshrc`: ROS 경로를 `ROS_action/install`로 수정
3. `.vla_aliases`: 함수 정의 전 `unalias` 추가

## 🎯 최종 확인 체크리스트

- [x] `source ~/.zshrc` 에러 없이 실행
- [x] `vla-env` 정상 작동 (API 환경 변수 표시)
- [x] `vla-help` 정상 작동
- [x] `vla-ros-env` ROS 경로 수정 완료
- [x] 모든 `vla-*` alias 정상 작동
- [x] API 서버 스크립트 정상 작동

## 💡 참고

### ROS2 환경을 사용할 때
```bash
# 옵션 1: vla-ros-env 사용 (권장)
vla-ros-env
ros2 topic list

# 옵션 2: 수동 설정
cd /home/soda/vla/ROS_action/install
source /opt/ros/humble/setup.bash
source local_setup.bash
```

### API 서버 빠른 시작
```bash
vla-start     # 서버 시작
vla-test      # 테스트
vla-status    # 상태 확인
```

---

**상태**: ✅ 완료  
**테스트**: ✅ 통과  
**버전**: 2.0 (최종)  
**작성**: 2025-12-17 18:55

# Plan: 시스템 문제점 인벤토리 + 점검 루틴

> 상태: **작성 완료 — 사용자 검토 대기**
> 작성: 2026-07-02 (하루 종일의 디버깅 세션에서 실측된 문제들 기반)

---

## 1. 문제점 인벤토리

### A. 해결됨 (오늘, 재발 감시 필요)

| # | 문제 | 수정 커밋 | 재발 감시 방법 |
|---|------|-----------|---------------|
| A1 | SYNC/PRE 모드 az=0 고정 (2D action 사용) → "도리도리" 실패군 | `45fcff07` | 세션 덤프에서 ROT_L/R 액션이 실제 기록되는지 |
| A2 | 프리뷰 R/L 교대 스윕 → 순 회전량 0 (RLRLR) | `db77ff12` | 서버 로그 `[CH54] preview ROT` 방향 확인 |
| A3 | 멀티프롬프트가 주행 중 매 스텝 8~12s 소모 | `8cfa7198` | live 프레임 lat이 2.1~2.2s로 고정인지 |
| A4 | 좀비 프로세스가 옛 MJPEG 스트림 계속 서빙 | `fdb85e59` | 헤더 PID/가동시간 배지 확인 |
| A5 | 카메라 멈춤인데 camera_ok=true (신선도 미확인) | `211e7fd9` | frame_age_s < 3s 판정 적용됨 |
| A6 | 재시작마다 로그 덮어쓰기 → 과거 분석 불가 | `744f4936` | logs/*.타임스탬프.log 보존 확인 |
| A7 | bbox 사각형 미표시 (xmin vs x1 필드명) | `2cb77dee` | — |
| A8 | 세션-메모 매칭 불가 (session_id 없음) | `1a0addff` | — |
| A9 | 런타임 설정이 세션에 안 남음 | `744f4936` | H5 attrs["runtime_config"] |

### B. 미해결 — 근본 원인 (우선순위순)

| # | 문제 | 증거 | 대응 방향 |
|---|------|------|-----------|
| **B1** | **PG2 그라운딩 탐지율 15~20%** — full-frame 환각, 저대비 씬, cx>0.7 학습분포 7.8% | 3세션 실측, obj_right 0/6 | minum PG2 파인튜닝 (진행 중). soda측: 종횡비(16:9→1:1 강제 리사이즈) 학습-추론 일치 여부 확인 |
| **B2** | **API 키 드리프트** — 서버/대시보드를 다른 셸에서 재시작하면 403 연쇄 → STOP만 발행 | 오늘 2회 발생 | go.sh에 기본값 넣었지만, `.zshrc` 키와 다른 셸에서 실행하면 여전히 어긋남. **키를 파일 하나(.vla_api_key 등)로 단일화** 필요 |
| **B3** | **usb_camera_service_server 간헐 행(hang)** — 응답은 하되 같은 프레임 반복 | 오늘 3회 재시작 필요 | 원인 미상 (장시간 가동 + DDS 유령 커넥션 추정). 카메라 서비스 자체에 워치독/자기재시작 필요 |
| B4 | 런타임 토글이 실제 적용됐는지 불투명 — "3세션 설정 다르게 했는데 동일" | skip_n [3,3,3] 실측 | H5 스냅샷(A9)으로 사후 확인은 가능해짐. **토글 직후 /health 재조회로 UI에 실제값 표시**가 다음 단계 |
| B5 | bbox 피처가 모델에 거의 기여 안 함 (ablation +1.1pp) | CLAUDE.md ablation | B1 해결 전엔 보류 (bbox 자체가 대부분 없음). 이후 게이팅 브랜치 검토 |
| B6 | 대시보드 단일 프로세스에 UI+ROS+하드웨어 결합 — UI 재시작 = 제어 중단 | 아키텍처 | 3-tier 분리 (기존 plan 문서 있음, Phase 2 미착수) |

### C. 잠재 리스크 (아직 안 터졌지만 터질 곳)

| # | 리스크 | 트리거 조건 | 선제 대응 |
|---|--------|------------|-----------|
| C1 | **디스크 고갈** — 세션 H5가 개당 30~50MB, 로그도 이제 무한 보존 | 세션 수십 개 누적 | 주간 정리 루틴에 포함 (아래 §3) |
| C2 | **Jetson 메모리 압박** — 15GB 중 13GB 사용 중, swap 1GB | 프로세스 추가/leak | precheck에 free 확인 포함 |
| C3 | episode_log.csv 스키마 변경(14컬럼) — minum측 구버전 파서 깨짐 가능 | minum이 receive 스킬로 파싱 시 | minum에게 session_id 컬럼 추가 공유 (이번 push에 포함됨) |
| C4 | 프리뷰 응답 grounding_latency_ms=0 하드코딩 — 분석 시 오독 | 세션 분석 때마다 | 낮은 우선순위 수정 대상 |
| C5 | camera_pub 재시작 시 대시보드 카메라 루프가 오래된 서비스 핸들 잡고 있을 가능성 | 카메라 재시작 직후 | 재시작 후 frame_age_s 확인 습관화 |
| C6 | `.menemory` 파일이 항상 dirty — 실수로 커밋에 섞일 위험 | git add -A 사용 시 | 커밋 시 명시적 파일 지정 유지 (현재 방식 OK) |

---

## 2. 주행 전 점검 루틴 (매 주행 세트 시작 전, ~30초)

```bash
# ① 서비스 3종 + 프로세스 신선도
bash scripts/run/go.sh --status
curl -s localhost:7800/health | python3 -m json.tool
#    확인: camera_ok=true, frame_age_s<1, pid/started_at이 방금 재시작한 값인지

# ② API 키 일치 (B2 재발 감시 — 403의 전조)
for p in $(pgrep -f "stage2_v2_inference_server|mona_dashboard.py"); do
  echo "PID $p: $(cat /proc/$p/environ | tr '\0' '\n' | grep VLA_API_KEY)"
done
#    확인: 두 값이 동일한지

# ③ 런타임 설정이 의도대로인지 (B4)
curl -s localhost:7800/infer/health | python3 -c "
import sys,json; d=json.load(sys.stdin)
print({k:d.get(k) for k in ['grounding_skip_n','multi_prompt','cx_jump_filter']})
print('preview:', d.get('preview'))"

# ④ 리소스 (C2)
free -h | head -2
df -h /home | tail -1
```

→ **하나라도 이상하면 주행 시작 전에 잡는다.** ①~④를 `scripts/run/precheck.sh`로
스크립트화하는 것을 제안 (승인 시 구현).

## 3. 주행 세트 후 루틴 (에피소드 기록 직후)

```bash
# ① 방금 세션 자동 분석 (런타임 설정 스냅샷 + 메모 + grounding 손실 구간 포함)
python3 scripts/analysis/analyze_episode_log.py --recent

# ② 확인 포인트
#    - "런타임 설정 스냅샷"이 의도한 설정과 일치하는가 (B4)
#    - live 프레임 lat이 2.1~2.2s인가 (8s+ 나오면 멀티프롬프트 어딘가 다시 켜진 것, A3)
#    - frame_duplicate_warning이 로그에 있는가 (B3 카메라 행 감지)
grep "동일한 이미지" logs/mona_dashboard.log | tail -3

# ③ 분석 가치 있는 세션은 minum 전송
bash scripts/sync/push_inference_session_to_minum.sh -n <개수>
```

## 4. 주간 루틴 (주 1회)

```bash
# ① 로그/세션 정리 (C1) — 2주 지난 타임스탬프 로그, 분석 끝난 세션 정리
find logs -name "*.2026*.log" -mtime +14 -ls        # 확인 후 삭제
du -sh docs/inference_sessions/                     # 총량 확인

# ② 에피소드 통계 리뷰 — 성공률 추세, 실패 유형 변화
python3 scripts/analysis/analyze_episode_log.py

# ③ 브랜치 동기화 — minum 커밋 확인 후 병합/푸시
git fetch origin && git log --oneline HEAD..origin/monavla-driving
```

## 5. 다음 액션 제안 (승인 대기)

1. **precheck.sh 스크립트화** (§2를 원커맨드로) — 작음, 즉시 가능
2. **API 키 단일화** (B2) — `.vla_api_key` 파일 하나에서 서버/대시보드/go.sh 모두 읽기
3. **camera_pub 워치독** (B3) — 대시보드가 frame_age_s>10s 지속 시 카메라 서비스 자동 재시작 (이미 /camera_proc/start API 있음 — 내부 호출만 추가)
4. **토글 직후 실제값 재조회 표시** (B4) — /config 응답의 applied를 UI에 그대로 보여주기
5. 종횡비 학습-추론 일치 여부 확인 (B1 보조) — minum과 협의 필요

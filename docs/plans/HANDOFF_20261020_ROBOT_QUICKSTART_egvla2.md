# 로봇/젯슨 세션 퀵스타트 — EGVLA2 점검 (2026-10-20, minum 작성)

> 상세 근거: `HANDOFF_20261013_continuous_collection_for_robot_session.md` §15, `plan_20261019_egvla2_inference_server.md`. **이 단계에서 로봇은 구동하지 않는다(로그만).**

## 0. 한 줄 요약
EGVLA2 배포 체크포인트(C + 서브골 드롭아웃30%)를 가져가 서버 경로 점검 → 지연 측정 → 드라이런(출력 로그만)까지 하고 결과를 회신해 달라. 구동·폐루프는 minum 승인 후.

## 1. 가져가기 (순서대로)
1. **코드 최신화**: `git pull` (branch `inference-integration`, 이 문서가 있는 커밋 이후). 로컬 수정(`mona_dashboard.py` 등)이 있으면 먼저 알려 줘. **서버 `/predict` 응답에서 `action_cont`·`drive`가 잘리던 버그를 수정한 버전이어야 한다(`InferenceResponse`에 필드 추가).**
2. **체크포인트**: `rsync -ah --progress minum:/home/minum/26CS/MoNaVLA/runs/egvla2_fm/deploy/ <로봇측 repo>/runs/egvla2_fm/deploy/` 후 `sha256sum *.pt`를 repo의 `runs/egvla2_fm/deploy/MANIFEST.json`과 대조(권장: `C_pi0_adaln_drop30_e120.pt`, sha256 `f88b944f…`). rsync에 `--delete` 금지.
3. Florence-2 base 가중치가 로컬 캐시에 있는지 확인(`HF_HUB_OFFLINE=1`로 로드되어야 함).

## 2. 필수 입력 조건 (안 지키면 결과가 틀어진다)
- 대시보드에서 서버로 보내는 JPEG: **`VLA_JPEG_QUALITY=95`** (기본 80이면 방향이 평균 4.1°·최대 17° 어긋남, q95는 0.2°).
- 색 보정 `apply_cc` **끔**. dtype은 fp32(메인), 서버·대시보드 모두 최신 코드.

## 3. 점검 3종 (구동 없음)
**A. 서버 경로 점검** — 서버를 평활 없이 띄우고(기존 서버 사용 시 `VLA_ACTION_MODE=continuous VLA_EGVLA2_CKPT=runs/egvla2_fm/deploy/C_pi0_adaln_drop30_e120.pt VLA_SPEED_SOURCE=shadow VLA_EGVLA2_EMA=0`, 단독 서버는 `python -m robovlm_nav.serve.egvla2_inference_server --port 8001`) 같은 프레임으로 비교:

`python3 scripts/robot_check_server_vs_local.py --h5 <수집 H5 폴더> --ckpt runs/egvla2_fm/deploy/C_pi0_adaln_drop30_e120.pt --server http://127.0.0.1:8001 --n 20 --quality 95 --dtype fp32` (API 키는 `VLA_API_KEY` 환경변수에서 읽음)

**합격**: `pass: true`(bbox 차 <0.01, 서버-로컬 방향 차 ≤ 로컬 샘플링 잡음×1.5+0.5°). 결과 JSON 회신. (minum에서 단독 서버로 확인: bbox 차 0.0005, 방향 차 2.6° vs 잡음 2.8° → pass)

**B. 지연 측정** — 젯슨 fp32, 위 서버로 프레임 100개의 `latency_ms`와 `grounding_latency_ms` 기록(평균·p95). 참고: minum 서버(fp32, HTTP·디코드 포함) 약 0.49~0.59초, Florence-2 단독 벤치 젯슨 817ms. **구간별(디코드/그라운딩/FM 샘플링) 시간이 있으면 같이.**

**C. 드라이런** — 서버 `VLA_EGVLA2_EMA=0.5`, 대시보드 **구동 없이**(로봇 정지) 사람이 조작하는 동안 응답의 `action_cont`·`drive`·`action_3d`·`subgoal_missing`·`bbox`·`chunk_first`·`spread`를 프레임마다 로그(시각, 프레임 번호, 사람 조작 값 포함). 최소 에피소드 5개(목표 위치 5개 각 1개).

## 4. 회신 항목
1. git 상태(`git log --oneline -3`, 로컬 수정 여부) 2. rsync·sha256 일치 3. A의 `pass`와 JSON 4. B의 지연 평균/p95와 구간별 5. C의 로그 파일 경로 6. 막힌 점(의존성, 가중치, 포트, API 키).

## 5. 하지 말 것 / 아직 모르는 것
- `VLA_DRIVE_SOURCE=egvla2_hybrid`로 **구동하지 말 것**(드라이런 로그 확인 후 minum이 승인). 폐루프(3 시작 위치×5 목표 위치×20)도 승인 후.
- 미확인: gain 1.15가 컨트롤러 lx/ly 단위와 맞는지(대각선 속도는 직진과 같다고 가정), EMA 값, 회전 탐색 중 대시보드 동작. 드라이런 로그로 확인.

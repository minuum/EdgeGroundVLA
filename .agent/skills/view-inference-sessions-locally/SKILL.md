---
name: view-inference-sessions-locally
description: inference_sessions_recv/에 받은 실주행 세션(H5 이미지 포함)을 mona_dashboard.py 로컬 웹뷰어로 프레임 단위 비교·대조한다. git-lfs 없이 이미지를 직접 보고 싶을 때 사용. "세션 웹으로 보여줘", "로컬에서 비교해줘", "대시보드 띄워줘" 요청에 사용.
---

# View Inference Sessions Locally

`receive-inference-session`으로 받은 H5 세션들을 커밋/git-lfs 없이 브라우저에서
프레임 단위(이미지+bbox+액션+episode_log 결과)로 열어보기 위한 절차.
`mona_dashboard.py`의 Session History / Frame-by-Frame Inspector를 재사용한다.

## 전제

- 세션 데이터는 `/home/minum/MoNaVLA/inference_sessions_recv/<날짜>/`에 있음 (git 저장소 밖,
  soda→minum rsync 랜딩존). 이 디렉토리 자체는 git 대상이 아니라 git-lfs 걱정 없음.
- 대시보드(`robovlm_nav/serve/mona_dashboard.py`)는 레포 안 `docs/inference_sessions/`
  (H5)와 `logs/episode_log.csv`만 읽는다 — 둘 다 `.gitignore`에 이미 포함되어 있어
  (`*.h5`, `logs/`) 심볼릭 링크/복사해도 git에 안 걸린다.

## Step 1. 신규 세션을 대시보드가 보는 경로로 연결

```bash
RECV_DATE=$(ls -t /home/minum/MoNaVLA/inference_sessions_recv | head -1)
RECV=/home/minum/MoNaVLA/inference_sessions_recv/$RECV_DATE

# H5는 심볼릭 링크 (복사 안 함 — 427M 등 대용량 중복 방지)
for f in "$RECV"/session_*.h5; do
  ln -sf "$f" docs/inference_sessions/"$(basename "$f")"
done

# episode_log.csv는 누적본이라 통째로 덮어써도 됨 (RECV 쪽이 항상 최신 누적본)
mkdir -p logs
cp "$RECV"/episode_log.csv logs/episode_log.csv
```

## Step 2. 의존성 확인 (최초 1회)

레포 기본 파이썬 환경에 `h5py`/`fastapi`가 없을 수 있음 — 전용 venv 사용:

```bash
python3 -m venv /tmp/h5env   # 이미 있으면 스킵
/tmp/h5env/bin/pip install -q h5py numpy fastapi uvicorn pillow python-multipart
```

## Step 3. 대시보드 실행

> ⚠️ Claude Code 자동실행 제한으로 `0.0.0.0` 바인딩 불가 — **`127.0.0.1`로 로컬 전용 실행**.

```bash
nohup /tmp/h5env/bin/python robovlm_nav/serve/mona_dashboard.py \
    --port 7800 --host 127.0.0.1 > /tmp/mona_dash.log 2>&1 &
disown
sleep 3 && curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:7800/
```

이미 떠 있으면(`WARNING: 중복 프로세스 감지`) 새로 띄우지 말고 기존 것 재사용.
확인:
```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:7800/ 2>/dev/null || echo down
```

## Step 4. 사용자 안내

접속 주소: **http://127.0.0.1:7800/** → Session History 탭 → 원하는 세션 클릭
→ Frame-by-Frame Inspector에서 슬라이더로 프레임별 이미지·bbox(cx,cy,area)·액션·
episode_log 결과(성공/실패, 메모)를 한 화면에서 비교.

## 종료

```bash
pkill -f "mona_dashboard.py --port 7800"
```

## 관련 스킬

- `receive-inference-session` — 세션 수신·수치 분석(이 스킬의 전 단계)
- `docs-server` — 정적 `docs/` 폴더 서빙 (이 스킬과는 다른 용도: 대시보드는 동적 API 필요)

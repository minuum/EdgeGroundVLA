---
name: 로컬뷰어
description: 사용자가 "로컬뷰어"라고만 입력하면 세션 대시보드(mona_dashboard, 7800)와 분석 리포트 정적서버(docs/, 8766)를 한 번에 띄우고 URL을 안내한다. 이미 떠있으면 재실행하지 않고 URL만 확인해서 알려준다.
---

# 로컬뷰어

`view-inference-sessions-locally`(세션 프레임 브라우징)와 분석 리포트(HTML, 예:
`docs/v5/analysis_reports/*.html`)를 매번 따로 켜지 않도록 묶은 원클릭 트리거.
**"로컬뷰어"** 한 마디만 입력해도 아래를 자동 수행.

## 트리거

사용자가 정확히 "로컬뷰어" (또는 "로컬뷰어 띄워줘/켜줘") 라고만 입력.

## Step 1. 이미 떠있는지 확인

```bash
curl -s -o /dev/null -w "dashboard(7800): %{http_code}\n" http://127.0.0.1:7800/ 2>/dev/null || echo "dashboard(7800): down"
curl -s -o /dev/null -w "report(8766): %{http_code}\n"    http://127.0.0.1:8766/ 2>/dev/null || echo "report(8766): down"
```

200이 아니면(down) 아래 Step 2/3에서 해당 서버만 새로 띄운다. 둘 다 200이면 바로
Step 4 URL 안내로 건너뛴다.

## Step 2. 세션 대시보드 (mona_dashboard, 7800) — down일 때만

```bash
# 최신 수신 세션을 대시보드가 읽는 경로에 연결 (최초 1회 또는 신규 세션 도착 시)
RECV_DATE=$(ls -t /home/minum/MoNaVLA/inference_sessions_recv | head -1)
RECV=/home/minum/MoNaVLA/inference_sessions_recv/$RECV_DATE
for f in "$RECV"/session_*.h5; do
  ln -sf "$f" ~/26CS/MoNaVLA/docs/inference_sessions/"$(basename "$f")"
done
mkdir -p ~/26CS/MoNaVLA/logs
cp "$RECV"/episode_log.csv ~/26CS/MoNaVLA/logs/episode_log.csv

# venv 없으면 최초 1회 생성
[ -x /tmp/h5env/bin/python ] || { python3 -m venv /tmp/h5env; /tmp/h5env/bin/pip install -q h5py numpy fastapi uvicorn pillow python-multipart; }

cd ~/26CS/MoNaVLA
nohup /tmp/h5env/bin/python robovlm_nav/serve/mona_dashboard.py \
    --port 7800 --host 127.0.0.1 > /tmp/mona_dash.log 2>&1 &
disown
```

## Step 3. 분석 리포트 정적 서버 (docs/, 8766) — down일 때만

```bash
cd ~/26CS/MoNaVLA/docs
nohup python3 -m http.server 8766 --bind 127.0.0.1 > /tmp/report_server.log 2>&1 &
disown
```

> ⚠️ 반드시 `127.0.0.1` 바인딩만 사용 (Claude Code 자동실행 정책상 `0.0.0.0` 불가 —
> 로컬 전용 뷰어 용도라 문제 없음).

## Step 4. 사용자에게 안내

```bash
sleep 2
curl -s -o /dev/null -w "dashboard: %{http_code}\n" http://127.0.0.1:7800/
curl -s -o /dev/null -w "report:    %{http_code}\n" http://127.0.0.1:8766/
```

두 URL을 함께 안내:
- **세션 프레임 브라우징**: http://127.0.0.1:7800/ (Session History 탭)
- **최신 분석 리포트**: http://127.0.0.1:8766/v5/analysis_reports/ 아래 최신 html
  (파일명이 바뀔 수 있으니 `ls -t docs/v5/analysis_reports/*.html | head -1`로 최신 것 확인 후 안내)

## 종료 (요청 시)

```bash
pkill -f "mona_dashboard.py --port 7800"
pkill -f "http.server 8766"
```

## 관련 스킬

- `view-inference-sessions-locally` — 세션 대시보드 단독 실행 절차(이 스킬의 Step 2 상세)
- `receive-inference-session` — 애초에 세션을 받아 수치 분석하는 전 단계
- `docs-server` — 8765 포트로 docs/ 전체를 서빙하는 별개 스킬(포트 다름, 이 스킬과 동시 사용 가능)

---
name: verify-local
description: MoNaVLA 로컬 검증 대시보드를 127.0.0.1:9001 에 띄우고 링크 안내. CH57 사진 확인, 반증 테스트 판단, GitHub Pages 배포 확인, 학습 로그 등을 브라우저에서 체크리스트로 검증. "검증 페이지", "verify", "로컬 대시보드" 등의 요청에 사용.
---

# verify-local 스킬

로컬에서만 접근 가능한 검증 대시보드를 생성하고 `127.0.0.1:9001` 에 서빙한다.
푸시하지 않음 — 개인 검증 전용.

## 실행 절차

### 1. 기존 서버 종료 (포트 충돌 방지)

```bash
pkill -f mona_verify_server.py 2>/dev/null; sleep 0.5
```

### 2. HTML 생성

`/tmp/mona_verify.html` 에 대시보드 HTML 작성.

포함 섹션:
- **§CH57** — `docs/v5/ch57_frames/` 이미지 6장 + ✅/❌/⬜ 체크버튼
- **§K-4** — bbox 반증 테스트 결과 판단
- **§Pages** — GitHub Pages 링크 (grounding_hub / research_story)
- **§학습** — `/tmp/ablate_ft_fix.log` 마지막 8KB 표시
- **요약** — 체크 결과 집계 + 클립보드 복사

### 3. 서버 실행

```python
# /tmp/mona_verify_server.py
import http.server, pathlib, urllib.parse

FRAMES_DIR = pathlib.Path("docs/v5/ch57_frames")  # 절대경로로 변환
HTML_FILE  = pathlib.Path("/tmp/mona_verify.html")
LOG_FILE   = "/tmp/ablate_ft_fix.log"

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        p = urllib.parse.urlparse(self.path).path
        if p in ("/", "/index.html"):
            self._send(HTML_FILE.read_bytes(), "text/html; charset=utf-8")
        elif p.startswith("/ch57_frames/"):
            fp = FRAMES_DIR / p.split("/ch57_frames/")[1]
            self._send(fp.read_bytes() if fp.exists() else b"", "image/jpeg")
        elif p == "/log":
            try: data = open(LOG_FILE,"rb").read()[-8000:]
            except: data = b"(no log)"
            self._send(data, "text/plain; charset=utf-8")
        else:
            self.send_response(404); self.end_headers()
    def _send(self, data, ct):
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.end_headers()
        self.wfile.write(data)
    def log_message(self, *a): pass

http.server.HTTPServer(("127.0.0.1", 9001), Handler).serve_forever()
```

백그라운드 실행:
```bash
python3 /tmp/mona_verify_server.py &
sleep 1
curl -s http://127.0.0.1:9001/ | head -1  # 확인
```

### 4. 링크 안내

```
http://127.0.0.1:9001
```

사용자에게 위 링크 전달 후 완료.

## 검증 항목 목록

| ID | 항목 | 판단 기준 |
|----|------|-----------|
| r0 | frame1 바스켓이 실제 오른쪽에 있는가 | 사진 육안 |
| r1 | frame0→1 장면 급변 여부 | 콜드스타트 흔적 |
| r2 | 정상 세션 frame1 예측 정상인가 | 비교군 확인 |
| r3 | CH57 bbox 오염 가설 폐기 동의 | 반증 테스트 기반 |
| r4 | grounding_hub §K 사진 노출 | Pages 배포 |
| r5 | research_story Hero 링크 노출 | Pages 배포 |
| r6 | OWL-v2 LoRA 5 seed 완료 | 학습 로그 |

## 업데이트 방법

새 검증 항목 추가 시:
1. HTML의 verdict div 추가 (`id="rN"` 증가)
2. `updateScore()` 의 `total` 값 갱신
3. 이 SKILL.md 의 검증 항목 테이블 업데이트

## 주의

- `127.0.0.1` 만 바인딩 — 로컬 전용, 외부 노출 없음
- `docs/v5/ch57_frames/` 에 이미지 없으면 사진 섹션 빈칸
- 서버 재기동: `pkill -f mona_verify_server.py && python3 /tmp/mona_verify_server.py &`

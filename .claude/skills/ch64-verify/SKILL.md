---
name: ch64-verify
description: exp73/CH64 실험 결과를 그림으로 만들고, 로컬호스트로 즉시 확인시켜주는 반복 워크플로우. "그림으로 보여줘", "로컬로 확인시켜줘", "CH64에 반영하고 보여줘" 같은 요청에 사용.
---

# CH64 실험 → 그림 → 로컬 검증 스킬

exp73/CH64 계열 실험(closed-loop 성능, cadence/HELD, 그라운더 비교 등)을 할 때마다
"숫자만 말로 설명" 대신 **그림으로 만들어서 로컬로 바로 보여주는** 표준 절차.

## 언제 쓰나
- 새 실험(예: 다른 stride, 다른 head, 다른 그라운더) 결과가 나왔을 때
- 사용자가 "예시 보여줘", "뭐가 맞는지 보여줘", "그림으로" 라고 요청할 때
- CH64에 새 finding-card를 추가하거나 기존 카드의 수치를 갱신할 때

## 절차 (매번 이 순서로)

1. **실험 스크립트 실행/확인**: `scripts/exp73_*.py` 계열(예: `exp73_window_cadence.py`,
   `gen_generalization_matrix.py`) 결과 로그 확인. 백그라운드로 돌린 경우
   완료 여부(`ps -p <PID>`)부터 확인 — 완료 전에 그림/카드를 만들지 말 것.

2. **그림 생성**: `scripts/gen_ch64_figs.py` 또는 `scripts/gen_ch64_cadence_diagram.py`
   패턴을 따라 신규 스크립트 작성(또는 기존 것에 함수 추가).
   - 필수: NanumGothic 폰트 등록(한글 깨짐 방지), Okabe-Ito 팔레트
     (`{"blue":"#0072B2","orange":"#E69F00","green":"#009E73","verm":"#D55E00","grey":"#999999"}`),
     `savefig.facecolor="white"`.
   - 출력 경로: `docs/v5/ch64_figs/fig_64_<N>_<설명>.png`

3. **CH64에 카드 삽입**: `docs/v5/research_story.html`에서 해당 finding-card를 찾아
   (`grep -n "64-N\."`) 표/설명 텍스트 갱신 + `<img src="ch64_figs/...">` 추가.
   기존 카드 스타일(`finding-card`, 표 포맷)을 그대로 따를 것 — 새 CSS 클래스 추가 금지.

4. **커밋(force-add 필요)**: `docs/**/*.png`는 `.gitignore` 대상이므로
   `git add -f docs/v5/ch64_figs/*.png` 로 명시적으로 강제 추가해야 함(잊기 쉬운 지점).
   ```bash
   git add -f docs/v5/ch64_figs/fig_64_<N>_*.png
   git add docs/v5/research_story.html scripts/<실험스크립트>.py
   git commit -m "docs(CH64): ..."
   git push origin HEAD:inference-integration
   ```

5. **로컬 서버로 즉시 확인**:
   - 서버가 이미 떠 있는지 먼저 확인: `ps aux | grep "http.server 8899"` 또는
     `curl -s -o /dev/null -w "%{http_code}" http://localhost:8899/v5/research_story.html`
   - 안 떠 있으면 **반드시 백그라운드로**(foreground로 실행하면 응답 못 하고 멈춤):
     ```bash
     cd /home/minum/26CS/MoNaVLA/docs && python3 -m http.server 8899 --bind 0.0.0.0 &
     ```
     (또는 `bash scripts/serve_research_story.sh &`)
   - 이미 떠 있으면 **재기동 불필요** — 파일시스템 직접 서빙이라 그림/HTML 저장 즉시 반영됨.
   - 확인 URL: `http://localhost:8899/v5/research_story.html#ch64` — 새 카드 번호(`#64-N`)까지
     명시해서 사용자에게 안내.
   - 이미지가 실제로 로드되는지 `curl -s -o /dev/null -w "%{http_code}"`로 200 확인 후 보고할 것
     (커밋만 하고 "됐다"고 하지 말고, 반드시 로컬에서 렌더 확인까지 마칠 것).

## 하지 말 것
- 로컬 서버를 매번 새로 띄우지 말 것(포트 충돌·좀비 프로세스 방지) — 살아있으면 재사용.
- `docs/**/*.png`가 gitignore 대상임을 잊고 `git add`만 해서 그림이 커밋 안 되는 실수
  반복 금지 — 반드시 `-f`.
- 그림 없이 표/텍스트만 카드에 넣고 "그림으로 보여줬다"고 하지 말 것 — 이 스킬의 존재
  이유가 "말 대신 그림"이므로, 매번 최소 1개 이상의 새 그림을 생성해야 함.

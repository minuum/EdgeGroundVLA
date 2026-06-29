# Plan — "4초 = 드리프트, 단발 latency 아님" 실시간 증명

> 작성: 2026-06-22 · 상태: **검토 대기 (v2 — 사용자 피드백으로 deliverable 변경)**
> 동기: `research_story.html` 미팅 준비 섹션 item 6에서 텍스트로만 설명한 두 결론
> (① 학습/추론 리사이즈는 원래 같았고 이제 명시적으로 강제됨, ② "4초"는 단발 latency가
> 아니라 1fps 가정 대비 처리속도 드리프트의 누적값)을 실제로 라이브 측정/시각화하는 형태로
> 증명.

## v2 변경 사항 (사용자 메모 반영)

- ~~독립 정적 HTML 페이지(`drift_latency_proof.html`)~~ → **`scripts/gradio_inference_dashboard.py`의
  기존 대시보드에 3번째 탭으로 통합.** 현재 탭 구조: `🤖 Drive / Inference` → `🔍 Grounding 검증`
  (line 1437, 1726) 바로 옆에 **`📊 Latency/Drift 진단`** 탭 신규 추가.
- 정적 과거 세션(`s6_cl_sim.json`) 재생이 아니라, **지금 떠 있는 운영 서버(exp66, port 8001)를
  실시간으로 호출해서 직접 데이터를 수집**하는 방식으로 변경 — "수집할 수 있게" 요구사항 반영.
- `Grounding 검증` 탭의 기존 패턴(카메라 프레임 소스, 단발/자동(1fps)/정지 버튼, `gr.Timer`,
  세션 JSONL 로깅, idle-gap 세션 분리)을 그대로 재사용해 일관성 유지.

---

## 0. 근거 데이터 (변경 없음 — v1에서 이미 확정)

`docs/v5/s6_cl_sim.json` (105프레임, 1fps 가정, 운영 파이프라인 기록):

| frame | 누적 실측(s) | 1fps 가정(s) | drift(s) |
|---|---|---|---|
| 1 | 1.85 | 1 | 0.85 |
| **10** | **13.96** | **10** | **3.96 ≈ "4.0초"** |
| 50 | 68.01 | 50 | 18.01 |
| 105 | 142.38 | 105 | 37.38 |

- 평균 처리시간 1356ms/frame vs 1fps 가정 1000ms/frame → 구조적 발산
- 단발 latency 최대 1852ms — 4초를 넘은 적 없음 → "4초"는 단발 문제가 아니라 누적 드리프트

이 표는 **새 탭에서 실시간으로 직접 재현**(과거 로그 재생이 아니라 지금 서버로 측정)하는 것이 목표.

---

## 1. 새 탭 — `📊 Latency/Drift 진단` (`gradio_inference_dashboard.py`)

`Grounding 검증` 탭(line 1726~2122 부근) 바로 다음에 `with gr.Tab("📊 Latency/Drift 진단"):` 블록 추가.

### UI 구성
- **좌측**: 버튼 행 — `▶ 단발 측정` / `🔄 자동 (1fps)` / `⏹ 정지` / `🆕 새 세션`
- **우측 상단 지표 텍스트박스 5개**:
  - 현재 시각 (1초 타이머, Grounding 탭과 동일 패턴)
  - 이번 호출 latency(ms) — `<1000ms PASS / <2000ms WARN / else FAIL`
  - 누적 프레임 수 / 누적 실제시간(s) / 누적 가정시간(s, =frame수×1.0)
  - **현재 drift(s)** — 강조 표시, 발산 추세면 색 변경(녹색→주황→빨강)
- **실시간 차트(`gr.Plot`, matplotlib, `traj_plot`과 동일 패턴 — line 1721 참고)**:
  - x=frame, y=누적시간(s) — "실제" 선 vs "가정(1fps)" 선, 매 tick마다 갱신
  - drift가 4.0s를 넘는 첫 frame에 점선+라벨로 마커 표시
- **이력 테이블**(`gr.Dataframe`, Grounding 탭의 `gnd_history`와 동일 패턴): 최근 10프레임 `[#, latency_ms, cum_real_s, cum_nominal_s, drift_s]`
- **JSONL 경로 표시 + 🆕 새 세션 버튼**: `logs/drift_sessions/drift_<ts>.jsonl`에 매 프레임 `{ts, frame, latency_ms, cum_real_s, cum_nominal_s, drift_s}` 기록 — Grounding 탭의 `_gnd_ensure_log()` idle-gap-split 로직 재사용
- **🩺 진단 실행 버튼**: 클릭 시 `scripts/eval/diagnose_pipeline_health.py`의 `check_latency()`/`check_resize()`를 직접 import해서 호출, PASS/WARN/FAIL 결과를 텍스트박스에 표시 (해당 스크립트는 이미 모듈 함수라 바로 import 가능, subprocess 불필요)

### 동작
- `▶ 단발 측정`: 운영 서버 `/predict`(또는 `/ground` — Grounding 탭과 동일 엔드포인트 사용해 두 탭이 같은 latency를 측정하는지 비교 가능하게) 1회 호출 → latency 기록 → 누적치 갱신 → 차트 갱신
- `🔄 자동 (1fps)`: `gr.Timer(1.0, active=False)`로 Grounding 탭과 동일하게 1초마다 반복 호출 — **"1fps로 가정하고 도는데 실제로는 더 걸린다"는 상황을 라이브로 그대로 재현**
- `⏹ 정지`: 타이머 정지, 세션 유지(이어서 재생 가능)
- `🆕 새 세션`: 누적치/이력/차트 리셋, 새 JSONL 파일 시작

---

## 2. 변경 파일

| 파일 | 작업 |
|---|---|
| `scripts/gradio_inference_dashboard.py` | `📊 Latency/Drift 진단` 탭 신규 추가 (UI + `_run_drift_check()` 함수 + 타이머/버튼 wiring) |
| `scripts/eval/diagnose_pipeline_health.py` | 변경 없음 — `check_latency`/`check_resize`를 그대로 import해서 재사용 |
| `docs/v5/research_story.html` | item 6 카드에 "📊 대시보드 3번째 탭에서 실시간 재현 가능" 한 줄 추가(선택) |

독립 HTML 페이지(`drift_latency_proof.html`)는 **만들지 않음** — 대시보드 탭으로 대체.

## 3. 트레이드오프
- 운영 서버(exp66, port 8001)에 실시간 부하를 추가로 줌 — Grounding 탭과 동시에 자동 모드를 둘 다 켜면 호출이 겹쳐 latency 측정이 왜곡될 수 있음. **두 탭의 자동 루프를 동시에 켜지 않는 걸 기본 가이드로 문서에 명시.**
- 과거 세션(`s6_cl_sim.json`) 재생 기능은 빠짐 — 필요하면 추후 "📁 세션 불러오기" 버튼으로 확장 가능(이번 범위 밖)

## 4. 완료 기준
- [ ] `📊 Latency/Drift 진단` 탭 UI 추가
- [ ] 단발/자동/정지/새세션 동작 확인
- [ ] 실시간 차트(실제 vs 가정 누적시간) 갱신 확인
- [ ] JSONL 세션 로깅 확인
- [ ] 🩺 진단 실행 버튼 — check_latency/check_resize 결과 표시 확인
- [ ] 로컬 대시보드(7865)에서 실제로 4초 부근 drift 재현되는지 라이브 확인
- [ ] git commit + push (monavla-driving, inference-integration 양 브랜치)

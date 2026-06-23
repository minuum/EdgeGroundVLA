# Plan — 대시보드 UX 개선 + 서빙 아키텍처 대안 + 교수 피드백 갭 정리 (2026-06-23)

> 상태: 리서치 완료, 구현 전. 이 문서는 검토/주석용 — 승인 전까지 코드 변경 없음.

---

## 1. 교수 피드백/실험에서 아직 부족한 부분 (리서치 결과)

서브에이전트로 `docs/v5/` 전체를 조사한 결과:

### 1-1. 명시적으로 요구됐지만 미완료
| 항목 | 출처 | 비고 |
|---|---|---|
| STOP 거리 캘리브레이션(cm 단위 매핑) | research_story.html CH44-3 | `scripts/eval/calibrate_stop_distance.py`는 작성됐지만(어제 커밋) 실측 미반영 |
| 다물체(brown pot/chair) goal-conditioned 데이터·학습 | PROF_QA_PREP_20260603.md Q3 | |
| 실로봇 물리 검증 | PROF_QA_PREP_20260603.md Q6 | 시뮬 closed-loop 100%뿐, 실로봇 미완 |
| center_straight 경로 추가 수집 | PROF_QA_PREP_20260603.md Q7 | jitter 증폭으로 구조적 미해결 |
| 4/24 미팅 "left-only→straight 출력" 정식 종료 보고 | APR24_PROF_TODO_FROM_TRANSCRIPT.md | 보고 누락 |

### 1-2. 스스로 "후속 검토 필요"로 남긴 항목
- 방향성(L/R/F) instruction 구분력 추가 검증
- 96.85%는 노이즈 상단값(평균 95.39%) — 추가 검증 진행 중
- chair V5-2 STOP 알고리즘 latch 재캘리브 미확정
- right_diagonal 시뮬 2/3 실패(FPE 0.517m) 실환경 재현 확인

### 1-3. 가장 시급 (공통점: 시뮬은 통과, 물리 세계 확인 누락)
- **`docs/v5/REAL_ROBOT_CHECKLIST_20260616.md`의 결과 기록표가 빈칸** — 체크리스트는 있는데 실제로 돌렸는지/결과가 뭔지 기록이 없음
- LoRA가 base PG2보다 grounding 악화시키는 원인은 규명됐지만 "base PG2로 단순화" 결정의 실로봇 검증 미완

**이 plan과는 별도 트랙**(실험/물리테스트)이라 여기선 우선순위 정리만 하고, 실행은 추론 인프라(2번) 정리 후로 미루는 걸 제안함 — 둘 다 동시에 진행하면 컨텍스트 분산이 큼.

---

## 2. 추론 대시보드 "모니터링/run" 화면 — 문제 진단

### 2-1. 현재 구조
`scripts/gradio_inference_dashboard.py` 2533줄, 3개 Tab:
- `🤖 Drive / Inference` (line 1457~1745, **288줄**) — 모델 로드, 실험모드 선택, manual WASD, 조이스틱, 속도 슬라이더, 라이브 카메라, run/stop/return 버튼, 상태 로그, latency/action 텍스트, trajectory plot **전부 한 화면에 압축**
- `🔍 Grounding 검증` (315줄)
- `📊 Latency/Drift 진단` (어제 신규)

### 2-2. 진단
"Drive / Inference" 탭이 **제어(조작)와 모니터링(관찰)이 분리 안 된 단일 화면**이라 실험 중 다음이 동시에 일어남:
- 모델/실험모드를 고르는 일회성 설정
- 매 프레임 갱신되는 라이브 카메라 + latency + action + trajectory (계속 움직이는 영역)
- 손으로 누르는 manual 컨트롤(WASD/조이스틱)

세 종류가 시각적으로 구분 없이 섞여 있어서, "지금 뭘 보고 있어야 하는지"와 "지금 뭘 눌러야 하는지"가 한 눈에 안 들어옴 — 이게 "화면구성이 너무 힘들다"의 핵심 원인으로 판단.

### 2-3. 제안 — 레이아웃 재구성 (같은 탭 안에서 영역만 분리, 새 프레임워크 도입 없음)
```
┌─────────────────────────────┬───────────────────────────┐
│  ① 설정 (접었다 펼 수 있는 Accordion)  │  ③ 모니터링 (항상 표시, 갱신 영역) │
│  - 모델 로드 / 실험모드 선택        │  - 라이브 카메라            │
│  - (자주 안 바꾸는 것들 위로 접기) │  - latency / action / chunk    │
├─────────────────────────────┤  - trajectory plot           │
│  ② 실행 제어 (항상 표시)         │  - 최근 N step 로그 테이블(신규)│
│  - Start/Stop/Return         │                           │
│  - Manual WASD + 조이스틱 상태   │                           │
│  - 속도 슬라이더              │                           │
└─────────────────────────────┴───────────────────────────┘
```
- `gr.Accordion`으로 ①을 기본 접어서 화면 차지 줄임
- ②③을 `gr.Row(equal_height=True)`의 좌/우 컬럼으로 명확히 분리 (지금도 Row/Column 쓰고 있지만 그룹 경계가 안 보임 — `gr.Group()` + 헤더로 시각적 구획 추가)
- **신규**: "최근 N step 로그 테이블" — 지금은 단일 텍스트박스(`status_log`)로 최신 한 줄만 보여서 추론 run 중 흐름을 못 따라감. drift 탭에 이미 만든 `gr.Dataframe` 히스토리 패턴을 재사용해서 step/action/latency/bbox area를 누적 표로 보여주면 모니터링이 훨씬 쉬워짐.

이 변경은 **파일 하나(`gradio_inference_dashboard.py`)의 레이아웃 재배치 + 작은 상태 누적 로직 추가**로 끝남 — 리스크 낮음, 별도 의존성 없음.

---

## 3. Gradio가 메모리상 최선인가? — 아키텍처 대안 검토

### 3-1. 현재 실측
```
시스템 전체: 15Gi total / 12Gi used / 379Mi free (여유 매우 적음)
- stage2_v2_inference_server.py : RES 1.2GB (모델 가중치 대부분)
- gradio_inference_dashboard.py : RES ~700~930MB (이 세션 중 변동 관측)
```
대시보드 프로세스의 900MB는 Gradio 자체보다는 **rclpy(ROS2 client), OpenCV, matplotlib, pygame, numpy를 한 프로세스에 다 같이 import**해서 누적된 비용에 가까움 — "Gradio라서 무겁다"는 정확한 진단이 아닐 수 있음.

### 3-2. 대안 비교

| 방식 | 메모리 영향 | 구현 비용 | 비고 |
|---|---|---|---|
| **A. 현재 유지 + 운영 방식만 정리** | 낮음(즉시 가능) | 거의 0 | 이미 `gradio_session_eval.py`/`gradio_grounding_demo.py`/`gradio_dataset_viewer.py` 등 용도별로 분리돼 있음(`gradio_hub.py`가 포털). **지금처럼 추론 대시보드를 상시 띄워두지 말고, 쓸 때만 켜고 끄기**만 해도 상시 점유 900MB를 회수 가능 |
| **B. FastAPI + HTMX/바닐라 JS 경량 프론트엔드** | 중간(서버 프로세스는 거의 FastAPI+필요 라이브러리만) | 중간~높음 | Gradio의 reactive wiring(콜백/큐) 없이 직접 REST+SSE/WS로 상태 push. ROS/카메라/조이스틱 로직은 그대로 백엔드에 두고 화면만 얇아짐. 다만 지금 짜여있는 수십 개 `.click()`/`gr.Timer()` 바인딩을 전부 수동 JS로 재작성해야 해서 **공수가 큼** |
| **C. 모델 서버와 "조작용 화면"을 완전히 분리** | 낮음(조작 화면이 모델/torch를 안 들고 있음) | 중간 | 지금도 API Server 모드면 대시보드 프로세스 자체는 torch 모델을 안 들고 있음(LocalInferenceBackend 안 쓰면). 진짜 무거운 건 ROS2/OpenCV/matplotlib 조합 — **카메라 프레임 디코딩이나 trajectory plot 렌더링을 백엔드(이미 떠 있는 stage2_v2 서버나 별도 경량 프로세스)로 옮기고, 대시보드는 결과 이미지/숫자만 받아 표시**하는 쪽으로 책임 분리 |
| **D. Streamlit/NiceGUI 등 타 프레임워크로 전면 교체** | 불확실(실측 필요) | 높음 | 프레임워크 자체보다 같이 import되는 ROS/CV 라이브러리가 메모리를 더 먹어서, 교체해도 큰 차이 안 날 가능성 — **권장 안 함** (검증 없는 재작성 리스크가 이득보다 큼) |

### 3-3. 권장
1. **즉시**: A (운영 습관) — 안 쓸 때 대시보드 프로세스 내려서 900MB 회수. 이번 세션에도 여러 번 재시작했는데, 매번 떠 있는 채로 진단 스크립트까지 별도 프로세스로 돌려서 메모리 압박이 더 컸음(조이스틱 충돌 원인도 이거였음)
2. **다음 단계 후보**: C 부분 적용 — 카메라 프레임을 대시보드가 직접 들고 처리(OpenCV 변환/annotate)하지 않고, 추론 서버 응답에 이미 annotate된 이미지를 포함해서 보내는 식으로 책임 이동 (작은 단위로 점진 적용 가능)
3. B/D는 **지금 당장은 보류** — 공수 대비 확실한 이득이 불확실. 메모리가 실제로 한계에 부딫혀서 OOM이 나는 시점이 오면 그때 B를 진지하게 검토하는 게 합리적 (지금은 379Mi free로 빠듯하지만 OOM은 아직 없었음)

---

## 4. 추가 기능 제안 (검토용, 우선순위 없음)

- **Run 히스토리 테이블** (2-3에서 언급) — step별 action/latency/bbox area 누적 표
- **세션 비교 뷰** — drift 탭에 만든 JSONL 세션 로그(`logs/drift_sessions/`)처럼, Drive/Inference 탭의 run도 세션별로 저장해서 두 run을 나란히 비교(이전 run vs 지금 run의 trajectory 오버레이)
- **"실로봇 체크리스트" 자동 기록 연동** — `REAL_ROBOT_CHECKLIST_20260616.md`의 빈 기록표(1-3에서 발견) 문제를 코드로 보완: run 종료 시 결과를 자동으로 그 체크리스트 형식의 markdown 행으로 append하는 버튼 추가 (사람이 손으로 표 채우는 걸 안 까먹게)
- **STOP 캘리브레이션 위젯** — `calibrate_stop_distance.py`가 이미 있으니, 그 출력(cm ↔ area 매핑)을 대시보드 모니터링 패널에 실시간으로 같이 표시(현재 area 값 옆에 "추정 거리: ~Xcm" 같이)
- **그라운딩 latency 분리 표시** — 이미 서버가 `grounding_latency_ms`를 주는데 대시보드 화면엔 합산 latency만 보임 — 모니터링 표에 "전체 / 그라운딩 / MLP" 분리 표시(이번 세션 분석 결과 직결)

---

## 5. 실행 순서 제안 (승인 시)

1. 2번(레이아웃 재구성 + run 히스토리 테이블) — 가장 즉각적 체감 개선, 리스크 낮음 — **완료 (2026-06-23)**
2. 4번 중 "그라운딩 latency 분리 표시" — 2번 작업과 같은 화면이라 묶어서 처리 효율적 — **완료 (2026-06-23, run 히스토리 표에 total/grounding/mlp 분리 컬럼으로 포함)**
3. 3-3의 운영 습관(A) 적용 — 코드 변경 아니라 즉시 적용 가능, 별도 합의 불필요
4. 1번(교수 피드백 갭) — 별도 트랙, 어떤 항목부터 다시 들어갈지 사용자 우선순위 필요
5. 4번 나머지(세션 비교, 체크리스트 자동기록, STOP 캘리브레이션 위젯) — 우선순위 정해서 순차 진행

---

## 6. 추가 발견 — `grounding_skip_n=3` 논의, 해결됨 (2026-06-24)

세션 중 `EXP_MODES`의 `grounding_skip_n: 3` 하드코딩이 "이동 시 초점이 바뀌는데 캐시된 bbox를 재사용하는 게 위험하지 않냐"는 의심으로 제기됐었음. `inference-integration` 브랜치에서 별도로 이미 정밀 검증됨(`docs/v5/research_story.html` CH49):

- skip_n=1(매프레임 그라운딩)은 실측 latency 1.3~1.4s/frame — 실시간 주행 자체가 불가능(s6/s7 실주행 세션 실측)
- skip_n=3의 bbox 캐시 재사용이 오히려 잡음에 대한 저역통과 필터처럼 작동해 baseline CL 93.1%→96.6%, FPE 0.145→0.119m로 개선
- area_delta 기법(CH47, FPE 0.098m 전체 최저)은 skip_n=1 조건에서만 유효하고 skip_n=3에선 역전(0.123m) — 미배포 확정
- **결론: 옵션 A(skip_n=3 유지, area_delta 미배포)로 메인 디폴트 확정, 추가 ablation 불필요**

조치: `gradio_inference_dashboard.py`의 `EXP_MODES` 위에 CH49 근거를 주석으로 남김(코드 변경 없음). `area_delta`는 운영 서버/대시보드/`model_registry.json` 어디에도 배포돼 있지 않음을 확인 — 추가 조치 불필요.

# Plan — CH37~CH39에 실제 이미지/시각자료 보강

> 작성: 2026-06-22 · 상태: **승인됨 — §1~3 구현·배포 완료, §4(CH37)는 보류**
> 동기: 사용자 요청 — "CH37~CH39 브리핑에서 실제 테스트하고 검증한 이미지들을 새 분석을 나타낼 때마다 모두 보여줬으면 좋겠다. 전체 프레임 가로 스크롤, 방향신호 같은 건 플로우차트보다 실제 이미지 예시(선호)로."

---

## 0. 리서치 결과 — 현재 상태

| 챕터 | 현재 이미지 | 실제 검증에 쓰인 원천 데이터 | 비고 |
|---|---|---|---|
| CH37 (STOP ablation) | 0개, 표만 | `scripts/ablate_stop_proximity.py` + 기존 V5 220개 에피소드 폐쇄루프 재평가(FPE/SR 수치) | 프레임/궤적 단위 시각화는 아직 만든 적 없음 |
| CH38 (VLA 비교) | 0개(본문), 단 **38-5는 `docs/today_visual_summary.html`에 실제 사진+raw 출력 이미 있음** | 38-4: `scripts/measure_hidden_state_pg2.py` (`docs/v5/grounding_frames/` 실제 세션 이미지 4장 S6/S7/S8), 38-5: `pg2_direction_output_test.json` | 이미지 자체는 존재, **research_story.html 본문에 박혀있지 않고 별도 페이지에만 있음** |
| CH39 (필터+probe) | 0개, 표만 | 39-1: soda 실측(이미지 미저장, 좌표/area 수치만 plan.md에 기록), 39-2: `scripts/eval/probe_v5_direction_hidden_state.py`(V5 h5 mid-frame, **메모리 내 처리만, png로 저장 안 함**) | 실제 프레임 이미지가 디스크에 없음 — 새로 추출해야 함 |

기존 CSS 클래스 재사용 가능: `frame-strip`(가로 스크롤), `fig-grid`, `overlay-grid`, `finding-grid` — 새 CSS 불필요.

---

## 1. CH38 — 기존 이미지를 본문에 끌어오기 (가장 저비용)

38-4/38-5에 쓰인 실제 이미지(`docs/v5/grounding_frames/` 내 S6/S7/S8 세션, `pg2_direction_output_test.json`에 기록된 4세션)를 `today_visual_summary.html`에서 research_story.html CH38 본문으로 직접 임베드.

- 38-4 finding-card 아래: 세션별 이미지 4장 + 코사인거리 라벨을 `frame-strip`(가로 스크롤)로 추가.
- 38-5 finding-card 아래: "방향 텍스트를 줘도 bbox가 거의 동일"을 보여주는 실제 이미지(같은 세션, 다른 prompt) 2~4장 비교 — bbox 좌표를 이미지 위에 박스로 오버레이해서 "거의 안 움직임"을 시각적으로 증명.
- **새 추출 작업 없음** — 기존 파일 경로 확인 후 `<img>` 태그만 추가. 단 bbox 오버레이가 안 그려진 원본이면 PIL로 박스 그려서 새 PNG 생성(가벼운 스크립트 1개).

## 2. CH39-1 (Step A) — 그라운딩 필터 보정 실측 이미지

soda 테스트 당시 이미지를 저장 안 해서 원본 캡처는 없음. 재캡처는 두 갈래:

- **콜라캔/의자/콘 (정상 케이스)**: GB10 로컬에서 같은 phrase로 PG2 그라운딩 재실행(soda 재접속 불필요, [[soda-pg2-concurrent-load-crash]] 회피) → bbox 오버레이 PNG 생성. `docs/v5/test_images/` 또는 `grounding_frames/`에 이미 있는 객체 이미지 재사용 가능한지 우선 확인, 없으면 동일 객체의 새 사진 1장씩만 사용(웹캠/기존 자산, 새 데이터 수집 아님 — 단발성 디버그 이미지).
- **사과/머그 (soda 전용 환각 케이스)**: GB10에서는 정상 박스가 나온다고 이미 확인됨 → GB10 결과 이미지(정상 박스) + 텍스트로 "soda(Jetson)에서는 같은 입력에 5/5 동일하게 풀프레임 환각(`<loc0000><loc0000><loc1006><loc1020>`)"이라고 대비. **soda 재접속해서 재캡처하지 않음** — 이미 plan.md에 기록된 raw_output 텍스트로 충분히 대비 가능하고, 굳이 다시 soda를 건드려 크래시 리스크를 또 만들지 않는 쪽을 권장.
- 5종 이미지를 `frame-strip`으로 가로 배치, 각 이미지 아래 area/상태(정상/환각) 라벨.

## 3. CH39-2 (Step B) — 방향 신호 probe 실제 예시 이미지 (플로우차트 대신 실제 이미지, 사용자 선호)

- 신규 스크립트 1개: V5 h5 220개 중 direction 라벨별(left/straight/right) 대표 에피소드 2~3개씩(가능하면 start 위치도 다양하게) mid-frame을 PNG로 추출 → `docs/v5/attention_analysis/direction_probe_examples/`에 저장.
- CH39-2 finding-card 아래에 9장(또는 6장) 이미지를 `frame-strip`(가로 스크롤)으로 배치, 각 이미지에 `start_direction` 라벨 캡션.
- 목적: "이미지가 이렇게 다르게 생겼으니 hidden state가 구분하는 게 직관적으로도 말이 된다"는 시각적 근거. probe 수치(90%, 99.1%, 92.3%) 표는 기존 그대로 유지.
- **새 학습/측정 없음** — 이미 끝난 probe 결과에 예시 이미지만 첨부.

## 4. CH37 — 폐쇄루프 STOP ablation 시각화 (선택, 비용 더 큼)

- V0(98.7%) vs V1(23.5%) 차이를 보여주려면 궤적(trajectory) plot이 필요 — `scripts/sim/evaluate_closed_loop_v5.py` 또는 `ablate_stop_proximity.py`가 step별 좌표를 로깅하는지 확인 필요(아직 미확인).
- 로깅이 있으면: matplotlib으로 V0/V1 궤적 2~4개 path_type 비교 플롯 생성 → PNG로 CH37에 추가.
- 로깅이 없으면: 이번 plan 범위에서 제외하고(재실행 필요 = 비용 큼) 추후 별도 작업으로 분리, CH37은 표만 유지.
- **사용자 확인 필요**: 궤적 로그가 없을 경우 폐쇄루프를 다시 돌려서까지 시각화를 만들지 여부.

---

## 5. 변경 파일 정리

| 파일 | 작업 |
|---|---|
| `scripts/eval/extract_direction_probe_examples.py` (신규) | V5 h5 → direction별 대표 mid-frame PNG 추출 |
| `docs/v5/attention_analysis/direction_probe_examples/*.png` (신규) | 위 스크립트 산출물, 6~9장 |
| `scripts/eval/draw_grounding_bbox_examples.py` (신규, 가벼움) | CH38/CH39용 bbox 오버레이 PNG 생성(기존 이미지 위에 박스만 그림) |
| `docs/v5/research_story.html` | CH38, CH39 본문에 `frame-strip` 이미지 블록 추가 (텍스트/표 구조는 유지, 이미지만 보강) |
| `docs/v5/grounding_frames/` 또는 `test_images/` | 필요한 객체 사진이 없으면 1~2장 보강 (새 "데이터 수집"이 아니라 디버그용 단발 사진) |

CH37은 궤적 로깅이 없어 재실행 필요(§4) — 사용자 승인("다시 돌려바 해도대 ㄱㄱㄱ")
받고 진행. `scripts/eval/visualize_stop_trajectories.py`(신규)가 기존 ckpt로 3개
val 에피소드(center_straight/left_left/right_right)만 재추론해 V0 vs V1 궤적을
matplotlib으로 그림(5-seed 풀 재집계 아님, 새 학습 없음) → `docs/v5/closed_loop_eval/trajectory_examples/*.png`.
CH37 L1 표 바로 아래에 frame-strip-scroll로 삽입 완료.

### 완료 기준 갱신
- [x] §1 CH38 이미지 임베드
- [x] §2 CH39-1 bbox 이미지
- [x] §3 CH39-2 방향 예시 프레임
- [x] §4 CH37 궤적 시각화 (3개 예시 episode 재추론)

---

## 6. 위험도 / 트레이드오프

- soda 재접속 없음 — 전부 GB10 로컬 작업 (크래시 리스크 0).
- 새 학습/재평가 없음 — 기존 산출물에 시각자료만 추가.
- CH37만 "궤적 로깅 있는지 확인 → 없으면 재실행 필요(비용 발생)"라는 불확정 요소가 있어, 그 부분만 사용자 결정 필요.
- 이미지 추가로 research_story.html 파일 용량 증가(특히 inline base64 쓰면 더 큼) → **상대경로 PNG 파일 방식 사용**(기존 다른 챕터의 패턴과 동일), base64 인라인은 피함.

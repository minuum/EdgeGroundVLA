# Plan — GroundingDINO zero-shot vs base PG2 zero-shot 그라운딩 비교

> 작성: 2026-06-21 · 상태: **검토 대기 (승인 전 본구현 금지)**
> 동기: PG2 224→448 해상도 ablation으로 "해상도는 답이 아니다"가 확정됨(S8 실측 + 로컬 ablation 독립 재확인).
> 다음 후보로 검토 중인 **GroundingDINO**를 실제로 떠보기 전에, 최근 VLA 5편 리서치 결과
> "거의 모든 케이스가 detector를 **frozen zero-shot**으로만 쓰고 fine-tune하지 않는다"는 결론을 반영해
> **GroundingDINO도 fine-tune 없이 zero-shot으로 base PG2 zero-shot과 동일 조건 비교**한다.
> 연관: `docs/v5/grounding_hub.html`(§A~E, exp56~64 LoRA 전부 grounding 악화), `docs/v5/robot_tests.html#resolution-analysis`, `docs/v5/pg2_resolution_ablation.json`

---

## 0. 배경 — 왜 GroundingDINO인가, 왜 zero-shot 비교인가

- `grounding_hub.html` §A/§E: PG2 LoRA(exp56~64) 전부 grounding을 **악화**시켰다. 레이어를 vision/LM 어디에 걸어도 동일 패턴 → "소규모 데이터 LoRA가 사전학습 grounding 표현을 깨뜨린다"는 구조적 결론.
- 224→448 해상도 ablation: bbox 거의 동일, S8 실측 정확도 51.4%→34.0% **하락**. 해상도도 답이 아님.
- 남은 미해결 실패 모드 2종 (실제 production 세션에서 확인됨):
  - **S6 dead-zone**: 정지장면에서 풀프레임 환각 (frame 56/70/85, `docs/v5/grounding_frames/s6/`)
  - **S7 near-miss**: 근접 상황에서 빈 바닥을 박싱 (frame 39/41, `docs/v5/grounding_frames/s7/`)
- 최근 VLA 논문 5편 조사 결과(이번 대화), grounding을 쓰는 VLA들은 거의 전부 **(a) 외부 open-vocab detector를 frozen zero-shot으로 bolt-on**, **(b) detector를 학습 데이터 큐레이션에만 씀(추론 시 미사용)**, **(c) detector 없이 end-to-end**, **(d) 거대 VLM을 통째로 frozen** 중 하나. **detector를 우리 소규모 데이터로 fine-tune하는 패턴은 없음.**
- → 이번 ablation도 같은 철학 유지: **GroundingDINO를 fine-tune하지 않는다.** zero-shot 대 zero-shot 비교로 "모델 교체"만의 효과를 분리.

---

## 1. 비교 대상

| | 모델 | 가중치 | 비고 |
|---|---|---|---|
| 기준 | base PG2 (`google/paligemma2-3b-mix-224`) | frozen, zero-shot | 현재 운영 중 모델 |
| 후보 A | `IDEA-Research/grounding-dino-tiny` | frozen, zero-shot | 172M, 경량 — Jetson 후보 |
| 후보 B | `IDEA-Research/grounding-dino-base` | frozen, zero-shot | 더 정확하지만 더 무거움 (latency 비교용, 시간 허용 시) |

HF `transformers`(현재 환경 4.49.0)에 `AutoModelForZeroShotObjectDetection`으로 GroundingDINO 네이티브 지원 확인됨 — 별도 `groundingdino` 레포 설치 불필요.

프롬프트: GroundingDINO 문법은 카테고리를 `.`로 구분 → `"gray basket."` (PG2는 `"<image>detect gray basket"` 그대로 유지).

---

## 2. 평가 데이터 (기존 자산 재사용, 신규 수집 없음)

`grounding_hub.html` 구축 시 이미 만든 표준 세트를 그대로 재사용 — 새로 라벨링하지 않음:

| 세트 | 경로 | 목적 |
|---|---|---|
| 표준 시점 49프레임 | `eval_exp64_grounding.py:build_basket_sample()` | hit/cx_MAE/full-frame 기본 비교 |
| OOD 의자 11장 | `docs/v5/chair_probe/images/` | 오탐률(미학습 객체 구분) |
| 측면 4경로 | `ROS_action/mobile_vla_dataset_v5` (left_left/right_right/left_right/right_left) | 꺾임 구간 추적 |
| **S6 dead-zone 3프레임** | `docs/v5/grounding_frames/s6/frame_{0056,0070,0085}.jpg` | **PG2가 실패하는 케이스 — 핵심 타겟** |
| **S7 near-miss 2프레임** | `docs/v5/grounding_frames/s7/frame_{0039,0041}.jpg` | **PG2가 실패하는 케이스 — 핵심 타겟** |

기존 hub와 동일 metric으로 측정해 같은 표에 한 행으로 추가할 수 있게 포맷을 맞춘다(`docs/v5/grounding_hub/hub_results.json` 스키마 재사용).

지표: hit / cx_MAE(HSV 근사 GT 대비) / full-frame율(area>0.9) / OOD FP율 / **S6+S7 실패케이스 hit&정답위치 일치율(신규 — 스폿체크 포함)** / latency(ms).

---

## 3. 실행 위치 — 로컬(GB10), soda 건드리지 않음

해상도 ablation 때 합의된 규칙 유지: **무거운 추론은 로컬, soda는 주행/카메라만.**
GroundingDINO-tiny(172M)는 PG2(3B)보다 훨씬 가벼워 향후 soda 탑재 시 메모리 부담이 적을 것으로 예상되나, 이번 단계는 비교 측정만 목적이므로 로컬에서 진행.

---

## 4. 신규/수정 파일

| 파일 | 작업 |
|---|---|
| `scripts/eval/eval_groundingdino_vs_pg2.py` | **신규** — GroundingDINO(tiny/base) + base PG2를 §2 데이터셋에 일괄 평가, `hub_results.json` 호환 포맷으로 산출 |
| `docs/v5/grounding_hub/hub_results.json` | GroundingDINO 행 추가 (기존 모델 행 보존) |
| `docs/v5/grounding_hub.html` | 비교 매트릭스(§B)에 GroundingDINO 행 추가 + 신규 섹션 "§H GroundingDINO zero-shot 비교"(S6/S7 실패케이스 시각 비교 포함) |
| `docs/v5/robot_tests.html` | `#resolution-analysis` 옆에 "다음 후보 검증" 링크 또는 짧은 결론 추가 (택1, 결과 보고 결정) |
| `docs/v5/research_story.html` | 결론에 따라 caveat box 갱신 (택1) |

---

## 5. 단계 (승인 후)

- **Phase 1:** `eval_groundingdino_vs_pg2.py` 작성 — `eval_grounding_hub.py`의 `make_pg_detect`/`build_basket_sample`/그리드 생성 로직 재사용, GroundingDINO용 `make_gdino_detect()` 추가(box_threshold=0.35, text_threshold=0.25 기본값, 가장 높은 confidence box만 채택).
- **Phase 2:** 표준 49프레임 + OOD 11장 + 측면 4경로에서 GroundingDINO-tiny vs base PG2 비교 실행. JSON 산출.
- **Phase 3:** S6 dead-zone 3프레임 + S7 near-miss 2프레임에 대해 **시각 스폿체크 포함**(grounding-session-pipeline 스킬의 "절대 생략 금지" 규칙 적용) — GroundingDINO가 실제로 다른/더 나은 박스를 내는지 직접 이미지로 확인.
- **Phase 4:** 시간 허용 시 grounding-dino-base 추가 비교 + latency 측정(soda 탑재 가능성 평가용 참고 수치).
- **Phase 5:** `grounding_hub.html` §H 섹션 작성 + 결론 반영, 필요시 robot_tests/research_story 갱신.

---

## 6. 결정 게이트

- GroundingDINO가 S6/S7 실패케이스에서 PG2보다 명확히 낫고(풀프레임 환각 없음 / 빈 바닥 오탐 없음) 표준세트·OOD에서 크게 밀리지 않으면 → **production 교체 후보로 승격**, 별도 plan으로 soda 배포 절차 수립.
- 차이가 없거나 GroundingDINO도 같은 실패 패턴을 보이면 → **"이 장면 자체가 모호해서 어떤 zero-shot detector도 못 푼다"**로 결론, 다음 후보(YOLO-World 등) 또는 수동 라벨링+소규모 fine-tune 트랙으로 전환.

---

## 7. 결정 사항 (2026-06-21 확정)

- ✅ **tiny + base 둘 다 포함.** latency/정확도 트레이드오프까지 같은 표에서 비교.
- ✅ §H 섹션은 `grounding_hub.html`에 통합 (별도 페이지 생성 안 함).
- ✅ 이번 라운드는 **로컬 zero-shot 비교까지만** — soda 실측(S8류) 재현은 결과에 따라 별도 plan으로 판단.

## 8. 완료 기준

- [x] `eval_groundingdino_vs_pg2.py` 작성 및 실행 (GB10 nvrtc `.prod()` JIT 버그 monkeypatch로 우회)
- [x] 표준 49프레임 + OOD 11장 + 측면 4경로 + S6/S7 실패케이스 비교 JSON (`docs/v5/grounding_hub/gdino_vs_pg2.json`)
- [x] S6/S7 결과 시각 스폿체크 (5건: s6 f56/70/85, s7 f39/41 — Read 툴로 원본 이미지 직접 확인)
- [x] `grounding_hub.html` §H 섹션 추가
- [x] 결정 게이트에 따른 다음 단계 명시

## 8.5 참고 — PG2 224/448/896 해상도 비교 (선행 ablation, 2026-06-20 완료)

이번 GroundingDINO 비교 전에 먼저 "해상도를 올리면 되는가"를 검증했고 결론은 부정이었다.
같은 7프레임(S6 baseline/dead-zone×3, S7 near-miss×2, S7 correct×1·교정 후 6종 사용)으로 실측.

| 체크포인트 | 이미지 토큰 | 토큰 배수 | grounding latency (실측) | GPU 메모리 (실측) | bbox 일치도 | 실서버 정확도 |
|---|---|---|---|---|---|---|
| **mix-224 (배포중)** | 256 | 1× | 1246~1264ms(soda) / 489ms(GB10) | 6.73GB(soda) / 6.14GB(GB10) | 기준 | **51.4%** (S6, n=105) |
| mix-448 | 1024 | 4× | ~2.1s(soda) / 858ms(GB10), 224 대비 ~1.75배 | 6.32GB(GB10), 224와 거의 동일(+0.18GB) | 거의 동일(예: s6:56 a=0.218/0.218, s7:39 a=0.078/0.078) | **34.0%** (S8, n=47) — 하락 |
| mix-896 | 4096 (16×) | — | 미시도 | 미시도 | — | — |

> 3B "mix"(downstream 튜닝) 라인업은 224/448까지만 존재 — 896은 pt(순수 사전학습)만 있고 `<loc>` grounding 포맷 출력을 보장하지 않아 시도하지 않음.

**결론**: bbox 값은 224/448에서 거의 동일(해상도를 올려도 PG2가 보는 "정답"이 안 바뀜), latency는 ~1.75배(추정했던 4~16배보다 가벼움), 메모리는 +0.18GB뿐(추정 +6GB보다 가벼움) — 그런데 실서버 배포 결과 정확도는 오히려 51.4%→34.0%로 **하락**. 비용은 추정보다 가벼웠지만 효과가 없었다는 게 최종 결론. 출처: `docs/v5/robot_tests.html#resolution-analysis`, `docs/v5/research_story.html`(PG2/PaliGemma 비교 챕터 caveat), `docs/v5/pg2_resolution_ablation.json`, `docs/v5/grounding_frames/s8/`.

이 결론과 이번 GroundingDINO 비교(§9)를 합치면: **해상도도 답이 아니고, 모델 교체(GroundingDINO)도 OOD 트레이드오프 때문에 아직 답이 아니다** — 두 후보 모두 기각, 남은 후보는 §9 결정 게이트에 명시.

## 9. 결과 요약 (2026-06-21)

- **표준세트**: base PG2(hit98%/cxMAE0.126/OOD9%) ≈ gdino-tiny/base(hit100%/cxMAE0.125) — 바스켓을 보는 능력 자체는 동급.
- **OOD 오탐 — GroundingDINO 탈락 사유**: "gray basket" 프롬프트에 의자 11장 중 tiny 100%/base 91% 오탐(threshold 0.5로 올려도 64%). base PG2는 9% — **negative 구분력에서 GroundingDINO가 크게 밀림**.
- **S6 dead-zone**: GroundingDINO가 base PG2보다 **명확히 개선** — 정지장면(f70≈f85, 이미지 직접 대조로 확인)에서 PG2만 area 0.13→0.25로 흔들리고 GDINO는 0.133~0.134로 일관.
- **S7 near-miss**: 3모델 전부 거의 동일한 박스 → **모델 교체로 해결 안 되는 장면 자체의 모호성**으로 결론.
- **결정**: 현재 임계값으로는 production 교체 부적합(OOD 게이트 탈락). 다음 후보 — 프롬프트 구체화 재시도, YOLO-World 비교, S7류는 STOP/근접 로직(Task #4)으로 별도 대응.
- ⚠️ 부작용 발견: `docs/v5/grounding_frames/s7/` 프레임 일부에 당시 대시보드 디버그 오버레이가 원본에 합성 저장됨 — 모델 간 상대비교는 유효하나 절대 판정은 raw 프레임 재추출 필요.

## 10. 후속 3단계 (2026-06-21 사용자 승인 — 순서대로 진행)

§9 결정 게이트 결과에 따른 다음 단계 3개를 순서대로 실행한다.

| 순서 | 작업 | 목적 | 산출 |
|---|---|---|---|
| A | GroundingDINO 프롬프트 구체화 재시도 | OOD 오탐(9%→91~100%)을 프롬프트만으로 줄일 수 있는지 확인 | 기존 `eval_groundingdino_vs_pg2.py`에 prompt variant 비교 추가 |
| B | YOLO-World 비교 | 관계형 표현 가능한 다른 open-vocab detector로 OOD 트레이드오프가 더 나은지 확인 | `ultralytics` 설치(YOLO-World 공식 지원) 후 동일 세트 평가 |
| C | STOP/근접 ablation 설계+실행 (Task #4) | S7 near-miss류는 detector가 아니라 STOP 로직 쪽 문제라는 §H4 결론을 검증 | learned-STOP/proximity-STOP/hybrid 비교 + near-miss 재현 테스트 |

세 단계 모두 **frozen zero-shot 비교 철학 유지**(A/B는 fine-tune 없음), 데이터는 기존 자산(49프레임/OOD11/S6·S7/세션 jsonl) 재사용, 실행은 로컬 GB10.

### 10-A 진행 로그 (완료)
- [x] 프롬프트 변형 4종(기본/구체화/경쟁클래스명시) OOD 11장 + basket 10장 재평가
- 결과: 최선("...chair." 경쟁클래스 명시)도 OOD FP 73% — PG2(9%)에 한참 못 미침. **프롬프트만으로 해결 불가.**

### 10-B 진행 로그 (완료)
- [x] `ultralytics` 설치, YOLO-World(yolov8s-worldv2) zero-shot 동일 세트 평가, conf 0.03/0.05/0.1/0.2 스윕
- 결과: conf=0.05 채택 시 hit92%/OOD18%/lat33ms — latency 18배 빠르고 OOD는 GDINO보다 훨씬 우수.
  단, **S6/S7 실패케이스 5건 중 3건 MISS**(PG2·GDINO는 5/5 HIT). recall 올리면(conf=0.03) tiny 노이즈로 가짜 hit 발생.
- **종합 결정**: GDINO·YOLO-World 둘 다 단독으로 PG2를 이기지 못함 — production 교체 보류, Phase C로 이동.

### 10-C 진행 로그 (완료 — 운영 파라미터 변경은 보류)
- [x] STOP 로직 코드 확인(`stage2_v2_inference_server.py:108-114,486-517`): `STOP_MODE`(proximity 기본값/learned) 토글 이미 존재, hybrid는 없음(추가 필요).
- [x] `scripts/eval/ablation_stop_mode.py` 작성 — 서버 미수정, 기존 `s6_cl_sim.json`/`s7_cl_sim.json` 기록을 재생(replay)해 learned/proximity/hybrid × GOAL_CONSEC_FRAMES(1~5) 비교.
- **핵심 재발견**: S7 "near-miss"는 grounding 실패가 아니었다 — frame 53 area=0.297로 GOAL_AREA(0.25) 실제 초과, 하지만 GOAL_CONSEC_FRAMES=3(연속조건) 때문에 단발성 스파이크로 걸러져 STOP 미발동 → 이후 바스켓 완전히 놓침. `n=1`일 때만 잡힘(`n=2`도 못 잡음 — 진짜 단일 프레임이었음).
- S6은 학습된 헤드가 frame 24에서 자체적으로 STOP 예측, 그 후 105프레임 끝까지 0/82 복귀 — proximity 발동 여부와 무관하게 동일 결과. **§9의 dead-zone 풀프레임 환각(f56/70/85)은 이미 STOP된 이후라 로봇 동작에 영향 없었음** — §9/§H4 결론 일부 수정 필요(grounding_hub.html §I1에 반영).
- **결정**: S7류 실패는 detector 교체(Phase A/B)로 원천 해결 불가능했던 문제였음을 확인(어떤 detector든 같은 연속프레임 필터에 걸렸을 것) — **STOP 로직 쪽 문제라는 가설이 맞았다.**
- ⚠️ **운영 서버(`GOAL_CONSEC_FRAMES` 등) 파라미터 변경은 보류** — 안전 관련 파라미터라 2개 세션 replay만으로 바로 적용하지 않음. 사용자 결정 필요: n=1 전면 적용 / n=2+GOAL_AREA 하향 절충안 / 단일 스파이크에 덜 취약한 대안 로직, 셋 중 방향 확정 후 추가 세션으로 false-positive 재검증 필요.

## 11. 3단계 종합 결론

- **Phase A(프롬프트)+B(YOLO-World)**: 둘 다 production 교체 부적합. GroundingDINO는 OOD에서, YOLO-World는 S6/S7 핵심 실패케이스 recall에서 막힘.
- **Phase C(STOP 로직)**: S7 near-miss는 애초에 detector 문제가 아니라 `GOAL_CONSEC_FRAMES=3`이 단발성 근접 스파이크를 걸러내는 부작용이었음을 확인. **grounding 모델 교체보다 STOP 임계값 튜닝이 훨씬 직접적인 레버.**
- **다음 결정 필요(사용자)**: GOAL_CONSEC_FRAMES/GOAL_AREA 조정 방향 확정 → 추가 세션으로 false-positive 재검증 → 운영 서버 적용.

## 12. GOAL_CONSEC_FRAMES=2 재검증 (2026-06-21, S8 추가) — n=2는 효과 없음으로 확인

S8 production jsonl(`logs/grounding_sessions/gnd_20260620_174429.jsonl`, n=47, **스폿체크로 확인된 거짓양성 frame 20** 포함)을
거짓양성 stress-test로 `scripts/eval/ablation_stop_mode.py`에 추가해 재생.

| consec_n | S7 near-miss(f53) | S8 거짓양성(f20) |
|---|---|---|
| n=1 | ✅ 잡음 | 🚨 거짓양성 STOP 발동 (f20/40/44) |
| **n=2** | ❌ 여전히 못 잡음 | 안전(발동 없음) |
| n=3(현재) | ❌ 못 잡음 | 안전(발동 없음) |

**결론: n=2는 n=3과 완전히 동일한 결과 — 원래 문제(S7)를 안 고치면서 안전하기만 하다.** S7의 f53 스파이크는 진짜 단일 프레임(앞뒤 area 0.07)이라 n=2도 통과 못 함. n=1만 잡는데, n=1은 S8의 실제 확인된 거짓양성에서 바로 false STOP을 낸다.
**`area+cx+연속프레임`만으로는 "진짜 단발 근접"과 "거짓양성 단발 스파이크"를 구분할 수 없다** — consec_n 조정은 더 이상 유효한 레버가 아님. 다른 신호(area 트렌드, 인접 프레임 평균 등) 설계가 필요. 운영값(n=3) 유지를 권장.

## 13. 부가 트랙 — PG2를 VLA(action) 백본으로 쓸 수 있는가 (2026-06-21)

§11 이후 대화에서 파생된 별도 질문: "지금 best 모델(decomposition)은 VLA가 아니다(언어가 action에 0% 기여) — VLA에 가깝게 가려면?"
조사 결과 Exp01~16(Google-robot+Kosmos-2 backbone) 시도는 **백본 자체의 text attention 구조적 붕괴(0.0000%, 전 24레이어)**로 실패했고
(Exp15 frozen-head-only로 LoRA 무관성 확정), PG2는 이 문제를 갖고 있는지 한 번도 검증된 적이 없었음.

**저비용 사전검증** (`scripts/measure_attention_pg2.py`, Exp15와 동일 방법론을 base PG2/frozen에 적용):

| 백본 | text attention(전 레이어) | 방향(L/R/F) 간 차이 |
|---|---|---|
| Google-robot(Kosmos-2 post-train) | **0.0000%** (전 24레이어 고정) | 없음 |
| base PG2(zero-shot) | **40~98%**(레이어별 변동, layer0부터 살아있음) | spread 1.4%p — 미약하지만 반응 |

양성대조군("detect gray basket" vs "detect red ball")도 text attention 87.5%/93.4% — Exp57의 출력레벨 증명(100% vs 0%)이 attention 레벨에서도 뒷받침됨.

**판단**: PG2는 Google-robot과 달리 text pathway가 구조적으로 살아있음 — "PG2 자체를 action backbone으로 재학습"이 Exp01~16과 같은 방식(구조적 text 붕괴)으로 실패할 가능성은 낮다. 단, 방향성 instruction 간 차이(1.4%p)는 객체유무 대비(~6%p)보다 작아 미약함 — **전체 end-to-end 재학습보다 risk가 낮은 절충안**(PG2는 grounding에서만 frozen으로 instruction 반영, action head는 현재처럼 geometry 기반 유지)을 우선 추천. 결과는 `docs/v5/research_story.html` CH2(Google-robot 실패 챕터) 직후에 비교 콜아웃으로 반영함. 전체 end-to-end 재시도는 별도 plan 필요 시 착수.

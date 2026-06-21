# Plan — PG2 instruction-conditioned grounding (VLA에 가장 가깝게, 절충안)

> 작성: 2026-06-21 · 상태: **검토 대기 (승인 전 본구현 금지)**
> 동기: "지금 best 모델(decomposition)은 VLA가 아니다(언어가 action에 0% 기여) — VLA에 가깝게 가려면?"에 대한 결론.
> Exp01~16(Google-robot+Kosmos-2) end-to-end는 백본 자체의 text attention 구조적 붕괴(0.0000%)로 실패.
> base PG2는 동일 방법으로 측정 시 text attention 40~98% — 구조적으로 살아있음 확인(`scripts/measure_attention_pg2.py`, plan §13 `plan_20260621_groundingdino_vs_pg2.md`).
> → **action head는 그대로 두고, 언어는 grounding 단계에서만 작동시킨다.** "find the X" → PG2 zero-shot이 X를 찾음 → 기존 geometric action head가 그 bbox로 주행.
> 연관: `docs/v5/research_story.html`(PG2 vs Google-robot attention 비교 콜아웃), `docs/v5/grounding_hub.html`(§A/E, base PG2 grounding 고정 결론)

---

## 0. 목표 — research_story.html이 정의한 "진짜 VLA"

> "find the gray basket" → grounding → navigation
> "find the brown pot"  → grounding → navigation  (같은 시스템, 텍스트만 변경)
> = Goal-Conditioned Navigation = 진짜 VLA  (research_story.html:9167-9170)

이 정의를 만족시키되, action head 재학습 없이 — 이미 검증된 모듈(PG2 zero-shot grounding, ActionMLP/LSTM)을 그대로 재사용한다.

---

## 1. 현재 코드 상태 (조사 완료)

`robovlm_nav/serve/stage2_v2_inference_server.py`:

- `predict(self, image_b64, instruction="basket")` (446행) — **`instruction` 파라미터가 존재하지만 액션 경로 어디에도 안 쓰임.**
- `PG2Grounder.run(self, image_rgb, ...)` (298행) — **`text="detect gray basket"` 완전 하드코딩**, phrase 파라미터 자체가 없음.
- `/ground` 엔드포인트(788~802행)는 이미 `request.prompt`를 받아 씀(테스트/오프라인 평가용 — 이번 GroundingDINO/YOLO-World ablation에서도 이 경로로 호출했음) — **`/predict`(실주행 경로)만 막혀있는 상태.**
- `ActionMLP`/`LinearHead`/`FCHead`/`LSTMHead` — 전부 `forward(self, x: torch.Tensor)`만 받음. 텍스트 입력 없음 → **이 부분은 그대로 둬도 무방**(객체가 바뀌어도 bbox 좌표/면적만 보고 동작하는 구조라 재학습 불필요할 가능성).

---

## 2. 변경 범위

| 파일 | 변경 |
|---|---|
| `robovlm_nav/serve/stage2_v2_inference_server.py` | ① `PG2Grounder.run()`에 `phrase: str = "gray basket"` 파라미터 추가, 호출부 `text=f"detect {phrase}"`로 교체. ② `predict()`에서 `self.grounder.run(image_rgb, phrase=instruction)`로 전달(현재 `instruction` 기본값 `"basket"` → `"gray basket"`으로 기본값 조정해 하위호환 유지). |
| `scripts/eval/eval_multi_object_grounding.py` | **신규** — 코드 변경 전, base PG2 zero-shot이 바스켓 외 객체에도 비슷한 품질로 작동하는지 먼저 검증(아래 §3). |

**스코프 제한**: 이번 단계는 grounding 프롬프트만 instruction에 연동한다. STOP 임계값(`GOAL_AREA` 등)은 바스켓 크기 기준이라 다른 객체로는 안 맞을 수 있음 — 이번 plan에서는 다루지 않고, 검증 단계에서 필요성만 확인한다(§5).

---

## 3. 검증 순서 (서버 수정 전에 먼저 — 위험 최소화)

1. **멀티 객체 zero-shot grounding 품질 확인** (`scripts/eval/eval_multi_object_grounding.py`, 신규):
   - 대상 이미지: `docs/object_test_images/`(apple, blue_mug, coke_can, chair_obstacle, cone_obstacle — 기존에 이미 있는 5장)
   - 각 이미지에 맞는 phrase로 base PG2 zero-shot 그라운딩(`"detect {object}"`) 실행, hit/area/cx 확인.
   - 비교 기준: §H/§B에서 측정한 바스켓 grounding 품질(hit 98%, cx_MAE 0.126)과 비슷한 수준이 나오는지.
   - **이게 실패하면(바스켓만 잘 잡고 다른 객체는 안 잡히면) 코드 변경 자체가 무의미** — 여기서 멈춤.
2. **GOAL_AREA 민감도 점검**: 검증된 객체들의 실제 화면 점유 면적대를 보고 바스켓 기준(0.25)과 크게 다른지 확인 — 다르면 §5에서 추가 작업 필요 표시만 해둠(이번 plan 범위 밖).
3. 통과 시에만 §2 코드 변경 진행.

---

## 4. 실행 위치

검증(§3)은 로컬 GB10. 코드 변경(§2) 적용 후 실제 동작 확인은 **soda에서 `/predict`를 다른 instruction으로 호출**해보는 단계가 필요한데, 이건 운영 서버 재시작을 수반하므로 **사용자 확인 후 진행**.

---

## 5. 결정 게이트

- §3 통과(멀티 객체 zero-shot이 바스켓 수준으로 작동) → §2 코드 변경 → soda에서 "find the X" 텍스트만 바꿔 같은 시스템이 다른 객체로 가는지 실증.
- §3 실패(바스켓 특화, 다른 객체 zero-shot 품질 낮음) → 코드 변경 보류, 객체별 소량 라벨링/프롬프트 튜닝 필요로 스코프 재정의.
- GOAL_AREA가 객체별로 크게 다르면 → 이번 plan과 별도로 "객체별 proximity 임계값" 후속 plan 필요(STOP 로직 plan §10-C/§12와 연동).

---

## 6. 완료 기준

- [x] `eval_multi_object_grounding.py` 작성 및 실행, 5개 객체 zero-shot 품질 확인
- [x] 결정 게이트 판정: **통과** — apple/mug/coke can/chair/cone 5/5 hit(100%), full-frame 0/5(0%). 바스켓 기준(hit 98%, full-frame 0%)과 동급 이상.
- [x] `PG2Grounder.run()` + `predict()` 코드 변경 — `phrase` 파라미터 추가, `instruction`("basket" 기본값은 "gray basket"로 매핑, 그 외는 그대로 phrase로 사용)을 grounding 프롬프트에 연동.
- [ ] soda 실증 테스트 — **사용자 확인 후 진행** (운영 서버 재시작 필요, 실제 로봇 동작에 영향)

## 7. §3 검증 결과 상세 (2026-06-21)

| 객체 | phrase | hit | area | cx | 판정 |
|---|---|---|---|---|---|
| 사과(녹색) | green apple | ✅ | 0.010 | 0.501 | 정상(작은 객체, 작은 박스) |
| 머그컵(파란) | blue mug | ✅ | 0.035 | 0.520 | 정상 |
| 콜라캔 | red coke can | ✅ | 0.063 | 0.500 | 정상 |
| 의자 | chair | ✅ | 0.125 | 0.518 | 정상 |
| 콘(주황) | orange cone | ✅ | 0.056 | 0.501 | 정상 |

5/5 모두 정상 크기 박스(area가 객체 크기에 비례, cx가 화면 중앙의 실제 위치와 일치), full-frame collapse 없음.
**base PG2 zero-shot grounding은 바스켓 특화가 아니라 일반 객체 인식 능력이다** — instruction 연동 코드 변경의 전제가 확인됨.
산출: `docs/v5/grounding_hub/multi_object_grounding.json`, `docs/v5/grounding_hub/grid_multi_object.png`.

## 8. §2 코드 변경 — 완료 (운영 서버 미배포, soda 미반영)

`robovlm_nav/serve/stage2_v2_inference_server.py`:
- `PG2Grounder.run(..., phrase: str = "gray basket")` — 하드코딩 제거, 파라미터화.
- `predict()` — `instruction`을 phrase로 매핑해 `self.grounder.run(image_rgb, phrase=phrase)` 호출.
- `/predict` API의 `instruction` 필드(이미 스키마에 존재)가 이제 실제로 grounding 프롬프트에 반영됨.

**⚠️ 알려진 제약(이번 범위 밖)**: `grounding_skip_n>1`(프레임 스킵 캐시) 설정 중 `instruction`을 바꾸면 캐시된 이전 객체의 bbox가 한 프레임 동안 재사용될 수 있음 — 현재 기본 동작(skip_n=1, 매 프레임 재계산)에서는 영향 없음. 동적 객체 전환을 자주 할 계획이면 캐시 무효화 로직 추가 필요.

git 커밋만 완료(inference-integration/monavla-driving) — **soda의 운영 서버 코드는 아직 이전 버전 그대로**, pull+재시작 전까지 실제 로봇 동작에는 영향 없음.

## 9. GOAL_AREA 객체별 매핑 (2026-06-21 추가)

전역 고정값(`GOAL_AREA_THRESHOLD=0.25`, 바스켓 기준) 하나로 모든 객체를 처리하면 작은 객체(사과 등)는
너무 늦게 멈춰 충돌 위험이 있음(§3 측정 area가 바스켓의 1/25 수준) — 객체별 매핑으로 전환.

- `robovlm_nav/serve/stage2_v2_inference_server.py`: `GOAL_AREA_MAP`(전역, `configs/goal_area_map.json`에서 로드) +
  `get_goal_area(phrase)` 헬퍼 추가. `predict()`의 proximity 체크가 전역 상수 대신 `get_goal_area(phrase)` 사용.
  매핑에 없는 phrase는 0.25로 폴백(기존 동작 보존).
- `scripts/calibrate_goal_area.py`: **신규** — 캘리브레이션 절차 스크립트. 객체를 "멈춰야 할 거리"에 실제로 놓고
  n회 캡처→`/ground` 측정→median을 `goal_area_map.json`에 저장. 바스켓의 0.25도 사진 환산이 아니라 실주행
  세션(S6~S8)에서 나온 경험값이라, 다른 객체도 동일 철학(실측, 수학적 추정 아님)으로 캘리브레이션해야 함.
- **한계**: `GOAL_AREA_MAP`은 모듈 로드 시 1회만 읆음 — `goal_area_map.json` 갱신 후 반영하려면 서버 재시작 필요(핫리로드 없음, 이번 범위 밖).
- **아직 미실행**: 실제 캘리브레이션(soda에서 물체를 들고 거리 맞춰 촬영)은 물리적 접근이 필요해 사용자 확인 후 진행.

## 10. 배경/장면 일반화 — 확인된 것과 안 된 것

- ✅ **확인됨**: §3의 5장은 서로 다른 복도/조명/배경에서 찍힌 사진이고(사과=유리문 복도, 머그컵=창문있는 회색타일, 콘=유리벽 복도 등)
  객체도 배경도 둘 다 다른 상태에서 5/5 hit — PG2 zero-shot이 학습 분포(바스켓+특정 복도) 밖으로 일반화함을 보임.
- ❌ **미확인**: "동일 배경에서 거리만 달라질 때" area가 얼마나 안정적으로 변하는지 — 이게 GOAL_AREA 캘리브레이션이 필요한 직접적 이유(§9).
  거리 안정성 자체를 별도로 검증하려면, 같은 객체를 여러 거리에서 찍은 시퀀스가 필요(현재 데이터에는 없음 — 캘리브레이션 단계에서 자연히 얻어짐).

# Plan — action head를 PG2 hidden state로 재학습 (T2 본 구현)

> 작성: 2026-06-22 · 상태: **승인됨 — §2 속도 테스트부터 진행**
> 동기: `plan_20260622_fundamental_vla.md` Step B에서 PG2 hidden state에 방향 신호가 90~99%로 인코딩됨을 확인 → "다음 plan"으로 명시했던 본 구현.
> 목표: T2(언어/장면이 경로에 영향) — 정확히는 "장면 정보가 경로 판단에 영향"을 action head 레벨에서 실제로 검증.

---

## 0. 리서치 결과

| 항목 | 사실 |
|---|---|
| 현재 best decomposition 모델 | Exp54 Step2 — `runs/v5_nav/mlp/exp54/stage2_v2/stage2_v2_mlp.pt`, PM 75.9%, closed-loop SR 66.7% |
| 학습 스크립트 | `scripts/train_exp54_stage2_v2_action.py` — AdamW(lr=1e-3, wd=1e-4), CosineAnnealingLR(300 epoch), batch=32, CrossEntropyLoss(class weight), `StratifiedShuffleSplit(test_size=0.2, random_state=42)` |
| 현재 입력 피처 (D_IN=288) | bbox 4종(cx,cy,area,has_bbox)×window(8) = 32 + Kosmos-2 vision 기반 image_proj 256 |
| 학습 데이터 | `docs/v5/bbox_nav_exp46/bbox_dataset_full.json` — **150개 에피소드**, 총 2,626프레임(8-class 분포는 CLAUDE.md 기준) |
| ⚠️ 데이터 불일치 | V5 H5 디렉토리는 현재 244개 파일(이번 세션 probe는 그중 패턴매칭된 220개 사용) — **bbox_dataset_full.json은 150개로 고정, 이후 늘어난 에피소드 미반영**. 이번 plan은 baseline(Exp54 Step2)과 **같은 150개·같은 split**으로 비교해야 공정하므로, 새 에피소드는 포함하지 않음(별도 이슈로 분리) |
| Stage1 인코더 vs PG2 | 서로 다른 모델. Stage1(image_proj 256차원)은 Kosmos-2 vision encoder 기반, hidden state는 PaliGemma2(`google/paligemma2-3b-mix-224`) 기반 — **완전히 별도 forward** |
| hidden state 추출 방식(Step B와 동일) | `model(**inp, output_hidden_states=True).hidden_states[-1][0,-1,:]` → 2304차원, prompt="detect gray basket" |
| ⚠️ 추출 속도 문제 | Step B에서 220회(에피소드당 1프레임) 추출에 GB10에서 **~68분**(프레임당 ~18.5초, CUDA그래프 동적 shape 재컴파일 오버헤드) 걸림. 이번엔 **전체 2,626프레임**(에피소드당 모든 프레임)이 필요할 수 있어 단순 추정 시 **13시간+** — §2에서 완화 방법 검증 필요 |
| 운영 서버 latency | `stage2_v2_inference_server.py`가 PG2Grounder를 이미 로드해서 grounding(bbox)에 쓰고 있음 → **실시간 추론에서는 hidden state를 "같은 forward에서 추가로 캡처"하면 추가 비용 거의 없음**(학습/오프라인 추출만 비용 발생) |

---

## 1. 피처 설계 — 무엇을, 얼마나 추가할 것인가

Step B는 **에피소드당 1개 중간 프레임**으로 "방향 신호가 hidden state에 있다"를 증명했다. 실제 action head는 **윈도우(8프레임) 단위로 매 타임스텝마다** 입력을 받는다. 두 설계안:

| 안 | 내용 | D_IN | 장단점 |
|---|---|---|---|
| **A. 현재 프레임만 추가(권장)** | 기존 288차원(bbox 32 + image 256) + **현재 타임스텝의 hidden state 2304차원** = 2592 | 2592 | 구현 단순, Step B와 동일한 "단일 프레임 hidden state" 가정 유지. 차원이 커서 과적합 위험 → MLP 첫 레이어 폭 늘릴 필요 |
| B. 윈도우 전체에 hidden state | 8프레임×2304 = 18432 + 기존 288 | 18672 | Step B가 검증 안 한 가정(윈도우 전체 신호 가산), 추출량 8배, 과적합 위험 큼 |
| C. hidden state 차원 축소 후 추가 | PCA 또는 학습 가능한 linear projection(2304→64) 후 concat | 288+64=352 | 가장 안전(기존 모델 규모 유지)하지만 PCA fit이 추가 단계, "원시 신호 그대로 넣었을 때도 되는가"를 먼저 보는 게 이번 plan의 본래 질문에 더 충실 |

**제안: A로 시작.** Step B가 검증한 가정과 가장 가깝고(단일 프레임 hidden state), 구현이 단순해 결과를 빨리 볼 수 있다. A가 잘 안 되면 C(차원축소)로 재시도 — B는 비용 대비 가설 검증 가치가 낮아 이번 plan에서 제외.

추가로 baseline과의 정확한 비교를 위해 **"bbox 32차원 유지 + hidden state 추가"** 버전과 **"bbox 32차원을 hidden state로 완전히 대체"** 버전 둘 다 학습해 비교(코드 동일, 플래그만 다름) — 어느 쪽이 나은지도 흥미로운 질문.

---

## 2. 오프라인 hidden state 추출 — 속도 문제 먼저 해결

Step B의 68분/220프레임은 **재학습용으로 쓰기엔 너무 느림**(2,626프레임이면 13시간+). 시작 전에 반드시:

1. **사전 타이밍 테스트**(10~20 프레임만, 새 스크립트 아님 — 콘솔에서 즉석 실행): `torch._dynamo` 관련 cudagraph 경고가 진짜 원인인지 확인하고, 아래 완화책 중 하나로 재측정:
   - `TORCHDYNAMO_DISABLE=1` 또는 `torch._dynamo.config.disable = True`로 eager 모드 강제
   - PG2 forward를 `generate()` 없이 순수 `model(**inp)`만 쓰는 경로(이미 그렇게 하고 있음 — `output_hidden_states=True`만 추가)라 generate 관련 그래프 캡처가 아닐 수도 있음, 정확한 원인 재확인 필요
   - 배치 처리(한 에피소드 내 여러 프레임을 batch로 묶어 한 번에 forward) — 이미지 크기가 동일하면 CUDA그래프 재사용 가능, 속도 개선 기대
2. **속도 개선이 안 되면**: 전체 2,626프레임 대신 **다운샘플링**(예: 윈도우 8프레임 중 마지막 프레임만, 또는 에피소드당 균등 샘플링한 일부 프레임)으로 축소해서 우선 작게 검증 후 확대 — 이 경우도 사용자에게 먼저 보고하고 승인받음.
3. 산출 캐시: `docs/v5/hidden_state_cache/v5_hidden_states.npz` (frame key → 2304-dim fp16 벡터) — fp16 기준 2,626 × 2304 × 2byte ≈ 12MB, 용량 문제 없음.

**이 단계가 끝나고 실제 추출 시간 추정치를 사용자에게 보고 후 진행**(시간이 많이 걸리면 진행 여부 재확인).

---

## 3. 학습 스크립트 변경

- `train_exp54_stage2_v2_action.py`를 직접 고치지 않고 **복제본** `scripts/train_hidden_state_action.py`로 분리(베이스라인 재현 가능성 보존).
- 변경점: `bbox_feat()` 호출부에 옵션 플래그(`--use_hidden_state {none,add,replace}`)로 hidden state 캐시를 로드해 concat/대체.
- **나머지는 100% 동일**(같은 seed=42 split, 같은 optimizer/scheduler/epoch) — 비교가 "피처만 다른" 순수 ablation이 되도록.

---

## 4. 평가

- PM(프레임 단위 정확도): 기존 `scripts/test_v5_pm_dm.py` 패턴 재사용 또는 학습 스크립트 내 val accuracy로 1차 확인.
- Closed-loop: `scripts/sim/evaluate_closed_loop_v5.py --model <new>` 또는 `ablate_stop_proximity.py` 패턴으로 SR/FPE 측정, Exp54 Step2(PM 75.9%, SR 66.7%)와 직접 비교.
- 비교 결과는 `docs/v5/research_story.html`에 새 챕터(CH40)로 기록 — 좋은 결과든 "차이 없음/악화"든 똑같이 기록(이미 CLAUDE.md 문서화 원칙).

---

## 5. 실행 순서 및 위험도

| 순서 | 내용 | 위험 | 비고 |
|---|---|---|---|
| 1 | 추출 속도 사전 테스트(10~20프레임) | 낮음, 수 분 | §2-1 |
| 2 | (속도 확인 후) 전체 150개 에피소드 hidden state 추출 | **불확실 — 최대 13시간**, 완화 성공 시 수십 분 | 사용자에게 추정 시간 보고 후 진행 |
| 3 | `train_hidden_state_action.py` 작성·학습(A안: add/replace 2개 변형) | 낮음 — 기존 학습 스크립트 복제, GPU 시간만 소요 | §3 |
| 4 | PM + closed-loop 평가, baseline과 비교 | 낮음 | §4 |
| 5 | 결과를 CH40으로 문서화 | 낮음 | — |

---

## 6. 완료 기준

- [x] 추출 속도 문제 원인 확인 및 완화 — 실측 결과 우려했던 18.5s/frame이 아니라 **0.22~0.4s/frame**(배치 처리 시 더 빠름).
      전체 2,626프레임 추정 10~17분 → 다운샘플링 불필요, 그대로 전체 추출 진행
- [x] 150개 에피소드(기존 baseline과 동일 집합) hidden state 캐시 생성 — 147/150 성공(3개는 h5 파일 자체가
      디스크에서 삭제/이동되어 없음, baseline 학습 때와 무관한 별개 이슈), 2,572프레임, 9.6분, 10.2MB
- [x] `train_hidden_state_action.py` 작성, add/replace 2 변형 학습 — 둘 다 300 epoch, 2.6분씩
- [x] PM 비교: ~~baseline 75.9% → add 89.2%(+13.3%p) / replace 87.8%(+11.9%p)~~ **정정(2026-06-22, 같은 날 발견)**:
      "baseline 75.9%"는 `train_hidden_state_action.py`에 박힌 하드코딩 참고 문자열이었고, 같은 코드로
      baseline을 직접 재학습하면 **89.76%** — add(89.17%, -0.6%p)/replace(87.80%, -2.0%p) 모두 baseline과
      같거나 약간 낮음. **hidden state 추가가 PM을 올린다는 결론은 착시였다.**
- [x] closed-loop SR 비교: baseline 96.6%(기존 운영 ckpt 기준 — 이 수치는 정정과 무관) → add 93.1%(-3.5%p) / replace 96.6%(동일)
      — PM 정정 후에는 "PM도 안 오르고 SR도 안 오른다"가 정확한 요약
- [x] CH40 결과 문서화 + 정정(40-1b 추가, 40-3 재작성) — 원래 기록은 지우지 않고 정정 카드로 남김

### 결론 (정정됨)
가설(PG2 hidden state에 방향 신호가 있다, CH39 Step B)은 frozen probe 레벨에서는 여전히 유효하다.
하지만 그 신호를 "원시 2304차원 그대로 작은 MLP에 concat/대체"하는 방식으로 재학습해서 꺼내 쓰는 건
**이번 실험에서 효과가 없었다**(PM 동일~소폭 하락, 단일 seed라 노이즈 가능성 있음). 최초 "+13%p"는
apples-to-apples가 아닌 비교(다른 평가 방식의 옛 수치 vs 새로 학습한 수치)에서 나온 착시였고,
hub 통합 작업 중 V5 실제 데이터로 점검하다 발견·역추적해서 즉시 정정함. 다음 후보: 차원축소 후
재시도, 5-seed 평균으로 노이즈 확인, 혹은 (사용자 지침대로) head보다 그라운딩/인식 품질 개선을
우선 — [[project_focus_grounding_for_direction]], `plan_20260622_grounding_quality_and_window_ablation.md`로 이어짐.

## 7. 트레이드오프

- 이번 plan의 가장 큰 불확실성은 "추출 속도"다 — Step B 때처럼 느리면 GPU 시간이 상당히 든다. 그래서 **속도 테스트를 가장 먼저, 별도 승인 게이트로** 둠.
- A안(현재 프레임만 추가)은 Step B가 직접 검증한 가정과 정확히 일치하지만, 윈도우 전체에 신호가 있을 가능성(B안)은 검증하지 않음 — 결과가 애매하면 후속 plan에서 B안 검토 가능.
- 데이터는 기존 150개로 한정(새 에피소드 미포함) — "데이터를 늘려서 좋아진 것"과 "피처를 바꿔서 좋아진 것"을 분리하기 위한 선택.

# 계획 — CH68 L0~L4 언어조건화 프로토콜 실행 (2단계·3단계부터)

## 배경 (리서치 요약, `docs/wiki/language-conditioning.md` 68-1~68-9)

- 트랙 B(61-12, "곡선 방향을 지시문으로")는 CH68에서 라벨 누출 위험이 재확인돼
  폐기 — "curve to the left"류 지시문은 정답을 자연어로 직접 알려줌.
- CH68이 대체 설계한 **L0~L4 프로토콜**(다중 객체 타겟 교체, "go to the chair"
  vs "go to the basket" — 좌/우를 말하지 않아 라벨 누출 원천 차단)이 지금
  가장 검증된 최신 안.
- 68-6 우선순위표: 1~3번(L1 데모, Florence-2 언어부 검정, 객체 전이 실험)은
  **로봇 없이·저비용**으로 가능. 4번(L2, 다중 객체 counterfactual 수집
  ~60ep)부터 물리 수집(로봇) 필요.
- 1번(L1 데모)은 **이미 완료**됨(68-7 성립 확인, 68-8 액션 레벨 확장은 실패,
  68-9에서 원인이 "헤드의 cx 응답 붕괴"로 규명됨).
- **이번 계획 범위**: 2번(Florence-2 언어부 검정) + 3번(객체 전이 실험, 기존
  chair 59ep 중 28ep 활용) — 물리 수집 결정을 내리기 전에 먼저 끝내야 하는
  저비용 선행 단계. 4번(L2 물리 수집)은 이 두 실험 결과를 보고 규모/필요성을
  다시 판단해서 별도 계획으로 진행.

## 왜 이 순서인가

68-5의 미해결 질문 4개 중 앞의 두 개("Florence-2 언어부가 쓸 만한가?",
"새 객체에 액션 헤드가 전이되는가?")에 답이 없으면 L2 수집 규모(몇 ep? 어떤
백본으로 학습할지?)를 정할 근거가 없다 — 지금 바로 soda에 60ep 수집을
요청하면 나중에 "Florence-2 언어부가 못 쓸 물건이었다"거나 "헤드가 새 객체에
전혀 전이 안 된다"는 걸 뒤늦게 알게 될 위험이 있다. 로봇 시간은 비용이 크므로
먼저 이 두 질문부터 답한다.

## 2단계 — Florence-2 언어부(140.2M) 텍스트 임베딩 품질 검정

**질문**: Florence-2의 언어 인코더가 "chair"와 "basket" 같은 타겟 명사를
서로 구분되는 임베딩으로 인코딩하는가? (B 경로: 헤드 concat 조건화의 백본 후보)

**방법**:
1. Florence-2 텍스트 인코더만 로드(`AutoModelForCausalLM`의 `language_model`
   또는 대응 서브모듈 — 정확한 접근 경로는 구현 시 `transformers` Florence-2
   구조 확인 필요)
2. 타겟 명사 후보(basket/chair/ball/cone/mug 등 CH27 zero-shot probe에서 쓴
   객체 리스트 재사용) + 액션 어휘가 섞인 대조군 문장(누출 방지 검증용
   "curve to the left" 등, 68-4 규칙 위반 예시)을 인코딩
3. 임베딩 간 코사인 거리로 ① 서로 다른 객체명 간 분리도, ② 객체명 vs
   방향/동작 어휘 간 분리도 측정 — CH38-4/38-5(hidden state cos_dist)와
   동일한 측정 프레임 재사용
4. 통과 기준: 객체명 간 분리도가 CH38의 "객체 대 방향" 격차(0.36~0.44 vs
   0.03~0.05, 약 10배)에 준하는 수준으로 나오는가

**산출물**: `scripts/probe_florence2_text_encoder.py`,
`docs/v5/detector/florence2_text_embedding_probe.json`

## 3단계 — 객체 전이 실험 (기존 chair 28ep 활용, 신규 수집 없음)

**질문**: 지금 basket 전용으로 학습된 액션 헤드(비전 피처 256d 의존)가
"chair" 장면에 얼마나 전이되는가? 안 되면 L2 수집 규모를 "적은 fine-tune용"이
아니라 "처음부터 균형 잡힌 학습셋"으로 키워야 한다.

**방법**:
1. `ROS_action/mobile_vla_dataset_v5_2/`의 chair 28ep(`docs/v5/bbox_chair/`
   주석 존재)를 exp77과 동일 파이프라인(Florence-2 phrase grounding + vis
   인코딩)으로 캐시 생성
2. 기존 exp77/mlp+soft(D) 체크포인트를 **재학습 없이 그대로** chair 28ep에
   추론만 실행 — basket 전용 헤드가 새 객체(다른 형태·다른 bbox 통계)에서도
   합리적인 액션을 내는지 확인(zero-shot transfer)
3. 별도로 chair 28ep만으로 처음부터 소규모 재학습(few-shot)해서 zero-shot과
   비교 — "아예 안 됨" vs "약간의 fine-tune으로 됨" vs "이미 잘 됨" 3갈래 판정
4. 주의: chair 28ep는 원래 "장애물"로 수집된 데이터라 타겟 궤적 자체가
   basket과 같은 의미인지(목표로 접근하는 궤적인지) 먼저 라벨을 확인해야 함
   (plan_20260707 §6에 이미 명시된 주의사항)

**산출물**: `scripts/eval_chair_transfer.py`,
`docs/v5/closed_loop_eval/chair_transfer_eval.json`

## 이후(별도 계획, 이번 범위 아님)

- 2·3단계 결과를 보고 L2(다중 객체 counterfactual, ~60ep, 좌우 교차 배치
  필수) 물리 수집 규모/필요성 재판단 → soda에 요청
- L3(언어 조건화 헤드 학습·검정, `train_step2_instr_head.py` 재사용) →
  no_text/shuffled_text/wrong_text 대조군 + counterfactual 방향 일치율 측정
- L3 실기 100건 검정(CH64 64-18 기준 프로토콜)

## 완료 기준

- [ ] Florence-2 언어부 텍스트 임베딩 분리도 측정, CH38 기준과 비교 판정
- [ ] chair 28ep 라벨(타겟 궤적 의미) 확인
- [ ] zero-shot 전이 성능 측정
- [ ] few-shot 재학습 성능 측정, zero-shot과 비교
- [ ] 두 결과 종합해 L2 수집 규모/우선순위 판단 → 다음 계획 문서로 넘김

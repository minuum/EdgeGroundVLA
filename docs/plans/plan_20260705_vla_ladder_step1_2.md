# VLA 사다리 ①+② — 언어→타겟 선택 + 언어→정책 조건화

> 작성일: 2026-07-05
> 상태: 승인됨 (사용자 "일단 ㄱㄱ 해보자 2개정도만") — ①, ② 진행
> 배경: docs/v5/grounding_benchmark/CONCLUSION.md — OWL 단일화 판정 완료.
> 현재 파이프라인의 언어 역할은 검출 쿼리뿐. VLA에 가깝게 가는 최소 2단계.

## ① 언어 → 타겟 선택 (OWL 쿼리 연결 정비 + 실증)

현황: 서버 `predict()`는 instruction을 phrase로 넘기지만, `OwlV2Grounder.run()`이
`query = phrase if "gray" in phrase else f"gray {phrase}"`로 **"gray"를 강제 접두** —
"red ball" → "gray red ball"이 되는 버그성 동작. 수정: 접두 제거, phrase 그대로 사용.

실증: 실주행 프레임 몇 장에 서로 다른 쿼리("gray basket"/"door"/"chair" 등)를 줘서
쿼리별로 다른 객체가 그라운딩되는 갤러리 생성 → 언어가 타겟을 바꾼다는 시각 증거.

산출물: 서버 수정(로컬만) + `scripts/demo_owl_query_switch.py` + 갤러리

## ② 언어 → 정책 조건화 (instruction 임베딩을 헤드에 직접 입력)

**아이디어**: OWL 내부의 CLIP 텍스트 인코더(추가 로드 0GB)로 instruction을 512차원
임베딩 → Step2 헤드 입력에 concat. "text attention 0%" 문제를 우회 — 언어가 명시적
feature라 무시 불가능한 경로.

**데이터**: 새 수집 없이 기존 43 에피소드의 path_type(9종)에서 instruction 합성:
  - `left_straight` → "approach the basket on your left side going straight" 등 9문장
  - gt 액션 분포가 path_type마다 다르므로 instruction-행동 상관이 존재

**학습**: bbox_dataset_owl.json 기반, 기존 78.4% 레시피에 text emb(512)만 추가.
비교군 3개 (5-seed 동일 split):
  1. no-text baseline (= 기존 78.4%)
  2. +text emb
  3. +text emb (shuffle control — 학습 시 임베딩을 에피소드간 무작위 섞음)

**핵심 검증 — 언어를 진짜 쓰는가 (PM만으론 판정 불가):**
  a. **Permutation test**: 학습된 ②모델의 eval에서 text를 다른 path_type 것으로
     바꿔치기 → acc 하락폭 = 텍스트 의존도
  b. **Counterfactual flip**: 같은 프레임에 "left쪽" vs "right쪽" instruction →
     예측 액션이 지시 방향으로 이동하는 비율

**성공 기준**: PM이 baseline -2%p 이내 유지하면서 permutation 하락 ≥5%p 또는
counterfactual flip이 방향 일치로 유의미하게 발생 → "언어가 행동에 인과적으로 개입"

**실패 시 해석**: 하락 ~0%p면 이미지가 instruction 정보를 이미 다 담고 있다는 뜻
(합성 instruction의 한계) → 진짜 이질적 instruction 데이터 수집(조이스틱 좌/중/우)이
필요하다는 근거로 기록

## DoD
- [ ] ① OwlV2Grounder gray-접두 제거 (로컬만, soda 보류)
- [ ] ① 쿼리 스위칭 갤러리 생성
- [ ] ② 텍스트 임베딩 생성 + 3비교군 학습 (5-seed)
- [ ] ② permutation/counterfactual 검증
- [ ] 결과를 CONCLUSION.md에 추가

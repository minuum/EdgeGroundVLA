# 리서치 — grounding-액션헤드 융합 레퍼런스 조사 (2026-08-26)

## 왜 다시 보나

CH70(exp78) 결과: deltacx가 무작위 split val_acc는 최고(+0.67%p)지만 R클래스가
5.4%p 하락 — "구조를 바꿔도 특정 클래스를 희생하는 트레이드오프"만 반복되는
패턴. 지금까지 시도한 것(cxgeom/hybrid/film/deltacx/cxaux)은 전부 **"cx를
concat이냐 곱셈이냐 보조손실이냐"** 축 안에서만 변주였다 — 최신 VLA 문헌에서
grounding과 action을 어떻게 융합하는지 참고해서 다른 축을 찾아본다.

## 조사한 레퍼런스 (2025~2026)

- **AffordanceVLA** (arxiv 2606.06155) — Mixture-of-Transformer로 Understanding→
  Affordance→Action 3단 파이프라인. Affordance expert가 Which2Act(grounding)·
  Where2Act(2D heatmap)·Where2Act(3D)를 병렬 서브모듈로 예측하고 **양방향
  cross-attention으로 서로 정제**한 뒤, Action expert가 (semantic context,
  affordance 예측, proprioception) 3개를 받아 실행만 담당. 핵심 아이디어:
  action expert는 "어디를 봐야 하는지 다시 찾지 않고, 이미 정제된 위치 신호를
  받아 실행에만 집중"한다.
- **ReconVLA** (arxiv 2508.10333) — grounded target에 대한 reconstruction
  auxiliary loss로 attention을 명시적으로 앵커링(우리 cxaux의 회귀손실과
  방향은 비슷하나, "위치 좌표"가 아니라 "grounded 영역의 패치 재구성"이 신호).
- **일반 로봇정책 융합 문헌(GateFusion 계열, 로봇 그립 검출)** — naive concat/
  전면 FiLM(입력 분포를 통째로 바꿈)은 사전학습 표현을 왜곡시켜 손해를 볼 수
  있다는 지적이 반복됨. 대안으로 **gated residual fusion**(게이트로 얼마나
  섞을지 스칼라/저차원으로 조절, 원래 표현은 최대한 보존)이 여러 도메인에서
  naive concat/full modulation보다 안정적이라고 보고됨 — 우리 film(FiLM
  전면 재스케일)이 R클래스를 가장 크게 무너뜨린 것과 방향이 일치하는 설명.

## 우리 스케일에 맞게 번역하면

우리는 대형 VLA가 아니라 **256d pooled vis + 4d bbox, window=6프레임의
경량 MLP/Transformer 헤드**다. 문헌의 큰 아이디어를 그대로 못 옮기지만
번역 가능한 것:

1. **게이트를 FiLM보다 약하게** — FiLM(전면 scale+shift, 512차원 파라미터)
   대신 **스칼라/저차원 게이트**로 "이 프레임에서 bbox를 얼마나 신뢰할지"만
   조절(`g = sigmoid(MLP(bbox))`, `fused = g⊙vis + (1-g)⊙vis_detached`
   또는 `g⊙bbox_proj + (1-g)⊙0`) — vis 표현 자체를 왜곡하지 않고 "섞는 비율"만
   학습. film의 R클래스 붕괴(50.8%)를 완화할 가능성.
2. **단일 학습 쿼리의 cross-attention("Perceiver 스타일")** — 지금 배포
   TransformerActionHead는 window 전체에 self-attention(파라미터 많음,
   실제로 mlp보다 계속 낮은 성능 — 과적합 신호일 수 있음). 대신 **학습 가능한
   쿼리 토큰 1개가 window의 [bbox;vis] 6개 토큰에 cross-attention만
   하는 구조**(self-attention 없음, 파라미터 훨씬 적음)로 "grounding이
   action을 찾는" 방향을 좁게 흉내낸다.
3. **체크포인트 선택 방식 자체가 낙관 편향의 일부일 수 있음** — 지금
   `train_one()`/`train_cxaux()` 등은 25epoch마다 val_acc를 재서 **최고
   시점의 state를 그대로 "best"로 채택**한다. 이건 val set에 대해 300/25=12번
   중 최댓값을 고르는 것과 같아서, 특히 val 표본이 작을수록(leave-one-
   direction-out처럼 held-out 방향 1개, ep수 적음) 낙관 편향을 만들 수 있다 —
   사용자가 느낀 "과적합돼 있는 것 같다"는 감각의 실제 원인 중 하나일 가능성.
   개선안: 최종 epoch 고정 채택, 또는 train 내부에서 별도 mini-val(예: train
   에피소드의 15%를 다시 떼어 조기종료 기준으로 쓰고, 원래 val은 순수
   테스트로만 사용)로 이중 분리.

## 제안 — 다음 실험 후보 (승인 후 구현)

| 후보 | 아이디어 | 우선순위 |
|---|---|---|
| A. gatefuse | 스칼라 게이트로 vis·bbox 약하게 결합(FiLM보다 순함) | 높음 — film의 R붕괴 직접 해소 시도 |
| B. actionquery | 학습 쿼리 1개 cross-attn(self-attn 없음, 저파라미터) | 중간 — 배포 transformer가 계속 지는 이유(과적합 가설) 검증 |
| C. 체크포인트 선택 방식 수정 | val 대신 train 내부 mini-val로 조기종료 | 높음 — 방법론 수정, 모든 헤드에 소급 적용 가능 |

C는 헤드 구조가 아니라 **평가 방법론 자체의 수정**이라, A/B보다 먼저 적용해서
지금까지의 exp78/CH69 결과가 실제로 얼마나 낙관적이었는지부터 재보는 게
우선순위가 높다 — "새 헤드가 좋아 보인 것도 이 편향 때문일 수 있다"는 가설을
지금 구조 그대로 반증/확증할 수 있다.

# Flow Matching / π0 스타일 연속 액션

> 이산 8-class 분류가 아니라 연속 action chunk를 flow matching으로 예측하는 MoNa-Pi 설계 사상과 실측(부진).

## 압축 요약

RT-2류 모델의 이산 조향 토큰 분류가 거동을 끊기게 만든다는 문제의식에서, Physical Intelligence의 **π0**를 모티브로 삼아 연속 액션 공간(`[linear_x, linear_y, angular_z]`)을 **Flow Matching**으로 직접 생성하는 **MoNa-Pi** 프레임워크를 설계했다. 핵심 아키텍처는 π0/DiT와 동일한 **AdaLN-Zero** 시간 컨디셔닝(timestep을 MLP로 scale/shift 파라미터로 매핑해 각 LayerNorm 블록에 동적 주입, γ zero-init으로 학습 초기 항등 매핑 유도)이며, 여기에 Action Chunking(16~50 step 동시 예측, 이론상 50Hz 이상 제어 주기)을 결합했다. MoNa-Pi의 차별점은 "Decomposed VLM + Flow Matching Head" 구조로, BBox 임프린팅을 통한 물리적 분해를 추가한 것이다. 그러나 π0/Octo가 수십만~수백만 에피소드 규모의 데이터로 학습되는 반면, 본 연구는 **243 ep**의 소규모 온디바이스 데이터만 사용했다는 근본적 데이터 규모 격차가 있다. 위키 서두에 명시된 대로 실측 결과는 **부진**했으며, 이는 8-class 이산 분류 대비 연속 액션 생성이 요구하는 데이터량을 충족하지 못한 데서 기인하는 것으로 정리된다. 결론적으로 flow matching 자체의 설계(AdaLN-Zero, chunking)는 π0 문헌을 충실히 따랐으나, 소규모 데이터 환경에서는 현재 프로젝트의 주력 방향(8-class 분류 + decomposition, BBox+Image MLP)이 여전히 더 실효적인 선택으로 남아 있다.

---

## 챕터별 원문 발췌 (시간순)

### CH 24 — MoNa-Pi: π0 (Pi-zero) 기반 Flow Matching VLA 프레임워크 설계 사상

① Motivation: π0 모델의 핵심 가치와 연속 액션 제어
기존 VLA 모델들(예: RT-1, RT-2, Kosmos-2)은 조향과 속도 명령을 유한한 이산 토큰(Discrete Tokens)으로 분류하여 출력하므로 거동이 끊기고 부드럽지 못한 조향 제어를 보였습니다.
본 연구는 이를 극복하기 위해 Physical Intelligence의 π0 (Pi-zero) 모델을 모티브로 삼아, Flow Matching(또는 Diffusion) 기술을 기반으로 연속 액션 공간(Continuous Action Space)에서 최적의 주행 제어 궤적을 직접 생성하는 프레임워크를 수립했습니다.
- Flow Matching Action Head: 9개의 불연속 조향 클래스로 분류하는 대신, 실시간 3차원 연속 액션 벡터 [linear_x, linear_y, angular_z]를 예측하여 부드럽고 자연스러운 물리 제어를 도출합니다.
- Action Chunking (Multi-step Prediction): 매 프레임 단일 액션만 예측하는 병목에서 탈피, 한 번에 미래의 N-step(예: 16~50 step) 액션 시퀀스를 동시에 생성해 냄으로써 제어 주기를 50Hz 이상으로 대폭 끌어올릴 수 있는 기틀을 마련했습니다.
② 핵심 아키텍처: AdaLN-Zero (Adaptive Layer Normalization) 컨디셔닝
VLA Flow Model의 핵심은 노이즈 제거 과정의 timestep $t$와 VLM이 추출한 시각/언어 조건부 임베딩 $c$를 어떻게 Action Expert 트랜스포머에 투입할 것인가입니다. MoNa-Pi는 π0 논문 및 DiT(Diffusion Transformer)의 핵심인 AdaLN-Zero 시간 컨디셔닝을 적용했습니다.
- 동적 스케일 및 쉬프트 주입: 단순히 입력 임베딩에 timestep 임베딩을 더하던 기존의 단순 덧셈 방식과 달리, $t$를 MLP에 통과시켜 각 Layer Normalization 블록마다 scale/shift 파라미터 $(\alpha, \beta, \gamma)$로 매핑해 동적으로 곱하고 더해줍니다.
// self-attention block
h = h + γ1 * self_attn( α1 * norm1(h) + β1 )
// mlp block
h = h + γ2 * mlp( α2 * norm3(h) + β2 )
- Zero Initialization 효과: 학습 초기 단계에서 게이팅 변수 $\gamma$를 0으로 초기화(Zero-init)하여 잔차 연결(Residual Block)이 항등 함수(Identity mapping)로 시작하도록 유도하여 학습 안정성을 극대화합니다.
③ 주요 VLA 프레임워크와의 구조적 비교 분석 (π0 기반)
프레임워크
모델 구조
시간 컨디셔닝 기법
제어 출력 및 빈도
필요 데이터 규모
학술적 근거 및 출처
RT-2
Single E2E Transformer
없음 (이산 토큰 분류)
이산 액션 / ~3Hz
수십만 ep + WebLI
RT-2 (CoRL 2023), Section 4.2
Octo Policy
Transformer + Diffusion Head
FiLM (Feature-wise Linear Modulation)
연속 액션 / ~10Hz
Open X-Embodiment
Octo Policy (CoRL 2024), Section 3.1
π0 (Pi-zero)
ViT + Gated LLM + Flow Matching (E2E)
AdaLN-Zero (Adaptive Layer Norm)
연속 액션 / 50Hz
수백만 ep + 교차 로봇 데이터
π0 (Physical Intelligence 2024), Section 3
MoNa-pi (제안)
Decomposed VLM + Flow Matching Head
AdaLN-Zero + BBox 임프린팅 (물리적 분해)
연속 액션 / 50Hz (5ms 이하)
243 ep (소규모 온디바이스)
본 연구, Section 3.1 & 3.2

[→ 원문 전체 보기(research_story.html#ch24)](../v5/research_story.html#ch24)

---

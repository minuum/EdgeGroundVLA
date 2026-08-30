# VLM 백본 비교 (PaliGemma/Kosmos-2/CLIP/Florence-2/Google-robot)

<p class="tagline">어떤 VLM을 백본으로 쓸지, LoRA가 실제로 무엇을 개선/손상시키는지, RoboVLMs 프레임워크 해부.</p>

<div class="summary-box" markdown="1">

**압축 요약**

**Kosmos-2(Google-robot post-train)는 text attention이 전 24레이어에서 0.0000%로 고정** — 이는 우리 LoRA 학습 탓이 아니라 Google-robot post-training 단계에서 이미 텍스트 경로가 구조적으로 붕괴된 것이며(Exp15 head-only 재확인), 언어 지시로 행동을 제어하는 E2E 경로가 이 backbone에서는 근본적으로 불가능함을 뜻한다. 반면 **base PaliGemma2(PG2, zero-shot, LoRA 없음)는 text attention이 레이어별 40~98%로 살아있고**, 방향(L/R/F) 간 spread는 1.4~1.84%p(V6 15개 path_type 평균)로 미약하지만 객체유무 대비(~6%p)보다는 작아 "방향 구분력"은 약하다는 결론이다. **RoboVLMs 프레임워크는 forward_continuous에서 `multimodal_embeds.requires_grad_(True)`가 중간 텐서를 leaf로 재등록시켜 vision_tower와의 계산 그래프를 끊는 구조적 버그**를 갖고 있어, E2E 8조합(top2/4/6/8 × frozen/tuned) ablation 전부 `lora_B=0.000000`으로 vision LoRA gradient가 전혀 도달하지 않았다(실질적으로는 frozen-vision E2E 8회 반복이었음). 우회책으로 grounding용 `generate()` 경로(exp64)를 쓰면 vision LoRA gradient가 정상 도달하지만(48 tensors 확인), **실측 결과 grounding 품질은 개선이 아니라 붕괴**했다 — full-frame 오탐이 전 시점 83~100%로 치솟고, base가 잘하던 중앙/근거리 정밀도(cx MAE)까지 악화시켰으며, 미학습 객체(Eames 의자) 오탐은 base와 동일해 일반화에도 기여하지 못했다. 결과적으로 **grounding은 base PG2(LoRA 없음)를 최종 채택**하고, action 예측은 decomposition 접근(bbox+16×16 이미지 MLP, PM 75.9%, closed-loop 66.7%)으로 우회하는 것이 현재 최선이며, RoboVLMs는 "E2E가 왜 안 되는지 증명한 baseline 인프라"로만 남기고 이후 모든 실험은 이를 우회한다. 한편 CH56(Exp63)에서는 **순수 HF Kosmos-2(Google-robot 아님) + PEFT LoRA에 image projection embeds 수동 추출 우회 패치**를 적용해 20 epoch E2E 학습 시 Val Acc 78.6%를 달성, 순수 backbone 기반이면 text-action 통합 경로 복구가 가능함을 시사했다. **미해결**: PG2를 실제 action backbone으로 재학습했을 때 방향 지시 구분력이 실사용에 충분한지, 그리고 Exp63 경로가 exp64/E2E 8조합의 구조적 문제를 실제로 회피했는지는 추가 검증이 필요하다.

</div>

## 챕터별 원문 발췌 (시간순)

<div class="chapter-block accent-a" markdown="1">

<div class="chapter-block-head"><span class="chapter-badge">CHAPTER 2</span> 첫 번째 접근: End-to-End Policy</div>

<div class="card" markdown="1">

🔴 핵심 발견: Google-robot backbone이 텍스트 경로를 구조적으로 파괴했다
Google robot post-training 단계에서 이미 Kosmos-2의 text attention이 붕괴되었다.
우리의 LoRA fine-tuning과 무관하다. Exp15에서 LoRA 없이 head만 학습해도 text=0% 재확인.
이는 언어 지시로 행동을 제어하는 end-to-end 경로가 현재 backbone에서 불가능함을 의미한다.

</div>

<div class="card" markdown="1">

🟢 2026-06-21 추가 검증 — "다른 백본이면 다를까?" PG2(PaliGemma2)로 동일 측정 재현
Exp15와 똑같은 방법(VLM 완전 frozen, output_attentions로 마지막 토큰의 image vs text 영역 attention 비율 측정)을
base PG2(LoRA 없음, zero-shot)에 적용했다 — scripts/measure_attention_pg2.py.
백본text attention(전 레이어)방향(L/R/F) 간 차이
Google-robot(Kosmos-2 post-train)
0.0000% (전 24레이어 고정)
차이 없음(소수점4자리까지 동일)
base PG2(zero-shot)
40~98% (레이어별 변동, layer0부터 살아있음)
spread 1.4%p — 미약하지만 반응함
양성대조군("detect gray basket" vs "detect red ball")도 text attention 87.5%/93.4%로 매우 높음 — Exp57에서 이미 출력 레벨로 증명된
text-conditioned 행동(100% vs 0% 검출)이 attention 레벨에서도 뒷받침됨. PG2는 Google-robot과 달리 text pathway가
구조적으로 살아있다 — "PG2를 action backbone으로 새로 시도해볼 가치가 있는가"라는 질문에 대한 저비용 사전 검증으로,
본 학습 투자 전에 최소한의 근거를 확보했다. 단, 방향(L/R/F) 간 차이(1.4%p)는 객체유무 대비(basket/ball, 차이 ~6%p)보다 작아
"방향성 instruction 구분력"은 추가 검증이 필요 — 전체 end-to-end 재학습보다는 grounding 단계에서만 instruction을 흘려보내는
절충안(§Section H 이후 논의)이 더 안전한 다음 단계로 판단됨.
2026-07-20 재현 확인: 위 spread(1.4%p)가 고정
이미지 1장의 우연이 아닌지, V6(트랙A+F, 15개 path_type — 극단cx 4곳 + center × 좌/직/우
곡선)의 각 대표 프레임으로 동일 측정을 반복(scripts/measure_attention_pg2_v6.py).
결과: 15개 path_type 평균 spread = 1.84%±1.07%p
(범위 0.29~3.74%p) — 기존 단일 프레임 값(1.40%p)과 같은 자릿수로 재현됨. "PG2가 방향
지시에 약하지만 실제로 반응한다"는 결론이 V6 전반에서 일관됨을 확인 — 다만 전체
end-to-end 재투자보다 grounding 절충안이 낫다는 기존 판단을 뒤집을 만큼 크지는 않음.

</div>

<a class="src-link" href="../v5/research_story.html#ch2">→ 원문 전체 보기 (research_story.html#ch2)</a>

</div>

<div class="chapter-block accent-b" markdown="1">

<div class="chapter-block-head"><span class="chapter-badge">CHAPTER 4</span> 돌파구 — Decomposition 접근법</div>

<div class="card" markdown="1">

**Step 1: bbox history MLP**

bbox center x/y + area history (3프레임)만 MLP에 넣음.
결과: PM 68.4%. 기존 end-to-end Exp11(58.6%)을 이미 넘어섬.
PM 68.4%

</div>

<div class="card" markdown="1">

**Step 2: bbox + 16×16 이미지**

bbox history에 현재 프레임 저해상도 이미지(16×16 gray)를 추가.
center_left/right 같은 경계 케이스 개선.
결과: PM 75.9% (5 seed 평균).
PM 75.9% | CL 66.7% ✅

</div>

<div class="card" markdown="1">

**Feature Ablation 결과**

bbox만: 67.4% ±9.8%
이미지만: 75.6% ±0.8%
bbox+이미지: 76.7% ±1.3%
→ 이미지가 핵심, bbox는 보조

</div>

<div class="card" markdown="1">

✅ 결론: 거대한 VLM 전체를 end-to-end로 다시 학습시키는 것보다,
VLM이 이미 잘 인코딩하고 있는 spatial information(bbox)을 명시적으로 꺼내서
작은 MLP head에 연결하는 것이 훨씬 효과적이다.

</div>

<a class="src-link" href="../v5/research_story.html#ch4">→ 원문 전체 보기 (research_story.html#ch4)</a>

</div>

<div class="chapter-block accent-c" markdown="1">

<div class="chapter-block-head"><span class="chapter-badge">CHAPTER 28</span> LoRA가 Vision을 개선하는가 — E2E는 학습 불가, Grounding은 학습되나 품질 붕괴</div>

<div class="card" markdown="1">

🔬 실험 목적 구조
E2E 8조합 Ablation (완료)
구성: top{2,4,6,8} × {frozen, tuned} = 8개
목적: 어떤 LoRA depth 조합이 action 예측에 유리한가
epoch: 5 epochs × ~84분 = 7시간/실험
총 학습: 56시간 (6/5~6/8, 3일)
⚠️ 핵심 발견
모든 8개 모델에서 lora_B = 0.000000
→ Vision LoRA가 E2E 경로에서 gradient 미도달
원인: forward_continuous의 multimodal_embeds.requires_grad_(True)가 vision_tower를 loss 그래프에서 분리
→ 실제로는 "frozen-vision E2E" 8조합 비교였음
✅ 그럼에도 유효한 기여
Vision은 frozen이었지만 action head + proj 조합에 따른 성능 차이는 실측됨
→ "어떤 depth의 LoRA가 MLP에 유리한가" 는 여전히 유효한 ablation
exp64 — Vision Grounding LoRA (진행 중)
구성: SigLIP layers 19-26, q/k/v만
목적: Grounding forward에서 vision LoRA가 실제 학습되는가
데이터: 8960샘플 (V5 pos 1500 + hard-neg 7500)
epoch: 15 epochs (진행 중)
✅ 핵심 차이
Grounding forward는 generate() 경로 사용
→ vision_tower gradient가 정상 역전파
trainable tensors: 48개 확인
epoch 1 loss = 1.4728 ✅ (학습 중)
🎯 완료 후 비교 대상
base PG2: full-frame 0%
→ [결과: CH31] exp64 full-frame 92% — 뛰어넘기는커녕 붕괴
📐 실험이 말하는 논리 흐름
1
E2E Ablation: "Vision LoRA는 E2E 경로에서 구조적으로 학습 불가"
8조합 전부 lora_B=0 → RoboVLMs forward 수술 없이는 불가. 재학습(6/9 게이트 테스트)에서도 재확인.
2
reframe: "Grounding forward 경로로 우회하면 되지 않나?"
Action이 아닌 BBox 탐지(grounding)는 generate() 경로 → vision까지 gradient 도달. 목적도 action이 아닌 grounding 품질 향상.
3
exp64: "Grounding LoRA는 실제로 작동하는가?" — 완료: 학습은 되나 품질 붕괴 (CH31)
trainable 48개 학습 확인(gradient 도달 O). 단 실측 결과 full-frame 92%로 박스 품질 붕괴 — base PG2가 더 정확. → vision LoRA는 grounding을 악화시킴.
4
6/4 교수님 피드백 대응 범위
이 두 실험이 직접 답하는 것: R2-2 LoRA 기여도, R6 Grounding 지터링 개선
답하지 않는 것: R4(조향 오실레이션 → Y-Center Gate로 해결), R3(단일 데이터 → 새 수집 필요)
📊 실험 현황 요약
실험
조합 수
Vision LoRA
상태
핵심 결론
E2E top2_proj_frozen
1/8
lora_B=0
완료
frozen-vision E2E baseline
E2E top2/4/6/8 × tuned
4/8
lora_B=0
완료
proj tuning 효과만 비교
E2E top4/6/8_proj_frozen
3/8
lora_B=0
완료
depth만 다른 frozen 비교
exp64 (Grounding LoRA)
1
48 tensors ✅
epoch 1/15
grounding 경로 vision LoRA 최초 작동 확인
판정 결과 (CH31 완료): exp64 full-frame 92% vs base PG2 0% — 미개선(붕괴).
→ base PG2를 최종 grounding 모델로 확정. "vision LoRA는 grounding 경로에서 gradient는 도달하나 박스 품질을 붕괴시킨다"가 결론. (전 모델 비교: Grounding Hub)

</div>

<a class="src-link" href="../v5/research_story.html#ch28">→ 원문 전체 보기 (research_story.html#ch28)</a>

</div>

<div class="chapter-block accent-d" markdown="1">

<div class="chapter-block-head"><span class="chapter-badge">CHAPTER 29</span> RoboVLMs 해부 — 의도한 실험과 실제 기여의 괴리</div>

<div class="card" markdown="1">

📄 RoboVLMs — 공식 저장소 팩트
논문
Towards Generalist Robot Policies: What Matters in Building Vision-Language-Action Models
Xinghang Li et al. · arXiv:2412.14058
Nature Machine Intelligence 게재 수락
설계 목표
타겟: CALVIN, SimplerEnv, Open X-Embodiment
설계: "30줄 코드로 임의 VLM → 로봇 정책"
HF 공개 체크포인트: KosMos 기반 3종만
마지막 업데이트: 2025년 초 이후 거의 없음
공식 지원 Backbone 현황 (GitHub README 기준)
Backbone
검증 상태
우리 프로젝트와 관련
KosMos-2
✅ 완전 검증
우리 Exp01~Exp16 (Kosmos backbone) — 공식 지원 범위 내
Flamingo
✅ 완전 검증
미사용
LLaVA
✅ 완전 검증
미사용
PaliGemma
⚠️ not fully tested
E2E 8조합이 이 상태로 돌아간 것 — robopaligemma.py 존재하지만 미검증
PaliGemma 2
❌ 언급 없음
exp64는 RoboVLMs를 우회한 이유가 여기에도 있음
Qwen, Uform, MoonDream
⚠️ 미완전
미사용
핵심: E2E 8조합 ablation은 공식적으로 "not fully tested" 상태의 PaliGemma backbone 위에서 돌렸다. lora_B=0 버그가 단순 우리 설정 문제가 아닌 이유 — PaliGemma 통합 자체가 미완성이었다.
⚡ 의도한 실험 vs 실제로 일어난 일
의도했던 것
- top{2,4,6,8} SigLIP 레이어에 LoRA 적용
- 각 depth에서 Vision 인코더가 얼마나 개선되는지 비교
- proj frozen vs tuned와 교차해 16조합 중 8개 측정
- "더 깊은 LoRA → 더 좋은 grounding?" 답하기
실제로 일어난 일
- 모든 8조합에서 lora_B = 0.000000
- Vision LoRA 파라미터 전혀 업데이트 안 됨
- 실제 변수는 projector frozen vs tuned 1개뿐
- 깊이별 차이 없음 — "frozen-vision E2E" 8회 반복
🔬 RoboVLMs 구조 결함 — 발견 과정
문제 코드: base_backbone.py › forward_continuous
# forward_continuous 내부 (~line 1294)
if multimodal_embeds.requires_grad is False:
multimodal_embeds.requires_grad_(True) # ← 이게 문제
# requires_grad_(True)를 중간 텐서에 직접 호출하면:
# → 해당 텐서가 leaf tensor로 재등록됨
# → vision_tower와의 계산 그래프 연결이 끊김
# → 역전파 시 vision_tower까지 gradient 미도달
# → lora_B에 쌓이는 gradient = 0
forward_continuous 경로
Action 예측용 (E2E 학습)
vision_tower → 인코딩 → re-leaf → LM → action head
→ vision LoRA gradient 0
generate() 경로
BBox 탐지용 (grounding)
vision_tower → 인코딩 → decoder → token
→ vision LoRA gradient 정상
확인 방법
학습 후 lora_B tensor 직접 출력
→ E2E 8조합 전부 0.000000
→ exp64는 epoch 2에 TP=100%
exp64가 generate() 경로를 쓰는 이유가 바로 이것. RoboVLMs를 우회해 직접 HF trainer로 돌려야 vision LoRA가 실제로 학습된다.
✅ RoboVLMs를 통해 실제로 얻은 것
①
E2E 0% → Decomposition 전환 근거 (음성 결과도 결과)
Exp11 closed-loop 0% vs Step2 66.7%. RoboVLMs 기반 E2E가 작동하지 않는다는 것을 실험으로 증명했기 때문에 decomposition 전환이 설득력을 가짐. 논문에서 "우리가 E2E를 시도했고 왜 안 되는지 알아냈다"는 서사로 쓸 수 있음.
②
구조 버그 발견 → exp64 방향 설정
lora_B=0을 발견하지 않았으면 "왜 vision LoRA가 안 먹히는지" 영원히 모른 채 E2E를 반복했을 것. 버그 발견이 generate() 경로 우회 → exp64 설계로 이어졌음.
③
text attention = 0% 원인 확정 (Google-robot pretrain 기인)
Exp15 head-only + E2E 8조합 모두에서 text=0% 재확인. 우리 LoRA 학습 방식 탓이 아닌 Google-robot post-training이 text 경로를 붕괴시킨 것으로 확정. Kosmos-2 backbone 선택의 타당성도 함께 재검토됨.
④
projector tuning 효과 (제한적이지만 유효)
vision LoRA depth 비교는 불가능했지만, frozen vs tuned projector 비교는 실제로 일어났음. action head 학습 시 projector를 같이 풀면 어떻게 되는지에 대한 데이터는 남아있음.
🗺 RoboVLMs — 앞으로의 포지션
현재 우리 결과를 내는 경로
Decomposition Step2 MLP → RoboVLMs 미사용
exp64 Grounding LoRA → RoboVLMs 미사용
E2E ablation → RoboVLMs 사용, vision LoRA 깨짐
RoboVLMs 한계
PG1만 지원 (PG2 없음)
CALVIN manipulation 설계 — 내비게이션 미고려
forward_continuous vision LoRA gradient 차단
수정 금지 (third_party)
결론: RoboVLMs는 E2E baseline 실험 인프라로서 역할을 마쳤다. 앞으로 새 실험(grounding LoRA, 의자 데이터 수집, MLP 재학습)은 모두 RoboVLMs를 우회하는 경로를 사용한다. 유지하되, 신뢰하지 않는다.

</div>

<a class="src-link" href="../v5/research_story.html#ch29">→ 원문 전체 보기 (research_story.html#ch29)</a>

</div>

<div class="chapter-block accent-e" markdown="1">

<div class="chapter-block-head"><span class="chapter-badge">CHAPTER 32</span> LoRA가 얻은 것과 실패한 데이터 — 시점별 해부</div>

<div class="card" markdown="1">

✅ LoRA로 실제 얻은 것 (정직하게)
① 방법론적 입증: vision LoRA가 generate() 경로로 실제 학습된다는 것 확인 (E2E의 lora_B=0과 대조). 이것이 exp64의 원래 목적이었고, 성공.
② 학습한 negative 억제: val에서 person/red_ball/brown_pot 오탐 0%. 단 학습한 3종에 한정 — 미학습 의자엔 일반화 안 됨(아래 ④).
③ 원거리에서만 우연한 cx 개선: base가 원래 못 잡던 far 시점(L-far MAE 0.30, R-far 0.37)에서 exp64가 약간 낮음(0.23, 0.34). 하지만 이는 실력이 아니라 full-frame 박스가 멀리 있는 작은 바구니를 "어쩌다" 덮어서 생긴 착시.
④ 못 얻은 것: 정밀 localization(전 시점 full-frame), 미학습 객체 오탐 감소(의자 FP 그대로), 근거리 검출 안정성(아래).
📊 시점 9버킷 — base vs exp64 (hit / full-frame / cx_MAE)
시점
base hit
exp64 hit
base full
exp64 full
base MAE
exp64 MAE
C-far
100%
100%
0%
100%
0.020
0.033
C-mid
100%
83%
0%
83%
0.020
0.033
C-near
100%
67%
0%
67%
0.073
0.009
L-far
100%
100%
0%
83%
0.300
0.231
L-mid
100%
100%
0%
100%
0.058
0.149
L-near
100%
100%
0%
100%
0.090
0.156
R-far
83%
100%
0%
100%
0.371
0.335
R-mid
100%
100%
0%
100%
0.135
0.186
R-near
100%
100%
0%
100%
0.013
0.141
full = full-frame율(area>0.9), MAE = cx 중심 오차(낮을수록 정확). exp64 full-frame이 전 시점 83~100%로 붕괴.
🎯 어떤 데이터를 못 잡았나
exp64의 가장 의외의 실패 — C-near
가장 쉬워야 할 가까운 중앙 바구니에서 검출 67%(2/6 miss). base는 100%.
→ 바구니가 화면을 크게 채우면 full-frame 박스 로직이 깨지면서 아예 박스를 못 내놓는 역설.
miss 프레임: C-mid#9, C-near#7, C-near#8
공통 오탐 — eames_zenith 의자
미학습 의자 11장 중 딱 1개(검정 가죽 Eames 체어)를 base·exp64 둘 다 basket으로 오탐(area 0.80).
→ PG2가 어두운 곡면 의자를 바구니로 착각. LoRA가 이 오탐을 전혀 못 고침(9%→9%).
base의 약점 — 원거리 정밀도
base는 far 시점에서 cx 오차 큼(L-far 0.30, R-far 0.37) + R-far 1프레임 miss.
→ 멀리 있는 작은 바구니의 정확한 좌측/우측 위치 판정이 base의 한계.
exp64가 망친 것 — 중앙/근거리 정밀도
base가 잘하던 C-mid(0.020)·L-mid(0.058)·L-near(0.090)에서 exp64는 0.033·0.149·0.156으로 모두 악화.
→ full-frame 박스라 중심이 화면 정중앙으로 고정되며 실제 바구니 위치를 놓침.
📐 한 줄 종합
LoRA는 "vision 경로 학습 가능"이라는 방법론과 학습한 3종 negative 억제를 얻었지만,
그 대가로 전 시점 full-frame 붕괴 + 근거리 검출 실패 + 중앙 정밀도 악화를 치렀다.
미학습 객체(Eames 의자) 오탐은 base와 동일 — LoRA가 일반화엔 기여하지 못함.
→ grounding은 base PG2, decomposition은 그 위에서. exp64는 "박스 크기 페널티 없는 grounding LoRA의 실패 사례"로 기록.
🔍 후속 발견 — exp59가 교수님이 본 "객체 인식 못함"의 정체
exp64(vision LoRA)는 전면 붕괴라 배포되지 않았지만, 실주행에 쓰인 exp59(LM LoRA)는 표준 프레임에선 멀쩡(full-frame 6%)해 보였다.
그러나 7모델을 측면·free극단·증강에 전수 비교(Grounding Hub)한 결과 — exp59는 실환경 변동(로봇거리 22%·저조도 17%·대각선 14%·회전 8~16%)에서 간헐적으로 full-frame 붕괴한다.
그 순간 action head는 "바구니=화면 전체" 신호를 받아 조향이 무너진다 → 교수님이 관찰하신 오예측의 실제 메커니즘.
base PG2는 같은 조건 full-frame 0~2% — grounding을 base PG2로 교체하면 이 간헐 붕괴가 사라진다.

</div>

<a class="src-link" href="../v5/research_story.html#ch32">→ 원문 전체 보기 (research_story.html#ch32)</a>

</div>

<div class="chapter-block accent-a" markdown="1">

<div class="chapter-block-head"><span class="chapter-badge">CH 56</span> Exp63 — 순수 HF Kosmos-2 E2E VLA 학습 완수</div>

<p class="chapter-subtitle-line">Google-robot post-trained 백본의 구조적 Text Attention 붕괴(0%)를 회피하여, 순수 백본 기반 튜닝으로 E2E VLA 복구 유효성 검증</p>

<div class="card" markdown="1">

**56-1. 학습 사양 및 정량 성과**

- 훈련 데이터셋: V5 Trajectory 140개 에피소드
- 학습 에폭: 20 epochs
- 우회 패치 적용: PEFT (LoRA) 래핑 시 vision model의 image projection embeds 수동 추출 매핑을 통해 backbone gradient 유실 방지.
- 최종 검증 정확도 (Val Acc): 78.6%
순수 VLM 백본 기반 파인튜닝이 텍스트-액션 통합 경로(Text Pathway)의 구조적 복구에 실제로 기여할 수 있음을 실증했습니다.
2026-06-28  |  관련: scripts/train_exp63_e2e_kosmos.py, runs/v5_nav/e2e/exp63

</div>

<a class="src-link" href="../v5/research_story.html#ch56">→ 원문 전체 보기 (research_story.html#ch56)</a>

</div>

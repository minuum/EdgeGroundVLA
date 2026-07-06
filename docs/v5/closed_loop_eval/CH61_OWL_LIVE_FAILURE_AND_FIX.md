# CH61 — 실로봇 OWL-v2 배포 첫날: 방향 편향 원인 규명 + vis_feat 정규화 버그 발견

> 작성일: 2026-07-07
> 배경: 7/6 OWL-v2(th=0.25) 실배포 첫 실로봇 테스트에서 obj_left/right 실패 다수 관측
> 선행: docs/v5/grounding_benchmark/CONCLUSION.md, docs/v5/closed_loop_eval/CH60_OWL_TEXT_CLOSED_LOOP.md

## 요약

실로봇 obj_right 테스트가 반복 실패(SR 0/16, 세션 로그 기준)한 원인을 추적한 결과,
**그라운더(OWL-v2) 문제가 아니라 두 가지 별개 이슈**로 좁혀졌다:
1. **exp71 헤드가 실전 flicker 분포(has_bbox 40~60% 결측)를 학습 때 거의 못 봄**
   (학습 데이터는 has_bbox 95.9~97.8%) — 재학습으로 개선 가능성 확인
2. **재현/검증 파이프라인 자체에 있던 vis_feat L2-정규화 누락 버그** 발견 — 오늘 만든
   "진짜 exp71 레시피" 실험 다수가 이 버그 상태로 진행됐다가 발견 후 재검증

## 1. 실로봇 실패 관측 (obj_left/right, 32 에피소드 누적)

| 경로 | n | 성공 | 특이사항 |
|---|---|---|---|
| obj_left (타겟 좌측) | 8 | 3 | top액션이 오히려 우측계열(ROT_R/FWD+R) 6/8 |
| obj_right (타겟 우측) | 16 | **0** | 방향 맞아도(7/16) 전부 실패 |

7/6 실측 OWL-v2 세션(171922 preview=True, 172030 preview=False) 직접 h5 분석:
- 172030: cx가 0.75→0.94로 실제 우측 드리프트가 있었는데도 **14프레임 전부 FORWARD 고정**
- 171922: preview 스캔 전체(10프레임)에서 탐지율 0%

## 2. 그라운더 vs 헤드 — 어느 쪽 문제인가

- **PG2 시절 세션 재확인**: "방향으로 간" 것처럼 보였던 세션들은 cx가 0.50 근처로 거의
  안 움직이는데도 FWD+L→RIGHT→FORWARD 블록으로 방향이 바뀜 → **grounding cx가 아니라
  raw image 자체로 모델이 판단한 것**으로 추정 (feature ablation: image_only 75.6% >
  bbox_only 67.4%와 일치)
- **preview 로직 코드 확인**: `inference_count==0`일 때만, 최대 5회, ROT_L/ROT_R만 반환
  가능 — FWD+L/RIGHT 같은 다중 블록은 preview가 만들 수 없는 값 → **PG2 시절 sweep은
  preview가 아니라 본 모델의 자체 판단이었다** (최초 가설 정정)
- **45ep 경량 프록시 실험**: exp71과 전혀 다른 아키텍처도 동일 cx 궤적에서 100% 동일하게
  FORWARD로 수렴 → 그라운더/헤드 특정 조합이 아니라 **데이터 분포 자체의 문제**

## 3. soda 관측: OWL-v2 flicker + 조건 무관 액션 수렴 (FINDING_20260706)

- 세션당 has_bbox=False 40~60%, 근접 직후(area 임계값 통과 직후) 검출이 끊기는 패턴
- 학습 데이터 has_bbox=95.9%(bbox_dataset_base_pg2_cx.json) / 97.8%(exp71 실제 소스
  bbox_dataset_pg448_cx.json) — 실전 flicker율과 **큰 분포 괴리**

## 4. 재학습 실험 (1차: 45ep 프록시 → 2차: 실제 exp71 레시피)

### 45ep 프록시 (decomposition MLP, IID flicker)
baseline이 랜덤 dropout에 이미 강건(78.5%→77.2%) — 그러나 IID flicker는 실전(상관형)과
다르므로 결론 보류.

### 상관형 flicker(근접 직후 집중) + 실제 exp71 레시피 (150ep, FrozenCLIPV2+Transformer)

| 변형 | val_acc | 진동율 |
|---|---|---|
| baseline (w6) | 98.4% | 3.0% |
| dropout_aug | 97.1% | 5.2%(최악) |
| sticky_aug | 98.1% | 1.9% |
| **window3** | **98.7%** | **1.9%** |

조합(sticky+window3, 확률 스윕)은 단일 window3보다 낫지 않음 → dropout류 증강은 폐기,
window3 단순 축소가 최선 후보.

**그러나 진짜 성공기준(FPE/SR/TLD, rollout_core 리플레이)으로 재확인하면 window3와
window6(운영)이 사실상 동률**(SR 97.7% 동일, FPE 0.087 vs 0.091m) — 진동율 지표로 낸
"window3 우세" 결론은 철회. 리플레이 자체가 카메라 피드백이 없는 근본적 한계
(CH60-c 기존 지적)라 더 이상 이 방식으론 판별 불가.

## 5. 치명적 버그 발견: vis_feat L2-정규화 누락

운영 서버(`Stage1Encoder.encode_image`, `stage2_v2_inference_server.py:381`)는
`F.normalize()`로 이미지 feature를 L2 정규화하는데, 연구용 재현 스크립트
(`train_exp71_stage2_transformer.py`의 `FrozenCLIPV2.encode_batch`)는 정규화를 안 함.

**soda 실제 7세션(72~78) 실측 bbox로 검증**:

| 세션 | 정규화 없이 재현 일치율 | 정규화 후 재현 일치율 |
|---|---|---|
| 231153 | 80.0% | 93.3% |
| 233327 | 25.0% | **91.7%** |
| 233424 | 11.1% | **88.9%** |
| 233159 | 46.7% | 86.7% |

**영향 범위**: 이번 세션에서 "진짜 exp71 레시피"라고 진행한 실험 다수(flicker
robustness, 헤드 구조 비교, truth_mini 검증, window3 vs window6 라이브 비교)가 전부
이 버그 상태로 이뤄짐 — 이후 실험(§4 후반부, §6)은 정규화 수정 후 재검증한 결과.

## 6. 셀프라벨링 데이터(bbox_truth_mini) 활용 — 오염 발견 및 정정

- 1차 시도: truth_mini(72프레임/18ep) clean-bbox 검증에서 **3개 헤드 전부 100%** —
  그러나 18ep 중 **15개가 이미 학습셋에 포함**돼 있어 오염(암기) 의심
- **격리 재학습**(truth_mini 18ep 완전 제외, 132ep 재분할) 후 재검증:

| | val_acc | truth_mini 진짜 held-out acc |
|---|---|---|
| baseline_w6 | 97.0% | 95.8% |
| window3 | 94.8% | **98.6%** |

진짜 held-out에서도 95.8~98.6%로 여전히 높음 — **헤드가 clean bbox에 대해 실제로
일반화한다는 것 자체는 신뢰성 있게 확인됨.** (다만 정적 분류 정확도라 폐루프 진동은
검증 불가 — §4 한계와 동일)

## 7. VLA 사다리 ② (언어 조건화) 재검증 — 버그 수정 후, 결론이 더 강하게 부정적으로 바뀜

| 비교군 | PM (43ep 프록시, 버그 상태) | PM (150ep 실제 레시피, 버그 수정) |
|---|---|---|
| no_text | 78.4% | 87.5% |
| with_text(real) | 81.8% (+3.4%p) | **85.0% (오히려 하락)** |
| shuffled_text | 77.5% | 80.6% |

- Permutation: 두 실험 모두 큰 하락폭(−14.6%p / **+16.1%p**) — 텍스트를 실제로 참조함
- **Counterfactual 변화율: 43ep 실험 20.3% → 150ep 실제 레시피 정확히 0.0%** —
  왼쪽/오른쪽 지시를 강제로 바꿔도 예측이 단 하나도 안 바뀜. 버그를 고치니 오히려
  "텍스트는 경로 맥락(prior)일 뿐 명령이 아니다"라는 결론이 더 명확해짐

**결론**: 조이스틱 이질 지시 데이터(장면-지시 상관이 깨진 데이터) 없이는 언어 조건화가
헤드 구조/레시피를 어떻게 고쳐도 명령 순응으로 이어지지 않는다는 게 이번 재검증으로
더 확고해짐.

## 8. 프리뷰 재설계 검토 (docs/plans/plan_20260706_preview_redesign.md)

옵션 A(제거)/B(양방향 탐색)/C(threshold 조정)/D(로깅 강화) 중 **D → A 순서 추천**.
다음 로봇 테스트(obj_right preview=False 5개 추가 수집)가 그대로 A의 실측 데이터가 됨.

## 종합 결론 및 다음 단계

1. **그라운더(OWL-v2)는 무죄에 가까움** — clean bbox 검증(§6)에서 헤드 자체 일반화力 확인
2. **진짜 병목은 (a) flicker 분포 불일치, (b) 언어 미조건화** 두 가지
3. **오프라인 리플레이 방법론은 한계에 도달** — 카메라 피드백 부재로 더 이상 판별 불가,
   실로봇 A/B(window3 재학습 헤드 vs 기존 window6)가 사실상 유일하게 남은 확정적 검증
4. **다음 우선순위**: (a) 실로봇 window3 vs window6 A/B, (b) 조이스틱 이질 지시 데이터
   수집 설계 착수, (c) preview 옵션 D(로깅 강화)부터

## 산출물

- `scripts/train_owl_flicker_robustness.py`, `scripts/eval_correlated_flicker_oscillation.py` — 45ep 프록시
- `scripts/train_exp71_flicker_robustness.py`, `scripts/train_exp71_multihead_truthmini.py` — 실제 exp71 레시피(버그 있는 상태)
- `scripts/eval_truthmini_holdout_and_cl.py` — truth_mini 격리 재학습 + FPE/SR/TLD 리플레이
- `scripts/eval_live_sessions_owl_flicker_windows.py` — soda 실제 세션 재현 + 정규화 버그 발견
- `scripts/train_exp71_instr_conditioning.py` — VLA 사다리 ② 정규화-수정 재검증
- `docs/plans/plan_20260706_preview_redesign.md` — 프리뷰 재설계 플랜

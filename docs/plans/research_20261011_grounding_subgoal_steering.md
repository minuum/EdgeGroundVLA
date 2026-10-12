# 리서치: 그라운딩 기반 서브골 조향 (Grounding-based Subgoal Steering)

> 작성일: 2026-10-11
> 배경: EGVLA3용 "FM이 엣지 내비게이션에서 더 나쁘다"는 주장은 우리가 진짜 연속 회귀 액션 데이터를 수집한 적이 없어 논거가 성립하지 않음(폐기). 대신 EGVLA2의 핵심 발견("그라운딩 품질이 성공을 좌우", gnd%≥80 → 98.8% 성공)과 π0.7의 "지각(월드모델) → 서브골 → 정책 조향" 구조 아이디어를 결합해, flow-matching 없이도 적용 가능한 방향을 찾는다.

## 1. 기존 데이터/코드 조사 결과

### 1.1 `docs/v5/bbox_nav_step1/bbox_dataset.json`
- 45개 에피소드(list), 각 에피소드에 `path_type`(9종: center/left/right × straight/left/right), `episode`(이름), `frames`(list) 보유
- 프레임 필드: `frame_idx`, `gt_class`(0-7, 8-class 정답), `cx`,`cy`(정규화 bbox 중심), `area`(정규화 bbox 면적), `has_bbox`(bool)
- raw xyxy 없음, 타임스탬프 없음 → **1회성 Kosmos-2 그라운딩 결과를 오프라인으로 캐시한 것**이지 실시간 추론 결과가 아님

### 1.2 Exp14 Step2 (BBox+Image MLP) — `scripts/test_v5_bbox_nav_step2.py`
- feature: window=3 bbox 히스토리(`cx,cy,area,has_bbox`×3=12차원) + 16×16 그레이스케일 이미지(256차원) **단순 concat**
- `nn.Sequential(Linear(d_in,256)→ReLU→Dropout→Linear(256,128)→ReLU→Dropout→Linear(128,64)→ReLU→Linear(64,8))`
- gating/cross-attention/분리 branch 없음 — "조향"이라 부를 구조적 장치 전혀 없음

### 1.3 `scripts/ablate_bbox_image_features.py`
- bbox_only 67.4%±9.8% / image_only 75.6%±0.8% / bbox+image 76.7%±1.3% (PM, 5 seeds)
- 동일하게 concat-then-shared-MLP. bbox 자체의 정보량이 16×16 이미지보다 낮다는 기존 결론(CLAUDE.md "핵심 발견 2")과 일치

### 1.4 `robovlm_nav/datasets/nav_h5_dataset_impl.py` — `grounding_aux`
- `self.grounding_aux_index` (`_load_grounding_aux_index()`, L315, config key `grounding_aux`) — 사전 라벨 truth 파일(`docs/v5/bbox_truth_mini.json`)에서 `bbox_xyxy_norm`, `coarse_label` 로드
- `__getitem__`(~L852-867)에서 윈도우 텐서 `grounding_bbox`, `grounding_bbox_mask`, `grounding_bbox_weight`, `grounding_coarse_label`, `grounding_coarse_mask` 생성
- **단, 현재는 보조 손실(aux loss) 타겟으로만 쓰이고, 메인 액션 헤드의 조건 입력으로는 쓰이지 않음** → 이것이 기존 인프라 중 "프레임별 타겟 bbox 신호"에 가장 가까운 것

### 1.5 "subgoal" 관련 기존 코드
- `grep -rn "subgoal\|goal_bbox\|target_bbox\|steer"` 결과: 실제 서브골 추상화는 없음
- `tools/model_analysis/medium_term/advanced_multimodal_model_v2.py`(L158-254)에 `GoalDecomposer`(연속 2D 액션 기반, `num_subgoals=3`) 존재하나 **현재 V5 파이프라인과 단절된 구버전 프로토타입** — 재사용 불가, 참고만 가능
- "steer"는 VLM 프롬프트 문자열("Steer left toward...")로만 등장, 제어 신호 아님

### 1.6 `scripts/sim/evaluate_closed_loop_v5.py` (step2 추론)
- `eval_step2_episode()`(L384-413): bbox는 **매 프레임 캐시된 `bbox_dataset.json`에서 읽음**, 이미지만 h5에서 매 프레임 새로 읽어 16×16 리사이즈
- **closed-loop 평가에서도 실시간 그라운딩 추론은 실행되지 않음** — 전부 오프라인 bbox 재생(replay) 기반
- exp67/exp71/exp72 등 다른 모드도 동일 패턴(`bbox_dataset_pg448_cx.json` 등 다른 캐시 파일 사용)

## 2. 타당성 판단

- **"그라운딩 출력 → 서브골 신호 → 이산 헤드 조향"은 기존 코드의 리팩토링이 아니라 신규 아키텍처**다. 재사용 가능한 부품은:
  - `grounding_aux`의 bbox/coarse_label 텐서 파이프라인(이미 VLM 학습 데이터셋 클래스에 통합됨) → 보조손실이 아니라 메인 헤드 조건 입력으로 전환하는 최소 변경 경로
  - Exp14 bbox_dataset.json과 Step2 MLP 코드 → concat 대신 "게이팅/조향" 메커니즘으로 대체할 베이스라인
- **"실시간 온라인 그라운딩 조향"은 범위 밖**이다. 현재 closed-loop 평가 전체가 오프라인 캐시 bbox 재생 구조이고, 실시간 그라운딩 모델을 루프에 붙이는 작업은 별도의 큰 엔지니어링 과제(레이턴시, 적시계산)이며 6장 단기 논문 범위를 넘어선다. **겸손하게 "오프라인/캐시 bbox 기반 ablation"이라고 명시해야 함.**
- 데이터 재사용성은 높음: 새 수집 없이 기존 `bbox_dataset.json`(45 ep) + Exp14 feature pipeline을 그대로 쓸 수 있음 → 턴어라운드 짧음.

## 3. 겸손하게 성립 가능한 주장 (초안)

> "동일한 그라운딩 출력(cx, cy, area)을 **단순 concat 피처**로 넣는 것과, **조향(steering) 신호로 명시적으로 분리해 게이팅/컨디셔닝**하는 것 사이에 차이가 있는가?"

- 비교 대상: (A) 기존 Exp14 concat-MLP (baseline, 이미 76.7%) vs (B) bbox를 별도 서브골 임베딩으로 만들어 FiLM/게이팅 방식으로 이미지 특징에 주입하는 구조
- 주장 범위: "오프라인 캐시 그라운딩 신호를 액션 헤드에 주입하는 방식(단순 concat vs 명시적 조향)이 성능에 미치는 영향" — FM/연속 액션과 무관, 온라인 추론 주장도 하지 않음
- 데이터/레이블 수집 불필요, 기존 `bbox_dataset.json` 재사용, 모델 변경만 필요

## 4. 다음 단계 제안

승인되면 `plan.md`에서 다음을 구체화:
1. (B) 구조 설계 — FiLM 스타일 bbox→이미지 특징 게이팅 or cross-attention 소형 버전 (경량 제약 고려)
2. leave-one-direction-out 또는 기존 5-seed PM 프로토콜 재사용한 평가 설계
3. 6장 논문 틀: 서론(그라운딩 중요성 기존 발견 재확인) → 관련연구(π0.7의 구조적 아이디어, 단 FM은 미사용 명시) → 방법(concat vs steering 두 구조) → 실험 → 고찰 → 결론

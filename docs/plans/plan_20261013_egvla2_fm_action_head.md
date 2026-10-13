# 계획: EGVLA2 — 진짜 Flow-Matching 액션헤드(π0 형태 등) 학습 가능성 검증

> 작성일: 2026-10-13 | 상태: **승인 대기 — 아직 코드 작성 안 함**
> 관련: [plan_20261012_egvla2_florence2_unification.md](plan_20261012_egvla2_florence2_unification.md), CH75(비교), 위키 flow-matching-vla / action-head-architecture CH63-6·CH73-1

## 0. 목적 (사용자 요청 재확인)

비교 실험이 아니라, **연속 액션 데이터로 FM 액션헤드(π0 형태 또는 다른 형태)를 실제로 구성해 학습이 되는지** 확인한다.
성능 우열 주장은 하지 않는다. 판정 대상은 "학습 루프가 수렴하는가 / 샘플링 액션이 정답 분포와 맞는가 / 어떤 형태가 이 데이터 규모에서 학습 가능한가"이다.

## 1. 리서치 결과 (읽기만 수행)

### 1.1 기존 FM 시도와 그 한계
- `scripts/train_exp73_stage1v3_heads.py:205` `FlowMatchingHead`(transformer 컨텍스트 + MLP 속도장, rectified flow, 10-step Euler)와 `scripts/pilot_flow_matching_head.py`(synthetic 검증만)가 이미 있다.
- 과거 flow 헤드 비교(CH63-6: 71~75%, 이산·연속·FM 비교 53.8%)는 **타깃이 8-class LUT 고정값(±1.15)** 이라 FM 평가로 **무효**(CH73-1에서 폐기). 즉 "FM이 부진했다"는 기록은 FM의 한계가 아니라 이산 타깃 위에서 FM을 돌린 결과다.
- 진짜 연속 타깃이 처음 필요했고, pilot 스크립트도 "실데이터 대기 중"이라고 명시했다.

### 1.2 신규 연속 데이터 (2026-10-13 수신, 읽기로 확인)
- 위치 `ROS_action/mobile_vla_dataset_egvla2/`, h5 23개(+json 2), 1,417프레임, 평균 62프레임/에피소드, 5Hz 추정.
- 속성 `control_mode="continuous"`, `action_space="egvla2_continuous_v1"`, `action_chunk_size=8`. 데이터셋 `actions (T,3)` 연속, `action_classes (T,)` 역산 8-class(참고용).
- 값: 고유값 170개, lin_x −0.87~1.0, lin_y −1~1, ang_z −1~1 (V6는 41개 이산값, ±1.15).
- **경로 유형이 center_right_curve 20개 + center_straight 3개뿐** — 다양성 매우 낮음.
- 이미지 720×1280, V6와 동일 포맷.

### 1.3 입력 특징 경로
- 현재 서브골/헤드 실험은 Florence-2 캐시(bbox 4-d×3프레임 + 768-d 풀링 특징)를 사용. 신규 데이터는 캐시가 없어 **소규모 전처리(23ep, 수 분)** 가 필요하다.
- 공간 특징(4×4 격자) 실험이 진행 중(CH75-8 후속). 결과에 따라 prefix 토큰 구성을 정한다.

## 2. 후보 구조 (3종, 모두 동일 인터페이스: 조건 입력 → 액션 chunk H×3 생성)

공통: rectified flow, `x_t=(1-t)x0+t·a`, 타깃 `a−x0`, 추론은 Euler N스텝. 액션은 ACTION_SCALE로 정규화(연속 범위가 ±1이라 1.0 사용, 확정은 구현 시 데이터로). H=8(수집 attrs의 action_chunk_size).

### A. MLP-FM (최소형, 기존 `FlowMatchingHead` 재사용·적응)
```python
v = velocity_net(cat[c, t_emb, x_t_flat])      # c: 조건 벡터, x_t_flat: (H*3)
```
- 장점: 구현 거의 없음, 학습 가능성 하한선 확인용. 단점: chunk 내부 구조 활용 안 함.

### B. FiLM-FM (팀 합의 "액션헤드 MLP→FM(FiLM)"에 가장 가까움)
```python
s = subgoal_enc(bbox_window)                   # 검증된 SubgoalSteered 방식
gamma, beta = film(cat[s, t_emb]).chunk(2)     # 시간까지 FiLM으로 주입
h = gamma * img_proj(img_feat) + beta
v = trunk(cat[h, x_t_flat])                    # 속도장 H*3
```
- 서브골(bbox)로 조향하는 기존 검증 구조에 flow 시간 조건만 확장. 가장 작은 변경.

### C. π0형 action expert (AdaLN-Zero 트랜스포머)
```python
prefix = [subgoal_token, vision_tokens(풀링 1 또는 4×4=16)]      # 조건 prefix, 고정
act_tokens = Linear(x_t) + pos_emb(H)                            # H개 액션 토큰
block: h = h + g1 * attn(a1 * norm(h) + b1, kv=[prefix, h]); h = h + g2 * mlp(a2*norm(h)+b2)
       (a,b,g) = MLP(time_emb);  g는 zero-init
v = Linear(h)                                                    # 토큰별 속도
```
- π0/DiT의 핵심(AdaLN-Zero, chunk 토큰, prefix 조건)을 경량(2층, d=128)으로 구현. 소규모 데이터에서 과적합 가능성이 가장 큼 → 학습 가능성 판정 대상.

## 3. 데이터 구성 옵션

| 옵션 | 내용 | 평가 | 판단 |
|---|---|---|---|
| 1 | 신규 연속 23ep만 | 학습 루프·수렴·샘플 품질 확인 가능, 일반화 주장 불가 | **기본** |
| 2 | V6 225ep(이산 LUT 타깃) FM | 타깃이 8개 점이라 FM의 **다봉 분포 생성** 능력만 시험(전환 모호성), 연속성은 시험 불가 | 보조 |
| 3 | 1+2 혼합 | action_space가 달라 스케일·의미 혼재 위험 | 이번 범위 제외 |

## 4. 검증 지표 ("학습 가능한가" 판정용, 우열 비교 아님)

1. 1-batch 과적합 테스트: loss가 0 근처로 떨어지는가 (구현 정상성)
2. 에피소드 단위 hold-out(5-fold 정도, 23ep): 학습/검증 flow loss 곡선
3. 샘플링 액션 vs 정답: 축별 MSE·R², 정답이 이산 램프일 때 전환 근처 거동
4. 기준선(비교가 아니라 sanity): "평균 액션", "ContReg MLP 회귀" 대비 FM이 무의미하게 나쁘지 않은가
5. 샘플 다양성: 같은 조건에서 K번 샘플 시 분산(다봉성) — FM의 의의 확인
6. `action_classes` 역산 라벨과의 일치(참고)

## 5. 생성/수정 파일 (승인 후)

| 파일 | 변경 |
|---|---|
| `scripts/precompute_egvla2_continuous_features.py` (신규) | 신규 h5 23개 → F2 bbox(어댑터)+768-d(+선택 4×4) 캐시, 프레임별 연속 액션 chunk 타깃 구성 |
| `scripts/train_egvla2_fm_head.py` (신규) | A/B/C 구조 + rectified flow 학습 + Euler 샘플링 + 지표 |
| `docs/v5/egvla2_fm/summary_*.json` (신규 출력) | 결과 |
| 기존 `FlowMatchingHead`, 로더 | **수정 없음** (A는 import/복사로 재사용) |

## 6. 리스크 / 트레이드오프

- **데이터 규모·다양성(가장 큼)**: 23ep, 사실상 단일 경로. 결과는 "파이프라인이 돈다/수렴한다"까지만 주장 가능. π0형 C는 파라미터 대비 데이터가 부족해 과적합 가능성 높음 → 그 자체가 판정 결과.
- 경로 유형이 한 종류라 방향 일반화는 검증 불가. 데이터 확대(방향·곡선 조합) 후 재판정 필요.
- 연속값은 조이스틱 아날로그 입력 → 사람 입력 잡음이 타깃에 포함. 샘플 품질 평가 시 이 잡음 수준을 먼저 추정(정답 자체의 프레임 간 변동).
- 닫힌 루프 성공률은 이번 범위 밖(PM 개념 없음). 실주행은 별도.
- 다른 세션(로봇)이 데이터를 계속 추가 중 — 이번 학습은 수신 시점 스냅샷(23ep)으로 고정하고 파일 목록을 결과에 기록.

## 7. 체크리스트 (승인 후 진행)

- [ ] 연속 액션 램프 형태·잡음 수준 확인 (정답 자체의 변동성, `action_classes` 일치율)
- [ ] `precompute_egvla2_continuous_features.py` 작성 + 캐시 생성
- [ ] 구조 A(MLP-FM) 구현 + 1-batch 과적합 테스트
- [ ] 구조 B(FiLM-FM) 구현
- [ ] 구조 C(π0형 action expert) 구현
- [ ] 5-fold 에피소드 hold-out 학습/지표 수집 (A/B/C)
- [ ] 옵션 2(V6 이산 타깃) 다봉 분포 시험 (보조)
- [ ] 결과를 CH75에 "FM 학습 가능성" 탭으로 추가 + 로컬호스트 링크

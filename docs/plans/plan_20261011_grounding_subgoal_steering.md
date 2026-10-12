# 계획: 그라운딩 기반 서브골 조향 (concat vs FiLM steering)

> 작성일: 2026-10-11 | 리서치: [research_20261011_grounding_subgoal_steering.md](research_20261011_grounding_subgoal_steering.md)
> 상태: **승인 대기 — 아직 코드 작성 안 함**

## 0. 범위 재확인 (리서치에서 합의된 것)

- 비교 대상은 **(A) 기존 concat-MLP(Exp14 baseline, 76.7%) vs (B) bbox를 명시적 서브골 임베딩으로 분리해 FiLM식 게이팅으로 이미지 특징을 조향하는 구조**
- 그라운딩 신호는 **오프라인 캐시(`bbox_dataset.json`)** 그대로 사용 — 실시간 그라운딩 추론 주장 안 함
- 연속 액션/flow-matching 관련 주장 안 함 — 여전히 8-class 이산 타겟
- 새 데이터 수집 없음

## 1. 접근 방식

### 1.1 베이스라인 (A) — 변경 없음
기존 `scripts/ablate_bbox_image_features.py` / `scripts/test_v5_bbox_nav_step2.py`의 concat-MLP를 그대로 재실행해 동일 시드/스플릿으로 재확인만 한다 (이미 있는 76.7% 수치 재현, 새 코드 아님).

### 1.2 신규 (B) — FiLM 스타일 서브골 게이팅
bbox 특징(cx, cy, area, has_bbox 히스토리, 12차원)을 **작은 MLP(서브골 인코더)**로 저차원 임베딩 `s`로 만들고, 이 `s`로 이미지 특징 경로에 FiLM(Feature-wise Linear Modulation) 변조를 적용한다:

```python
class SubgoalSteeredHead(nn.Module):
    def __init__(self, img_dim=256, bbox_dim=12, hidden=128, n_classes=8):
        super().__init__()
        self.subgoal_enc = nn.Sequential(
            nn.Linear(bbox_dim, 32), nn.ReLU(),
            nn.Linear(32, hidden),
        )
        # FiLM: subgoal embedding -> (gamma, beta) per image-feature channel
        self.film = nn.Linear(hidden, hidden * 2)
        self.img_proj = nn.Linear(img_dim, hidden)
        self.trunk = nn.Sequential(
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(self, bbox_feat, img_feat):
        s = self.subgoal_enc(bbox_feat)
        gamma, beta = self.film(s).chunk(2, dim=-1)
        h = self.img_proj(img_feat)
        h = gamma * h + beta          # 서브골이 이미지 특징을 "조향"
        return self.trunk(h)
```

- 파라미터 수는 기존 concat-MLP와 거의 동일하게 맞춤(공정 비교, "더 큰 모델이라 이겼다" 반박 방지)
- 학습/평가 프로토콜은 기존 `ablate_bbox_image_features.py`의 5-seed PM 방식을 그대로 재사용 (새 프로토콜 설계 안 함)

### 1.3 평가 지표
- PM(5 seeds, 기존과 동일 split) — (A) vs (B) 직접 비교
- 추가로 "has_bbox=False" 프레임(그라운딩 실패 프레임)에서의 각 방식 강건성 별도 집계 — 서브골 신호가 없을 때 게이팅이 identity에 가깝게 행동하는지 확인 (FiLM의 안전장치 역할 검증)
- closed-loop 평가는 **이번 단기 논문 범위에 포함하지 않음** (기존 `evaluate_closed_loop_v5.py`는 step2 전용으로 구조가 다름 — 별도 통합 작업 필요, 범위 초과)

## 2. 수정/생성될 파일

| 파일 | 변경 |
|---|---|
| `scripts/eval_subgoal_steering_vs_concat.py` (신규) | (A) 기존 concat-MLP 재실행 + (B) FiLM steering 신규 학습, 동일 split/seed로 PM 비교, JSON 저장 |
| `docs/v5/closed_loop_eval/subgoal_steering_vs_concat.json` (신규 출력) | 결과 저장 |
| 기존 `scripts/ablate_bbox_image_features.py`, `scripts/test_v5_bbox_nav_step2.py` | **수정 없음** — 베이스라인 재현용으로 import/참조만 |

## 3. 트레이드오프

- **장점**: 새 데이터/레이블 불필요, 기존 인프라 100% 재사용, 6장 논문에 적합한 분량, FM 관련 잘못된 주장 리스크 없음
- **한계 (논문에 명시할 것)**:
  - 45 에피소드/794프레임은 통계적으로 작은 표본 — 논문에서 "예비 결과(preliminary)"로 명시
  - FiLM 구조가 이길지는 불확실 — 리서치 단계에서 bbox_only(67.4%)가 image_only(75.6%)보다 낮다는 기존 결론을 보면, bbox 자체의 정보량이 낮아 게이팅으로도 큰 개선이 없을 수 있음 → **결과가 음성(negative)이어도 "단순 concat으로 충분하다"는 결론으로 논문화 가능**(이것도 유효한 기여)
  - closed-loop(실제 주행) 통합은 범위 밖 — PM(오프라인 지표)만으로 결론 내림을 명시

## 4. 체크리스트 (승인 후 진행)

- [x] `scripts/eval_subgoal_steering_vs_concat.py` 작성
- [x] (A) 베이스라인 재현 수치 확인 — 76.32%±0.99% (기존 ablation 76.7%±1.3%와 사실상 일치, 재현됨)
- [x] (B) FiLM steering 학습/평가, 5 seeds — **78.21%±1.61%**, +1.9%p, 파라미터는 더 적음(79,336 vs 110,536)
- [x] has_bbox=False 프레임 강건성 비교 — 두 방식 동일(75.0%), 차이 없음
- [x] 결과 JSON 저장 (`docs/v5/subgoal_steering_eval/summary.json`) + HTML 리포트(`docs/v5/subgoal_steering_eval/index.html`)

## 5. 결과 (2026-10-11)

| 방식 | PM (5-seed) | 파라미터 수 | has_bbox=False 정확도 |
|---|---|---|---|
| Concat-MLP (baseline) | 76.32%±0.99% | 110,536 | 75.0% |
| **Subgoal-Steered FiLM** | **78.21%±1.61%** | **79,336** | 75.0% |

- FiLM 서브골 조향이 concat보다 **+1.9%p 우세하면서 파라미터는 28% 더 적음** — 경량화 방향과 정합적인 긍정 결과
- 단, 45 에피소드 규모의 예비 실험이며, 오프라인 캐시 그라운딩 재생 기반 — 실시간 온라인 조향이 아님을 반드시 명시
- has_bbox=False(그라운딩 실패) 프레임에서는 두 방식 차이 없음 — FiLM이 "그라운딩 실패 시 안전하게 identity로 수렴"하는 효과까지는 확인되지 않음(향후 과제)

## 6. 확장 실험 (2026-10-11, 사용자 요청 1·2·3·4) — `scripts/eval_subgoal_steering_extended.py`

15 seed로 확장, train/test loss 추가 기록, 파라미터 매칭 베이스라인(concat_small) 및 그라운딩 실패 identity 정규화(steered_reg) 변형 추가. paired t-test(df=14) 유의성 검증.

| 조건 | PM(15-seed) | params | test_loss |
|---|---|---|---|
| Concat (원본) | 76.15%±1.94% | 110,536 | 1.471 |
| Concat-Small (steered와 params 매칭) | 76.24%±1.69% | 72,304 | 1.429 |
| **Steered (FiLM)** | **78.97%±2.43%** | 79,336 | 2.091 |
| **Steered + Identity 정규화** | **81.28%±2.11%** | 79,336 | 1.837 |

- steered vs concat: Δ=+2.82%p, t=4.310, **p=0.0007** (유의)
- steered vs concat_small(파라미터 매칭): Δ=+2.73%p, t=5.559, **p=0.00007** (유의 — 파라미터 수 차이로 설명 안 됨, 구조 자체의 이득)
- steered_reg vs steered: Δ=+2.31%p, t=3.230, **p=0.006** (유의 — 그라운딩 실패 시 identity 유도 정규화가 실제로 효과 있음, 73-3에서 "향후 과제"로 남겼던 가설이 확인됨)
- 관찰된 이상 현상: steered 계열이 test_loss는 오히려 더 큼(acc는 높은데 확신도 분포가 다름) — class-weighted CE 특성으로 추정, 과적합 신호는 아니나 원인 미확인 → 향후 과제
- 결과: `docs/v5/subgoal_steering_eval/summary_extended.json`, `docs/v5/subgoal_steering_eval/extended.html`

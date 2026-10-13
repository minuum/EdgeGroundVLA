# 계획: EGVLA2 — 서브골 인코더에 텍스트 결합 (FiLM 확장) + 액션 헤드 MLP→FM

> 작성일: 2026-10-12 | 리서치: [research_20261012_egvla2_text_subgoal.md](research_20261012_egvla2_text_subgoal.md) §1-4
> 상태: **구현·실험 완료 (2026-10-13) — 결과: 텍스트 결합 PM 하락(채택 근거 없음)**

## 0. 범위 재확인

- 팀 합의(2026-10-12): 액션 헤드 MLP → **FM(FiLM)** — 이미 검증된 `SubgoalSteeredMLP` 방향과 동일, 여기서는 이 FiLM에 **텍스트를 추가 조건**으로 넣는 확장만 다룸.
- Florence-2 백본 전환은 별도 plan(`plan_20261012_egvla2_florence2_unification.md`)에서 다룸 — 이 plan은 **현재 OWLv2+Kosmos-2 파이프라인 위에서 텍스트 FiLM만 우선 검증**(백본 교체와 독립적으로 먼저 결론 낼 수 있음).
- 자유형 자연어 지시 학습은 범위 밖 — 템플릿 기반 지시문(V6 5-direction 반영 신규 템플릿)까지만.
- 텍스트 인코더는 **frozen**으로 고정, 학습 안 함.

## 1. 접근 방식

### 1.1 텍스트 인코더
- HF `transformers.CLIPTextModel` + `CLIPTokenizer` 사용 (frozen, `.venv`에 `transformers==4.49.0` 설치 확인됨). 로컬 가중치 없으면 `openai/clip-vit-base-patch32` 다운로드.
- **환경 이슈**: 현재 eval 스크립트들은 `.venv_lerobot`(transformers/clip 미설치)로 실행 중. 두 옵션:
  - (A) 텍스트 임베딩을 **사전 추출해 npz 캐시**(Stage 1 패턴 재사용)로 저장한 뒤, 학습/평가는 기존 `.venv_lerobot`에서 캐시만 로드 — **권장**(환경 변경 없이 기존 파이프라인 유지)
  - (B) `.venv_lerobot`에 패키지 설치 — 환경 오염 리스크, 비권장
- 채택: (A). `scripts/precompute_v6_text_embeddings.py` (신규)가 `.venv`로 실행되어 텍스트 임베딩 캐시 생성.

### 1.2 V6 방향별 텍스트 템플릿 (신규)
기존 `PATH_TYPE_INSTRUCTIONS`(left/right/straight 3종)는 weak/strong 구분이 없어 V6의 5-direction을 못 담음. 신규 5종 템플릿 작성:

```python
V6_DIRECTION_INSTRUCTIONS = {
    "center":       ["Go straight toward the basket", "바구니를 향해 똑바로 가"],
    "weak_left":    ["Steer slightly left toward the basket", "바구니 쪽으로 살짝 왼쪽으로 틀어"],
    "strong_left":  ["Turn sharply left toward the basket", "바구니 쪽으로 크게 왼쪽으로 꺾어"],
    "weak_right":   ["Steer slightly right toward the basket", "바구니 쪽으로 살짝 오른쪽으로 틀어"],
    "strong_right": ["Turn sharply right toward the basket", "바구니 쪽으로 크게 오른쪽으로 꺾어"],
}
```
각 에피소드의 `direction` 필드로 템플릿 선택, 프레임마다 `random.choice`로 변형 샘플링(기존 관례와 동일).

### 1.3 구조 — FiLM에 텍스트 결합 (옵션 (a): concat 후 단일 FiLM, 리서치에서 선정한 단순안)
```python
class TextBBoxSteeredMLP(nn.Module):
    def __init__(self, bbox_dim=BBOX_DIM, text_dim=512, hidden=HIDDEN):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, 32)       # CLIP text(512d) 축소 투영
        self.subgoal_enc = nn.Sequential(
            nn.Linear(bbox_dim + 32, 32), nn.ReLU(),
            nn.Linear(32, hidden),
        )
        self.film = nn.Linear(hidden, hidden * 2)
        self.img_proj = nn.Linear(IMG_DIM, hidden)
        self.trunk = nn.Sequential(
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, 64), nn.ReLU(),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, bbox_feat, text_feat, img_feat):
        t = self.text_proj(text_feat)
        s = self.subgoal_enc(torch.cat([bbox_feat, t], dim=-1))
        gamma, beta = self.film(s).chunk(2, dim=-1)
        h = self.img_proj(img_feat)
        h = gamma * h + beta
        return self.trunk(h), (gamma, beta)
```
- `text_proj`로 512d→32d 축소해 bbox_feat(4~6d)와 균형 맞춤(텍스트가 FiLM을 지배하지 않도록, 리서치 §3 우려 반영).
- 기존 `steered_reg`의 identity-regularization은 bbox `has_bbox=False`일 때만 적용(텍스트는 항상 존재하므로 조건 변경 없음).

### 1.4 "텍스트가 실제로 영향을 주는가" 검증 프로브 (리서치 §4-1, 필수)
동일 bbox + 다른 text를 주입했을 때 예측 action이 달라지는지 확인하는 synthetic ablation:
```
for 고정된 bbox 샘플 1개:
    for text in [center, weak_left, strong_left, weak_right, strong_right 템플릿 각 1개]:
        model(bbox, text_embed(text), img) 예측 action 기록
→ 5개 예측이 서로 다르면 "텍스트가 실제로 조향에 반영됨" 지지
→ 5개가 전부 같으면 bbox가 지배적 → Exp54 masking 검증과 유사한 패턴으로 "text 무시" 결론
```
Exp54 5-Track 검증([[project_exp54_5track_validation]])과 동일한 masking/probe 패턴 재사용.

### 1.5 평가 지표
- PM(10 seeds, V6 기준, 기존 `eval_subgoal_steering_extended_v6.py` 프로토콜 그대로) — `steered_reg`(bbox만) vs `text_bbox_steered`(bbox+text) 비교
- §1.4 텍스트 영향력 프로브 결과(정성 + flip rate 정량)
- closed-loop(offline replay)는 참고 지표로만 표기, PM이 주 지표(사용자 피드백 반영: "PM으로 봐야겠지")

## 2. 수정/생성될 파일

| 파일 | 변경 |
|---|---|
| `scripts/precompute_v6_text_embeddings.py` (신규, `.venv`로 실행) | CLIP text encoder로 V6 5-direction 템플릿 임베딩 사전 추출, npz 캐시 저장 |
| `docs/v5/bbox_nav_owl/v6_text_embed_cache.npz` (신규 출력) | 텍스트 임베딩 캐시 |
| `scripts/eval_subgoal_steering_text_v6.py` (신규) | `TextBBoxSteeredMLP` 학습/평가, 기존 `eval_subgoal_steering_extended_v6.py` 구조 복제+확장 |
| `scripts/eval_text_influence_probe.py` (신규) | §1.4 텍스트 영향력 검증 |
| 기존 `eval_subgoal_steering_extended_v6.py` | **수정 없음** — 베이스라인(bbox만) 비교 기준으로 유지 |

## 3. 트레이드오프

- **장점**: 기존 FiLM 인프라 최소 확장, 텍스트 인코더 frozen이라 학습 비용 거의 안 늘어남(캐시 재사용), Florence-2 전환과 독립적으로 먼저 결론 가능
- **리스크**:
  - 템플릿이 direction과 1:1 고정이라 "텍스트 자체의 일반화 능력"은 검증 못 함 — §1.4 프로브로 "최소한 text가 무시되지 않는다"까지만 증명 가능, "임의 지시를 이해한다"는 주장은 안 함(overclaim 방지)
  - CLIP text(512d)가 커서 축소 투영 차원(32d) 선택이 결과에 민감할 수 있음 — 민감도 체크 필요시 16/32/64 간단 스윕
  - V6 템플릿과 PATH_TYPE_INSTRUCTIONS(기존 3종)가 공존하게 되어 추후 학습 파이프라인(`nav_h5_dataset_impl.py`)과의 일관성 정리 필요(이 plan 범위에서는 eval 스크립트 전용으로만 사용, 메인 학습 코드는 건드리지 않음)

## 4. 체크리스트 (승인 후 진행)

- [x] V6 5-direction 텍스트 템플릿 작성
- [x] `precompute_v6_text_embeddings.py` 작성 + 캐시 생성 (`.venv`)
- [x] `TextBBoxSteeredMLP` 구현
- [x] 10-seed PM 비교: `steered_reg`(bbox만) vs `text_bbox_steered`(bbox+text)
- [x] 텍스트 영향력 프로브 실행 + flip rate 기록
- [x] 결과 CH73에 반영(PM 주지표, closed-loop는 참고로 명시) + 로컬호스트 링크

## 5. 결과 (2026-10-13, V6 10-seed, F2 캐시 입력)

- steered_reg(bbox만) 78.33±1.28% vs text_steered_reg 76.92±0.80% — Δ−1.41%p, p=0.0003 (텍스트가 오히려 하락)
- 프로브(테스트셋 전체, 학습 종료 가중치): 정상 75.4% / 다른 방향 text 69.9%(−5.5%p, p=0.015) / 0벡터 72.4%, flip 13.2%±10.6%(seed별 1.8~24.6%) → 텍스트는 쓰이지만 불안정
- CLIP 방향 임베딩 코사인(weak_left↔weak_right 0.98)이 left/right를 거의 구분 못 함 — 유력 원인(미검증)
- 계획 대비 변경: CLIP-L/14 text 768-d(로컬 캐시, B/32 아님), 베이스라인을 F2 캐시 입력으로(빠른 paired 비교), 영어 템플릿 3종/방향, 프로브는 단일 샘플이 아닌 테스트셋 전체 교체로 `eval_subgoal_steering_text_v6.py`에 통합(`eval_text_influence_probe.py` 별도 파일은 만들지 않음)
- 산출: `docs/v5/subgoal_steering_eval/summary_text_steering_v6.json`
- 후속 대조 완료 (one-hot 5-d, `TEXT_MODE=onehot`): 78.07% vs bbox-only 78.36% (Δ−0.29%p, p=0.24), 교체 프로브 −2.2%p(p=0.0005)·flip 12.6%±3.3 → 방향 신호는 안정적으로 쓰이나 PM 이득 없음. CLIP은 여기에 하락·불안정을 추가. 결론: 이 구조에서 텍스트 서브골은 bbox+이미지 대비 추가 정보 없음. 산출 `summary_text_steering_v6_onehot.json`
- 미착수: text_proj 차원 스윕(결론 바뀔 가능성 낮아 보류)

### 5.1 정직한 체크포인트 선택 재측정 (2026-10-13)
inner-val(학습 에피소드 20%)로 epoch 선택, 테스트 미사용. 낙관 편향 0.7~1.2%p, 결론 유지: CLIP −1.18%p(p=0.005), one-hot −0.35%p(p=0.31), 곡선 종류 오라클 +1.64%p(p=0.003), 경로 +1.13%p(p=0.035). `summary_honest_f2_v6_*.json`, 스크립트 `eval_honest_selection_f2_v6.py`. 상세 CH75-8.

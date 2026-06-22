# Plan — hidden state 차원축소 + 학습 가능한 가중치로 재시도 (C안)

> 작성: 2026-06-22 · 상태: **승인됨 — 구현 진행**
> 동기: CH40/41에서 원시 2304차원 hidden state를 그대로 concat/대체했을 때 baseline을 못 넘었다(plan_20260622_hidden_state_action_head.md §1의 "C안"으로 미뤘던 경로). 사용자가 "가중치 차이를 나게 할 방법"을 물어 — 차원을 맞추고 학습 가능한 가중치로 비중을 조절하는 실험으로 확정.

---

## 0. 가설

CH40/41의 실패 원인 후보: bbox(32차원)+image(256차원)=288차원 입력에 2304차원 hidden state를 그대로 얹으면(add: 2592차원) 너무 고차원이라 작은 MLP(256→128→64)가 제대로 못 배운다(과적합/최적화 어려움). **hidden state를 학습 가능한 linear projection으로 32~128차원 정도로 줄이면**, 같은 정보를 MLP가 더 잘 활용할 수 있을 것이라는 가설.

## 1. 설계

`ActionMLP` 앞에 학습 가능한 projection을 추가:
```python
class ProjectedHiddenHead(nn.Module):
    def __init__(self, bbox_img_dim, hidden_dim=2304, proj_dim=64, mode="add"):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, proj_dim)
        d_in = (bbox_img_dim + proj_dim) if mode == "add" else proj_dim + PROJ_DIM_IMG
        self.head = ActionMLP(d_in=d_in)
    def forward(self, bbox_img, hidden_raw):
        h = self.proj(hidden_raw)   # 2304 -> proj_dim, 역전파로 학습
        x = torch.cat([bbox_img, h], dim=-1)
        return self.head(x)
```
`self.proj`가 학습되면서 "hidden state의 어떤 차원/조합을 얼마나 쓸지"가 사실상 가중치 역할을 한다 — 명시적 스칼라 가중치(`w1*bbox + w2*hidden`)보다 표현력이 높고 구현도 단순.

### ablation 변수
- proj_dim ∈ {32, 64, 128, 256}
- mode ∈ {add, replace}
- window = 8(CH41에서 hidden state 변형의 최선 윈도우)로 고정 — 조합 폭증 방지

총 4×2 = 8개 조합, 매번 baseline도 동일 비교 기준(CH40/41의 window=8 baseline 91.54%) 재사용.

## 2. 변경 파일

| 파일 | 작업 |
|---|---|
| `scripts/train_hidden_state_projected.py` (신규) | `train_hidden_state_action.py` 복제 + ProjectedHiddenHead로 교체 |
| `docs/v5/research_story.html` | CH43으로 결과 기록 |

## 3. 완료 기준
- [ ] 8개 조합 학습, PM 비교
- [ ] window=8 baseline(91.54%)을 넘는 조합이 있는지 확인
- [ ] CH43 문서화(긍정/부정 모두)

## 4. 위험도
- 새 데이터 수집 없음, 기존 150개 재사용. 8개 조합 각 ~2.5분, 총 20분 내.
- 그래도 baseline을 못 넘으면 — "head 구조를 더 손보는 것보다 그라운딩 품질 개선이 우선"이라는 CH41 결론이 다시 한번 확인되는 것으로 정리.

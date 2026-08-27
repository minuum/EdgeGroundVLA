# 계획 — cx를 강하게 반영하는 액션헤드 신규 실험

리서치(`research_20260826_cx_emphasis_head.md`): cx 브랜치를 "추가"하는 기존
접근(cxgeom, hybrid, bbox_scale↑)은 이미 다 해봤고 효과가 ±1p 이내였다. 이번엔
**cx와 vis를 곱셈적으로 결합**하거나 **cx의 시간적 변화**를 명시 입력하거나
**gradient 수준에서 cx를 강제**하는, 구조적으로 다른 3가지 헤드를 새로 만들어
`train_exp73_stage1v3_heads.py`와 동일 프로토콜(V6 고정 split, seed 0/1/2,
epoch 300, pg448 그라운더 캐시)로 비교한다.

## 접근 방식

`scripts/train_exp73_stage1v3_heads.py`에 아래 3개 헤드 클래스를 추가하고
`HEADS` 딕셔너리 및 `main()`의 분기에 등록한다. 기존 mlp/cxgeom/hybrid 결과와
같은 로그 파일(`exp73_stage1v3_trackA_heads.json`)에 append.

### 1) FiLM 헤드 — cx가 vis를 곱셈적으로 변조

```python
class FiLMHead(nn.Module):
    """cx(4d)로 vis(256d)에 FiLM(scale·shift) 변조를 가한 뒤 flatten+FC.
    지금까지의 concat 방식과 달리 cx가 "vis를 어떻게 해석할지"를 조절."""
    def __init__(self, frame_dim=FRAME_DIM, window=WINDOW, vis_dim=PROJ_DIM):
        super().__init__()
        self.film = nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, vis_dim * 2))
        self.trunk = nn.Sequential(
            nn.Linear(vis_dim * window, 512), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, NUM_CLASSES))

    def forward(self, x):
        bbox, vis = x[..., :4], x[..., 4:]              # (B,W,4), (B,W,256)
        gb = self.film(bbox)                              # (B,W,512)
        gamma, beta = gb.chunk(2, dim=-1)                 # (B,W,256) each
        modulated = vis * (1 + gamma) + beta
        return self.trunk(modulated.flatten(1))
```

### 2) Δcx 헤드 — cx 시간 변화율을 명시 채널로 추가

```python
class DeltaCxHead(nn.Module):
    """window 내 인접 프레임 cx 차분(Δcx, Δcy, Δarea)을 3개 추가 채널로 붙여
    "커지는 중/줄어드는 중"을 명시 입력. F↔FR 경계 판단(69-6①-b)을 겨냥."""
    def __init__(self, frame_dim=FRAME_DIM, window=WINDOW):
        super().__init__()
        in_dim = (frame_dim + 3) * window
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, NUM_CLASSES))

    def forward(self, x):
        bbox = x[..., :3]                                 # cx,cy,area (has_bbox 제외)
        delta = bbox[:, 1:] - bbox[:, :-1]                 # (B,W-1,3)
        delta = F.pad(delta, (0, 0, 1, 0))                 # 첫 프레임은 0으로 패딩 → (B,W,3)
        return self.net(torch.cat([x, delta], dim=-1).flatten(1))
```

### 3) cx 보조 손실 헤드 — R/FR/F 3-way 보조 분류로 gradient 강제

```python
class CxAuxHead(nn.Module):
    """메인 8-class 로짓 외에 cx 회귀값(마지막 프레임)을 보조 출력으로 함께 반환.
    학습 루프에서 main_CE + 0.3*aux_MSE(cx_pred, cx_true)로 gradient에 cx 활용을 강제."""
    def __init__(self, frame_dim=FRAME_DIM, window=WINDOW):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(frame_dim * window, 512), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(512, 128), nn.ReLU())
        self.cls_head = nn.Linear(128, NUM_CLASSES)
        self.cx_head = nn.Linear(128, 1)

    def forward(self, x):
        h = self.trunk(x.flatten(1))
        return self.cls_head(h), self.cx_head(h).squeeze(-1)
```
학습 함수 `train_cxaux()`를 `train_hybrid()` 패턴으로 새로 작성 —
`loss = CE(logit, y) + 0.3 * MSE(cx_pred, x[:, -1, 0])`.

## 수정 파일

- `scripts/train_exp73_stage1v3_heads.py` — 3개 헤드 클래스 + `train_cxaux()` +
  `HEADS`/`main()` 분기 추가 (기존 헤드/함수는 건드리지 않음)

## 실행 계획

```bash
.venv/bin/python3 scripts/train_exp73_stage1v3_heads.py \
  --heads film,deltacx,cxaux --arms v6 --seeds 0,1,2 --epochs 300 \
  --ann-v6 docs/v5/bbox_frame_level/bbox_dataset_v6_florence2_phrase_cx.json \
  --tag pg448
```
(정확한 `--ann-v6` 경로는 exp77 학습 때 쓴 phrase 그라운딩 주석 파일로 맞춘다 —
구현 단계에서 `exp77_florence2_phrase_full_stage2.json`의 `cache` 필드 확인 후 지정)

## 트레이드오프

- FiLM/Δcx/cxaux 셋 다 파라미터 증가 미미(수만 개), 학습 시간은 기존 mlp/cxgeom과
  거의 동일(GPU 캐시 재사용, epoch당 수 초 수준)
- 리서치에서 이미 "cx 브랜치 추가"류는 효과가 작다고 나왔으므로, 이번에도 mlp
  대비 유의미한 개선(>1~2p, seed 분산 밖)이 없으면 **"MLP 헤드 구조를 바꿔서
  cx 활용도를 올리는 건 한계에 부딪혔다"**로 결론짓고 다른 축(데이터 증강,
  앙상블)으로 넘어가는 게 합리적 — 이 계획 자체가 그 가설을 검증하는 실험

## 완료 기준

- [x] 헤드 3종 구현 + 문법 검증
- [x] pg448/v6 조건으로 seed 0/1/2 학습 완료 — 실행은 `train_exp73_stage1v3_heads.py`의
      CLI 대신 exp77 vis 캐시를 직접 재사용하는 `scripts/train_exp78_cx_emphasis_heads.py`로
      진행(헤드 클래스는 계획대로 `train_exp73_stage1v3_heads.py`에 구현, 캐시만 exp77
      것을 그대로 사용해 재인코딩 생략 — apples-to-apples 비교 목적에는 동일)
- [x] mlp/cxgeom 대비 비교표 작성 → `docs/v5/research_story.html#ch70` 신규 챕터 기록
      (결과: deltacx가 val_acc 최고 +0.67%p지만 R클래스 -5.4%p 하락 — 개선 단정 보류)
- [x] leave-one-direction-out으로 deltacx 방향별 일반화 재검증 — 개선 소멸 확인
      (LOO 평균 mlp 54.00% vs deltacx 53.83%, 70-3)
- [x] honest checkpoint selection(C), 학습곡선(과적합 진단), actionquery(경량
      cross-attn), ordinal soft label(D) 추가 실행 — D가 유일하게 LOO 평균 개선
      (+3.71~3.99%p, 70-4~70-7)
- [ ] v6v5, trackF 조건까지 확장 실행 — D+actionquery 결합 검증 이후 판단
- [ ] 궤적 재생 근사(rollout_core.py)로 D 후보 재평가, 이후 소규모 실기 A/B

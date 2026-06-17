# 논문 Table 초안 (2026-06-17 갱신)

> 평가 기준: Closed-Loop Offline Replay (FPE < 0.5m AND TLD ∈ [0.7, 1.5])  
> 벤치마크: V5 데이터셋 150ep, stratified split seed=42, val 29~30ep  
> 실로봇 결과: 2026-06-17 테스트 예정 → Table 5에 반영

---

## Table 1 — 주요 방법 비교 (Main Results)

| Method | Architecture | CL (↑) | FPE (↓) | TLD |
|--------|-------------|--------|---------|-----|
| E2E VLA (Exp11) | Kosmos-2 + LoRA, 8-class | 0.0% | 1.454 m | 1.026 |
| Decomposition v1 (Exp14) | CLIP + BBox MLP | 66.7% | 0.555 m | 1.034 |
| Simple MLP (Exp65b) | CLIP + plain MLP (no L2, no aug) | 10.3% | 0.941 m | — |
| **Ours (Exp66) ★** | **CLIP + L2-norm + bbox aug, MLP w=4** | **96.6%** | **0.094 m** | **1.003** |

> Exp11: text attention = 0% (Google-robot post-training으로 언어 경로 구조적 사망).  
> Simple MLP vs Ours: 동일 cx 소스(base PG2), 파이프라인만 다름 → ×9.4 성능 차이.

---

## Table 2 — Ablation: 파이프라인 × cx 소스

### 2-A. 파이프라인이 성능을 결정한다 (cx 소스 고정 = base PG2)

| Pipeline | cx Source | val_acc | CL (↑) | FPE (↓) |
|----------|-----------|---------|--------|---------|
| Simple MLP (Exp65b) | base PG2 | 90.2% | 10.3% | 0.941 m |
| **L2-norm + aug (Exp66)** | base PG2 | **93.5%** | **96.6%** | **0.094 m** |

파이프라인만 교체: 10.3% → 96.6% (×9.4배)

### 2-B. cx 소스는 성능에 영향을 주지 않는다 (파이프라인 고정 = L2+aug)

| cx Source | Grounding Quality | val_acc | CL (↑) | FPE (↓) |
|-----------|------------------|---------|--------|---------|
| HSV (Exp54) | hit 97%, std 0.159 | 92.6% | 96.6% | 0.110 m |
| base PG2 (Exp66) | hit 97%, std 0.056 | 93.5% | 96.6% | 0.094 m |
| Exp59 LoRA (Exp67) | hit 94%, full-frame 6% | 94.5% | 96.6% | 0.111 m |

cx 소스 변경 시 CL 성능 변화 없음 → grounding LoRA 개선이 action에 기여하지 않음.

---

## Table 2-C — Action Head Ablation

파이프라인 고정(L2-norm + bbox aug), cx 소스 고정(HSV), window=8:

| Exp | Head | Type | val_acc | CL (↑) | FPE (↓) |
|-----|------|------|---------|--------|---------|
| Exp68 | Linear | 1-layer FC | 76.8% | 69.0% | 0.377 m |
| Exp69 | FCHead | RoboVLMs FCDecoder | 95.3% | 93.1% | 0.109 m |
| Exp70 | LSTMHead | RoboVLMs MobileVLAClassificationDecoder | 95.7% | 96.6% | 0.112 m |
| **Exp54** | **ActionMLP (Ours)** | **3-layer MLP, window-flat** | **92.6%** | **96.6%** | **0.110 m** |

LSTM (RoboVLMs) = ActionMLP (ours): CL 96.6% 동일. Window-flattened MLP가 temporal LSTM과 등가, 더 경량.

---

## Table 2-D — Window Size Ablation

| Head | Window | val_acc | CL (↑) | FPE (↓) | 비고 |
|------|--------|---------|--------|---------|------|
| MLP | 2 | 94.88% | 93.1% | 0.145 m | 방향 파악 실패 |
| **MLP** | **4 ★** | **92.91%** | **96.6%** | **0.094 m** | MLP 최소 필요 window |
| MLP | 8 | 92.6% | 96.6% | 0.110 m | baseline |
| MLP | 16 | 89.57% | 96.6% | 0.102 m | val_acc 하락, CL 유지 |
| LSTM | 4 | 95.08% | 96.6% | 0.123 m | — |
| LSTM | 8 | 95.7% | 96.6% | 0.112 m | baseline |
| **LSTM** | **16 ★** | **96.85%** | **96.6%** | **0.080 m** | 최저 FPE |

MLP: w≥4에서 포화. LSTM: w=16에서 FPE 0.080 m 최저.

---

## Table 3 — Basket Localization Proof

### 3-A. Zero-shot Linear Probe

| Model | Features | Protocol | Accuracy |
|-------|----------|----------|---------|
| Frozen CLIP (Stage1 v2) | 256-dim L2-norm | 3-way (left/center/right), LOO | **96.6%** |
| Random baseline | — | — | 33.3% |

### 3-B. Basket Masking Ablation (Exp66, base PG2)

| Condition | 프레임 수 | 행동 반전 | 반전율 |
|-----------|---------|---------|------|
| Original → Basket masked | 9 | 9 | **100% (9/9)** |

두 증거: CLIP encoder가 basket을 독립적으로 인식하며, action이 이에 인과적으로 의존.

---

## Table 4 — Feature Ablation (Image vs BBox)

| Input | val_acc | 비고 |
|-------|---------|------|
| BBox only (cx, cy, area ×8) | 67.4% ± 9.8% | 방향 정보 부족 |
| Image only (CLIP 256-dim) | 75.6% ± 0.8% | 안정적 |
| **Image + BBox (Ours)** | **76.7% ± 1.3%** | BBox는 보조적 기여 |

---

## Table 5 — 실로봇 결과 [2026-06-17 테스트 예정]

| Model | 경로 유형 | 시도 | 성공 | CL% | FPE (m) |
|-------|---------|------|------|-----|---------|
| Exp66 (Stage2 v2) | left_curve | — | — | — | — |
| Exp66 (Stage2 v2) | right_curve | — | — | — | — |
| Exp66 (Stage2 v2) | center_straight | — | — | — | — |

> STOP 조건: area ≥ 0.50 AND |cx−0.5| ≤ 0.30, 2 consecutive frames  
> 결과 나오면: `mona-sync --add-exp` 로 experiment_history 추가

---

## 핵심 서술 포인트

1. **E2E VLA 실패**: text attention 0% (Google-robot backbone 기인, LoRA로 복구 불가)
2. **Decomposition 필요성**: E2E 0% → 96.6% CL
3. **Pipeline = 유일 결정 변수**: L2-norm + aug, ×9.4. cx 소스 무관 (음성 결과)
4. **RoboVLMs 비교**: LSTM = ActionMLP = 96.6%, window-flat MLP가 더 경량
5. **Basket localization 이중 증명**: zero-shot 96.6% + masking 9/9 flip

---

## 체크포인트 (재구성 후 기준)

| 실험 | 경로 |
|------|------|
| Stage1 v2 (공유) | `runs/v5_nav/mlp/shared/stage1_v2_projs.pt` |
| **Exp66 ★** | **`runs/v5_nav/mlp/exp66/action_mlp.pt`** |
| Exp54 (HSV baseline) | `runs/v5_nav/mlp/exp54/action_mlp.pt` |
| Exp65b (Simple MLP) | `runs/v5_nav/mlp/exp65/action_mlp.pt` |
| Exp67~70 | `runs/v5_nav/mlp/exp{67,68,69,70}/action_mlp.pt` |
| Window ablation | `runs/v5_nav/mlp/ablation_window/` |

# 논문 Table 초안 (2026-06-12)

> 평가 기준: Closed-Loop Offline Replay (FPE < 0.5m AND TLD ∈ [0.7, 1.5])
> 벤치마크: V5 데이터셋 150ep, stratified split seed=42, val 29~30ep

---

## Table 1 — 주요 방법 비교 (Main Results)

| Method | Architecture | CL (↑) | FPE (↓) | TLD |
|--------|-------------|--------|---------|-----|
| E2E VLA (Exp11) | Kosmos-2 + LoRA, 8-class | 0.0% | 1.454m | 1.026 |
| Decomposition Step1 (Exp14) | CLIP + BBox MLP | 66.7% | 0.555m | 1.034 |
| **Ours (Exp54/66)** | **CLIP + BBox MLP + L2-norm + aug** | **96.6%** | **0.10m** | **1.00** |

> Exp11은 text attention = 0% (Google-robot post-training으로 언어 경로 사망). decomposition이 필요한 이유.

---

## Table 2 — Ablation: 파이프라인 × cx 소스

### 2-A. 파이프라인이 성능을 결정한다 (cx 소스 고정 = base PG2)

| Pipeline | cx Source | val_acc | CL (↑) | FPE (↓) |
|----------|-----------|---------|--------|---------|
| Simple MLP (exp65b) | base PG2 | 90.2% | 10.3% | 0.941m |
| **L2-norm + aug (exp66)** | base PG2 | 93.5% | **96.6%** | 0.102m |

파이프라인만 바꿨을 때: 10.3% → 96.6% (×9.4배)

### 2-B. cx 소스는 성능에 영향을 주지 않는다 (파이프라인 고정 = L2+aug)

| cx Source | Grounding Quality | val_acc | CL (↑) | FPE (↓) |
|-----------|------------------|---------|--------|---------|
| HSV (exp54) | hit 97%, std 0.159 | 92.6% | 96.6% | 0.110m |
| base PG2 (exp66) | hit 97%, std 0.056 | 93.5% | 96.6% | 0.102m |
| exp59 LoRA (exp67) | hit 94%, full-frame 6% | 94.5% | 96.6% | 0.111m |

cx 소스를 바꿔도 CL 성능 변화 없음 → grounding LoRA 개선이 action에 기여하지 않음.

---

## Table 3 — Augmentation Ablation (참고)

| Experiment | Val acc | CL | 비고 |
|------------|---------|-----|------|
| exp49 (no aug) | — | 96.7% | 초기 HSV 기반 |
| exp54 Stage2 v2 (L2+aug, HSV) | 92.6% | 96.6% | 현재 최선 |
| exp53 (CLIP LoRA) | — | 96.6% | |
| exp55 (free ep 포함) | — | 96.7% | |
| exp50 (flip aug) | — | 83.3% | flip aug 오히려 하락 |

---

## 핵심 서술 포인트 (논문 본문용)

1. **E2E VLA 실패 원인**: Google-robot pretrained backbone의 언어 경로 구조적 사망 (text attention 0%, Exp15 head-only에서도 동일 확인). LoRA 학습과 무관한 모델 구조 기인.

2. **Decomposition의 필요성**: Stage1(vision encoder) + Stage2(BBox cx + image → action) 분리로 E2E 0% → 96.6% 달성.

3. **파이프라인 기여**: L2 정규화와 PG2 grounding 분포 모사 증강이 핵심. 단순 MLP 대비 ×9.4배. grounding 품질 자체(cx 소스)는 무관.

4. **Grounding 음성 결과**: LoRA grounding 개선(exp59, exp64 등)이 downstream action 성능에 기여하지 않음. 이는 현재 파이프라인에서 cx 신호가 포화 상태임을 시사.

---

## 체크포인트 경로

| 실험 | 경로 |
|------|------|
| exp54 Stage2 v2 | `runs/v5_nav/mlp/exp54/stage2_v2/stage2_v2_mlp.pt` |
| exp66 (base PG2) | `runs/v5_nav/mlp/exp54/stage2_v2/stage2_v2_mlp_base_pg2_aug.pt` |
| exp67 (exp59 LoRA) | `runs/v5_nav/mlp/exp54/stage2_v2/stage2_v2_mlp_exp59_aug.pt` |
| exp65b (단순 MLP) | `runs/v5_nav/mlp/exp65/` |

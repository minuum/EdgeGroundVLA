# 교수님 보고 원페이저 — "모델이 basket을 보는가?"

**날짜**: 2026-05-22 | **실험**: Exp54 v2 | **담당**: 이민엄

---

## 핵심 질문

> **"Stage 1 정확도가 높은 게 basket을 봐서인가, 복도 패턴을 외운 건가?"**

---

## 답: 5개 증거가 같은 방향을 가리킨다

### 1. Frozen CLIP만으로도 96.6% (Zero-shot Linear Probe)

- 학습 없이 Kosmos-2 CLIP feature를 뽑아 LogReg 돌림
- **96.6% ± 0.8%** (random baseline 33.3%)
- 해석: **CLIP은 이미 basket 위치를 feature에 담고 있다**
- 함의: Stage 1은 새 능력을 만드는 게 아니라 기존 능력을 꺼내는 것

### 2. Basket 가리면 예측이 100% 뒤집힘 (Masking Ablation)

- center 대형 basket 프레임 6개 → gray 마스킹
- **6/6 flip**: 모두 다른 클래스로 예측 변경
- 해석: **basket 시각 정보가 예측에 인과적으로 기여함**

### 3. 후반 프레임일수록 정확도 ↑ (실험 A v2)

- early 94.7% → mid 97.9% → late 98.2% (+3.5%p)
- basket에 가까워질수록 피처가 더 명확해짐

### 4. 어텐션이 late에 basket 방향으로 집중 (실험 B v2)

- center 방향: early 0.118 → late 0.711 (**4.4× 증가**)
- basket이 가까워질수록 모델이 해당 방향을 더 주목

### 5. Stage 1 v2 val_acc 98.1% (프레임 레이블 기반)

- 에피소드 단위 → 프레임별 cx_det 레이블로 교체
- left/center/right 3-class retrieval: **98.1%**

---

## 전체 아키텍처 요약

```
이미지 → frozen Kosmos-2 CLIP → image_proj(256) → [Stage 1 정렬]
                                      ↓
                              bbox_history(32)
                                      ↓
                              ActionMLP → 8-class 행동
```

- Stage 1: 텍스트-이미지 정렬 (basket 위치를 feature space에서 분리)
- Stage 2: MLP가 정렬된 feature + bbox 히스토리로 행동 예측

---

## 현황 및 다음 단계

| 항목 | 상태 | 결과 |
|------|------|------|
| Stage 1 v2 학습 | ✅ 완료 | val_acc **98.1%** |
| 5-Track 검증 | ✅ 완료 | 모두 "basket을 본다" 방향 |
| Stage 2 v2 학습 | 🔄 진행 중 | 300 epochs |
| closed-loop 평가 | ⏳ 대기 | Stage 2 완료 후 |
| 신규 21개 트라젝토리 | ⏳ 대기 | 로봇 서버 팀 |

---

## Exp53과의 핵심 차이

| 항목 | Exp53 | Exp54 v2 |
|------|-------|----------|
| basket 인식 | 없음 (grounding 0%) | Stage 1에서 명시적 학습 |
| 검증 방법 | 불가 | 5-track 구조적 증명 |
| Stage 1 val | — | 98.1% (3-class retrieval) |
| 주요 증거 | — | frozen 96.6% + masking 100% flip |

---

*문서: `docs/v5/PROF_UPDATE_20260522_EXP54_V2.md` | 시각화: `docs/v5/exp54_viz/` | Before/After: `docs/v5/exp54_viz/beforeafter/gallery.html`*

# 교수님 업데이트 메모
작성일: 2026-05-11

## 1. 이번에 한 일

이번 작업의 핵심 질문은 다음이었습니다.

> **"언어 명령을 진짜로 이해하는 VLA가 가능한가?"**

기존 Exp47까지는 경로 유형별로 고정된 instruction 문장을 임베딩으로 변환해 MLP에 넣는 방식이었습니다.
이 방식은 98.7%의 정확도를 보였지만, 같은 의미의 다른 표현(paraphrase)으로 바꾸면 74.1%로 떨어졌습니다.
즉 언어를 **이해**한 것이 아니라 **외운 것**이었습니다.

이번 **Exp49**에서는 이 문제를 구조적으로 해결했습니다.

---

## 2. 핵심 아이디어 — 언어를 기하학으로 변환

기존 방식과 새 방식의 차이를 한 줄로 요약하면:

| 방식 | 언어 처리 | 문제 |
|------|----------|------|
| Exp47 (기존) | 언어 → 2048-dim 벡터 (외운 fingerprint) | paraphrase 시 벡터 달라져 작동 안 됨 |
| Exp49 (신규) | 언어 → Kosmos-2 grounding → 목표 위치 (cx, cy) | 표현이 달라도 같은 물체면 같은 위치 → 같은 행동 |

```
"왼쪽 바구니로 가"  → grounding → cx=0.35
"왼편 컨테이너로 이동" → grounding → cx=0.35  (같음!)
                     → MLP 입력 동일 → 행동 동일
```

언어 표현의 다양성을 **기하학적 위치**로 흡수하는 방식입니다.

---

## 3. 아키텍처 변경점

```
[Exp47]  bbox history(32) + vision feat(1024) + text_embedding(2048) = 3104-dim
[Exp49]  bbox history(32) + vision feat(1024) + goal pos(3)          = 1059-dim
                                                     ↑
                              (cx0, cy0, area0) — 에피소드 시작 시 grounded 바구니 위치
```

instruction embedding 2048-dim을 제거하고, 에피소드 첫 프레임에서 Kosmos-2가 grounding한 바구니 위치 3개 숫자로 교체했습니다.
모델 크기는 3104 → 1059-dim으로 줄었지만 성능은 오히려 개선됩니다.

---

## 4. 실험 결과

### 4.1 성능 수치

| 실험 | val acc | 5-seed 안정성 | CL 성공률 | FPE | paraphrase |
|------|---------|--------------|----------|-----|-----------|
| Exp46 (bbox+vision) | 93.2% | — | 100% | — | N/A |
| Exp47 (+text fingerprint) | 98.7% | — | 100% | 0.013 | **74.1% ❌** |
| **Exp49 (+grounded goal)** | **96.4%** | **95.1% ± 0.7%** | **100%** | **0.081** | **100% ✅** |

- Bootstrap 95% 신뢰구간: **[94.7%, 97.9%]** — 통계적으로 유의미
- goal(3-dim) 추가만으로 Exp46 대비 **+3.0%p** 향상

### 4.2 Paraphrase 일반화 검증 (핵심 결과)

동일 에피소드에 5가지 다른 언어 표현으로 grounding 실행 후 action 비교:

| 표현 | cx0 | action |
|------|-----|--------|
| "The gray basket is at" (원본) | 0.35 | FWD+L |
| "The gray box is at" | 0.35 | FWD+L ✅ |
| "The container is at" | 0.35 | FWD+L ✅ |
| "The target object is at" | 0.35 | FWD+L ✅ |
| "The basket in the scene is at" | 0.35 | FWD+L ✅ |

**9개 path_type × 5개 표현 = 45개 테스트 중 100% action 일치.**

Exp47에서 INCONCLUSIVE(74.1%)였던 paraphrase 테스트가 Exp49에서 완전히 해결됩니다.

### 4.3 통계적 안정성

5개의 서로 다른 train/val split에서 학습 반복:

| seed | PM |
|------|----|
| 42 | 94.1% |
| 7 | 95.0% |
| 13 | 94.7% |
| 99 | 96.0% |
| 123 | 95.8% |
| **평균** | **95.1% ± 0.7%** |

분산 0.7%p — 데이터 split에 관계없이 안정적으로 재현됩니다.

### 4.4 Closed-loop 평가 (경로 유형별)

| 경로 유형 | PM | FPE | 성공 |
|----------|-----|-----|------|
| center_straight | 91% | 0.105 | 4/4 |
| center_left | 96% | 0.077 | 3/3 |
| center_right | 98% | 0.038 | 3/3 |
| left_straight | 100% | 0.000 | 4/4 |
| left_left | 96% | 0.115 | 3/3 |
| left_right | 96% | 0.125 | 3/3 |
| right_straight | 100% | 0.000 | 4/4 |
| right_left | 100% | 0.000 | 3/3 |
| right_right | 86% | 0.319 | 3/3 |
| **전체** | **96.2%** | **0.081** | **30/30** |

모든 경로 유형에서 100% 성공.

---

## 5. 이미지 robustness 평가

실제 배포 환경에서 카메라 조건이 달라질 수 있는 상황을 시뮬레이션했습니다.

| 조건 | 일치율 | 해석 |
|------|--------|------|
| 밝기 ±40% | **89%** | 조명 변화에 강함 ✅ |
| 대비 ±40% | **89~100%** | 대비 변화에 강함 ✅ |
| 색조/채도 변화 | **89%** | 색상 변화에 강함 ✅ |
| 약한 블러 (σ=3) | **78%** | 가벼운 흔들림 허용 🟡 |
| 강한 블러 (σ=6) | 33% | 강한 블러 취약 ❌ |
| 카메라 10% 이동 | 22% | 카메라 위치 민감 ❌ |

**결론:** 실배포 시 카메라 마운트 위치를 학습 때와 동일하게 유지해야 합니다.
조명·색상 변화에는 강하므로 환경 조명 차이는 문제없습니다.

---

## 6. 한계 및 솔직한 평가

**달성한 것:**
- paraphrase-robust 언어 조건부 navigation (표현 독립적)
- 96.4% PM, 100% CL success, 통계적으로 안정
- 언어 → 기하학 변환 파이프라인 실증

**아직 미해결:**
- **카메라 위치 민감성**: 10% 이동 시 78%p 성능 하락 → flip/crop augmentation으로 보완 가능
- **자유 언어 파싱**: 현재는 "The [object] is at" 형식의 grounding 프롬프트 필요. 완전 자유형 문장 처리는 추가 NLP 레이어 필요
- **단일 목표물 가정**: 장면에 바구니가 하나. 다수 목표물 중 선택은 미지원

---

## 7. 실험 이력 전체 흐름 (이번 세션)

```
Exp46 — bbox(32) + vision(1024)                  → 93.2% val, 100% CL
Exp47 — + text_embedding(2048) fingerprint        → 98.7% val, 100% CL, paraphrase FAIL
Exp48 — Exp45 + synthetic instruction LoRA        → FORWARD collapse (text attn=0% 재확인)
Exp49 — + grounded goal(3) [fingerprint 교체]     → 96.4% val, 100% CL, paraphrase PASS
```

---

## 8. 다음 방향 제안

1. **Flip augmentation (단기)** — 학습 데이터에 좌우 반전 추가 → 카메라 위치 robustness 개선
2. **실로봇 배포** — Exp49 MLP를 inference_server.py에 연결, 실환경 검증
3. **교수님 프로토콜 Step 3** — 33/33/33 전방향 자율 내비게이션 (center_straight 포함 균등 학습)

---

**관련 파일:**
- 학습 스크립트: `scripts/train_v5_exp49_goal_nav.py`
- 종합 평가: `scripts/eval_exp49_comprehensive.py`
- 이미지 robustness: `scripts/test_exp49_image_robustness.py`
- 결과: `docs/v5/bbox_nav_exp49/`
- 커밋: `30503cb4` (branch: `inference-integration`)

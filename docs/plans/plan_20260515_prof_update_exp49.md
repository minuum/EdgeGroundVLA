# Plan — 교수님 업데이트: Exp49 결과 보고

작성: 2026-05-15  
브랜치: `monavla-driving`

---

## 0. 한 줄 요약

Exp49(GoalNav)가 오프라인 CL 96.7%를 달성했으며, 이는 기존 최고(Exp14 Step2 66.7%)를 +30%p 초과한다.

---

## 1. 교수 프로토콜 현황

| 단계 | 조건 | 결과 |
|------|------|------|
| Step 1 | 곡선만 학습 → 직선도 됨 | ✅ Exp11 (PM 58.6%) |
| **Step 2** | **50/50 비율 → 동작** | **🔄 Exp49로 우회 해결** |
| Step 3 | 33/33/33 완전 자율 내비 | ⬜ |

> Step 2에서 Exp16(전체 150ep) collapse → Exp49(GoalNav) 접근으로 우회.  
> 성능은 Step 2 기준을 충족하나, "50/50 직접 학습" 방식이 아니라는 점을 솔직히 보고.

---

## 2. 핵심 결과 테이블

### 2.1 오프라인 평가

| 실험 | 접근 방식 | val acc | CL 성공 | FPE |
|------|----------|---------|---------|-----|
| Exp11 (end-to-end) | Google-robot LoRA | 58.6% PM | 0% | 1.45m |
| Exp14 Step2 | BBox+Image MLP | 75.9% PM | 66.7% | 0.55m |
| Exp25 (Pure HF) | balanced objective | 52.4% PM | 55.6% | 0.48m |
| **Exp49 (GoalNav)** | **BBox+Vis+goal_pos** | **96.4% PM** | **96.7%** | **0.08m** |

### 2.2 Exp49 상세

| 항목 | 값 |
|------|-----|
| 모델 | MLP (d_in=1059) |
| 입력 | bbox(8×4=32) + vision(1024) + goal_pos(3) |
| goal_pos | 에피소드 첫 프레임 grounding 결과 (cx0, cy0, area0) |
| val acc | 96.4% (bootstrap 95% CI: [94.7%, 97.9%]) |
| 오프라인 CL | 96.7% (30 에피소드, 9 path type 전부 포함) |
| FPE | 0.08m (이전 최고 0.55m 대비 86% 감소) |

### 2.3 Path Type별 CL 성공 (Exp49)

| Path Type | PM | FPE | 성공 |
|-----------|-----|-----|------|
| center_straight | 91.1% | 0.10m | ✅ |
| center_left | 96.3% | 0.08m | ✅ |
| center_right | 98.1% | 0.04m | ✅ |
| left_straight | 100% | 0.00m | ✅ |
| left_left | 96.4% | 0.11m | ✅ |
| left_right | 96.5% | 0.12m | ✅ |
| right_straight | 100% | 0.00m | ✅ |
| right_left | 100% | 0.00m | ✅ |
| right_right | 86.2% | 0.32m | ✅ |

---

## 3. Exp49 접근 방식 설명

### "왜 GoalNav인가?"

기존 실패 원인 분석:
- End-to-end (Exp11): text attention = 0% → 언어 경로 사망, FORWARD bias
- Balanced objective (Exp25): PM 52%, CL 55.6% — rollout에서 누적 오류
- Instruction MLP (Exp47): instruction 임베딩이 path_type과 완전 매칭될 때만 동작

Exp49 해법:
```
"go to the basket on the left" → Kosmos-2 grounding → cx0=0.3
                                                        ↓
                                               MLP(bbox, vision, cx0=0.3) → LEFT action
```
- 언어를 직접 처리하지 않고, grounding을 통해 **공간 좌표로 변환**
- 다른 표현도 같은 물체 → 같은 cx0 → 같은 action (paraphrase-robust)
- val 데이터의 cx0 분산 0.25 이내에서 100% 행동 일치 확인

### augmentation 실험 결과

| 실험 | 변경 | val acc | CL |
|------|------|---------|-----|
| Exp49 | baseline | 96.4% | 96.7% |
| Exp50 | flip aug (+2626 frames) | 92.0% | 83.3% |
| Exp51 | crop aug (+7878 frames) | 93.4% | 96.7% |
| Exp52 | lang-vis 2048 (joint forward) | 93.9% | 93.3% |

결론: augmentation 효과 없음. 원본 데이터 2626 frames만으로 충분.

---

## 4. 남은 과제

### 4.1 단기 (이번 미팅 전)
- [ ] 실로봇 Exp49 테스트 (계획: `plan_20260515_real_robot_eval_exp49.md`)
- [ ] 실로봇 성공률 측정 → 교수님께 보고할 실제 숫자 확보

### 4.2 Step 3 진입 조건
Step 3(33/33/33)을 시작하려면 교수님 결정이 필요한 항목:

| 질문 | 선택지 |
|------|--------|
| GoalNav 방식으로 Step 3 진행? | A: Exp49 확장 (start_pos 3종 grounding) |
| 아니면 진짜 end-to-end 재도전? | B: 새 backbone (TICVLA / MobilityVLA) |
| 실로봇 성공 ≥ 80% 확인 후 Step 3? | C: 실로봇 결과 기다림 |

### 4.3 GoalNav Step 3 설계 (A안 참고용)

```
현재 Exp49: path_type 9종 → goal_pos(cx0, cy0, area0) 고정 → MLP
Step 3 확장: 
  - start_pos 3종(left/center/right) × goal_pos → 더 일반화된 MLP
  - 또는: 실시간 grounding (매 프레임 bbox 재추출) → goal_pos 갱신
```

---

## 5. 발표 구성 제안 (10분 기준)

1. **(2분) 이전 상황 리마인드**: Exp11 0%, Step2 66.7%, Step3(end-to-end) 55.6%
2. **(3분) Exp49 소개**: GoalNav 아이디어 + 결과 테이블
3. **(2분) Path Type별 성공**: 9종 모두 ✅ (right_right FPE 0.32m 제외하고 전부 < 0.13m)
4. **(2분) 실로봇 결과** (미팅 전 확보 시 포함)
5. **(1분) 다음 단계 결정 요청**: Step 3 방향

---

## 6. 파일 위치

| 항목 | 경로 |
|------|------|
| Exp49 종합 eval | `docs/v5/bbox_nav_exp49/comprehensive_eval.json` |
| CL 전체 결과 | `docs/v5/closed_loop_eval/rollout_metrics.json` |
| CL 대시보드 | `docs/v5/closed_loop_eval/index.html` |
| 학습 스크립트 | `scripts/train_v5_exp49_goal_nav.py` |
| 체크포인트 | `runs/v5_nav/mlp/exp49/exp49_mlp.pt` |

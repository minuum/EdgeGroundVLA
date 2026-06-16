# EXP54 진행 상황 — 2026-05-19

## 핵심 요약

> **교수님 질문: "박스를 본 건가, 텍스트를 외운 건가?"**  
> → 이를 구조적으로 증명하기 위해 **2-Stage 학습**으로 전환

---

## 이전까지 문제 (Exp53 진단 결과)

| 항목 | 결과 |
|------|------|
| Kosmos-2 live grounding 탐지율 | **0%** (basket 못 찾음) |
| bbox_dataset 탐지 방법 | 99.1% 중 실제 entity match **17%**, 나머지는 쓰레기통/에어컨 bbox 대리 사용 |
| Exp53 학습 신호 | action 8-class 분류 — basket 어디 있는지 학습 안 됨 |
| val_acc 94.7% 의미 | "복도 패턴으로 action 맞추기", basket 인식 아님 |

---

## Exp54: 2-Stage 분리 구조

```
Stage 1: CLIP LoRA → 텍스트-이미지 정렬 학습
         "The gray basket is on the left"  ↔  left 이미지
         "The gray basket is in the center" ↔ center 이미지
         "The gray basket is on the right"  ↔  right 이미지

Stage 2: Stage 1 LoRA frozen → MLP action head 학습
         basket 인식이 된 CLIP 위에 navigation 붙이기
```

---

## 현재 진행 상황 (2026-05-19 기준)

### Stage 1 ✅ 완료

| 항목 | 결과 |
|------|------|
| 학습 시간 | 57.5분 (25 epoch) |
| val_acc (retrieval) | **100%** (목표 80%) |
| left 방향 | 9/9 ✅ |
| center 방향 | 9/9 ✅ |
| right 방향 | 9/9 ✅ |
| 저장 위치 | `runs/v5_nav/mlp/exp54/stage1/` |

**중립 앵커 ablation**: "An object is visible" (방향 없는 텍스트) → 3개가 동일 → random 33%  
→ 방향 텍스트가 핵심임을 확인

### Stage 2 🔄 학습 중

| 항목 | 상태 |
|------|------|
| 구조 | Stage 1 LoRA frozen + MLP(d_in=1056) |
| goal 벡터 | **제거** (fake bbox였으므로) |
| 학습 중 | 300 epoch, 현재 진행 중 |
| 비교 기준 | Exp53 94.7% / Exp49 96.4% |

---

## 교수님 미팅 이후 변경 예정

### 현재 한계

1. **레이블이 에피소드 단위** — `path_type`(left/center/right)으로 전체 에피소드를 one-label 처리
   - 초반 프레임은 basket이 멀리 있어 노이즈 있음
   - "복도 분위기 학습"과 "basket 위치 인식" 구분 불가

2. **Stage 1 val_acc 100%의 의미 불명확**
   - 같은 배포에서 나온 val 데이터 → over-optimistic일 수 있음
   - 완전히 새로운 환경/각도에서 테스트 필요

3. **30 트라젝토리 신규 데이터 미수집** (교수님 5/15 지시사항)
   - 조이스틱 비동기 수집: 좌/중/우 각 10개
   - 현재는 기존 150ep로 학습

### 미팅 후 할 일 (우선순위 순)

```
1. 신규 30 트라젝토리 수집 (조이스틱)
   → 데이터 다양성 확보 + 교수님 지시 이행

2. Stage 2 완료 후 closed-loop 평가
   → 실로봇에서 "go to the box" 명령 성공률

3. 방향어 있음/없음 비교 테이블 작성
   프롬프트 타입 | 방향 | 시도 | 성공
   방향어 포함   | 왼쪽 |  5  |  ?
   방향어 없음   | 왼쪽 |  5  |  ?
   ...

4. 프레임별 bbox 어노테이션 (선택)
   → path_type 레이블 노이즈 해결
   → 진짜 basket 위치 기반 Stage 1 재학습
```

---

## 실험 흐름

```
Exp49 (baseline MLP, 96.4%)
  ↓
Exp53 (CLIP LoRA end-to-end, 94.7%) — basket 인식 안 됨 진단
  ↓
Exp54 Stage 1 (CLIP contrastive, 100% retrieval) ✅
  ↓
Exp54 Stage 2 (action head, 학습 중) 🔄
  ↓
실로봇 평가 → 교수님 보고
```

---

## 관련 파일

| 파일 | 설명 |
|------|------|
| `scripts/train_exp54_stage1_contrastive.py` | Stage 1 학습 |
| `scripts/test_exp54_stage1_retrieval.py` | Stage 1 검증 |
| `scripts/train_exp54_stage2_action.py` | Stage 2 학습 |
| `runs/v5_nav/mlp/exp54/stage1/` | Stage 1 저장 |
| `logs/exp54_stage1.log` | Stage 1 학습 로그 |
| `logs/exp54_stage2.log` | Stage 2 학습 로그 (진행 중) |

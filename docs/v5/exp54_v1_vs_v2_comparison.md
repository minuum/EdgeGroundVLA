# Exp54 v1 vs v2 비교 정리

작성일: 2026-05-22

---

## 개요

Exp54는 basket 인식 강제 학습(Stage 1 contrastive)을 통해  
"박스를 본 건가, 복도를 외운 건가?" 를 구조적으로 답하기 위한 2-stage 접근입니다.

v1 → v2의 핵심 변화는 **Stage 1 레이블 품질** 개선입니다.

---

## Stage 1 비교

| 항목 | v1 | v2 |
|------|----|----|
| 레이블 방식 | 에피소드 경로(`path_type`) | 프레임별 `cx_det` (실제 basket 위치) |
| 레이블 생성 | 에피소드 전체에 동일 레이블 | cx_det < 0.40 → left, 0.40~0.60 → center, > 0.60 → right |
| center 데이터 | 68 프레임 (path_type=center 에피소드만) | 775 프레임 (Kosmos-2 bbox cx 기반) |
| 사용 프레임 | consistent=True 전체 | consistent=True + cx_det 기반 재분류 |
| val_acc | ~60% 추정 (center 거의 0%) | **98.1%** (left 99.4%, center 96.7%, right 100.0%) |
| 인코더 출력 | LoRA-CLIP (1024-dim) | frozen CLIP + image_proj (256-dim, L2-norm) |
| 저장 경로 | `stage1/clip_lora_adapter/` | `stage1_v2/stage1_v2_projs.pt` |

### 왜 v2가 98.1%인가?

v1은 에피소드 초반에 basket이 멀리 있어도 레이블 노이즈가 심했습니다.  
v2는 Kosmos-2 bbox 검출 결과(`cx_det`)를 직접 사용해 프레임 단위로 정확한 위치 정보를 부여.  
→ left/right는 HSV 기반 검출, center는 Kosmos-2 bbox 중심 x 좌표 사용 (하이브리드 방식)

---

## Stage 2 비교 (Stage 2 v2 학습 완료 후 채워넣기)

| 항목 | v1 | v2 |
|------|----|----|
| 인코더 | CLIP LoRA (1024-dim) | frozen CLIP + image_proj (256-dim) |
| D_IN | 1056 (bbox32 + vis1024) | 288 (bbox32 + proj256) |
| MLP 구조 | 1056→512→256→128→64→8 | 288→256→128→64→8 |
| 학습 대상 | MLP만 (LoRA frozen) | MLP만 (FrozenCLIPV2 frozen) |
| val_acc | TBD | **TBD (학습 중)** |
| 참고: Exp49 | — | 96.4% |
| 참고: Exp53 | — | 94.7% |

---

## 5-Track 추가 검증 (v2 기준)

| Track | 결과 | 해석 |
|-------|------|------|
| 실험 A v2 | early 94.7% → late 98.2% (+3.5%p) | 후반 프레임에서 basket 가까워질수록 정확도 ↑ |
| 실험 B v2 | center 어텐션 early 0.118 → late 0.711 (4.4×) | basket 근접 시 attention 집중 |
| Track 1 | keyword hit ~0% (trash can/AC로 인식) | Exp53 grounding 실패 원인 확인 |
| Track 2 | **frozen CLIP linear probe 96.6%** | 학습 전부터 이미 basket 위치 인코딩 |
| Track 3 | center 대형 basket 마스킹 → **6/6 flip** | 인과적 증거: basket을 실제로 보고 있음 |

---

## 핵심 논리 흐름

```
frozen CLIP이 basket 위치 안다 (96.6%)
    → Stage 1은 이 능력을 꺼내는 것 (정렬)
    → Stage 1이 꺼내면 Stage 2가 활용
    → basket 가리면 예측 뒤집힘 (100% flip)
    → "basket을 본다" ✅
```

---

## 파일 경로

| 항목 | 경로 |
|------|------|
| Stage 1 v1 LoRA | `runs/v5_nav/mlp/exp54/stage1/clip_lora_adapter/` |
| Stage 1 v2 체크포인트 | `runs/v5_nav/mlp/exp54/stage1_v2/stage1_v2_projs.pt` |
| Stage 2 v1 체크포인트 | `runs/v5_nav/mlp/exp54/stage2/stage2_mlp.pt` |
| Stage 2 v2 체크포인트 | `runs/v5_nav/mlp/exp54/stage2_v2/stage2_v2_mlp.pt` (학습 중) |
| v1 평가 스크립트 | `scripts/eval_exp54_stage2.py` |
| v2 평가 스크립트 | `scripts/eval_exp54_stage2_v2.py` |

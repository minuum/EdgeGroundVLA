# Plan: image_proj(Stage 1) 5-class 재학습 + 좌측 실기 재검증

**날짜:** 2026-08-06
**목적:** 배포 중인 image_proj(`stage1_v2_projs.pt`)는 2026-05-21에 150ep·HSV 라벨로 학습된
구형 체크포인트다. 현재 액션 헤드(exp73 cadence-aligned, 2026-07-23)는 225ep로 학습됐는데
image_proj는 재학습된 적이 없어 세대가 다르다. 225ep 데이터셋(V6)의 네이티브 5-class 구조
(강좌/약좌/중앙/약우/강우, 각 45ep)를 그대로 살려 image_proj를 재학습하고, 실기 재검증은
좌측(약좌·강좌)만 수행한다(교수님 지시 — 중앙/약우/강우는 재검증 없이 기존 실기 결과 재사용).

---

## 배경 및 근거

| | 배포 중(150ep) | 재학습 목표(225ep) |
|---|---|---|
| 체크포인트 | `runs/v5_nav/mlp/shared/stage1_v2_projs.pt` (2026-05-21) | 신규, 이름 미정 (아래 Step 4) |
| 데이터 | `bbox_dataset_frame_level.json`, 150ep/2,431프레임 | `bbox_nav_owl/bbox_dataset_v6_owl.json`, 225ep/16,599프레임 (전부 consistent) |
| 라벨 소스 | HSV+connected component 휴리스틱(고전 CV), center만 예외로 옛 Kosmos-2 cx 재사용 | **OWL-v2**(서빙 검출기와 일치) — PaliGemma2 버전(`bbox_dataset_v6_pg448_cx.json`)과 `label` 필드 100% 동일 확인됨(cx_det만 1.7% 미세 차이), 학습 스크립트가 `label`만 쓰므로 어느 쪽을 써도 무방하지만 서빙 일치성 위해 OWL-v2 버전 채택 |
| 클래스 | 3-class (left/center/right) | **5-class 유지** (strong_left/weak_left/center/weak_right/strong_right) — 데이터셋 자체가 이 구조로 수집됐으므로 합치지 않음 |
| 클래스 분포 | left=319·center=775·right=750 (좌 편중 불균형) | strong_left 3611·weak_left 3299·center 2819·weak_right 3268·strong_right 3602 (훨씬 균형) |

**실기 재검증 범위 축소 (교수님 지시)**: 5개 위치를 전부 재검증하면 시간이 너무 오래 걸리므로,
**약좌·강좌 2곳만** soda에서 재검증하고 중앙/약우/강우는 기존 실기 결과를 그대로 재사용한다.
최종 표는 "약좌·강좌=신규, 중앙·약우·강우=기존"을 합쳐 **하나의 모델의 공식 결과**로 보고한다.

---

## Step 1: 학습 스크립트 작성

`scripts/train_exp54_stage1_v2_frame_level.py`를 복제해
`scripts/train_stage1_v3_5cls_owl.py`로 저장. 원본은 보존(비교/롤백용).

변경점 **3곳만**:

```python
# 1. 데이터 경로
DATA_PATH = ROOT / "docs" / "v5" / "bbox_nav_owl" / "bbox_dataset_v6_owl.json"
OUT_DIR   = ROOT / "runs" / "v5_nav" / "mlp" / "stage1_v3_5cls"

# 2. 5-class 라벨 매핑 (기존 3-class에서 확장)
DIR_IDX = {
    "strong_left": 0, "weak_left": 1, "center": 2,
    "weak_right": 3, "strong_right": 4,
}
ANCHOR_TEXTS = {
    "strong_left":  "The gray basket is strongly on the left side of the image",
    "weak_left":    "The gray basket is slightly on the left side of the image",
    "center":       "The gray basket is in the center of the image",
    "weak_right":   "The gray basket is slightly on the right side of the image",
    "strong_right": "The gray basket is strongly on the right side of the image",
}

# 3. compute_text_anchors()의 anchors 개수: 3 → 5, cat 결과 shape (5, 2048)
```

`load_frame_level_data()`, `encode_images()`, 학습 루프(AdamW, CosineAnnealingLR, 30 epoch,
batch 16, lr 3e-4)는 원본 그대로 — 클래스 수만 3→5로 바뀌어도 구조 변경 불필요
(`nn.Linear(LM_DIM, PROJ_DIM)`, 코사인 유사도 기반 CE 그대로 5-class로 확장됨).

class weight도 원본처럼 빈도 역수로 자동 계산(코드 그대로 유지, 5-class 분포에 맞춰 자동 재계산됨).

**실행:**
```bash
.venv/bin/python3 scripts/train_stage1_v3_5cls_owl.py
# 예상 소요: ~1시간 (150ep/2,431프레임 기준 55.3분이었으므로 225ep/16,599프레임은
# 프레임 수 6.8배 증가 — batch 16 기준 시간도 비례 증가 예상, 2~4시간 가능성 있음.
# 실측 후 아래 "리스크"에 갱신)
```

---

## Step 2: val 평가 및 혼동행렬 확인

학습 스크립트 내 `evaluate()` 함수가 이미 5-class 혼동행렬을 출력하도록 구조는 동일
(3-class용 로직을 5-class로 그대로 재사용, 라벨 수만 늘어남). 학습 완료 후 로그
(`logs/train_stage1_v3_5cls_owl.log`)에서 다음을 확인:

- 전체 val retrieval accuracy (150ep 기준 98.11%와 비교)
- **strong_left ↔ weak_left 혼동** — 실기에서 좌측이 약했던 원인이 두 서브클래스 간
  혼동인지, 좌/우 자체의 혼동인지 구분 가능해짐
- weak_right/strong_right 정확도도 참고용으로 기록(재검증은 안 하지만 offline 지표는 남김)

---

## Step 3: Stage 2(행동 헤드) 재학습 필요 여부 판단

image_proj의 출력 차원(256)과 정규화(L2 norm)는 변경되지 않으므로 **Stage 2 헤드 구조는
그대로**. 다만 image_proj 가중치가 바뀌면 256차원 임베딩의 분포가 달라질 수 있어, 기존
`exp73_..._holdaware_seed0.pt`(cadence-aligned 헤드)를 그대로 재사용할지, 새 image_proj
위에서 재학습할지 결정 필요.

- 옵션 A: 헤드는 그대로 두고 새 image_proj만 교체해서 val_acc 변화 확인 (빠름, 위험— 분포
  shift로 성능 저하 가능)
- 옵션 B: 새 image_proj 위에서 헤드도 재학습 (안전, 시간 추가)

→ **결정: 시간 차이가 크지 않으므로 A/B 둘 다 진행.** 옵션 A(헤드 그대로) val_acc를 먼저
확인해 참고치로 남기고, 옵션 B(헤드도 재학습)까지 완료해 두 버전 모두의 val_acc/PM을 비교
기록한다. 최종 배포 후보는 두 결과 중 더 좋은 쪽으로 선택.

---

## Step 4: 체크포인트 명명 및 배포 경로

```
runs/v5_nav/mlp/stage1_v3_5cls/stage1_v3_5cls_owl_projs.pt   ← 신규
runs/v5_nav/mlp/shared/stage1_v2_projs.pt                     ← 기존, 그대로 유지(롤백용)
```

`scripts/run/go.sh`의 `S1_PT` 및 `stage2_v2_inference_server.py`의 `DEFAULT_STAGE1`은
**이 단계에서는 변경하지 않음** — soda 실기 재검증(약좌·강좌)이 통과한 뒤에만 배포 경로 전환.

---

## Step 5: soda 실기 재검증 (좌측만)

soda에 신규 체크포인트(`stage1_v3_5cls_owl_projs.pt` + Step 3 결과 헤드) 전달 후:

- **약좌·강좌 위치 2곳만** 재검증 실행 (기존 실기 테스트와 동일 프로토콜)
- 중앙·약우·강우는 **재검증하지 않고 기존 결과값 재사용**
- 결과 통합 표: 약좌·강좌=신규, 중앙·약우·강우=기존 → "하나의 모델"로 공식 보고

---

## ⚠️ 발견된 버그 — BGR→RGB 반전 누락 (수정 완료)

V6 raw H5(`f["images"]`)는 BGR로 저장되어 있음(`gen_v6_owl_annotation.py`,
`train_exp73_trackA_heads.py`의 "colorfixed 규격"에서 확인 — 실제 배포 모델(exp73)도 이
반전을 적용해 학습됨). `train_stage1_v3_5cls_owl.py`의 최초 버전은 이 반전이 빠져 있어
1차 학습 실행분(중단함)이 잘못된 색으로 학습되고 있었음. `load_image()`에
`[:, :, ::-1]` 추가 후 재시작.

## 완료 표시

- [x] Step 1: 학습 스크립트 작성 (`scripts/train_stage1_v3_5cls_owl.py`, 원본 보존)
- [x] Step 1.5: BGR→RGB 버그 수정 후 재시작 (위 참조)
- [ ] Step 2: 학습 실행 + val 평가
- [ ] Step 3: Stage 2 A/B 둘 다 실행(헤드 유지 vs 재학습), val_acc/PM 비교
- [ ] Step 4: 체크포인트 명명/보존
- [ ] Step 5: soda 실기 재검증(좌측만)

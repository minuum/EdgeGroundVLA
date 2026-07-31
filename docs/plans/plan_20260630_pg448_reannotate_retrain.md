# Plan: PG448 재-어노테이션 + Stage2 MLP 재학습 (Exp67)

**날짜:** 2026-06-30  
**목적:** 현재 Stage2 MLP(exp65/66)는 PG2-**224**로 생성된 어노테이션으로 학습됨. PG2-**448**로 재-그라운딩하여 더 깨끗한 bbox 어노테이션 생성 후 MLP 재학습.

---

## 배경 및 근거

| 항목 | 현재 (exp65/66) | 목표 (exp67) |
|------|-----------------|--------------|
| 어노테이션 그라운더 | PG2-224 | PG2-448 |
| V5 전체 검출률 | 95.9% (2519/2626) | ~99% 예상 |
| cx≈0.5 fallback 의심 | 11.8% (296프레임) | 감소 예상 |
| val_acc (MLP) | 96.8% (exp65) | ? |

셀프라벨링 185프레임 결과:
- PG224 방향 정확도 98.7%, 검출률 73.5%
- PG448 방향 정확도 99.1%, 검출률 98.9%
→ 학습 어노테이션의 cx 품질이 더 높아지면 MLP도 더 정확한 bbox 신호를 학습

---

## Step 1: gen_pg448_annotation.py 작성

`scripts/gen_base_pg2_annotation.py`를 복사 후 수정.

변경점 **2곳만**:

```python
# 1. 모델 경로
PG2 = Path.home() / ".cache/huggingface/hub" \
      / "models--google--paligemma2-3b-mix-448" \
      / "snapshots/1406c92ec87d32cc6b983239278901b904ba7a51"

# 2. 출력 경로
OUT = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_pg448_cx.json"
```

이미지 리사이즈는 PaliGemmaProcessor가 자동으로 448×448 처리 → 코드 변경 불필요.

**실행:**
```bash
.venv/bin/python3 scripts/gen_pg448_annotation.py
# 예상 소요: ~30분 (2626프레임 × ~0.7초)
```

---

## Step 2: train_exp67_stage2_pg448.py 작성

`scripts/train_exp65_stage2_basepg2.py`를 복사 후 수정.

변경점 **3곳만**:

```python
# 1. 어노테이션 경로
ANN_PG2 = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_pg448_cx.json"

# 2. 출력 디렉토리
OUT_DIR = ROOT / "runs/v5_nav/mlp/exp67"

# 3. 체크포인트 소스 기록 (저장 시)
torch.save({..., "source": "pg448", "exp": "exp67"}, ...)
```

나머지 아키텍처(d_in=288, WINDOW=8, ActionMLP 구조) 동일 유지.

**실행:**
```bash
.venv/bin/python3 scripts/train_exp67_stage2_pg448.py
# 예상 소요: ~10분
```

---

## Step 3: 셀프라벨링 185프레임으로 검증

```bash
# 기존 평가 스크립트에 exp67 체크포인트 연결
.venv/bin/python3 scripts/test_v5_pm_dm.py --model exp67
```

또는 서버의 `DEFAULT_STAGE2` 경로를 exp67로 바꿔서 실제 추론 테스트.

**비교 기준:**
- PM (Direction Match): exp65 vs exp67
- 셀프라벨 185프레임 L/R 방향 일치율 (ground truth 있음)

---

## 파일 목록

| 파일 | 상태 | 비고 |
|------|------|------|
| `scripts/gen_pg448_annotation.py` | 신규 작성 | gen_base_pg2_annotation.py 기반 |
| `docs/v5/bbox_frame_level/bbox_dataset_pg448_cx.json` | 생성됨 | Step1 결과물 |
| `scripts/train_exp67_stage2_pg448.py` | 신규 작성 | train_exp65 기반 |
| `runs/v5_nav/mlp/exp67/action_mlp.pt` | 생성됨 | Step2 결과물 |

---

## 트레이드오프

- PG448은 PG224보다 검출률 높지만, 학습/추론 분포 일치가 핵심 — **추론 서버도 PG448 쓰도록 이미 변경됨** (6/30 커밋)
- 185 셀프라벨 프레임이 validation set으로 쓰기엔 V5 train에 포함된 프레임과 겹칠 수 있음 → 세션 프레임(39개)만 독립 validation으로 사용하는 게 더 엄밀
- exp65 val_acc 96.8% 이미 높음 — 향상폭은 크지 않을 수 있으나 **학습-추론 분포 일치** 자체가 목적

---

## 승인 후 구현 순서

1. `gen_pg448_annotation.py` 작성
2. 실행 → json 생성 확인
3. `train_exp67_stage2_pg448.py` 작성
4. 실행 → val_acc 확인
5. 서버 DEFAULT_STAGE2 경로 exp67로 교체 → 실주행 테스트

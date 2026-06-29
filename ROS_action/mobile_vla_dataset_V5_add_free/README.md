# mobile_vla_dataset_V5_add_free

V5 기본 데이터셋(`mobile_vla_dataset_v5`)의 균형 보정 + free 에피소드 추가 버전.

생성 스크립트: `scripts/build_dataset_v5_add_free.py` (seed=42)

---

## 구성 요약

| 항목 | 수 |
|------|-----|
| 총 에피소드 | **221** |
| structured (서브샘플) | 200 |
| free (STOP 제거) | 21 |
| 총 프레임 (추정) | ~3,700 |

---

## Structured 에피소드 분포

| 경로 타입 | 원본 | 이 데이터셋 | 비고 |
|-----------|------|------------|------|
| right_straight | 55 | **25** | 서브샘플 |
| right_left | 53 | **25** | 서브샘플 |
| right_right | 35 | **25** | 서브샘플 |
| left_left | 26 | 26 | 전량 |
| left_straight | 25 | 25 | 전량 |
| left_right | 23 | 23 | 전량 |
| center_straight | 21 | 21 | 전량 |
| center_right | 15 | 15 | 전량 |
| center_left | 15 | 15 | 전량 |

right_* 3종을 25개로 상한 적용해 균형 맞춤.

---

## Free 에피소드 (21개)

5/21~5/22 수집. 바구니 위치·조명·거리 다변화 시나리오.

| 변형 | left | right | center |
|------|------|-------|--------|
| basket_left_extreme | 1 | 1 | 1 |
| basket_right_extreme | 1 | 1 | 1 |
| robot_close | 1 | 1 | 1 |
| robot_far | 1 | 1 | 1 |
| diagonal_left | 1 | 1 | 1 |
| diagonal_right | 1 | 2 | 1 |
| lighting_diff | 1 | 1 | 0 |

**STOP 프레임 처리:** 에피소드 중간의 정지 프레임(x=0,y=0,z=0) 84개 제거.
마지막 프레임 STOP은 유지(navigation 완료 신호).

---

## 액션 분포 (예상)

```
FORWARD   ~65%   (원본 75.8% → 개선)
FWD+L     ~12%
FWD+R     ~9%
LEFT      ~4%
RIGHT     ~4%
ROT_L     ~2%    (right/left_straight 드리프트 보정 + free diagonal)
ROT_R     ~1%
STOP      ~0%    (합성 라벨로 추가)
```

---

## 비서브샘플 버전 (미생성)

`mobile_vla_dataset_v5`(원본 291개) + free 에피소드 전량을 쓰되
**서브샘플 없이** 사용하는 버전.

- 에피소드 수: ~289개 (free STOP 제거 후)
- 특징: right_* 편중 유지 (55+53+35 = 143개, 53%)
- 권장 사용처: Exp60 flip augmentation 전용
  - flip aug가 right_straight(55) → left_straight(55 mirrored) 를 자동 생성하므로 불균형이 학습 시 해소됨
  - `scripts/train_exp60_flip_aug.py`와 조합 시 서브샘플 불필요

서브샘플 없는 버전이 필요하면 `build_dataset_v5_add_free.py`의
`SUBSAMPLE_CAPS` 딕셔너리를 비워서 재실행:
```python
SUBSAMPLE_CAPS = {}  # 서브샘플 없음 → 전량 유지
```

---

## 학습 연결

```bash
# nav_h5_dataset_impl.py 사용 시 data_dir 경로만 변경
python3 robovlm_nav/train.py configs/xxx.json \
    --data_dir ROS_action/mobile_vla_dataset_V5_add_free
```

Exp60 (`train_exp60_flip_aug.py`) 과 함께 사용 시
PG2 cx 재주석 파일(`bbox_dataset_pg2_cx.json`)에 free 에피소드 항목 추가 필요.

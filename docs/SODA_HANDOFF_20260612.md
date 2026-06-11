# SODA → Minum 인수인계 (2026-06-12)

> **작성:** SODA 서버 (100.85.118.58) — 로봇 데이터 수집 담당  
> **수신:** Minum 서버 — 학습/평가 담당

---

## 1. V5-2 데이터셋 현황

**위치:** `/home/minum/26CS/MoNaVLA/ROS_action/mobile_vla_dataset_v5_2/`  
**전송일:** 2026-06-12 rsync 완료 (613MB, 60파일)

### 수집 분포

| ID | 시나리오 | 수집/목표 | 상태 |
|----|----------|----------|------|
| 1 | 좌측 - 왼쪽 곡선 | 0/15 | ❌ 미수집 |
| 2 | 좌측 - 직선 | 0/20 | ❌ 미수집 |
| 3 | 좌측 - 오른쪽 곡선 | 0/15 | ❌ 미수집 |
| 4 | 중앙 - 왼쪽 곡선 | 10/15 | 🟡 66% |
| 5 | 중앙 - 직선 | 10/20 | 🟡 50% |
| 6 | 중앙 - 오른쪽 곡선 | 10/15 | 🟡 66% |
| 7 | 우측 - 왼쪽 곡선 | 1/15 | 🔴 6% |
| 8 | 우측 - 직선 | 0/20 | ❌ 미수집 |
| 9 | 우측 - 오른쪽 곡선 | 0/15 | ❌ 미수집 |
| FL | 자유-좌측 | 10/7 | ✅ 초과 |
| FC | 자유-중앙 | 9/7 | ✅ 초과 |
| FR | 자유-우측 | 9/7 | ✅ 초과 |

**합계: 59/171 에피소드 (34.5%), 3618 프레임**

### 액션 분포
```
FORWARD   51.4%  ██████████
FWD+R     17.6%  ███
FWD+L     16.6%  ███
STOP       9.8%  █
RIGHT      2.3%
LEFT       1.4%
ROT_L/R    1.0%
```

---

## 2. 자유 수집 데이터 주의사항

- `free_left/center/right` 에피소드 28개는 파일명에 `chair_left_extreme` 태그가 붙어 있음
- 의자가 항상 좌측에 위치한 편향된 조건에서 수집됨
- **학습 시 별도 분리 or 제외 권장**
- 파일명 패턴: `*free_{left,center,right}__chair_left_extreme*`

---

## 3. 즉시 학습 가능 여부 판단

| 조건 | 현황 |
|------|------|
| 좌측 데이터 | 0개 — 학습 시 심각한 편향 |
| 우측 데이터 | 1개 — 사실상 없음 |
| 중앙 데이터 | 30개 — 사용 가능 |

**권장:**
- 중앙 데이터 30개만으로 **pilot 학습** 가능 (overfitting 위험 있음)
- 좌측/우측 수집 완료 후 **full 학습** 권장
- 자유 수집 제외 기준: `free_` prefix 파일 필터링

---

## 4. 데이터 포맷 (V5-2)

```python
with h5py.File(ep, 'r') as f:
    images  = f['observations']['images'][:]  # (N, H, W, 3) uint8
    actions = f['actions'][:]                  # (N, 3) float32 — [linear_x, linear_y, angular_z]
```

- **이미지 키:** `f['observations']['images']` (V4는 `f['images']` — 다름 주의)
- **액션:** `(linear_x, linear_y, angular_z)` — 8-class 분류 기준 동일
- **STOP 프레임:** 각 에피소드 끝에 STOP 액션 3프레임 자동 주입됨
- **저장 포맷:** JPEG (q=90) 압축 vlen — 기존 raw gzip과 혼재 가능, 로더 자동 인식

---

## 5. 학습 설정 참고

- **기준 config:** `configs/mobile_vla_v5_exp11_google_robot_8cls.json`
- **현재 best E2E:** Exp11 (PM 58.6%, closed-loop 0%) — text attention 0% 확인됨
- **현재 best decomp:** Exp14 Step2 MLP (PM 75.9%, closed-loop 66.7%)
- `generate()` 절대 호출 금지 — Google-robot backbone에서 무한루프

---

## 6. SODA 수집 계속 진행 중

다음 수집 예정 (우선순위순):
1. 좌측 3종 (1/2/3) — 0개, 시급
2. 우측 직선+우곡선 (8/9) — 0개
3. 우측 좌곡선 (7) — 14개 추가 필요
4. 중앙 3종 (4/5/6) — 각 5~10개 추가

수집 완료 후 다시 rsync 예정.

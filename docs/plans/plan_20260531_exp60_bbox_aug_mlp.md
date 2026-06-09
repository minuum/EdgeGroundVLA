# Plan — Exp60: Stage2 MLP를 VLM bbox 분포에 강건하게 재학습

> 작성: 2026-05-31 · 상태: **✅ 구현·검증 완료** (하이브리드 접근) · 결과: [exp60_report.md](../v5/exp60_report.md)
> 결론: CL 4.5% → **36.4% (champion aug2.0)**, FPE 4.075m → 0.575m
> 선행 진단: Exp59 CL 0~4.5% — grounding은 98% 성공하나 MLP가 drift

---

## 1. 문제 정의 (확정된 진단)

Exp59가 파이프라인을 연결하면서 병목이 명확해졌다:

```
PaliGemma2 grounding 성공률  : 98%   ✅  (basket을 거의 안 놓침)
Closed-Loop 성공률           : 0~4.5% ❌  (drift)
참고: Exp54 CL (HSV bbox)    : 96.7%      (같은 MLP, bbox만 다름)
```

**같은 MLP인데 bbox 소스만 바뀌니 96.7% → 4.5%로 붕괴.** 원인은 분포 불일치:

| | 학습 시 bbox | 추론 시 bbox |
|---|---|---|
| 소스 | HSV GT (픽셀 임계값) | PaliGemma2 grounding |
| 코드 | [train_exp54_stage2_v2_action.py:106-111](../../scripts/train_exp54_stage2_v2_action.py#L106-L111) `bbox_feat()` | [eval_exp59_closedloop.py:101-116](../../scripts/eval_exp59_closedloop.py#L101-L116) `detect()` |
| cx 오차 | 0 (GT) | cx_err 0.075~0.286 (계통 오프셋) |

MLP는 **깨끗한 HSV 분포만** 봤고, VLM bbox는 한 번도 못 봄 → OOD → 극단 action → 누적 drift.
EMA smoothing(α=0.5)으로도 정체됨 → **지터가 아니라 systematic offset이 원인**임이 검증됨 (CURRENT_STATE_SNAPSHOT §1).

---

## 2. 목표

Stage2 MLP가 VLM bbox 분포에서도 동작하도록 재학습하여 **CL 성공률 회복** (목표: ≥ 50%, 이상적 70%+).
PM(프레임 정확도)는 유지 (현재 ~96%).

**비목표:** Stage1 인코더·grounder 재학습 (이건 이미 잘 동작). 액션 공간 변경 없음.

---

## 3. 접근 방식 (2안 + 권장 하이브리드)

### 접근 A — 합성 노이즈 증강 (빠름)
학습 데이터는 HSV GT 유지, `bbox_feat()`에서 학습 시에만 VLM-style 왜곡 주입.

```python
# bbox_feat() 안, 학습 중에만 적용 (eval/추론 시 비활성)
def bbox_feat(frames, t, augment=False, rng=None):
    arr = []
    for k in range(WINDOW):
        fr = frames[max(0, t - (WINDOW - 1 - k))]
        cx, cy, area, has = fr["cx"], fr["cy"], fr["area"], float(fr["has_bbox"])
        if augment and has > 0.5:
            cx   += rng.normal(OFFSET_MU_X, OFFSET_SD)   # 계통 오프셋 + 지터
            cy   += rng.normal(OFFSET_MU_Y, OFFSET_SD)
            area *= rng.uniform(1 - SCALE_J, 1 + SCALE_J)  # 스케일 jitter
            cx, cy = np.clip([cx, cy], 0, 1)
            if rng.random() < MISS_P:                      # 가끔 미검출 모사
                has = 0.0
        arr.extend([cx, cy, area, has])
    return np.array(arr, dtype=np.float32)
```

- **파라미터 출처:** §4 측정 단계에서 실제 VLM−GT delta 분포로 캘리브레이션
- **장점:** 빠름(재학습 ~수분, 재grounding 불필요), 노이즈 강도 sweep 쉬움
- **단점:** 가우시안 가정이 실제 VLM 오차 모양과 다를 수 있음

### 접근 B — 실제 VLM bbox로 학습 데이터 재구축 (충실)
PaliGemma2 Exp59로 **학습 에피소드 전체를 re-ground** → bbox feature를 VLM 출력으로 교체 → 그 분포로 MLP 학습.

- **장점:** 배포 분포와 정확히 일치 (train == test distribution)
- **단점:** PaliGemma2 3B 추론 느림 (150 ep × N frames, 수십 분~시간), grounding 실패 프레임 처리 필요

### 권장 — 하이브리드 (B로 측정 → A로 학습)
1. **측정만** B 방식으로: 학습셋 일부를 re-ground해서 (cx_vlm−cx_gt) 등 delta **통계만** 수집 (전체 재학습 X)
2. 그 통계로 A의 `OFFSET_MU/SD/SCALE_J/MISS_P` 캘리브레이션
3. A로 빠르게 재학습 + 노이즈 강도 sweep
4. Exp59 CL로 검증, 안 되면 B 전면 도입

> **기술 선택은 사용자 결정** — 위 3안 중 어느 것으로 갈지 지정 요망. (Claude 권장: 하이브리드)

---

## 4. 구현 단계 (승인 후)

```
[ ] S1. 측정 스크립트: 학습셋 N개 ep re-ground → (Δcx, Δcy, area_ratio, miss_rate) 통계 산출
        산출물: docs/v5/exp60_bbox_offset_stats.json
[ ] S2. train_exp54_stage2_v2_action.py 에 --augment + 노이즈 파라미터 인자 추가
        - bbox_feat(augment=True) 학습 경로에만 (line 227), evaluate(line 161)는 augment=False
        - rng = np.random.default_rng(seed) 재현성
[ ] S3. 재학습: 노이즈 강도 2~3개 sweep (약/중/강)
        출력: runs/v5_nav/mlp/exp54/stage2_v2/stage2_v2_mlp_aug{level}.pt
[ ] S4. PM 검증: eval_exp54_stage2_v2.py 로 frame acc 유지 확인 (≥95%)
[ ] S5. CL 검증: eval_exp59_closedloop.py 의 STAGE2_PT를 aug 모델로 → 성공률 측정
[ ] S6. 최선 모델 선정 + 결과 문서화 (exp60_report.md)
```

---

## 5. 수정/생성 파일

| 파일 | 변경 |
|---|---|
| `scripts/train_exp54_stage2_v2_action.py` | `bbox_feat()` augment 인자, argparse 노이즈 파라미터 |
| `scripts/measure_exp60_bbox_offset.py` | **신규** — VLM−GT delta 통계 (접근 B 측정용) |
| `scripts/eval_exp59_closedloop.py` | `STAGE2_PT` 경로 인자화 (aug 모델 평가용) |
| `docs/v5/exp60_bbox_offset_stats.json` | **신규** 측정 결과 |
| `docs/v5/exp60_report.md` | **신규** 결과 보고 |
| `runs/v5_nav/mlp/exp54/stage2_v2/stage2_v2_mlp_aug*.pt` | **신규** 재학습 가중치 |

> ⚠️ 기존 `stage2_v2_mlp.pt`는 **덮어쓰지 않음** — `_aug` suffix로 분리 보관 (롤백 가능)

---

## 6. 트레이드오프 / 리스크

- **과도한 노이즈** → PM 하락 (학습이 너무 흐려짐). sweep으로 최적점 탐색.
- **노이즈로도 안 되면** → 실제 오차가 가우시안이 아니거나(접근 B 필요) MLP 용량 부족.
- **CL 평가 비용** → PaliGemma2 3B 추론으로 22 ep ~3-4분. sweep 시 반복 → 측정 ep 수 제한 가능.
- **근본 한계** R3(basket-only)는 이 실험으로 안 풀림 — 별도 데이터 수집 필요(축 C).

---

## 7. 교수님 질문과의 연결

- Q1~Q4 (grounding "본다") → Exp57/59로 이미 답함 ✅
- 이 실험 = **"그래서 실제로 가는가?"** 에 대한 답 — grounding을 action까지 닫는 마지막 연결고리
- 성공 시: "텍스트 목표 → grounding → 실제 주행" end-to-end 데모 가능 (R2-4 강화)

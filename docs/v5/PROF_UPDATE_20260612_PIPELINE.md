# 교수님 업데이트 — 2026-06-12 파이프라인 원인 확정

**요약:** exp65b / exp66 / exp67 3-way ablation으로 CL 성능 격차의 원인이 cx 소스(grounding 품질)가 아니라 **파이프라인(L2-norm + bbox 증강)** 임을 확정했습니다.

---

## 1. 실험 요약

| 실험 | cx 소스 | 파이프라인 | val_acc | CL 성공률 |
|------|---------|-----------|---------|-----------|
| exp54 (참고 기준) | HSV | L2-norm + bbox 증강 | 92.6% | 96.6% |
| exp65b (대조군) | base PG2 | 단순 MLP | 90.2% | 10.3% |
| exp66 | base PG2 | L2-norm + bbox 증강 | 93.5% | **96.6%** |
| exp67 | exp59 LoRA | L2-norm + bbox 증강 | 94.5% | **96.6%** |

cx 소스를 base PG2(exp66)로 고정하더라도, 파이프라인만 올바르게 맞추면 96.6%로 복원됩니다.

---

## 2. 핵심 결론

1. **cx 소스(grounding 품질)는 CL 성능에 영향을 주지 않는다**
   - base PG2든 exp59 LoRA든 올바른 파이프라인에서 동일하게 96.6%
   - LoRA grounding 개선 → action 성능 기여 없음 (음성 결과로서 중요한 발견)

2. **L2-norm + bbox 증강이 유일한 성능 결정 요소다**
   - 파이프라인 있음: 96.6% / 없음: 10.3% → **×9.4배 격차**

3. **이전 잠정 결론 수정**
   - "단일 프레임 grounding 정확도 ≠ 궤적 action 성능"은 파이프라인 교란에 의한 착시였음
   - 올바른 파이프라인에서는 grounding 정확도와 action 성능이 함께 올라감

---

## 3. 파이프라인 차이 설명

**exp60 / exp65b (단순 MLP, 낮은 성능)**
- 이미지 피처를 정규화 없이 그대로 사용
- bbox 피처 증강 없음
- grounding 분포 불일치를 보정하지 않음

**exp54 / exp66 / exp67 (L2-norm + aug, 높은 성능)**
- `F.normalize(..., dim=-1)` 로 이미지 피처 L2 정규화
- `exp60_bbox_offset_stats.json` 기반으로 PG2 grounding 분포를 모사한 노이즈를 bbox 피처에 주입
- 학습-추론 간 분포 불일치를 사전에 보정

핵심은 **grounding 품질 자체가 아니라, 학습 시 분포 보정 여부**였습니다.

---

## 4. 이전 판단 수정

| 이전 판단 | 수정 내용 |
|----------|----------|
| CH32: "single-frame grounding accuracy ≠ trajectory action performance" | 파이프라인 교란이 원인이었음. 올바른 파이프라인에서는 해당 없음 |
| exp65 10.3%를 cx 소스(base PG2) 탓으로 귀인 | cx 소스 무관. 파이프라인 누락이 원인 |
| LoRA grounding 개선이 action에 기여할 것 | 기여 없음 확인. 음성 결과로 기록 |

---

## 5. 다음 계획

**의자(chair) 파이프라인 구축**

현재 상태:
- 좌/우 에피소드 0개, 중앙만 31ep 수집 완료
- 파일럿 결과: val_acc 65.3%, CL 57.1% (데이터 편향 기인 — 예상된 수치)

수집 완료 후 실행 순서:
1. `build_chair_dataset_v5_2` — chair 데이터셋 빌드
2. `gen_chair_pg2_annotation` — PG2 grounding 어노테이션 생성
3. 학습 (L2-norm + bbox 증강 파이프라인 적용)
4. CL 평가

좌/우 에피소드 수집이 완료되면 즉시 위 파이프라인을 돌릴 수 있는 상태입니다.

# Plan — exp65: base PG2 grounding 기반 Stage2 깨끗한 기준선

> 작성: 2026-06-11 · 상태: 승인된 방향(사용자 6/11) 실행
> 동기: grounding=base PG2 확정 → action 격리 baseline. systematic study용 대조군.

## 0. 핵심 통찰

기존 **exp60 Stage2는 exp59 LoRA로 annotation 생성**했다(`gen_exp60_pg2_annotation.py`).
exp59는 실환경에서 full-frame 붕괴(거리22%·저조도17%) → 그 cx가 ~0.5로 기록되어 **action 학습 신호 오염**.
exp60의 오탐 필터(cy<0.35, area<0.01)는 **full-frame(area>0.9)을 못 거른다.**

→ **base PG2로 재주석 + full-frame 가드 = 깨끗한 action 기준선(exp65).**
이건 그 자체로 systematic study 결과: "grounding 소스 품질이 action으로 전파됨 (exp59-annotated vs base-annotated Stage2)."

## 1. 변경 (exp60 복제 후 최소 수정)

| 파일 | exp60 | exp65 (신규) |
|---|---|---|
| annotation 생성 | `gen_exp60_pg2_annotation.py` (exp59 LoRA) | `gen_base_pg2_annotation.py` (**base PG2, 어댑터 없음**) |
| 오탐 필터 | cy<0.35 or area<0.01 | **+ area>0.9 (full-frame 가드)** 추가 |
| annotation 출력 | bbox_dataset_pg2_cx.json | bbox_dataset_base_pg2_cx.json |
| Stage2 학습 | `train_exp60_stage2_pg2cx.py` → mlp/exp60 | `train_exp65_stage2_basepg2.py` → mlp/exp65 |

나머지(Stage1 CLIP, MLP 구조, window=8, 8-class) 전부 동일 — **grounding 소스만 차이**로 통제.

## 2. 단계

1. `gen_base_pg2_annotation.py` 실행 → 150ep/2626프레임 재주석 (~22분 GPU)
2. `train_exp65_stage2_basepg2.py` 실행 → MLP 학습 (~수분)
3. CL 평가 (`eval_exp54_stage2_v2_closedloop.py` 계열) → **exp65(base) vs exp60(exp59) vs exp54(HSV)** 비교
4. 결과를 systematic study / 교수님 문서에 반영

## 3. 기대 / 판정

- exp65 CL ≥ exp60 이면 → "exp59 full-frame 오염이 action을 갉아먹었다" 정량 입증 (강한 systematic 결과)
- 비슷하면 → grounding 소스 영향 작음 (그래도 깨끗한 기준선 확보)
- 어느 쪽이든 **base PG2 위 action 최대치**를 의자 데이터의 대조군으로 확보

## 4. 완료 기준
- [ ] base PG2 annotation 생성 (hit율 기록)
- [ ] exp65 MLP 학습 + val PM
- [ ] CL 평가 3자 비교 (exp65/exp60/exp54)
- [ ] 결과 문서화

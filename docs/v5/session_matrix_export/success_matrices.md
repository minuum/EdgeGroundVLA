# 실주행 성공률 매트릭스 (5~7월, soda→minum, 2026-07-22)

세션 254개 · 경로검증 에피소드 124행

## path_type별 성공률

| path_type | 성공/시도 |
|---|---|
| center_straight | 2/2 (100%) |
| left_left | 0/1 (0%) |
| obj_center | 4/11 (36%) |
| obj_left | 3/8 (38%) |
| obj_right | 21/68 (31%) |
| right_left | 34/34 (100%) |

## checkpoint별 성공률 (session_id 조인된 55행)

| checkpoint | 성공/시도 |
|---|---|
| action_transformer.pt | 20/52 (38%) |
| exp73_pg448_v6_mlp.pt | 1/3 (33%) |

## grounder별 성공률 (조인된 행)

| grounder | 성공/시도 |
|---|---|
| OWL-v2 | 20/52 (38%) |
| PG2-448 | 1/3 (33%) |

## checkpoint × path_type 교차 성공률

| checkpoint | path_type | 성공/시도 |
|---|---|---|
| action_transformer.pt | left_left | 0/1 (0%) |
| action_transformer.pt | obj_right | 20/51 (39%) |
| exp73_pg448_v6_mlp.pt | obj_center | 1/3 (33%) |

## 세션 status 분포 (전체 세션, path_type 무관)

| status | 세션 수 |
|---|---|
| manual_stop | 163 |
| completed | 78 |
| stopped | 13 |

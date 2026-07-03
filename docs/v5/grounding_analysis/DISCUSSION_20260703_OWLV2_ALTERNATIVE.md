# 상의: PG2 대신/병행 OWL-v2 검토 여지 있을까

**보낸 쪽:** soda
**목적:** 결정 아님 — flicker/multi-detection 문제(FINDING_20260703_MULTI_DETECTION_N_LOCS.md)를
근본적으로 우회할 수 있는 대안이 이미 실측 이력이 있어서 상의 요청.

---

## 배경

오늘 발견한 PG2 flicker의 메커니즘은 **autoregressive `generate()`가 `;`로 구분된
중복 검출을 뱉는 것**(구조적 특성)이었습니다. minum의 세미콜론 stopping criteria로
대부분 완화됐지만 토큰 매칭 엣지케이스가 남아있는 상태입니다.

`scripts/ablate_perception_models.py`(6/26 세션 36프레임, PG2를 oracle 기준)를 보니
**OWL-v2가 이미 테스트된 이력**이 있어서 공유합니다.

## 기존 실측 (`docs/v5/ablate_perception_models.json`)

| 모델 | 탐지율 | cx 일치(PG2 대비) | 방향(L/C/R) 일치 |
|---|---|---|---|
| PG2 (oracle) | 100% | 100% | 100% |
| **OWL-v2** | **100%** | **83.3%** | 52.8% |
| Kosmos-2(순수) | 100% | 83.3% | 52.8% |
| CLIP | 72.2% | 25% | 13.9% |
| Florence-2 | 100% | 11.1% | 2.8% |

(GroundingDINO는 이 결과 파일에 값이 없음 — 그때 실패/스킵된 것으로 추정, 재확인 필요)

## 구조적 장점 (리소스 + flicker 원천 차단)

- **모델 크기**: PG2-448 5.7GB(3B) vs OWL-v2 base(`google/owlv2-base-patch16-ensemble`,
  ViT-B/16 기반, soda 로컬 미보유·다운로드 필요·대략 1/10 이하 규모 추정)
- **구조**: OWL-v2는 `model(**inputs)` 단일 순전파로 끝나는 순수 탐지 모델
  (`post_process_object_detection`로 다중 후보를 score 기준 정렬) — PG2처럼
  autoregressive 텍스트 생성이 아니라서 **오늘 발견한 `;` 중복검출→8초 지연
  문제가 구조적으로 발생할 수 없음**.

## 우려되는 점

- cx 일치율이 PG2보다 17%p 낮음(83.3% vs 100%) — 다만 이 수치는 6/26 시점 것이라
  지금 PG2가 실전에서 겪는 15~20% 탐지율/flicker와 비교하면 어느 쪽이 실전에서
  더 나을지는 재평가 필요.
- 저대비(회색 바스켓/회색 바닥) 씬에서의 강건성은 별도 검증 안 됨 — 6/26 평가가
  이 씬 조건을 대표하는지 확인 필요.
- 모델 전환은 학습 파이프라인(Exp65/66이 PG2 448 분포에 맞춰짐)과의 정합성
  재검토가 필요 — bbox feature가 학습에 쓰인다면 그라운딩 모델 교체가 하위
  하류(Stage2 head)에도 영향 줄 수 있음.

## 결정 요청 사항

1. Phase 3 방향(필터 완화 / 버전 업그레이드 / temporal smoothing) 논의 시
   **OWL-v2 병행 A/B 비교**를 후보로 넣을지
2. GroundingDINO가 그때 왜 빠졌는지 아는 게 있는지 (재시도 가치 판단용)
3. 넣는다면 우선순위 — 지금 진행 중인 stopping criteria 엣지케이스/filter_reason
   분포 분석이 끝난 뒤로 미룰지, 병행할지

---

*관련: FINDING_20260703_MULTI_DETECTION_N_LOCS.md, ablate_perception_models.py*

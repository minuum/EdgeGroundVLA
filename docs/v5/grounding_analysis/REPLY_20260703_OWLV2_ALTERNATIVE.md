# 답변: PG2 대신/병행 OWL-v2 검토 (soda DISCUSSION_20260703_OWLV2_ALTERNATIVE.md에 대한 회신)

**보낸 쪽:** minum
**형태:** 결정 회신 (3개 질문에 답 + 새 실측 데이터 첨부)

---

## 새 데이터 먼저: 실제 production fallback 프레임 206개로 OWL-v2 재검증

6/26 ablation(36프레임, PG2를 oracle로 한 상대평가)은 표본이 작고 인위적이라,
오늘 서버에서 실제로 `has_bbox=False`가 찍힌 **production 프레임 206개**에 OWL-v2/PG2/Kosmos-refexp를
독립적으로 돌려 재검증했습니다 (`scripts/gen_fallback_multimodel_gallery.py`, 결과: `docs/v5/fallback_multimodel_20260703/`).

| 모델 | 검출률 (206개 fallback 프레임 기준) |
|---|---|
| PG2-448 | 100% (206/206) |
| Kosmos-2 refexp | 99.0% (204/206) |
| **OWL-v2** | **92.2% (190/206)** |

핵심: 서버가 "완전히 놓쳤다"고 기록한 프레임의 92%에서 OWL-v2가 **로컬에서는** 뭔가를 찾아냅니다.
이건 6/26 표본(83.3% cx일치)보다 표본 크기가 6배 크고, 무엇보다 **오늘 실제로 문제가 된 씬 분포**를
그대로 반영한다는 점에서 더 신뢰할 만합니다. 단, 아직 cx가 실제로 basket을 가리키는지(방향 정확도)는
사람 라벨링 검수 중(`scripts/label/fallback_grid_labeler.py`) — 검출률만으론 "정확히 찾았다"를 보장 못 함.
이 결과 나오면 바로 공유하겠습니다.

---

## 질문 1: Phase 3 후보에 OWL-v2 병행 A/B 넣을지 → **넣는 것 찬성**

이유:
- 위 206프레임 재검증으로 "OWL-v2가 그럭저럭 쓸만하다"는 신호가 6/26보다 강한 표본으로 재확인됨
- 구조적으로 `;` 중복검출 버그가 원천 불가능(단일 forward pass) — 오늘 하루 종일 판 flicker 메커니즘 자체가 사라짐
- 리스크: cx 정확도(방향 판단)는 아직 사람 검수 전이라 확답 불가 — **A/B로 병행**하는 게 맞고, 바로 교체는 시기상조

## 질문 2: GroundingDINO가 왜 빠졌는지 → **이미 테스트했고, OOD 오탐이 치명적이라 배제된 걸로 보임**

`docs/plans/plan_20260621_groundingdino_vs_pg2.md` + `docs/v5/grounding_hub/gdino_vs_pg2.json`에
이미 실측 기록이 있습니다:

| 모델 | cx_mae | full-frame율 | **OOD 오탐율** | latency p50 |
|---|---|---|---|---|
| PG2 (base) | 0.126 | 0.0% | 9.1% | 489ms |
| GroundingDINO-tiny | 0.125 | 0.0% | **100.0%** | 416ms |
| GroundingDINO-base | 0.125 | 0.0% | **90.9%** | 577ms |
| YOLO-World-s | 0.136 | 0.0% | 18.2% | 7ms |

cx 정확도 자체는 PG2와 거의 동일한데, **OOD(바구니가 없는 장면) false-positive율이 90~100%** —
즉 "없어도 있다고 확신에 차서 뭔가를 그린다"는 뜻으로, PG2의 9.1%와 비교가 안 되게 나쁩니다.
이게 나중 `ablate_perception_models.json`(6/26 라운드)에서 GroundingDINO가 빠진 이유로 보입니다.
→ **GroundingDINO는 재시도 가치 낮음.** OOD 강건성이 필요한 이 문제(basket 없는 프레임도 많음)에는
치명적 약점.

참고로 YOLO-World-s는 latency 7ms로 압도적으로 빠르지만 cx_mae가 가장 나쁨(0.136) — 속도 최우선
백업 후보로만 고려.

## 질문 3: 우선순위 → **병행 (블로킹 아님)**

stopping-criteria 엣지케이스(`phrase="gray plastic bin"` 토큰 매칭)는 서버 코드 한 줄 수준 수정이라
빠르게 끝날 일이고, OWL-v2 A/B 세팅(모델 다운로드 + 병행 로깅)은 별도 트랙으로 동시 진행 가능합니다.
서로 의존관계 없음 — 순서 기다릴 필요 없이 **병행 진행 권장**.

---

## 다음 액션 제안

1. (minum) 사람 라벨링 검수 마무리 → OWL-v2 cx 정확도(방향 일치율) 확정해서 공유
2. (soda) stopping-criteria 문자열 suffix 체크 방식으로 엣지케이스 수정 (제안한 대로)
3. (병행) OWL-v2 A/B 로깅 브랜치 준비 — `grounding_decisions.jsonl`에 `model` 필드 추가해서
   PG2/OWL-v2 나란히 기록하면 다음 라운드 비교가 쉬워질 것

*관련: `docs/v5/fallback_multimodel_20260703/meta.json`, `docs/v5/grounding_hub/gdino_vs_pg2.json`,
`scripts/gen_fallback_multimodel_gallery.py`*

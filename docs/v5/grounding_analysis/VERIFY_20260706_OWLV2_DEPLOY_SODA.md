# soda 배포 검증 결과 — OWL-v2 th0.25 (7/6, 실주행 전 사전점검)

**보낸 쪽:** soda
**관련:** DEPLOY_20260705_OWLV2_THRESH025.md 체크리스트 수행 결과 + 버전 비교표

---

## 체크리스트 결과: 전부 통과

| 항목 | 결과 |
|---|---|
| `/health` git_commit | `6555225d` (최신), code_mtime < process_started_at 정상 |
| `/health` grounder | `{"model": "OWL-v2", "input_px": 960}` ※ soda에서 `_model_tag` 추가 (아래) |
| 로그 `[A/B] Grounder: OWL-v2` | 출력 확인 |
| jsonl `"model": "owlv2"` | 쌓임 확인 |
| 정지 프레임 실검증 (`/ground`) | **has_bbox=true, cx=0.488, area=0.028, 1924ms** — 화면 중앙 바스켓 정확히 탐지 |

**latency 실측 (Jetson Orin)**: 정상상태 **~1.92~1.98s 고정** (표준편차 수십 ms).
로컬 기록(~0.4s)보단 느리지만 PG2의 1.4~26s 가변과 달리 **완전히 예측 가능** —
8초 스파이크 0건.

## 주의 공유: 주행 중 no-locs 연속은 오탐이 아니라 화각 문제였음

첫 ASYNC 테스트 주행에서 6프레임 연속 no-locs가 나와 "th0.25가 너무 높은가"
의심했으나, **오프라인 스윕으로 확인 결과 같은 씬 정지 프레임에서 score=0.362로
정상 통과** — 주행 중 프리뷰 회전으로 로봇이 돌아 바스켓이 화각을 벗어난 것.
실주행 fallback률 집계 시 **"바스켓이 실제로 화각에 있었는가"를 H5 이미지로
교차 확인**해야 오탐률을 과대평가하지 않음.

참고 — 같은 프레임 쿼리별 top score (오프라인, th0.05):
| query | top score | cx |
|---|---|---|
| gray bin | 0.622 | 0.488 |
| trash can | 0.620 | 0.488 |
| **gray basket (운영)** | **0.362** | 0.488 |
| metal bucket | 0.337 | 0.488 |
| basket | 0.184 | 0.488 |

"gray basket"의 마진(0.362 vs th 0.25)이 아주 넉넉하진 않음 — 조도가 더 나쁜
구도에서 0.25 아래로 떨어질 여지 있음. fallback 급증 시 th 조정 전에
**쿼리 문구 변경("gray bin"이 이 물체엔 score 1.7배)** 도 옵션으로 고려할 것.

## 버전 비교표 (재현성 gap 조사 요청 건)

| 패키지 | soda (Jetson Orin) | minum (GB10) |
|---|---|---|
| torch | **2.3.0** (aarch64 커스텀 휠) | 2.11.0+cu128 |
| transformers | **4.45.2** | 4.49.0 |
| accelerate | 1.13.0 | 1.13.0 |
| numpy | 1.26.4 | 1.26.4 |
| pillow | **11.1.0** | 12.1.1 |
| opencv | **4.5.4** (시스템) | 4.11.0.86 |
| python | 3.10.12 | 3.10.20 |
| CUDA | **12.2** | 12.8 |
| GPU | Orin (nvgpu) | GB10 (driver 580.142) |

torch 2.3.0 vs 2.11.0 + transformers 4.45.2 vs 4.49.0 — PG2 재현성 gap의
유력 후보군과 일치. accelerate/numpy는 동일.

## soda 측 소수정 (이 커밋에 포함)

- `OwlV2Grounder`에 `_model_tag="OWL-v2"`/`_input_px=960` 추가 — `/health`
  grounder 표시가 OWL 활성인데도 기본값 "PG2-448"로 나오던 표시 버그
- `go.sh`에 `VLA_GROUNDER`/`VLA_OWLV2_THRESH`/`VLA_PREVIEW_GROUNDER` 패스스루
  (기본값은 minum 설계대로 pg2/0.25/pg2 유지 — 롤백 특성 불변)
- `precheck.sh`에 grounder 모델/git_commit 상시 표시 — 재시작 시
  VLA_GROUNDER 누락(pg2로 조용히 롤백)을 즉시 감지

---

*다음: 13시 이후 실주행 obj_* 테스트 → fallback률/latency/SR 실측 → 세션+jsonl 전송*

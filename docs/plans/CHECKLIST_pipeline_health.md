# 파이프라인 건강진단 체크리스트

> 작성: 2026-06-22 · 이번 세션에 실제로 겪은 문제(latency 드리프트 오인, 그라운딩
> 품질, 리사이즈 일관성)를 다음에도 빠르게 감지할 수 있게 체크리스트로 정리.
> 실행 스크립트: `scripts/eval/diagnose_pipeline_health.py`

## 사용법

```bash
# 전체 체크 (서버 + 세션 로그 둘 다 있을 때)
.venv/bin/python3 scripts/eval/diagnose_pipeline_health.py \
  --server http://localhost:8001 --api-key $VLA_API_KEY \
  --session docs/v5/s6_cl_sim.json --fps 1.0

# 서버만 (latency만)
.venv/bin/python3 scripts/eval/diagnose_pipeline_health.py --server http://localhost:8001 --api-key $VLA_API_KEY

# 세션 로그만 (드리프트/연속성/그라운딩)
.venv/bin/python3 scripts/eval/diagnose_pipeline_health.py --session docs/v5/s6_cl_sim.json --fps 1.0
```

---

## A. Latency — 단발 호출이 목표(1초) 안에 들어오는가

- **체크**: `/predict`를 N회 호출(warm 상태), 평균 latency 측정.
- **기준**: 평균 < 1000ms = PASS, < 2000ms = WARN, 그 이상 FAIL.
- **이번 세션 실측값**(참고, 시점 고정값 — 재측정 권장):
  - GB10(로컬): 560~650ms
  - soda(Jetson, 운영): ~1,310~1,325ms — **WARN 수준, 목표 미달**
- **다음에 이게 또 느려지면**: PG2 cold-start(43초, lazy-load) 때문인지 먼저 확인 — 첫 호출만 비정상으로 느리면 정상(캐시 워밍 전), 반복 호출도 계속 느리면 진짜 문제.

## B. Drift — "N초 차이"가 단발 latency 문제인지 누적 드리프트인지 구분

- **배경**: 1fps로 추출한 영상을 시뮬레이션 재생할 때, 실제 처리속도가 1fps보다
  느리면 "처리시간 누적값"과 "영상 nominal 시간"의 차이가 **프레임이 갈수록 계속 커진다**
  — 이걸 단발 호출 latency로 착각하기 쉽다(이번 세션에 실제로 이렇게 착각했었음).
- **체크**: 세션 JSON의 `total_latency_ms`를 누적해서 nominal 시간(`frame/fps`)과 비교.
- **기준**: 평균 처리시간 < `1/fps` 초 = PASS(드리프트 안 쌓임), 그 이상이면 FAIL(드리프트 발산).
- **실측 사례**(S6, gnd_20260618_172621, 105프레임, 1fps): 평균 1.36s/frame > 1.0s/frame
  → 마지막 프레임에서 37.4초 드리프트. **"3~4초 차이"로 보였던 건 frame 10번 근처의 드리프트값(4.0초)이었다** — 단발 latency 아님.

## C. Continuity — 프레임 누락 확인

- **체크**: 세션 JSON의 `frame` 번호가 1..N 연속인지.
- **기준**: 누락 0개 = PASS.
- **실측 사례**: S6 세션 105/105 전부 연속, 누락 없음 — "동시 수집/추론" 의심도 기각(코드 자체가 순차 호출이라 동시성 문제 자체가 구조적으로 발생 불가).

## D. Resize — 학습/추론 이미지 전처리가 224x224로 일관되는가

- **체크**: `robovlm_nav.image_preprocess.resize_for_vlm()`이 1280x720 입력을 224x224로 만드는지.
- **기준**: 출력 size == (224,224) = PASS.
- **배경**: `plan_20260622_train_inference_image_pipeline_unify.md` — 학습/추론 둘 다 HF
  AutoProcessor의 암묵적 리사이즈에 의존하던 걸 명시적 함수로 통일(2026-06-22).

## E. Grounding — 그라운딩 신뢰도(has_bbox/area) 분포

- **체크**: 세션 JSON의 `has_bbox=False` 비율, 검출된 것의 평균 area.
- **기준**: `has_bbox=False` 비율 < 30% = PASS, 그 이상 WARN.
- **배경**: CH41에서 `has_bbox=False` 프레임의 행동 오예측률이 3~5배 높다는 게 확인됨
  — 그라운딩 신뢰도가 낮은 세션은 행동 예측도 같이 불안정할 가능성이 큼.

---

## 다음에 새 세션/실험을 검증할 때

1. 위 스크립트를 **항상 먼저** 돌려서 5개 체크 통과 여부를 본다.
2. B(드리프트)가 FAIL이면 — 절대 "단발 호출이 N초 걸린다"고 결론 내리지 말고,
   반드시 A(단발 latency)를 따로 측정해서 둘을 분리한다.
3. E(그라운딩)가 WARN이면 — 행동 예측 결과를 너무 믿지 말고 그라운딩 자체부터 점검.

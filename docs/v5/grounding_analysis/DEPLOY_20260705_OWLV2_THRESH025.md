# 배포 안내 — OWL-v2 threshold 0.25 확정 + 쿼리 버그 수정

> 작성일: 2026-07-05
> 대상: soda (실주행 테스트 7/6 월 13시 이후)

## 이번 커밋 변경 사항

1. **OWL-v2 threshold 0.1 → 0.25** — `owlv2_threshold_roc.py` 실측 확정값.
   정탐 유지율 95.3%, 오탐(객체없음 프레임에서 검출) **74.7% → 0%**.
   환경변수로 조절 가능: `VLA_OWLV2_THRESH` (기본값 0.25)
2. **OwlV2Grounder 쿼리 버그 수정** — `phrase`에 "gray"가 없으면 강제로 "gray {phrase}"를
   접두하던 로직 제거. `instruction="red ball"` 등 임의 객체를 넘겨도 정상 그라운딩되도록.
3. **PG2 stopping-criteria 세미콜론 병합 버그 수정** (지난 커밋 bc8f310d, 재확인차 포함)
4. **robovlm_nav/perception/hsv_basket.py 신규** — 서버가 무조건 import하므로 필수 동봉.
   HSV 프리뷰 대체안은 검증 후 기각(정확도 1~3%)됐고, `VLA_PREVIEW_GROUNDER` 기본값은
   여전히 `pg2`라 실사용엔 영향 없음. import만 되면 됨.

## 활성화/확인 방법 (기존과 동일)

```bash
export VLA_GROUNDER=owlv2          # 메인 그라운더 A/B
# export VLA_OWLV2_THRESH=0.25     # 기본값이라 생략 가능
```

확인 체크리스트:
- `/health`에서 git_commit 최신인지
- 로그에 `[A/B] Grounder: OWL-v2` 출력
- `grounding_decisions.jsonl`에 `"model": "owlv2"` 항목 쌓이는지
- 부재판정 동작: 바구니 없는 프레임에서 `has_bbox: false` 나오는지 (이전엔 여기서 오탐)

## 로컬(minum)에서 검증된 근거 (실로봇 테스트 전 참고)

- `docs/v5/grounding_benchmark/CONCLUSION.md` — 정확도/오탐/IoU 3지표 벤치마크
- `docs/v5/closed_loop_eval/CH60_OWL_TEXT_CLOSED_LOOP.md` — 운영 계보(exp66/71) drop-in
  스왑 검증: 재학습 없이 그라운더만 교체해도 SR 완전 유지 (96.6%/100%)
- **단, 위는 전부 오프라인 리플레이 기준** — 이 계보는 리플레이 CL이 포화 상태라 실로봇
  fallback률/latency/실제 SR로 최종 확인 필요 (7/6 테스트의 목적)

## 롤백

`VLA_GROUNDER` 미설정(또는 `pg2`) 시 기존 동작 완전 동일. `VLA_OWLV2_THRESH`도
0.1로 되돌리면 이전 동작 재현 가능(비권장 — 오탐 급증).

## 별건 — Jetson vs 로컬 환경 버전 비교 요청 (재현성 gap 미해결 건)

`docs/v5/grounding_analysis/CONCLUSION_20260704_fallback_repro_gap.md`에서 확인된
"서버 fallback 프레임 100%가 로컬 재실행 시 탐지됨" 문제의 근본 원인 후보가
Jetson-vs-local 환경 버전 차이입니다. OWL-v2 전환으로 이 실패 경로 자체는 우회되지만,
근본 원인은 아직 안 밝혀졌으니 시간 될 때 아래 부탁드립니다:

```bash
.venv/bin/python3 -m pip freeze | grep -iE "^torch|^transformers|^accelerate|^pillow|^numpy|^opencv"
python3 --version
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader  # 또는 jetson_release
```

로컬(minum) 기준값:
```
torch==2.11.0+cu128 / transformers==4.49.0 / accelerate==1.13.0
numpy==1.26.4 / opencv-python==4.11.0.86 / pillow==12.1.1
python 3.10.20, CUDA 12.8, GPU NVIDIA GB10 (driver 580.142)
```

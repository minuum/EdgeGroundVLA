# fallback 프레임 206개 — 로컬 재분석 결론 (soda 없이 자체 확정)

> 작성일: 2026-07-04
> 데이터: `docs/v5/fallback_multimodel_20260703/meta.json` (206 프레임, PG2/Kr/Ow 3모델 로컬 재실행 결과)

## 결론: (A) Jetson-vs-local 재현성 gap이 지배적 원인. (B) 진짜 타겟 안 보임은 사실상 배제.

## 근거 (사람 라벨링 없이 3모델 교차검증으로 확정 가능한 수치)

| 지표 | 값 |
|---|---|
| 서버가 fallback(has_bbox=False) 처리한 206프레임 중, 로컬에서 **1개 이상** 모델이 탐지 | **206/206 (100%)** |
| 로컬에서 **완전히 탐지 실패**(진짜 안 보임 후보) | **0/206 (0%)** |
| PG2 vs OWL-v2 둘 다 탐지된 경우, cx 오차 0.08 이내로 **일치** | **160/190 (84.2%)** |
| 3모델 중 2개 이상이 cx 0.08 이내로 일치 (독립 모델 교차확정) | **151/206 (73.3%)** |
| Kosmos-refexp(kr)만 유독 cx가 좁은 구간(0.10~0.17)에 몰림 — 자체 편향/기본값 의심 | 19/206 (9.2%) |

## 해석

- 서버가 "안 보인다"고 fallback 처리한 프레임의 **100%**에서 로컬 재실행 시 최소 하나의 모델이 그럴듯한 bbox를 찾음.
- 그 중 84%는 PG2와 OWL-v2라는 **서로 다른 아키텍처**가 cx 0.08 이내로 합의 → 우연이 아니라 실제로 타겟이 그 위치에 있다는 강한 증거.
- 이견(53건)의 상당수는 kr(Kosmos-refexp)이 이상치인 경우가 많음 — kr 자체의 알려진 약점(구 ablation에서도 dir-match 52.8%로 낮음)과 일치, PG2/OWL-v2 쌍은 여전히 잘 맞는 경우가 많음.
- 즉 "타겟이 화면에 없어서 fallback났다"는 가설은 이 데이터로 사실상 기각됨. 남은 설명은 Jetson 서버의 PG2 실행 환경(torch/transformers 버전, 또는 우리가 이미 알고 있는 `;` 중복생성/8초 지연/full-frame hallucination 버그)이 동일 입력에 대해 로컬과 다른 출력을 낸다는 것.

## 사람 라벨링(그리드 툴)의 남은 역할

자동 교차검증으로 "전체 결론"은 이미 확정 가능하지만, 이견 53건(디스크: 위 표 세번째 줄 미충족 프레임)에 대해서만 사람이 직접 봐서 "그림상 실제로 어디 있는지" 확인하면 kr의 이상치 여부를 완전히 확증할 수 있음. 전체 206건을 다 볼 필요는 없어짐.

- 그리드 라벨러: http://localhost:7791 (포트 7791, `scripts/label/fallback_grid_labeler.py`, 현재 실행 중)
- 라벨 저장: `docs/v5/fallback_multimodel_20260703/human_labels.json`

## 추가 (2026-07-05) — Jetson vs 로컬 환경 버전 비교 요청

이 문서 작성 당시 "torch/transformers 버전 비교가 다음 우선순위"라고만 적어두고 로컬
버전 정보 자체를 안 남겨서 지금 추가. **재현성 gap의 근본 원인은 아직 미해결.**

### 로컬(minum) 환경

```
torch==2.11.0+cu128
torchvision==0.26.0+cu128
transformers==4.49.0
accelerate==1.13.0
numpy==1.26.4
opencv-python==4.11.0.86
pillow==12.1.1
python 3.10.20 (venv), CUDA 12.8
GPU: NVIDIA GB10 (driver 580.142)
```

### soda(Jetson)에 요청할 것

```bash
.venv/bin/python3 -m pip freeze | grep -iE "^torch|^transformers|^accelerate|^pillow|^numpy|^opencv"
python3 --version
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader  # 또는 jetson_release
```

버전이 다르면 → 동일 버전으로 맞춰서 fallback 재현율 변화 확인.
버전이 같으면 → 하드웨어(Jetson fp16/양자화) 또는 이미 수정된 PG2 stopping-criteria
세미콜론 버그(`bc8f310d`)가 원인일 가능성으로 좁혀짐. 단, OWL-v2 전환(`d07ead37`)으로
이 실패 경로 자체가 우회되므로 실무 우선순위는 낮아졌음 — 그래도 근본 원인 규명 자체는
별도로 남겨둘 가치 있음(다른 유사 재현성 이슈 예방).

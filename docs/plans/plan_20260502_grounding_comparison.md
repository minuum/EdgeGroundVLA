# Plan — Grounding 모델 비교 테스트 (Kosmos-2 vs PaliGemma vs Moondream2)

작성: 2026-05-02
브랜치: inference-integration

## 0. 목표

Kosmos-2의 "gray basket 오인식" 문제를 다른 모델과 정량 비교.
- Kosmos-2: "trash can", "air conditioner" 등으로 잘못 명명하지만 coarse 방향은 맞음
- 질문: PaliGemma / Moondream2가 더 나은 레이블 + 동등 이상의 위치 정확도를 보이는가?

테스트 모델 (총 4개):
- `kosmos` — baseline
- `paligemma-mix` — `google/paligemma-3b-mix-224` (1세대 mix, detect 지원 확실)
- `paligemma2-mix` — `google/paligemma2-3b-mix-224` (2세대, detect 지원 여부 포함 검증)
- `moondream` — `vikhyatk/moondream2`

## 1. 비교 기준 (GT bbox 없으므로 proxy 지표 사용)

| 지표 | 설명 |
|---|---|
| **Detection rate** | 프레임당 bbox를 하나라도 찾는 비율 |
| **Label accuracy** | "basket" 관련 단어가 캡션에 포함되는 비율 |
| **Direction agreement** | cx 기반 left/center/right가 Kosmos-2 결과와 일치하는 비율 |
| **cx 분포** | 전체 프레임 cx 평균/표준편차 (치우침 없는지) |
| **Latency** | 프레임당 추론 시간 (ms) |

Direction agreement 기준: Kosmos-2를 seed_coarse_agreement=1.0 신뢰 기반으로 pseudo-GT로 사용.

## 2. 샘플 프레임 전략

- `bbox_dataset.json` 45 에피소드 × has_bbox=True 프레임 중 **50프레임 랜덤 샘플**
- path_type 균형: straight / left / right 포함
- H5 경로: `/home/minum/minum/26CS/MoNa-pi/mobile_vla_dataset_v5/`
- 이미지 키: `f['observations']['images'][frame_idx]` (V5 포맷)

## 3. 모델별 grounding 방식

### Kosmos-2 (baseline, 재실행)
```python
PROMPT = "<grounding>The gray basket is at"
# AutoModelForVision2Seq + processor.post_process_generation()
# → entities 리스트에서 bbox 파싱 (기존 GroundingBackend 로직 재사용)
```

### PaliGemma 1세대 (`google/paligemma-3b-mix-224`)
```python
PROMPT = "detect gray basket\n"
# AutoProcessor + PaliGemmaForConditionalGeneration
# 출력: "<loc_0123><loc_0456><loc_0789><loc_0987> gray basket"
# loc 토큰: 0~1023 → 0.0~1.0 정규화
```

### PaliGemma 2세대 (`google/paligemma2-3b-mix-224`)
```python
PROMPT = "detect gray basket\n"
# 동일 방식 시도. detect 미지원 시 fallback: "Where is the gray basket?"
# 결과에 detect 지원 여부도 함께 기록
```

> ⚠️ `paligemma-3b-pt-224` (pretrained-only)는 detect 태스크 미지원. mix 버전 사용.

### Moondream2 (`vikhyatk/moondream2`)
```python
# revision="2025-01-09" 권장
model = AutoModelForCausalLM.from_pretrained("vikhyatk/moondream2", 
    trust_remote_code=True, revision="2025-01-09")
result = model.detect(image, "gray basket")
# → [{"x_min":0.1, "y_min":0.2, "x_max":0.4, "y_max":0.6}, ...]
```

## 4. 구현: `scripts/test_grounding_comparison.py`

단일 스크립트로 3개 모델 순차 실행 후 결과 JSON + 콘솔 표 출력.

```
python3 scripts/test_grounding_comparison.py \
  --dataset docs/v5/bbox_nav_step1/bbox_dataset.json \
  --data-dir /home/minum/minum/26CS/MoNa-pi/mobile_vla_dataset_v5 \
  --n-frames 50 \
  --models kosmos paligemma-mix paligemma2-mix moondream \
  --output docs/v5/grounding_comparison/results.json
```

### 출력 형식
```json
{
  "kosmos":         {"detection_rate": 0.94, "label_acc": 0.33, "mean_cx": 0.50, "latency_ms": 280},
  "paligemma-mix":  {"detection_rate": 0.XX, "label_acc": 0.XX, "mean_cx": 0.XX, "latency_ms": XX},
  "paligemma2-mix": {"detection_rate": 0.XX, "label_acc": 0.XX, "mean_cx": 0.XX, "latency_ms": XX, "detect_supported": true/false},
  "moondream":      {"detection_rate": 0.XX, "label_acc": 0.XX, "mean_cx": 0.XX, "latency_ms": XX}
}
```

콘솔에는 모델별 5개 샘플 캡션도 출력.

## 5. 수정 파일

| 파일 | 변경 |
|---|---|
| `scripts/test_grounding_comparison.py` | **신규** 생성 (~200줄) |
| `docs/v5/grounding_comparison/results.json` | 신규 (실행 후 생성) |

기존 코드 수정 없음.

## 6. 참고

- **HF 다운로드**: PaliGemma 1세대 ~3GB, PaliGemma2 ~3GB, Moondream2 ~2GB. 첫 실행 시 자동 다운로드.
- **소요 시간 추정**: 모델당 50프레임 × ~5초 = ~4분. 4개 모델 합계 ~20분 (다운로드 별도).
- **paligemma2-mix detect 미지원 시**: fallback으로 `"Where is the gray basket?"` VQA 방식으로 전환하고 결과에 `detect_supported: false` 기록.

## 7. 완료 조건

- [ ] 스크립트 작성
- [ ] 3개 모델 실행 완료
- [ ] 결과 JSON 저장
- [ ] 콘솔 비교 표 출력

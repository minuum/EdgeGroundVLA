# Plan — 작은/먼 객체(area<0.05) 2배 줌 재그라운딩 ablation

> 작성: 2026-06-24 · 상태: **승인됨(사용자 "ㄱㄱㄱㄱㄱ") — 바로 구현**
> 동기: CH46-4 — 오류 프레임의 area가 정답 프레임보다 약 40% 작음(3개 모드 전부). "탐지는 되지만 작게
> 잡히면 액션 예측이 불안정해진다"가 현재 결론. 작은 객체에 한해 중앙부 2배 줌 크롭으로 재그라운딩하면
> bbox 정밀도(나아가 액션 정확도)가 개선되는지 검증.

---

## 0. 리서치 결과

| 항목 | 사실 |
|---|---|
| 대상 규모 | `bbox_dataset_full_pg2.json`의 has_bbox=True 2,621프레임 중 **area&lt;0.05가 36.2%**(949프레임) |
| 현재 그라운딩 입력 | `PG2Grounder.run()` — `resize_for_vlm`로 224×224 축소 후 추론. 작은 객체는 224×224로 다운샘플될 때 디테일 손실이 더 큼 |
| 줌 전략 | area&lt;0.05인 프레임만 원본 h5 이미지(1280×720)에서 현재 (cx,cy) 중심으로 가로/세로 1/2 크기 크롭 → 그 크롭을 다시 `PG2Grounder.run()`에 통과(크롭 내부에서 또 224로 리사이즈됨, 실효 확대 2배) → 결과 bbox를 크롭 좌표→원본 정규화 좌표로 역변환 |
| 실패 시 폴백 | 줌 재시도에서 has_bbox=False가 나오면 **원본(줌 전) 어노테이션 유지** — 악화 방지 |

---

## 1. 줌 재그라운딩 스크립트

`scripts/eval/regroun_zoom_small.py`(신규):
```python
def zoom_crop(img, cx, cy, zoom=2.0):
    H, W = img.shape[:2]
    cw, ch = W / zoom, H / zoom
    x0 = int(np.clip(cx * W - cw / 2, 0, W - cw))
    y0 = int(np.clip(cy * H - ch / 2, 0, H - ch))
    crop = img[y0:y0+int(ch), x0:x0+int(cw)]
    return crop, x0, y0, cw, ch

# 재그라운딩 후 좌표 역변환
x1_full = (x0 + bbox["x1"] * cw) / W   # 등등 y1/x2/y2 동일 패턴
```
- 대상: area&lt;0.05인 프레임만(나머지는 그대로 복사)
- 출력: `bbox_dataset_full_pg2_zoomsmall.json` (기존 파일 보존)

## 2. 비교

CH46의 LSTM-none(plain, area_delta 없음 — 현재 메인 디폴트 계열)과 같은 코드/split으로 비교:

| 구성 | val_acc | closed-loop FPE |
|---|---|---|
| PG2만(CH46, 줌 없음) | 94.88% | 0.145m |
| PG2 + 줌 재그라운딩(이번) | 측정 | 측정 |

`--use_hidden_state none`만 1차 검증(기존 ablation들과 동일 범위 원칙).

## 3. 문서화

CH50으로 추가. 개선 있으면 closed-loop까지, 없으면(또는 악화) 정직하게 기록 후 종료.

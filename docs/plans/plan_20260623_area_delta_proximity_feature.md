# Plan — area 변화율(proximity proxy) feature 추가 ablation

> 작성: 2026-06-23 · 상태: **승인됨(사용자 "ㄱㄱ") — 바로 구현**
> 동기: CH46-5에서 확인됨 — PG2 재주석 후 LEFT/RIGHT/ROT_L 클래스의 절대 area가 거의 무분산(p90≈mean)이 되어
> "박스가 커짐=가까워짐" 신호가 사라짐 → closed-loop FPE 악화(46-3). 절대 area 대신 **윈도우 내 변화율**을
> 추가하면, 회전 직전 프레임 자체엔 신호가 없어도 그 직전 FORWARD 구간의 area 상승 추세는 살아있을 수 있음
> (FORWARD는 area mean/p90=0.144/0.304로 여전히 변동폭이 큼) — 이 추세를 윈도우가 포착할 수 있는지 검증.

---

## 0. 확인된 사실

| 항목 | 값 |
|---|---|
| cy도 같은 문제 확인 | LEFT/RIGHT/ROT_L/ROT_R cy std 0.002~0.006 — area와 마찬가지로 거의 무분산, cy 단독 대체는 의미 없음 |
| FORWARD area 변동폭 | mean=0.144, p90=0.304 — 윈도우가 회전 직전 FORWARD 프레임들을 포함하므로 그 구간의 추세는 살아있을 가능성 있음 |
| 현재 feature 구조 | `bbox_feat()`: window(8) × [cx,cy,area,has_bbox] = 32차원, 절대값만 사용, 프레임간 변화율 없음 |

## 1. 변경 사항

`bbox_feat()`에 윈도우 내 각 스텝의 `area_delta = area[k] - area[k-1]`(k=0은 윈도우 밖 직전 프레임과 비교, 없으면 0)을 5번째 채널로 추가 → window(8) × 5 = 40차원.

```python
def bbox_feat_v2(frames, t, window=WINDOW):
    arr = []
    prev_area = None
    for k in range(window):
        idx = max(0, t - (window - 1 - k))
        fr = frames[idx]
        cx, cy, area = fr.get("cx", 0.5), fr.get("cy", 0.5), fr.get("area", 0.05)
        has = float(fr.get("has_bbox", False))
        prev_idx = max(0, idx - 1)
        delta = area - frames[prev_idx].get("area", area)
        arr.extend([cx, cy, area, has, delta])
    return np.array(arr, dtype=np.float32)
```

- `train_hidden_state_action.py` → 복제본 `scripts/train_hidden_state_action_areadelta.py` (D_IN 32→40 반영)
- `train_hidden_state_lstm.py` → 복제본 `scripts/train_hidden_state_lstm_areadelta.py`
- 둘 다 `--data docs/v5/bbox_nav_exp46/bbox_dataset_full_pg2.json --use_hidden_state none`만 테스트 (이번 ablation 핵심 질문은 "hidden state 없이도 area_delta가 회전 직전 근접 신호를 복원하는가" — CH43/CH46과 같은 6configs 전부 반복은 범위 밖, none만으로 1차 검증)

## 2. 비교 기준

| 구성 | CH46 PG2(절대값만) | 이번(절대값+delta) |
|---|---|---|
| MLP none val_acc | 93.90% | 측정 |
| LSTM none val_acc | 94.88% | 측정 |
| LSTM none closed-loop FPE | 0.145m | 측정 |

closed-loop 재평가는 LSTM-none만 (`closed_loop_eval_lstm_pg2.py`의 D_IN 변경 버전 복제).

## 3. 문서화

CH46 뒤에 CH47("area 변화율로 회전 직전 근접 신호 복원 시도")로 추가, 같은 타임스탬프 표 형식 유지.

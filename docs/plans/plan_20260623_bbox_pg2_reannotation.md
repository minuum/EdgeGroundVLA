# Plan — bbox_dataset_full.json을 현재 PG2 모델로 재주석 + CH43 재학습

> 작성: 2026-06-23 · 상태: **리서치 완료 — 승인 대기**
> 동기: CH45-2("그라운딩 실패 패턴") 진단의 has_bbox=False 5건을 현재 운영 PG2Grounder로 재실행하니 5건 전부 정상 탐지됨. 추적 결과 `bbox_dataset_full.json`은 2026-05-08(커밋 77683562)에 **Kosmos-2 기반 Tier1/Tier3 그라운딩**으로 생성된 데이터이고, 현재 운영 모델(PaliGemma2, exp59~)은 그 이후 도입됨 — CH43의 모든 MLP/LSTM hidden-state 실험이 이 낡은 bbox 라벨로 학습/평가되고 있었음.

---

## 0. 리서치 결과

| 항목 | 사실 |
|---|---|
| 현재 데이터 출처 | `docs/v5/bbox_nav_exp46/bbox_dataset_full.json` — 150 ep, 2,626 frames. git 추적상 2026-05-08 생성, 이후 한 번도 갱신 안 됨 |
| 생성 당시 모델 | Kosmos-2 Tier1(LoRA)/Tier3(coarse direction clf) — `scripts/finetune_kosmos2_grounding.py`, `scripts/train_coarse_direction_clf.py` (2026-05-06 커밋) |
| 현재 운영 모델 | PaliGemma2 (`PG2Grounder`, `stage2_v2_inference_server.py:336`) — `detect gray basket` 프롬프트, `resize_for_vlm`(224×224) 적용, 필터 `min_area=0.01, min_cy=0.35, area>0.9, x-full-width` |
| 재현성 확인 | CH45-2의 has_bbox=False 5프레임을 PG2Grounder.run()으로 재실행 → **5/5 has_bbox=True** (해상도 224/1280 둘 다 동일하게 탐지 성공) |
| 영향 범위 | `grounding_quality_vs_error.py`, `train_hidden_state_action.py`, `train_hidden_state_projected.py`, `train_hidden_state_lstm.py` 전부 `DATA_PATH`로 이 파일을 기본 사용 — **CH43 전체(LSTM/MLP, none/add/replace, 5-seed 포함)가 낡은 bbox 라벨 기반** |
| 스크립트 변경 필요 여부 | **불필요** — 3개 학습 스크립트 모두 이미 `--data` 인자로 경로 override 가능 (`train_hidden_state_action.py:357`, `_lstm.py:313`, `_projected.py:335`) |
| 1프레임 PG2 grounding 속도 | 실측 ~1.0s/frame → 2,626프레임 전체 재주석 시 **약 44분** (GPU, batch 없이 순차) |

---

## 1. 재주석 스크립트 (신규)

`scripts/eval/reannotate_bbox_pg2.py`

- 입력: 기존 `bbox_dataset_full.json`의 150 episode 구조(`{path_type, episode, frames:[{frame_idx, gt_class}, ...]}`)를 그대로 순회 — **episode/frame_idx 매핑은 유지**, cx/cy/area/has_bbox만 새로 채움.
- 각 프레임: 해당 h5의 `frame_idx` 이미지를 운영 코드와 동일한 `PG2Grounder.run(img, phrase="gray basket")` (resize_for_vlm + 운영 필터 4종)으로 재추론.
- 출력: **새 파일** `docs/v5/bbox_nav_exp46/bbox_dataset_full_pg2.json` — 기존 파일은 보존(히스토리/재현용으로 그대로 둠, 삭제하지 않음).
- 진행 중 sanity 로그: has_bbox 비율, area 분포(p25/median/p75)를 기존 파일과 비교 출력.

```python
# 핵심 루프 (의사코드)
for ep in old_data:
    h5_path = Path(ep["episode"])
    with h5py.File(h5_path) as f:
        imgs = f["observations"]["images"]
        for fr in ep["frames"]:
            img = imgs[fr["frame_idx"]]
            bbox = grounder.run(np.asarray(img), phrase="gray basket")
            fr["cx"], fr["cy"], fr["area"], fr["has_bbox"] = bbox["cx"], bbox["cy"], bbox["area"], bbox["has_bbox"]
```

---

## 2. 비교 + 재학습

같은 seed=42 split, 같은 optimizer/epoch — **데이터만 다른** 순수 ablation으로 비교:

| 모델 | 기존 데이터(헤드라인) | 신규 PG2 데이터 |
|---|---|---|
| MLP none (baseline) | 89.76% (CH40-1b) | 재측정 |
| LSTM none | 95.87% (CH43-2b) | 재측정 |
| LSTM add (5-seed) | 95.39%±0.20%p (CH43-2d) | 재측정 — **변화 크면 5-seed로 재확인** (CH43-2d와 같은 실수 반복 방지) |

명령 예시(스크립트 수정 없이 `--data`만 교체):
```bash
.venv/bin/python3 scripts/train_hidden_state_lstm.py \
  --data docs/v5/bbox_nav_exp46/bbox_dataset_full_pg2.json \
  --use_hidden_state none
```

---

## 3. closed-loop 재평가 + CH45-2 재진단

- `scripts/eval/closed_loop_eval_lstm.py` 동일 val 29 episodes로 새 ckpt 재평가.
- `scripts/eval/grounding_quality_vs_error.py`의 `DATA_PATH`를 새 파일로 바꿔 재실행 → has_bbox=False 비율이 실제로 줄어드는지(혹은 사라지는지) 확인. **이게 CH45-2가 다시 유효한 결론을 낼지의 직접 검증.**

---

## 4. 문서화

- CH45-2 옆에 **삭제하지 않고 정정 카드 추가**(CH45-2b): "위 진단은 2026-05-08 Kosmos-2 시절 데이터 기준 — 현재 PG2 모델로 재현 안 됨" + 재주석 후 실제 수치.
- 새 챕터(CH46): 재주석 절차 + Before/After 표(헤드라인 정확도, closed-loop SR, has_bbox 비율) + 결론(데이터 신선도가 모델 구조보다 더 큰 factor였는지 여부).
- TODO 섹션(`todo-ch44`)의 관련 항목 상태 갱신.

---

## 5. 리스크/주의

- 44분짜리 GPU job — soda 운영 서버와 무관(로컬 GB10에서 오프라인 실행), 운영 충돌 없음.
- 재학습은 6configs(none/add/replace × MLP/LSTM) 각각 300 epoch — 기존 CH43 실험과 동일 규모, 추가 시간 크지 않음(개별 학습은 분 단위).
- 기존 `bbox_dataset_full.json`은 절대 덮어쓰지 않음 — 새 파일명으로만 저장, 비교 기준선 보존.

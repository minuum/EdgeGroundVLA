# CH54 프리뷰 정렬 루프 — PG2 → HSV 룰 교체 계획

> 작성일: 2026-07-03
> 상태: 리서치 완료, 사용자 승인 대기 (코드 미착수)
> 배경: `grounding_decisions_20260703.jsonl` 분석 결과, 8~26초대 PG2 지연/환각(full-frame, 세미콜론 중복)이
> 전부 CH54 "첫 프레임 정렬 재시도 루프"(`_ground_multi` 호출)에서 발생. 이 루프는 정밀한 방향 인식이 아니라
> "그냥 basket이 대충 어느 쪽에 있나"만 알면 되는 coarse 판단이라, PG2 대신 이미 검증된 HSV 룰
> (`scripts/extract_basket_cx_frame_level.py::detect_basket_cx`)로 대체하자는 제안.

---

## 1. 현재 구조 (리서치)

`robovlm_nav/serve/stage2_v2_inference_server.py`:

- `predict()` (826행 부근): `inference_count == 0`이고 `_preview_attempt < max_retry`일 때만 진입
  ```python
  first_bbox = self._ground_multi(image_rgb, phrase)   # PG2, 최대 4개 프롬프트 순차 재시도
  if self._needs_preview(first_bbox):                  # has_bbox=False
      preview_rot = self._preview_rot_from_bbox(first_bbox)
      ...
  else:
      self._grounding_cache = first_bbox                # 성공 시 캐시 → 이후 정상 추론 첫 스텝에 재사용
  ```
- `_ground_multi`: PG2 1차 phrase 실패 시 `_fallback_prompts`(예: "laundry basket", "gray bin" 등)로 최대 4회 순차 재시도 → 이게 jsonl에서 본 phrase 다양성의 정체
- `_needs_preview`: `has_bbox=False`일 때만 True
- `_preview_rot_from_bbox`: `has_bbox=True`면 cx<0.4→ROT_L, else ROT_R. `has_bbox=False`면 hint_cx 또는 고정 스윕 방향
- `preview_align()` (외부 `/preview_align` 엔드포인트용): 위와 동일 로직 별도 진입점

이미 `docs/plans/plan_ch54_yolo_preview.md`(2026-06-26)에서 "YOLO로 이 루프 대체" 아이디어가 나왔으나
"basket zero-shot 안 되면 color-based fallback 검토"라고 적어두고 미착수 상태였음 — 이번 제안이 그 색상 기반 대안.

`scripts/extract_basket_cx_frame_level.py::detect_basket_cx(img_rgb)`:
- HSV 회색 마스크(S/V 임계값) → 천장/바닥 컷 → connected components → 최대 블롭(복도 배경 오탐 시 2위 사용)
- 반환: `(cx, cy, area_ratio, confidence)` 또는 `None`
- 이미 V5 구조화 데이터 라벨링(`cx_det_hsv`)에 실사용 중 — 새 코드 아님, 검증된 룰
- 순수 OpenCV, GPU/VLM 호출 없음 → 지연시간 수 ms 수준 (PG2 대비 1000배 이상 빠름)

---

## 2. 제안: 프리뷰 루프만 HSV로 교체 (메인 그라운딩은 PG2 유지)

**범위를 프리뷰(콜드스타트 정렬)로 한정하는 이유**: [[project_focus_grounding_for_direction]]에 따라
실주행 방향판단 품질의 1차 원인은 그라운딩 정확도 — 학습된 액션 헤드는 PG2 bbox 분포로 학습됨.
HSV 룰을 메인 grounding까지 대체하면 헤드 입력 분포가 바뀌어 회귀 위험이 큼.
반면 프리뷰 루프는 "정렬 필요 여부 + 대략 방향"만 판단하는 게이트라서 정밀도 요구가 낮고, HSV로도 충분.

### 2-1. 변경 파일: `robovlm_nav/serve/stage2_v2_inference_server.py`

1. `detect_basket_cx`를 서버 모듈에서 import (scripts 경로 의존 제거 위해 함수를 `robovlm_nav/` 하위로 이동하거나
   `sys.path`에 `scripts/` 추가 — 택1, 사용자 의견 필요)
2. 신규 메서드 `_ground_hsv(image_rgb) -> dict`:
   ```python
   def _ground_hsv(self, image_rgb: np.ndarray) -> dict:
       det = detect_basket_cx(image_rgb)
       if det is None:
           return {"has_bbox": False, "cx": 0.5, "cy": 0.6, "area": 0.0}
       cx, cy, area, conf = det
       return {"has_bbox": True, "cx": cx, "cy": cy, "area": area, "hsv_confidence": conf}
   ```
   (`_needs_preview`/`_preview_rot_from_bbox`는 `has_bbox`/`cx` 키만 보므로 **그대로 재사용 가능**, 수정 불필요)
3. `predict()`의 프리뷰 분기에서 `self._ground_multi(image_rgb, phrase)` → `self._ground_hsv(image_rgb)` 로 교체
4. **중요**: 프리뷰가 "정렬 완료"로 판단해 정상 추론으로 넘어갈 때, `self._grounding_cache = first_bbox`에
   HSV 결과를 그대로 넣지 않는다. 정상 추론 첫 스텝은 기존처럼 PG2를 호출해 헤드 입력 분포를 유지한다.
   → `_grounding_cache` 캐싱 라인은 HSV 분기에서 제거하고, 다음 정상 predict() 호출에서 PG2가 정상적으로
   1회 실행되도록 그대로 둔다 (즉 프리뷰 성공 시에도 캐시 스킵, PG2는 다음 스텝에 자연히 실행됨).
5. `preview_align()` 외부 엔드포인트도 동일하게 `_ground_hsv`로 교체 (일관성)

### 2-2. 환경변수 (선택, 롤백용)

```bash
export VLA_PREVIEW_GROUNDER=hsv   # 기본값 "pg2" 유지 시 기존 동작, "hsv" 설정 시 이 경로 활성
```
`__init__`에서 `self._preview_grounder = os.getenv("VLA_PREVIEW_GROUNDER", "pg2")` 읽어서 분기.

---

## 3. 트레이드오프

| 항목 | PG2 (현재) | HSV 룰 (제안) |
|---|---|---|
| 지연시간 | 1.4~26s (flicker/환각 시 최악) | 수 ms |
| 안정성 | 세미콜론 중복·full-frame 환각 있음 (07-03 jsonl 확인) | 결정론적, 환각 없음 |
| 일반화 | 임의 phrase(자연어) 가능 | **회색 basket 색상에 튜닝됨** — 다른 색 물체/phrase엔 무효 |
| 복도 배경 오탐 | 없음(별도 케이스) | BG_RATIO 휴리스틱으로 대응하나 완벽하지 않음 |
| 유지보수 | VLM 재현성 이슈(버전/HW差) | 조명 조건 바뀌면 임계값 재튜닝 필요 |

**리스크**: instruction이 "gray basket"이 아닌 다른 phrase일 때 HSV 룰이 무의미해짐.
현재 프리뷰 루프는 콜드스타트 상황에서만 쓰이고 기본 phrase가 "gray basket"이므로 실사용 범위 내에선 괜찮으나,
다른 색 물체 실험(빨간 공 등) 진행 시 이 프리뷰만 별도로 PG2로 되돌리는 스위치(`VLA_PREVIEW_GROUNDER=pg2`)가 필요.

---

## 4. 검증 계획

1. 로컬: 06-27/07-02/07-03 수신 세션의 frame 0~2에 대해 `detect_basket_cx` vs PG2 첫 프레임 판정 일치율 확인
2. soda: `VLA_PREVIEW_GROUNDER=hsv`로 obj_left/center/right 재주행 → 프리뷰 단계 latency 및 정렬 성공률 비교
3. 롤백: env var 미설정 시 기존 PG2 경로 완전 유지

---

## 5. 완료 기준 (DoD)

- [ ] `detect_basket_cx` 서버에서 import 가능하도록 위치 정리 (scripts 의존 여부 결정)
- [ ] `_ground_hsv()` 추가, 프리뷰 분기에서 교체
- [ ] `_grounding_cache` HSV 오염 방지 확인 (정상 추론 첫 스텝은 항상 PG2)
- [ ] `VLA_PREVIEW_GROUNDER` 환경변수로 롤백 가능
- [ ] soda A/B 테스트: 프리뷰 latency, 정렬 성공률, 세션 전체 SR 비교

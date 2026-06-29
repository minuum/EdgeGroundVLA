# CH54 — YOLO 프리뷰 모델 구현 계획

> **작성일:** 2026-06-26  
> **상태:** 아이디어 확정, 구현 미착수  
> **배경:** 6/26 미팅 — 시작 위치 좌우 오프셋 시 PG2 그라운딩 실패 → 수동 각도 조정 의존 문제 해결

---

## 1. 목표

첫 프레임 PG2 그라운딩 실패(has_bbox=False 또는 area < 임계값) 시,
YOLO 탐지로 타겟 방향을 추정해 로봇을 자동 회전시키고 PG2 재시도.
성공 후 정상 항법(Exp66 ActionMLP)으로 진입.

**기대 효과:** 측면 시작 SR 70% → 90% (오늘 수동 각도 조정 시 실측치)

---

## 2. 트리거 조건

```python
AREA_THRESHOLD = float(os.getenv("VLA_PREVIEW_AREA_THRESH", "0.03"))

def needs_preview(bbox: tuple) -> bool:
    cx, cy, area, has_bbox = bbox
    return (has_bbox < 0.5) or (area < AREA_THRESHOLD)
```

---

## 3. 실행 플로우

```
[시작] 로봇 출발 위치 고정
  ↓
① PG2 그라운딩 시도
  ├─ 성공 (has_bbox=True && area ≥ 0.03) → Exp66 정상 항법
  └─ 실패
       ↓
② YOLO 탐지 (YOLOv8n)
   ├─ bbox.cx < 0.4 → ROT_L × N_ROT 스텝
   ├─ bbox.cx > 0.6 → ROT_R × N_ROT 스텝
   ├─ 0.4 ≤ cx ≤ 0.6 → PG2 재시도 (①로)
   └─ 탐지 실패 → ROT_R 기본 방향 소량 회전 후 재시도
       ↓ (최대 MAX_RETRY=5회)
   성공 → 정상 항법 / 실패 → STOP
```

---

## 4. 수정 파일

### 4-1. `robovlm_nav/serve/stage2_v2_inference_server.py`

`InferenceServer.__init__`에 추가:
```python
# CH54: YOLO 프리뷰 모델 (VLA_PREVIEW_MODEL 설정 시 활성)
_preview_model_name = os.getenv("VLA_PREVIEW_MODEL", "")
if _preview_model_name:
    from ultralytics import YOLO
    self._yolo = YOLO(_preview_model_name)  # e.g. "yolov8n.pt"
    self._preview_target_classes = [
        int(c) for c in os.getenv("VLA_PREVIEW_CLASSES", "56").split(",")
    ]  # COCO 56=chair, basket은 없으므로 chair 시나리오 우선
    self._preview_area_thresh = float(os.getenv("VLA_PREVIEW_AREA_THRESH", "0.03"))
    self._preview_max_retry   = int(os.getenv("VLA_PREVIEW_MAX_RETRY", "5"))
    self._preview_n_rot       = int(os.getenv("VLA_PREVIEW_N_ROT", "2"))
    log.info(f"[CH54] Preview model 활성: {_preview_model_name}")
else:
    self._yolo = None
```

새 메서드 `_preview_align(frame: np.ndarray) -> list[int]` 추가:
```python
def _preview_align(self, frame: np.ndarray) -> list[int]:
    """
    YOLO로 타겟 방향 탐지 후 ROT 명령 시퀀스 반환.
    반환값: [ROT_L | ROT_R | 0(정렬됨)] 리스트
    """
    results = self._yolo(frame, classes=self._preview_target_classes,
                          conf=0.4, verbose=False)
    if not results or len(results[0].boxes) == 0:
        # 탐지 실패 → 기본 ROT_R
        return [7] * self._preview_n_rot  # ROT_R
    
    box = results[0].boxes[0]
    cx_norm = float((box.xyxy[0][0] + box.xyxy[0][2]) / 2) / frame.shape[1]
    
    if cx_norm < 0.4:
        return [6] * self._preview_n_rot  # ROT_L
    elif cx_norm > 0.6:
        return [7] * self._preview_n_rot  # ROT_R
    else:
        return []  # 정렬 완료
```

`predict()` 또는 해당 추론 진입점에 프리뷰 루프 삽입:
```python
# CH54: 첫 프레임 그라운딩 실패 시 프리뷰 정렬
if self._yolo is not None and self._frame_count == 0:
    bbox = self._grounding_hub.get_bbox(frame_rgb)
    if needs_preview(bbox):
        for _ in range(self._preview_max_retry):
            rot_cmds = self._preview_align(frame_np)
            if not rot_cmds:
                break  # 정렬 완료, 정상 항법
            for cmd in rot_cmds:
                yield cmd  # ROT_L/ROT_R 전송
            frame_np, frame_rgb = self._capture_frame()
            bbox = self._grounding_hub.get_bbox(frame_rgb)
            if not needs_preview(bbox):
                break
```

### 4-2. 신규 파일 (선택): `robovlm_nav/serve/preview_model.py`

YOLO 로직 분리 시 생성. 규모 작으면 inference_server.py에 인라인으로 통합.

---

## 5. 의존성

```bash
# soda 서버에서
pip install ultralytics  # YOLOv8n 포함
# 또는 이미 설치됐으면 확인:
python3 -c "from ultralytics import YOLO; print('ok')"
```

---

## 6. 환경변수 (soda 배포 시)

```bash
# .env 또는 서버 실행 시 추가
export VLA_PREVIEW_MODEL=yolov8n.pt       # 활성화 (비우면 비활성)
export VLA_PREVIEW_CLASSES=56             # COCO chair=56, 복수 시 "56,39"
export VLA_PREVIEW_AREA_THRESH=0.03       # 그라운딩 실패 판정 임계값
export VLA_PREVIEW_MAX_RETRY=5            # 최대 재시도 횟수
export VLA_PREVIEW_N_ROT=2               # 회전 스텝 수 (1스텝 ≈ 5~10°)
```

---

## 7. 검증 계획

1. **단위 테스트**: YOLO 탐지 결과 → ROT 방향 매핑 함수 (`_preview_align`) 오프라인 테스트
2. **시뮬 테스트**: 측면 시작 에피소드에서 프리뷰 루프 동작 확인
3. **실로봇 A/B**: 프리뷰 ON/OFF 각 10회 → SR 비교 (기대: 70%→90%)
4. **basket 시나리오**: YOLO basket zero-shot 안 되면 custom class 파인튜닝 또는 color-based fallback 검토

---

## 8. 롤백 전략

`VLA_PREVIEW_MODEL` 미설정 시 기존 동작 완전 유지.
별도 브랜치 `feat/ch54-preview-model`에서 개발 후 PR.

---

## 9. 완료 기준 (DoD)

- [ ] `_preview_align()` 구현 + 단위 테스트
- [ ] inference_server.py 통합
- [ ] 실로봇 측면 시작 SR ≥ 85% 확인
- [ ] rollback 검증 (VLA_PREVIEW_MODEL 미설정 시 기존 동작 동일)

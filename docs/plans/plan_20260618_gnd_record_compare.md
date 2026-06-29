# Plan: 그라운딩 녹화 + 서버 비교

**작성일**: 2026-06-18  
**브랜치**: monavla-driving  
**상태**: 📋 검토 대기

---

## 목표

1. **녹화**: 그라운딩 탭에서 bbox 오버레이 영상을 MP4로 저장  
2. **서버 비교**: `/ground` (PG2 bbox) + `/predict` (MLP 액션) 동시 호출 → 결과 나란히 표시  
   - "PG2가 cx=0.4로 왼쪽 치우침 → 서버는 FWD+LEFT 예측?" 확인
   - 이동 없이 순수 예측만 (execute_move=False)

---

## 변경 내용

### 1. 녹화 기능

**버튼 추가** (기존 버튼 Row 오른쪽):
```
▶ 단발 검증 | 🔄 자동 (1fps) | ⏹ 정지 | 🔴 녹화 시작
```

**내부 동작:**
```python
_gnd_video_writer: list = [None]   # cv2.VideoWriter 래퍼
_gnd_video_path:   list = [None]

def _gnd_start_record():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = _gnd_log_dir / f"gnd_{ts}.mp4"
    _gnd_video_path[0] = path
    _gnd_video_writer[0] = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (W, H)
    )
    return gr.update(value="⏹ 녹화 중지", variant="stop"), str(path)

def _gnd_stop_record():
    if _gnd_video_writer[0]:
        _gnd_video_writer[0].release()
        _gnd_video_writer[0] = None
    return gr.update(value="🔴 녹화 시작", variant="secondary"), "저장 완료"
```

**`_run_grounding()` 끝에 프레임 추가:**
```python
# 녹화 중이면 현재 프레임 추가
if _gnd_video_writer[0] is not None:
    bgr = cv2.cvtColor(np.array(img_draw), cv2.COLOR_RGB2BGR)
    _gnd_video_writer[0].write(bgr)
```

**새 UI 컴포넌트:**
```python
gnd_rec_btn      = gr.Button("🔴 녹화 시작", variant="secondary", scale=1)
gnd_rec_display  = gr.Textbox(label="영상 저장", value="—", interactive=False, scale=3)
```

---

### 2. 서버 비교 (inference 병행)

**추가 입력** — `_run_grounding`에 `backend_mode`, `api_url`, `instr` 주입:

```python
def _run_grounding(api_url, backend_mode, instr, history_rows, count):
    ...
    # /ground 호출 (기존)
    gnd_result = _call_ground(api_url, frame)

    # /predict 호출 (execute_move=False — 이동 없음)
    pred_result = None
    try:
        pred_result = run_backend_inference(
            frame, instr, backend_mode, api_url, execute_move=False
        )
    except Exception:
        pass
    ...
```

**표시 방식** — `gnd_has_bbox` 박스에 PG2 + 서버 예측 합쳐서:
```
✅ 검출됨  area=0.142  cx=0.48
↳ 서버: FWD  (lat=348ms)
```

**JSONL 레코드에 필드 추가:**
```python
record = {
    ...,                           # 기존 필드
    "pred_label":  "FORWARD",      # 서버 예측 액션 레이블
    "pred_lat_ms": 348.2,          # 서버 예측 레이턴시
    "pred_goal_near": False,       # 서버 goal_near 여부
}
```

**inputs 변경:**
```python
# 기존
inputs=[api_url_box, _gnd_history_rows, _gnd_count]
# 변경
inputs=[api_url_box, backend_radio, instr_box_real, _gnd_history_rows, _gnd_count]
```

---

## 수정 파일

| 파일 | 변경 |
|---|---|
| `scripts/gradio_inference_dashboard.py` | 녹화 버튼/로직, 서버 비교 로직 |

추가 패키지 없음 — `cv2` 이미 의존성에 있음.

---

## 트레이드오프

| 항목 | 고려사항 |
|---|---|
| `/predict` 병행 호출 | 그라운딩 1회당 PG2(1.2s) + 서버(0.35s) → 순차 시 ~1.5s. 병렬 스레드로 분리하면 1.2s 유지 가능 |
| VideoWriter fps=1 | 자동 1fps와 맞춤. 단발 검증 시엔 프레임 1개 저장 |
| 서버 비교 선택 여부 | 서버 모델이 로드 안 된 상태면 비교 skip (예외 처리) |
| 영상 해상도 | 카메라 원본 크기(1398×720)로 저장. bbox 오버레이 포함 |

---

## 질문

1. `/predict` 병행 — 순차(구현 단순) vs 병렬 스레드(1fps 유지)?  
2. 서버 비교를 항상 켜놓을지, 별도 체크박스 on/off로 할지?  
3. 녹화 중 자동 자막(타임스탬프, cx, has_bbox)을 프레임에 넣을지?


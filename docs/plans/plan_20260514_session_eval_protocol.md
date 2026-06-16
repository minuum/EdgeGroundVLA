# Plan — Session Eval Protocol (VLM 기반 수집 품질 / 모델 오류 분리)

작성: 2026-05-14
브랜치: `monavla-driving`

## 0. 한 줄 요약

Pure HF Kosmos-2 grounding으로 H5 에피소드를 자동 평가하는 독립 Gradio 서버를 만들어,
**"수집 문제 (basket 비가시)"** vs **"모델 문제 (방향 불일치)"** 를 프레임 단위로 분리한다.

---

## 1. 목표

- **입력**: `ROS_action/mobile_vla_dataset_v5/*.h5` (기존 150ep + 신규 수집분)
- **평가**: 프레임마다 Kosmos-2 grounding → basket 위치(left/center/right) → 로그된 액션과 비교
- **출력**: 에피소드별 점수 + 프레임 뷰어 + 판정 (`collection_issue` / `model_issue` / `ok`)
- **형태**: 독립 Gradio 서버 (`scripts/gradio_session_eval.py`, 포트 7861)

---

## 2. 핵심 개념 — 3가지 판정 기준

### 2.1 grounding_success_rate (수집 품질)

Kosmos-2가 basket BBox를 찾은 프레임 비율.

- **< 50% → `collection_issue`**: basket이 프레임 밖이거나 조명/블러 문제
- **≥ 50%** → grounding 정상, 다음 기준으로 진행

### 2.2 action_agreement_rate (모델 품질)

Kosmos-2가 본 basket 방향과 로그된 액션의 일치율.

| Kosmos-2 basket 위치 | 기대 액션 |
|----------------------|----------|
| cx < 0.4 (left) | LEFT / FWD+L / ROT_L (class 2, 4, 6) |
| cx > 0.6 (right) | RIGHT / FWD+R / ROT_R (class 3, 5, 7) |
| 0.4 ≤ cx ≤ 0.6 (center) | FORWARD (class 1) |
| bbox 없음 | STOP or FORWARD — 평가 제외 |

- grounding_success ≥ 50% 인데 **agreement < 50% → `model_issue`**
- 둘 다 ≥ 50% → **`ok`**

### 2.3 raw 액션 → 방향 매핑

H5에는 `[lx, ly, az]` 연속값이 저장됨. 아래 임계값으로 이산 방향으로 변환:

```python
def raw_action_to_direction(act: list[float]) -> str:
    lx, ly, az = act
    if abs(az) > 0.15:
        return "left" if az < 0 else "right"   # ROT_L=left, ROT_R=right
    if abs(ly) > 0.1:
        return "left" if ly < 0 else "right"   # strafe
    if lx > 0.05:
        return "center"                          # FORWARD
    return "none"                                # STOP or near-zero
```

> 임계값 0.15 / 0.1 / 0.05는 V5 수집 로봇의 실제 cmd_vel 스케일 기반.
> UI 슬라이더로 조정 가능하게 노출.

---

## 3. 아키텍처

```
scripts/gradio_session_eval.py
├── KosmosEvaluator                 # 평가 엔진
│   ├── __init__                    # GroundingBackend 재사용
│   ├── eval_episode(h5_path)       # → EpisodeReport
│   └── eval_batch(h5_paths, cb)    # 배치 평가 (progress callback)
├── EpisodeReport (dataclass)       # 에피소드별 집계
│   ├── path_type, n_frames
│   ├── grounding_success_rate
│   ├── action_agreement_rate
│   ├── verdict: str
│   └── frames: list[FrameReport]
├── FrameReport (dataclass)         # 프레임별 원본 데이터
│   ├── img_rgb (np.ndarray)
│   ├── action_raw [lx, ly, az]
│   ├── action_dir ("left"/"center"/"right"/"none")
│   ├── bbox (dict | None)
│   ├── grounding_dir ("left"/"center"/"right" | None)
│   └── agree (bool | None)
└── Gradio UI (포트 7861)
    ├── [좌] 에피소드 선택 패널
    │   ├── 스캔 디렉터리 입력 + 스캔 버튼
    │   ├── 에피소드 목록 (Dropdown)
    │   ├── 평가 시작 버튼 + 진행률 표시
    │   └── 임계값 슬라이더 (선택)
    └── [우] 결과 패널
        ├── 요약 테이블 (에피소드별 점수 + 판정)
        ├── 프레임 뷰어 (슬라이더, 어노테이션 오버레이)
        └── JSON 내보내기 버튼
```

---

## 4. 코드 설계

### 4.1 KosmosEvaluator — GroundingBackend 재사용

`proxy_inference_server.py`의 `GroundingBackend`를 직접 import:

```python
import sys
sys.path.insert(0, str(ROOT / "robovlm_nav" / "serve"))
from proxy_inference_server import GroundingBackend

class KosmosEvaluator:
    def __init__(self, model_path: Path, device: str = "cuda"):
        self.grounding = GroundingBackend(model_path, device)

    def eval_episode(self, h5_path: Path) -> EpisodeReport:
        with h5py.File(h5_path, "r") as f:
            # V5/V4 자동 감지
            if "observations" in f and "images" in f["observations"]:
                images = f["observations"]["images"][:]   # (N, H, W, 3) uint8
            else:
                images = f["images"][:]
            actions = f["actions"][:]                      # (N, 3)
            instr = f["language_instruction"][0]
            instr = instr.decode() if isinstance(instr, bytes) else str(instr)

        frames = []
        for t, (img, act) in enumerate(zip(images, actions)):
            g = self.grounding.run(img)                   # img: uint8 RGB
            bbox = g.get("bbox")
            grounding_dir = bbox_to_direction(bbox) if bbox else None
            action_dir = raw_action_to_direction(act.tolist())
            agree = (grounding_dir == action_dir) if (grounding_dir and action_dir != "none") else None

            frames.append(FrameReport(
                img_rgb=img, action_raw=act.tolist(),
                action_dir=action_dir, bbox=bbox,
                grounding_dir=grounding_dir, agree=agree,
            ))

        return build_report(h5_path, frames)
```

### 4.2 bbox_to_direction

```python
def bbox_to_direction(bbox: dict) -> str:
    cx = bbox["cx"]
    if cx < 0.4:
        return "left"
    elif cx > 0.6:
        return "right"
    else:
        return "center"
```

### 4.3 EpisodeReport 집계

```python
@dataclass
class EpisodeReport:
    episode_id: str
    path_type: str
    n_frames: int
    grounding_success_rate: float   # bbox 찾은 프레임 / 전체
    action_agreement_rate: float    # agree==True / grounding 성공 프레임
    verdict: str                    # "collection_issue" | "model_issue" | "ok"
    frames: list[FrameReport]

def build_report(h5_path: Path, frames: list[FrameReport]) -> EpisodeReport:
    n = len(frames)
    grounded = [f for f in frames if f.grounding_dir is not None]
    agreed   = [f for f in grounded if f.agree is True]

    gsr = len(grounded) / n if n > 0 else 0.0
    aar = len(agreed) / len(grounded) if grounded else 0.0

    if gsr < 0.5:
        verdict = "collection_issue"
    elif aar < 0.5:
        verdict = "model_issue"
    else:
        verdict = "ok"

    # path_type은 파일명 파싱 또는 attrs에서 추출
    attrs = extract_attrs(h5_path)

    return EpisodeReport(
        episode_id=h5_path.stem,
        path_type=attrs.get("pattern", "unknown"),
        n_frames=n,
        grounding_success_rate=round(gsr, 3),
        action_agreement_rate=round(aar, 3),
        verdict=verdict,
        frames=frames,
    )
```

### 4.4 Gradio UI 핵심

```python
import gradio as gr

with gr.Blocks(title="Session Eval") as demo:
    state = gr.State({"reports": [], "selected": None})

    with gr.Row():
        # ── 좌측: 선택 패널 ──
        with gr.Column(scale=1):
            scan_dir = gr.Textbox(
                value="ROS_action/mobile_vla_dataset_v5",
                label="Dataset Directory"
            )
            btn_scan = gr.Button("📁 Scan Episodes")
            ep_dropdown = gr.Dropdown(label="Select Episode(s)", multiselect=True)
            btn_eval   = gr.Button("▶️ Run Evaluation", variant="primary")
            progress   = gr.Textbox(label="Progress", interactive=False)

        # ── 우측: 결과 패널 ──
        with gr.Column(scale=2):
            summary_table = gr.Dataframe(
                headers=["episode", "path_type", "n_frames", "gsr", "aar", "verdict"],
                label="Episode Summary",
            )
            frame_slider = gr.Slider(minimum=0, maximum=0, step=1, label="Frame")
            frame_img    = gr.Image(label="Frame + BBox Overlay", interactive=False)
            frame_info   = gr.Textbox(label="Frame Info", interactive=False)
            btn_export   = gr.Button("💾 Export JSON")

    # 콜백 연결
    btn_scan.click(fn=scan_episodes, inputs=[scan_dir], outputs=[ep_dropdown])
    btn_eval.click(fn=run_eval, inputs=[ep_dropdown, state], outputs=[summary_table, progress, state])
    summary_table.select(fn=on_episode_select, inputs=[state], outputs=[frame_slider, frame_img, frame_info])
    frame_slider.change(fn=on_frame_change, inputs=[frame_slider, state], outputs=[frame_img, frame_info])
    btn_export.click(fn=export_json, inputs=[state], outputs=[gr.File()])
```

### 4.5 프레임 어노테이션 오버레이

```python
def annotate_frame(frame: FrameReport) -> np.ndarray:
    img = frame.img_rgb.copy()
    h, w = img.shape[:2]

    # BBox 그리기
    if frame.bbox:
        x1 = int(frame.bbox["x1"] * w)
        y1 = int(frame.bbox["y1"] * h)
        x2 = int(frame.bbox["x2"] * w)
        y2 = int(frame.bbox["y2"] * h)
        color = (0, 255, 0) if frame.agree else (255, 0, 0)   # 녹색=일치, 적색=불일치
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"G:{frame.grounding_dir} A:{frame.action_dir}"
        cv2.putText(img, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 액션 방향 화살표
    cx, cy = w // 2, h // 2
    arrow_map = {
        "left":   (-60, 0),
        "center": (0, -60),
        "right":  (60, 0),
        "none":   (0, 0),
    }
    dx, dy = arrow_map.get(frame.action_dir, (0, 0))
    if dx or dy:
        cv2.arrowedLine(img, (cx, cy), (cx + dx, cy + dy), (255, 255, 0), 3, tipLength=0.3)

    return img
```

---

## 5. 파일 구성

| 파일 | 변경 | 비고 |
|------|------|------|
| `scripts/gradio_session_eval.py` | **신규 ✅ 완료** | 독립 Gradio 서버 (포트 7861) |
| `robovlm_nav/serve/proxy_inference_server.py` | **변경 없음** | `GroundingBackend` 재사용만 |

새 파일만 추가, 기존 코드 수정 없음.

---

## 6. 실행 방법

```bash
# 독립 서버 실행
python3 scripts/gradio_session_eval.py

# 접속: http://localhost:7861
# 또는 --share 플래그로 외부 공유
python3 scripts/gradio_session_eval.py --share
```

---

## 7. 검증 계획

| 체크 | 기준 |
|------|------|
| H5 로드 | 150ep 전부 오류 없이 읽힘 |
| Grounding 실행 | 프레임당 평균 < 1초 (GPU 기준) |
| BBox 오버레이 | 시각적으로 basket 위치에 box 표시됨 |
| 판정 타당성 | `center_straight` ep → `ok` or `model_issue` (basket은 항상 정면) |
| JSON 내보내기 | `eval_results_YYMMDD_HHMMSS.json` 정상 저장 |

---

## 8. 트레이드오프

| 장점 | 단점 |
|------|------|
| 기존 GroundingBackend 재사용 → 구현 빠름 | Kosmos-2 grounding IoU 0.679 — 완벽하지 않아 false alarm 가능 |
| 수집/모델 문제 자동 분리 | action threshold (0.15/0.1) 는 수동 튜닝 필요 |
| 기존 코드 수정 없음 (신규 파일만) | GPU 메모리 GroundingBackend + 기존 서버와 충돌 가능 (별도 프로세스 권장) |
| 에피소드 단위 + 프레임 단위 양쪽 확인 가능 | 처리 속도: 1ep(~18프레임) × 150ep = ~45분 full scan |

---

## 9. 미결 결정 사항 (사용자 확인 필요)

1. **포트 충돌**: 기존 대시보드(7860)와 분리해 7861로 띄울지, 아니면 탭으로 통합할지
2. **샘플링 모드**: 전체 프레임 평가 vs 에피소드당 N프레임(예: 5프레임) 샘플링
3. **action threshold 기본값**: `az > 0.15` 기준이 실제 로봇 cmd_vel과 맞는지 확인 필요
4. **결과 저장 경로**: `docs/v5/eval/` 아래 저장할지 별도 디렉터리로 할지

---

## 10. CLAUDE.md 준수

이 plan은 사용자 검토/주석 → 승인 후에만 §5 파일을 생성한다. **승인 전 코드 0줄.**

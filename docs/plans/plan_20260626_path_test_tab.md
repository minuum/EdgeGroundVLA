# Plan — Gradio 추론 대시보드 4번째 탭: "🧪 경로 검증 (Path Test)"

> 작성: 2026-06-26 · 상태: **리서치 완료, 승인 대기**

---

## 1. 목적

`REAL_ROBOT_CHECKLIST_20260616.md`의 11-episode 테스트 매트릭스(center_straight 3 /
left_diagonal 3 / right_diagonal 3 / center_curve 2)를 실로봇 테스트 중 탭 전환 없이
한 화면에서 진행 + 즉시 기록할 수 있게 한다.

---

## 2. 리서치 결과 — 왜 "탭 1 통째 복사"가 아닌가

`scripts/gradio_inference_dashboard.py`의 탭 1(`🤖 Drive / Inference`, 1493~1820행)은:

- 카메라/상태/액션 등 모든 표시가 **단일 전역 `state` dict** + **단일 `gr.Timer(0.5)`**
  (`timer.tick(fn=update_ui, ...)`, 2496~2497행)에 묶여 있음
- `update_ui()`(1242행~)는 분기마다 9-튜플을 반환하는 긴 함수 — 출력 컴포넌트를
  통째로 복제하면 이 함수의 모든 `return` 지점을 같이 고쳐야 해서 회귀 위험이 큼
- `set_running` / `return_to_start` / `reset_model_wrapper`는 이미 전역 함수라
  **탭 1 컴포넌트값을 입력으로 그대로 재사용 가능** — 새로 안 만들어도 됨

**결론**: 탭 4는 탭 1의 핵심 로직을 그대로 호출하되, 출력 표시만 가볍게 미러링하고
경로검증 전용 UI(매트릭스 표 + 기록 테이블)를 추가하는 방식으로 구현. 기존 탭 1
코드는 1줄도 수정하지 않음 (회귀 위험 0).

---

## 3. 설계

### 3-1. 라이브 카메라/상태 미러링 (새 타이머 호출 추가, 기존 `update_ui` 무수정)

```python
camera_output_test = gr.Image(label="Live Camera", interactive=False)
bbox_area_display_test = gr.Textbox(label="bbox area/cx", value="—", interactive=False)

timer.tick(fn=lambda: state.get("last_img"), outputs=camera_output_test)
timer.tick(fn=_get_bbox_area_display, outputs=bbox_area_display_test)
```

- `state["last_img"]`는 `update_ui()`가 이미 매 tick 갱신하는 값 — 추가 ROS 호출 없음
- `_get_bbox_area_display`는 기존 함수 재사용, `/recent` 호출 1회 추가(0.5s 주기, 무시 가능한 부하)

### 3-2. 시작/정지/복귀 — 탭 1 컴포넌트 입력 재사용

```python
btn_start_test = gr.Button("▶️ START", variant="primary")
btn_stop_test  = gr.Button("⏹️ STOP", variant="stop")
btn_return_test = gr.Button("🔄 복귀")
run_status_test = gr.Textbox(label="Run Status", value="Stopped", interactive=False)

btn_start_test.click(
    fn=lambda mode, url, instr, gt, cc: set_running(True, mode, url, instr, gt, apply_cc=cc),
    inputs=[backend_radio, api_url_box, instr_box_real, gt_object_box, toggle_cc],  # 탭1 컴포넌트 그대로
    outputs=run_status_test,
)
btn_stop_test.click(fn=lambda: set_running(False, "", "", ""), outputs=run_status_test)
btn_return_test.click(fn=return_to_start, outputs=run_status_test)
```

- `inputs`가 탭 1의 `instr_box_real`/`api_url_box` 등을 그대로 참조 — Gradio는 컴포넌트가
  어느 탭에 시각적으로 속했는지와 무관하게 이벤트 입력으로 참조 가능 (검증됨, 기존
  코드에서도 `stop_apply_btn.click`이 탭1 내부에서 동일 패턴 사용 중)
- 즉 탭 4에서 START를 눌러도 탭 1과 동일한 `instr_box_real` 값(현재 "the gray basket")을
  그대로 사용 — 별도 instruction 입력 불필요(이미 결정된 prompt 통일과 일치)

### 3-3. 테스트 매트릭스 표시 (정적 Markdown)

```python
gr.Markdown("""
### 📋 실로봇 11-Episode 테스트 매트릭스

| 경로 타입 | 목표 횟수 | 성공 기준 |
|---|---|---|
| center_straight | 3회 | TLD 0.7~1.5, FPE<0.5m, STOP 필수 |
| left_diagonal | 3회 | TLD 0.7~1.5, FPE<0.5m |
| right_diagonal | 3회 (★우선) | TLD 0.7~1.5, FPE<0.5m |
| center_curve | 2회 (옵션) | TLD 0.7~1.5, FPE<0.5m |

**목표: 7/11 (63.6%) 이상 성공**
""")
```

### 3-4. 에피소드 기록 테이블 + 진행률

```python
path_type_test = gr.Dropdown(
    choices=["center_straight", "left_diagonal", "right_diagonal", "center_curve"],
    value="right_diagonal",  # 우선순위 경로 기본값
    label="이번 에피소드 경로 타입",
)
fpe_test = gr.Number(label="FPE 추정(육안, m)", value=0.0)
success_test = gr.Radio(choices=["성공", "실패"], value="성공", label="결과")
note_test = gr.Textbox(label="특이사항", value="")
btn_log_episode = gr.Button("📝 에피소드 기록")

episode_log_table = gr.Dataframe(
    headers=["#", "경로타입", "결과", "STOP", "area", "cx", "FPE(m)", "특이사항"],
    datatype=["number", "str", "str", "str", "number", "number", "number", "str"],
    label="에피소드 기록 (누적)",
    row_count=11,
    interactive=False,
)
progress_test = gr.Textbox(label="진행률", value="0/11 (목표 7/11)", interactive=False)

_episode_log_state = gr.State([])  # 세션 내 누적 (페이지 새로고침 시 초기화 — 4-3 참고)

def log_episode(path_type, success, fpe, note, log_list):
    import requests as _req
    area, cx = 0.0, 0.5
    try:
        r = _req.get(f"{DEFAULT_API_URL}/recent", timeout=2)
        preds = r.json().get("predictions", [])
        if preds:
            bbox = preds[0].get("bbox", {})
            area, cx = bbox.get("area", 0.0), bbox.get("cx", 0.5)
    except Exception:
        pass
    stop_flag = "Y" if area >= 0.18 else "N"  # 현재 STOP threshold 기준 추정
    row = [len(log_list) + 1, path_type, success, stop_flag, round(area, 3), round(cx, 2), fpe, note]
    log_list = log_list + [row]

    targets = {"center_straight": 3, "left_diagonal": 3, "right_diagonal": 3, "center_curve": 2}
    done = {k: 0 for k in targets}
    success_count = 0
    for r in log_list:
        done[r[1]] = done.get(r[1], 0) + 1
        if r[2] == "성공":
            success_count += 1
    total = len(log_list)
    prog = f"{total}/11 (성공 {success_count}, 목표 7) | " + ", ".join(
        f"{k}:{done.get(k,0)}/{v}" for k, v in targets.items()
    )
    return log_list, log_list, prog

btn_log_episode.click(
    fn=log_episode,
    inputs=[path_type_test, success_test, fpe_test, note_test, _episode_log_state],
    outputs=[_episode_log_state, episode_log_table, progress_test],
)
```

### 3-5. 결과 저장 (CSV export)

```python
btn_export_test = gr.Button("💾 CSV로 저장")
export_status_test = gr.Textbox(label="", value="", interactive=False)

def export_episode_log(log_list):
    import csv, datetime as _dt
    path = PROJECT_ROOT / "logs" / f"realtest_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["#", "경로타입", "결과", "STOP", "area", "cx", "FPE(m)", "특이사항"])
        w.writerows(log_list)
    return f"✅ 저장: {path}"

btn_export_test.click(fn=export_episode_log, inputs=[_episode_log_state], outputs=export_status_test)
```

---

## 4. 트레이드오프 / 알려진 제약

1. **새로고침 시 기록 초기화**: `gr.State`는 브라우저 세션 단위라 페이지 새로고침하면
   누적 기록이 날아감. 테스트 중간에 저장 버튼(3-5)을 자주 눌러 CSV로 백업 권장.
   (영구 저장이 필요하면 추후 서버사이드 파일 append로 변경 가능 — 이번 스코프 밖)
2. **STOP 여부 추정**: `log_episode`의 `stop_flag`는 기록 시점의 `/recent` 최신 bbox
   area만 보고 추정 — 실제 STOP 발생 여부의 정확한 판단은 여전히 서버 로그
   (`PROXIMITY STOP` 그치만 `VLA_STOP_MODE=learned`라 해당 로그 자체가 안 찍힐 수 있음 —
   `learned` 모드는 모델이 class 0(STOP)을 예측하면 정지하므로 별도 텍스트 로그가 없을 수 있음)
   를 직접 보고 사용자가 기록표의 "특이사항"에 적어주는 게 더 정확함. UI 추정값은 참고용.
3. **기존 탭 1 코드 무수정** — 회귀 위험 없음. 추가 컴포넌트만 신설.

---

## 5. 영향 범위

- 파일: `scripts/gradio_inference_dashboard.py` 1개만 수정
  - 탭 추가 위치: 기존 탭 3(`📊 Latency/Drift 진단`, 2162행) 다음, `with gr.Tabs():` 블록 안
  - 이벤트 바인딩(`btn_start_test.click` 등)은 기존 이벤트 바인딩 구역(2430행 근처) 옆에 추가
- 재시작 필요: `go.sh --dashboard`
- 백엔드/서버(`stage2_v2_inference_server.py`) 무수정

---

## 6. 실행 순서

1. (승인 시) 탭 4 추가 — Markdown 매트릭스 + 카메라/상태 미러 + Start/Stop/복귀 + 기록 테이블 + CSV 저장
2. `go.sh --dashboard` 재시작
3. 브라우저에서 탭 4 열어 카메라 미러링 동작 확인, START 눌러 탭1과 동일하게 동작하는지 확인
4. 더미 기록 1건 입력 → 진행률/표 갱신 확인 → CSV 저장 확인

#!/usr/bin/env python3
"""
MoNaVLA CL Dashboard — Closed-Loop 평가 결과 + 실로봇 주행 로그
Port: 7867

탭 구성:
  1. CL Results    — rollout_metrics.json 파싱, 모델별 성공률/FPE 테이블
  2. Run Eval      — 모델 선택 후 CL eval 백그라운드 실행 + 로그 스트리밍
  3. Robot Log     — 실로봇 주행 세션 기록 (날짜/모델/경로/성공여부/메모)

오른쪽 사이드에 TODO 패널: 현재 진행 작업 상태 팝업 표시
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

METRICS_PATH      = ROOT / "docs" / "v5" / "closed_loop_eval" / "rollout_metrics.json"
ROBOT_LOG_PATH    = ROOT / "docs" / "v5" / "real_robot_sessions.json"
CL_EVAL_SCRIPT    = ROOT / "scripts" / "sim" / "evaluate_closed_loop_v5.py"

GOAL_NAV_MODELS = ["exp49", "exp50", "exp51", "exp52", "exp53"]
PATH_TYPES = [
    "center_straight", "center_left", "center_right",
    "left_straight",   "left_left",   "left_right",
    "right_straight",  "right_left",  "right_right",
]

TODO_ITEMS = [
    {"label": "exp53 feature 추출 (150 ep)",      "status": "running"},
    {"label": "CL eval 스크립트 exp53 지원",       "status": "done"},
    {"label": "exp53 CL eval 실행 (30 ep)",        "status": "pending"},
    {"label": "CL 대시보드 UI 구축",               "status": "running"},
    {"label": "gradio_hub.py 등록",                "status": "pending"},
]

# ─── 데이터 로드 유틸 ────────────────────────────────────────────────────────

def load_metrics() -> dict:
    if not METRICS_PATH.exists():
        return {}
    try:
        return json.loads(METRICS_PATH.read_text())
    except Exception:
        return {}


def load_robot_log() -> list[dict]:
    if not ROBOT_LOG_PATH.exists():
        return []
    try:
        return json.loads(ROBOT_LOG_PATH.read_text()).get("sessions", [])
    except Exception:
        return []


def save_robot_log(sessions: list[dict]) -> None:
    ROBOT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ROBOT_LOG_PATH.write_text(json.dumps({"sessions": sessions}, indent=2, ensure_ascii=False))


# ─── CL 결과 테이블 ──────────────────────────────────────────────────────────

def build_summary_table() -> list[list]:
    metrics = load_metrics()
    summary = metrics.get("summary", {})
    per_path = metrics.get("per_path", {})

    rows = []
    model_order = ["exp49", "exp51", "exp54_s2v2", "exp52", "exp53", "exp50",
                   "step2", "step3", "exp19", "exp11"]
    seen = set()
    ordered = [m for m in model_order if m in summary] + \
              [m for m in summary if m not in model_order]

    for model in ordered:
        if model in seen:
            continue
        seen.add(model)
        s = summary[model]
        sr = s.get("success_rate", 0)
        fpe = s.get("mean_fpe", 0)
        n = s.get("n_episodes") or s.get("n_seeds", "?")

        # per-path breakdown
        pp = per_path.get(model, {})
        path_cells = []
        for pt in PATH_TYPES:
            eps = pp.get(pt, [])
            if eps:
                succ = sum(1 for e in eps if e.get("success"))
                path_cells.append(f"{succ}/{len(eps)}")
            else:
                path_cells.append("-")

        rows.append([model, f"{sr:.1%}", f"{fpe:.3f}", str(n)] + path_cells)

    return rows


def get_summary_html() -> str:
    metrics = load_metrics()
    summary = metrics.get("summary", {})
    if not summary:
        return "<p>No CL results found. Run an evaluation first.</p>"

    rows_html = ""
    model_order = ["exp49", "exp51", "exp54_s2v2", "exp52", "exp53", "exp50",
                   "step2", "step3", "exp19", "exp11"]
    seen: set = set()
    ordered = [m for m in model_order if m in summary] + \
              [m for m in summary if m not in model_order]

    for model in ordered:
        if model in seen:
            continue
        seen.add(model)
        s = summary[model]
        sr = s.get("success_rate", 0)
        fpe = s.get("mean_fpe", 0)
        n = s.get("n_episodes") or s.get("n_seeds", "?")
        color = "#2ecc71" if sr >= 0.9 else "#f39c12" if sr >= 0.7 else "#e74c3c"
        rows_html += f"""
        <tr>
          <td><b>{model}</b></td>
          <td style="color:{color};font-weight:bold">{sr:.1%}</td>
          <td>{fpe:.3f}</td>
          <td>{n}</td>
        </tr>"""

    return f"""
    <table style="width:100%;border-collapse:collapse;font-size:14px">
      <thead>
        <tr style="background:#2c3e50;color:white">
          <th style="padding:8px;text-align:left">모델</th>
          <th style="padding:8px">성공률</th>
          <th style="padding:8px">FPE</th>
          <th style="padding:8px">에피소드</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>"""


def get_perpath_html(model: str) -> str:
    metrics = load_metrics()
    per_path = metrics.get("per_path", {}).get(model, {})
    if not per_path:
        return f"<p>No per-path data for <b>{model}</b></p>"

    rows_html = ""
    for pt in PATH_TYPES:
        eps = per_path.get(pt, [])
        if not eps:
            continue
        succ = sum(1 for e in eps if e.get("success"))
        fpe_vals = [e.get("fpe", 0) for e in eps]
        mean_fpe = sum(fpe_vals) / len(fpe_vals) if fpe_vals else 0
        color = "#2ecc71" if succ == len(eps) else "#f39c12" if succ > 0 else "#e74c3c"
        rows_html += f"""
        <tr>
          <td>{pt}</td>
          <td style="color:{color};font-weight:bold">{succ}/{len(eps)}</td>
          <td>{mean_fpe:.3f}</td>
        </tr>"""

    return f"""
    <b>{model} per-path 상세</b>
    <table style="width:100%;border-collapse:collapse;font-size:13px;margin-top:8px">
      <thead>
        <tr style="background:#34495e;color:white">
          <th style="padding:6px;text-align:left">path_type</th>
          <th style="padding:6px">성공/전체</th>
          <th style="padding:6px">mean FPE</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>"""


# ─── CL eval 실행 ────────────────────────────────────────────────────────────

_eval_proc: subprocess.Popen | None = None


def run_eval(model: str):
    global _eval_proc
    if _eval_proc and _eval_proc.poll() is None:
        yield "⚠️ 이미 실행 중입니다. 완료 후 다시 시도하세요.", gr.update()
        return

    npz_check = ROOT / "docs" / "v5" / f"bbox_nav_{model}" / "vision_features.npz"
    if not npz_check.exists() and model in GOAL_NAV_MODELS:
        lang_check = ROOT / "docs" / "v5" / f"bbox_nav_{model}" / "lang_vis_features.npz"
        if not lang_check.exists():
            yield f"⚠️ {model} feature npz 없음. 먼저 추출 스크립트를 실행하세요.", gr.update()
            return

    cmd = [sys.executable, str(CL_EVAL_SCRIPT), "--model", model]
    yield f"▶ 실행: {' '.join(cmd)}\n", gr.update()

    _eval_proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    log = f"▶ {model} CL eval 시작 ({datetime.now().strftime('%H:%M:%S')})\n"
    for line in _eval_proc.stdout:
        log += line
        yield log, gr.update()

    _eval_proc.wait()
    rc = _eval_proc.returncode
    status = "✅ 완료" if rc == 0 else f"❌ 실패 (exit {rc})"
    log += f"\n{status} ({datetime.now().strftime('%H:%M:%S')})\n"
    yield log, gr.update(value=get_summary_html())


# ─── 실로봇 주행 로그 ────────────────────────────────────────────────────────

def add_session(model, path_type, success, notes):
    sessions = load_robot_log()
    entry = {
        "timestamp": datetime.now().isoformat(),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "model": model,
        "path_type": path_type,
        "success": success == "성공",
        "notes": notes or "",
    }
    sessions.insert(0, entry)
    save_robot_log(sessions)
    return build_session_table(sessions), "✅ 저장됨"


def build_session_table(sessions: list[dict] | None = None) -> list[list]:
    if sessions is None:
        sessions = load_robot_log()
    rows = []
    for s in sessions[:50]:
        icon = "✅" if s.get("success") else "❌"
        rows.append([
            s.get("date", ""),
            s.get("model", ""),
            s.get("path_type", ""),
            icon,
            s.get("notes", ""),
        ])
    return rows


# ─── TODO 패널 HTML ──────────────────────────────────────────────────────────

def build_todo_html() -> str:
    items_html = ""
    for item in TODO_ITEMS:
        st = item["status"]
        if st == "done":
            icon, color = "✅", "#2ecc71"
        elif st == "running":
            icon, color = "🔄", "#f39c12"
        else:
            icon, color = "⬜", "#95a5a6"
        items_html += f'<div style="padding:4px 0;color:{color}">{icon} {item["label"]}</div>'

    return f"""
    <div style="font-size:13px;line-height:1.6">
      <b style="font-size:14px">📋 진행 상황</b>
      <hr style="margin:6px 0;border-color:#444">
      {items_html}
      <hr style="margin:6px 0;border-color:#444">
      <div style="font-size:11px;color:#888">{datetime.now().strftime('%H:%M:%S')} 기준</div>
    </div>"""


# ─── UI ──────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="MoNaVLA CL Dashboard", theme=gr.themes.Soft()) as demo:
        # 헤더 + TODO 사이드패널
        with gr.Row():
            with gr.Column(scale=4):
                gr.Markdown("## MoNaVLA Closed-Loop Dashboard")
            with gr.Column(scale=1, min_width=220):
                todo_panel = gr.HTML(value=build_todo_html(), label="")
                todo_refresh_btn = gr.Button("↻ 새로고침", size="sm", variant="secondary")

        # ── 탭 ──────────────────────────────────────────────────────────────
        with gr.Tabs():

            # Tab 1: CL Results
            with gr.Tab("📊 CL Results"):
                result_html = gr.HTML(value=get_summary_html())
                with gr.Row():
                    refresh_btn = gr.Button("🔄 결과 새로고침", variant="secondary")
                    model_detail_dd = gr.Dropdown(
                        choices=[""] + list(load_metrics().get("summary", {}).keys()),
                        label="모델 선택 → per-path 상세",
                        value="",
                        scale=2,
                    )
                perpath_html = gr.HTML()

                refresh_btn.click(
                    fn=lambda: (get_summary_html(),
                                gr.update(choices=[""] + list(load_metrics().get("summary", {}).keys()))),
                    outputs=[result_html, model_detail_dd],
                )
                model_detail_dd.change(
                    fn=lambda m: get_perpath_html(m) if m else "",
                    inputs=model_detail_dd,
                    outputs=perpath_html,
                )

            # Tab 2: Run Eval
            with gr.Tab("▶ Run CL Eval"):
                gr.Markdown("### Closed-Loop 평가 실행")
                with gr.Row():
                    eval_model_dd = gr.Dropdown(
                        choices=GOAL_NAV_MODELS,
                        value="exp53",
                        label="모델",
                        scale=2,
                    )
                    run_btn = gr.Button("▶ 평가 시작", variant="primary", scale=1)
                eval_log = gr.Textbox(
                    label="실행 로그",
                    lines=20,
                    max_lines=40,
                    interactive=False,
                )
                eval_result_html = gr.HTML()

                run_btn.click(
                    fn=run_eval,
                    inputs=eval_model_dd,
                    outputs=[eval_log, eval_result_html],
                )

            # Tab 3: Robot Log
            with gr.Tab("🤖 Real Robot Log"):
                gr.Markdown("### 실로봇 주행 기록")
                with gr.Row():
                    log_model = gr.Dropdown(choices=GOAL_NAV_MODELS, value="exp49", label="모델")
                    log_path  = gr.Dropdown(choices=PATH_TYPES, value="center_straight", label="경로 유형")
                    log_succ  = gr.Radio(choices=["성공", "실패"], value="성공", label="결과")
                with gr.Row():
                    log_notes = gr.Textbox(label="메모", placeholder="특이사항 기록...", scale=4)
                    save_btn  = gr.Button("💾 저장", variant="primary", scale=1)
                save_status = gr.Textbox(label="", lines=1, interactive=False)

                gr.Markdown("#### 최근 주행 기록 (최대 50건)")
                session_table = gr.Dataframe(
                    value=build_session_table(),
                    headers=["날짜", "모델", "경로 유형", "결과", "메모"],
                    datatype=["str", "str", "str", "str", "str"],
                    interactive=False,
                    wrap=True,
                )

                save_btn.click(
                    fn=add_session,
                    inputs=[log_model, log_path, log_succ, log_notes],
                    outputs=[session_table, save_status],
                )

        # TODO 패널 새로고침
        todo_refresh_btn.click(fn=build_todo_html, outputs=todo_panel)

    return demo


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=7867)
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()

    print(f"[CL Dashboard] http://0.0.0.0:{args.port}")
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()

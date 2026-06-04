#!/usr/bin/env python3
"""
Real Robot Trial Logger — 실로봇 주행 평가 기록기
포트 7862 독립 Gradio 서버

평가 기준 (카메라 최종 프레임 기반, 2×2m 맵):
  S: 바스켓 중앙 + bbox 60%+  (~0.1m 이내)
  A: 바스켓 중앙 1/3 + bbox 40%+  (~0.1~0.3m)
  B: 바스켓 보이지만 중앙 이탈 or 40% 미만  (~0.3~0.7m)
  F: 바스켓 프레임 밖 or 안 보임  (>0.7m or 이탈)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
EVAL_OUT_DIR = ROOT / "docs" / "v5" / "eval"
EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["exp49", "exp53", "exp54_s2v2"]

POSITIONS = {
    "A": "A — 정면 가까움 (~1.5m)",
    "B": "B — 정면 멀리 (~2.5m)",
    "C": "C — 측면 비스듬 (~2.0m, 30~45°)",
}

FAILURE_REASONS = [
    "(해당 없음)",
    "FORWARD_BIAS — 항상 직진",
    "WRONG_DIR — 반대 방향",
    "EARLY_STOP — 중간에 멈춤",
    "DRIFT — 누적 편차 이탈",
    "GROUNDING_FAIL — bbox 미검출",
    "COLLISION — 충돌/걸림",
    "OTHER",
]

# 오프라인 CL 베이스라인
OFFLINE_CL = {
    "exp49":       {"n_ok": 29, "n": 30, "rate": 29/30},
    "exp53":       {"n_ok": 29, "n": 30, "rate": 29/30},
    "exp54_s2v2":  {"n_ok": 29, "n": 30, "rate": 29/30},
}

GRADE_DESC = {
    "S": "S — 중앙 + bbox 60%+ (~0.1m)",
    "A": "A — 중앙 1/3 + bbox 40%+ (~0.3m)",
    "B": "B — 보이지만 이탈 or 작음 (~0.7m)",
    "F": "F — 프레임 밖 or 안 보임",
}


# ── 등급 자동 계산 ────────────────────────────────────────────────────────

def compute_grade(visible: bool, center: bool, bbox_pct: float) -> str:
    if not visible:
        return "F"
    if center and bbox_pct >= 60:
        return "S"
    if center and bbox_pct >= 40:
        return "A"
    return "B"


def grade_label(visible: str, center: str, bbox_pct: float) -> tuple[str, str]:
    v = visible == "Y"
    c = center == "Y"
    g = compute_grade(v, c, bbox_pct)
    color = {"S": "🟢", "A": "🟡", "B": "🟠", "F": "🔴"}[g]
    desc = GRADE_DESC[g]
    return f"{color} **{g}등급** — {desc}", g


# ── 상태 ─────────────────────────────────────────────────────────────────

def empty_state() -> dict:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        "trials": [],
        "save_path": str(EVAL_OUT_DIR / f"real_robot_{ts}.json"),
    }


# ── 로직 ─────────────────────────────────────────────────────────────────

def log_trial(state: dict, model: str, position_key: str,
              visible: str, center: str, bbox_pct: float,
              failure_reason: str, notes: str):
    g = compute_grade(visible == "Y", center == "Y", bbox_pct)
    success = g in ("S", "A")
    trial = {
        "trial_id":       len(state["trials"]) + 1,
        "timestamp":      datetime.now().isoformat(),
        "model":          model,
        "position":       position_key,
        "grade":          g,
        "success":        success,
        "basket_visible": visible == "Y",
        "bbox_center":    center == "Y",
        "bbox_pct":       bbox_pct,
        "failure_reason": None if success else failure_reason,
        "notes":          notes.strip(),
    }
    state["trials"].append(trial)
    _autosave(state)
    pos_label = POSITIONS[position_key]
    status = f"✅ #{trial['trial_id']} [{model}] [{pos_label}] {g}등급 기록됨"
    return state, _summary_md(state["trials"]), _grade_table(state["trials"]), status, ""


def undo_last(state: dict):
    if not state["trials"]:
        return state, _summary_md([]), _grade_table([]), "⚠️ 취소할 기록 없음"
    t = state["trials"].pop()
    _autosave(state)
    return state, _summary_md(state["trials"]), _grade_table(state["trials"]), \
           f"↩️ #{t['trial_id']} [{t['model']}] {t['grade']}등급 취소됨"


def _autosave(state: dict):
    trials = state["trials"]
    n = len(trials)
    n_ok = sum(1 for t in trials if t["success"])
    payload = {
        "date":          datetime.now().strftime("%Y-%m-%d"),
        "server":        "soda@100.85.118.58",
        "map_size":      "2x2m",
        "success_def":   "S or A grade (basket visible + centered + bbox>=40%)",
        "total_trials":  n,
        "success_rate":  round(n_ok / n, 4) if n else 0,
        "trials":        trials,
    }
    Path(state["save_path"]).write_text(
        json.dumps(payload, indent=2, ensure_ascii=False)
    )


# ── 렌더링 ───────────────────────────────────────────────────────────────

def _summary_md(trials: list[dict]) -> str:
    if not trials:
        return "_아직 기록된 시도 없음_"
    n = len(trials)
    n_ok = sum(1 for t in trials if t["success"])
    rate = n_ok / n

    grade_counts = {"S": 0, "A": 0, "B": 0, "F": 0}
    for t in trials:
        grade_counts[t["grade"]] = grade_counts.get(t["grade"], 0) + 1

    icon = "🟢" if rate >= 0.8 else ("🟡" if rate >= 0.5 else "🔴")
    lines = [
        f"## {icon} 전체 {n_ok}/{n} = **{rate:.1%}** (S+A 기준)",
        f"🟢S:{grade_counts['S']}  🟡A:{grade_counts['A']}  🟠B:{grade_counts['B']}  🔴F:{grade_counts['F']}",
        "",
    ]

    # 모델별 집계
    lines.append("### 모델별")
    for m in MODELS:
        mt = [t for t in trials if t["model"] == m]
        if not mt:
            continue
        mn = len(mt)
        mok = sum(1 for t in mt if t["success"])
        off = OFFLINE_CL.get(m, {})
        off_str = f"CL {off['rate']:.1%}" if off else ""
        diff = (mok/mn) - off.get("rate", 0) if off else 0
        diff_str = f"({diff:+.1%})" if off else ""
        gc = {"S":0,"A":0,"B":0,"F":0}
        for t in mt:
            gc[t["grade"]] += 1
        lines.append(f"- **{m}**: {mok}/{mn}={mok/mn:.0%} {diff_str} vs {off_str}  "
                     f"[S:{gc['S']} A:{gc['A']} B:{gc['B']} F:{gc['F']}]")

    # 실패 원인
    fails = [t for t in trials if not t["success"]]
    if fails:
        lines += ["", "### 실패 원인"]
        counts: dict[str, int] = {}
        for t in fails:
            r = t.get("failure_reason") or "OTHER"
            counts[r] = counts.get(r, 0) + 1
        for r, c in sorted(counts.items(), key=lambda x: -x[1]):
            lines.append(f"- {r}: {c}회")

    return "\n".join(lines)


def _grade_table(trials: list[dict]) -> list[list]:
    """Position × Grade 분포표."""
    rows = []
    for pk, plabel in POSITIONS.items():
        pt = [t for t in trials if t["position"] == pk]
        if not pt:
            rows.append([plabel, "—", "—", "—", "—", "—"])
            continue
        gc = {"S": 0, "A": 0, "B": 0, "F": 0}
        for t in pt:
            gc[t["grade"]] += 1
        n = len(pt)
        n_ok = sum(1 for t in pt if t["success"])
        rows.append([
            plabel,
            f"{gc['S']}",
            f"{gc['A']}",
            f"{gc['B']}",
            f"{gc['F']}",
            f"{n_ok}/{n} ({n_ok/n:.0%})" if n else "—",
        ])
    return rows


# ── UI ───────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Real Robot Trial Logger") as demo:
        state = gr.State(empty_state())

        gr.Markdown(
            "# 🤖 실로봇 주행 평가 Logger\n"
            "최종 프레임 카메라 기준 S/A/B/F 등급 — 2×2m 맵\n\n"
            "**성공 = S 또는 A** (바스켓 프레임 중앙 + bbox 40% 이상)"
        )

        with gr.Row():
            # ── 입력 패널 ─────────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("## 시도 기록")

                model_dd = gr.Dropdown(
                    choices=MODELS,
                    value="exp54_s2v2",
                    label="모델",
                )
                position_dd = gr.Radio(
                    choices=list(POSITIONS.keys()),
                    value="A",
                    label="시작 위치",
                    info="A=정면 가까움(1.5m) / B=정면 멀리(2.5m) / C=측면 비스듬(2m, 30~45°)",
                )

                gr.Markdown("---\n### 📷 최종 프레임 평가")

                visible_r = gr.Radio(
                    choices=["Y", "N"],
                    value="Y",
                    label="바스켓이 프레임에 보이는가?",
                )
                center_r = gr.Radio(
                    choices=["Y", "N"],
                    value="Y",
                    label="바스켓 bbox가 화면 중앙 1/3 안에 있는가?",
                    visible=True,
                )
                bbox_slider = gr.Slider(
                    minimum=0, maximum=100, step=5, value=50,
                    label="bbox 화면 점유율 (%)",
                    info="S≥60% / A≥40% / B<40%",
                    visible=True,
                )

                grade_display = gr.Markdown("🟡 **A등급** — A — 바스켓 중앙 1/3 + bbox 40%+ (~0.3m)")
                grade_hidden  = gr.Textbox(value="A", visible=False)

                failure_reason = gr.Dropdown(
                    choices=FAILURE_REASONS,
                    value=FAILURE_REASONS[0],
                    label="실패 원인 (B/F 등급시)",
                    visible=False,
                )
                notes = gr.Textbox(
                    label="메모 (선택)",
                    placeholder="예: bbox 왼쪽 치우침, 조명 어두움...",
                    lines=2,
                )

                with gr.Row():
                    btn_log  = gr.Button("📝 기록", variant="primary", scale=3)
                    btn_undo = gr.Button("↩ 취소", variant="secondary", scale=1)

                status_box = gr.Textbox(label="상태", interactive=False, lines=1)

            # ── 요약 패널 ─────────────────────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("## 실시간 집계")

                summary_md = gr.Markdown("_아직 기록된 시도 없음_")

                gr.Markdown("### 위치별 등급 분포")
                grade_tbl = gr.Dataframe(
                    headers=["위치", "🟢S", "🟡A", "🟠B", "🔴F", "성공(S+A)"],
                    value=_grade_table([]),
                    interactive=False,
                    wrap=True,
                )

                gr.Markdown(
                    "---\n**등급 기준 (2×2m 맵)**\n"
                    "| 등급 | 거리 | 바스켓 | 중앙 | bbox% |\n"
                    "|---|---|---|---|---|\n"
                    "| 🟢 S | ~0.1m | 보임 | ✅ | 60%+ |\n"
                    "| 🟡 A | ~0.3m | 보임 | ✅ | 40%+ |\n"
                    "| 🟠 B | ~0.7m | 보임 | ❌ or <40% | any |\n"
                    "| 🔴 F | >0.7m | ❌ 안 보임 | — | — |"
                )

        # ── 이벤트 ──────────────────────────────────────────────────────

        def on_visible_change(v):
            hidden = v == "Y"
            return gr.update(visible=hidden), gr.update(visible=hidden)

        visible_r.change(
            fn=on_visible_change,
            inputs=[visible_r],
            outputs=[center_r, bbox_slider],
        )

        def on_grade_inputs(v, c, pct):
            label, g = grade_label(v, c, pct)
            show_fail = g in ("B", "F")
            return label, g, gr.update(visible=show_fail)

        for comp in [visible_r, center_r, bbox_slider]:
            comp.change(
                fn=on_grade_inputs,
                inputs=[visible_r, center_r, bbox_slider],
                outputs=[grade_display, grade_hidden, failure_reason],
            )

        btn_log.click(
            fn=log_trial,
            inputs=[state, model_dd, position_dd,
                    visible_r, center_r, bbox_slider,
                    failure_reason, notes],
            outputs=[state, summary_md, grade_tbl, status_box, notes],
        )

        btn_undo.click(
            fn=undo_last,
            inputs=[state],
            outputs=[state, summary_md, grade_tbl, status_box],
        )

    return demo


# ── 진입점 ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7862)
    args = ap.parse_args()

    print(f"📊 Trial Logger → http://{args.host}:{args.port}")
    print(f"📂 저장 → {EVAL_OUT_DIR}/real_robot_*.json")
    build_ui().launch(server_name=args.host, server_port=args.port)

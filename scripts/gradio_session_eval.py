#!/usr/bin/env python3
"""
Session Eval Server — Pure HF Kosmos-2로 H5 에피소드 품질 자동 평가
"수집 문제(basket 비가시) vs 모델 문제(방향 불일치)" 분리
포트 7861 독립 Gradio 서버
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import h5py
import numpy as np
import torch
import gradio as gr
from PIL import Image

# ── 프로젝트 경로 설정 ───────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KOSMOS_MODEL_PATH = ROOT / ".vlms" / "kosmos-2-patch14-224"
DATASET_DIR = ROOT / "ROS_action" / "mobile_vla_dataset_v5"
EVAL_OUT_DIR = ROOT / "docs" / "v5" / "eval"
EVAL_OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 방향 판정 임계값 (UI 슬라이더로 조정 가능)
_AZ_DEFAULT  = 0.15   # angular_z 회전 임계값
_LY_DEFAULT  = 0.10   # lateral_y 스트레이프 임계값
_LX_DEFAULT  = 0.05   # linear_x 전진 임계값
_CX_LEFT     = 0.38   # grounding cx < 이 값 → basket LEFT
_CX_RIGHT    = 0.62   # grounding cx > 이 값 → basket RIGHT


# ── 데이터 클래스 ────────────────────────────────────────────────────

@dataclass
class FrameReport:
    frame_idx: int
    action_raw: list[float]       # [lx, ly, az]
    action_dir: str               # "left" / "center" / "right" / "none"
    bbox: Optional[dict]          # {"cx", "cy", ...} or None
    grounding_dir: Optional[str]  # "left" / "center" / "right" or None
    agree: Optional[bool]
    caption: str = ""
    latency_ms: float = 0.0


@dataclass
class EpisodeReport:
    episode_id: str
    path_type: str
    n_frames: int
    grounding_success_rate: float
    action_agreement_rate: float
    verdict: str                  # "ok" / "model_issue" / "collection_issue"
    frames: list[FrameReport] = field(default_factory=list)


# ── 방향 변환 헬퍼 ───────────────────────────────────────────────────

def raw_action_to_dir(act: list[float],
                      az_thr: float = _AZ_DEFAULT,
                      ly_thr: float = _LY_DEFAULT,
                      lx_thr: float = _LX_DEFAULT) -> str:
    lx, ly, az = act[0], act[1], act[2]
    # 회전이 가장 강한 신호 (ROT_L/ROT_R)
    if abs(az) > az_thr:
        return "left" if az > 0 else "right"
    # 스트레이프
    if abs(ly) > ly_thr:
        return "left" if ly > 0 else "right"
    # 전진
    if lx > lx_thr:
        return "center"
    return "none"


def bbox_to_dir(bbox: dict, cx_left: float = _CX_LEFT, cx_right: float = _CX_RIGHT) -> str:
    cx = bbox["cx"]
    if cx < cx_left:
        return "left"
    if cx > cx_right:
        return "right"
    return "center"


def _parse_path_type(stem: str) -> str:
    """파일명 episode_<ts>_<scenario>__<pattern>__<dist>_<tag>.h5 에서 pattern 추출."""
    parts = stem.split("__")
    return parts[1] if len(parts) >= 2 else "unknown"


def _build_report(episode_id: str, path_type: str,
                  frames: list[FrameReport]) -> EpisodeReport:
    n = len(frames)
    grounded = [f for f in frames if f.grounding_dir is not None]
    agreed = [f for f in grounded if f.agree is True]

    gsr = len(grounded) / n if n > 0 else 0.0
    aar = len(agreed) / len(grounded) if grounded else 0.0

    if gsr < 0.5:
        verdict = "collection_issue"
    elif aar < 0.5:
        verdict = "model_issue"
    else:
        verdict = "ok"

    return EpisodeReport(
        episode_id=episode_id,
        path_type=path_type,
        n_frames=n,
        grounding_success_rate=round(gsr, 3),
        action_agreement_rate=round(aar, 3),
        verdict=verdict,
        frames=frames,
    )


# ── 평가 엔진 ────────────────────────────────────────────────────────

class EvalEngine:
    """Kosmos-2 grounding을 lazy-load해서 H5 에피소드를 평가한다."""

    def __init__(self):
        self._grounding = None

    def _load(self):
        if self._grounding is not None:
            return
        if not KOSMOS_MODEL_PATH.exists():
            raise FileNotFoundError(f"Kosmos-2 모델 없음: {KOSMOS_MODEL_PATH}")
        from robovlm_nav.serve.proxy_inference_server import GroundingBackend
        self._grounding = GroundingBackend(KOSMOS_MODEL_PATH, torch.device(DEVICE))

    def eval_episode(self, h5_path: Path,
                     az_thr: float = _AZ_DEFAULT,
                     ly_thr: float = _LY_DEFAULT,
                     lx_thr: float = _LX_DEFAULT) -> tuple[EpisodeReport, list[np.ndarray]]:
        """평가 실행. (EpisodeReport, raw_images) 반환."""
        self._load()

        with h5py.File(h5_path, "r") as f:
            if "observations" in f and "images" in f["observations"]:
                images = f["observations"]["images"][:]
            else:
                images = f["images"][:]
            actions = f["actions"][:]
            path_type = dict(f.attrs).get("pattern", _parse_path_type(h5_path.stem))

        frames: list[FrameReport] = []
        for t, (img, act) in enumerate(zip(images, actions)):
            img_rgb = img.astype(np.uint8)
            act_list = act[:3].tolist()
            action_dir = raw_action_to_dir(act_list, az_thr, ly_thr, lx_thr)

            result = self._grounding.run(img_rgb)
            bbox = result.get("bbox")
            grounding_dir = bbox_to_dir(bbox) if bbox else None

            if grounding_dir is not None and action_dir != "none":
                agree = grounding_dir == action_dir
            else:
                agree = None

            frames.append(FrameReport(
                frame_idx=t,
                action_raw=act_list,
                action_dir=action_dir,
                bbox=bbox,
                grounding_dir=grounding_dir,
                agree=agree,
                caption=result.get("caption", ""),
                latency_ms=result.get("latency_ms", 0.0),
            ))

        report = _build_report(h5_path.stem, path_type, frames)
        return report, [img.astype(np.uint8) for img in images]


# 전역 엔진 (Gradio 재시작 없이 모델 재사용)
_engine = EvalEngine()


# ── 프레임 어노테이션 ────────────────────────────────────────────────

def annotate_frame(img_rgb: np.ndarray, frame: FrameReport) -> np.ndarray:
    img = img_rgb.copy()
    h, w = img.shape[:2]

    if frame.bbox:
        x1 = int(frame.bbox.get("x1", frame.bbox["cx"] - 0.06) * w)
        y1 = int(frame.bbox.get("y1", frame.bbox["cy"] - 0.06) * h)
        x2 = int(frame.bbox.get("x2", frame.bbox["cx"] + 0.06) * w)
        y2 = int(frame.bbox.get("y2", frame.bbox["cy"] + 0.06) * h)
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, w), min(y2, h)
        color = (30, 200, 30) if frame.agree else (200, 50, 50)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"G:{frame.grounding_dir} A:{frame.action_dir}"
        cv2.putText(img, label, (max(x1, 2), max(y1 - 5, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)
    else:
        cv2.putText(img, f"A:{frame.action_dir}  G:none",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 130, 0), 1, cv2.LINE_AA)

    # 액션 방향 화살표
    cy_px, cx_px = h // 2, w // 2
    arrows = {"left": (-55, 0), "center": (0, -55), "right": (55, 0)}
    dx, dy = arrows.get(frame.action_dir, (0, 0))
    if dx or dy:
        cv2.arrowedLine(img, (cx_px, cy_px), (cx_px + dx, cy_px + dy),
                        (255, 220, 0), 2, tipLength=0.3)

    # 프레임 인덱스
    cv2.putText(img, f"#{frame.frame_idx}", (w - 40, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)
    return img


# ── Gradio 콜백 ──────────────────────────────────────────────────────

def scan_episodes(scan_dir: str) -> gr.update:
    d = Path(scan_dir.strip())
    if not d.exists():
        return gr.update(choices=[], value=None)
    h5s = sorted(d.glob("episode_*.h5"), key=lambda p: p.stat().st_mtime, reverse=True)
    names = [p.name for p in h5s]
    return gr.update(choices=names, value=names[:1] if names else None)


def run_eval(ep_names: list[str], scan_dir: str,
             az_thr: float, ly_thr: float, lx_thr: float,
             state: dict):
    """배치 평가 — generator로 진행률 업데이트."""
    if not ep_names:
        yield [], "에피소드를 선택하세요.", state
        return

    data_dir = Path(scan_dir.strip())
    reports: dict[str, EpisodeReport] = dict(state.get("reports", {}))
    images: dict[str, list[np.ndarray]] = dict(state.get("images", {}))

    for i, name in enumerate(ep_names):
        status = f"[{i+1}/{len(ep_names)}] 평가 중: {name}"
        yield _build_table(list(reports.values())), status, state

        h5_path = data_dir / name
        try:
            report, imgs = _engine.eval_episode(h5_path, az_thr, ly_thr, lx_thr)
            reports[report.episode_id] = report
            images[report.episode_id] = imgs
        except Exception as e:
            yield _build_table(list(reports.values())), f"오류: {name} — {e}", state
            continue

    new_state = {**state, "reports": reports, "images": images, "selected": None}
    total = len(reports)
    ok = sum(1 for r in reports.values() if r.verdict == "ok")
    mi = sum(1 for r in reports.values() if r.verdict == "model_issue")
    ci = sum(1 for r in reports.values() if r.verdict == "collection_issue")
    summary = f"완료 {total}ep | ok:{ok}  model_issue:{mi}  collection_issue:{ci}"
    yield _build_table(list(reports.values())), summary, new_state


def _build_table(reports: list[EpisodeReport]) -> list[list]:
    rows = []
    for r in reports:
        verdict_emoji = {"ok": "✅", "model_issue": "⚠️", "collection_issue": "❌"}.get(r.verdict, "?")
        rows.append([
            r.episode_id[:50],
            r.path_type,
            r.n_frames,
            f"{r.grounding_success_rate:.1%}",
            f"{r.action_agreement_rate:.1%}",
            f"{verdict_emoji} {r.verdict}",
        ])
    return rows


def on_table_select(evt: gr.SelectData, state: dict):
    """테이블 행 클릭 → 해당 에피소드 프레임 뷰어 초기화."""
    reports: dict = state.get("reports", {})
    images: dict = state.get("images", {})
    if not reports:
        return None, gr.update(maximum=0, value=0), "에피소드 없음", state

    ep_ids = list(reports.keys())
    row_idx = evt.index[0] if hasattr(evt, "index") else 0
    if row_idx >= len(ep_ids):
        return None, gr.update(maximum=0, value=0), "", state

    ep_id = ep_ids[row_idx]
    new_state = {**state, "selected": ep_id}
    report = reports[ep_id]
    imgs = images.get(ep_id, [])

    if not imgs or not report.frames:
        return None, gr.update(maximum=0, value=0), "이미지 없음", new_state

    annotated = annotate_frame(imgs[0], report.frames[0])
    info = _frame_info(report.frames[0])
    return annotated, gr.update(maximum=len(imgs) - 1, value=0), info, new_state


def on_frame_slide(frame_idx: int, state: dict):
    """슬라이더 이동 → 해당 프레임 표시."""
    ep_id = state.get("selected")
    if not ep_id:
        return None, ""
    reports: dict = state.get("reports", {})
    images: dict = state.get("images", {})
    report = reports.get(ep_id)
    imgs = images.get(ep_id, [])
    if not report or frame_idx >= len(imgs):
        return None, ""

    frame = report.frames[frame_idx]
    annotated = annotate_frame(imgs[frame_idx], frame)
    return annotated, _frame_info(frame)


def _frame_info(f: FrameReport) -> str:
    bbox_str = "없음"
    if f.bbox:
        cx, cy = f.bbox.get("cx", 0), f.bbox.get("cy", 0)
        entity = f.bbox.get("entity", "")
        bbox_str = f"cx={cx:.2f} cy={cy:.2f} [{entity}]"
    agree_str = {True: "✅ 일치", False: "❌ 불일치", None: "— (평가 제외)"}.get(f.agree, "?")
    return (
        f"프레임 #{f.frame_idx}\n"
        f"액션: {f.action_dir}  [{f.action_raw[0]:.2f}, {f.action_raw[1]:.2f}, {f.action_raw[2]:.2f}]\n"
        f"Grounding: {f.grounding_dir or '실패'}  {bbox_str}\n"
        f"판정: {agree_str}  ({f.latency_ms:.0f}ms)\n"
        f"Caption: {f.caption[:100]}"
    )


def export_json(state: dict):
    reports: dict = state.get("reports", {})
    if not reports:
        return None

    out = {
        "generated": datetime.now().isoformat(),
        "episodes": [
            {
                "episode_id": r.episode_id,
                "path_type": r.path_type,
                "n_frames": r.n_frames,
                "grounding_success_rate": r.grounding_success_rate,
                "action_agreement_rate": r.action_agreement_rate,
                "verdict": r.verdict,
                "frames": [
                    {
                        "frame_idx": f.frame_idx,
                        "action_dir": f.action_dir,
                        "grounding_dir": f.grounding_dir,
                        "agree": f.agree,
                        "bbox_cx": f.bbox["cx"] if f.bbox else None,
                        "caption": f.caption,
                    }
                    for f in r.frames
                ],
            }
            for r in reports.values()
        ],
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = EVAL_OUT_DIR / f"eval_results_{ts}.json"
    with open(out_path, "w") as fp:
        json.dump(out, fp, indent=2, ensure_ascii=False)
    return str(out_path)


# ── Gradio UI ────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="VLA Session Evaluator", theme=gr.themes.Soft()) as demo:
        state = gr.State({"reports": {}, "images": {}, "selected": None})

        gr.Markdown("## 🔍 VLA Session Evaluator\nPure HF Kosmos-2 grounding으로 수집 품질 / 모델 오류를 자동 분리합니다.")

        with gr.Row():
            # ── 좌측: 설정 + 선택 ──
            with gr.Column(scale=1, min_width=280):
                scan_dir = gr.Textbox(
                    value=str(DATASET_DIR),
                    label="Dataset Directory",
                    placeholder="/path/to/mobile_vla_dataset_v5",
                )
                btn_scan = gr.Button("📁 Scan Episodes", variant="secondary")
                ep_dropdown = gr.Dropdown(
                    label="평가할 에피소드 (복수 선택 가능)",
                    multiselect=True,
                    choices=[],
                )

                with gr.Accordion("⚙️ 임계값 설정", open=False):
                    az_slider = gr.Slider(0.05, 0.5, value=_AZ_DEFAULT, step=0.01,
                                         label="AZ 임계값 (회전 판정)")
                    ly_slider = gr.Slider(0.05, 0.5, value=_LY_DEFAULT, step=0.01,
                                         label="LY 임계값 (스트레이프 판정)")
                    lx_slider = gr.Slider(0.01, 0.3, value=_LX_DEFAULT, step=0.01,
                                         label="LX 임계값 (전진 판정)")

                btn_eval = gr.Button("▶️ 평가 시작", variant="primary")
                progress_box = gr.Textbox(label="진행 상황", interactive=False, lines=2)

            # ── 우측: 결과 ──
            with gr.Column(scale=2):
                summary_table = gr.Dataframe(
                    headers=["episode_id", "path_type", "frames", "grounding%", "agreement%", "verdict"],
                    label="에피소드 요약 (행 클릭 → 프레임 뷰어)",
                    interactive=False,
                    wrap=True,
                )

                with gr.Row():
                    frame_img = gr.Image(label="Frame + BBox Overlay", interactive=False,
                                         height=360)
                    frame_info = gr.Textbox(label="프레임 정보", interactive=False,
                                            lines=7)

                frame_slider = gr.Slider(minimum=0, maximum=0, step=1,
                                         label="Frame Index", value=0)

                with gr.Row():
                    btn_export = gr.Button("💾 JSON 내보내기", variant="secondary")
                    export_path = gr.Textbox(label="저장 경로", interactive=False)

        # ── 이벤트 연결 ──
        btn_scan.click(
            fn=scan_episodes,
            inputs=[scan_dir],
            outputs=[ep_dropdown],
        )

        btn_eval.click(
            fn=run_eval,
            inputs=[ep_dropdown, scan_dir, az_slider, ly_slider, lx_slider, state],
            outputs=[summary_table, progress_box, state],
        )

        summary_table.select(
            fn=on_table_select,
            inputs=[state],
            outputs=[frame_img, frame_slider, frame_info, state],
        )

        frame_slider.change(
            fn=on_frame_slide,
            inputs=[frame_slider, state],
            outputs=[frame_img, frame_info],
        )

        btn_export.click(
            fn=export_json,
            inputs=[state],
            outputs=[export_path],
        )

    return demo


# ── 진입점 ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)

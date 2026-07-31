"""
필터 OFF 상태로 PG2-448 grounding을 재현해서 확인하는 로컬 Gradio 대시보드.

목적: session_20260702_212648.h5의 t9/t12/t15 (동일 구도인데 has_bbox flicker)를
      raw locs 그대로 보여줘서 min_area/min_cy 필터에 걸린 건지, 진짜 미탐지인지 확인.

실행: python3 scripts/debug_pg2_filter_off.py
"""
import re
import sys
from pathlib import Path

import gradio as gr
import h5py
import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor

PG2_PATH = "google/paligemma2-3b-mix-448"
_LOC_RE = re.compile(r"<loc(\d{4})>")

RECV_ROOT = Path("/home/minum/MoNaVLA/inference_sessions_recv")
RECV_DIRS = [RECV_ROOT / "20260702", RECV_ROOT / "20260703"]

print("PG2-448 로딩 중...")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32
proc = PaliGemmaProcessor.from_pretrained(PG2_PATH)
model = PaliGemmaForConditionalGeneration.from_pretrained(PG2_PATH, torch_dtype=dtype).to(device).eval()
print(f"로딩 완료. device={device}")


def run_pg2_raw(pil_img: Image.Image, phrase: str = "gray basket"):
    """필터 없이 raw PG2 출력 그대로 반환."""
    inp = proc(text=f"detect {phrase}", images=pil_img, return_tensors="pt").to(device)
    inp["pixel_values"] = inp["pixel_values"].to(dtype)
    with torch.no_grad():
        gen = model.generate(**inp, max_new_tokens=48, min_new_tokens=1, do_sample=False)
    raw = proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=False)[0]
    locs = [int(v) / 1023.0 for v in _LOC_RE.findall(raw)]

    result = {"raw": raw, "locs": locs}
    if len(locs) >= 4:
        y1, x1, y2, x2 = locs[:4]
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        area = (x2 - x1) * (y2 - y1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        result.update(x1=x1, y1=y1, x2=x2, y2=y2, area=area, cx=cx, cy=cy)
        # 필터 조건 각각 개별 평가 (참고용, 실제로는 적용 안 함)
        result["would_filter_fullframe"] = area > 0.9
        result["would_filter_tiny"] = area < 0.01
        result["would_filter_top"] = cy < 0.35
        result["would_filter_xfull"] = (x1 < 0.02 and x2 > 0.98)
    else:
        result["note"] = "loc 토큰 4개 미만 — 파싱 실패 (진짜 미탐지)"
    return result


def draw_bbox(pil_img: Image.Image, result: dict) -> Image.Image:
    img = pil_img.copy()
    if "x1" not in result:
        return img
    draw = ImageDraw.Draw(img)
    w, h = img.size
    x1, y1, x2, y2 = result["x1"] * w, result["y1"] * h, result["x2"] * w, result["y2"] * h
    draw.rectangle([x1, y1, x2, y2], outline="red", width=4)
    draw.ellipse([result["cx"] * w - 6, result["cy"] * h - 6,
                  result["cx"] * w + 6, result["cy"] * h + 6], fill="yellow")
    return img


def _resolve_session(session_name: str) -> Path | None:
    for d in RECV_DIRS:
        p = d / session_name
        if p.exists():
            return p
    return None


def load_session_frame(session_name: str, frame_idx: int):
    fpath = _resolve_session(session_name)
    if fpath is None:
        return None, f"세션 없음: {session_name}"
    with h5py.File(str(fpath), "r") as h:
        n = h["observations/images"].shape[0]
        if frame_idx >= n:
            return None, f"frame_idx {frame_idx} >= N({n})"
        img = h["observations/images"][frame_idx]
        cached = h["grounding/cached"][frame_idx]
        hasbbox = h["grounding/bbox"][frame_idx, 3]
        cx_orig = h["grounding/bbox"][frame_idx, 0]
        status = "LIVE" if cached == 0 else ("CACHE" if cached == 1 else "NONE")
        meta = f"서버 기록: {status}, has_bbox={bool(hasbbox)}, cx={cx_orig:.3f}"
    return Image.fromarray(img.astype(np.uint8)).convert("RGB"), meta


def analyze(session_name, frame_idx, phrase):
    pil, meta = load_session_frame(session_name, int(frame_idx))
    if pil is None:
        return None, meta, "{}"
    result = run_pg2_raw(pil, phrase)
    drawn = draw_bbox(pil, result)

    lines = [meta, f"phrase='{phrase}'", f"raw_output: {result['raw'][:200]}", f"locs: {result['locs']}"]
    if "area" in result:
        lines.append(f"cx={result['cx']:.4f} cy={result['cy']:.4f} area={result['area']:.4f}")
        lines.append(f"필터 없이 순수 판정: has_bbox=True (필터 적용시 지금 서버 로직 기준)")
        lines.append(f"  full-frame(>0.9): {result['would_filter_fullframe']}")
        lines.append(f"  tiny(<0.01): {result['would_filter_tiny']}")
        lines.append(f"  top(cy<0.35): {result['would_filter_top']}")
        lines.append(f"  x-full-width: {result['would_filter_xfull']}")
        any_filter = any([result['would_filter_fullframe'], result['would_filter_tiny'],
                           result['would_filter_top'], result['would_filter_xfull']])
        lines.append(f"→ 필터 적용시 최종 판정: has_bbox={'False (필터 걸림!)' if any_filter else 'True'}")
    else:
        lines.append(result.get("note", ""))

    return drawn, "\n".join(lines), str(result)


with gr.Blocks(title="PG2 필터 OFF 디버그") as demo:
    gr.Markdown("# PG2-448 필터 OFF 원본 탐지 확인\n"
                "t9/t12/t15 동일 구도 flicker 원인 확인용. "
                "필터 적용 전 raw locs를 그대로 보여주고, 4개 필터 각각의 통과 여부를 개별 표시.")
    with gr.Row():
        with gr.Column():
            _choices = sorted(p.name for d in RECV_DIRS for p in d.glob("session_*.h5"))
            session_dd = gr.Dropdown(
                choices=_choices,
                value=_choices[-1] if _choices else None,
                label="세션 파일",
            )
            frame_slider = gr.Slider(0, 40, value=9, step=1, label="frame_idx")
            phrase_tb = gr.Textbox(value="gray basket", label="grounding phrase")
            run_btn = gr.Button("분석 실행", variant="primary")
        with gr.Column():
            out_img = gr.Image(label="bbox 시각화")
    out_text = gr.Textbox(label="분석 결과", lines=12)
    out_raw = gr.Textbox(label="raw dict", lines=4)

    run_btn.click(analyze, [session_dd, frame_slider, phrase_tb], [out_img, out_text, out_raw])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)

#!/usr/bin/env python3
"""MoNaVLA 데이터셋 뷰어 (독립 페이지) — 포트 8083.

H5 에피소드 파일을 리스트업 → 선택 → 실제 프레임을 액션 라벨과 함께 스크롤 갤러리로 확인.
ROS/카메라 불필요 (디스크의 H5만 읽음). JPEG(vlen) / raw 저장 둘 다 자동 디코딩.

실행:
  python3 scripts/gradio_dataset_viewer.py
  (VLA_VIEWER_PORT 로 포트 변경 가능, 기본 8083)
"""
import os
import glob
import socket

import cv2
import h5py
import numpy as np
import gradio as gr
from PIL import Image

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROS_ROOT = os.path.join(_PROJECT_ROOT, "ROS_action")

CLASS_NAMES_8 = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "ROT_L", "ROT_R"]
CLASS_SYMBOLS = {0: "●", 1: "▲", 2: "◀", 3: "▶", 4: "↖", 5: "↗", 6: "↺", 7: "↻"}
THUMB_W = 240  # 갤러리 썸네일 가로 px


def classify_8class(action):
    x = float(action[0]); y = float(action[1])
    az = float(action[2]) if len(action) > 2 else 0.0
    is_x, is_y = abs(x) > 0.3, abs(y) > 0.3
    if not is_x and not is_y:
        if az > 0.1:  return 6
        if az < -0.1: return 7
        return 0
    if x > 0.3:
        if y > 0.3:  return 4
        if y < -0.3: return 5
        return 1
    if abs(x) < 0.3:
        if y > 0.3:  return 2
        if y < -0.3: return 3
        return 0
    return 0  # backward 등 → STOP (학습 분류와 동일)


def list_dataset_roots():
    """ROS_action/mobile_vla_dataset_* 디렉토리 목록 (H5 보유한 것)."""
    roots = []
    for d in sorted(glob.glob(os.path.join(_ROS_ROOT, "mobile_vla_dataset_*"))):
        if os.path.isdir(d) and glob.glob(os.path.join(d, "*.h5")):
            roots.append(os.path.basename(d))
    return roots


def list_episodes(root_name):
    """선택 데이터셋의 H5 파일 (최신순)."""
    if not root_name:
        return []
    root = os.path.join(_ROS_ROOT, root_name)
    files = glob.glob(os.path.join(root, "*.h5"))
    files.sort(key=os.path.getmtime, reverse=True)
    return [os.path.basename(f) for f in files]


def _decode_frame(src, i):
    """vlen JPEG / raw 자동 디코딩 → RGB uint8 array."""
    raw = src[i]
    if getattr(raw, "ndim", 3) == 1:  # JPEG bytes
        arr = cv2.imdecode(np.asarray(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        return arr  # save_h5가 RGB를 인코딩했으므로 그대로 RGB
    return np.asarray(raw)


def load_episode(root_name, fname):
    """선택 에피소드 → (갤러리 항목 리스트, 정보 markdown)."""
    if not root_name or not fname:
        return [], "_(에피소드를 선택하세요)_"
    path = os.path.join(_ROS_ROOT, root_name, fname)
    if not os.path.exists(path):
        return [], f"⚠️ 파일 없음: {fname}"
    try:
        with h5py.File(path, "r") as h:
            src = h["observations"]["images"] if "observations" in h else h["images"]
            n = len(src)
            acts = h["actions"][:] if "actions" in h else np.zeros((n, 3))
            fmt = src.attrs.get("format", "raw")
            instr = ""
            if "language_instruction" in h:
                v = h["language_instruction"][()]
                try:
                    instr = (v[0] if hasattr(v, "__len__") and not isinstance(v, bytes) else v).decode("utf-8")
                except Exception:
                    instr = str(v)
            attrs = {k: h.attrs[k] for k in h.attrs}

            gallery = []
            counts = [0] * 8
            for i in range(n):
                img = _decode_frame(src, i)
                if img is None:
                    continue
                a = acts[i] if i < len(acts) else [0, 0, 0]
                cls = classify_8class(a)
                counts[cls] += 1
                # 후진(backward) 별도 표기
                bk = " ⚠후진" if float(a[0]) < -0.3 else ""
                thumb = cv2.resize(img, (THUMB_W, int(THUMB_W * img.shape[0] / img.shape[1])))
                cap = f"[{i}] {CLASS_SYMBOLS[cls]}{CLASS_NAMES_8[cls]}{bk}"
                gallery.append((Image.fromarray(thumb.astype(np.uint8)), cap))
    except Exception as e:
        return [], f"❌ 로드 실패: {e}"

    total = sum(counts) or 1
    dist = "\n".join(
        f"{CLASS_SYMBOLS[i]} {CLASS_NAMES_8[i]:8s} {c/total*100:5.1f}% {'█'*int(c/total*100/5)} ({c})"
        for i, c in enumerate(counts) if c > 0
    )
    size_kb = os.path.getsize(path) // 1024
    info = (
        f"### 📄 {fname}\n"
        f"- 프레임: **{n}**  ·  저장포맷: **{fmt}**  ·  파일크기: {size_kb}KB\n"
        f"- scenario: `{attrs.get('scenario','?')}`  end_pos: `{attrs.get('end_pos','?')}`  "
        f"pattern: `{attrs.get('pattern','?')}`\n"
        f"- instruction: {instr or '—'}\n\n"
        f"**액션 분포**\n```\n{dist}\n```"
    )
    return gallery, info


def delete_episode(root_name, fname):
    """선택 에피소드 파일 삭제 → 리스트에서 즉시 제거하고 다음 에피소드 로드."""
    if not root_name or not fname:
        return gr.update(), [], "_(선택된 에피소드 없음)_"
    path = os.path.join(_ROS_ROOT, root_name, fname)
    try:
        if os.path.exists(path):
            os.remove(path)
            msg = f"🗑️ 삭제 완료: {fname}"
        else:
            msg = f"⚠️ 이미 없음: {fname}"
    except Exception as e:
        return gr.update(), [], f"❌ 삭제 실패: {e}"
    eps = list_episodes(root_name)          # 삭제 후 재조회 → 즉시 반영
    nxt = eps[0] if eps else None
    gallery, info = load_episode(root_name, nxt) if nxt else ([], "_(에피소드 없음)_")
    return gr.update(choices=eps, value=nxt), gallery, f"{msg}\n\n{info}"


def show_full(root_name, fname, evt: gr.SelectData):
    """갤러리에서 클릭한 프레임을 원본 화질(1280×720)로 로드."""
    if evt is None or not root_name or not fname:
        return None, ""
    i = int(evt.index)
    path = os.path.join(_ROS_ROOT, root_name, fname)
    try:
        with h5py.File(path, "r") as h:
            src = h["observations"]["images"] if "observations" in h else h["images"]
            img = _decode_frame(src, i)
            acts = h["actions"][:] if "actions" in h else None
        a = acts[i] if (acts is not None and i < len(acts)) else [0.0, 0.0, 0.0]
        cls = classify_8class(a)
        bk = " ⚠후진" if float(a[0]) < -0.3 else ""
        cap = (f"### 🔍 프레임 [{i}]  {CLASS_SYMBOLS[cls]} {CLASS_NAMES_8[cls]}{bk}\n"
               f"action `[{a[0]:+.2f}, {a[1]:+.2f}, {a[2]:+.2f}]`  ·  원본 {img.shape[1]}×{img.shape[0]}")
        return Image.fromarray(img.astype(np.uint8)), cap
    except Exception as e:
        return None, f"❌ {e}"


def refresh_roots():
    roots = list_dataset_roots()
    default = "mobile_vla_dataset_v5_2" if "mobile_vla_dataset_v5_2" in roots else (roots[0] if roots else None)
    eps = list_episodes(default)
    return (gr.update(choices=roots, value=default),
            gr.update(choices=eps, value=(eps[0] if eps else None)))


def on_root_change(root_name):
    eps = list_episodes(root_name)
    return gr.update(choices=eps, value=(eps[0] if eps else None))


def pick_port(default_port, span=20):
    for p in range(default_port, default_port + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", p)); return p
            except OSError:
                continue
    return default_port


CSS = """
.gradio-container { background:#0d1117 !important; color:#c9d1d9 !important; }
"""

with gr.Blocks(title="MoNaVLA Dataset Viewer", css=CSS) as demo:
    gr.Markdown("# 📂 MoNaVLA 데이터셋 뷰어 — 에피소드 프레임 확인")
    _roots = list_dataset_roots()
    _default_root = "mobile_vla_dataset_v5_2" if "mobile_vla_dataset_v5_2" in _roots else (_roots[0] if _roots else None)
    _default_eps = list_episodes(_default_root)

    with gr.Row():
        root_dd = gr.Dropdown(choices=_roots, value=_default_root, label="📁 데이터셋", scale=2)
        episode_dd = gr.Dropdown(choices=_default_eps,
                                 value=(_default_eps[0] if _default_eps else None),
                                 label="🎬 에피소드 (최신순)", scale=4)
        refresh_btn = gr.Button("🔄 새로고침", scale=1)
        del_btn = gr.Button("🗑️ 이 에피소드 삭제", variant="stop", scale=1)

    info_md = gr.Markdown("_(에피소드를 선택하세요)_")
    gallery = gr.Gallery(label="프레임 (액션 라벨 · ⚠후진 표시) — 클릭하면 아래 원본 화질", columns=6,
                         height=460, object_fit="contain", show_label=True)

    full_info = gr.Markdown("_(썸네일을 클릭하면 원본 화질로 표시)_")
    full_view = gr.Image(label="🔍 원본 화질 프레임", interactive=False, height=520)

    # 이벤트
    root_dd.change(fn=on_root_change, inputs=root_dd, outputs=episode_dd)
    episode_dd.change(fn=load_episode, inputs=[root_dd, episode_dd], outputs=[gallery, info_md])
    refresh_btn.click(fn=refresh_roots, outputs=[root_dd, episode_dd])
    del_btn.click(fn=delete_episode, inputs=[root_dd, episode_dd], outputs=[episode_dd, gallery, info_md])
    gallery.select(fn=show_full, inputs=[root_dd, episode_dd], outputs=[full_view, full_info])
    # 시작 시 첫 에피소드 자동 로드
    demo.load(fn=load_episode, inputs=[root_dd, episode_dd], outputs=[gallery, info_md])


if __name__ == "__main__":
    port = pick_port(int(os.getenv("VLA_VIEWER_PORT", "8083")))
    demo.launch(server_name="0.0.0.0", server_port=port)

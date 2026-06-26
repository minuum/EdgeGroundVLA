#!/usr/bin/env python3
"""MoNaVLA 데이터셋 뷰어 v2 — 포트 8083.

지원 포맷:
  - V3  : images (N,H,W,3) raw  /  action_event_types
  - V5  : observations/images (N,H,W,3) raw  /  scenario attrs
  - V5_2: observations/images (N,) object JPEG vlen

UX 개선:
  - 시나리오 필터 + 에피소드 페이지네이션
  - 데이터셋 전체 통계 (시나리오 분포)
  - 이전/다음 에피소드 버튼
  - 프레임 범위 슬라이더 (긴 에피소드 빠른 탐색)
  - 클릭 → 원본 화질 즉시 표시
  - 삭제 확인 없이 빠른 정리 가능
"""
import os, glob, socket
from collections import Counter
from pathlib import Path

import cv2
import h5py
import numpy as np
import gradio as gr
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ROS_ROOT     = _PROJECT_ROOT / "ROS_action"

CLASS_NAMES = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "ROT_L", "ROT_R"]
CLASS_SYM   = {0:"●", 1:"▲", 2:"◀", 3:"▶", 4:"↖", 5:"↗", 6:"↺", 7:"↻"}
THUMB_W = 200

# ── 유틸 ────────────────────────────────────────────────────────────────────

def classify_8(action):
    x, y = float(action[0]), float(action[1])
    az = float(action[2]) if len(action) > 2 else 0.0
    if abs(x) < 0.3 and abs(y) < 0.3:
        return 6 if az > 0.1 else (7 if az < -0.1 else 0)
    if x > 0.3:
        return 4 if y > 0.3 else (5 if y < -0.3 else 1)
    return 2 if y > 0.3 else (3 if y < -0.3 else 0)


def _decode(src, i):
    raw = src[i]
    if getattr(raw, "ndim", 3) == 1:
        arr = cv2.imdecode(np.asarray(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(arr, cv2.COLOR_BGR2RGB) if arr is not None else None
    return np.asarray(raw)


def _thumb(img, w=THUMB_W):
    h = int(w * img.shape[0] / img.shape[1])
    return cv2.resize(img, (w, h))


def _detect_version(h5):
    if "observations" in h5:
        return "v5"
    if "images" in h5:
        return "v3"
    return "unknown"


# ── 데이터셋 목록 ─────────────────────────────────────────────────────────

def list_datasets():
    results = []
    for d in sorted(_ROS_ROOT.glob("mobile_vla_dataset*")):
        if not d.is_dir():
            continue
        h5s = list(d.glob("*.h5"))
        if h5s:
            results.append((d.name, len(h5s)))
    return results  # [(name, count), ...]


def dataset_display_choices():
    ds = list_datasets()
    return [f"{name}  ({cnt}ep)" for name, cnt in ds], [name for name, _ in ds]


# ── 시나리오 통계 ─────────────────────────────────────────────────────────

def dataset_stats(ds_name: str) -> tuple[str, list]:
    """(markdown 통계 문자열, scenario 선택지 리스트)"""
    if not ds_name:
        return "_(데이터셋을 선택하세요)_", ["(전체)"]
    root = _ROS_ROOT / ds_name
    files = list(root.glob("*.h5"))
    scenarios = Counter()
    bad = 0
    for f in files:
        try:
            with h5py.File(f) as h:
                sc = h.attrs.get("scenario", "")
                # V3: scenario 없으면 파일명에서 추출
                if not sc:
                    parts = f.stem.split("_")
                    sc = "_".join(parts[2:5]) if len(parts) > 4 else "unknown"
                scenarios[sc] += 1
        except Exception:
            bad += 1

    total = sum(scenarios.values())
    lines = [f"### 📊 {ds_name}", f"총 **{total}** 에피소드  (손상: {bad})\n"]
    bar_max = max(scenarios.values()) if scenarios else 1
    for sc, cnt in sorted(scenarios.items(), key=lambda x: -x[1]):
        bar = "█" * int(cnt / bar_max * 20)
        lines.append(f"`{sc:40s}` {cnt:3d}  {bar}")

    sc_choices = ["(전체)"] + sorted(scenarios.keys())
    return "\n".join(lines), sc_choices


# ── 에피소드 목록 ────────────────────────────────────────────────────────

def list_episodes(ds_name: str, scenario_filter: str = "(전체)") -> list[str]:
    if not ds_name:
        return []
    root = _ROS_ROOT / ds_name
    files = sorted(root.glob("*.h5"), key=os.path.getmtime, reverse=True)
    result = []
    for f in files:
        if scenario_filter and scenario_filter != "(전체)":
            if scenario_filter not in f.name:
                # attrs 확인
                try:
                    with h5py.File(f) as h:
                        if h.attrs.get("scenario", "") != scenario_filter:
                            continue
                except Exception:
                    continue
        result.append(f.name)
    return result


# ── 에피소드 로드 ────────────────────────────────────────────────────────

def load_episode(ds_name: str, fname: str, frame_start: int = 0, frame_end: int = -1):
    if not ds_name or not fname:
        return [], "_(에피소드를 선택하세요)_", 0, 1
    path = _ROS_ROOT / ds_name / fname
    if not path.exists():
        return [], f"⚠️ 파일 없음: {fname}", 0, 1
    try:
        with h5py.File(path) as h:
            ver = _detect_version(h)
            src = h["observations"]["images"] if ver == "v5" else h["images"]
            n_total = len(src)
            acts = h["actions"][:] if "actions" in h else np.zeros((n_total, 3))
            fmt = src.attrs.get("format", "raw") if hasattr(src, "attrs") else "raw"

            instr = ""
            if "language_instruction" in h:
                v = h["language_instruction"][()]
                try:
                    instr = (v[0] if hasattr(v, "__len__") and not isinstance(v, bytes) else v).decode()
                except Exception:
                    instr = str(v)

            attrs = dict(h.attrs)
            act_types = None
            if "action_event_types" in h:
                act_types = [x.decode() if isinstance(x, bytes) else str(x) for x in h["action_event_types"][:]]

            fe = n_total if frame_end < 0 else min(frame_end, n_total)
            fs = max(0, min(frame_start, fe - 1))

            gallery = []
            counts = [0] * 8
            for i in range(n_total):
                a = acts[i] if i < len(acts) else [0, 0, 0]
                cls = classify_8(a)
                counts[cls] += 1

            for i in range(fs, fe):
                img = _decode(src, i)
                if img is None:
                    continue
                a = acts[i] if i < len(acts) else [0, 0, 0]
                cls = classify_8(a)
                bk = " ⚠" if float(a[0]) < -0.3 else ""
                label_extra = f" [{act_types[i]}]" if act_types and i < len(act_types) else ""
                th = _thumb(img)
                cap = f"[{i}] {CLASS_SYM[cls]}{CLASS_NAMES[cls]}{bk}{label_extra}"
                gallery.append((Image.fromarray(th.astype(np.uint8)), cap))

    except Exception as e:
        return [], f"❌ 로드 실패: {e}", 0, 1

    total = sum(counts) or 1
    dist = "\n".join(
        f"{CLASS_SYM[i]} {CLASS_NAMES[i]:8s} {c/total*100:5.1f}% {'█'*int(c/total*100/5)} ({c})"
        for i, c in enumerate(counts) if c > 0
    )
    size_kb = path.stat().st_size // 1024
    scenario = attrs.get("scenario", Path(fname).stem)
    info = (
        f"### 📄 `{fname}`\n"
        f"- 포맷: **{ver}** / 이미지저장: **{fmt}**  ·  크기: {size_kb} KB\n"
        f"- 전체 프레임: **{n_total}**  (표시: {fs}~{fe-1})\n"
        f"- scenario: `{scenario}`  ·  pattern: `{attrs.get('pattern','?')}`  "
        f"·  end_pos: `{attrs.get('end_pos','?')}`\n"
        f"- instruction: {instr or '—'}\n\n"
        f"**전체 액션 분포**\n```\n{dist}\n```"
    )
    return gallery, info, n_total, fe


# ── 원본 프레임 ──────────────────────────────────────────────────────────

def show_full(ds_name: str, fname: str, frame_start: int, evt: gr.SelectData):
    if evt is None or not ds_name or not fname:
        return None, ""
    i = int(evt.index) + frame_start
    path = _ROS_ROOT / ds_name / fname
    try:
        with h5py.File(path) as h:
            ver = _detect_version(h)
            src = h["observations"]["images"] if ver == "v5" else h["images"]
            img = _decode(src, i)
            acts = h["actions"][:] if "actions" in h else None
            act_types = None
            if "action_event_types" in h:
                act_types = [x.decode() if isinstance(x, bytes) else str(x) for x in h["action_event_types"][:]]
        a = acts[i] if (acts is not None and i < len(acts)) else [0., 0., 0.]
        cls = classify_8(a)
        bk = "  ⚠️ 후진" if float(a[0]) < -0.3 else ""
        at = f"  |  event: `{act_types[i]}`" if act_types and i < len(act_types) else ""
        cap = (
            f"### 🔍 프레임 [{i}]  {CLASS_SYM[cls]} **{CLASS_NAMES[cls]}**{bk}\n"
            f"action `[{a[0]:+.3f}, {a[1]:+.3f}, {a[2]:+.3f}]`{at}  ·  {img.shape[1]}×{img.shape[0]}"
        )
        return Image.fromarray(img.astype(np.uint8)), cap
    except Exception as e:
        return None, f"❌ {e}"


# ── 삭제 ────────────────────────────────────────────────────────────────

def delete_episode(ds_name, fname, scenario_filter):
    if not ds_name or not fname:
        return gr.update(), [], "_(선택 없음)_"
    path = _ROS_ROOT / ds_name / fname
    try:
        if path.exists():
            path.unlink()
            msg = f"🗑️ 삭제: `{fname}`"
        else:
            msg = f"⚠️ 이미 없음: `{fname}`"
    except Exception as e:
        return gr.update(), [], f"❌ 삭제 실패: {e}"
    eps = list_episodes(ds_name, scenario_filter)
    nxt = eps[0] if eps else None
    gal, info, n, fe = load_episode(ds_name, nxt) if nxt else ([], "_(없음)_", 0, 1)
    return gr.update(choices=eps, value=nxt), gal, f"{msg}\n\n{info}"


# ── 이전/다음 에피소드 ───────────────────────────────────────────────────

def nav_episode(ds_name, fname, scenario_filter, direction: int):
    eps = list_episodes(ds_name, scenario_filter)
    if not eps or fname not in eps:
        return gr.update(), [], "_(없음)_", 0, 1
    idx = eps.index(fname)
    nidx = max(0, min(len(eps) - 1, idx + direction))
    nxt = eps[nidx]
    gal, info, n, fe = load_episode(ds_name, nxt)
    return gr.update(value=nxt), gal, info, n, fe


# ── 포트 선택 ────────────────────────────────────────────────────────────

def pick_port(default, span=20):
    for p in range(default, default + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("0.0.0.0", p)); return p
            except OSError:
                continue
    return default


# ── UI ──────────────────────────────────────────────────────────────────

_CSS = """
.gradio-container { max-width: 1600px !important; }
#stats-md { background: #0d1117; border-radius: 8px; padding: 12px; font-family: monospace; font-size: 0.85rem; }
#ep-info  { font-size: 0.9rem; }
#gallery  { min-height: 300px; }
.nav-btn  { min-width: 80px !important; }
"""

_ds_labels, _ds_values = dataset_display_choices()
_default_ds = next((v for v in _ds_values if "v5_2" in v), (_ds_values[0] if _ds_values else None))
_init_sc    = ["(전체)"]   # 실제 통계는 demo.load에서 lazy 로드
_init_eps   = []
_init_ep    = None

with gr.Blocks(
    title="MoNaVLA Dataset Viewer",
    css=_CSS,
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.indigo,
        neutral_hue=gr.themes.colors.slate,
    ),
) as demo:
    gr.Markdown("# 📂 MoNaVLA 데이터셋 뷰어")

    # ── 데이터셋 선택 행 ─────────────────────────────────────────────
    with gr.Row():
        ds_dd = gr.Dropdown(
            choices=_ds_labels, value=(_ds_labels[_ds_values.index(_default_ds)] if _default_ds else None),
            label="📁 데이터셋", scale=3,
        )
        sc_dd = gr.Dropdown(choices=_init_sc, value="(전체)", label="🗂 시나리오 필터", scale=2)
        ep_dd = gr.Dropdown(choices=_init_eps, value=_init_ep, label="🎬 에피소드", scale=4)
        refresh_btn = gr.Button("🔄", scale=0, min_width=50)

    # ── 프레임 범위 + 이전/다음/삭제 ────────────────────────────────
    with gr.Row():
        frame_range = gr.Slider(minimum=0, maximum=1, value=0, step=1,
                                label="표시 시작 프레임", scale=5, interactive=True)
        frame_end   = gr.Slider(minimum=1, maximum=1, value=1, step=1,
                                label="표시 끝 프레임",  scale=5, interactive=True)
        prev_btn = gr.Button("⬅ 이전", elem_classes="nav-btn", scale=1)
        next_btn = gr.Button("다음 ➡", elem_classes="nav-btn", scale=1)
        del_btn  = gr.Button("🗑️ 삭제", variant="stop", scale=1)

    with gr.Row(equal_height=False):
        # 왼쪽: 갤러리
        with gr.Column(scale=3):
            gallery = gr.Gallery(
                label="프레임 갤러리 (클릭 → 원본 화질)", elem_id="gallery",
                columns=5, height=420, object_fit="contain", show_label=True,
            )
            full_info = gr.Markdown("_(썸네일 클릭 → 원본 화질)_")
            full_view = gr.Image(label="🔍 원본 화질", interactive=False, height=480)

        # 오른쪽: 에피소드 정보 + 데이터셋 통계
        with gr.Column(scale=1):
            ep_info_md  = gr.Markdown("_(에피소드를 선택하세요)_", elem_id="ep-info")
            gr.Markdown("---")
            stats_md = gr.Markdown("_(데이터셋 선택 후 통계 로드)_", elem_id="stats-md")

    # ── 내부 상태 ────────────────────────────────────────────────────
    _ds_map   = gr.State({v: v for v, l in zip(_ds_values, _ds_labels)})  # label→name
    _n_frames = gr.State(1)

    # ── 헬퍼: label→name 변환 ────────────────────────────────────────
    def _ds_name(label):
        for v, l in zip(_ds_values, _ds_labels):
            if l == label or v == label:
                return v
        return label  # fallback

    # ── 이벤트: 데이터셋 변경 ────────────────────────────────────────
    def on_ds_change(ds_label):
        name = _ds_name(ds_label)
        stats, scs = dataset_stats(name)
        eps = list_episodes(name, "(전체)")
        ep = eps[0] if eps else None
        return (
            gr.update(choices=scs, value="(전체)"),
            gr.update(choices=eps, value=ep),
            stats,
        )

    ds_dd.change(
        fn=on_ds_change, inputs=ds_dd,
        outputs=[sc_dd, ep_dd, stats_md],
    )

    # ── 이벤트: 시나리오 필터 변경 ───────────────────────────────────
    def on_sc_change(ds_label, sc):
        name = _ds_name(ds_label)
        eps = list_episodes(name, sc)
        return gr.update(choices=eps, value=(eps[0] if eps else None))

    sc_dd.change(fn=on_sc_change, inputs=[ds_dd, sc_dd], outputs=ep_dd)

    # ── 이벤트: 에피소드 변경 ────────────────────────────────────────
    def on_ep_change(ds_label, fname):
        name = _ds_name(ds_label)
        gal, info, n, fe = load_episode(name, fname, 0, -1)
        return gal, info, gr.update(minimum=0, maximum=max(0,n-1), value=0), gr.update(minimum=1, maximum=n, value=n), n

    ep_dd.change(
        fn=on_ep_change, inputs=[ds_dd, ep_dd],
        outputs=[gallery, ep_info_md, frame_range, frame_end, _n_frames],
    )

    # ── 이벤트: 프레임 범위 변경 ─────────────────────────────────────
    def on_range_change(ds_label, fname, fs, fe):
        name = _ds_name(ds_label)
        gal, info, n, _ = load_episode(name, fname, int(fs), int(fe))
        return gal, info

    frame_range.release(fn=on_range_change, inputs=[ds_dd, ep_dd, frame_range, frame_end], outputs=[gallery, ep_info_md])
    frame_end.release(fn=on_range_change,   inputs=[ds_dd, ep_dd, frame_range, frame_end], outputs=[gallery, ep_info_md])

    # ── 이벤트: 원본 화질 ─────────────────────────────────────────────
    def _show_full(ds_label, fname, fs, evt: gr.SelectData):
        return show_full(_ds_name(ds_label), fname, int(fs), evt)

    gallery.select(fn=_show_full, inputs=[ds_dd, ep_dd, frame_range], outputs=[full_view, full_info])

    # ── 이벤트: 삭제 ─────────────────────────────────────────────────
    def on_delete(ds_label, fname, sc):
        name = _ds_name(ds_label)
        ep_upd, gal, info = delete_episode(name, fname, sc)
        return ep_upd, gal, info

    del_btn.click(fn=on_delete, inputs=[ds_dd, ep_dd, sc_dd], outputs=[ep_dd, gallery, ep_info_md])

    # ── 이벤트: 이전/다음 ────────────────────────────────────────────
    def on_prev(ds_label, fname, sc):
        name = _ds_name(ds_label)
        ep_upd, gal, info, n, fe = nav_episode(name, fname, sc, -1)
        return ep_upd, gal, info, gr.update(minimum=0, maximum=max(0,n-1), value=0), gr.update(minimum=1, maximum=n, value=n)

    def on_next(ds_label, fname, sc):
        name = _ds_name(ds_label)
        ep_upd, gal, info, n, fe = nav_episode(name, fname, sc, +1)
        return ep_upd, gal, info, gr.update(minimum=0, maximum=max(0,n-1), value=0), gr.update(minimum=1, maximum=n, value=n)

    prev_btn.click(fn=on_prev, inputs=[ds_dd, ep_dd, sc_dd], outputs=[ep_dd, gallery, ep_info_md, frame_range, frame_end])
    next_btn.click(fn=on_next, inputs=[ds_dd, ep_dd, sc_dd], outputs=[ep_dd, gallery, ep_info_md, frame_range, frame_end])

    # ── 새로고침 ──────────────────────────────────────────────────────
    def on_refresh(ds_label):
        name = _ds_name(ds_label)
        stats, scs = dataset_stats(name)
        eps = list_episodes(name, "(전체)")
        return gr.update(choices=scs, value="(전체)"), gr.update(choices=eps, value=(eps[0] if eps else None)), stats

    refresh_btn.click(fn=on_refresh, inputs=ds_dd, outputs=[sc_dd, ep_dd, stats_md])

    # ── 초기 로드 (lazy) ─────────────────────────────────────────────
    def _init():
        name = _default_ds
        stats, scs = dataset_stats(name)
        eps = list_episodes(name, "(전체)")
        ep  = eps[0] if eps else None
        gal, info, n, fe = load_episode(name, ep, 0, -1) if ep else ([], "_(없음)_", 0, 1)
        return (
            gr.update(choices=scs, value="(전체)"),
            gr.update(choices=eps, value=ep),
            gal, info, stats,
            gr.update(minimum=0, maximum=max(0,n-1), value=0),
            gr.update(minimum=1, maximum=n, value=n),
            n,
        )

    demo.load(fn=_init, outputs=[sc_dd, ep_dd, gallery, ep_info_md, stats_md, frame_range, frame_end, _n_frames])


if __name__ == "__main__":
    port = pick_port(int(os.getenv("VLA_VIEWER_PORT", "8083")))
    print(f"✅ Dataset Viewer → http://0.0.0.0:{port}")
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)

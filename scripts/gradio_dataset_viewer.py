#!/usr/bin/env python3
"""MoNaVLA 데이터셋 뷰어 v3 — 포트 8083.

탭 1  📂 데이터셋  — H5 에피소드 갤러리
탭 2  📊 세션 로그 — episode_log.csv / grounding / drift JSONL

지원 H5 포맷:
  V3  : images (N,H,W,3) raw  /  action_event_types
  V5  : observations/images (N,H,W,3) raw  /  scenario attrs
  V5_2: observations/images (N,) object JPEG vlen
"""
import csv
import json
import os
import socket
from collections import Counter
from pathlib import Path

import cv2
import h5py
import numpy as np
import gradio as gr
from PIL import Image

_PROJECT_ROOT     = Path(__file__).resolve().parents[1]
_ROS_ROOT         = _PROJECT_ROOT / "ROS_action"
_LOG_DIR          = _PROJECT_ROOT / "logs"
_EPISODE_CSV      = _LOG_DIR / "episode_log.csv"
_GND_DIR          = _LOG_DIR / "grounding_sessions"
_DRIFT_DIR        = _LOG_DIR / "drift_sessions"
_INFER_API_URL    = os.getenv("VLA_API_URL", "http://localhost:8001")
_INFER_API_KEY    = os.getenv("VLA_API_KEY", "")
_CALIB_DIR        = _LOG_DIR / "calib_sessions"
_INFER_H5_DIR     = _PROJECT_ROOT / "docs" / "inference_sessions"   # H5
_INFER_REPORT_DIR = _PROJECT_ROOT / "docs" / "inference_reports"    # JSON

_INFER_DS_KEY = "inference_sessions"   # 데이터셋 드롭다운에 표시되는 키

def _ds_root(ds_name: str) -> Path:
    """ds_name → 실제 디렉터리 경로."""
    if ds_name == _INFER_DS_KEY:
        return _INFER_H5_DIR
    return _ROS_ROOT / ds_name

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
    # 추론 세션 H5
    if _INFER_H5_DIR.exists():
        h5s = list(_INFER_H5_DIR.glob("session_*.h5"))
        if h5s:
            results.append((_INFER_DS_KEY, len(h5s)))
    return results


def dataset_display_choices():
    ds = list_datasets()
    return [f"{name}  ({cnt}ep)" for name, cnt in ds], [name for name, _ in ds]


# ── 시나리오 통계 ─────────────────────────────────────────────────────────

def dataset_stats(ds_name: str):
    if not ds_name:
        return "_(데이터셋을 선택하세요)_", ["(전체)"]
    root = _ds_root(ds_name)
    is_infer = ds_name == _INFER_DS_KEY
    files = list(root.glob("*.h5"))
    scenarios = Counter()
    bad = 0
    for f in files:
        try:
            with h5py.File(f) as h:
                if is_infer:
                    sc = h.attrs.get("model_name", "unknown")[:30] or "unknown"
                else:
                    sc = h.attrs.get("scenario", "")
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

    return "\n".join(lines), ["(전체)"] + sorted(scenarios.keys())


# ── 에피소드 목록 ────────────────────────────────────────────────────────

def list_episodes(ds_name: str, scenario_filter: str = "(전체)"):
    if not ds_name:
        return []
    root = _ds_root(ds_name)
    files = sorted(root.glob("*.h5"), key=os.path.getmtime, reverse=True)
    result = []
    is_infer = ds_name == _INFER_DS_KEY
    for f in files:
        if scenario_filter and scenario_filter != "(전체)":
            try:
                with h5py.File(f) as h:
                    attr_val = h.attrs.get("model_name", "") if is_infer else h.attrs.get("scenario", "")
                    if attr_val != scenario_filter and scenario_filter not in f.name:
                        continue
            except Exception:
                continue
        result.append(f.name)
    return result


# ── 에피소드 로드 ────────────────────────────────────────────────────────

def load_episode(ds_name: str, fname: str, frame_start: int = 0, frame_end: int = -1):
    if not ds_name or not fname:
        return [], "_(에피소드를 선택하세요)_", 0, 1
    path = _ds_root(ds_name) / fname
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

            counts = [0] * 8
            for i in range(n_total):
                counts[classify_8(acts[i] if i < len(acts) else [0,0,0])] += 1

            gallery = []
            for i in range(fs, fe):
                img = _decode(src, i)
                if img is None:
                    continue
                a = acts[i] if i < len(acts) else [0, 0, 0]
                cls = classify_8(a)
                bk = " ⚠" if float(a[0]) < -0.3 else ""
                le = f" [{act_types[i]}]" if act_types and i < len(act_types) else ""
                gallery.append((Image.fromarray(_thumb(img).astype(np.uint8)),
                                f"[{i}] {CLASS_SYM[cls]}{CLASS_NAMES[cls]}{bk}{le}"))
    except Exception as e:
        return [], f"❌ 로드 실패: {e}", 0, 1

    total = sum(counts) or 1
    dist = "\n".join(
        f"{CLASS_SYM[i]} {CLASS_NAMES[i]:8s} {c/total*100:5.1f}% {'█'*int(c/total*100/5)} ({c})"
        for i, c in enumerate(counts) if c > 0
    )
    size_kb = path.stat().st_size // 1024
    if ds_name == _INFER_DS_KEY:
        instr_txt = attrs.get("instruction", "—")
        info = (
            f"### 🤖 `{fname}`\n"
            f"- 포맷: **{ver}**  ·  크기: {size_kb} KB\n"
            f"- 전체 프레임: **{n_total}**  (표시: {fs}~{fe-1})\n"
            f"- model: `{attrs.get('model_name','?')}`\n"
            f"- instruction: {instr_txt}\n"
            f"- status: `{attrs.get('status','?')}`\n\n"
            f"**전체 액션 분포**\n```\n{dist}\n```"
        )
    else:
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
    path = _ds_root(ds_name) / fname
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
    path = _ds_root(ds_name) / fname
    try:
        path.unlink(missing_ok=True)
        msg = f"🗑️ 삭제: `{fname}`"
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
    nxt = eps[max(0, min(len(eps) - 1, idx + direction))]
    gal, info, n, fe = load_episode(ds_name, nxt)
    return gr.update(value=nxt), gal, info, n, fe


# ══════════════════════════════════════════════════════════════════════════════
# 탭 2: 세션 로그
# ══════════════════════════════════════════════════════════════════════════════

_EP_COLS = ["#", "경로", "결과", "steps", "lat(ms)", "top액션", "gnd%", "area", "cx", "STOP", "FPE", "메모", "날짜"]


def load_episode_csv():
    if not _EPISODE_CSV.exists():
        return [], "_(기록 없음)_"
    try:
        rows = []
        with open(_EPISODE_CSV) as f:
            for r in csv.DictReader(f):
                rows.append([r.get(c, "") for c in _EP_COLS])
        total = len(rows)
        ok = sum(1 for r in rows if r[2] == "성공")
        summary = f"총 **{total}** 에피소드  |  성공 **{ok}** / 실패 **{total-ok}**  |  성공률 **{ok/total*100:.1f}%**" if total else "_(기록 없음)_"
        return rows, summary
    except Exception as e:
        return [], f"❌ 로드 실패: {e}"


def _list_jsonl(directory: Path, prefix: str):
    if not directory.exists():
        return []
    return sorted([f.name for f in directory.glob(f"{prefix}*.jsonl")], reverse=True)


def _ts_from_fname(fname: str) -> int:
    """파일명에서 YYYYmmdd_HHMMSS 타임스탬프 → 초 단위 정수. 실패 시 0."""
    import re, datetime as _dt
    m = re.search(r"(\d{8}_\d{6})", fname)
    if not m:
        return 0
    try:
        return int(_dt.datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").timestamp())
    except Exception:
        return 0


def _find_matching_mp4(directory: Path, jsonl_fname: str, max_delta: int = 60) -> Path | None:
    """같은 디렉터리에서 타임스탬프가 가장 가까운 .mp4 반환 (delta ≤ max_delta초)."""
    ts_ref = _ts_from_fname(jsonl_fname)
    if ts_ref == 0:
        return None
    best, best_delta = None, max_delta + 1
    for mp4 in directory.glob("*.mp4"):
        delta = abs(_ts_from_fname(mp4.name) - ts_ref)
        if delta < best_delta:
            best, best_delta = mp4, delta
    return best if best_delta <= max_delta else None


def _extract_frames(mp4_path: Path, max_frames: int = 30) -> list:
    """MP4에서 균등 간격으로 최대 max_frames장 PIL 썸네일 추출."""
    cap = cv2.VideoCapture(str(mp4_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    step = max(1, total // max_frames)
    frames = []
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frm = cap.read()
        if not ok:
            continue
        frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        th  = _thumb(frm, w=180)
        frames.append((Image.fromarray(th.astype(np.uint8)), f"[{i}]"))
        if len(frames) >= max_frames:
            break
    cap.release()
    return frames


def load_jsonl_session(directory: Path, fname: str):
    if not fname:
        return [], "_(세션을 선택하세요)_"
    path = directory / fname
    try:
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows, path
    except Exception as e:
        return [], f"❌ {e}"


def gnd_session_to_table(rows):
    """grounding JSONL → Dataframe rows + summary markdown"""
    if not rows:
        return [], "_(없음)_"
    table = []
    for r in rows:
        table.append([
            r.get("n", ""),
            "✅" if r.get("has_bbox") else "❌",
            round(r.get("area", 0), 4),
            round(r.get("cx", 0), 4),
            round(r.get("cy", 0), 4),
            round(r.get("latency_ms", 0), 1),
            r.get("pred_label", r.get("raw", "")[:30]),
        ])
    n = len(rows)
    has = sum(1 for r in rows if r.get("has_bbox"))
    lats = [r["latency_ms"] for r in rows if isinstance(r.get("latency_ms"), (int, float))]
    avg_lat = round(sum(lats) / len(lats), 1) if lats else 0
    areas = [r["area"] for r in rows if r.get("area")]
    avg_area = round(sum(areas) / len(areas), 4) if areas else 0
    summary = (
        f"**{n}** 스텝  |  grounding **{has}/{n}** ({has/n*100:.1f}%)  "
        f"|  평균 latency **{avg_lat}ms**  |  평균 area **{avg_area}**"
    )
    return table, summary


def _list_infer_reports() -> list[str]:
    if not _INFER_REPORT_DIR.exists():
        return []
    return sorted([f.name for f in _INFER_REPORT_DIR.glob("session_*.json")], reverse=True)


def load_infer_report(fname: str):
    """추론 세션 JSON → (table_rows, summary_md, matching_h5_name_or_None)"""
    if not fname:
        return [], "_(선택하세요)_", None
    path = _INFER_REPORT_DIR / fname
    try:
        with open(path) as f:
            d = json.load(f)
    except Exception as e:
        return [], f"❌ {e}", None

    history = d.get("history", [])
    sm = d.get("summary", {})

    # 테이블: step, action, latency, predicted_label, bbox area/cx
    table = []
    for h in history:
        a = h.get("action", [0, 0, 0])
        cls = classify_8(a)
        bbox = h.get("bbox") or {}
        table.append([
            h.get("step", ""),
            f"{CLASS_SYM[cls]} {CLASS_NAMES[cls]}",
            f"[{a[0]:+.2f},{a[1]:+.2f},{a[2]:+.2f}]",
            round(h.get("latency_ms", 0), 1),
            h.get("predicted_label", ""),
            round(bbox.get("area", 0), 4) if bbox else 0,
            round(bbox.get("cx", 0), 3) if bbox else 0,
            "✅" if bbox.get("has_bbox") else ("—" if not bbox else "❌"),
            str(h.get("timestamp", ""))[:19],
        ])

    # action 분포
    act_counts: Counter = Counter()
    for h in history:
        act_counts[classify_8(h.get("action", [0, 0, 0]))] += 1
    dist = " | ".join(
        f"{CLASS_SYM[i]}{CLASS_NAMES[i]}: {c}"
        for i, c in sorted(act_counts.items()) if c > 0
    )

    summary = (
        f"**{d.get('session_id','')}**  |  "
        f"model: `{d.get('model_name','?')}`  |  "
        f"steps: **{len(history)}**  |  "
        f"avg lat: **{sm.get('avg_latency_ms',0)}ms**  |  "
        f"status: `{d.get('status','?')}`\n\n"
        f"instruction: {d.get('instruction','—')}\n\n"
        f"액션 분포: {dist or '—'}"
    )

    # 매칭 H5 (session_id 기반)
    sid = d.get("session_id", "")
    h5_name = f"session_{sid}.h5" if sid else None
    h5_exists = h5_name and (_INFER_H5_DIR / h5_name).exists()

    return table, summary, (h5_name if h5_exists else None)


def calib_session_to_table(rows):
    """calib JSONL → Dataframe rows + summary + 추천 임계값"""
    if not rows:
        return [], "_(없음)_"
    table = []
    for r in rows:
        table.append([
            r.get("n", ""),
            round(r.get("area", 0), 4),
            round(r.get("cx", 0), 3),
            round(r.get("cy", 0), 3),
            round(r.get("latency_ms", 0), 1),
            "Y" if r.get("stop_triggered") else "N",
            str(r.get("ts", "")),
            str(r.get("pred_label", "")),
        ])
    n = len(rows)
    areas = sorted(r.get("area", 0) for r in rows if r.get("area", 0) > 0.01)
    stop_n = sum(1 for r in rows if r.get("stop_triggered"))
    if areas:
        p85 = areas[int(len(areas) * 0.85)]
        avg_a = round(sum(areas) / len(areas), 4)
        rec = f"추천 임계값: **`{p85:.3f}`** (85퍼센타일, {len(areas)}개)  |  avg: {avg_a}  min: {areas[0]:.4f}  max: {areas[-1]:.4f}"
    else:
        rec = "_(area 데이터 없음)_"
    summary = f"**{n}** 샘플  |  STOP 발생 **{stop_n}회**  |  {rec}"
    return table, summary


def drift_session_to_table(rows):
    """drift JSONL → Dataframe rows + summary markdown"""
    if not rows:
        return [], "_(없음)_"
    table = []
    for r in rows:
        table.append([
            r.get("frame", ""),
            r.get("ts", "")[:19],
            round(r.get("latency_ms", 0), 1),
            round(r.get("cum_real_s", 0), 2),
            round(r.get("cum_nominal_s", 0), 2),
            round(r.get("drift_s", 0), 3),
        ])
    n = len(rows)
    lats = [r["latency_ms"] for r in rows if isinstance(r.get("latency_ms"), (int, float))]
    avg_lat = round(sum(lats) / len(lats), 1) if lats else 0
    max_drift = max((r.get("drift_s", 0) for r in rows), default=0)
    summary = (
        f"**{n}** 프레임  |  평균 latency **{avg_lat}ms**  |  최대 drift **{max_drift:.3f}s**"
    )
    return table, summary


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


# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

_CSS = """
.gradio-container { max-width: 100% !important; padding: 0 16px !important; }
footer { display: none !important; }
#stats-md  { background: #0d1117; border-radius: 8px; padding: 12px;
             font-family: monospace; font-size: 0.85rem; }
#ep-info   { font-size: 0.9rem; }
#gallery   { min-height: 300px; }
#ep-table  { font-size: 0.82rem; }
#gnd-table { font-size: 0.82rem; }
#dft-table { font-size: 0.82rem; }
.nav-btn   { min-width: 80px !important; }
"""

_ds_labels, _ds_values = dataset_display_choices()
_default_ds = next((v for v in _ds_values if "v5_2" in v), (_ds_values[0] if _ds_values else None))

with gr.Blocks(
    title="MoNaVLA Dataset Viewer",
    css=_CSS,
    theme=gr.themes.Soft(
        primary_hue=gr.themes.colors.indigo,
        neutral_hue=gr.themes.colors.slate,
    ),
) as demo:
    gr.Markdown("# 📂 MoNaVLA 데이터셋 · 세션 뷰어")

    with gr.Tabs():

        # ══════════════════════════════════════════════════════════════════
        # 탭 1: 데이터셋
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("📂 데이터셋"):

            with gr.Row():
                ds_dd = gr.Dropdown(
                    choices=_ds_labels,
                    value=(_ds_labels[_ds_values.index(_default_ds)] if _default_ds else None),
                    label="📁 데이터셋", scale=3,
                )
                sc_dd = gr.Dropdown(choices=["(전체)"], value="(전체)",
                                    label="🗂 시나리오 필터", scale=2)
                ep_dd = gr.Dropdown(choices=[], value=None,
                                    label="🎬 에피소드", scale=5)
                refresh_btn = gr.Button("🔄", scale=0, min_width=50)

            with gr.Row():
                frame_range = gr.Slider(minimum=0, maximum=1, value=0, step=1,
                                        label="시작 프레임", scale=5, interactive=True)
                frame_end   = gr.Slider(minimum=1, maximum=1, value=1, step=1,
                                        label="끝 프레임",   scale=5, interactive=True)
                prev_btn = gr.Button("⬅ 이전", elem_classes="nav-btn", scale=1)
                next_btn = gr.Button("다음 ➡", elem_classes="nav-btn", scale=1)
                del_btn  = gr.Button("🗑️ 삭제", variant="stop", scale=1)

            with gr.Row(equal_height=False):
                with gr.Column(scale=4):
                    gallery = gr.Gallery(
                        label="프레임 갤러리  (클릭 → 원본 화질)", elem_id="gallery",
                        columns=6, height=440, object_fit="contain",
                    )
                    full_info = gr.Markdown("_(썸네일 클릭 → 원본 화질)_")
                    full_view = gr.Image(label="🔍 원본 화질", interactive=False, height=500)

                    # ── 그라운딩 + 추론 패널 ───────────────────────────────
                    with gr.Accordion("🔍 그라운딩 / 🤖 추론 (추론 서버 8001)", open=False):
                      with gr.Row():
                        infer_phrase = gr.Textbox(
                            value="gray basket", label="그라운딩 phrase / instruction",
                            scale=4, max_lines=1,
                        )
                        btn_ground  = gr.Button("🔍 그라운딩",  variant="secondary", scale=1)
                        btn_predict = gr.Button("🤖 추론",       variant="primary",   scale=1)
                      with gr.Row():
                        infer_annotated = gr.Image(
                            label="결과 이미지 (bbox 오버레이)", interactive=False, height=360, scale=3,
                        )
                        with gr.Column(scale=2):
                          infer_result_md = gr.Markdown("_(실행 전)_")

                with gr.Column(scale=1):
                    ep_info_md = gr.Markdown("_(에피소드를 선택하세요)_", elem_id="ep-info")
                    gr.Markdown("---")
                    stats_md = gr.Markdown("_(데이터셋 선택 후 통계 로드)_", elem_id="stats-md")

            _n_frames = gr.State(1)

            # 헬퍼
            def _ds_name(label):
                for v, l in zip(_ds_values, _ds_labels):
                    if l == label or v == label:
                        return v
                return label

            def on_ds_change(ds_label):
                name = _ds_name(ds_label)
                stats, scs = dataset_stats(name)
                eps = list_episodes(name, "(전체)")
                ep = eps[0] if eps else None
                return gr.update(choices=scs, value="(전체)"), gr.update(choices=eps, value=ep), stats

            def on_sc_change(ds_label, sc):
                name = _ds_name(ds_label)
                eps = list_episodes(name, sc)
                return gr.update(choices=eps, value=(eps[0] if eps else None))

            def on_ep_change(ds_label, fname):
                name = _ds_name(ds_label)
                gal, info, n, fe = load_episode(name, fname, 0, -1)
                return (gal, info,
                        gr.update(minimum=0, maximum=max(0,n-1), value=0),
                        gr.update(minimum=1, maximum=n, value=n), n)

            def on_range_change(ds_label, fname, fs, fe):
                name = _ds_name(ds_label)
                gal, info, n, _ = load_episode(name, fname, int(fs), int(fe))
                return gal, info

            def on_prev(ds_label, fname, sc):
                name = _ds_name(ds_label)
                eu, gal, info, n, fe = nav_episode(name, fname, sc, -1)
                return eu, gal, info, gr.update(minimum=0,maximum=max(0,n-1),value=0), gr.update(minimum=1,maximum=n,value=n)

            def on_next(ds_label, fname, sc):
                name = _ds_name(ds_label)
                eu, gal, info, n, fe = nav_episode(name, fname, sc, +1)
                return eu, gal, info, gr.update(minimum=0,maximum=max(0,n-1),value=0), gr.update(minimum=1,maximum=n,value=n)

            def on_delete(ds_label, fname, sc):
                name = _ds_name(ds_label)
                return delete_episode(name, fname, sc)

            def on_refresh(ds_label):
                name = _ds_name(ds_label)
                stats, scs = dataset_stats(name)
                eps = list_episodes(name, "(전체)")
                return gr.update(choices=scs, value="(전체)"), gr.update(choices=eps, value=(eps[0] if eps else None)), stats

            def _show_full(ds_label, fname, fs, evt: gr.SelectData):
                return show_full(_ds_name(ds_label), fname, int(fs), evt)

            ds_dd.change(on_ds_change, ds_dd, [sc_dd, ep_dd, stats_md])
            sc_dd.change(on_sc_change, [ds_dd, sc_dd], ep_dd)
            ep_dd.change(on_ep_change, [ds_dd, ep_dd], [gallery, ep_info_md, frame_range, frame_end, _n_frames])
            frame_range.release(on_range_change, [ds_dd, ep_dd, frame_range, frame_end], [gallery, ep_info_md])
            frame_end.release(on_range_change,   [ds_dd, ep_dd, frame_range, frame_end], [gallery, ep_info_md])
            gallery.select(_show_full, [ds_dd, ep_dd, frame_range], [full_view, full_info])
            del_btn.click(on_delete, [ds_dd, ep_dd, sc_dd], [ep_dd, gallery, ep_info_md])
            prev_btn.click(on_prev, [ds_dd, ep_dd, sc_dd], [ep_dd, gallery, ep_info_md, frame_range, frame_end])
            next_btn.click(on_next, [ds_dd, ep_dd, sc_dd], [ep_dd, gallery, ep_info_md, frame_range, frame_end])
            refresh_btn.click(on_refresh, ds_dd, [sc_dd, ep_dd, stats_md])

            # ── 그라운딩 / 추론 핸들러 ────────────────────────────────────
            def _img_to_b64(pil_img) -> str:
                import base64, io as _io
                buf = _io.BytesIO()
                pil_img.save(buf, format="PNG")
                return base64.b64encode(buf.getvalue()).decode()

            def _draw_bbox(pil_img, bbox: dict) -> Image.Image:
                arr = np.array(pil_img)
                h, w = arr.shape[:2]
                # 3×3 격자
                for x in [w//3, 2*w//3]:
                    cv2.line(arr, (x,0),(x,h),(100,255,100),1)
                for y in [h//3, 2*h//3]:
                    cv2.line(arr, (0,y),(w,y),(100,255,100),1)
                if bbox.get("has_bbox"):
                    cx_px = int(bbox["cx"]*w); cy_px = int(bbox["cy"]*h)
                    if "x1" in bbox and bbox["x1"] is not None:
                        x1,y1,x2,y2 = int(bbox["x1"]*w),int(bbox["y1"]*h),int(bbox["x2"]*w),int(bbox["y2"]*h)
                        cv2.rectangle(arr,(x1,y1),(x2,y2),(255,80,80),2)
                    cv2.circle(arr,(cx_px,cy_px),5,(255,80,80),-1)
                    cv2.putText(arr, f"cx={bbox['cx']:.3f} area={bbox['area']:.4f}",
                                (max(cx_px-60,0), max(cy_px-10,12)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,80,80), 1, cv2.LINE_AA)
                return Image.fromarray(arr)

            def _run_ground(img, phrase):
                import requests as _rq
                if img is None:
                    return None, "❌ 이미지 없음 — 썸네일을 먼저 클릭하세요"
                pil = Image.fromarray(img) if isinstance(img, np.ndarray) else img
                b64 = _img_to_b64(pil)
                try:
                    _hdrs = {"X-API-Key": _INFER_API_KEY} if _INFER_API_KEY else {}
                    r = _rq.post(f"{_INFER_API_URL}/ground",
                                 json={"image": b64, "prompt": f"detect {phrase}"},
                                 headers=_hdrs, timeout=15)
                    b = r.json()
                    annotated = _draw_bbox(pil, b)
                    md = (
                        f"**🔍 그라운딩 결과**\n\n"
                        f"- phrase: `detect {phrase}`\n"
                        f"- has_bbox: `{b.get('has_bbox')}`\n"
                        f"- cx / cy: `{b.get('cx',0):.4f}` / `{b.get('cy',0):.4f}`\n"
                        f"- area: `{b.get('area',0):.5f}`\n"
                        f"- bbox: x1={b.get('x1','?')}  x2={b.get('x2','?')}\n"
                        f"- raw_output: `{b.get('raw_output','')[:80]}`\n"
                        f"- latency: `{b.get('latency_ms','?')} ms`"
                    )
                    return annotated, md
                except Exception as e:
                    return None, f"❌ 그라운딩 실패: {e}"

            def _run_predict(img, phrase):
                import requests as _rq
                if img is None:
                    return None, "❌ 이미지 없음 — 썸네일을 먼저 클릭하세요"
                pil = Image.fromarray(img) if isinstance(img, np.ndarray) else img
                b64 = _img_to_b64(pil)
                try:
                    _hdrs = {"X-API-Key": _INFER_API_KEY} if _INFER_API_KEY else {}
                    # 1. 추론 (predict)
                    r = _rq.post(f"{_INFER_API_URL}/predict",
                                 json={"image": b64, "instruction": phrase},
                                 headers=_hdrs, timeout=20)
                    p = r.json()
                    bbox = p.get("bbox", {}) or {}
                    annotated = _draw_bbox(pil, bbox)

                    action_3d = p.get("action_3d") or p.get("action") or [0,0,0]
                    md = (
                        f"**🤖 추론 결과**\n\n"
                        f"- instruction: `{phrase}`\n"
                        f"- **predicted_label: `{p.get('predicted_label','?')}`**\n"
                        f"- action_3d: `{[round(v,3) for v in action_3d]}`\n"
                        f"- grounding_cached: `{p.get('grounding_cached','?')}`\n"
                        f"- latency: `{p.get('latency_ms','?')} ms` "
                        f"(grounding: `{p.get('grounding_latency_ms','?')} ms`)\n\n"
                        f"**bbox:**\n"
                        f"- entity: `{bbox.get('entity','?')}`\n"
                        f"- cx/cy: `{bbox.get('cx',0):.4f}` / `{bbox.get('cy',0):.4f}`\n"
                        f"- area: `{bbox.get('area',0):.5f}`\n"
                        f"- has_bbox: `{bbox.get('has_bbox','?')}`"
                    )
                    return annotated, md
                except Exception as e:
                    return None, f"❌ 추론 실패: {e}"

            btn_ground.click(_run_ground,   inputs=[full_view, infer_phrase], outputs=[infer_annotated, infer_result_md])
            btn_predict.click(_run_predict, inputs=[full_view, infer_phrase], outputs=[infer_annotated, infer_result_md])

        # ══════════════════════════════════════════════════════════════════
        # 탭 2: 세션 로그
        # ══════════════════════════════════════════════════════════════════
        with gr.Tab("📊 세션 로그"):

            # ── Episode CSV ───────────────────────────────────────────────
            with gr.Row():
                gr.Markdown("## 📝 에피소드 기록 (`episode_log.csv`)")
                ep_csv_refresh = gr.Button("🔄 새로고침", scale=0, min_width=100)

            ep_csv_summary = gr.Markdown("_(로드 중...)_")
            ep_csv_table = gr.Dataframe(
                headers=_EP_COLS,
                datatype=["number","str","str","number","number","str","number",
                          "number","number","str","number","str","str"],
                label="에피소드 기록", elem_id="ep-table",
                interactive=False, wrap=True,
            )

            gr.Markdown("---")

            # ── Grounding Sessions ────────────────────────────────────────
            gr.Markdown("## 🎯 Grounding 세션 (`grounding_sessions/`)")
            with gr.Row():
                gnd_dd = gr.Dropdown(
                    choices=_list_jsonl(_GND_DIR, "gnd_"),
                    value=None, label="세션 파일", scale=5,
                )
                gnd_load_btn = gr.Button("불러오기", scale=1)

            gnd_summary = gr.Markdown("_(세션을 선택하세요)_")
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    gnd_video = gr.Video(label="📹 세션 영상", interactive=False, height=360)
                    gnd_gallery = gr.Gallery(
                        label="프레임 갤러리 (균등 추출, 클릭 → 확대)",
                        columns=6, height=220, object_fit="contain",
                    )
                with gr.Column(scale=2):
                    gnd_table = gr.Dataframe(
                        headers=["n","bbox","area","cx","cy","lat(ms)","pred"],
                        datatype=["number","str","number","number","number","number","str"],
                        label="스텝별 Grounding 데이터", elem_id="gnd-table",
                        interactive=False,
                    )

            gr.Markdown("---")

            # ── Drift Sessions ────────────────────────────────────────────
            gr.Markdown("## ⏱️ Drift 세션 (`drift_sessions/`)")
            with gr.Row():
                dft_dd = gr.Dropdown(
                    choices=_list_jsonl(_DRIFT_DIR, "drift_"),
                    value=None, label="세션 파일", scale=5,
                )
                dft_load_btn = gr.Button("불러오기", scale=1)

            dft_summary = gr.Markdown("_(세션을 선택하세요)_")
            dft_table = gr.Dataframe(
                headers=["frame","timestamp","lat(ms)","cum_real(s)","cum_nom(s)","drift(s)"],
                datatype=["number","str","number","number","number","number"],
                label="스텝별 Drift 데이터", elem_id="dft-table",
                interactive=False,
            )

            gr.Markdown("---")

            # ── Calib Sessions ────────────────────────────────────────────
            gr.Markdown("## 🔧 Calib 세션 (`calib_sessions/`)")
            with gr.Row():
                calib_dd = gr.Dropdown(
                    choices=_list_jsonl(_CALIB_DIR, "calib_"),
                    value=None, label="세션 파일", scale=5,
                )
                calib_load_btn   = gr.Button("불러오기",     scale=1)
                calib_refresh_btn = gr.Button("🔄 목록 새로고침", scale=1)

            calib_summary_md = gr.Markdown("_(세션을 선택하세요)_")
            with gr.Row(equal_height=False):
                with gr.Column(scale=3):
                    calib_video = gr.Video(label="📹 세션 영상", interactive=False, height=360)
                    calib_gallery = gr.Gallery(
                        label="프레임 갤러리 (균등 추출)",
                        columns=6, height=220, object_fit="contain",
                    )
                with gr.Column(scale=2):
                    calib_table = gr.Dataframe(
                        headers=["n","area","cx","cy","lat(ms)","STOP?","시각","메모"],
                        datatype=["number","number","number","number","number","str","str","str"],
                        label="스텝별 캘리브레이션 데이터", elem_id="calib-table",
                        interactive=False,
                    )

            # ── 이벤트 ───────────────────────────────────────────────────
            def refresh_ep_csv():
                rows, summary = load_episode_csv()
                return rows, summary

            def load_gnd(fname):
                rows, _ = load_jsonl_session(_GND_DIR, fname)
                if isinstance(rows, str):
                    return [], rows, None, []
                table, summary = gnd_session_to_table(rows)
                mp4 = _find_matching_mp4(_GND_DIR, fname)
                video_path = str(mp4) if mp4 else None
                frames = _extract_frames(mp4) if mp4 else []
                return table, summary, video_path, frames

            def load_dft(fname):
                rows, _ = load_jsonl_session(_DRIFT_DIR, fname)
                if isinstance(rows, str):
                    return [], rows
                table, summary = drift_session_to_table(rows)
                return table, summary

            def load_calib(fname):
                rows, _ = load_jsonl_session(_CALIB_DIR, fname)
                if isinstance(rows, str):
                    return [], rows, None, []
                table, summary = calib_session_to_table(rows)
                mp4 = _find_matching_mp4(_CALIB_DIR, fname)
                video_path = str(mp4) if mp4 else None
                frames = _extract_frames(mp4) if mp4 else []
                return table, summary, video_path, frames

            def refresh_calib_list():
                files = _list_jsonl(_CALIB_DIR, "calib_")
                return gr.update(choices=files, value=(files[0] if files else None))

            ep_csv_refresh.click(refresh_ep_csv, outputs=[ep_csv_table, ep_csv_summary])
            gnd_load_btn.click(
                load_gnd, inputs=gnd_dd,
                outputs=[gnd_table, gnd_summary, gnd_video, gnd_gallery],
            )
            dft_load_btn.click(load_dft, inputs=dft_dd, outputs=[dft_table, dft_summary])
            calib_load_btn.click(
                load_calib, inputs=calib_dd,
                outputs=[calib_table, calib_summary_md, calib_video, calib_gallery],
            )
            calib_refresh_btn.click(refresh_calib_list, outputs=calib_dd)

            gr.Markdown("---")

            # ── Inference Sessions ────────────────────────────────────────
            gr.Markdown("## 🤖 추론 세션 (`inference_reports/`)\n"
                        f"> H5 이미지는 **Tab 1 → `{_INFER_DS_KEY}`** 데이터셋에서 볼 수 있습니다.")
            with gr.Row():
                infer_dd = gr.Dropdown(
                    choices=_list_infer_reports(),
                    value=None, label="세션 JSON", scale=5,
                )
                infer_load_btn    = gr.Button("불러오기",         scale=1)
                infer_refresh_btn = gr.Button("🔄 목록 새로고침", scale=1)

            infer_summary_md = gr.Markdown("_(세션을 선택하세요)_")
            infer_h5_link_md = gr.Markdown("")
            infer_table = gr.Dataframe(
                headers=["step", "cls", "action", "lat(ms)", "predicted_label",
                         "bbox area", "bbox cx", "has_bbox", "timestamp"],
                datatype=["number","str","str","number","str","number","number","str","str"],
                label="스텝별 추론 데이터", elem_id="infer-table",
                interactive=False,
            )

            def load_infer(fname):
                table, summary, h5_name = load_infer_report(fname)
                if h5_name:
                    link = (f"**H5 이미지 데이터**: Tab 1에서 "
                            f"`{_INFER_DS_KEY}` 선택 후 `{h5_name}` 에피소드를 열면 실제 영상 확인 가능")
                else:
                    link = "_(이 세션에 대응하는 H5 이미지 파일 없음)_"
                return table, summary, link

            def refresh_infer_list():
                files = _list_infer_reports()
                return gr.update(choices=files, value=(files[0] if files else None))

            infer_load_btn.click(
                load_infer, inputs=infer_dd,
                outputs=[infer_table, infer_summary_md, infer_h5_link_md],
            )
            infer_refresh_btn.click(refresh_infer_list, outputs=infer_dd)

    # ── 초기 로드 (lazy) ─────────────────────────────────────────────
    def _init():
        name = _default_ds
        stats, scs = dataset_stats(name) if name else ("", ["(전체)"])
        eps = list_episodes(name, "(전체)") if name else []
        ep  = eps[0] if eps else None
        gal, info, n, fe = load_episode(name, ep, 0, -1) if ep else ([], "_(없음)_", 0, 1)
        ep_rows, ep_summary = load_episode_csv()
        return (
            gr.update(choices=scs, value="(전체)"),
            gr.update(choices=eps, value=ep),
            gal, info, stats,
            gr.update(minimum=0, maximum=max(0,n-1), value=0),
            gr.update(minimum=1, maximum=n, value=n),
            n,
            ep_rows, ep_summary,
        )

    demo.load(
        fn=_init,
        outputs=[sc_dd, ep_dd, gallery, ep_info_md, stats_md,
                 frame_range, frame_end, _n_frames,
                 ep_csv_table, ep_csv_summary],
    )


if __name__ == "__main__":
    port = pick_port(int(os.getenv("VLA_VIEWER_PORT", "8083")))
    print(f"✅ Dataset Viewer → http://0.0.0.0:{port}")
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)

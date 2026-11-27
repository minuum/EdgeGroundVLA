#!/usr/bin/env python3
"""연구·엔지니어링 포트폴리오용 고해상도 시각 카드(SVG + PNG 5종) 생성 스크립트.

원칙:
  - 개인 진학 계획/GPA/내부 미팅 문구 제거 -> 실제 구현·검증한 성과만 직관적으로 제시
  - 처음 보는 사람도 5초 안에 이해하는 Problem -> Solution -> Result 구조
출력 디렉토리: docs/v5/portfolio_kit/
"""
from __future__ import annotations

import base64
import html
import json
import math
from pathlib import Path

import cairosvg

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "docs/v5/portfolio_kit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FONT = "NanumBarunGothic, Noto Sans CJK KR, sans-serif"
MONO = "Noto Sans Mono CJK KR, DejaVu Sans Mono, monospace"


def _b64_img(rel_path: str) -> str:
    fp = ROOT / rel_path
    ext = fp.suffix.lower().lstrip(".")
    mime = "image/jpeg" if ext in ("jpg", "jpeg") else f"image/{ext}"
    data = base64.b64encode(fp.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def _save_svg_and_png(name: str, svg_str: str, scale: float = 2.0) -> None:
    svg_path = OUT_DIR / f"{name}.svg"
    png_path = OUT_DIR / f"{name}.png"
    svg_path.write_text(svg_str, encoding="utf-8")
    cairosvg.svg2png(bytestring=svg_str.encode("utf-8"), write_to=str(png_path), scale=scale)
    print(f"[OK] {png_path.name} ({png_path.stat().st_size // 1024} KB)")


def _svg_shell(w: int, h: int, body: str, defs: str = "") -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#ffffff"/>
      <stop offset="100%" stop-color="#f8fafc"/>
    </linearGradient>
    <linearGradient id="blueGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#1d4ed8"/>
      <stop offset="100%" stop-color="#2563eb"/>
    </linearGradient>
    <linearGradient id="greenGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#059669"/>
      <stop offset="100%" stop-color="#10b981"/>
    </linearGradient>
    <linearGradient id="purpleGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#6d28d9"/>
      <stop offset="100%" stop-color="#8b5cf6"/>
    </linearGradient>
    <marker id="arrGray" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
      <path d="M0,1 L7,4 L0,7 Z" fill="#64748b"/>
    </marker>
    <marker id="arrBlue" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
      <path d="M0,1 L7,4 L0,7 Z" fill="#2563eb"/>
    </marker>
    {defs}
  </defs>
  <rect x="1" y="1" width="{w-2}" height="{h-2}" rx="18" fill="url(#bgGrad)" stroke="#cbd5e1" stroke-width="1.5"/>
  {body}
</svg>'''


PUB_FONT = "Liberation Sans, DejaVu Sans, Helvetica, Arial, sans-serif"
PUB_MONO = "DejaVu Sans Mono, Liberation Mono, monospace"


def _draw_pub_frame(
    x0: float, y0: float, w: float, h: float, img_uri: str, cid: str,
    bbox: tuple[float, float, float], col: str, dash: str,
    title_lbl: str, box_lbl: str, ang_deg: float, out_lbl: str,
) -> str:
    """CVPR/ICRA 논문 스타일의 동일 프레임 추론 패널 (직각 프레임, Okabe-Ito 컬러, 깔끔한 캡션)."""
    cx, cy, area = bbox
    side = math.sqrt(max(0.002, area))
    bw = min(0.85, side * 1.15) * w
    bh = min(0.85, side / 1.15) * h
    bx = x0 + max(2.0, min(w - bw - 2.0, cx * w - bw / 2.0))
    by = y0 + max(2.0, min(h - bh - 2.0, cy * h - bh / 2.0))
    da = f' stroke-dasharray="{dash}"' if dash else ""

    ox, oy = x0 + w * 0.5, y0 + h - 8.0
    rad = math.radians(ang_deg)
    length = 56.0
    tx = ox + math.sin(rad) * length
    ty = oy - math.cos(rad) * length

    return f'''
    <text x="{x0}" y="{y0 - 8}" font-family="{PUB_FONT}" font-size="11.5" font-weight="700" fill="{col}">{html.escape(title_lbl)}</text>
    <clipPath id="{cid}"><rect x="{x0}" y="{y0}" width="{w}" height="{h}"/></clipPath>
    <image href="{img_uri}" x="{x0}" y="{y0}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid slice" clip-path="url(#{cid})"/>
    <rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="none" stroke="#111827" stroke-width="0.9"/>
    <rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="none" stroke="{col}" stroke-width="2.2"{da}/>
    <rect x="{bx:.1f}" y="{max(y0+2, by-15):.1f}" width="106" height="14" fill="rgba(255,255,255,0.88)"/>
    <text x="{bx+3:.1f}" y="{max(y0+12, by-5):.1f}" font-family="{PUB_MONO}" font-size="8.8" font-weight="700" fill="{col}">{html.escape(box_lbl)}</text>
    <line x1="{ox:.1f}" y1="{oy:.1f}" x2="{tx:.1f}" y2="{ty:.1f}" stroke="{col}" stroke-width="3.2" stroke-linecap="round"/>
    <circle cx="{tx:.1f}" cy="{ty:.1f}" r="3.8" fill="{col}"/>
    <text x="{x0 + w/2}" y="{y0 + h + 16}" text-anchor="middle" font-family="{PUB_MONO}" font-size="9.6" font-weight="700" fill="#111827">{html.escape(out_lbl)}</text>
    '''


def _lock_icon(x: float, y: float) -> str:
    """CVPR 논문 스타일 미니 자물쇠(Frozen) 벡터 아이콘."""
    return (
        f'<g transform="translate({x:.1f},{y:.1f})">'
        f'<path d="M2,5 V3.5 A2.5,2.5 0 0 1 7,3.5 V5" fill="none" stroke="#6B7280" stroke-width="1.1"/>'
        f'<rect x="1" y="5" width="7" height="5.5" rx="1" fill="#6B7280"/>'
        f'</g>'
    )


def _train_icon(x: float, y: float, col: str) -> str:
    """CVPR 논문 스타일 미니 학습 가능(Trainable) 벡터 아이콘."""
    return (
        f'<g transform="translate({x:.1f},{y:.1f})">'
        f'<circle cx="4.5" cy="5.5" r="3.8" fill="none" stroke="{col}" stroke-width="1.2"/>'
        f'<circle cx="4.5" cy="5.5" r="1.6" fill="{col}"/>'
        f'</g>'
    )


def _arch_box(x: float, y: float, w: float, h: float, title: str, sub: str, frozen: bool, col: str) -> str:
    """아키텍처 다이어그램 모듈 박스 (Frozen=회색+자물쇠, Trainable=컬러+원형 마커)."""
    bg = "#F3F4F6" if frozen else "#FFFFFF"
    stroke = "#9CA3AF" if frozen else col
    sw = "1.0" if frozen else "1.5"
    tc = "#374151" if frozen else "#111827"
    sc = "#6B7280" if frozen else col
    icon = _lock_icon(x + w - 13, y + 4) if frozen else _train_icon(x + w - 13, y + 4, col)
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="2" fill="{bg}" stroke="{stroke}" stroke-width="{sw}"/>'
        f'{icon}'
        f'<text x="{x + 8}" y="{y + 18}" font-family="{PUB_FONT}" font-size="9.8" font-weight="700" fill="{tc}">{html.escape(title)}</text>'
        f'<text x="{x + 8}" y="{y + 32}" font-family="{PUB_FONT}" font-size="8.6" fill="{sc}">{html.escape(sub)}</text>'
    )


def _sg_badge(cx: float, cy: float) -> str:
    """학회 표준 Stop-Gradient (SG) 마커."""
    return (
        f'<rect x="{cx - 11}" y="{cy - 8}" width="22" height="16" rx="2" fill="#FFFFFF" stroke="#6B7280" stroke-width="1.0"/>'
        f'<text x="{cx}" y="{cy + 3.2}" text-anchor="middle" font-family="{PUB_MONO}" font-size="8.2" font-weight="700" fill="#4B5563">SG</text>'
    )


def build_01_profile_roadmap() -> None:
    """Figure 1: CVPR/ICRA Publication Style — (a) Same-Frame Inference, (b) Unified Table, (c) Architecture & Gradient Flow."""
    W, H = 1100, 920
    same_img_uri = _b64_img("docs/v5/figures/inline/99f9060660c1394c.jpg")

    C1, C2, C3 = "#0072B2", "#009E73", "#D55E00"
    INK, MUT, RULE = "#111827", "#4B5563", "#D1D5DB"

    # (a) 3개 동일 프레임 패널
    pw, ph = 326, 148
    px1, px2, px3 = 36, 387, 738
    py = 68

    f1 = _draw_pub_frame(
        px1, py, pw, ph, same_img_uri, "pubG1",
        (0.281, 0.645, 0.027), C1, "",
        "EGVLA-1  (Modular VLA · Discrete MLP)",
        "OWLv2 cx=0.281", 0.0,
        "Output: FORWARD  [1.15, 0.00]  (0.0 deg)",
    )
    f2 = _draw_pub_frame(
        px2, py, pw, ph, same_img_uri, "pubG2",
        (0.280, 0.645, 0.030), C2, "",
        "EGVLA-2  (Modular VLA · Flow Matching)",
        "F2 <loc> cx=0.280", -26.4,
        "Output: [0.64, -0.32]  (-26.4 deg, cont.)",
    )
    f3 = _draw_pub_frame(
        px3, py, pw, ph, same_img_uri, "pubG3",
        (0.294, 0.646, 0.023), C3, "5 3",
        "EGVLA-3  (End-to-End VLA · Soft-Loc)",
        "Soft-Loc cx=0.294", -28.8,
        "Output: [0.59, -0.32]  (-28.8 deg, E2E)",
    )

    # (b) LaTeX Booktabs 통일 기준 비교표 (표준 용어: Modular VLA vs End-to-End VLA)
    rows = [
        ("VLA Paradigm", "Modular VLA (2-Stage)", "Modular VLA (2-Stage)", "End-to-End VLA"),
        ("VLM Backbone", "OWLv2 + Kosmos-2 (Frozen)", "Florence-2 (Frozen)", "Florence-2 (Decoder LoRA)"),
        ("Grounding Interface", "2D BBox + Visual Proj. (SG)", "Text-Parsed <loc> BBox (SG)", "Soft-Loc Expectation (Diff.)"),
        ("Action Head", "Discrete 8-Class MLP (1.128M)", "Flow-Matching Head (PiZeroFM)", "Flow-Matching Head (PiZeroFM)"),
        ("Action Space", "Discrete 8-Way (3-DoF)", "Continuous 2D Velocity [vx, vy]", "Continuous 2D Velocity [vx, vy]"),
        ("Deployment / Eval", "Real-Robot · Jetson Orin NX (On-Device)", "Real-Robot · Serbot 2 + GPU Server", "Offline Replay · 1,458 Real Frames"),
        ("Real-Robot Success", "95.0% (Seen) / 92.0% (300 trials)", "45/45 (0 contact; unseen chair 15/15)", "Pending (Offline Verified)"),
        ("Unseen-Object Stop", "N/A (Single-Object Basket)", "93.0% (area >= 0.25, MOD-A3-AG)", "100.0% (Soft-Loc+) / 97.7% (Joint)"),
        ("Visual-Swap Flip", "N/A", "27.4% -> 0.0% (a3 -> a3-ag23)", "14.9% -> 0.0% (Hidden -> Soft-Loc)"),
        ("Inference Latency", "On-Device 10 Hz Async (Jitter Hold)", "397.1 ms Real (285.3 ms GB10)", "296.6 ms GB10 (283.8 ms Core)"),
    ]

    tx0, tx1, tx2, tx3 = 36, 232, 514, 796
    ty0 = 296
    rh = 29
    tbl_svg = [
        f'<text x="36" y="268" font-family="{PUB_FONT}" font-size="12.5" font-weight="700" fill="{INK}">(b) Unified Architectural &amp; Empirical Comparison Across Generations</text>',
        f'<line x1="36" y1="{ty0-17}" x2="1064" y2="{ty0-17}" stroke="{INK}" stroke-width="1.6"/>',
        f'<text x="{tx0}" y="{ty0}" font-family="{PUB_FONT}" font-size="10.8" font-weight="700" fill="{INK}">Criterion</text>',
        f'<text x="{tx1}" y="{ty0}" font-family="{PUB_FONT}" font-size="10.8" font-weight="700" fill="{C1}">EGVLA-1 (Gen 1)</text>',
        f'<text x="{tx2}" y="{ty0}" font-family="{PUB_FONT}" font-size="10.8" font-weight="700" fill="{C2}">EGVLA-2 (Gen 2)</text>',
        f'<text x="{tx3}" y="{ty0}" font-family="{PUB_FONT}" font-size="10.8" font-weight="700" fill="{C3}">EGVLA-3 (Gen 3)</text>',
        f'<line x1="36" y1="{ty0+9}" x2="1064" y2="{ty0+9}" stroke="{INK}" stroke-width="0.9"/>',
    ]

    for i, (crit, v1, v2, v3) in enumerate(rows):
        ry = ty0 + 30 + i * rh
        if i % 2 == 1:
            tbl_svg.append(f'<rect x="36" y="{ry-18}" width="1028" height="{rh}" fill="#F9FAFB"/>')
        if i == 5:
            tbl_svg.append(f'<line x1="36" y1="{ry-18}" x2="1064" y2="{ry-18}" stroke="#9CA3AF" stroke-width="0.7" stroke-dasharray="3 2"/>')
        tbl_svg.append(f'<text x="{tx0}" y="{ry}" font-family="{PUB_FONT}" font-size="10.4" font-weight="700" fill="{INK}">{html.escape(crit)}</text>')
        for xv, val in ((tx1, v1), (tx2, v2), (tx3, v3)):
            fill_c = MUT if val.startswith(("N/A", "Pending")) else INK
            fw = "700" if any(k in val for k in ("95.0%", "45/45", "100.0%", "0.0%", "End-to-End VLA")) else "400"
            tbl_svg.append(f'<text x="{xv}" y="{ry}" font-family="{PUB_FONT}" font-size="10.2" font-weight="{fw}" fill="{fill_c}">{html.escape(val)}</text>')

    bot_y = ty0 + 30 + len(rows) * rh - 13
    tbl_svg.append(f'<line x1="36" y1="{bot_y}" x2="1064" y2="{bot_y}" stroke="{INK}" stroke-width="1.6"/>')

    # (c) CVPR 스타일 아키텍처 & Gradient Flow 다이어그램 (3행 수평 비교)
    cy_hdr = bot_y + 34
    r1y, r2y, r3y = cy_hdr + 26, cy_hdr + 92, cy_hdr + 158
    bh = 42

    def _arrow(x1: float, y: float, x2: float, col: str = "#374151") -> str:
        return f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{col}" stroke-width="1.3" marker-end="url(#pubArr)"/>'

    arch_svg = [
        f'<text x="36" y="{cy_hdr}" font-family="{PUB_FONT}" font-size="12.5" font-weight="700" fill="{INK}">(c) Architecture &amp; Gradient Flow Comparison</text>',
        # Legend (우상단)
        _lock_icon(646, cy_hdr - 10),
        f'<text x="659" y="{cy_hdr - 1}" font-family="{PUB_FONT}" font-size="9.5" fill="{MUT}">Frozen</text>',
        _train_icon(708, cy_hdr - 10, INK),
        f'<text x="721" y="{cy_hdr - 1}" font-family="{PUB_FONT}" font-size="9.5" fill="{MUT}">Trainable</text>',
        _sg_badge(790, cy_hdr - 4),
        f'<text x="807" y="{cy_hdr - 1}" font-family="{PUB_FONT}" font-size="9.5" fill="{MUT}">Stop-Gradient</text>',
        f'<line x1="892" y1="{cy_hdr - 4}" x2="922" y2="{cy_hdr - 4}" stroke="{C3}" stroke-width="1.5" stroke-dasharray="4 2" marker-end="url(#gradArr)"/>',
        f'<text x="928" y="{cy_hdr - 1}" font-family="{PUB_FONT}" font-size="9.5" font-weight="700" fill="{C3}">Action Backprop</text>',
        f'<line x1="36" y1="{cy_hdr + 9}" x2="1064" y2="{cy_hdr + 9}" stroke="{RULE}" stroke-width="0.8"/>',

        # Row 1: EGVLA-1
        f'<text x="36" y="{r1y + 18}" font-family="{PUB_FONT}" font-size="10.8" font-weight="700" fill="{C1}">EGVLA-1</text>',
        f'<text x="36" y="{r1y + 32}" font-family="{PUB_FONT}" font-size="8.6" fill="{MUT}">Modular</text>',
        f'<rect x="116" y="{r1y}" width="110" height="{bh}" rx="2" fill="#FFFFFF" stroke="#6B7280" stroke-width="1.0"/>',
        f'<text x="171" y="{r1y + 18}" text-anchor="middle" font-family="{PUB_FONT}" font-size="9.5" font-weight="700" fill="{INK}">RGB + Prompt</text>',
        f'<text x="171" y="{r1y + 32}" text-anchor="middle" font-family="{PUB_MONO}" font-size="8.2" fill="{MUT}">fixed basket</text>',
        _arrow(226, r1y + 21, 252),
        _arch_box(254, r1y, 204, bh, "OWLv2 + Kosmos-2", "Dual Frozen Encoders (458M)", True, C1),
        _arrow(458, r1y + 21, 514),
        _sg_badge(486, r1y + 21),
        _arch_box(516, r1y, 178, bh, "2D BBox + Proj.", "6-frame window (280-d)", False, C1),
        _arrow(694, r1y + 21, 722),
        _arch_box(724, r1y, 178, bh, "Discrete MLP Head", "8-Class (1.128M params)", False, C1),
        _arrow(902, r1y + 21, 928),
        f'<rect x="930" y="{r1y}" width="134" height="{bh}" rx="2" fill="#F8FAFC" stroke="{C1}" stroke-width="1.2"/>',
        f'<text x="997" y="{r1y + 18}" text-anchor="middle" font-family="{PUB_FONT}" font-size="9.5" font-weight="700" fill="{C1}">8-Way Discrete</text>',
        f'<text x="997" y="{r1y + 32}" text-anchor="middle" font-family="{PUB_MONO}" font-size="8.2" fill="{MUT}">Jetson 10 Hz</text>',

        # Row 2: EGVLA-2
        f'<text x="36" y="{r2y + 18}" font-family="{PUB_FONT}" font-size="10.8" font-weight="700" fill="{C2}">EGVLA-2</text>',
        f'<text x="36" y="{r2y + 32}" font-family="{PUB_FONT}" font-size="8.6" fill="{MUT}">Modular</text>',
        f'<rect x="116" y="{r2y}" width="110" height="{bh}" rx="2" fill="#FFFFFF" stroke="#6B7280" stroke-width="1.0"/>',
        f'<text x="171" y="{r2y + 18}" text-anchor="middle" font-family="{PUB_FONT}" font-size="9.5" font-weight="700" fill="{INK}">RGB + Prompt</text>',
        f'<text x="171" y="{r2y + 32}" text-anchor="middle" font-family="{PUB_MONO}" font-size="8.2" fill="{MUT}">open-vocab</text>',
        _arrow(226, r2y + 21, 252),
        _arch_box(254, r2y, 204, bh, "Florence-2 (0.23B)", "Frozen Encoder + Decoder", True, C2),
        _arrow(458, r2y + 21, 514),
        _sg_badge(486, r2y + 21),
        _arch_box(516, r2y, 178, bh, "Text-Parsed <loc>", "Regex BBox + Vis (768-d)", True, C2),
        _arrow(694, r2y + 21, 722),
        _arch_box(724, r2y, 178, bh, "Flow-Matching Head", "PiZeroFM + 50% Swap", False, C2),
        _arrow(902, r2y + 21, 928),
        f'<rect x="930" y="{r2y}" width="134" height="{bh}" rx="2" fill="#F8FAFC" stroke="{C2}" stroke-width="1.2"/>',
        f'<text x="997" y="{r2y + 18}" text-anchor="middle" font-family="{PUB_FONT}" font-size="9.5" font-weight="700" fill="{C2}">Cont. [vx, vy]</text>',
        f'<text x="997" y="{r2y + 32}" text-anchor="middle" font-family="{PUB_MONO}" font-size="8.2" fill="{MUT}">+ S2 Decel</text>',

        # Row 3: EGVLA-3 (with End-to-End Gradient Backprop Path)
        f'<text x="36" y="{r3y + 18}" font-family="{PUB_FONT}" font-size="10.8" font-weight="700" fill="{C3}">EGVLA-3</text>',
        f'<text x="36" y="{r3y + 32}" font-family="{PUB_FONT}" font-size="8.6" fill="{MUT}">End-to-End</text>',
        f'<rect x="116" y="{r3y}" width="110" height="{bh}" rx="2" fill="#FFFFFF" stroke="#6B7280" stroke-width="1.0"/>',
        f'<text x="171" y="{r3y + 18}" text-anchor="middle" font-family="{PUB_FONT}" font-size="9.5" font-weight="700" fill="{INK}">RGB + Prompt</text>',
        f'<text x="171" y="{r3y + 32}" text-anchor="middle" font-family="{PUB_MONO}" font-size="8.2" fill="{MUT}">open-vocab</text>',
        _arrow(226, r3y + 21, 252),
        _arch_box(254, r3y, 204, bh, "Florence-2 (Dec. LoRA)", "Jointly Adapted VLM", False, C3),
        _arrow(458, r3y + 21, 514, C3),
        _arch_box(516, r3y, 178, bh, "Soft-Loc (Expected)", "Softmax <loc> (0% Parse)", False, C3),
        _arrow(694, r3y + 21, 722, C3),
        _arch_box(724, r3y, 178, bh, "Flow-Matching Head", "PiZeroFM (Joint Loss)", False, C3),
        _arrow(902, r3y + 21, 928, C3),
        f'<rect x="930" y="{r3y}" width="134" height="{bh}" rx="2" fill="#FFF7ED" stroke="{C3}" stroke-width="1.4"/>',
        f'<text x="997" y="{r3y + 18}" text-anchor="middle" font-family="{PUB_FONT}" font-size="9.5" font-weight="700" fill="{C3}">Cont. [vx, vy]</text>',
        f'<text x="997" y="{r3y + 32}" text-anchor="middle" font-family="{PUB_MONO}" font-size="8.2" fill="{C3}">End-to-End VLA</text>',

        # Gradient Backprop Arc (Action Head -> Soft-Loc -> Florence-2 Decoder LoRA)
        f'<path d="M 813,{r3y + bh} L 813,{r3y + bh + 18} L 356,{r3y + bh + 18} L 356,{r3y + bh + 3}" fill="none" stroke="{C3}" stroke-width="1.5" stroke-dasharray="4 2.5" marker-end="url(#gradArr)"/>',
        f'<rect x="505" y="{r3y + bh + 10}" width="200" height="15" fill="#FFFFFF"/>',
        f'<text x="605" y="{r3y + bh + 21}" text-anchor="middle" font-family="{PUB_FONT}" font-size="9.0" font-weight="700" fill="{C3}">End-to-End Gradient Flow (L_action)</text>',
    ]

    svg_str = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <defs>
    <marker id="pubArr" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
      <path d="M0,0.8 L5.2,3 L0,5.2 Z" fill="#374151"/>
    </marker>
    <marker id="gradArr" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
      <path d="M0,0.8 L5.2,3 L0,5.2 Z" fill="{C3}"/>
    </marker>
  </defs>
  <rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>
  <text x="36" y="34" font-family="{PUB_FONT}" font-size="12.5" font-weight="700" fill="{INK}">(a) Same-Frame Policy Inference  —  Input Prompt: &quot;go to the gray basket&quot;  (Target at left, cx = 0.28)</text>
  <line x1="36" y1="43" x2="1064" y2="43" stroke="{RULE}" stroke-width="0.8"/>
  {f1}
  {f2}
  {f3}
  {"".join(tbl_svg)}
  {"".join(arch_svg)}
  <text x="36" y="{H - 14}" font-family="{PUB_FONT}" font-size="9.1" fill="{MUT}">Figure 1. Evolution from Modular VLA (EGVLA-1/2) to End-to-End VLA (EGVLA-3): (a) same-frame inference, (b) unified evaluation, and (c) module-level architecture &amp; gradient flow.</text>
</svg>'''
    _save_svg_and_png("01_profile_roadmap", svg_str)


def build_02_edgeground_vla_v1_v2() -> None:
    """카드 2: EdgeGround-VLA 아키텍처 & 실기 정량 성과 (쉬운 용어 중심)."""
    W, H = 1100, 510
    body = f'''
    <rect x="28" y="24" width="170" height="24" rx="12" fill="#dbeafe"/>
    <text x="113" y="40" text-anchor="middle" font-family="{FONT}" font-size="11" font-weight="700" fill="#1d4ed8">FLAGSHIP PROJECT</text>
    <text x="210" y="41" font-family="{FONT}" font-size="12.5" font-weight="700" fill="#475569">EdgeGround-VLA — 클라우드·지도 없이 엣지 로봇에서 직접 주행하는 경량 VLA</text>
    <text x="28" y="76" font-family="{FONT}" font-size="22" font-weight="700" fill="#0f172a">경량 아키텍처 진화 및 실기(Real-Robot) 검증 결과</text>

    <!-- 좌측: 아키텍처 -->
    <rect x="28" y="96" width="516" height="390" rx="14" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5"/>
    <text x="46" y="124" font-family="{FONT}" font-size="14" font-weight="700" fill="#0f172a">A. 시스템 구조: 동결(Frozen) VLM + 경량 정책 헤드</text>

    <!-- v1 -->
    <rect x="44" y="140" width="484" height="148" rx="12" fill="#f8fafc" stroke="#cbd5e1"/>
    <rect x="56" y="150" width="128" height="22" rx="11" fill="#2563eb"/>
    <text x="120" y="165" text-anchor="middle" font-family="{FONT}" font-size="11" font-weight="700" fill="#ffffff">v1 · On-Device 경량</text>
    <text x="196" y="165" font-family="{FONT}" font-size="11.5" font-weight="700" fill="#334155">전체의 0.25%(1.128M)만 학습 · 10Hz 비동기 제어</text>

    <rect x="56" y="184" width="136" height="56" rx="9" fill="#e2e8f0" stroke="#94a3b8"/>
    <text x="124" y="207" text-anchor="middle" font-family="{FONT}" font-size="11.5" font-weight="700" fill="#0f172a">OWLv2 + Kosmos-2</text>
    <text x="124" y="225" text-anchor="middle" font-family="{FONT}" font-size="10" fill="#475569">이중 동결 비전 인코더</text>

    <line x1="192" y1="212" x2="216" y2="212" stroke="#2563eb" stroke-width="2" marker-end="url(#arrBlue)"/>

    <rect x="220" y="184" width="136" height="56" rx="9" fill="#dbeafe" stroke="#60a5fa"/>
    <text x="288" y="207" text-anchor="middle" font-family="{FONT}" font-size="11.5" font-weight="700" fill="#1e3a8a">경량 액션 헤드</text>
    <text x="288" y="225" text-anchor="middle" font-family="{FONT}" font-size="10" fill="#1d4ed8">8방향 이산 제어 (1.1M)</text>

    <line x1="356" y1="212" x2="380" y2="212" stroke="#2563eb" stroke-width="2" marker-end="url(#arrBlue)"/>

    <rect x="384" y="184" width="132" height="56" rx="9" fill="#eff6ff" stroke="#93c5fd"/>
    <text x="450" y="207" text-anchor="middle" font-family="{FONT}" font-size="11.5" font-weight="700" fill="#0f172a">Jetson Orin NX</text>
    <text x="450" y="225" text-anchor="middle" font-family="{FONT}" font-size="10" fill="#475569">지도·클라우드 없이 구동</text>

    <text x="56" y="268" font-family="{FONT}" font-size="11" fill="#475569">결과: 훈련 위치 95.0%(95/100), 미학습 위치 포함 300회 92.0%(276/300)</text>

    <!-- v2 -->
    <rect x="44" y="304" width="484" height="166" rx="12" fill="#f0fdf4" stroke="#86efac" stroke-width="1.5"/>
    <rect x="56" y="314" width="144" height="22" rx="11" fill="#059669"/>
    <text x="128" y="329" text-anchor="middle" font-family="{FONT}" font-size="11" font-weight="700" fill="#ffffff">v2 · 다물체 일반화·안전</text>
    <text x="212" y="329" font-family="{FONT}" font-size="11.5" font-weight="700" fill="#065f46">연속 속도 생성 + 추론 2배 가속 + 무충돌 정지</text>

    <rect x="56" y="348" width="136" height="60" rx="9" fill="#dcfce7" stroke="#4ade80"/>
    <text x="124" y="371" text-anchor="middle" font-family="{FONT}" font-size="11.5" font-weight="700" fill="#064e3b">Florence-2 탐지</text>
    <text x="124" y="388" text-anchor="middle" font-family="{FONT}" font-size="10" fill="#047857">FP16 고속화 (285ms)</text>
    <text x="124" y="402" text-anchor="middle" font-family="{FONT}" font-size="9.5" fill="#065f46">임의 물체 위치 추출</text>

    <line x1="192" y1="378" x2="216" y2="378" stroke="#059669" stroke-width="2" marker-end="url(#arrGray)"/>

    <rect x="220" y="348" width="136" height="60" rx="9" fill="#d1fae5" stroke="#10b981"/>
    <text x="288" y="371" text-anchor="middle" font-family="{FONT}" font-size="11.5" font-weight="700" fill="#064e3b">Flow Matching 헤드</text>
    <text x="288" y="388" text-anchor="middle" font-family="{FONT}" font-size="10" fill="#047857">외형 편향 차단 학습</text>
    <text x="288" y="402" text-anchor="middle" font-family="{FONT}" font-size="9.5" fill="#065f46">부드러운 연속 2D 속도</text>

    <line x1="356" y1="378" x2="380" y2="378" stroke="#059669" stroke-width="2" marker-end="url(#arrGray)"/>

    <rect x="384" y="348" width="132" height="60" rx="9" fill="#ecfdf5" stroke="#34d399"/>
    <text x="450" y="371" text-anchor="middle" font-family="{FONT}" font-size="11.5" font-weight="700" fill="#064e3b">안전 감속 제어기</text>
    <text x="450" y="388" text-anchor="middle" font-family="{FONT}" font-size="10" fill="#047857">근접 선형 감속·필터</text>
    <text x="450" y="402" text-anchor="middle" font-family="{FONT}" font-size="9.5" fill="#065f46">실기 접촉 0건 달성</text>

    <text x="56" y="435" font-family="{FONT}" font-size="11" font-weight="700" fill="#065f46">결과: 바구니만 학습하고도 미학습 의자 15/15 포함 실기 45/45 성공 · 접촉 0건</text>
    <text x="56" y="454" font-family="{FONT}" font-size="10.5" fill="#047857">언어 안전 게이트: 눈앞 물체와 다른 명령 입력 시 99.5% 제자리 정지</text>

    <!-- 우측: KPI & 바차트 -->
    <rect x="560" y="96" width="512" height="390" rx="14" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5"/>
    <text x="578" y="124" font-family="{FONT}" font-size="14" font-weight="700" fill="#0f172a">B. 실기 정량 성과 요약 (Serbot 2 + Jetson Orin NX)</text>

    <rect x="578" y="140" width="232" height="80" rx="10" fill="#eff6ff" stroke="#bfdbfe"/>
    <text x="592" y="160" font-family="{FONT}" font-size="11" font-weight="700" fill="#1d4ed8">v1 · 온디바이스 주행 성공률</text>
    <text x="592" y="190" font-family="{FONT}" font-size="24" font-weight="700" fill="#0f172a">95.0% / 92.0%</text>
    <text x="592" y="209" font-family="{FONT}" font-size="10.5" fill="#475569">훈련 95/100 · 미학습 위치 포함 276/300</text>

    <rect x="822" y="140" width="232" height="80" rx="10" fill="#ecfdf5" stroke="#a7f3d0"/>
    <text x="836" y="160" font-family="{FONT}" font-size="11" font-weight="700" fill="#047857">v2 · 미학습 물체 실기 검증</text>
    <text x="836" y="190" font-family="{FONT}" font-size="24" font-weight="700" fill="#0f172a">45/45 · 접촉 0건</text>
    <text x="836" y="209" font-family="{FONT}" font-size="10.5" fill="#475569">바구니 30/30 + 처음 본 의자 15/15</text>

    <rect x="578" y="230" width="232" height="72" rx="10" fill="#f8fafc" stroke="#cbd5e1"/>
    <text x="592" y="250" font-family="{FONT}" font-size="11" font-weight="700" fill="#334155">실시간 추론 지연 단축</text>
    <text x="592" y="277" font-family="{FONT}" font-size="20" font-weight="700" fill="#059669">591ms → 285ms</text>
    <text x="592" y="294" font-family="{FONT}" font-size="10.5" fill="#475569">2.09× 가속 (정확도 99.7% 유지)</text>

    <rect x="822" y="230" width="232" height="72" rx="10" fill="#f8fafc" stroke="#cbd5e1"/>
    <text x="836" y="250" font-family="{FONT}" font-size="11" font-weight="700" fill="#334155">조향 정밀도 · 학습 효율</text>
    <text x="836" y="277" font-family="{FONT}" font-size="20" font-weight="700" fill="#2563eb">0.54° · 0.25%</text>
    <text x="836" y="294" font-family="{FONT}" font-size="10.5" fill="#475569">방향 오차 1° 미만 · 1.128M/459M 학습</text>

    <text x="578" y="330" font-family="{FONT}" font-size="12" font-weight="700" fill="#334155">주요 개선 지표 (기존 기준선 → 개선 모델)</text>

    <text x="578" y="354" font-family="{FONT}" font-size="11" fill="#475569">미학습 의자 출발</text>
    <rect x="692" y="342" width="286" height="16" rx="4" fill="#f1f5f9"/>
    <rect x="692" y="342" width="286" height="16" rx="4" fill="url(#greenGrad)"/>
    <text x="986" y="354" font-family="{FONT}" font-size="11" font-weight="700" fill="#047857">0% → 100%</text>

    <text x="578" y="384" font-family="{FONT}" font-size="11" fill="#475569">근접 무충돌 정지</text>
    <rect x="692" y="372" width="286" height="16" rx="4" fill="#f1f5f9"/>
    <rect x="692" y="372" width="80" height="16" rx="4" fill="#fb923c"/>
    <rect x="692" y="372" width="266" height="16" rx="4" fill="url(#greenGrad)" fill-opacity="0.88"/>
    <text x="986" y="384" font-family="{FONT}" font-size="11" font-weight="700" fill="#047857">28% → 93%</text>

    <text x="578" y="414" font-family="{FONT}" font-size="11" fill="#475569">오명령 안전 차단</text>
    <rect x="692" y="402" width="286" height="16" rx="4" fill="#f1f5f9"/>
    <rect x="692" y="402" width="284" height="16" rx="4" fill="url(#blueGrad)"/>
    <text x="986" y="414" font-family="{FONT}" font-size="11" font-weight="700" fill="#1d4ed8">99.5% 정지</text>

    <text x="578" y="444" font-family="{FONT}" font-size="11" fill="#475569">프레임 추론 지연</text>
    <rect x="692" y="432" width="286" height="16" rx="4" fill="#fee2e2"/>
    <rect x="692" y="432" width="138" height="16" rx="4" fill="url(#purpleGrad)"/>
    <text x="986" y="444" font-family="{FONT}" font-size="11" font-weight="700" fill="#6d28d9">285ms (-52%)</text>
    '''
    _save_svg_and_png("02_edgeground_vla_v1_v2", _svg_shell(W, H, body))


def _draw_scene_overlay(x0: float, y0: float, w: float, h: float, sc: dict, items: list[tuple[str, str, str, float]]) -> str:
    img_uri = _b64_img(f"docs/v5/{sc['img_src']}")
    cid = f"clip_{sc['id']}"
    elems = [
        f'<clipPath id="{cid}"><rect x="{x0}" y="{y0}" width="{w}" height="{h}" rx="10"/></clipPath>',
        f'<image href="{img_uri}" x="{x0}" y="{y0}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid slice" clip-path="url(#{cid})"/>',
        f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" rx="10" fill="none" stroke="#94a3b8" stroke-width="1.2"/>',
    ]
    bb = sc["bbox"]
    side = math.sqrt(max(0.002, float(bb[2])))
    bw = min(0.92, side * 1.15) * w
    bh = min(0.92, side / 1.15) * h
    bx = x0 + max(3.0, min(w - bw - 3.0, float(bb[0]) * w - bw / 2.0))
    by = y0 + max(3.0, min(h - bh - 3.0, float(bb[1]) * h - bh / 2.0))
    elems.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="none" stroke="#38bdf8" stroke-width="2.5"/>')
    elems.append(f'<rect x="{bx:.1f}" y="{max(y0+4, by-18):.1f}" width="154" height="17" rx="4" fill="rgba(15,23,42,0.82)"/>')
    elems.append(f'<text x="{bx+5:.1f}" y="{max(y0+16, by-6):.1f}" font-family="{FONT}" font-size="10" font-weight="700" fill="#7dd3fc">Target BBox (area={bb[2]:.2f})</text>')

    ox, oy, scale = x0 + w * 0.5, y0 + h - 22.0, 128.0
    ms = sc["models"]
    for key, label, col, yoff in items:
        m = ms[key]
        lx, ly = float(m["cmd"][0]), float(m["cmd"][1])
        if m["stop"]:
            r = 8.0 + abs(yoff) * 0.35
            elems.append(f'<circle cx="{ox:.1f}" cy="{oy:.1f}" r="{r:.1f}" fill="none" stroke="{col}" stroke-width="3.0"/>')
            lx_pos = ox + r + 8 if yoff >= 0 else ox - r - 122
            elems.append(f'<rect x="{lx_pos:.1f}" y="{oy - 9 + yoff*0.5:.1f}" width="116" height="18" rx="4" fill="rgba(15,23,42,0.84)"/>')
            elems.append(f'<text x="{lx_pos+5:.1f}" y="{oy + 3 + yoff*0.5:.1f}" font-family="{FONT}" font-size="10" font-weight="700" fill="{col}">{html.escape(label)}: STOP</text>')
        else:
            tx = max(x0 + 18, min(x0 + w - 102, ox - ly * scale))
            ty = max(y0 + 24, min(y0 + h - 18, oy - lx * scale))
            elems.append(f'<line x1="{ox:.1f}" y1="{oy:.1f}" x2="{tx:.1f}" y2="{ty:.1f}" stroke="{col}" stroke-width="3.4" stroke-linecap="round"/>')
            elems.append(f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="4.5" fill="{col}"/>')
            elems.append(f'<rect x="{tx+7:.1f}" y="{ty - 10 + yoff*0.4:.1f}" width="124" height="18" rx="4" fill="rgba(15,23,42,0.84)"/>')
            elems.append(f'<text x="{tx+12:.1f}" y="{ty + 2 + yoff*0.4:.1f}" font-family="{FONT}" font-size="10" font-weight="700" fill="{col}">{html.escape(label)}: GO</text>')
    return "\n".join(elems)


def build_03_research_question_diagnostic() -> None:
    """카드 3: 왜 로봇 정책은 낯선 물체에서 실패하는가? (실기 카메라 오버레이 + 반사실 감사)."""
    W, H = 1100, 550
    fam = json.loads((ROOT / "docs/v5/egvla2_fm/family_arch_compare.json").read_text(encoding="utf-8"))
    sc_map = {s["id"]: s for s in fam["scenes"]}
    sc1 = sc_map["sc1_chair_center"]
    sc3 = sc_map["sc3_chair_close"]

    ov1 = _draw_scene_overlay(
        44, 152, 484, 236, sc1,
        [("MOD-A3", "기존 정책", "#f87171", -14), ("MOD-A3-AG", "제안 정책", "#4ade80", 0)],
    )
    ov2 = _draw_scene_overlay(
        572, 152, 484, 236, sc3,
        [("VLM-E2E-Hidden", "일반 E2E", "#f87171", -10), ("VLM-E2E-SoftLoc", "공간 병목 E2E", "#4ade80", 12)],
    )

    body = f'''
    <rect x="28" y="24" width="176" height="24" rx="12" fill="#fef3c7"/>
    <text x="116" y="40" text-anchor="middle" font-family="{FONT}" font-size="11" font-weight="700" fill="#b45309">FAILURE DIAGNOSIS</text>
    <text x="216" y="41" font-family="{FONT}" font-size="12.5" font-weight="700" fill="#475569">실기 오류 630건 중 81.4%가 대체 가능 행동 → 외형 지름길(Visual Shortcut) 규명</text>
    <text x="28" y="76" font-family="{FONT}" font-size="22" font-weight="700" fill="#0f172a">왜 로봇은 처음 보는 물체 앞에서 멈추거나 부딪힐까?</text>

    <!-- 좌측 장면 -->
    <rect x="28" y="96" width="516" height="334" rx="14" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5"/>
    <text x="44" y="122" font-family="{FONT}" font-size="13.5" font-weight="700" fill="#0f172a">① 출발 실패 진단 (처음 본 의자 정중앙 · 실기 카메라)</text>
    <text x="44" y="140" font-family="{FONT}" font-size="11" fill="#475569">명령: "go to the brown wooden chair" (바구니 데이터로만 학습)</text>
    {ov1}
    <text x="44" y="408" font-family="{FONT}" font-size="11.5" fill="#334155"><tspan font-weight="700" fill="#dc2626">기존 정책</tspan>: 배경 외형에 과적합돼 이미 도착으로 오인 → <tspan font-weight="700">실기 0/5 비출발</tspan></text>
    <text x="44" y="423" font-family="{FONT}" font-size="11.5" fill="#334155"><tspan font-weight="700" fill="#059669">제안 정책</tspan>: 50% 시각 특징 교체 + 기하 정합 → <tspan font-weight="700">100% 정상 출발</tspan></text>

    <!-- 우측 장면 -->
    <rect x="556" y="96" width="516" height="334" rx="14" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5"/>
    <text x="572" y="122" font-family="{FONT}" font-size="13.5" font-weight="700" fill="#0f172a">② 근접 충돌 진단 (처음 본 의자 코앞 · area=0.36)</text>
    <text x="572" y="140" font-family="{FONT}" font-size="11" fill="#475569">End-to-End 정책에서 명시적 공간 병목(Spatial Bottleneck) 유무 비교</text>
    {ov2}
    <text x="572" y="408" font-family="{FONT}" font-size="11.5" fill="#334155"><tspan font-weight="700" fill="#dc2626">일반 E2E</tspan>: 공간 병목 없이 특징만 연결 → <tspan font-weight="700">의자 정지 18.6% (충돌 위험)</tspan></text>
    <text x="572" y="423" font-family="{FONT}" font-size="11.5" fill="#334155"><tspan font-weight="700" fill="#059669">공간 병목 E2E (Soft-Loc)</tspan>: 미분가능 위치 표현 → <tspan font-weight="700">100% 정확 정지</tspan></text>

    <!-- 하단 정량 요약 -->
    <rect x="28" y="444" width="1044" height="84" rx="12" fill="#f8fafc" stroke="#cbd5e1"/>
    <text x="48" y="470" font-family="{FONT}" font-size="13" font-weight="700" fill="#0f172a">반사실 검증 (Counterfactual Test) — 물체 위치는 고정하고 배경 시각 특징만 무작위 교체</text>

    <rect x="48" y="484" width="320" height="32" rx="8" fill="#fef2f2" stroke="#fecaca"/>
    <text x="208" y="504" text-anchor="middle" font-family="{FONT}" font-size="12" font-weight="700" fill="#b91c1c">배경 교체 시 오판율: 27.4% → 0.0%</text>

    <rect x="386" y="484" width="324" height="32" rx="8" fill="#ecfdf5" stroke="#a7f3d0"/>
    <text x="548" y="504" text-anchor="middle" font-family="{FONT}" font-size="12" font-weight="700" fill="#047857">물체 근접 반응성: -0.3%p → +100%p</text>

    <rect x="728" y="484" width="324" height="32" rx="8" fill="#f5f3ff" stroke="#ddd6fe"/>
    <text x="890" y="504" text-anchor="middle" font-family="{FONT}" font-size="12" font-weight="700" fill="#6d28d9">E2E 미학습 정지율: 18.6% → 100.0%</text>
    '''
    _save_svg_and_png("03_research_question_diagnostic", _svg_shell(W, H, body))


def build_04_eg1_vs_eg2_protocol_y2() -> None:
    """카드 4: 물체 크기 편차 해결(바닥 접지선 y2) & 공정 실기 벤치마크 (내부 미팅 문구 제거)."""
    W, H = 1100, 450
    basket_uri = _b64_img("docs/v5/figures/real_1125/20261125_152041_바구니_trackF_center.jpg")
    chair_uri = _b64_img("docs/v5/figures/real_1125/20261125_155938_의자_trackF_center.jpg")
    defs = (
        '<clipPath id="clipBsk"><rect x="576" y="150" width="228" height="126" rx="8"/></clipPath>'
        '<clipPath id="clipChr"><rect x="820" y="150" width="228" height="126" rx="8"/></clipPath>'
    )

    body = f'''
    <rect x="28" y="24" width="196" height="24" rx="12" fill="#ede9fe"/>
    <text x="126" y="40" text-anchor="middle" font-family="{FONT}" font-size="11" font-weight="700" fill="#6d28d9">GEOMETRIC CONTROL</text>
    <text x="236" y="41" font-family="{FONT}" font-size="12.5" font-weight="700" fill="#475569">물체 크기에 무관한 일관된 정지 거리 제어 및 공정 실기 평가 프로토콜</text>
    <text x="28" y="76" font-family="{FONT}" font-size="22" font-weight="700" fill="#0f172a">바닥 접지선(y2) 기반 정지 정합 &amp; 다물체 비교 벤치마크</text>

    <!-- 좌측: 3x3 공정 평가 매트릭스 -->
    <rect x="28" y="96" width="512" height="332" rx="14" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5"/>
    <text x="46" y="124" font-family="{FONT}" font-size="14" font-weight="700" fill="#0f172a">A. 동일 조건 3×3 실기 평가 매트릭스 (9경로 × 3객체)</text>

    <rect x="46" y="138" width="476" height="182" rx="10" fill="#f8fafc" stroke="#cbd5e1"/>
    <rect x="82" y="152" width="104" height="34" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="1.5"/>
    <text x="134" y="174" text-anchor="middle" font-family="{FONT}" font-size="12" font-weight="700" fill="#14532d">목표 위치 A</text>
    <rect x="232" y="152" width="104" height="34" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="1.5"/>
    <text x="284" y="174" text-anchor="middle" font-family="{FONT}" font-size="12" font-weight="700" fill="#14532d">목표 위치 B</text>
    <rect x="382" y="152" width="104" height="34" rx="8" fill="#dcfce7" stroke="#16a34a" stroke-width="1.5"/>
    <text x="434" y="174" text-anchor="middle" font-family="{FONT}" font-size="12" font-weight="700" fill="#14532d">목표 위치 C</text>

    <g stroke="#3b82f6" stroke-opacity="0.45" stroke-width="1.6" stroke-dasharray="4 3">
      <line x1="134" y1="272" x2="134" y2="186"/><line x1="134" y1="272" x2="284" y2="186"/><line x1="134" y1="272" x2="434" y2="186"/>
      <line x1="284" y1="272" x2="134" y2="186"/><line x1="284" y1="272" x2="284" y2="186"/><line x1="284" y1="272" x2="434" y2="186"/>
      <line x1="434" y1="272" x2="134" y2="186"/><line x1="434" y1="272" x2="284" y2="186"/><line x1="434" y1="272" x2="434" y2="186"/>
    </g>
    <rect x="174" y="214" width="220" height="28" rx="14" fill="#ffffff" stroke="#93c5fd"/>
    <text x="284" y="232" text-anchor="middle" font-family="{FONT}" font-size="11.5" font-weight="700" fill="#1d4ed8">공통 9경로 (출발 3곳 × 도착 3곳)</text>

    <rect x="82" y="272" width="104" height="34" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>
    <text x="134" y="294" text-anchor="middle" font-family="{FONT}" font-size="12" font-weight="700" fill="#1e3a8a">출발 1 (좌)</text>
    <rect x="232" y="272" width="104" height="34" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>
    <text x="284" y="294" text-anchor="middle" font-family="{FONT}" font-size="12" font-weight="700" fill="#1e3a8a">출발 2 (중)</text>
    <rect x="382" y="272" width="104" height="34" rx="8" fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>
    <text x="434" y="294" text-anchor="middle" font-family="{FONT}" font-size="12" font-weight="700" fill="#1e3a8a">출발 3 (우)</text>

    <rect x="46" y="332" width="232" height="80" rx="10" fill="#eff6ff" stroke="#bfdbfe"/>
    <text x="60" y="354" font-family="{FONT}" font-size="12" font-weight="700" fill="#1d4ed8">v1 기준선 (9조건)</text>
    <text x="60" y="374" font-family="{FONT}" font-size="11" fill="#334155">• 학습 객체 1종(바구니) 평가</text>
    <text x="60" y="393" font-family="{FONT}" font-size="11" fill="#334155">• 고정 지시어 · 이산 8방향 제어</text>

    <rect x="290" y="332" width="232" height="80" rx="10" fill="#ecfdf5" stroke="#a7f3d0"/>
    <text x="304" y="354" font-family="{FONT}" font-size="12" font-weight="700" fill="#047857">v2 제안 모델 (27조건)</text>
    <text x="304" y="374" font-family="{FONT}" font-size="11" fill="#334155">• 바구니 + 미학습 의자·책상 3종</text>
    <text x="304" y="393" font-family="{FONT}" font-size="11" fill="#334155">• 성공률·접촉·잔여거리(cm)·지연 측정</text>

    <!-- 우측: 바닥 접지선 y2 -->
    <rect x="558" y="96" width="514" height="332" rx="14" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5"/>
    <text x="576" y="124" font-family="{FONT}" font-size="14" font-weight="700" fill="#0f172a">B. 물체 크기 편차 해결 — 바닥 접지선 (y2 = cy + h/2)</text>
    <text x="576" y="142" font-family="{FONT}" font-size="11" fill="#475569">같은 화면 면적이어도 큰 의자는 멀리 멈춤 → 물체 바닥선(y2)으로 거리 통일</text>

    <image href="{basket_uri}" x="576" y="150" width="228" height="126" preserveAspectRatio="xMidYMid slice" clip-path="url(#clipBsk)"/>
    <rect x="576" y="150" width="228" height="126" rx="8" fill="none" stroke="#94a3b8"/>
    <line x1="576" y1="268" x2="804" y2="268" stroke="#22c55e" stroke-width="3" stroke-dasharray="6 3"/>
    <rect x="584" y="157" width="136" height="20" rx="5" fill="rgba(15,23,42,0.82)"/>
    <text x="590" y="171" font-family="{FONT}" font-size="10.5" font-weight="700" fill="#86efac">바구니 · 바닥선 y2=1.00</text>

    <image href="{chair_uri}" x="820" y="150" width="228" height="126" preserveAspectRatio="xMidYMid slice" clip-path="url(#clipChr)"/>
    <rect x="820" y="150" width="228" height="126" rx="8" fill="none" stroke="#94a3b8"/>
    <line x1="820" y1="250" x2="1048" y2="250" stroke="#f59e0b" stroke-width="3" stroke-dasharray="6 3"/>
    <rect x="828" y="157" width="130" height="20" rx="5" fill="rgba(15,23,42,0.82)"/>
    <text x="834" y="171" font-family="{FONT}" font-size="10.5" font-weight="700" fill="#fde68a">의자 · 바닥선 y2=0.82</text>

    <rect x="576" y="290" width="472" height="122" rx="10" fill="#f8fafc" stroke="#cbd5e1"/>
    <text x="592" y="314" font-family="{FONT}" font-size="12.5" font-weight="700" fill="#0f172a">바닥 접지선(y2) 정지 정합 검증 결과</text>
    <text x="592" y="338" font-family="{FONT}" font-size="11.5" fill="#334155">• <tspan font-weight="700" fill="#059669">작은 물체 (바구니 y2=1.00)</tspan>: 100.0% 정확 정지 유지</text>
    <text x="592" y="359" font-family="{FONT}" font-size="11.5" fill="#334155">• <tspan font-weight="700" fill="#2563eb">큰 물체 (의자 y2=0.82)</tspan>: 원거리 조기 정지 100% → <tspan font-weight="700">0.0%</tspan> 개선</text>
    <text x="592" y="380" font-family="{FONT}" font-size="11.5" fill="#334155">• <tspan font-weight="700" fill="#6d28d9">배경 교체 강건성</tspan>: 반사실 시각 셔플 뒤집힘 <tspan font-weight="700">0.2%</tspan> 유지</text>
    <text x="592" y="401" font-family="{FONT}" font-size="10.5" fill="#64748b">실기 로그(.h5 / CSV) 기반 자동 정량 집계 파이프라인 구축</text>
    '''
    _save_svg_and_png("04_eg1_vs_eg2_protocol_y2", _svg_shell(W, H, body, defs))


def build_05_publications_awards_stack() -> None:
    """카드 5: 논문 · 수상 · 프로젝트 및 기술 스택."""
    W, H = 1100, 410
    body = f'''
    <rect x="28" y="24" width="210" height="24" rx="12" fill="#e0f2fe"/>
    <text x="133" y="40" text-anchor="middle" font-family="{FONT}" font-size="11" font-weight="700" fill="#0369a1">TRACK RECORD &amp; STACK</text>
    <text x="250" y="41" font-family="{FONT}" font-size="12.5" font-weight="700" fill="#475569">논문 연구 · ROS2/Jetson 실기 배포 · Multi-Agent/RAG 엔지니어링</text>
    <text x="28" y="76" font-family="{FONT}" font-size="22" font-weight="700" fill="#0f172a">논문 · 수상 · 프로젝트 및 기술 스택</text>

    <rect x="28" y="96" width="528" height="290" rx="14" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5"/>
    <text x="46" y="124" font-family="{FONT}" font-size="14" font-weight="700" fill="#0f172a">Publications &amp; Awards</text>

    <rect x="44" y="138" width="496" height="60" rx="10" fill="#eff6ff" stroke="#bfdbfe"/>
    <rect x="56" y="148" width="84" height="19" rx="9" fill="#2563eb"/>
    <text x="98" y="161" text-anchor="middle" font-family="{FONT}" font-size="10" font-weight="700" fill="#ffffff">1저자 · 심사중</text>
    <text x="150" y="162" font-family="{FONT}" font-size="12" font-weight="700" fill="#0f172a">EdgeGround-VLA (JKSCI, KCI 등재지)</text>
    <text x="56" y="184" font-family="{FONT}" font-size="10.8" fill="#475569">On-Device Goal-Directed Navigation via Dual Frozen Grounding Encoders (2026)</text>

    <rect x="44" y="208" width="496" height="60" rx="10" fill="#ecfdf5" stroke="#a7f3d0"/>
    <rect x="56" y="218" width="84" height="19" rx="9" fill="#059669"/>
    <text x="98" y="231" text-anchor="middle" font-family="{FONT}" font-size="10" font-weight="700" fill="#ffffff">2저자 · 우수상</text>
    <text x="150" y="232" font-family="{FONT}" font-size="12" font-weight="700" fill="#0f172a">RAG-Based Loneliness Analysis (JCCT 10(6), 2024)</text>
    <text x="56" y="254" font-family="{FONT}" font-size="10.8" fill="#475569">IPACT 2024 Outstanding Paper Award · RAG 기반 한국인 외로움 분석 대화형 AI</text>

    <rect x="44" y="278" width="496" height="94" rx="10" fill="#f8fafc" stroke="#cbd5e1"/>
    <text x="58" y="300" font-family="{FONT}" font-size="12" font-weight="700" fill="#0f172a">주요 수상 및 시스템 프로젝트</text>
    <text x="58" y="321" font-family="{FONT}" font-size="11.2" fill="#334155">• <tspan font-weight="700">2026 캡스톤 은상</tspan> · <tspan font-weight="700">2025 강남톤 대상(공과대학장상)</tspan></text>
    <text x="58" y="341" font-family="{FONT}" font-size="11.2" fill="#334155">• <tspan font-weight="700">Ladi (최우수상)</tspan>: KiwiBM25+FAISS+CrossEncoder · <tspan font-weight="700">Liberty (우수상)</tspan>: LangGraph</text>
    <text x="58" y="361" font-family="{FONT}" font-size="11.2" fill="#334155">• <tspan font-weight="700">Serbot 2 통합</tspan>: Jetson Orin Linux + ROS2 드라이버·주행 제어 스택</text>

    <rect x="572" y="96" width="500" height="290" rx="14" fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5"/>
    <text x="590" y="124" font-family="{FONT}" font-size="14" font-weight="700" fill="#0f172a">Core Technical Stack</text>

    <rect x="588" y="138" width="468" height="64" rx="10" fill="#f5f3ff" stroke="#ddd6fe"/>
    <text x="602" y="159" font-family="{FONT}" font-size="11.5" font-weight="700" fill="#6d28d9">AI · Vision-Language-Action (VLA)</text>
    <text x="602" y="180" font-family="{MONO}" font-size="11" fill="#1e293b">PyTorch · HuggingFace · LoRA/PEFT · Flow Matching</text>
    <text x="602" y="194" font-family="{MONO}" font-size="10.5" fill="#475569">Florence-2 · OWLv2 · Kosmos-2 · SigLIP · PaliGemma2</text>

    <rect x="588" y="212" width="468" height="56" rx="10" fill="#ecfdf5" stroke="#a7f3d0"/>
    <text x="602" y="233" font-family="{FONT}" font-size="11.5" font-weight="700" fill="#047857">Robotics · On-Device Deployment</text>
    <text x="602" y="254" font-family="{MONO}" font-size="11" fill="#1e293b">ROS2 · Jetson Orin NX · Async Control (10Hz) · Linux</text>

    <rect x="588" y="278" width="468" height="48" rx="10" fill="#fffbeb" stroke="#fde68a"/>
    <text x="602" y="298" font-family="{FONT}" font-size="11.5" font-weight="700" fill="#b45309">Agent Orchestration &amp; Retrieval</text>
    <text x="602" y="317" font-family="{MONO}" font-size="11" fill="#1e293b">LangGraph · FAISS · KiwiBM25 · CrossEncoder · FastAPI</text>

    <rect x="588" y="336" width="468" height="36" rx="10" fill="#0f172a"/>
    <text x="822" y="358" text-anchor="middle" font-family="{MONO}" font-size="11" font-weight="700" fill="#f8fafc">minuum.github.io/EdgeGroundVLA  |  github.com/minuum  |  minwool0357@gmail.com</text>
    '''
    _save_svg_and_png("05_publications_awards_stack", _svg_shell(W, H, body))


def main() -> None:
    build_01_profile_roadmap()
    build_02_edgeground_vla_v1_v2()
    build_03_research_question_diagnostic()
    build_04_eg1_vs_eg2_protocol_y2()
    build_05_publications_awards_stack()


if __name__ == "__main__":
    main()

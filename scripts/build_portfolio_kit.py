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


def _mini_icon(kind: str, x: float, y: float, col: str) -> str:
    """모듈별 특성을 직관적으로 보여주는 논문용 미니 벡터 아이콘."""
    if kind == "dual_enc":
        # 이중 비전 인코더 스택 + 패치 그리드
        return (
            f'<g transform="translate({x},{y})">'
            f'<rect x="0" y="3" width="16" height="16" rx="1.5" fill="#E5E7EB" stroke="#6B7280" stroke-width="1"/>'
            f'<rect x="5" y="0" width="16" height="16" rx="1.5" fill="#FFFFFF" stroke="{col}" stroke-width="1.1"/>'
            f'<line x1="10.3" y1="0" x2="10.3" y2="16" stroke="{col}" stroke-width="0.6" opacity="0.6"/>'
            f'<line x1="15.6" y1="0" x2="15.6" y2="16" stroke="{col}" stroke-width="0.6" opacity="0.6"/>'
            f'<line x1="5" y1="8" x2="21" y2="8" stroke="{col}" stroke-width="0.6" opacity="0.6"/>'
            f'</g>'
        )
    if kind == "florence":
        # Encoder -> Decoder 블록
        return (
            f'<g transform="translate({x},{y})">'
            f'<rect x="0" y="2" width="10" height="16" rx="1.5" fill="#E5E7EB" stroke="#6B7280" stroke-width="1"/>'
            f'<rect x="13" y="2" width="10" height="16" rx="1.5" fill="#FFFFFF" stroke="{col}" stroke-width="1.1"/>'
            f'<line x1="10" y1="10" x2="13" y2="10" stroke="#6B7280" stroke-width="1.1"/>'
            f'</g>'
        )
    if kind == "florence_lora":
        # Encoder -> Decoder + LoRA 저랭크 어댑터 (A·B 다이아몬드)
        return (
            f'<g transform="translate({x},{y})">'
            f'<rect x="0" y="2" width="9" height="16" rx="1.5" fill="#E5E7EB" stroke="#6B7280" stroke-width="1"/>'
            f'<rect x="11" y="2" width="11" height="16" rx="1.5" fill="#FFF7ED" stroke="{col}" stroke-width="1.2"/>'
            f'<polygon points="16.5,5 20,10 16.5,15 13,10" fill="{col}" opacity="0.85"/>'
            f'</g>'
        )
    if kind == "bbox_win":
        # 6-프레임 BBox 스택
        return (
            f'<g transform="translate({x},{y})">'
            f'<rect x="0" y="4" width="18" height="14" rx="1" fill="#F9FAFB" stroke="#9CA3AF" stroke-width="0.9"/>'
            f'<rect x="3" y="1" width="18" height="14" rx="1" fill="#FFFFFF" stroke="{col}" stroke-width="1.1"/>'
            f'<rect x="7" y="4" width="8" height="7" fill="none" stroke="{col}" stroke-width="1.2"/>'
            f'</g>'
        )
    if kind == "text_parse":
        # 문자열 토큰 <loc> 파싱 아이콘
        return (
            f'<g transform="translate({x},{y})">'
            f'<rect x="0" y="1" width="23" height="18" rx="1.5" fill="#F3F4F6" stroke="#6B7280" stroke-width="1"/>'
            f'<text x="11.5" y="13" text-anchor="middle" font-family="{PUB_MONO}" font-size="7.5" font-weight="700" fill="{col}">&lt;/&gt;</text>'
            f'</g>'
        )
    if kind == "softloc_dist":
        # Softmax 확률 분포 종형 곡선 (Soft-Loc Expectation)
        return (
            f'<g transform="translate({x},{y})">'
            f'<line x1="0" y1="18" x2="24" y2="18" stroke="#9CA3AF" stroke-width="0.9"/>'
            f'<path d="M1,18 C6,18 8,3 12,3 C16,3 18,18 23,18" fill="{col}" fill-opacity="0.22" stroke="{col}" stroke-width="1.5"/>'
            f'<line x1="12" y1="3" x2="12" y2="18" stroke="{col}" stroke-width="1.0" stroke-dasharray="2 1.5"/>'
            f'</g>'
        )
    if kind == "compass8":
        # 8방향 이산 나침반
        return (
            f'<g transform="translate({x},{y})">'
            f'<circle cx="11" cy="10" r="9" fill="#F8FAFC" stroke="#9CA3AF" stroke-width="0.9"/>'
            f'<line x1="11" y1="3" x2="11" y2="17" stroke="#CBD5E1" stroke-width="0.8"/>'
            f'<line x1="4" y1="10" x2="18" y2="10" stroke="#CBD5E1" stroke-width="0.8"/>'
            f'<line x1="6" y1="5" x2="16" y2="15" stroke="#CBD5E1" stroke-width="0.8"/>'
            f'<line x1="16" y1="5" x2="6" y2="15" stroke="#CBD5E1" stroke-width="0.8"/>'
            f'<line x1="11" y1="10" x2="11" y2="3" stroke="{col}" stroke-width="2.2" stroke-linecap="round"/>'
            f'<circle cx="11" cy="10" r="2" fill="{col}"/>'
            f'</g>'
        )
    if kind == "flow_traj":
        # 연속 Flow Matching 궤적 벡터 필드
        return (
            f'<g transform="translate({x},{y})">'
            f'<rect x="0" y="1" width="23" height="18" rx="1.5" fill="#FFFFFF" stroke="#E5E7EB" stroke-width="0.9"/>'
            f'<path d="M3,16 Q10,13 19,4" fill="none" stroke="{col}" stroke-width="1.7" stroke-linecap="round"/>'
            f'<path d="M3,12 Q11,10 18,6" fill="none" stroke="{col}" stroke-width="1.0" opacity="0.45"/>'
            f'<circle cx="3" cy="16" r="1.8" fill="#9CA3AF"/>'
            f'<circle cx="19" cy="4" r="2.2" fill="{col}"/>'
            f'</g>'
        )
    return ""


def _arch_box(x: float, y: float, w: float, h: float, title: str, data_ex: str, sub: str, frozen: bool, col: str, icon_kind: str = "") -> str:
    """3단 고밀도 아키텍처 박스: [제목] -> [실제 값/데이터 구조 코드 칩] -> [간단 설명]."""
    bg = "#F3F4F6" if frozen else "#FFFFFF"
    stroke = "#9CA3AF" if frozen else col
    sw = "1.0" if frozen else "1.4"
    tc = "#374151" if frozen else "#111827"
    sc = "#6B7280" if frozen else "#4B5563"
    chip_bg = "#E5E7EB" if frozen else "#F8FAFC"
    chip_fg = "#1F2937" if frozen else col
    chip_bd = "#D1D5DB" if frozen else "#E2E8F0"
    status_icon = _lock_icon(x + w - 13, y + 5) if frozen else _train_icon(x + w - 13, y + 5, col)
    left_icon = _mini_icon(icon_kind, x + 7, y + (h - 20) / 2, col) if icon_kind else ""
    tx = x + (35 if icon_kind else 8)
    cw = w - (42 if icon_kind else 16)
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="2" fill="{bg}" stroke="{stroke}" stroke-width="{sw}"/>'
        f'{left_icon}{status_icon}'
        f'<text x="{tx}" y="{y + 16}" font-family="{PUB_FONT}" font-size="9.4" font-weight="700" fill="{tc}">{html.escape(title)}</text>'
        f'<rect x="{tx}" y="{y + 21}" width="{cw}" height="17" rx="2" fill="{chip_bg}" stroke="{chip_bd}" stroke-width="0.7"/>'
        f'<text x="{tx + 4}" y="{y + 32.5}" font-family="{PUB_MONO}" font-size="7.5" font-weight="700" fill="{chip_fg}">{html.escape(data_ex)}</text>'
        f'<text x="{tx}" y="{y + 51}" font-family="{PUB_FONT}" font-size="8.0" fill="{sc}">{html.escape(sub)}</text>'
    )


def _sg_badge(cx: float, cy: float) -> str:
    """학회 표준 Stop-Gradient (SG) 마커."""
    return (
        f'<rect x="{cx - 11}" y="{cy - 8}" width="22" height="16" rx="2" fill="#FFFFFF" stroke="#6B7280" stroke-width="1.0"/>'
        f'<text x="{cx}" y="{cy + 3.2}" text-anchor="middle" font-family="{PUB_MONO}" font-size="8.2" font-weight="700" fill="#4B5563">SG</text>'
    )


def _sparkbar(x: float, y: float, w: float, pct: float, col: str) -> str:
    """테이블 핵심 지표용 미니 인라인 프로그레스 바 (높이 4px)."""
    fw = max(2.0, w * max(0.0, min(1.0, pct)))
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="4" rx="1" fill="#E5E7EB"/>'
        f'<rect x="{x}" y="{y}" width="{fw:.1f}" height="4" rx="1" fill="{col}"/>'
    )


def build_01_profile_roadmap() -> None:
    """Figure 1: CVPR/ICRA Publication Style — (a) Same-Frame Inference, (b) Unified Table + Sparkbars, (c) High-Density Architecture & Gradient Flow."""
    W, H = 1100, 995
    same_img_uri = _b64_img("docs/v5/figures/inline/99f9060660c1394c.jpg")
    serbot_uri = _b64_img("docs/v5/figures/figure2_v2_hardware_web.png")

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

    # (b) LaTeX Booktabs 통일 기준 비교표 (EGVLA-1 Jetson 실측 지연 정정: 1.02s fp16 / 528ms cache avg)
    rows = [
        ("VLA Paradigm", "Modular VLA (2-Stage)", "Modular VLA (2-Stage)", "End-to-End VLA", None),
        ("VLM Backbone", "OWLv2 + Kosmos-2 (Frozen)", "Florence-2 (Frozen)", "Florence-2 (Decoder LoRA)", None),
        ("Grounding Interface", "2D BBox + Visual Proj. (SG)", "Text-Parsed <loc> BBox (SG)", "Soft-Loc Expectation (Diff.)", None),
        ("Action Head", "Discrete 8-Class MLP (1.128M)", "Flow-Matching Head (PiZeroFM)", "Flow-Matching Head (PiZeroFM)", None),
        ("Action Space", "Discrete 8-Way (3-DoF)", "Continuous 2D Velocity [vx, vy]", "Continuous 2D Velocity [vx, vy]", None),
        ("Deployment / Eval", "Real-Robot · Jetson Orin NX (On-Device)", "Real-Robot · Serbot 2 + GPU Server", "Offline Replay · 1,458 Real Frames", None),
        ("Real-Robot Success", "95.0% (Seen) / 92.0% (300 trials)", "45/45 (0 contact; unseen chair 15/15)", "Pending (Offline Verified)", (0.95, 1.0, None)),
        ("Unseen-Object Stop", "N/A (Single-Object Basket)", "93.0% (area >= 0.25, MOD-A3-AG)", "100.0% (Soft-Loc+) / 97.7% (Joint)", (None, 0.93, 1.0)),
        ("Visual-Swap Flip", "N/A", "27.4% -> 0.0% (a3 -> a3-ag23)", "14.9% -> 0.0% (Hidden -> Soft-Loc)", None),
        ("Inference Latency", "1.02s fp16 / ~528ms avg w/ cache", "397.1 ms Real (285.3 ms GB10)", "296.6 ms GB10 (283.8 ms Core)", None),
    ]

    tx0, tx1, tx2, tx3 = 36, 232, 514, 796
    ty0 = 296
    rh = 30
    tbl_svg = [
        f'<text x="36" y="268" font-family="{PUB_FONT}" font-size="12.5" font-weight="700" fill="{INK}">(b) Unified Architectural &amp; Empirical Comparison Across Generations</text>',
        f'<line x1="36" y1="{ty0-17}" x2="1064" y2="{ty0-17}" stroke="{INK}" stroke-width="1.6"/>',
        f'<text x="{tx0}" y="{ty0}" font-family="{PUB_FONT}" font-size="10.8" font-weight="700" fill="{INK}">Criterion</text>',
        f'<text x="{tx1}" y="{ty0}" font-family="{PUB_FONT}" font-size="10.8" font-weight="700" fill="{C1}">EGVLA-1 (Gen 1)</text>',
        f'<text x="{tx2}" y="{ty0}" font-family="{PUB_FONT}" font-size="10.8" font-weight="700" fill="{C2}">EGVLA-2 (Gen 2)</text>',
        f'<text x="{tx3}" y="{ty0}" font-family="{PUB_FONT}" font-size="10.8" font-weight="700" fill="{C3}">EGVLA-3 (Gen 3)</text>',
        f'<line x1="36" y1="{ty0+9}" x2="1064" y2="{ty0+9}" stroke="{INK}" stroke-width="0.9"/>',
    ]

    for i, (crit, v1, v2, v3, bars) in enumerate(rows):
        ry = ty0 + 30 + i * rh
        if i % 2 == 1:
            tbl_svg.append(f'<rect x="36" y="{ry-18}" width="1028" height="{rh}" fill="#F9FAFB"/>')
        if i == 5:
            tbl_svg.append(f'<line x1="36" y1="{ry-18}" x2="1064" y2="{ry-18}" stroke="#9CA3AF" stroke-width="0.7" stroke-dasharray="3 2"/>')
        tbl_svg.append(f'<text x="{tx0}" y="{ry}" font-family="{PUB_FONT}" font-size="10.4" font-weight="700" fill="{INK}">{html.escape(crit)}</text>')
        for j, (xv, val, col) in enumerate(((tx1, v1, C1), (tx2, v2, C2), (tx3, v3, C3))):
            fill_c = MUT if val.startswith(("N/A", "Pending")) else INK
            fw = "700" if any(k in val for k in ("95.0%", "45/45", "100.0%", "0.0%", "End-to-End VLA")) else "400"
            tbl_svg.append(f'<text x="{xv}" y="{ry - (2 if bars and bars[j] is not None else 0)}" font-family="{PUB_FONT}" font-size="10.2" font-weight="{fw}" fill="{fill_c}">{html.escape(val)}</text>')
            if bars and bars[j] is not None:
                tbl_svg.append(_sparkbar(xv, ry + 3, 120, bars[j], col))

    bot_y = ty0 + 30 + len(rows) * rh - 12
    tbl_svg.append(f'<line x1="36" y1="{bot_y}" x2="1064" y2="{bot_y}" stroke="{INK}" stroke-width="1.6"/>')

    # (c) 3줄 고밀도 아키텍처 & 텐서 차원 & Gradient Flow 다이어그램
    cy_hdr = bot_y + 34
    bh = 60
    r1y, r2y, r3y = cy_hdr + 26, cy_hdr + 110, cy_hdr + 194

    def _arrow(x1: float, y: float, x2: float, col: str = "#374151", lbl: str = "") -> str:
        s = f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{col}" stroke-width="1.3" marker-end="url(#pubArr)"/>'
        if lbl:
            s += f'<text x="{(x1+x2)/2:.1f}" y="{y - 5}" text-anchor="middle" font-family="{PUB_MONO}" font-size="7.4" fill="{MUT}">{html.escape(lbl)}</text>'
        return s

    def _input_thumb(y: float, cid: str, prompt_txt: str) -> str:
        return (
            f'<rect x="104" y="{y}" width="122" height="{bh}" rx="2" fill="#FFFFFF" stroke="#6B7280" stroke-width="1.0"/>'
            f'<clipPath id="{cid}"><rect x="109" y="{y+7}" width="44" height="46" rx="1"/></clipPath>'
            f'<image href="{same_img_uri}" x="109" y="{y+7}" width="44" height="46" preserveAspectRatio="xMidYMid slice" clip-path="url(#{cid})"/>'
            f'<rect x="109" y="{y+7}" width="44" height="46" fill="none" stroke="#9CA3AF" stroke-width="0.6"/>'
            f'<text x="158" y="{y+18}" font-family="{PUB_FONT}" font-size="9.0" font-weight="700" fill="{INK}">RGB+Text</text>'
            f'<text x="158" y="{y+33}" font-family="{PUB_FONT}" font-size="8.0" fill="{MUT}">{html.escape(prompt_txt)}</text>'
            f'<text x="158" y="{y+49}" font-family="{PUB_MONO}" font-size="7.5" fill="#6B7280">1280x720</text>'
        )

    def _output_robot(y: float, cid: str, col: str, l1: str, l2: str, l3: str) -> str:
        return (
            f'<rect x="926" y="{y}" width="138" height="{bh}" rx="2" fill="#FFFFFF" stroke="{col}" stroke-width="1.3"/>'
            f'<clipPath id="{cid}"><rect x="931" y="{y+7}" width="36" height="46" rx="1"/></clipPath>'
            f'<image href="{serbot_uri}" x="931" y="{y+7}" width="36" height="46" preserveAspectRatio="xMidYMid cover" clip-path="url(#{cid})"/>'
            f'<text x="972" y="{y+18}" font-family="{PUB_FONT}" font-size="9.0" font-weight="700" fill="{col}">{html.escape(l1)}</text>'
            f'<text x="972" y="{y+33}" font-family="{PUB_FONT}" font-size="8.0" fill="{INK}">{html.escape(l2)}</text>'
            f'<text x="972" y="{y+49}" font-family="{PUB_MONO}" font-size="7.4" fill="{MUT}">{html.escape(l3)}</text>'
        )

    arch_svg = [
        f'<text x="36" y="{cy_hdr}" font-family="{PUB_FONT}" font-size="12.5" font-weight="700" fill="{INK}">(c) Architecture, Tensor Flow &amp; Gradient Path Comparison</text>',
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
        f'<text x="36" y="{r1y + 24}" font-family="{PUB_FONT}" font-size="10.8" font-weight="700" fill="{C1}">EGVLA-1</text>',
        f'<text x="36" y="{r1y + 39}" font-family="{PUB_FONT}" font-size="8.5" fill="{MUT}">Modular</text>',
        _input_thumb(r1y, "inG1", "basket only"),
        _arrow(226, r1y + 30, 248),
        _arch_box(250, r1y, 206, bh, "OWLv2 + Kosmos-2", "box=[0.28,0.65,0.03]", "Dual Frozen (458M · 1.02s)", True, C1, "dual_enc"),
        _arrow(456, r1y + 30, 512),
        _sg_badge(484, r1y + 30),
        _arch_box(514, r1y, 180, bh, "2D BBox + Proj.", "x: float[6x260]=1560d", "6-Frame Window Fusion", False, C1, "bbox_win"),
        _arrow(694, r1y + 30, 718),
        _arch_box(720, r1y, 182, bh, "Discrete MLP Head", "argmax(p[8]) -> cls=1", "1.128M (0.25%) · <1ms", False, C1, "compass8"),
        _arrow(902, r1y + 30, 924),
        _output_robot(r1y, "outG1", C1, "FORWARD", "[1.15, 0.0, 0.0]", "Jetson 10Hz ctrl"),

        # Row 2: EGVLA-2
        f'<text x="36" y="{r2y + 24}" font-family="{PUB_FONT}" font-size="10.8" font-weight="700" fill="{C2}">EGVLA-2</text>',
        f'<text x="36" y="{r2y + 39}" font-family="{PUB_FONT}" font-size="8.5" fill="{MUT}">Modular</text>',
        _input_thumb(r2y, "inG2", "open-vocab"),
        _arrow(226, r2y + 30, 248),
        _arch_box(250, r2y, 206, bh, "Florence-2 (0.23B)", "\"<loc_217><loc_540>..\"", "Frozen Enc+Dec (283ms)", True, C2, "florence"),
        _arrow(456, r2y + 30, 512),
        _sg_badge(484, r2y + 30),
        _arch_box(514, r2y, 180, bh, "Text-Parsed <loc>", "b=[0.28,0.65,0.03,1]", "Regex 3-win(12d)+768d", True, C2, "text_parse"),
        _arrow(694, r2y + 30, 718),
        _arch_box(720, r2y, 182, bh, "Flow-Matching Head", "a: float[8x2] (K=8)", "PiZeroFM + 50% Swap", False, C2, "flow_traj"),
        _arrow(902, r2y + 30, 924),
        _output_robot(r2y, "outG2", C2, "[0.64, -0.32]", "-26.4 deg + S2", "Real 397ms · 0c"),

        # Row 3: EGVLA-3
        f'<text x="36" y="{r3y + 24}" font-family="{PUB_FONT}" font-size="10.8" font-weight="700" fill="{C3}">EGVLA-3</text>',
        f'<text x="36" y="{r3y + 39}" font-family="{PUB_FONT}" font-size="8.5" fill="{MUT}">End-to-End</text>',
        _input_thumb(r3y, "inG3", "open-vocab"),
        _arrow(226, r3y + 30, 248),
        _arch_box(250, r3y, 206, bh, "Florence-2 (LoRA)", "z: float[4x1000] logit", "Prefix-Forced KV-Cache", False, C3, "florence_lora"),
        _arrow(456, r3y + 30, 512, C3, "logits"),
        _arch_box(514, r3y, 180, bh, "Soft-Loc (Expected)", "u=sum(softmax(z)*i/1k)", "0% Parse · Diff. BBox", False, C3, "softloc_dist"),
        _arrow(694, r3y + 30, 718, C3, "feat"),
        _arch_box(720, r3y, 182, bh, "Flow-Matching Head", "a: float[8x2] (Joint)", "PiZeroFM + KL Reg.", False, C3, "flow_traj"),
        _arrow(902, r3y + 30, 924, C3),
        _output_robot(r3y, "outG3", C3, "[0.59, -0.32]", "-28.8 deg E2E", "GB10 297ms"),

        # Gradient Backprop Arc
        f'<path d="M 811,{r3y + bh} L 811,{r3y + bh + 20} L 353,{r3y + bh + 20} L 353,{r3y + bh + 3}" fill="none" stroke="{C3}" stroke-width="1.5" stroke-dasharray="4 2.5" marker-end="url(#gradArr)"/>',
        f'<rect x="502" y="{r3y + bh + 12}" width="206" height="15" fill="#FFFFFF"/>',
        f'<text x="605" y="{r3y + bh + 23}" text-anchor="middle" font-family="{PUB_FONT}" font-size="9.0" font-weight="700" fill="{C3}">End-to-End Gradient Flow (L_action)</text>',
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
  <text x="36" y="{H - 12}" font-family="{PUB_FONT}" font-size="9.1" fill="{MUT}">Figure 1. Evolution from Modular VLA (EGVLA-1/2) to End-to-End VLA (EGVLA-3): (a) same-frame inference, (b) unified evaluation, and (c) detailed tensor &amp; gradient flow.</text>
</svg>'''
    _save_svg_and_png("01_profile_roadmap", svg_str)


def build_02_edgeground_vla_v1_v2() -> None:
    """Figure 2: CVPR/ICRA Publication Style — Real-Robot Benchmarks, Latency Separation & Offline Generalization."""
    W, H = 1100, 710
    bsk_uri = _b64_img("docs/v5/figures/real_1125/20261125_152041_바구니_trackF_center.jpg")
    chr_uri = _b64_img("docs/v5/figures/real_1125/20261125_155938_의자_trackF_center.jpg")

    C1, C2, C3 = "#0072B2", "#009E73", "#D55E00"
    INK, MUT, RULE = "#111827", "#4B5563", "#D1D5DB"

    # ── Panel (a): EGVLA-1 On-Device Real-Robot Success (N=300) ──
    # 바 차트 좌표 (x=68..490, y=96..270)
    ay_top, ay_bot = 104, 272
    ah = ay_bot - ay_top
    bars_a = [
        ("pos1\n(Seen)", 95.0, "95/100", C1, 1.0),
        ("pos2\n(-50 cm)", 91.0, "91/100", C1, 0.78),
        ("pos3\n(+50 cm)", 90.0, "90/100", C1, 0.78),
        ("Overall\n(N=300)", 92.0, "276/300", "#111827", 0.92),
    ]
    a_svg = [
        f'<text x="36" y="34" font-family="{PUB_FONT}" font-size="12.2" font-weight="700" fill="{INK}">(a) EGVLA-1: On-Device Real-Robot Success (Jetson Orin NX, N = 300)</text>',
        f'<line x1="36" y1="43" x2="526" y2="43" stroke="{RULE}" stroke-width="0.8"/>',
        # Parameter Efficiency 인셋 콜아웃 (우상단)
        f'<rect x="272" y="54" width="254" height="36" rx="2" fill="#F8FAFC" stroke="{C1}" stroke-width="1.0"/>',
        f'<text x="280" y="69" font-family="{PUB_FONT}" font-size="9.1" font-weight="700" fill="{C1}">Trained: 1.128M / 459M (0.25%)</text>',
        f'<text x="280" y="83" font-family="{PUB_FONT}" font-size="8.2" fill="{MUT}">OWLv2 962ms fp16 (~528ms cache avg) · 10Hz ctrl</text>',
    ]
    # Y축 눈금 (0, 50, 80, 100%)
    for pct in (0, 50, 80, 100):
        yy = ay_bot - ah * (pct / 100.0)
        dash = ' stroke-dasharray="3 2"' if pct > 0 else ""
        scol = "#111827" if pct == 0 else "#E5E7EB"
        a_svg.append(f'<line x1="72" y1="{yy:.1f}" x2="514" y2="{yy:.1f}" stroke="{scol}" stroke-width="0.9"{dash}/>')
        a_svg.append(f'<text x="64" y="{yy+3:.1f}" text-anchor="end" font-family="{PUB_MONO}" font-size="8.5" fill="{MUT}">{pct}%</text>')
    a_svg.append(f'<line x1="72" y1="{ay_top}" x2="72" y2="{ay_bot}" stroke="{INK}" stroke-width="1.0"/>')

    for i, (lbl, val, sub, col, op) in enumerate(bars_a):
        bx = 98 + i * 104
        bw = 56
        bh = ah * (val / 100.0)
        by = ay_bot - bh
        a_svg.append(f'<rect x="{bx}" y="{by:.1f}" width="{bw}" height="{bh:.1f}" fill="{col}" fill-opacity="{op}"/>')
        a_svg.append(f'<text x="{bx+bw/2}" y="{by-5:.1f}" text-anchor="middle" font-family="{PUB_MONO}" font-size="9.6" font-weight="700" fill="{INK}">{val:.1f}%</text>')
        a_svg.append(f'<text x="{bx+bw/2}" y="{by+14:.1f}" text-anchor="middle" font-family="{PUB_MONO}" font-size="8.0" fill="#FFFFFF">{sub}</text>')
        l1, l2 = lbl.split("\n")
        a_svg.append(f'<text x="{bx+bw/2}" y="{ay_bot+15}" text-anchor="middle" font-family="{PUB_FONT}" font-size="9.4" font-weight="700" fill="{INK}">{l1}</text>')
        a_svg.append(f'<text x="{bx+bw/2}" y="{ay_bot+28}" text-anchor="middle" font-family="{PUB_FONT}" font-size="8.3" fill="{MUT}">{l2}</text>')

    # ── Panel (b): EGVLA-2 Multi-Object Real-Robot Eval (11/25, N=45) ──
    b_svg = [
        f'<text x="564" y="34" font-family="{PUB_FONT}" font-size="12.2" font-weight="700" fill="{INK}">(b) EGVLA-2: Multi-Object Real-Robot Arrival (11/25, N = 45, 0 Contact)</text>',
        f'<line x1="564" y1="43" x2="1064" y2="43" stroke="{RULE}" stroke-width="0.8"/>',
        # 사진 1: 바구니
        f'<text x="564" y="61" font-family="{PUB_FONT}" font-size="9.8" font-weight="700" fill="{C2}">Seen Object: Basket (30/30 · 0 Contact)</text>',
        f'<clipPath id="fig2Bsk"><rect x="564" y="68" width="240" height="132"/></clipPath>',
        f'<image href="{bsk_uri}" x="564" y="68" width="240" height="132" preserveAspectRatio="xMidYMid slice" clip-path="url(#fig2Bsk)"/>',
        f'<rect x="564" y="68" width="240" height="132" fill="none" stroke="{INK}" stroke-width="0.9"/>',
        f'<rect x="568" y="180" width="154" height="16" fill="rgba(255,255,255,0.90)"/>',
        f'<text x="572" y="191" font-family="{PUB_MONO}" font-size="8.2" font-weight="700" fill="{C2}">max_area=0.295 · 396.5 ms</text>',
        # 사진 2: 미학습 의자
        f'<text x="824" y="61" font-family="{PUB_FONT}" font-size="9.8" font-weight="700" fill="{C2}">Unseen Object: Chair (15/15 · 0 Contact)</text>',
        f'<clipPath id="fig2Chr"><rect x="824" y="68" width="240" height="132"/></clipPath>',
        f'<image href="{chr_uri}" x="824" y="68" width="240" height="132" preserveAspectRatio="xMidYMid slice" clip-path="url(#fig2Chr)"/>',
        f'<rect x="824" y="68" width="240" height="132" fill="none" stroke="{INK}" stroke-width="0.9"/>',
        f'<rect x="828" y="180" width="154" height="16" fill="rgba(255,255,255,0.90)"/>',
        f'<text x="832" y="191" font-family="{PUB_MONO}" font-size="8.2" font-weight="700" fill="{C2}">max_area=0.287 · 410.8 ms</text>',
        # 하단 접촉 감소 대조 미니 바 (11/16 a3-s2 vs 11/25 MOD-DR-Safe)
        f'<text x="564" y="224" font-family="{PUB_FONT}" font-size="9.8" font-weight="700" fill="{INK}">Real-Robot Contact Events (45 trials each, Serbot 2 + GPU Server)</text>',
        f'<text x="564" y="247" font-family="{PUB_FONT}" font-size="9.2" fill="{MUT}">a3-s2 (11/16, no S2 decel)</text>',
        f'<rect x="720" y="236" width="220" height="14" fill="#F3F4F6"/>',
        f'<rect x="720" y="236" width="220" height="14" fill="{C3}"/>',
        f'<text x="948" y="247" font-family="{PUB_MONO}" font-size="9.4" font-weight="700" fill="{C3}">6 contacts</text>',
        f'<text x="564" y="274" font-family="{PUB_FONT}" font-size="9.2" font-weight="700" fill="{INK}">MOD-DR-Safe (11/25)</text>',
        f'<rect x="720" y="263" width="220" height="14" fill="#F3F4F6"/>',
        f'<rect x="720" y="263" width="4" height="14" fill="{C2}"/>',
        f'<text x="948" y="274" font-family="{PUB_MONO}" font-size="9.4" font-weight="700" fill="{C2}">0 contacts (-100%)</text>',
        f'<text x="564" y="298" font-family="{PUB_FONT}" font-size="8.5" fill="{MUT}">* Model preemptive stop before near_area=0.25 in 45/45 runs (mean 4.1 decel steps).</text>',
    ]

    # ── Panel (c): Inference Latency Acceleration (Hardware Separated) ──
    c_svg = [
        f'<text x="36" y="352" font-family="{PUB_FONT}" font-size="12.2" font-weight="700" fill="{INK}">(c) EGVLA-2 Latency Acceleration (Hardware Explicitly Separated)</text>',
        f'<line x1="36" y1="361" x2="526" y2="361" stroke="{RULE}" stroke-width="0.8"/>',
    ]
    lat_items = [
        ("GB10 GPU Benchmark (330 frames, beam_bench.json)", None, None, None, None),
        ("beam=3 fp32 (Baseline)", 590.7, 960.0, "#9CA3AF", "590.7 ms"),
        ("beam=1 fp16 (MOD-DR-Fast)", 283.0, 960.0, C1, "283.0 ms  (2.09x)"),
        ("Real-Robot Server (soda GPU, 11/25 45 real runs)", None, None, None, None),
        ("beam=3 fp32 (Prior Real)", 945.0, 960.0, "#9CA3AF", "~945 ms"),
        ("beam=1 fp16 (11/25 Real)", 397.1, 960.0, C2, "397.1 ms  (2.3x+)"),
    ]
    ly = 386
    for lbl, val, vmax, col, vtxt in lat_items:
        if val is None:
            c_svg.append(f'<text x="36" y="{ly}" font-family="{PUB_FONT}" font-size="9.6" font-weight="700" fill="{INK}">{html.escape(lbl)}</text>')
            ly += 24
        else:
            bw = 240.0 * (val / vmax)
            c_svg.append(f'<text x="44" y="{ly}" font-family="{PUB_FONT}" font-size="9.2" fill="{MUT}">{html.escape(lbl)}</text>')
            c_svg.append(f'<rect x="196" y="{ly-10}" width="240" height="13" fill="#F3F4F6"/>')
            c_svg.append(f'<rect x="196" y="{ly-10}" width="{bw:.1f}" height="13" fill="{col}"/>')
            c_svg.append(f'<text x="444" y="{ly}" font-family="{PUB_MONO}" font-size="9.2" font-weight="700" fill="{col if col != "#9CA3AF" else MUT}">{vtxt}</text>')
            ly += 28
    c_svg.append(f'<text x="36" y="618" font-family="{PUB_FONT}" font-size="8.6" fill="{MUT}">* beam=1 fp16 preserves 99.7% BBox agreement vs. beam=3 (cx MAE 0.0016, E2E serve 285.3 ms on GB10).</text>')

    # ── Panel (d): Offline Generalization & Safety Benchmarks (Booktabs) ──
    d_rows = [
        ("Unseen Chair Center-Start Go", "0.0% (a3)", "100.0% (a3-ag23)", 1.0, C2),
        ("Unseen Chair Close-Stop (a>=0.25)", "30.2% (a3-s2)", "93.0% (a3-ag23)", 0.93, C2),
        ("Steering Alignment vs. Baseline", "25.13 deg (B1)", "0.54 deg (98.7% L/R)", 0.987, C1),
        ("Visual-Swap Stop Flip Rate", "27.4% (a3)", "0.0% (a3-ag23)", 0.0, C2),
        ("Negative-Prompt Safety Stop", "0.0% (no text)", "99.5% (MOD-V4R)", 0.995, C1),
        ("BBox Horizontal-Flip Defense", "86.0% (V4)", "94.3% (MOD-V4R)", 0.943, C1),
    ]
    dy0 = 398
    drh = 35
    d_svg = [
        f'<text x="564" y="352" font-family="{PUB_FONT}" font-size="12.2" font-weight="700" fill="{INK}">(d) Offline Generalization &amp; Safety Audit (1,458 Real Frames)</text>',
        f'<line x1="564" y1="{dy0-17}" x2="1064" y2="{dy0-17}" stroke="{INK}" stroke-width="1.5"/>',
        f'<text x="564" y="{dy0}" font-family="{PUB_FONT}" font-size="9.8" font-weight="700" fill="{INK}">Metric (Offline Replay)</text>',
        f'<text x="792" y="{dy0}" font-family="{PUB_FONT}" font-size="9.8" font-weight="700" fill="{MUT}">Baseline</text>',
        f'<text x="904" y="{dy0}" font-family="{PUB_FONT}" font-size="9.8" font-weight="700" fill="{C2}">Proposed (EGVLA-2)</text>',
        f'<line x1="564" y1="{dy0+8}" x2="1064" y2="{dy0+8}" stroke="{INK}" stroke-width="0.9"/>',
    ]
    for i, (m, base_v, prop_v, pct, col) in enumerate(d_rows):
        ry = dy0 + 28 + i * drh
        if i % 2 == 1:
            d_svg.append(f'<rect x="564" y="{ry-17}" width="500" height="{drh}" fill="#F9FAFB"/>')
        d_svg.append(f'<text x="564" y="{ry}" font-family="{PUB_FONT}" font-size="9.4" font-weight="700" fill="{INK}">{html.escape(m)}</text>')
        d_svg.append(f'<text x="792" y="{ry}" font-family="{PUB_MONO}" font-size="8.8" fill="{MUT}">{html.escape(base_v)}</text>')
        d_svg.append(f'<text x="904" y="{ry-2}" font-family="{PUB_MONO}" font-size="9.0" font-weight="700" fill="{col}">{html.escape(prop_v)}</text>')
        d_svg.append(_sparkbar(904, ry + 3, 140, pct, col))
    dbot = dy0 + 28 + len(d_rows) * drh - 12
    d_svg.append(f'<line x1="564" y1="{dbot}" x2="1064" y2="{dbot}" stroke="{INK}" stroke-width="1.5"/>')

    svg_str = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>
  {"".join(a_svg)}
  {"".join(b_svg)}
  {"".join(c_svg)}
  {"".join(d_svg)}
  <text x="36" y="{H - 14}" font-family="{PUB_FONT}" font-size="9.1" fill="{MUT}">Figure 2. Real-robot evaluation of EGVLA-1 (on-device Jetson Orin NX) and EGVLA-2 (Serbot 2 + GPU server), hardware-separated latency acceleration, and offline generalization audit.</text>
</svg>'''
    _save_svg_and_png("02_edgeground_vla_v1_v2", svg_str)


def _draw_pub_diag_scene(
    x0: float, y0: float, w: float, h: float, sc: dict,
    items: list[tuple[str, str, str, float]],
) -> str:
    """CVPR/ICRA 논문 스타일 실기 실패 진단 카메라 오버레이 (직각 프레임, Okabe-Ito, 깔끔한 라벨)."""
    img_uri = _b64_img(f"docs/v5/{sc['img_src']}")
    cid = f"pubdiag_{sc['id']}"
    elems = [
        f'<clipPath id="{cid}"><rect x="{x0}" y="{y0}" width="{w}" height="{h}"/></clipPath>',
        f'<image href="{img_uri}" x="{x0}" y="{y0}" width="{w}" height="{h}" preserveAspectRatio="xMidYMid slice" clip-path="url(#{cid})"/>',
        f'<rect x="{x0}" y="{y0}" width="{w}" height="{h}" fill="none" stroke="#111827" stroke-width="0.9"/>',
    ]
    bb = sc["bbox"]
    side = math.sqrt(max(0.002, float(bb[2])))
    bw = min(0.90, side * 1.15) * w
    bh = min(0.90, side / 1.15) * h
    bx = x0 + max(3.0, min(w - bw - 3.0, float(bb[0]) * w - bw / 2.0))
    by = y0 + max(3.0, min(h - bh - 3.0, float(bb[1]) * h - bh / 2.0))
    elems.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="none" stroke="#0072B2" stroke-width="2.2"/>')
    elems.append(f'<rect x="{bx:.1f}" y="{max(y0+3, by-16):.1f}" width="132" height="15" fill="rgba(255,255,255,0.90)"/>')
    elems.append(f'<text x="{bx+4:.1f}" y="{max(y0+13, by-5):.1f}" font-family="{PUB_MONO}" font-size="8.6" font-weight="700" fill="#0072B2">BBox area={bb[2]:.2f}</text>')

    ox, oy, scale = x0 + w * 0.5, y0 + h - 16.0, 118.0
    ms = sc["models"]
    for key, label, col, yoff in items:
        m = ms[key]
        lx, ly = float(m["cmd"][0]), float(m["cmd"][1])
        if m["stop"]:
            r = 8.0 + abs(yoff) * 0.25
            elems.append(f'<circle cx="{ox:.1f}" cy="{oy:.1f}" r="{r:.1f}" fill="none" stroke="{col}" stroke-width="2.8"/>')
            lx_pos = ox + r + 8 if yoff >= 0 else ox - r - 144
            elems.append(f'<rect x="{lx_pos:.1f}" y="{oy - 9 + yoff*0.4:.1f}" width="136" height="16" fill="rgba(255,255,255,0.92)" stroke="{col}" stroke-width="0.8"/>')
            elems.append(f'<text x="{lx_pos+4:.1f}" y="{oy + 2 + yoff*0.4:.1f}" font-family="{PUB_MONO}" font-size="8.4" font-weight="700" fill="{col}">{html.escape(label)}: STOP</text>')
        else:
            tx = max(x0 + 18, min(x0 + w - 135, ox + ly * scale))
            ty = max(y0 + 22, min(y0 + h - 18, oy - lx * scale))
            elems.append(f'<line x1="{ox:.1f}" y1="{oy:.1f}" x2="{tx:.1f}" y2="{ty:.1f}" stroke="{col}" stroke-width="3.2" stroke-linecap="round"/>')
            elems.append(f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="4.0" fill="{col}"/>')
            elems.append(f'<rect x="{tx+7:.1f}" y="{ty - 9 + yoff*0.35:.1f}" width="132" height="16" fill="rgba(255,255,255,0.92)" stroke="{col}" stroke-width="0.8"/>')
            elems.append(f'<text x="{tx+11:.1f}" y="{ty + 2 + yoff*0.35:.1f}" font-family="{PUB_MONO}" font-size="8.4" font-weight="700" fill="{col}">{html.escape(label)}: GO</text>')
    return "\n".join(elems)


def build_03_research_question_diagnostic() -> None:
    """Figure 3: CVPR/ICRA Publication Style — Visual Shortcut Diagnosis & Counterfactual Audit."""
    W, H = 1100, 745
    fam = json.loads((ROOT / "docs/v5/egvla2_fm/family_arch_compare.json").read_text(encoding="utf-8"))
    sc_map = {s["id"]: s for s in fam["scenes"]}
    sc1 = sc_map["sc1_chair_center"]
    sc3 = sc_map["sc3_chair_close"]

    C1, C2, C3, WARN = "#0072B2", "#009E73", "#D55E00", "#E69F00"
    INK, MUT, RULE = "#111827", "#4B5563", "#D1D5DB"

    ov1 = _draw_pub_diag_scene(
        36, 68, 494, 214, sc1,
        [("MOD-A3", "a3 (Baseline)", C3, -10), ("MOD-A3-AG", "a3-ag23", C2, 0)],
    )
    ov2 = _draw_pub_diag_scene(
        570, 68, 494, 214, sc3,
        [("VLM-E2E-Hidden", "EG3-08 Hidden", C3, -6), ("VLM-E2E-SoftLoc", "Soft-Loc / AG23", C2, 10)],
    )

    # ── Panel (c): Counterfactual Protocol + Visual-Swap Flip Rate Bar Chart ──
    c_svg = [
        f'<text x="36" y="362" font-family="{PUB_FONT}" font-size="12.2" font-weight="700" fill="{INK}">(c) Counterfactual Visual-Swap Audit (Lower Flip Rate = Less Shortcut)</text>',
        f'<line x1="36" y1="371" x2="530" y2="371" stroke="{RULE}" stroke-width="0.8"/>',
        # 미니 반사실 개입 도식 (BBox Fixed + Vis Shuffled -> Measure Flip)
        f'<rect x="36" y="382" width="144" height="32" rx="2" fill="#F8FAFC" stroke="{C1}" stroke-width="1.0"/>',
        f'<text x="108" y="396" text-anchor="middle" font-family="{PUB_MONO}" font-size="8.2" font-weight="700" fill="{C1}">BBox: Fixed</text>',
        f'<text x="108" y="408" text-anchor="middle" font-family="{PUB_FONT}" font-size="7.8" fill="{MUT}">[cx, cy, area, has]</text>',
        f'<text x="190" y="402" text-anchor="middle" font-family="{PUB_FONT}" font-size="11" font-weight="700" fill="{INK}">+</text>',
        f'<rect x="200" y="382" width="154" height="32" rx="2" fill="#FFF7ED" stroke="{C3}" stroke-width="1.0"/>',
        f'<text x="277" y="396" text-anchor="middle" font-family="{PUB_MONO}" font-size="8.2" font-weight="700" fill="{C3}">Vis 768-d: Shuffled</text>',
        f'<text x="277" y="408" text-anchor="middle" font-family="{PUB_FONT}" font-size="7.8" fill="{MUT}">Random frame feature</text>',
        f'<line x1="354" y1="398" x2="374" y2="398" stroke="{INK}" stroke-width="1.2" marker-end="url(#pubArr)"/>',
        f'<rect x="376" y="382" width="154" height="32" rx="2" fill="#F9FAFB" stroke="#6B7280" stroke-width="1.0"/>',
        f'<text x="453" y="396" text-anchor="middle" font-family="{PUB_MONO}" font-size="8.2" font-weight="700" fill="{INK}">Stop-Flip Rate (%)</text>',
        f'<text x="453" y="408" text-anchor="middle" font-family="{PUB_FONT}" font-size="7.8" fill="{MUT}">Ideal = 0.0% (Zero bias)</text>',
    ]
    swap_bars = [
        ("MOD-A3 (a3 · Baseline)", 27.4, C3),
        ("VLM-E2E-Hidden (EG3-08)", 14.9, C3),
        ("MOD-A3-s2 (11/16 Real)", 11.0, WARN),
        ("MOD-A3-AG (a3-ag23)", 0.0, C2),
        ("VLM-E2E-SoftLoc (EG3-10/11)", 0.0, C2),
    ]
    sy = 446
    for lbl, val, col in swap_bars:
        bw = max(3.0, 230.0 * (val / 30.0))
        c_svg.append(f'<text x="36" y="{sy}" font-family="{PUB_FONT}" font-size="9.4" font-weight="{"700" if val==0 else "400"}" fill="{INK}">{html.escape(lbl)}</text>')
        c_svg.append(f'<rect x="218" y="{sy-10}" width="230" height="14" fill="#F3F4F6"/>')
        c_svg.append(f'<rect x="218" y="{sy-10}" width="{bw:.1f}" height="14" fill="{col}"/>')
        c_svg.append(f'<text x="456" y="{sy}" font-family="{PUB_MONO}" font-size="9.4" font-weight="700" fill="{col}">{val:.1f}%</text>')
        sy += 36
    c_svg.append(f'<text x="36" y="646" font-family="{PUB_FONT}" font-size="8.6" fill="{MUT}">* Motivation (EGVLA-1): 81.4% of 630 offline errors were substitutable actions (1.97% critical L/R flip),</text>')
    c_svg.append(f'<text x="44" y="660" font-family="{PUB_FONT}" font-size="8.6" fill="{MUT}">revealing that label-only imitation memorizes appearance shortcuts rather than target geometry.</text>')

    # ── Panel (d): Full Counterfactual Audit Matrix (Booktabs) ──
    d_rows = [
        ("MOD-A3 (a3)", "0.0%", "46.5%", "27.4%", "+2.7%p", "Shortcut", C3),
        ("MOD-A3-s2", "100.0%", "30.2%", "11.0%", "-0.3%p", "Partial", WARN),
        ("MOD-A3-AG", "100.0%", "93.0%", "0.0%", "+100.0%p", "Clean", C2),
        ("EG3-08 Hidden", "100.0%", "18.6%", "14.9%", "0.0%p", "Shortcut", C3),
        ("EG3-10 SoftLoc+", "100.0%", "100.0%", "0.0%", "+100.0%p", "Clean", C2),
        ("EG3-11 Joint", "100.0%", "97.7%", "0.0%", "+100.0%p", "Clean", C2),
    ]
    dy0 = 408
    drh = 34
    d_svg = [
        f'<text x="570" y="362" font-family="{PUB_FONT}" font-size="12.2" font-weight="700" fill="{INK}">(d) Counterfactual Audit Matrix (1,458 Real Frames, Unseen Chair)</text>',
        f'<line x1="570" y1="{dy0-17}" x2="1064" y2="{dy0-17}" stroke="{INK}" stroke-width="1.5"/>',
        f'<text x="570" y="{dy0}" font-family="{PUB_FONT}" font-size="9.4" font-weight="700" fill="{INK}">Model</text>',
        f'<text x="688" y="{dy0}" font-family="{PUB_FONT}" font-size="9.4" font-weight="700" fill="{INK}">Chair Go</text>',
        f'<text x="756" y="{dy0}" font-family="{PUB_FONT}" font-size="9.4" font-weight="700" fill="{INK}">Chair Stop</text>',
        f'<text x="832" y="{dy0}" font-family="{PUB_FONT}" font-size="9.4" font-weight="700" fill="{INK}">Swap Flip</text>',
        f'<text x="904" y="{dy0}" font-family="{PUB_FONT}" font-size="9.4" font-weight="700" fill="{INK}">S_area</text>',
        f'<text x="984" y="{dy0}" font-family="{PUB_FONT}" font-size="9.4" font-weight="700" fill="{INK}">Verdict</text>',
        f'<line x1="570" y1="{dy0+8}" x2="1064" y2="{dy0+8}" stroke="{INK}" stroke-width="0.9"/>',
    ]
    for i, (m, cgo, cst, flp, sa, verd, col) in enumerate(d_rows):
        ry = dy0 + 28 + i * drh
        if i % 2 == 1:
            d_svg.append(f'<rect x="570" y="{ry-17}" width="494" height="{drh}" fill="#F9FAFB"/>')
        if i == 3:
            d_svg.append(f'<line x1="570" y1="{ry-17}" x2="1064" y2="{ry-17}" stroke="#9CA3AF" stroke-width="0.7" stroke-dasharray="3 2"/>')
        d_svg.append(f'<text x="570" y="{ry}" font-family="{PUB_FONT}" font-size="9.2" font-weight="700" fill="{INK}">{html.escape(m)}</text>')
        d_svg.append(f'<text x="688" y="{ry}" font-family="{PUB_MONO}" font-size="8.8" fill="{C3 if cgo=="0.0%" else INK}">{cgo}</text>')
        d_svg.append(f'<text x="756" y="{ry}" font-family="{PUB_MONO}" font-size="8.8" font-weight="{"700" if col==C2 else "400"}" fill="{col if col!=WARN else INK}">{cst}</text>')
        d_svg.append(f'<text x="832" y="{ry}" font-family="{PUB_MONO}" font-size="8.8" font-weight="{"700" if flp=="0.0%" else "400"}" fill="{C2 if flp=="0.0%" else C3}">{flp}</text>')
        d_svg.append(f'<text x="904" y="{ry}" font-family="{PUB_MONO}" font-size="8.8" fill="{C2 if "+100" in sa else MUT}">{sa}</text>')
        d_svg.append(f'<text x="984" y="{ry}" font-family="{PUB_FONT}" font-size="9.0" font-weight="700" fill="{col}">{verd}</text>')
    dbot = dy0 + 28 + len(d_rows) * drh - 12
    d_svg.append(f'<line x1="570" y1="{dbot}" x2="1064" y2="{dbot}" stroke="{INK}" stroke-width="1.5"/>')
    d_svg.append(f'<text x="570" y="{dbot+20}" font-family="{PUB_FONT}" font-size="8.6" fill="{MUT}">* S_area: stop-rate sensitivity when BBox area increases from &lt;0.10 to &gt;=0.25 (fixed visual feat.).</text>')

    svg_str = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <defs>
    <marker id="pubArr" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
      <path d="M0,0.8 L5.2,3 L0,5.2 Z" fill="{INK}"/>
    </marker>
  </defs>
  <rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>
  <!-- Panel (a) -->
  <text x="36" y="34" font-family="{PUB_FONT}" font-size="12.2" font-weight="700" fill="{INK}">(a) Unseen Start Failure (sc1_chair_center, area = 0.06)</text>
  <line x1="36" y1="43" x2="530" y2="43" stroke="{RULE}" stroke-width="0.8"/>
  <text x="36" y="59" font-family="{PUB_FONT}" font-size="9.2" fill="{MUT}">Prompt: &quot;go to the brown wooden chair&quot; (Trained on basket only, 300 ep)</text>
  {ov1}
  <rect x="36" y="290" width="494" height="36" rx="2" fill="#F8FAFC" stroke="#E5E7EB" stroke-width="0.9"/>
  <text x="44" y="305" font-family="{PUB_MONO}" font-size="8.4" fill="{C3}">a3: [0.04,-0.02] STOP (Real 0/5)  —  768-d appearance shortcut</text>
  <text x="44" y="319" font-family="{PUB_MONO}" font-size="8.4" font-weight="700" fill="{C2}">a3-ag23: [0.68,-0.02] GO (100%)   —  50% Feature Swap + Area Align</text>

  <!-- Panel (b) -->
  <text x="570" y="34" font-family="{PUB_FONT}" font-size="12.2" font-weight="700" fill="{INK}">(b) Unseen Close-Stop Ablation (sc3_chair_close, area = 0.36)</text>
  <line x1="570" y1="43" x2="1064" y2="43" stroke="{RULE}" stroke-width="0.8"/>
  <text x="570" y="59" font-family="{PUB_FONT}" font-size="9.2" fill="{MUT}">End-to-End VLA Ablation: Latent Hidden State vs. Differentiable Spatial Bottleneck</text>
  {ov2}
  <rect x="570" y="290" width="494" height="36" rx="2" fill="#F8FAFC" stroke="#E5E7EB" stroke-width="0.9"/>
  <text x="578" y="305" font-family="{PUB_MONO}" font-size="8.4" fill="{C3}">EG3-08 (Hidden): [0.26,+0.15] GO (18.6% stop)  —  no spatial bottleneck</text>
  <text x="578" y="319" font-family="{PUB_MONO}" font-size="8.4" font-weight="700" fill="{C2}">Soft-Loc / AG23: [0.00, 0.00] STOP (93-100%)   —  geometric grounding</text>

  {"".join(c_svg)}
  {"".join(d_svg)}
  <text x="36" y="{H - 14}" font-family="{PUB_FONT}" font-size="9.1" fill="{MUT}">Figure 3. Diagnosis of visual shortcut overfitting on unseen objects via real-camera overlays (a, b) and counterfactual visual-feature swap audit across 1,458 frames (c, d).</text>
</svg>'''
    _save_svg_and_png("03_research_question_diagnostic", svg_str)


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

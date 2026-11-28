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


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson 신뢰구간(백분율)을 계산한다."""
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    h = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return max(0.0, c - h) * 100.0, min(1.0, c + h) * 100.0


def _omni_robot_icon(x: float, y: float, col: str) -> str:
    """선명한 3-DoF Kiwi-Drive 옴니휠 로봇 벡터 아이콘 (뭉개진 저해상도 썸네일 대체)."""
    return (
        f'<g transform="translate({x},{y})">'
        f'<circle cx="17" cy="21" r="13" fill="#F8FAFC" stroke="{col}" stroke-width="1.3"/>'
        # 3개 옴니휠 (120도 간격)
        f'<rect x="13" y="5" width="8" height="3.2" rx="1" fill="{col}"/>'
        f'<rect x="3" y="25" width="8" height="3.2" rx="1" fill="{col}" transform="rotate(-60 7 26.6)"/>'
        f'<rect x="23" y="25" width="8" height="3.2" rx="1" fill="{col}" transform="rotate(60 27 26.6)"/>'
        # 진행 방향 화살표
        f'<line x1="17" y1="21" x2="17" y2="11" stroke="#111827" stroke-width="1.6" stroke-linecap="round"/>'
        f'<polygon points="17,9 14.5,13 19.5,13" fill="#111827"/>'
        f'<circle cx="17" cy="21" r="2.2" fill="#111827"/>'
        f'</g>'
    )


def _output_robot_box(y: float, bh: float, col: str, l1: str, l2: str, l3: str) -> str:
    """Figure 1 (c) 우측 Output 모듈 박스 (선명한 3-DoF 옴니휠 벡터 아이콘)."""
    return (
        f'<rect x="926" y="{y}" width="138" height="{bh}" rx="2" fill="#FFFFFF" stroke="{col}" stroke-width="1.3"/>'
        f'{_omni_robot_icon(932, y + 9, col)}'
        f'<text x="971" y="{y+18}" font-family="{PUB_FONT}" font-size="9.0" font-weight="700" fill="{col}">{html.escape(l1)}</text>'
        f'<text x="971" y="{y+33}" font-family="{PUB_MONO}" font-size="7.6" fill="#111827">{html.escape(l2)}</text>'
        f'<text x="971" y="{y+49}" font-family="{PUB_FONT}" font-size="7.6" fill="#4B5563">{html.escape(l3)}</text>'
    )


def build_01_profile_roadmap() -> None:
    """Figure 1: CVPR/ICRA Publication Style — (a) Same-Frame Inference, (b) Unified Table, (c) Detailed Tensor & Gradient Flow."""
    W, H = 1100, 995
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
        "Output: FORWARD  [1.15, 0.00]  (0.0 deg, 8-way)",
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

    # (b) LaTeX Booktabs 통일 기준 비교표
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

    # (c) 3단 고밀도 아키텍처 & 텐서 흐름 & 선명한 옴니휠 아이콘
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
        _output_robot_box(r1y, bh, C1, "FORWARD", "[1.15, 0.0, 0.0]", "Jetson 10Hz ctrl"),

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
        _output_robot_box(r2y, bh, C2, "[0.64, -0.32]", "-26.4 deg + S2", "Real 397ms · 0c"),

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
        _output_robot_box(r3y, bh, C3, "[0.59, -0.32]", "-28.8 deg E2E", "GB10 297ms"),

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
  <text x="36" y="34" font-family="{PUB_FONT}" font-size="12.5" font-weight="700" fill="{INK}">(a) Same-Frame Policy Inference (Single-Frame Snapshot)  —  Prompt: &quot;go to the gray basket&quot; (cx = 0.28)</text>
  <line x1="36" y1="43" x2="1064" y2="43" stroke="{RULE}" stroke-width="0.8"/>
  {f1}
  {f2}
  {f3}
  {"".join(tbl_svg)}
  {"".join(arch_svg)}
  <text x="36" y="{H - 12}" font-family="{PUB_FONT}" font-size="9.0" fill="{MUT}">Figure 1. Evolution from Modular VLA (EGVLA-1/2) to End-to-End VLA (EGVLA-3). (a) Single-frame snapshot illustrating discrete 8-way quantization vs. continuous 2D steering; (b) unified evaluation; (c) tensor &amp; gradient flow.</text>
</svg>'''
    _save_svg_and_png("01_profile_roadmap", svg_str)


def build_02_edgeground_vla_v1_v2() -> None:
    """Figure 2: CVPR/ICRA Publication Style — Real-Robot Benchmarks, Latency Separation & Offline Generalization."""
    W, H = 1100, 710
    bsk_uri = _b64_img("docs/v5/figures/real_1125/20261125_152041_바구니_trackF_center.jpg")
    chr_uri = _b64_img("docs/v5/figures/real_1125/20261125_155938_의자_trackF_center.jpg")

    C1, C2, C3 = "#0072B2", "#009E73", "#D55E00"
    INK, MUT, RULE = "#111827", "#4B5563", "#D1D5DB"

    # ── Panel (a): EGVLA-1 On-Device Real-Robot Success + 95% Wilson CI Error Bars ──
    ay_top, ay_bot = 104, 272
    ah = ay_bot - ay_top
    bars_a = [
        ("pos1\n(Seen)", 95, 100, C1, 1.0),
        ("pos2\n(-50 cm)", 91, 100, C1, 0.78),
        ("pos3\n(+50 cm)", 90, 100, C1, 0.78),
        ("Overall\n(N=300)", 276, 300, "#111827", 0.92),
    ]
    a_svg = [
        f'<text x="36" y="34" font-family="{PUB_FONT}" font-size="12.2" font-weight="700" fill="{INK}">(a) EGVLA-1: On-Device Real-Robot Success (Jetson Orin NX, 95% Wilson CI)</text>',
        f'<line x1="36" y1="43" x2="526" y2="43" stroke="{RULE}" stroke-width="0.8"/>',
        f'<rect x="268" y="52" width="258" height="36" rx="2" fill="#F8FAFC" stroke="{C1}" stroke-width="1.0"/>',
        f'<text x="276" y="67" font-family="{PUB_FONT}" font-size="9.0" font-weight="700" fill="{C1}">Trained: 1.128M / 459M (0.25%)</text>',
        f'<text x="276" y="81" font-family="{PUB_FONT}" font-size="8.1" fill="{MUT}">OWLv2 962ms fp16 (~528ms cache avg) · 10Hz ctrl</text>',
    ]
    for pct in (0, 50, 80, 100):
        yy = ay_bot - ah * (pct / 100.0)
        dash = ' stroke-dasharray="3 2"' if pct > 0 else ""
        scol = "#111827" if pct == 0 else "#E5E7EB"
        a_svg.append(f'<line x1="72" y1="{yy:.1f}" x2="514" y2="{yy:.1f}" stroke="{scol}" stroke-width="0.9"{dash}/>')
        a_svg.append(f'<text x="64" y="{yy+3:.1f}" text-anchor="end" font-family="{PUB_MONO}" font-size="8.5" fill="{MUT}">{pct}%</text>')
    a_svg.append(f'<line x1="72" y1="{ay_top}" x2="72" y2="{ay_bot}" stroke="{INK}" stroke-width="1.0"/>')

    for i, (lbl, k, n, col, op) in enumerate(bars_a):
        val = 100.0 * k / n
        lo, hi = _wilson_ci(k, n)
        bx = 98 + i * 104
        bw = 56
        bh = ah * (val / 100.0)
        by = ay_bot - bh
        ylo = ay_bot - ah * (lo / 100.0)
        yhi = ay_bot - ah * (hi / 100.0)
        cx_b = bx + bw / 2
        a_svg.append(f'<rect x="{bx}" y="{by:.1f}" width="{bw}" height="{bh:.1f}" fill="{col}" fill-opacity="{op}"/>')
        # 95% Wilson CI error bar
        a_svg.append(f'<line x1="{cx_b}" y1="{yhi:.1f}" x2="{cx_b}" y2="{ylo:.1f}" stroke="#111827" stroke-width="1.4"/>')
        a_svg.append(f'<line x1="{cx_b-6}" y1="{yhi:.1f}" x2="{cx_b+6}" y2="{yhi:.1f}" stroke="#111827" stroke-width="1.4"/>')
        a_svg.append(f'<line x1="{cx_b-6}" y1="{ylo:.1f}" x2="{cx_b+6}" y2="{ylo:.1f}" stroke="#111827" stroke-width="1.4"/>')
        a_svg.append(f'<text x="{cx_b}" y="{yhi-5:.1f}" text-anchor="middle" font-family="{PUB_MONO}" font-size="9.4" font-weight="700" fill="{INK}">{val:.1f}%</text>')
        a_svg.append(f'<text x="{cx_b}" y="{ylo+13:.1f}" text-anchor="middle" font-family="{PUB_MONO}" font-size="7.8" fill="#FFFFFF">{k}/{n}</text>')
        l1, l2 = lbl.split("\n")
        a_svg.append(f'<text x="{cx_b}" y="{ay_bot+15}" text-anchor="middle" font-family="{PUB_FONT}" font-size="9.4" font-weight="700" fill="{INK}">{l1}</text>')
        a_svg.append(f'<text x="{cx_b}" y="{ay_bot+28}" text-anchor="middle" font-family="{PUB_FONT}" font-size="8.3" fill="{MUT}">{l2}</text>')

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
        ("Offline GPU Benchmark (GB10, 330 real frames)", None, None, None, None),
        ("beam=3 fp32 (Baseline)", 590.7, 960.0, "#9CA3AF", "590.7 ms"),
        ("beam=1 fp16 (MOD-DR-Fast)", 283.0, 960.0, C1, "283.0 ms  (2.09x)"),
        ("Remote GPU Server (Real-Robot Loop, 11/25, N = 45)", None, None, None, None),
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

    # ── Panel (d): S2 Safe Deceleration Profile + Language Gate (Fig 3 중복 완전 제거!) ──
    gx0, gy0, gw, gh = 610, 386, 444, 108
    # X축: BBox area 0.00 -> 0.30, Y축: Velocity scale 0.0 -> 1.0
    def _gx(a: float) -> float:
        return gx0 + gw * (a / 0.30)
    def _gy(v: float) -> float:
        return gy0 + gh * (1.0 - v)

    d_svg = [
        f'<text x="564" y="352" font-family="{PUB_FONT}" font-size="12.2" font-weight="700" fill="{INK}">(d) S2 Safe Deceleration Profile &amp; Language Safety Gate (MOD-V4R)</text>',
        f'<line x1="564" y1="361" x2="1064" y2="361" stroke="{RULE}" stroke-width="0.8"/>',
        # 감속 그래프 배경 & 축
        f'<rect x="{gx0}" y="{gy0}" width="{gw}" height="{gh}" fill="#FAFAFA" stroke="#E5E7EB" stroke-width="0.8"/>',
        # 감속 구간 음영 (0.14 -> 0.25)
        f'<rect x="{_gx(0.14):.1f}" y="{gy0}" width="{_gx(0.25)-_gx(0.14):.1f}" height="{gh}" fill="#ECFDF5" opacity="0.8"/>',
        f'<line x1="{gx0}" y1="{gy0+gh}" x2="{gx0+gw}" y2="{gy0+gh}" stroke="{INK}" stroke-width="1.0"/>',
        f'<line x1="{gx0}" y1="{gy0}" x2="{gx0}" y2="{gy0+gh}" stroke="{INK}" stroke-width="1.0"/>',
        # Baseline (no decel: 1.0 until hard stop at 0.25)
        f'<polyline points="{_gx(0.0):.1f},{_gy(1.0):.1f} {_gx(0.25):.1f},{_gy(1.0):.1f} {_gx(0.25):.1f},{_gy(0.0):.1f}" fill="none" stroke="{C3}" stroke-width="1.6" stroke-dasharray="4 3"/>',
        # S2 Linear Decel (1.0 -> linear ramp 0.14..0.25 -> 0.0) + Model Preemptive Stop (0.238)
        f'<polyline points="{_gx(0.0):.1f},{_gy(1.0):.1f} {_gx(0.14):.1f},{_gy(1.0):.1f} {_gx(0.238):.1f},{_gy(0.12):.1f} {_gx(0.238):.1f},{_gy(0.0):.1f}" fill="none" stroke="{C2}" stroke-width="2.4"/>',
        f'<circle cx="{_gx(0.238):.1f}" cy="{_gy(0.12):.1f}" r="3.8" fill="{C2}"/>',
        f'<text x="{_gx(0.238)-10:.1f}" y="{_gy(0.12)+3:.1f}" text-anchor="end" font-family="{PUB_MONO}" font-size="7.8" font-weight="700" fill="{C2}">Model Stop (a=0.238)</text>',
        f'<text x="{_gx(0.02):.1f}" y="{gy0+14}" font-family="{PUB_FONT}" font-size="8.0" fill="{C3}">--- No Decel (a3-s2: 6 contacts)</text>',
        f'<text x="{_gx(0.02):.1f}" y="{gy0+26}" font-family="{PUB_FONT}" font-size="8.0" font-weight="700" fill="{C2}">— S2 Linear Decel (0.14-&gt;0.25, 0 contacts)</text>',
        # X축 눈금
        f'<text x="{_gx(0.0):.1f}" y="{gy0+gh+13}" text-anchor="middle" font-family="{PUB_MONO}" font-size="7.8" fill="{MUT}">0.00</text>',
        f'<text x="{_gx(0.14):.1f}" y="{gy0+gh+13}" text-anchor="middle" font-family="{PUB_MONO}" font-size="7.8" fill="{C2}">0.14 (Ramp)</text>',
        f'<text x="{_gx(0.25):.1f}" y="{gy0+gh+13}" text-anchor="middle" font-family="{PUB_MONO}" font-size="7.8" fill="{INK}">0.25 (Near)</text>',
        f'<text x="{_gx(0.30):.1f}" y="{gy0+gh+13}" text-anchor="middle" font-family="{PUB_MONO}" font-size="7.8" fill="{MUT}">0.30</text>',
        f'<text x="{gx0-6}" y="{gy0+6}" text-anchor="end" font-family="{PUB_MONO}" font-size="7.8" fill="{MUT}">1.0x</text>',
        f'<text x="{gx0-6}" y="{gy0+gh}" text-anchor="end" font-family="{PUB_MONO}" font-size="7.8" fill="{MUT}">0.0x</text>',
        # 하단 언어 게이트 & 조향 정렬 미니 Booktabs (3행)
        f'<line x1="564" y1="528" x2="1064" y2="528" stroke="{INK}" stroke-width="1.3"/>',
        f'<text x="564" y="544" font-family="{PUB_FONT}" font-size="9.0" font-weight="700" fill="{INK}">Safety / Precision Check (Offline)</text>',
        f'<text x="808" y="544" font-family="{PUB_FONT}" font-size="9.0" font-weight="700" fill="{MUT}">Baseline</text>',
        f'<text x="916" y="544" font-family="{PUB_FONT}" font-size="9.0" font-weight="700" fill="{C1}">Proposed</text>',
        f'<line x1="564" y1="550" x2="1064" y2="550" stroke="{INK}" stroke-width="0.8"/>',
        f'<text x="564" y="568" font-family="{PUB_FONT}" font-size="9.0" font-weight="700" fill="{INK}">Steering Alignment vs. Baseline</text>',
        f'<text x="808" y="568" font-family="{PUB_MONO}" font-size="8.6" fill="{MUT}">25.13 deg (B1)</text>',
        f'<text x="916" y="568" font-family="{PUB_MONO}" font-size="8.8" font-weight="700" fill="{C1}">0.54 deg (98.7%)</text>',
        f'<rect x="564" y="576" width="500" height="24" fill="#F9FAFB"/>',
        f'<text x="564" y="592" font-family="{PUB_FONT}" font-size="9.0" font-weight="700" fill="{INK}">Negative-Prompt Safety Stop</text>',
        f'<text x="808" y="592" font-family="{PUB_MONO}" font-size="8.6" fill="{MUT}">0.0% (no text)</text>',
        f'<text x="916" y="592" font-family="{PUB_MONO}" font-size="8.8" font-weight="700" fill="{C1}">99.5% (MOD-V4R)</text>',
        f'<text x="564" y="616" font-family="{PUB_FONT}" font-size="9.0" font-weight="700" fill="{INK}">BBox Horizontal-Flip Defense</text>',
        f'<text x="808" y="616" font-family="{PUB_MONO}" font-size="8.6" fill="{MUT}">86.0% (V4)</text>',
        f'<text x="916" y="616" font-family="{PUB_MONO}" font-size="8.8" font-weight="700" fill="{C1}">94.3% (MOD-V4R)</text>',
        f'<line x1="564" y1="626" x2="1064" y2="626" stroke="{INK}" stroke-width="1.3"/>',
    ]

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
        [("VLM-E2E-Hidden", "EG3-08 Hidden", C3, -6), ("VLM-E2E-SoftLoc", "EG3-11 Soft-Loc", C2, 10)],
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
    # X축 수직 눈금선 (0%, 10%, 20%, 30%)
    for tick in (0, 10, 20, 30):
        tx_t = 218.0 + 230.0 * (tick / 30.0)
        c_svg.append(f'<line x1="{tx_t:.1f}" y1="432" x2="{tx_t:.1f}" y2="612" stroke="{"#111827" if tick==0 else "#E5E7EB"}" stroke-width="{"1.0" if tick==0 else "0.8"}"{" stroke-dasharray=\"2 2\"" if tick>0 else ""}/>')
        c_svg.append(f'<text x="{tx_t:.1f}" y="625" text-anchor="middle" font-family="{PUB_MONO}" font-size="8.0" fill="{MUT}">{tick}%</text>')
    for lbl, val, col in swap_bars:
        bw = max(3.0, 230.0 * (val / 30.0))
        c_svg.append(f'<text x="36" y="{sy}" font-family="{PUB_FONT}" font-size="9.4" font-weight="{"700" if val==0 else "400"}" fill="{INK}">{html.escape(lbl)}</text>')
        c_svg.append(f'<rect x="218" y="{sy-10}" width="230" height="14" fill="#F3F4F6" fill-opacity="0.6"/>')
        c_svg.append(f'<rect x="218" y="{sy-10}" width="{bw:.1f}" height="14" fill="{col}"/>')
        c_svg.append(f'<text x="456" y="{sy}" font-family="{PUB_MONO}" font-size="9.4" font-weight="700" fill="{col}">{val:.1f}%</text>')
        sy += 35
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
  <text x="578" y="319" font-family="{PUB_MONO}" font-size="8.4" font-weight="700" fill="{C2}">EG3-11 (Soft-Loc): [0.00, 0.00] STOP (97.7% / SoftLoc+ 100%) — spatial bottleneck</text>

  {"".join(c_svg)}
  {"".join(d_svg)}
  <text x="36" y="{H - 14}" font-family="{PUB_FONT}" font-size="9.1" fill="{MUT}">Figure 3. Diagnosis of visual shortcut overfitting on unseen objects via real-camera overlays (a, b) and counterfactual visual-feature swap audit across 1,458 frames (c, d).</text>
</svg>'''
    _save_svg_and_png("03_research_question_diagnostic", svg_str)


def build_04_eg1_vs_eg2_protocol_y2() -> None:
    """Figure 4: CVPR/ICRA Publication Style — Side-View Pinhole Geometry, Ground-Line (y2) Verification & 3x3 Benchmark."""
    W, H = 1100, 745
    basket_uri = _b64_img("docs/v5/figures/real_1125/20261125_152041_바구니_trackF_center.jpg")
    chair_uri = _b64_img("docs/v5/figures/real_1125/20261125_155938_의자_trackF_center.jpg")

    C1, C2, C3 = "#0072B2", "#009E73", "#D55E00"
    INK, MUT, RULE = "#111827", "#4B5563", "#D1D5DB"

    # ── Panel (a): Real Arrival Photos + Side-View Pinhole Geometry Diagram ──
    a_svg = [
        f'<text x="36" y="34" font-family="{PUB_FONT}" font-size="12.2" font-weight="700" fill="{INK}">(a) Object-Size Bias under Area Stopping vs. Ground-Line (y2 = cy + h/2)</text>',
        f'<line x1="36" y1="43" x2="536" y2="43" stroke="{RULE}" stroke-width="0.8"/>',
        # Basket photo
        f'<text x="36" y="61" font-family="{PUB_FONT}" font-size="9.5" font-weight="700" fill="{C2}">Small Object: Basket (area=0.28, y2=1.00)</text>',
        f'<clipPath id="pubBsk"><rect x="36" y="68" width="242" height="128"/></clipPath>',
        f'<image href="{basket_uri}" x="36" y="68" width="242" height="128" preserveAspectRatio="xMidYMid slice" clip-path="url(#pubBsk)"/>',
        f'<rect x="36" y="68" width="242" height="128" fill="none" stroke="{INK}" stroke-width="0.9"/>',
        f'<line x1="36" y1="190" x2="278" y2="190" stroke="{C2}" stroke-width="2.6" stroke-dasharray="6 3"/>',
        f'<rect x="40" y="170" width="122" height="15" fill="rgba(255,255,255,0.90)"/>',
        f'<text x="44" y="181" font-family="{PUB_MONO}" font-size="8.1" font-weight="700" fill="{C2}">y2 = 1.00 (Close)</text>',
        # Chair photo
        f'<text x="294" y="61" font-family="{PUB_FONT}" font-size="9.5" font-weight="700" fill="{C3}">Tall Object: Chair (area=0.29, y2=0.82)</text>',
        f'<clipPath id="pubChr"><rect x="294" y="68" width="242" height="128"/></clipPath>',
        f'<image href="{chair_uri}" x="294" y="68" width="242" height="128" preserveAspectRatio="xMidYMid slice" clip-path="url(#pubChr)"/>',
        f'<rect x="294" y="68" width="242" height="128" fill="none" stroke="{INK}" stroke-width="0.9"/>',
        f'<line x1="294" y1="172" x2="536" y2="172" stroke="{C3}" stroke-width="2.6" stroke-dasharray="6 3"/>',
        f'<rect x="298" y="152" width="118" height="15" fill="rgba(255,255,255,0.90)"/>',
        f'<text x="302" y="163" font-family="{PUB_MONO}" font-size="8.1" font-weight="700" fill="{C3}">y2 = 0.82 (Far)</text>',
        # 측면 핀홀 기하 단면도 (Side-View Pinhole Geometry)
        f'<rect x="36" y="206" width="500" height="118" rx="2" fill="#FAFAFA" stroke="#D1D5DB" stroke-width="0.9"/>',
        f'<text x="46" y="222" font-family="{PUB_FONT}" font-size="9.0" font-weight="700" fill="{INK}">Side-View Pinhole Geometry: Why Ground-Line (y2) Tracks Physical Distance</text>',
        # 바닥선
        f'<line x1="56" y1="306" x2="516" y2="306" stroke="{INK}" stroke-width="1.2"/>',
        # 로봇 카메라
        f'<rect x="64" y="266" width="32" height="40" rx="2" fill="#E5E7EB" stroke="{INK}" stroke-width="1.0"/>',
        f'<circle cx="96" cy="276" r="4" fill="{C1}"/>',
        f'<text x="80" y="260" text-anchor="middle" font-family="{PUB_FONT}" font-size="7.8" font-weight="700" fill="{INK}">Robot Cam</text>',
        # 이미지 평면 (점선)
        f'<line x1="146" y1="234" x2="146" y2="304" stroke="{C1}" stroke-width="1.2" stroke-dasharray="3 2"/>',
        f'<text x="146" y="230" text-anchor="middle" font-family="{PUB_MONO}" font-size="7.4" fill="{C1}">Image Plane</text>',
        # 가까운 낮은 바구니
        f'<rect x="242" y="274" width="38" height="32" fill="#ECFDF5" stroke="{C2}" stroke-width="1.4"/>',
        f'<text x="261" y="269" text-anchor="middle" font-family="{PUB_FONT}" font-size="7.8" font-weight="700" fill="{C2}">Basket</text>',
        f'<line x1="96" y1="276" x2="242" y2="306" stroke="{C2}" stroke-width="1.3" stroke-dasharray="4 2"/>',
        f'<circle cx="146" cy="286" r="2.8" fill="{C2}"/>',
        f'<rect x="150" y="291" width="68" height="11" fill="#FAFAFA" opacity="0.92"/>',
        f'<text x="152" y="299.5" font-family="{PUB_MONO}" font-size="7.3" font-weight="700" fill="{C2}">y2=1.00 (near)</text>',
        # 먼 키 큰 의자 (같은 시야각 = 같은 BBox area!)
        f'<rect x="424" y="236" width="40" height="70" fill="#FFF7ED" stroke="{C3}" stroke-width="1.4"/>',
        f'<text x="444" y="275" text-anchor="middle" font-family="{PUB_FONT}" font-size="7.8" font-weight="700" fill="{C3}">Chair</text>',
        f'<line x1="96" y1="276" x2="424" y2="306" stroke="{C3}" stroke-width="1.3" stroke-dasharray="4 2"/>',
        f'<circle cx="146" cy="280" r="2.8" fill="{C3}"/>',
        f'<text x="152" y="274" font-family="{PUB_MONO}" font-size="7.3" font-weight="700" fill="{C3}">y2=0.82 (far)</text>',
        f'<text x="286" y="319" text-anchor="middle" font-family="{PUB_MONO}" font-size="7.5" fill="{MUT}">Equal BBox area (~0.29), but ground contact y2 maps monotonically to floor distance d</text>',
    ]

    # ── Panel (b): Offline Verification Table (내부 파일명 제거 + 트레이드오프 명시) ──
    b_rows = [
        ("Basket Arrival (y2=1.00) Stop", "100.0%", "100.0%", C2),
        ("Chair Far Arrival (y2=0.82) Stop", "100.0% (Early)", "0.0% (Closer)", C1),
        ("Mid-Range (y2 0.78-0.86) Early Stop", "26.3%", "3.0%", C2),
        ("Near-Floor (y2 >= 0.95) Stop Rate", "24.8%", "85.7%", C2),
        ("Visual-Swap Stop Flip Rate", "0.0%", "0.2%", C2),
    ]
    by0 = 86
    brh = 33
    b_svg = [
        f'<text x="568" y="34" font-family="{PUB_FONT}" font-size="12.2" font-weight="700" fill="{INK}">(b) Ground-Line Policy Verification (Offline Replay, 11/25 Frames)</text>',
        f'<line x1="568" y1="{by0-17}" x2="1064" y2="{by0-17}" stroke="{INK}" stroke-width="1.5"/>',
        f'<text x="568" y="{by0}" font-family="{PUB_FONT}" font-size="9.6" font-weight="700" fill="{INK}">Evaluation Metric</text>',
        f'<text x="802" y="{by0}" font-family="{PUB_FONT}" font-size="9.6" font-weight="700" fill="{MUT}">AG23 (Area)</text>',
        f'<text x="926" y="{by0}" font-family="{PUB_FONT}" font-size="9.6" font-weight="700" fill="{C1}">A3C_Y2 (y2)</text>',
        f'<line x1="568" y1="{by0+8}" x2="1064" y2="{by0+8}" stroke="{INK}" stroke-width="0.9"/>',
    ]
    for i, (m, v_ag, v_y2, col) in enumerate(b_rows):
        ry = by0 + 28 + i * brh
        if i % 2 == 1:
            b_svg.append(f'<rect x="568" y="{ry-18}" width="496" height="{brh}" fill="#F9FAFB"/>')
        b_svg.append(f'<text x="568" y="{ry}" font-family="{PUB_FONT}" font-size="9.3" font-weight="700" fill="{INK}">{html.escape(m)}</text>')
        b_svg.append(f'<text x="802" y="{ry}" font-family="{PUB_MONO}" font-size="8.8" fill="{MUT}">{html.escape(v_ag)}</text>')
        b_svg.append(f'<text x="926" y="{ry}" font-family="{PUB_MONO}" font-size="9.0" font-weight="700" fill="{col}">{html.escape(v_y2)}</text>')
    bbot = by0 + 28 + len(b_rows) * brh - 12
    b_svg.append(f'<line x1="568" y1="{bbot}" x2="1064" y2="{bbot}" stroke="{INK}" stroke-width="1.5"/>')
    b_svg.append(f'<rect x="568" y="{bbot+12}" width="496" height="44" rx="2" fill="#F8FAFC" stroke="#E5E7EB" stroke-width="0.9"/>')
    b_svg.append(f'<text x="578" y="{bbot+28}" font-family="{PUB_FONT}" font-size="8.4" font-weight="700" fill="{INK}">Honest Trade-Off &amp; Status:</text>')
    b_svg.append(f'<text x="578" y="{bbot+42}" font-family="{PUB_FONT}" font-size="8.2" fill="{MUT}">Offline verified (real-robot test pending). Basket center-go: 84.9% (A3C_Y2) vs. 97.1% (AG23);</text>')
    b_svg.append(f'<text x="578" y="{bbot+52}" font-family="{PUB_FONT}" font-size="8.2" fill="{MUT}">chair center-go remains 100.0%. Candidate AG28 (area=0.28) is evaluated alongside A3C_Y2.</text>')

    # ── Panel (c): Common 3x3 Real-Robot Protocol (EGVLA-1 vs EGVLA-2) ──
    cy_hdr = 368
    c_svg = [
        f'<text x="36" y="{cy_hdr}" font-family="{PUB_FONT}" font-size="12.2" font-weight="700" fill="{INK}">(c) Fair Real-Robot Comparison Protocol — Common 3x3 Route Matrix (Ongoing Benchmark)</text>',
        f'<line x1="36" y1="{cy_hdr+9}" x2="1064" y2="{cy_hdr+9}" stroke="{RULE}" stroke-width="0.8"/>',
        f'<rect x="36" y="{cy_hdr+22}" width="494" height="270" rx="2" fill="#F9FAFB" stroke="#D1D5DB" stroke-width="0.9"/>',
        f'<text x="52" y="{cy_hdr+44}" font-family="{PUB_FONT}" font-size="9.8" font-weight="700" fill="{INK}">Common 9 Routes (3 Start Positions x 3 Random Goal Positions)</text>',
    ]
    gxs = [110, 283, 456]
    gy, sy_pos = cy_hdr + 72, cy_hdr + 226
    for gx, gl in zip(gxs, ("G6 (Goal A)", "G7 (Goal B)", "G8 (Goal C)")):
        c_svg.append(f'<rect x="{gx-56}" y="{gy}" width="112" height="32" rx="2" fill="#FFFFFF" stroke="{C2}" stroke-width="1.4"/>')
        c_svg.append(f'<text x="{gx}" y="{gy+20}" text-anchor="middle" font-family="{PUB_FONT}" font-size="9.8" font-weight="700" fill="{C2}">{gl}</text>')
    for sx in gxs:
        for gx in gxs:
            c_svg.append(f'<line x1="{sx}" y1="{sy_pos}" x2="{gx}" y2="{gy+32}" stroke="#9CA3AF" stroke-width="1.0" stroke-dasharray="3 3"/>')
    c_svg.append(f'<rect x="193" y="{cy_hdr+150}" width="180" height="24" rx="2" fill="#FFFFFF" stroke="#D1D5DB"/>')
    c_svg.append(f'<text x="283" y="{cy_hdr+166}" text-anchor="middle" font-family="{PUB_MONO}" font-size="9.0" font-weight="700" fill="{INK}">3 x 3 = 9 Common Routes</text>')
    for sx, sl in zip(gxs, ("pos1 (Start 1)", "pos2 (Start 2)", "pos3 (Start 3)")):
        c_svg.append(f'<rect x="{sx-56}" y="{sy_pos}" width="112" height="32" rx="2" fill="#FFFFFF" stroke="{C1}" stroke-width="1.4"/>')
        c_svg.append(f'<text x="{sx}" y="{sy_pos+20}" text-anchor="middle" font-family="{PUB_FONT}" font-size="9.8" font-weight="700" fill="{C1}">{sl}</text>')

    p_rows = [
        ("Start / Goal Grid", "pos1-3 -> G6-8 (9 routes)", "pos1-3 -> G6-8 (9 routes)"),
        ("Target Objects", "Basket only (1 seen)", "Basket, Chair, Desk (2 unseen)"),
        ("Total Conditions", "9 conditions (3x3x1)", "27 conditions (3x3x3)"),
        ("Prompt Format", "\"gray basket\" (fixed)", "\"go to the <object>\""),
        ("Checkpoint", "exp73_owl_stage1v3 (3.4M)", "A3C_AG28 / A3C_Y2 (MOD-DR-Safe)"),
        ("Logged Metrics", "Success, Contact, Dist (cm)", "Same + Auto .h5 / CSV Agg."),
    ]
    py0 = cy_hdr + 44
    prh = 37
    c_svg += [
        f'<line x1="564" y1="{py0-17}" x2="1064" y2="{py0-17}" stroke="{INK}" stroke-width="1.5"/>',
        f'<text x="564" y="{py0}" font-family="{PUB_FONT}" font-size="9.6" font-weight="700" fill="{INK}">Protocol Item</text>',
        f'<text x="692" y="{py0}" font-family="{PUB_FONT}" font-size="9.6" font-weight="700" fill="{C1}">EGVLA-1 (Baseline)</text>',
        f'<text x="874" y="{py0}" font-family="{PUB_FONT}" font-size="9.6" font-weight="700" fill="{C2}">EGVLA-2 (Proposed)</text>',
        f'<line x1="564" y1="{py0+8}" x2="1064" y2="{py0+8}" stroke="{INK}" stroke-width="0.9"/>',
    ]
    for i, (item, v1, v2) in enumerate(p_rows):
        ry = py0 + 29 + i * prh
        if i % 2 == 1:
            c_svg.append(f'<rect x="564" y="{ry-18}" width="500" height="{prh}" fill="#F9FAFB"/>')
        c_svg.append(f'<text x="564" y="{ry}" font-family="{PUB_FONT}" font-size="9.2" font-weight="700" fill="{INK}">{html.escape(item)}</text>')
        c_svg.append(f'<text x="692" y="{ry}" font-family="{PUB_FONT}" font-size="8.8" fill="{MUT}">{html.escape(v1)}</text>')
        c_svg.append(f'<text x="874" y="{ry}" font-family="{PUB_FONT}" font-size="8.8" font-weight="700" fill="{INK}">{html.escape(v2)}</text>')
    pbot = py0 + 29 + len(p_rows) * prh - 14
    c_svg.append(f'<line x1="564" y1="{pbot}" x2="1064" y2="{pbot}" stroke="{INK}" stroke-width="1.5"/>')

    svg_str = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>
  {"".join(a_svg)}
  {"".join(b_svg)}
  {"".join(c_svg)}
  <text x="36" y="{H - 14}" font-family="{PUB_FONT}" font-size="9.1" fill="{MUT}">Figure 4. Side-view pinhole geometry &amp; ground-line stopping (y2) offline verification (a, b), paired with a unified 3x3 real-robot benchmark protocol (c).</text>
</svg>'''
    _save_svg_and_png("04_eg1_vs_eg2_protocol_y2", svg_str)


def build_05_publications_awards_stack() -> None:
    """Figure 5: CVPR/ICRA Academic Summary — Research Timeline, Publications, Systems & Technical Stack."""
    W, H = 1100, 610
    C1, C2, C3 = "#0072B2", "#009E73", "#D55E00"
    INK, MUT, RULE = "#111827", "#4B5563", "#D1D5DB"

    # (a) Research & Engineering Progression Timeline (2024 - 2026)
    milestones = [
        ("2024.05", "Liberty", "LangGraph Multi-Agent", "Excellence Award", C3),
        ("2024.10", "Ladi & JCCT", "RAG Retrieval & Paper", "IPACT Outstanding", C2),
        ("2025.03", "Serbot 2", "Jetson + ROS2 Stack", "Platform Integration", C1),
        ("2026.08", "EGVLA-1", "On-Device 95%/92%", "JKSCI 1st Author", C1),
        ("2026.11", "EGVLA-2 / 3", "45/45 0c & Soft-Loc", "Comparison Paper", C2),
    ]
    tl_svg = [
        f'<text x="36" y="34" font-family="{PUB_FONT}" font-size="12.2" font-weight="700" fill="{INK}">(a) Research &amp; Engineering Progression (2024 – 2026)</text>',
        f'<line x1="36" y1="43" x2="1064" y2="43" stroke="{RULE}" stroke-width="0.8"/>',
        f'<line x1="88" y1="74" x2="1012" y2="74" stroke="#9CA3AF" stroke-width="1.6"/>',
    ]
    for i, (dt, nm, sub, tag, col) in enumerate(milestones):
        mx = 128 + i * 211
        tl_svg.append(f'<circle cx="{mx}" cy="74" r="5.5" fill="{col}" stroke="#FFFFFF" stroke-width="1.5"/>')
        tl_svg.append(f'<text x="{mx}" y="61" text-anchor="middle" font-family="{PUB_MONO}" font-size="8.4" font-weight="700" fill="{col}">{dt}</text>')
        tl_svg.append(f'<rect x="{mx-88}" y="86" width="176" height="44" rx="2" fill="#F9FAFB" stroke="#E5E7EB" stroke-width="0.9"/>')
        tl_svg.append(f'<text x="{mx}" y="102" text-anchor="middle" font-family="{PUB_FONT}" font-size="9.4" font-weight="700" fill="{INK}">{html.escape(nm)}</text>')
        tl_svg.append(f'<text x="{mx}" y="114" text-anchor="middle" font-family="{PUB_FONT}" font-size="8.0" fill="{MUT}">{html.escape(sub)}</text>')
        tl_svg.append(f'<text x="{mx}" y="125" text-anchor="middle" font-family="{PUB_MONO}" font-size="7.5" font-weight="700" fill="{col}">{html.escape(tag)}</text>')

    # (b) Publications Booktabs
    oy = 126
    p_svg = [
        f'<text x="36" y="{34+oy}" font-family="{PUB_FONT}" font-size="12.2" font-weight="700" fill="{INK}">(b) Publications &amp; Manuscripts</text>',
        f'<line x1="36" y1="{45+oy}" x2="1064" y2="{45+oy}" stroke="{INK}" stroke-width="1.5"/>',
        f'<text x="36" y="{63+oy}" font-family="{PUB_FONT}" font-size="9.6" font-weight="700" fill="{INK}">Venue</text>',
        f'<text x="150" y="{63+oy}" font-family="{PUB_FONT}" font-size="9.6" font-weight="700" fill="{INK}">Role</text>',
        f'<text x="244" y="{63+oy}" font-family="{PUB_FONT}" font-size="9.6" font-weight="700" fill="{INK}">Title &amp; Key Contribution</text>',
        f'<text x="930" y="{63+oy}" font-family="{PUB_FONT}" font-size="9.6" font-weight="700" fill="{INK}">Status</text>',
        f'<line x1="36" y1="{71+oy}" x2="1064" y2="{71+oy}" stroke="{INK}" stroke-width="0.9"/>',
        f'<text x="36" y="{94+oy}" font-family="{PUB_MONO}" font-size="9.2" font-weight="700" fill="{C1}">JKSCI (KCI)</text>',
        f'<text x="150" y="{94+oy}" font-family="{PUB_FONT}" font-size="9.2" font-weight="700" fill="{INK}">1st Author</text>',
        f'<text x="244" y="{94+oy}" font-family="{PUB_FONT}" font-size="9.4" font-weight="700" fill="{INK}">EdgeGround-VLA: On-Device Goal-Directed Navigation via Dual Frozen Grounding Encoders</text>',
        f'<text x="244" y="{109+oy}" font-family="{PUB_FONT}" font-size="8.6" fill="{MUT}">1.128M/459M trained · Jetson Orin NX + Serbot 2 · 95.0% seen / 92.0% overall (300 trials)</text>',
        f'<text x="930" y="{94+oy}" font-family="{PUB_MONO}" font-size="8.8" font-weight="700" fill="{C1}">Under Review (2026)</text>',
        f'<rect x="36" y="{118+oy}" width="1028" height="40" fill="#F9FAFB"/>',
        f'<text x="36" y="{136+oy}" font-family="{PUB_MONO}" font-size="9.2" font-weight="700" fill="{C2}">JCCT 10(6)</text>',
        f'<text x="150" y="{136+oy}" font-family="{PUB_FONT}" font-size="9.2" font-weight="700" fill="{INK}">2nd Author</text>',
        f'<text x="244" y="{136+oy}" font-family="{PUB_FONT}" font-size="9.4" font-weight="700" fill="{INK}">Designing Conversational AI Services Using RAG-Based Loneliness Analysis for Koreans</text>',
        f'<text x="244" y="{151+oy}" font-family="{PUB_FONT}" font-size="8.6" fill="{MUT}">JCCT 10(6):625-633 · IPACT 2024 Outstanding Paper Award</text>',
        f'<text x="930" y="{136+oy}" font-family="{PUB_MONO}" font-size="8.8" font-weight="700" fill="{C2}">Published (2024)</text>',
        f'<line x1="36" y1="{162+oy}" x2="1064" y2="{162+oy}" stroke="{INK}" stroke-width="1.5"/>',
    ]

    # (c) Systems & Awards + (d) Technical Stack
    b_rows = [
        ("Serbot 2 Integration", "Jetson Orin Linux + ROS2 driver & locomotion stack", "2025.01-03"),
        ("Ladi (RAG System)", "KiwiBM25 + FAISS + CrossEncoder routine recommender", "Best Project"),
        ("Liberty (Multi-Agent)", "LangGraph legal QA multi-agent (PM & Lead Architect)", "Excellence"),
        ("Honors & Competitions", "Capstone Silver (2026) · Kangnengthon Grand Prize (2025)", "SW 1st · AI Exc."),
    ]
    by0 = 222 + oy
    brh = 39
    b_svg = [
        f'<text x="36" y="{192+oy}" font-family="{PUB_FONT}" font-size="12.2" font-weight="700" fill="{INK}">(c) Robotics / Agent Systems &amp; Honors</text>',
        f'<line x1="36" y1="{by0-17}" x2="574" y2="{by0-17}" stroke="{INK}" stroke-width="1.5"/>',
        f'<text x="36" y="{by0}" font-family="{PUB_FONT}" font-size="9.4" font-weight="700" fill="{INK}">Project / Honor</text>',
        f'<text x="178" y="{by0}" font-family="{PUB_FONT}" font-size="9.4" font-weight="700" fill="{INK}">Scope</text>',
        f'<text x="488" y="{by0}" font-family="{PUB_FONT}" font-size="9.4" font-weight="700" fill="{INK}">Outcome</text>',
        f'<line x1="36" y1="{by0+8}" x2="574" y2="{by0+8}" stroke="{INK}" stroke-width="0.9"/>',
    ]
    for i, (nm, sc, out) in enumerate(b_rows):
        ry = by0 + 29 + i * brh
        if i % 2 == 1:
            b_svg.append(f'<rect x="36" y="{ry-18}" width="538" height="{brh}" fill="#F9FAFB"/>')
        b_svg.append(f'<text x="36" y="{ry}" font-family="{PUB_FONT}" font-size="9.2" font-weight="700" fill="{INK}">{html.escape(nm)}</text>')
        b_svg.append(f'<text x="178" y="{ry}" font-family="{PUB_FONT}" font-size="8.6" fill="{MUT}">{html.escape(sc)}</text>')
        b_svg.append(f'<text x="488" y="{ry}" font-family="{PUB_MONO}" font-size="8.4" font-weight="700" fill="{C1}">{html.escape(out)}</text>')
    b_svg.append(f'<line x1="36" y1="{by0 + 29 + len(b_rows)*brh - 14}" x2="574" y2="{by0 + 29 + len(b_rows)*brh - 14}" stroke="{INK}" stroke-width="1.5"/>')

    c_svg = [
        f'<text x="604" y="{192+oy}" font-family="{PUB_FONT}" font-size="12.2" font-weight="700" fill="{INK}">(d) Core Technical Stack &amp; Links</text>',
        f'<line x1="604" y1="{205+oy}" x2="1064" y2="{205+oy}" stroke="{INK}" stroke-width="1.5"/>',

        f'<rect x="604" y="{215+oy}" width="460" height="52" rx="2" fill="#F8FAFC" stroke="{C1}" stroke-width="1.0"/>',
        f'<text x="616" y="{233+oy}" font-family="{PUB_FONT}" font-size="9.6" font-weight="700" fill="{C1}">AI &amp; Vision-Language-Action (VLA)</text>',
        f'<text x="616" y="{249+oy}" font-family="{PUB_MONO}" font-size="8.4" fill="{INK}">PyTorch · HuggingFace · LoRA/PEFT · Flow Matching</text>',
        f'<text x="616" y="{262+oy}" font-family="{PUB_MONO}" font-size="8.2" fill="{MUT}">Florence-2 · OWLv2 · Kosmos-2 · SigLIP · PaliGemma2</text>',

        f'<rect x="604" y="{274+oy}" width="460" height="44" rx="2" fill="#F8FAFC" stroke="{C2}" stroke-width="1.0"/>',
        f'<text x="616" y="{291+oy}" font-family="{PUB_FONT}" font-size="9.6" font-weight="700" fill="{C2}">Robotics &amp; On-Device Deployment</text>',
        f'<text x="616" y="{308+oy}" font-family="{PUB_MONO}" font-size="8.4" fill="{INK}">ROS2 · Jetson Orin NX · Async Control (10Hz) · Ubuntu Linux</text>',

        f'<rect x="604" y="{325+oy}" width="460" height="44" rx="2" fill="#F8FAFC" stroke="{C3}" stroke-width="1.0"/>',
        f'<text x="616" y="{342+oy}" font-family="{PUB_FONT}" font-size="9.6" font-weight="700" fill="{C3}">Agent Orchestration &amp; Retrieval</text>',
        f'<text x="616" y="{359+oy}" font-family="{PUB_MONO}" font-size="8.4" fill="{INK}">LangGraph · FAISS · KiwiBM25 · CrossEncoder · FastAPI · Git</text>',

        f'<rect x="604" y="{378+oy}" width="460" height="30" rx="2" fill="#111827"/>',
        f'<text x="834" y="{397+oy}" text-anchor="middle" font-family="{PUB_MONO}" font-size="8.8" font-weight="700" fill="#FFFFFF">minuum.github.io/EdgeGroundVLA  |  github.com/minuum  |  minwool0357@gmail.com</text>',
    ]

    svg_str = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">
  <rect x="0" y="0" width="{W}" height="{H}" fill="#FFFFFF"/>
  {"".join(tl_svg)}
  {"".join(p_svg)}
  {"".join(b_svg)}
  {"".join(c_svg)}
  <text x="36" y="{H - 14}" font-family="{PUB_FONT}" font-size="9.1" fill="{MUT}">Figure 5. Research &amp; engineering progression (2024-2026), publications, real-robot &amp; agent systems, honors, and technical stack.</text>
</svg>'''
    _save_svg_and_png("05_publications_awards_stack", svg_str)


def main() -> None:
    build_01_profile_roadmap()
    build_02_edgeground_vla_v1_v2()
    build_03_research_question_diagnostic()
    build_04_eg1_vs_eg2_protocol_y2()
    build_05_publications_awards_stack()


if __name__ == "__main__":
    main()

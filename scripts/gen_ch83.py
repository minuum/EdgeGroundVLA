#!/usr/bin/env python3
"""research_story_src.html에 CH83(E2E 대체 가능성 심층 분석 · 6장면 사진 오버레이 · 3계열 파레토 프론티어, 2026-11-22)을 (재)생성하고 발행한다. 멱등.
실행: .venv/bin/python3 scripts/gen_ch83.py
"""
from __future__ import annotations

import html
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import ch_common as CC  # noqa: E402
import gen_ch77 as G77  # noqa: E402

HTML, tbl, H, st = G77.HTML, G77.tbl, G77.H, G77.st
FM = ROOT / "docs/v5/egvla2_fm"


def _load(name: str) -> dict:
    p = FM / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _scene_svg(sc: dict) -> str:
    """실제 로봇 세션 사진 위에 BBox(Florence-2 / Soft-Loc)와 핵심 4개 모델 방향 화살표를 SVG로 오버레이."""
    W, Hh = 416, 234
    elems = []

    def add_box(bb: list[float] | None, col: str, dash: str, lbl: str, dy: int = 0) -> None:
        if not bb or bb[3] < 0.3:
            return
        side = math.sqrt(max(0.002, float(bb[2])))
        bw = min(0.92, side * 1.15) * W
        bh = min(0.92, side / 1.15) * Hh
        x = max(2.0, min(W - bw - 2.0, float(bb[0]) * W - bw / 2.0))
        y = max(2.0, min(Hh - bh - 2.0, float(bb[1]) * Hh - bh / 2.0))
        d_attr = f' stroke-dasharray="{dash}"' if dash else ""
        elems.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="none" stroke="{col}" stroke-width="2.2"{d_attr}/>')
        ty = max(12.0, y - 4.0 + dy)
        elems.append(f'<text x="{x+3:.1f}" y="{ty:.1f}" fill="{col}" font-size="9.5" font-weight="700">{html.escape(lbl)}</text>')

    # Florence-2 BBox + Soft-Loc BBox
    bb = sc.get("bbox")
    add_box(bb, "#38bdf8", "", f"F2 BBox (cx={bb[0]:.2f}, a={bb[2]:.2f})", 0)
    ms = sc.get("models", {})
    sl_m = ms.get("VLM-E2E-SoftLoc", {})
    if sl_m.get("softloc_bbox"):
        sb = sl_m["softloc_bbox"]
        add_box(sb, "#e879f9", "4 3", f"Soft-Loc (cx={sb[0]:.2f}, a={sb[2]:.2f})", 12)

    # 핵심 4개 모델 화살표: MOD-A3-AG, ENC-E2E-1(B1), VLM-E2E-Hidden, VLM-E2E-SoftLoc
    show = [
        ("MOD-A3-AG", "A3-AG", "#4ade80", 0),
        ("ENC-E2E-1", "B1(ENC)", "#f87171", -10),
        ("VLM-E2E-Hidden", "Hidden", "#c084fc", 10),
        ("VLM-E2E-SoftLoc", "SoftLoc", "#e879f9", 0),
    ]
    ox, oy, scale = 208.0, 214.0, 112.0
    for key, tag, col, yoff in show:
        m = ms.get(key)
        if not m:
            continue
        lx, ly = float(m["cmd"][0]), float(m["cmd"][1])
        if m["stop"]:
            r = 6 + abs(yoff) * 0.25
            elems.append(f'<circle cx="{ox:.1f}" cy="{oy:.1f}" r="{r:.1f}" fill="none" stroke="{col}" stroke-width="2.3"/>')
        else:
            tx = max(16.0, min(W - 44.0, ox - ly * scale))
            ty = max(16.0, min(Hh - 12.0, oy - lx * scale))
            elems.append(f'<line x1="{ox:.1f}" y1="{oy:.1f}" x2="{tx:.1f}" y2="{ty:.1f}" stroke="{col}" stroke-width="3.0" stroke-linecap="round"/>')
            elems.append(f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="3.8" fill="{col}"/>')
            elems.append(f'<text x="{tx+5:.1f}" y="{ty+3+yoff*0.5:.1f}" fill="{col}" font-size="9.5" font-weight="800">{tag}</text>')

    return (
        f'<div style="position:relative;width:100%;aspect-ratio:16/9;border-radius:8px;overflow:hidden;background:#000;border:1px solid #233252">'
        f'<img src="{html.escape(sc["img_src"])}" alt="{html.escape(sc["tab"])}" loading="lazy" decoding="async" style="width:100%;height:100%;object-fit:cover;display:block"/>'
        f'<svg viewBox="0 0 {W} {Hh}" style="position:absolute;inset:0;width:100%;height:100%;pointer-events:none">'
        + "".join(elems)
        + "</svg></div>"
    )


def _scene_cards_html(scenes: list[dict]) -> str:
    """실기 6장면 사진 + 모델별 수치/판정 카드 2열 그리드."""
    if not scenes:
        return '<div style="color:#94a3b8">장면 데이터 없음</div>'

    verdicts = {
        "sc1_chair_center": "미학습 의자 정중앙 원거리(a=0.06): <b>MOD-A3-AG·Hidden·SoftLoc 모두 정상 전진</b>. 반면 기존 A3는 비출발, <b>B1(ENC-E2E-1)</b>은 방향 이탈.",
        "sc2_basket_close": "바구니 초근접(a=0.31): <b>MOD-A3-AG · VLM-E2E-Hidden · VLM-E2E-SoftLoc 모두 정확히 STOP!</b> 반면 <code>a3-s2</code>와 <code>B1</code>은 계속 전진(충돌 위험).",
        "sc3_chair_close": "<b>결정적 차이 장면 (미학습 의자 초근접, a=0.33)</b>: <b>VLM-E2E-SoftLoc</b>과 <b>MOD-A3-AG</b>는 정확히 <b>STOP</b>하지만, 기하 병목 없는 <b>VLM-E2E-Hidden</b>은 계속 전진!",
        "sc4_desk_center": "미학습 책상 정중앙 원거리(a=0.05): <b>Hidden·SoftLoc·A3-AG 모두 정상 출발</b>. Soft-Loc이 미학습 책상 박스도 정확히 추적.",
        "sc5_side_steer": "바구니 좌측 조향(cx=0.14): <b>MOD-A3-AG</b>와 <b>VLM-E2E-SoftLoc</b>이 좌측으로 정밀 조향. <b>B1(ENC-E2E-1)</b>은 직진/역방향으로 붕괴.",
        "sc6_pot_center": "미학습 화분 정중앙(a=0.08): <b>VLM-E2E-SoftLoc</b>이 화분 Soft-Loc 박스를 잡아 <b>A3-AG</b>와 동일한 직진 방향 출력.",
    }

    cards = []
    for sc in scenes:
        ms = sc.get("models", {})
        mini_rows = []
        for k, tag, col in (
            ("MOD-A3-AG", "MOD-A3-AG (기준)", "#4ade80"),
            ("ENC-E2E-1", "ENC-E2E-1 (구 B1)", "#f87171"),
            ("VLM-E2E-Hidden", "VLM-E2E-Hidden", "#c084fc"),
            ("VLM-E2E-SoftLoc", "VLM-E2E-SoftLoc", "#e879f9"),
        ):
            m = ms.get(k)
            if not m:
                continue
            badge = '<span style="color:#fca5a5;font-weight:700">STOP</span>' if m["stop"] else '<span style="color:#86efac;font-weight:700">DRIVE</span>'
            mini_rows.append(
                f'<tr><td style="color:{col};font-weight:700;padding:3px 6px">{tag}</td>'
                f'<td style="padding:3px 6px"><code>[{m["cmd"][0]:+.2f}, {m["cmd"][1]:+.2f}]</code> ({m["ang_deg"]:+.0f}°)</td>'
                f'<td style="padding:3px 6px">{badge}</td></tr>'
            )
        vd = verdicts.get(sc["id"], sc.get("real_outcome", ""))
        cards.append(
            '<div style="background:#0c1527;border:1px solid #1e3050;border-radius:10px;padding:10px;display:flex;flex-direction:column;gap:7px">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;gap:6px">'
            f'<b style="font-size:0.84rem;color:#f8fafc">{html.escape(sc["tab"])}</b>'
            f'<span style="font-size:0.72rem;color:#94a3b8"><code>{html.escape(sc["sid"])}</code> #{sc["fi"]}</span></div>'
            f'{_scene_svg(sc)}'
            f'<div style="font-size:0.74rem;color:#94a3b8">지시어: <code>"{html.escape(sc["ins"])}"</code> · 물체: {html.escape(sc["obj"])} ({html.escape(sc["pos"])})</div>'
            f'<table style="width:100%;border-collapse:collapse;font-size:0.74rem;background:#08101f"><tbody>{"".join(mini_rows)}</tbody></table>'
            f'<div style="font-size:0.77rem;line-height:1.55;color:#cbd5e1;border-left:3px solid #c084fc;padding-left:8px">{vd}</div>'
            '</div>'
        )
    leg = (
        '<div style="display:flex;flex-wrap:wrap;gap:10px;font-size:0.77rem;margin:6px 0 10px;padding:7px 10px;background:#0a1628;border:1px solid #1e3050;border-radius:8px">'
        '<span><b style="color:#4ade80">━ MOD-A3-AG</b> (모듈러 기준)</span>'
        '<span><b style="color:#f87171">━ ENC-E2E-1</b> (구 B1, 디코더 제거)</span>'
        '<span><b style="color:#c084fc">━ VLM-E2E-Hidden</b> (BBox=0 순수 1-Pass)</span>'
        '<span><b style="color:#e879f9">━ VLM-E2E-SoftLoc</b> (미분가능 Soft-Loc)</span>'
        '<span><b style="color:#94a3b8">○ 원형</b> = STOP 판정</span>'
        '</div>'
    )
    return leg + '<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:12px">' + "".join(cards) + "</div>"


def _pareto_svg(points: list[tuple[str, float, float, str, str]]) -> str:
    """지연시간(ms, x축) vs A3 대비 조향 오차(°, y축) 파레토 프론티어 SVG 차트."""
    W, Hh, L, R, Tp, Bm = 760, 270, 56, 24, 24, 46
    xmax, ymax = 650.0, 28.0
    px = lambda ms: L + (W - L - R) * min(ms, xmax) / xmax
    py = lambda deg: Tp + (Hh - Tp - Bm) * (1.0 - min(deg, ymax) / ymax)

    g = []
    for ms in (0, 100, 200, 300, 400, 500, 600):
        x = px(ms)
        g.append(f'<line x1="{x:.1f}" y1="{Tp}" x2="{x:.1f}" y2="{Hh-Bm}" stroke="#1e293b" stroke-dasharray="3 3"/>')
        g.append(f'<text x="{x:.1f}" y="{Hh-Bm+15}" text-anchor="middle" fill="#94a3b8" font-size="10">{ms}ms</text>')
    for deg in (0, 5, 10, 15, 20, 25):
        y = py(deg)
        g.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W-R}" y2="{y:.1f}" stroke="#1e293b" stroke-dasharray="3 3"/>')
        g.append(f'<text x="{L-6}" y="{y+3:.1f}" text-anchor="end" fill="#94a3b8" font-size="10">{deg}°</text>')
    g.append(f'<line x1="{L}" y1="{Hh-Bm}" x2="{W-R}" y2="{Hh-Bm}" stroke="#475569"/>')
    g.append(f'<line x1="{L}" y1="{Tp}" x2="{L}" y2="{Hh-Bm}" stroke="#475569"/>')
    g.append(f'<text x="{(L+W-R)/2:.1f}" y="{Hh-8}" text-anchor="middle" fill="#cbd5e1" font-size="11">추론 지연시간 Latency (ms, 낮을수록 빠름 ←)</text>')
    g.append(f'<text x="14" y="{(Tp+Hh-Bm)/2:.1f}" text-anchor="middle" fill="#cbd5e1" font-size="11" transform="rotate(-90 14 {(Tp+Hh-Bm)/2:.1f})">A3 대비 각도 오차 (°, 낮을수록 정확 ↓)</text>')
    g.append(f'<rect x="{L}" y="{py(4):.1f}" width="{px(310)-L:.1f}" height="{Hh-Bm-py(4):.1f}" fill="#4ade80" fill-opacity="0.06" stroke="#4ade80" stroke-dasharray="4 3" stroke-opacity="0.4"/>')
    g.append(f'<text x="{L+8}" y="{py(3.2):.1f}" fill="#4ade80" font-size="9.5" font-weight="700">목표 파레토 영역 (≤300ms · ≤4°)</text>')

    for label, ms, deg, col, sub in points:
        x, y = px(ms), py(deg)
        g.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.5" fill="{col}" stroke="#0b1120" stroke-width="1.5"/>')
        dx = -8 if ms > 500 else 9
        anc = "end" if ms > 500 else "start"
        g.append(f'<text x="{x+dx:.1f}" y="{y-4:.1f}" text-anchor="{anc}" fill="{col}" font-size="10.5" font-weight="700">{label}</text>')
        g.append(f'<text x="{x+dx:.1f}" y="{y+8:.1f}" text-anchor="{anc}" fill="#94a3b8" font-size="9">{sub}</text>')

    return (
        '<div style="font-size:0.82rem;font-weight:800;color:#fcd34d;margin:8px 0 4px">'
        '3계열 아키텍처 파레토 프론티어: 추론 지연(ms) vs 조향 오차(°)</div>'
        f'<svg viewBox="0 0 {W} {Hh}" style="width:100%;height:auto;background:#0a1628;border:1px solid #1e3050;border-radius:10px">'
        + "".join(g) + "</svg>"
    )


def main() -> None:
    beam = _load("beam_bench.json")
    ta = _load("track_a_dec1pass_results.json")
    fam_cmp = _load("family_arch_compare.json")
    scenes = fam_cmp.get("scenes", [])

    b3_ms = beam.get("beam3_fp32 (현행 기준)", {}).get("latency_med_ms", 590.7)
    b1_ms = beam.get("beam1_fp16 (Track B)", {}).get("latency_med_ms", 283.0)
    b1_sp = beam.get("beam1_fp16 (Track B)", {}).get("speedup_vs_beam3", 2.09)
    sl_ms = beam.get("softloc_fp16 (Track A2)", {}).get("latency_med_ms", 283.8)
    sl_sp = beam.get("softloc_fp16 (Track A2)", {}).get("speedup_vs_beam3", 2.08)

    a1 = ta.get("Track A1 (Decoder 1-Pass 순수 E2E)", {})
    a2 = ta.get("Track A2 (Soft-Loc 미분가능 E2E)", {})
    a2_zs = ta.get("VLM-E2E-SoftLoc (Zero-Shot A3C_AG23)", {})
    a2_v2 = ta.get("VLM-E2E-SoftLoc+ (EG3-10, 3F-Win+Swap80)", {})

    f_pct = lambda d, k: f"{d[k]*100:.1f}%" if k in d and d[k] is not None else "—"
    f_deg = lambda d, k: f"{d[k]:.2f}°" if k in d and d[k] is not None else "—"

    P = lambda t: f'<div style="font-size:0.83rem;line-height:1.8;margin:6px 0">{t}</div>'

    # ── Tab 1: 실기 6장면 사진 위 E2E 화살표 비교 ──
    tab1 = [
        P('<b>실제 로봇 카메라 사진 6장 위에서 각 E2E 모델이 어떻게 보고 어디로 가는지</b>를 모듈러 1순위(`MOD-A3-AG`)와 나란히 겹쳐 확인한다. 특히 <b>장면 ③(의자 초근접)</b>이 <code>VLM-E2E-Hidden</code>과 <code>VLM-E2E-SoftLoc</code>의 결정적 차이를 한눈에 보여준다. (단계별 텐서·레이어·입출력 흐름은 <a href="phase14_serving_timeline.html"><b>⏱️ Phase 1~4 서빙 가로 타임라인</b></a>, 전체 8모델 토글은 <a href="family_arch_compare.html"><b>🔀 3계열 비교 스튜디오</b></a>에서 확인)'),
        _scene_cards_html(scenes),
    ]

    # ── Tab 2: E2E 대체 가능성 판정 & 모델별 특징 심층 분석 ──
    verdict_box = (
        '<div style="border-left:4px solid #e879f9;background:#1b1030;padding:10px 14px;border-radius:0 10px 10px 0;font-size:0.84rem;line-height:1.75;margin:8px 0">'
        '<b>🎯 핵심 결론 — 현행 A3(모듈러)를 E2E로 대체할 수 있는가?</b><br>'
        '• <b>기존 인코더 단독 E2E (`ENC-E2E-1` / 구 B1, `ENC-E2E-Aux`)</b>: <b style="color:#f87171">대체 불가 (No-Go)</b> — VLM 디코더를 버리면 300ep 바구니 외 미학습 물체(의자·책상·화분) 일반화가 무너짐.<br>'
        '• <b>순수 Hidden 1-Pass E2E (`VLM-E2E-Hidden`, π0형)</b>: <b style="color:#fbbf24">바구니 단일 작업만 가능, 다물체 대체 불가</b> — 속도는 <code>~70ms</code>로 최강이고 바구니 정지 100%이나, 기하 병목이 없어 <b>미학습 의자 근접 정지 18.6%</b>로 충돌 위험.<br>'
        '• <b>미분가능 Soft-Loc E2E (`VLM-E2E-SoftLoc+` / `Zero-Shot AG23`)</b>: <b style="color:#4ade80">조건부 실기 대체 가능 (학술 E2E 1순위!)</b> — 문자열 파싱 0% 완전 미분 E2E이면서 지연 <code>283.8ms</code>(2.08x), 각도차 <code>2.71°</code>, 11/13 다물체 중앙 출발 <b>100%</b>, 11/16 바구니·의자 근접 정지 <b>100% / 100%</b> 달성. 단, 실기 1회(15세션) 검증 후 전환 권장.<br>'
        '• <b>즉시 실기 운영 1순위 (`MOD-DR-Fast/Safe` + `MOD-A3-AG`)</b>: <b style="color:#38bdf8">즉시 대체 가능 (Go)</b> — 지연 <code>283.0ms</code>(2.09x), 각도차 <code>0.54°</code>(좌우 98.7%), 기존 서버 코드 100% 호환.'
        '</div>'
    )
    tab2 = [
        verdict_box,
        tbl(
            ["E2E / 비교 모델", "대체 가능 판정", "핵심 메커니즘 (특징)", "강점", "치명적 약점 / 잔여 조건"],
            [
                ["ENC-E2E-1 (`B1`, EG3-04)", "❌ 대체 불가", "DaViT 마지막 단 LoRA → FM (디코더 제거)", "초고속 (~30ms)", "11/13·16 각도 25.13°·좌우 43.3% 붕괴, Open-Vocab 상실"],
                ["ENC-E2E-Aux (`EG3-07`)", "❌ 다물체 불가", "DaViT LoRA + 인코더 Aux BBox 헤드", "generate() 없이 바구니 cx MAE 0.007 (~32ms)", "바구니 300ep만 학습 → 의자·책상·화분 지시어 이해 불가"],
                ["VLM-E2E-Hidden (`EG3-08`, π0형)", "⚠️ 단일물체만", "Florence-2 Enc + Decoder(6L) 1-Pass Hidden → FM (BBox=0)", "빠름(~70ms), B1 대비 오차 3.3배 감소(7.70°), 바구니 정지 100%", "기하 병목 부재로 <b>미학습 의자 근접 정지 18.6%</b> (충돌 위험)"],
                ["VLM-E2E-SoftLoc (`EG3-09/10` + `AG23`)", "✅ 조건부 대체 가능", "Decoder `<loc_0~999>` Softmax 기댓값(미분가능) → FM", "문자열 파싱 0%, 283.8ms(2.08x), 각도 2.71°, 바구니·의자 정지 <b>100%/100%</b>", "서빙 코드에 Soft-Loc 래퍼 연결 + 실기 15세션 검증 필요"],
                ["MOD-DR-Fast/Safe (`a3-ag23`, `beam=1 fp16`)", "✅ 즉시 대체 가능", "Decoder `beam=1 fp16` + Area-Grounded FM + S2 감속", "283.0ms(2.09x), 각도 0.54°(좌우 98.7%), 바구니·의자 정지 90%/93%", "문자열 파싱 경계에서 미분 단절 (성능 손실은 0.3% 미만)"],
            ],
        ),
        P("<b>왜 `VLM-E2E-SoftLoc`이 학술적으로 가장 강력한 E2E 해법인가?</b><br>"
          "1) <b>VLM 사전학습 지식 100% 보존</b>: Florence-2 Decoder의 `<loc_0>~<loc_999>` 토큰 분포를 그대로 활용하므로 300ep 단일 물체 데이터로도 의자·책상·화분 Zero-Shot 검출률 <b>98.8%</b> 유지.<br>"
          "2) <b>명시적 기하 병목의 필요성 입증</b>: 동일한 Decoder에서 Hidden State만 넘긴 `EG3-08`(의자 정지 18.6%) vs Softmax 기댓값 좌표를 넘긴 `EG3-09/10`(의자 정지 93~100%)의 <b>+74~81%p 격차</b>가 소규모 VLA에서 기하 병목이 필수임을 직접 증명.<br>"
          "3) <b>완전 미분 + 속도 2배</b>: 문자열 `generate()`와 정규식 파싱을 제거해 액션 손실이 VLM Decoder까지 역전파 가능하며, Prefix-Forced KV-Cache 4스텝으로 `590.7ms → 283.8ms` (2.08배) 단축."),
    ]

    # ── Tab 3: 3계열 파레토 & 전체 정량표 ──
    pts = [
        ("MOD-A3-AG (a3-ag23)", b3_ms, 0.54, "#38bdf8", f"{b3_ms}ms · 0.54° · 정지 90/93%"),
        ("MOD-DR-Fast (beam=1 fp16)", b1_ms, 0.60, "#4ade80", f"{b1_ms}ms({b1_sp}x) · cx MAE 0.0016"),
        ("ENC-E2E-1 (구 B1)", 30.0, 25.13, "#f87171", "~30ms · 25.13° (시뮬 붕괴)"),
        ("VLM-E2E-Hidden (EG3-08)", 70.0, a1.get("ang_med_vs_a3", 7.7), "#c084fc", f"~70ms · {f_deg(a1,'ang_med_vs_a3')} · 의자정지 {f_pct(a1,'chair_ge25_stop_16')}"),
        ("VLM-E2E-SoftLoc (EG3-09)", sl_ms, a2.get("ang_med_vs_a3", 6.0), "#e879f9", f"{sl_ms}ms · {f_deg(a2,'ang_med_vs_a3')} · 의자정지 {f_pct(a2,'chair_ge25_stop_16')}"),
    ]
    if a2_zs:
        pts.append(("SoftLoc + AG23 (29k)", sl_ms, a2_zs["ang_med_vs_a3"], "#f472b6", f"{sl_ms}ms · {f_deg(a2_zs,'ang_med_vs_a3')} · 바구니/의자 100/100%"))

    t3_rows = [
        ["MOD", "MOD-A3 (`a3`, EG2-09)", f"{b3_ms}ms (1.00x)", "0.00° (100%)", "0% / 4%", "0.0%", "80% / 58%", "현행 배포 (beam=3)"],
        ["MOD", "MOD-A3-AG (`a3-ag23`, EG2-16)", f"{b3_ms}ms (1.00x)", "0.54° (98.7%)", "100% / 100%", "0.0%", "90% / 93%", "Area-Grounded 드롭인 1순위"],
        ["MOD", "MOD-DR-Fast (`a3-dr-fast`, EG2-17)", f"{b1_ms}ms ({b1_sp}x)", "AG23과 99.7% 일치", "100% / 100%", "0.0%", "90% / 93%", "beam=1 fp16 (cx MAE 0.0016)"],
        ["ENC-E2E", "ENC-E2E-1 (`B1`, EG3-04)", "~30ms", "25.13° (43.3%)", "붕괴", "—", "45% / 26%", "디코더 제거 (실기 시뮬 붕괴)"],
        ["VLM-E2E", "VLM-E2E-Hidden (`EG3-08`, π0형)", "~70ms (1-Pass)", f"{f_deg(a1,'ang_med_vs_a3')} ({f_pct(a1,'lat_agree_vs_a3')})", f"{f_pct(a1,'chair_center_13')} / {f_pct(a1,'desk_center_13')}", f_pct(a1,'stop_lt10_16'), f"{f_pct(a1,'basket_ge25_stop_16')} / {f_pct(a1,'chair_ge25_stop_16')}", "외부 BBox=0 완전 차단 E2E"],
        ["VLM-E2E", "VLM-E2E-SoftLoc (`EG3-09`)", f"{sl_ms}ms ({sl_sp}x)", f"{f_deg(a2,'ang_med_vs_a3')} ({f_pct(a2,'lat_agree_vs_a3')})", f"{f_pct(a2,'chair_center_13')} / {f_pct(a2,'desk_center_13')}", f_pct(a2,'stop_lt10_16'), f"{f_pct(a2,'basket_ge25_stop_16')} / {f_pct(a2,'chair_ge25_stop_16')}", "4,860장 120ep, 미분 Soft-Loc"],
    ]
    if a2_v2:
        t3_rows.append(["VLM-E2E", "VLM-E2E-SoftLoc+ (`EG3-10`)", f"{sl_ms}ms ({sl_sp}x)", f"{f_deg(a2_v2,'ang_med_vs_a3')} ({f_pct(a2_v2,'lat_agree_vs_a3')})", f"{f_pct(a2_v2,'chair_center_13')} / {f_pct(a2_v2,'desk_center_13')}", f_pct(a2_v2,'stop_lt10_16'), f"{f_pct(a2_v2,'basket_ge25_stop_16')} / {f_pct(a2_v2,'chair_ge25_stop_16')}", "3프레임 시계열 + 80% Swap"])
    if a2_zs:
        t3_rows.append(["VLM-E2E", "SoftLoc + ZeroShot AG23", f"{sl_ms}ms ({sl_sp}x)", f"{f_deg(a2_zs,'ang_med_vs_a3')} ({f_pct(a2_zs,'lat_agree_vs_a3')})", f"{f_pct(a2_zs,'chair_center_13')} / {f_pct(a2_zs,'desk_center_13')}", f_pct(a2_zs,'stop_lt10_16'), f"{f_pct(a2_zs,'basket_ge25_stop_16')} / {f_pct(a2_zs,'chair_ge25_stop_16')}", "29,161장 AG23 + 미분 Soft-Loc"])

    tab3 = [
        _pareto_svg(pts),
        tbl(["계열", "축약명 (파트너 ID)", "지연 (배속)", "A3 대비 각도차(좌우)", "11/13 의자·책상 중앙", "11/16 원거리 오정지", "11/16 근접정지(바구니/의자)", "비고"], t3_rows),
    ]

    # ── Tab 4: MOD-DR-Fast/Safe & 실기·YahBoom 이식 ──
    b_rows = [
        [k, f"{v['latency_med_ms']}ms ({v['speedup_vs_beam3']}x)", f"{v['has_bbox_rate']*100:.1f}%", f"{v['has_agree_vs_beam3']*100:.1f}%", f"{v['cx_mae_vs_beam3']:.4f}", f"{v['area_mae_vs_beam3']:.4f}"]
        for k, v in beam.items()
    ]
    tab4 = [
        P("실기 즉시 배포(`MOD-DR-Fast/Safe`)와 YahBoom 이식을 위한 디코딩 속도 벤치마크(330프레임) 및 컨트롤러 안전 패치 내역이다."),
        tbl(["디코딩 모드", "중앙값 지연 (배속)", "검출률", "beam=3 일치율", "cx MAE", "area MAE"], b_rows),
        P("<b>MOD-DR-Safe 컨트롤러 패치 (`robovlm_nav/serve/egvla2_only_logic.py`, 기본 끔 · 단위 테스트 통과):</b><br>"
          "• <b>S2 연속 접근 감속 (`decel_start_area=0.14`, `decel_min_scale=0.35`)</b>: 근접할수록 선형 감속해 관성 접촉 억제.<br>"
          "• <b>단발성 면적 스파이크 필터 (`jump_area_ratio=2.5`)</b>: 1프레임 오검출 스파이크로 인한 조기 오정지 방지.<br>"
          "• <b>YahBoom 이식</b>: Serbot 정지 검증 완료 후 `[lx, ly] → [v, ω]` 구동계 어댑터와 카메라 1프레임 기준 면적 보정만 교체."),
    ]

    # ── Tab 5: EGVLA2 vs EGVLA3 세대 분리 계획 & 각각의 강점 ──
    tab5 = [
        P('한 저장소 안에서 <b>실기 안정화 트랙(EGVLA2)</b>과 <b>차세대 완전 미분 E2E 연구 트랙(EGVLA3)</b>이 혼동되지 않도록, <b>"미분 단절 유무(문자열 정규식 파싱 vs 텐서·확률 분포 직결)"</b>를 기준으로 두 세대의 정체성·강점·역할을 분리했다. (전문: <a href="../plans/plan_20261124_egvla2_vs_egvla3_separation.md"><b>EGVLA2 vs EGVLA3 세대 분리 계획서</b></a>)'),
        tbl(
            ["비교 축", "🔵 EGVLA2 (2세대 · MOD-* 계열)", "🟣 EGVLA3 (3세대 · VLM-E2E-* 계열)"],
            [
                ["핵심 정의", "Explicit Modular Grounding + Flow-Matching VLA", "Differentiable Spatial-Bottleneck End-to-End VLA"],
                ["대표 모델", "`MOD-A3`(`a3`), `MOD-A3-AG`(`a3-ag23`), `MOD-DR-Safe`, `MOD-V4R`", "`VLM-E2E-Hidden`(`EG3-08`), `VLM-E2E-SoftLoc+`(`EG3-09/10` + `AG23`)"],
                ["VLM ↔ Action 경계", "Decoder `generate()` → 문자열 `<loc>` 정규식 파싱 (비미분)", "Prefix-Forced `<loc_0~999>` Softmax 기댓값 $\\hat{u}=\\sum p_i(i/1000)$ (완전 미분)"],
                ["대응 SOTA 패러다임", "NaVILA (arXiv:2412.04453), RT-H (arXiv:2403.01823)", "π0 (arXiv:2410.24164), ECoT (arXiv:2407.08693), SpatialVLA (arXiv:2501.15830)"],
                ["종단 서빙 지연 (GB10)", "`590.7ms`(beam=3) → <b>`285.3ms`</b>(`beam=1 fp16`, 2.09x)", "<b>`~70ms`</b>(`Hidden`) / <b>`296.6ms`</b>(`Soft-Loc fp16`, 2.08x)"],
                ["조향 정밀도 / 근접 정지", "<b>`0.54°` (좌우 98.7%)</b> / 바구니 90% · 의자 93%", "<b>`2.71°` (좌우 82.8%)</b> / <b>바구니 100% · 의자 100%</b>"],
                ["독보적 4대 강점", "① 실기 560+ 세션 검증력<br>② 이산 토큰 절단으로 최고 조향 정밀도(`0.54°`)<br>③ Dual-Shortcut 해소(`a3-ag23`) + 언어 게이트(`V4R`)<br>④ S2 감속·스파이크 필터·YahBoom 무재학습 즉시 이식", "① VLM Decoder 6L 완전 미분 관통(파싱 0%)<br>② 300ep에서 π0형 Hidden E2E 한계(`18.6%`) 규명 및 Soft-Loc(`100%`) 돌파 서사<br>③ Prefix-Forced KV-Cache 4스텝 2배 가속<br>④ Action Loss → Decoder LoRA 공동 역전파 확장성"],
                ["단일 서버 스위치", "`VLA_EGVLA2_DECODE_MODE=beam1_fp16` (실기 주력 트랙)", "`VLA_EGVLA2_DECODE_MODE=softloc_fp16` (차세대 E2E 연구 트랙)"],
            ],
        ),
    ]

    tabs = [
        ("photos", "📸 실기 6장면 E2E 비교", [("html", re.sub(r"`([^`]+)`", r"<code>\1</code>", "".join(tab1)))]),
        ("verdict", "🔍 E2E 대체 가능성·특징", [("html", re.sub(r"`([^`]+)`", r"<code>\1</code>", "".join(tab2)))]),
        ("gen_sep", "🏛️ EGVLA2 vs EGVLA3 분리", [("html", re.sub(r"`([^`]+)`", r"<code>\1</code>", "".join(tab5)))]),
        ("pareto", "📊 3계열 파레토·정량표", [("html", re.sub(r"`([^`]+)`", r"<code>\1</code>", "".join(tab3)))]),
        ("deploy", "⚡ MOD-DR-Fast/Safe", [("html", re.sub(r"`([^`]+)`", r"<code>\1</code>", "".join(tab4)))]),
    ]
    inner = CC.tabs_html("ch83", tabs, "", {})
    title = "E2E 대체 가능성 심층 분석 & 3계열 파레토 — MOD vs ENC-E2E vs VLM-E2E (Soft-Loc)"
    sub = (
        "실제 로봇 세션 사진 6장 위에서 MOD · ENC-E2E · VLM-E2E(Hidden/Soft-Loc)의 방향과 정지 판단을 직접 대조하고, "
        "현행 A3를 E2E로 대체할 수 있는지 모델별 특징·한계·조건을 정량 판정했다."
    )
    blk = CC.chapter_block(83, "#c084fc", "#e9d5ff", title, sub, "2026-11-22", inner)

    v2_str = f", SoftLoc+(EG3-10) {f_deg(a2_v2,'ang_med_vs_a3')}·바구니/의자 정지 100%/100%, ZeroShot AG23 결합 시 {f_deg(a2_zs,'ang_med_vs_a3')}" if a2_v2 and a2_zs else ""
    summary_text = (
        f"실기 6장면 사진 위 E2E vs 모듈러 직접 대조 및 대체 가능성 판정. "
        f"ENC-E2E-1(B1, 25.13°)·ENC-E2E-Aux는 Open-Vocab 상실로 대체 불가; "
        f"VLM-E2E-Hidden(BBox=0, {f_deg(a1,'ang_med_vs_a3')})은 바구니 정지 100%이나 미학습 의자 정지 {f_pct(a1,'chair_ge25_stop_16')}로 다물체 대체 불가; "
        f"미분가능 VLM-E2E-SoftLoc({sl_ms}ms, 2.08x)은 의자 정지 {f_pct(a2,'chair_ge25_stop_16')}로 폭등{v2_str}하며 조건부 실기 대체 가능(학술 E2E 1순위). "
        f"즉시 운영 1순위는 MOD-DR-Fast/Safe(beam=1 fp16 {b1_ms}ms, 2.09x + S2 접근 감속)."
    )
    row = CC.summary_row(83, "E2E 대체 가능성 · 6장면 사진 대조(11/22)", summary_text, "#c084fc")

    s = HTML.read_text(encoding="utf-8")
    s = CC.insert_chapter(s, 83, blk, "<!--/CH82-->", title, "🧬 CH83 E2E 분석·사진", "E2E·파레토", "#c084fc", "#e9d5ff", "#c084fc", "#240f3b", row, 82)
    s = CC.replace_toc_label(s, 83, title)
    s = CC.replace_summary_row(s, 83, "E2E 대체 가능성 · 6장면 사진 대조(11/22)", summary_text, "#c084fc")
    s = re.sub(r"현재 상태 \(CH1→CH\d+ · [0-9-]+\)", "현재 상태 (CH1→CH83 · 2026-11-22)", s, count=1)
    HTML.write_text(s, encoding="utf-8")

    __import__("story_publish").publish()
    for t, a, b in CC.balance(s):
        print(t, a, b)
    print("CH83 with 6 real-session photo overlays & E2E verdict published cleanly!")


if __name__ == "__main__":
    main()

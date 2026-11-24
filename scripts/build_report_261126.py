#!/usr/bin/env python3
"""2026-11-26 EGVLA 실험보고 발표용 페이지 생성기.

실제 로봇 카메라 사진(figures/inline/*.jpg) + GPU 실측 JSON + 과적합 감사 결과를 통합해
발표 1시간 전에도 그대로 띄워 설명할 수 있는 5단계 스토리라인 리포트를 생성한다.

출력: docs/v5/report_261126_egvla.html
"""
from __future__ import annotations

import datetime
import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FM = ROOT / "docs/v5/egvla2_fm"
OUT_HTML = ROOT / "docs/v5/report_261126_egvla.html"


def _scene_svg(sc: dict) -> str:
    """실제 로봇 사진 위 BBox(F2 + Soft-Loc) 및 4대 핵심 모델 주행/정지 벡터 오버레이."""
    W, H = 416, 234
    elems = []

    def draw_box(bb, col, dash, lbl, dy=0):
        if not bb or bb[3] < 0.3:
            return
        side = math.sqrt(max(0.002, float(bb[2])))
        bw = min(0.92, side * 1.15) * W
        bh = min(0.92, side / 1.15) * H
        x = max(2.0, min(W - bw - 2.0, float(bb[0]) * W - bw / 2.0))
        y = max(2.0, min(H - bh - 2.0, float(bb[1]) * H - bh / 2.0))
        da = f' stroke-dasharray="{dash}"' if dash else ""
        elems.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="none" stroke="{col}" stroke-width="2.2"{da}/>')
        elems.append(f'<text x="{x+3:.1f}" y="{max(12.0, y-4.0+dy):.1f}" fill="{col}" font-size="9.5" font-weight="700">{html.escape(lbl)}</text>')

    bb = sc.get("bbox")
    draw_box(bb, "#38bdf8", "", f"F2 BBox (cx={bb[0]:.2f}, a={bb[2]:.2f})", 0)
    ms = sc.get("models", {})
    sl_m = ms.get("VLM-E2E-SoftLoc", {})
    if sl_m.get("softloc_bbox"):
        sb = sl_m["softloc_bbox"]
        draw_box(sb, "#e879f9", "4 3", f"Soft-Loc (cx={sb[0]:.2f}, a={sb[2]:.2f})", 12)

    ox, oy, scale = 208.0, 214.0, 112.0
    for key, tag, col, yoff in (
        ("MOD-A3", "a3(현배포)", "#94a3b8", -12),
        ("MOD-A3-AG", "a3-ag23", "#4ade80", 0),
        ("VLM-E2E-Hidden", "Hidden(π0형)", "#c084fc", 10),
        ("VLM-E2E-SoftLoc", "SoftLoc(E2E)", "#e879f9", 0),
    ):
        m = ms.get(key)
        if not m:
            continue
        lx, ly = float(m["cmd"][0]), float(m["cmd"][1])
        if m["stop"]:
            r = 6 + abs(yoff) * 0.25
            elems.append(f'<circle cx="{ox:.1f}" cy="{oy:.1f}" r="{r:.1f}" fill="none" stroke="{col}" stroke-width="2.4"/>')
        else:
            tx = max(16.0, min(W - 56.0, ox - ly * scale))
            ty = max(16.0, min(H - 12.0, oy - lx * scale))
            elems.append(f'<line x1="{ox:.1f}" y1="{oy:.1f}" x2="{tx:.1f}" y2="{ty:.1f}" stroke="{col}" stroke-width="3.0" stroke-linecap="round"/>')
            elems.append(f'<circle cx="{tx:.1f}" cy="{ty:.1f}" r="4" fill="{col}"/>')
            elems.append(f'<text x="{tx+5:.1f}" y="{ty+3+yoff*0.4:.1f}" fill="{col}" font-size="9.5" font-weight="800">{tag}</text>')

    return (
        f'<div class="cam-wrap">'
        f'<img src="{html.escape(sc["img_src"])}" alt="{html.escape(sc["tab"])}" loading="lazy" decoding="async">'
        f'<svg viewBox="0 0 {W} {H}">{"".join(elems)}</svg>'
        f'</div>'
    )


def main() -> None:
    fam = json.loads((FM / "family_arch_compare.json").read_text(encoding="utf-8"))
    beam = json.loads((FM / "beam_bench.json").read_text(encoding="utf-8"))
    ta = json.loads((FM / "track_a_dec1pass_results.json").read_text(encoding="utf-8"))
    audit = json.loads((FM / "overfitting_audit.json").read_text(encoding="utf-8"))

    scenes = fam["scenes"]
    b3_ms = beam["beam3_fp32 (현행 기준)"]["latency_med_ms"]
    b1_ms = beam["beam1_fp16 (Track B)"]["latency_med_ms"]
    b1_sp = beam["beam1_fp16 (Track B)"]["speedup_vs_beam3"]
    sl_ms = beam["softloc_fp16 (Track A2)"]["latency_med_ms"]
    sl_sp = beam["softloc_fp16 (Track A2)"]["speedup_vs_beam3"]

    eg11 = ta["VLM-E2E-SoftLoc-Joint (EG3-11, DecLoRA+KL)"]
    eg08 = ta["Track A1 (Decoder 1-Pass 순수 E2E)"]
    eg09 = ta["Track A2 (Soft-Loc 미분가능 E2E)"]
    zs = ta["VLM-E2E-SoftLoc (Zero-Shot A3C_AG23)"]

    scene_takeaways = {
        "sc1_chair_center": "<b>[출발 기로]</b> 의자가 정중앙에 있어도 기존 <code>a3</code>는 시각 지름길 때문에 <b>정지(0/5)</b>. 반면 <code>a3-ag23</code>·<code>Hidden</code>·<code>SoftLoc</code>은 모두 정상 출발!",
        "sc2_basket_close": "<b>[바구니 초근접]</b> 면적 0.31에서 <code>a3-ag23</code>·<code>Hidden</code>·<code>SoftLoc</code> 모두 <b>정확히 STOP</b>. 반면 <code>a3-s2</code>와 <code>B1</code>은 계속 전진(충돌 원인).",
        "sc3_chair_close": "<b>[결정적 증거 ★]</b> 미학습 의자 초근접(a=0.33): <b><code>VLM-E2E-Hidden</code>(π0형)은 계속 전진</b>하지만, <b><code>SoftLoc</code>과 <code>a3-ag23</code>은 정확히 STOP</b>!",
        "sc4_desk_center": "<b>[낯선 물체]</b> 작은 책상 정중앙: <code>Soft-Loc</code>이 책상 위치(cx=0.49)를 정확히 잡아 <code>a3-ag23</code>과 동일하게 직진.",
        "sc5_side_steer": "<b>[측면 조향]</b> 바구니가 좌측(cx=0.14)에 있을 때 <code>a3-ag23</code>·<code>SoftLoc</code> 모두 좌측으로 예리하게 조향(<code>B1</code>은 이탈).",
        "sc6_pot_center": "<b>[화분 일반화]</b> 300ep 학습 때 한 번도 못 본 화분도 <code>Soft-Loc</code> 박스로 잡아 <code>a3-ag23</code>과 일치하게 직진.",
    }

    cards_html = []
    for sc in scenes:
        ms = sc["models"]
        rows = []
        for k, lbl, col in (
            ("MOD-A3", "a3 (현 배포)", "#94a3b8"),
            ("MOD-A3-s2", "a3-s2 (11/16실기)", "#fb923c"),
            ("MOD-A3-AG", "a3-ag23 (EGVLA2 1순위)", "#4ade80"),
            ("VLM-E2E-Hidden", "EG3-08 (Hidden π0형)", "#c084fc"),
            ("VLM-E2E-SoftLoc", "EG3-09/11 (Soft-Loc)", "#e879f9"),
        ):
            m = ms.get(k)
            if not m:
                continue
            st = '<span class="bd b-bad">STOP</span>' if m["stop"] else '<span class="bd b-ok">DRIVE</span>'
            rows.append(
                f'<tr><td style="color:{col};font-weight:700">{lbl}</td>'
                f'<td><code>[{m["cmd"][0]:+.2f}, {m["cmd"][1]:+.2f}]</code> ({m["ang_deg"]:+.0f}°)</td>'
                f'<td>{st}</td></tr>'
            )
        cards_html.append(
            f'<div class="sc-card">'
            f'<div class="sc-top"><b>{html.escape(sc["tab"])}</b><span><code>{html.escape(sc["sid"])}</code> #{sc["fi"]}</span></div>'
            f'{_scene_svg(sc)}'
            f'<div class="sc-ins">지시어: <code>"{html.escape(sc["ins"])}"</code> · {html.escape(sc["obj"])} ({html.escape(sc["pos"])})</div>'
            f'<table class="mini-tbl"><tbody>{"".join(rows)}</tbody></table>'
            f'<div class="sc-note">{scene_takeaways.get(sc["id"], "")}</div>'
            f'</div>'
        )

    audit_rows = []
    for mname, r in audit.items():
        vcol = "b-ok" if "없음" in r["verdict"] or "완료" in r["verdict"] else ("b-bad" if "과적합" in r["verdict"] else "b-warn")
        audit_rows.append(
            f'<tr><td><b>{html.escape(mname)}</b></td>'
            f'<td>{r["basket_center_go"]:.1f}% vs {r["chair_center_go"]:.1f}% (<b>{r["unseen_go_gap_pp"]:.1f}%p</b>)</td>'
            f'<td>{r["basket_ge25_stop"]:.1f}% vs {r["chair_ge25_stop"]:.1f}% (<b>{r["unseen_stop_gap_pp"]:.1f}%p</b>)</td>'
            f'<td>{r["vis_swap_ang_deg"]:.1f}° / <b>{r["vis_swap_stop_flip_pct"]:.1f}%</b></td>'
            f'<td>{r["S_cx_deg"]:+.1f}° / <b>{r["S_area_pp"]:+.1f}%p</b></td>'
            f'<td><span class="bd {vcol}">{html.escape(r["verdict"])}</span></td></tr>'
        )

    gen_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    doc = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>261126 EGVLA 실험보고 — EGVLA2 실기 완성 &amp; EGVLA3 완전 미분 E2E</title>
<style>
:root{{--bg:#080e1c;--card:#10192e;--head:#16233f;--fg:#e2e8f0;--mut:#94a3b8;--line:#223354;--acc:#38bdf8;--ok:#4ade80;--warn:#fbbf24;--bad:#f87171;--purp:#c084fc;--pink:#e879f9}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,'Noto Sans KR',sans-serif}}
main{{max-width:1320px;margin:0 auto;padding:22px 18px 80px}}
.top-nav{{position:sticky;top:0;z-index:30;background:rgba(8,14,28,.94);backdrop-filter:blur(8px);border-bottom:1px solid var(--line);padding:9px 0;margin:-22px -18px 18px;padding-left:18px;padding-right:18px;display:flex;flex-wrap:wrap;gap:8px;align-items:center;font-size:.8rem}}
.top-nav a{{border:1px solid var(--line);background:var(--card);color:var(--acc);padding:4px 10px;border-radius:99px;text-decoration:none;font-weight:700}}
.hero{{background:linear-gradient(135deg,#0f1a33,#1e1136,#0c1e2e);border:1px solid var(--line);border-radius:16px;padding:22px 24px;margin-bottom:20px}}
.hero h1{{margin:0 0 6px;font-size:1.68rem;color:#fff}}
.hero p{{margin:0;color:#cbd5e1;font-size:.92rem}}
.kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin-top:14px}}
.kpi{{background:rgba(12,20,38,.85);border:1px solid var(--line);border-radius:11px;padding:10px 13px}}
.kpi .t{{font-size:.72rem;font-weight:800;color:var(--acc);text-transform:uppercase}}
.kpi .v{{font-size:1.25rem;font-weight:800;color:#fff;margin:2px 0}}
.kpi .d{{font-size:.77rem;color:var(--mut)}}

section.act{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px 20px;margin:18px 0;scroll-margin-top:58px}}
section.act h2{{margin:0 0 8px;font-size:1.22rem;color:#fcd34d;display:flex;align-items:center;gap:8px}}
.lead{{font-size:.88rem;color:#cbd5e1;margin:0 0 12px}}
code{{background:#1b2944;padding:1px 6px;border-radius:4px;font-size:.82em;color:#f1f5f9}}
.bd{{display:inline-block;padding:2px 8px;border-radius:99px;font-size:.72rem;font-weight:700;white-space:nowrap}}
.b-ok{{background:#14532d;color:#86efac}}.b-bad{{background:#450a0a;color:#fca5a5}}.b-warn{{background:#451a03;color:#fcd34d}}.b-blue{{background:#0c2d48;color:#7dd3fc}}.b-purp{{background:#2e1065;color:#e9d5ff}}

.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
@media(max-width:860px){{.two-col{{grid-template-columns:1fr}}}}
.box{{background:#0b1324;border:1px solid var(--line);border-radius:12px;padding:13px 15px;font-size:.84rem;line-height:1.65}}
.box.mod{{border-top:4px solid var(--acc)}}.box.e2e{{border-top:4px solid var(--pink)}}

.sc-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(370px,1fr));gap:14px}}
.sc-card{{background:#0b1324;border:1px solid var(--line);border-radius:12px;padding:11px;display:flex;flex-direction:column;gap:7px}}
.sc-top{{display:flex;justify-content:space-between;align-items:center;font-size:.84rem;color:#fff}}
.sc-top span{{font-size:.72rem;color:var(--mut)}}
.cam-wrap{{position:relative;width:100%;aspect-ratio:16/9;border-radius:8px;overflow:hidden;background:#000;border:1px solid #24375a}}
.cam-wrap img{{width:100%;height:100%;object-fit:cover;display:block}}
.cam-wrap svg{{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}}
.sc-ins{{font-size:.75rem;color:var(--mut)}}
.mini-tbl{{width:100%;border-collapse:collapse;font-size:.75rem;background:#070d19}}
.mini-tbl td{{border:1px solid #1d2c49;padding:3px 6px}}
.sc-note{{font-size:.78rem;color:#e2e8f0;border-left:3px solid var(--pink);padding-left:8px;line-height:1.5}}

table.main-tbl{{width:100%;border-collapse:collapse;font-size:.82rem;background:#0b1324;border-radius:10px;overflow:hidden;margin:10px 0}}
table.main-tbl th,table.main-tbl td{{border:1px solid var(--line);padding:7px 9px;text-align:left;vertical-align:top}}
table.main-tbl th{{background:var(--head);color:#cbd5e1;font-size:.77rem}}
@media print{{.top-nav{{display:none}}body{{background:#fff;color:#111}}section.act,.box,.sc-card,table.main-tbl{{background:#fff!important;color:#111!important;border-color:#ccc!important}}}}
</style>
</head>
<body>
<main>
  <nav class="top-nav">
    <span style="font-weight:800;color:#fff">📑 261126 EGVLA 실험보고</span>
    <a href="#act1">1. 배경·문제</a>
    <a href="#act2">2. 실기 6장면 증거</a>
    <a href="#act3">3. EGVLA2 vs 3</a>
    <a href="#act4">4. 과적합·SOTA 비교</a>
    <a href="#act5">5. 로드맵</a>
    <span style="margin-left:auto;color:var(--mut)">생성 {html.escape(gen_ts)} · <a href="phase14_serving_timeline.html" style="border:none;background:none;padding:0">상세 타임라인 ↗</a></span>
  </nav>

  <div class="hero">
    <h1>2026-11-26 EGVLA 실험보고: 실기 안정화(EGVLA2)와 완전 미분 E2E VLA(EGVLA3)</h1>
    <p>300 에피소드(바구니 단일 물체) 소규모 데이터 환경에서 <b>실기 주행 안정성·속도 2배 개선</b>과 <b>VLM Decoder 완전 미분 End-to-End 학습·추론</b>을 동시에 달성한 전체 실험 보고서</p>
    <div class="kpi-grid">
      <div class="kpi"><div class="t">EGVLA2 실기 1순위</div><div class="v">0.54° · 정지 90/93%</div><div class="desc"><code>A3C_AG23_s0</code> (중앙 출발 0%→100%)</div></div>
      <div class="kpi"><div class="t">추론 속도 2배 가속</div><div class="v">{b3_ms}ms → {b1_ms}ms ({b1_sp}x)</div><div class="desc"><code>beam=1 fp16</code> (BBox 일치율 99.7%)</div></div>
      <div class="kpi"><div class="t">EGVLA3 최종 완성</div><div class="v">2.90° · 의자정지 98%</div><div class="desc"><code>EG3-11</code> (Action→DecLoRA + KL증류)</div></div>
      <div class="kpi"><div class="t">Open-Vocab · 과적합</div><div class="v">검출 99.8% · 뒤집힘 0%</div><div class="desc">미학습 의자·책상·화분 망각 0% 검증</div></div>
    </div>
  </div>

  <section class="act" id="act1">
    <h2>Act 1. 출발점과 문제 정의 — 왜 잘 가던 A3가 의자 앞에서 멈췄는가?</h2>
    <p class="lead">현행 배포 모델 <b><code>a3</code> (EGVLA2)</b>는 바구니 실기 <b>455/455</b>를 성공했지만, 11/13·11/16 다물체 실기 테스트에서 <b>3가지 구조적 병목</b>이 확인되었다.</p>
    <div class="two-col">
      <div class="box">
        <b>① Dual-Shortcut (이중 지름길) 발견</b><br>
        • <code>a3</code>는 조향 각도는 BBox 중심(<code>cx</code>)에 정합됐지만, <b>출발/정지 크기는 768-d 이미지 외형에 의존</b>했다.<br>
        • 그 결과 낯선 의자·책상 정중앙에서 <i>"이미 도착했나?"</i>로 오인해 <b>의자 중앙 출발 0/5 (0%)</b> 발생.<br>
        • 시드2(<code>a3-s2</code>)는 중앙 출발(5/5)은 풀었으나 근접 정지가 둔해져(정지율 12~22%) <b>접촉 6건</b> 발생.
      </div>
      <div class="box">
        <b>② 속도 병목(591ms) &amp; 비미분 모듈러 한계</b><br>
        • 전체 추론 지연의 94%가 Florence-2 <code>beam=3 fp32</code> 문자열 생성에서 발생.<br>
        • 문자열 정규식 파싱 경계에서 역전파가 끊어져 VLM과 Action Expert가 공동 학습되지 못함.<br>
        • 반대로 Decoder를 제거한 기존 <code>B1</code>(<code>ENC-E2E-1</code>)은 각도 오차 <b>25.13°</b>로 붕괴.
      </div>
    </div>
  </section>

  <section class="act" id="act2">
    <h2>Act 2. 실제 로봇 카메라 6장면 시각 증거 (11/13 · 11/16 실기 프레임 GPU 실측)</h2>
    <p class="lead">실제 로봇이 촬영한 카메라 프레임 6장 위에서 각 모델의 <b>BBox 검출</b>과 <b>출력 주행 벡터 <code>[lx, ly]</code></b>를 직접 비교한다. (범례: <b style="color:#94a3b8">회색=a3</b> · <b style="color:#4ade80">초록=a3-ag23</b> · <b style="color:#c084fc">보라=EG3-08(Hidden)</b> · <b style="color:#e879f9">분홍=Soft-Loc(E2E)</b> · <b>원(○)=STOP</b>)</p>
    <div class="sc-grid">{"".join(cards_html)}</div>
  </section>

  <section class="act" id="act3">
    <h2>Act 3. 두 세대(EGVLA2 vs EGVLA3) 분리 전략 및 핵심 성과</h2>
    <div class="two-col">
      <div class="box mod">
        <b style="color:#7dd3fc;font-size:.98rem">🔵 EGVLA2 (MOD-* · 실기 완성형)</b><br>
        • <b><code>A3C_AG23_s0</code></b>: 조향(<code>cx</code>) + 정지(<code>area≥0.23</code>) 동시 기하 정합 + 시각 특징 50% 교체 → T7 <b>0.54°</b>, 중앙 출발 <b>100%</b>, 근접 정지 <b>90%/93%</b>.<br>
        • <b><code>MOD-DR-Fast/Safe</code></b>: <code>beam=1 fp16</code>으로 <b>{b1_ms}ms ({b1_sp}x)</b> 가속 + S2 접근 감속(<code>0.14→0.25</code>) + 스파이크 필터(<code>2.5x</code>) + <b>YahBoom 어댑터</b>.<br>
        • <b><code>V4 / V4R</code></b>: 바구니 전용 언어 게이트(비목표 <b>99.5% 정지</b>, 반전 방어 <b>94.3%</b>).
      </div>
      <div class="box e2e">
        <b style="color:#f5d0fe;font-size:.98rem">🟣 EGVLA3 (VLM-E2E-* · 100% 미분 E2E)</b><br>
        • <b><code>EG3-08</code> (π0형 Hidden)</b>: <code>~70ms</code> 초고속이지만 기하 병목 부재로 의자 정지 <b>{eg08['chair_ge25_stop_16']*100:.1f}%</b> 한계 노출.<br>
        • <b><code>EG3-09/10</code> (Soft-Loc)</b>: Prefix-Forced 4스텝 <code>&lt;loc&gt;</code> Softmax 기댓값(파싱 0%, <b>{sl_ms}ms</b>)으로 의자 정지 <b>93~100%</b> 돌파.<br>
        • <b><code>EG3-11</code> (SoftLoc-Joint ★)</b>: Action Loss → Decoder LoRA(<code>r=4</code>) 실제 역전파(<code>grad=0.142</code>) + KL 증류 → 각도 <b>{eg11['ang_med_vs_a3']}°</b>, 바구니/의자 정지 <b>100%/98%</b>, <b>Open-Vocab {eg11['open_vocab_1113_det_rate']*100:.1f}%</b>!
      </div>
    </div>
  </section>

  <section class="act" id="act4">
    <h2>Act 4. 과적합 정밀 감사 &amp; 2026 최신 SOTA VLA와의 위치 비교</h2>
    <p class="lead">300에피소드 소규모 학습에서 발생할 수 있는 <b>시각 외형 과적합</b>을 반사실 실험(동일 BBox에서 이미지 특징만 무작위 셔플)으로 검증한 결과, <code>a3-ag23</code>과 <code>EG3-11</code> 모두 <b>시각 셔플 뒤집힘 0.0% · 바구니-의자 정지 격차 0~3%p</b>로 과적합이 완전히 해소되었다.</p>
    <table class="main-tbl">
      <thead><tr><th>모델</th><th>출발 격차 (바구니 vs 의자)</th><th>정지 격차 (바구니 vs 의자)</th><th>시각 셔플 민감도 (각도 / 뒤집힘)</th><th>기하 반응성 (S_cx / S_area)</th><th>감사 판정</th></tr></thead>
      <tbody>{"".join(audit_rows)}</tbody>
    </table>
    <table class="main-tbl">
      <thead><tr><th>구분</th><th>2026 SOTA (π0 · GR00T N1 · OpenVLA-OFT · SpatialVLA)</th><th>우리 EGVLA2 (MOD-DR-Safe)</th><th>우리 EGVLA3 (EG3-11 SoftLoc-Joint)</th></tr></thead>
      <tbody>
        <tr><td><b>데이터 / VLM</b></td><td>수십만~110만 ep / 3B~7B 대형 VLM</td><td>300 ep / 0.23B Florence-2 (동결)</td><td>300 ep / 0.23B (Decoder LoRA 295k + KL)</td></tr>
        <tr><td><b>공간 추론 경계</b></td><td>Hidden State 직접 전달(π0) 또는 텍스트 CoT(ECoT)</td><td><code>beam=1 fp16</code> 문자열 파싱 (<code>0.54°</code>, 285ms)</td><td><b>미분가능 Soft-Loc 기댓값</b> (파싱 0%, 297ms)</td></tr>
        <tr><td><b>저데이터 핵심 장치</b></td><td>대규모 다물체 사전학습으로 자연 해소</td><td>Area-Grounded 라벨 + 50% Feature Swap</td><td>Soft-Loc 기하 병목 + Action→LoRA 역전파 + KL 증류</td></tr>
      </tbody>
    </table>
  </section>

  <section class="act" id="act5">
    <h2>Act 5. 완료 현황 &amp; 실기 로드맵 (Next Steps)</h2>
    <div class="box" style="border-left:5px solid var(--ok)">
      <b>1. 현재까지 100% 완료 (오프라인 1,458프레임 + GPU 종단 서빙 실측):</b><br>
      • <code>MOD-A3-AG</code>(<code>A3C_AG23_s0.pt</code>), <code>MOD-DR-Safe</code>(285ms · S2 감속 · YahBoom 어댑터), <code>EG3-11</code>(297ms · 2.90° · Open-Vocab 99.8%) 모두 검증 완료 및 <code>soda</code> 인박스 전달(<code>SHA256 9/9 OK</code>).<br><br>
      <b>2. 남은 실기 검증 (로봇 하드웨어 일정):</b><br>
      • <b>Step 1 (EGVLA2)</b>: <code>soda</code>에서 <code>V4/V4R</code> 언어 게이트 테스트 → <code>A3C_AG23_s0</code> + <code>MOD-DR-Safe</code> 실기 정지 확정 → <b>YahBoom 카 이식</b>.<br>
      • <b>Step 2 (EGVLA3)</b>: 동일 서버에서 <code>VLA_EGVLA2_DECODE_MODE=softloc_fp16</code> 스위치로 <code>EG3-11</code> 파일럿 실기 15세션 검증 후 논문 표 완성.
    </div>
  </section>
</main>
</body>
</html>
"""
    OUT_HTML.write_text(doc, encoding="utf-8")
    print(f"Generated {OUT_HTML} ({OUT_HTML.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

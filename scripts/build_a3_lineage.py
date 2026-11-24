#!/usr/bin/env python3
"""A3 계열(a3 -> a3-s2 -> a3-ag23 -> MOD-DR-Fast/Safe) 전용 상세 페이지 생성기.

출력:
  - docs/v5/a3_lineage.html
"""
from __future__ import annotations

import datetime
import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FM = ROOT / "docs/v5/egvla2_fm"
OUT = ROOT / "docs/v5/a3_lineage.html"


def _a3_scene_svg(sc: dict) -> str:
    """실기 촬영 사진 위 F2 BBox 및 A3 3세대(a3, a3-s2, a3-ag23) 예측 오버레이."""
    W, H = 416, 234
    elems = []
    bb = sc.get("bbox")
    if bb and bb[3] >= 0.3:
        side = math.sqrt(max(0.002, float(bb[2])))
        bw = min(0.92, side * 1.15) * W
        bh = min(0.92, side / 1.15) * H
        x = max(2.0, min(W - bw - 2.0, float(bb[0]) * W - bw / 2.0))
        y = max(2.0, min(H - bh - 2.0, float(bb[1]) * H - bh / 2.0))
        elems.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="none" stroke="#38bdf8" stroke-width="2.2"/>')
        elems.append(f'<text x="{x+3:.1f}" y="{max(12.0, y-4.0):.1f}" fill="#38bdf8" font-size="9.5" font-weight="700">F2 BBox (cx={bb[0]:.2f}, a={bb[2]:.2f})</text>')

    ms = sc.get("models", {})
    arrows = [
        ("MOD-A3", "a3(현배포)", "#94a3b8", -12),
        ("MOD-A3-s2", "a3-s2(11/16)", "#fb923c", 8),
        ("MOD-A3-AG", "a3-ag23(완성형)", "#4ade80", 0),
    ]
    ox, oy, scale = 208.0, 214.0, 112.0
    for key, tag, col, yoff in arrows:
        m = ms.get(key)
        if not m:
            continue
        lx, ly = float(m["cmd"][0]), float(m["cmd"][1])
        if m["stop"]:
            r = 6 + abs(yoff) * 0.25
            elems.append(f'<circle cx="{ox:.1f}" cy="{oy:.1f}" r="{r:.1f}" fill="none" stroke="{col}" stroke-width="2.4"/>')
        else:
            tx = max(16.0, min(W - 64.0, ox - ly * scale))
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
    audit = json.loads((FM / "overfitting_audit.json").read_text(encoding="utf-8"))

    b3_ms = beam["beam3_fp32 (현행 기준)"]["latency_med_ms"]
    b1_ms = beam["beam1_fp16 (Track B)"]["latency_med_ms"]
    b1_sp = beam["beam1_fp16 (Track B)"]["speedup_vs_beam3"]

    takeaways = {
        "sc1_chair_center": "<b>[출발 기로]</b> 의자 정중앙: <code>a3</code>는 768-d 시각 지름길 때문에 <b>정지(실기 0/5)</b>. <code>a3-s2</code>와 <code>a3-ag23</code>은 정상 직진 출발.",
        "sc2_basket_close": "<b>[바구니 초근접]</b> 면적 0.31: <code>a3-ag23</code>은 <b>정확히 STOP</b>(90%)하지만, <code>a3-s2</code>는 계속 전진(11/16 접촉 6건의 원인).",
        "sc3_chair_close": "<b>[의자 초근접]</b> 면적 0.33: <code>a3-ag23</code>은 미학습 의자에서도 면적 기하 정합(<code>area≥0.23</code>)으로 <b>정확히 STOP</b>(93%).",
        "sc4_desk_center": "<b>[책상 정중앙]</b> <code>a3</code>는 비출발(STOP)로 멈추지만, <code>a3-s2</code>와 <code>a3-ag23</code>은 정상 직진 출발.",
        "sc5_side_steer": "<b>[측면 조향]</b> 바구니 좌측(cx=0.14): <code>a3</code>·<code>a3-s2</code>·<code>a3-ag23</code> 모두 좌측으로 일관되게 조향(T7 <code>0.54°</code>).",
        "sc6_pot_center": "<b>[화분 정중앙]</b> 미학습 화분: <code>a3-ag23</code>이 바구니와 동일하게 안정적 직진 출발.",
    }

    cards_html = []
    for sc in fam["scenes"]:
        sid = sc["sid"]
        shot_dt = f"{sid[:4]}-{sid[4:6]}-{sid[6:8]} {sid[9:11]}:{sid[11:13]}:{sid[13:15]}" if len(sid) >= 15 else sid
        ms = sc["models"]

        def row_m(k: str, lbl: str, col: str) -> str:
            m = ms.get(k, {})
            lx, ly = float(m.get("cmd", [0, 0])[0]), float(m.get("cmd", [0, 0])[1])
            st = '<span class="bd b-bad">STOP</span>' if m.get("stop") else '<span class="bd b-ok">GO</span>'
            return f'<tr><td style="color:{col};font-weight:700">{lbl}</td><td><code>[{lx:+.3f}, {ly:+.3f}]</code></td><td>{st}</td></tr>'

        bb = sc.get("bbox")
        cards_html.append(f"""
      <div class="sc-card">
        <div class="sc-top">
          <b>{html.escape(sc['tab'])}</b>
          <span>📸 실기 촬영 <b>{html.escape(shot_dt)}</b> · #{sc.get('fi', '')}</span>
        </div>
        {_a3_scene_svg(sc)}
        <div class="sc-ins">지시어: <code>{html.escape(sc['ins'])}</code> · F2 BBox: <code>cx={bb[0]:.2f}, area={bb[2]:.2f}</code></div>
        <table class="mini-tbl">
          {row_m("MOD-A3", "EG2-9 (a3)", "#94a3b8")}
          {row_m("MOD-A3-s2", "EG2-14 (a3-s2)", "#fb923c")}
          {row_m("MOD-A3-AG", "EG2-16 (a3-ag23)", "#4ade80")}
        </table>
        <div class="sc-note">{takeaways.get(sc['id'], '')}</div>
      </div>""")

    audit_rows = []
    for mname, r in audit.items():
        if not any(tag in mname for tag in ("A3", "a3")):
            continue
        verdict = r["verdict"]
        vc = "b-ok" if "없음" in verdict or "완료" in verdict else ("b-bad" if "과적합" in verdict else "b-warn")
        audit_rows.append(f"""
        <tr>
          <td><b>{html.escape(mname)}</b></td>
          <td>{r['basket_center_go']:.1f}% vs {r['chair_center_go']:.1f}%<br><span style="color:#94a3b8;font-size:.74rem">격차 {r['unseen_go_gap_pp']:.1f}%p</span></td>
          <td>{r['basket_ge25_stop']:.1f}% vs {r['chair_ge25_stop']:.1f}%<br><span style="color:#94a3b8;font-size:.74rem">격차 {r['unseen_stop_gap_pp']:.1f}%p</span></td>
          <td>{r['vis_swap_ang_deg']:.1f}° / <b>{r['vis_swap_stop_flip_pct']:.1f}%</b></td>
          <td><code>{r['S_cx_deg']:+.1f}°</code> / <code>{r['S_area_pp']:+.1f}%p</code></td>
          <td><span class="bd {vc}">{html.escape(verdict)}</span></td>
        </tr>""")

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M KST")
    doc = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>A3 계열 진화 계보: a3 → a3-s2 → a3-ag23 → MOD-DR-Fast/Safe</title>
<style>
:root{{--bg:#080e1c;--card:#10192e;--fg:#e2e8f0;--mut:#94a3b8;--line:#223354;--acc:#38bdf8;--ok:#4ade80;--warn:#fbbf24;--bad:#f87171;--purp:#c084fc}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,'Noto Sans KR',sans-serif}}
main{{max-width:1320px;margin:0 auto;padding:22px 18px 80px}}
.top-nav{{position:sticky;top:0;z-index:30;background:rgba(8,14,28,.94);backdrop-filter:blur(8px);border-bottom:1px solid var(--line);padding:9px 18px;margin:-22px -18px 18px;display:flex;flex-wrap:wrap;gap:8px;align-items:center;font-size:.8rem}}
.top-nav a{{border:1px solid var(--line);background:var(--card);color:var(--acc);padding:4px 10px;border-radius:99px;text-decoration:none;font-weight:700}}
.hero{{background:linear-gradient(135deg,#0f1a33,#132238,#0c1e2e);border:1px solid var(--line);border-radius:16px;padding:22px 24px;margin-bottom:20px}}
.hero h1{{margin:0 0 6px;font-size:1.58rem;color:#fff}}
.hero p{{margin:0;color:#cbd5e1;font-size:.92rem}}
.kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px;margin-top:14px}}
.kpi{{background:rgba(12,20,38,.85);border:1px solid var(--line);border-radius:11px;padding:10px 13px}}
.kpi .t{{font-size:.72rem;font-weight:800;color:var(--acc);text-transform:uppercase}}
.kpi .v{{font-size:1.22rem;font-weight:800;color:#fff;margin:2px 0}}
.kpi .d{{font-size:.77rem;color:var(--mut)}}
section.act{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px 20px;margin:18px 0;scroll-margin-top:58px}}
section.act h2{{margin:0 0 8px;font-size:1.2rem;color:#fcd34d;display:flex;align-items:center;gap:8px}}
.lead{{font-size:.88rem;color:#cbd5e1;margin:0 0 12px}}
code{{background:#1b2944;padding:1px 6px;border-radius:4px;font-size:.82em;color:#f1f5f9}}
.bd{{display:inline-block;padding:2px 8px;border-radius:99px;font-size:.72rem;font-weight:700;white-space:nowrap}}
.b-ok{{background:#14532d;color:#86efac}}.b-bad{{background:#450a0a;color:#fca5a5}}.b-warn{{background:#451a03;color:#fcd34d}}.b-blue{{background:#0c2d48;color:#7dd3fc}}.b-purp{{background:#2e1065;color:#e9d5ff}}
.flow-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin-top:10px}}
.flow-card{{background:#0b1324;border:1px solid var(--line);border-radius:12px;padding:13px;font-size:.81rem;line-height:1.58}}
.flow-card h3{{margin:0 0 6px;font-size:.92rem}}
table.main-tbl{{width:100%;border-collapse:collapse;font-size:.82rem;background:#0b1324;border-radius:10px;overflow:hidden;margin:10px 0}}
table.main-tbl th,table.main-tbl td{{border:1px solid var(--line);padding:7px 9px;text-align:left;vertical-align:top}}
table.main-tbl th{{background:#16233f;color:#93c5fd}}
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
.sc-note{{font-size:.78rem;color:#e2e8f0;border-left:3px solid var(--acc);padding-left:8px;line-height:1.5}}
.note-box{{background:#0b1324;border:1px solid var(--line);border-left:4px solid var(--acc);border-radius:10px;padding:12px 14px;font-size:.83rem;line-height:1.65;margin-top:12px}}
</style>
</head>
<body>
<main>
  <nav class="top-nav">
    <a href="#sec1">1. 진화 계보 &amp; ag23 의도</a>
    <a href="#sec2">2. 세대별 정밀 비교</a>
    <a href="#sec3">3. 실기 사진 위 3세대 비교</a>
    <a href="#sec4">4. 과적합 감사 &amp; 속도</a>
    <a href="report_261126_egvla.html" style="border-color:#fbbf24;color:#fbbf24">← 261126 보고서 (EGVLA2)</a>
    <a href="report_261126_egvla_with_egvla3.html" style="border-color:#c084fc;color:#e9d5ff">261126 확장판 (EGVLA3)</a>
    <a href="model_registry.html">모델 레지스트리</a>
    <a href="research_story_egvla2.html#ch83">리서치 히스토리 (#ch83)</a>
  </nav>

  <header class="hero">
    <h1>🧬 A3 계열 진화 계보: <code>a3</code> → <code>a3-s2</code> → <code>a3-ag23</code> → <code>MOD-DR-Fast/Safe</code></h1>
    <p>300에피소드 바구니 데이터(<code>PiZeroFM(12,768)</code>)에서 출발해 <b>중앙 비출발 문제</b>와 <b>근접 미정지(접촉 6건)</b>를 단계적으로 해결하고, <b><code>beam=1 fp16</code>(285ms)</b> 및 YahBoom 이식으로 EGVLA2를 완결한 전체 계보. · 생성 {now_str}</p>
    <div class="kpi-grid">
      <div class="kpi"><div class="t">① EG2-9 (a3) · 현 배포</div><div class="v">455 / 455</div><div class="d">바구니 실기(10/29~11/04) · 11/13 의자 중앙 0/5 비출발</div></div>
      <div class="kpi"><div class="t">② EG2-14 (a3-s2) · 시드 2</div><div class="v">중앙 5/5 (100%)</div><div class="d">11/16 실기 45/45 출발 복구 · 근접 정지 55%/28%(접촉 6건)</div></div>
      <div class="kpi"><div class="t">③ EG2-16 (a3-ag23) · 정지·일반화</div><div class="v">정지 90%/93% · 0.54°</div><div class="d">area≥0.23 + 50% Feature Swap · 시각 셔플 뒤집힘 0.0%</div></div>
      <div class="kpi"><div class="t">④ EG2-17 (MOD-DR-Fast/Safe)</div><div class="v">{b3_ms:.1f}ms → 285.3ms</div><div class="d">beam=1 fp16({b1_sp:.2f}x) + S2 감속 + 스파이크 필터(EGVLA2 엔딩)</div></div>
    </div>
  </header>

  <section class="act" id="sec1">
    <h2>1. 왜 이렇게 진화했는가? (Causal Flow &amp; <code>ag23</code> 명명 의도)</h2>
    <p class="lead">각 세대는 이전 실기 주행에서 드러난 <b>단 하나의 명확한 병목</b>을 겨냥해 설계되었다.</p>
    <div class="flow-grid">
      <div class="flow-card" style="border-top:3px solid #93c5fd">
        <h3 style="color:#93c5fd">① EG2-9 (<code>a3</code>) — 기준선</h3>
        <span class="bd b-blue">실기 2026-10-29 ~ 11-13</span>
        <p><b>설계</b>: Florence-2 BBox 중심(<code>cx</code>)과 조향(<code>linear_y</code>)을 기하 서보로 정합(시드 0).</p>
        <p><b>성과</b>: 학습 물체인 바구니 실기 <b>455/455</b> 안정 추종.</p>
        <p><b>발견된 문제</b>: 11/13 의자 중앙에서 <b>0/5 비출발</b>(768-d 시각 지름길 과적합, 시각 셔플 뒤집힘 <b>27.4%</b>).</p>
      </div>
      <div class="flow-card" style="border-top:3px solid #fcd34d">
        <h3 style="color:#fcd34d">② EG2-14 (<code>a3-s2</code>) — 시드 전환</h3>
        <span class="bd b-blue">실기 2026-11-16 (45/45)</span>
        <p><b>변경</b>: <code>a3</code>와 동일한 아키텍처·레시피에서 <b>랜덤 시드만 0→2</b>로 변경.</p>
        <p><b>성과</b>: 11/16 실기에서 의자·책상 중앙 출발 <b>5/5 (100%)</b> 즉시 복구.</p>
        <p><b>발견된 문제</b>: 정지 라벨이 기하 면적과 묶이지 않아 근접 정지 <b>55%/28%</b>(<code>area 0.25~0.40</code> 12~22%), 실기 <b>접촉 6건</b>.</p>
      </div>
      <div class="flow-card" style="border-top:3px solid #4ade80">
        <h3 style="color:#4ade80">③ EG2-16 (<code>a3-ag23</code>) — 기하·시각 완결</h3>
        <span class="bd b-warn">오프라인/GPU 검증</span>
        <p><b>변경</b>: <b>정지 면적 <code>0.23</code> 기하 정합(<code>area≥0.23</code>)</b> + <b>시각 특징 50% 교체</b>.</p>
        <p><b>성과</b>: 중앙 <b>100%</b> · 근접 정지 <b>90%/93%</b> · T7 각도 <b>0.54°</b>(좌우 98.7%) · 시각 셔플 뒤집힘 <b>0.0%</b> · <code>S_area=+100.0%p</code>.</p>
        <p><b>남은 숙제</b>: <code>beam=3 fp32</code>(<b>{b3_ms:.1f}ms</b>) 지연 병목 및 실기 주행 미검증.</p>
      </div>
      <div class="flow-card" style="border-top:3px solid #c084fc">
        <h3 style="color:#c084fc">④ EG2-17 (<code>MOD-DR-Fast/Safe</code>) — 서빙 완결</h3>
        <span class="bd b-purp">GPU 종단 서빙</span>
        <p><b>변경</b>: <code>a3-ag23</code> 위에 <b><code>beam=1 fp16</code></b> + S2 연속 감속(<code>0.14→0.25</code>) + 스파이크 필터(<code>2.5x</code>) 결합.</p>
        <p><b>성과</b>: 종단 지연 <b>{b3_ms:.1f}ms → 285.3ms ({b1_sp:.2f}x)</b>, Serbot → YahBoom 카 이식(<b>EGVLA2 엔딩</b>).</p>
        <p><b>이후 방향</b>: 오픈보캡 실기·멈춤 문구 E2E는 <b>EGVLA3(<code>EG3-11</code>)</b>로 분리 진행.</p>
      </div>
    </div>

    <div class="note-box">
      <b>💡 이름 의도 — 왜 <code>a3-ag23</code> 인가?</b><br>
      • <b><code>a3</code></b>: <code>A</code>rchitecture 3 (<code>PiZeroFM(12,768)</code> 기반 모듈러 Flow-Matching 계열) · <b><code>s2</code></b>: Seed 2<br>
      • <b><code>ag23</code></b>: <b><u>A</u>rea-<u>G</u>rounded 0.<u>23</u></b> — 기존 <code>a3</code>가 조향(<code>cx</code>)만 기하에 정렬해 근접 시 충돌하던 문제를 풀기 위해, <b>정지(STOP) 기준을 바운딩박스 면적 임계값 <code>area ≥ 0.23</code>에 기하학적으로 정렬(Area-Grounded)</b>하고 시각 특징 50% 교체로 배경 편향을 끊은 모델이라는 의미다.
    </div>
  </section>

  <section class="act" id="sec2">
    <h2>2. 세대별 구조·학습·성능 정밀 비교표</h2>
    <table class="main-tbl">
      <thead><tr><th>세대 (ID / 별칭)</th><th>검증 구분</th><th>핵심 변경점</th><th>정량 지표 (출발 / 정지 / 조향 / 속도)</th><th>해결점 vs 남은 과제</th></tr></thead>
      <tbody>
        <tr>
          <td><b><code>EG2-9 (a3)</code></b><br><span style="color:#94a3b8;font-size:.75rem">MOD-A3</span></td>
          <td><span class="bd b-blue">실기 완료</span><br>2026-10-29~11-13</td>
          <td>조향(<code>cx</code>)만 기하 서보 정합, 시드 0, <code>beam=3 fp32</code></td>
          <td>• 바구니 실기 <b>455/455</b><br>• 11/13 의자 중앙 <b>0/5</b><br>• 11/16 근접 정지 80%/58%<br>• 지연 <b>{b3_ms:.1f}ms</b></td>
          <td>바구니 주행 안정적이나 시각 지름길(뒤집힘 27.4%)로 낯선 물체 중앙 비출발.</td>
        </tr>
        <tr>
          <td><b><code>EG2-14 (a3-s2)</code></b><br><span style="color:#94a3b8;font-size:.75rem">MOD-A3-s2</span></td>
          <td><span class="bd b-blue">실기 완료</span><br>2026-11-16 (45/45)</td>
          <td><code>a3</code>와 동일 레시피 · <b>시드 0→2</b></td>
          <td>• 의자·책상 중앙 <b>5/5 (100%)</b><br>• 11/16 근접 정지 <b>55%/28%</b><br>• T7 각도 <code>0.70°</code></td>
          <td>중앙 출발 해결, 그러나 근접 정지 약화로 실기 <b>접촉 6건</b> 발생.</td>
        </tr>
        <tr style="background:#0c231a">
          <td><b><code>EG2-16 (a3-ag23)</code></b><br><span style="color:#6ee7b7;font-size:.75rem">MOD-A3-AG</span></td>
          <td><span class="bd b-warn">오프라인/GPU</span><br>2026-11-21</td>
          <td><b>정지 면적 <code>0.23</code> 기하 정합</b> + <b>시각 특징 50% 교체</b></td>
          <td>• 중앙 <b>100%</b> · 근접 정지 <b>90%/93%</b><br>• T7 <b><code>0.54°</code></b> (좌우 98.7%)<br>• 셔플 뒤집힘 <b>0.0%</b></td>
          <td>실기 주행 미검증 + <code>beam=3</code> 지연({b3_ms:.1f}ms) → <code>MOD-DR-Fast/Safe</code>로 해결.</td>
        </tr>
        <tr style="background:#17102b">
          <td><b><code>EG2-17 (MOD-DR-Fast/Safe)</code></b><br><span style="color:#c4b5fd;font-size:.75rem">EGVLA2 최종 서빙</span></td>
          <td><span class="bd b-purp">GPU 종단 서빙</span><br>2026-11-24</td>
          <td><code>a3-ag23</code> + <b><code>beam=1 fp16</code></b> + S2 감속(<code>0.14→0.25</code>) + 스파이크 필터(<code>2.5x</code>)</td>
          <td>• 종단 지연 <b>285.3ms</b> ({b1_sp:.2f}x)<br>• <code>beam=3</code> 일치율 <b>99.7%</b> (<code>cx MAE 0.0016</code>)</td>
          <td><code>soda</code> 인박스 전달 완료(SHA256 9/9 OK). Serbot→YahBoom 실기 이식 대기.</td>
        </tr>
      </tbody>
    </table>
  </section>

  <section class="act" id="sec3">
    <h2>3. 실제 실기 촬영 사진(날짜·시각) 위 A3 3세대 행동 비교</h2>
    <p class="lead">범례: <b style="color:#94a3b8">회색=a3</b> · <b style="color:#fb923c">주황=a3-s2</b> · <b style="color:#4ade80">초록=a3-ag23</b> · <b>원(○)=STOP</b> (각 사진은 실기 날짜·시각이 기록된 프레임 위 오프라인 추론 재생)</p>
    <div class="sc-grid">{"".join(cards_html)}</div>
  </section>

  <section class="act" id="sec4">
    <h2>4. 과적합 정밀 감사 (A3 계열) &amp; 디코더 추론 속도 개선</h2>
    <table class="main-tbl">
      <thead><tr><th>모델</th><th>출발 격차 (바구니 vs 의자)</th><th>정지 격차 (바구니 vs 의자)</th><th>시각 셔플 (각도 / 뒤집힘)</th><th>기하 반응성 (S_cx / S_area)</th><th>감사 판정</th></tr></thead>
      <tbody>{"".join(audit_rows)}</tbody>
    </table>
    <div class="note-box">
      <b>⏱️ 추론 속도 병목 해소 (<code>MOD-DR-Fast/Safe</code>)</b>:
      <code>beam=3 fp32</code> (<b>{b3_ms:.1f}ms</b>) → <code>beam=1 fp16</code> (순수 디코더 <b>{b1_ms:.1f}ms</b>, 종단 서빙 <b>285.3ms</b>, <b>{b1_sp:.2f}x</b> 가속), <code>beam=3</code> 대비 중심점 오차 <code>cx MAE = 0.0016</code> (99.7% 일치).
    </div>
  </section>
</main>
</body>
</html>
"""
    OUT.write_text(doc, encoding="utf-8")
    print(f"Generated {OUT} ({OUT.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

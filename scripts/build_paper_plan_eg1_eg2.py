#!/usr/bin/env python3
"""2026-11-26 교수님 미팅 확정: EGVLA1 vs EGVLA2 비교 논문 실기 실험 프로토콜 & 계획 전용 페이지 생성기.

출력:
  - docs/v5/paper_egvla1_vs_egvla2_plan.html
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FM = ROOT / "docs/v5/egvla2_fm"
OUT = ROOT / "docs/v5/paper_egvla1_vs_egvla2_plan.html"


def main() -> None:
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M KST")
    r1125 = json.loads((FM / "real_1125_analysis.json").read_text(encoding="utf-8"))
    sbc = json.loads((FM / "step_bc_results.json").read_text(encoding="utf-8"))
    sb = sbc["Step B (A3C_Y2_s0)"]

    doc = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>EGVLA1 vs EGVLA2 비교 논문 — 공통 실기 실험 프로토콜 &amp; 실행 계획</title>
<style>
:root{{--bg:#080e1c;--card:#10192e;--head:#16233f;--fg:#e2e8f0;--mut:#94a3b8;--line:#223354;--acc:#38bdf8;--ok:#4ade80;--warn:#fbbf24;--bad:#f87171;--purp:#c084fc}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,'Noto Sans KR',sans-serif}}
main{{max-width:1280px;margin:0 auto;padding:22px 18px 80px}}
.top-nav{{position:sticky;top:0;z-index:30;background:rgba(8,14,28,.94);backdrop-filter:blur(8px);border-bottom:1px solid var(--line);padding:9px 18px;margin:-22px -18px 18px;display:flex;flex-wrap:wrap;gap:8px;align-items:center;font-size:.8rem}}
.top-nav a{{border:1px solid var(--line);background:var(--card);color:var(--acc);padding:4px 10px;border-radius:99px;text-decoration:none;font-weight:700}}
.hero{{background:linear-gradient(135deg,#0f1a33,#132238,#0c1e2e);border:1px solid var(--line);border-radius:16px;padding:22px 24px;margin-bottom:20px}}
.hero h1{{margin:0 0 6px;font-size:1.56rem;color:#fff}}
.hero p{{margin:0;color:#cbd5e1;font-size:.91rem}}
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
table.main-tbl{{width:100%;border-collapse:collapse;font-size:.82rem;background:#0b1324;border-radius:10px;overflow:hidden;margin:10px 0}}
table.main-tbl th,table.main-tbl td{{border:1px solid var(--line);padding:7px 9px;text-align:left;vertical-align:top}}
table.main-tbl th{{background:var(--head);color:#93c5fd;font-size:.78rem}}
.box{{background:#0b1324;border:1px solid var(--line);border-left:4px solid var(--acc);border-radius:10px;padding:12px 14px;font-size:.83rem;line-height:1.65;margin-top:10px}}
.g2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}}
</style>
</head>
<body>
<main>
  <nav class="top-nav">
    <span style="font-weight:800;color:#fff">📝 논문 실기 계획 (EG1 vs EG2)</span>
    <a href="#sec1">1. 미팅 확정 방향</a>
    <a href="#sec2">2. 공통 경로 매트릭스</a>
    <a href="#sec3">3. 선행 정지거리 후보</a>
    <a href="#sec4">4. 논문 지표·체크리스트</a>
    <a href="report_261126_egvla.html" style="border-color:#fbbf24;color:#fbbf24">← 261126 보고서</a>
    <a href="a3_lineage.html" style="border-color:#4ade80;color:#86efac">🧬 A3 전용</a>
    <a href="research_story_egvla2.html#ch83">리서치 히스토리</a>
  </nav>

  <header class="hero">
    <h1>📝 EGVLA1 vs EGVLA2 비교 논문 — 공통 실기 실험 프로토콜 &amp; 실행 계획</h1>
    <p>2026-11-26 교수님 구조 설명 미팅에서 확정된 <b>동일 출발(pos1~3)·동일 도착(G6~8) 기반 모델 비교 논문</b> 실험 설계 및 단계별 실행 로드맵 · 생성 {now_str}</p>
    <div class="kpi-grid">
      <div class="kpi"><div class="t">논문 핵심 주제</div><div class="v">EGVLA1 vs EGVLA2</div><div class="d">동일 출발·도착 조건에서 기존 대비 일반화·정지·속도 비교</div></div>
      <div class="kpi"><div class="t">공통 경로 매트릭스</div><div class="v">출발 3 × 도착 3 = 9경로</div><div class="d">출발 <code>pos1, pos2, pos3</code> → 랜덤 도착 <code>G6, G7, G8</code></div></div>
      <div class="kpi"><div class="t">총 평가 조건 수</div><div class="v">EG1 9조건 · EG2 27조건</div><div class="d">EG1: 바구니(1종) · EG2: 바구니·의자·책상(3종)</div></div>
      <div class="kpi"><div class="t">본 실기 진입 조건</div><div class="v">AG28 / A3C_Y2 확인</div><div class="d">너무 멀지 않은 정지 거리 확보 직후 논문용 본 실기 착수</div></div>
    </div>
  </header>

  <!-- 1. 교수님 미팅 확정 방향 & 전체 워크플로우 -->
  <section class="act" id="sec1">
    <h2>1. 2026-11-26 미팅 확정 방향 &amp; 논문 집필 워크플로우</h2>
    <div class="box" style="border-left-color:#4ade80">
      <b>🗣️ 교수님 피드백 요약 (2026-11-26):</b><br>
      • 현재 구조에서 <b>“정지 거리만 너무 멀지 않게 적정 수준으로 잡히면”</b>, 기존 <b><code>EGVLA1</code></b>과 현재 <b><code>EGVLA2</code></b>를 정면 비교하는 논문으로 작성 진행 확정.<br>
      • 두 모델이 <b>공통 출발 지점(<code>pos1, pos2, pos3</code>)</b>과 <b>공통 도착 지점(<code>G6, G7, G8</code> 랜덤 3곳)</b>에서 공정하게 비교되도록 프로토콜 통일.<br>
      • 논문에 최종 수록할 상세 정량 지표(수치 항목)는 추후 교수님께서 별도 지정 예정.
    </div>

    <svg viewBox="0 0 920 135" style="width:100%;height:auto;display:block;background:#070d19;border:1px solid #1e293b;border-radius:10px;margin-top:12px">
      <defs><marker id="pa" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,1 L6,3.5 L0,6 Z" fill="#64748b"/></marker></defs>
      <!-- Step 1 -->
      <rect x="14" y="18" width="160" height="98" rx="9" fill="#1c1507" stroke="#eab308" stroke-width="1.8"/>
      <rect x="24" y="26" width="96" height="17" rx="8" fill="#451a03"/><text x="72" y="38" text-anchor="middle" fill="#fcd34d" font-size="8.8" font-weight="700">1단계 · 실기 대기</text>
      <text x="94" y="59" text-anchor="middle" fill="#fde047" font-size="11.2" font-weight="800">AG28 정지 확인</text>
      <text x="94" y="77" text-anchor="middle" fill="#cbd5e1" font-size="9.2">의자 정지거리 적정화</text>
      <text x="94" y="94" text-anchor="middle" fill="#94a3b8" font-size="8.8">(백업: A3C_Y2_s0)</text>
      <line x1="174" y1="67" x2="196" y2="67" stroke="#64748b" stroke-width="2" marker-end="url(#pa)"/>

      <!-- Step 2 -->
      <rect x="198" y="18" width="160" height="98" rx="9" fill="#0f172a" stroke="#38bdf8" stroke-width="1.6"/>
      <rect x="208" y="26" width="84" height="17" rx="8" fill="#0c2d48"/><text x="250" y="38" text-anchor="middle" fill="#7dd3fc" font-size="8.8" font-weight="700">2단계 · 점검</text>
      <text x="278" y="59" text-anchor="middle" fill="#7dd3fc" font-size="11.2" font-weight="800">EGVLA1 구동 점검</text>
      <text x="278" y="77" text-anchor="middle" fill="#cbd5e1" font-size="9.2">soda 프로필·로깅 확인</text>
      <text x="278" y="94" text-anchor="middle" fill="#94a3b8" font-size="8.8">공통 포맷(h5/json)</text>
      <line x1="358" y1="67" x2="380" y2="67" stroke="#64748b" stroke-width="2" marker-end="url(#pa)"/>

      <!-- Step 3 -->
      <rect x="382" y="18" width="170" height="98" rx="9" fill="#06281e" stroke="#4ade80" stroke-width="1.8"/>
      <rect x="392" y="26" width="94" height="17" rx="8" fill="#14532d"/><text x="439" y="38" text-anchor="middle" fill="#86efac" font-size="8.8" font-weight="700">3단계 · 본 실기</text>
      <text x="467" y="59" text-anchor="middle" fill="#4ade80" font-size="11.2" font-weight="800">공통 9경로 실기</text>
      <text x="467" y="77" text-anchor="middle" fill="#cbd5e1" font-size="9.2">EG1(9조건) vs EG2(27조건)</text>
      <text x="467" y="94" text-anchor="middle" fill="#86efac" font-size="8.8">pos1~3 → G6~8 공통</text>
      <line x1="552" y1="67" x2="574" y2="67" stroke="#64748b" stroke-width="2" marker-end="url(#pa)"/>

      <!-- Step 4 -->
      <rect x="576" y="18" width="156" height="98" rx="9" fill="#17102b" stroke="#a855f7" stroke-width="1.6"/>
      <rect x="586" y="26" width="96" height="17" rx="8" fill="#2e1065"/><text x="634" y="38" text-anchor="middle" fill="#e9d5ff" font-size="8.8" font-weight="700">4단계 · 수치 확정</text>
      <text x="654" y="59" text-anchor="middle" fill="#d8b4fe" font-size="11.2" font-weight="800">교수님 지정 지표</text>
      <text x="654" y="77" text-anchor="middle" fill="#cbd5e1" font-size="9.2">상세 수치 자동 집계</text>
      <text x="654" y="94" text-anchor="middle" fill="#94a3b8" font-size="8.8">표·히트맵·통계 생성</text>
      <line x1="732" y1="67" x2="754" y2="67" stroke="#64748b" stroke-width="2" marker-end="url(#pa)"/>

      <!-- Step 5 -->
      <rect x="756" y="18" width="150" height="98" rx="9" fill="#0f1f3d" stroke="#60a5fa" stroke-width="1.8"/>
      <rect x="766" y="26" width="84" height="17" rx="8" fill="#172554"/><text x="808" y="38" text-anchor="middle" fill="#93c5fd" font-size="8.8" font-weight="700">5단계 · 집필</text>
      <text x="831" y="59" text-anchor="middle" fill="#93c5fd" font-size="11.2" font-weight="800">논문 작성 완료</text>
      <text x="831" y="77" text-anchor="middle" fill="#cbd5e1" font-size="9.2">EGVLA1 vs EGVLA2</text>
      <text x="831" y="94" text-anchor="middle" fill="#94a3b8" font-size="8.8">정량·정성 비교 투고</text>
    </svg>
  </section>

  <!-- 2. 공통 경로 매트릭스 & EGVLA1 vs EGVLA2 비교 프로토콜 -->
  <section class="act" id="sec2">
    <h2>2. 공통 출발·도착 경로 매트릭스 &amp; 모델별 실험 조건</h2>
    <p class="lead">두 모델 모두 <b>동일한 출발지 3곳(<code>pos1, pos2, pos3</code>)</b>과 <b>동일한 랜덤 도착지 3곳(<code>G6, G7, G8</code>)</b>을 사용하여 공정한 비교 기준을 맞춘다.</p>
    <div class="g2">
      <!-- 좌측: 3x3 경로 SVG 다이어그램 -->
      <div style="background:#0b1324;border:1px solid #334155;border-radius:12px;padding:12px">
        <b style="font-size:.88rem;color:#7dd3fc;display:block;margin-bottom:8px">🗺️ 공통 3×3 경로 배치 도식 (출발 3곳 → 랜덤 도착 3곳)</b>
        <svg viewBox="0 0 460 225" style="width:100%;height:auto;display:block;background:#070d19;border:1px solid #1e293b;border-radius:8px">
          <!-- 도착 지점 G6, G7, G8 (상단) -->
          <rect x="45" y="18" width="90" height="40" rx="8" fill="#14532d" stroke="#4ade80" stroke-width="1.8"/>
          <text x="90" y="36" text-anchor="middle" fill="#fff" font-size="11" font-weight="800">G6 (도착 1)</text>
          <text x="90" y="50" text-anchor="middle" fill="#86efac" font-size="8.8">랜덤 위치 A</text>

          <rect x="185" y="18" width="90" height="40" rx="8" fill="#14532d" stroke="#4ade80" stroke-width="1.8"/>
          <text x="230" y="36" text-anchor="middle" fill="#fff" font-size="11" font-weight="800">G7 (도착 2)</text>
          <text x="230" y="50" text-anchor="middle" fill="#86efac" font-size="8.8">랜덤 위치 B</text>

          <rect x="325" y="18" width="90" height="40" rx="8" fill="#14532d" stroke="#4ade80" stroke-width="1.8"/>
          <text x="370" y="36" text-anchor="middle" fill="#fff" font-size="11" font-weight="800">G8 (도착 3)</text>
          <text x="370" y="50" text-anchor="middle" fill="#86efac" font-size="8.8">랜덤 위치 C</text>

          <!-- 9개 연결선 (pos1~3 -> G6~8) -->
          <g stroke="#38bdf8" stroke-opacity="0.38" stroke-width="1.6" stroke-dasharray="4 3">
            <line x1="90" y1="168" x2="90" y2="58"/><line x1="90" y1="168" x2="230" y2="58"/><line x1="90" y1="168" x2="370" y2="58"/>
            <line x1="230" y1="168" x2="90" y2="58"/><line x1="230" y1="168" x2="230" y2="58"/><line x1="230" y1="168" x2="370" y2="58"/>
            <line x1="370" y1="168" x2="90" y2="58"/><line x1="370" y1="168" x2="230" y2="58"/><line x1="370" y1="168" x2="370" y2="58"/>
          </g>

          <!-- 중앙 라벨 -->
          <rect x="135" y="96" width="190" height="32" rx="8" fill="rgba(15,23,42,0.92)" stroke="#334155"/>
          <text x="230" y="116" text-anchor="middle" fill="#fcd34d" font-size="10.5" font-weight="800">공통 9개 조합 (출발 3 × 도착 3)</text>

          <!-- 출발 지점 pos1, pos2, pos3 (하단) -->
          <rect x="45" y="168" width="90" height="38" rx="8" fill="#0c2d48" stroke="#38bdf8" stroke-width="1.8"/>
          <text x="90" y="191" text-anchor="middle" fill="#e0f2fe" font-size="11" font-weight="800">pos1 (출발 1)</text>

          <rect x="185" y="168" width="90" height="38" rx="8" fill="#0c2d48" stroke="#38bdf8" stroke-width="1.8"/>
          <text x="230" y="191" text-anchor="middle" fill="#e0f2fe" font-size="11" font-weight="800">pos2 (출발 2)</text>

          <rect x="325" y="168" width="90" height="38" rx="8" fill="#0c2d48" stroke="#38bdf8" stroke-width="1.8"/>
          <text x="370" y="191" text-anchor="middle" fill="#e0f2fe" font-size="11" font-weight="800">pos3 (출발 3)</text>
        </svg>
      </div>

      <!-- 우측: 모델별 대조표 -->
      <div>
        <table class="main-tbl" style="margin-top:0">
          <thead><tr><th>구분</th><th><code>EGVLA1</code> (기존 기준선)</th><th><code>EGVLA2</code> (제안 모델 · <code>MOD-DR-Safe</code>)</th></tr></thead>
          <tbody>
            <tr><td><b>공통 출발지</b></td><td><code>pos1, pos2, pos3</code> (3곳)</td><td><code>pos1, pos2, pos3</code> (3곳)</td></tr>
            <tr><td><b>공통 도착지</b></td><td><code>G6, G7, G8</code> (랜덤 3곳)</td><td><code>G6, G7, G8</code> (랜덤 3곳)</td></tr>
            <tr><td><b>지시어 형식</b></td><td><code>"gray basket"</code> 고정</td><td><code>"go to the &lt;객체&gt;"</code></td></tr>
            <tr><td><b>평가 객체</b></td><td><b>바구니 1종</b> (학습 객체)</td><td><b>바구니 · 의자 · 책상 3종</b> (미학습 2종 포함!)</td></tr>
            <tr><td><b>총 조건 수</b></td><td><code>3 × 3 × 1 =</code> <b>9 조건</b></td><td><code>3 × 3 × 3 =</code> <b>27 조건</b></td></tr>
            <tr><td><b>비교 핵심</b></td><td>단일 객체·고정 지시어 기준선</td><td><b>미학습 다물체 일반화 + 무충돌 정지 + 고속화</b></td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </section>

  <!-- 3. 본 실기 진입 전 선행 후보 (AG23 vs AG28 vs A3C_Y2) -->
  <section class="act" id="sec3">
    <h2>3. 선행 과제: “너무 멀지 않은 정지 거리” 확정 후보 비교 (<code>soda</code> 전송 완료)</h2>
    <p class="lead">논문 본 실기(<code>pos1~3 × G6~8</code>)에 올릴 <code>EGVLA2</code> 최종 체크포인트를 확정하기 위한 3개 후보 비교. 3종 모두 <code>soda</code> 서버에 전송되어 즉시 테스트 가능하다.</p>
    <table class="main-tbl">
      <thead><tr><th>후보 체크포인트</th><th>상태</th><th>정지 기준 설계</th><th>핵심 결과 (바구니 vs 의자)</th><th><code>soda</code> 권장 테스트 환경변수</th></tr></thead>
      <tbody>
        <tr>
          <td><b><code>A3C_AG23_s0.pt</code></b></td>
          <td><span class="bd b-ok">11/25 실기 완료</span></td>
          <td><code>area ≥ 0.23</code> (`ramp 0.18`)</td>
          <td>• 실기 <b>{r1125['total']['n']}/{r1125['total']['n']} (100%, 접촉 0건, {r1125['total']['lat_med_ms']}ms)</b><br>• 단, 의자 정지 시 <code>y2=0.82</code>로 다소 멀리 멈춤</td>
          <td><code>beam1_fp16</code>, <code>decel=0.14</code>, <code>near_area=0.25</code></td>
        </tr>
        <tr style="background:#1c1507">
          <td><b><code>A3C_AG28_s0.pt</code> ★</b><br><span style="color:#fcd34d;font-size:.75rem">동환님 1순위 테스트</span></td>
          <td><span class="bd b-warn">학습완료·실기예정</span></td>
          <td><code>area ≥ 0.28</code> (`ramp 0.22`)</td>
          <td>• <code>0.20~0.25</code> 조기 정지 <b>25.8% → 0.0%</b><br>• <code>≥0.30</code> 정지 <b>89.1%</b>, 중앙 <b>100%</b>, 셔플 <b>0.0%</b></td>
          <td><code>beam1_fp16</code>, <code>decel=0.18</code>, <code>near_area=0.30</code>, <code>jump=2.5</code></td>
        </tr>
        <tr style="background:#0c231a">
          <td><b><code>A3C_Y2_s0.pt</code></b><br><span style="color:#6ee7b7;font-size:.75rem">물체크기 무관 백업(2순위)</span></td>
          <td><span class="bd b-warn">학습완료·시뮬통과</span></td>
          <td>바닥선 <code>y2 ≥ 0.95</code> (`ramp 0.86`)</td>
          <td>• 11/25 바구니(<code>y2=1.00</code>) <b>{sb['stop_1125_basket_arrival_pct']}% 정지</b><br>• 11/25 의자(<code>y2=0.82</code>) <b>{sb['stop_1125_chair_arrival_pct']}% 정지</b>(더 가까이 진입!), 셔플 <b>{sb['vis_shuffle_flip_pct']}%</b></td>
          <td><code>beam1_fp16</code>, <code>decel=0.18</code>, <code>near_area=0.35</code> (또는 <code>off</code>), <code>jump=2.5</code></td>
        </tr>
      </tbody>
    </table>
  </section>

  <!-- 4. 논문 수록 예정 정량 지표 슬롯 & 체크리스트 -->
  <section class="act" id="sec4">
    <h2>4. 논문 수록 예정 정량 지표 &amp; 실기 준비 체크리스트</h2>
    <p class="lead">상세 논문 수치는 교수님께서 추후 확정해 주실 예정이며, 아래 지표들이 <code>.h5</code> / <code>session_*.json</code> / <code>episode_log.csv</code>에서 자동 집계될 수 있도록 사전 준비한다.</p>
    <table class="main-tbl">
      <thead><tr><th>지표 후보</th><th>측정 / 추출 방법</th><th>비교 포인트 (<code>EGVLA1</code> vs <code>EGVLA2</code>)</th><th>상태</th></tr></thead>
      <tbody>
        <tr><td><b>주행 성공률 (%)</b></td><td><code>episode_log.csv</code> + 최종 도착 프레임 검증</td><td>바구니 동등 이상 + <b>의자·책상 Zero-Shot 성공률</b></td><td><span class="bd b-blue">자동 집계 준비</span></td></tr>
        <tr><td><b>접촉 · 충돌 횟수</b></td><td>실기 마킹 + <code>.h5 max_area ≥ 0.50 / 0.90</code></td><td><code>MOD-DR-Safe</code> 2중 감속으로 <b>접촉 0건</b> 입증</td><td><span class="bd b-blue">자동 집계 준비</span></td></tr>
        <tr><td><b>도착 후 잔여 거리 (cm)</b></td><td>현장 줄자 측정 기록 (바구니·의자·책상)</td><td><b>교수님 핵심 확인 지표</b> (너무 멀지도, 부딪히지도 않게)</td><td><span class="bd b-warn">현장 측정 필요</span></td></tr>
        <tr><td><b>평균 추론 지연 (ms) / Hz</b></td><td><code>session_*.json</code> <code>avg_latency_ms</code></td><td><code>beam3</code>(~945ms) vs <code>beam1_fp16</code>(~400ms, 2.3x+)</td><td><span class="bd b-blue">자동 집계 준비</span></td></tr>
        <tr><td><b>경로별(9조합) 히트맵</b></td><td><code>start_position</code>(pos1~3) × <code>target_position</code>(G6~8)</td><td>출발·도착 위치 편향 없이 균일한 주행 안정성 입증</td><td><span class="bd b-purp">실기 후 생성</span></td></tr>
      </tbody>
    </table>
    <div class="box">
      <b>✅ 다음 액션 체크리스트:</b><br>
      1. <b>동환님</b>: <code>soda</code>에서 <code>A3C_AG28_s0.pt</code> 정지 거리 확인 (필요 시 <code>A3C_Y2_s0.pt</code> 5분 대조) → 교수님 보고.<br>
      2. <b>사전 점검</b>: 본 실기 전 <code>soda</code>에서 <code>EGVLA1</code> 모델 구동 및 로그 저장 포맷 확인.<br>
      3. <b>본 실기 &amp; 논문 집계</b>: 교수님 상세 수치 확정 후 <code>pos1~3 × G6~8</code> 실기 데이터 일괄 수집 및 논문 표·그림 자동 생성.
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

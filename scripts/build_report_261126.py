#!/usr/bin/env python3
"""2026-11-26 EGVLA 실험보고 발표용 페이지 생성기 (메인: EGVLA2 집중판 / 확장판: EGVLA3 포함).

전개 순서 (발표 최적화):
  1. 핵심 결과 요약 (Result First)
  2. 모델별 상세 설명 & 정량 비교 (a3 · a3-s2 · a3-ag23 · MOD-DR-Fast/Safe · V4/V4R [+ EGVLA3])
  3. 실기(날짜·시각 명시) vs 오프라인 추론 6장면 비교 (실제 카메라 사진 오버레이)
  4. 과적합 정밀 감사 (시각 셔플 & 기하 반응성)
  5. 실기 현황 & 로드맵 (soda Serbot → YahBoom)

출력:
  - docs/v5/report_261126_egvla.html (메인: EGVLA3 제외, EGVLA2 중심)
  - docs/v5/report_261126_egvla_with_egvla3.html (확장판: EGVLA3 포함)
"""
from __future__ import annotations

import datetime
import html
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FM = ROOT / "docs/v5/egvla2_fm"
OUT_MAIN = ROOT / "docs/v5/report_261126_egvla.html"
OUT_FULL = ROOT / "docs/v5/report_261126_egvla_with_egvla3.html"


def _scene_svg(sc: dict, include_egvla3: bool) -> str:
    """실제 로봇 사진 위 BBox(F2 [+ Soft-Loc]) 및 모델별 주행/정지 벡터 오버레이."""
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
    if include_egvla3:
        sl_m = ms.get("VLM-E2E-SoftLoc", {})
        if sl_m.get("softloc_bbox"):
            sb = sl_m["softloc_bbox"]
            draw_box(sb, "#e879f9", "4 3", f"Soft-Loc (cx={sb[0]:.2f}, a={sb[2]:.2f})", 12)

    arrows = [
        ("MOD-A3", "a3(현배포)", "#94a3b8", -12),
        ("MOD-A3-s2", "a3-s2(11/16)", "#fb923c", 8),
        ("MOD-A3-AG", "a3-ag23(1순위)", "#4ade80", 0),
    ]
    if include_egvla3:
        arrows += [
            ("VLM-E2E-Hidden", "Hidden(π0형)", "#c084fc", 12),
            ("VLM-E2E-SoftLoc", "SoftLoc(E2E)", "#e879f9", -6),
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
            tx = max(16.0, min(W - 58.0, ox - ly * scale))
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


def build_one(include_egvla3: bool, out_path: Path, data: dict) -> None:
    scenes = data["scenes"]
    v4_by_id = data["v4_by_id"]
    b3_ms, b1_ms, b1_sp = data["b3_ms"], data["b1_ms"], data["b1_sp"]
    sl_ms, sl_sp = data["sl_ms"], data["sl_sp"]
    eg11, eg08 = data["eg11"], data["eg08"]
    audit = data["audit"]

    takeaways_egvla2 = {
        "sc1_chair_center": "<b>[출발 기로]</b> 의자 정중앙: <code>a3</code>는 768-d 시각 지름길 때문에 <b>정지(실기 0/5)</b>. <code>a3-s2</code>와 <code>a3-ag23</code>은 정상 출발!",
        "sc2_basket_close": "<b>[바구니 초근접]</b> 면적 0.31: <code>a3-ag23</code>은 <b>정확히 STOP</b>(90%)하지만, <code>a3-s2</code>는 계속 전진(11/16 접촉 보고의 원인).",
        "sc3_chair_close": "<b>[의자 초근접]</b> 면적 0.33: <code>a3-ag23</code>은 낯선 의자에서도 면적 기하 정합으로 <b>정확히 STOP</b>(93%). <code>a3-s2</code>는 미정지.",
        "sc4_desk_center": "<b>[책상 정중앙]</b> <code>a3</code>는 비출발(STOP)로 멈추지만, <code>a3-ag23</code>과 <code>a3-s2</code>는 정상 직진 출발.",
        "sc5_side_steer": "<b>[측면 조향]</b> 바구니 좌측(cx=0.14): <code>a3</code>·<code>a3-s2</code>·<code>a3-ag23</code> 모두 좌측으로 정확히 조향(0.54° 정렬).",
        "sc6_pot_center": "<b>[화분 정중앙]</b> 300ep 미학습 화분: <code>a3-ag23</code>이 바구니와 동일하게 안정적 직진 출발.",
    }
    takeaways_egvla3 = {
        **takeaways_egvla2,
        "sc3_chair_close": "<b>[결정적 증거 ★]</b> 미학습 의자 초근접(a=0.33): <b><code>EG3-08</code>(π0형 Hidden)은 계속 전진</b>하지만, <b><code>EG3-09/11</code>(Soft-Loc)과 <code>a3-ag23</code>은 정확히 STOP</b>!",
    }
    takeaways = takeaways_egvla3 if include_egvla3 else takeaways_egvla2

    cards_html = []
    for sc in scenes:
        sid = sc["sid"]
        shot_dt = f"{sid[:4]}-{sid[4:6]}-{sid[6:8]} {sid[9:11]}:{sid[11:13]}:{sid[13:15]}" if len(sid) >= 15 else sid
        ms = sc["models"]
        v4d = v4_by_id.get(sc["id"], {})
        v4ro = (v4d.get("v4r", {}).get("orig") or [[0, 0], 0])
        v4rn = (v4d.get("v4r", {}).get("neg") or [[0, 0], 0])

        model_list = [
            ("MOD-A3", "EG2-9 (a3)", "#94a3b8", "실기(11/13)"),
            ("MOD-A3-s2", "EG2-14 (a3-s2)", "#fb923c", "실기(11/16)"),
            ("MOD-A3-AG", "EG2-16 (a3-ag23)", "#4ade80", "오프라인 추론"),
        ]
        if include_egvla3:
            model_list += [
                ("VLM-E2E-Hidden", "EG3-8 (Hidden)", "#c084fc", "오프라인 추론"),
                ("VLM-E2E-SoftLoc", "EG3-11 (SoftLoc)", "#e879f9", "오프라인 추론"),
            ]

        rows = []
        for k, lbl, col, tag_kind in model_list:
            m = ms.get(k)
            if not m:
                continue
            st = '<span class="bd b-bad">STOP</span>' if m["stop"] else '<span class="bd b-ok">DRIVE</span>'
            k_bd = (
                f'<span class="bd b-blue" style="font-size:.66rem;padding:1px 5px">{tag_kind}</span>'
                if "실기" in tag_kind
                else f'<span class="bd b-warn" style="font-size:.66rem;padding:1px 5px">{tag_kind}</span>'
            )
            rows.append(
                f'<tr><td style="color:{col};font-weight:700">{lbl} {k_bd}</td>'
                f'<td><code>[{m["cmd"][0]:+.2f}, {m["cmd"][1]:+.2f}]</code> ({m["ang_deg"]:+.0f}°)</td>'
                f'<td>{st}</td></tr>'
            )
        # V4R 요약 행 추가
        v4_stop = v4ro[1] < 0.2
        v4_badge = '<span class="bd b-bad">STOP</span>' if v4_stop else '<span class="bd b-ok">DRIVE</span>'
        rows.append(
            f'<tr><td style="color:#fbbf24;font-weight:700">EG2-T7 (v4r) <span class="bd b-warn" style="font-size:.66rem;padding:1px 5px">오프라인 추론</span></td>'
            f'<td><code>[{v4ro[0][0]:+.2f}, {v4ro[0][1]:+.2f}]</code> · 부정 <code>[{v4rn[0][0]:+.2f}, {v4rn[0][1]:+.2f}]</code></td>'
            f'<td>{v4_badge}</td></tr>'
        )

        cards_html.append(
            f'<div class="sc-card">'
            f'<div class="sc-top"><b>{html.escape(sc["tab"])}</b><span>📅 촬영: <b>{shot_dt}</b> · #{sc["fi"]}</span></div>'
            f'{_scene_svg(sc, include_egvla3)}'
            f'<div class="sc-ins">지시어: <code>"{html.escape(sc["ins"])}"</code> · {html.escape(sc["obj"])} ({html.escape(sc["pos"])})</div>'
            f'<div style="font-size:.73rem;color:#93c5fd;background:#09152b;padding:4px 7px;border-radius:6px"><b>실제 로봇 주행 기록({shot_dt[:10]}):</b> {html.escape(sc.get("real_outcome", ""))}</div>'
            f'<table class="mini-tbl"><tbody>{"".join(rows)}</tbody></table>'
            f'<div class="sc-note">{takeaways.get(sc["id"], "")}</div>'
            f'</div>'
        )

    audit_rows = []
    for mname, r in audit.items():
        if not include_egvla3 and ("VLM-E2E" in mname or "EG3-" in mname):
            continue
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
    title_main = "261126 EGVLA 실험보고 — EGVLA2 실기 개선·고속화·언어게이트" if not include_egvla3 else "261126 EGVLA 실험보고 (확장판) — EGVLA2 + EGVLA3 E2E 통합"
    ver_switch = (
        '<a href="report_261126_egvla_with_egvla3.html" style="border-color:#c084fc;color:#e9d5ff">🟣 EGVLA3 포함 확장판 보기 ↗</a>'
        if not include_egvla3
        else '<a href="report_261126_egvla.html" style="border-color:#38bdf8;color:#7dd3fc">🔵 EGVLA2 메인판 보기 ↗</a>'
    )

    eg3_timing_rows = ""
    if include_egvla3:
        eg3_timing_rows = """
        <tr>
          <td><b>Phase 2 특성 추출+학습</b><br>(<code>EG3-08</code>, <code>EG3-09</code>)</td>
          <td>학습 4,984장 + 11/13(870장) + 11/16(588장) = <b>6,442장</b> (<code>fp16</code>)</td>
          <td><b>51.9분</b> (<code>3,115.4초</code>)<br><span style="color:#94a3b8">추출 49.4분 + 학습 2.6분</span></td>
          <td>120ep 캐시 학습</td>
        </tr>
        <tr>
          <td><b>Phase 3 공동학습+전수평가</b><br>(<code>EG3-11</code>, Joint)</td>
          <td>1,500장 추출 + 8ep Decoder LoRA + <b>1,458장(11/13·16)</b> Soft-Loc 추론</td>
          <td><b>21.1분</b> (<code>1,264.2초</code>)<br><span style="color:#94a3b8">추출 11.1분 · 학습 2.9분 · <b>1,458장 추론 7.1분</b></span></td>
          <td>11/13(65세션) 4.2분 · 11/16(45세션) 2.8분 (~0.29초/장)</td>
        </tr>"""

    timing_box = f"""
    <h3 style="margin:16px 0 6px;font-size:1.02rem;color:#7dd3fc">⏱️ 실제 추론 기준 및 실측 소요 시간 (하드웨어: minum GB10 · 11/13 65세션 870장 · 11/16 45세션 588장)</h3>
    <table class="main-tbl">
      <thead><tr><th>측정 대상 / 구간</th><th>측정 기준 (샘플 수 · 정밀도 · 입력 조건)</th><th>실측 소요 시간</th><th>프레임당 지연 (ms) / 비고</th></tr></thead>
      <tbody>
        <tr>
          <td><b>현행 기준 (<code>a3</code>)</b><br><code>beam=3 fp32</code></td>
          <td>11/13+11/16 다물체·근접 <b>330프레임</b> (문자열 BBox + <code>PiZeroFM</code>)</td>
          <td>330장 기준 <b>~3.2분</b></td>
          <td>중앙값 <b><code>{b3_ms}ms</code></b> (p90 <code>597.6ms</code>, ~1.7Hz)</td>
        </tr>
        <tr>
          <td><b><code>MOD-DR-Fast/Safe</code></b><br><code>beam=1 fp16</code></td>
          <td>동일 <b>330프레임</b> 벤치마크 &amp; <code>EgVla2Service</code> 6장면 종단 서빙</td>
          <td>330장 기준 <b>~1.6분</b></td>
          <td>벤치 <b><code>{b1_ms}ms</code></b> ({b1_sp}x) · 종단 서빙 <b><code>285.3ms</code></b> (웜업 제외)</td>
        </tr>
        <tr>
          <td><b><code>VLM-E2E-SoftLoc</code></b><br><code>softloc_fp16</code></td>
          <td>동일 <b>330프레임</b> + <code>EgVla2Service</code> 6장면 종단 서빙 (파싱 0%)</td>
          <td>330장 기준 <b>~1.6분</b></td>
          <td>벤치 <b><code>{sl_ms}ms</code></b> ({sl_sp}x) · 종단 서빙 <b><code>296.6ms</code></b> (웜업 제외)</td>
        </tr>
        {eg3_timing_rows}
        <tr>
          <td><b>과적합 정밀 감사</b><br>(7개 모델 전수)</td>
          <td>11/13(870장) + 11/16(588장) 캐시 위 반사실 셔플 + 기하 반응성</td>
          <td><b>5.6초</b> (0.09분)</td>
          <td>캐시 특징 사용 시 초고속 검증 가능</td>
        </tr>
      </tbody>
    </table>"""

    eg3_model_rows = ""
    if include_egvla3:
        eg3_model_rows = f"""
      <tr>
        <td><b><code>EG3-08</code></b><br><code>VLM-E2E-Hidden</code></td>
        <td><span class="bd b-warn">오프라인 추론</span><br>(2026-11-22)</td>
        <td><b>π0형 순수 1-Pass E2E</b><br>(<code>BBox=0</code>, Decoder Hidden 직접 연결)</td>
        <td>• <code>~70ms</code> 초고속, <code>B1</code>(25.1°) 대비 각도 <code>7.70°</code>로 개선<br>• 바구니 정지 100% vs <b>의자 정지 {eg08['chair_ge25_stop_16']*100:.1f}%</b></td>
        <td>❌ 기하 병목 부재로 미학습 의자 정지 실패 (Ablation 대조군)</td>
      </tr>
      <tr style="background:#190f2b">
        <td><b><code>EG3-11</code> ★</b><br><code>VLM-E2E-Joint</code></td>
        <td><span class="bd b-warn">오프라인 추론</span><br>(2026-11-24)</td>
        <td><b>100% 미분 E2E 공동 학습</b><br>(Decoder LoRA + Soft-Loc + KL 증류)</td>
        <td>• 각도 <b><code>{eg11['ang_med_vs_a3']}°</code></b>(82.8%), <b>{sl_ms}ms</b>({sl_sp}x)<br>• 바구니/의자 정지 <b>100%/98%</b>, <b>Open-Vocab {eg11['open_vocab_1113_det_rate']*100:.1f}%</b></td>
        <td>🏆 Action→LoRA 역전파(<code>grad=0.142</code>) + 망각 0% (파일럿 실기 예정)</td>
      </tr>"""

    sota_sec = ""
    if include_egvla3:
        sota_sec = """
    <table class="main-tbl" style="margin-top:12px">
      <thead><tr><th>구분</th><th>2026 SOTA (π0 · GR00T N1 · OpenVLA-OFT · SpatialVLA)</th><th>우리 EGVLA2 (MOD-DR-Safe)</th><th>우리 EGVLA3 (EG3-11 SoftLoc-Joint)</th></tr></thead>
      <tbody>
        <tr><td><b>데이터 / VLM</b></td><td>수십만~110만 ep / 3B~7B 대형 VLM</td><td>300 ep / 0.23B Florence-2 (동결)</td><td>300 ep / 0.23B (Decoder LoRA 295k + KL)</td></tr>
        <tr><td><b>공간 추론 경계</b></td><td>Hidden State 직접 전달(π0) 또는 텍스트 CoT(ECoT)</td><td><code>beam=1 fp16</code> 문자열 파싱 (<code>0.54°</code>, 285ms)</td><td><b>미분가능 Soft-Loc 기댓값</b> (파싱 0%, 297ms)</td></tr>
        <tr><td><b>저데이터 핵심 장치</b></td><td>대규모 다물체 사전학습으로 자연 해소</td><td>Area-Grounded 라벨 + 50% Feature Swap</td><td>Soft-Loc 기하 병목 + Action→LoRA 역전파 + KL 증류</td></tr>
      </tbody>
    </table>"""

    legend_extra = ' · <b style="color:#c084fc">보라=EG3-08(Hidden)</b> · <b style="color:#e879f9">분홍=Soft-Loc(E2E)</b>' if include_egvla3 else ""

    doc = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title_main)}</title>
<style>
:root{{--bg:#080e1c;--card:#10192e;--head:#16233f;--fg:#e2e8f0;--mut:#94a3b8;--line:#223354;--acc:#38bdf8;--ok:#4ade80;--warn:#fbbf24;--bad:#f87171;--purp:#c084fc;--pink:#e879f9}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,'Noto Sans KR',sans-serif}}
main{{max-width:1320px;margin:0 auto;padding:22px 18px 80px}}
.top-nav{{position:sticky;top:0;z-index:30;background:rgba(8,14,28,.94);backdrop-filter:blur(8px);border-bottom:1px solid var(--line);padding:9px 0;margin:-22px -18px 18px;padding-left:18px;padding-right:18px;display:flex;flex-wrap:wrap;gap:8px;align-items:center;font-size:.8rem}}
.top-nav a{{border:1px solid var(--line);background:var(--card);color:var(--acc);padding:4px 10px;border-radius:99px;text-decoration:none;font-weight:700}}
.hero{{background:linear-gradient(135deg,#0f1a33,#132238,#0c1e2e);border:1px solid var(--line);border-radius:16px;padding:22px 24px;margin-bottom:20px}}
.hero h1{{margin:0 0 6px;font-size:1.64rem;color:#fff}}
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
table.main-tbl{{width:100%;border-collapse:collapse;font-size:.82rem;background:#0b1324;border-radius:10px;overflow:hidden;margin:10px 0}}
table.main-tbl th,table.main-tbl td{{border:1px solid var(--line);padding:7px 9px;text-align:left;vertical-align:top}}
table.main-tbl th{{background:var(--head);color:#cbd5e1;font-size:.77rem}}
@media print{{.top-nav{{display:none}}body{{background:#fff;color:#111}}section.act,.sc-card,table.main-tbl{{background:#fff!important;color:#111!important;border-color:#ccc!important}}}}
</style>
</head>
<body>
<main>
  <nav class="top-nav">
    <span style="font-weight:800;color:#fff">📑 261126 EGVLA 실험보고</span>
    <a href="#sec1">1. 핵심 결과</a>
    <a href="#sec2">2. 모델별 설명·비교</a>
    <a href="#sec3">3. 실기 vs 오프라인 6장면</a>
    <a href="#sec4">4. 과적합 감사</a>
    <a href="#sec5">5. 실기 로드맵</a>
    {ver_switch}
    <span style="margin-left:auto;color:var(--mut)">생성 {html.escape(gen_ts)}</span>
  </nav>

  <div class="hero">
    <h1>{html.escape(title_main)}</h1>
    <p>결과(Result First) → 모델별 구조·성능 비교 → 실기(날짜·시각 표기) vs 오프라인 추론 6장면 대조 → 과적합 감사 순서로 정리한 발표용 보고서</p>
    <div class="kpi-grid">
      <div class="kpi"><div class="t">핸들 방향 정확도 (a3-ag23)</div><div class="v">방향 오차 0.54° (98.7% 일치)</div><div class="desc">기존 잘 달리던 로봇과 핸들 각도 차이 <b>1° 미만</b> [오프라인]</div></div>
      <div class="kpi"><div class="t">출발 · 브레이크 안전율</div><div class="v">출발 100% · 멈춤 90~93%</div><div class="desc">의자 정면 출발 <b>0%→100%</b> · 충돌 전 정지 <b>90~93%</b></div></div>
      <div class="kpi"><div class="t">반응 속도 2배 가속</div><div class="v">{b3_ms}ms → {b1_ms}ms ({b1_sp}x)</div><div class="desc">초당 판단 1.7회 → <b>3.5회</b> (정확도 99.7% 유지)</div></div>
      <div class="kpi"><div class="t">{"EGVLA3 E2E (EG3-11)" if include_egvla3 else "엉뚱한 명령 차단 (V4R)"}</div><div class="v">{"방향 2.90° · 멈춤 98% · 검출 99.8%" if include_egvla3 else "부정 명령 99.5% 제자리 정지"}</div><div class="desc">{"VLM 전체를 한 번에 학습한 차세대 E2E" if include_egvla3 else "바구니가 보여도 다른 말이면 출발 안 함"}</div></div>
    </div>
  </div>

  <!-- 처음 보시는 분을 위한 지표 10초 가이드 -->
  <section class="act" style="border-left:5px solid #38bdf8;background:#0b1629">
    <h2 style="color:#7dd3fc">📖 숫자 &amp; 지표 읽는 법 (처음 보시는 분을 위한 10초 설명)</h2>
    <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px;font-size:.83rem;line-height:1.58">
      <div style="background:#08101f;border:1px solid #1e3154;border-radius:10px;padding:10px 12px">
        <b style="color:#4ade80">① 방향 오차 0.54° (핸들 정렬도)</b><br>
        기존 기준 로봇(<code>a3</code>)이 꺾으려는 주행 방향과 비교했을 때 <b>겨우 0.54°밖에 안 틀어진다</b>는 뜻입니다.<br>
        <span style="color:#94a3b8">• 비유: 시계 1분 눈금이 6°이므로, <b>1분 눈금의 1/10도 안 될 만큼 똑같은 길로 간다</b>는 의미입니다. (반면 나쁜 모델 <code>B1</code>은 <code>25.13°</code>로 딴 길로 샙니다.)</span>
      </div>
      <div style="background:#08101f;border:1px solid #1e3154;border-radius:10px;padding:10px 12px">
        <b style="color:#38bdf8">② 중앙 출발 100% (낯선 물체 출발률)</b><br>
        로봇이 한 번도 배운 적 없는 <b>의자·책상 정중앙</b> 앞에 섰을 때, 머뭇거리지 않고 출발한 비율입니다.<br>
        <span style="color:#94a3b8">• 기존 <code>a3</code>는 <b>0% (5번 다 제자리 멈춤)</b>였지만, 개선 모델은 <b>100% 출발</b>합니다.</span>
      </div>
      <div style="background:#08101f;border:1px solid #1e3154;border-radius:10px;padding:10px 12px">
        <b style="color:#fbbf24">③ 근접 정지 90% / 93% (충돌 전 브레이크)</b><br>
        목표물(바구니 90% · 의자 93%)에 <b>부딪히기 직전 거리(화면의 25% 이상 크기)</b>까지 갔을 때, 스스로 멈춘 비율입니다.<br>
        <span style="color:#94a3b8">• 기존 <code>a3-s2</code>는 55%/28%밖에 못 멈춰 실제로 6번 부딪혔습니다.</span>
      </div>
      <div style="background:#08101f;border:1px solid #1e3154;border-radius:10px;padding:10px 12px">
        <b style="color:#e879f9">④ 부정 명령 정지 99.5% (말귀 이해)</b><br>
        눈앞에 바구니가 있어도, 사람이 <b>"빨간 콘으로 가"</b>처럼 다른 명령을 내리면 속지 않고 멈춘 비율입니다.<br>
        <span style="color:#94a3b8">• <code>a3</code>는 텍스트를 못 읽어 무조건 출발하지만, <code>V4R</code>은 99.5% 멈춥니다.</span>
      </div>
    </div>
  </section>

  <!-- 1. 핵심 결과 요약 (Result First) -->
  <section class="act" id="sec1">
    <h2>1. 핵심 결과 요약 — 무엇을 발견했고 무엇이 개선되었는가?</h2>
    <table class="main-tbl">
      <thead><tr><th>핵심 문제 (실기 관측)</th><th>원인 규명 (반사실 실험)</th><th>해결 모델 &amp; 기법</th><th>정량 개선 결과 (Before → After)</th></tr></thead>
      <tbody>
        <tr>
          <td><b>① 낯선 물체 중앙 비출발</b><br>(<code>a3</code>, 2026-11-13 실기 의자 <b>0/5</b>)</td>
          <td><b>시각 외형 지름길</b>: 768-d 이미지 특징이 "중앙 큰 물체=이미 도착"으로 오인</td>
          <td><b><code>a3-ag23</code></b> (<code>MOD-A3-AG</code>)<br>50% Feature Swap + <code>area</code> 기하 정합</td>
          <td>의자·책상 중앙 출발 <b>0% → 100%</b><br>시각 셔플 뒤집힘 <b>27.4% → 0.0%</b></td>
        </tr>
        <tr>
          <td><b>② 근접 미정지 · 접촉</b><br>(<code>a3-s2</code>, 2026-11-16 실기 <b>접촉 6건</b>)</td>
          <td><code>area ≥ 0.25</code> 정지 반응성 부족 (<code>S_area = -0.3%p</code>)</td>
          <td><b><code>a3-ag23</code> + <code>MOD-DR-Safe</code></b><br><code>area≥0.23</code> 정합 + S2 선형 감속</td>
          <td>11/16 근접 정지 <b>55%/28% → 90%/93%</b><br>기하 면적 반응성 <b>-0.3%p → +100.0%p</b></td>
        </tr>
        <tr>
          <td><b>③ 추론 지연 (590.7ms)</b><br>(제어 주기 ~1.7Hz 병목)</td>
          <td>Florence-2 <code>beam=3 fp32</code> 탐색이 전체 지연의 94% 차지</td>
          <td><b><code>MOD-DR-Fast</code></b><br><code>beam=1 fp16</code> + 중복 인코딩 제거</td>
          <td>종단 서빙 지연 <b>590.7ms → 285.3ms (2.09x)</b><br><code>beam=3</code> 일치율 <b>99.7%</b> (<code>cx MAE 0.0016</code>)</td>
        </tr>
        <tr>
          <td><b>④ 비목표 지시어 오출발</b><br>(<code>a3</code>는 텍스트 입력 없음)</td>
          <td><code>PiZeroFM(12,768)</code>에 지시어 특징이 없어 <code>"red cone"</code>에도 출발</td>
          <td><b><code>V4 / V4R</code></b> (<code>MOD-V4R</code>)<br><code>fdim=1536</code>(이미지+문구) + 박스훼손 증강</td>
          <td>비바구니/부정 지시어 <b>99.5% 제자리 정지</b><br>박스 좌우 반전 방어 <b>86.0% → 94.3%</b></td>
        </tr>
      </tbody>
    </table>
  </section>

  <!-- 2. 모델별 상세 설명 & 비교 -->
  <section class="act" id="sec2">
    <h2>2. 모델별 구조 설명 및 정량 비교 (실기 기준선 vs 개선 후보)</h2>
    <p class="lead">각 모델이 <b>무엇을 입력받고(구조)</b>, <b>어떤 강점과 한계</b>를 가지며, <b>실기 주행인지 오프라인 추론인지</b>를 한 표에 정리한다.</p>
    <table class="main-tbl">
      <thead><tr><th>모델명 (ID)</th><th>검증 구분 (날짜)</th><th>아키텍처 / 핵심 특징</th><th>핵심 정량 지표</th><th>강점 / 한계 및 역할</th></tr></thead>
      <tbody>
        <tr>
          <td><b><code>EG2-9 (a3)</code></b><br><span style="color:#94a3b8;font-size:.75rem">MOD-A3 · 현 배포</span></td>
          <td><span class="bd b-blue">실기 완료</span><br>2026-10-29~11-13</td>
          <td>Florence-2(<code>beam=3 fp32</code>) → <code>PiZeroFM(12,768)</code><br>(조향 <code>cx</code>만 서보 정합)</td>
          <td>• 바구니 실기 <b>455/455</b><br>• 11/13 의자 중앙 <b>0/5</b><br>• 11/16 근접 정지 80%/58%</td>
          <td><b>현 배포 기준선</b>. 바구니 주행 안정적이나 시각 지름길로 낯선 물체 중앙 비출발.</td>
        </tr>
        <tr>
          <td><b><code>EG2-14 (a3-s2)</code></b><br><span style="color:#94a3b8;font-size:.75rem">MOD-A3-s2</span></td>
          <td><span class="bd b-blue">실기 완료</span><br>2026-11-16 (45/45)</td>
          <td><code>a3</code>와 동일 레시피 · 시드 2</td>
          <td>• 의자·책상 중앙 <b>5/5 (100%)</b><br>• 11/16 근접 정지 <b>55%/28%</b><br>• T7 각도 <code>0.70°</code></td>
          <td>중앙 출발 문제는 풀렸으나 근접 정지가 약해 <b>접촉 6건</b> 발생 → <code>a3-ag23</code> 개발 계기.</td>
        </tr>
        <tr style="background:#0d1f18">
          <td><b><code>EG2-16 (a3-ag23)</code> ★</b><br><span style="color:#94a3b8;font-size:.75rem">MOD-A3-AG</span></td>
          <td><span class="bd b-warn">오프라인 추론</span><br>2026-11-17~24</td>
          <td>동일 <code>PiZeroFM(12,768)</code><br><b>조향(cx) + 정지(area≥0.23)</b> 동시 정합 + 50% Feature Swap</td>
          <td>• T7 각도 <b><code>0.54°</code></b> (좌우 98.7%)<br>• 의자·책상 중앙 <b>100%/100%</b><br>• 근접 정지 <b>90%/93%</b>, 오정지 <b>0.0%</b></td>
          <td>🏆 <b>EGVLA2 실기 1순위</b>. 기존 서버 코드 수정 없이 <code>.pt</code> 드롭인 교체 가능.</td>
        </tr>
        <tr>
          <td><b><code>EG2-17 (a3-dr-fast)</code></b><br><span style="color:#94a3b8;font-size:.75rem">MOD-DR-Fast/Safe</span></td>
          <td><span class="bd b-warn">GPU 서빙 검증</span><br>2026-11-22~24</td>
          <td><code>beam=1 fp16</code> 고속 추출 + <b>S2 감속(0.14→0.25)</b> + 스파이크 필터 + <b>YahBoom 어댑터</b></td>
          <td>• 지연 <b><code>285.3ms</code> (2.09x)</b><br>• <code>beam=3</code> 일치 <b>99.7%</b><br>• <code>cx MAE 0.0016</code></td>
          <td>⚡ <b>EGVLA2 엔딩 완성 계층</b>. 관성 접촉 억제 및 YahBoom 카 무재학습 이식 담당.</td>
        </tr>
        <tr>
          <td><b><code>EG2-T4/T7 (v4 / v4r)</code></b><br><span style="color:#94a3b8;font-size:.75rem">MOD-V4R · 기존 EG4-04/07</span></td>
          <td><span class="bd b-warn">오프라인 추론</span><br>(실기 2종 예정)</td>
          <td><code>"gray basket"</code> 고정 BBox + <b><code>TextCondPolicy(12,1536)</code></b><br>(이미지 768 + 문구 768)</td>
          <td>• 비목표 지시어 <b>99.5% 정지</b><br>• <code>v4r</code> 박스 반전 방어 <b>94.3%</b><br>• T7 각도 <code>3.17°</code> (좌우 94.5%)</td>
          <td>🟠 <b>바구니 전용 안전 차단기 (일반화 모델 아님!)</b><br>의자·책상 주행 일반화가 아니라, 바구니가 아닌 명령일 때 멈추게 하는 전용 모델 (다물체 일반화는 <code>EG2-16</code> 담당).</td>
        </tr>
        {eg3_model_rows}
      </tbody>
    </table>
    {timing_box}
  </section>

  <!-- 3. 실기 vs 오프라인 추론 6장면 비교 -->
  <section class="act" id="sec3">
    <h2>3. 오프라인 추론 재생 — 실제 실기 촬영 사진(날짜·시각 표기) 위 모델별 예측 비교</h2>
    <div style="background:#171207;border:1px solid #b45309;border-left:4px solid #fbbf24;border-radius:8px;padding:8px 12px;font-size:.81rem;margin-bottom:12px;line-height:1.55">
      <b>⚠️ 실기 주행 vs 오프라인 추론 구분:</b>
      <span class="bd b-blue">실기(날짜)</span>는 실제 로봇 바퀴로 주행한 기록(<code>a3</code>: 2026-10-29~11-13, <code>a3-s2</code>: 2026-11-16)이고,
      <span class="bd b-warn">오프라인 추론</span>은 위 실기 촬영 사진(<b>날짜·시각 명시</b>)을 GPU에 넣어 모델별 출력 명령을 재생·비교한 결과다.
    </div>
    <p class="lead">범례: <b style="color:#94a3b8">회색=a3</b> · <b style="color:#fb923c">주황=a3-s2</b> · <b style="color:#4ade80">초록=a3-ag23</b>{legend_extra} · <b>원(○)=STOP</b></p>
    <div class="sc-grid">{"".join(cards_html)}</div>
  </section>

  <!-- 4. 과적합 정밀 감사 -->
  <section class="act" id="sec4">
    <h2>4. 과적합 정밀 감사 (반사실 시각 셔플 &amp; 미학습 물체 일반화 격차)</h2>
    <p class="lead">300에피소드(바구니만 학습)에서 이미지 배경 편향에 과적합되었는지 확인하기 위해 <b>동일 BBox에서 시각 특징만 무작위 셔플</b>한 결과, <code>a3</code>는 정지 판정이 <b>27.4%</b> 뒤집힌 반면 <code>a3-ag23</code>{" 및 <code>EG3-11</code>" if include_egvla3 else ""}은 <b>0.0%</b>로 완벽히 일반화되었다.</p>
    <table class="main-tbl">
      <thead><tr><th>모델</th><th>출발 격차 (바구니 vs 의자)</th><th>정지 격차 (바구니 vs 의자)</th><th>시각 셔플 민감도 (각도 / 뒤집힘)</th><th>기하 반응성 (S_cx / S_area)</th><th>감사 판정</th></tr></thead>
      <tbody>{"".join(audit_rows)}</tbody>
    </table>
    {sota_sec}
  </section>

  <!-- 5. 실기 현황 & 로드맵 -->
  <section class="act" id="sec5">
    <h2>5. 완료 현황 (실기 vs 오프라인) &amp; 다음 실기 로드맵</h2>
    <div class="box" style="border-left:5px solid var(--ok);background:#0b1324;border:1px solid var(--line);border-radius:12px;padding:13px 15px;font-size:.85rem;line-height:1.7">
      <b>1. 완료된 실제 로봇 주행 (날짜별):</b>
      <code>a3</code> (<b>2026-10-29~11-04</b> 바구니 455/455, <b>2026-11-13</b> 다물체 65세션) · <code>a3-s2</code> (<b>2026-11-16</b> 45/45, 접촉 6건)<br>
      <b>2. 완료된 오프라인 추론 &amp; GPU 종단 서빙 (2026-11-22~24, 실기 미실시):</b>
      <code>A3C_AG23_s0</code>, <code>MOD-DR-Safe</code>(285ms · S2 감속 · YahBoom 어댑터){", <code>EG3-11</code>(297ms · 2.90° · Open-Vocab 99.8%)" if include_egvla3 else ""} → <code>soda</code> 인박스 전달 완료(<code>SHA256 9/9 OK</code>).<br><br>
      <b>3. 확정된 세대별 다음 실기 계획 (EGVLA2 엔딩 → EGVLA3 자율정지 E2E):</b><br>
      • <b>🔵 EGVLA2 엔딩 (모듈러 실용화 완결)</b>: <code>soda</code>에서 <code>V4/V4R</code> 실기 2종 → <code>A3C_AG23_s0</code> + <b><code>beam=1 fp16</code> (285ms)</b> + S2 감속으로 Serbot 정지 확정 → <b>YahBoom 카 이식</b>으로 EGVLA2 마무리.<br>
      • <b>🟣 EGVLA3 본편 (자율 멈춤 E2E VLA)</b>: 하드코딩 <code>near_area=0.25</code> 규칙을 끄고(<code>VLA_EGVLA2_NEAR_AREA=off</code>, <code>STOP_DEBOUNCE=2</code>), <b><code>EG3-11</code>(<code>softloc_fp16</code>) 모델 자체가 멈춤(<code>[0, 0]</code>)을 직접 출력</b>해 정지하는 순수 E2E 실기 검증 진행.
    </div>
  </section>
</main>
</body>
</html>
"""
    out_path.write_text(doc, encoding="utf-8")
    print(f"Generated {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


def main() -> None:
    fam = json.loads((FM / "family_arch_compare.json").read_text(encoding="utf-8"))
    v4_cmp = json.loads((FM / "v4_vs_a3_compare.json").read_text(encoding="utf-8"))
    beam = json.loads((FM / "beam_bench.json").read_text(encoding="utf-8"))
    ta = json.loads((FM / "track_a_dec1pass_results.json").read_text(encoding="utf-8"))
    audit = json.loads((FM / "overfitting_audit.json").read_text(encoding="utf-8"))

    data = dict(
        scenes=fam["scenes"],
        v4_by_id={x["id"]: x for x in v4_cmp},
        b3_ms=beam["beam3_fp32 (현행 기준)"]["latency_med_ms"],
        b1_ms=beam["beam1_fp16 (Track B)"]["latency_med_ms"],
        b1_sp=beam["beam1_fp16 (Track B)"]["speedup_vs_beam3"],
        sl_ms=beam["softloc_fp16 (Track A2)"]["latency_med_ms"],
        sl_sp=beam["softloc_fp16 (Track A2)"]["speedup_vs_beam3"],
        eg11=ta["VLM-E2E-SoftLoc-Joint (EG3-11, DecLoRA+KL)"],
        eg08=ta["Track A1 (Decoder 1-Pass 순수 E2E)"],
        audit=audit,
    )
    build_one(False, OUT_MAIN, data)
    build_one(True, OUT_FULL, data)


if __name__ == "__main__":
    main()

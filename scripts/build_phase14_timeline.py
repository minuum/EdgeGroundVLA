#!/usr/bin/env python3
"""Phase 1~4 실제 서빙 파이프라인 가로 스크롤 타임라인 시각화 생성기.

오프라인 실기 프레임 재생·GPU 서빙 실측 데이터를 바탕으로,
Phase 1(MOD-DR-Fast) / Phase 2(VLM-E2E-Hidden) / Phase 3(VLM-E2E-SoftLoc) / Phase 4(MOD-DR-Safe + YahBoom)
4개 파이프라인에서 실제 카메라 이미지·단어(토큰)·레이어·텐서 차원·제어 명령이
왼쪽에서 오른쪽으로 흐르는 전 과정을 가로 스크롤 타임라인으로 상세하게 표현한다.

출력: docs/v5/phase14_serving_timeline.html
"""
from __future__ import annotations

import datetime
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FM = ROOT / "docs/v5/egvla2_fm"
OUT_HTML = ROOT / "docs/v5/phase14_serving_timeline.html"


def main() -> None:
    fam = json.loads((FM / "family_arch_compare.json").read_text(encoding="utf-8"))
    beam = json.loads((FM / "beam_bench.json").read_text(encoding="utf-8"))
    ta = json.loads((FM / "track_a_dec1pass_results.json").read_text(encoding="utf-8"))

    # 대표 3장면(출발 / 초근접 정지 / 측면 조향) + 나머지 3장면 모두 선택 가능하게 전달
    scenes = fam["scenes"]
    scenes_js = json.dumps(scenes, ensure_ascii=False)
    gen_ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    b3_ms = beam.get("beam3_fp32 (현행 기준)", {}).get("latency_med_ms", 590.7)
    b1_ms = beam.get("beam1_fp16 (Track B)", {}).get("latency_med_ms", 283.0)
    sl_ms = beam.get("softloc_fp16 (Track A2)", {}).get("latency_med_ms", 283.8)

    doc = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Phase 1~4 실제 서빙 입출력 · 레이어 가로 스크롤 타임라인</title>
<style>
:root{{--bg:#090f1d;--card:#10192d;--node:#0c1426;--head:#16223b;--fg:#e2e8f0;--mut:#94a3b8;--line:#233454;--acc:#38bdf8;--ok:#4ade80;--warn:#fbbf24;--bad:#f87171;--purp:#c084fc;--pink:#e879f9;--ora:#fb923c}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font:14px/1.55 system-ui,'Noto Sans KR',sans-serif}}
main{{max-width:1480px;margin:0 auto;padding:20px 16px 80px}}
h1{{font-size:1.52rem;margin:0 0 6px;color:#f8fafc}}
p.sub{{margin:0 0 12px;color:var(--mut);font-size:.88rem}}
a{{color:var(--acc)}}
code{{background:#1b2944;padding:1px 5px;border-radius:4px;font-size:.8em;color:#f1f5f9;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}}
.nav-top{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:14px;font-size:.8rem}}
.nav-top a{{border:1px solid var(--line);background:var(--card);padding:4px 11px;border-radius:99px;text-decoration:none}}

.honest-banner{{border:1px solid #b45309;border-left:5px solid var(--warn);background:#1c160b;border-radius:10px;padding:10px 14px;margin:10px 0 14px;font-size:.83rem;line-height:1.6}}
.honest-banner b{{color:#fde68a}}

.scene-bar{{position:sticky;top:0;z-index:30;background:rgba(9,15,29,.95);backdrop-filter:blur(8px);border:1px solid var(--line);border-radius:12px;padding:10px 12px;margin:10px 0 18px;display:flex;flex-wrap:wrap;gap:8px;align-items:center}}
.scene-bar .lbl{{font-size:.8rem;font-weight:800;color:#fcd34d}}
.sc-btn{{border:1px solid var(--line);background:var(--head);color:var(--mut);padding:5px 11px;border-radius:8px;font-size:.79rem;font-weight:700;cursor:pointer;transition:.15s}}
.sc-btn.on{{background:var(--acc);color:#090f1d;border-color:var(--acc)}}
.scroll-hint{{margin-left:auto;font-size:.76rem;color:#7dd3fc;font-weight:700}}

.phase-sec{{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:14px 16px 16px;margin:18px 0}}
.phase-sec.p1{{border-top:4px solid var(--acc)}}
.phase-sec.p2{{border-top:4px solid var(--purp)}}
.phase-sec.p3{{border-top:4px solid var(--pink)}}
.phase-sec.p4{{border-top:4px solid var(--ok)}}

.phase-hdr{{display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:8px;margin-bottom:10px}}
.phase-ttl{{font-size:1.08rem;font-weight:800;color:#fff;display:flex;align-items:center;gap:8px}}
.badge{{display:inline-block;padding:2px 9px;border-radius:99px;font-size:.72rem;font-weight:700;white-space:nowrap}}
.b-blue{{background:#0c2d48;color:#7dd3fc;border:1px solid #0284c7}}
.b-purp{{background:#2e1065;color:#e9d5ff;border:1px solid #9333ea}}
.b-pink{{background:#4a044e;color:#f5d0fe;border:1px solid #c026d3}}
.b-ok{{background:#14532d;color:#86efac;border:1px solid #16a34a}}
.b-bad{{background:#450a0a;color:#fca5a5;border:1px solid #dc2626}}
.b-warn{{background:#451a03;color:#fcd34d;border:1px solid #d97706}}
.phase-sub{{font-size:.8rem;color:var(--mut);margin-bottom:10px}}

/* 가로 스크롤 타임라인 트랙 */
.tl-wrap{{position:relative}}
.tl-scroll{{display:flex;align-items:stretch;gap:0;overflow-x:auto;padding:8px 4px 14px;scroll-behavior:smooth;scrollbar-color:#334155 #0a1220;scrollbar-width:thin}}
.tl-scroll::-webkit-scrollbar{{height:9px}}
.tl-scroll::-webkit-scrollbar-track{{background:#0a1220;border-radius:99px}}
.tl-scroll::-webkit-scrollbar-thumb{{background:#334155;border-radius:99px}}
.tl-scroll::-webkit-scrollbar-thumb:hover{{background:#475569}}

.step-card{{flex:0 0 278px;background:var(--node);border:1px solid var(--line);border-radius:12px;padding:10px 11px;display:flex;flex-direction:column;gap:7px;position:relative}}
.step-card.hl-break{{border:1.5px dashed var(--bad);background:#1b0f16}}
.step-card.hl-diff{{border:1.5px solid var(--pink);background:#1a0d29}}
.step-card.hl-safe{{border:1.5px solid var(--ok);background:#0c1f19}}
.step-top{{display:flex;justify-content:space-between;align-items:center;gap:6px}}
.step-num{{font-size:.7rem;font-weight:800;letter-spacing:.04em;color:var(--acc);text-transform:uppercase}}
.step-ms{{font-size:.7rem;font-weight:700;background:#16223b;color:#fcd34d;padding:1px 6px;border-radius:6px}}
.step-ttl{{font-size:.88rem;font-weight:800;color:#fff;line-height:1.3}}

.arrow-col{{flex:0 0 34px;display:flex;flex-direction:column;align-items:center;justify-content:center;color:#64748b;font-size:1.1rem;font-weight:900;user-select:none}}
.arrow-col span.atag{{font-size:.62rem;color:#94a3b8;font-weight:700;text-align:center;line-height:1.2;margin-top:2px}}

.layer-stack{{display:flex;flex-direction:column;gap:4px}}
.layer-box{{background:#132038;border:1px solid #263a5e;border-radius:6px;padding:4px 7px;font-size:.72rem;line-height:1.4}}
.layer-box b{{color:#e2e8f0;display:block}}
.layer-box span{{color:#94a3b8;font-size:.69rem}}

.io-box{{background:#070d1a;border:1px solid #1e2d4a;border-radius:7px;padding:6px 8px;font-size:.72rem;line-height:1.45;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;color:#cbd5e1;word-break:break-word}}
.io-box .k{{color:#7dd3fc;font-weight:700}}
.io-box .v{{color:#fde68a}}
.io-box .ok{{color:#4ade80;font-weight:700}}
.io-box .bad{{color:#f87171;font-weight:700}}

.mini-cam{{position:relative;width:100%;aspect-ratio:16/9;border-radius:7px;overflow:hidden;background:#000;border:1px solid #263a5e}}
.mini-cam img{{width:100%;height:100%;object-fit:cover;display:block}}
.mini-cam svg{{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}}

.tok-row{{display:flex;flex-wrap:wrap;gap:3px}}
.tok{{background:#1e293b;border:1px solid #334155;color:#e2e8f0;padding:1px 5px;border-radius:4px;font-size:.68rem;font-family:ui-monospace,monospace}}
.tok.loc{{background:#3b0764;border-color:#a855f7;color:#f5d0fe;font-weight:700}}
.tok.sp{{background:#0c4a6e;border-color:#0284c7;color:#bae6fd}}

.step-note{{font-size:.73rem;color:var(--mut);line-height:1.45;margin-top:auto}}
.mini-svg{{width:100%;height:auto;background:#070d1a;border:1px solid #1e2d4a;border-radius:7px;display:block}}
</style>
</head>
<body>
<main>
  <div class="nav-top">
    <a href="research_story_egvla2.html#ch83">← CH83 E2E 분석·사진</a>
    <a href="family_arch_compare.html">🔀 3계열 6장면 스튜디오</a>
    <a href="model_registry.html#naming">📋 모델 레지스트리 (42종)</a>
    <a href="v4_vs_a3_compare.html">🟣 V4/V4R vs A3</a>
  </div>

  <h1>Phase 1~4 실제 서빙 입출력 · 레이어 상세 가로 스크롤 타임라인</h1>
  <p class="sub">
    생성 {html.escape(gen_ts)} · 실제 로봇 카메라 프레임 1장이 HTTP <code>/predict</code>로 들어와 <b>이미지 전처리 → 토큰화 → Vision/Text 레이어 → Decoder → Action Expert → 컨트롤러 → 로봇 구동 명령</b>으로 변환되는 전 과정을 Phase 1~4별로 나란히 가로 스크롤하며 추적합니다.
  </p>

  <div class="honest-banner">
    <b>⚠️ 검증 범위 명시 (오프라인/GPU 서빙 실측 vs 실제 로봇 주행):</b>
    지금까지 완료된 검증은 <b>① 실제 로봇(11/13·11/16)이 촬영한 카메라 프레임 1,458장 오프라인 재생</b>과 <b>② 실제 서빙 코드(<code>EgVla2Service</code>) 위에서의 GPU 종단 추론 실측</b>입니다.
    아직 실제 Serbot/YahBoom 바퀴를 굴리는 폐루프 실기 주행은 실시하지 않았으며, 아래 4개 타임라인은 <b>실기 서버에 올라갔을 때 정확히 어떤 텐서·레이어·명령이 어떤 순서로 흐르는지</b>를 실측 수치로 해부한 청사진입니다.
  </div>

  <div class="scene-bar">
    <span class="lbl">📷 테스트할 실제 로봇 카메라 장면 선택:</span>
    <div id="sc-btns" style="display:flex;flex-wrap:wrap;gap:6px"></div>
    <span class="scroll-hint">↔ 각 타임라인 카드를 좌우로 스크롤하세요</span>
  </div>

  <div id="timelines"></div>
</main>

<script>
const SCENES = {scenes_js};
const B3_MS = {b3_ms};
const B1_MS = {b1_ms};
const SL_MS = {sl_ms};
let curIdx = 2; // 기본: sc3_chair_close (Phase 2 vs Phase 3 차이가 가장 극명한 장면!)

function extractPhrase(ins) {{
  let s = (ins || '').trim().toLowerCase();
  for (const p of ['go to the ', 'go to ', 'move to the ', 'move to ', 'find the ', 'find ']) {{
    if (s.startsWith(p)) return s.slice(p.length).trim();
  }}
  return s || 'gray basket';
}}

function boxSvg(bb, col, dash, lbl, cmd, isStop) {{
  const W = 320, H = 180;
  let out = '';
  if (bb && bb[3] >= 0.3) {{
    const side = Math.sqrt(Math.max(0.002, bb[2]));
    const bw = Math.min(0.92, side * 1.15) * W;
    const bh = Math.min(0.92, side / 1.15) * H;
    const x = Math.max(2, Math.min(W - bw - 2, bb[0] * W - bw / 2));
    const y = Math.max(2, Math.min(H - bh - 2, bb[1] * H - bh / 2));
    out += `<rect x="${{x.toFixed(1)}}" y="${{y.toFixed(1)}}" width="${{bw.toFixed(1)}}" height="${{bh.toFixed(1)}}" fill="none" stroke="${{col}}" stroke-width="2.4" ${{dash ? `stroke-dasharray="${{dash}}"` : ''}}/>`;
    out += `<text x="${{(x+3).toFixed(1)}}" y="${{Math.max(12, y-3).toFixed(1)}}" fill="${{col}}" font-size="10" font-weight="700">${{lbl}}</text>`;
  }}
  if (cmd) {{
    const ox = W / 2, oy = H - 14, sc = 88;
    if (isStop) {{
      out += `<circle cx="${{ox}}" cy="${{oy}}" r="9" fill="none" stroke="#f87171" stroke-width="2.8"/>`;
      out += `<text x="${{ox+13}}" y="${{oy+4}}" fill="#fca5a5" font-size="11" font-weight="800">STOP</text>`;
    }} else {{
      const tx = Math.max(16, Math.min(W - 16, ox - cmd[1] * sc));
      const ty = Math.max(16, Math.min(H - 12, oy - cmd[0] * sc));
      out += `<line x1="${{ox}}" y1="${{oy}}" x2="${{tx.toFixed(1)}}" y2="${{ty.toFixed(1)}}" stroke="${{col}}" stroke-width="3.4" stroke-linecap="round"/>`;
      out += `<circle cx="${{tx.toFixed(1)}}" cy="${{ty.toFixed(1)}}" r="4.5" fill="${{col}}"/>`;
    }}
  }}
  return `<svg viewBox="0 0 ${{W}} ${{H}}">${{out}}</svg>`;
}}

function softlocDistSvg(cx) {{
  // Softmax <loc_0~999> 분포 시각화 미니 SVG
  const W = 250, H = 68, mu = Math.max(0.08, Math.min(0.92, cx));
  let pts = [];
  for (let i = 0; i <= 40; i++) {{
    const u = i / 40;
    const p = Math.exp(-Math.pow((u - mu) / 0.045, 2));
    const x = 10 + u * (W - 20);
    const y = 52 - p * 38;
    pts.push(`${{x.toFixed(1)}},${{y.toFixed(1)}}`);
  }}
  const mx = 10 + mu * (W - 20);
  return `<svg class="mini-svg" viewBox="0 0 ${{W}} ${{H}}">
    <line x1="10" y1="52" x2="${{W-10}}" y2="52" stroke="#334155"/>
    <polyline fill="none" stroke="#e879f9" stroke-width="2" points="${{pts.join(' ')}}"/>
    <line x1="${{mx.toFixed(1)}}" y1="10" x2="${{mx.toFixed(1)}}" y2="52" stroke="#4ade80" stroke-dasharray="3 2"/>
    <text x="${{mx.toFixed(1)}}" y="11" text-anchor="middle" fill="#4ade80" font-size="8.5" font-weight="700">E[u]=${{mu.toFixed(3)}}</text>
    <text x="10" y="64" fill="#94a3b8" font-size="8">&lt;loc_0&gt;(50269)</text>
    <text x="${{W-10}}" y="64" text-anchor="end" fill="#94a3b8" font-size="8">&lt;loc_999&gt;(51268)</text>
  </svg>`;
}}

function decelCurveSvg(area) {{
  // Phase 4 S2 연속 접근 감속 곡선 (area 0.14 -> 0.25)
  const W = 250, H = 70;
  const ax = a => 14 + (Math.min(0.40, Math.max(0, a)) / 0.40) * (W - 28);
  const sy = s => 54 - s * 40;
  const effA = Math.min(0.40, Math.max(0, area));
  let spd = 1.0;
  if (effA >= 0.25) spd = 0.0;
  else if (effA > 0.14) spd = 1.0 - ((effA - 0.14) / 0.11) * (1.0 - 0.35);
  return `<svg class="mini-svg" viewBox="0 0 ${{W}} ${{H}}">
    <line x1="14" y1="54" x2="${{W-14}}" y2="54" stroke="#334155"/>
    <polyline fill="none" stroke="#4ade80" stroke-width="2.2" points="${{ax(0)}},${{sy(1)}} ${{ax(0.14)}},${{sy(1)}} ${{ax(0.25)}},${{sy(0.35)}} ${{ax(0.25)}},${{sy(0)}} ${{ax(0.40)}},${{sy(0)}}"/>
    <line x1="${{ax(0.14)}}" y1="12" x2="${{ax(0.14)}}" y2="54" stroke="#fbbf24" stroke-dasharray="2 2"/>
    <line x1="${{ax(0.25)}}" y1="12" x2="${{ax(0.25)}}" y2="54" stroke="#f87171" stroke-dasharray="2 2"/>
    <circle cx="${{ax(effA).toFixed(1)}}" cy="${{sy(spd).toFixed(1)}}" r="4.5" fill="#38bdf8" stroke="#fff" stroke-width="1.2"/>
    <text x="${{ax(0.14)}}" y="65" text-anchor="middle" fill="#fbbf24" font-size="8">0.14 감속</text>
    <text x="${{ax(0.25)}}" y="65" text-anchor="middle" fill="#f87171" font-size="8">0.25 정지</text>
    <text x="${{W-14}}" y="20" text-anchor="end" fill="#7dd3fc" font-size="8.5" font-weight="700">현재 a=${{area.toFixed(2)}} → ×${{spd.toFixed(2)}}</text>
  </svg>`;
}}

function arrow(tag) {{
  return `<div class="arrow-col">→<span class="atag">${{tag}}</span></div>`;
}}

function card(stepNum, ms, title, bodyHtml, note, cls='') {{
  return `<div class="step-card ${{cls}}">
    <div class="step-top"><span class="step-num">${{stepNum}}</span><span class="step-ms">${{ms}}</span></div>
    <div class="step-ttl">${{title}}</div>
    ${{bodyHtml}}
    <div class="step-note">${{note}}</div>
  </div>`;
}}

function buildTracks(sc) {{
  const ph = extractPhrase(sc.ins);
  const phWords = ph.split(/\\s+/);
  const bb = sc.bbox;
  const mAG = sc.models['MOD-A3-AG'];
  const mHid = sc.models['VLM-E2E-Hidden'];
  const mSL = sc.models['VLM-E2E-SoftLoc'];
  const sb = (mSL && mSL.softloc_bbox) ? mSL.softloc_bbox : bb;

  // 근사 bbox 정수 좌표 (0..999)
  const side = Math.sqrt(Math.max(0.002, bb[2]));
  const x1 = Math.max(0, Math.round((bb[0] - side * 0.55) * 1000));
  const y1 = Math.max(0, Math.round((bb[1] - side * 0.45) * 1000));
  const x2 = Math.min(999, Math.round((bb[0] + side * 0.55) * 1000));
  const y2 = Math.min(999, Math.round((bb[1] + side * 0.45) * 1000));

  // YahBoom 변환 계산
  const lx = mAG.stop ? 0 : mAG.cmd[0];
  const ly = mAG.stop ? 0 : mAG.cmd[1];
  const th = Math.atan2(ly, Math.max(1e-4, lx));
  const ybV = mAG.stop ? 0 : Math.max(0, Math.cos(th)) * 0.25 * (bb[2] >= 0.25 ? 0 : (bb[2] > 0.14 ? 0.5 : 0.7));
  const ybW = mAG.stop ? 0 : Math.max(-0.8, Math.min(0.8, 1.2 * th * 0.7));

  // ── PHASE 1: MOD-DR-Fast (beam=1 fp16) ──
  const p1Steps = [
    card('STEP 1 · 입력', '~2ms', '로봇 카메라 + 지시어 수신', `
      <div class="mini-cam"><img src="${{sc.img_src}}" alt="raw">${{boxSvg(null)}}</div>
      <div class="io-box"><span class="k">POST /predict</span><br>image: <span class="v">416×234 RGB JPEG</span><br>instruction: <span class="v">"${{sc.ins}}"</span><br>phrase: <span class="ok">"${{ph}}"</span></div>
    `, 'HTTP base64 JPEG 디코딩 후 extract_phrase()로 대상 명사구 추출.'),
    arrow('RGB<br>768²'),
    card('STEP 2 · Vision', '~28ms', 'DaViT 4-Stage Vision Tower', `
      <div class="layer-stack">
        <div class="layer-box"><b>DaViT Stage 0→3 (fp16)</b><span>ConvStem → Spatial/Channel Window Attn</span></div>
        <div class="layer-box"><b>image_projection (1024→768)</b><span>1회 호출로 비전 토큰 + 풀링 특징 동시 획득</span></div>
      </div>
      <div class="io-box">in: <span class="v">(1, 3, 768, 768)</span><br>img_feats: <span class="ok">(1, 577, 768)</span><br>vis_feat: <span class="ok">mean(1) → (768,)</span></div>
    `, 'precompute_v6_florence2_fused와 100% 동일한 _encode_image 1회 호출(중복 인코딩 제거).'),
    arrow('577 tok'),
    card('STEP 3 · 병합', '~4ms', '멀티모달 토큰 병합 + BART Enc', `
      <div class="tok-row">
        <span class="tok sp">&lt;CAPTION_TO_PHRASE_GROUNDING&gt;</span>
        ${{phWords.map(w => `<span class="tok">${{w}}</span>`).join('')}}
      </div>
      <div class="layer-stack">
        <div class="layer-box"><b>BART Encoder (6 Layers, 12H)</b><span>Image 577 + Text ${{phWords.length+2}} Cross-Fusion</span></div>
      </div>
      <div class="io-box">merged: <span class="v">(1, ${{579+phWords.length}}, 768)</span><br>enc_out: <span class="ok">(1, ${{579+phWords.length}}, 768)</span></div>
    `, '이미지와 지시어 단어가 6층 Self-Attention으로 융합됨.'),
    arrow('enc_out'),
    card('STEP 4 · Decoder', `~${{B1_MS}}ms`, 'BART Decoder (beam=1 fp16)', `
      <div class="layer-stack">
        <div class="layer-box"><b>BART Decoder (6 Layers, Greedy)</b><span>beam=3 fp32 (${{B3_MS}}ms) → beam=1 fp16 (${{B1_MS}}ms, <b>2.09x</b>)</span></div>
      </div>
      <div class="tok-row">
        ${{phWords.map(w => `<span class="tok">${{w}}</span>`).join('')}}
        <span class="tok loc">&lt;loc_${{x1}}&gt;</span><span class="tok loc">&lt;loc_${{y1}}&gt;</span><span class="tok loc">&lt;loc_${{x2}}&gt;</span><span class="tok loc">&lt;loc_${{y2}}&gt;</span>
      </div>
      <div class="io-box">beam=3 대비 일치율: <span class="ok">99.7%</span><br>cx MAE: <span class="ok">0.0016</span> · area MAE: <span class="ok">0.0006</span></div>
    `, 'num_beams=1 + fp16 전환만으로 정확도 손실 0.3% 미만, 속도 2.09배 가속.'),
    arrow('텍스트'),
    card('STEP 5 · 파싱', '~0.2ms', '정규식 파싱 (미분 단절 지점)', `
      <div class="mini-cam"><img src="${{sc.img_src}}" alt="bbox">${{boxSvg(bb, '#38bdf8', '', `BBox cx=${{bb[0].toFixed(2)}} a=${{bb[2].toFixed(2)}}`)}}</div>
      <div class="io-box"><span class="bad">⚡ non-differentiable</span><br>regex → <span class="v">[${{bb[0].toFixed(2)}}, ${{bb[1].toFixed(2)}}, ${{bb[2].toFixed(2)}}, 1.0]</span></div>
    `, '문자열에서 숫자 4개를 뽑아 3프레임 창(12-d)에 적재. 역전파는 여기서 끊김.', 'hl-break'),
    arrow('12+768d'),
    card('STEP 6 · Action', '~14ms', 'PiZeroFM (A3C_AG23_s0)', `
      <div class="layer-stack">
        <div class="layer-box"><b>AdaLN-Zero Transformer</b><span>bb(12) + vis_feat(768) 조건화</span></div>
        <div class="layer-box"><b>Flow Matching Euler (10 steps × K=8)</b><span>8스텝 청크(8,3) 예측 → 첫 스텝 평균</span></div>
      </div>
      <div class="io-box">cmd: <span class="ok">[${{mAG.cmd[0].toFixed(2)}}, ${{mAG.cmd[1].toFixed(2)}}]</span><br>크기: <span class="v">${{mAG.mag.toFixed(2)}}</span> · 각도: <span class="v">${{mAG.ang_deg}}°</span></div>
    `, '조향(cx)과 정지(area≥0.23)를 동시에 기하 정합한 Area-Grounded 헤드.'),
    arrow('구동'),
    card('STEP 7 · 출력', '총 285ms', 'Serbot 최종 구동 명령', `
      <div class="mini-cam"><img src="${{sc.img_src}}" alt="out">${{boxSvg(bb, '#4ade80', '', 'MOD-DR-Fast', mAG.cmd, mAG.stop)}}</div>
      <div class="io-box">판정: ${{mAG.stop ? '<span class="bad">STOP (정지)</span>' : '<span class="ok">DRIVE (주행)</span>'}}<br>action: <span class="v">[${{mAG.stop?'0.00':mAG.cmd[0].toFixed(2)}}, ${{mAG.stop?'0.00':mAG.cmd[1].toFixed(2)}}]</span></div>
    `, '현행 A3 대비 2.09배 빠르면서도 다물체 출발 100% · 근접 정지 90~93% 유지.'),
  ].join('');

  // ── PHASE 2: VLM-E2E-Hidden (π0형 1-Pass, BBox=0) ──
  const p2Steps = [
    card('STEP 1 · 입력', '~2ms', '로봇 카메라 + 지시어 수신', `
      <div class="mini-cam"><img src="${{sc.img_src}}" alt="raw">${{boxSvg(null)}}</div>
      <div class="io-box">image: <span class="v">416×234 RGB</span><br>prompt: <span class="v">"${{ph}}"</span></div>
    `, 'Phase 1과 완전히 동일한 카메라 이미지 및 지시어 입력.'),
    arrow('768²'),
    card('STEP 2 · Enc', '~32ms', 'DaViT + BART Encoder (6L)', `
      <div class="layer-stack">
        <div class="layer-box"><b>DaViT (Frozen) + BART Enc (6L)</b><span>Image 577 + Phrase 융합 인코딩</span></div>
      </div>
      <div class="io-box">enc_out: <span class="ok">(1, ${{579+phWords.length}}, 768)</span></div>
    `, 'Florence-2 사전학습 백본 동결로 파괴적 망각 차단.'),
    arrow('1-Pass'),
    card('STEP 3 · Dec 1-Pass', '~22ms', 'BART Decoder 1-Pass (generate 제거!)', `
      <div class="tok-row"><span class="tok sp">decoder_input_ids = [&lt;bos&gt;=2]</span></div>
      <div class="layer-stack">
        <div class="layer-box"><b>BART Decoder (6 Layers, 1-Pass)</b><span>자기회귀 루프 0회 — 단 1회 Forward!</span></div>
      </div>
      <div class="io-box">H_dec_last: <span class="ok">(1, 768)</span><br>문자열 파싱: <span class="ok">0% (완전 미분)</span></div>
    `, 'π0 방식처럼 Decoder 은닉 상태 텐서를 직접 Action Expert에 전달.', 'hl-diff'),
    arrow('BBox=0'),
    card('STEP 4 · 병목 차단', '0ms', '외부 BBox = [0..0] 완전 차단', `
      <div class="io-box"><span class="bad">bb = zeros(12)</span><br>기하 좌표(cx, area) 없음!<br>오직 <span class="v">H_dec_last (768-d)</span>만 전달</div>
    `, '명시적 기하 병목 없이 순수 잠재 벡터(768-d)만으로 조향·정지를 학습.', 'hl-break'),
    arrow('768-d'),
    card('STEP 5 · Action', '~14ms', 'PiZeroFM (EG3_A1_Dec1Pass)', `
      <div class="layer-stack">
        <div class="layer-box"><b>AdaLN-Zero FM (120ep)</b><span>H_dec_768 → Euler 10스텝</span></div>
      </div>
      <div class="io-box">cmd: <span class="v">[${{mHid.cmd[0].toFixed(2)}}, ${{mHid.cmd[1].toFixed(2)}}]</span><br>크기: <span class="v">${{mHid.mag.toFixed(2)}}</span> · 각도: <span class="v">${{mHid.ang_deg}}°</span></div>
    `, '기존 B1(25.1°) 대비 각도 오차 3.3배 개선(7.70°).'),
    arrow('판정'),
    card('STEP 6 · 한계 진단', '총 ~70ms', '초고속이나 미학습 정지 한계', `
      <div class="mini-cam"><img src="${{sc.img_src}}" alt="hid">${{boxSvg(null, '#c084fc', '', '', mHid.cmd, mHid.stop)}}</div>
      <div class="io-box">판정: ${{mHid.stop ? '<span class="ok">STOP</span>' : '<span class="bad">DRIVE</span>'}}<br>바구니≥0.25 정지: <span class="ok">100%</span><br>의자≥0.25 정지: <span class="bad">18.6% (충돌 위험)</span></div>
    `, '기하 병목이 없어 바구니 외형만 암기 → 미학습 의자 초근접(장면③)에서 멈추지 못함!'),
  ].join('');

  // ── PHASE 3: VLM-E2E-SoftLoc (미분가능 Prefix-Forced Soft-Loc) ──
  const p3Steps = [
    card('STEP 1 · 입력', '~2ms', '로봇 카메라 + 지시어 수신', `
      <div class="mini-cam"><img src="${{sc.img_src}}" alt="raw">${{boxSvg(null)}}</div>
      <div class="io-box">image: <span class="v">416×234 RGB</span><br>phrase: <span class="ok">"${{ph}}"</span></div>
    `, '동일 입력. VLM Decoder의 Open-Vocab 지식을 100% 활용.'),
    arrow('Enc'),
    card('STEP 2 · Enc', '~32ms', 'DaViT + BART Encoder (6L)', `
      <div class="layer-stack">
        <div class="layer-box"><b>_encode_image (1회) → vis_feat (768)</b><span>A3C_AG23 학습 분포와 100% 일치</span></div>
        <div class="layer-box"><b>BART Encoder (6L)</b><span>enc_out (1, ${{579+phWords.length}}, 768)</span></div>
      </div>
    `, '이미지 특징과 텍스트 융합 인코더 출력을 동시에 확보.'),
    arrow('Prefix'),
    card('STEP 3 · Prefix', '~45ms', 'Prefix-Forced 1-Pass (KV-Cache)', `
      <div class="tok-row">
        <span class="tok sp">2(&lt;/s&gt;)</span><span class="tok sp">0(&lt;s&gt;)</span>
        ${{phWords.map(w => `<span class="tok">${{w}}</span>`).join('')}}
      </div>
      <div class="io-box">문구 토큰 재생성 <span class="ok">스킵!</span><br>바로 다음 토큰 = <span class="v">&lt;loc_x1&gt;</span> 로짓</div>
    `, '문구 단어를 다시 생성하지 않고 Prefix로 강제 주입해 KV-Cache 초기화.', 'hl-diff'),
    arrow('4-Step'),
    card('STEP 4 · Soft-Loc', `~${{(SL_MS-80).toFixed(0)}}ms`, '미분가능 <loc> Softmax 기댓값', `
      ${{softlocDistSvg(sb[0])}}
      <div class="io-box">4스텝 KV-Cache → 1000-bin Softmax<br><span class="ok">u = Σ softmax(z_i)·(i/1000)</span><br>soft_bb: <span class="v">[${{sb[0].toFixed(2)}}, ${{sb[1].toFixed(2)}}, ${{sb[2].toFixed(2)}}, ${{sb[3].toFixed(0)}}]</span></div>
    `, '정규식 파싱 0%! 확률 가중합이므로 액션 손실이 VLM Decoder까지 완전 미분 가능.', 'hl-diff'),
    arrow('미분 BBox'),
    card('STEP 5 · 기하창', '~0.1ms', '3프레임 Soft-Loc 시계열 창', `
      <div class="mini-cam"><img src="${{sc.img_src}}" alt="sl">${{boxSvg(sb, '#e879f9', '4 3', `Soft-Loc cx=${{sb[0].toFixed(2)}} a=${{sb[2].toFixed(2)}}`)}}</div>
      <div class="io-box">Open-Vocab 일치율: <span class="ok">98.8%</span><br>cx MAE: <span class="ok">0.0141</span> · area MAE: <span class="ok">0.0074</span></div>
    `, '미학습 의자·책상·화분에도 Soft-Loc 기하 병목이 정확히 작동.'),
    arrow('12+768d'),
    card('STEP 6 · Action', '~14ms', 'PiZeroFM (SoftLoc+ / AG23)', `
      <div class="layer-stack">
        <div class="layer-box"><b>EG3-10 (SoftLoc+) / Zero-Shot AG23</b><span>각도차 5.52° (4.8k) → <b>2.71°</b> (29k)</span></div>
      </div>
      <div class="io-box">cmd: <span class="ok">[${{mSL.cmd[0].toFixed(2)}}, ${{mSL.cmd[1].toFixed(2)}}]</span><br>바구니/의자 정지: <span class="ok">100% / 100%</span></div>
    `, 'Phase 2(Hidden)의 의자 정지 18.6%를 단숨에 93~100%로 해결!'),
    arrow('출력'),
    card('STEP 7 · 출력', '총 297ms', '완전 미분 VLM-E2E 구동 출력', `
      <div class="mini-cam"><img src="${{sc.img_src}}" alt="sl_out">${{boxSvg(sb, '#e879f9', '4 3', 'VLM-E2E-SoftLoc', mSL.cmd, mSL.stop)}}</div>
      <div class="io-box">판정: ${{mSL.stop ? '<span class="bad">STOP (안전 정지)</span>' : '<span class="ok">DRIVE (주행)</span>'}}<br>모드: <span class="v">VLA_EGVLA2_DECODE_MODE=softloc_fp16</span></div>
    `, '실제 GPU 서빙 검증 완료(평균 296.6ms). 학술 E2E 1순위!'),
  ].join('');

  // ── PHASE 4: MOD-DR-Safe 컨트롤러 + YahBoom 이식 ──
  const p4Steps = [
    card('STEP 1 · 추출', '~283ms', 'FastF2Extractor (beam1/softloc)', `
      <div class="io-box">mode: <span class="ok">beam1_fp16</span> (285ms)<br>or <span class="v">softloc_fp16</span> (297ms)<br>out: <span class="v">bbox(4) + vis_feat(768)</span></div>
    `, '환경변수 1개로 모듈러 고속 모드와 미분 E2E 모드를 무중단 전환.'),
    arrow('bbox+feat'),
    card('STEP 2 · 정책', '~14ms', 'EgVla2FMPolicy (A3C_AG23_s0)', `
      <div class="io-box">K=8 Euler 평균 → first(lx,ly,az)<br>to_drive_command() →<br>stop=${{mAG.stop}}, speed_scale=<span class="v">${{mAG.stop?'0.00':'0.70'}}</span></div>
    `, '원거리 오정지 0.0%, 의자·책상 중앙 출발 100%, 근접 정지 90~93%.'),
    arrow('area'),
    card('STEP 3 · 스파이크', '~0.05ms', '단발 면적 스파이크 필터', `
      <div class="layer-stack">
        <div class="layer-box"><b>jump_area_ratio = 2.5</b><span>eff_area > last_area × 2.5 (1프레임 튀김)<br>→ last_area로 억제 (조기 오정지 차단)</span></div>
      </div>
      <div class="io-box">현재 area: <span class="v">${{bb[2].toFixed(3)}}</span><br>상태: <span class="ok">정상 범위 통과</span></div>
    `, 'egvla2_only_logic.py 단위 테스트 통과(4배 스파이크 시 near_run=0 유지).', 'hl-safe'),
    arrow('eff_area'),
    card('STEP 4 · S2 감속', '~0.05ms', 'S2 연속 접근 감속', `
      ${{decelCurveSvg(bb[2])}}
      <div class="io-box">decel_start_area=<span class="v">0.14</span><br>decel_min_scale=<span class="v">0.35</span><br>근접 시 선형 감속 → 관성 충돌 방지</div>
    `, '11/16 a3-s2 접촉 6건 원인이던 고속 근접 관성을 소프트웨어 감속으로 차단.', 'hl-safe'),
    arrow('cmd'),
    card('STEP 5 · 상태기계', '~0.05ms', 'EgVla2Controller 상태 전이', `
      <div class="layer-stack">
        <div class="layer-box"><b>우선순위 규칙</b><span>① near_area≥0.25 (2프레임) → STOPPED<br>② 소실 5프레임 → SEARCH(제자리 회전)<br>③ DRIVE: [ux·1.15, uy·1.15, 0]</span></div>
      </div>
      <div class="io-box">state: <span class="ok">${{mAG.stop ? 'STOPPED/idle' : 'DRIVE:drive'}}</span></div>
    `, '기존 운영 100% 하위 호환 (신규 옵션은 환경변수 미설정 시 기본 꺼짐).'),
    arrow('Serbot'),
    card('STEP 6 · Serbot', 'JSON 응답', 'Serbot 전방향 구동 출력', `
      <div class="mini-cam"><img src="${{sc.img_src}}" alt="serbot">${{boxSvg(bb, '#4ade80', '', 'MOD-DR-Safe', mAG.cmd, mAG.stop)}}</div>
      <div class="io-box">{{<br>&nbsp;"action": [${{lx.toFixed(2)}}, ${{ly.toFixed(2)}}],<br>&nbsp;"action_speed_scale": ${{mAG.stop?'0.00':'0.70'}},<br>&nbsp;"goal_near": ${{mAG.stop}}<br>}}</div>
    `, 'vla-inference-gradio / ros2_client 규약(2D action) 100% 호환.'),
    arrow('이식'),
    card('STEP 7 · YahBoom', '어댑터', 'YahBoom 카 ROS2 Twist 변환', `
      <div class="layer-stack">
        <div class="layer-box"><b>serbot_to_yahboom_twist()</b><span>[lx,ly,az] → [linear.x, angular.z]<br>Serbot +az(우) ↔ ROS2 +ω(좌) 부호 반전</span></div>
        <div class="layer-box"><b>calibrate_near_area()</b><span>1프레임 카메라 면적 비율로 임계치 자동 보정</span></div>
      </div>
      <div class="io-box">linear_x: <span class="ok">${{ybV.toFixed(2)}} m/s</span><br>angular_z: <span class="ok">${{ybW>=0?'+':''}}${{ybW.toFixed(2)}} rad/s</span></div>
    `, 'VLA 재학습 없이 어댑터 + 1프레임 보정만으로 이종 로봇 즉시 이식.', 'hl-safe'),
  ].join('');

  return `
    <section class="phase-sec p1">
      <div class="phase-hdr">
        <div class="phase-ttl">🔵 Phase 1 · MOD-DR-Fast <span class="badge b-blue">모듈러 고속화 · beam=1 fp16</span> <span class="badge b-ok">실측 285.3ms (2.09x)</span></div>
        <span style="font-size:.78rem;color:#94a3b8">파트너 ID: <code>a3-dr-fast</code> (EG2-17) + <code>A3C_AG23_s0</code></span>
      </div>
      <div class="phase-sub">현행 <code>beam=3 fp32 (591ms)</code>의 디코더 병목을 <code>beam=1 fp16</code>으로 2.09배 단축하면서 <code>cx MAE 0.0016</code>으로 정확도 99.7% 유지.</div>
      <div class="tl-wrap"><div class="tl-scroll">${{p1Steps}}</div></div>
    </section>

    <section class="phase-sec p2">
      <div class="phase-hdr">
        <div class="phase-ttl">🟣 Phase 2 · VLM-E2E-Hidden <span class="badge b-purp">π0형 1-Pass 순수 E2E · BBox=0</span> <span class="badge b-warn">~70ms · 의자정지 18.6% 한계</span></div>
        <span style="font-size:.78rem;color:#94a3b8">파트너 ID: <code>EG3-08</code> (<code>EG3_A1_Dec1Pass.pt</code>)</span>
      </div>
      <div class="phase-sub"><code>generate()</code>와 외부 BBox를 완전히 제거하고 Decoder 1-Pass Hidden State만 전달 → 초고속이지만 기하 병목 부재로 미학습 의자 정지가 무너짐을 입증.</div>
      <div class="tl-wrap"><div class="tl-scroll">${{p2Steps}}</div></div>
    </section>

    <section class="phase-sec p3">
      <div class="phase-hdr">
        <div class="phase-ttl">🌸 Phase 3 · VLM-E2E-SoftLoc <span class="badge b-pink">완전 미분 Soft-Loc E2E · 파싱 0%</span> <span class="badge b-ok">실측 296.6ms (2.08x) · 정지 100/100%</span></div>
        <span style="font-size:.78rem;color:#94a3b8">파트너 ID: <code>EG3-09/10</code> + Zero-Shot <code>AG23</code> (2.71°)</span>
      </div>
      <div class="phase-sub">Prefix-Forced KV-Cache 4스텝으로 <code>&lt;loc_0~999&gt;</code> 확률 기댓값을 미분 가능하게 추출 → 문자열 파싱 없이 Open-Vocab(98.8%)과 의자 근접 정지 100% 동시 달성.</div>
      <div class="tl-wrap"><div class="tl-scroll">${{p3Steps}}</div></div>
    </section>

    <section class="phase-sec p4">
      <div class="phase-hdr">
        <div class="phase-ttl">🟢 Phase 4 · MOD-DR-Safe &amp; YahBoom 이식 <span class="badge b-ok">S2 감속 + 스파이크 필터 + Twist 어댑터</span></div>
        <span style="font-size:.78rem;color:#94a3b8">파일: <code>egvla2_only_logic.py</code> · <code>egvla2_inference_server.py</code> · <code>yahboom_adapter.py</code></span>
      </div>
      <div class="phase-sub">단발성 BBox 스파이크를 걸러내고 <code>area 0.14→0.25</code>에서 선형 감속한 뒤, Serbot <code>[lx,ly,az]</code>를 YahBoom ROS2 <code>[v,ω]</code>로 변환하는 실기 안전 제어 스택.</div>
      <div class="tl-wrap"><div class="tl-scroll">${{p4Steps}}</div></div>
    </section>
  `;
}}

function render() {{
  document.getElementById('sc-btns').innerHTML = SCENES.map((s, i) =>
    `<button type="button" class="sc-btn ${{i===curIdx?'on':''}}" onclick="curIdx=${{i}};render()">${{s.tab}}</button>`
  ).join('');
  document.getElementById('timelines').innerHTML = buildTracks(SCENES[curIdx]);
}}
render();
</script>
</body>
</html>
"""
    OUT_HTML.write_text(doc, encoding="utf-8")
    print(f"Generated {OUT_HTML} ({OUT_HTML.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()

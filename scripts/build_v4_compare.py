#!/usr/bin/env python3
"""V4 · V4R vs A3 · a3-s2 · a3-ag23 정확한 차이 비교 HTML 생성기.

입력:
  - docs/v5/egvla2_fm/v4_vs_a3_compare.json (실기 6개 장면 위 A3/a3-s2/a3-ag23/V4/V4R 실측 추론 결과)
  - docs/v5/egvla2_fm/t7_all_seeds.json (T7 452세션 10시드 통계)
출력:
  - docs/v5/v4_vs_a3_compare.html
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "docs/v5/egvla2_fm/v4_vs_a3_compare.json"
T7_PATH = ROOT / "docs/v5/egvla2_fm/t7_all_seeds.json"
OUT_HTML = ROOT / "docs/v5/v4_vs_a3_compare.html"


def main() -> None:
    scenes = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    t7 = json.loads(T7_PATH.read_text(encoding="utf-8"))

    v4_s = t7["V4"]
    v4r_s = t7["V4R"]
    v0_s = t7["V0"]

    scenes_js = json.dumps(scenes, ensure_ascii=False)

    html_doc = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>V4 · V4R vs A3 계열 — 무엇이 어떻게 다른가?</title>
<style>
:root{{--bg:#0b1120;--card:#111a2e;--head:#16223b;--fg:#e2e8f0;--mut:#94a3b8;--line:#233252;--acc:#38bdf8;--ok:#4ade80;--warn:#fbbf24;--bad:#f87171;--purp:#c084fc;--pink:#f472b6;--ora:#fb923c}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,'Noto Sans KR',sans-serif}}
main{{max-width:1320px;margin:0 auto;padding:22px 18px 80px}}
h1{{font-size:1.6rem;margin:0 0 6px;color:#f8fafc}}
h2{{font-size:1.18rem;margin:28px 0 10px;padding-bottom:6px;border-bottom:2px solid #233252;color:#fcd34d}}
p.sub{{margin:0 0 16px;color:var(--mut);font-size:.9rem}}
a{{color:var(--acc)}}
code{{background:#1e293b;padding:2px 6px;border-radius:4px;font-size:.82em;color:#e2e8f0}}
.nav-top{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px;font-size:.82rem}}
.nav-top a{{border:1px solid var(--line);background:var(--card);padding:4px 10px;border-radius:99px;text-decoration:none}}
.kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin:14px 0}}
.kpi-card{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:13px 15px}}
.kpi-card .k-tag{{font-size:.73rem;font-weight:800;color:var(--acc);text-transform:uppercase;letter-spacing:.04em}}
.kpi-card .k-ttl{{font-size:1rem;font-weight:700;margin:3px 0 6px;color:#fff}}
.kpi-card .k-body{{font-size:.83rem;color:var(--mut);line-height:1.55}}
.kpi-card .k-body b{{color:var(--fg)}}

.arch-grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin:12px 0}}
@media(max-width:860px){{.arch-grid{{grid-template-columns:1fr}}}}
.arch-box{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px}}
.arch-box.v4{{border-color:var(--purp)}}
.arch-box.a3{{border-color:var(--acc)}}
.arch-ttl{{font-weight:800;font-size:1rem;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center}}
.flow-row{{display:flex;flex-wrap:wrap;align-items:center;gap:6px;margin:10px 0}}
.fnode{{background:var(--head);border:1px solid var(--line);border-radius:8px;padding:6px 10px;font-size:.78rem;line-height:1.35}}
.fnode b{{display:block;color:#fff}}
.farr{{color:var(--mut);font-weight:700}}

table{{width:100%;border-collapse:collapse;font-size:.83rem;background:var(--card);border-radius:10px;overflow:hidden;margin:10px 0}}
th,td{{border:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}}
th{{background:var(--head);color:#cbd5e1;font-size:.78rem}}
.badge{{display:inline-block;padding:2px 8px;border-radius:99px;font-size:.73rem;font-weight:700}}
.b-ok{{background:#14532d;color:#86efac}}
.b-warn{{background:#451a03;color:#fcd34d}}
.b-bad{{background:#450a0a;color:#fca5a5}}
.b-blue{{background:#0c4a6e;color:#7dd3fc}}

.studio{{background:var(--card);border:2px solid var(--purp);border-radius:14px;padding:16px 18px;margin:18px 0}}
.tabs{{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0}}
.tab-btn{{background:var(--head);border:1px solid var(--line);color:var(--mut);padding:6px 12px;border-radius:8px;font-size:.82rem;font-weight:700;cursor:pointer}}
.tab-btn.on{{background:var(--purp);color:#0b1120;border-color:var(--purp)}}
.prompt-bar{{display:flex;flex-wrap:wrap;gap:8px;align-items:center;background:#0a1220;border:1px solid var(--line);padding:9px 12px;border-radius:10px;margin:10px 0}}
.prompt-bar span.lbl{{font-size:.8rem;font-weight:700;color:#fcd34d}}
.p-btn{{background:#16223b;border:1px solid #334155;color:#cbd5e1;padding:5px 11px;border-radius:7px;font-size:.8rem;font-weight:600;cursor:pointer}}
.p-btn.on{{background:#0284c7;border-color:#38bdf8;color:#fff}}
.p-btn.neg.on{{background:#be123c;border-color:#fb7185;color:#fff}}

.sim-grid{{display:grid;grid-template-columns:460px 1fr;gap:16px;align-items:start;margin-top:12px}}
@media(max-width:960px){{.sim-grid{{grid-template-columns:1fr}}}}
.cam-card{{background:#070c17;border:1px solid var(--line);border-radius:12px;padding:10px}}
.cam-wrap{{position:relative;width:100%;aspect-ratio:16/9;border-radius:8px;overflow:hidden;background:#000}}
.cam-wrap img{{width:100%;height:100%;object-fit:cover;display:block}}
.cam-wrap svg{{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}}
.cam-leg{{font-size:.76rem;color:var(--mut);margin-top:8px;line-height:1.55}}
.note-box{{border-left:4px solid var(--warn);background:#1c1917;padding:10px 14px;border-radius:0 10px 10px 0;font-size:.85rem;margin:10px 0;line-height:1.6}}
</style>
</head>
<body>
<main>
  <div class="nav-top">
    <a href="family_arch_compare.html">🔀 3계열(MOD·ENC-E2E·VLM-E2E) 비교</a>
    <a href="research_story_egvla2.html#ch81">← CH81 V4R·서빙 브리핑</a>
    <a href="research_story_egvla2.html#ch82">← CH82 실기·재생 대조</a>
    <a href="model_registry.html#decision-studio">🧠 모델 레지스트리 스튜디오</a>
    <a href="object_test_compare_1113.html">🎮 11/13·11/16 동시 재생기</a>
  </div>

  <h1>V4 · V4R (<code>MOD-V4/V4R</code>) vs A3 계열 (<code>MOD-A3</code>) — 정확히 무엇이 다른가?</h1>
  <p class="sub">
    두 모델군은 모두 <b>모듈러 2단계(<code>MOD-*</code>, 비-E2E)</b> 계열이며 <b>대체 관계가 아니라 역할이 다릅니다.</b>
    A3 계열(<code>a3</code> / <code>MOD-A3</code>, <code>a3-s2</code> / <code>MOD-A3-s2</code>, <code>a3-ag23</code> / <code>MOD-A3-AG</code>)은 <b>지시어 그라운딩으로 새 물체까지 주행</b>하는 모델이고,
    V4 · V4R(<code>EG4-04</code> / <code>MOD-V4</code>, <code>EG4-07</code> / <code>MOD-V4R</code>)은 <b>텍스트 조건화(언어 게이트)로 '목표가 아니면 제자리 정지 + 박스 오검출 방어'</b>를 검증하는 모델입니다.
  </p>

  <div class="kpi-grid">
    <div class="kpi-card">
      <div class="k-tag">차이 ① · 그라운딩(박스) 대상</div>
      <div class="k-ttl">지시어 박스 vs 'gray basket' 고정</div>
      <div class="k-body">
        • <b>A3 계열</b>: 요청 <code>instruction</code>(의자·책상·화분 등)으로 Florence-2 박스를 찾음<br>
        • <b>V4 · V4R</b>: 박스는 <b>항상 <code>"gray basket"</code> 고정</b>으로 찾고, <code>instruction</code>은 <b>헤드 내부 텍스트 특징(768-d)</b>으로만 전달
      </div>
    </div>
    <div class="kpi-card">
      <div class="k-tag">차이 ② · 언어 이해 (Language Gate)</div>
      <div class="k-ttl">텍스트 무시 vs 비목표 99.5% 정지</div>
      <div class="k-body">
        • <b>A3 계열</b>: 액션 헤드에 텍스트 입력이 없음(<code>fdim=768</code>). <code>"red cone"</code>이라고 해도 박스가 잡히면 출발함<br>
        • <b>V4 · V4R</b>: 헤드 입력 <code>fdim=1536</code>(이미지 768 + 문구 768). 바구니가 아닌 지시어는 <b>99.5% 제자리 정지</b>
      </div>
    </div>
    <div class="kpi-card">
      <div class="k-tag">차이 ③ · 다른 물체 주행 가능 여부</div>
      <div class="k-ttl">의자·책상 주행 vs 바구니 전용 정지</div>
      <div class="k-body">
        • <b>A3 계열(s2·ag23)</b>: 의자·책상 중앙 <b>100% 출발·주행</b><br>
        • <b>V4 · V4R</b>: 의자·책상·화분 지시 시 <b>출발하지 않고 정지하는 것이 설계된 정상 동작</b> (실패가 아님)
      </div>
    </div>
    <div class="kpi-card">
      <div class="k-tag">차이 ④ · V4 vs V4R (박스 오검출 방어)</div>
      <div class="k-ttl">시드 편차(42~93%) vs 안정(94.3%)</div>
      <div class="k-body">
        • <b>박스 좌우 반전 테스트</b>(T7 452세션, 10시드 평균): A3 계열(시각만 의존 시) <code>9%</code> → V4 <b>{v4_s['m_lat']['mean']*100:.1f}%</b>(최저 {v4_s['m_lat']['min']*100:.0f}%, 편차 큼) → <b>V4R {v4r_s['m_lat']['mean']*100:.1f}%</b>(최저 {v4r_s['m_lat']['min']*100:.0f}%, SD {v4r_s['m_lat']['std']*100:.1f}%p)
      </div>
    </div>
  </div>

  <h2>1. 아키텍처 &amp; 서빙 파이프라인 비교</h2>
  <div class="arch-grid">
    <div class="arch-box a3">
      <div class="arch-ttl"><span>🔵 A3 계열 (<code>a3</code>, <code>a3-s2</code>, <code>a3-ag23</code>)</span><span class="badge b-blue">fdim = 768</span></div>
      <div class="flow-row">
        <div class="fnode"><b>입력 이미지 + 지시어</b><code>phrase_from_request=true</code></div>
        <div class="farr">→</div>
        <div class="fnode"><b>Florence-2 그라운딩</b>지시어 문구로 BBox 검출<br>(의자·책상·바구니 모두 검출)</div>
        <div class="farr">→</div>
        <div class="fnode"><b>PiZeroFM (12, 768)</b>BBox(12) + 이미지(768)<br><i>텍스트 입력 없음</i></div>
      </div>
      <div style="font-size:.8rem;color:var(--mut)">
        • <b>강점</b>: Open-Vocabulary — 지시어만 바꾸면 학습 때 못 본 의자·책상으로도 주행 가능.<br>
        • <b>약점</b>: 헤드가 지시어 의미를 모르므로, 그라운더가 엉뚱한 물체를 잡거나 지시어가 틀려도 그대로 감.
      </div>
    </div>

    <div class="arch-box v4">
      <div class="arch-ttl"><span>🟣 V4 · V4R (<code>EG4-04_s1</code>, <code>EG4-07_s1</code>)</span><span class="badge b-ok">fdim = 1536</span></div>
      <div class="flow-row">
        <div class="fnode"><b>입력 이미지</b></div>
        <div class="farr">→</div>
        <div class="fnode"><b>Florence-2 그라운딩</b>박스는 항상 <code>"gray basket"</code> 고정<br>+ 융합 인코더로 <b>지시어 특징(768)</b> 추출</div>
        <div class="farr">→</div>
        <div class="fnode"><b>TextCondPolicy (12, 1536)</b>BBox(12) + [이미지 768 | <b>문구 768</b>]<br>목표 일치 시 주행 / 불일치 시 정지</div>
      </div>
      <div style="font-size:.8rem;color:var(--mut)">
        • <b>강점</b>: <b>① 언어 게이트</b>(목표 아닌 지시어 99.5% 정지) + <b>② V4R 박스 훼손 증강</b>(박스 드롭30·지터20·반전10·정상40)으로 오검출 방어.<br>
        • <b>한계</b>: 학습 데이터가 바구니뿐(프레임 1/9 = 3,377장)이고 박스가 바구니에 고정되어 있어 다른 물체로 주행 못 함.
      </div>
    </div>
  </div>

  <h2>2. 실제 로봇 세션 6개 장면 위에서의 동작 비교 (GPU 실측 시뮬레이터)</h2>
  <div class="studio">
    <p style="margin:0 0 8px;font-size:.86rem;color:#cbd5e1">
      실제 11/13 · 11/16 로봇 카메라 사진 위에서 <b>A3 계열</b>과 <b>V4 · V4R (시드 1)</b>이 각각 어떤 박스를 보고, 지시어를 바꿨을 때 어떻게 반응하는지 직접 눌러 확인하세요.
    </p>

    <div class="tabs" id="sc-tabs"></div>

    <div class="prompt-bar">
      <span class="lbl">🔤 V4/V4R에 넣을 지시어 조건 선택:</span>
      <button type="button" class="p-btn on" data-p="orig" id="btn-p-orig">① 장면 원본 지시어</button>
      <button type="button" class="p-btn" data-p="basket" id="btn-p-basket">② "go to the gray basket" (긍정)</button>
      <button type="button" class="p-btn neg" data-p="neg" id="btn-p-neg">③ "go to the red cone" (부정/미학습)</button>
      <button type="button" class="p-btn" data-p="mir" id="btn-p-mir">④ 박스 좌우 반전 (오검출 테스트)</button>
    </div>

    <div class="sim-grid">
      <div class="cam-card">
        <div class="cam-wrap">
          <img id="cam-img" src="" alt="실기 장면">
          <svg id="cam-svg" viewBox="0 0 416 234"></svg>
        </div>
        <div class="cam-leg" id="cam-leg"></div>
      </div>
      <div>
        <div id="sc-desc" style="font-size:.86rem;margin-bottom:10px;line-height:1.55"></div>
        <table>
          <thead>
            <tr>
              <th>모델</th>
              <th>참조하는 BBox</th>
              <th>출력 명령 <code>[lx, ly]</code> (크기)</th>
              <th>판단 결과</th>
              <th>왜 이렇게 나왔는가?</th>
            </tr>
          </thead>
          <tbody id="cmp-tbody"></tbody>
        </table>
      </div>
    </div>
  </div>

  <h2>3. 모델별 전체 스펙 · 오프라인 · 실기 성능 비교표</h2>
  <table>
    <thead>
      <tr>
        <th>비교 항목</th>
        <th><code>a3</code> · <code>MOD-A3</code> (EG2-09, 현 배포)</th>
        <th><code>a3-s2</code> · <code>MOD-A3-s2</code> (EG2-14, 11/16 실기)</th>
        <th><code>a3-ag23</code> · <code>MOD-A3-AG</code> (EG2-16, 신규 후보)</th>
        <th><code>V4</code> · <code>MOD-V4</code> (EG4-04, 시드1)</th>
        <th><code>V4R</code> · <code>MOD-V4R</code> (EG4-07, 시드1)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td><b>체크포인트 파일</b></td>
        <td><code>C_pi0_servolabel1.0_drop30_e120.pt</code></td>
        <td><code>A3C_D30_s2.pt</code></td>
        <td><code>A3C_AG23_s0.pt</code></td>
        <td><code>EG4-04_V4_full_s1.pt</code></td>
        <td><code>EG4-07_V4R_full_s1.pt</code></td>
      </tr>
      <tr>
        <td><b>입력 차원 (bdim, fdim)</b></td>
        <td>12, 768 (이미지)</td>
        <td>12, 768 (이미지)</td>
        <td>12, 768 (이미지)</td>
        <td>12, <b>1536 (이미지+텍스트)</b></td>
        <td>12, <b>1536 (이미지+텍스트)</b></td>
      </tr>
      <tr>
        <td><b>그라운딩 박스 문구</b></td>
        <td>요청 지시어 (가변)</td>
        <td>요청 지시어 (가변)</td>
        <td>요청 지시어 (가변)</td>
        <td><code>"gray basket"</code> 고정</td>
        <td><code>"gray basket"</code> 고정</td>
      </tr>
      <tr>
        <td><b>학습 데이터 · 기법</b></td>
        <td>300ep 전체 (29,161장)<br>서보 라벨 λ=1.0, 드롭 30%</td>
        <td>300ep 전체 (29,161장)<br>A3와 동일, 시드 2</td>
        <td>300ep 전체 (29,161장)<br>조향(cx)+<b>면적(area) 정지 정합</b>, 특징교체 50%</td>
        <td>300ep 1/9 (3,377장)<br>어휘 40+40 + 부정 20%</td>
        <td>300ep 1/9 (3,377장)<br>V4 + <b>박스 훼손 증강</b>(드롭30·지터20·반전10)</td>
      </tr>
      <tr>
        <td><b>T7 452세션 (A3 대비 각도)</b></td>
        <td>기준 (0.00°)</td>
        <td>0.70° (좌우 98.3%)</td>
        <td>0.54° (좌우 98.7%)</td>
        <td>{v4_s['ang']['mean']:.2f}±{v4_s['ang']['std']:.2f}° (좌우 {v4_s['lat']['mean']*100:.1f}%)</td>
        <td>{v4r_s['ang']['mean']:.2f}±{v4r_s['ang']['std']:.2f}° (좌우 {v4r_s['lat']['mean']*100:.1f}%)</td>
      </tr>
      <tr>
        <td><b>부정 지시어 정지율</b></td>
        <td>불가 (헤드 텍스트 없음)</td>
        <td>불가 (헤드 텍스트 없음)</td>
        <td>불가 (헤드 텍스트 없음)</td>
        <td><span class="badge b-ok">99.5% 정지</span> (긍정 오정지 6.4%)</td>
        <td><span class="badge b-ok">99.5% 정지</span> (긍정 오정지 6.7%)</td>
      </tr>
      <tr>
        <td><b>합성 박스 반전 좌우 일치</b></td>
        <td>~9% (이미지만 의존 시 10%)</td>
        <td>—</td>
        <td>—</td>
        <td><span class="badge b-warn">{v4_s['m_lat']['mean']*100:.1f}%</span> (시드0 42% · 시드1 93%)</td>
        <td><span class="badge b-ok">{v4r_s['m_lat']['mean']*100:.1f}%</span> (최저 {v4r_s['m_lat']['min']*100:.0f}%, 편차 1.5%p)</td>
      </tr>
      <tr>
        <td><b>11/13 의자·책상 중앙 출발</b></td>
        <td><span class="badge b-bad">0/5 (0%)</span></td>
        <td><span class="badge b-ok">5/5 (100%)</span></td>
        <td><span class="badge b-ok">100% (오프라인)</span></td>
        <td>해당 없음 (비바구니 정지 설계)</td>
        <td>해당 없음 (비바구니 정지 설계)</td>
      </tr>
      <tr>
        <td><b>11/16 근접(0.25~0.40) 정지</b></td>
        <td>바구니 80% · 의자 58%</td>
        <td><span class="badge b-bad">바구니 55% · 의자 28%</span> (접촉 6건)</td>
        <td><span class="badge b-ok">바구니 90% · 의자 93%</span></td>
        <td>바구니 0.36에서 0.054 (정지)</td>
        <td>바구니 0.36에서 0.111 (정지)</td>
      </tr>
      <tr>
        <td><b>실기 이력</b></td>
        <td>455/455 + 11/13 49/65</td>
        <td>11/16 45/45 (접촉 7건 보고)</td>
        <td>미검증 (패키지 전달됨)</td>
        <td><b>미검증</b> (동환 테스트 예정)</td>
        <td><b>미검증</b> (패치 v2로 교체 가능)</td>
      </tr>
    </tbody>
  </table>

  <div class="note-box">
    <b>📋 동환 V4/V4R 실기 주행 시 꼭 확인할 체크리스트 (오해 방지용)</b><br>
    1. <b>바구니 지시어(<code>"go to the gray basket"</code>)</b>를 주면 A3처럼 바구니로 주행해야 정상입니다.<br>
    2. <b>의자·책상·화분 지시어(<code>"go to the brown wooden chair"</code> 등)</b>를 주면 <b>출발하지 않고 제자리에 정지해야 정상(성공)</b>입니다! ("의자로 안 가네? 실패!"로 판정하면 안 됩니다.)<br>
    3. <code>switch_family</code>로 V4/V4R을 올릴 때는 <b>반드시 시드 1(<code>_s1.pt</code>)</b>을 쓰고, <code>/config {{"phrase_from_request": true}}</code> 설정 후 <code>/health</code>에서 <code>active_family="egvla2"</code>, <code>phrase_from_request=true</code>, <code>egvla2.variant</code>가 뜨는지 확인하세요. (<code>changeA2_auto_text_ckpt.patch</code> 반영을 위해 서버가 1회 재시작된 상태여야 합니다.)
  </div>
</main>
<script>
(function(){{
  var SC = {scenes_js};
  var curSc = 0;
  var promptMode = "orig";

  function arrow(svg, ox, oy, cmd, col, label) {{
    var mag = Math.hypot(cmd[0], cmd[1]);
    if (mag < 0.2) return false;
    var len = Math.min(mag, 1.0) * 95;
    var ux = cmd[1] / Math.max(mag, 1e-6);
    var uy = -cmd[0] / Math.max(mag, 1e-6);
    var x2 = ox + ux * len, y2 = oy + uy * len;
    var px = -uy, py = ux;
    svg.push('<line x1="'+ox+'" y1="'+oy+'" x2="'+x2.toFixed(1)+'" y2="'+y2.toFixed(1)+'" stroke="#000" stroke-width="5.5" stroke-linecap="round" opacity="0.6"/>');
    svg.push('<line x1="'+ox+'" y1="'+oy+'" x2="'+x2.toFixed(1)+'" y2="'+y2.toFixed(1)+'" stroke="'+col+'" stroke-width="3.5" stroke-linecap="round"/>');
    svg.push('<polygon points="'+(x2+ux*8).toFixed(1)+','+(y2+uy*8).toFixed(1)+' '+(x2+px*5).toFixed(1)+','+(y2+py*5).toFixed(1)+' '+(x2-px*5).toFixed(1)+','+(y2-py*5).toFixed(1)+'" fill="'+col+'" stroke="#000" stroke-width="0.8"/>');
    svg.push('<text x="'+(x2+ux*13).toFixed(1)+'" y="'+(y2+uy*10).toFixed(1)+'" text-anchor="middle" fill="'+col+'" stroke="#000" stroke-width="2.5" paint-order="stroke" font-size="10.5" font-weight="800">'+label+' ('+mag.toFixed(2)+')</text>');
    return true;
  }}

  function boxRect(svg, b, col, dash, tag, dy) {{
    if (!b || b[3] < 0.5) return;
    var W = 416, H = 234;
    var side = Math.sqrt(b[2] * W * H);
    var x0 = b[0] * W - side / 2, y0 = b[1] * H - side / 2;
    svg.push('<rect x="'+x0.toFixed(1)+'" y="'+y0.toFixed(1)+'" width="'+side.toFixed(1)+'" height="'+side.toFixed(1)+'" fill="none" stroke="'+col+'" stroke-width="2.2" '+(dash?'stroke-dasharray="6 3"':'')+'/>');
    var ly = Math.max(13, y0 + dy);
    svg.push('<rect x="'+Math.max(2, x0).toFixed(1)+'" y="'+(ly-12).toFixed(1)+'" width="152" height="14" rx="3" fill="rgba(0,0,0,0.78)"/>');
    svg.push('<text x="'+(Math.max(2, x0)+4).toFixed(1)+'" y="'+(ly-2).toFixed(1)+'" fill="'+col+'" font-size="9.5" font-weight="700">'+tag+' cx='+b[0].toFixed(2)+' area='+b[2].toFixed(3)+'</text>');
  }}

  function render() {{
    var s = SC[curSc];
    document.getElementById('sc-tabs').innerHTML = SC.map(function(x, i) {{
      return '<button type="button" class="tab-btn '+(i===curSc?'on':'')+'" data-i="'+i+'">'+x.tab+'</button>';
    }}).join('');
    document.querySelectorAll('#sc-tabs .tab-btn').forEach(function(b) {{
      b.onclick = function() {{ curSc = parseInt(b.dataset.i, 10); render(); }};
    }});

    document.querySelectorAll('.p-btn').forEach(function(b) {{
      b.classList.toggle('on', b.dataset.p === promptMode);
    }});

    document.getElementById('cam-img').src = s.img_src;
    var svg = [];
    var a3b = s.a3_bbox.slice(), v4b = s.v4_gb_bbox.slice();
    if (promptMode === "mir") {{
      a3b[0] = 1 - a3b[0];
      v4b[0] = 1 - v4b[0];
    }}
    boxRect(svg, a3b, "#facc15", false, "A3 지시어박스", -4);
    boxRect(svg, v4b, "#38bdf8", true, "V4 바구니고정박스", 14);

    var ox = 208, oy = 220;
    svg.push('<circle cx="'+ox+'" cy="'+oy+'" r="4.5" fill="#fff" stroke="#000" stroke-width="1.5"/>');

    var c_a3 = promptMode === "mir" ? s.a3.mir : s.a3.cmd;
    var c_s2 = promptMode === "mir" ? s.a3s2.mir : s.a3s2.cmd;
    var c_ag = promptMode === "mir" ? s.ag23.mir : s.ag23.cmd;
    var v4_key = promptMode === "mir" ? "mir_basket" : promptMode;
    var c_v4 = s.v4[v4_key];
    var c_v4r = s.v4r[v4_key];

    var items = [
      {{name:"A3", col:"#38bdf8", box:"지시어 ("+s.ins+")", res:c_a3, why: promptMode==="mir" ? "반전 박스를 따라 방향 전환" : (c_a3[1]<0.2 ? "시각 외형 지름길로 정지" : "지시어 박스 따라 주행")}},
      {{name:"a3-s2", col:"#fb923c", box:"지시어 ("+s.ins+")", res:c_s2, why: c_s2[1]<0.2 ? "정지" : "지시어 박스 따라 주행 (근접 정지 둔화 주의)"}},
      {{name:"a3-ag23", col:"#4ade80", box:"지시어 ("+s.ins+")", res:c_ag, why: c_ag[1]<0.2 ? "BBox 면적 ≥0.23 기하 정지" : "기하 BBox 조건으로 안정적 출발"}},
      {{name:"V4 (s1)", col:"#c084fc", box:"gray basket 고정 + 텍스트", res:c_v4, why: promptMode==="neg" ? "부정 지시어(red cone) → 언어 게이트 정지" : (promptMode==="orig" && s.ins!=="go to the gray basket" ? "비바구니 지시어 → 언어 게이트 정지(정상)" : (c_v4[1]<0.2 ? "바구니 근접 정지" : "바구니 지시어 + 바구니 박스 주행"))}},
      {{name:"V4R (s1)", col:"#f472b6", box:"gray basket 고정 + 텍스트", res:c_v4r, why: promptMode==="mir" ? "박스 좌우 반전에서도 바구니 주행" : (promptMode==="neg" ? "부정 지시어(red cone) → 언어 게이트 정지" : (promptMode==="orig" && s.ins!=="go to the gray basket" ? "비바구니 지시어 → 언어 게이트 정지(정상)" : (c_v4r[1]<0.2 ? "바구니 근접 정지" : "바구니 지시어 + 바구니 박스 주행")))}}
    ];

    var stops = [];
    items.forEach(function(it) {{
      if (!arrow(svg, ox, oy, it.res[0], it.col, it.name)) {{
        stops.push('<tspan fill="'+it.col+'" font-weight="700">■ '+it.name+'('+it.res[1].toFixed(2)+')</tspan>');
      }}
    }});
    if (stops.length) {{
      svg.push('<rect x="6" y="208" width="404" height="20" rx="4" fill="rgba(0,0,0,0.78)"/>');
      svg.push('<text x="12" y="222" font-size="10">정지: '+stops.join(' · ')+'</text>');
    }}
    document.getElementById('cam-svg').innerHTML = svg.join('');

    var insTxt = promptMode==="orig" ? s.ins : (promptMode==="neg" ? "go to the red cone" : "go to the gray basket");
    document.getElementById('cam-leg').innerHTML =
      '<b>세션:</b> <code>'+s.sid+'</code> · <b>실제 물체:</b> '+s.obj+' ('+s.pos+')<br>'+
      '<b>노란 실선:</b> A3용 지시어 박스 · <b>하늘 점선:</b> V4/V4R용 <code>"gray basket"</code> 고정 박스';

    document.getElementById('sc-desc').innerHTML =
      '<b>'+s.title+'</b><br>'+
      '<span style="color:var(--mut)">현재 V4/V4R 입력 지시어: <code>"'+insTxt+'"</code>'+(promptMode==="mir"?' (박스 cx 좌우 반전 적용)':'')+'</span>';

    document.getElementById('cmp-tbody').innerHTML = items.map(function(it) {{
      var st = it.res[1] < 0.2;
      return '<tr>'+
        '<td><b style="color:'+it.col+'">'+it.name+'</b></td>'+
        '<td style="font-size:.78rem">'+it.box+'</td>'+
        '<td><code>['+it.res[0][0].toFixed(2)+', '+it.res[0][1].toFixed(2)+']</code> (<b>'+it.res[1].toFixed(3)+'</b>)</td>'+
        '<td>'+(st ? '<span class="badge b-bad">정지 (&lt;0.2)</span>' : '<span class="badge b-ok">이동 (≥0.2)</span>')+'</td>'+
        '<td style="font-size:.8rem;color:#cbd5e1">'+it.why+'</td>'+
      '</tr>';
    }}).join('');
  }}

  document.querySelectorAll('.p-btn').forEach(function(b) {{
    b.onclick = function() {{ promptMode = b.dataset.p; render(); }};
  }});
  render();
}})();
</script>
</body>
</html>
"""
    OUT_HTML.write_text(html_doc, encoding="utf-8")
    print(f"Generated {OUT_HTML} ({len(html_doc)/1024:.1f} KB)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""VLAops 대시보드(docs/v5/vlaops_dashboard.html)에 '🔤 텍스트 조건화' 탭을 (재)생성한다. 멱등: 마커 사이만 교체.
내용: 학습 유효 지표(정의·방향·유효성·이번 값), 주요 피처, 변경사항(감사 기록), 변형별 결과. 값은 docs/v5/egvla2_fm/text_cond_t2.json에서 읽는다.
plan: docs/plans/plan_20261106_text_conditioned_e2e_vla.md. 실행: python3 scripts/add_vlaops_textcond_tab.py"""
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / "docs/v5/vlaops_dashboard.html"
T2 = ROOT / "docs/v5/egvla2_fm/text_cond_t2.json"
VS = ["V0_img_only", "V1_text_const", "V2_paraphrase_aug", "V3_aug_neg"]
LAB = {"V0_img_only": "V0 이미지 전용", "V1_text_const": "V1 문구 고정", "V2_paraphrase_aug": "V2 +패러프레이즈", "V3_aug_neg": "V3 +부정 지시"}
BTN = '<button class="tab-btn" onclick="switchTab(\'tab-textcond\')"><span>🔤 텍스트 조건화·학습 지표</span><span class="tab-count" style="color: #fbbf24;">T0~T2</span></button>'
B0, B1 = "<!--TEXTCOND-BEGIN-->", "<!--TEXTCOND-END-->"


def agg(res, key):
    out = {}
    for v in VS:
        x = [r[key] for k, r in res.items() if k.split("|")[1] == v and key in r and r[key] == r[key]]
        out[v] = (np.mean(x), np.std(x), len(x)) if x else None
    return out


def cell(a, pct=False, nd=1):
    if a is None: return "–"
    return f"{a[0] * 100:.1f}±{a[1] * 100:.1f}%" if pct else f"{a[0]:.{nd}f}±{a[1]:.{nd}f}"


def table(head, rows):
    th = "".join(f"<th>{h}</th>" for h in head)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="ops-table"><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>'


def card(title, inner, sub=""):
    s = f'<div style="font-size:0.76rem;color:#94a3b8;margin-top:10px;line-height:1.6">{sub}</div>' if sub else ""
    return f'<section class="card"><div class="card-header"><div class="card-title"><span>{title}</span></div></div><div style="overflow-x:auto">{inner}</div>{s}</section>'


def panel():
    t2 = json.loads(T2.read_text()) if T2.exists() else {}
    n = len(t2)
    badge = lambda t, c: f'<span class="status-badge {c}">{t}</span>'
    A = {k: agg(t2, k) for k in ("ang_ref", "ang_para", "para_effect", "stop_neg_nobbox", "stop_neg_bbox", "stop_pos_bbox", "text_effect",
                                  "unseen_ang_para", "unseen_para_effect", "unseen_stop_pos", "unseen_stop_neg_nobbox", "unseen_stop_neg_bbox")}
    val = lambda key, pct=False, nd=1: f'<span class="tc-val" data-key="{key}" data-pct="{int(pct)}" data-nd="{nd}">…</span>'
    metrics = table(["지표", "정의", "좋은 방향", "유효성·주의", "V0 · V1 · V2 · V3 (분할×시드 평균±sd, 라이브)"], [
        ["방향 오차(기준 문구)", "첫 스텝 명령과 사람 시연의 각도 차(°), 둘 다 이동 중인 프레임", "↓", badge("유효(변형 간 비교)", "badge-completed") + " 서보 라벨 학습이라 절대값은 이전 19°대와 직접 비교 불가", val("ang_ref")],
        ["방향 오차(패러프레이즈)", "같은 의미 문구 5종에서의 방향 오차", "↓", badge("유효", "badge-completed") + " 기준 문구와 같아야 불변", val("ang_para")],
        ["패러프레이즈 명령 변화", "같은 의미 문구로 바꿨을 때 첫 스텝 명령 변화 크기", "↓ (불변)", badge("유효", "badge-completed") + " V1에서 커지면 문구 고정 학습의 취약성", val("para_effect", nd=3)],
        ["<b>부정 정지율</b>(박스 없음)", "장면에 없는 물체 문구에서 |명령|&lt;0.2 비율", "↑", badge("부분 유효", "badge-running") + " 학습에 쓴 문구면 암기와 구분 불가", val("stop_neg_nobbox", True)],
        ["<b>부정 정지율</b>(박스 남음)", "오검출로 박스가 남은 상황에서의 정지율", "↑", badge("부분 유효", "badge-running") + " 텍스트가 박스를 이기는지의 직접 지표", val("stop_neg_bbox", True)],
        ["정상 문구 정지율", "정상 문구에서 멈춘 비율(오작동)", "↓", badge("유효", "badge-completed"), val("stop_pos_bbox", True)],
        ["<b>미지 부정 정지율</b>(박스 없음)", "학습에 없던 문구(green plant·black laptop)에서의 정지율", "↑", badge("핵심", "badge-completed") + " 교수님 시험 ②의 오프라인 대응. 3분할×3시드, 문구 2개뿐", val("unseen_stop_neg_nobbox", True)],
        ["<b>미지 부정 정지율</b>(박스 남음)", "같은 문구에서 오검출 박스가 남은 상황", "↑", badge("핵심", "badge-completed"), val("unseen_stop_neg_bbox", True)],
        ["미지 패러프레이즈 정지율", "학습에 없던 동의 표현(grey bucket·the bin)에서 정지", "↓", badge("핵심", "badge-completed"), val("unseen_stop_pos", True)],
        ["학습 loss·val loss", "flow-matching MSE", "–", badge("무효", "badge-danger") + " 폐루프 성공·텍스트 사용 여부를 말해주지 못함(CH79·메모리 결론)", "미보고"],
        ["오프라인 지표 전반", "프레임 단위 방향·정지", "–", badge("주의", "badge-running") + " 폐루프 성공을 예측하지 못함. 최종 판단은 실기 5~10회", "—"]])
    feats = table(["피처", "차원", "출처", "텍스트 의존", "비고"], [
        ["이미지 풀링", "768", "Florence-2 이미지 인코더(DaViT) 출력 평균", "없음(cos 1.000)", "현재 A3 헤드 입력"],
        ["텍스트 토큰 풀링", "768", "융합(BART) 인코더 은닉 중 텍스트 토큰 평균", "있음(go to the gray bin 0.773)", "T1·T2 추가 입력. 문구 토큰이 소수라 평균이 뭉개질 수 있음"],
        ["융합 은닉 전체/이미지 토큰 평균", "768", "융합 인코더 은닉", "약함(0.92~0.99)", "이미지 토큰 577개가 지배 → 입력으로 쓰지 않음"],
        ["bbox 창", "12", "[cx,cy,area,has]×3프레임", "있음(문구가 박스를 정함)", "서브골 드롭아웃 30%(없으면 0벡터)"],
        ["서보 라벨 λ", "–", "방향을 cx 서보 방향과 혼합(k=2, max 0.7)", "–", "A3와 동일 λ=1.0, 120 epoch, PiZeroFM"],
        ["부정 라벨", "–", "장면에 없는 물체 문구 → 명령 0", "있음(정답이 문구에 의존)", "V3: 20%, 그중 절반은 박스 유지"]])
    changes = table(["일시", "변경", "상세·소스", "상태"], [
        ["2026-11-07", "T1·T2 학습(4변형×3분할×3시드) 및 미지 문구 평가 추가", "<code>scripts/eval_text_cond_t2.py</code>, 보류 문구 캐시 <code>precompute_text_fused_heldout.py</code>", badge("RUNNING" if n < 36 else "COMPLETED", "badge-running" if n < 36 else "badge-completed")],
        ["2026-11-07", "인코더별 문구 민감도·사진 확인·CH79 탭", "<code>eval_text_encoders.py</code>, <code>eval_text_visual.py</code>, <code>ch79_textcond.py</code> (커밋 2c4c26f3, bb948ea0)", badge("COMPLETED", "badge-completed")],
        ["2026-11-06", "문구 변형 오프라인 점검(T0) + E2E 텍스트 조건화 계획서", "<code>eval_text_robustness.py</code>, <code>plan_20261106_text_conditioned_e2e_vla.md</code> (543b3418, 9498efd0)", badge("COMPLETED", "badge-completed")],
        ["2026-11-07", "라이브 물체 촬영 전달 문서(초안)", "<code>HANDOFF_20261107_TEXT_OBJECT_SWAP_PHOTOS.md</code> — 다른 물체가 없는 장면이라 평가가 약함", badge("대기: 사용자 전달", "badge-running")],
        ["미정", "서버 문구 인자화(<code>egvla2_inference_server.py:143</code> 현재 무시)·T3 수집", "계획서 §2 T3 · 승인 필요", badge("미착수", "badge-danger")]])
    res = table(["지표"] + [LAB[v] for v in VS], [
        ["완료", f"{n}/36", "", "", ""],
        ["방향 오차 기준(°)"] + [cell(A["ang_ref"][v]) for v in VS], ["패러프레이즈 방향 오차(°)"] + [cell(A["ang_para"][v]) for v in VS],
        ["부정 정지율(박스 없음)"] + [cell(A["stop_neg_nobbox"][v], True) for v in VS], ["부정 정지율(박스 남음)"] + [cell(A["stop_neg_bbox"][v], True) for v in VS],
        ["미지 부정 정지율(박스 없음)"] + [cell(A["unseen_stop_neg_nobbox"][v], True) for v in VS], ["미지 부정 정지율(박스 남음)"] + [cell(A["unseen_stop_neg_bbox"][v], True) for v in VS],
        ["미지 패러프레이즈 정지율(↓)"] + [cell(A["unseen_stop_pos"][v], True) for v in VS]]) if t2 else '<div style="font-size:0.84rem">학습 진행 중</div>'
    hero = ('<div style="font-size:0.84rem;line-height:1.8;margin-bottom:6px">교수님(11/05) 질문: 지시문 변경·물체 교체에도 가는가 → <b>텍스트가 행동에 직접 들어가는 E2E</b>가 되는지 오프라인으로 점검. '
            '상세 해설·사진: <a href="research_story.html#ch79" style="color:#38bdf8">CH79 🔤 탭</a> · 계획서 docs/plans/plan_20261106_text_conditioned_e2e_vla.md. 현재 파이프라인은 텍스트→bbox만 텍스트 의존.</div>')
    return (f'{B0}<div id="tab-textcond" class="tab-panel">' + card("🔤 텍스트 조건화(T0~T2) — 개요", hero) + card("📏 학습 관련 유효 지표", metrics, "표의 마지막 열은 분할 3 × 시드 3의 평균±표준편차(V0~V3 순). 단일 학습 설정·3프레임 간격 부분집합. 표본 문구가 적어 '핵심' 표기는 잠정.")
            + LIVE + card("🧬 주요 피처", feats) + card("🛠 변경사항 (감사 기록)", changes) + JS + f'</div>{B1}')



LIVE = """
<section class="card" id="tc-live"><div class="card-header"><div class="card-title"><span>⏱ 학습 진행 (라이브 — 8초마다 <code>egvla2_fm/text_cond_t2.json</code> 갱신)</span></div><span id="tc-state" class="status-badge badge-running">…</span></div>
<div style="background:#0f172a;border:1px solid #1e293b;border-radius:8px;height:14px;overflow:hidden"><div id="tc-bar" style="height:100%;width:0;background:linear-gradient(90deg,#38bdf8,#34d399)"></div></div>
<div id="tc-prog" style="font-size:0.8rem;color:#94a3b8;margin:8px 0">…</div>
<table class="ops-table"><thead><tr><th>지표 (분할×시드 평균±sd)</th><th>V0 이미지 전용</th><th>V1 문구 고정</th><th>V2 +패러프레이즈</th><th>V3 +부정 지시</th></tr></thead><tbody id="tc-res"></tbody></table>
<div style="font-size:0.74rem;color:#64748b;margin-top:6px">서버(<code>python3 -m http.server 8001</code>)로 열어야 갱신됩니다(file:// 불가). 완료 수는 36 = 3분할 × 4변형 × 시드 3.</div></section>

<section class="card"><div class="card-header"><div class="card-title"><span>📊 퍼센트 지표 (36/36 최종)</span></div></div><img src="figures/ch79/text_cond_bars.png" style="width:100%;border-radius:8px"><div style="font-size:0.75rem;color:#94a3b8;margin-top:6px">입력은 타이핑한 텍스트 문구(음성 아님). 부정 = 장면에 없는 물체 문구, 정답 정지. ②가 핵심(V3 82%), ④는 동의 표현에서 오정지(V3 12%).</div></section>
<section class="card"><div class="card-header"><div class="card-title"><span>🧭 학습 흐름 대조 — 같은 사진, 변형마다 무엇이 다른가</span></div></div>
<div style="overflow-x:auto"><table class="ops-table" style="min-width:880px" id="tc-flow"><thead><tr><th style="width:130px"></th><th>V0 이미지 전용 (A3 방식)</th><th>V1 문구 고정</th><th>V2 +패러프레이즈 증강</th><th>V3 +부정 지시</th></tr></thead><tbody>
<tr><td><b>① 입력 사진</b><br><span style="font-size:0.7rem;color:#94a3b8">실제 학습 프레임(데이터셋 H5)</span></td><td colspan="4" id="tc-flow-photo" style="text-align:center">…</td></tr>
<tr><td><b>② 문구</b></td><td style="color:#94a3b8">(문구 입력 없음)</td><td><code>gray basket</code> 고정</td><td>6종 무작위<br><span style="font-size:0.72rem;color:#94a3b8">gray basket · gray bin · gray container · the gray basket · go to the gray bin · basket</span></td><td>V2의 80% + <b>20% 부정</b><br><span style="font-size:0.72rem;color:#fca5a5">red chair · person · blue bottle</span></td></tr>
<tr><td><b>③ 이미지 인코더</b></td><td colspan="4" style="text-align:center;color:#94a3b8">Florence-2 DaViT → 풀링 768-d (문구와 무관 · 4변형 공통)</td></tr>
<tr><td><b>④ 융합 인코더</b></td><td style="color:#64748b">사용 안 함</td><td colspan="3" style="text-align:center;color:#94a3b8">이미지+문구 → 텍스트 토큰 풀링 768-d (문구별로 달라짐)</td></tr>
<tr><td><b>⑤ 헤드 입력</b></td><td>[이미지 768] + bbox 창 12</td><td colspan="3" style="text-align:center">[이미지 768 ; 텍스트 768] + bbox 창 12 <span style="color:#94a3b8">(서브골 드롭아웃 30%)</span></td></tr>
<tr><td><b>⑥ 학습 정답</b><br><span style="font-size:0.7rem;color:#94a3b8">이 사진의 첫 스텝</span></td><td colspan="3" style="text-align:center"><span id="tc-flow-gt">…</span><br><span style="font-size:0.72rem;color:#94a3b8">사람 시연(회색) · 서보 라벨(초록)</span></td><td style="text-align:center"><span id="tc-flow-neg">…</span><br><span style="font-size:0.72rem;color:#fca5a5">부정 문구면 정답 = 정지(0) · 절반은 박스가 남아 있어도</span></td></tr>
<tr><td><b>⑦ 텍스트가 쓰일 이유</b></td><td style="color:#64748b">없음</td><td style="color:#fca5a5">없음 — 문구가 상수(귀무)</td><td style="color:#fbbf24">불변성만 요구</td><td style="color:#34d399"><b>있음</b> — 문구가 바뀌면 정답이 바뀜</td></tr>
</tbody></table></div></section>

<section class="card"><div class="card-header"><div class="card-title"><span>📷 원본 사진으로 비교 — 학습된 헤드는 이 사진에서 어떻게 움직이나 (fold0 시험 프레임, 시드 평균)</span></div></div>
<div style="font-size:0.78rem;color:#94a3b8;margin-bottom:8px;line-height:1.7">화살표: 위 = 전진(lx), 오른쪽 = 우측 이동(ly). 점 = 정지(|명령|&lt;0.2). 회색 = 사람 시연, 초록 = 서보 라벨(학습 정답). 사진의 붉은 상자는 bbox 중심·면적(근사). 문구가 달라도 정상 문구면 화살표가 같아야 하고, 부정 문구면 점(정지)이어야 한다.</div>
<div id="tc-probes">…</div></section>
"""
JS = r"""
<script>
(function(){
  const VS=['V0_img_only','V1_text_const','V2_paraphrase_aug','V3_aug_neg'];
  const KEYS=[['ang_ref','방향 오차 기준(°)',0,1],['ang_para','패러프레이즈 방향 오차(°)',0,1],['para_effect','패러프레이즈 명령 변화',0,3],['stop_neg_nobbox','부정 정지율(박스 없음)',1,1],['stop_neg_bbox','부정 정지율(박스 남음)',1,1],['stop_pos_bbox','정상 문구 정지율(↓)',1,1],['unseen_ang_para','미지 패러프레이즈 방향 오차(°)',0,1],['unseen_stop_pos','미지 패러프레이즈 정지율(↓)',1,1],['unseen_stop_neg_nobbox','미지 부정 정지율(박스 없음)',1,1],['unseen_stop_neg_bbox','미지 부정 정지율(박스 남음)',1,1]];
  const PH=[['ref','gray basket','학습 문구','#34d399'],['para_seen','go to the gray bin','학습한 패러프레이즈','#34d399'],['para_unseen','grey bucket','학습에 없던 패러프레이즈','#fbbf24'],['neg_seen_bbox','red chair + 박스 유지','학습한 부정(오검출 상황)','#fca5a5'],['neg_seen_nobbox','red chair + 박스 없음','학습한 부정','#fca5a5'],['neg_unseen_bbox','green plant + 박스 유지','학습에 없던 부정(오검출 상황)','#f87171'],['neg_unseen_nobbox','green plant + 박스 없음','학습에 없던 부정','#f87171']];
  const mean=a=>a.reduce((x,y)=>x+y,0)/a.length, sd=a=>{const m=mean(a);return Math.sqrt(mean(a.map(x=>(x-m)**2)))};
  function arrow(v,col,sz){sz=sz||58;const c=sz/2,s=sz*0.42,m=Math.hypot(v[0],v[1]);
    if(m<0.2) return `<svg width="${sz}" height="${sz}"><circle cx="${c}" cy="${c}" r="6" fill="${col}"/><text x="${c}" y="${sz-4}" font-size="9" fill="#94a3b8" text-anchor="middle">정지 ${m.toFixed(2)}</text></svg>`;
    const dx=v[1]*s, dy=-v[0]*s, x2=c+dx, y2=c+dy, a=Math.atan2(dy,dx);
    const h=(t)=>`${x2-7*Math.cos(a-t)},${y2-7*Math.sin(a-t)}`;
    return `<svg width="${sz}" height="${sz}"><line x1="${c}" y1="${c}" x2="${x2}" y2="${y2}" stroke="${col}" stroke-width="3"/><polygon points="${x2},${y2} ${h(0.45)} ${h(-0.45)}" fill="${col}"/><circle cx="${c}" cy="${c}" r="2.5" fill="#64748b"/></svg>`;}
  function photo(m,w){const b=m.bbox,bw=Math.sqrt(b[2]*9/16)*100,bh=Math.sqrt(b[2]*16/9)*100;
    return `<div style="position:relative;display:inline-block;width:${w}px"><img src="figures/textcond/probe_${m.i}.jpg" style="width:100%;border-radius:6px;display:block"><div style="position:absolute;left:${b[0]*100-bw/2}%;top:${b[1]*100-bh/2}%;width:${bw}%;height:${bh}%;border:2px solid #ef4444;box-sizing:border-box"></div></div>`;}
  async function J(u){const r=await fetch(u+'?t='+Date.now());if(!r.ok)throw 0;return r.json();}
  let probes=null;
  async function tick(){
    let res={}; try{res=await J('egvla2_fm/text_cond_t2.json');}catch(e){}
    try{ if(!probes) probes=await J('egvla2_fm/text_cond_probes.json'); }catch(e){}
    const ks=Object.keys(res), n=ks.length;
    document.getElementById('tc-bar').style.width=(n/36*100)+'%';
    const last=ks.sort((a,b)=>(res[a].t||0)-(res[b].t||0)).pop(); const ts=ks.length?ks.map(k=>res[k].t||0):[];
    let eta='';
    if(n>=2){const all=Object.values(res).map(r=>r.t).filter(Boolean).sort((a,b)=>a-b);const per=(all[all.length-1]-all[0])/(all.length-1);eta=` · 1개당 약 ${per.toFixed(0)}초 · 남은 시간 약 ${Math.max(0,((36-n)*per/60)).toFixed(0)}분`;}
    document.getElementById('tc-prog').textContent=`완료 ${n}/36 · 최근: ${last||'-'}${eta}`;
    const st=document.getElementById('tc-state'); st.textContent=n>=36?'COMPLETED':'RUNNING'; st.className='status-badge '+(n>=36?'badge-completed':'badge-running');
    const rows=KEYS.map(([k,l,pct,nd])=>'<tr><td>'+l+'</td>'+VS.map(v=>{const x=Object.entries(res).filter(([kk,r])=>kk.split('|')[1]===v&&typeof r[k]==='number'&&r[k]===r[k]).map(([kk,r])=>r[k]);
      if(!x.length)return '<td>–</td>';const f=pct?100:1;return `<td>${(mean(x)*f).toFixed(nd)}±${(sd(x)*f).toFixed(nd)}${pct?'%':''} <span style="color:#64748b;font-size:0.7rem">n=${x.length}</span></td>`}).join('')+'</tr>').join('');
    document.getElementById('tc-res').innerHTML=rows;
    document.querySelectorAll('.tc-val').forEach(el=>{const k=el.dataset.key,pct=+el.dataset.pct,nd=+el.dataset.nd,f=pct?100:1;
      el.innerHTML=VS.map(v=>{const x=Object.entries(res).filter(([kk,r])=>kk.split('|')[1]===v&&typeof r[k]==='number'&&r[k]===r[k]).map(([kk,r])=>r[k]);return x.length?`${(mean(x)*f).toFixed(nd)}±${(sd(x)*f).toFixed(nd)}${pct?'%':''}`:'–'}).join(' · ')});
    if(!probes) return;
    const m0=probes[0]; document.getElementById('tc-flow-photo').innerHTML=photo(m0,300)+`<div style="font-size:0.72rem;color:#94a3b8">${m0.ep} · 프레임 ${m0.frame} · ${m0.cell}</div>`;
    document.getElementById('tc-flow-gt').innerHTML=arrow(m0.human,'#94a3b8')+arrow(m0.servo,'#34d399'); document.getElementById('tc-flow-neg').innerHTML=arrow([0,0],'#f87171');
    let h='';
    probes.forEach(m=>{
      h+=`<div style="display:flex;gap:16px;flex-wrap:wrap;border-top:1px solid #1e293b;padding:12px 0"><div style="flex:0 0 260px">${photo(m,260)}<div style="font-size:0.7rem;color:#94a3b8;margin-top:4px">${m.ep} · f${m.frame} · ${m.cell}<br>정답(사람/서보): ${arrow(m.human,'#94a3b8',46)}${arrow(m.servo,'#34d399',46)}</div></div>`;
      h+='<div style="flex:1;overflow-x:auto"><table class="ops-table" style="min-width:560px"><thead><tr><th>문구 조건</th>'+['V0','V1','V2','V3'].map(x=>'<th>'+x+'</th>').join('')+'</tr></thead><tbody>';
      PH.forEach(([pk,pl,pd,col])=>{h+=`<tr><td><code>${pl}</code><br><span style="font-size:0.68rem;color:#94a3b8">${pd}</span></td>`;
        VS.forEach(v=>{const rs=Object.entries(res).filter(([kk,r])=>kk.startsWith('fold0|'+v+'|')&&r.probe&&r.probe[pk]).map(([kk,r])=>r.probe[pk][m.i]);
          if(!rs.length){h+='<td>–</td>';return;} const a=[mean(rs.map(x=>x[0])),mean(rs.map(x=>x[1]))]; h+=`<td style="text-align:center">${arrow(a,col,52)}<div style="font-size:0.62rem;color:#64748b">n=${rs.length}</div></td>`;});
        h+='</tr>';});
      h+='</tbody></table></div></div>';});
    document.getElementById('tc-probes').innerHTML=h;
  }
  tick(); setInterval(tick,8000);
})();
</script>
"""


def main():
    s = HTML.read_text(encoding="utf-8")
    if B0 in s: s = re.sub(re.escape(B0) + r".*?" + re.escape(B1), lambda m: panel(), s, flags=re.S)
    else: s = s.replace("  </main>", panel() + "\n  </main>", 1)
    if "tab-textcond')" not in s: s = s.replace("      </button>\n    </nav>", "      </button>\n      " + BTN + "\n    </nav>", 1)
    HTML.write_text(s, encoding="utf-8"); print("ok", "tab-textcond" in s, "switchTab('tab-textcond')" in s)


main()

"""정지 정책 시뮬레이션 시각화 페이지 생성 → docs/v5/stop_policy_sim.html (+ 프레임 썸네일 docs/v5/figures/stop_sim/).
실제 사진(11/16 a3-s2 실기 프레임 중 바구니·의자 30세션) 위에 정책별 정지 시점·추정 간격(cm)을 얹는다.
면적→간격은 11/17 실측 쌍 회귀(stop_policy_sim_v2.json), 정지 시점은 stop_policy_sim.py의 규칙 재현(11/17 15/15 일치).
한계: 열린 루프·손 측정·물체별 5~10점 보정. 11/17 사진은 아직 없음(소다 전송 대기)."""
import csv, json, sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import stop_policy_sim as S

A = ROOT / "docs/v5/egvla2_fm/a3c"
FR = ROOT / "docs/v5/figures/objtest_1116"
OUT_IMG = ROOT / "docs/v5/figures/stop_sim"
V2 = json.load(open(A / "stop_policy_sim_v2.json"))
S.LAG = 0
POL = {  # id: (라벨, {물체: (T, 예측사용)})
    "P0a": ("현행 near 0.25", {"basket": (0.25, False), "chair": (0.25, False)}),
    "P0b": ("11/17 near 0.18", {"basket": (0.18, False), "chair": (0.18, False)}),
    "P1": ("물체별 near (바구니 0.15 · 의자 0.24)", {"basket": (0.15, False), "chair": (0.24, False)}),
    "P2": ("다음 스텝 예측 (바구니 0.19 · 의자 0.28)", {"basket": (0.19, True), "chair": (0.28, True)}),
}
NOTES = {  # 동환 구두 관찰(DRIVE_NOTES.md) — CSV에는 없음
    "20261116_154830": "부딪혀서 사용자가 정지시킴", "20261116_155042": "살짝 부딪히고 바로 멈춤",
    "20261116_155253": "아주 살짝 부딪히고 멈춤", "20261116_155332": "아주 살짝 부딪히고 멈춤", "20261116_155403": "아주 살짝 부딪히고 멈춤",
    "20261116_155433": "안 부딪힘", "20261116_155504": "5cm 넘게 밀었음",
}
POSN = {"trackF_center": "가운데", "trackA_weak_right": "오른쪽", "trackA_weak_left": "왼쪽"}
pos = {r["session_id"]: POSN.get(r["경로"], r["경로"]) for r in csv.DictReader(open(A / "real_1116_episode_log_slice.csv", encoding="utf-8"))}

data = []
OUT_IMG.mkdir(parents=True, exist_ok=True)
for sid, v in S.SESS.items():
    o = S.OBJ(v["instruction"])
    if o == "desk":
        continue
    a, h = S.series(sid)
    ae = S.extend(a)
    pol = {}
    for pid, (lab, cfg) in POL.items():
        T, pr = cfg[o]
        i, why = S.stop_index(ae, h, T, pred=pr)
        pol[pid] = {"idx": i, "why": why, "extrap": i >= len(a)}
    n = len(a)
    for i in range(n):
        dst = OUT_IMG / f"{sid}_{i}.jpg"
        src = FR / f"{sid}_{i}.jpg"
        if not dst.exists() and src.exists():
            Image.open(src).convert("RGB").resize((384, 216), Image.LANCZOS).save(dst, quality=70)
    data.append({"sid": sid, "obj": o, "pos": pos.get(sid, "?"), "n": n, "area": [round(x, 4) for x in ae], "rec": n,
                 "cx": v["cx"][:n], "cy": v["cy"][:n], "head": [round(x, 3) for x in h], "note": NOTES.get(sid, ""), "pol": pol})
data.sort(key=lambda d: (d["obj"], d["pos"], d["sid"]))
payload = {"sessions": data, "calib": V2["calib"], "summary": {o: V2[o] for o in ("basket", "chair")},
           "pol": {k: v[0] for k, v in POL.items()}, "target": S.TARGET, "band": S.BAND}

HTML = r'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>정지 정책 시뮬레이션 — 실제 사진으로 보기</title>
<style>
:root{--bg:#0a0f1a;--card:#111827;--card2:#0f172a;--line:#1f2937;--tx:#e2e8f0;--mut:#94a3b8;--a:#f87171;--b:#fbbf24;--c:#34d399;--d:#60a5fa;--real:#e2e8f0}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--tx);font:14px/1.6 system-ui,-apple-system,"Noto Sans KR",sans-serif;padding:0 16px 80px}
.w{max-width:1100px;margin:0 auto}h1{font-size:24px;margin:28px 0 4px}h2{font-size:17px;margin:34px 0 8px;display:flex;gap:8px;align-items:center}
.sub{color:var(--mut);margin:0 0 14px}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:10px 0}
.warn{border-color:#92400e;background:#1c1304}.warn b{color:#fbbf24}
.grid{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);gap:14px}@media(max-width:820px){.grid{grid-template-columns:1fr}}
select,button{background:#1e293b;color:var(--tx);border:1px solid #334155;border-radius:8px;padding:6px 12px;font:inherit;cursor:pointer}button:hover{background:#334155}
.photo{position:relative;background:#000;border-radius:10px;overflow:hidden}.photo img{display:block;width:100%}
.hud{position:absolute;left:8px;top:8px;background:rgba(0,0,0,.65);padding:4px 10px;border-radius:8px;font-size:12px;line-height:1.5}
.tag{display:inline-block;padding:1px 9px;border-radius:999px;font-size:12px;font-weight:700;margin-right:4px}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left}th{color:var(--mut);font-weight:600}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
.strip{display:flex;gap:6px;overflow-x:auto;padding:6px 0}.th{flex:0 0 118px;position:relative;cursor:pointer;border:2px solid transparent;border-radius:8px;overflow:hidden}.th.on{border-color:var(--real)}.th img{display:block;width:100%}
.th .t{position:absolute;left:0;right:0;bottom:0;background:rgba(0,0,0,.7);font-size:11px;padding:1px 4px}.th .b{position:absolute;top:3px;left:3px;display:flex;gap:2px;flex-wrap:wrap}
.th .b i{font-style:normal;font-size:10px;font-weight:800;padding:0 5px;border-radius:6px;color:#0a0f1a}
.bar{height:8px;background:#1e293b;border-radius:6px;overflow:hidden;margin-top:3px}.bar i{display:block;height:100%}
svg text{fill:var(--mut);font-size:11px}.big{font-size:22px;font-weight:800}
.ok{color:#34d399}.no{color:#f87171}.mid{color:#fbbf24}
</style></head><body><div class="w">
<h1>정지 정책 시뮬레이션 — 실제 사진으로 보기</h1>
<p class="sub">11/16 <code>a3-s2</code> 실기 프레임(바구니·의자 30세션) 위에, 정지 규칙별로 "어느 프레임에서 멈추고 그때 간격이 몇 cm쯤인지"를 얹었습니다. 2026-11-18 · 계획서 <code>plan_20261117_stop_policy.md</code> §9~§10</p>
<div class="card warn"><b>읽기 전에</b> — ① 사진은 <b>11/16 실기</b>(near 0.25)이고, 11/17(near 0.18) 사진은 아직 못 받았습니다. ② 정책별 정지 프레임은 <b>열린 루프 재생</b>입니다(정지 이후 경로가 바뀌지 않음). ③ cm는 11/17 동환 <b>손 측정 5~10점</b>으로 보정한 근사이고 순위·방향 참고용입니다. ④ 박스 위치는 십자(중심)와 면적 게이지로만 표시합니다(박스 모양은 기록에 없음).</div>

<h2>① 한 세션씩 보기</h2>
<div class="card"><select id="sel"></select> <button id="prev">◀</button> <button id="play">▶ 재생</button> <button id="next">▶|</button> <span id="info" style="color:var(--mut);margin-left:8px"></span></div>
<div class="grid">
 <div><div class="photo"><img id="img" alt=""><div class="hud" id="hud"></div></div>
  <div class="card"><div style="color:var(--mut);font-size:12px">위에서 본 거리 (추정, 목표=왼쪽 0cm)</div><svg id="scene" viewBox="0 0 520 120" width="100%"></svg></div></div>
 <div><div class="card"><div style="color:var(--mut);font-size:12px">박스 면적 · 프레임별</div><svg id="chart" viewBox="0 0 520 220" width="100%"></svg></div>
  <div class="card"><div id="cur"></div></div></div>
</div>
<div class="card"><div style="color:var(--mut);font-size:12px">프레임 필름스트립 — 색 배지 = 그 정책이 멈추는 프레임 (클릭하면 이동)</div><div class="strip" id="strip"></div></div>
<div class="card"><table id="ptab"></table><div id="legend" style="margin-top:6px;color:var(--mut);font-size:12px"></div></div>

<h2>② 면적→간격 보정 (11/17 실측)</h2>
<p class="sub">동환 손 측정 cm와 첫 STOPPED 박스 면적을 짝지어 회귀. 점이 선에서 얼마나 떨어졌는지가 불확실성입니다.</p>
<div class="grid"><div class="card"><svg id="cal1" viewBox="0 0 520 280" width="100%"></svg></div><div class="card"><svg id="cal2" viewBox="0 0 520 280" width="100%"></svg></div></div>
<div class="card" id="calnote"></div>

<h2>③ 정책 비교 (11/16 경로 전체, 목표: 바구니 5cm · 의자 17.5cm · 허용 ±5cm)</h2>
<div class="card"><table id="sum"></table></div>
<div class="card warn"><b>한계</b> — 의자는 정지 부근에서 한 스텝에 약 25cm 움직여(회귀 기준) 임계값만으로는 이상 범위(15~20cm)를 정밀하게 못 맞춥니다 → 접근 감속(설계안 3)이 필요합니다. 바구니·의자의 스텝 길이 추정이 서로 달라(약 6cm vs 25cm) 물리적으로 일관되지 않으므로 cm는 방향 참고용입니다. 첫 STOPPED 이후에도 면적이 커지는 현상(11/17)은 소다 확인 대기입니다.</div>
</div>
<script>
const D=__DATA__;
const COL={P0a:'#f87171',P0b:'#fbbf24',P1:'#34d399',P2:'#60a5fa'};
const $=id=>document.getElementById(id);
const OBJ={basket:'바구니',chair:'의자'};
function gap(o,a){const c=D.calib[o];return c.alpha+c.beta/Math.sqrt(a)}
function verdict(o,g){const t=D.target[o],b=D.band;if(g<0)return['접촉','no'];if(Math.abs(g-t)<=b)return['대역 안','ok'];return g>t?['너무 일찍','mid']:['대역 아래(가까움)','mid']}
let S=null,fi=0,timer=null;
const sel=$('sel');D.sessions.forEach((s,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${OBJ[s.obj]} · ${s.pos} · ${s.sid.slice(9,13)}:${s.sid.slice(13)}${s.note?' ⚠ '+s.note:''}`;sel.appendChild(o)});
sel.onchange=()=>load(+sel.value);$('prev').onclick=()=>show(Math.max(0,fi-1));$('next').onclick=()=>show(Math.min(S.n-1,fi+1));
$('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;$('play').textContent='▶ 재생';return}$('play').textContent='⏸ 정지';if(fi>=S.n-1)show(0);timer=setInterval(()=>{if(fi>=S.n-1){clearInterval(timer);timer=null;$('play').textContent='▶ 재생';return}show(fi+1)},700)};
function load(i){S=D.sessions[i];sel.value=i;
 $('info').textContent=`${S.n}프레임 · ${S.note?'동환 구두: '+S.note:'구두 접촉 보고 없음'}`;
 // 필름스트립
 const st=$('strip');st.innerHTML='';
 for(let k=0;k<S.n;k++){const d=document.createElement('div');d.className='th';d.dataset.k=k;
  const badges=Object.keys(D.pol).filter(p=>S.pol[p].idx===k).map(p=>`<i style="background:${COL[p]}">${p}</i>`).join('');
  d.innerHTML=`<img loading="lazy" src="figures/stop_sim/${S.sid}_${k}.jpg"><div class="b">${badges}</div><div class="t">#${k} · ${S.area[k].toFixed(2)}</div>`;d.onclick=()=>show(k);st.appendChild(d)}
 // 표
 const rows=Object.keys(D.pol).map(p=>{const r=S.pol[p],a=S.area[r.idx],g=gap(S.obj,a),v=verdict(S.obj,g);
  return `<tr><td><span class="dot" style="background:${COL[p]}"></span>${D.pol[p]}</td><td>#${r.idx}${r.extrap?' <span class="mid">(기록 끝 너머·외삽)</span>':''}</td><td>${a.toFixed(3)}</td><td>${g.toFixed(1)} cm</td><td class="${v[1]}"><b>${v[0]}</b></td><td style="color:var(--mut)">${r.why==='near'?'면적 규칙':'헤드/기록 종료'}</td></tr>`}).join('');
 $('ptab').innerHTML=`<tr><th>정책</th><th>정지 프레임</th><th>그때 면적</th><th>추정 간격</th><th>판정</th><th>원인</th></tr>${rows}<tr><td><span class="dot" style="background:#e2e8f0"></span>실제(기록 마지막 프레임)</td><td>#${S.rec-1}</td><td>${S.area[S.rec-1].toFixed(3)}</td><td>${gap(S.obj,S.area[S.rec-1]).toFixed(1)} cm</td><td>${S.note||'-'}</td><td style="color:var(--mut)">운영자/컨트롤러</td></tr>`;
 $('legend').textContent=`목표 간격 ${D.target[S.obj]}cm ±${D.band}cm · 보정 잔차 SD ${D.calib[S.obj].resid_sd_cm}cm(n=${D.calib[S.obj].n})`;
 show(Math.min(fi,S.n-1)===fi&&fi<S.n?fi:0)}
function show(k){fi=k;$('img').src=`figures/stop_sim/${S.sid}_${k}.jpg`;
 const a=S.area[k],g=gap(S.obj,a);
 $('hud').innerHTML=`프레임 #${k}/${S.n-1}<br>박스 면적 ${a.toFixed(3)}<br>헤드 출력 ${S.head[k]!==undefined?S.head[k]:'-'}`;
 // 십자
 const ph=document.querySelector('.photo');ph.querySelectorAll('.cross').forEach(e=>e.remove());
 const cx=S.cx[k],cy=S.cy[k];if(cx!=null){const e=document.createElement('div');e.className='cross';e.style.cssText=`position:absolute;left:${cx*100}%;top:${cy*100}%;width:26px;height:26px;margin:-13px 0 0 -13px;border:2px solid #f43f5e;border-radius:50%;box-shadow:0 0 0 2px rgba(0,0,0,.5)`;ph.appendChild(e)}
 document.querySelectorAll('.th').forEach(e=>e.classList.toggle('on',+e.dataset.k===k));
 const sc=document.querySelector(`.th[data-k="${k}"]`);if(sc)sc.scrollIntoView({block:'nearest',inline:'center'});
 const v=verdict(S.obj,g);
 $('cur').innerHTML=`<div class="big ${v[1]}">${g.toFixed(1)} cm <span style="font-size:13px;font-weight:600">· ${v[0]}</span></div><div style="color:var(--mut);font-size:12px">이 프레임에서 멈춘다면 추정 간격(근사)</div><div class="bar"><i style="width:${Math.min(100,a*100/0.6)}%;background:#60a5fa"></i></div><div style="color:var(--mut);font-size:12px">면적 ${a.toFixed(3)} / 0.6</div>`;
 drawChart();drawScene()}
function drawChart(){const W=520,H=220,L=40,R=10,T=10,B=26;const n=S.n,mx=Math.max(0.6,...S.area.slice(0,n));const x=i=>L+(W-L-R)*i/Math.max(n-1,1),y=a=>T+(H-T-B)*(1-a/mx);
 let s='';for(let t=0;t<=mx;t+=0.1){s+=`<line x1="${L}" x2="${W-R}" y1="${y(t)}" y2="${y(t)}" stroke="#1f2937"/><text x="4" y="${y(t)+4}">${t.toFixed(1)}</text>`}
 for(let i=0;i<n;i++)s+=`<text x="${x(i)-3}" y="${H-8}">${i}</text>`;
 const th=[[0.25,'#f87171','0.25'],[0.18,'#fbbf24','0.18']];const o=S.obj,p1=o==='basket'?0.15:0.24;th.push([p1,'#34d399',String(p1)]);
 th.forEach(t=>s+=`<line x1="${L}" x2="${W-R}" y1="${y(t[0])}" y2="${y(t[0])}" stroke="${t[1]}" stroke-dasharray="4 4" opacity=".7"/><text x="${W-R-30}" y="${y(t[0])-3}" style="fill:${t[1]}">${t[2]}</text>`);
 s+=`<polyline fill="none" stroke="#e2e8f0" stroke-width="2" points="${S.area.slice(0,n).map((a,i)=>x(i)+','+y(a)).join(' ')}"/>`;
 Object.keys(D.pol).forEach(p=>{const r=S.pol[p];if(r.idx<n){s+=`<circle cx="${x(r.idx)}" cy="${y(S.area[r.idx])}" r="6" fill="${COL[p]}" stroke="#0a0f1a" stroke-width="2"/>`}});
 s+=`<line x1="${x(fi)}" x2="${x(fi)}" y1="${T}" y2="${H-B}" stroke="#94a3b8" stroke-dasharray="2 3"/>`;$('chart').innerHTML=s}
function drawScene(){const W=520,H=120;const o=S.obj;const gs=S.area.slice(0,S.n).map(a=>gap(o,a));const mx=Math.max(40,gs[0]>200?200:gs[0]);const X0=70,X1=W-20;
 const x=g=>X0+(X1-X0)*Math.min(Math.max(g,-5),mx)/(mx+5);
 let s=`<rect x="10" y="30" width="${o==='basket'?34:46}" height="60" rx="6" fill="${o==='basket'?'#94a3b8':'#a16207'}"/><text x="14" y="26">${OBJ[o]}</text>`;
 const t=D.target[o],b=D.band;s+=`<rect x="${x(Math.max(t-b,0))}" y="20" width="${x(t+b)-x(Math.max(t-b,0))}" height="80" fill="#34d39922"/><text x="${x(t-b)}" y="14" style="fill:#34d399">목표 ${t}±${b}cm</text>`;
 s+=`<line x1="${X0-20}" x2="${X1}" y1="110" y2="110" stroke="#334155"/>`;
 for(let k=0;k<S.n;k++){const g=gs[k];s+=`<circle cx="${x(g)}" cy="60" r="${k===fi?7:3}" fill="${k===fi?'#e2e8f0':'#475569'}"/>`}
 Object.keys(D.pol).forEach((p,j)=>{const r=S.pol[p];const g=gs[Math.min(r.idx,S.n-1)];if(r.idx<S.n)s+=`<line x1="${x(g)}" x2="${x(g)}" y1="${34+j*3}" y2="94" stroke="${COL[p]}" stroke-width="2"/>`});
 const g=gs[fi];s+=`<text x="${x(g)-14}" y="82" style="fill:#e2e8f0;font-weight:700">${g.toFixed(0)}cm</text>`;$('scene').innerHTML=s}
// 보정 산점도
const PAIRS=[['basket',[[0.241,5],[0.345,1],[0.245,5],[0.319,3],[0.386,-3],[0.293,1],[0.333,2],[0.31,-2],[0.245,5],[0.298,-3]]],['chair',[[0.287,30],[0.307,25],[0.269,35],[0.307,25],[0.302,25]]]];
function cal(id,o,pts){const W=520,H=280,L=44,R=12,T=14,B=34;const c=D.calib[o];const ar=pts.map(p=>p[0]);const amin=0.2,amax=o==='basket'?0.42:0.46;
 const gs=pts.map(p=>p[1]);const gmin=-10,gmax=o==='basket'?12:50;const x=a=>L+(W-L-R)*(a-amin)/(amax-amin),y=g=>T+(H-T-B)*(1-(g-gmin)/(gmax-gmin));
 let s=`<text x="${L}" y="12" style="fill:#e2e8f0;font-weight:700">${OBJ[o]} (n=${c.n}) — 간격 = ${c.alpha} + ${c.beta}/√면적 · 잔차 SD ${c.resid_sd_cm}cm</text>`;
 for(let g=Math.ceil(gmin/10)*10;g<=gmax;g+=10)s+=`<line x1="${L}" x2="${W-R}" y1="${y(g)}" y2="${y(g)}" stroke="${g===0?'#f87171':'#1f2937'}" ${g===0?'stroke-dasharray="4 3"':''}/><text x="6" y="${y(g)+4}">${g}</text>`;
 for(let a=amin;a<=amax+1e-9;a+=0.05)s+=`<text x="${x(a)-10}" y="${H-14}">${a.toFixed(2)}</text>`;
 let pts2=[];for(let a=amin;a<=amax;a+=0.005)pts2.push(x(a)+','+y(c.alpha+c.beta/Math.sqrt(a)));
 s+=`<polyline fill="none" stroke="#60a5fa" stroke-width="2" points="${pts2.join(' ')}"/>`;
 const t=D.target[o];s+=`<line x1="${L}" x2="${W-R}" y1="${y(t)}" y2="${y(t)}" stroke="#34d399" stroke-dasharray="4 4"/><text x="${W-R-110}" y="${y(t)-4}" style="fill:#34d399">목표 ${t}cm → 면적 ${c.target_area}</text>`;
 pts.forEach(p=>s+=`<circle cx="${x(p[0])}" cy="${y(p[1])}" r="5" fill="#e2e8f0" opacity=".85"/>`);
 s+=`<text x="${L}" y="${H-1}">첫 STOPPED 박스 면적 →</text><text x="6" y="${T+2}">cm</text>`;$(id).innerHTML=s}
cal('cal1','basket',PAIRS[0][1]);cal('cal2','chair',PAIRS[1][1]);
$('calnote').innerHTML=`바구니는 점이 흩어져(순위상관 ${D.calib.basket.spearman}) 신뢰도가 낮고, 의자는 25~35cm 구간 5점뿐이라 목표 17.5cm는 <b>관측 범위 밖 외삽</b>입니다. 접촉(간격 0)에 해당하는 면적은 바구니 약 ${D.calib.basket.area_at_gap0}, 의자 약 ${D.calib.chair.area_at_gap0}로 추정됩니다(의자는 접촉 데이터 0건).`;
// 요약 표
const names={'P0a 현행 near=0.25':'현행 near 0.25','P0b 오늘 near=0.18':'near 0.18 (11/17)','P1 물체별 near(LOO)':'물체별 near','P2 다음스텝 예측+물체별(LOO)':'다음 스텝 예측'};
let h='<tr><th>물체</th><th>정책</th><th>간격 중앙</th><th>대역 안</th><th>접촉</th><th>임계값(중앙)</th></tr>';
['basket','chair'].forEach(o=>Object.entries(D.summary[o]).forEach(([k,m])=>{h+=`<tr><td>${OBJ[o]}</td><td>${names[k]||k}</td><td>${m.gap_median} cm</td><td>${Math.round(m.in_band*100)}%</td><td class="${m.contact>0.2?'no':''}">${Math.round(m.contact*100)}%</td><td>${m.T_median}</td></tr>`}));
$('sum').innerHTML=h;
load(0);
</script></body></html>'''
out = ROOT / "docs/v5/stop_policy_sim.html"
out.write_text(HTML.replace("__DATA__", json.dumps(payload, ensure_ascii=False)), encoding="utf-8")
n_img = len(list(OUT_IMG.glob("*.jpg")))
print(f"{out} {out.stat().st_size/1e3:.0f}KB · 세션 {len(data)} · 썸네일 {n_img}장 {sum(f.stat().st_size for f in OUT_IMG.glob('*.jpg'))/1e6:.1f}MB")

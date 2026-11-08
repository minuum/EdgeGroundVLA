"""모델 세대 레지스트리: 스펙(아래 SPEC) + 파일 해시·크기 + 로봇 episode_log 집계 -> docs/v5/MODEL_REGISTRY.json, model_registry.html.
실기 수치는 origin/monavla-driving:logs/episode_log.csv의 체크포인트 열 집계(운영자 수동 판정). 스펙에 없는 값은 None(미확인)으로 둔다.
실행: .venv/bin/python3 scripts/build_model_registry.py  (git fetch 후)"""
import csv, collections, hashlib, io, json, pathlib, subprocess, datetime, html
ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "docs/v5"
GENS = {
 "G0": ("V5 시기(Kosmos-2)", "순수 E2E → 박스 분해. 폐기, 기록만."),
 "G1": ("EGVLA1", "OWLv2 + Kosmos-2 + MLP(exp73). 실기 기준선."),
 "G2": ("EGVLA2", "Florence-2 단일 + FM 헤드, 300에피소드. 현재 배포(A3=EG2-09)."),
 "G3": ("E2E LoRA", "Florence-2 LoRA + FM. 오프라인 부진, 실기 없음, 보류."),
 "G4": ("텍스트 조건화", "V0~V5. 평가 중 학습·폐기, 체크포인트 없음. 오프라인만."),
}
def E(pn, gen, name, ck=None, path=None, arch="", train="", offline="", status="", note="", key=None):
    return dict(pn=pn, gen=gen, name=name, ckpt=ck, path=path, arch=arch, train=train, offline=offline, status=status, note=note, log_key=key if key is not None else ck)
SPEC = [
 E("EG0-01","G0","V5 Exp11","epoch=14 ckpt","runs/v5_nav/kosmos/mobile_vla_v5_exp11/2026-04-16/v5-exp11-google-robot-8cls","Kosmos-2(고정)+LoRA, 구글로봇 사전학습, 8클래스, 순수 E2E","","val_loss 1.010, PM 58.6%","폐기","폐루프(시뮬) 0%, FPE 1.45m",key="__none__"),
 E("EG0-02","G0","Exp14 Step2","","docs/v5/bbox_nav_step1","박스+이미지 MLP(분해)","45ep 794프레임","PM 75.9%(5시드 76.6±1.6%)","폐기","폐루프(시뮬) 66.7%, FPE 0.55m",key="__none__"),
 E("EG1-00","G1","구버전(체크포인트 미기록)","", None,"","","","추적 불가","로그 6/26~7/23, 체크포인트 열 공란",key=""),
 E("EG1-01","G1","OWL 트랙 hold-aware","exp73_owl_trackF_v6_mlp_holdaware_seed0.pt",None,"OWLv2 박스 + MLP","","","기록"),
 E("EG1-02","G1","EGVLA1 기준선","exp73_owl_stage1v3_v6_mlp.pt","runs/v5_nav/mlp/exp73_stage1v3/exp73_owl_stage1v3_v6_mlp.pt","MLP 1560→512→128→8(8클래스), window 6, bbox_scale 3, V6 arm","데이터: mobile_vla_dataset_v5(h5 515개, CH76 ④·사용자 확인). 에폭: CH70이 exp73 원본 값 300을 언급(이 체크포인트 자체에는 기록 없음). CH78: EGVLA1·2 모두 pos1만 학습","val_acc 74.1%(V6 split, 체크포인트 기록)","기준선","3×5×20 폐루프 기준선. 위치별 pos1 95·pos2 91·pos3 90 (episode_log·메모리)"),
 E("EG1-03","G1","Florence-2 grounder + MLP","exp73_florence2_grounder_v6_mlp.pt","runs/v5_nav/mlp/exp73_stage1v3/exp73_florence2_grounder_v6_mlp.pt","Florence-2 그라운더 + MLP(V6)","미확인","미확인","실기 로그 없음","8/20 파일. EGVLA2 전환 이전의 중간 산물로 보이나 미확인",key="__none__"),
 E("EG1-04","G1","Florence-2 phrase grounder + MLP","exp73_florence2_phrase_grounder_v6_mlp.pt","runs/v5_nav/mlp/exp73_stage1v3/exp73_florence2_phrase_grounder_v6_mlp.pt","Florence-2 문구 그라운더 + MLP(V6)","미확인","미확인","실기 로그 없음","8/21 파일",key="__none__"),
 E("EG2-01","G2","A_mlp_fm","A_mlp_fm.pt","runs/egvla2_fm/A_mlp_fm.pt","MLP+FM","250ep, 300ep","각도 11.26°(층화) / 20.28°(LOPO)","오프라인",key="__none__"),
 E("EG2-02","G2","B2_film_res_fm","B2_film_res_fm.pt","runs/egvla2_fm/B2_film_res_fm.pt","FiLM 잔차+FM","250ep, 300ep","10.81° / 17.91°","오프라인",key="__none__"),
 E("EG2-03","G2","C_pi0_adaln","C_pi0_adaln.pt","runs/egvla2_fm/C_pi0_adaln.pt","PiZeroFM AdaLN-Zero","250ep, 300ep","10.39° / LOPO 미출력","오프라인",key="__none__"),
 E("EG2-04","G2","MLP 회귀(결정적)","MLP_regression_drop30_e60.pt","runs/egvla2_fm/deploy/MLP_regression_drop30_e60.pt","MLP 회귀, 드롭아웃 30%","60ep, 300ep","","실기 비교군 후보",key="__none__"),
 E("EG2-05","G2","C adaln 드롭0","C_pi0_adaln_e120.pt","runs/egvla2_fm/deploy/C_pi0_adaln_e120.pt","adaln, 드롭아웃 0","120ep, 300ep","","대조",key="__none__"),
 E("EG2-06","G2","C adaln 드롭30 (CH79의 A1)","C_pi0_adaln_drop30_e120.pt","runs/egvla2_fm/deploy/C_pi0_adaln_drop30_e120.pt","adaln, 드롭아웃 30%","120ep, 300ep","","실기 부진","오프라인 좋았으나 실기 낮음"),
 E("EG2-07","G2","C e1b tuned","C_pi0_e1b_tuned_drop30_e120.pt","runs/egvla2_fm/deploy/C_pi0_e1b_tuned_drop30_e120.pt","서보 τ=0.22 gain=2.6","120ep, 300ep","","실기 기록 없음",key="__none__"),
 E("EG2-08","G2","서보 라벨 λ=0.5 (CH79의 A2)","C_pi0_servolabel0.5_drop30_e120.pt","runs/egvla2_fm/deploy/C_pi0_servolabel0.5_drop30_e120.pt","서보 라벨 λ=0.5","120ep, 300ep","","시험"),
 E("EG2-09","G2","A3 서보 라벨 λ=1.0","C_pi0_servolabel1.0_drop30_e120.pt","runs/egvla2_fm/deploy/C_pi0_servolabel1.0_drop30_e120.pt","서보 라벨 λ=1.0","120ep, 300ep","","현재 배포","실기 로그 중 11/04 253건은 pos4~6×G6~G8 수집(EG1과 프로토콜 다름)"),
 E("EG2-10","G2","299건 버전 A/B2/C","", "runs/egvla2_fm_299/","A_mlp_fm·B2·C_pi0_adaln","299건(29,062프레임)","","보존","300번째 반영 이전의 학습 결과를 보존한 폴더(계획서 §119). 서빙에 쓰이는지는 미확인",key="__none__"),
 E("EG3-01","G3","E2E LoRA 스모크","e2e_vla_florence2_lora_latest.pt","runs/egvla2_e2e_lora/e2e_vla_florence2_lora_latest.pt","Florence-2 LoRA r=8(0.72%)+PiZeroFM","25ep, 15에피소드","in-sample 2.82°, 미학습 4개 15.78°","보류","412.95ms",key="__none__"),
 E("EG3-02","G3","A_frozen_fused","","docs/v5/egvla2_fm/e2e_general.json","frozen 융합 특징","","held-out 방향 19.6°(3분할 평균)","결과만",key="__none__"),
 E("EG3-03","G3","B0 DaViT frozen","","docs/v5/egvla2_fm/e2e_general.json","인코더 고정 120ep/20ep","","29.6° / 32.7°","결과만",key="__none__"),
 E("EG3-04","G3","B1 LoRA 마지막 단계","","docs/v5/egvla2_fm/e2e_general.json","LoRA 마지막 단계 20ep","","27.7°","결과만",key="__none__"),
 E("EG3-05","G3","B2 LoRA 전 단계 384px","","docs/v5/egvla2_fm/e2e_fullstage.json","LoRA 전 단계 8ep","","30.9°, 라이브 36.1°","결과만",key="__none__"),
]
for pn,nm,ar,of,nt in [("EG4-00","V0 이미지만","A3 레시피","부정 정지 8%, 박스 반대편 시 서보각 20.3°·좌우 9%",""),("EG4-01","V1 고정 문구","","결과 JSON 참고(text_cond_t2)",""),("EG4-02","V2 패러프레이즈","","결과 JSON 참고(text_cond_t2)",""),
    ("EG4-03","V3 +부정 20%","","부정 정지 81.5%",""),("EG4-04","V4 어휘 40+40","","부정 정지 99.5%, 긍정 +3.1pp(기준 +3 미달), C 정상 2.1°·반대편 4.8°",""),
    ("EG4-05","V0R 박스 훼손 증강","","반대편 −12.8°, 정상 +4.6° 악화",""),("EG4-06","V5 토큰 어텐션","MultiheadAttention 768/4head","학습 중(T5)","")]:
    SPEC.append(E(pn,"G4",nm,"", "docs/v5/egvla2_fm/text_cond_t4.json" if pn!="EG4-06" else "docs/v5/egvla2_fm/text_cond_t5.json",ar,"3분할×시드3, 120ep, 프레임 1/9",of,"오프라인만",nt,key="__none__"))

RS = "research_story.html"; VO = "vlaops_dashboard.html"
LINKS = {  # PN -> [(라벨, v5 기준 상대 URL)]
 "EG1-02": [("CH70 헤드 구조·손실 실험(exp73, epoch 300 언급)", RS+"#ch70"), ("CH78 ⑥-2 EGVLA1 vs EGVLA2(276/300)", RS+"#ch78"), ("CH77 EGVLA1 비교", RS+"#ch77"), ("CH76 ④ V6 학습 데이터", RS+"#ch76")],
 "EG2-01": [("CH77 학습·오프라인(FM-MLP)", RS+"#ch77")], "EG2-02": [("CH77 (FM-FiLM)", RS+"#ch77")], "EG2-03": [("CH77 (FM-π0)", RS+"#ch77")],
 "EG2-04": [("CH77 MLP 회귀 대조", RS+"#ch77"), ("CH78 실기 비교군", RS+"#ch78")], "EG2-05": [("CH77", RS+"#ch77"), ("CH79", RS+"#ch79")],
 "EG2-06": [("CH79 폐루프(10/22~28, A1)", RS+"#ch79")], "EG2-07": [("CH79", RS+"#ch79")], "EG2-08": [("CH79 10/28 비교(A2)", RS+"#ch79")],
 "EG2-09": [("CH79 A3 채택·10/29·11/04 일반화", RS+"#ch79"), ("VLAops", VO)],
 "EG3-01": [("CH79 E2E 일반 비교", RS+"#ch79")], "EG3-02": [("CH79 E2E 일반 비교", RS+"#ch79")], "EG3-03": [("CH79 E2E 일반 비교", RS+"#ch79")], "EG3-04": [("CH79 E2E 일반 비교(마지막 단계 LoRA)", RS+"#ch79")], "EG3-05": [("CH79 E2E 일반 비교(전 단계 LoRA@384)", RS+"#ch79")],
}
for _pn in ("EG4-00","EG4-01","EG4-02","EG4-03","EG4-04","EG4-05","EG4-06"):
    LINKS[_pn] = [("CH79 🔤 텍스트 조건화 탭", RS+"#ch79"), ("VLAops 🔤 문구·표현별 테스트", VO)]
DATASETS = [  # id, 이름, 경로, 설명, 사용 모델(PN), 결과 링크
 ("DS-01","EGVLA2 학습셋 300건","ROS_action/mobile_vla_dataset_egvla2","10/13~10/16 수집(디렉터리 H5 371개 중 학습 세트 300건 = meta/egvla2_v3_main.json, 29,161프레임). 속성: 시작 위치·경로 라벨","EG2-01~09, EG3-*, EG4-*", [("CH76 수집 점검", RS+"#ch76"),("CH77", RS+"#ch77")]),
 ("DS-02","V5 데이터셋(EGVLA1 학습)","ROS_action/mobile_vla_dataset_v5","h5 515개 로컬 확인. CH76 ④: V6 학습 데이터, minum 쪽이 유일 원본(로봇 쪽 복사본은 정리됨, 별도 백업 없음). CLAUDE.md의 V5 150건은 이 중 일부로 보이나 미확인","EG1-02(사용자 확인), EG0-*(추정)", [("CH76 ④ 데이터 보관 상태", RS+"#ch76")]),
 ("DS-03","폐루프 드라이런 10/22·10/28","ROS_action/dryrun_closedloop_20261022, ..._20261028","EGVLA2 폐루프 드라이런 원본(각각 657MB, 2.1GB)","EG2-06, EG2-08", [("CH79", RS+"#ch79")]),
 ("DS-04","시작 위치 드라이런","ROS_action/dryrun_startpos","시작 위치 재생용 드라이런","EG2-*", [("CH78 시작 위치 재생", RS+"#ch78")]),
 ("DS-05","실기 세션 H5(10/22~10/29)","docs/inference_sessions/session_20261022~29_*.h5","A3·adaln·서보라벨 실기 세션(10/22 19, 10/28 102, 10/29 96)","EG2-06, EG2-08, EG2-09", [("CH79 📅 10/29", RS+"#ch79")]),
 ("DS-06","11/04·11/05 수집 297건","docs/inference_sessions/session_20261104_*.h5, session_20261105_*.h5","pos4~6 × G6~G8 × 3궤적 × 11. 11/04 252 + 11/05 45(로그 행 없음)","EG2-09", [("CH79 📆 11/04", RS+"#ch79")]),
 ("DS-07","텍스트 어휘·특징 캐시","runs/egvla2_fm/text_vocab_cache.npz, text_vocab_tokens.npz","T4/T5용 문구 116개의 융합 인코더 특징(stride 9, 프레임 약 3.2k)","EG4-*", [("VLAops T4·T5", VO)]),
 ("DS-08","EGVLA1 실기 세션(8/7~9/19)","docs/inference_sessions/ (로컬 없음)","episode_log에 session_id 300건이 있으나 H5는 이 머신에 없음(billy 쪽 보관 추정, 미확인)","EG1-02", []),
]

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()

def robot():
    try:
        txt = subprocess.check_output(["git", "show", "origin/monavla-driving:logs/episode_log.csv"], cwd=ROOT).decode("utf-8")
    except Exception as e:
        return {}, f"episode_log 읽기 실패: {e}"
    g = collections.defaultdict(list)
    for x in csv.DictReader(io.StringIO(txt)): g[x["체크포인트"]].append(x)
    out = {}
    for k, v in g.items():
        ds = sorted(x["날짜"] for x in v if x["날짜"]); ok = sum(x["결과"] == "성공" for x in v)
        out[k] = dict(n=len(v), ok=ok, rate=round(ok / len(v) * 100, 1), first=ds[0][:10] if ds else None, last=ds[-1][:10] if ds else None)
    return out, None

def sessions():
    """checkpoint -> 세션 목록(날짜·결과·로컬 H5 유무·필름스트립 유무). docs/v5/egvla2_fm/model_sessions.json 저장."""
    try:
        txt = subprocess.check_output(["git", "show", "origin/monavla-driving:logs/episode_log.csv"], cwd=ROOT).decode("utf-8")
    except Exception:
        return {}
    fs = set()
    for f in ("filmstrips1029.json", "filmstrips1104.json"):
        try: fs |= {x["sid"] for x in json.load(open(D / "egvla2_fm" / f))}
        except Exception: pass
    out = collections.defaultdict(list)
    for x in csv.DictReader(io.StringIO(txt)):
        sid = x["session_id"]
        out[x["체크포인트"] or "(공란)"].append(dict(sid=sid or None, date=x["날짜"][:16], ok=x["결과"] == "성공", route=x["경로"], steps=x["steps"],
            h5=bool(sid) and (ROOT / f"docs/inference_sessions/session_{sid}.h5").exists(), film=sid in fs))
    return out

def main():
    rb, err = robot(); ss = sessions(); rows = []
    for e in SPEC:
        r = dict(e); p = ROOT / e["path"] if e["path"] else None
        r["exists"] = bool(p and p.exists()); r["bytes"] = p.stat().st_size if r["exists"] and p.is_file() and p.suffix in (".pt", ".ckpt") else None
        r["sha256"] = sha(p) if r["bytes"] and r["bytes"] < 5e8 and p.suffix in (".pt", ".ckpt") else None
        r["robot"] = rb.get(e["log_key"]) if e["log_key"] not in (None, "__none__") else None
        r["links"] = LINKS.get(e["pn"], [])
        sl = ss.get(e["log_key"] or "(공란)", []) if e["log_key"] not in (None, "__none__") else []
        r["sessions"] = dict(n=len(sl), h5_local=sum(x["h5"] for x in sl), film=sum(x["film"] for x in sl)) if sl else None
        rows.append(r)
    (D / "egvla2_fm" / "model_sessions.json").write_text(json.dumps(ss, ensure_ascii=False), encoding="utf-8")
    (D / "MODEL_REGISTRY.json").write_text(json.dumps(dict(generated=datetime.datetime.now().isoformat(timespec="minutes"), gens={k: list(v) for k, v in GENS.items()}, datasets=[dict(id=d[0], name=d[1], path=d[2], desc=d[3], used_by=d[4], links=d[5]) for d in DATASETS], models=rows, robot_log_error=err), ensure_ascii=False, indent=1), encoding="utf-8")
    esc = html.escape
    def links_cell(r):
        a = "<br>".join(f'<a href="{u}">{esc(l)}</a>' for l, u in r["links"])
        se = r["sessions"]
        if se: a += f'<br><span class=m>세션 {se["n"]}건 · 로컬 H5 {se["h5_local"]} · 필름스트립 {se["film"]}</span> <a href="#ses-{r["pn"]}">목록</a>'
        return a
    def cell(r):
        rr = r["robot"]
        if rr: return f'<b>{rr["ok"]}/{rr["n"]} = {rr["rate"]}%</b><br><span class=m>{rr["first"]}~{rr["last"]}</span>'
        return '<span class=m>없음</span>'
    css = """:root{--bg:#fff;--fg:#1d2330;--mut:#5b6577;--line:#d9dee7;--head:#eef1f6;--acc:#1f5fbf;--ok:#1b7f4b;--warn:#a55a00}
@media(prefers-color-scheme:dark){:root{--bg:#14171d;--fg:#e4e8ef;--mut:#9aa4b5;--line:#2c3340;--head:#1d222b;--acc:#7aa7ee;--ok:#5cc489;--warn:#e0a050}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,'Noto Sans KR',sans-serif}main{max-width:1250px;margin:0 auto;padding:20px 16px 60px}
h1{font-size:1.45rem}h2{margin:2rem 0 .2rem;font-size:1.1rem;border-bottom:1px solid var(--line);padding-bottom:.3rem}p.d{margin:.2rem 0 .6rem;color:var(--mut);font-size:.88rem}
.note{border-left:3px solid var(--warn);padding:.3rem .9rem;margin:.8rem 0;color:var(--mut);font-size:.86rem}
table{border-collapse:collapse;width:100%;display:block;overflow-x:auto;font-size:.82rem}th,td{border:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}th{background:var(--head);position:sticky;top:0}
code{font-size:.78em;background:var(--head);padding:1px 4px;border-radius:3px;word-break:break-all}.m{color:var(--mut);font-size:.78rem}.pn{font-weight:700;color:var(--acc);white-space:nowrap}
.no{color:var(--warn)}details{margin:.5rem 0;border:1px solid var(--line);border-radius:6px;padding:4px 10px}summary{cursor:pointer;padding:4px 0}input{padding:6px 9px;border:1px solid var(--line);border-radius:6px;background:var(--bg);color:var(--fg);width:min(420px,100%)}"""
    body = [f'<h1>모델 세대 레지스트리</h1><p class=d>생성 {datetime.date.today()} · 출처 MANIFEST.json, episode_log.csv(origin/monavla-driving), docs/v5/egvla2_fm/*.json · PN은 제안안 EG{{세대}}-{{순번}}</p>',
            '<div class=note>실기 = 운영자 수동 판정(블라인드 아님, 성공 기준 문서화 안 됨, 좌우 정렬 미반영). 세대 간 오프라인 수치는 평가 방식이 달라 직접 비교하지 않는다. 같은 3×5×20 프로토콜에서의 실기 비교만 세대를 잇는다.</div>',
            '<p><input id=q placeholder="PN·이름·체크포인트 검색"></p>']
    for gk, (gn, gd) in GENS.items():
        rs = [r for r in rows if r["gen"] == gk]
        trs = "".join(f'<tr><td class=pn>{r["pn"]}</td><td><b>{esc(r["name"])}</b><br><span class=m>{esc(r["status"])}</span></td><td>{("<code>"+esc(r["ckpt"])+"</code>") if r["ckpt"] else ""}{("<br><span class=m>"+esc(r["path"])+"</span>") if r["path"] else ""}</td><td>{esc(r["arch"])}</td><td>{esc(r["train"])}</td><td>{esc(r["offline"])}</td><td>{cell(r)}</td><td>{esc(r["note"])}</td><td>{links_cell(r)}</td><td>{("<span class=m>"+str(round(r["bytes"]/1e6,1))+"MB</span>") if r["bytes"] else ("<span class=no>파일 없음</span>" if r["path"] and not r["exists"] else "")}{("<br><code>"+r["sha256"][:10]+"</code>") if r["sha256"] else ""}</td></tr>' for r in rs)
        body.append(f'<h2>{gk} · {gn}</h2><p class=d>{gd}</p><table><thead><tr><th>PN</th><th>모델</th><th>체크포인트/경로</th><th>구조</th><th>학습</th><th>오프라인</th><th>실기(로그)</th><th>비고</th><th>결과·세션</th><th>파일</th></tr></thead><tbody>{trs}</tbody></table>')
    def alinks(ls): return "<br>".join('<a href="' + u + '">' + esc(l) + '</a>' for l, u in ls)
    dsr = "".join(f'<tr><td class=pn>{d[0]}</td><td><b>{esc(d[1])}</b></td><td><code>{esc(d[2])}</code></td><td>{esc(d[3])}</td><td>{esc(d[4])}</td><td>{alinks(d[5])}</td></tr>' for d in DATASETS)
    body.append(f'<h2>데이터셋</h2><p class=d>경로는 저장소 기준. 사용 모델 열은 학습·평가에 쓰인 데이터. 확인 못한 항목은 (추정)·미확인으로 표시.</p><table><thead><tr><th>ID</th><th>이름</th><th>경로</th><th>설명</th><th>사용 모델</th><th>결과 페이지</th></tr></thead><tbody>{dsr}</tbody></table>')
    sec = []
    for r in rows:
        sl = ss.get(r["log_key"] or "(공란)", []) if r["log_key"] not in (None, "__none__") else []
        if not sl: continue
        items = "".join(f'<tr><td><code>{x["sid"] or "-"}</code></td><td>{x["date"]}</td><td>{esc(x["route"])}</td><td>{"성공" if x["ok"] else "<span class=no>실패</span>"}</td><td>{x["steps"]}</td><td>{"✓" if x["h5"] else "—"}</td><td>{"✓" if x["film"] else ""}</td></tr>' for x in sl)
        sec.append(f'<details id="ses-{r["pn"]}"><summary><b class=pn>{r["pn"]}</b> {esc(r["name"])} — 세션 {len(sl)}건</summary><table><thead><tr><th>session_id</th><th>시각</th><th>경로 라벨</th><th>결과</th><th>steps</th><th>로컬 H5</th><th>필름스트립</th></tr></thead><tbody>{items}</tbody></table></details>')
    body.append('<h2>모델별 실기 세션</h2><p class=d>episode_log의 session_id 기준. 로컬 H5 = 이 머신 docs/inference_sessions에 파일이 있는 세션. 필름스트립 = CH79에 사진이 있는 세션(10/29·11/04분).</p>' + "".join(sec))
    body.append("<script>document.getElementById('q').oninput=e=>{const s=e.target.value.toLowerCase();document.querySelectorAll('tbody tr').forEach(r=>r.style.display=r.textContent.toLowerCase().includes(s)?'':'none')}</script>")
    (D / "model_registry.html").write_text(f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>모델 세대 레지스트리</title><style>{css}</style></head><body><main>{"".join(body)}</main></body></html>', encoding="utf-8")
    print(len(rows), "models; robot log error:", err)
main()

"""VLAops 🔤 탭용 '문구·표현별 테스트 — 어떻게 했고 결과는' 내부 탭 카드(정적 HTML). 원자료: text_robustness.json, text_cond_t2.json, text_cond_t4(_phrases).json, text_cond_t5.json(있으면)"""
import json, numpy as np
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; FM = ROOT / "docs/v5/egvla2_fm"
def _j(n):
    p = FM / n; return json.loads(p.read_text()) if p.exists() else None
def _flow(steps):
    cells = []
    for i, (t, b) in enumerate(steps):
        cells.append(f'<div style="flex:1;min-width:150px;background:#0f1d33;border:1px solid #1e3a5f;border-radius:8px;padding:8px 10px"><div style="font-size:0.7rem;color:#38bdf8">{t}</div><div style="font-size:0.78rem;line-height:1.5">{b}</div></div>')
        if i < len(steps) - 1: cells.append('<div style="align-self:center;color:#64748b;font-size:1.2rem">▶</div>')
    return '<div style="display:flex;gap:6px;flex-wrap:wrap;margin:8px 0 12px">' + "".join(cells) + '</div>'
def _tbl(h, rows):
    return '<div style="overflow-x:auto"><table class="ops-table" style="min-width:640px;font-size:0.78rem"><thead><tr>' + "".join(f"<th>{x}</th>" for x in h) + '</tr></thead><tbody>' + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows) + '</tbody></table></div>'
def _img(src, cap=""): return f'<img src="{src}" style="width:100%;max-width:1000px;border-radius:8px;display:block;margin:6px 0"><div style="font-size:0.72rem;color:#94a3b8;margin-bottom:8px">{cap}</div>'

def build():
    T0 = _j("text_robustness.json")["summary"]; t2 = _j("text_cond_t2.json") or {}; ph = _j("text_cond_t4_phrases.json") or []; t5 = _j("text_cond_t5.json")
    # ── T0
    nm = {"ref": "gray basket(기준)", "para_bin": "gray bin", "para_container": "gray container", "para_the": "the gray basket", "para_sentence": "go to the gray bin", "generic_basket": "basket", "wrong_chair": "red chair", "wrong_person": "person", "wrong_box": "white box"}
    r0 = [[f"<code>{nm.get(k, v['phrase'])}</code>", f"{v['det_rate'] * 100:.0f}%", (f"{v['d_cx']:.3f}" if v.get('d_cx') is not None else "–"), (f"{v['d_first_ly']:.3f}" if v.get('d_first_ly') is not None else "–"), (f"{v['az_sign_agree'] * 100:.0f}%" if v.get('az_sign_agree') is not None else "–")] for k, v in T0.items()]
    tab0 = (_flow([("① 사진", "10/29 실기 H5에서 24세션 × 3프레임 = 72장(RGB)"), ("② 문구", "9종(동의 표현 5, 정상 변형 1, 다른 물체 3)"), ("③ 처리", "Florence-2 phrase grounding → bbox, 같은 A3 헤드로 첫 스텝 명령 계산 (학습 없음)"), ("④ 판정", "기준 문구 대비 검출률·Δcx·첫 스텝 차이")]) +
            _img("figures/ch79/text_phrases_grid.png", "실제 프레임 4장 × 문구 9종. 빨간 박스 = 그라운딩. 장면에 다른 물체가 없어 red chair도 회색 통을 추측") +
            _tbl(["문구", "검출률", "Δcx", "첫 스텝 ly 차이", "방향 부호 일치"], r0) + '<div style="font-size:0.76rem;color:#94a3b8;margin-top:6px">패러프레이즈는 박스가 완전히 같다(Δ0). 이 경로(모듈형)에서는 헤드가 문구를 보지 않아 지시문을 바꿔도 명령이 같다.</div>')
    # ── T2
    def a2(k, v):
        x = [r[k] for kk, r in t2.items() if kk.split("|")[1] == v and k in r]; return f"{np.mean(x) * 100:.0f}%" if x else "–"
    r2 = [["학습에 쓴 부정 3개 (red chair·person·blue bottle) 정지율"] + [a2("stop_neg_nobbox", v) for v in ("V1_text_const", "V2_paraphrase_aug", "V3_aug_neg")],
          ["안 본 부정 2개 (green plant·black laptop) 정지율"] + [a2("unseen_stop_neg_nobbox", v) for v in ("V1_text_const", "V2_paraphrase_aug", "V3_aug_neg")],
          ["안 본 동의 표현 2개 (grey bucket·the bin)에서 정지"] + [a2("unseen_stop_pos", v) for v in ("V1_text_const", "V2_paraphrase_aug", "V3_aug_neg")]]
    tab2 = (_flow([("① 사진", "학습 프레임(데이터셋 H5, BGR→RGB), 3프레임마다 1개 ≈ 9.7k"), ("② 문구", "학습: 정상 6 + 부정 3 / 평가: 학습에 없던 4개"), ("③ 처리", "융합 인코더 텍스트 풀링 → 헤드 → 첫 스텝 명령"), ("④ 판정", "|명령| &lt; 0.2 이면 정지. 부정은 높을수록, 정상은 낮을수록 좋음")]) +
            _img("figures/textcond/probe_0.jpg", "같은 사진에서 문구만 바꿔 본 예(아래 📷 카드에 6장 전체)") + _tbl(["지표", "V1 문구 고정", "V2 +패러프레이즈", "V3 +부정 지시"], r2) +
            '<div style="font-size:0.76rem;color:#94a3b8;margin-top:6px">V1·V2는 부정 신호가 없어 문구를 무시(≈V0). 안 본 부정은 V3만 멈춘다(82%). 문구 4개뿐이라 일반화 결론은 약함 → T4에서 36개로 확장.</div>')
    # ── T4
    def grp(g, key="V4"): x = [r[key] for r in ph if r["group"].startswith(g)]; return f"{np.mean(x):.0f}%" if x else "–"
    r4 = [[f"<code>{r['phrase']}</code>", r["group"].split(":")[0], f"{r['V0']:.0f}%", f"{r['V3']:.0f}%", f"<b>{r['V4']:.0f}%</b>"] for r in ph]
    tab4 = (_flow([("① 사진", "같은 학습 프레임(3.2k, 간격 9)"), ("② 문구", "학습 정상 40 + 부정 40 / 평가 36개는 <b>학습에 전혀 없음</b>: 동의 표현 16, 새 물체 14, 같은 색·다른 물체 4, 속성만 다름 2"), ("③ 처리", "V3(6+3)와 V4(40+40)를 같은 조건·3분할×시드3으로 학습"), ("④ 판정", "문구별 정지율 평균. 기준: 부정 ≥90%, 동의 표현 ≤V0+3%p")]) +
            _img("figures/ch79/text_cond_t4_phrases.png", "가는 선 = V0 기준 정지율. 정상(파랑)은 짧을수록, 부정(빨강·주황)은 길수록 좋다") +
            f'<div style="font-size:0.78rem;margin:6px 0">V4 평균 — 동의 표현 {grp("정상")} · 새 물체 {grp("부정: 새")} · 같은 색 다른 물체 {grp("부정: 색")} · 속성만 다름 {grp("모호")}</div>' + _tbl(["문구", "유형", "V0", "V3", "V4"], r4))
    # ── T5
    if t5 and len(t5) >= 27:
        VS5 = [("V0_img_only", "V0 이미지 전용"), ("V4_big_vocab", "V4 평균 풀링"), ("V5_token_attn", "V5 토큰 어텐션")]
        def a5(k, pct=True, nd=1):
            o = []
            for v, _ in VS5:
                x = [r[k] for kk, r in t5.items() if kk.split("|")[1] == v]; f = 100 if pct else 1; o.append(f"{np.mean(x) * f:.{nd}f}" + ("%" if pct else "") + f" ± {np.std(x) * f:.{nd}f}")
            return o
        r5 = [["안 본 부정 14개 정지율"] + a5("held_neg_stop_bbox"), ["같은 색·다른 물체 4개 정지율"] + a5("hard_neg_stop_bbox"), ["안 본 동의 표현 16개에서 정지"] + a5("held_pos_stop"), ["동의 표현으로 바뀐 명령 크기"] + a5("held_pos_eff", False, 3), ["기준 문구 방향 오차 (°)"] + a5("ang_ref", False, 1)]
        body5 = _tbl(["지표"] + [n for _, n in VS5], r5) + '<div style="font-size:0.76rem;color:#94a3b8;margin-top:6px">같은 어휘·분할·시드·평가. 학습에 없는 문구만 평가.</div>'
    else:
        body5 = '<div style="font-size:0.82rem;color:#fbbf24">학습 진행 전(토큰 캐시 계산 중) — 결과가 나오면 이 탭이 채워진다.</div>'
    tab5 = (_flow([("① 사진", "T4와 같은 프레임"), ("② 문구", "T4와 같은 정상 40 + 부정 40"), ("③ 처리", "융합 인코더 마지막 6개 토큰 → <b>이미지 특징이 질의인 크로스 어텐션(4 head)</b> → 768-d. 평균 풀링 대신 토큰을 직접 읽음"), ("④ 판정", "T4와 같은 지표·기준")]) + body5)
    import subprocess
    def dates(*files):
        """git 로그로 작성일(첫 커밋)·갱신일(마지막 커밋). 커밋 전이면 '아직 없음'."""
        ds = []
        for f in files:
            out = subprocess.run(["git", "log", "--format=%ad", "--date=format:%Y-%m-%d", "--", f], cwd=ROOT, capture_output=True, text=True).stdout.split()
            ds += out
        if not ds: return "작성: 아직 없음(결과 대기) "
        a, b = min(ds), max(ds); return f"작성 {a} · 최종 갱신 {b}"
    def _pl(code, title, status, color, rows):
        return (f'<div style="border:1px solid #1e3a5f;border-radius:8px;padding:10px 12px;margin:0 0 12px;background:#0b1730"><div style="display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap">'
                f'<b style="font-size:0.88rem">{code} — {title}</b><span style="font-size:0.72rem;padding:2px 8px;border-radius:10px;background:{color}22;color:{color};border:1px solid {color}55">{status}</span></div>'
                + _tbl(["항목", "내용"], [[f"<b>{a}</b>", b] for a, b in rows]) + '</div>')
    G_, Y_, B_, R_ = "#34d399", "#fbbf24", "#38bdf8", "#f87171"
    plans = (
        '<div style="font-size:0.78rem;color:#94a3b8;margin-bottom:10px">원문: docs/plans/plan_20261106_text_conditioned_e2e_vla.md · plan_20261108_text_vocab_expansion.md · plan_20261108_text_token_attention_head.md · HANDOFF_20261107_TEXT_OBJECT_SWAP_PHOTOS.md. 모든 오프라인 지표는 폐루프 성공을 예측하지 않는다(CH79).</div>' +
        _pl("T0", "문구 민감도 측정 (학습 없음)", "완료 11/06", G_, [
            ("목적", "지금 파이프라인에서 텍스트가 어디에 영향을 주는지(bbox 한 경로뿐인지) 수치화"),
            ("방법", "10/29 실기 H5 24세션×3프레임=72장 × 문구 9종 → Florence-2 그라운딩 + A3 헤드. 인코더 부분별(이미지·융합 전체·융합 이미지 토큰·융합 텍스트 토큰) 코사인"),
            ("파일", "scripts/eval_text_robustness.py · eval_text_visual.py · eval_text_encoders.py"),
            ("성공 기준", "융합 특징의 기준 대비 cos &lt; 0.99(텍스트가 특징에 실림)"),
            ("결과", "패러프레이즈는 박스 Δ0(모듈형은 지시문 변경에 강건). 이미지 풀링 특징은 문구와 무관(cos 1.000) → 현 헤드는 텍스트를 못 봄. 텍스트 토큰만 풀링하면 문구 차이가 가장 선명(go to the gray bin 0.773)")]) +
        _pl("T1·T2", "헤드에 텍스트 넣기 + 부정 지시 학습", "완료 11/07", G_, [
            ("목적", "텍스트를 쓰게 만드는 학습 신호가 필요한지 확인(귀무 가설: 문구가 상수면 무시)"),
            ("방법", "V0 이미지 전용 / V1 문구 고정 / V2 +패러프레이즈 6종 / V3 +부정 지시 20%(정답 정지, 절반은 박스 유지). A3 학습법 동일(서보 라벨 1.0·드롭아웃 30%·120 epoch), 3분할×시드3"),
            ("데이터", "기존 300 에피소드 재라벨(새 수집 없음), 3프레임 간격 ≈9.7k"),
            ("파일", "scripts/precompute_text_fused(_heldout).py · eval_text_cond_t2.py"),
            ("성공 기준", "부정 정지 ≥90%, 정상 방향 오차 ≤ 기준+1°, 패러프레이즈 명령 불변"),
            ("결과", "V1·V2는 텍스트 무시(귀무 확인), V3만 부정 문구에 정지(안 본 문구 82%). 동의 표현 취약(명령 변화 0.23)")]) +
        _pl("T4", "어휘 확장 (정상 40 + 부정 40)", "완료 11/08", G_, [
            ("목적", "T2의 취약성이 어휘가 적어서인지 확인"),
            ("방법", "학습 80개, 평가는 학습에 전혀 없는 36개(동의 표현 16·새 물체 14·같은 색 다른 물체 4·속성만 다름 2). 프레임 간격 9로 V0·V3·V4를 같은 조건에서 새로 학습(27회)"),
            ("파일", "scripts/text_vocab.py · precompute_text_vocab.py · eval_text_cond_t4.py"),
            ("성공 기준", "안 본 부정 정지 ≥90% · 동의 표현 명령 변화 ≤0.10 · 동의 표현 정지 ≤ V0+3%p"),
            ("결과", "99.5% 충족 · 0.077 충족 · +3.1%p 근소 미충족. 같은 색 다른 물체도 100% 정지, grey bucket·gray pail은 22~24% 오정지")]) +
        _pl("T5", "텍스트 토큰 어텐션 헤드", "진행 중 (토큰 캐시 계산)", Y_, [
            ("목적", "평균 풀링이 흐리는 문구 순서·핵심 단어를 어텐션으로 읽으면 동의 표현 취약성이 줄어드는가"),
            ("방법", "융합 인코더 마지막 6토큰 → 이미지 특징이 질의인 크로스 어텐션(4 head) → 768-d. V4(평균 풀링)와 같은 어휘·분할·시드·평가, V0·V4·V5 27회"),
            ("파일", "scripts/precompute_text_tokens.py(약 2시간) → eval_text_cond_t5.py → text_cond_t5.json"),
            ("성공 기준", "T4와 같은 3기준 + V4 대비 동의 표현 정지율 감소. 개선이 없으면 \"어텐션 헤드 이득 없음\"으로 기록"),
            ("다음", "결과가 나오면 이 문서 4번 섹션과 CH79에 자동 반영")]) +
        _pl("T5-b", "지연·로드시간 측정", "대기 (GPU 점유 해제 후)", Y_, [
            ("목적", "E2E 변형(V4 평균 풀링·V5 어텐션)의 추가 비용 확인"),
            ("방법", "같은 GPU·fp16/fp32에서 헤드 forward 지연, 모델 로드 시간 각 50회. 기존 값(Florence-2 fused 서버 180~201ms, Jetson fp16 372ms, 헤드 약 54ms)과 같은 표에 정리"),
            ("파일", "scripts/bench_egvla2_unified.py 확장 예정(미작성)"),
            ("한계", "Jetson 수치는 로봇 쪽 측정 필요")]) +
        _pl("T6", "서버 지시문 인자화 + 로봇 사진 촬영", "대기 (사용자 전달·승인)", B_, [
            ("목적", "교수님 시험 ①②를 실제 장면으로 확인할 수 있게 함. 현재 서버는 문구를 gray basket으로 고정(egvla2_inference_server.py:143)"),
            ("1단계", "정지 사진 촬영(S0 통만 / S1 다른 물체만 / S2 통+물체 / S3 빈 바닥) 위치 좌·중·우 각 3장, 물체 후보 흰 상자·컵·의자·파란 병"),
            ("2단계", "서버를 요청 문구 → Florence-2 프롬프트로 전달(기본 gray basket 유지, 플래그 기본 꺼짐). 실기 각 5~10회"),
            ("파일", "docs/plans/HANDOFF_20261107_TEXT_OBJECT_SWAP_PHOTOS.md (로봇에 전달 필요)"),
            ("성공 기준", "물체 교체: 새 물체 단어로 도달, 빈 장면·없는 단어는 정지. 지시문 변경: 동의 표현에서 기준과 같은 도달")]) +
        _pl("T7", "기록 세션 재생 평가 (제안)", "제안 · 승인 필요", B_, [
            ("목적", "프레임 단위 지표의 한계를 줄이기 위해, 기록된 11/04 세션(297개) 프레임을 시간순으로 재생하며 정지 판정이 처음 나오는 시점을 비교"),
            ("방법", "A3·V4·V5가 같은 프레임열에서 명령을 내게 하고, 정상 문구는 실기 정지 시점 근처에서 멈추는지, 부정 문구는 처음부터 멈추는지 확인. 정지 후 재출발은 모사하지 못함"),
            ("한계", "관측이 실제 로봇의 행동에 의존하지 않는 오픈 루프 재생이라 폐루프 성공을 대신하지 못함")]) +
        _pl("T3", "물체 2개 중 고르는 데이터 (새 수집)", "미착수 · 결정 필요", R_, [
            ("목적", "\"단어로 목표를 고르는\" E2E 주장의 유일한 근거. 이것 없이는 불변성·거절만 보인 것"),
            ("방법", "장면에 회색 통 + 다른 물체를 두고, 문구가 가리키는 쪽으로 조이스틱 시연. 문구당 약 20 에피소드(기존 수집 절차 재사용)"),
            ("전제", "T5·T6 결과로 E2E 방향을 계속할지 먼저 판단"),
            ("결정 필요", "물체 종류, 문구 목록, 수집 일정(로봇 쪽 협의)")]))
    secs = [("pt-m", "0. 방법 한눈에", None, dates("docs/plans/plan_20261106_text_conditioned_e2e_vla.md")),
            ("pt-p", "T별 계획 (로드맵)", plans, dates("docs/plans/plan_20261108_text_vocab_expansion.md", "docs/plans/plan_20261108_text_token_attention_head.md", "docs/plans/HANDOFF_20261107_TEXT_OBJECT_SWAP_PHOTOS.md")),
            ("pt-0", "1. T0 그라운딩 9문구", tab0, dates("docs/v5/egvla2_fm/text_robustness.json", "scripts/eval_text_robustness.py")),
            ("pt-2", "2. T2 학습/미지 문구", tab2, dates("docs/v5/egvla2_fm/text_cond_t2.json")),
            ("pt-4", "3. T4 문구별 36개", tab4, dates("docs/v5/egvla2_fm/text_cond_t4.json")),
            ("pt-5", "4. T5 토큰 어텐션", tab5, dates("docs/v5/egvla2_fm/text_cond_t5.json", "docs/plans/plan_20261108_text_token_attention_head.md"))]
    meth = _tbl(["단계", "무엇을 바꿨나", "문구는 어떻게", "측정", "섹션"], [
        ["T0", "학습 없음(현 A3 모듈형)", "9종 직접 입력", "그라운딩 박스·명령 변화", "1"], ["T2", "헤드에 텍스트 특징 추가", "정상 6 + 부정 3 학습, 미지 4 평가", "정지율·명령 변화", "2"],
        ["T4", "어휘 40+40로 확장", "미지 36 평가(동의 표현·새 물체·같은 색·속성)", "문구별 정지율", "3"], ["T5", "평균 풀링 → 크로스 어텐션", "T4와 동일", "T4와 동일", "4"]]) + '<div style="font-size:0.76rem;color:#94a3b8;margin-top:8px">입력은 모두 타이핑한 텍스트. 부정 = 장면에 없는 물체 문구(정답 정지). 모든 지표는 오프라인(프레임 단위)이며 폐루프 성공을 예측하지 않는다.</div>'
    nav = "".join(f'<a class="ptn" href="#{i}" data-t="{i}" onclick="ptGo(event,\'{i}\')" style="display:block;padding:7px 10px;border-radius:6px;color:#cbd5e1;text-decoration:none;font-size:0.8rem;margin-bottom:2px">{n}</a>' for i, n, _, _ in secs)
    body = "".join(f'<div class="pts" id="{i}" style="padding:4px 4px 22px;border-bottom:1px solid #1e293b;margin-bottom:18px"><h3 style="margin:6px 0 4px;font-size:1rem;color:#e2e8f0">{n}</h3>{meth if b is None else b}<div style="font-size:0.7rem;color:#64748b;margin-top:10px;border-top:1px dashed #1e293b;padding-top:6px">🕒 {d}</div></div>' for i, n, b, d in secs)
    js = ("<script>(function(){const box=document.getElementById('pt-scroll');if(!box)return;"
          "window.ptGo=function(e,i){e.preventDefault();const t=document.getElementById(i);box.scrollTo({top:t.offsetTop-box.offsetTop-4,behavior:'smooth'})};"
          "function spy(){const ps=[...document.querySelectorAll('.pts')];let cur=ps[0].id;ps.forEach(p=>{if(p.offsetTop-box.offsetTop-12<=box.scrollTop)cur=p.id});"
          "document.querySelectorAll('.ptn').forEach(a=>{a.style.background=a.dataset.t===cur?'#1d4ed8':'transparent';a.style.color=a.dataset.t===cur?'#fff':'#cbd5e1'})}"
          "box.addEventListener('scroll',spy);spy()})()</script>")
    return (f'<section class="card"><div class="card-header"><div class="card-title"><span>🗂 문구·표현별 테스트 — 각각 어떻게 했고 결과는 (왼쪽 목차 · 스크롤)</span></div></div>'
            f'<div style="display:flex;gap:14px;align-items:flex-start"><nav style="flex:0 0 190px;position:sticky;top:0;padding-top:4px">{nav}</nav>'
            f'<div id="pt-scroll" style="flex:1;min-width:0;max-height:78vh;overflow-y:auto;padding-right:8px;position:relative">{body}</div></div>{js}</section>\n')

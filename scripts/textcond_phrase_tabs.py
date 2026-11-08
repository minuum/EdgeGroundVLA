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
    tabs = [("pt-m", "방법 한눈에", None), ("pt-0", "T0 그라운딩 9문구", tab0), ("pt-2", "T2 학습/미지 문구", tab2), ("pt-4", "T4 문구별 36개", tab4), ("pt-5", "T5 토큰 어텐션", tab5)]
    meth = _tbl(["단계", "무엇을 바꿨나", "문구는 어떻게", "측정", "탭"], [
        ["T0", "학습 없음(현 A3 모듈형)", "9종 직접 입력", "그라운딩 박스·명령 변화", "T0"], ["T2", "헤드에 텍스트 특징 추가", "정상 6 + 부정 3 학습, 미지 4 평가", "정지율·명령 변화", "T2"],
        ["T4", "어휘 40+40로 확장", "미지 36 평가(동의 표현·새 물체·같은 색·속성)", "문구별 정지율", "T4"], ["T5", "평균 풀링 → 크로스 어텐션", "T4와 동일", "T4와 동일", "T5"]]) + '<div style="font-size:0.76rem;color:#94a3b8;margin-top:8px">입력은 모두 타이핑한 텍스트. 부정 = 장면에 없는 물체 문구(정답 정지). 모든 지표는 오프라인(프레임 단위)이며 폐루프 성공을 예측하지 않는다.</div>'
    btn = "".join(f'<button class="ptb" data-t="{i}" onclick="ptShow(\'{i}\')" style="background:#0f1d33;color:#e2e8f0;border:1px solid #1e3a5f;border-radius:6px;padding:5px 12px;margin:0 6px 6px 0;cursor:pointer;font-size:0.78rem">{n}</button>' for i, n, _ in tabs)
    pan = "".join(f'<div class="ptp" id="{i}" style="display:{"block" if i == "pt-m" else "none"}">{meth if b is None else b}</div>' for i, n, b in tabs)
    js = "<script>function ptShow(i){document.querySelectorAll('.ptp').forEach(e=>e.style.display=e.id===i?'block':'none');document.querySelectorAll('.ptb').forEach(b=>{b.style.background=b.dataset.t===i?'#1d4ed8':'#0f1d33'})}ptShow('pt-m')</script>"
    return f'<section class="card"><div class="card-header"><div class="card-title"><span>🗂 문구·표현별 테스트 — 각각 어떻게 했고 결과는 (탭)</span></div></div>{btn}{pan}{js}</section>\n'

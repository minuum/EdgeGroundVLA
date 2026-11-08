"""CH79 🔤 지시문·E2E 탭(T0~T2): 지시문/물체 교체 시험 구조 진단, 인코더별 문구 민감도, 사진 확인, 텍스트 조건화 학습(T1·T2) 결과.
수치는 text_robustness.json · text_encoder_parts.json · text_cond_t2.json에서 읽는다(없으면 진행 중으로 표시).
자산: eval_text_robustness.py, eval_text_visual.py, eval_text_encoders.py, precompute_text_fused*.py, eval_text_cond_t2.py. plan: docs/plans/plan_20261106_text_conditioned_e2e_vla.md"""
import json

import numpy as np

import gen_ch76_viz as V
from ch79_closedloop import chip

FM = V.FM
A, G, R, B, P, MUT = V.A, V.G, V.R, V.B, V.P, V.MUT


def _load(name):
    p = FM / name
    return json.loads(p.read_text()) if p.exists() else None


def _agg(res, key, variants):
    out = {}
    for v in variants:
        vals = [r[key] for k, r in res.items() if k.split("|")[1] == v and key in r and r[key] == r[key]]
        out[v] = (np.mean(vals), np.std(vals), len(vals)) if vals else None
    return out


def _arrow(v, col, sz=52):
    c, sc, m = sz / 2, sz * 0.42, float(np.hypot(v[0], v[1]))
    if m < 0.2:
        return f'<svg width="{sz}" height="{sz}"><circle cx="{c}" cy="{c}" r="6" fill="{col}"/><text x="{c}" y="{sz - 3}" font-size="9" fill="#94a3b8" text-anchor="middle">정지</text></svg>'
    dx, dy = v[1] * sc, -v[0] * sc; x2, y2 = c + dx, c + dy; a = float(np.arctan2(dy, dx))
    h = lambda t: f"{x2 - 7 * np.cos(a - t):.1f},{y2 - 7 * np.sin(a - t):.1f}"
    return f'<svg width="{sz}" height="{sz}"><line x1="{c}" y1="{c}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{col}" stroke-width="3"/><polygon points="{x2:.1f},{y2:.1f} {h(0.45)} {h(-0.45)}" fill="{col}"/></svg>'


def _probe_html(t2, probes):
    vs = ["V0_img_only", "V1_text_const", "V2_paraphrase_aug", "V3_aug_neg"]
    ph = [("ref", "gray basket", "#34d399"), ("para_seen", "go to the gray bin", "#34d399"), ("para_unseen", "grey bucket(미학습)", "#fbbf24"), ("neg_seen_bbox", "red chair+박스 유지", "#fca5a5"),
          ("neg_seen_nobbox", "red chair+박스 없음", "#fca5a5"), ("neg_unseen_bbox", "green plant+박스 유지(미학습)", "#f87171"), ("neg_unseen_nobbox", "green plant+박스 없음(미학습)", "#f87171")]
    out = ""
    for m in probes:
        b = m["bbox"]; bw, bh = np.sqrt(b[2] * 9 / 16) * 100, np.sqrt(b[2] * 16 / 9) * 100
        photo = (f'<div style="position:relative;width:260px"><img src="figures/textcond/probe_{m["i"]}.jpg" style="width:100%;border-radius:6px;display:block">'
                 f'<div style="position:absolute;left:{b[0] * 100 - bw / 2:.1f}%;top:{b[1] * 100 - bh / 2:.1f}%;width:{bw:.1f}%;height:{bh:.1f}%;border:2px solid #ef4444;box-sizing:border-box"></div></div>')
        rows = ""
        for pk, pl, col in ph:
            cells = ""
            for v in vs:
                rs = [r["probe"][pk][m["i"]] for k, r in t2.items() if k.startswith(f"fold0|{v}|") and "probe" in r]
                cells += f'<td style="text-align:center">{_arrow([np.mean([x[0] for x in rs]), np.mean([x[1] for x in rs])], col) if rs else "–"}</td>'
            rows += f"<tr><td><code>{pl}</code></td>{cells}</tr>"
        out += (f'<div style="display:flex;gap:16px;flex-wrap:wrap;border-top:1px solid #1e293b;padding:12px 0"><div style="flex:0 0 260px">{photo}'
                f'<div style="font-size:0.7rem;color:#94a3b8;margin-top:4px">{m["ep"]} · f{m["frame"]} · {m["cell"]}<br>정답: 사람 {_arrow(m["human"], "#94a3b8", 40)} 서보 {_arrow(m["servo"], "#34d399", 40)}</div></div>'
                f'<div style="flex:1;overflow-x:auto"><table style="width:100%;font-size:0.76rem"><tr><th></th><th>V0</th><th>V1</th><th>V2</th><th>V3</th></tr>{rows}</table></div></div>')
    return out


def block(tbl, H, st):
    rob, enc, t2 = _load("text_robustness.json"), _load("text_encoder_parts.json"), _load("text_cond_t2.json")
    o = [H("0. 질문과 현재 구조 진단"),
         V.wrap("교수님 질문(11/05)과 지금 구조", '<div style="font-size:0.84rem;line-height:1.85">'
                '교수님 시험: <b>① 지시문을 바꿔도(예 “go to the gray bin”) 가는가</b> · <b>② 회색 통을 다른 물체로 바꾸고 그 물체 단어를 주면 가는가</b>(학습 없이 인식하는지). 사용자 방향: 텍스트가 행동에 직접 들어가는 <b>진짜 E2E VLA</b>.<br>'
                '현재(EGVLA2) 파이프라인: <b>텍스트 → Florence-2 디코더 → bbox</b> + <b>이미지 인코더 풀링 768-d(텍스트 무관)</b> → FM 헤드. 헤드는 텍스트를 직접 받지 않고, 학습 300에피소드의 문구도 <code>gray basket</code> 하나뿐이다.<br>'
                '→ 시험 ①·②는 <b>VLA가 아니라 Florence-2 그라운딩</b>이 문구를 따르는지를 묻는 시험이고, 텍스트를 헤드에 그대로 넣어도 <b>쓸 이유(그래디언트)가 없다</b>(Kosmos Exp17~41에서 text attention 0%였던 것과 같은 함정). 그래서 T0(측정) → T1(입력 교체) → T2(텍스트가 정답을 바꾸는 학습 신호)로 나눠 확인한다.</div>',
                "코드 근거: <code>precompute_v6_florence2_fused.run_batch</code>(풀링은 <code>img_feats.mean</code>), <code>egvla2_inference_server.py:143</code>(instruction 무시). 계획서: docs/plans/plan_20261106_text_conditioned_e2e_vla.md", "flow"),
         H("1. T0 — 측정(학습 없음)")]
    if rob:
        s = rob["summary"]
        rows = [[f"<code>{v['phrase']}</code>", f"{v['det_rate'] * 100:.0f}%", "–" if v["d_cx"] is None else f"{v['d_cx']:.3f}", f"{v['d_first_ly']:.3f}", f"{v['az_sign_agree'] * 100:.0f}%"] for v in s.values()]
        o.append(V.wrap(f"문구별 bbox·A3 명령 변화 (10/29 H5 {rob['n_frames']}프레임, 기준 = gray basket)", tbl(["문구", "검출률", "bbox cx 변화", "첫 스텝 ly 변화", "회전 부호 일치"], rows),
                        "패러프레이즈 5종은 기준과 bbox까지 같다(시험 ① 사실상 통과). <code>white box</code>도 회색 통을 잡는다 → 장면에 다른 물체가 없으면 오검출과 일반화를 구분할 수 없다. <code>red chair</code>는 4장 중 2장에서 왼쪽 위 붉은 물체(실제 붉은 의자일 수 있음), 1장은 오른쪽 가장자리 띠, 1장은 후보가 없자 <b>회색 통을 추측</b>(대상이 없어도 박스를 강제로 내는 경향). <code>person</code>은 화면 전체 박스(어댑터가 미검출 처리)."))
    o.append(V.wrap("실제 사진으로 확인 — 같은 사진, 문구만 변경(빨간 박스 = Florence-2 그라운딩)", V.img("figures/ch79/text_phrases_grid.png", "서로 다른 4세션 × 문구 9종. 장면에 회색 통 외 물체가 거의 없어 <b>이 사진만으로는 일반화와 오검출을 구분할 수 없다</b> — 다른 물체를 놓은 라이브 촬영이 필요(아래 한계·로봇 전달 문서). 화면 전체 박스(person)는 제목을 빨갛게 표시. 10/29 H5는 RGB 저장이라 채널을 뒤집지 않았다.")))
    if enc:
        names = [("vision", "① 이미지 인코더 풀링"), ("fused_all", "② 융합 인코더 전체 평균"), ("fused_img", "③ 융합 은닉 중 이미지 토큰"), ("fused_text", "④ 융합 은닉 중 텍스트 토큰")]
        rows = [[f"<code>{p}</code>"] + [f"{d[k]:.3f}" for k, _ in names] for p, d in list(enc.items())[1:]]
        o.append(V.wrap("인코더별 문구 민감도 (기준 문구 대비 코사인, 72프레임 평균 — 낮을수록 문구가 특징에 많이 실림)", V.img("figures/ch79/text_encoder_parts.png", "①은 모든 문구에서 1.000 → 현재 헤드 입력은 텍스트를 전혀 모른다. ②·③은 이미지 토큰 577개가 지배해 희석된다. ④ 텍스트 토큰만 풀링하면 문구 차이가 가장 선명하다(go to the gray bin 0.773).") + tbl(["문구"] + [n for _, n in names], rows),
                        "그래서 T1 입력은 [이미지 인코더 풀링 ; 융합 인코더의 텍스트 토큰 풀링](1536-d)로 정했다."))
    o.append(H("2. T1·T2 — 텍스트가 정답을 바꾸게 만들기"))
    o.append(V.wrap("변형 정의 (학습법은 모두 A3와 동일: 서보 라벨 1.0 · 서브골 드롭아웃 30% · 120 epoch · PiZeroFM)", tbl(["변형", "헤드 입력", "학습 때 문구", "정답"], [
        [chip("V0", MUT) + " 이미지 전용", "이미지 인코더 풀링 768-d", "—(텍스트 없음)", "사람 시연(서보 라벨)"],
        [chip("V1", B) + " T1: 텍스트 특징, 문구 고정", "[이미지 ; 텍스트 풀링] 1536-d", "gray basket만", "사람 시연 — <b>텍스트가 상수라 무시해도 됨</b>(귀무 가설)"],
        [chip("V2", G) + " T2a: 패러프레이즈 증강", "동일", "gray basket 외 5종 무작위(gray bin·container·the gray basket·go to the gray bin·basket)", "사람 시연 — 불변성"],
        [chip("V3", R) + " T2b: + 부정 지시", "동일", "V2 + 20%는 장면에 없는 물체(red chair·person·blue bottle)", "<b>정지(0)</b> · 그중 절반은 박스가 남아 있어도(오검출 상황) 정지"]]),
        "3프레임마다 1개(약 9.7k프레임)만 사용(캐시 계산 0.36초/프레임). 분할 fold0·fold1·lopo_center × 시드 0·1·2. 지표는 오프라인이며 폐루프 성공을 예측하지 못한다(CH79 결론)."))
    if t2:
        vs = ["V0_img_only", "V1_text_const", "V2_paraphrase_aug", "V3_aug_neg"]; lab = {"V0_img_only": "V0 이미지 전용", "V1_text_const": "V1 문구 고정", "V2_paraphrase_aug": "V2 +패러프레이즈", "V3_aug_neg": "V3 +부정 지시"}
        n_done = len(t2)
        def row(title, key, pct=False, fmt="{:.1f}"):
            a = _agg(t2, key, vs); cells = []
            for v in vs:
                x = a[v]; cells.append("–" if x is None else (f"{x[0] * 100:.1f}±{x[1] * 100:.1f}%" if pct else (fmt + "±" + fmt).format(x[0], x[1])))
            return [title] + cells
        seen = [row("방향 오차 — 기준 문구 (°, 낮을수록 좋음)", "ang_ref"), row("방향 오차 — 패러프레이즈 5종 (°)", "ang_para"),
                row("패러프레이즈로 바뀐 명령 크기 (낮을수록 불변)", "para_effect", fmt="{:.3f}"),
                row("<b>부정 문구 → 정지율</b> (박스 없음, 높을수록 좋음)", "stop_neg_nobbox", True), row("<b>부정 문구 → 정지율</b> (박스 남음, 오검출 상황)", "stop_neg_bbox", True),
                row("정상 문구에서 정지(낮아야 함)", "stop_pos_bbox", True), row("문구만 바꿨을 때 명령 변화 크기", "text_effect", fmt="{:.3f}")]
        o.append(V.wrap(f"학습에 쓴 문구로 평가 (분할×시드 평균±표준편차, 완료 {n_done}/36)", tbl(["지표"] + [lab[v] for v in vs], seen),
                        "부정 3문구는 V3 학습에도 쓰였으므로 이 표만으로는 '단어를 이해'했는지 '외웠는지' 구분되지 않는다 → 아래 미지 문구 표가 핵심."))
        if any("unseen_stop_neg_bbox" in r for r in t2.values()):
            un = [row("미지 패러프레이즈(grey bucket·the bin) 방향 오차 (°)", "unseen_ang_para"), row("미지 패러프레이즈로 바뀐 명령 크기", "unseen_para_effect", fmt="{:.3f}"),
                  row("미지 패러프레이즈에서 정지(낮아야 함)", "unseen_stop_pos", True),
                  row("<b>미지 부정(green plant·black laptop) → 정지율</b> (박스 없음)", "unseen_stop_neg_nobbox", True), row("<b>미지 부정 → 정지율</b> (박스 남음)", "unseen_stop_neg_bbox", True)]
            o.append(V.wrap("학습에 없던 문구로 평가 — 교수님 시험 ②에 가장 가까운 오프라인 대응", tbl(["지표"] + [lab[v] for v in vs], un),
                            "평가 프레임: 시험 분할의 3번째 프레임마다 1개. V0은 텍스트를 받지 않으므로 문구가 바뀌어도 출력이 같다."))
    else:
        o.append(V.wrap("T1·T2 결과", '<div style="font-size:0.84rem">학습 진행 중 — 보류 문구 캐시 완료 후 4변형 × 3분할 × 3시드 실행. 이 탭은 결과가 나오면 다시 생성된다.</div>', "", "flow"))
    if t2 and len(t2) >= 36:
        o.append(V.wrap("퍼센트 지표 한눈에 보기", V.img("figures/ch79/text_cond_bars.png", "입력은 전부 타이핑한 텍스트 문구이며 음성(소리) 입력은 없다. '부정'은 장면에 없는 물체 문구(red chair 등)를 뜻하고 정답은 정지. ②가 핵심: 학습에 없던 green plant·black laptop에서 V3만 정지(82%). ④는 학습에 없던 동의 표현(grey bucket·the bin)에서 정상인데 멈추는 비율 — V3가 12%로 가장 높아 패러프레이즈에 약함.")))
        g = lambda k, v, pct=False: (lambda x: f"{x[0] * (100 if pct else 1):.{1 if pct else 3}f}" + ("%" if pct else ""))(_agg(t2, k, [v])[v])
        o.append(V.wrap("결과 해석 (36/36: 3분할 × 시드 3, 오프라인)", '<div style="font-size:0.84rem;line-height:1.9">'
            f'<b>① 텍스트를 쓰게 만드는 데 성공.</b> V1·V2(부정 신호 없음)는 부정 문구에서도 정지율이 {g("stop_neg_nobbox", "V1_text_const", True)}·{g("stop_neg_nobbox", "V2_paraphrase_aug", True)}로 V0({g("stop_neg_nobbox", "V0_img_only", True)})와 같아 <b>텍스트를 무시</b>한다(귀무 가설 확인). 부정 지시를 학습한 V3는 학습한 부정 문구에서 {g("stop_neg_nobbox", "V3_aug_neg", True)}, <b>박스가 남아 있어도</b> {g("stop_neg_bbox", "V3_aug_neg", True)} 정지 → 텍스트가 박스를 이긴다.<br>'
            f'<b>② 안 본 단어에도 일부 일반화.</b> 학습에 없던 부정 문구(green plant·black laptop)에서 V3 정지율 {g("unseen_stop_neg_nobbox", "V3_aug_neg", True)}(박스 남음 {g("unseen_stop_neg_bbox", "V3_aug_neg", True)}) 대 V0~V2 약 10%. 단 <b>미지 문구가 2개뿐</b>이라 어휘 전반으로 일반화한다고 말할 수 없다.<br>'
            f'<b>③ 방향 정확도 비용은 보이지 않는다.</b> 기준 문구 방향 오차 V0 {g("ang_ref", "V0_img_only")}° / V3 {g("ang_ref", "V3_aug_neg")}°(분할 간 sd 약 3.7° 안쪽).<br>'
            f'<b>④ 그러나 패러프레이즈에 취약해진다 — 모듈형(V0)이 더 강건.</b> V0은 문구를 안 받으므로 명령 변화 0. 텍스트를 받는 변형은 학습한 패러프레이즈에서는 변화 {g("para_effect", "V2_paraphrase_aug")}(증강 효과, V1은 {g("para_effect", "V1_text_const")})로 줄지만, <b>학습에 없던 동의 표현(grey bucket·the bin)에서는 {g("unseen_para_effect", "V3_aug_neg")} 수준으로 여전히 크고</b> 방향 오차도 {g("unseen_ang_para", "V0_img_only")}° → {g("unseen_ang_para", "V3_aug_neg")}°로 나빠진다. 증강 5종은 외우는 수준이다.<br>'
            '<b>⑤ 사진에서 확인되는 것(fold0 프레임 6장, 시드 평균).</b> 문구를 고정해 학습한 V1은 텍스트 임베딩이 멀리 벗어난 <code>go to the gray bin</code>(텍스트 풀링 cos 0.77)에서 <b>6장 모두 명령 크기가 0.02~0.07(정지)</b>로 무너진다(기준 문구에서는 0.7~0.87). 패러프레이즈 증강(V2)·부정 지시(V3)는 이를 막지만, V3는 학습에 없던 <code>grey bucket</code>에서 명령이 0.36~0.74로 <b>머뭇거린다</b>(V0·V2는 0.65~0.92) → 문장형 지시와 미지 표현에 가장 약하다.<br>'
            '<b>결론(교수님 시험 관점):</b> 시험 ①(지시문 변경)은 <b>모듈형(현재 A3)이 오히려 안전</b>하다(Florence-2가 같은 박스를 주고 헤드는 문구를 안 봄). E2E(텍스트→헤드)는 "없는 물체는 안 간다"는 선택성을 얻는 대신 표현 변화에 흔들린다 → 어휘를 크게 늘린 증강(수십~수백 표현)과 다른 물체가 실제로 있는 장면 데이터가 있어야 이득을 주장할 수 있다. 실기 확인 전이며 오프라인 지표다.</div>', "한계: 부정 정답을 \"정지\"로 정한 설계, 미지 문구 4개, 3프레임 간격 부분집합, 단일 학습 설정. 미지 문구 정지 수치는 3분할 × 시드 3 평균(분할별 73~90%).", "flow"))
    t4 = _load("text_cond_t4.json")
    if t4 and len(t4) >= 27:
        V4s = ["V0_img_only", "V3_old_vocab", "V4_big_vocab"]; L4 = ["V0 이미지 전용", "V3 기존 어휘(정상6+부정3)", "V4 확장 어휘(정상40+부정40)"]
        def a4(k, pct=True, nd=1):
            out = []
            for v in V4s:
                x = [r[k] for kk, r in t4.items() if kk.split("|")[1] == v]; f = 100 if pct else 1
                out.append(f"{np.mean(x) * f:.{nd}f}±{np.std(x) * f:.{nd}f}" + ("%" if pct else ""))
            return out
        r4 = [["<b>안 본 부정 문구 14개 정지율</b> (박스 남음)"] + a4("held_neg_stop_bbox"), ["같은 색·다른 물체 4개 정지율(gray chair·cat·laptop·sofa)"] + a4("hard_neg_stop_bbox"),
              ["<b>안 본 동의 표현 16개에서 정지</b>(낮아야 함)"] + a4("held_pos_stop"), ["안 본 동의 표현으로 바뀐 명령 크기"] + a4("held_pos_eff", False, 3),
              ["안 본 동의 표현 방향 오차 (°)"] + a4("held_pos_ang", False, 1), ["기준 문구 방향 오차 (°)"] + a4("ang_ref", False, 1), ["속성만 다른 2개(red basket·blue bin) 정지 — 참고"] + a4("ambig_stop")]
        o.append(V.wrap("T4 어휘 확장 — 학습 어휘를 6+3개에서 40+40개로 늘리면? (평가는 학습에 없는 문구 36개만, 27/27)",
            V.img("figures/ch79/text_cond_t4_bars.png", "초록 점선 = 사전 선언한 성공 기준(plan_20261108). 입력은 모두 타이핑한 텍스트. 프레임 간격 9로 계산(약 3.2k프레임)해 V0·V3·V4를 같은 조건으로 새로 학습했으므로 위 T2 표와 수치를 직접 비교하지 않는다.") +
            tbl(["지표"] + L4, r4) +
            V.img("figures/ch79/text_cond_t4_phrases.png", "문구별(학습에 전혀 없던 36개) 정지율. 가는 선 = V0 기준. 파랑 = 정상 동의 표현(짧을수록 좋음), 빨강·주황 = 부정(길수록 좋음), 보라 = 속성만 다름(참고). 같은 문구 표는 VLAops 🔤 탭의 '문구·표현별 테스트' 탭에 있다.") +
            '<div style="font-size:0.84rem;line-height:1.9;margin-top:10px"><b>사전 기준 판정:</b> ① 안 본 부정 정지 ≥90% → <b>충족</b>(V4 99.5%, V3 81.5%) · ② 안 본 동의 표현 명령 변화 ≤0.10 → <b>충족</b>(V4 0.077, V3 0.136) · ③ 안 본 동의 표현 정지율 ≤ V0+3%p → <b>근소하게 미충족</b>(V4 10.8% vs V0 7.7% = +3.1%p).<br>'
            '<b>해석:</b> 어휘를 늘리자 안 본 부정 문구와 "색은 같고 물체가 다른" 문구(gray chair 100%, gray laptop 98%)까지 거의 다 멈추고, 동의 표현에 대한 흔들림도 V3보다 줄었다. 방향 오차 비용은 보이지 않는다(31.2° vs 30.7°, 분할 간 sd 3.3° 안). 그러나 동의 표현 정지율이 V0보다 약 3%p 높고 <code>gray pail</code>·<code>grey bucket</code> 같은 변형에서 22~24%가 멈춰, <b>패러프레이즈 취약성이 사라진 것은 아니다</b>.<br>'
            '<b>주의:</b> 속성만 다른 <code>blue bin</code>은 18%, <code>red basket</code>은 52% 멈춘다(정답이 불명확해 판정 제외). 오프라인·프레임 단위이고 장면에 다른 물체가 없는 사진이라, 선택 능력(두 물체 중 고르기)이나 폐루프 성공을 뜻하지 않는다.</div>',
            "plan: docs/plans/plan_20261108_text_vocab_expansion.md · 로그 runs/e2e_cache/logs/text_cond_t4.log · 원자료 docs/v5/egvla2_fm/text_cond_t4.json"))
    br = _load("bbox_robustness.json")
    if br and len(br) >= 27:
        Vb = [("V0_img_only", "V0 현재 A3 방식"), ("V0R_bbox_aug", "V0R 훼손 증강(보완안)"), ("V4_big_vocab", "V4 텍스트 조건화")]
        Cb = [("clean", "정상 박스"), ("drop50", "50% 제거"), ("drop100", "전부 제거"), ("jitter", "흔들림"), ("wrong", "반대편 오검출")]
        def m_(v, c, k):
            x = [r[c][k] for kk, r in br.items() if kk.split("|")[1] == v]; return float(np.mean(x)), float(np.std(x))
        rb = [[f"<b>{n}</b> 서보 라벨 대비 각도(°)"] + [f"{m_(v, c, 'servo_ang')[0]:.1f}±{m_(v, c, 'servo_ang')[1]:.1f}" for v, _ in Vb] for c, n in Cb] + [[f"<b>{n}</b> 박스 의존도(명령 변화)"] + [f"{m_(v, c, 'd_clean')[0]:.3f}" for v, _ in Vb] for c, n in Cb[1:]] + [[f"<b>{n}</b> 좌우 부호 일치"] + [f"{m_(v, c, 'ly_sign')[0] * 100:.1f}%" for v, _ in Vb] for c, n in (Cb[0], Cb[2], Cb[4])]
        dw = m_("V0_img_only", "wrong", "servo_ang")[0] - m_("V0R_bbox_aug", "wrong", "servo_ang")[0]; dc = m_("V0R_bbox_aug", "clean", "servo_ang")[0] - m_("V0_img_only", "clean", "servo_ang")[0]
        ok = dw >= 5 and dc <= 1
        o.append(V.wrap("C 상황 — 박스를 훼손했을 때 방향을 유지하나, 훼손 증강(V0R) 보완안 (27/27)", V.img("figures/ch79/bbox_robustness.png", "입력은 같은 프레임에서 박스만 합성 훼손(제거·흔들림·반대편 반전). 정지율은 정상 장면이라 낮아야 한다. 사전 선언 기준: wrong에서 V0 대비 서보 라벨 대비 각도 ≥5° 감소 + clean 악화 ≤1°. 지표는 첫 시도(사람 명령 대비 방향 오차)가 박스 의존성을 못 재서 교체했고 그 사실을 계획서에 기록했다.") + tbl(["조건"] + [n for _, n in Vb], rb) +
            f'<div style="font-size:0.84rem;line-height:1.9;margin-top:10px"><b>사전 기준 판정:</b> wrong 조건 V0→V0R 서보 라벨 대비 각도 변화 {dw:+.1f}°(기준 ≥5°), clean 악화 {dc:+.1f}°(기준 ≤1°) → <b>{"충족" if ok else "미충족"}</b>.<br>이 평가는 오프라인·합성 훼손이며 실제 검출 실패 분포와 다르다. 실기 세션의 그라운딩 실패 프레임으로 검증하는 것은 후속.</div>',
            "plan: docs/plans/plan_20261108_bbox_robustness_and_supplement.md · 원자료 docs/v5/egvla2_fm/bbox_robustness.json"))
    probes = _load("text_cond_probes.json")
    if t2 and probes and any("probe" in r for r in t2.values()):
        o.append(V.wrap("원본 사진으로 비교 — 같은 학습 프레임에서 변형별 첫 스텝 명령 (fold0 시험 프레임, 시드 평균)",
                        '<div style="font-size:0.78rem;color:#94a3b8;margin-bottom:8px;line-height:1.7">화살표: 위=전진, 오른쪽=우측 이동, 점=정지. 회색=사람 시연, 초록=서보 라벨(학습 정답). 정상 문구면 화살표가 정답과 같아야 하고 부정 문구면 점이어야 한다. 사진은 실험에 쓴 원본 프레임(붉은 상자=bbox 중심·면적 근사). 실시간 진행·흐름 대조: <a href="vlaops_dashboard.html" style="color:#38bdf8">VLAops 🔤 탭</a></div>' + _probe_html(t2, probes), "", "img"))
    o.append(V.wrap("한계", '<div style="font-size:0.82rem;line-height:1.8">· 오프라인(프레임 단위) 지표이며 폐루프 성공과 다름 · 부정 지시의 정답을 “정지”로 정한 것은 설계 선택(물체를 찾는 탐색 행동 등은 미포함) · 단어를 가리켜 <b>목표를 고르는</b> 능력(물체 2개 중 선택)은 데이터가 없어 T3(새 수집)이 필요 · <b>오프라인 사진에는 다른 물체가 없어</b> 부정·물체 교체 평가가 약함(대상이 없어도 박스를 추측해 냄) → 다른 물체를 실제로 놓고 찍은 사진·실기가 필요(docs/plans/HANDOFF_20261107_TEXT_OBJECT_SWAP_PHOTOS.md) · 실기 물체 교체 시험은 회색 통을 치운 장면에서 해야 의미(white box 사례) · 3프레임 간격 부분집합 사용</div>', "", "flow"))
    return "".join(o)

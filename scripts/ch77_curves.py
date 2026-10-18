"""CH77 ⑥-6: 서브골 ablation의 학습 곡선·과적합 점검(SVG 선 그래프 + 요약표). 데이터: ablation_curves.json"""
import json

import gen_ch76_viz as V

FM = V.FM
A, G, R, B, P, MUT = V.A, V.G, V.R, V.B, V.P, V.MUT
VC = {"full": B, "nobbox": R, "drop30": G}
VL = {"full": "서브골 있음", "nobbox": "서브골 없음", "drop30": "드롭아웃30%"}
FIX = {"B2_film_res_fm": 220, "C_pi0_adaln": 120}
SPN = {"fold0": "5-fold fold0 (같은 목표 위치)", "lopo_center": "LOPO center (안 본 목표 위치)", "lopo_strong_right": "LOPO strong_right (안 본 목표 위치)"}


def chart(series, title, ylabel, w=330, h=210, vline=None, ymax=None, ymin=None):
    """series: [(label, color, dash, [(x,y)...])]"""
    L, Rr, Tt, Bb = 42, 8, 22, 30
    xs = [p[0] for s in series for p in s[3]]; ys = [p[1] for s in series for p in s[3]]
    y0 = min(ys) if ymin is None else ymin; y1 = max(ys) if ymax is None else ymax
    pad = (y1 - y0) * 0.08 or 0.01; y0 -= pad; y1 += pad; x1 = max(xs)
    X = lambda x: L + (w - L - Rr) * x / x1
    Y = lambda y: h - Bb - (h - Bb - Tt) * (y - y0) / (y1 - y0)
    o = f'<svg viewBox="0 0 {w} {h}" style="width:100%;max-width:{w}px;background:#0a1220;border:1px solid #1e293b;border-radius:8px">'
    o += f'<text x="{L}" y="14" fill="#e2e8f0" font-size="10" font-weight="700">{title}</text>'
    for k in range(5):
        yv = y0 + (y1 - y0) * k / 4
        o += f'<line x1="{L}" y1="{Y(yv):.1f}" x2="{w - Rr}" y2="{Y(yv):.1f}" stroke="#1e293b"/><text x="{L - 4}" y="{Y(yv) + 3:.1f}" fill="#64748b" font-size="8" text-anchor="end">{yv:.3g}</text>'
    for xv in (0, 50, 100, 150, 200, 250):
        o += f'<text x="{X(xv):.1f}" y="{h - 16}" fill="#64748b" font-size="8" text-anchor="middle">{xv}</text>'
    o += f'<text x="{(L + w) / 2:.0f}" y="{h - 4}" fill="#64748b" font-size="8" text-anchor="middle">epoch · {ylabel}</text>'
    if vline:
        o += f'<line x1="{X(vline):.1f}" y1="{Tt}" x2="{X(vline):.1f}" y2="{h - Bb}" stroke="#fbbf24" stroke-dasharray="3 3"/><text x="{X(vline) + 3:.1f}" y="{Tt + 9}" fill="#fbbf24" font-size="8">ablation {vline}</text>'
    for lab, col, dash, pts in series:
        o += f'<polyline points="{" ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in pts)}" fill="none" stroke="{col}" stroke-width="1.8" {dash} stroke-linejoin="round"/>'
    o += '</svg>'
    return o


def legend(items):
    return '<div style="font-size:0.72rem;margin:4px 0">' + " &nbsp; ".join(f'<span style="color:{c}"><b>{"┅" if d else "━"}</b> {t}</span>' for t, c, d in items) + '</div>'


def summ(res):
    """분할 하나의 요약 히트 표: 행=헤드, 그룹=변형(3), 열=고정 epoch / 최적 epoch / 250 epoch."""
    import ch77_viz as Z
    grp = [(VL[v], ["고정 epoch", "최적(낙관)", "250"], [Z.VC[VL[v].replace("서브골 있음", "서브골 있음")] if False else VC[v]] * 3) for v in VC]
    rows = []
    for h, hn in (("B2_film_res_fm", "B2"), ("C_pi0_adaln", "C"), ("MLP_regression", "회귀")):
        fx = FIX.get(h, 60); vals, eps_ = [], []
        for v in VC:
            c = res[f"{h}|{v}"]; at = min(c, key=lambda r: abs(r["ep"] - fx)); ba = min(c, key=lambda r: r["angle"])
            vals.append([at["angle"], ba["angle"], c[-1]["angle"]]); eps_.append([f"ep {fx}", f"ep {ba['ep']}", "ep 250"])
        rows.append((f"{hn} 방향 오차", vals, "low")); rows.append((f'<span style="color:#64748b">{hn} epoch</span>', eps_, None))
    return Z.heat_table(grp, rows, fmt="{:.1f}°", corner="방향 오차(°)")


def block(tbl, H, st):
    f = FM / "ablation_curves.json"
    if not f.exists():
        return ""
    d = json.load(open(f)); out = H("⑥-6 학습 곡선과 과적합 점검 — 서브골 ablation은 과적합 때문인가") + (
        '<div style="font-size:0.8rem;line-height:1.6;margin:6px 0">ablation 결과(서브골을 빼도 정확도 동급)가 <b>고정 epoch 때문에 생긴 착시</b>인지 확인하려고, 같은 변형·헤드를 250 epoch까지 돌리며 '
        'train/hold-out 손실과 hold-out 방향 오차를 10 epoch마다 기록했다. 노란 점선은 ablation에서 쓴 고정 epoch(B2 220, C 120)이다. seed 0, 분할 3개.</div>')
    rows = []
    for sp, res in d.items():
        out += f'<div style="font-weight:800;color:#fcd34d;margin:14px 0 4px;font-size:0.85rem">{SPN.get(sp, sp)}</div>'
        figs = []
        # ① 학습 vs hold-out 손실 (full)
        ser = []
        for h, hn in (("B2_film_res_fm", "B2"), ("C_pi0_adaln", "C")):
            c = res[f"{h}|full"]
            col = B if hn == "B2" else A
            ser += [(f"{hn} hold-out", col, "", [(r["ep"], r["val"]) for r in c]), (f"{hn} train", col, 'stroke-dasharray="4 3" opacity="0.6"', [(r["ep"], r["train"]) for r in c])]
        figs.append(chart(ser, "학습 vs hold-out flow loss (서브골 있음)", "손실", ymin=0))
        # ② 방향 오차 B2, C (3 변형)
        for h, hn in (("B2_film_res_fm", "B2"), ("C_pi0_adaln", "C")):
            ser = [(VL[v], VC[v], "", [(r["ep"], r["angle"]) for r in res[f"{h}|{v}"]]) for v in VC]
            figs.append(chart(ser, f"{hn}: hold-out 방향 오차(°)", "방향 오차", vline=FIX[h]))
        # ③ 회귀
        c = res["MLP_regression|full"]
        figs.append(chart([("회귀 hold-out", MUT, "", [(r["ep"], r["val"]) for r in c]), ("회귀 train", MUT, 'stroke-dasharray="4 3" opacity="0.6"', [(r["ep"], r["train"]) for r in c])], "MLP 회귀: train vs hold-out MSE", "MSE", ymin=0))
        out += f'<div style="display:flex;gap:10px;flex-wrap:wrap">{"".join(f"<div style=flex:1 1 300px>{x}</div>" for x in figs)}</div>'
        out += summ(res)
    out += legend([("서브골 있음", B, False), ("서브골 없음", R, False), ("드롭아웃30%", G, False), ("train(점선)", MUT, True)])
    out += f'{st("읽는 법", A)} <b>손실</b>: train은 계속 내려가는데 hold-out이 올라가기 시작하면 과적합이다(최저 epoch과 마지막 차이가 +로 커질수록). <b>방향 오차</b>: 변형 간 순서(서브골 있음/없음/드롭아웃)가 모든 epoch에서 같은 모양이면 ablation 결론이 epoch 선택 때문이 아니다. 최적 epoch은 hold-out으로 고른 낙관적 값이다.'
    out += (f'{st("판정(이번 결과)", G)}<ul style="margin:6px 0 6px 18px;font-size:0.8rem">'
            '<li><b>과적합은 있지만 중간 정도다.</b> train 손실은 hold-out보다 크게 낮다(C: train 0.07 vs hold-out 0.13~0.25). hold-out 손실은 최저(C 약 60~140, B2 약 90~230 epoch) 이후 250까지 +0.00~0.07 오른다. 같은 장소(fold0)는 거의 평평하고 <b>안 본 목표 위치(LOPO)에서 더 뚜렷</b>하다.</li>'
            '<li><b>회귀가 가장 심하게 과적합한다</b>: LOPO에서 최적이 5~10 epoch인데 250 epoch은 +3~5° 나빠진다(예: strong_right 19.4° → 24.4°). FM 헤드(B2·C)는 250 epoch에서도 +1~3° 정도만 나빠져 <b>긴 학습에 더 덜 민감</b>하다.</li>'
            '<li><b>ablation의 고정 epoch은 대체로 합리적이었다</b>: C는 고정 120이 최적(약 80~160) 안에 있고, B2 220은 최적(90~240) 안에 있다. 다만 LOPO 일부(C strong_right, B2 center)는 고정 epoch이 최적보다 2~3° 나쁘다.</li>'
            '<li><b>서브골 효과는 epoch을 바꿔도 일관되지 않고 분할마다 다르다</b>: center를 뺐을 때는 B2에서 서브골을 <b>빼는 쪽이 3~5° 낫고</b>(고정 22.4 → 17.8°, 최적 19.9 → 16.6°), strong_right에서는 서브골을 빼면 0.5~1° 나쁘다. C는 ±1~2°로 일관된 방향이 없다. 즉 ablation의 "서브골은 정확도를 일관되게 올리지 않는다"는 결론은 <b>epoch 선택 때문이 아니다</b>.</li>'
            '<li><b>epoch마다 흔들림이 크다</b>: 같은 모델의 방향 오차가 10 epoch 간격으로 ±1~3° 오르내린다. 이 흔들림이 변형 간 차이(대부분 1° 안팎)와 같은 크기라, <b>seed·분할을 늘리지 않으면 작은 차이는 구분할 수 없다</b>.</li>'
            '<li>헤드 비교(최적 epoch, 서브골 있음): fold0 회귀 11.1 / B2 10.6 / C 10.0°, center 16.8 / 19.9 / 15.5°, strong_right 19.4 / 18.7 / 17.5° → <b>C가 세 분할 모두 최저</b>(최적 epoch 기준은 낙관적).</li></ul>')
    return out

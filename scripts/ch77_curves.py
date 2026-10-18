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
        for h, hn in (("B2_film_res_fm", "B2"), ("C_pi0_adaln", "C"), ("MLP_regression", "회귀")):
            for v in VC:
                c = res[f"{h}|{v}"]; at = lambda e: min(c, key=lambda r: abs(r["ep"] - e))
                bv, ba = min(c, key=lambda r: r["val"]), min(c, key=lambda r: r["angle"])
                fx = FIX.get(h, 60)
                rows.append([SPN[sp].split(" ")[0] + " " + SPN[sp].split(" ")[1], hn, VL[v], f'{at(fx)["angle"]:.1f}°', f'{ba["angle"]:.1f}° (ep {ba["ep"]})', f'{c[-1]["angle"]:.1f}°',
                             f'ep {bv["ep"]}', f'{c[-1]["val"] - bv["val"]:+.3f}'])
    out += legend([("서브골 있음", B, False), ("서브골 없음", R, False), ("드롭아웃30%", G, False), ("train(점선)", MUT, True)])
    out += H("곡선 요약 — 고정 epoch vs 최적 epoch vs 250") + tbl(["분할", "헤드", "변형", "고정 epoch 방향 오차", "최적 방향 오차(epoch)", "250 epoch 방향 오차", "hold-out 손실 최저 epoch", "250 vs 최저 손실 차"], rows)
    out += f'{st("읽는 법", A)} <b>손실</b>: train은 계속 내려가는데 hold-out이 올라가기 시작하면 과적합이다(최저 epoch과 마지막 차이가 +로 커질수록). <b>방향 오차</b>: 변형 간 순서(서브골 있음/없음/드롭아웃)가 모든 epoch에서 같은 모양이면 ablation 결론이 epoch 선택 때문이 아니다. 최적 epoch은 hold-out으로 고른 낙관적 값이다.'
    return out

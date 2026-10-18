"""CH77 시각화 헬퍼: 그룹 헤더 + 열별 색 히트 표(최고=초록, 최악=빨강 농도), 가로 막대 비교, 헤드·변형 색 팔레트."""
HC = {"MLP 회귀": "#94a3b8", "A": "#38bdf8", "B2": "#4ade80", "C": "#fbbf24"}      # 헤드 색
VC = {"서브골 있음": "#38bdf8", "서브골 없음": "#f87171", "드롭아웃30%": "#4ade80"}   # 변형 색
MUT, AL = "#94a3b8", "#fcd34d"


def _bg(v, lo, hi, better):
    """값 v를 같은 비교 묶음 안에서 0(좋음)~1(나쁨)로 정규화해 초록~빨강 배경을 낸다."""
    if hi - lo < 1e-12:
        return "transparent"
    x = (v - lo) / (hi - lo)
    if better == "high":
        x = 1 - x
    if x < 0.34:
        return f"rgba(74,222,128,{0.32 - 0.5 * x:.2f})"
    if x > 0.66:
        return f"rgba(248,113,113,{0.12 + 0.30 * (x - 0.66) / 0.34:.2f})"
    return "transparent"


def th(txt, color=None, span=1, left=False):
    c = f"border-bottom:3px solid {color};" if color else "border-bottom:2px solid #334155;"
    return f'<th colspan="{span}" style="padding:6px 8px;{c}color:{color or "#cbd5e1"};text-align:{"left" if left else "center"};font-size:0.76rem;white-space:nowrap">{txt}</th>'


def heat_table(groups, rows, fmt="{:.1f}", corner="", ref_label=None, note="", by="row"):
    """groups: [(그룹명, [열 라벨...], [열 색...]|None)]
    rows: [(행 라벨, [그룹별 값 리스트...], better('low'|'high'|None), [ref값...]|None)]  값은 숫자 또는 문자열(색 없음).
    숫자 값은 같은 행·같은 그룹 안에서 상대 색칠. ref는 사람 기준 같은 참조 열(색칠 제외)."""
    h1 = th(corner, left=True) + "".join(th(g, None, len(cols)) for g, cols, _ in groups)
    h2 = '<th style="padding:3px 8px"></th>'
    for g, cols, colors in groups:
        for i, c in enumerate(cols):
            h2 += th(c, (colors or [None] * len(cols))[i])
    if ref_label:
        h1 += th(ref_label, None, 1); h2 += '<th style="padding:3px 8px"></th>'
    out = (f'<table style="width:100%;border-collapse:collapse;margin:8px 0;font-size:0.8rem"><tr style="background:#0a1220">{h1}</tr><tr style="background:#0a1220">{h2}</tr>')
    colv = {}
    if by == "col":
        for r_ in rows:
            for gi_, vs_ in enumerate(r_[1]):
                for ci_, v_ in enumerate(vs_):
                    if isinstance(v_, (int, float)):
                        colv.setdefault((gi_, ci_), []).append(v_)
    for lab, vals, better, *rest in rows:
        out += f'<tr><td style="padding:6px 8px;border-bottom:1px solid #1e293b;color:#e2e8f0;font-weight:700;white-space:nowrap">{lab}</td>'
        for gi, (g, cols, _) in enumerate(groups):
            vs = vals[gi]
            nums = [v for v in vs if isinstance(v, (int, float))]
            lo, hi = (min(nums), max(nums)) if nums else (0, 0)
            for ci, v in enumerate(vs):
                if by == "col" and (gi, ci) in colv:
                    nums = colv[(gi, ci)]; lo, hi = min(nums), max(nums)
                if isinstance(v, (int, float)):
                    best = better and abs(v - (lo if better == "low" else hi)) < 1e-12 and len(nums) > 1
                    out += (f'<td style="padding:6px 8px;border-bottom:1px solid #1e293b;text-align:center;background:{_bg(v, lo, hi, better) if better else "transparent"};'
                            f'{"font-weight:900;color:#bbf7d0;" if best else "color:#e2e8f0;"}border-left:{"1px solid #1e293b" if gi else "none"}">{fmt.format(v)}</td>')
                else:
                    out += f'<td style="padding:6px 8px;border-bottom:1px solid #1e293b;text-align:center;color:{MUT}">{v}</td>'
        if ref_label:
            r = rest[0] if rest and rest[0] is not None else None
            out += f'<td style="padding:6px 8px;border-bottom:1px solid #1e293b;text-align:center;color:{AL};font-weight:700">{r if r is not None else "—"}</td>'
        out += "</tr>"
    out += "</table>"
    if note:
        out += f'<div style="font-size:0.72rem;color:#64748b;margin:-2px 0 8px">{note}</div>'
    return out


LEG = ('<div style="font-size:0.72rem;color:#94a3b8;margin:4px 0"><span style="background:rgba(74,222,128,0.35);padding:1px 8px;border-radius:3px">같은 묶음에서 가장 좋음</span> '
       '<span style="background:rgba(248,113,113,0.35);padding:1px 8px;border-radius:3px">가장 나쁨</span> (비교 묶음 안에서 상대 비교)</div>')


def bars(rows, maxv, unit="", ref=None, w=130):
    """rows: [(라벨, 값, 색)], ref: (라벨, 값) 기준선. 가로 막대 + 기준선."""
    o = ""
    for lab, v, col in rows:
        pct = min(100, v / maxv * 100)
        o += (f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;font-size:0.76rem"><div style="width:{w}px;color:#cbd5e1;font-weight:700">{lab}</div>'
              f'<div style="flex:1;position:relative;background:#0a1220;border-radius:4px;height:16px"><div style="width:{pct:.1f}%;background:{col};height:16px;border-radius:4px"></div>'
              + (f'<div style="position:absolute;left:{min(100, ref[1] / maxv * 100):.1f}%;top:-2px;height:20px;border-left:2px dashed #fcd34d"></div>' if ref else "")
              + f'</div><div style="width:70px;color:#e2e8f0;text-align:right">{v:.3g}{unit}</div></div>')
    if ref:
        o += f'<div style="font-size:0.7rem;color:#fcd34d;margin-left:{w + 8}px">┆ {ref[0]} {ref[1]:.3g}{unit}</div>'
    return o

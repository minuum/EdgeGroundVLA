"""CH78 ⚙️ 탭 하단: 오프라인 ablation 결과(runtime_ablation.json, fm_train_ablation.json) — 있는 결과만 그림/표로. 분할 평균(fold5 / 안 본 위치 5)."""
import json

import numpy as np

import gen_ch76_viz as V

FM = V.FM
A, G, R, B, P, MUT = V.A, V.G, V.R, V.B, V.P, V.MUT
TX, GRID = "#e2e8f0", "#334155"


def load(n):
    f = FM / n
    return json.load(open(f)) if f.exists() else {}


def agg(res, getter):
    """분할별 값 → (fold 평균, lopo 평균, 분할 수)"""
    fo, lo = [], []
    for s, v in res.items():
        try:
            x = getter(v)
        except (KeyError, TypeError):
            continue
        if x is None or (isinstance(x, float) and np.isnan(x)):
            continue
        (fo if s.startswith("fold") else lo).append(x)
    return (float(np.mean(fo)) if fo else None, float(np.mean(lo)) if lo else None, len(fo), len(lo))


def f(x):
    return "—" if x is None else f"{x:.2f}"


def paired(res, key, base, getter):
    """변형 key와 기준 base가 둘 다 있는 분할만 모아 (변형 fold, 변형 lopo, 기준 fold, 기준 lopo, fold 수, lopo 수) — 분할이 덜 찬 상태에서도 공정 비교."""
    vf, vl, bf, bl = [], [], [], []
    for sp, v in res.items():
        if key in v and base in v:
            try:
                a, b = getter(v[key]), getter(v[base])
            except (KeyError, TypeError):
                continue
            (vf if sp.startswith("fold") else vl).append(a); (bf if sp.startswith("fold") else bl).append(b)
    m = lambda x: float(np.mean(x)) if x else None
    return m(vf), m(vl), m(bf), m(bl), len(vf), len(vl)


def dif(a, b):
    return "" if a is None or b is None else f"{a - b:+.2f}"


def lines_svg(xs, series, ymin, ymax, xlabel):
    W, x0, y0, pw, ph = 760, 56, 14, 640, 160
    X = lambda i: x0 + i / (len(xs) - 1) * pw
    Y = lambda v: y0 + (ymax - v) / (ymax - ymin) * ph
    o = [f'<svg viewBox="0 0 {W} {ph + 52}" style="width:100%;max-width:{W}px" font-family="sans-serif">']
    for v in np.linspace(ymin, ymax, 5):
        o.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x0 + pw}" y2="{Y(v):.1f}" stroke="{GRID}" stroke-dasharray="2 4"/><text x="{x0 - 6}" y="{Y(v) + 4:.1f}" fill="{MUT}" font-size="11" text-anchor="end">{v:.1f}°</text>')
    for i, x in enumerate(xs):
        o.append(f'<text x="{X(i):.1f}" y="{y0 + ph + 16}" fill="{MUT}" font-size="11" text-anchor="middle">{x}</text>')
    o.append(f'<text x="{x0 + pw / 2}" y="{y0 + ph + 34}" fill="{MUT}" font-size="11" text-anchor="middle">{xlabel}</text>')
    for k, (name, vals, col, dash) in enumerate(series):
        pts = [(X(i), Y(v)) for i, v in enumerate(vals) if v is not None]
        if len(pts) < 2:
            continue
        o.append(f'<polyline points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}" fill="none" stroke="{col}" stroke-width="2.2" stroke-dasharray="{dash}"/>'
                 f'<text x="{pts[-1][0] + 5:.1f}" y="{pts[-1][1] + 4:.1f}" fill="{col}" font-size="10.5">{vals[-1]:.1f}</text>'
                 f'<text x="{x0 + 8 + k * 160}" y="{y0 + 12}" fill="{col}" font-size="10.5">{name}</text>')
    o.append("</svg>")
    return "".join(o)


def block(tbl, H, st):
    ra, ft, bd = load("runtime_ablation.json"), load("fm_train_ablation.json"), load("bold_ablation.json")
    ax = load("aux_ablation.json")
    grid, b1 = load("sampling_grid.json"), load("bold_ablation_seed1.json")
    out = ""
    if ra:
        d = {s: v["drop30"] for s, v in ra.items()}
        n = max(agg(d, lambda v: v["chunk_n1|vs_first"])[2:])
        out += H(f"오프라인 ablation 결과 — 런타임 요인 (분할 {len(ra)}/10 반영, 방향 오차 °)")
        ns = list(range(1, 9))
        ser = []
        for tag, key, col, dash in (("같은 위치·현재 스텝 기준", "vs_first", G, ""), ("같은 위치·이동 구간 3스텝 평균 기준", "vs_burst3", R, ""),
                                      ("안 본 위치·현재 스텝 기준", "vs_first", G, "5 4"), ("안 본 위치·이동 구간 3스텝 평균 기준", "vs_burst3", R, "5 4")):
            idx = 0 if "같은" in tag else 1
            ser.append((tag[:6] + tag[7:12], [agg(d, lambda v, n_=nn: v[f"chunk_n{n_}|{key}"])[idx] for nn in ns], col, dash))
        allv = [v for _, vals, _, _ in ser for v in vals if v is not None]
        if allv:
            out += V.wrap("청크 앞 n스텝 평균 — 정답을 무엇으로 보느냐에 따라 최적 n이 달라지나", lines_svg(ns, ser, min(allv) - 0.5, max(allv) + 0.5, "평균에 쓴 청크 스텝 수 n"),
                          "초록=정답이 현재 스텝, 빨강=정답이 이동 구간(3스텝) 평균. 실선=같은 위치, 점선=안 본 위치", kind="chart")
        rows = []
        for lab, gen in (("K 샘플 수", [(f"K={k}", f"K{k}|first") for k in (1, 2, 4, 8, 16)]), ("Euler 스텝", [(f"{s}스텝", f"euler{s}|first") for s in (1, 2, 3, 5, 10, 20)])):
            for nm, key in gen:
                a = agg(d, lambda v, k_=key: v[k_])
                rows.append([lab, nm, f(a[0]), f(a[1])])
        out += V.wrap("K·Euler 스텝 — 더 쓰면 좋아지나", tbl(["항목", "값", "같은 위치", "안 본 위치"], rows), "현재 서빙 K=8, Euler 10", kind="table")
        rows = []
        for al in (0.0, 0.3, 0.5, 0.7, 0.85):
            a = agg(d, lambda v, a_=al: v[f"ema{a_}|n1"]["vs_first"]); j = agg(d, lambda v, a_=al: v[f"ema{a_}|n1"]["jitter"])
            rows.append([f"α={al}(과거 가중)", f(a[0]), f(a[1]), f"{j[0]:.3f}" if j[0] is not None else "—"])
        out += V.wrap("EMA — 방향 오차와 저크의 맞바꿈", tbl(["EMA", "같은 위치", "안 본 위치", "저크"], rows), "서버 ema_alpha 0.5 = 과거 가중 0.5", kind="table")
        rows = []
        for dr in ("drop30", "drop50"):
            dd = {s: v[dr] for s, v in ra.items() if dr in v}
            a, b = agg(dd, lambda v: v["miss0|first"]), agg(dd, lambda v: v["miss1|first"])
            rows.append([dr.replace("drop", "드롭아웃 ") + "%", f(a[0]), f(b[0]), f(a[1]), f(b[1])])
        out += V.wrap("서브골 드롭아웃 30% vs 50%", tbl(["", "같은 위치·서브골 있음", "같은 위치·결측", "안 본 위치·서브골 있음", "안 본 위치·결측"], rows), kind="table")
    if ft:
        out += H("오프라인 ablation 결과 — FM-π0 학습 설정 (π0와 다른 점을 하나씩 바꿈)")
        base = "uniform|L2|d128"
        names = [k for k in ["uniform|L2|d128", "beta|L2|d128", "logitnormal|L2|d128", "uniform|L1|d128", "uniform|L4|d128", "uniform|L2|d64", "uniform|L2|d256"] if any(k in v for v in ft.values())]
        rows = []
        for vn in names:
            a = paired(ft, vn, base, lambda v: v["first"]); m_ = paired(ft, vn, base, lambda v: v["miss_first"]); pr = agg({s_: v for s_, v in ft.items() if vn in v}, lambda v: v[vn]["params"])
            rows.append([vn.replace("|", " · "), f"{int(pr[0]):,}" if pr[0] else "—", f(a[0]), dif(a[0], a[2]), f(a[1]), dif(a[1], a[3]), dif(m_[0], m_[2]), f"{a[4]}+{a[5]}"])
        out += V.wrap("시간 샘플링·층수·폭 변형", tbl(["변형(τ·층·폭)", "파라미터", "같은 위치", "차이", "안 본 위치", "차이", "같은 위치·서브골 결측 차이", "분할(fold+lopo)"], rows),
                      "uniform·L2·d128이 현재 배포 구성(차이는 같은 분할끼리의 기준 대비, 음수=좋음). beta는 π0 논문 방식. 모두 120 epoch 고정이라 0.2° 안팎 차이는 해석 주의", kind="table")
    if bd:
        out += H(f"오프라인 ablation 결과 — 덜 보수적인 후보 (분할 {len(bd)}/10 반영, 방향 오차 °)")
        lab = {"base": "기준(FM-π0, 현재 배포 구성)", "ep250": "학습 2배 길게(240 epoch)", "H4": "청크 길이 4로 재학습", "H16": "청크 길이 16으로 재학습", "w_first": "첫 스텝 손실 4배",
               "unit_target": "방향만 학습(크기 1)", "smooth_labels": "사람 조작 5프레임 이동평균", "feat_prev3": "3프레임 전 특징 추가(시간 맥락)", "bbox_noise": "bbox 입력 잡음", "w_curved": "곡선 구간 프레임 가중 3배"}
        rows = []
        for k, nm in lab.items():
            if not any(k in v for v in bd.values()):
                continue
            a = paired(bd, k, "base", lambda v: v["first"]); b = paired(bd, k, "base", lambda v: v["n1_vs_burst"])
            rows.append([nm, f(a[0]), "" if k == "base" else dif(a[0], a[2]), f(a[1]), "" if k == "base" else dif(a[1], a[3]), "" if k == "base" else dif(b[0], b[2]), f"{a[4]}+{a[5]}"])
        out += V.wrap("학습·데이터·목표 변형 (음수=기준보다 좋음)", tbl(["변형", "같은 위치", "차이", "안 본 위치", "차이", "같은 위치·이동 구간 기준 차이", "분할(fold+lopo)"], rows),
                      "정답은 모두 원본 사람 조작. 모든 변형 120 epoch 고정(최적 epoch가 변형마다 다를 수 있어 0.2° 안팎 차이는 변형 효과와 구분 불가). 같은 seed·같은 분할이라 학습 잡음은 약 ±0.1°", kind="table")
        rows = []
        for sg in (0.0, 0.5, 1.0, 1.5):
            a = agg(bd, lambda v, s_=sg: v["base"][f"sigma{s_}|first"]); rows.append([f"σ={sg}" + (" (현재)" if sg == 1.0 else ""), f(a[0]), f(a[1])])
        out += V.wrap("흐름 시작 노이즈 크기 (평가만 변경)", tbl(["초기 노이즈", "같은 위치", "안 본 위치"], rows), kind="table")
    if ax:
        merged = {sp: dict(v, **({"base": bd[sp]["base"]} if sp in bd and "base" in bd[sp] else {})) for sp, v in ax.items()}
        out += H(f"오프라인 ablation 결과 — 학습 전용 보조 목표(추론 비용 0), 분할 {len(ax)}개 반영")
        lab = {"aux_f3_w0.2": "3프레임 뒤 특징 예측(가중 0.2)", "aux_f8_w0.5": "8프레임 뒤 특징 예측(가중 0.5)", "aux_bbox3_w0.5": "3프레임 뒤 bbox 예측(가중 0.5)"}
        rows = []
        for k, nm in lab.items():
            if not any(k in v for v in ax.values()):
                continue
            a = paired(merged, k, "base", lambda v: v["first"]); b = paired(merged, k, "base", lambda v: v["n1_vs_burst"])
            rows.append([nm, f(a[0]), dif(a[0], a[2]), f(a[1]), dif(a[1], a[3]), dif(b[0], b[2]), f"{a[4]}+{a[5]}"])
        out += V.wrap("보조 목표 변형 (음수=기준보다 좋음)", tbl(["변형", "같은 위치", "차이", "안 본 위치", "차이", "같은 위치·이동 구간 기준 차이", "분할(fold+lopo)"], rows),
                      "보조 헤드는 학습에만 쓰고 배포 모델은 그대로(추론 비용 0). 기준은 덜 보수적 후보의 base와 같은 설정·같은 분할. 120 epoch 고정", kind="table")
    if grid:
        out += H(f"오프라인 ablation 결과 — 서빙 샘플링 조합 (분할 {len(grid)}개, 재학습 없이 평가만 변경, 방향 오차 °)")
        rows = []
        for sig in (0.0, 0.25, 0.5, 1.0):
            for k in ((1,) if sig == 0 else (1, 4, 8, 16)):
                cells = []
                for grp in ("fold", "lopo"):
                    for st in (3, 5, 10):
                        a = agg(grid, lambda v, key=f"s{sig}|K{k}|e{st}": v[key]["first"])
                        cells.append(f(a[0] if grp == "fold" else a[1]))
                rows.append([f"σ={sig}" + (" (현재)" if sig == 1.0 and k == 8 else ""), f"K={k}"] + cells)
        out += V.wrap("σ × K × Euler 스텝", tbl(["시작 노이즈", "샘플 수", "같은 위치·3스텝", "5스텝", "10스텝", "안 본 위치·3스텝", "5스텝", "10스텝"], rows),
                      "현재 서빙 = σ=1.0·K=8·10스텝. σ=0은 결정적 출력(조건부 평균에 가까움)이라 시작 구간 다봉에서 모드를 뭉갤 수 있음. 폐루프 확인 전 배포 변경 금지", kind="table")
    if b1 and bd:
        rows = []
        for tag, res in (("seed 0", bd), ("seed 1", b1)):
            a = paired(res, "w_first", "base", lambda v: v["first"])
            rows.append([tag, f(a[0]), dif(a[0], a[2]), f(a[1]), dif(a[1], a[3]), f"{a[4]}+{a[5]}"])
        out += V.wrap("첫 스텝 손실 가중 — seed별 재현", tbl(["", "같은 위치", "차이", "안 본 위치", "차이", "분할(fold+lopo)"], rows), "차이가 seed 사이에서 같은 부호이면 재현됨", kind="table")
    return out

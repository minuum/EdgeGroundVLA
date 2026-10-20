"""CH77 ⑥-11: 방향 오차 해부 — 지표의 출처·한계, 오차 분포, 진행도별 오차, 사람 방향 전환 근처 비중, 최선/중앙/최악 에피소드(썸네일·방향 시계열·궤적). 데이터: error_anatomy.json"""
import json

import numpy as np

import ch77_viz as Z
import gen_ch76_viz as V

FM = V.FM
A, G, R, B, P, MUT = V.A, V.G, V.R, V.B, V.P, V.MUT
VA, VB = 6.59, 19.85
SPN = {"fold0": "같은 목표 위치(fold0)", "lopo_strong_right": "안 본 목표 위치(strong_right)"}
BINS = [(0, 5), (5, 10), (10, 20), (20, 30), (30, 45), (45, 90), (90, 181)]
BCOL = [G, G, B, A, "#fb923c", R, R]


def ticks(lo, hi, k=5):
    return [lo + (hi - lo) * i / (k - 1) for i in range(k)]


def linechart(series, w=330, h=190, ymin=None, ymax=None, xmax=None, ylabel="°", xlabel="프레임", title=""):
    L, Rr, Tt, Bb = 40, 8, 20, 28
    xs = [p[0] for s in series for p in s[3] if p[1] is not None]; ys = [p[1] for s in series for p in s[3] if p[1] is not None]
    y0 = min(ys) if ymin is None else ymin; y1 = max(ys) if ymax is None else ymax; pad = (y1 - y0) * 0.08 or 1; y0 -= pad; y1 += pad; x1 = xmax or max(xs)
    X = lambda x: L + (w - L - Rr) * x / x1; Y = lambda y: h - Bb - (h - Bb - Tt) * (y - y0) / (y1 - y0)
    o = f'<svg viewBox="0 0 {w} {h}" style="width:100%;max-width:{w}px;background:#0a1220;border:1px solid #1e293b;border-radius:8px"><text x="{L}" y="13" fill="#e2e8f0" font-size="10" font-weight="700">{title}</text>'
    for yv in ticks(y0 + pad, y1 - pad):
        o += f'<line x1="{L}" y1="{Y(yv):.1f}" x2="{w - Rr}" y2="{Y(yv):.1f}" stroke="#1e293b"/><text x="{L - 4}" y="{Y(yv) + 3:.1f}" fill="#64748b" font-size="8" text-anchor="end">{yv:.0f}</text>'
    for xv in ticks(0, x1, 5):
        o += f'<text x="{X(xv):.1f}" y="{h - 14}" fill="#64748b" font-size="8" text-anchor="middle">{xv:.0f}</text>'
    o += f'<text x="{(L + w) / 2:.0f}" y="{h - 3}" fill="#64748b" font-size="8" text-anchor="middle">{xlabel} · {ylabel}</text>'
    for lab, col, dash, pts in series:
        seg = []
        for a, b in pts + [(None, None)]:
            if b is None:
                if len(seg) > 1:
                    o += f'<polyline points="{" ".join(seg)}" fill="none" stroke="{col}" stroke-width="2" {dash} stroke-linejoin="round"/>'
                elif len(seg) == 1:
                    cx, cy = seg[0].split(","); o += f'<circle cx="{cx}" cy="{cy}" r="1.6" fill="{col}"/>'
                seg = []
            else:
                seg.append(f"{X(a):.1f},{Y(b):.1f}")
    return o + "</svg>"


def path_svg(it, w=300, h=250):
    dt = np.array(it["dt"]); hu = np.array(it["human"]); pr = np.array(it["pred"])
    hm = np.linalg.norm(hu, axis=1); v_h = np.where(hm[:, None] > 1e-6, hu / np.maximum(hm[:, None], 1e-9) * (VA + VB * hm)[:, None], 0)
    pm = np.linalg.norm(pr, axis=1); go = (pm >= 0.2)[:, None]; v_p = np.where(go, pr / np.maximum(pm[:, None], 1e-9) * (VA + VB * 0.7), 0)
    ph = np.vstack([[0, 0], np.cumsum(v_h * dt[:, None], 0)]); pp = np.vstack([[0, 0], np.cumsum(v_p * dt[:, None], 0)])
    allp = np.vstack([ph, pp]); lat0, lat1 = allp[:, 1].min(), allp[:, 1].max(); fw = max(allp[:, 0].max(), 1)
    sc = min((w - 50) / max(lat1 - lat0, 60), (h - 50) / max(fw, 60))
    Pt = lambda p: f"{w - 25 - (p[1] - lat0) * sc:.1f},{h - 25 - p[0] * sc:.1f}"
    o = f'<svg viewBox="0 0 {w} {h}" style="width:100%;max-width:{w}px;background:#0a1220;border:1px solid #1e293b;border-radius:8px"><text x="10" y="14" fill="#e2e8f0" font-size="10" font-weight="700">적분 궤적(위 = 전진, 왼쪽 = 좌)</text>'
    o += f'<polyline points="{" ".join(Pt(p) for p in ph)}" fill="none" stroke="{G}" stroke-width="2.4"/><polyline points="{" ".join(Pt(p) for p in pp)}" fill="none" stroke="{A}" stroke-width="2.4" stroke-dasharray="6 3"/>'
    o += f'<circle cx="{Pt(ph[0]).split(",")[0]}" cy="{Pt(ph[0]).split(",")[1]}" r="4" fill="#fbbf24"/><circle cx="{Pt(ph[-1]).split(",")[0]}" cy="{Pt(ph[-1]).split(",")[1]}" r="4" fill="{G}"/><circle cx="{Pt(pp[-1]).split(",")[0]}" cy="{Pt(pp[-1]).split(",")[1]}" r="4" fill="{A}"/></svg>'
    return o, float(np.linalg.norm(ph[-1] - pp[-1]))


def block(tbl, H, st):
    f = FM / "error_anatomy.json"
    if not f.exists():
        return ""
    d = json.load(open(f)); S = d["splits"]; EX = d["examples"]
    src = tbl(["지표", "어디서 쓰나", "무엇을 재나", "이번 작업에서의 위치"], [
        ["SR · CR · SPL · NE", "내비게이션 VLA 논문(NaVILA, TIC-VLA 등 — 이번 원문 대조로 확인)", "성공률·충돌률·경로 효율·종점 오차(폐루프)", "<b>최종 성능 기준</b> → CH78 R7 폐루프 300회"],
        ["ADE / FDE", "궤적 예측·계획 분야(일반 지식, 이번 원문 확인은 안 함)", "예측 궤적과 정답 궤적의 평균·종점 거리", "리플레이 시뮬레이션의 ADE/FDE와 같은 계열(CH77 이산 vs 연속 탭)"],
        ["조향각 오차·코사인 유사도", "종단 주행 모방학습의 오프라인 평가(일반 지식)", "정답 조작과 예측 조작의 방향 일치", "<b>우리 방향 오차(°)가 이 계열</b>"],
        ["액션 MSE / R²", "행동 복제(BC) 전반", "값 자체의 거리", "함께 보고(방향이 맞아도 크기가 다를 수 있어서)"]])
    note = (f'{st("출처 솔직히", A)} <b>방향 오차(°)는 내비 논문의 표준 지표가 아니다.</b> 모방학습·종단 주행에서 쓰는 "조작 방향 일치도" 계열을 가져와 <b>내가(분석 쪽에서) 정한 진단 지표</b>이고(plan_20261015 §4), '
            '근거는 "방향이 핵심 결정이고 속력은 별도 규칙으로 다룬다"는 설계 때문이다. 한계: 성공과 직접 연결되지 않고 사람 조작이 정답이라 잡음이 있다 → 그래서 <b>사람 일치도 천장, 궤적 적분, 폐루프 SR</b>과 함께 읽는다.')
    ang_def = ('<svg viewBox="0 0 300 150" style="width:100%;max-width:300px;background:#0a1220;border:1px solid #1e293b;border-radius:8px"><text x="10" y="14" fill="#e2e8f0" font-size="10" font-weight="700">방향 오차 = 두 조작 벡터 사이각</text>'
               f'<line x1="40" y1="125" x2="230" y2="55" stroke="{G}" stroke-width="3"/><text x="232" y="52" fill="{G}" font-size="9">사람 (lx,ly)</text>'
               f'<line x1="40" y1="125" x2="215" y2="85" stroke="{A}" stroke-width="3" stroke-dasharray="6 3"/><text x="218" y="92" fill="{A}" font-size="9">모델 첫 스텝</text>'
               '<path d="M 100 107 A 62 62 0 0 0 100 117" fill="none" stroke="#f87171" stroke-width="1.6"/><text x="108" y="118" fill="#f87171" font-size="9">θ</text>'
               '<text x="10" y="145" fill="#64748b" font-size="8">둘 다 이동 중(크기>0.2)인 프레임만 평균 · 정지·속력 차이는 제외</text></svg>')
    # 분포
    def share(items):
        e = np.array([x for it in items for x in it["err"] if x is not None], float)
        return [float(((e >= a) & (e < b)).mean() * 100) for a, b in BINS], float(e.mean()), float(np.median(e)), len(e)
    dist = ""
    for sp in S:
        sh, mean, med, n = share(S[sp])
        rows = [(f"{a}~{b if b < 181 else '180'}°", v, c, f"{v:.1f}%") for (a, b), c, v in zip(BINS, BCOL, sh)]
        dist += f'<div style="flex:1 1 300px"><div style="font-weight:800;color:#fcd34d;font-size:0.8rem;margin:4px 0">{SPN[sp]} · 평균 {mean:.1f}° / 중앙값 {med:.1f}° · {n:,}프레임</div>{Z.bars(rows, 60, "", w=84)}</div>'
    # 진행도
    ser = []
    for sp, col in zip(S, (B, R)):
        pts = []
        for k in range(10):
            e = [x for it in S[sp] for t, x in enumerate(it["err"]) if x is not None and k <= t / it["n"] * 10 < k + 1]
            pts.append((k * 10 + 5, float(np.mean(e)) if e else None))
        ser.append((SPN[sp], col, "", pts))
    prog = linechart(ser, w=340, h=200, xmax=100, ymin=0, xlabel="에피소드 진행도(%)", ylabel="평균 방향 오차(°)", title="진행도별 평균 방향 오차")
    # 전환
    trow = []
    for sp in S:
        inn, out, hi_in, hi_tot = [], [], 0, 0
        for it in S[sp]:
            for t, x in enumerate(it["err"]):
                if x is None:
                    continue
                (inn if it["chg"][t] else out).append(x)
                if x > 30:
                    hi_tot += 1; hi_in += it["chg"][t]
        trow.append([SPN[sp], f"{len(inn) / (len(inn) + len(out)) * 100:.0f}%", f"{np.mean(inn):.1f}° / {np.mean(out):.1f}°", f"{hi_in / max(hi_tot, 1) * 100:.0f}% ({hi_in}/{hi_tot})"])
    trans = tbl(["평가 세트", "사람 방향 전환 근처 프레임 비중(±3프레임, 5프레임 전 대비 30° 이상 변화)", "전환 근처 / 그 외 평균 오차", "30° 초과 프레임 중 전환 근처 비율"], trow)
    # 발견(계산)
    fnd = []
    for sp in S:
        e = np.array([x for it in S[sp] for x in it["err"] if x is not None], float)
        first = [x for it in S[sp] for t, x in enumerate(it["err"]) if x is not None and t / it["n"] < 0.1]
        rest = [x for it in S[sp] for t, x in enumerate(it["err"]) if x is not None and t / it["n"] >= 0.1]
        hi = [it["chg"][t] for it in S[sp] for t, x in enumerate(it["err"]) if x is not None and x > 30]
        fnd.append(f"<li><b>{SPN[sp]}</b>: 10° 미만 {float((e < 10).mean() * 100):.0f}% · 30° 초과 {float((e > 30).mean() * 100):.0f}% · 에피소드 첫 10% 구간 평균 {np.mean(first):.0f}°, 나머지 {np.mean(rest):.0f}° · 30° 초과 프레임 중 사람 방향 전환 근처 {float(np.mean(hi) * 100):.0f}%</li>")
    finds = '<ul style="margin:6px 0 6px 18px;font-size:0.8rem">' + "".join(fnd) + "</ul>"
    # 예시
    cards = ""
    for sp in S:
        cards += f'<div style="margin:14px 0 4px;color:#fcd34d;font-weight:800;font-size:0.86rem">{SPN[sp]} — 최선 / 중앙 / 최악 에피소드</div><div style="display:flex;gap:12px;flex-wrap:wrap">'
        for role, lab, col in (("best", "최선", G), ("median", "중앙", B), ("worst", "최악", R)):
            nm = EX[sp][role]; it = next(x for x in S[sp] if x["name"] == nm)
            hu = np.array(it["human"]); pr = np.array(it["pred"]); hm = np.linalg.norm(hu, axis=1); pm = np.linalg.norm(pr, axis=1)
            ha = np.degrees(np.arctan2(hu[:, 1], hu[:, 0])); pa = np.degrees(np.arctan2(pr[:, 1], pr[:, 0]))
            sh = [(t, float(ha[t]) if hm[t] > 0.2 else None) for t in range(it["n"])]; sp_ = [(t, float(pa[t]) if pm[t] > 0.2 else None) for t in range(it["n"])]
            ts = linechart([("사람", G, "", sh), ("모델", A, 'stroke-dasharray="5 3"', sp_)], w=320, h=170, xmax=it["n"] - 1, title="조작 방향 시계열(°, 0=전진, +=좌)")
            ps, gap = path_svg(it)
            imgs = "".join(f'<img src="egvla2_fm/anatomy/{sp}_{role}_{k}.jpg" style="width:32%;border-radius:4px;margin-right:1%">' for k in range(3))
            cards += (f'<div style="flex:1 1 330px;background:#0d1420;border:2px solid {col};border-radius:10px;padding:8px 10px"><div style="font-size:0.8rem;font-weight:800;color:{col}">{lab} · 평균 {it["mean_err"]:.1f}° · 이동 프레임 {it["n_moving"]}/{it["n"]} · 종점 차 {gap:.0f}cm</div>'
                      f'<div style="font-size:0.7rem;color:#64748b;margin-bottom:4px">{nm[:-3]}</div><div style="margin-bottom:4px">{imgs}</div>{ts}<div style="margin-top:6px">{ps}</div>'
                      f'<div style="font-size:0.7rem;color:#94a3b8">초록 = 사람, 노랑 점선 = 모델(EMA 0.5). 궤적은 사람 라벨의 속력 단계와 모델 방향×0.7 속력(섀도)로 적분한 추정치.</div></div>')
        cards += "</div>"
    return (H("⑥-11 방향 오차 해부 — 지표 출처, 오차가 적은/많은 경우, 데이터와 모델 흐름")
            + '<div style="font-size:0.8rem;line-height:1.6;margin:6px 0">방향 오차가 어디서 나온 지표인지, 숫자가 실제 데이터에서 어떤 모습인지(오차가 작은 에피소드와 큰 에피소드)를 한곳에 모은 해부 탭이다. 설정은 배포와 같다(C, 드롭아웃 30%, 120 epoch, K=8, EMA 0.5), 안 본 에피소드 60개.</div>'
            + '<div style="margin:10px 0 4px;font-size:0.84rem;font-weight:800;color:#fcd34d">① 이 지표는 어디서 나왔나 — 다른 지표와 비교</div>' + src + note
            + '<div style="margin:12px 0 4px;font-size:0.84rem;font-weight:800;color:#fcd34d">② 어떻게 재나</div><div style="display:flex;gap:14px;flex-wrap:wrap;align-items:center">' + ang_def
            + '<div style="flex:1 1 300px;font-size:0.78rem;line-height:1.6;color:#cbd5e1">프레임마다 <b>사람 조작 (lx,ly)</b>와 <b>모델 첫 스텝 (lx,ly)</b>의 사이각을 잰다(둘 다 크기 0.2 초과일 때). 에피소드 평균을 다시 평균한 것이 우리 방향 오차다. 크기(속력)는 이 지표에 들어 있지 않고 R²·MSE로 따로 본다.</div></div>'
            + '<div style="margin:12px 0 4px;font-size:0.84rem;font-weight:800;color:#fcd34d">③ 오차 분포 — 대부분 작고 일부 프레임이 크다</div><div style="display:flex;gap:14px;flex-wrap:wrap">' + dist + '</div>'
            + '<div style="margin:12px 0 4px;font-size:0.84rem;font-weight:800;color:#fcd34d">④ 오차는 에피소드 어디에 몰리나</div><div style="display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start">' + prog
            + '<div style="flex:1 1 300px">' + trans + '<div style="font-size:0.74rem;color:#94a3b8">사람 조작이 갑자기 바뀌는 순간(예: 직진→사선)은 시점을 예측하기 어려워 오차가 몰린다. 이 구간이 큰 오차의 주된 위치인지 표로 확인.</div></div></div>'
            + '<div style="margin:12px 0 4px;font-size:0.84rem;font-weight:800;color:#fcd34d">발견(계산)</div>' + finds
            + '<div style="margin:12px 0 4px;font-size:0.84rem;font-weight:800;color:#fcd34d">⑤ 오차가 적은 경우와 많은 경우 — 실제 에피소드</div>' + cards
            + f'{st("읽는 법", G)} 사진은 에피소드의 시작·중간(최악은 오차 최대 프레임)·끝이다. 최선 에피소드는 사람과 모델의 방향 선이 거의 겹치고 궤적도 비슷하게 나온다. 최악 에피소드는 방향 선이 벌어지는 구간(어느 프레임인지 사진과 비교)이 있다. '
              f'{st("한계", R)} 평균 오차 수치는 에피소드 선택에 민감하고(seed 하나), 궤적은 라벨 속력 단계 기반 추정이다.')

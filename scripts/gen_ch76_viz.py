#!/usr/bin/env python3
"""research_story.html CH76 각 탭에 그래프·구조도·플로우차트를 (재)주입한다. 멱등: <!--VIZ76-->…<!--/VIZ76--> 블록을 교체.
데이터: /tmp/ch76_hist.json(스틱 크기 분포; 아래 compute_hist로 재생성 가능), docs/v5/egvla2_fm/summary_*.json.
실행: .venv/bin/python3 scripts/gen_ch76_viz.py
"""
import glob
import json
import re
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
HTML = ROOT / "docs/v5/research_story.html"
FM = ROOT / "docs/v5/egvla2_fm"
A, AL, G, R, B, P, MUT = "#fbbf24", "#fcd34d", "#4ade80", "#f87171", "#38bdf8", "#a78bfa", "#94a3b8"


def compute_hist():
    D = ROOT / "ROS_action/mobile_vla_dataset_egvla2/"
    S = json.load(open(ROOT / "docs/v5/egvla2_fm/meta/manual_dataset_sets.json"))["sets"]["pilot_v2_gauge_20261013"]
    names = {x if x.endswith(".h5") else x + ".h5" for x in S}
    bins = np.arange(0, 1.6, 0.1)
    v1, v2 = [], []
    for f in sorted(glob.glob(str(D / "*.h5"))):
        with h5py.File(f) as h:
            if h.attrs.get("action_space") == "egvla2_continuous_v1":
                v1.append(np.array(h["actions"]))
            elif Path(f).name in names:
                v2.append(np.array(h["actions_raw"]))
    def hist(L):
        m = np.max(np.abs(np.concatenate(L)), axis=1); m = m[m > 1e-3]
        return np.histogram(m, bins=bins)[0].tolist(), len(m)
    h1, n1 = hist(v1); h2, n2 = hist(v2)
    return dict(bins=bins.tolist(), v1=h1, n1=n1, v2=h2, n2=n2)


KINDS = {"chart": ("📊 그래프", B), "flow": ("🔀 구조도·플로우", P), "img": ("🖼 실제 데이터", G), "table": ("▦ 표·비교", A)}


def badge(kind):
    t, c = KINDS[kind]
    return f'<span style="display:inline-block;background:{c}22;color:{c};border:1px solid {c};border-radius:999px;padding:1px 10px;font-size:0.68rem;font-weight:800;margin-right:8px;vertical-align:middle">{t}</span>'


def wrap(title, inner, note="", kind=None):
    if kind is None:
        kind = "img" if "<img" in inner else "table" if "<table" in inner else "chart" if ("<svg" in inner or "height:16px" in inner) else "flow"
    c = KINDS[kind][1]
    n = f'<div style="font-size:0.74rem;color:#64748b;margin-top:8px;border-top:1px dashed #334155;padding-top:6px">※ {note}</div>' if note else ""
    return (f'<div style="background:#0a1628;border:1px solid #1e3050;border-left:6px solid {c};border-radius:10px;padding:14px 18px;margin:18px 0">'
            f'<div style="margin-bottom:10px">{badge(kind)}<span style="color:{AL};font-size:0.95rem;font-weight:800">{title}</span></div>{inner}{n}</div>')


def img(src, cap):
    return (f'<a href="{src}" target="_blank"><img src="{src}" style="width:100%;border-radius:8px;border:1px solid #1e3050;display:block"></a>'
            f'<div style="font-size:0.78rem;color:#94a3b8;margin-top:6px;line-height:1.6">{cap}</div>')


def hbars(rows, maxv, unit="", w=170):
    out = ""
    for label, val, col, *txt in rows:
        t = txt[0] if txt else f"{val:g}{unit}"
        out += (f'<div style="display:flex;align-items:center;gap:8px;margin:5px 0;font-size:0.76rem"><div style="width:{w}px;color:{MUT}">{label}</div>'
                f'<div style="flex:1;background:#0a1220;border-radius:4px;height:16px"><div style="width:{max(0.0, min(100.0, val / maxv * 100)):.1f}%;background:{col};height:16px;border-radius:4px"></div></div>'
                f'<div style="width:96px;color:#e2e8f0;text-align:right">{t}</div></div>')
    return out


def legend(items):
    return '<div style="display:flex;gap:14px;flex-wrap:wrap;font-size:0.74rem;color:%s;margin-bottom:8px">' % MUT + "".join(
        f'<span><span style="display:inline-block;width:10px;height:10px;background:{c};border-radius:2px;margin-right:5px"></span>{t}</span>' for t, c in items) + "</div>"


def vgroups(cats, series, maxv, h=150, fmt="{:g}", bw=16):
    """cats: 라벨 리스트, series: [(이름,색,[값...])]"""
    gw = (bw + 3) * len(series) + 14
    W = gw * len(cats) + 30
    svg = f'<svg viewBox="0 0 {W} {h + 34}" style="width:100%;max-width:{W * 1.6:.0f}px;height:auto" xmlns="http://www.w3.org/2000/svg">'
    svg += f'<line x1="26" y1="{h}" x2="{W}" y2="{h}" stroke="#334155"/>'
    for ci, c in enumerate(cats):
        x0 = 30 + ci * gw
        for si, (_, col, vals) in enumerate(series):
            v = vals[ci]; bh = max(0.0, v / maxv * (h - 14))
            x = x0 + si * (bw + 3)
            svg += f'<rect x="{x}" y="{h - bh:.1f}" width="{bw}" height="{bh:.1f}" fill="{col}" rx="2"/>'
            if bh > 0:
                svg += f'<text x="{x + bw / 2}" y="{h - bh - 3:.1f}" font-size="7.5" fill="#cbd5e1" text-anchor="middle">{fmt.format(v)}</text>'
        svg += f'<text x="{x0 + (gw - 14) / 2 - 2}" y="{h + 13}" font-size="8" fill="#94a3b8" text-anchor="middle">{c}</text>'
    return svg + "</svg>"


def flow(steps, direction="row"):
    """steps: [(텍스트, 색, 부제)] — 박스 + 화살표"""
    arrow = f'<div style="color:{MUT};font-size:1.1rem;align-self:center;padding:0 2px">{"→" if direction == "row" else "↓"}</div>'
    parts = []
    for t, col, *sub in steps:
        s = f'<div style="font-size:0.68rem;color:{MUT};margin-top:3px;font-weight:400">{sub[0]}</div>' if sub else ""
        parts.append(f'<div style="border:2px solid {col};background:{col}1a;border-radius:8px;padding:7px 10px;font-size:0.76rem;font-weight:700;color:#e2e8f0;text-align:center;min-width:92px">{t}{s}</div>')
    joined = arrow.join(parts)
    fd = "row" if direction == "row" else "column"
    return f'<div style="display:flex;flex-direction:{fd};flex-wrap:wrap;gap:6px;align-items:{"stretch" if fd == "row" else "center"}">{joined}</div>'


def heat(rows, cols):
    sym = {2: ("●", G, "좋음"), 1: ("◐", A, "보통"), 0: ("○", R, "나쁨")}
    t = f'<table style="width:100%;border-collapse:collapse;font-size:0.78rem"><tr style="background:#0a1220"><th style="padding:6px 10px;text-align:left;color:{AL}">기준</th>'
    t += "".join(f'<th style="padding:6px 10px;text-align:center;color:{AL}">{c}</th>' for c in cols) + "</tr>"
    for name, scores in rows:
        t += f'<tr><td style="padding:6px 10px;border-bottom:1px solid #1e293b;color:{MUT}">{name}</td>'
        for sc in scores:
            s, c, l = sym[sc]
            t += f'<td style="padding:6px 10px;border-bottom:1px solid #1e293b;text-align:center;background:{c}22;color:{c};font-weight:800">{s} {l}</td>'
        t += "</tr>"
    return t + "</table>"


def speed_chart():
    r = json.loads((ROOT / "docs/v5/egvla2_fm/meta/speed_calibration.json").read_text())["records"]
    sc = {0: 0.4, 1: 0.7, 2: 1.0}
    W, H = 520, 190
    svg = f'<svg viewBox="0 0 {W} {H}" style="width:100%;max-width:640px" xmlns="http://www.w3.org/2000/svg"><line x1="40" y1="{H - 28}" x2="{W - 10}" y2="{H - 28}" stroke="#334155"/>'
    X = lambda sc_: 70 + (sc_ - 0.3) / 0.8 * (W - 100)
    Y = lambda v: (H - 28) - v / 30 * (H - 50)
    for v in (10, 20, 30):
        svg += f'<line x1="40" y1="{Y(v):.1f}" x2="{W - 10}" y2="{Y(v):.1f}" stroke="#1e293b"/><text x="34" y="{Y(v) + 3:.1f}" font-size="8" fill="#94a3b8" text-anchor="end">{v}</text>'
    xs = np.array([sc[a["level"]] for a in r]); ys = np.array([a["speed_cmps"] for a in r]); b1, a0 = np.polyfit(xs, ys, 1)
    svg += f'<line x1="{X(0.3):.1f}" y1="{Y(a0 + b1 * 0.3):.1f}" x2="{X(1.1):.1f}" y2="{Y(a0 + b1 * 1.1):.1f}" stroke="{P}" stroke-dasharray="4 3" stroke-width="1.5"/>'
    for L, col in ((0, B), (1, G), (2, A)):
        v = np.array([a["speed_cmps"] for a in r if a["level"] == L]); x0 = X(sc[L])
        for i, y in enumerate(sorted(v)):
            svg += f'<circle cx="{x0 - 14 + (i % 5) * 7:.1f}" cy="{Y(y):.1f}" r="2.6" fill="{col}" opacity="0.75"/>' if False else f'<circle cx="{x0 + ((i * 37) % 21 - 10):.1f}" cy="{Y(y):.1f}" r="2.8" fill="{col}" opacity="0.8"/>'
        svg += f'<rect x="{x0 - 19:.1f}" y="{Y(v.mean() + v.std(ddof=1)):.1f}" width="38" height="{Y(v.mean() - v.std(ddof=1)) - Y(v.mean() + v.std(ddof=1)):.1f}" fill="{col}" opacity="0.15"/>'
        svg += f'<line x1="{x0 - 19:.1f}" y1="{Y(v.mean()):.1f}" x2="{x0 + 19:.1f}" y2="{Y(v.mean()):.1f}" stroke="{col}" stroke-width="2.2"/>'
        svg += f'<text x="{x0:.1f}" y="{H - 14}" font-size="9" fill="#cbd5e1" text-anchor="middle">레벨{L} (scale {sc[L]:.2f})</text>'
        svg += f'<text x="{x0:.1f}" y="{Y(v.mean() + v.std(ddof=1)) - 5:.1f}" font-size="8.5" fill="{col}" text-anchor="middle">{v.mean():.2f} ± {v.std(ddof=1):.2f}</text>'
    svg += f'<text x="{W - 12}" y="14" font-size="8.5" fill="{P}" text-anchor="end">선형근사 v = {a0:.2f} + {b1:.2f}×scale (cm/s)</text><text x="4" y="12" font-size="8" fill="#94a3b8">cm/s</text></svg>'
    return svg + f'<div style="font-size:0.74rem;color:{MUT}">점 = 개별 측정(각 레벨 10회, 200cm), 굵은 선 = 평균, 음영 = ±1 표준편차, 점선 = 30건 전체 선형 적합. 원본: <code>docs/v5/egvla2_fm/meta/speed_calibration.json</code></div>'


def load(name):
    p = FM / name
    return json.loads(p.read_text()) if p.exists() else None


def build(hist):
    V = {}
    # ---------- t1 ----------
    V["t1"] = wrap("① 파일럿 v1 → v2 (같은 판정 기준, <code>actions_raw</code>)",
        legend([("파일럿 v1 10ep (게이지 전)", R), ("파일럿 v2 15ep (게이지 후)", G)])
        + vgroups(["lx 중간값", "ly 중간값", "크기 중간값", "코너", "az 스파이크"],
                  [("v1", R, [14, 8, 1, 25, 33]), ("v2", G, [78, 34, 75, 1, 18])], 100, fmt="{:g}%", bw=20),
        "합격 기준: 크기 중간값 ≥50% · 코너 ≤20% · az 스파이크 ≤25% — v1 미달, v2 합격. 출처 76-6") + wrap(
        "② 문제 → 원인 → 조치",
        flow([("약한 스틱(<0.5) 입력", MUT), ("<code>if key:</code> 게이트 (임계 0.5)", R, "발행·기록 안 됨"), ("v1 비영 2,048프레임 중 3개", R, "max|v|&lt;0.5"),
              ("게이트 → 크기&gt;0 발행", G, "수정"), ("+ 게이지·방사형 데드존", G, "중간값 75%")]))
    V["t1"] += wrap("③ 스틱 평면에서 본 변화 — v1 수집 40건 vs 파일럿 v2 15건 (실제 데이터)", img("egvla2_fm/figures/ch76_stick_plane.jpg",
        "가로=좌우, 세로=전후. 노란 사각형은 v1 게이트(<code>if key:</code>)가 지우던 영역(세 축 모두 |값|&lt;0.5). v1은 점이 모서리·끝에 몰리고 사각형 안은 비어 있다. v2는 사각형 안과 원 안쪽 전 구간에 입력이 있다."),
        "비영 프레임만 표시, 겹침을 줄이려 점마다 ±0.012 지터")
    # ---------- t2 ----------
    V["t2"] = wrap("① 수집 경로 비교 — 이전(v1) vs 지금(v3)",
        '<div style="font-size:0.74rem;color:%s;margin-bottom:4px">이전 (v1, 40건)</div>' % R
        + flow([("스틱 3축", MUT), ("축별 데드존 0.15", R, "방향 왜곡"), ("<code>if key:</code> 게이트", R, "임계 0.5"), ("각도 + 고정 throttle", R, "속력 무반영"), ("<code>actions</code>=구동값", R, "대각선 최대 1.41")])
        + '<div style="font-size:0.74rem;color:%s;margin:12px 0 4px">지금 (v3, 28건)</div>' % G
        + flow([("스틱 3축", MUT), ("방사형 데드존 + 재조정", G), ("센터링 0.03 스냅", G, "직진 드리프트 제거"), ("<code>has_input</code>", G, "크기&gt;0"), ("원형게이트", G, "크기≤1"), ("구동 C", G, "throttle×3단계"), ("raw/scaled/exec", G, "+타임스탬프")]),
        "초록 = 이번에 바뀐 단계, 빨강 = 이전의 문제 단계") + wrap(
        "② 저장 필드 변화",
        flow([("v1: <code>actions</code> 1개", R, "의미 모호"), ("v2: raw + scaled", A, "버전 분리"), ("v3: raw + scaled + exec + frame_timestamps", G, "derive_version, 속도 보정 attrs")]))
    V["t2"] += wrap("③ 실제 에피소드 한 개의 기록 — 프레임, 스틱 HUD, 속력 단계, 시계열 (파일럿 v2)", img("egvla2_fm/figures/ch76_pilot_episode.jpg",
        "위: 8프레임(초록 + = Florence-2가 찾은 바구니 중심). 가운데: 그 프레임의 스틱(화살표=방향, 길이=크기)과 속력 단계. 아래: raw lx·ly와 exec 속력 단계(노랑) 시계열 — 음영은 v1 게이트가 지우던 구간. 한 에피소드 안에 느림·보통·빠름 단계가 모두 기록된다."),
        "에피소드는 속력 단계 수·스틱 변화량이 가장 큰 것으로 자동 선택 (<code>scripts/gen_ch76_figures.py</code>)")
    # ---------- t3 ----------
    bins = hist["bins"]; n1, n2 = hist["n1"], hist["n2"]
    cats = [f"{bins[i]:.1f}" for i in range(len(hist["v1"]))][:11]
    s1 = [100 * x / n1 for x in hist["v1"][:11]]; s2 = [100 * x / n2 for x in hist["v2"][:11]]
    V["t3"] = wrap("① 스틱 입력 크기 분포 — 비영 프레임의 max(|lx|,|ly|,|az|), 구간별 비율(%)",
        legend([(f"v1 40건 (n={n1})", R), (f"파일럿 v2 15건 (n={n2})", G)]) + vgroups(cats, [("v1", R, s1), ("v2", G, s2)], 100, h=170, fmt="{:.0f}", bw=11)
        + f'<div style="font-size:0.74rem;color:{MUT}">가로축: 구간 시작값(0.0~1.0, 각 0.1폭). v1은 0.9 이상이 99%, 0.5 미만은 3프레임 — 게이트(0.5)가 약한 입력을 지운 증거. v2는 0~1.0 전 구간에 퍼짐.</div>',
        "v1은 <code>actions</code>, v2는 <code>actions_raw</code> 기준. v2의 1.0 구간은 정확히 1.0 포함") + wrap(
        "② 문제 심각도 × 수정 난이도",
        hbars([("1. key 게이트가 약한 입력 제거", 100, R, "치명 · 수정 쉬움"), ("2. 구동은 각도만 사용(속력 무반영)", 85, R, "큼 · 결정 필요"), ("3. 회전+이동 동시 입력 불일치", 55, A, "중간 · 규칙 정의"),
               ("4. raw 크기 최대 1.46", 30, A, "낮음 · 문서화"), ("5. az 200ms 홀드", 25, A, "낮음 · 80ms 복귀")], 100, w=250),
        "막대 길이 = 학습 데이터에 미치는 영향(평가자 판단), 문구 = 심각도·수정 난이도")
    V["t3"] += wrap("③ 약한 스틱 입력이 실제로 기록된 프레임 4개 (v2)", img("egvla2_fm/figures/ch76_weak_stick_frames.jpg",
        "각 프레임에서 스틱 크기가 0.37~0.45로 v1 게이트였다면 발행·기록되지 않았을 입력이다. 스틱을 조금만 밀어 천천히 움직이는 장면에서 실제로 나타난다."),
        "v2 raw 크기 0.15~0.45 구간 프레임 중 에피소드별 1개 선택")
    # ---------- t4 ----------
    V["t4"] = wrap("① 선택지 비교 매트릭스 (평가자 판단 기준, 데이터 근거는 76-3·76-6)",
        heat([("라벨 = 실제 구동 일치", [0, 2, 2]), ("감속(속력) 정보 학습", [2, 0, 1]), ("odom 없이 효과 검증 가능", [0, 2, 2]), ("구현·보정 비용 낮음", [1, 2, 1]), ("라벨 노이즈 위험 낮음", [0, 2, 1])],
             ["A 크기→throttle 연속", "B 각도 + 고정 속력", "C 속력 3단계 + 각도"]),
        "A는 구동이 크기를 쓰지 않는 한 라벨이 실제와 어긋남. C는 보정(스톱워치) 이후 일치.") + wrap(
        "② 결정 흐름 — 채택은 C(보정 완료)",
        flow([("① 게이트 수정", G, "완료"), ("② B로 파일럿 (라벨=실제)", MUT, "건너뜀"), ("③ 스톱워치 보정", G, "완료 14.45/20.64/26.36 cm/s"), ("④ C로 확장 (v3)", G, "채택"), ("A", R, "odom 전까지 보류")]))
    V["t4"] += wrap("③ 같은 프레임에서 A/B/C 라벨이 어떻게 달라지는가 (실제 데이터)", img("egvla2_fm/figures/ch76_conversion_example.jpg",
        "raw 한 개에서 A(원형게이트), B(각도+GO), C(각도+속력 단계)가 결정적으로 파생되고, 실제 구동값(exec)은 구동 방식 C가 만든 값이다. 크기가 0.66이면 B는 GO, C는 ‘보통(0.7)’이 된다."),
        "A는 크기를 보존하지만 실제 속도는 단계로만 반영됨 → A 라벨의 크기는 구동과 일치하지 않는다")
    # ---------- t5 ----------
    V["t5"] = wrap("① 로드맵 — 완료 / 대기",
        flow([("게이트 수정", G, "✅"), ("구동 C + 보정", G, "✅"), ("v3 파일럿 15ep 합격", G, "✅"), ("지표 정의 고정 (raw)", G, "✅"), ("FM A/B/C 5-fold", G, "✅"), ("B2·조기 종료", A, "진행"), ("waypoint 적분 시험", A, "진행"),
              ("에피소드·az 데이터 확대", MUT, "대기"), ("서브골 FiLM + 연속 PM", MUT, "대기"), ("closed-loop", MUT, "대기")]),
        "초록 완료 · 노랑 진행 · 회색 대기") + wrap(
        "② 선행사례 → 우리 설계 대응",
        flow([("Unity radial deadzone (크기 임계 + 재조정)", B), ("→ 방사형 데드존 + (mag−0.15)/0.85", G)])
        + '<div style="height:8px"></div>' + flow([("RL 단위원 정규화 (2407.01991)", B, "원문 미확인"), ("→ 원형게이트 (크기≤1)", G)])
        + '<div style="height:8px"></div>' + flow([("궤적/waypoint 타깃 (UNI·TONAV·Hydra)", B, "초록 기준"), ("→ 명령 적분 waypoint 시험 (plan 8.4)", A, "진행")]))
    # ---------- t6 ----------
    V["t6"] = wrap("① 보고값 vs 학습 서버 재계산 — 파일럿 v1 → v2 (raw 기준, 동일 정의)",
        legend([("v1", R), ("v2", G)]) + vgroups(["lx 중간값", "ly 중간값", "코너", "az 스파이크"], [("v1", R, [14, 8, 25, 33]), ("v2", G, [78, 34, 1, 18])], 100, fmt="{:g}%", bw=22),
        "raw 기준 재계산은 로봇 세션 보고와 정확히 일치(lx 78%·ly 34%·코너 1%·스파이크 5/28). 초기 불일치는 scaled 기준으로 계산해서 생긴 정의 차이였음") + wrap(
        "② 시간 순서",
        flow([("v2 코드 푸시 62e722bc", MUT), ("리뷰 §9 (문제 5)", R), ("제안 §10", A), ("§11 반영: 게이트·구동 C·보정·게이지", G), ("v3 데이터 28건 수신", G), ("공식 판정 (파일럿 v2 합격)", G), ("FM 학습 시험", A)]))
    # ---------- t7 ----------
    s_e = load("summary_exec.json"); s_b2 = load("summary_exec_b2.json"); s_wp = load("summary_v6_waypoint.json")
    if s_e:
        a = s_e["aggregate"]
        names = [("평균 기준선", MUT), ("MLP 회귀", P), ("A MLP-FM", G), ("B FiLM-FM", R), ("C π0형", B)]
        def mse(ax): return [a["A_mlp_fm"]["base_mean_mse_axis"][ax], a["MLP_regression"]["mse_axis"][ax], a["A_mlp_fm"]["mse_mean_axis"][ax], a["B_film_fm"]["mse_mean_axis"][ax], a["C_pi0_adaln"]["mse_mean_axis"][ax]]
        series = [(names[i][0], names[i][1], [mse(0)[i], mse(1)[i], mse(2)[i] * 5]) for i in range(5)]
        extra = ""
        if s_b2 and "B2_film_res_fm" in s_b2["aggregate"]:
            b2 = s_b2["aggregate"]["B2_film_res_fm"]
            series.append(("B2 FiLM+잔차", "#fb923c", [b2["mse_mean_axis"][0], b2["mse_mean_axis"][1], b2["mse_mean_axis"][2] * 5]))
        leg = legend([(n, c) for n, c, *_ in series])
        V["t7"] = wrap("① 축별 MSE — 낮을수록 좋음 (exec, 5-fold, 샘플 평균)", leg + vgroups(["lx", "ly", "az (×5)"], series, 0.16, fmt="{:.3f}", bw=15),
            "az는 값 범위가 작아 5배로 표시. 기준선 대비 낮으면 학습된 것")
        r2 = [("A MLP-FM", G, a["A_mlp_fm"]["r2_mean_axis"]), ("B FiLM-FM", R, a["B_film_fm"]["r2_mean_axis"]), ("C π0형", B, a["C_pi0_adaln"]["r2_mean_axis"])]
        if s_b2 and "B2_film_res_fm" in s_b2["aggregate"]: r2.append(("B2 FiLM+잔차", "#fb923c", s_b2["aggregate"]["B2_film_res_fm"]["r2_mean_axis"]))
        V["t7"] += wrap("② 축별 R² (0 = 평균 기준선, 음수 = 기준선보다 나쁨)", legend([(n, c) for n, c, _ in r2])
            + vgroups(["lx", "ly", "az (음수는 0 표시)"], [(n, c, [max(v[0], 0), max(v[1], 0), max(v[2], 0)]) for n, c, v in r2], 1.0, fmt="{:.2f}", bw=15),
            "음수 R²는 막대가 0으로 표시됨 — 실제 값은 76-7 표 참조(az: A −0.34, B −5.2, C −0.11)")
        tr = [(k, a[k]["train_flow_final"], a[k]["test_flow_final"]) for k in ("A_mlp_fm", "B_film_fm", "C_pi0_adaln")]
        V["t7"] += wrap("③ 과적합 — 최종 epoch flow loss (train vs test)", legend([("train", G), ("test(hold-out)", R)])
            + vgroups(["A MLP-FM", "B FiLM-FM", "C π0형"], [("train", G, [t[1] for t in tr]), ("test", R, [t[2] for t in tr])], 0.5, fmt="{:.2f}", bw=22),
            "C는 train 0.08 대 test 0.24로 격차가 가장 큼")
        ce = s_b2["aggregate"]["A_mlp_fm"].get("curve_epochs") if s_b2 else None
        if ce and s_b2:
            lines = ""
            cols = {"A_mlp_fm": G, "B_film_fm": R, "C_pi0_adaln": B, "B2_film_res_fm": "#fb923c"}
            W, Hh = 520, 150; allv = []
            for k in cols:
                if k in s_b2["aggregate"]: allv += s_b2["aggregate"][k]["curve_test"]
            lo, hi = min(allv), max(allv)
            for k, c in cols.items():
                if k not in s_b2["aggregate"]: continue
                ys = s_b2["aggregate"][k]["curve_test"]; xs = s_b2["aggregate"][k]["curve_epochs"]
                pts = " ".join(f"{30 + x / max(xs) * (W - 40):.1f},{Hh - 10 - (y - lo) / (hi - lo + 1e-9) * (Hh - 24):.1f}" for x, y in zip(xs, ys))
                lines += f'<polyline fill="none" stroke="{c}" stroke-width="2" points="{pts}"/>'
            V["t7"] += wrap("④ hold-out flow loss 학습 곡선 (epoch 1·10·25…150)", legend([(k, c) for k, c in cols.items() if k in s_b2["aggregate"]])
                + f'<svg viewBox="0 0 {W} {Hh + 14}" style="width:100%;max-width:700px"><line x1="28" y1="{Hh - 8}" x2="{W}" y2="{Hh - 8}" stroke="#334155"/>{lines}'
                + f'<text x="28" y="{Hh + 8}" font-size="8" fill="#94a3b8">epoch 1</text><text x="{W - 38}" y="{Hh + 8}" font-size="8" fill="#94a3b8">150</text>'
                + f'<text x="2" y="12" font-size="8" fill="#94a3b8">{hi:.2f}</text><text x="2" y="{Hh - 10}" font-size="8" fill="#94a3b8">{lo:.2f}</text></svg>',
                "최저점 이후 올라가면 과적합. 아래로 갈수록 좋음")
    V["t7"] = V.get("t7", "") + wrap("⑤ 생성된 액션 vs 정답 — 학습에 쓰지 않은 에피소드 (실제 데이터)", img("egvla2_fm/figures/ch76_fm_prediction.jpg",
        "노랑 = 정답(exec), 색 영역 = 샘플 16개 범위, 선 = 샘플 평균. 이 에피소드는 좌우(ly)가 거의 0인 직진인데 모델이 ly≈0.3을 섞어 내는 구간이 있다 — 학습 데이터가 우곡선 위주라 직진 에피소드 일반화가 약하다는 것을 보여 주는 정직한 예시. A는 전후(lx) 추세를 따라가지만 단계 값(0.4/0.7/1.0)은 부드럽게 평균낸다."),
        "5-fold 중 fold 3의 hold-out 1ep, 60 epoch, seed 0 (별도 실행)")
    if s_wp:
        r = s_wp["results"]
        rows = []
        for k, c in (("chunk|A_mlp_fm", MUT), ("waypoint|A_mlp_fm", G), ("chunk|C_pi0_adaln", MUT), ("waypoint|C_pi0_adaln", B)):
            if k in r: rows.append((k.replace("|", " · "), c, r[k]["r2"]))
        if rows:
            V["t7"] = V.get("t7", "") + wrap("⑥ 명령 적분 → waypoint 타깃 시험 (V6 225ep, 마지막 waypoint의 R²)", legend([(a_, c) for a_, c, _ in rows])
                + vgroups(["lx", "ly", "az"], [(a_, c, [max(v[0], 0), max(v[1], 0), max(v[2], 0)]) for a_, c, v in rows], 1.0, fmt="{:.2f}", bw=18),
                "같은 분할·입력에서 chunk(명령 8개)와 waypoint(누적합) 타깃을 같은 척도(마지막 waypoint)로 비교. V6 명령은 ±1.15 이산값, 속도·시간 미보정(프레임 단위)")
    # ---------- t8 ----------
    V["t8"] = wrap("① 입력 → 저장 구조도", flow([("축0 (왼쪽 X)", MUT, "→ ly"), ("축1 (왼쪽 Y)", MUT, "→ lx"), ("축2 (오른쪽 X)", MUT, "→ az"), ("방사형 데드존+재조정", G, "0.15"), ("센터링 스냅", G, "0.03"), ("<code>actions_raw</code>", A, "기준값")])
        + '<div style="height:8px"></div>' + flow([("A: 원형게이트", B, "scaled"), ("B: 각도 + GO/STOP", B), ("C: 각도 + 속력 3단계", B, "exec")])
        + '<div style="height:8px"></div>' + flow([("구동 C", G, "driver.move(각도, throttle×단계)"), ("<code>actions_exec</code>", A, "실제로 한 일"), ("+ frame_timestamps · derive_version", MUT)]),
        "축 3개만 데이터가 되고, 버튼·D-pad는 메타데이터. az는 변환 없이 저장, 구동은 |az|&gt;0.1이면 회전만") + wrap(
        "② 속력 3단계 실측 보정 (2m 코스, 레벨당 10회 = 30건 원본)",
        speed_chart(),
        "원본 30건 재계산이 보고값(14.45/20.64/26.36)과 일치. 선형 적합 6.59 + 19.85 × scale — 레벨1 예측 20.5 vs 실측 20.64. 레벨0 한 건에 ‘완전 직선은 아니었음’ 메모. 모터 기저속도(데드밴드) 추정") + wrap(
        "③ 추론 속력 보간 규칙 (결정: 연속값, 임의 보간)", flow([("모델 출력 (lx, ly)", MUT), ("m = |(lx, ly)|", P), ("m &lt; 0.2 → STOP", R), ("s = clamp(m, 0.4, 1.0)", G, "0.4 미만은 최저 속도로"), ("throttle = 50 × s", G, "20 ~ 50 연속"), ("예상 v ≈ 6.59 + 19.85·s cm/s", A, "측정점 사이는 보간 가정")]),
        "plan_20261014_inference_continuous_speed.md. 한 스텝 동안 균일 속도, 중간 속력은 별도 검증하지 않음") + wrap(
        "④ 버튼·입력 구성", hbars([("주행 아날로그 축", 3, G, "3개 → 학습 입력"), ("기능 버튼", 8, MUT, "8개"), ("보조 버튼 (복귀)", 1, MUT, "1개"), ("D-pad", 1, MUT, "1개 (시나리오 선택)")], 10, w=190),
        "학습 데이터에 들어가는 입력은 축 3개뿐")
    return V



def speed_area_stats():
    from scipy.stats import spearmanr
    S = json.load(open(ROOT / "docs/v5/egvla2_fm/meta/manual_dataset_sets.json"))["sets"]["pilot_v2_gauge_20261013"]
    C = np.load(ROOT / "docs/v5/egvla2_fm/v3_florence2_cache.npz")
    A, L, T, EP = [], [], [], []
    for n in S:
        n = n if n.endswith(".h5") else n + ".h5"
        with h5py.File(ROOT / "ROS_action/mobile_vla_dataset_egvla2" / n) as f:
            e = np.array(f["actions_exec"])
        bb = C[n + "|bbox"]; m = np.hypot(e[:, 0], e[:, 1]); mv = m > 1e-3
        A += bb[mv, 2].tolist(); L += m[mv].tolist(); T += (np.arange(len(e)) / len(e))[mv].tolist(); EP += [n] * int(mv.sum())
    A, L, T = np.array(A), np.array(L), np.array(T)
    qs = np.quantile(A, [0, .25, .5, .75, 1]); rows = []
    for i in range(4):
        mk = (A >= qs[i]) & (A <= qs[i + 1])
        rows.append((f"{qs[i]:.3f}~{qs[i + 1]:.3f}", [float(np.mean(np.isclose(L[mk], v, atol=.05))) for v in (.4, .7, 1.0)], float(L[mk].mean()), int(mk.sum())))
    ep_means = [float(np.mean([l for e_, l in zip(EP, L) if e_ == n])) for n in sorted(set(EP))]
    return dict(rows=rows, rho_area=float(spearmanr(L, A)[0]), rho_t=float(spearmanr(L, T)[0]), n=len(A), ep_min=min(ep_means), ep_max=max(ep_means))


def stacked(rows):
    out = legend([("느림 0.4", B), ("보통 0.7", G), ("빠름 1.0", A)])
    for name, fr, mean, n in rows:
        segs = "".join(f'<div style="width:{f * 100:.1f}%;background:{c};height:20px;font-size:0.68rem;color:#0a0f1a;text-align:center;line-height:20px;font-weight:800">{f * 100:.0f}%</div>' if f > 0.04 else f'<div style="width:{f * 100:.1f}%;background:{c};height:20px"></div>' for f, c in zip(fr, (B, G, A)))
        out += (f'<div style="display:flex;align-items:center;gap:8px;margin:5px 0;font-size:0.76rem"><div style="width:150px;color:{MUT}">바구니 면적 {name}<br><span style="font-size:0.66rem">n={n}, 평균 {mean:.2f}</span></div>'
                f'<div style="flex:1;display:flex;border-radius:4px;overflow:hidden;background:#0a1220">{segs}</div></div>')
    return out


def t9_card():
    st = speed_area_stats()
    pill_ = lambda l, c: f'<span style="display:inline-block;background:{c}22;color:{c};border:1px solid {c};border-radius:6px;padding:0 8px;font-size:0.76rem;font-weight:800;margin-right:4px">{l}</span>'
    H = lambda t: f'<div style="margin:16px 0 6px;padding-bottom:4px;border-bottom:2px solid {A}55;color:{AL};font-size:0.92rem;font-weight:800">{t}</div>'
    def tbl(head, rows):
        h = "".join(f'<th style="text-align:left;padding:7px 10px;border-bottom:2px solid {A};color:{AL}">{x}</th>' for x in head)
        o = f'<table style="width:100%;border-collapse:collapse;margin:10px 0;font-size:0.8rem"><tr style="background:#0a1220">{h}</tr>'
        for r in rows: o += "<tr>" + "".join(f'<td style="padding:6px 10px;border-bottom:1px solid #1e293b">{x}</td>' for x in r) + "</tr>"
        return o + "</table>"
    body = (pill_("결론", G) + "속력은 <b>모델이 마음대로 학습하게 두지 말고</b>, ‘연속 출력 경로는 만들되 속력의 출처는 설정(<code>speed_source=model|rule</code>)으로 고르는’ 방식으로 간다. 지금 데이터에서는 속력이 거리 단서와 거의 상관이 없어 <b>첫 적용은 규칙 속력(또는 모델 속력을 기록만 하는 섀도 모드)</b>, 수집 규칙을 정해 데이터를 모은 뒤 상관이 생기면 모델 속력으로 전환한다."
        + H("① 파일럿 데이터가 말해 주는 것 — 속력 단계 vs 바구니 거리(F2 bbox 면적)")
        + wrap("바구니 면적 4분위별 속력 단계 비율 (파일럿 v2, 이동 프레임 %d개)" % st["n"], stacked(st["rows"]),
               f"순위상관: 속력~면적 ρ={st['rho_area']:+.2f}, 속력~진행시간 ρ={st['rho_t']:+.2f} (둘 다 거의 0). 에피소드별 평균 속력은 {st['ep_min']:.2f}~{st['ep_max']:.2f}로 운전자 습관에 따라 흩어짐. 면적은 거리의 대리값", kind="chart")
        + "느림→빠름→느림의 U자 모양(출발·도착 부근 느림)은 보이지만 ‘가까울수록 감속’하는 일관된 규칙이 아니다. 지금의 속력 라벨은 시각 단서보다 <b>운전 습관</b>의 영향이 크다."
        + H("② 다른 최신 연구는 속력을 어떻게 다루는가 (초록·요약 기준, 본문 미확인)")
        + tbl(["연구", "방식", "속력 처리", "우리와의 관계"], [
            ["ABot-N0 (2602.11598)", "플로우 매칭 헤드가 로컬 waypoint 5개(x, y, yaw) 생성", "속도 명령이 아니라 궤적", "waypoint 타깃 시험과 같은 방향"],
            ["ViNT/NoMaD → MetricNet (2509.13965)", "정규화 waypoint × (v_max/f)", "스케일이 틀리면 명령 속도가 한계를 크게 초과", "우리는 보정값(cm/s)이 있어 이 문제를 피할 수 있음"],
            ["VISTA (2606.17294)", "정규화 액션 이력을 입력으로 추가", "이력으로 물리 변위와 정렬", "우리 모델은 이력 입력이 없음(남은 후보)"],
            ["NaVILA (2412.04453)", "상위 VLA는 ‘75cm 전진’ 같은 중간 지시", "하위 보행 정책이 속도 담당", "속력을 상위 학습에서 분리하는 선례"],
            ["VEGA (2606.18426)", "영상에서 복원한 기하 궤적을 감독", "궤적 기반", "수집 방식 대안"],
          ])
        + "공통 경향: 대부분 조이스틱 <b>속력 라벨을 직접 회귀하지 않고</b> 궤적·waypoint로 만들거나 속도를 하위 제어기로 분리한다."
        + H("③ 우리가 취할 방침 — 단계별")
        + wrap("단계 흐름 (속력 출처를 데이터가 허락할 때 모델로 전환)", flow([
            ("① 지금: 연속 출력 경로 구현", G, "action_cont + throttle_scale"), ("속력은 규칙 또는 섀도", A, "speed_source=rule"), ("② 수집 가이드 정하기", B, "거리별 속력 가이드"),
            ("③ 일관 데이터 60~100ep", B), ("④ 판정: |ρ(속력, 면적)| ≥ 0.3?", P, "속력이 거리와 연결됐나"), ("예 → speed_source=model", G), ("아니오 → 규칙 유지, 위치(waypoint) 타깃으로", R)]), "판정 기준 0.3은 임의로 정한 제안값(다음 단계로 넘어갈지 보는 통과 기준). 데이터가 늘면 같은 분석으로 재측정", kind="flow")
        + tbl(["영역", "하는 일", "이유"], [
            ["수집", "속력 규칙 정하기: 출발·정지 램프, 먼 구간 보통~빠름, 목표 근처 느림. 게이지에 목표 속력 가이드 표시", "라벨 노이즈(운전 습관)를 줄여 속력이 거리 단서와 연결되게 함"],
            ["학습", "방향(각도)·GO/STOP을 주 타깃, 속력은 별도 보조 손실(가중치 낮게). 보정값으로 waypoint(미터) 타깃도 시험", "속력 노이즈가 방향 학습을 해치지 않게 분리, 연구 경향(궤적 타깃)과 정렬"],
            ["추론", "연속 경로(<code>action_cont</code>, <code>throttle_scale</code>)는 구현하되 속력 출처를 설정으로 선택. 첫 적용은 s≤0.7, 규칙 또는 섀도(예측만 기록)", "중간 속력 미검증, STOP 거리·지연 위험, 현재 데이터로는 모델 속력을 신뢰하기 이르다"],
          ])
        + H("④ 결정이 필요한 것")
        + "<ul style='margin:6px 0 6px 18px'><li>수집 속력 규칙의 구체안(거리 구간별 목표 단계) — 로봇 세션과 합의</li><li>첫 적용 속력 출처: 규칙 vs 섀도 (<code>plan_20261014_inference_continuous_speed.md</code> §2에 <code>speed_source</code> 추가 제안)</li><li>판정 기준(제안 0.3)과 필요한 에피소드 수(제안 60~100) — 수집이 끝난 뒤 정해도 된다</li></ul>"
        + pill_("한계", R) + "분석은 파일럿 15ep(이동 프레임 %d개)이고 면적은 거리의 대리값이다. 문헌은 초록 수준이라 인용 전 원문 확인이 필요하다." % st["n"])
    return f'<div class="finding-card" style="border-color:{A}"><div class="finding-title" style="color:{A}">76-9. 우리는 어떻게 할까 — 속력 정책 방향 (수집·학습·추론)</div><div style="font-size:0.85rem;line-height:1.8;color:#94a3b8">{body}</div></div>'


def ep250_block(a1, pill_, tbl):
    f = FM / "summary_s1_ep250.json"
    if not f.exists():
        return ""
    a3 = json.load(open(f))["aggregate"]; names = ["A_mlp_fm", "B2_film_res_fm", "C_pi0_adaln"]
    rows = []
    for n in names:
        rows.append([n, *[f"{a1[n]['r2_mean_axis'][i]:.2f} → {a3[n]['r2_mean_axis'][i]:.2f}" for i in (0, 1)], f"{a1[n]['angle_deg']:.1f}° → {a3[n]['angle_deg']:.1f}°", f"{a1[n]['go_f1']:.3f} → {a3[n]['go_f1']:.3f}", f"{a1[n]['best_epoch']:.0f} → {a3[n]['best_epoch']:.0f}"])
    return (f"<br>{pill_('epoch 확대(250, 3 seed)', A)}100 epoch(5 seed) → 250 epoch(3 seed) 비교(seed 수가 달라 대략 비교):"
            + tbl(["모델", "lx R²", "ly R²", "방향 오차", "GO/STOP F1", "최적 epoch"], rows))


def t10_card():
    V = json.load(open(FM / "validation_batch.json"))
    S1 = json.load(open(FM / "summary_s1.json")); S2 = json.load(open(FM / "summary_s1_azmask.json"))
    ag = V["aggregate"]
    pill_ = lambda l, c: f'<span style="display:inline-block;background:{c}22;color:{c};border:1px solid {c};border-radius:6px;padding:0 8px;font-size:0.76rem;font-weight:800;margin-right:4px">{l}</span>'
    H = lambda t: f'<div style="margin:16px 0 6px;padding-bottom:4px;border-bottom:2px solid {A}55;color:{AL};font-size:0.92rem;font-weight:800">{t}</div>'
    def tbl(head, rows):
        h = "".join(f'<th style="text-align:left;padding:7px 10px;border-bottom:2px solid {A};color:{AL}">{x}</th>' for x in head)
        o = f'<table style="width:100%;border-collapse:collapse;margin:10px 0;font-size:0.8rem"><tr style="background:#0a1220">{h}</tr>'
        for r in rows: o += "<tr>" + "".join(f'<td style="padding:6px 10px;border-bottom:1px solid #1e293b">{x}</td>' for x in r) + "</tr>"
        return o + "</table>"
    iso, nz = ag["az_spike"]
    lv = ag["level_dist"]
    cells = {}
    for k, v in ag["paths"].items(): cells[k] = v
    pos = ["center", "weak_left", "strong_left", "weak_right"]; pth = ["left_curve", "straight", "right_curve"]
    grid = f'<table style="border-collapse:collapse;font-size:0.78rem;margin:6px 0"><tr><th style="padding:5px 10px;color:{AL}">위치 \\ 경로</th>' + "".join(f'<th style="padding:5px 10px;color:{AL}">{x}</th>' for x in pth) + "</tr>"
    for ps in pos + ["strong_right"]:
        grid += f'<tr><td style="padding:5px 10px;color:{MUT}">{ps}</td>'
        for ph in pth:
            n = cells.get(str((ps, ph)), 0); col = G if n >= 20 else (A if n else R)
            grid += f'<td style="padding:5px 10px;text-align:center;background:{col}22;color:{col};font-weight:800;border:1px solid #1e293b">{n}/20</td>'
        grid += "</tr>"
    grid += "</table>"
    names = ["STOP", "FORWARD", "LEFT", "RIGHT", "FWD+L", "FWD+R", "ROT_L", "ROT_R"]
    cls = vgroups(names, [("신규 v3", B, [ag_ * 100 for ag_ in [V["class_dist_new"][n] for n in names]]), ("V6", MUT, [V["class_dist_v6"][n] * 100 for n in names])], 50, fmt="{:.1f}")
    prof = V["aggregate"]["profile"]
    pr = vgroups(["0~10%", "10~30%", "30~70%", "70~90%", "90~100%"], [("신규 평균 크기", B, [q["mean_mag"] for q in prof]), ("가이드 목표(§12)", A, [0.42, 0.67, 0.72, 0.62, 0.44])], 1.0, fmt="{:.2f}")
    a1, a2 = S1["aggregate"], S2["aggregate"]
    arch = ["A_mlp_fm", "B2_film_res_fm", "C_pi0_adaln"]
    r2 = vgroups(["lx", "ly", "az"], [("MLP 회귀(MSE 기준 환산 아님)", MUT, [0, 0, 0])] and [(n.split("_")[0] + " FM", c, a1[n]["r2_mean_axis"]) for n, c in zip(arch, (B, G, A))], 1.0, fmt="{:.2f}")
    mse_rows = [["평균 기준선", *[f"{x:.3f}" for x in a1["A_mlp_fm"]["base_mean_mse_axis"]], "-", "-"], ["MLP 회귀", *[f"{x:.3f}" for x in a1["MLP_regression"]["mse_axis"]], "-", "-"]]
    for n in arch:
        mse_rows.append([n, *[f"{x:.3f}" for x in a1[n]["mse_mean_axis"]], f"{a1[n]['angle_deg']:.1f}°", f"{a1[n]['go_f1']:.3f}"])
    body = (pill_("결론", G) + f"신규 v3 {ag['files']}건(10/14분, 18,176프레임)은 <b>읽기 문제 0건</b>이고 라벨 구조도 정상이다. az 단독 스파이크는 <b>{iso}/{nz}={iso / nz:.0%}</b>로 합격 기준(≤25%) 안이라 재수집이 필요 없다. 다만 가이드 이전 수집이라 <b>속력은 아직 거리와 무관(ρ=-0.02)</b>하다. 본학습 세트 218건(20,357프레임)에서 FM 헤드는 평균 기준선보다 방향 축을 잘 맞히지만 <b>단순 MLP 회귀와는 차이가 없다</b>."
        + H("① 수집 현황 — 5 위치 × 3 경로 (300ep 설계)") + wrap("셀별 도착 수(신규 191건 기준)", grid, "weak_right는 수집 진행 중, strong_right는 미수집. 10/13분 28건은 위치·경로 속성이 비어 있어 ‘unlabeled’로 별도 취급", kind="table")
        + H("② 검증 결과 — 파일·라벨·품질")
        + tbl(["항목", "결과", "판정"], [
            ["파일 읽기·스키마·NaN·타임스탬프", f"{ag['files']}건 모두 통과, 경고 5건(연속 동일 프레임 4건, 주행 중 간격 0.54s 1건)", "정상"],
            ["제외", "weak_right_straight 1건(18프레임·연속 동일 프레임 33%·NaN/길이 검사 미달)", "세트에서 제외"],
            ["중간 크기 비율 / 코너 비율", f"{ag['mid_ratio']:.0%} / {ag['corner']:.0%}", "합격(≥50%, ≤20%)"],
            ["az 단독 스파이크", f"{iso}/{nz} = {iso / nz:.0%}", "합격(≤25%). 초기 32건 시점의 36%는 일부 셀 편중이었고 전체에서는 낮아짐"],
            ["회전 프레임", f"{ag['spin']['frames']}개 중 이동 입력 동시 {ag['spin']['with_translation']}개 → exec는 회전만 남음(회전 우선 규칙)", "라벨 특성(주의)"],
            ["속력 단계 분포", f"느림 {lv['0.4']:.0%} / 보통 {lv['0.7']:.0%} / 빠름 {lv['1.0']:.0%} (목표 15~25 / 35~45 / 35~45)", "보통 많고 빠름 적음"],
            ["속력~바구니 면적 ρ", f"{V['speed_distance']['rho_speed_area']:+.2f} (판정 기준 |ρ|≥0.3)", "미충족 — 가이드 이전 수집"],
          ])
        + wrap("진행도별 평균 이동 크기 — 가이드가 의도한 램프와 비교", pr + legend([("신규", B), ("가이드 목표", A)]), "신규는 램프가 약하고 중반이 낮다(0.63 vs 0.72). 가이드(§12.3) 적용 전 수집이라 예상된 결과")
        + wrap("8-class 분포 — 신규 v3 vs 기존 V6", legend([("신규 v3", B), ("V6", MUT)]) + cls, "FORWARD가 줄고 회전(ROT_L/R)이 늘었다(회전 합 4.7% vs 1.7%). 회전 증가의 일부는 회전 우선 규칙 때문", kind="chart")
        + H("③ 기존 데이터와의 차이")
        + tbl(["데이터", "프레임", "고유 raw 액션", "중간 크기 비율", "비고"], [
            ["V6 (키보드)", "—", "18종 (±1.15 / 0)", "0%", "이산 LUT, 속력 정보 없음"],
            ["v1 아카이브(40ep)", "2,292", "185", "1%", "약한 스틱이 게이트에 제거, 구동 방식 다름"],
            ["파일럿 v2(15ep)", "1,236", "356", "75%", "v3 첫 합격 세트"],
            ["신규 v3(191ep)", f"{ag['frames']:,}", "4,312", f"{ag['mid_ratio']:.0%}", "속력 3단계 구동 C, raw/exec/scaled 3종 저장"]])
        + H("④ S1 학습 결과 — 본학습 세트 218ep, 5-fold × 5 seed, inner-val 조기 종료")
        + wrap("축별 R² (평균 기준선 대비 설명 비율)", legend([("A MLP-FM", B), ("B2 FiLM-FM", G), ("C π0형", A)]) + r2, "az는 희소해 R²≈0.05. lx 0.45~0.46, ly 0.67~0.69", kind="chart")
        + tbl(["모델", "lx MSE", "ly MSE", "az MSE", "방향 오차", "GO/STOP F1"], mse_rows)
        + f"방향 오차는 이동 중인 프레임의 첫 스텝 방향 각도 차이다. <b>FM 헤드 3종과 MLP 회귀의 MSE 차이는 0.003 이하</b>로 우열을 말할 수 없고, 평균 기준선 대비 개선(lx 0.093→0.05, ly 0.149→0.046)이 핵심이다. 구조 중 C가 방향 오차 {a1['C_pi0_adaln']['angle_deg']:.1f}°로 가장 낮다(A {a1['A_mlp_fm']['angle_deg']:.1f}°, B2 {a1['B2_film_res_fm']['angle_deg']:.1f}°)."
        + f"<br>{pill_('조기 종료', A)}최적 epoch 평균이 A·B2 {a1['A_mlp_fm']['best_epoch']:.0f}, C {a1['C_pi0_adaln']['best_epoch']:.0f}로 상한(100)에 가까워 <b>epoch을 늘리면 더 좋아질 수 있다</b>(미확인)."
        + f"<br>{pill_('az 마스크', P)}고립 az 스파이크를 마스킹한 타깃에서는 az MSE가 {a1['A_mlp_fm']['mse_mean_axis'][2]:.3f}→{a2['A_mlp_fm']['mse_mean_axis'][2]:.3f}로 줄었고(기준선도 {a1['A_mlp_fm']['base_mean_mse_axis'][2]:.3f}→{a2['A_mlp_fm']['base_mean_mse_axis'][2]:.3f}) R²는 {a1['A_mlp_fm']['r2_mean_axis'][2]:.2f}→{a2['A_mlp_fm']['r2_mean_axis'][2]:.2f}로 거의 같다. 마스킹으로 az가 학습되는 건 아니다."
        + ep250_block(a1, pill_, tbl)
        + H("⑤ 구현·결정 (plan_20261014 · plan_20261015 · plan_20261016)")
        + tbl(["항목", "내용"], [
            ["사용자 결정(10/15)", "폐루프는 15칸×20회=300회로 확대, 속력 출처는 첫 적용 섀도, az는 첫 적용 0, 판정 기준 |ρ|≥0.3·60~100ep 수용, az 스파이크는 재수집 없이 진행"],
            ["구현 완료", "속력 보간 규칙 <code>continuous_speed.py</code>(테스트 6개), 세트·S1 스크립트, <b>300ep 원샷 파이프라인</b> <code>run_egvla2_pipeline.py</code>, FM 헤드 공유 모듈 <code>egvla2_fm_heads.py</code>(분리 전후 출력 비트 동일), 서빙 정책 <code>egvla2_fm_policy.py</code>(F2 추출기 포함), 서버 <code>VLA_ACTION_MODE=continuous</code>(기본 꺼짐), 대시보드 <code>VLA_APPLY_CONT_SPEED</code>(기본 0), 오프라인 재생 <code>replay_egvla2_policy.py</code>"],
            ["검증됨", "서빙 추출기 vs 캐시: bbox 차이 0.0, 특징 차이 0.002(fp16 수준). 점검용 체크포인트로 재생 경로 동작 확인(성능 수치는 의미 없음)"],
            ["구현 미완", "최종 체크포인트(300ep 후) 재생·hold-out 비교, 젯슨·서버 지연 측정, 헤드 최종 선택, 드라이런·폐루프(300회)"],
            ["다음", "S2 서브골 FiLM(연속 타깃), epoch 확대 재시험, 남은 셀(weak_right, strong_right) 수집 후 세트 갱신, 가이드(§12.3) 적용 후 속력 판정 재측정"]])
        + pill_("한계", R) + "S1은 한 장소에서 수집한 데이터의 에피소드 단위 hold-out이라 장소·환경 일반화는 보지 못한다. 위치별 일반화(leave-one-position-out)와 폐루프는 아직이다.")
    return f'<div class="finding-card" style="border-color:{A}"><div class="finding-title" style="color:{A}">76-10. 신규 배치 검증 · 기존 대비 차이 · S1 학습 결과</div><div style="font-size:0.85rem;line-height:1.8;color:#94a3b8">{body}</div></div>'


PILLS = {"판단": G, "추천": B, "한계": R, "다음": A, "남은 것": A, "실측 근거": MUT, "설정": MUT, "FM 학습에 미치는 영향": P, "동시 수집 구조(제안)": P,
         "결과": G, "조기 종료": A, "B2 시험(후속)": B}


def pill(label, col):
    return f'<span style="display:inline-block;background:{col}22;color:{col};border:1px solid {col};border-radius:6px;padding:0 8px;font-size:0.76rem;font-weight:800;margin-right:4px">{label}</span>'


def style_text(seg):
    for label, col in PILLS.items():
        seg = seg.replace(f"<b>{label}</b>", pill(label, col))
    # 번호 소제목(<b>① …</b>) → 구분선 있는 소제목 블록
    seg = re.sub(r"<b>([①②③④⑤⑥])\s?([^<]{2,90})</b>",
                 lambda m: f'<div style="margin:16px 0 6px;padding-bottom:4px;border-bottom:2px solid {A}55;color:{AL};font-size:0.92rem;font-weight:800">{m.group(1)} {m.group(2)}</div>', seg)
    return seg


LEGEND = ('<!--LEG76--><div style="display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:10px 0 4px;font-size:0.76rem;color:#94a3b8">'
          '<span style="font-weight:800;color:#e2e8f0">읽는 법</span>' + "".join(badge(k) for k in KINDS)
          + '<span style="display:inline-block;background:#33415522;color:#cbd5e1;border:1px solid #475569;border-radius:999px;padding:1px 10px;font-size:0.68rem;font-weight:800">📝 설명(왼쪽 선 없는 본문)</span>'
          + pill("판단", G) + pill("추천", B) + pill("한계", R) + pill("다음", A) + "</div><!--/LEG76-->")



def main():
    hist = compute_hist()
    V = build(hist)
    s = HTML.read_text(encoding="utf-8")
    a = s.index('<div class="chapter" id="ch76">'); b = s.index("/CH76 ═")
    seg = s[a:b]
    seg = re.sub(r"<!--VIZ76-->.*?<!--/VIZ76-->", "", seg, flags=re.S)
    seg = re.sub(r"<!--LEG76-->.*?<!--/LEG76-->", "", seg, flags=re.S)
    if 'data-t="t10"' not in seg:
        seg = seg.replace('🧭 속력 정책 방향</button>', '🧭 속력 정책 방향</button><button class="ctab-btn" data-t="t10" onclick="ctabShow(\'ch76\',\'t10\')">🧪 신규 배치 검증·S1</button>', 1)
        seg = seg.replace('  <script>ctabInit("ch76")</script>', '  <div class="ctab-panel" data-t="t10" style="display:none">T10PLACEHOLDER</div>\n  <script>ctabInit("ch76")</script>', 1)
    if 'data-t="t9"' not in seg:
        seg = seg.replace('🕹 입력·변환 설계</button>', '🕹 입력·변환 설계</button><button class="ctab-btn" data-t="t9" onclick="ctabShow(\'ch76\',\'t9\')">🧭 속력 정책 방향</button>', 1)
        seg = seg.replace('  <script>ctabInit("ch76")</script>', '  <div class="ctab-panel" data-t="t9" style="display:none">T9PLACEHOLDER</div>\n  <script>ctabInit("ch76")</script>', 1)
    seg = re.sub(r'(<div class="ctab-panel" data-t="t9" style="display:none">).*?(</div>\n  <div class="ctab-panel" data-t="t10")', lambda m: m.group(1) + t9_card() + m.group(2), seg, count=1, flags=re.S)
    seg = re.sub(r'(<div class="ctab-panel" data-t="t10" style="display:none">).*?(</div>\n  <script>ctabInit)', lambda m: m.group(1) + t10_card() + m.group(2), seg, count=1, flags=re.S)
    starts = [(m.start(), m.group(1)) for m in re.finditer(r'  <div class="ctab-panel" data-t="(t\d+)"', seg)]
    tail = seg.index('  <script>ctabInit("ch76")</script>')
    bounds = [st for st, _ in starts][1:] + [tail]
    out = seg[:starts[0][0]]
    for (st, tid), en in zip(starts, bounds):
        chunk = seg[st:en]
        k = chunk.rstrip().rfind("</div>")
        if tid in V:
            chunk = chunk[:k] + "<!--VIZ76-->" + V[tid] + "<!--/VIZ76-->" + chunk[k:]
        out += chunk
    seg = out + seg[tail:]
    seg = style_text(seg)
    s = s[:a] + seg + s[b:]
    HTML.write_text(s, encoding="utf-8")
    for t in ["div", "svg", "table", "tr", "button"]:
        print(t, len(re.findall(f"<{t}[ >]", s)), len(re.findall(f"</{t}>", s)))


if __name__ == "__main__":
    main()

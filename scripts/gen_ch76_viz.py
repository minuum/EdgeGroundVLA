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
    S = json.load(open(ROOT / "logs/manual_dataset_sets.json"))["sets"]["pilot_v2_gauge_20261013"]
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


def wrap(title, inner, note=""):
    n = f'<div style="font-size:0.74rem;color:#64748b;margin-top:8px">{note}</div>' if note else ""
    return (f'<div style="background:#0a1628;border:1px solid #1e3050;border-radius:10px;padding:14px 16px;margin:14px 0">'
            f'<div style="color:{AL};font-size:0.82rem;font-weight:800;margin-bottom:10px">{title}</div>{inner}{n}</div>')


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
    # ---------- t2 ----------
    V["t2"] = wrap("① 수집 경로 비교 — 이전(v1) vs 지금(v3)",
        '<div style="font-size:0.74rem;color:%s;margin-bottom:4px">이전 (v1, 40건)</div>' % R
        + flow([("스틱 3축", MUT), ("축별 데드존 0.15", R, "방향 왜곡"), ("<code>if key:</code> 게이트", R, "임계 0.5"), ("각도 + 고정 throttle", R, "속력 무반영"), ("<code>actions</code>=구동값", R, "대각선 최대 1.41")])
        + '<div style="font-size:0.74rem;color:%s;margin:12px 0 4px">지금 (v3, 28건)</div>' % G
        + flow([("스틱 3축", MUT), ("방사형 데드존 + 재조정", G), ("센터링 0.03 스냅", G, "직진 드리프트 제거"), ("<code>has_input</code>", G, "크기&gt;0"), ("원형게이트", G, "크기≤1"), ("구동 C", G, "throttle×3단계"), ("raw/scaled/exec", G, "+타임스탬프")]),
        "초록 = 이번에 바뀐 단계, 빨강 = 이전의 문제 단계") + wrap(
        "② 저장 필드 변화",
        flow([("v1: <code>actions</code> 1개", R, "의미 모호"), ("v2: raw + scaled", A, "버전 분리"), ("v3: raw + scaled + exec + frame_timestamps", G, "derive_version, 속도 보정 attrs")]))
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
    # ---------- t4 ----------
    V["t4"] = wrap("① 선택지 비교 매트릭스 (평가자 판단 기준, 데이터 근거는 76-3·76-6)",
        heat([("라벨 = 실제 구동 일치", [0, 2, 2]), ("감속(속력) 정보 학습", [2, 0, 1]), ("odom 없이 효과 검증 가능", [0, 2, 2]), ("구현·보정 비용 낮음", [1, 2, 1]), ("라벨 노이즈 위험 낮음", [0, 2, 1])],
             ["A 크기→throttle 연속", "B 각도 + 고정 속력", "C 속력 3단계 + 각도"]),
        "A는 구동이 크기를 쓰지 않는 한 라벨이 실제와 어긋남. C는 보정(스톱워치) 이후 일치.") + wrap(
        "② 결정 흐름 — 채택은 C(보정 완료)",
        flow([("① 게이트 수정", G, "완료"), ("② B로 파일럿 (라벨=실제)", MUT, "건너뜀"), ("③ 스톱워치 보정", G, "완료 14.45/20.64/26.36 cm/s"), ("④ C로 확장 (v3)", G, "채택"), ("A", R, "odom 전까지 보류")]))
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
    if s_wp:
        r = s_wp["results"]
        rows = []
        for k, c in (("chunk|A_mlp_fm", MUT), ("waypoint|A_mlp_fm", G), ("chunk|C_pi0_adaln", MUT), ("waypoint|C_pi0_adaln", B)):
            if k in r: rows.append((k.replace("|", " · "), c, r[k]["r2"]))
        if rows:
            V["t7"] = V.get("t7", "") + wrap("⑤ 명령 적분 → waypoint 타깃 시험 (V6 225ep, 마지막 waypoint의 R²)", legend([(a_, c) for a_, c, _ in rows])
                + vgroups(["lx", "ly", "az"], [(a_, c, [max(v[0], 0), max(v[1], 0), max(v[2], 0)]) for a_, c, v in rows], 1.0, fmt="{:.2f}", bw=18),
                "같은 분할·입력에서 chunk(명령 8개)와 waypoint(누적합) 타깃을 같은 척도(마지막 waypoint)로 비교. V6 명령은 ±1.15 이산값, 속도·시간 미보정(프레임 단위)")
    # ---------- t8 ----------
    V["t8"] = wrap("① 입력 → 저장 구조도", flow([("축0 (왼쪽 X)", MUT, "→ ly"), ("축1 (왼쪽 Y)", MUT, "→ lx"), ("축2 (오른쪽 X)", MUT, "→ az"), ("방사형 데드존+재조정", G, "0.15"), ("센터링 스냅", G, "0.03"), ("<code>actions_raw</code>", A, "기준값")])
        + '<div style="height:8px"></div>' + flow([("A: 원형게이트", B, "scaled"), ("B: 각도 + GO/STOP", B), ("C: 각도 + 속력 3단계", B, "exec")])
        + '<div style="height:8px"></div>' + flow([("구동 C", G, "driver.move(각도, throttle×단계)"), ("<code>actions_exec</code>", A, "실제로 한 일"), ("+ frame_timestamps · derive_version", MUT)]),
        "축 3개만 데이터가 되고, 버튼·D-pad는 메타데이터. az는 변환 없이 저장, 구동은 |az|&gt;0.1이면 회전만") + wrap(
        "② 속력 3단계 실측 보정 (2m 코스, 레벨당 10회)",
        hbars([("레벨0 느림 (scale 0.40)", 14.45, B, "14.45 cm/s"), ("레벨1 보통 (scale 0.70)", 20.64, G, "20.64 cm/s"), ("레벨2 빠름 (scale 1.00)", 26.36, A, "26.36 cm/s")], 30, w=190),
        "선형근사: 속도 ≈ 6.5 + 19.85 × scale (cm/s) — 레벨1 예측 20.4 vs 실측 20.64 (1% 이내). 모터 기저속도(데드밴드) 추정") + wrap(
        "③ 버튼·입력 구성", hbars([("주행 아날로그 축", 3, G, "3개 → 학습 입력"), ("기능 버튼", 8, MUT, "8개"), ("보조 버튼 (복귀)", 1, MUT, "1개"), ("D-pad", 1, MUT, "1개 (시나리오 선택)")], 10, w=190),
        "학습 데이터에 들어가는 입력은 축 3개뿐")
    return V


def main():
    hist = compute_hist()
    V = build(hist)
    s = HTML.read_text(encoding="utf-8")
    a = s.index('<div class="chapter" id="ch76">'); b = s.index("/CH76 ═")
    seg = s[a:b]
    seg = re.sub(r"<!--VIZ76-->.*?<!--/VIZ76-->", "", seg, flags=re.S)
    starts = [(m.start(), m.group(1)) for m in re.finditer(r'  <div class="ctab-panel" data-t="(t\d)"', seg)]
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
    s = s[:a] + seg + s[b:]
    HTML.write_text(s, encoding="utf-8")
    for t in ["div", "svg", "table", "tr", "button"]:
        print(t, len(re.findall(f"<{t}[ >]", s)), len(re.findall(f"</{t}>", s)))


if __name__ == "__main__":
    main()

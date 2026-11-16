"""CH82 — a3 vs a3-s2: 실기(11/13·11/16)와 시뮬레이션(재생) 대조 브리핑. 데이터는 docs/v5/egvla2_fm/a3c/*.json에서 읽는다."""
import base64, io, json, math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parent.parent; A = ROOT / "docs/v5/egvla2_fm/a3c"; FIG = ROOT / "docs/v5/figures/objtest"
J = lambda n: json.loads((A / n).read_text(encoding="utf-8"))
G, Y, R, B = "#4ade80", "#fbbf24", "#f87171", "#38bdf8"
P = lambda t: f'<div style="font-size:0.82rem;line-height:1.8;margin:6px 0">{t}</div>'
COL = {"a3": "#38bdf8", "a3-s1": "#c084fc", "a3-s2": "#fb923c", "a3-ens3": "#4ade80"}


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h) * 100, min(1, c + h) * 100


def _bars(title, groups, a, b, la, lb, ymax=0.6):
    W, Hh, L, Bm = 760, 240, 40, 62; n = len(groups); bw = (W - L - 10) / n; y = lambda v: 10 + (Hh - Bm - 10) * (1 - min(v, ymax) / ymax)
    g = f'<line x1="{L}" y1="{y(.25):.1f}" x2="{W-10}" y2="{y(.25):.1f}" stroke="{R}" stroke-dasharray="4 3"/><text x="{W-12}" y="{y(.25)-3:.1f}" text-anchor="end" fill="{R}" font-size="9">near_area 0.25</text>'
    g += f'<line x1="{L}" y1="{y(0):.1f}" x2="{W-10}" y2="{y(0):.1f}" stroke="#334155"/>'
    for i, k in enumerate(groups):
        x0 = L + i * bw + bw * .12; w = bw * .34
        for j, (v, c) in enumerate(((a.get(k), COL["a3"]), (b.get(k), COL["a3-s2"]))):
            if v is None: continue
            g += f'<rect x="{x0 + j * (w + 3):.1f}" y="{y(v):.1f}" width="{w:.1f}" height="{y(0) - y(v):.1f}" fill="{c}"/><text x="{x0 + j * (w + 3) + w / 2:.1f}" y="{y(v) - 3:.1f}" text-anchor="middle" fill="#e2e8f0" font-size="9">{v:.2f}</text>'
        g += f'<text x="{L + i * bw + bw / 2:.1f}" y="{Hh - 40}" text-anchor="middle" fill="#94a3b8" font-size="10" transform="rotate(-20 {L + i * bw + bw / 2:.1f} {Hh - 40})">{k}</text>'
    g += f'<text x="{L}" y="{Hh - 8}" font-size="10"><tspan fill="{COL["a3"]}">■ {la}</tspan>  <tspan fill="{COL["a3-s2"]}">■ {lb}</tspan></text>'
    return f'<div style="font-size:0.8rem;font-weight:800;color:#fcd34d;margin:6px 0">{title}</div><svg viewBox="0 0 {W} {Hh}" style="width:100%;height:auto;background:#0a1628;border:1px solid #1e3050;border-radius:8px">{g}</svg>'


def _lines(title, series, xlab, ymax, note, hline=None, W=520, Hh=190):
    L, Bm = 36, 34; n = max(len(v) for v in series.values()); x = lambda i: L + (W - L - 10) * (i / max(n - 1, 1)); y = lambda v: 8 + (Hh - Bm - 8) * (1 - min(max(v, 0), ymax) / ymax)
    g = f'<line x1="{L}" y1="{y(0):.1f}" x2="{W-10}" y2="{y(0):.1f}" stroke="#334155"/><text x="{L-4}" y="{y(0)+3:.1f}" text-anchor="end" fill="#64748b" font-size="9">0</text><text x="{L-4}" y="{y(ymax)+8:.1f}" text-anchor="end" fill="#64748b" font-size="9">{ymax:g}</text>'
    if hline: g += f'<line x1="{L}" y1="{y(hline[0]):.1f}" x2="{W-10}" y2="{y(hline[0]):.1f}" stroke="{R}" stroke-dasharray="4 3"/><text x="{W-12}" y="{y(hline[0])-3:.1f}" text-anchor="end" fill="{R}" font-size="9">{hline[1]}</text>'
    for k, v in series.items():
        g += f'<polyline fill="none" stroke="{COL.get(k, "#94a3b8")}" stroke-width="2" points="' + " ".join(f"{x(i):.1f},{y(t):.1f}" for i, t in enumerate(v)) + '"/>'
    leg = "  ".join(f'<tspan fill="{COL.get(k, "#94a3b8")}">━ {k}</tspan>' for k in series)
    g += f'<text x="{L}" y="{Hh - 20}" fill="#94a3b8" font-size="9.5">{xlab}</text><text x="{L}" y="{Hh - 6}" font-size="10">{leg}</text>'
    return f'<div><div style="font-size:0.78rem;font-weight:800;color:#fcd34d;margin:4px 0">{title}</div><svg viewBox="0 0 {W} {Hh}" style="width:100%;height:auto;background:#0a1628;border:1px solid #1e3050;border-radius:8px">{g}</svg><div style="font-size:0.72rem;color:#94a3b8;line-height:1.55">{note}</div></div>'


def _b64(im, q=70):
    b = io.BytesIO(); im.save(b, "JPEG", quality=q); return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def _arrow_img(sid, i, v, col, size=(224, 126), area=None):
    im = Image.open(FIG / f"{sid}_{i}.jpg").convert("RGB").resize(size); d = ImageDraw.Draw(im); w, h = size; cx, cy = w / 2, h - 10; n = float(np.hypot(*v))
    if n < .2:
        d.text((cx - 22, cy - 12), f"정지 {n:.2f}", fill=(255, 90, 90))
    else:
        L = min(n, 1.2) * h * .65; ux, uy = v[1] / n, -v[0] / n; x2, y2 = cx + ux * L, cy + uy * L
        d.line([(cx, cy), (x2, y2)], fill=(0, 0, 0), width=7); d.line([(cx, cy), (x2, y2)], fill=col, width=4); d.ellipse([x2 - 5, y2 - 5, x2 + 5, y2 + 5], fill=col)
    d.text((4, 3), f"{n:.2f}" + (f" · 면적 {area:.2f}" if area is not None else ""), fill=(255, 255, 255))
    return im


def _strip(sim, sid, frames, heads, title, note):
    s = sim[sid]; hexc = {"a3": (56, 189, 248), "a3-s1": (192, 132, 252), "a3-s2": (251, 146, 60), "a3-ens3": (74, 222, 128)}
    rows = ""
    for h in heads:
        cells = "".join(f'<div style="flex:0 0 auto"><img src="{_b64(_arrow_img(sid, i, s["cmd"][h][i], hexc[h], area=s["area"][i]))}" style="width:168px;height:auto;border-radius:6px;display:block"><div style="font-size:0.68rem;color:#94a3b8;text-align:center">프레임 {i + 1}</div></div>' for i in frames)
        rows += f'<div style="margin:4px 0"><div style="font-size:0.74rem;font-weight:800;color:{COL[h]}">{h}</div><div style="display:flex;gap:6px;overflow-x:auto">{cells}</div></div>'
    return f'<div style="margin:10px 0;padding:8px 10px;border:1px solid #1e3050;border-radius:10px;background:#0a1220"><div style="font-size:0.84rem;font-weight:800;color:#fcd34d">{title}</div>{rows}<div style="font-size:0.74rem;color:#94a3b8;line-height:1.6;margin-top:4px">{note}</div></div>'


def tabs(tbl, H, st):
    sim = J("ch82_sim.json"); RV = J("real_1116_vs_1113.json"); VS = J("verify_seeds.json"); DD = J("dirdiff_all.json"); SE = J("eval_seeds.json")
    s2, a3r = RV["a3-s2(11/16)"], RV["a3(11/13)"]
    ts = A / "ch82_train_stoparea.json"; TS = json.loads(ts.read_text(encoding="utf-8")) if ts.exists() else None
    groups = ["바구니·가운데", "바구니·왼쪽", "바구니·오른쪽", "의자·가운데", "의자·오른쪽", "의자·왼쪽", "책상·가운데", "책상·왼쪽", "책상·오른쪽"]
    fa = {k: v["마지막면적"] for k, v in a3r.items()}; fb = {k: v["마지막면적"] for k, v in s2.items()}
    lo_c, hi_c = wilson(5, 5); lo_0, hi_0 = wilson(0, 5)

    # ── t1 요약
    t1 = [H("1. 한눈에 — 두 가지 질문에 대한 답이 서로 반대로 나왔다"),
          tbl(["질문", "결과", "근거와 수준"], [
              ["가운데 위치의 새 물체에서 출발하는가", st("해결", G) + " 의자·가운데 5/5 (a3 11/13: 0/5), 책상·가운데 5/5 (a3: 0/1)", f"실기 11/16, n=5. 5/5의 95% 구간 {lo_c:.0f}~{hi_c:.0f}% vs 0/5 {lo_0:.0f}~{hi_0:.0f}%(겹치지 않음). 운영자 수동 판정·성공 기준 없음·비맹검 여부 미확인"],
              ["도착할 때 제때 멈추는가", st("악화 가능", R) + " 바구니 종료 면적 0.30~0.45 (a3 0.23), 바구니 옆 10회 중 접촉 보고 6건", "실기 11/16 + 동환 구두 관찰(CSV 미반영). a3 쪽 접촉 기록은 없어 직접 비교 불가. 원인은 미확정"],
              ["방향(어느 쪽으로 가는가)", st("같음", G) + " a3 대비 방향 차이 중앙값 0.7°(a3-s2)", "오프라인 재생 226세션. 시드끼리 차이(1.4°) 수준"],
              ["성공률", st("판단 보류", Y) + " 45/45 (a3 455/455와 11/13 49/65)", "접촉이 CSV 성공에 포함됨. 성공률만으로 비교하면 안 됨"],
          ]),
          H("2. 판단에 필요한 것"),
          P("• <b>(A) 현재 배포 <code>a3</code> 유지</b> — 바구니 접촉은 줄지만 가운데 새 물체는 계속 출발하지 못함.<br>"
            "• <b>(B) <code>a3-s2</code>로 교체</b> — 가운데는 풀리지만 도착이 거칠어짐(접촉 보고).<br>"
            "• <b>(C) 정지를 보강한 새 후보</b> — 시드 평균·정지 전용 학습·<code>near_area</code> 조정 중 선택 후 오프라인 비교, 실기 재시험 필요.<br>"
            "각 선택지의 근거·위험·필요한 시험은 <b>⚖ 선택지·판단</b> 탭에 정리했다. <b>어떤 쪽이든 먼저 정해야 할 것: 성공 기준(접촉 포함 여부)과 맹검.</b>"),
          P(f'링크: <a href="object_test_compare_1113.html" style="color:{B}">11/13 동시 비교 재생</a> · <a href="model_registry.html#EG2-14" style="color:{B}">레지스트리 a3-s2</a> · 계획서 <code>plan_20261115_a3_improvement.md</code> §8~11')]

    # ── t2 실기 비교
    tr = []
    for g in groups:
        a, b = a3r.get(g), s2.get(g)
        tr.append([g, f'{a["성공"]}/{a["n"]}' if a else "시험 없음", f'{b["성공"]}/{b["n"]}' if b else "시험 없음", f'{a["마지막면적"]:.3f}' if a else "–", f'{b["마지막면적"]:.3f} ({b["마지막면적_범위"][0]:.2f}~{b["마지막면적_범위"][1]:.2f})' if b else "–"])
    def aligned(sessions, n=10):
        out = []
        for a in sessions:
            a = np.array(a)
            if len(a) >= 2: out.append(np.pad(a, (max(0, n - len(a)), 0), mode="edge")[-n:])
        return np.mean(out, 0).tolist()
    S16 = J("real_1116_sessions_bbox_area.json"); S16c = {k: v for k, v in S16.items() if v["instruction"] == "go to the gray basket"}
    a3_bk = [v["area"] for v in sim.values() if v["obj"] == "바구니"]; s2_bk = [v["bbox_area_seq"] for v in S16c.values()]
    import csv as _csv
    route = {r["session_id"]: r["경로"] for r in _csv.DictReader(open(A / "real_1116_episode_log_slice.csv", encoding="utf-8"))}
    a3_ch = [v["area"] for v in sim.values() if v["obj"] == "의자" and "가운" not in v["pos"]]; s2_ch = [v["bbox_area_seq"] for k, v in S16.items() if v["instruction"] == "go to the brown wooden chair" and route.get(k) != "trackF_center"]
    t2 = [H("1. 11/13 a3 vs 11/16 a3-s2 — 같은 지시어·같은 서버 컨트롤러(near_area 0.25)"),
          tbl(["물체·위치", "a3 성공(11/13)", "a3-s2 성공(11/16)", "a3 종료 박스 면적", "a3-s2 종료 박스 면적(범위)"], tr),
          P("<b>읽는 법</b>: 종료 면적은 세션 마지막 프레임의 박스 면적(지시어로 찾은 박스)이다. 0.25는 컨트롤러 정지 규칙(연속 2프레임)의 문턱이다. 같은 컨트롤러에서 a3는 바구니에서 약 0.23에서 끝났고 a3-s2는 0.30~0.45에서 끝났다. 책상은 둘 다 약 0.215로 차이가 없다. 세션 종료는 전부 운영자 수동이다."),
          _bars("물체·위치별 종료 박스 면적 (파랑 a3, 주황 a3-s2)", groups, fa, fb, "a3 11/13", "a3-s2 11/16", 0.6),
          H("2. 끝에서 거슬러 본 박스 면적 변화 (마지막 10프레임 평균)"),
          '<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px">' +
          _lines("바구니 (a3 15세션 vs a3-s2 15세션)", {"a3": aligned(a3_bk), "a3-s2": aligned(s2_bk)}, "프레임 −9 → 0(마지막)", 0.6, "a3-s2는 마지막 몇 프레임에서 면적이 0.25를 넘어 계속 커진다. a3는 0.2 안팎에서 끝난다.", (0.25, "near_area 0.25")) +
          _lines("의자 옆 (a3 오른쪽 10세션 vs a3-s2 오른쪽·왼쪽 10세션)", {"a3": aligned(a3_ch), "a3-s2": aligned(s2_ch)}, "프레임 −9 → 0(마지막)", 0.6, "a3-s2의 의자 옆 종료 면적이 더 크다(표 참고). 위치(오른쪽/왼쪽)가 섞여 있어 대략적 비교다.", (0.25, "near_area 0.25")) + "</div>",
          H("3. 접촉 관찰 (동환 구두, CSV 미반영)"),
          tbl(["묶음", "관찰", "비고"], [["바구니 오른쪽 5회", "1회 부딪혀서 정지시킴, 5회 살짝 부딪힘 (2~4회는 언급 없음)", "한 세션 마지막 프레임 박스 면적 0.996"], ["바구니 왼쪽 5회", "1·2·3회 아주 살짝 부딪힘, 4회 안 부딪힘, 5회 5cm 넘게 밀음", ""], ["책상 오른쪽 4회차", "살짝 부딪힘 (세션 20261116_153524)", "책상 15회 중 1건"], ["의자", "'몇 번 제외하고 알아서 정지'", "횟수·세션 미정리"]]),
          P("a3의 접촉 여부는 11/13에 기록한 적이 없어 비교할 수 없다. 접촉 기록 방법(예: 시험마다 접촉/비접촉 열 추가)을 다음 시험 전에 정하는 것이 좋다.")]

    # ── t3 시뮬레이션 대조
    heads = ["a3", "a3-s1", "a3-s2", "a3-ens3"]; nm = lambda c: np.linalg.norm(np.array(c), axis=1)
    succ = [s for s, v in sim.items() if v["res"] == "성공"]; rowsA = []
    for h in heads:
        sa, cnt = [], 0
        for s in succ:
            v = sim[s]; n = nm(v["cmd"][h]); a = np.array(v["area"]); mv = np.where(n > .2)[0]
            if len(mv) == 0: continue
            lt = np.where(n[mv[0]:] < .2)[0]
            if len(lt): sa.append(a[mv[0] + lt[0]]); cnt += 1
        rowsA.append([h, f"{cnt}/{len(succ)}", f"{np.median(sa):.3f}" if sa else "–"])
    ch = [("20261113_150911", "바구니·왼쪽 (a3 성공)", "프레임별 명령 크기. 점선 0.2 아래면 정지."), ("20261113_154226", "의자·오른쪽 (a3 성공)", ""), ("20261113_170121", "의자·가운데 (a3 실패, 내내 정지)", "a3만 0, 나머지는 움직인다."), ("20261113_162953", "책상·가운데 (a3 실패)", "")]
    charts = ""
    for sid, lab, nt in ch:
        if sid not in sim: continue
        v = sim[sid]; charts += _lines(lab, {h: nm(v["cmd"][h]).tolist() for h in heads}, f"프레임 1 → {v['n']}", 1.0, nt, (0.2, "정지 기준 0.2"))
    t3 = [H("1. 먼저 알아둘 한계 — 11/13 프레임은 a3가 멈춘 곳에서 끝난다"),
          P("11/13 기록 프레임은 <b>a3가 실제로 달린 경로</b>라서 박스 면적이 대부분 0.23~0.29 이하에서 끝난다. a3-s2가 실기에서 도달한 0.3~0.5 구간의 프레임은 없으므로, 이 재생으로는 a3-s2가 가까이 갔을 때 정지하는지를 직접 볼 수 없다(검열). 아래 표는 '볼 수 있는 구간 안에서'의 비교다."),
          H("2. 11/13 성공 세션 49개 재생 — 이동 후 처음 정지(크기<0.2)를 내는 박스 면적"),
          tbl(["헤드", "정지를 낸 세션 수", "그때의 박스 면적 중앙값"], rowsA),
          P("관찰 구간 안에서는 a3·a3-s2의 정지 면적이 비슷하다(차이가 작음). 즉 <b>이 재생만으로는 'a3-s2의 정지 능력이 약하다'를 확인하지 못한다.</b> 이전에 근거로 쓴 '11/13 바구니 15세션 끝 3프레임 정지 a3 20%, a3-s2 11%'도 같은 검열 구간의 값이라 약한 증거다.")]
    if TS:
        t3 += [H("3. 면적이 더 큰 구간을 가진 학습 에피소드로 확인 (학습에 쓴 데이터, 150개)"),
               tbl(["헤드/라벨", "정지를 낸 에피소드", "정지 면적 중앙값", "정지 면적 평균", "면적 0.25 이전에 정지한 비율"],
                   [[k, f'{TS[k]["정지_있는_에피소드"]}/{TS[k]["n"]}', TS[k]["정지면적_중앙값"], TS[k]["정지면적_평균"], TS[k]["이전_정지_비율_면적025"]] for k in ["라벨", "a3", "a3-s1", "a3-s2"]]),
               P(f"에피소드 마지막 박스 면적 중앙값 {TS['_에피소드_마지막_박스면적_중앙값']}. 학습에 쓴 데이터이므로 일반화 지표가 아니라 '학습된 정지 면적이 어디에 있는가'의 참고다."),
               P(f"<b>해석</b>: a3-s2는 면적 0.25 이전에 멈추는 비율이 {TS['a3-s2']['이전_정지_비율_면적025']:.0%}로 a3({TS['a3']['이전_정지_비율_면적025']:.0%})보다 낮고 정지 면적 평균도 더 크다({TS['a3-s2']['정지면적_평균']} 대 {TS['a3']['정지면적_평균']}). <b>방향은 '조금 늦게 멈춘다'와 같지만 차이가 작다</b> — 실기 종료 면적 차이(바구니 약 0.41 대 0.23)를 이 정도로는 설명하지 못한다. 중앙값은 거의 같다({TS['a3-s2']['정지면적_중앙값']} 대 {TS['a3']['정지면적_중앙값']}). 실기 차이에는 시험일·배치·운영자 정지 시점 같은 요인이 섞였을 수 있고, 어느 쪽이 얼마나 기여하는지는 이 데이터로 가를 수 없다. 정답 라벨의 정지 면적(중앙값 {TS['라벨']['정지면적_중앙값']})은 헤드들보다 훨씬 작아, 세 헤드 모두 사람 시연보다 늦게 멈추도록 학습된 편이다(라벨 정지에는 시연 중 일시 정지가 섞여 있어 단순 비교는 조심).")]
    t3 += [H("4. 프레임별 명령 크기 — a3·a3-s1·a3-s2·앙상블"), '<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin:8px 0">' + charts + "</div>",
           H("5. 오프라인 지표 요약"),
           tbl(["지표", "a3", "a3-s1", "a3-s2", "앙상블(3)"], [
               ["T7 452세션: 기록된 실제 a3와 이동/정지 일치", f'{VS["A3(시드0,배포)"]["T7_기록(실제 A3)과_이동정지_일치"]:.3f}', f'{VS["시드1"]["T7_기록(실제 A3)과_이동정지_일치"]:.3f}', f'{VS["시드2"]["T7_기록(실제 A3)과_이동정지_일치"]:.3f}', f'{VS["앙상블(3)"]["T7_기록(실제 A3)과_이동정지_일치"]:.3f}'],
               ["T7: 세션 끝 3프레임 정지", f'{VS["A3(시드0,배포)"]["T7_세션끝_정지"]:.3f}', f'{VS["시드1"]["T7_세션끝_정지"]:.3f}', f'{VS["시드2"]["T7_세션끝_정지"]:.3f}', f'{VS["앙상블(3)"]["T7_세션끝_정지"]:.3f}'],
               ["11/13 의자·가운데 이동", VS["A3(시드0,배포)"]["11/13_그룹별_이동"]["의자·가운데"], VS["시드1"]["11/13_그룹별_이동"]["의자·가운데"], VS["시드2"]["11/13_그룹별_이동"]["의자·가운데"], VS["앙상블(3)"]["11/13_그룹별_이동"]["의자·가운데"]],
               ["a3 대비 방향 차이 중앙값(도)", "기준", DD["A3 레시피 시드1"]["중앙값"], DD["A3 레시피 시드2"]["중앙값"], f'{VS["앙상블(3)"]["T7_A3시드0와_방향차이_중앙값"]}']]),
           P("<b>같은 레시피 시드 편차</b>: 의자·가운데 이동은 박스 가림 30%에서 시드 0/1/2가 0/100/100%였다(<code>eval_seeds.json</code>). 시드 0 하나가 이 장면에서 정지하는 쪽으로 학습된 개별 체크포인트로 보인다.")]

    # ── t4 프레임 대조
    sc = [_strip(sim, "20261113_170121", [0, 3, 6, 9, 12, 15], ["a3", "a3-s2", "a3-ens3"], "의자·가운데 (11/13 a3 실패 세션): 같은 프레임에서 모델별 첫 명령", "a3는 모든 프레임에서 정지(크기 0). a3-s2·앙상블은 전진 명령을 낸다. 노란 박스가 아닌 화살표만 표시(크기 그대로, 0.2 미만이면 '정지').")]
    bs = "20261113_150911"; n = sim[bs]["n"]; sc.append(_strip(sim, bs, list(range(max(0, n - 6), n)), ["a3", "a3-s2", "a3-ens3"], "바구니·왼쪽 (11/13 a3 성공 세션): 도착 직전 프레임", "이 구간(면적 약 0.2~0.26)에서는 세 헤드 모두 크기가 줄어든다. a3-s2가 실기에서 더 가까이 간 구간(면적 0.3 이상)의 프레임은 없다."))
    t4 = [H("프레임 대조 — 같은 프레임, 모델별 명령(재생)"), P("화살표는 위가 전진, 오른쪽이 우측 이동, 길이가 명령 크기다. 11/16 실기 프레임은 받지 못해(면적 시계열만) 11/13 프레임으로 대조한다.")] + sc

    # ── t5 선택지
    t5 = [H("선택지와 근거 — 판단은 사용자가"),
          tbl(["선택지", "얻는 것", "위험·한계", "필요한 시험", "비용"], [
              ["<b>A. a3 유지</b>", "바구니 도착이 안정적(455/455, 11/13 바구니 종료 면적 0.23)", "가운데 새 물체(의자·책상)에서 출발 못 함(0/5)", "없음", "없음"],
              ["<b>B. a3-s2 교체</b>", "가운데 새 물체 출발(5/5, 5/5)", "바구니·의자에서 더 가까이 가서 끝남, 접촉 보고 6건. a3-s2 하나만 실기 시험됨", "접촉을 기록하는 재시험(바구니·의자)", "시험 1회"],
              ["<b>C-1. 시드 평균(앙상블) 또는 a3-s1</b>", "방향 동일, 가운데 풀림(오프라인). a3-s1은 정지 면적이 큰 쪽(0.241)", "서빙 코드에 평균이 없음(구현 필요). 도착 정지를 개선한다는 증거는 아직 없음", "오프라인 정지 비교(검열 한계) + 실기", "구현 + 시험"],
              ["<b>C-2. 정지 전용 보강 학습</b>", "도착 판단을 따로 학습", "새 학습·검증 필요, 개선 보장 없음", "오프라인 + 실기", "학습 + 시험"],
              ["<b>C-3. near_area 일괄 낮춤</b>", "접촉 감소 가능(a3 바구니 종료 면적 ~0.23 근처 기준)", "물체마다 크기가 달라 일괄 적용은 책상 등을 일찍 세울 수 있음. 물체별 분기는 일반성이 떨어짐(로봇 쪽 제안은 보류)", "실기(바구니·의자·책상)", "서버 설정 1줄 수준"],
          ]),
          H("판단 전에 정해야 할 것"),
          tbl(["항목", "현재", "필요한 결정"], [["성공 기준", "없음(운영자 수동 판정)", "접촉을 성공으로 볼지, 도착 거리 기준은 무엇인지"], ["맹검", "11/13은 비맹검, 11/16은 미확인", "운영자에게 체크포인트를 숨길 수 있는지"], ["접촉 기록", "구두 관찰만(CSV 미반영)", "시험마다 접촉/비접촉을 기록하는 열 추가"], ["지시어 형태", "11/13과 같은 문구 고정", "유지"]]),
          P("<b>제안(근거 수준 낮음):</b> 접촉을 기록하면서 a3-s2의 바구니·의자 재시험을 먼저 해서 문제가 재현되는지 보고, 재현되면 C 중 하나를 오프라인 비교 후 선택. 이 제안은 판단 보조일 뿐이며 결정은 사용자의 몫이다.")]

    # ── t6 근거
    t6 = [H("근거 파일과 링크"),
          tbl(["내용", "위치"], [["11/16 a3-s2 실기 45세션(로봇 서버 보고)", "<code>docs/v5/egvla2_fm/a3c/real_1116_*</code>, 원본 <code>_received_1116_eg2_14/</code>(미커밋)"], ["a3 vs a3-s2 비교 계산", "<code>real_1116_vs_1113.json</code> (<code>scripts/a3c_real_1116_analysis.py</code>)"], ["시뮬레이션(재생) 값", "<code>ch82_sim.json</code> (<code>scripts/ch82_compute.py</code>)"],
                      ["학습 에피소드 정지 면적", "<code>ch82_train_stoparea.json</code> (<code>scripts/ch82_train_stoparea.py</code>)"], ["시드·가림 비율 비교, 후보 검증, 방향 차이", "<code>eval_seeds.json</code>, <code>verify_seeds.json</code>, <code>dirdiff_all.json</code>"], ["계획서", "<code>plan_20261115_a3_improvement.md</code> §8~11, <code>plan_20261116_soda_test_protocol.md</code>"]]),
          P(f'페이지: <a href="object_test_compare_1113.html" style="color:{B}">11/13 동시 비교 재생</a> · <a href="object_test_compare_1113_standalone.html" style="color:{B}">단독 실행판(실기 프레임 870장 내장, 14MB, 폴더 없이 열림)</a> · <a href="model_registry.html" style="color:{B}">모델 레지스트리</a> · <a href="research_story.html#ch81" style="color:{B}">CH81 종합 브리핑</a>')]
    return [("t1", "📌 요약·판단", t1), ("t2", "🤖 실기 비교(11/13 vs 11/16)", t2), ("t3", "🧪 시뮬레이션 대조", t3), ("t4", "🖼 프레임 대조", t4), ("t5", "⚖ 선택지·판단", t5), ("t6", "📎 근거·링크", t6)]

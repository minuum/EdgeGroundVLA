"""CH81 탭 ⚖ A3 대비·대체 이유·시뮬 대조·반영 가능성·오차. 원자료: t7_all_seeds.json, t7_replay.json, objtest_data.json, logs/episode_log.csv(성패는 objtest_data의 res)"""
import json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FM = ROOT / "docs/v5/egvla2_fm"
GO = 0.2
P = lambda t: f'<div style="font-size:0.82rem;line-height:1.8;margin:6px 0">{t}</div>'


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h) * 100, min(1, c + h) * 100


def _chart(title, a3, v4r, note):
    W, Hh, L, B = 460, 150, 34, 24
    n = len(a3); x = lambda i: L + (W - L - 10) * (i / max(n - 1, 1)); y = lambda v: 8 + (Hh - B - 8) * (1 - min(v, 1.3) / 1.3)
    pl = lambda s, col: f'<polyline fill="none" stroke="{col}" stroke-width="2" points="' + " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(s)) + '"/>'
    g = f'<line x1="{L}" y1="{y(GO):.1f}" x2="{W - 10}" y2="{y(GO):.1f}" stroke="#f87171" stroke-dasharray="4 3"/><text x="{W - 12}" y="{y(GO) - 3:.1f}" text-anchor="end" fill="#f87171" font-size="9">정지 기준 0.2</text>'
    g += f'<line x1="{L}" y1="{y(0):.1f}" x2="{W - 10}" y2="{y(0):.1f}" stroke="#334155"/><text x="{L - 4}" y="{y(0) + 3:.1f}" text-anchor="end" fill="#64748b" font-size="9">0</text><text x="{L - 4}" y="{y(1.0) + 3:.1f}" text-anchor="end" fill="#64748b" font-size="9">1.0</text>'
    g += f'<text x="{L}" y="{Hh - 8}" fill="#94a3b8" font-size="9.5">프레임 0 → {n - 1}</text>'
    g += f'<text x="{W - 10}" y="{Hh - 8}" text-anchor="end" font-size="9.5"><tspan fill="#38bdf8">━ A3 기록(실기)</tspan>  <tspan fill="#4ade80">━ V4R 재생(서빙 설계, 10시드 평균)</tspan></text>'
    svg = f'<svg viewBox="0 0 {W} {Hh}" style="width:100%;height:auto;background:#0a1628;border:1px solid #1e3050;border-radius:8px">{g}{pl(a3, "#38bdf8")}{pl(v4r, "#4ade80")}</svg>'
    return f'<div><div style="font-size:0.78rem;font-weight:800;color:#fcd34d;margin:4px 0">{title}</div>{svg}<div style="font-size:0.72rem;color:#94a3b8;line-height:1.55">{note}</div></div>'


def tab(tbl, H, st):
    G, Y, R = "#4ade80", "#fbbf24", "#f87171"
    S = json.loads((FM / "t7_all_seeds.json").read_text()); T = json.loads((FM / "t7_replay.json").read_text())
    D = {s["sid"]: s for s in json.loads((FM / "objtest_data.json").read_text())}
    m = lambda k, key: S[k][key]["mean"]; sd = lambda k, key: S[k][key]["std"]; pct = lambda v, nd=1: f"{v * 100:.{nd}f}%"
    a3self = T["mirror"]["a3_lat_agree"]

    # ① 대체 이유
    t1 = [H("1. 왜 A3만으로는 부족한가 — 대체(보완) 이유와 한계"),
          tbl(["요구·상황", "A3 (EG2-09)", "V4", "V4R", "근거 수준"], [
              ["지시어를 보고 목표가 아니면 정지", "불가 — 헤드에 텍스트 입력이 없고 박스가 곧 목표", f"{pct(m('V4', 'neg_stop'))} 정지", f"{pct(m('V4R', 'neg_stop'))} 정지", "오프라인(실기 프레임 + 가짜 문구). 실기 미검증"],
              ["박스가 반대편일 때 좌우 방향 (합성 반전)", f"일치 {pct(a3self)} — 박스를 그대로 따라 뒤집힘", f"{pct(m('V4', 'm_lat'))} (최소 {pct(S['V4']['m_lat']['min'], 0)})", f"{pct(m('V4R', 'm_lat'))} (최소 {pct(S['V4R']['m_lat']['min'], 0)})", "합성 조건(정상 이미지에 박스만 뒤집음). 실제 오검출 분포와 다름"],
              ["정상 장면에서 A3 명령 재현", "(기준 자신)", f"각도 차 {m('V4', 'ang'):.2f}° · 좌우 {pct(m('V4', 'lat'))}", f"각도 차 {m('V4R', 'ang'):.2f}° · 좌우 {pct(m('V4R', 'lat'))}", "A3가 낸 명령과의 유사도. 품질 아님"],
              ["정상 지시에서 오정지(낮을수록 좋음)", "해당 없음", pct(m("V4", "pos_stop")), f"{pct(m('V4R', 'pos_stop'))} (최대 {pct(S['V4R']['pos_stop']['max'])})", "대체 시 비용. 시드별 편차"],
              ["지시어로 다른 물체에 가기", "박스가 그 물체를 잡으면 감 — 11/13 옆 위치 39/40", "<b style='color:#f87171'>못 감</b> — 박스 고정, 비바구니 문구는 정지", "<b style='color:#f87171'>못 감</b> — 동일", "11/13 재생: 의자 0%, 책상 13~24%, 화분 5~15% 이동"]]),
          P("<b>정리</b>: V4/V4R로 바꾸는 이유는 <b>'지시어를 이해해 목표가 아니면 멈추고, 박스가 잘못 잡혀도 덜 휘둘리는' 텍스트 조건화 E2E</b>를 만들기 위해서다. "
            "<b>'지시어로 새 물체까지 가는 것'은 V4/V4R이 하지 못한다</b> — 그 능력은 11/13에서 A3 + soda의 지시어 박스(그라운딩) 조합이 이미 보여 준 것이다. 즉 두 모델은 대체 관계가 아니라 <b>역할이 다르다</b>. "
            "교수님의 '3물체를 지시어로 검증' 요청에 V4를 같은 방식으로 돌리면, 서빙 설계상 비바구니 지시에는 정지가 나오는 것이 정상이다(실패가 아니라 설계된 동작)."),
          ]

    # ② A3 실기 vs 재생
    g = defaultdict(list)
    for s in D.values():
        sd_ = "가운데" if "center" in s["route"] else ("옆" if "weak" in s["route"] else "기록 없음")
        g[(s["obj"], sd_)].append(s)
    rows = []
    for (o, sdn), ss in sorted(g.items()):
        if sdn == "기록 없음":
            continue
        n = len(ss); k = sum(x["res"] == "성공" for x in ss); lo, hi = wilson(k, n)
        a3go = np.mean([x["go"] for x in ss]); v4r = np.mean([1 - np.mean(x["V4R_ins_gb_stop"]) for x in ss]); v4 = np.mean([1 - np.mean(x["V4_ins_gb_stop"]) for x in ss])
        ok = (k / n >= 0.5) == (v4r >= 0.5)
        rows.append([o, sdn, f"{k}/{n}", f"{lo:.0f}~{hi:.0f}%", pct(a3go, 0), pct(v4, 0), pct(v4r, 0), st("부합", G) if ok else st("어긋남", R)])
    t2 = [H("2. A3 실기 결과와 시뮬레이션(재생) 대조 — 11/13, 물체·위치별"),
          tbl(["물체", "위치", "A3 실기 성공", "95% 구간(Wilson)", "A3 이동 프레임", "V4 재생 이동", "V4R 재생 이동", "성공 쪽 일치(단순)"], rows),
          P("'이동 프레임'은 명령 크기가 정지 기준(0.2) 이상인 프레임 비율이다. A3 열은 <b>실기에서 기록된 값</b>, V4·V4R 열은 서빙 설계 입력(gray basket 박스 + 지시어 문구 특징)으로 <b>재생한 값</b>. "
            "'부합'은 실기 성공률 ≥50%와 재생 이동 ≥50%가 같은 쪽인지 본 단순 판정이며 모델 성능 판단이 아니다. <b>화분·가운데의 '부합'은 우연에 가깝다</b>: A3는 81% 이동하고도 실패했고 V4R은 지시어 때문에 멈춘다(이유가 다름). "
            "<b>어긋남이 나오는 곳(의자·옆, 책상)은 오류가 아니라 설계 차이</b>다: A3는 지시어 박스가 의자를 잡아 갔고, V4R은 박스가 바구니에 고정돼 지시어 때문에 멈춘다.")]
    charts = ""
    sel = [("20261113_165500", "바구니·가운데 (A3 성공)"), ("20261113_154226", "의자·옆 (A3 성공)"), ("20261113_170121", "의자·가운데 (A3 실패)"), ("20261113_151741", "화분·가운데 (A3 실패)")]
    notes = {"20261113_165500": "둘 다 이동 기준(0.2) 이상이다. 크기는 A3 약 1.15, V4R 약 0.5로 다르다(속도 후처리 차이 가능, 미확인).", "20261113_154226": "A3는 이동했지만 V4R 재생은 정지(지시어가 의자라서). 설계된 차이.",
             "20261113_170121": "A3 명령이 처음부터 0. V4R도 정지 — 이 세션은 어느 쪽으로도 못 간다.", "20261113_151741": "A3는 둘째 프레임에서 한 번 멈춘 뒤 끝까지 이동했는데도 실패 판정이다 — 이 실패는 출발 문제가 아니다(원인 미확정). V4R은 대부분 정지."}
    for sid, lab in sel:
        s = D[sid]
        a = [float(np.linalg.norm(v[:2])) for v in s["a3"]]; v = [float(np.linalg.norm(x[:2])) for x in s["V4R_ins_gb"]]
        charts += _chart(lab, a, v, notes[sid])
    t2 += [H("2-1. 프레임별 명령 크기: A3 기록 vs V4R 재생"), f'<div style="display:grid;grid-template-columns:repeat(2,1fr);gap:10px;margin:8px 0">{charts}</div>',
           P("선이 겹치면 재생이 A3와 같은 판단을 한 것이다. 이 그래프는 세션 4개의 대표 사례이며 65건 전체 평균이 아니다.")]

    # ③ 반영 가능성
    t3 = [H("3. 시뮬레이션 결과를 실기 판단에 반영해도 되나 — 항목별"),
          tbl(["주장", "반영", "이유"], [
              ["V4R이 바구니 주행에서 A3에 가깝다", st("조건부 가능", Y), f"정상 장면 A3 명령과 각도 차 {m('V4R', 'ang'):.2f}°(452세션 5,845프레임, 10시드). 단 첫 스텝 비교이고 닫힌고리 성공 여부는 별개. 실기 시험 필요"],
              ["박스 오검출에 V4R이 더 강건", st("조건부 가능", Y), f"합성 반전에서 좌우 일치 {pct(m('V4R', 'm_lat'))}(A3 {pct(a3self)}). 실제 오검출 형태는 다르므로 실기 확인 필요"],
              ["재생 결과로 11/13 실기 성공률을 예측", st("불가", R), "열린고리(오차가 누적·보정되지 않음) + soda 후처리 규칙(near_area 0.25, lost 5 등) 미반영 + 재생 입력(gray 박스)이 비바구니 실기 입력(지시어 박스)과 다름"],
              ["V4/V4R로 의자·책상·화분에 성공", st("근거 반대", R), "재생상 의자 0%, 책상 13~24%, 화분 5~15% 이동. 설계상 비바구니 문구는 정지가 정상"],
              ["V4R이 가운데 실패(의자 0/5, 화분 5/13)를 해결", st("근거 없음", R), "가운데 위치의 의자는 V4R도 0% 이동. 화분은 재생상 대부분 정지. 해결 증거가 없다"],
              ["서빙 경로가 학습 경로와 동등", st("가능(바구니 기준)", G), "문구 특징 코사인 ≥0.9998, 각도 중앙값 0.08°/0.07°, 정지 일치 100%. 비바구니 장면 동등성은 비교 불가"]]),
          ]

    # ④ 오차
    A = T["a3_replay_vs_stored"]
    lo1, hi1 = wilson(49, 65); lo2, hi2 = wilson(39, 40); lo3, hi3 = wilson(10, 24); lo4, hi4 = wilson(455, 455)
    t4 = [H("4. 오차와 불확실성 — 크기와 영향"),
          tbl(["원인", "크기", "영향·주의"], [
              ["실기 표본 크기", f"49/65: {lo1:.0f}~{hi1:.0f}% · 옆 39/40: {lo2:.0f}~{hi2:.0f}% · 가운데 10/24: {lo3:.0f}~{hi3:.0f}% · A3 전체 455/455: {lo4:.1f}~{hi4:.0f}%", "의자 5건·책상 가운데 1건 등은 구간이 매우 넓다. 물체·위치 비교는 방향만 참고"],
              ["재생 환경 충실도(오차 하한)", f"A3 재생 vs 기록 명령: 각도 평균 {A['ang_mean']:.2f}°(90% {A['ang_p90']:.2f}° 이내), 좌우 일치 {pct(A['lat_agree'])}", "재생 도구가 A3를 거의 재현하므로, 그 위의 V4/V4R 비교 수치는 이 오차 이상으로만 의미"],
              ["서빙–학습 경로 차이", "각도 중앙값 0.08°(V4)·0.07°(V4R), JPEG q95, 109프레임", "무시할 수준. 비교 가능한 이동 프레임은 44~45개뿐"],
              ["시드 편차 — V4", f"각도 {m('V4', 'ang'):.2f}±{sd('V4', 'ang'):.2f}° · 반전 좌우 일치 {pct(m('V4', 'm_lat'))}±{sd('V4', 'm_lat') * 100:.0f}%p (최소 {pct(S['V4']['m_lat']['min'], 0)})", "시드에 따라 결과가 크게 달라진다. 서빙은 시드 1(시드 0이 가장 약함)"],
              ["시드 편차 — V4R", f"각도 {m('V4R', 'ang'):.2f}±{sd('V4R', 'ang'):.2f}° · 반전 좌우 일치 {pct(m('V4R', 'm_lat'))}±{sd('V4R', 'm_lat') * 100:.1f}%p", "편차가 작다. 단 정상 오정지 최대 12.9%"],
              ["입력 불일치(재생 vs 실기)", "비바구니 세션 박스 불일치 76~100%(바구니 0%)", "재생 입력이 실제 서빙 입력과 다르므로 비바구니 장면 수치는 동등성 확인 없이 해석 금지"],
              ["합성 반전", "정상 이미지에 박스만 좌우 반전", "실제 오검출 분포와 다름 — 강건성의 상한이 아닌 대략적 지표"],
              ["성공/실패 라벨", "operator 수기, 비맹검, 기준 문서 없음", "경계 사례(책상 1건 실패 등)의 일관성 보장 안 됨"],
              ["미반영 요소", "후처리 규칙·닫힌고리·지연 약 0.9초", "정량 불가. 실기 시험으로만 확인"]]),
          H("5. 한 문단 결론 (교수님·동환 전달용)"),
          P("11/13 실기(A3 + 지시어 박스)는 <b>옆 위치 39/40, 가운데 10/24</b>였고 가운데에서는 의자(0/5)와 화분(5/13)이 실패했다. V4/V4R은 <b>지시어를 이해해 목표가 아니면 멈추고 박스 오검출에 덜 흔들리는</b> 텍스트 조건화 모델이라는 점에서 A3를 보완하지만, "
            "<b>다른 물체로 가는 능력은 없고</b> 가운데 실패를 해결한다는 증거도 아직 없다. 재생(시뮬레이션)은 A3 명령과의 유사도(V4R 약 3.2°, 반전 시 좌우 94%)까지만 말해 주며, "
            "열린고리·후처리 미반영·입력 불일치 때문에 <b>실기 성공률은 예측하지 못한다</b>. 실기 시험(바구니 → 4물체 × 3위치, V4·V4R 시드 1)이 다음 판단 근거다.")]
    return [H("0. 이 탭에서 답하려는 질문"), P("A3(실험에 쓴 모델)와 비교해 V4/V4R로 바꾸려는 이유는 무엇이고, A3로 한 실기 결과와 시뮬레이션을 대조하면 어디까지 반영할 수 있으며 오차는 얼마인가.")] + t1 + t2 + t3 + t4

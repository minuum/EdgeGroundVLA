"""CH77 ⑥ 학습 대조 해설: 패러다임·평가 축·서브골 입력·과적합 대조 + 방향 오차에 미치는 요인별 효과 막대. 수치는 JSON에서 읽는다."""
import json

import ch77_viz as Z
import gen_ch76_viz as V

FM = V.FM
A, G, R, B, P, MUT = V.A, V.G, V.R, V.B, V.P, V.MUT


def J(n):
    return json.load(open(FM / n))


def block(tbl, H, st):
    s100, s250, s300, l300 = J("summary_s1.json")["aggregate"], J("summary_s1_ep250.json")["aggregate"], J("summary_s1_full300.json")["aggregate"], J("summary_s1_full300_lopo.json")["aggregate"]
    ab, sw, hc = J("subgoal_ablation.json"), J("smoothing_sweep.json"), J("human_ceiling.json")["results"]["같은 셀(목표 위치+경로)"]
    ang = lambda a, k: a[k]["angle_deg"]
    mean = lambda key, sel: sum(x[key]["normal"]["angle_deg"] for k, x in ab.items() if sel(k)) / sum(1 for k in ab if sel(k))
    fo, lo = (lambda k: k.startswith("fold")), (lambda k: k.startswith("lopo"))
    eff = [("목표 위치 일반화(같은 위치 → 안 본 위치, C)", ang(l300, "C_pi0_adaln") - ang(s300, "C_pi0_adaln"), R),
           ("K 샘플 1 → 8 평균(같은 위치, 평활 없음)", sw["fold0"]["K1|a0.0"]["angle"] - sw["fold0"]["K8|a0.0"]["angle"], G),
           ("헤드 A → C (안 본 위치)", ang(l300, "A_mlp_fm") - ang(l300, "C_pi0_adaln"), G),
           ("헤드 A → C (같은 위치)", ang(s300, "A_mlp_fm") - ang(s300, "C_pi0_adaln"), G),
           ("학습 데이터 218 → 300건 (C, 250 epoch)", ang(s250, "C_pi0_adaln") - ang(s300, "C_pi0_adaln"), G),
           ("epoch 100 → 250 (A, 218건)", ang(s100, "A_mlp_fm") - ang(s250, "A_mlp_fm"), G),
           ("EMA 0.5 평활(K=8)", sw["fold0"]["K8|a0.0"]["angle"] - sw["fold0"]["K8|a0.5"]["angle"], G),
           ("서브골 없음 → 있음 (B2, 같은 위치, 3 seed)", mean("B2_film_res_fm|nobbox", fo) - mean("B2_film_res_fm|full", fo), B),
           ("서브골 없음 → 있음 (C, 같은 위치, 3 seed)", mean("C_pi0_adaln|nobbox", fo) - mean("C_pi0_adaln|full", fo), B),
           ("서브골 드롭아웃 30% (C, 같은 위치)", mean("C_pi0_adaln|full", fo) - mean("C_pi0_adaln|drop30", fo), B)]
    bars = Z.bars([(l, abs(v), c if v >= 0 else "#475569") for l, v, c in eff], 7, "°", w=300)
    box = lambda t, b: f'<div style="margin:12px 0 4px;font-size:0.84rem;font-weight:800;color:#fcd34d">{t}</div>{b}'
    paradigm = tbl(["", "이산 분류(EGVLA1)", "연속 회귀(MLP 회귀)", "FM 헤드 A / B2 / C"], [
        ["출력", "8-class 중 하나(+속력 고정)", "(lx, ly, az) 청크 H=8 한 개", "같은 청크를 노이즈에서 10스텝으로 샘플링(여러 개)"],
        ["손실", "교차 엔트로피", "MSE", "flow-matching(속도장 MSE)"],
        ["다봉 분포", "클래스로 표현(경계가 거침)", "<b>평균으로 뭉개짐</b>", "<b>이론상 표현 가능</b> — 이 데이터는 단봉이라 이점 안 드러남(⑥-3)"],
        ["부드러움", "계단식(8방향)", "<b>가장 부드러움</b>(저크 0.085)", "샘플 잡음으로 흔들림 큼(저크 0.12) → K 평균+EMA 필요"],
        ["불확실도", "소프트맥스 확률", "없음", "샘플 분산(오차와 약한 상관 ρ 0.1~0.3)"],
        ["서브골 조건", "bbox concat", "bbox concat", "A concat / B2 FiLM / C prefix(AdaLN)"],
        ["정확도(방향 오차, 같은 위치/안 본 위치)", "측정 불가(표현이 다름)", "— (MSE 동급)", f"{ang(s300, 'A_mlp_fm'):.1f}°·{ang(l300, 'A_mlp_fm'):.1f}° / {ang(s300, 'B2_film_res_fm'):.1f}°·{ang(l300, 'B2_film_res_fm'):.1f}° / <b>{ang(s300, 'C_pi0_adaln'):.1f}°·{ang(l300, 'C_pi0_adaln'):.1f}°</b>"],
        ["언제 쓰나", "단순·안전, 계단 궤적 허용", "<b>정확도만 보면 충분</b>, 구현 가장 단순", "연속 샘플링·불확실도·청크 표현이 필요할 때, 비용은 저크·연산(+54ms)"]])
    axes = tbl(["평가 축", "무엇이 달라지나", "수치(C 방향 오차)", "의미", "상태"], [
        ["같은 목표 위치(5-fold)", "학습에 없던 에피소드, 같은 목표 위치·같은 시작 위치", f"{ang(s300, 'C_pi0_adaln'):.1f}°", "사람 변동성 한계(천장 9.8~12.8°)와 같은 수준", st("측정", G)],
        ["안 본 목표 위치(LOPO)", "목표(바구니)가 화면의 다른 위치에 보임", f"{ang(l300, 'C_pi0_adaln'):.1f}°", f"천장 대비 +{ang(l300, 'C_pi0_adaln') - hc['consensus_mean']:.0f}° — 일반화 여지", st("측정", G)],
        ["안 본 시작 위치(pos2·pos3)", "시작을 좌우 50cm 옮김(화면·원근·배경 변화)", "—", "학습은 pos1뿐 — 오프라인으로 첫 측정 예정(CH78 ⑥-10)", st("수집 대기", A)],
        ["실기 폐루프", "모델 출력이 다음 입력을 바꿈, 지연·속력·미끄럼 포함", "—", "성공률은 여기서만 나옴(EGVLA1 92.0%)", st("미측정", R)]])
    sub = tbl(["서브골 입력 방식", "무엇을 넣나", "같은 위치", "안 본 위치", "결측 시(안 본 위치)", "한 줄 평"], [
        [f'<span style="color:{R}">●</span> 없음', "bbox 0(학습·평가 모두)", f'{mean("C_pi0_adaln|nobbox", fo):.1f}°', f'{mean("C_pi0_adaln|nobbox", lo):.1f}°', "—", "이미지 특징에 위치 정보가 이미 있어 정확도 유지"],
        [f'<span style="color:{B}">●</span> 있음(C, 창3)', "현재+이전 2프레임 bbox", f'{mean("C_pi0_adaln|full", fo):.1f}°', f'{mean("C_pi0_adaln|full", lo):.1f}°', f'{sum(x["C_pi0_adaln|full"]["missing"]["angle_deg"] for k, x in ab.items() if lo(k)) / sum(1 for k in ab if lo(k)):.1f}°', "정확도 이득 ≤0.1°"],
        [f'<span style="color:{B}">●</span> 있음(B2, FiLM)', "FiLM으로 이미지 특징 조향", f'{mean("B2_film_res_fm|full", fo):.1f}°', f'{mean("B2_film_res_fm|full", lo):.1f}°', f'{sum(x["B2_film_res_fm|full"]["missing"]["angle_deg"] for k, x in ab.items() if lo(k)) / sum(1 for k in ab if lo(k)):.1f}°', "같은 위치 +0.4° 이득, <b>결측에 크게 취약</b>"],
        [f'<span style="color:{G}">●</span> 드롭아웃 30%(C)', "학습 중 30% 확률로 서브골을 0으로", f'{mean("C_pi0_adaln|drop30", fo):.1f}°', f'{mean("C_pi0_adaln|drop30", lo):.1f}°', f'{sum(x["C_pi0_adaln|drop30"]["missing"]["angle_deg"] for k, x in ab.items() if lo(k)) / sum(1 for k in ab if lo(k)):.1f}°', "<b>손실 없이 결측 견고 → 배포 권장</b>"],
        [f'<span style="color:{P}">●</span> 미래 bbox(π0.7식 단순화)', "8프레임 뒤 bbox를 예측해 조건", "10.3°", "16.0°", "—", "예측은 이득 ≤0.4°, 정답을 줘도 0.6~2.3°(누설 상한)"]])
    cv = J("ablation_curves.json"); FIXE = {"B2_film_res_fm": 220, "C_pi0_adaln": 120, "MLP_regression": 60}
    def inc(h, sps):
        v = []
        for sp in sps:
            c = cv[sp][f"{h}|full"]; a = min(c, key=lambda r: abs(r["ep"] - FIXE[h]))["angle"]; v.append(c[-1]["angle"] - a)
        return f"{min(v):+.1f}~{max(v):+.1f}°"
    over = tbl(["대조", "같은 목표 위치(5-fold)", "안 본 목표 위치(LOPO)", "해석"], [
        ["C의 hold-out 손실 최저 epoch", "약 90~120", "약 20~140(분할마다)", "새 위치일수록 일찍 과적합"],
        ["MLP 회귀 hold-out 손실 최저 epoch", "약 50", "약 5~40", "회귀가 가장 일찍 과적합"],
        ["250 epoch − 고정 epoch 방향 오차(MLP 회귀)", inc("MLP_regression", ["fold0"]), inc("MLP_regression", ["lopo_center", "lopo_strong_right"]), "회귀는 안 본 위치에서 가장 크게 악화"],
        ["250 epoch − 고정 epoch 방향 오차(B2)", inc("B2_film_res_fm", ["fold0"]), inc("B2_film_res_fm", ["lopo_center", "lopo_strong_right"]), "FM 헤드는 악화가 작거나 분할에 따라 다름"],
        ["250 epoch − 고정 epoch 방향 오차(C)", inc("C_pi0_adaln", ["fold0"]), inc("C_pi0_adaln", ["lopo_center", "lopo_strong_right"]), "곡선이 흔들려(±1~3°) 방향은 일정치 않음"],
        ["배포 epoch 결정", "120(C)", "—", "안 본 위치 기준 최적과 같은 범위(약 80~160)"]])
    return (H("⑥ 학습 분야 대조 해설 — 무엇이 성능을 좌우했나")
            + '<div style="font-size:0.8rem;line-height:1.6;margin:6px 0">이 탭은 학습 분야의 결과를 <b>무엇과 무엇을 비교했는가</b> 중심으로 다시 풀어 쓴 것이다. 먼저 어떤 요인이 방향 오차를 얼마나 바꿨는지 한 장으로 보고, 이어서 학습 방식·평가 축·서브골 입력·과적합을 짝으로 대조한다.</div>'
            + box("방향 오차를 바꾼 요인별 효과 (°, 막대가 길수록 영향이 큼)", bars + '<div style="font-size:0.72rem;color:#64748b;margin-top:4px">빨강 = 악화 요인(위치 일반화), 초록 = 개선 요인, 파랑 = 서브골. 사람 조작 천장은 약 10~13°.</div>')
            + f'{st("핵심", R)} <b>안 본 목표 위치로 일반화할 때의 악화(약 6.5°)가 다른 모든 요인의 효과(대부분 1° 안팎)를 합친 것보다 크다.</b> 개선 노력의 우선순위는 헤드·서브골 튜닝이 아니라 위치 일반화와 평활이다.'
            + box("① 학습 방식 대조 — 이산 분류 / 연속 회귀 / FM 헤드", paradigm) + box("② 평가 축 대조 — 어디까지 일반화되는가", axes)
            + box("③ 서브골 입력 대조 — 방향 오차(°)", sub) + box("④ 과적합 대조 — 같은 위치 vs 안 본 위치", over))

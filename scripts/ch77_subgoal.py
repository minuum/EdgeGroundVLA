"""CH77: 서브골 형태 비교(논문별 서브골 형태 + 우리 ablation·미래 서브골 실험). 데이터: subgoal_ablation.json, subgoal_future.json (있는 분할만 집계)."""
import json

import numpy as np

import gen_ch76_viz as V

FM = V.FM
A, G, R, B, P, MUT = V.A, V.G, V.R, V.B, V.P, V.MUT
HN = {"B2_film_res_fm": "B2", "C_pi0_adaln": "C"}
VN = {"full": "서브골 있음(창3, 기준)", "nobbox": "서브골 없음", "w1": "현재 프레임만(창1)", "w5": "창 5", "drop30": "드롭아웃 30%로 학습",
      "fut_oracle": "미래 bbox 정답 입력(누설 상한)", "fut_pred": "미래 bbox 생성기 예측(π0.7식 단순화)"}


def agg(d, key, cond, metric):
    v = [x[key][cond][metric] for s, x in d.items() if key in x]
    return (float(np.mean(v)), len(v)) if v else (None, 0)


def block(tbl, H, st):
    fa, ff = FM / "subgoal_ablation.json", FM / "subgoal_future.json"
    lit = tbl(["서브골 형태", "대표 사례(원문 확인 2026-10-20)", "서브골의 정체", "우리와의 관계"], [
        ["생성 이미지", "π0.7 (arXiv 2604.15483, 2026-04-16)", "BAGEL 기반 경량 월드모델이 서브태스크 지시에서 근미래 서브골 이미지를 생성, flow-matching action expert를 가진 정책이 이를 <b>프롬프트 조건으로 입력</b>(“조향”보다 “조건화”가 정확)", "서브골 조건화 아이디어의 출처. 이미지 생성은 하지 않고 미래 bbox 예측으로 단순화해 시험(이득 없음)"],
        ["중간 수준 언어 행동", "NaVILA (2412.04453)", "상위 VLM이 “전진 75cm” 같은 언어 행동을 내고 저수준 시각 보행 RL 정책이 실행", "텍스트 서브골은 우리 실험에서 이득 없음(CLIP −1.41%p, one-hot ±0)"],
        ["공간 지점", "Gemini Robotics-ER 1.5 (2510.03342)", "점을 경량 표현·중간 추론 도구·경로 표현으로 서술. <b>서브골 표현으로 쓴다는 명시 문서는 확인 못 함</b>", "<b>우리 bbox와 가장 가까운 형태</b>(부분 확인)"],
        ["잠재 의미 인터페이스", "TIC-VLA (2602.02459)", "지연된 VLM의 KV 캐시 특징을 행동 전문가가 cross-attention으로 읽고, 지연 시간 Δt와 지연 중 이동량을 명시 입력", "우리는 단일 호출이라 구조는 다르나 지연 문제는 공통(서버 180ms vs 젯슨 817ms)"],
        ["목표 프레임·위상 그래프", "MobilityVLA (2407.07775)", "장문맥 VLM이 투어 영상에서 <b>목표 프레임 인덱스</b>를 고르고 위상 그래프 위 저수준 정책이 웨이포인트 생성(프레임=노드)", "목표 위치 표현 방식이 다름(지도 없음)"]])
    met = tbl(["논문", "SR 정의", "CR 정의", "실기 시행 수", "시뮬레이션 병행"], [
        ["NaVILA", "VLN-CE 표준, 실기는 SR+NE", "—", "25 지시 × 3회 = 75회(Go2), 전체 88%(복잡 지시 75%)", "있음(R2R/RxR, VLN-CE-Isaac)"],
        ["TIC-VLA", "목표 1.5m 이내 정지", "에피소드 중 <b>1회 이상 충돌</b>한 에피소드 비율", "4과제 × 5회(Go2), 구성별 SR 0.70~0.85. DynaNav SR 55.29 / CR 28.24", "있음(DynaNav)"],
        ["CrossTracer", "임계 거리 이내, 충돌·수동 개입 없음", "—", "방법당 40회(4과제×5회×2로봇)", "NaviTrace 병행"],
        ["VEGA", "임계 거리 이내 시행 비율", "벤치: 궤적이 점유·미지 영역 진입 / 실기: 접촉·안전 개입 시행 비율", "방법당 15회(3시나리오×5회)", "있음(VEGA-Bench)"],
        ["MM-Nav·MobilityVLA·OmniVLA", "논문별", "논문별", "총 시행 수 확인 못 함", "논문별"],
        ["<b>EGVLA1 / EGVLA2(우리)</b>", "기존 프로토콜(EGVLA1과 동일 정의)", "충돌·개입 횟수 기록 예정", "<b>300회</b> (3 시작 위치 × 5 목표 위치 × 20), EGVLA1은 276/300=92.0%", "없음(서봇2 모델 부재)"]])
    out = (H("⑥-5 서브골 형태 비교 — 논문들의 방식과 우리 bbox 서브골의 위치")
           + '<div style="font-size:0.8rem;line-height:1.6;margin:6px 0"><b>EGVLA1의 bbox와 같은 신호</b>다. EGVLA1은 OWLv2 bbox를 이미지 특징과 concat해 MLP에 넣었고(bbox_scale 무영향), EGVLA2는 같은 bbox를 “서브골”로 보고 FiLM·prefix로 <b>조향</b>한다(V6 steered 78.41 vs concat 76.76). '
              '“서브골”이라는 이름은 π0.7의 구조 아이디어를 따라 붙인 것이다. 논문 정보는 원문(arXiv·공식 페이지)으로 대조했다(2026-10-20, <code>docs/plans/research_20261020_subgoal_paper_check.md</code>). 확인하지 못한 항목은 표에 그렇게 적었다.</div>' + lit + '<div style="margin:10px 0 4px;font-size:0.82rem;font-weight:800;color:#fcd34d">평가 지표·실기 규모 비교 — 정의가 논문마다 달라 숫자를 한 열에서 직접 비교하지 말 것</div>' + met)
    import ch77_viz as Z
    fo, lo = (lambda k: k.startswith("fold")), (lambda k: k.startswith("lopo"))
    HB = [("B2_film_res_fm", "B2"), ("C_pi0_adaln", "C")]
    hc = [Z.HC["B2"], Z.HC["C"]]
    colored = lambda t, c: f'<span style="color:{c}">●</span> {t}'
    if fa.exists():
        d = json.load(open(fa))
        ag = lambda key, sel, cond="normal": np.mean([x[key][cond]["angle_deg"] for k, x in d.items() if sel(k) and key in x])
        VR = [("full", "서브골 있음(창3, 기준)", B), ("nobbox", "서브골 없음", R), ("w1", "현재 프레임만(창1)", MUT), ("w5", "창 5", MUT), ("drop30", "드롭아웃 30%로 학습", G)]
        rows = []
        for v, lab, c in VR:
            rows.append((colored(lab, c), [[ag(f"{h}|{v}", fo) for h, _ in HB], [ag(f"{h}|{v}", lo) for h, _ in HB],
                                        [("—" if v == "nobbox" else ag(f"{h}|{v}", lo, "missing")) for h, _ in HB]], "low"))
        out += (H("서브골 ablation — 30분할 (5-fold 5 + 목표 위치별 LOPO 5, seed 3개 평균)")
                + '<div style="font-size:0.8rem;margin:4px 0">방향 오차(°). 초록=그 열에서 가장 낮은 변형, 빨강=가장 높은 변형(열 단위 비교). 같은 열끼리만 비교한다.</div>'
                + Z.LEG
                + Z.heat_table([("같은 목표 위치 (5-fold)", ["B2", "C"], hc), ("안 본 목표 위치 (LOPO)", ["B2", "C"], hc), ("서브골 결측 입력 (LOPO)", ["B2", "C"], hc)], rows, fmt="{:.1f}°", corner="변형", by="col",
                                note="서브골 결측 = 평가 때 bbox 입력을 전부 0으로. 서브골 없음은 학습·평가 모두 0이라 해당 없음(—).")
                + '<div style="font-size:0.78rem;color:#94a3b8">핵심(3 seed): ① 서브골을 빼도 같은 목표 위치에서 B2는 +0.4°(15쌍 모두 악화), C는 +0.1°로 효과가 작고, 안 본 목표 위치에서는 B2가 오히려 0.8° 낫다(7/15쌍 악화 = 잡음 수준) ② 서브골로 학습한 B2는 결측에 크게 취약(5-fold 17.8°, 안 본 위치 24.7°) ③ 드롭아웃30%로 학습한 C는 <b>정상 입력에서 손실 없이(−0.2°, −0.4°) 결측에서도 거의 같다</b>(10.1°, 16.8°).</div>')
    if ff.exists():
        f2 = json.load(open(ff)); fg = [x["generator"] for x in f2.values()]
        ga = lambda key, sel: np.mean([x[key]["normal"]["angle_deg"] for k, x in f2.items() if sel(k)])
        rows = [(colored("서브골 없음", R), [[ag(f"{h}|nobbox", fo) for h, _ in HB], [ag(f"{h}|nobbox", lo) for h, _ in HB]], "low"),
                (colored("현재 bbox만(기준)", B), [[ag(f"{h}|full", fo) for h, _ in HB], [ag(f"{h}|full", lo) for h, _ in HB]], "low"),
                (colored("미래 bbox 생성기 예측(π0.7식 단순화)", P), [[ga(f"{h}|fut_pred", fo) for h, _ in HB], [ga(f"{h}|fut_pred", lo) for h, _ in HB]], "low"),
                (colored("미래 bbox 정답(누설 상한)", A), [[ga(f"{h}|fut_oracle", fo) for h, _ in HB], [ga(f"{h}|fut_oracle", lo) for h, _ in HB]], "low")]
        out += (H("π0.7식 미래 서브골 단순화 (K=8프레임 뒤 bbox, 10분할 seed 0)")
                + Z.LEG + Z.heat_table([("같은 목표 위치 (5-fold)", ["B2", "C"], hc), ("안 본 목표 위치 (LOPO)", ["B2", "C"], hc)], rows, fmt="{:.1f}°", corner="서브골 형태", by="col")
                + f'<div style="font-size:0.8rem;margin:4px 0">생성기 품질(미래 bbox 예측 MSE): 5-fold {np.mean([x["gen_mse"] for k, x in f2.items() if fo(k) for x in [x["generator"]]]):.4f} vs 현재 유지 기준선 '
                  f'{np.mean([x["persist_mse"] for k, y in f2.items() if fo(k) for x in [y["generator"]]]):.4f} / LOPO {np.mean([y["generator"]["gen_mse"] for k, y in f2.items() if lo(k)]):.4f} vs {np.mean([y["generator"]["persist_mse"] for k, y in f2.items() if lo(k)]):.4f}(기준선과 같음 = 안 본 위치에서 일반화 못 함). '
                  '정답 미래 bbox는 로봇이 움직인 결과가 새어 들어가는 <b>누설 상한</b>이라 공정한 비교가 아니다.</div>')
    return out + f'{st("읽는 법", A)} 방향 오차 차이가 1° 미만이면 seed 변동(±0.3°)과 구분이 어렵다. 서브골 결측 입력은 실기에서 그라운딩이 실패했을 때를 가정한 스트레스 테스트다. 판단은 실기 폐루프에서.'

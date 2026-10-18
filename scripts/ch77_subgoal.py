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
    lit = tbl(["서브골 형태", "대표 사례", "서브골의 정체", "우리와의 관계"], [
        ["생성 이미지", "π0.7", "월드모델이 만든 미래 목표 이미지가 정책을 조향", "bbox를 서브골로 보는 아이디어의 출처. 이미지 생성은 하지 않음(미래 bbox 예측으로 단순화해 시험)"],
        ["중간 수준 언어 행동", "NaVILA", "고수준 VLM이 '전진 75cm' 같은 언어 행동을 내고 저수준 정책 실행", "텍스트 서브골은 우리 실험에서 이득 없음(CLIP −1.41%p, one-hot ±0)"],
        ["공간 지점·박스", "Gemini Robotics-ER 계열(지점 지정)", "이미지 위의 위치가 중간 표현", "<b>우리 bbox와 가장 가까움</b>"],
        ["잠재 의미 인터페이스", "TIC-VLA", "느린 추론의 KV 캐시를 빠른 제어가 읽음, 지연 정보 명시 입력", "우리는 단일 호출이라 구조는 다르나 지연 문제는 공통"],
        ["위상 그래프 목표 노드", "MobilityVLA", "장기 맥락 VLM이 목표 노드를 고르고 그래프로 이동", "목표 위치 표현 방식이 다름(지도 없음)"]])
    out = (H("⑥-5 서브골 형태 비교 — 논문들의 방식과 우리 bbox 서브골의 위치")
           + '<div style="font-size:0.8rem;line-height:1.6;margin:6px 0"><b>EGVLA1의 bbox와 같은 신호</b>다. EGVLA1은 OWLv2 bbox를 이미지 특징과 concat해 MLP에 넣었고(bbox_scale 무영향), EGVLA2는 같은 bbox를 “서브골”로 보고 FiLM·prefix로 <b>조향</b>한다(V6 steered 78.41 vs concat 76.76). '
              '“서브골”이라는 이름은 π0.7의 구조 아이디어를 따라 붙인 것이다. 논문 정보는 레포 스터디 노트와 검색 요약 기준이며 원문 대조 전이다.</div>' + lit)
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
        out += (H("서브골 ablation — 10분할 (5-fold 5 + 목표 위치별 LOPO 5, seed 0)")
                + '<div style="font-size:0.8rem;margin:4px 0">방향 오차(°). 초록=그 열에서 가장 낮은 변형, 빨강=가장 높은 변형(열 단위 비교). 같은 열끼리만 비교한다.</div>'
                + Z.LEG
                + Z.heat_table([("같은 목표 위치 (5-fold)", ["B2", "C"], hc), ("안 본 목표 위치 (LOPO)", ["B2", "C"], hc), ("서브골 결측 입력 (LOPO)", ["B2", "C"], hc)], rows, fmt="{:.1f}°", corner="변형", by="col",
                                note="서브골 결측 = 평가 때 bbox 입력을 전부 0으로. 서브골 없음은 학습·평가 모두 0이라 해당 없음(—).")
                + '<div style="font-size:0.78rem;color:#94a3b8">핵심: ① 서브골을 빼도 같은 목표 위치에서는 거의 같다 ② 서브골로 학습한 B2는 결측에 크게 취약(25.9°) ③ 드롭아웃30%로 학습한 C는 결측에서도 거의 손실이 없다(16.2°).</div>')
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

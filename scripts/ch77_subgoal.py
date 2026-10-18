"""CH77: 서브골 형태 비교(논문별 서브골 형태 + 우리 ablation·미래 서브골 실험). 데이터: subgoal_ablation.json, subgoal_future.json (있는 분할만 집계)."""
import json

import numpy as np

import gen_ch76_viz as V

FM = V.FM
A, G, R, B, P = V.A, V.G, V.R, V.B, V.P
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
    if fa.exists():
        d = json.load(open(fa)); rows = []
        n_sp = len(d)
        for v in ("full", "nobbox", "w1", "w5", "drop30"):
            for h, hn in HN.items():
                a, n1 = agg(d, f"{h}|{v}", "normal", "angle_deg"); m, _ = agg(d, f"{h}|{v}", "missing", "angle_deg")
                f1, _ = agg(d, f"{h}|{v}", "normal", "go_f1"); ms, _ = agg(d, f"{h}|{v}", "normal", "mse_xy")
                if a is not None:
                    rows.append([VN[v], hn, f"{a:.1f}°", f"{m:.1f}°" if v != "nobbox" else "—", f"{f1:.3f}", f"{ms:.4f}"])
        out += (H(f"서브골 ablation (분할 {n_sp}개 집계, seed 0)") + tbl(["변형", "헤드", "방향 오차(서브골 정상)", "방향 오차(서브골 결측 입력)", "GO/STOP F1", "청크 MSE(xy)"], rows)
                + '<div style="font-size:0.78rem;color:#94a3b8">분할은 5-fold + 목표 위치별 LOPO를 섞어 평균한 값. "서브골 없음"은 학습·평가 모두 서브골 0.</div>')
    if ff.exists():
        d = json.load(open(ff)); rows = []
        gq = [x["generator"] for x in d.values()]
        for v in ("fut_oracle", "fut_pred"):
            for h, hn in HN.items():
                a, n1 = agg(d, f"{h}|{v}", "normal", "angle_deg"); ms, _ = agg(d, f"{h}|{v}", "normal", "mse_xy")
                if a is not None:
                    rows.append([VN[v], hn, f"{a:.1f}°", f"{ms:.4f}"])
        out += (H(f"π0.7식 미래 서브골 단순화 (분할 {len(d)}개 집계, K=8프레임 뒤 bbox)") + tbl(["변형", "헤드", "방향 오차", "청크 MSE(xy)"], rows)
                + f'<div style="font-size:0.8rem;margin:4px 0">생성기 품질: 미래 bbox 예측 MSE {np.mean([g["gen_mse"] for g in gq]):.4f} vs "현재 bbox 유지" 기준선 {np.mean([g["persist_mse"] for g in gq]):.4f}. '
                  '정답 미래 bbox 입력은 로봇이 움직인 결과가 새어 들어가는 <b>누설 상한</b>이라 공정한 비교가 아니다.</div>')
    return out + f'{st("읽는 법", A)} 방향 오차 차이가 1° 미만이면 seed 변동(±0.3°)과 구분이 어렵다. 서브골 결측 입력은 실기에서 그라운딩이 실패했을 때를 가정한 스트레스 테스트다. 판단은 실기 폐루프에서.'

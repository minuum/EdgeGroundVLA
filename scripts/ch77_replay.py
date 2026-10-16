"""CH77: 리플레이 시뮬레이션(오프라인 궤적 적분) EGVLA1식 이산 vs 연속 누적 위치 오차. 데이터: replay_sim_cont.json"""
import json

import gen_ch76_viz as V

FM = V.FM
A, G, R = V.A, V.G, V.R
VAR = {"cont_shadow": "연속 + 속력 0.7 고정(섀도)", "cont_model": "연속 + 모델 속력", "disc": "이산화(8방향 반올림 + 0.7 고정, EGVLA1식)", "gt_disc": "정답 라벨만 이산화(예측 무관)"}
SP = {"fold0": "같은 장소(fold0)", "lopo_strong_right": "안 본 위치(strong_right)"}


def block(tbl, H, st):
    f = FM / "replay_sim_cont.json"
    if not f.exists():
        return ""
    d = json.load(open(f)); rows = []
    for s, v in d.items():
        for m, r in v.items():
            for k, x in r.items():
                if k == "gt_disc" and m != "C_pi0_adaln":
                    continue
                rows.append([SP[s], "C (FM)" if m.startswith("C") else "MLP 회귀", VAR[k], f"{x['fpe_cm']:.0f} (중앙 {x['fpe_median']:.0f})", f"{x['ade_cm']:.0f}", f"{x['lateral_err_cm']:.0f}", f"{x['path_len_diff_cm']:+.0f}"])
    c0, c1 = d["fold0"]["C_pi0_adaln"], d["lopo_strong_right"]["C_pi0_adaln"]
    return (H("⑥-4 리플레이 시뮬레이션 — 이산 vs 연속 누적 위치 오차 (EGVLA1식 액션 표현 비교)")
            + '<div style="font-size:0.8rem;line-height:1.6;margin:6px 0">안 본 60개 에피소드의 기록 프레임마다 모델 예측(첫 스텝)을 내고, <b>예측 이동을 적분</b>해 사람이 실제로 간 궤적과 비교했다. '
              '같은 지각(Florence-2)·같은 헤드에서 <b>액션 표현만</b> 바꾼다(연속 vs 8방향 반올림). 단위는 cm, 에피소드 평균(FPE=종점 오차, ADE=경로 평균 오차, 좌우=종점 좌우 오차, 길이차=예측−사람 경로 길이).</div>'
            + tbl(["평가 세트", "헤드", "액션 표현", "FPE", "ADE", "좌우", "길이차"], rows)
            + f'{st("읽는 법", A)}<ul style="margin:6px 0 6px 18px;font-size:0.8rem">'
              f'<li><b>예측을 이산화해도 오차가 거의 안 늘어난다</b>: C 기준 FPE {c0["cont_shadow"]["fpe_cm"]:.0f}→{c0["disc"]["fpe_cm"]:.0f}cm(같은 장소), {c1["cont_shadow"]["fpe_cm"]:.0f}→{c1["disc"]["fpe_cm"]:.0f}cm(안 본 위치). 오차의 대부분은 <b>액션 표현이 아니라 모델 예측 오차</b>에서 온다.</li>'
              f'<li>정답 라벨만 이산화해도 종점이 {c0["gt_disc"]["fpe_cm"]:.0f}~{c1["gt_disc"]["fpe_cm"]:.0f}cm 어긋난다 → 이산 표현 자체의 손실 하한. 연속 표현이면 이 손실은 없다(상한이 아니라 표현의 한계).</li>'
              f'<li><b>모델 속력</b>을 쓰면 경로 길이 차이가 크게 줄고(+{c0["cont_shadow"]["path_len_diff_cm"]:.0f}→+{c0["cont_model"]["path_len_diff_cm"]:.0f}cm) FPE도 낮다({c0["cont_shadow"]["fpe_cm"]:.0f}→{c0["cont_model"]["fpe_cm"]:.0f}cm). 단 사람 속력에 맞추는 것이라 <b>첫 적용은 섀도(0.7 고정)</b> 방침과는 별개로 속력 모델링의 잠재 이득을 보여 준다.</li>'
              f'<li><b>안 본 위치는 오차가 크다</b>(FPE {c1["cont_shadow"]["fpe_cm"]:.0f}cm, 대부분 좌우 {c1["cont_shadow"]["lateral_err_cm"]:.0f}cm). 보정 없는 열린 고리 적분이라 실제 폐루프에서는 줄어들 수 있으나, 이 위치(strong_right)는 EGVLA1 실기에서도 가장 낮았던 곳(85%)이다.</li>'
              f'<li>같은 입력에서 <b>C가 MLP 회귀보다 궤적 오차가 낮다</b>(안 본 위치 FPE {c1["cont_shadow"]["fpe_cm"]:.0f} vs {d["lopo_strong_right"]["MLP_regression"]["cont_shadow"]["fpe_cm"]:.0f}cm). 프레임 MSE는 비슷했는데 누적하면 차이가 난다.</li></ul>'
            + f'{st("한계", R)} EGVLA1의 실제 모델(OWLv2+Kosmos-2)이 아니라 <b>같은 지각 위에서 액션 표현만 이산화</b>한 비교다. 영상은 원본이라 예측 오차가 다음 입력을 바꾸지 않는 열린 고리이며, <b>폐루프 성공률이 아니다</b>(사람 궤적과의 일치도). 회전(az)은 이동 0으로 처리, seed 하나. 성공률은 실기 폐루프에서만 나온다.')

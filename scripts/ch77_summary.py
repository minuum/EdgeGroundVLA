"""CH77 종합 대조 정리(⓪). 모든 수치는 JSON에서 읽는다. gen_ch77.py가 본문 앞부분에 삽입."""
import json

import numpy as np

import gen_ch76_viz as V

FM = V.FM
A, G, R, B, P, MUT = V.A, V.G, V.R, V.B, V.P, V.MUT


def J(n):
    return json.load(open(FM / n))


def block(tbl, H, st):
    s100, s250, s300, l300 = J("summary_s1.json")["aggregate"], J("summary_s1_ep250.json")["aggregate"], J("summary_s1_full300.json")["aggregate"], J("summary_s1_full300_lopo.json")["aggregate"]
    cmp_, ex, rp = J("validation_batch.json")["compare"], J("fm_head_extras.json"), J("replay_sim_cont.json")
    ab = J("subgoal_ablation.json")
    ang = lambda a, k: f"{a[k]['angle_deg']:.1f}°"
    jit = lambda sp, k: ex[sp]["metrics"][k]["frame_jitter"]
    c0, c1 = rp["fold0"]["C_pi0_adaln"], rp["lopo_strong_right"]["C_pi0_adaln"]

    def sg(h, v, sel, cond="normal"):
        xs = [x for s, x in ab.items() if sel(s) and f"{h}|{v}" in x]
        return np.mean([x[f"{h}|{v}"][cond]["angle_deg"] for x in xs]) if xs else float("nan")
    fo, lo = (lambda s: s.startswith("fold")), (lambda s: s.startswith("lopo"))
    H_ = ["MLP_regression", "A_mlp_fm", "B2_film_res_fm", "C_pi0_adaln"]

    data = tbl(["", "v1 아카이브", "파일럿 v2", "<b>v3 본수집(현재)</b>", "V6(기존 이산)"], [
        ["에피소드·프레임", "40ep / 2,292", "15ep / 1,236", f"<b>300ep / {cmp_['신규 배치']['frames']:,}</b>", "60ep(이산)"],
        ["액션", "이산에 가까움", "연속", "<b>연속(조이스틱)</b>", "키보드 LUT ±1.15/0"],
        ["고유 raw 액션 수", f"{cmp_['v1 아카이브(40ep)']['unique_raw']}", f"{cmp_['파일럿 v2(15ep)']['unique_raw']}", f"<b>{cmp_['신규 배치']['unique_raw']:,}</b>", "18"],
        ["중간 크기 입력 비율", f"{cmp_['v1 아카이브(40ep)']['mid']:.0%}", f"{cmp_['파일럿 v2(15ep)']['mid']:.0%}", f"<b>{cmp_['신규 배치']['mid']:.0%}</b>", "—"],
        ["위치·경로", "단일", "단일 경로", "<b>5위치×3경로×20</b>", "5방향"]])
    act = tbl(["", "이산 8-class (EGVLA1식)", "연속 (EGVLA2)"], [
        ["표현", "8방향 중 하나 + 속력 고정", "(lx, ly) 연속 + 속력 규칙"],
        ["방향 해상도", "0°·45°·90°… (코너에서 최대 22.5° 어긋남)", "스틱 각도 그대로"],
        ["정답 라벨만 이산화했을 때 종점 오차", f"{c0['gt_disc']['fpe_cm']:.0f}~{c1['gt_disc']['fpe_cm']:.0f}cm (표현 자체의 손실)", "0 (손실 없음)"],
        ["모델 예측을 이산화했을 때 FPE (C)", f"{c0['disc']['fpe_cm']:.0f}cm / {c1['disc']['fpe_cm']:.0f}cm", f"{c0['cont_shadow']['fpe_cm']:.0f}cm / {c1['cont_shadow']['fpe_cm']:.0f}cm (같은 장소 / 안 본 위치)"],
        ["해석", "예측 오차가 지배적이라 이산화 영향은 작음", "표현 한계가 없고 속력 모델링 여지(모델 속력 시 FPE " + f"{c0['cont_model']['fpe_cm']:.0f}/{c1['cont_model']['fpe_cm']:.0f}cm)"]])
    heads = tbl(["", "MLP 회귀", "A (MLP-FM)", "B2 (FiLM-FM)", "C (π0형)"], [
        ["5-fold 방향 오차(같은 장소)", "—"] + [ang(s300, k) for k in H_[1:]],
        ["LOPO 방향 오차(안 본 위치)", "—"] + [ang(l300, k) for k in H_[1:]],
        ["청크 MSE(5-fold, lx/ly)", " / ".join(f"{x:.3f}" for x in s300["MLP_regression"]["mse_axis"][:2])] + [" / ".join(f"{x:.3f}" for x in s300[k]["mse_mean_axis"][:2]) for k in H_[1:]],
        ["최적 epoch(5-fold)", "10~50(조기 포화)", f"{s300['A_mlp_fm']['best_epoch']:.0f}", f"{s300['B2_film_res_fm']['best_epoch']:.0f}", f"{s300['C_pi0_adaln']['best_epoch']:.0f}"],
        ["프레임 저크(낮을수록 부드러움, 사람 0.045~0.063)"] + [f"{jit('fold0', k):.3f}" for k in H_],
        ["서브골 결측 시 방향 오차(안 본 위치)", "—"] + [f"{sg(k, 'full', lo, 'missing'):.1f}°" for k in H_[2:]][:0] + ["—", f"{sg('B2_film_res_fm', 'full', lo, 'missing'):.1f}°", f"{sg('C_pi0_adaln', 'full', lo, 'missing'):.1f}°"],
        ["한 줄 평", "가장 부드럽고 단순, 정확도 동급", "가장 단순한 FM, 수렴 느림", "FiLM 조향, 서브골 의존 큼", "<b>안 본 위치 최우수, 서브골 결측에 강함</b>"]])
    growth = tbl(["학습 세트", "에피소드", "방향 오차 C (같은 장소)", "A", "비고"], [
        ["이전(100epoch)", "218", ang(s100, "C_pi0_adaln"), ang(s100, "A_mlp_fm"), "위치 라벨 없는 28건 포함"],
        ["epoch 확대(250)", "218", ang(s250, "C_pi0_adaln"), ang(s250, "A_mlp_fm"), "epoch만 변경"],
        ["<b>300건 최종</b>", "300", f"<b>{ang(s300, 'C_pi0_adaln')}</b>", ang(s300, "A_mlp_fm"), "라벨 없음 28건 제외, 신규 81건 추가"]])
    sgt = tbl(["서브골 변형", "B2 같은 장소", "B2 안 본 위치", "C 같은 장소", "C 안 본 위치", "C 서브골 결측(안 본 위치)"], [
        [lab] + [f"{sg(h, v, s):.1f}°" for h, s in (("B2_film_res_fm", fo), ("B2_film_res_fm", lo), ("C_pi0_adaln", fo), ("C_pi0_adaln", lo))] + [("—" if v == "nobbox" else f"{sg('C_pi0_adaln', v, lo, 'missing'):.1f}°")]
        for v, lab in (("full", "서브골 있음(창3)"), ("nobbox", "서브골 없음"), ("w1", "현재 프레임만"), ("w5", "창 5"), ("drop30", "<b>드롭아웃 30%</b>"))])
    scope = tbl(["평가 범위", "무엇을 보는가", "결과(C)", "상태"], [
        ["5-fold(같은 장소)", "안 본 에피소드", ang(s300, "C_pi0_adaln"), st("측정", G)],
        ["LOPO(안 본 위치)", "안 본 위치까지 일반화", ang(l300, "C_pi0_adaln"), st("측정", G)],
        ["리플레이 시뮬레이션", "예측을 적분한 누적 위치 오차", f"FPE {c0['cont_shadow']['fpe_cm']:.0f} / {c1['cont_shadow']['fpe_cm']:.0f}cm", st("측정(열린 고리)", A)],
        ["실기 폐루프(EGVLA2)", "바구니 도달 성공률", "—", st("미측정", R)],
        ["실기 폐루프(EGVLA1)", "같은 정의의 성공률", "92.0% (276/300)", st("측정", G)]])
    claim = tbl(["주장해도 되는 것", "주장하면 안 되는 것"], [
        ["연속 조이스틱 데이터로 방향이 학습된다(R²: lx 0.5, ly 0.7)", "FM 헤드가 MLP 회귀보다 정확하다(차이 ≤0.003, 부드러움은 오히려 열세)"],
        ["Florence-2 단일화로 젯슨 fp32 지연 2.41배 감소(1971→817ms)", "EGVLA2가 EGVLA1보다 실기 성공률이 높다(미측정)"],
        ["안 본 위치에서는 방향 오차가 커지고(10.4→16.9°) C가 가장 낫다", "bbox 서브골이 정확도를 올린다(서브골 없어도 동급, B2는 안 본 위치에서 오히려 낫다)"],
        ["서브골 드롭아웃이 결측 입력에 대한 안전장치로 유효하다(C)", "π0.7을 재현했다(서브골 조향 아이디어와 π0형 헤드의 경량 변형)"],
        ["az(회전)는 이 수집 방식으로 학습되지 않아 규칙으로 둔다", "시뮬레이션 벤치마크 성능(해당 환경에 서봇2 없음)"]])
    ind = tbl(["", "논문들의 일반 평가", "우리 계획"], [
        ["주 지표", "SR(성공률), CR(충돌률), SPL, 이동 시간", "SR(EGVLA1 92.0%와 같은 정의), 위치별 60회 비교"],
        ["시뮬레이션", "IsaacLab·Habitat·DynaNav 벤치마크 병행", "없음(서봇2 모델 부재) → 한계로 명시, 리플레이 시뮬레이션은 보조"],
        ["실기 규모", "시나리오 소수, 규모가 작음", "300회 목표(상대적으로 큼, 단 타 논문과 규모 비교는 미확인)"],
        ["오프라인 지표", "보조 분석", "R²·방향 오차는 학습 가능성 확인용(성공과의 연결 없음)"],
        ["지연 대응", "TIC-VLA가 핵심 주제", "서버 180ms vs 젯슨 817ms 비교 예정"]])
    return (H("⓪ 지금까지의 정리 — 대조표 모음")
            + '<div style="font-size:0.8rem;line-height:1.6;margin:6px 0">아래는 이번 작업에서 나온 결과를 <b>대조 형식</b>으로 한곳에 모은 것이다. 세부 근거는 ①~⑦과 CH76-10에 있다. 모든 오프라인 수치는 안 본 에피소드(hold-out) 기준이고 seed·분할은 각 항목에 따른다.</div>'
            + H("A. 데이터 대조 — 무엇이 달라졌나") + data
            + H("B. 액션 표현 대조 — 이산 vs 연속") + act
            + H("C. 학습량 대조 — 데이터가 늘면 얼마나 좋아지나") + growth
            + H("D. 헤드 대조 — MLP 회귀 / A / B2 / C") + heads
            + H("E. 서브골 대조 — 있음/없음/창/드롭아웃") + sgt
            + '<div style="font-size:0.78rem;color:#94a3b8">방향 오차 평균(5분할). 서브골 결측은 입력을 전부 0으로 한 스트레스 테스트. seed 0 하나.</div>'
            + H("F. 평가 범위 대조 — 어디까지 측정됐나") + scope
            + H("G. 주장 가능 / 불가 대조") + claim
            + H("H. 평가 방식 대조 — 일반 논문 vs 우리") + ind)

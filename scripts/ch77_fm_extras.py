"""CH77: FM 헤드 보조 평가(부드러움·불확실도·다봉·학습 곡선) 섹션. 데이터: docs/v5/egvla2_fm/fm_head_extras.json (seed 0 단일)."""
import json

import ch77_viz as Z
import gen_ch76_viz as V

FM = V.FM
A, G, R, B, P = V.A, V.G, V.R, V.B, V.P
KEYS = ["MLP_regression", "A_mlp_fm", "B2_film_res_fm", "C_pi0_adaln"]
LAB = ["MLP 회귀", "FM-MLP", "FM-FiLM", "FM-π0"]
COL = [Z.HC[x] for x in LAB]
SPL = [("fold0", "같은 목표 위치 (fold0)"), ("lopo_strong_right", "안 본 목표 위치 (strong_right)")]


def block(tbl, H, st):
    f = FM / "fm_head_extras.json"
    if not f.exists():
        return ""
    d = json.load(open(f)); M = lambda sp, k, m: d[sp]["metrics"][k].get(m)
    grp = [(nm, LAB, COL) for _, nm in SPL]
    grp3 = [(nm, LAB[1:], COL[1:]) for _, nm in SPL]
    row = lambda m, k_list=KEYS: [[M(sp, k, m) for k in k_list] for sp, _ in SPL]
    gt = lambda sp, m: d[sp]["metrics"]["gt"][m]

    acc = Z.heat_table(grp, [("전체 청크 MSE (xy)", row("mse_xy"), "low"), ("다봉 후보 프레임 MSE", row("mse_xy_multimodal"), "low"), ("그 외 프레임 MSE", row("mse_xy_other"), "low")], fmt="{:.4f}",
                       corner="낮을수록 좋음", note=f"다봉 후보 = 비슷한 학습 장면 10개의 사람 조작 방향이 갈리는 프레임(fold0 {gt('fold0','multimodal_frac'):.0%}, 안 본 위치 {gt('lopo_strong_right','multimodal_frac'):.0%})")
    sm = Z.heat_table(grp, [("청크 TV · K=8 평균", row("chunk_tv"), "low", [f"{gt('fold0','chunk_tv'):.3f} / {gt('lopo_strong_right','chunk_tv'):.3f}"]),
                            ("청크 TV · 단일 샘플", row("chunk_tv_single"), "low", [""]),
                            ("프레임 저크 · K=8 평균", row("frame_jitter"), "low", [f"{gt('fold0','frame_jitter'):.3f} / {gt('lopo_strong_right','frame_jitter'):.3f}"]),
                            ("프레임 저크 · 단일 샘플", row("frame_jitter_single"), "low", [""])], fmt="{:.3f}", corner="낮을수록 부드러움", ref_label="사람(같은목표/안본목표)")
    jit = Z.bars([(l, M("fold0", k, "frame_jitter"), c) for l, k, c in zip(LAB, KEYS, COL)] + [(l + " 단일", M("fold0", k, "frame_jitter_single"), c) for l, k, c in zip(LAB[1:], KEYS[1:], COL[1:])],
                 0.3, ref=("사람", gt("fold0", "frame_jitter")))
    unc = Z.heat_table(grp3, [("순위 상관 ρ (높을수록 신호 좋음)", row("unc_err_spearman", KEYS[1:]), "high"),
                              ("오차 · 불확실도 하위 25%", row("err_low_unc", KEYS[1:]), None), ("오차 · 불확실도 상위 25%", row("err_high_unc", KEYS[1:]), None),
                              ("불확실도 · 다봉 후보", row("unc_multimodal", KEYS[1:]), None), ("불확실도 · 그 외", row("unc_other", KEYS[1:]), None)], fmt="{:.2f}", corner="K=8 샘플 표준편차 → 오차 예측")
    cur = []
    for k, lab in zip(KEYS, LAB):
        pass
    cv = lambda sp, k: d[sp]["curves"][k]
    mn = lambda sp, k: min(cv(sp, k), key=lambda x: x[2])
    curve = Z.heat_table(grp, [("hold-out 최저 손실", [[mn(sp, k)[2] for k in KEYS] for sp, _ in SPL], "low"),
                               ("최저 epoch", [[f"ep {mn(sp, k)[0]}" for k in KEYS] for sp, _ in SPL], None),
                               ("마지막 epoch hold-out 손실", [[cv(sp, k)[-1][2] for k in KEYS] for sp, _ in SPL], "low"),
                               ("마지막 epoch train 손실", [[cv(sp, k)[-1][1] for k in KEYS] for sp, _ in SPL], None)], fmt="{:.3f}", corner="손실(회귀는 MSE)")
    box = lambda t, body: f'<div style="margin:12px 0 4px;font-size:0.82rem;font-weight:800;color:#fcd34d">{t}</div>{body}'
    return (H("⑥-3 FM 헤드 보조 평가 — 부드러움·불확실도·다봉·학습 곡선 (오프라인, seed 0)")
            + '<div style="font-size:0.8rem;line-height:1.6;margin:6px 0">FM 헤드를 방향 오차 말고 <b>다른 관점</b>으로 본 결과. 학습에 쓰지 않은 에피소드 60개로 평가했고 헤드별 epoch은 고정(FM-MLP 240, FM-FiLM 220, FM-π0 120, 회귀 60), <b>seed 하나</b>라 경향만 읽는다.</div>'
            + Z.LEG
            + box("① 정확도 — 청크 이동 MSE", acc) + box("② 부드러움 — 청크 변화량(TV)과 프레임 간 저크", sm)
            + box("프레임 저크 비교 (K=8 평균 · 같은 목표 위치) — 점선이 사람", jit)
            + box("③ 불확실도 신호 — 샘플 분산이 클수록 오차도 큰가", unc) + box("④ 학습 곡선 요약", curve)
            + f'{st("읽는 법", A)}<ul style="margin:6px 0 6px 18px;font-size:0.8rem">'
              '<li><b>정확도</b>: FM이 회귀보다 낫지 않다(초록=가장 좋음 열 확인). 다봉 후보(5~7%)는 모든 헤드 오차가 약 2배로 커지지만 FM 우위는 없다 → <b>이 데이터에서는 FM의 다봉 이점이 보이지 않는다</b>(사실상 단봉). <b style="color:#f87171">[정정 2026-10-21]</b> 에피소드 <b>시작 구간</b>은 경로 모호성으로 실제 다봉이며 FM 샘플이 여러 모드로 갈린다(⑥-13). 이 판단은 시작 구간을 제외한 대부분 프레임에 대한 것이다.</li>'
              '<li><b>부드러움</b>: 회귀가 가장 부드럽고 FM 단일 샘플은 흔들림이 3배 가까이 크다. K=8 평균 후에도 사람의 2~3배 → <b>서빙에서 K 평균과 시간 평활 필수</b>. 반대로 청크 내부 변화량은 FM이 사람에 더 가깝다.</li>'
              '<li><b>불확실도</b>: 상관은 약하나 방향은 맞다(ρ 0.1~0.3). 상위 25% 프레임의 오차가 하위 25%의 1.2~1.9배, 다봉 후보에서 분산이 커진다 → 속력을 낮추는 <b>보조 신호</b> 후보.</li>'
              '<li><b>학습 곡선</b>: 안 본 위치에서 hold-out이 훨씬 일찍 최저가 된 뒤 오른다 → 250 epoch은 새 위치에 과적합(자세한 곡선은 ⑥-6).</li></ul>'
            + f'{st("결론", R)} FM 헤드를 쓸 <b>정확도상의 이유는 없다</b>(회귀와 동급 또는 약간 낮음). 이득은 연속 샘플링·불확실도·청크 표현력이고, 비용은 저크와 연산이다. 헤드 선택은 실기에서 하되 FM이면 <b>K 평균 + 평활 + 일찍 멈춘 체크포인트</b>를 기본으로, 회귀 헤드를 비교군으로 함께 평가하길 권한다.')

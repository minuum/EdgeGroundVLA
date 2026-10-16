"""CH77: FM 헤드 보조 평가(부드러움·불확실도·다봉·학습 곡선) 섹션. 데이터: docs/v5/egvla2_fm/fm_head_extras.json (seed 0 단일)."""
import json

import gen_ch76_viz as V

FM = V.FM
A, G, R, B, P = V.A, V.G, V.R, V.B, V.P
NM = {"A_mlp_fm": "A (MLP-FM)", "B2_film_res_fm": "B2 (FiLM-FM)", "C_pi0_adaln": "C (π0형)", "MLP_regression": "MLP 회귀"}
SP = {"fold0": "5-fold fold0 (같은 장소, 60ep 평가)", "lopo_strong_right": "LOPO strong_right (안 본 위치, 60ep 평가)"}


def block(tbl, H, st):
    f = FM / "fm_head_extras.json"
    if not f.exists():
        return ""
    d = json.load(open(f))
    acc, smooth, unc, curve = [], [], [], []
    for s, v in d.items():
        m = v["metrics"]; g = m["gt"]
        for k in NM:
            r = m[k]
            acc.append([SP[s], NM[k], f"{r['mse_xy']:.4f}", f"{r['mse_xy_multimodal']:.4f} / {r['mse_xy_other']:.4f}"])
            smooth.append([SP[s], NM[k], f"{r['chunk_tv']:.3f}" + (f" (단일 {r['chunk_tv_single']:.3f})" if k != "MLP_regression" else ""),
                           f"{r['frame_jitter']:.3f}" + (f" (단일 {r['frame_jitter_single']:.3f})" if k != "MLP_regression" else "")])
            if "unc_err_spearman" in r:
                unc.append([SP[s], NM[k], f"{r['unc_err_spearman']:.2f}", f"{r['err_low_unc']:.2f} → {r['err_high_unc']:.2f}", f"{r['unc_multimodal']:.3f} / {r['unc_other']:.3f}"])
        smooth.append([SP[s], "<b>사람 조작(정답)</b>", f"<b>{g['chunk_tv']:.3f}</b>", f"<b>{g['frame_jitter']:.3f}</b>"])
        for k, c in v["curves"].items():
            best = min(c, key=lambda x: x[2])
            curve.append([SP[s], NM[k], f"{best[0]} ({best[2]:.3f})", f"{c[-1][0]} ({c[-1][2]:.3f})", f"{c[-1][1]:.3f}"])
    nm = {s: f"{v['metrics']['gt']['multimodal_frac']:.0%}" for s, v in d.items()}
    return (H("⑥-3 FM 헤드 보조 평가 — 부드러움·불확실도·다봉·학습 곡선 (오프라인, seed 0)")
            + '<div style="font-size:0.8rem;line-height:1.6;margin:6px 0">FM 헤드를 방향 오차 말고 <b>다른 관점</b>으로 본 결과다. 학습에 쓰지 않은 에피소드(60ep)로 평가했고, 헤드별 epoch은 이전 최적값(A 240, B2 220, C 120, MLP 60), <b>seed 하나</b>라 경향만 읽는다.</div>'
            + f'<div style="font-size:0.8rem;margin:4px 0"><b>① 정확도(청크 이동 MSE, 낮을수록 좋음)와 다봉 후보 프레임</b> — 다봉 후보 = 비슷한 학습 장면 10개의 사람 조작 방향이 갈리는 프레임(fold0 {nm["fold0"]}, LOPO {nm["lopo_strong_right"]})</div>'
            + tbl(["평가 세트", "모델", "전체 MSE", "다봉 후보 / 그 외 MSE"], acc)
            + '<div style="font-size:0.8rem;margin:10px 0 4px"><b>② 부드러움</b> — 청크 내부 변화량(TV)과 연속 프레임 간 첫 스텝 변화량(저크). 정답에 가까울수록 사람처럼 움직임</div>'
            + tbl(["평가 세트", "모델", "청크 TV (K=8 평균 / 단일 샘플)", "프레임 간 저크 (K=8 평균 / 단일 샘플)"], smooth)
            + '<div style="font-size:0.8rem;margin:10px 0 4px"><b>③ 불확실도 신호</b> — K=8 샘플 표준편차가 클수록 실제 오차도 큰가</div>'
            + tbl(["평가 세트", "모델", "순위 상관 ρ", "오차: 불확실도 하위25% → 상위25%", "불확실도: 다봉 후보 / 그 외"], unc)
            + '<div style="font-size:0.8rem;margin:10px 0 4px"><b>④ 학습 곡선</b> — hold-out flow loss(회귀는 MSE)가 최저인 epoch과 마지막 값</div>'
            + tbl(["평가 세트", "모델", "hold-out 최저 epoch (값)", "마지막 epoch (hold-out)", "마지막 train loss"], curve)
            + f'{st("읽는 법", A)}<ul style="margin:6px 0 6px 18px;font-size:0.8rem">'
              '<li><b>정확도</b>: FM이 MLP 회귀보다 낫지 않다(fold0 회귀 0.039 vs FM 0.040~0.043, LOPO 회귀 0.069 ≈ B2 0.069). 다봉 후보 프레임(5~7%)에서 모든 헤드의 오차가 약 2배로 커지는데, fold0에서는 회귀가 더 낫고 LOPO에서는 B2만 소폭 낫다(0.082 vs 0.086) → <b>이 데이터에서는 FM의 다봉 이점이 보이지 않는다</b>(예상대로 사실상 단봉).</li>'
              '<li><b>부드러움</b>: 회귀가 가장 부드럽다. FM 단일 샘플은 프레임 간 저크가 0.20~0.26으로 회귀(0.07~0.085)의 약 3배이고, K=8 평균해도 0.11~0.12로 사람(0.045~0.063)의 2~3배. 반대로 청크 내부 변화량은 FM이 사람(0.066)에 더 가깝고 회귀(0.019)는 지나치게 평평하다. → <b>서빙에서는 K 평균과 시간 평활(EMA 등)이 필수</b>.</li>'
              '<li><b>불확실도</b>: 상관은 약하지만 방향이 맞다(ρ 0.1~0.3). 불확실도 상위 25% 프레임의 오차는 하위 25%의 약 1.2~1.9배이고, 다봉 후보에서 샘플 분산이 커진다. → 불확실할 때 속력을 낮추는 <b>보조 신호</b>로는 가능성이 있으나 단독 신뢰는 어렵다.</li>'
              '<li><b>학습 곡선</b>: 안 본 위치(LOPO)에서는 hold-out이 훨씬 일찍 최저(회귀 10, C 20, A 80~, B2 100 부근)가 되고 이후 올라간다. 즉 <b>250 epoch 학습은 새 위치에는 과적합</b>이고, 같은 장소 안(fold0)에서는 A·B2는 계속 줄고 C·회귀는 50~90에서 포화한다. 앞 S1/LOPO의 epoch은 학습 위치 inner-val로 골랐기 때문에 새 위치 기준으로는 최적이 아닐 수 있다.</li></ul>'
            + f'{st("결론", R)} FM 헤드를 쓸 <b>정확도상의 이유는 없다</b>(회귀와 동급 또는 약간 낮음). 이득은 연속 샘플링·불확실도·청크 표현력이고, 비용은 저크와 연산이다. 헤드 선택은 실기(폐루프)에서 하되 FM을 쓴다면 <b>K 평균 + 평활 + 일찍 멈춘 체크포인트</b>를 기본으로 하고, 회귀(MLP) 헤드를 비교군으로 함께 실기 평가하는 것을 권한다. seed 하나·2개 hold-out의 결과다.')

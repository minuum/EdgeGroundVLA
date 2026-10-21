"""CH77 ⑥-18: 액션 표현·데이터량·에이전트/미래 서브골 심화 실험. 있는 JSON만 그린다(실험이 끝나는 대로 재생성)."""
import json

import numpy as np

import ch77_viz as Z
import gen_ch76_viz as V

FM = V.FM
A, G, R, B, P, MUT = V.A, V.G, V.R, V.B, V.P, V.MUT


def J(n):
    f = FM / n
    return json.load(open(f)) if f.exists() else None


def sec_repr(tbl, H, st):
    s = J("action_repr_stats.json")
    if not s:
        return ""
    q = s["quant_err"]; hist = q["hist"]; tot = sum(hist); labs = ["0~2.5°", "2.5~5°", "5~10°", "10~15°", "15~22.5°"]
    bars = Z.bars([(l, c / tot * 100, B if i < 3 else R, f"{c / tot * 100:.0f}%") for i, (l, c) in enumerate(zip(labs, hist))], 70, w=70)
    ent = s["entropy_bits"]; sp = s["speed"]; sc = s["staircase"]; bn = s["boundary_near"]
    out = (H("⑥-18a 액션 표현 심화 — 사람 조작을 8방향으로 바꾸면 무엇이 사라지나")
           + '<div style="display:flex;gap:16px;flex-wrap:wrap"><div style="flex:1 1 320px"><div style="font-weight:800;color:#fcd34d;font-size:0.8rem;margin-bottom:4px">사람 조작 방향 → 8방향 반올림 오차 분포(이동 프레임 25,386개)</div>' + bars
           + f'<div style="font-size:0.74rem;color:#94a3b8;margin-top:4px">중앙값 {q["median"]:.1f}° · 평균 {q["mean"]:.1f}° · 상위 10%는 {q["p90"]:.0f}° 이상. 반올림 경계(±22.5°, ±67.5°) 5° 안에 있는 프레임 {bn["within_5deg"] * 100:.0f}%.</div></div>'
           + '<div style="flex:1 1 320px">' + tbl(["항목", "값", "해석"], [
               ["방향 정보량(5° 해상도 / 1° 해상도)", f'{ent["angle_5deg"]:.1f} / {ent["angle_1deg"]:.1f} bit', "연속 방향이 담는 정보"], ["8-class가 남기는 정보량", f'{ent["class8"]:.1f} bit', "사실상 5방향(전진·사선 2·좌우)만 사용"],
               ["속력 단계", f'느림 {sp["exec_levels"]["0.4"] * 100:.0f}% / 보통 {sp["exec_levels"]["0.7"] * 100:.0f}% / 빠름 {sp["exec_levels"]["1.0"] * 100:.0f}%', "라벨(exec)은 3단계"], ["속력 0.7 고정 가정의 평균 상대 오차", f'{sp["rel_err_fixed07"]["mean"] * 100:.0f}%', "이산 표현은 속력도 고정"],
               ["클래스 전환 / 프레임당 방향 변화", f'{sc["class_switch_per_frame"] * 100:.1f}% / {sc["mean_abs_angle_step_deg"]:.1f}°', "연속 방향은 매끄럽게 변함, 이산은 계단"]]) + '</div></div>')
    d = J("discrete_head.json")
    if d:
        names = ["이산 8-class 헤드", "FM-π0 (연속 그대로)", "FM-π0 (8방향으로 양자화)", "MLP 회귀 (연속 그대로)", "MLP 회귀 (8방향으로 양자화)"]
        grp = [(g, ["이산", "FM 연속", "FM 양자화", "회귀 연속", "회귀 양자화"], [R, A, A, MUT, MUT]) for g in ("같은 목표 위치(5-fold 평균)", "안 본 목표 위치(LOPO 5개 평균)")]
        rows = []
        for met, lab, better, fmt in (("angle", "방향 오차(°)", "low", "{:.1f}"), ("acc", "8-class 일치율", "high", "{:.2f}"), ("go_f1", "GO/STOP F1", "high", "{:.3f}"), ("fpe", "궤적 종점 오차(cm)", "low", "{:.0f}")):
            vals = []
            for pref in ("fold", "lopo"):
                ks = [k for k in d if k.startswith(pref)]
                vals.append([float(np.mean([d[k][n][met] for k in ks])) for n in names])
            rows.append((lab, vals, better))
        out += ('<div style="margin:12px 0 4px;font-size:0.84rem;font-weight:800;color:#fcd34d">같은 Florence-2 특징·bbox로 학습한 이산 헤드 vs 연속 헤드 (분할 ' + f'{len(d)}개 완료)</div>' + Z.LEG
                + Z.heat_table(grp, rows, fmt="{:.2f}", corner="지표", by="row")
                + f'{st("읽는 법", A)} <b>표현(양자화) 효과와 학습 방식 효과를 분리</b>했다. 연속 헤드의 예측을 8방향으로 양자화해도 오차가 거의 안 변하고, 이산 헤드(교차 엔트로피)도 비슷하다 → <b>이 데이터에서는 액션 표현보다 모델 예측 오차가 지배적</b>이다. 이산 표현의 실제 손실은 정답 라벨 자체를 이산화해도 종점이 19~23cm 어긋나는 것(표현 한계)에 있다.')
    return out


def sec_scaling(tbl, H, st):
    d = J("data_scaling.json")
    if not d:
        return ""
    dens = {}
    for k, v in d.items():
        if k.startswith("density"):
            dens.setdefault(float(k.split("|")[1][1:]), []).append(v["angle"])
    div = {}
    for k, v in d.items():
        if k.startswith("diversity"):
            div.setdefault(int(k.split("|")[2][1:]), []).append(v["angle"])
    out = H("⑥-18b 데이터량 곡선 — 에피소드를 더 모으면 얼마나, 위치를 다양하게 모으면 얼마나")
    if dens:
        fr = sorted(dens); xs = np.array([f * 240 for f in fr]); ys = np.array([np.mean(dens[f]) for f in fr])
        rows = [[f"{f:.1f}", f"{int(f * 240)}", f"{np.mean(dens[f]):.2f}°", f"{np.std(dens[f]):.2f}", f"{len(dens[f])}"] for f in fr]
        slope = ""
        if len(fr) >= 3:
            b = np.polyfit(np.log(xs), np.log(ys), 1)[0]; slope = f" 로그-로그 기울기 {b:.2f}(오차 ∝ N^{b:.2f}) → 데이터를 2배로 하면 오차가 약 {(1 - 2 ** b) * 100:.0f}% 줄어드는 추세."
        out += ('<div style="margin:8px 0 4px;font-size:0.84rem;font-weight:800;color:#fcd34d">① 밀도 — 같은 목표 위치에서 학습 에피소드 수를 늘렸을 때 (안 본 에피소드 방향 오차)</div>'
                + tbl(["학습 비율", "학습 에피소드(약)", "방향 오차 평균", "표준편차", "실행 수"], rows) + f'<div style="font-size:0.78rem;color:#94a3b8">최적화 스텝 수는 맞춤(epoch=120/비율).{slope}</div>')
    if div:
        rows = [[f"{k}개", f"{np.mean(v):.2f}°", f"{np.std(v):.2f}", f"{len(v)}"] for k, v in sorted(div.items())]
        out += ('<div style="margin:12px 0 4px;font-size:0.84rem;font-weight:800;color:#fcd34d">② 다양성 — 같은 에피소드 수(60개)를 몇 개 목표 위치에 나눠 모았을 때 (안 본 목표 위치 방향 오차)</div>'
                + tbl(["학습에 쓴 목표 위치 수", "안 본 위치 방향 오차 평균", "표준편차", "실행 수"], rows)
                + f'{st("읽는 법", A)} 에피소드 수를 고정하고 위치 수만 바꾼 비교라 <b>다양성의 순수 효과</b>다. 위치 수가 늘수록 오차가 줄면 pos2·pos3 같은 새 조건을 모으는 쪽이 같은 에피소드 수를 한 조건에 더 쌓는 것보다 낫다는 근거가 된다. 문헌(로봇 모방학습 데이터 스케일링 연구, ICLR 2025)도 환경·객체 다양성이 데모 수보다 중요하다고 보고했다(원문 요약 수준).')
    return out


def sec_subgoal(tbl, H, st):
    a, f = J("agent_subgoal.json"), J("subgoal_future_feature.json")
    out = ""
    if a:
        ks = list(a); mean = lambda c, m: float(np.mean([a[k][c][m] for k in ks]))
        rows = [[c, f'{mean(c, "all"):.1f}°', f'{mean(c, "first10"):.1f}°', f'{mean(c, "rest"):.1f}°'] for c in ("정답 의도", "의도 없음", "틀린 의도")]
        out += (H("⑥-18c 에이전트 서브골 상한 — 경로 의도(좌커브/직진/우커브)를 두 번째 서브골 채널로")
                + f'<div style="font-size:0.8rem;margin:6px 0">bbox 외에 상위 에이전트가 주는 경로 의도 원핫을 FM-π0의 서브골 입력에 추가(학습 시 이 채널만 30% 드롭아웃). 같은 모델을 세 조건으로 평가(분할 {len(ks)}개 평균, seed 0).</div>'
                + tbl(["에이전트가 준 의도", "방향 오차(전체)", "에피소드 첫 10%", "나머지"], rows)
                + f'{st("읽는 법", G)} 에이전트가 맞으면 {mean("정답 의도", "all"):.1f}°(의도 없음 {mean("의도 없음", "all"):.1f}°), <b>틀리면 {mean("틀린 의도", "all"):.1f}°로 크게 나빠진다</b>. 이득의 상한은 작고 위험은 크므로 에이전트 서브골은 <b>정확도가 충분히 높을 때(그리고 틀렸을 때를 견디도록 학습할 때)</b>만 가치가 있다.')
    if f:
        ks = list(f); m = lambda v, k: float(np.mean([f[x][v][k] for x in ks]))
        rows = [["현재 특징만(기준, 서브골 없음 아님: bbox만)", "—", "—"], ["미래 임베딩 정답(누설 상한)", f'{m("fut_feat_oracle", "angle"):.1f}°', f'{m("fut_feat_oracle", "go_f1"):.3f}'], ["미래 임베딩 예측(생성기)", f'{m("fut_feat_pred", "angle"):.1f}°', f'{m("fut_feat_pred", "go_f1"):.3f}']]
        out += (H("⑥-18d '다음 이미지 예측' 서브골 — 잠재 공간 버전(약 1초 뒤 프레임의 Florence-2 임베딩)")
                + f'<div style="font-size:0.8rem;margin:6px 0">π0.7의 "서브골 이미지"를 이미지 임베딩(768-d)으로 옮긴 버전. 생성기 품질: 예측 vs 정답 코사인 {m("generator", "cos_pred"):.3f}, "현재 그대로" 기준선 {m("generator", "cos_stay"):.3f}. 분할 {len(ks)}개 평균.</div>'
                + tbl(["서브골", "방향 오차", "GO/STOP F1"], rows[1:])
                + f'{st("읽는 법", A)} 미래 임베딩을 정답으로 줘도(누설 상한) 이득이 어느 정도인지, 예측으로 주면 그 이득이 남는지를 본다. 앞의 미래 bbox 실험과 같은 구조라 비교 가능하다.')
    return out


def block(tbl, H, st):
    body = sec_repr(tbl, H, st) + sec_scaling(tbl, H, st) + sec_subgoal(tbl, H, st)
    if not body:
        return ""
    return H("⑥-18 심화 실험 — 액션 표현·데이터량·에이전트/미래 서브골") + '<div style="font-size:0.8rem;color:#94a3b8;margin:4px 0">진행 중인 실험은 끝나는 대로 이 섹션이 채워진다(분할 수를 표에 표시). seed 1~2개, 경향 파악용.</div>' + body

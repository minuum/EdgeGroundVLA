"""CH77 ⑥-17: 방향 오차를 성공 기준에 가깝게 — 다중 정답 오차, 종점 허용 범위, 사람 데모의 종점 이미지 허용치. 데이터: outcome_metrics.json"""
import json

import numpy as np

import ch77_viz as Z
import gen_ch76_viz as V

FM = V.FM
A, G, R, B, P, MUT = V.A, V.G, V.R, V.B, V.P, V.MUT
SPN = {"fold0": "같은 목표 위치(fold0)", "lopo_strong_right": "안 본 목표 위치(strong_right)"}
BN = ["0~5°", "5~10°", "10~15°", "15~20°", "20~30°", "30~45°", "45°+"]


def endpoints_svg(o, cell, ce, w=330, h=250):
    hum = np.array(ce[cell]); mod = np.array([e[1] for e in o["ends"] if e[0] == cell]) if o["ends"] else np.zeros((0, 2))
    allp = np.vstack([hum] + ([mod] if len(mod) else [])); lat0, lat1 = allp[:, 1].min() - 15, allp[:, 1].max() + 15; f0, f1 = allp[:, 0].min() - 15, allp[:, 0].max() + 15
    sc = min((w - 50) / max(lat1 - lat0, 40), (h - 50) / max(f1 - f0, 40)); X = lambda p: w - 25 - (p[1] - lat0) * sc; Y = lambda p: h - 25 - (p[0] - f0) * sc
    s = f'<svg viewBox="0 0 {w} {h}" style="width:100%;max-width:{w}px;background:#0a1220;border:1px solid #1e293b;border-radius:8px"><text x="10" y="14" fill="#e2e8f0" font-size="10" font-weight="700">종점(cm): {cell} — 사람 {len(hum)}명 vs 모델</text>'
    s += "".join(f'<circle cx="{X(p):.1f}" cy="{Y(p):.1f}" r="4.5" fill="{G}" opacity="0.8"/>' for p in hum) + "".join(f'<path d="M {X(p) - 5:.1f} {Y(p) - 5:.1f} l 10 10 m -10 0 l 10 -10" stroke="{A}" stroke-width="2.4"/>' for p in mod)
    return s + '<text x="10" y="' + str(h - 8) + f'" fill="#94a3b8" font-size="8">● 사람 종점(20개), ✕ 모델(안 본 에피소드) · 위=전진, 왼쪽=좌</text></svg>'


def block(tbl, H, st):
    f = FM / "outcome_metrics.json"
    if not f.exists():
        return ""
    d = json.load(open(f)); ff = d["final_frame"]
    rows = []
    for sp in SPN:
        o = d[sp]
        rows.append([SPN[sp], f'{o["single_ref_model"]["mean"]:.1f}°', f'<b>{o["multi_ref_model"]["mean"]:.1f}°</b>', f'{o["multi_ref_human"]["mean"]:.1f}°', f'{o["multi_ref_model"]["le10"] * 100:.0f}% / {o["multi_ref_human"]["le10"] * 100:.0f}%',
                     f'{o["endpoint_nn_model"]["median"]:.0f} / {o["endpoint_nn_human"]["median"]:.0f}cm', f'{o["frac_model_end_within_human_p90"] * 100:.0f}%'])
    t = tbl(["평가 세트", "단일 정답(그 사람 조작) 오차", "<b>다중 정답</b> 오차(모델)", "다중 정답 오차(사람 자신)", "10° 이내 프레임(모델 / 사람)", "종점 최근접 사람과의 거리 중앙값(모델 / 사람)", "모델 종점이 사람 종점 분포(p90) 안인 비율"], rows)
    hist = ""
    for sp in SPN:
        o = d[sp]; tot_m, tot_h = sum(o["hist"]["model"]), sum(o["hist"]["human"])
        hist += f'<div style="flex:1 1 330px"><div style="font-weight:800;color:#fcd34d;font-size:0.8rem;margin:4px 0">{SPN[sp]} — 다중 정답 오차 분포(프레임 비율)</div>' + Z.bars([(f"{b} 모델", m / tot_m * 100, B, f"{m / tot_m * 100:.0f}%") for b, m in zip(BN, o["hist"]["model"])], 100, w=70) + "</div>"
    cell = "center|left_curve"
    pts = '<div style="display:flex;gap:12px;flex-wrap:wrap">' + "".join(f'<div style="flex:1 1 330px">{endpoints_svg(d[sp], cell, d["cell_ends"])}</div>' for sp in ("fold0",) if any(e[0] == cell for e in d[sp]["ends"])) + "</div>"
    tol = tbl(["사람 데모의 마지막 프레임(300건)", "값", "의미"], [
        ["bbox 검출", f'{ff["detected"] * 100:.0f}%', "모든 에피소드 끝에서 바구니가 보임"], ["가로 위치 cx 평균 ± 표준편차", f'{ff["cx_mean"]:.2f} ± {ff["cx_std"]:.2f}', "정중앙(0.5)에서 벗어나 끝나는 에피소드가 다수"],
        ["cx 5~95% 범위", f'{ff["cx_p05"]:.2f} ~ {ff["cx_p95"]:.2f}', "도착 위치 허용 폭이 넓음"], ["정중앙 ±0.1 / ±0.2 이내", f'{ff["within_cx_0p1"] * 100:.0f}% / {ff["within_cx_0p2"] * 100:.0f}%', "'정확히 가운데 도착'이 아니어도 데이터로 채택"],
        ["bbox 면적 평균(5~95%)", f'{ff["area_mean"]:.2f} ({ff["area_p05"]:.2f}~{ff["area_p95"]:.2f})', "가까이 도착하면 종료(근접 규칙과 같은 기준)"]])
    return (H("⑥-17 방향 오차는 맞는 지표인가 — 성공 기준에 가깝게 보기(다중 정답·종점 허용)")
            + '<div style="font-size:0.8rem;line-height:1.6;margin:6px 0">사용자 지적: 수집 의도는 <b>정확히 가운데 도착이 아니어도 성공으로 치는 에피소드</b>이고 사람마다 다른 경로를 택한다. 그렇다면 <b>한 사람의 조작을 정답으로 삼는 방향 오차는 정당한 다른 선택을 오답으로 친다</b>. '
              '이를 반영해 (1) 같은 상황(목표 위치·경로·진행도)에서 <b>다른 사람들이 낸 방향 중 가장 가까운 것</b>과의 오차, (2) 종점이 사람 종점 구름 안에 드는지, (3) 사람 데모의 도착 허용 폭을 재 보았다.</div>'
            + '<div style="margin:10px 0 4px;font-size:0.84rem;font-weight:800;color:#fcd34d">① 단일 정답 vs 다중 정답</div>' + t
            + f'{st("결론", G)} 같은 목표 위치에서 평균 방향 오차 {d["fold0"]["single_ref_model"]["mean"]:.1f}° 중 <b>대부분이 "다른 정당한 선택"이었다</b>(다중 정답 {d["fold0"]["multi_ref_model"]["mean"]:.1f}°, 사람 자신 {d["fold0"]["multi_ref_human"]["mean"]:.1f}°). 안 본 위치는 {d["lopo_strong_right"]["single_ref_model"]["mean"]:.1f}° → {d["lopo_strong_right"]["multi_ref_model"]["mean"]:.1f}°로 줄지만 사람 자신({d["lopo_strong_right"]["multi_ref_human"]["mean"]:.1f}°)보다 여전히 크다 → <b>일반화 gap은 실재</b>한다.'
            + '<div style="display:flex;gap:14px;flex-wrap:wrap;margin-top:8px">' + hist + '</div>'
            + '<div style="margin:12px 0 4px;font-size:0.84rem;font-weight:800;color:#fcd34d">② 종점 — 모델이 낸 명령을 적분한 도착점이 사람 도착점 구름 안에 드는가</div>' + pts
            + f'{st("읽는 법", A)} 모델 종점이 사람 종점(초록)에서 가장 가까운 사람까지 중앙값 {d["fold0"]["endpoint_nn_model"]["median"]:.0f}cm(사람끼리는 {d["fold0"]["endpoint_nn_human"]["median"]:.0f}cm), 모델 종점의 {d["fold0"]["frac_model_end_within_human_p90"] * 100:.0f}%가 사람 종점 분포 안이다. 열린 고리 적분이라 보정이 없어 실제 폐루프보다 보수적이다.'
            + '<div style="margin:12px 0 4px;font-size:0.84rem;font-weight:800;color:#fcd34d">③ 수집 데이터가 허용하는 도착 범위</div>' + tol
            + f'{st("판단", G)} <b>방향 오차(단일 정답)는 성공 지표로 부적절하다</b> — 진단 지표로만 쓴다. 성공에 가까운 순서로 ① 폐루프 SR·충돌·개입, ② <b>종점·도착 허용 범위</b>(사람 데모의 cx 허용 폭 ±0.2 안에 84% 도착), ③ 다중 정답 방향 오차, ④ 단일 정답 방향 오차 순으로 위계를 두자. 다만 ②③도 사람 데모가 정답 집합이라는 가정이고 실제 장애물·충돌은 반영하지 못한다.')

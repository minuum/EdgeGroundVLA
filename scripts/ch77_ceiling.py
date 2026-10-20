"""CH77 ⑥-8: 사람 조작 일치도 천장 — 모델 방향 오차를 해석하는 기준. 데이터: human_ceiling.json, summary_s1_full300*.json"""
import json

import ch77_viz as Z
import gen_ch76_viz as V

FM = V.FM
A, G, R, B, P, MUT = V.A, V.G, V.R, V.B, V.P, V.MUT


def block(tbl, H, st):
    c = json.load(open(FM / "human_ceiling.json"))["results"]; s = json.load(open(FM / "summary_s1_full300.json"))["aggregate"]; l = json.load(open(FM / "summary_s1_full300_lopo.json"))["aggregate"]
    cell, pos = c["같은 셀(목표 위치+경로)"], c["같은 목표 위치(경로 무관)"]
    bars = Z.bars([("사람끼리 한 프레임(같은 셀·진행도)", cell["pairwise_mean"], MUT), ("사람 합의 방향(오라클: 셀·진행도 앎)", cell["consensus_mean"], G),
                   ("모델 C · 같은 목표 위치(5-fold)", s["C_pi0_adaln"]["angle_deg"], B), ("모델 C · 안 본 목표 위치(LOPO)", l["C_pi0_adaln"]["angle_deg"], R),
                   ("참고: 경로가 다른 사람끼리(같은 목표 위치)", pos["pairwise_mean"], "#475569")], 45, "°", w=250)
    tb = tbl(["비교 대상", "평균", "중앙값", "p90", "프레임 수"], [
        ["사람 합의 방향 vs 사람(같은 셀·진행도)", f"{cell['consensus_mean']:.1f}°", f"{cell['consensus_median']:.1f}°", f"{cell['consensus_p90']:.1f}°", f"{cell['n']:,}"],
        ["사람 한 프레임 vs 사람 한 프레임(같은 셀·진행도)", f"{cell['pairwise_mean']:.1f}°", f"{cell['pairwise_median']:.1f}°", "—", f"{cell['n']:,}"],
        ["사람 합의 방향 vs 사람(같은 목표 위치, 경로 무관)", f"{pos['consensus_mean']:.1f}°", f"{pos['consensus_median']:.1f}°", f"{pos['consensus_p90']:.1f}°", f"{pos['n']:,}"]])
    return (H("⑥-8 사람 조작 일치도 천장 — 방향 오차 10°는 높은가 낮은가")
            + '<div style="font-size:0.8rem;line-height:1.6;margin:6px 0">같은 상황에서 <b>사람끼리</b> 방향이 몇 도 다른지 재서, 모델 방향 오차를 해석하는 기준으로 삼았다. 각 에피소드의 이동 프레임을 진행도(t/n) ±4% 안의 다른 에피소드 프레임과 비교했다. '
              '사람 합의 방향은 진행도·경로를 아는 오라클 예측기에 해당한다.</div>'
            + '<div style="margin:10px 0 4px;font-size:0.82rem;font-weight:800;color:#fcd34d">방향 차이(°) 비교</div>' + bars + tb
            + f'{st("해석", G)}<ul style="margin:6px 0 6px 18px;font-size:0.8rem">'
              f'<li><b>모델의 같은 목표 위치 오차({s["C_pi0_adaln"]["angle_deg"]:.1f}°)는 사람끼리의 차이({cell["pairwise_mean"]:.1f}°)와 같은 수준</b>이고, 진행도·경로를 아는 오라클 합의({cell["consensus_mean"]:.1f}°)와도 거의 같다. 즉 같은 목표 위치에서의 오프라인 오차는 <b>사람 조작 자체의 변동성 한계 근처</b>라서 더 줄이는 것은 의미가 작다.</li>'
              f'<li>안 본 목표 위치(LOPO {l["C_pi0_adaln"]["angle_deg"]:.1f}°)는 천장({cell["consensus_mean"]:.1f}°)보다 약 {l["C_pi0_adaln"]["angle_deg"] - cell["consensus_mean"]:.0f}° 높다 → <b>줄일 수 있는 여지는 일반화</b>에 있다.</li>'
              f'<li>경로가 다른 사람끼리는 {pos["pairwise_mean"]:.0f}°로 크게 갈린다. 모델은 경로를 입력으로 받지 않는데도 같은 위치 오차가 천장 수준이므로 <b>영상·bbox 이력에서 경로 의도를 읽고 있다</b>고 볼 수 있다(이 점은 별도 검증 필요).</li></ul>'
            + f'{st("한계", R)} 사람 일치도는 진행도 정렬(±4%)에 의존하는 근사이고, 같은 셀의 에피소드가 20개뿐이라 합의 방향이 불안정한 구간이 있다. 천장은 <b>하한 추정</b>이 아니라 "사람끼리 이 정도 차이"의 기준선이다.')

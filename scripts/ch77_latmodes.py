"""CH77 ⑥-13: 지연 보상과 다봉(경로 모호성) 점검. 데이터: latency_modes.json (FM-π0, 드롭아웃30%, 120ep, K=8, seed 0)."""
import json

import ch77_viz as Z
import gen_ch76_viz as V

FM = V.FM
A, G, R, B, P, MUT = V.A, V.G, V.R, V.B, V.P, V.MUT
SPN = {"fold0": "같은 목표 위치(fold0)", "lopo_strong_right": "안 본 목표 위치(strong_right)"}


def block(tbl, H, st):
    f = FM / "latency_modes.json"
    if not f.exists():
        return ""
    d = json.load(open(f))
    import ch77_anatomy as AN
    lat = ""
    for sp, v in d.items():
        L = v["latency"]
        lat += '<div style="flex:1 1 320px">' + AN.linechart([("보상: 청크 k번째 스텝 사용", A, "", [(int(k), x) for k, x in L["compensated"].items()]),
                                                              ("현재: 첫 스텝을 k프레임 늦게 사용", G, 'stroke-dasharray="5 3"', [(int(k), x) for k, x in L["stale"].items()])],
                                                             w=330, h=190, xmax=7, xlabel="지연(프레임, 젯슨 892ms ≈ 7)", ylabel="방향 오차(°)", title=SPN[sp]) + '</div>'
    mrows = []
    for sp, v in d.items():
        M = v["modes"]
        for key, lab in (("mean_of_k", "K=8 평균(현재 서빙)"), ("single", "샘플 1개"), ("chained", "모드 유지(이전 선택에 가까운 샘플)"), ("best_of_k", "가장 가까운 샘플(오라클, 상한)")):
            mrows.append((f"{SPN[sp].split('(')[0]} · {lab}", [[M[key]["first10"], M[key]["rest"]]], "low"))
    mt = Z.heat_table([("방향 오차(°)", ["에피소드 첫 10%", "나머지"], [R, B])], mrows, fmt="{:.1f}", corner="평가 · 방식", by="col")
    sp_t = tbl(["평가 세트", "샘플 8개 방향 퍼짐(최대 차, 첫 10% / 나머지)", "샘플이 45° 이상 갈린 프레임(첫 10% / 나머지)"],
               [[SPN[sp], f'{v["modes"]["spread_deg"]["first10"]:.0f}° / {v["modes"]["spread_deg"]["rest"]:.0f}°', f'{v["modes"]["split45_frac"]["first10"] * 100:.0f}% / {v["modes"]["split45_frac"]["rest"] * 100:.0f}%'] for sp, v in d.items()])
    f0 = d["fold0"]["modes"]
    return (H("⑥-13 지연 보상과 다봉(경로 모호성) 점검")
            + '<div style="font-size:0.8rem;line-height:1.6;margin:6px 0">두 가설을 오프라인으로 확인했다. ① 젯슨 지연(약 7프레임)만큼 뒤의 청크 스텝을 쓰면 더 맞는가. ② 같은 장면에서 사람이 좌커브/직진/우커브 중 하나를 고르는 <b>경로 모호성</b> 때문에 FM 샘플이 여러 방향으로 갈리는데, K 평균이 이를 뭉개고 있는가. 배포 설정(FM-π0, 드롭아웃 30%, 120 epoch, K=8), seed 1개.</div>'
            + '<div style="margin:10px 0 4px;font-size:0.84rem;font-weight:800;color:#fcd34d">① 지연 보상 — 청크 뒤쪽 스텝을 쓰면 오히려 나빠진다</div><div style="display:flex;gap:12px;flex-wrap:wrap">' + lat + '</div>'
            + f'{st("결론", R)} <b>가설 기각.</b> 7프레임 뒤를 예측한 스텝(보상)이 현재 예측을 7프레임 늦게 쓰는 것보다 오차가 크다(같은 위치 {d["fold0"]["latency"]["compensated"]["7"]:.1f}° vs {d["fold0"]["latency"]["stale"]["7"]:.1f}°). 사람 조작이 1초 안에 거의 바뀌지 않아 "늦은 현재 예측"이 더 정확하다. '
              '지연의 실제 비용은 방향이 아니라 <b>폐루프의 반응 지연(그 사이 이동 거리)</b>이므로 fp16 서빙 등 지연 자체를 줄이는 쪽이 맞다. 서빙은 청크 첫 스텝을 그대로 쓴다.'
            + '<div style="margin:14px 0 4px;font-size:0.84rem;font-weight:800;color:#fcd34d">② 다봉 — 에피소드 시작에서 샘플이 갈린다</div>' + Z.LEG + mt + sp_t
            + f'{st("결론", G)} <b>에피소드 시작 구간은 실제로 다봉이다.</b> 첫 10% 프레임의 {f0["split45_frac"]["first10"] * 100:.0f}%에서 샘플이 45° 이상 갈리고(나머지 {f0["split45_frac"]["rest"] * 100:.0f}%), 가장 가까운 샘플은 {f0["best_of_k"]["first10"]:.1f}°인데 K 평균은 {f0["mean_of_k"]["first10"]:.1f}°다. '
              '즉 <b>FM 헤드는 경로 모호성을 여러 모드로 표현하고 있고, K 평균이 그것을 중간 방향으로 뭉갠다.</b> 앞서 쓴 "데이터가 사실상 단봉이라 FM 다봉 이점이 안 보인다"(⑥-3)는 <b>시작 구간에 대해서는 틀린 결론</b>이었다(이웃 장면 방식은 시작 프레임이 적어 놓쳤음). '
              f'다만 이전 선택을 잇는 단순 모드 유지는 첫 선택이 사람의 경로와 다르면 계속 틀려서 개선이 없다({f0["chained"]["first10"]:.1f}°). '
              f'{st("한계", R)} 사람이 고른 경로만 정답으로 보는 방향 오차는 <b>다른 정당한 경로를 오답으로 친다</b>. 시작 구간에서 어느 모드를 고르는 것이 좋은지는 폐루프(성공·충돌)로만 판단할 수 있다.')

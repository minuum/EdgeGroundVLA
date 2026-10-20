"""CH77 ⑥-10: 시작 위치(pos1/2/3)별 서버 재생 결과 — 시작 위치 일반화(오프라인). 데이터: startpos_replay.json (없으면 빈 문자열)."""
import json

import ch77_viz as Z
import gen_ch76_viz as V

FM = V.FM
A, G, R = V.A, V.G, V.R
TG = ["strong_left", "weak_left", "center", "weak_right", "strong_right"]


def block(tbl, H, st):
    f = FM / "startpos_replay.json"
    if not f.exists():
        return ""
    d = json.load(open(f)); b = d["baseline"]; S = d["starts"]
    rows = []
    for k, v in S.items():
        s = v["summary"]
        rows.append([k, s["episodes"], f'{s["angle_deg_mean"]:.1f}°', f'{s["go_f1_mean"]:.3f}', f'{s["jitter_mean"]:.3f}', f'{s["server_ms_mean"]:.0f}ms', f'{s["subgoal_missing_frames"]}'])
    base = tbl(["시작 위치", "에피소드", "방향 오차(평균)", "GO/STOP F1", "저크", "서버 왕복", "서브골 결측 프레임"], rows)
    grp = [(k, ["방향 오차(°)"], [A]) for k in S]
    hrows = []
    for t in TG:
        vals = []
        for k, v in S.items():
            e = [x for x in v["episodes"] if x.get("pos") == t and x["angle_deg"] is not None]
            vals.append([sum(x["angle_deg"] for x in e) / len(e)] if e else ["—"])
        hrows.append((t, vals, "low"))
    heat = Z.heat_table(grp, hrows, fmt="{:.1f}°", corner="목표 위치 × 시작 위치", by="col")
    ref = tbl(["기준", "방향 오차", "GO/STOP F1", "저크", "비고"], [
        ["오프라인 5-fold(학습 시작 위치 pos1, 안 본 에피소드)", f'{b["fold0_angle"]:.1f}°', f'{b["fold0_f1"]:.3f}', f'{b["fold0_jitter"]:.3f}', "K=8, EMA 새 값 0.5, 같은 평활"],
        ["오프라인 LOPO strong_right(안 본 목표 위치)", f'{b["lopo_angle"]:.1f}°', f'{b["lopo_f1"]:.3f}', f'{b["lopo_jitter"]:.3f}', "목표 위치 일반화"],
        ["EGVLA1 실기 성공률(참고, 다른 지표)", "pos1 95 / pos2 91 / pos3 90 %", "—", "—", "학습은 pos1만, pos2·3은 제로샷"]])
    return (H("⑥-10 시작 위치별 서버 재생 — 시작 위치 일반화(오프라인)")
            + '<div style="font-size:0.8rem;line-height:1.6;margin:6px 0">로봇 쪽에서 pos2·pos3 시작으로 새로 수집한 에피소드를 서버(<code>replay_episodes_through_server.py</code>, JPEG q95, EMA 0.5)에 프레임 순서대로 재생하고 사람 조작과 비교했다. '
              '학습 데이터는 pos1 하나라 pos2·pos3는 <b>처음 보는 시작 위치</b>다(EGVLA1도 같은 조건).</div>'
            + Z.LEG + base + '<div style="margin:10px 0 4px;font-size:0.82rem;font-weight:800;color:#fcd34d">목표 위치 × 시작 위치별 방향 오차</div>' + heat
            + '<div style="margin:10px 0 4px;font-size:0.82rem;font-weight:800;color:#fcd34d">기준선</div>' + ref
            + f'{st("읽는 법", A)} pos1 기준선({b["fold0_angle"]:.1f}°)보다 얼마나 올라가는지가 시작 위치 일반화의 지표다. 안 본 목표 위치(LOPO {b["lopo_angle"]:.1f}°)와 비슷하면 "시작을 50cm 옮기는 것은 타깃이 화면에서 다른 위치에 보이는 것과 비슷한 정도의 어려움"으로 해석할 수 있다. '
              f'{st("한계", R)} 시작 위치당 에피소드가 5건뿐이라 목표 위치별 값은 참고용이며, 방향 오차는 성공률이 아니다(성공률은 폐루프에서).')

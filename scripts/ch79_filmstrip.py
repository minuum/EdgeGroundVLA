"""CH79 🎞 필름스트립·순서 점검: 프레임 순서(bbox·특징·액션·영상) 정렬 결과 + 학습 4개 에피소드와 실기 세션 전체를 가로 스크롤로. 자산: make_filmstrips.py, analyze_alignment.py"""
import json

import gen_ch76_viz as V

FM = V.FM
A, G, R, B, P, MUT = V.A, V.G, V.R, V.B, V.P, V.MUT
TX, GRID = "#e2e8f0", "#334155"


def chip(t, col):
    return f'<span style="display:inline-block;border:1px solid {col};background:{col}22;color:{col};border-radius:999px;padding:1px 9px;font-size:0.72rem;font-weight:700;white-space:nowrap">{t}</span>'


def lag_svg(lag, title, hi_key):
    ks = sorted(int(k) for k in lag); W, h, x0, bw = 330, 150, 34, 38
    o = [f'<svg viewBox="0 0 {W} {h + 28}" style="width:100%;max-width:{W}px" font-family="sans-serif"><text x="{W / 2}" y="14" fill="{TX}" font-size="12" text-anchor="middle" font-weight="700">{title}</text>']
    base = 30 + 0
    for k_i, k in enumerate(ks):
        v = lag[str(k)] if str(k) in lag else lag[k]; col = R if k == hi_key else B
        hh = abs(v) / 0.8 * 100; x = x0 + k_i * (bw + 6)
        o.append(f'<rect x="{x}" y="{base}" width="{bw}" height="{hh:.0f}" fill="{col}" fill-opacity="0.8" rx="3"/><text x="{x + bw / 2}" y="{base + hh + 12:.0f}" fill="{TX}" font-size="10" text-anchor="middle">{v:+.2f}</text>'
                 f'<text x="{x + bw / 2}" y="{h + 22}" fill="{MUT}" font-size="11" text-anchor="middle">k={k:+d}</text>')
    o.append("</svg>")
    return "".join(o)


def row(title_html, items):
    cells = []
    for it in items:
        cx = "소실" if it["cx"] is None else f"cx{it['cx']:.2f} 면적{it['area']:.3f}"
        dx = "" if it["dx"] is None else f" · 장면이동 {it['dx']:+.0f}px"
        cells.append(f'<div style="flex:0 0 320px"><a href="egvla2_fm/filmstrips/{it["f"]}" target="_blank"><img src="egvla2_fm/filmstrips/{it["f"]}" loading="lazy" style="width:320px;border-radius:4px;display:block"></a>'
                     f'<div style="font-size:0.66rem;color:#94a3b8;line-height:1.35;margin-top:2px">{cx}<br>lx{it["lx"]:+.2f} ly{it["ly"]:+.2f} az{it["az"]:+.2f}{dx}</div></div>')
    return f'<div style="margin:14px 0 4px;font-size:0.84rem">{title_html}</div><div style="display:flex;gap:6px;overflow-x:auto;padding-bottom:10px;border-bottom:1px solid #1e293b">{"".join(cells)}</div>'


def block(tbl, H, st):
    al1 = json.load(open(FM / "alignment_check_a1.json")); al2 = json.load(open(FM / "alignment_check_a2a3.json")); fs = json.load(open(FM / "filmstrips.json"))
    a1 = al1["A1_cache_alignment"]; a2 = al2["A2_train_action_vs_image_motion_corr_by_lag"]; a3 = al2["A3_live_action_vs_image_motion_corr_by_lag"]
    legend = ('<div style="font-size:0.84rem;line-height:1.9"><span style="color:#ffdc28;font-weight:700">━▶ 노랑</span> = 명령(위쪽 가로 화살표: 회전, +az 오른쪽=우회전 / 아래 중앙에서 뻗는 화살표: 이동, 위=전진·가로=ly(ly&lt;0이면 왼쪽으로 이동))<br>'
              '<span style="color:#3cdcff;font-weight:700">━▶ 청록</span> = 실제로 관측된 장면의 수평 이동(다음 프레임과 비교한 위상 상관). 로봇이 우회전하면 장면은 왼쪽(←)으로, 로봇이 왼쪽으로 가면 장면은 오른쪽(→)으로 움직임<br>'
              '<span style="color:#ff5050;font-weight:700">□ 빨강</span> = 실기 bbox, <span style="color:#50dc50;font-weight:700">□ 초록</span> = 학습 bbox. 각 사진 아래: cx·면적·명령값·장면 이동량(px@폭 320). 가로로 스크롤하면 처음부터 끝까지 볼 수 있고, 사진을 누르면 큰 원본(448×252)이 새 탭에 열림</div>')
    chk = tbl(["점검", "방법", "결과", "판정"], [
        [chip("A1 캐시 순서", B), "학습 프레임 42장의 bbox·특징을 다시 계산해 캐시[t], t±1, t±2와 비교", f"오프셋 0이 {a1['best_offset_counts']['0']}/{a1['n']}장(나머지 2장은 정지 구간으로 인접 프레임과 동일), 오프셋 0 bbox 오차 {a1['bbox_abs_err_by_offset']['0']:.4f} vs ±1 {a1['bbox_abs_err_by_offset']['1']:.3f}, 특징 코사인 {a1['feat_cos_by_offset']['0']:.4f} vs ±1 {a1['feat_cos_by_offset']['1']:.4f}", chip("정렬 정상", G)],
        [chip("A2 학습 액션↔영상", P), "사람 az[t+k]와 영상 회전량 dx(t→t+1)의 상관(70개 에피소드)", f"가장 강한 지연 k=−2({a2['-2']:+.2f}), k=−1({a2['-1']:+.2f}), k=0은 {a2['0']:+.2f}, k=+1은 {a2['1']:+.2f}", chip("액션이 영상보다 약 2프레임(≈0.26초) 앞섬", A)],
        [chip("A3 실기 액션↔영상", R), "SYNC 스텝의 az[i+k]와 dx(i→i+1)의 상관", f"k=0 {a3['0']:+.2f}, k=−1 {a3['-1']:+.2f}, k=+1 {a3['1']:+.2f}. H5 actions와 리포트 action 최대 차 {al2['A3_h5_vs_report_action_maxdiff']:.0e}", chip("정렬 정상(지연 0)", G)]])
    note = ('<div style="font-size:0.82rem;line-height:1.7;margin:6px 0">해석: <b>bbox·특징·영상의 순서는 맞다</b>(밀리거나 당겨진 것 없음). 다만 학습 데이터에서 <b>사람 명령은 영상 변화보다 약 2프레임(≈0.26초) 먼저</b> 기록돼 있다 — 명령의 효과(회전)가 영상에 나타나려면 구동기·카메라 지연이 필요하기 때문(정상 지연이며 버그는 아님). 즉 학습 프레임 t의 영상은 이미 직전 명령들(a[t−2], a[t−1])의 결과를 담고 있다(회전 중 번짐·진행 중인 이동). '
            '반면 실기 SYNC는 명령이 끝난 뒤 정지 상태에서 다음 프레임을 찍어 지연 0이다. 이 시간 구조 차이는 학습 입력(움직이는 중 영상)과 서빙 입력(정지 후 영상)의 차이로 이어질 수 있는 가설이며 영향은 측정하지 않았다.</div>')
    out = [V.wrap("프레임 순서 점검 결과", chk + note, "도구: scripts/analyze_alignment.py (A1은 .venv, A2·A3는 .venv_lerobot)", kind="table"),
           V.wrap("액션-영상 지연: 학습은 k=−2·−1에서 최대, 실기는 k=0", '<div style="display:flex;flex-wrap:wrap;gap:20px">' + lag_svg(a2, "학습: corr(az[t+k], 영상 dx(t→t+1))", -2) + lag_svg(a3, "실기 SYNC: corr(az[i+k], dx(i→i+1))", 0) + "</div>", "음수 = 우회전(+az)일 때 장면이 왼쪽(−dx)으로 이동. 빨강 = 최대 상관 지연", kind="chart"),
           V.wrap("화살표 읽는 법", legend, kind="flow"), H("학습(사람) 에피소드 4개 — 처음부터 끝까지")]
    for ep in fs["train"]:
        out.append(row(f'{chip("학습", G)} <b>{ep["cell"]}</b> · {len(ep["items"])}프레임', ep["items"]))
    out.append(H("실기 세션 — 처음부터 끝까지 (SYNC 16개, 저장 오류가 있는 세션 3개 제외: 194447·201157 정지 이미지/액션 0, 193630 0프레임)"))
    for s_ in fs["live"]:
        mode = chip(s_["mode"], P if s_["mode"] == "SYNC" else B)
        tg = (s_["target"] or "-").replace("trackA_", "")
        out.append(row(f'{chip("실기", A)} <b>{s_["sid"]}</b> {mode} {chip("목표 " + tg, MUT)} · {len(s_["items"])}스텝 · {s_["model"]}', s_["items"]))
    return "".join(out)

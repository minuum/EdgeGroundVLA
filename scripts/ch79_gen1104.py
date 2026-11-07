"""CH79 📅 탭의 11/04 2차 일반화(pos4~6 x G6~G8) 하위 블록. 수치는 gen_1104_summary.json·gen_1104_meta.json·filmstrips1104.json에서 읽는다(make_gen1104.py가 생성)."""
import json
import numpy as np
import gen_ch76_viz as V
from ch79_closedloop import chip
from ch79_gen1029 import strip as _strip

FM = V.FM
A, G, R, B, P, MUT = V.A, V.G, V.R, V.B, V.P, V.MUT


def strip(title_html, items):
    return _strip(title_html, items).replace("filmstrips1029", "filmstrips1104")


def block(tbl, H, st):
    rows = json.load(open(FM / "gen_1104_summary.json")); meta = json.load(open(FM / "gen_1104_meta.json")); fs = json.load(open(FM / "filmstrips1104.json"))
    n = len(rows); fcx = np.array([r["final_cx"] for r in rows]); far = np.array([r["final_area"] for r in rows])
    comb = []
    for p in ("pos4", "pos5", "pos6"):
        for t in ("G6", "G7", "G8"):
            m = [r for r in rows if r["start"] == p and r["target"] == t]
            comb.append([p, t, f"{len(m)} / 33 ({len(m)/33*100:.0f}%)", f"{sum(r['in_log'] for r in m)} · {sum(r['ok'] for r in m)} / {sum(not r['in_log'] for r in m)}", f"{np.mean([r['cx0'] for r in m]):.2f}", f"{np.median([r['final_cx'] for r in m]):.2f}",
                         f"{min(r['final_area'] for r in m):.2f}~{max(r['final_area'] for r in m):.2f}", f"{int(np.median([r['n'] for r in m]))}"])
    comb.append(["합계", "", f"{n} / 297 ({n/297*100:.1f}%)", f"{sum(r['in_log'] for r in rows)} · {sum(r['ok'] for r in rows)} / {sum(not r['in_log'] for r in rows)}", "", f"{np.median(fcx):.2f}", f"{far.min():.2f}~{far.max():.2f}", ""])
    tab = tbl(["출발", "목표(임시)", "H5 수 / 계획", "로그 일치 · 성공 / 로그 없음", "시작 cx 평균", "마지막 cx 중앙값", "마지막 면적 범위", "스텝 중앙값"], comb)
    nl = meta["log_without_h5"][0]
    m4 = [r for r in rows if r["sid"].startswith("20261104")]; m5 = [r for r in rows if r["sid"].startswith("20261105")]
    ar = lambda m: f"{np.median([r['final_area'] for r in m]):.3f}"; cm = lambda m: f"{np.median([r['final_cx'] for r in m]):.2f}"
    recon = tbl(["항목", "수", "설명"], [
        ["계획", "297", "pos4~6 × G6~G8 × 33회(로봇 쪽 계획). 속성(start_position·target_position)으로 선택한 H5가 9개 조합 모두 33개 = <b>297/297</b>"],
        ["파일명 11/04", f"{meta['n_h5_1104']}", f"episode_log 11/04 행 {meta['n_log']}건(전부 '성공', 종료방식 manual)과 세션 단위로 {meta['n_in_log']}건 일치. 모두 체크포인트 C_pi0_servolabel1.0_drop30_e120.pt, status manual_stop. H5 속성 pos·target과 로그 라벨 불일치 0"],
        ["파일명 11/05", f"{meta['n_h5_1105']}", "<b>나중에 추가된 45개</b>(pos6 × G7 22, pos6 × G8 22, pos4 × G7 1). 이 페이지의 앞선 판(252개 기준)에서는 이 45개가 빠져 있어 '252/297, pos6 G7·G8 11건'으로 적혀 있었음(정정)"],
        ["로그에 행이 있는 H5", f"{meta['n_in_log']} (운영자 '성공' {meta['h5_ok']})", "session_id로 매칭(날짜 아님)"],
        ["로그에 행이 없는 H5", f"{meta['n_not_in_log']} (전부 파일명 11/05)", "<b>로그 기록 없음(성공 판정 미확인)</b>. origin/monavla-driving의 episode_log.csv에서 session_id가 일치하는 행이 없음. 성공으로 가정하지 않고 따로 집계"],
        ["이상: 같은 라벨의 시작 시야 차이", "pos6×G8, pos4×G7", "pos6 × G8의 시작 cx 평균이 11/04분 0.62 대 11/05분 0.25, pos4 × G7의 11/05분 1건은 0.37(11/04분 0.61). 같은 (출발, 목표) 라벨인데 시작 시야가 서로 다르다. 물체 배치가 달랐는지는 데이터로 구분 못 함 → 조합별 시작 cx 평균은 두 묶음이 섞인 값"],
        ["H5가 없는 로그 행", "1", f"#{nl['no']} · 세션 {nl['sid']} · 경로 라벨 {nl['route']} · steps {nl['steps']} · {nl['t']} — 스텝 0인 저장 기록(계획에 없는 라벨). 로봇 쪽에 확인하지 않았음"],
    ])
    v = lambda tag: next(x for x in fs if tag in x["tag"] and x["tag"].startswith("이상치"))
    out = [H("5. 📆 11/04 2차 일반화 — 출발 pos4~6 × 임시 목표 G6~G8 (H5 297회)"),
           V.wrap("수집 현황과 로그·H5 대조", tab, "성공 = 로봇 쪽 episode_log 결과 열(운영자 수동 판정, 블라인드 아님, 문서화된 성공 기준 없음, 좌우 정렬은 판정에 안 들어감). 모델은 모두 A3, 같은 체크포인트. H5 기준 타깃 소실 프레임 0. <b>계획 297회 중 252회만 수집</b>(pos6 × G7·G8 미완료) → 아래 수치는 297회에 대한 것이고 pos6 두 조합은 n=11이라 해석 폭이 좁다", kind="table"),
           V.wrap("로그 253행 vs H5 297개 — 정직한 대조", recon, "로그 253행 중 252행이 H5와 일치하고, 남은 한 행은 스텝 0짜리 빈 저장이다. 반대로 H5 297개 중 45개(11/05분)는 로그에 없다. 참고: 로그 steps는 H5 프레임 수보다 대체로 1 크다(252개 중 175개) — 집계 방식 차이로 보이며 영향은 확인하지 않음", kind="table"),
           V.wrap("건수 · 시작 cx → 마지막 cx · 마지막 면적(11/04·11/05, 297회)", V.img("figures/ch79/generalization_1104.png", f"왼쪽: 조합별 수집률(계획 33회 대비 %, 9개 조합 모두 100%). 가운데: 시작 cx에 따라 마지막 cx가 중앙(점선) 쪽으로 모이는 정도. 오른쪽: 마지막 면적 {far.min():.2f}~{far.max():.2f}, 소실 0. 10/29(0.22~0.41)와 거의 같은 범위(11/05분에서 최대 0.46 한 건). 11/05분의 마지막 면적 중앙값은 {ar(m5)}로 11/04분 {ar(m4)}보다 크고 마지막 cx 중앙값도 {cm(m5)} 대 {cm(m4)}로 다르다 — 원인은 데이터로 구분 못 함. 성공 기준이 '근접·정지'이므로 '일반화에 성공했다'고 확정하는 근거가 되지 못하고, 단 이 범위에서 H5상 타깃을 놓친 세션은 없었다는 사실만 말할 수 있다"), kind="img"),
           V.wrap("시작 방향 확인과 마지막 정렬(퍼센트)", V.img("figures/ch79/counts_1104.png", f"왼쪽: 시작 cx가 좌(<0.4)·우(>0.6)인 237회 중 첫 스텝 횡이동 명령(ly)이 바구니 쪽(ly<0 = 왼쪽)인 비율 99.2%(235/237). 예외 2건(20261104_111656 시작 cx 0.77, 20261104_112506 시작 cx 0.13)은 첫 명령 ly가 0.0이라 반대 방향이 아니라 '횡이동 없음'이다. 오른쪽: 마지막 cx는 중앙 ±0.1 안 {149/297*100:.0f}%, 0.6 이상 {106/297*100:.0f}%, 0.4 이하 {42/297*100:.0f}%(중앙값 {np.median(fcx):.2f}). 10/29(중앙값 0.58)보다 중앙에 가깝지만 오른쪽 치우침 경향은 남음. 첫 스텝 방향이 맞는다는 것은 시작 반응일 뿐 정지 위치의 좌우 정렬이나 실기 성공을 예측하지 않는다"), kind="img"),
           V.wrap("마지막 프레임 사진 — 이상치와 가장 중앙인 건(원본 RGB)", V.img("figures/ch79/alignment_1104.png", f"마지막 면적 최소·cx 0.83(20261104_162044): 바구니가 화면 오른쪽 가장자리에 반쯤 걸린 채 정지 — 성공으로 기록됐지만 정렬 기준으로는 이상치. 마지막 면적 최대 0.46(20261105_184459, 로그 없음), cx 최좌측 0.28(153434), 스텝 최다 18(113255)도 함께 표시. 이상치 선택은 지표 극값 기준이며 나머지 세션을 대표하지 않음"), kind="img"),
           V.wrap("읽는 법과 선택 규칙 (11/04)", '<div style="font-size:0.82rem;line-height:1.8">10/29와 같은 필름스트립 규약(노랑 = 명령, 청록 = 관측 장면 이동, 빨강 □ = 모델이 잡은 바구니). <b>선택 규칙</b>: (출발, 목표) 9개 조합마다 프레임 수 중앙값에 가까운 세션 1개(결과를 보고 고르지 않음, 동률은 이른 시각) + 지표 극값 이상치 4개(마지막 면적 최대·최소, 마지막 cx 최좌측, 스텝 최다; cx 최우측은 면적 최소와 같은 세션). 로그에 행이 있는 세션은 운영자 판정 성공, 11/05분은 판정 미확인(제목에 표시). 사진은 로봇 카메라 원본 RGB.</div>', kind="flow")]
    for s_ in fs:
        tg = s_["target"]; ol = s_["tag"].startswith("이상치")
        extra = f" · {s_['tag']} (마지막 cx {s_['final_cx']:.2f}, 면적 {s_['final_area']:.3f})" if ol else ""
        nolog = next(r for r in rows if r["sid"] == s_["sid"])["in_log"] is False
        extra += " · 로그 기록 없음(성공 판정 미확인)" if nolog else ""
        out.append(strip(f'{chip("11/05" if s_["sid"].startswith("20261105") else "11/04", R if ol else G)} <b>{s_["sid"]}</b> {chip(s_["start"], P)} {chip("목표 " + tg, MUT)} · {len(s_["items"])}스텝{extra}', s_["items"]))
    return "".join(out)

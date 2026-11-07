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
            comb.append([p, t, f"{len(m)} / 33 ({len(m)/33*100:.0f}%)", f"{sum(r['ok'] for r in m)}/{len(m)}", f"{np.mean([r['cx0'] for r in m]):.2f}", f"{np.median([r['final_cx'] for r in m]):.2f}",
                         f"{min(r['final_area'] for r in m):.2f}~{max(r['final_area'] for r in m):.2f}", f"{int(np.median([r['n'] for r in m]))}"])
    comb.append(["합계", "", f"{n} / 297 ({n/297*100:.1f}%)", f"{sum(r['ok'] for r in rows)}/{n}", "", f"{np.median(fcx):.2f}", f"{far.min():.2f}~{far.max():.2f}", ""])
    tab = tbl(["출발", "목표(임시)", "H5 수집 / 계획", "운영자 성공(로그)", "시작 cx 평균", "마지막 cx 중앙값", "마지막 면적 범위", "스텝 중앙값"], comb)
    nl = meta["log_without_h5"][0]
    recon = tbl(["항목", "수", "설명"], [
        ["계획", "297", "pos4~6 × G6~G8 × 33회(로봇 쪽 계획)"],
        ["episode_log 11/04 행", f"{meta['n_log']} (전부 '성공', 종료방식 manual)", "검증 스크리닝(L2 저장) 메모. 로봇 쪽 문서: 정리 후 253/253"],
        ["이 저장소의 H5", f"{meta['n_h5']}", "모두 체크포인트 C_pi0_servolabel1.0_drop30_e120.pt, status manual_stop. H5 속성 pos·target과 로그 라벨은 세션 단위로 전부 일치(불일치 0)"],
        ["H5가 없는 로그 행", "1", f"#{nl['no']} · 세션 {nl['sid']} · 경로 라벨 {nl['route']} · steps {nl['steps']} · {nl['t']} — <b>스텝 0인 저장 기록</b>(trackF_center는 이번 pos4~6×G6~8 계획에 없는 라벨). 시도 데이터가 아니라 빈 저장으로 보이나, 로봇 쪽에 확인하지 않았음"],
        ["로그 없는 H5", "0", "H5 252개 전부 로그에 있음"],
        ["계획 대비 미수집", "45 = 297 − 252", "pos6 × G7, pos6 × G8이 각각 11/33만 수집됨(미완료 상태), pos4 × G7 32/33. 로봇 쪽이 의도적으로 지운 오염 샘플(커밋 6f8f2079)이 있다고 문서화했지만, 이 45건 중 몇 건이 삭제분인지는 데이터로 구분 못 함"],
    ])
    v = lambda tag: next(x for x in fs if tag in x["tag"] and x["tag"].startswith("이상치"))
    out = [H("5. 📆 11/04 2차 일반화 — 출발 pos4~6 × 임시 목표 G6~G8 (H5 252회)"),
           V.wrap("수집 현황과 로그·H5 대조", tab, "성공 = 로봇 쪽 episode_log 결과 열(운영자 수동 판정, 블라인드 아님, 문서화된 성공 기준 없음, 좌우 정렬은 판정에 안 들어감). 모델은 모두 A3, 같은 체크포인트. H5 기준 타깃 소실 프레임 0. <b>계획 297회 중 252회만 수집</b>(pos6 × G7·G8 미완료) → 아래 수치는 252회에 대한 것이고 pos6 두 조합은 n=11이라 해석 폭이 좁다", kind="table"),
           V.wrap("로그 253행 vs H5 252개 — 정직한 대조", recon, "한 건 차이는 스텝 0짜리 빈 저장 행이다. 이 한 건을 제외하면 로그와 H5는 세션 단위로 일치한다. 참고: 로그 steps는 H5 프레임 수보다 대체로 1 크다(252개 중 175개) — 집계 방식 차이로 보이며 영향은 확인하지 않음", kind="table"),
           V.wrap("건수 · 시작 cx → 마지막 cx · 마지막 면적(11/04, 252회)", V.img("figures/ch79/generalization_1104.png", f"왼쪽: 조합별 수집률(계획 33회 대비 %). 가운데: 시작 cx에 따라 마지막 cx가 중앙(점선) 쪽으로 모이는 정도. 오른쪽: 마지막 면적 {far.min():.2f}~{far.max():.2f}, 소실 0. 10/29(0.22~0.41)와 같은 범위. 성공 기준이 '근접·정지'이므로 '일반화에 성공했다'고 확정하는 근거가 되지 못하고, 단 이 범위에서 H5상 타깃을 놓친 세션은 없었다는 사실만 말할 수 있다"), kind="img"),
           V.wrap("시작 방향 확인과 마지막 정렬(퍼센트)", V.img("figures/ch79/counts_1104.png", f"왼쪽: 시작 cx가 좌(<0.4)·우(>0.6)인 198회 중 첫 스텝 횡이동 명령(ly)이 바구니 쪽(ly<0 = 왼쪽)인 비율 99.0%(196/198). 예외 2건(20261104_111656 시작 cx 0.77, 20261104_112506 시작 cx 0.13)은 첫 명령 ly가 0.0이라 반대 방향이 아니라 '횡이동 없음'이다. 오른쪽: 마지막 cx는 중앙 ±0.1 안 52%, 0.6 이상 34%, 0.4 이하 15%(중앙값 {np.median(fcx):.2f}). 10/29(중앙값 0.58)보다 중앙에 가깝지만 오른쪽 치우침 경향은 남음. 첫 스텝 방향이 맞는다는 것은 시작 반응일 뿐 정지 위치의 좌우 정렬이나 실기 성공을 예측하지 않는다"), kind="img"),
           V.wrap("마지막 프레임 사진 — 이상치와 가장 중앙인 건(원본 RGB)", V.img("figures/ch79/alignment_1104.png", f"마지막 면적 최소·cx 0.83(20261104_162044): 바구니가 화면 오른쪽 가장자리에 반쯤 걸린 채 정지 — 성공으로 기록됐지만 정렬 기준으로는 이상치. 마지막 면적 최대 0.41(153338), cx 최좌측 0.28(153434), 스텝 최다 18(113255)도 함께 표시. 이상치 선택은 지표 극값 기준이며 나머지 세션을 대표하지 않음"), kind="img"),
           V.wrap("읽는 법과 선택 규칙 (11/04)", '<div style="font-size:0.82rem;line-height:1.8">10/29와 같은 필름스트립 규약(노랑 = 명령, 청록 = 관측 장면 이동, 빨강 □ = 모델이 잡은 바구니). <b>선택 규칙</b>: (출발, 목표) 9개 조합마다 프레임 수 중앙값에 가까운 세션 1개(결과를 보고 고르지 않음, 동률은 이른 시각) + 지표 극값 이상치 4개(마지막 면적 최대·최소, 마지막 cx 최좌측, 스텝 최다; cx 최우측은 면적 최소와 같은 세션). 모두 운영자 판정 성공. 사진은 로봇 카메라 원본 RGB.</div>', kind="flow")]
    for s_ in fs:
        tg = s_["target"]; ol = s_["tag"].startswith("이상치")
        extra = f" · {s_['tag']} (마지막 cx {s_['final_cx']:.2f}, 면적 {s_['final_area']:.3f})" if ol else ""
        out.append(strip(f'{chip("11/04", R if ol else G)} <b>{s_["sid"]}</b> {chip(s_["start"], P)} {chip("목표 " + tg, MUT)} · {len(s_["items"])}스텝{extra}', s_["items"]))
    return "".join(out)

"""CH79 📅 10/29 위치 일반화 탭: 용어 정의(pos·G·단계)·날짜별 시도 정리·96회 결과·성공 사례 필름스트립. 수치는 gen_1029_summary.json·filmstrips1029.json에서 읽는다.
자산: ch79_generalization_figs.py, ch79_alignment_figs.py, make_filmstrips_1029.py"""
import json

import numpy as np

import gen_ch76_viz as V
from ch79_closedloop import chip

FM = V.FM
A, G, R, B, P, MUT = V.A, V.G, V.R, V.B, V.P, V.MUT


def strip(title_html, items):
    cells = []
    for it in items:
        cx = "소실" if it["cx"] is None else f"cx{it['cx']:.2f} 면적{it['area']:.3f}"
        dx = "" if it["dx"] is None else f" · 장면이동 {it['dx']:+.0f}px"
        cells.append(f'<div style="flex:0 0 320px"><a href="egvla2_fm/filmstrips1029/{it["f"]}" target="_blank"><img src="egvla2_fm/filmstrips1029/{it["f"]}" loading="lazy" style="width:320px;border-radius:4px;display:block"></a>'
                     f'<div style="font-size:0.66rem;color:#94a3b8;line-height:1.35;margin-top:2px">{cx}<br>lx{it["lx"]:+.2f} ly{it["ly"]:+.2f} az{it["az"]:+.2f}{dx}</div></div>')
    return f'<div style="margin:14px 0 4px;font-size:0.84rem">{title_html}</div><div style="display:flex;gap:6px;overflow-x:auto;padding-bottom:10px;border-bottom:1px solid #1e293b">{"".join(cells)}</div>'


def block(tbl, H, st):
    rows = json.load(open(FM / "gen_1029_summary.json")); fs = json.load(open(FM / "filmstrips1029.json"))
    # 2단계 G6~G8 시작 cx 평균(pos별) — 정의 표에 실측으로 넣는다
    g68 = {}
    for r in rows:
        if r["stage"] == 2: g68.setdefault((r["start"], r["target"]), []).append(r["cx0"])
    g68txt = " / ".join(f"{p}: " + ", ".join(f"{g} {np.mean(g68[(p, g)]):.2f}" for g in ("G6", "G7", "G8")) for p in ("pos1", "pos2", "pos3"))
    defs = tbl(["용어", "뜻", "근거·상태"], [
        [chip("pos1 · pos2 · pos3", B), "로봇의 <b>시작 위치</b>. 2m×2m 맵에서 pos1 = (−1, 0) 기본, pos2 = pos1에서 왼쪽 50 cm, pos3 = 오른쪽 50 cm", "프로젝트 정의. <b>EGVLA2 학습 데이터(300건)는 pos1 하나에서 수집</b> → pos2·pos3은 학습에 쓰지 않은 시작 위치"],
        [chip("pos4 · pos5 · pos6", P), "10/29에 추가한 <b>새 시작 위치</b> 3곳(로봇 쪽 표기 '6개 물리 시작 위치' 중 신규). 계획상 pos2~pos3 사이(가로)로 정하기로 했음", f"<b>좌표 기록 없음</b>(로봇 쪽 확인). 같은 목표 라벨의 시작 cx가 pos마다 다름(pos4 중앙 라벨 0.55, pos6 0.28)으로 시야가 다름만 확인"],
        [chip("G1 ~ G5", G), "<b>목표(바구니)의 화면상 위치</b> 5곳 = 기존 <code>cx_position</code>: G1 strong_left(강좌) · G2 weak_left(약좌) · G3 center(중앙) · G4 weak_right(약우) · G5 strong_right(강우)", "pos1 시작 때 바구니 cx(10/28 A3): 0.12 / 0.21 / 0.48 / 0.74 / 0.85. G2~G4 대응은 로봇이 적은 라벨 교체 목록에서 확인, G1·G5는 같은 순서의 연장"],
        [chip("G6 · G7 · G8", R), "2단계 <b>임시 목표 위치</b> 3곳(강좌와 강우 사이로 하기로 함). 물체를 놓은 뒤 기존 G2·G3·G4 버튼으로 기록하고 시험 후 라벨만 G6·G7·G8로 교체", f"<b>바닥 좌표 기록 없음</b>. 시작 cx(평균) — {g68txt}. 10/28 5곳 범위(0.03~0.94)와 겹침"],
        [chip("1단계", G), "시작 pos4·5·6 × 목표 G1~G5(기존 5곳) × 4회 = 60회", "교수님 방침 ①: 출발 위치를 바꾸고 목표는 고정"],
        [chip("2단계", B), "시작 pos1·2·3 × 목표 G6·G7·G8 × 4회 = 36회", "교수님 방침 ②: 출발 위치는 기존, 목표 위치를 바꿈"],
        [chip("3단계", MUT), "출발 3곳 × 물체 3곳 모두 랜덤(횟수 미정)", "아직 안 함. 판정 기준·블라인드·횟수를 먼저 정해야 함"],
        [chip("성공", A), "로봇 쪽 운영자가 저장 시점에 화면에서 '바구니 근접·정지'를 보고 수동 기록(<code>episode_log</code> 결과 열)", "문서화된 기준 없음, 블라인드 아님, 좌우 정렬은 판정에 안 들어감"],
    ])
    timeline = tbl(["날짜", "무엇을 했나", "결과(출처)", "해석"], [
        ["10/22~23", "A0(기준) 폐루프 점검, SYNC", "정지 위치 불량·타깃 소실 다수(CH79 🔬 탭)", "원인 후보: 서빙 쪽 지연·보정, 모델의 bbox 무반응"],
        ["10/28", "pos1 × 목표 5곳. A0 0/1, A1(서버 서보 혼합 0.5) 11/23, A2(서보 라벨 0.5) 14/20, <b>A3(서보 라벨 1.0) 63/63</b>", "수동 성공 기록 + H5 교차검증(📍 탭 5·5-2절)", "서보 라벨 비중을 키울수록 좋아짐. A3 채택. 오프라인 사람 방향 오차는 A3에서 오히려 +22°"],
        ["10/28~29", "일반화에 도움 될 만한 오프라인 시도: E2E DaViT LoRA, 월드모델 보조 손실, 풀링 비교(feature-gap)", "E2E 27.7~33.1° 대 기존 19.6°(개선 없음) · 월드모델 S 미개선으로 중단 · feature-gap R1 19.1°, 3×3 격자 풀링 16.1°(−3.0°, 경계)·S 미개선", "오프라인 지표는 실기와 일치하지 않음. 일반화 개선 근거로 쓰지 않음"],
        ["10/29", "<b>1단계</b>: 새 출발 pos4~6 × 기존 목표 5곳 60회", "운영자 판정 60/60 · H5 소실 0 · 마지막 면적 0.22~0.41", "출발 위치를 바꿔도 같은 모델이 근접·정지에 도달"],
        ["10/29", "<b>2단계</b>: 기존 pos1~3 × 임시 목표 G6~8 36회", "운영자 판정 36/36 · H5 소실 0 · 면적 0.22~0.36", "화면상 위치가 기존 범위와 겹쳐 '학습 밖'이라고 단정 못 함"],
        ["미정", "3단계(출발·목표 모두 랜덤), 판정 기준·블라인드 확정 후", "—", "교수님께 횟수와 정렬 포함 여부 확인 필요"],
    ])
    out = [H("1. 용어 정의 — pos·G·단계"), V.wrap("pos와 G가 가리키는 것", defs, "pos = 출발 위치(로봇이 어디서 시작), G = 목표 위치(바구니가 화면 어디에 보이는가). 학습 때는 pos1 하나, G1~G5 다섯 곳. 로봇 쪽은 H5 속성 <code>start_position</code>·<code>target_position</code>을 시험 뒤 일괄 패치했다(처음에는 모두 pos1로 저장).", kind="table"),
           H("2. 날짜별 시도 — 성공 사례와 일반화를 위한 시도"), V.wrap("10/22부터 10/29까지", timeline, "성공 = 수동 기록. 모든 실기 수치는 ASYNC·EMA 0.0, 모델은 10/28의 A3 이후 동일(<code>C_pi0_servolabel1.0_drop30_e120.pt</code>, sha256 5248ea15…)", kind="table"),
           H("3. 10/29 일반화 96회 결과"),
           V.wrap("1·2단계 구성과 결과(성공 = 로봇 쪽 episode_log 결과 열, 운영자 수동 판정)", _gen_tab(tbl), "모델은 모두 A3, ASYNC, EMA 0.0. 100%여서 변별력이 낮고(두 날 모두 전수 성공), 판정은 블라인드가 아니며 문서화된 성공 기준은 없었다(운영자가 저장 시점에 화면에서 '바구니 근접·정지'를 보고 수동 판정, 자동 정지 규칙 near_area 0.25와는 별개). 로봇 쪽도 '판정이 느슨했을 가능성'을 열어 두었다", kind="table"),
           V.wrap("시작 시야와 마지막 프레임", V.img("figures/ch79/generalization_1029.png", "왼쪽: 새 출발 위치(pos4~6)와 임시 목표 G6~8의 시작 cx는 10/28 pos1 목표 5곳의 범위와 겹친다 → 화면상으로는 '학습 분포 밖'이라고 주장할 근거가 없다. 오른쪽: 타깃 소실 0, 면적 0.22~0.41. 0.41 한 건(200929)과 cx 0.7 이상 6건은 로봇 쪽이 영상으로 재판정해 7건 모두 '근접 성공'으로 유지했고(200618만 '중앙 정렬이 기준이면 재검토'), 같은 3건을 직접 비교해도 바구니가 오른쪽으로 치우쳐 프레임 가장자리에 걸린 모습이 같다. 즉 판정은 '근접·정지'이고 좌우 정렬은 보지 않았다. CSV의 FPE·area·cx·gnd% 열은 기본값(0, 0, 0.5, 100)이라 쓰지 않음"), kind="img"),
           V.wrap("좌우 정렬 여부 — 마지막 프레임 사진(10/29 96회)", V.img("figures/ch79/alignment_1029.png", "96회 마지막 cx 중앙값 0.58(범위 0.35~0.76). 중앙 ±0.1 안 48회, 0.6 이상 42회로 오른쪽에 치우쳐 멈추는 경향이 있다. 재판정 7건 중 3건을 중앙에 가까운 건·왼쪽 건과 나란히 놓았다. 성공 기준이 '근접·정지'라 이 치우침은 성공 판정에 반영되지 않았다. 카메라 장착 위치 때문인지 모델 때문인지는 미구분"), kind="img"),
           H("4. 성공 사례 필름스트립 — 처음부터 끝까지 (10/29)"),
           V.wrap("읽는 법과 선택 규칙", '<div style="font-size:0.82rem;line-height:1.8"><span style="color:#ffdc28;font-weight:700">━▶ 노랑</span> = 명령(위쪽 가로 화살표: 회전, 아래 중앙에서 뻗는 화살표: 이동) · <span style="color:#3cdcff;font-weight:700">━▶ 청록</span> = 관측된 장면 수평 이동 · <span style="color:#ff5050;font-weight:700">□ 빨강</span> = 모델이 잡은 바구니(bbox). 사진 아래 = cx·면적·명령값·장면 이동량. 가로 스크롤, 사진을 누르면 원본.<br><b>선택 규칙</b>: 시작 위치(pos1~6)마다 프레임 수 중앙값에 가까운 세션 1개(결과를 보고 고르지 않음) + 재판정 대상 200618. 모두 운영자 판정 성공.</div>', kind="flow")]
    for s_ in fs:
        stg = "1단계" if s_["stage"] == 1 else "2단계"; tg = s_["target"].replace("trackA_", "").replace("trackF_", "")
        extra = " · 재판정 대상(최종 cx 0.70, 바구니 우측)" if s_["sid"].endswith("200618") else ""
        out.append(strip(f'{chip(stg, G if s_["stage"] == 1 else B)} <b>{s_["sid"]}</b> {chip(s_["start"], P)} {chip("목표 " + tg, MUT)} · {len(s_["items"])}스텝{extra}', s_["items"]))
    return "".join(out)


def _gen_tab(tbl):
    return tbl(["단계", "출발 위치", "목표 위치(라벨)", "회수", "수동 성공", "H5 소실·마지막 면적"], [
        ["1", "pos4·5·6 (새 출발 위치)", "10/28과 같은 5곳(G1~G5 = trackA 4종 + 중앙)", "60 (15조합 × 4)", "60/60 (구간 94~100%)", "소실 0 · 0.22~0.41"],
        ["2", "pos1·2·3 (기존 출발 위치)", "임시 위치 G6~G8 (좌표 미확정, G2·G3·G4 버튼으로 기록 후 라벨 교체)", "36 (9조합 × 4)", "36/36 (구간 90~100%)", "소실 0 · 0.22~0.36"],
    ])

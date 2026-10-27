"""CH79 📍 현재 상황(2026-10-28): 진행 상태·신뢰도 구분(확정/가설/정정)·지표 트레이드오프·E2E 일반 비교·로봇 대기·결정 필요. 수치는 JSON 원본에서 읽는다."""
import json
import os
import re
from pathlib import Path

import numpy as np

import gen_ch76_viz as V
from ch79_closedloop import chip, TEAL

FM = V.FM
A, G, R, B, P, MUT = V.A, V.G, V.R, V.B, V.P, V.MUT
TX, GRID = "#e2e8f0", "#334155"
ROOT = Path(__file__).resolve().parent.parent


def load(name):
    f = FM / name
    return json.load(open(f)) if f.exists() else {}


def avg(d, var, key, live=False, splits=None):
    xs = []
    for s, v in d.items():
        if splits and s not in splits:
            continue
        if var in v:
            r = v[var]["live"] if live else v[var]
            x = r.get(key)
            if x is not None:
                xs.append(x)
    return float(np.mean(xs)) if xs else None


def scatter(points):
    W, x0, y0, pw, ph = 760, 54, 16, 670, 300
    xs = [p[1] for p in points]; ys = [p[2] for p in points]
    X = lambda v: x0 + (v - 12) / (36 - 12) * pw
    Y = lambda v: y0 + (22 - v) / (22 + 8) * ph
    o = [f'<svg viewBox="0 0 {W} {ph + 54}" style="width:100%;max-width:{W}px" font-family="sans-serif">']
    for v in (-5, 0, 5, 10, 15, 20):
        o.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x0 + pw}" y2="{Y(v):.1f}" stroke="{GRID}" stroke-dasharray="2 4"/><text x="{x0 - 6}" y="{Y(v) + 4:.1f}" fill="{MUT}" font-size="11" text-anchor="end">{v}°</text>')
    for v in (12, 16, 20, 24, 28, 32, 36):
        o.append(f'<text x="{X(v):.1f}" y="{y0 + ph + 16}" fill="{MUT}" font-size="11" text-anchor="middle">{v}°</text>')
    o.append(f'<line x1="{x0}" y1="{Y(10):.1f}" x2="{x0 + pw}" y2="{Y(10):.1f}" stroke="{G}" stroke-dasharray="6 4" opacity="0.7"/><text x="{x0 + pw - 4}" y="{Y(10) - 4:.1f}" fill="{G}" font-size="11" text-anchor="end">목표 S ≥ +10°</text>')
    o.append(f'<text x="{x0 + pw / 2}" y="{y0 + ph + 36}" fill="{MUT}" font-size="11.5" text-anchor="middle">사람 방향 오차 (°, 낮을수록 사람과 같음) →   세로: 타깃 반응 S (°, 높을수록 타깃을 따라감) ↑</text>')
    for lab, x, y, col, hollow in points:
        o.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="7" fill="{"none" if hollow else col}" fill-opacity="0.85" stroke="{col}" stroke-width="2"/><text x="{X(x) + 10:.1f}" y="{Y(y) + 4:.1f}" fill="{TX}" font-size="10.5">{lab}</text>')
    o.append("</svg>")
    return "".join(o)


def e2e_bars(d):
    arms = [("A 고정 융합(기존)", "A_frozen_fused", B), ("B0 고정 DaViT 120ep", "B0_frozen_davit_120ep", MUT), ("B0 고정 DaViT 20ep", "B0_frozen_davit_20ep", "#64748b"), ("B1 마지막 단계 LoRA 20ep", "B1_lora_last_stage_20ep", A)]
    splits = ["fold0", "lopo_center", "lopo_strong_right"]; W, x0, bw = 760, 190, 400
    o = [f'<svg viewBox="0 0 {W} {len(arms) * 34 + 56}" style="width:100%;max-width:{W}px" font-family="sans-serif">']
    for k, (s, c) in enumerate(zip(["fold0(같은 위치)", "center(안 본)", "strong_right(안 본)"], [0.95, 0.65, 0.4])):
        o.append(f'<rect x="{x0 + k * 150}" y="6" width="12" height="12" fill="{TX}" fill-opacity="{c}"/><text x="{x0 + k * 150 + 17}" y="16" fill="{TX}" font-size="11.5">{s}</text>')
    for i, (lab, key, col) in enumerate(arms):
        y = 30 + i * 34; o.append(f'<text x="8" y="{y + 18}" fill="{TX}" font-size="12.5">{lab}</text>')
        for k, s in enumerate(splits):
            v = d[s][key]["dir_err"]; w = v / 36 * bw
            o.append(f'<rect x="{x0}" y="{y + k * 9}" width="{w:.0f}" height="8" rx="2" fill="{col}" fill-opacity="{[0.95, 0.65, 0.4][k]}"/><text x="{x0 + w + 5:.0f}" y="{y + k * 9 + 8}" fill="{TX}" font-size="10">{v:.1f}°</text>')
    o.append(f'<text x="{x0}" y="{len(arms) * 34 + 48}" fill="{MUT}" font-size="11">사람 방향 오차(°, 낮을수록 좋음). 같은 6프레임 간격 프레임·같은 분할·같은 라벨·같은 지표</text></svg>')
    return "".join(o)


def block(tbl, H, st):
    gs, e1, e1b, tc, e2e, fs = load("gap_suite.json"), load("gap_e1.json"), load("gap_e1b.json"), load("target_cond.json"), load("e2e_general.json"), load("e2e_fullstage.json")
    chain = ""
    lg = ROOT / "runs/e2e_cache/logs/chain.log"
    if lg.exists():
        chain = lg.read_text().strip().splitlines()[-1]
    fs_prog = ""
    flog = ROOT / "runs/e2e_cache/logs/fullstage.log"
    if flog.exists():
        ls = [l for l in flog.read_text().splitlines() if "B2 epoch" in l or "done" in l]
        fs_prog = ls[-1] if ls else ""
    status = tbl(["영역", "상태", "내용", "다음"], [
        [chip("완료", G), "오프라인 학습·평가", "헤드 비교, 서브골, 이산/연속, 데이터량, 서빙 파라미터·학습 설정·보조 목표 ablation(200회 이상), 간극 지표 5종, 서보 라벨, E1/E1b", "—"],
        [chip("완료", G), "실기 드라이런 분석(10/22)", "유효 SYNC 16세션(저장 오류 3개 제외): 시작 회전 누락·부호·컨트롤러 환경·타깃 서보 부재·전처리 동등·순서 정렬", "—"],
        [chip("완료", G), "서빙 수정", "회전 상태, 부호 정정(+az=우회전), 크기 기반 정지 래치 끔, 타깃 서보 환경변수(기본 끔)", "로봇 pull·재시작 확인 대기"],
        [chip("완료", G), "E2E 일반 비교(마지막 단계 LoRA)", "같은 조건 3개 분할 비교 완료 — 결과는 아래", "—"],
        [chip("진행 중", B), "E2E 일반 비교(전 단계 LoRA @384)", f"분할 3개 × 8에폭(에폭당 약 10분). {fs_prog or '시작됨'}", "약 10/28 05:30 종료 예상"],
        [chip("대기", A), "로봇 폐루프 비교 팔", "체크포인트 5종·환경변수·기록 항목 전달 문서 작성(미전달 가능성: 사용자 전달 여부 미확인)", "로봇 회신·승인"],
        [chip("대기", A), "pos2·pos3 드라이런 10건", "`ROS_action/dryrun_startpos/` 미수신. 서버 재부팅(10/27 14:27)으로 자동 반영 스크립트 소실 → 수동 확인 필요", "수신 후 분석"],
        [chip("미측정", R), "실기 폐루프 성공률", "3 시작 위치 × 5 목표 위치 × 20 = 300회 미실시. EGVLA1 기준선 92.0%(276/300)", "최종 목표"],
    ])
    trust = tbl(["구분", "항목", "근거"], [
        [chip("확정(측정)", G), "서빙 전처리는 학습과 동일, bbox·특징·영상 순서 정렬 정상", "서버 bbox와 재계산 bbox 일치, 오프셋 0 최선(40/42)"],
        [chip("확정(측정)", G), "학습의 53.7%가 제자리 회전으로 시작, 이 로봇 +az=우회전", "데이터 분석 + 실기 회전 부호 관측"],
        [chip("확정(측정)", G), "하드웨어는 방향각·부호·throttle만 사용, 학습 속력 라벨은 스틱 크기에서 파생한 값", "코드 + 300 에피소드 `joystick_speed_setting`=1.15"],
        [chip("확정(측정)", G), "현재 모델은 타깃 bbox 위치에 반응하지 않음(S ≈ −0.7°)", "반사실 시험 + 실기 raw 방향이 cx와 무관(Spearman −0.07)"],
        [chip("확정(측정)", G), "서보 라벨·E1b로 타깃 반응을 만들 수 있으나 사람 방향 오차가 증가", "S +3~+20°, 방향 오차 +9~+22° (3~4개 분할)"],
        [chip("확정(측정)", G), "마지막 단계 LoRA E2E는 기존 EGVLA2보다 낫지 않음", "같은 조건 3개 분할: 방향 오차 27.8° vs 19.6°"],
        [chip("가설", A), "실기 이탈의 원인이 서보 부재", "상관·반사실 근거는 있으나 인과는 폐루프 개입(비교 팔)으로만 증명"],
        [chip("가설", A), "분포 이동(바닥·조명·접근 거리)의 영향", "특징 코사인 차 0.004(작음), 효과 크기 미측정"],
        [chip("가설", A), "학습의 2프레임 명령 앞섬과 서빙의 정지 후 영상의 차이", "정렬 점검으로 현상만 확인, 영향 미측정"],
        [chip("가설", A), "연속 구동(ASYNC)·속도 정렬(throttle 50)의 이득", "계획·시뮬뿐, 실기 미시험"],
        [chip("정정·철회", R), "Gemini 세션의 E2E '2.82° vs 10.42°, 72.9% 급감, BBox 결측 100% 생존'", "in-sample 값 대 근거 없는 상수. 대시보드·CH79·리포트 정정"],
        [chip("정정·철회", R), "대시보드 '실시간 학습 스트림/텔레메트리'", "수식·난수로 만든 가짜 곡선. 제거, 실제 상태 파일 폴링으로 교체"],
        [chip("정정·철회", R), "E1b_tuned '프로덕션/챔피언'", "실기 미배포·미검증, held-out 방향 오차 22~26°로 기준보다 큼. 후보로 정정"],
        [chip("정정·철회", R), "이전 분석: '바구니 색이 다르다', 'az 단위당 4배', '현재는 정지-이동'", "각각 표시 오류(BGR 변환 누락)/하드웨어가 az 크기를 무시/ASYNC가 API 기본값 — 정정 완료"],
    ])
    pts = []
    def gs_row(d, var):
        return avg(d, var, "dir_err"), avg(d, var, "S")
    for lab, d, var, col in (("기준", gs, "base", MUT), ("E2 라벨", gs, "E2_hw_labels", "#64748b"), ("E5 가중", gs, "E5_edge_w", "#64748b"), ("E2+E1 이동 증강(상대 라벨)", e1, "E2+E1", P),
                             ("E1b 절대 서보", e1b, "E1b", A), ("E1b_tuned", e1b, "E1b_tuned", "#fb923c"), ("E1b+E8+E5", e1b, "E1b+E8+E5", A)):
        x, y = gs_row(d, var)
        if x is not None:
            pts.append((lab, x, y, col, False))
    for lab, var, col in (("서보 라벨 λ=0.5", "servo_label0.5", "#fbbf24"), ("서보 라벨 λ=1.0", "servo_label1.0", "#f97316")):
        xs = [tc[s][var]["first"] for s in tc if var in tc[s]]; ys = [tc[s][var]["sens"] for s in tc if var in tc[s]]
        if xs:
            pts.append((lab + " (다른 분할 집합)", float(np.mean(xs)), float(np.mean(ys)), col, True))
    sc = scatter(pts)
    rows = []
    for nm, key in (("A 고정 융합 특징(기존 EGVLA2)", "A_frozen_fused"), ("B0 고정 DaViT 풀링 120ep", "B0_frozen_davit_120ep"), ("B0 고정 DaViT 풀링 20ep", "B0_frozen_davit_20ep"), ("B1 마지막 단계 LoRA(E2E) 20ep", "B1_lora_last_stage_20ep")):
        rows.append([nm, f"{avg(e2e, key, 'dir_err'):.1f}°", f"{avg(e2e, key, 'S'):+.1f}°", f"{avg(e2e, key, 'spin_recall'):.2f}", f"{avg(e2e, key, 'edge_dir_err'):.1f}°", f"{avg(e2e, key, 'move_angle_err', True):.0f}°"])
    e2e_tab = tbl(["팔(같은 데이터·분할·라벨·지표)", "사람 방향 오차", "S", "회전 재현", "가장자리 오차", "실기 규칙 전문가 각오차"], rows)
    fs_rows = ""
    if fs:
        fr = []
        for s in fs:
            for k, v in fs[s].items():
                fr.append([s, k, f"{v['dir_err']:.1f}°", f"{v['S']:+.1f}°", f"{v['spin_recall']:.2f}"])
        fs_rows = tbl(["분할", "팔(384px)", "방향 오차", "S", "회전 재현"], fr) if fr else ""
    robot = tbl(["항목", "상태", "비고"], [
        ["폐루프 비교 팔 전달 문서", chip("작성·푸시", B), "`HANDOFF_20261027_ROBOT_MODEL_ARMS.md` (체크포인트 5종 sha 대조 완료, E2E 제외)"],
        ["회전·서보 서버 코드", chip("푸시", B), "inference-integration ≥ f280b33f. 로봇의 pull·재시작 확인은 회신 대기"],
        ["물리 보정·로깅 보강 요청", chip("작성·푸시", B), "`HANDOFF_20261023_ROBOT_PHYSICAL_CALIBRATION.md`, `..._LOGGING_and_followup.md`"],
        ["새 실기 데이터", chip("없음", A), "마지막 수신: 10/22 드라이런 20세션. 이후 로봇 쪽 새 세션·회신 미확인"],
    ])
    decide = tbl(["결정", "권장", "이유"], [
        ["폐루프 비교 팔 구성·순서", "A0 기준 → A1 기준+외부 서보 → A2 서보 라벨 0.5 (3팔 우선)", "서보 효과를 가장 싸게 분리. 나머지는 결과 후 추가"],
        ["E2E 방향", "전 단계 결과(05:30) 확인 전 보류, 마지막 단계만으로는 기존 방식 유지", "현재 근거로는 E2E 이득 없음"],
        ["속도 정렬(throttle 50) 시험", "서보 팔 결과 후 별도 축", "한 번에 하나만 변경"],
        ["교수님 보고 문구", "실기 성공률 미측정·E2E 정정 명시", "과대 보고 방지"],
    ])
    n_e1b = len(e1b)
    note = ('<div style="font-size:0.8rem;color:#94a3b8;margin:6px 0">갱신: 2026-10-28. 이 탭의 수치는 `docs/v5/egvla2_fm/*.json` 원본에서 생성. '
            f'체인 마지막 줄: {chain or "—"}. 전 단계 E2E는 완료 후 같은 표로 갱신.</div>')
    return (note + H("1. 어디까지 왔나") + status + H("2. 무엇이 확정이고 무엇이 가설·정정인가") + trust
            + H("3. 지표 트레이드오프 — 타깃에 반응하게 만들수록 사람과 멀어진다")
            + V.wrap("사람 방향 오차 vs 타깃 반응 S (3개 분할 평균, 속이 빈 점은 분할 집합이 달라 참고용)", sc, "기준 = 현재 배포 모델. 목표 영역은 오른쪽이 아니라 위쪽-왼쪽(S 높고 방향 오차 낮음)이며 아직 비어 있다. 방향 오차 증가 ≤ 3° 목표는 어느 변형도 달성 못함", kind="chart")
            + H("4. E2E 일반 비교 (같은 조건)")
            + V.wrap("사람 방향 오차 — 기존 EGVLA2 vs E2E 계열", e2e_bars(e2e), "B1은 같은 DaViT 특징의 고정(B0)보다 2~5° 낫지만 기존 융합 특징(A)에는 못 미침. A와 B의 특징 출처가 달라(융합 768 vs DaViT 풀링 1024) A 대 B0 격차에는 특징 차이가 섞임", kind="chart")
            + V.wrap("E2E 일반 비교 표 (3개 분할 평균)", e2e_tab, "전 단계 LoRA(384px)는 진행 중", kind="table")
            + (V.wrap("전 단계 LoRA @384 (완료된 부분)", fs_rows, kind="table") if fs_rows else "")
            + H("5. 로봇 쪽 상태") + robot
            + H("6. 결정이 필요한 것") + decide)

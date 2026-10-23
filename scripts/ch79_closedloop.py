"""CH79 🔬 폐루프 점검(10/22~23): 용어·발견 타임라인·실제 세션 사진·방향 오차와 실제 액션의 관계·학습 vs 실기 대조·타깃 반응 시험. 자산: make_closedloop_assets.py"""
import json

import numpy as np

import gen_ch76_viz as V

FM = V.FM
A, G, R, B, P, MUT = V.A, V.G, V.R, V.B, V.P, V.MUT
TX, GRID = "#e2e8f0", "#334155"


def cx_series_svg(sess):
    W, x0, y0, pw, ph = 760, 44, 14, 690, 190
    X = lambda i, n=26: x0 + min(i, n) / n * pw
    Y = lambda v: y0 + (1 - v) * ph
    o = [f'<svg viewBox="0 0 {W} {ph + 54}" style="width:100%;max-width:{W}px" font-family="sans-serif">']
    o.append(f'<rect x="{x0}" y="{Y(0.58):.1f}" width="{pw}" height="{(0.16) * ph:.1f}" fill="#22c55e" fill-opacity="0.10"/><text x="{x0 + 4}" y="{Y(0.58) - 3:.1f}" fill="#22c55e" font-size="10">중앙 구간(0.42~0.58)</text>')
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        o.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x0 + pw}" y2="{Y(v):.1f}" stroke="{GRID}" stroke-dasharray="2 4"/><text x="{x0 - 5}" y="{Y(v) + 4:.1f}" fill="{MUT}" font-size="11" text-anchor="end">{v:g}</text>')
    for i in range(0, 27, 5):
        o.append(f'<text x="{X(i):.1f}" y="{y0 + ph + 16}" fill="{MUT}" font-size="11" text-anchor="middle">{i}</text>')
    o.append(f'<text x="{x0 + pw / 2}" y="{y0 + ph + 34}" fill="{MUT}" font-size="11" text-anchor="middle">스텝 번호(SYNC, 스텝 간격 약 1.5초)    ● 회전 명령 스텝    × 타깃 소실</text>')
    feat = {"20261022_204813": G, "20261022_205118": A, "20261022_210745": R}
    for sid, rows in sess.items():
        col = feat.get(sid, "#475569"); wd = 2.6 if sid in feat else 1.2; op = 1.0 if sid in feat else 0.6
        pts = [(X(i), Y(r["cx"])) for i, r in enumerate(rows[:27]) if r["cx"] is not None]
        if len(pts) > 1:
            o.append(f'<polyline points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}" fill="none" stroke="{col}" stroke-width="{wd}" opacity="{op}"/>')
        for i, r in enumerate(rows[:27]):
            if r["label"] == "rotate" and r["cx"] is not None:
                o.append(f'<circle cx="{X(i):.1f}" cy="{Y(r["cx"]):.1f}" r="4.5" fill="{col}"/>')
            if r["cx"] is None or r["label"] == "bbox_missing":
                o.append(f'<text x="{X(i):.1f}" y="{Y(0.5) + 4:.1f}" fill="{col}" font-size="12" text-anchor="middle" opacity="{op}">×</text>')
    o.append(f'<text x="{x0 + pw - 4}" y="{y0 + 12}" fill="{G}" font-size="11" text-anchor="end">초록=204813(한 번 회전 후 중앙)</text><text x="{x0 + pw - 4}" y="{y0 + 26}" fill="{A}" font-size="11" text-anchor="end">주황=205118(두 번 회전 → 오버슈트)</text><text x="{x0 + pw - 4}" y="{y0 + 40}" fill="{R}" font-size="11" text-anchor="end">빨강=210745(직진 중 화면 밖으로 이탈)</text></svg>')
    return "".join(o)


def scatter_svg(d):
    W, x0, y0, pw, ph = 760, 50, 14, 680, 230
    X = lambda v: x0 + v * pw
    Y = lambda a: y0 + (90 - max(-90, min(90, a))) / 180 * ph
    o = [f'<svg viewBox="0 0 {W} {ph + 52}" style="width:100%;max-width:{W}px" font-family="sans-serif">']
    for a in (-90, -45, 0, 45, 90):
        o.append(f'<line x1="{x0}" y1="{Y(a):.1f}" x2="{x0 + pw}" y2="{Y(a):.1f}" stroke="{GRID}" stroke-dasharray="2 4"/><text x="{x0 - 5}" y="{Y(a) + 4:.1f}" fill="{MUT}" font-size="11" text-anchor="end">{a}°</text>')
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        o.append(f'<text x="{X(v):.1f}" y="{y0 + ph + 16}" fill="{MUT}" font-size="11" text-anchor="middle">{v:g}</text>')
    o.append(f'<text x="{x0 + pw / 2}" y="{y0 + ph + 36}" fill="{MUT}" font-size="11" text-anchor="middle">타깃 위치 cx (0=왼쪽 끝, 1=오른쪽 끝)   세로: 명령 방향각 atan2(ly,lx) (0°=전진, 양수=ly 양의 방향)</text>')
    for cx, a in zip(d["train_scatter"]["cx"][::2], d["train_scatter"]["ang"][::2]):
        o.append(f'<circle cx="{X(cx):.1f}" cy="{Y(a):.1f}" r="1.8" fill="#64748b" opacity="0.35"/>')
    for rows in d["sessions"].values():
        for r in rows:
            if r["label"] == "drive" and r["cx"] is not None and r["raw_angle"] is not None:
                o.append(f'<circle cx="{X(r["cx"]):.1f}" cy="{Y(r["raw_angle"]):.1f}" r="3" fill="{R}" opacity="0.65"/>')
    o.append(f'<text x="{x0 + pw - 4}" y="{y0 + 12}" fill="#94a3b8" font-size="11" text-anchor="end">회색=학습(사람 조작)</text><text x="{x0 + pw - 4}" y="{y0 + 26}" fill="{R}" font-size="11" text-anchor="end">빨강=실기(모델 raw 출력)</text></svg>')
    return "".join(o)


def block(tbl, H, st):
    d = json.load(open(FM / "closedloop_summary.json")); tc = json.load(open(FM / "target_cond.json")); od = json.load(open(FM / "rotation_odometry.json"))
    terms = tbl(["용어", "뜻", "주의"], [
        ["lx · ly · az", "전진/좌우 이동·회전 명령. 이 로봇에서 <b>+az=우회전, −az=좌회전</b>(학습 기록 규약과 같음). ly<0이면 타깃 cx가 늘어남(데이터·실기 공통)", "ROS 관례(+az=좌회전)와 반대라고 가정했다가 실기에서 정정"],
        ["cx · area", "Florence-2가 찾은 타깃 bbox의 가로 중심(0~1)·이미지 대비 면적. 소실이면 has_bbox=false", "면적이 0.25 이상이면 도착 정지"],
        ["방향각", "명령 (lx,ly)의 방향 atan2(ly,lx)", "하드웨어는 이 각도와 throttle만 쓴다(크기 무시)"],
        ["방향 오차(°)", "모델 방향과 사람 방향의 각도 차(이동 프레임만, 오프라인)", "사람과의 일치 지표일 뿐 성공 지표가 아님 (CH77 ⑥-17)"],
        ["raw(chunk_first) · 최종 action", "모델이 낸 첫 스텝 값 · EMA·단위 방향 정규화·gain 1.15 뒤 구동값", "정규화로 크기 정보가 버려져 최종값이 평탄해 보임"],
        ["throttle · rot_throttle", "구동기 속도 단계(직진 50, 회전 17.5 × speed_scale)", "수집 시 50·17.5 고정, 서빙은 ×0.7 → 약 30% 느림"],
        ["SYNC · ASYNC", "SYNC: 0.4초 이동 후 정지(스텝 약 1.55초). ASYNC: 다음 응답까지 직전 명령 유지", "폐루프 17/20 세션이 SYNC"],
        ["상태: rotate · idle · drive · search", "시작 회전 · 이동 0 대기 · 구동 · 소실 탐색", "크기 기반 stop 래치는 끔(idle 20 연속에서만 안전망)"],
        ["서보", "타깃이 중앙에서 벗어난 만큼 방향을 중앙 쪽으로 보정(모델 밖 규칙)", "방향각만 바꾸므로 하드웨어에 유효"],
        ["반사실 민감도 S", "타깃 cx를 ±0.2 옮겼을 때 예측 방향이 교정 방향으로 바뀐 각도(양수=타깃 쪽)", "현재 배포 모델 −0.7°(반응 없음)"],
        ["최근접 코사인", "라이브 특징과 가장 비슷한 학습 특징의 유사도", "학습 보류 에피소드 0.9974 vs 실기 0.9939"],
        ["BGR · RGB", "학습 H5 images는 BGR로 저장, 특징 추출 때 RGB로 변환", "표시할 때 변환을 빠뜨리면 색이 반전"]])
    flow = V.flow([("① 첫 프레임에서 안 움직임", R, "원인: az=0 버림 + stop 래치"), ("② 회전 상태 추가", G, "az 부호 반대 → 실기 정정"), ("③ 회전은 맞게 돎", G, "오버슈트·0.9초 지연"),
                   ("④ 직진 중 cx 이탈", R, "모델은 타깃 위치를 안 봄"), ("⑤ 컨트롤러 환경 대조", B, "크기 무시·속도 30% 느림"), ("⑥ 서보·학습 보정 시험", A, "S +3~+20°")])
    s_od = od
    ctrl = tbl(["항목", "수집(학습 데이터)", "추론(실기 서빙)", "영향"], [
        ["하드웨어 명령", "방향각·부호·throttle만 사용 (spin/move)", "같은 함수", "크기는 어느 쪽도 의미 없음"],
        ["직진 throttle", "50 (전 300 에피소드 joystick_speed 1.15)", "50 × 0.7 = 35", "서빙이 약 30% 느림(26.4 → 20.6cm/s)"],
        ["회전 throttle", "17.5", "17.5 × 0.7 ≈ 12", "느림, 단 관측 회전은 더 큼(아래)"],
        ["'속력 3단계' 라벨", "스틱 크기에서 파생(0.4/0.7/1.0), 물리 속도 아님", "서버가 속도 0.7 고정", "모델 속력 예측은 물리와 무관"],
        ["구동 시간 구조", "스틱 쥐는 동안 10Hz 연속 발행", "SYNC 0.4초 버스트 + 정지", "회전 한 번 0.5초 vs 0.4초 + 0.9초 정지"],
        ["회전 이동량(영상 px@320)", f"{s_od['train_az+']['px_per_s']:.0f}px/s (한 번 {s_od['train_az+']['dx_total_px']:.0f}px)", f"{s_od['robot_burst_az+']['px_per_s_if_0p4']:.0f}px/s (한 번 {s_od['robot_burst_az+']['dx_px_per_burst']:.0f}px)", "약 1.7배, 정지 후 잔여 약 20%"],
        ["입력 분포(특징 최근접 코사인)", "0.9974 (보류 에피소드끼리)", "0.9939", "작은 이동"],
        ["bbox 검출률 · 면적 중앙값", "99.96% · 0.059", "89.9% · 0.034", "소실이 10%, 접근이 덜 됨"]])
    rel = tbl(["측정", "값", "해석"], [
        ["오프라인 방향 오차(사람 대비)", "5-fold 10.4° / 안 본 위치 17.6°", "사람과 같은 방향을 내는가: 양호"],
        ["라이브: 최근접 학습 프레임의 사람 방향과의 차", "중앙값 8.6° / 평균 19.7°", "장면에 맞는 학습 행동을 재생(조회형)"],
        ["라이브 raw 방향각 vs 타깃 cx", "Spearman −0.07, 중앙값 −48°", "타깃 위치와 무관한 대각선"],
        ["타깃 bbox 반사실 민감도 S", "−0.7°(cx ±0.2에도 방향 1° 미만 변화)", "모델이 타깃 위치를 안 씀"],
        ["중앙 쪽으로 보정한 비율(cx 0.1 이상 벗어났을 때)", "사람 44.5% / 모델 36%", "데이터가 서보형이 아님"],
        ["실제 이탈", "210745: cx 0.36→0.96 단조 증가 후 소실", "방향 오차가 작아도 폐루프는 이탈"]])
    def row(v):
        f = [tc[s][v] for s in tc if s.startswith("fold") and v in tc[s]]; l = [tc[s][v] for s in tc if s.startswith("lopo") and v in tc[s]]
        return [f"{np.mean([x['first'] for x in f]):.1f}°", f"{np.mean([x['sens'] for x in f]):+.1f}°", f"{np.mean([x['first'] for x in l]):.1f}°", f"{np.mean([x['sens'] for x in l]):+.1f}°"]
    tcrows = [[n] + row(v) for n, v in (("기준(현재 배포)", "base"), ("이미지 특징 50% 드롭", "featdrop50"), ("서보 라벨 λ=0.5", "servo_label0.5"), ("서보 라벨 λ=1.0", "servo_label1.0"))]
    tct = tbl(["변형(학습)", "같은 위치 방향 오차", "민감도 S", "안 본 위치 방향 오차", "민감도 S"], tcrows)
    return (H("1. 폐루프 점검에서 발견한 순서") + V.wrap("발견 → 수정 흐름", flow, "근거 문서: plans/finding_20261022_start_rotation_mismatch.md, analysis_20261023_*.md", kind="flow")
            + H("2. 용어") + terms
            + H("3. 실제 세션 사진") + V.wrap("학습의 시작 구간(사람): 먼저 돌아서 타깃을 중앙에 두고 직진", V.img("egvla2_fm/closedloop_photos/train_weak_right_start.jpg", "학습 weak_right|straight. 프레임 0~4 사람 az=+0.4(우회전) → cx 0.76→0.48, 이후 lx로 직진하며 cx 유지. 초록 상자 = bbox"), kind="img")
            + V.wrap("실기 204813: 한 번 회전으로 중앙", V.img("egvla2_fm/closedloop_photos/live_204813_center.jpg", "스텝 0 cx 0.76 → rotate 한 번 → 0.54 → 중앙 0.50. 이후 직진하는 동안 cx가 0.53→0.72로 점진 이탈"), kind="img")
            + V.wrap("실기 205118: 두 번 회전하면 오버슈트", V.img("egvla2_fm/closedloop_photos/live_205118_overshoot.jpg", "시작 cx 0.88에서 rotate 두 번(0.88→0.64→0.36), 중앙을 지나 반대쪽으로. 이후 ly≈−0.8 strafe가 이어짐"), kind="img")
            + V.wrap("실기 210745: 직진 중 타깃이 화면 밖으로", V.img("egvla2_fm/closedloop_photos/live_210745_drift.jpg", "회전 후 cx 0.36에서 직진하는 동안 cx가 0.36→0.96으로 단조 증가해 화면 밖으로 나가고 bbox 소실 → 탐색"), kind="img")
            + H("4. 방향 오차와 실제 액션의 관계") + V.wrap("스텝별 타깃 cx — 회전은 중앙으로, 이후 직진은 이탈", cx_series_svg(d["sessions"]), "SYNC EGVLA2 세션 17개. 초록 띠 = 중앙 구간", kind="chart")
            + V.wrap("타깃 cx vs 명령 방향각 — 학습은 퍼져 있고 실기는 한 대각선", scatter_svg(d), "실기 raw 방향이 cx와 무관하게 약 −48°에 모임(회색 점은 학습 이동 프레임 일부)", kind="chart")
            + V.wrap("오프라인 방향 오차는 이 문제를 못 본다", rel, "방향 오차는 '사람 방향과의 일치'만 재므로, 타깃을 향한 피드백 부재는 별도 지표(민감도 S)로 봐야 함", kind="table")
            + H("5. 학습 vs 실기 대조 (컨트롤러 환경 포함)") + V.wrap("무엇이 같고 무엇이 다른가", ctrl, "하드웨어는 명령 크기를 무시: driver.spin(±rot_throttle), driver.move(각도, throttle)", kind="table")
            + H("6. 타깃에 반응하게 만드는 시험") + V.wrap("서보 라벨 학습 (오프라인, 2+2 분할)", tct, "S: 타깃 cx ±0.2 이동 시 교정 방향 각도 변화. 방향 오차는 사람 대비라 서보 효과는 폐루프로만 판단. 같은 효과를 재학습 없이 주는 외부 서보(SERVO_BLEND)가 구현됨(기본 끔)", kind="table"))

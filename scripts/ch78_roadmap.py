"""CH78(실기 전 단계) 개요·로드맵. 단계별 상태·담당·선행조건·합격 기준·산출물과 다음 액션."""
import json

import gen_ch76_viz as V

FM = V.FM
A, G, R, B, P, MUT = V.A, V.G, V.R, V.B, V.P, V.MUT
ST = {"done": ("완료", G), "wait": ("대기(사용자·로봇)", A), "todo": ("미착수", MUT), "gate": ("게이트", R)}


def badge(k):
    t, c = ST[k]
    return f'<span style="display:inline-block;background:{c}22;color:{c};border:1px solid {c};border-radius:6px;padding:0 8px;font-size:0.72rem;font-weight:800">{t}</span>'


STAGES = [
    ("R0", "데이터 수집·검증", "done", "로봇 세션", "—", "읽기 문제 0, az 스파이크 ≤25%", "300건(15칸×20), validation_batch.json"),
    ("R1", "오프라인 학습·평가", "done", "minum", "R0", "방향 오차·R²·LOPO·서브골·곡선·평활·천장(CH77)", "summary_s1_full300*.json, CH77"),
    ("R2", "배포 산출물·서빙 코드", "done", "minum", "R1", "체크포인트 sha256·정책 점검·서버 단계 A/B 테스트 통과", "deploy/C_pi0_adaln_drop30_e120.pt, 서버 코드, 핸드오프 §15·§16"),
    ("R3", "젯슨 기술 점검(서버 경로·지연)", "done", "로봇 세션", "R2", "점검 A pass, 지연 측정", "A: 방향 차 3.01°, B: 왕복 892ms(Florence-2 821ms + FM 54ms)"),
    ("R4", "드라이런 = 에피소드 서버 재생(pos2·pos3 신규 10건)", "wait", "사용자 수집 → 로봇 세션 재생·전송", "R3", "재생 JSON·CSV와 신규 H5 도착", "startpos_replay.json, CH78 ⑥-10"),
    ("R5", "구동 전 게이트(승인 판단)", "gate", "minum 분석 + 사용자 승인", "R4", "pos2·3 방향 오차가 LOPO 기준(약 18°) 이내, 정지 판단·서브골 결측 로그 이상 없음, EMA·정지·탐색 값 확정(제안, 임의)", "게이트 판정표, 단계 B 점검 결과"),
    ("R6", "하이브리드 단계 A 짧은 구동 시험", "todo", "로봇 세션(승인 후)", "R5", "안전 설정(s≤0.7, az=0)으로 몇 에피소드 구동, 이상 동작 없음", "구동 로그"),
    ("R7", "폐루프 300회 (3 시작 위치×5 목표 위치×20)", "todo", "로봇 세션", "R6", "시작 위치별 SR, 충돌·개입·시간 기록", "SR 표(EGVLA1 95/91/90과 같은 축)"),
    ("R8", "분석·논문", "todo", "minum + 사용자", "R7", "EGVLA1 대비 SR·지연·견고성 서술", "논문 초안, CH79+"),
]


def block(tbl, H, st):
    nd = sum(1 for s in STAGES if s[2] == "done")
    call = ('<div class="callout critical">🧭 <strong>CH78 = 실기 주행 전 단계 정리 (서빙·배포·점검·로드맵·시사점·논문 방향)</strong><br>'
            f'① 로드맵 {len(STAGES)}단계 중 <strong>{nd}단계 완료</strong>(수집, 오프라인 학습·평가, 배포 산출물, 젯슨 서버 경로·지연 점검). 현재는 <strong>R4 드라이런</strong>(pos2·pos3 신규 에피소드 10건 수집 대기)이다.<br>'
            '② 젯슨 왕복 지연은 <strong>892ms(Florence-2 821ms + FM 헤드 54ms)</strong>이고 서버 경로 점검은 pass. 서버로 보내는 JPEG는 q95 이상이어야 한다.<br>'
            '③ <strong>구동 전 게이트(R5)</strong>를 통과해야 하이브리드 구동(R6)과 폐루프 300회(R7)로 간다. 오프라인 성능은 사람 변동성 한계 수준이지만 <strong>실기 성공률은 아직 모른다</strong>.</div>')
    flow = V.flow([(f"{s[0]} {s[1].split('(')[0].split('=')[0].strip()}", ST[s[2]][1], ST[s[2]][0]) for s in STAGES])
    cards = ""
    for sid, name, k, owner, pre, gate, out in STAGES:
        col = ST[k][1]
        cards += (f'<div style="display:flex;gap:10px;align-items:stretch;margin:6px 0"><div style="width:6px;background:{col};border-radius:3px"></div>'
                  f'<div style="flex:1;background:#0d1420;border:1px solid #1e293b;border-radius:8px;padding:8px 12px;font-size:0.78rem;line-height:1.55">'
                  f'<div><b style="color:#fcd34d">{sid}</b> <b style="color:#e2e8f0">{name}</b> {badge(k)} <span style="color:#64748b;margin-left:6px">담당: {owner} · 선행: {pre}</span></div>'
                  f'<div style="color:#94a3b8"><b>합격 기준</b> {gate}</div><div style="color:#94a3b8"><b>산출물</b> {out}</div></div></div>')
    nxt = tbl(["지금 할 일", "누가", "언제까지", "결과가 오면"], [
        ["pos2 시작 × 목표 위치 5개 × 1에피소드, pos3 시작 × 5개 × 1에피소드 = 10건 수집(시작 위치 pos를 로봇 세션에 알려 파일명 정리)", "사용자(로봇 조작)", "로봇 세션 대기 중", "—"],
        ["수집분을 서버(EMA 0.5)에 재생해 JSON·CSV와 H5를 minum으로 전송", "로봇 세션", "수집 직후", "—"],
        ["시작 위치별 방향 오차·F1·저크 분석, CH78 ⑥-10 자동 반영, 게이트 판정", "minum(제가)", "전송 직후", "구동 승인 여부 판단"],
        ["EMA·정지 디바운스·소실 탐색 값 확정, 단계 B 서버 점검", "minum", "게이트 전", "R6 구동 시험 설계"]])
    dec = tbl(["결정(사용자)", "내용"], [
        ["폐루프 규모·구성", "300회 = 3 시작 위치(pos1~3) × 5 목표 위치 × 20 (EGVLA1 276/300=92.0%와 같은 구성, 시작 위치별 보고)"],
        ["속력 출처", "첫 적용은 섀도(규칙 속력 0.7 고정으로 구동, 모델 속력은 로그), s≤0.7"],
        ["az", "첫 적용은 az=0(이동만 모델, 회전은 기존 규칙)"],
        ["드라이런 방식", "조이스틱 동시 로그 대신 수집 에피소드를 서버에 재생(로봇 정지, 사람 조작과 프레임 단위 비교)"],
        ["구동 승인", "하이브리드·폐루프 구동은 R5 결과를 본 뒤 minum이 승인(그 전 금지)"],
        ["서빙 입력", "서버 전송 JPEG q95 이상, 색 보정 끔, 최신 서버 코드(응답 잘림 버그 수정)"]])
    blk = ("<ul style='margin:6px 0 6px 18px'><li><b>로봇 세션</b>: 구동·폐루프는 사람이 직접 몰아야 해서 자동화 불가(수집 10건 대기). 대시보드에 시작 위치(pos) 속성을 남기는 기능이 없어 수동 리네임 필요.</li>"
           "<li><b>미확인 항목</b>: gain 1.15가 컨트롤러 lx/ly 단위와 맞는지(대각선 속도는 직진과 같다고 가정), 회전 탐색 중 대시보드의 <code>move_and_stop_ramped</code> 동작, 탐색 회전 크기·부호.</li>"
           "<li><b>단계 B(EGVLA2 단독 서버)</b>: 정지·소실·평활 값은 드라이런 로그로 확정해야 한다(스켈레톤은 구현됨, 기본 꺼짐).</li></ul>")
    return (call + H("실기 전 로드맵") + V.wrap("단계 흐름 (색 = 상태)", flow, "초록 완료 · 노랑 대기 · 빨강 게이트 · 회색 미착수", kind="flow") + cards
            + H("③ 다음 액션") + nxt + H("④ 결정 기록") + dec + H("막힌 것·의존") + blk)

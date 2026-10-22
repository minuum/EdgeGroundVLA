"""CH78 ⚙️ 런타임 요인 — 한 주기 플로우, 시간표, 청크 사용, 낡은 명령 그래프, 끌/남길 요인, 정지 면적 (2026-10-22). 글은 최소로, 그림 중심."""
import json

import gen_ch76_viz as V

FM = V.FM
A, G, R, B, P, MUT = V.A, V.G, V.R, V.B, V.P, V.MUT
TX, GRID = "#e2e8f0", "#334155"


def timeline_svg():
    """서버 추론 / 로봇 이동을 두 모델에 대해 같은 시간축(초)으로."""
    sc, x0, W = 250, 130, 760
    lanes = [("EGVLA2 서버 추론", 30, [(0.0, 0.90, B, "추론 0.9초"), (1.35, 2.25, B, "추론 0.9초")]),
             ("EGVLA2 로봇", 66, [(0.90, 1.30, G, "이동 0.4"), (2.25, 2.65, G, "이동 0.4")]),
             ("EGVLA1 서버 추론", 118, [(0.0, 0.53, A, "0.53"), (0.98, 1.51, A, "0.53"), (1.96, 2.49, A, "0.53")]),
             ("EGVLA1 로봇", 154, [(0.53, 0.93, G, "이동"), (1.51, 1.91, G, "이동"), (2.49, 2.89, G, "이동")])]
    o = [f'<svg viewBox="0 0 {W} 210" style="width:100%;max-width:{W}px" font-family="sans-serif">']
    k = 190   # px/초
    for s in range(0, 7):
        x = x0 + s * 0.5 * k
        o.append(f'<line x1="{x}" y1="14" x2="{x}" y2="178" stroke="{GRID}" stroke-dasharray="3 4"/><text x="{x}" y="196" fill="{MUT}" font-size="11" text-anchor="middle">{s * 0.5:g}초</text>')
    for name, y, bars in lanes:
        o.append(f'<text x="{x0 - 8}" y="{y + 19}" fill="{TX}" font-size="12" text-anchor="end">{name}</text>')
        for a, b, col, lab in bars:
            o.append(f'<rect x="{x0 + a * k:.1f}" y="{y}" width="{(b - a) * k:.1f}" height="26" rx="4" fill="{col}" fill-opacity="0.35" stroke="{col}"/>'
                     f'<text x="{x0 + (a + b) / 2 * k:.1f}" y="{y + 17}" fill="{TX}" font-size="11" text-anchor="middle">{lab}</text>')
    o.append(f'<text x="{x0}" y="112" fill="{MUT}" font-size="10.5">※ EGVLA1 값은 평균(캐시 재사용 반영), 비교용 단순화. 로봇은 추론 중 정지</text></svg>')
    return "".join(o)


def chunk_svg():
    o = ['<svg viewBox="0 0 760 110" style="width:100%;max-width:760px" font-family="sans-serif">']
    for i in range(8):
        col, lab = (G, "사용") if i == 0 else (A, "후보") if i < 3 else (MUT, "버림")
        o.append(f'<rect x="{20 + i * 90}" y="30" width="80" height="36" rx="5" fill="{col}" fill-opacity="0.3" stroke="{col}"/>'
                 f'<text x="{60 + i * 90}" y="52" fill="{TX}" font-size="12" text-anchor="middle">스텝 {i + 1}</text>'
                 f'<text x="{60 + i * 90}" y="82" fill="{col}" font-size="11" text-anchor="middle">{lab}</text>')
    o.append(f'<line x1="20" y1="20" x2="380" y2="20" stroke="{G}" stroke-width="2"/><text x="200" y="14" fill="{G}" font-size="11" text-anchor="middle">실제 이동 0.4초 ≈ 3.2스텝</text>')
    o.append(f'<text x="20" y="104" fill="{MUT}" font-size="10.5">청크 8스텝 = 약 1초(0.125초/스텝). 지금은 스텝 1만 사용, 앞 3스텝 평균은 오프라인 검증 후보</text></svg>')
    return "".join(o)


def lines_svg(lm):
    f0, lo = lm["fold0"]["latency"], lm["lopo_strong_right"]["latency"]
    W, H0, x0, y0, pw, ph, lo_y, hi_y = 760, 230, 56, 16, 660, 170, 10.0, 21.0
    X = lambda d: x0 + d / 7 * pw
    Y = lambda v: y0 + (hi_y - v) / (hi_y - lo_y) * ph
    o = [f'<svg viewBox="0 0 {W} {H0}" style="width:100%;max-width:{W}px" font-family="sans-serif">']
    for v in (10, 12, 14, 16, 18, 20):
        o.append(f'<line x1="{x0}" y1="{Y(v):.1f}" x2="{x0 + pw}" y2="{Y(v):.1f}" stroke="{GRID}" stroke-dasharray="2 4"/><text x="{x0 - 6}" y="{Y(v) + 4:.1f}" fill="{MUT}" font-size="11" text-anchor="end">{v}°</text>')
    for d in range(8):
        o.append(f'<text x="{X(d):.1f}" y="{y0 + ph + 16}" fill="{MUT}" font-size="11" text-anchor="middle">{d}</text>')
    o.append(f'<text x="{x0 + pw / 2}" y="{y0 + ph + 32}" fill="{MUT}" font-size="11" text-anchor="middle">명령이 오래된 프레임 수 (8fps)</text>')
    for name, series, col, dash in (("같은 위치·직전 명령 유지", f0["stale"], G, ""), ("같은 위치·청크 뒤 스텝", f0["compensated"], R, ""),
                                     ("안 본 위치·직전 명령 유지", lo["stale"], G, "5 4"), ("안 본 위치·청크 뒤 스텝", lo["compensated"], R, "5 4")):
        pts = " ".join(f"{X(d):.1f},{Y(series[str(d)]):.1f}" for d in range(8))
        o.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.2" stroke-dasharray="{dash}"/>')
        o.append(f'<text x="{X(7) + 4:.1f}" y="{Y(series["7"]) + 4:.1f}" fill="{col}" font-size="10.5">{series["7"]:.1f}</text>')
    o.append(f'<text x="{x0 + 6}" y="{y0 + 12}" fill="{G}" font-size="11">초록=유지 / 빨강=청크 뒤 스텝 / 실선=같은 위치 / 점선=안 본 위치</text></svg>')
    return "".join(o)


def area_svg(st):
    e, c = st["edges"], st["counts"]
    W, H0, x0, y0, pw, ph = 760, 190, 40, 14, 680, 130
    mx = max(c); bw = pw / len(c)
    o = [f'<svg viewBox="0 0 {W} {H0}" style="width:100%;max-width:{W}px" font-family="sans-serif">']
    for i, n in enumerate(c):
        h = n / mx * ph; col = G if e[i] >= 0.25 else R
        o.append(f'<rect x="{x0 + i * bw + 2:.1f}" y="{y0 + ph - h:.1f}" width="{bw - 4:.1f}" height="{h:.1f}" fill="{col}" fill-opacity="0.45" stroke="{col}"/>')
        if i % 2 == 0:
            o.append(f'<text x="{x0 + i * bw:.1f}" y="{y0 + ph + 14}" fill="{MUT}" font-size="10.5" text-anchor="middle">{e[i]:.2f}</text>')
    xt = x0 + (0.25 - e[0]) / (e[1] - e[0]) * bw
    o.append(f'<line x1="{xt:.1f}" y1="{y0}" x2="{xt:.1f}" y2="{y0 + ph}" stroke="{A}" stroke-width="2"/><text x="{xt + 4:.1f}" y="{y0 + 12}" fill="{A}" font-size="11">near_area 0.25</text>')
    o.append(f'<text x="{x0}" y="{y0 + ph + 32}" fill="{MUT}" font-size="11">에피소드 마지막 프레임의 bbox 면적 (중앙값 {st["median"]:.2f}, 0.25 이상 {st["frac_ge"] * 100:.0f}%, 도달 못함 {st["n_never"]}개) — 사람은 이 면적에서 멈춤</text></svg>')
    return "".join(o)


def chips(items, col):
    return "".join(f'<span style="display:inline-block;border:1px solid {col};background:{col}1a;border-radius:999px;padding:2px 10px;margin:3px 4px 3px 0;font-size:0.76rem;color:{TX}">{t}</span>' for t in items)


def block(tbl, H, st):
    lm = json.load(open(FM / "latency_modes.json"))
    ar = json.load(open(FM / "near_area_stats.json"))
    cycle = V.flow([("① 프레임 1장", B, "로봇 정지, 최신 이미지 JPEG q95"), ("② Florence-2", B, "그라운딩 + 특징, 0.9초 (지연 94%)"), ("③ bbox 창", P, "최근 3번 판단, 결측이면 0"),
                    ("④ FM 헤드", P, "K=8 샘플 → 청크 평균 (8스텝)"), ("⑤ 첫 스텝 + EMA 0.5", A, "나머지 7스텝 버림"), ("⑥ 방향·속도·정지", A, "속도 0.7 고정, 크기<0.2 정지, 면적≥0.25 도착"),
                    ("⑦ 0.4초 이동", G, "약 8cm 후 자동 정지"), ("⑧ 정지·대기", MUT, "0.45초 → ①로")])
    versus = tbl(["", "EGVLA1", "EGVLA2"], [["한 주기", "약 1.0초", "약 1.35초"], ["그라운딩 캐시", "있음(3프레임, 50.8%)", "없음"], ["지연 지배", "OWLv2 97%", "Florence-2 94%"],
                                              ["회복", "재그라운딩·회전·학습 정지", "소실 5 → 탐색, 면적 정지"], ["판단 횟수", "평균 11.6회", "약 14회(추정)"]])
    off = ["preview·hint_cx", "cx 점프 필터", "multi-prompt", "미검출 재그라운딩", "stop_on_miss", "stop_mode(learned)", "OWLv2 임계값·면적", "action_speed_scale"]
    keep = ["ckpt sha·arch", "dtype", "K=8", "EMA 0.5", "gain 1.15", "디바운스 3", "near_area 0.25 · 2", "lost 5 · search 40", "JPEG q95", "apply_cc 끔"]
    A_ = ["청크 앞 n스텝 평균", "EMA 0·0.3·0.7", "K 1·4·16", "Euler 3·5", "bbox 창 1·5", "드롭아웃 0.5"]
    B_ = ["gain", "near_area/frames", "STOP_M 0.2", "디바운스·소실·탐색", "search_az", "has_bbox 놓침"]
    C_ = ["move_duration 0.3·0.6", "이동 후 대기", "RULE_S 0.5·1.0", "그라운딩 N프레임마다", "beam·토큰 축소", "fp16"]
    return (V.wrap("한 주기 — 프레임이 명령이 되기까지", cycle, "정지 → 추론 → 0.4초 이동 → 정지. 추론 중 로봇은 서 있고, 청크 8스텝 중 구동에는 첫 스텝만 쓴다", kind="flow")
            + V.wrap("시간표 — 같은 시간축에서 추론과 이동", timeline_svg(), "EGVLA2는 추론이 길어 한 주기가 더 길다. 로봇은 둘 다 추론 중 정지", kind="chart")
            + V.wrap("EGVLA1 vs EGVLA2 요약", versus, kind="table")
            + V.wrap("청크는 어떻게 쓰이나", chunk_svg(), "뒤 스텝을 현재 프레임 정답과 비교하면 나쁨(아래 그래프). 이동 0.4초 구간 평균 방향과 비교하는 평가는 아직 안 함", kind="flow")
            + V.wrap("참고 — 낡은 명령을 쓰면 방향 오차가 얼마나 늘어나나", lines_svg(lm), "현재 루프(정지 후 추론)에서는 발생하지 않음. 연속 구동으로 바꿀 때 대비. 유지는 +0.5° 이내, 청크 뒤 스텝은 +1.6~1.8°", kind="chart")
            + V.wrap("런타임 요인 — 끌 것 / 남길 것", f'<div style="font-size:0.78rem;color:{MUT};margin-bottom:2px">끔(EGVLA1 전용)</div>{chips(off, R)}<div style="font-size:0.78rem;color:{MUT};margin:8px 0 2px">남기고 세션 H5에 기록</div>{chips(keep, G)}', "현재 세션 스냅샷은 EGVLA1 키만 읽음 → EGVLA2 키 추가 필요", kind="flow")
            + V.wrap("정지 판정 근거 — 학습 데이터 마지막 프레임 면적", area_svg(ar), f"0.25 임계는 거의 항상 발동하지만 사람이 멈춘 곳보다 약 {ar['rem_median']:.0f}프레임(약 1.2초) 앞서 멈춤. 실제 거리는 드라이런에서 확인", kind="chart")
            + V.wrap("조정·ablation 후보 — 어디서 검증하나", f'<div style="font-size:0.78rem;color:{G}">A 오프라인(오늘)</div>{chips(A_, G)}<div style="font-size:0.78rem;color:{A};margin-top:6px">B 드라이런 로그 필요</div>{chips(B_, A)}<div style="font-size:0.78rem;color:{R};margin-top:6px">C 실기에서만</div>{chips(C_, R)}', "내일 폐루프는 기본값 고정, 비교군(회귀)만 바꿈. 한 번에 하나만 변경", kind="flow"))

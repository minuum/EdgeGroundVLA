"""CH77 ⑥-16: 방향 오차의 '각도'와 'cm'가 무엇을 뜻하는가 — 좌표계 설명, 실제 프레임 오버레이(bbox·정면축·명령 나침반), bbox 위치 vs 명령 각도 산점도, cm 단위의 계산 기준."""
import json

import numpy as np

import ch77_viz as Z
import gen_ch76_viz as V

FM = V.FM
A, G, R, B, P, MUT = V.A, V.G, V.R, V.B, V.P, V.MUT
SPN = {"fold0": "같은 목표 위치(fold0)", "lopo_strong_right": "안 본 목표 위치(strong_right)"}
RN = {"best": ("최선", G), "median": ("중앙", B), "worst": ("최악", R)}


def frame_diagram():
    return ('<svg viewBox="0 0 640 230" style="width:100%;max-width:640px;background:#0a1220;border:1px solid #1e293b;border-radius:8px">'
            '<text x="12" y="18" fill="#e2e8f0" font-size="11" font-weight="700">① 카메라 이미지 (픽셀)</text><rect x="20" y="30" width="250" height="140" fill="#111a2b" stroke="#334155"/>'
            '<line x1="145" y1="30" x2="145" y2="170" stroke="#94a3b8" stroke-dasharray="4 3"/><text x="150" y="44" fill="#94a3b8" font-size="9">정면(전진 축)</text>'
            '<circle cx="205" cy="120" r="16" fill="none" stroke="#38bdf8" stroke-width="2.5"/><text x="188" y="152" fill="#38bdf8" font-size="9">bbox 중심 (cx, cy)</text>'
            '<line x1="145" y1="120" x2="205" y2="120" stroke="#38bdf8" stroke-dasharray="3 3"/><text x="150" y="184" fill="#64748b" font-size="9">cx = 0.5가 정면, cx &gt; 0.5면 오른쪽</text>'
            '<text x="330" y="18" fill="#e2e8f0" font-size="11" font-weight="700">② 로봇 몸체 좌표의 속도 명령 (lx, ly)</text>'
            '<line x1="440" y1="190" x2="440" y2="40" stroke="#94a3b8"/><line x1="340" y1="190" x2="540" y2="190" stroke="#334155"/>'
            '<text x="446" y="46" fill="#94a3b8" font-size="9">전진 lx (θ=0°)</text><text x="342" y="184" fill="#94a3b8" font-size="9">좌 ly &gt; 0 (θ=+90°)</text><text x="478" y="184" fill="#94a3b8" font-size="9">우 (−90°)</text>'
            f'<line x1="440" y1="190" x2="395" y2="100" stroke="{G}" stroke-width="4"/><text x="352" y="96" fill="{G}" font-size="10">사람 θ=+27°</text>'
            f'<line x1="440" y1="190" x2="470" y2="95" stroke="{A}" stroke-width="4" stroke-dasharray="6 3"/><text x="474" y="92" fill="{A}" font-size="10">모델 θ=−18°</text>'
            f'<path d="M 440 150 A 40 40 0 0 0 424 154" fill="none" stroke="{R}" stroke-width="3"/><text x="372" y="214" fill="{R}" font-size="10">방향 오차 = |27° − (−18°)| = 45°</text>'
            '<text x="12" y="214" fill="#64748b" font-size="9">각도 θ = atan2(ly, lx). 이미지의 픽셀 각도가 아니라 <tspan fill="#e2e8f0">로봇이 어느 방향으로 가라는 명령</tspan>의 방향이다.</text></svg>')


def scatter(items, cache, w=330, h=250):
    pts = []
    for it in items:
        bb = cache[it["name"] + "|bbox"]
        for t in range(it["n"]):
            hu, pr = np.array(it["human"][t]), np.array(it["pred"][t])
            if np.linalg.norm(hu) >= 0.2 and np.linalg.norm(pr) >= 0.2 and bb[t][3] >= 0.5:
                pts.append((bb[t][0], np.degrees(np.arctan2(hu[1], hu[0])), np.degrees(np.arctan2(pr[1], pr[0]))))
    pts = np.array(pts); rng = np.random.RandomState(0); sel = pts[rng.choice(len(pts), min(700, len(pts)), replace=False)]
    X = lambda x: 40 + (w - 52) * (x - 0.1) / 0.8; Y = lambda y: h - 34 - (h - 56) * (y + 120) / 240
    o = f'<svg viewBox="0 0 {w} {h}" style="width:100%;max-width:{w}px;background:#0a1220;border:1px solid #1e293b;border-radius:8px">'
    for yv in (-90, -45, 0, 45, 90):
        o += f'<line x1="40" y1="{Y(yv):.0f}" x2="{w - 12}" y2="{Y(yv):.0f}" stroke="#1e293b"/><text x="36" y="{Y(yv) + 3:.0f}" fill="#64748b" font-size="8" text-anchor="end">{yv}</text>'
    for xv in (0.2, 0.35, 0.5, 0.65, 0.8):
        o += f'<text x="{X(xv):.0f}" y="{h - 20}" fill="#64748b" font-size="8" text-anchor="middle">{xv}</text>'
    o += f'<line x1="{X(0.5):.0f}" y1="20" x2="{X(0.5):.0f}" y2="{h - 34}" stroke="#475569" stroke-dasharray="3 3"/>'
    for x, a, b in sel:
        o += f'<circle cx="{X(x):.1f}" cy="{Y(a):.1f}" r="1.7" fill="{G}" opacity="0.45"/><circle cx="{X(x):.1f}" cy="{Y(b):.1f}" r="1.7" fill="{A}" opacity="0.45"/>'
    from scipy.stats import spearmanr
    rh, rp = spearmanr(pts[:, 0], pts[:, 1])[0], spearmanr(pts[:, 0], pts[:, 2])[0]
    o += f'<text x="44" y="13" fill="#e2e8f0" font-size="9" font-weight="700">bbox x(가로) vs 명령 각도(°)  ρ 사람 {rh:.2f} · 모델 {rp:.2f}</text></svg>'
    return o, rh, rp


def block(tbl, H, st):
    f = FM / "error_anatomy.json"
    if not f.exists():
        return ""
    d = json.load(open(f)); cache = np.load(FM / "v3_florence2_cache.npz")
    meta = json.load(open(FM / "anatomy/overlay_meta.json"))
    gal = ""
    for sp in d["splits"]:
        gal += f'<div style="margin:14px 0 4px;color:#fcd34d;font-weight:800;font-size:0.85rem">{SPN[sp]}</div><div style="display:flex;gap:12px;flex-wrap:wrap">'
        for role, (lab, col) in RN.items():
            m = meta[f"{sp}|{role}"]; it = next(x for x in d["splits"][sp] if x["name"] == m["name"])
            imgs = "".join(f'<img src="egvla2_fm/anatomy/overlay_{sp}_{role}_{k}.jpg" style="width:49%;border-radius:4px;margin:0 0.5% 4px 0">' for k in range(4))
            gal += (f'<div style="flex:1 1 380px;background:#0d1420;border:2px solid {col};border-radius:10px;padding:8px"><div style="font-size:0.8rem;font-weight:800;color:{col}">{lab} · 평균 {it["mean_err"]:.1f}° · {m["name"][:-3]}</div>{imgs}</div>')
        gal += "</div>"
    sc = {}
    for sp in d["splits"]:
        sc[sp] = scatter(d["splits"][sp], cache)
    scat = '<div style="display:flex;gap:12px;flex-wrap:wrap">' + "".join(f'<div style="flex:1 1 330px"><div style="font-size:0.78rem;color:#fcd34d;font-weight:800;margin-bottom:3px">{SPN[sp]}</div>{sc[sp][0]}</div>' for sp in sc) + '</div>'
    cal = tbl(["속력 단계(s)", "측정 속도", "측정 방법"], [["0.4 (느림)", "14.45 cm/s", "2m 직진 10회 평균"], ["0.7 (보통)", "20.64 cm/s", "2m 직진 10회 평균"], ["1.0 (빠름)", "26.36 cm/s", "2m 직진 10회 평균"], ["<b>선형식</b>", "v = 6.59 + 19.85·s (cm/s)", "30회(레벨당 10회) 적합"]])
    units = tbl(["이 문서의 cm 수치", "계산 방식", "근거·가정", "불확실성"], [
        ["<b>FPE</b> (종점 오차)", "프레임마다 (이동 방향 단위벡터 × 속도 × 프레임 간격 dt)를 누적해 경로를 만들고, 사람 경로와 모델 경로의 <b>종점 사이 거리</b>", "속도는 위 보정식, 모델·이산은 s=0.7 고정(섀도 방침), 사람은 라벨의 속력 단계. dt는 프레임 시각(중앙값 약 0.128초)", "직진 2m 측정만 있고 좌우·대각선 속도는 같다고 가정, 회전은 이동 0으로 처리, 바퀴 미끄럼·지연 미반영"],
        ["<b>ADE</b>", "같은 두 경로의 대응하는 점 사이 거리의 평균", "FPE와 같음", "FPE와 같음"],
        ["<b>좌우 오차</b>", "두 경로 종점의 좌우 방향 차이", "FPE와 같음", "FPE와 같음"],
        ["<b>1m당 옆 이탈</b>", "1m × sin(방향 오차)", "순수 기하(속도 보정 불필요)", "보정 없는 열린 고리 상한"],
        ["<b>지연 중 이동</b>", "지연(초) × 속도(cm/s). 예: 0.892초 × 20.6 = 약 18cm, × 26.4 = 약 24cm", "젯슨 왕복 892ms 실측, 속도 보정", "ASYNC(기본) 모드에서는 응답이 올 때까지 직전 명령을 유지하므로 이 거리만큼 이동. SYNC 모드(0.4초 이동 후 정지)에서는 해당 없음(CH78 ⚙️ 탭 참고)"],
        ["<b>경로 길이</b>", "프레임별 이동 거리의 합. 예: 평균 약 180cm(2m 직진 과제와 일치)", "속도 보정, dt", "—"]])
    ex = ("<div style='font-size:0.8rem;line-height:1.7;margin:6px 0'><b>예(한 프레임)</b>: 속력 단계 0.7이면 v = 6.59 + 19.85×0.7 = <b>20.5cm/s</b>, 프레임 간격 0.128초 → 한 프레임에 약 <b>2.6cm</b>. 모든 프레임이 0.7 단계로 이동한다면 100프레임에 약 2.6m이지만, 실제로는 "
          "이동하지 않는 프레임(약 13%)과 느린 단계가 섞여 평균 경로 길이는 약 1.8m다. FPE 31cm는 그 경로 길이의 약 <b>17%</b>이고, 방향이 틀린 시간이 길수록 이 값이 커진다.</div>")
    return (H("⑥-16 방향 오차의 '각도'와 '단위(cm)'는 무엇을 뜻하나 — 이미지 위에 그려 보기")
            + '<div style="font-size:0.8rem;line-height:1.6;margin:6px 0">방향 오차(°)는 <b>이미지 안의 각도가 아니다</b>. 모델은 이미지(Florence-2 특징과 bbox)를 보고 <b>로봇 몸체 좌표의 속도 명령 (lx, ly)</b>를 내고, 그 명령의 방향 θ=atan2(ly, lx)를 사람 조작과 비교한 값이다. 아래에서 (1) 좌표계, (2) 실제 프레임 위의 표시, (3) 이미지 위치와 명령 각도의 관계, (4) cm 단위의 계산 기준을 차례로 설명한다.</div>'
            + '<div style="margin:12px 0 4px;font-size:0.84rem;font-weight:800;color:#fcd34d">① 이미지 → 명령 각도: 무엇을 무엇으로 바꿨나</div>' + frame_diagram()
            + '<div style="margin:12px 0 4px;font-size:0.84rem;font-weight:800;color:#fcd34d">② 실제 프레임 위에 그린 것 — 파란 원=Florence-2 bbox, 흰 점선=정면 축, 우하단 나침반=명령 방향</div>' + gal
            + f'{st("읽는 법", A)} 파란 원은 모델이 본 목표(바구니)이고, 우하단 나침반의 초록·노랑 선이 사람·모델이 낸 <b>명령 방향</b>(위=전진, 왼쪽=좌)이다. 빨간 호가 방향 오차다. bbox가 정면에 있어도 사람이 좌커브를 택하면 명령은 왼쪽으로 가므로(최악 사례의 시작 프레임) <b>이미지 위치만으로 명령 각도가 정해지지 않는다</b>.'
            + '<div style="margin:12px 0 4px;font-size:0.84rem;font-weight:800;color:#fcd34d">③ 이미지 위치(bbox 가로)와 명령 각도의 관계</div>' + scat
            + f'{st("발견", G)} bbox가 오른쪽에 있을수록 명령이 오른쪽으로 향하는 음의 상관이 있지만 <b>약하다</b>(ρ 사람 약 {sc["fold0"][1]:.2f}·모델 {sc["fold0"][2]:.2f}). 같은 bbox 위치에서도 명령 각도가 넓게 퍼진다 = 경로 의도(좌커브·직진·우커브)가 각도를 좌우한다. 안 본 위치에서는 모델의 상관이 더 약해진다(ρ {sc["lopo_strong_right"][2]:.2f}).'
            + '<div style="margin:12px 0 4px;font-size:0.84rem;font-weight:800;color:#fcd34d">④ cm는 어떤 기준으로 나왔나</div>' + ex + cal + units
            + f'{st("한계", R)} cm는 <b>속도 보정(2m 직진 30회)과 프레임 간격으로 만든 추정</b>이지 실제 주행 경로를 측정한 값이 아니다. 실기 폐루프에서 로봇 위치 기록(오도메트리나 마커)이 있어야 같은 정의로 실측할 수 있다.')

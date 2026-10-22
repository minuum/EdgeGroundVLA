"""CH78 ⑥-14: EGVLA1 vs EGVLA2 출력 액션 비교(두 데이터셋, 실제 프레임·화살표·시계열·궤적)와 방향 오차 이외의 판단 기준. 데이터: egvla1_vs_egvla2.json"""
import json

import numpy as np

import ch77_viz as Z
import gen_ch76_viz as V

FM = V.FM
A, G, R, B, P, MUT = V.A, V.G, V.R, V.B, V.P, V.MUT
DSN = {"v3": "v3 연속 데이터(EGVLA2 수집, fold0 hold-out)", "v6": "V6 데이터(EGVLA1 수집, EGVLA1 검증 분할)"}
VA, VB = 6.59, 19.85


def angd(a, b):
    a, b = np.array(a[:2], float), np.array(b[:2], float); na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 0.2 or nb < 0.2:
        return None
    return float(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1, 1))))


def f1(pred, gt):
    tp = sum(p and g for p, g in zip(pred, gt)); fp = sum(p and not g for p, g in zip(pred, gt)); fn = sum(g and not p for p, g in zip(pred, gt))
    return 2 * tp / max(2 * tp + fp + fn, 1)


def path(vs, dts):
    p = [np.zeros(2)]
    for v, dt in zip(vs, dts):
        m = np.linalg.norm(v[:2])
        step = np.zeros(2) if m < 0.2 else np.array([v[0], v[1]]) / m * (VA + VB * 0.7) * dt
        p.append(p[-1] + step)
    return np.array(p)


def metrics(eps):
    out = {}
    for key, cls_key, has_key in (("e1", "e1cls", "e1_has"), ("e2", "e2cls", "e2_has")):
        fr = [r for e in eps for r in e["frames"]]
        agree = np.mean([r[cls_key] == r["hcls"] for r in fr]) * 100
        ang = [x for x in (angd(r[key], r["human"]) for r in fr) if x is not None]
        go = f1([np.linalg.norm(r[key][:2]) >= 0.2 for r in fr], [np.linalg.norm(r["human"][:2]) >= 0.2 for r in fr])
        rot = np.mean([r[cls_key] in (6, 7) for r in fr]) * 100
        sw = np.mean([e["frames"][t][cls_key] != e["frames"][t - 1][cls_key] for e in eps for t in range(1, e["n"])]) * 100
        has = np.mean([r[has_key] for r in fr]) * 100
        fpe = np.mean([np.linalg.norm(path([r[key] for r in e["frames"]], [r["dt"] for r in e["frames"]])[-1] - path([r["human"] for r in e["frames"]], [r["dt"] for r in e["frames"]])[-1]) for e in eps])
        out[key] = dict(agree=agree, ang=np.mean(ang), ang_med=np.median(ang), go=go, rot=rot, sw=sw, has=has, fpe=fpe, n=len(fr))
    fr = [r for e in eps for r in e["frames"]]
    out["human"] = dict(rot=np.mean([r["hcls"] in (6, 7) for r in fr]) * 100, sw=np.mean([e["frames"][t]["hcls"] != e["frames"][t - 1]["hcls"] for e in eps for t in range(1, e["n"])]) * 100)
    out["e1_ms"] = np.mean([r["e1_ms"] for r in fr]); out["e1_cached"] = np.mean([r["e1_cached"] for r in fr]) * 100; out["e1_latched"] = np.mean([r["e1_stop_latched"] for r in fr]) * 100
    return out


def obs(M, st):
    """수치에서 바로 읽히는 관찰(자동)."""
    v3, v6 = M.get("v3"), M.get("v6")
    if not (v3 and v6):
        return ""
    return (f'{st("관찰", G)}<ul style="margin:6px 0 6px 18px;font-size:0.8rem">'
            f'<li><b>자기 데이터에서</b>: EGVLA2(v3) 8-class 일치 {v3["e2"]["agree"]:.0f}%·방향 중앙값 {v3["e2"]["ang_med"]:.1f}°, EGVLA1(V6) 일치 {v6["e1"]["agree"]:.0f}%·방향 중앙값 {v6["e1"]["ang_med"]:.1f}°(이산이라 맞으면 0°).</li>'
            f'<li><b>교차 데이터에서</b>: EGVLA1은 v3에서 크게 흔들린다(일치 {v3["e1"]["agree"]:.0f}%, 방향 평균 {v3["e1"]["ang"]:.0f}°). EGVLA2는 V6에서 중간 수준(일치 {v6["e2"]["agree"]:.0f}%, 방향 평균 {v6["e2"]["ang"]:.0f}°) — 두 모델 모두 다른 수집 조건에 약하지만 EGVLA2 쪽이 덜 무너진다.</li>'
            f'<li><b>EGVLA1은 자기 데이터에서도 프레임 일치가 {v6["e1"]["agree"]:.0f}%인데 실기는 92%다.</b> 프레임 단위 일치는 성공률을 예측하지 못한다는 레포의 반복 결론과 같은 방향이다(매 프레임 다시 보고 고치는 폐루프가 오차를 흡수). 기록된 val_acc 74.1%와의 차이는 채점 방식(오프라인 OWL 주석 bbox·stride 채점 vs 이번 온라인 그라운딩·JPEG q80·재사용 3)에서 온다.</li>'
            f'<li><b>정지</b>: GO/STOP F1이 EGVLA1 {v3["e1"]["go"]:.2f}/{v6["e1"]["go"]:.2f}, EGVLA2 {v3["e2"]["go"]:.2f}/{v6["e2"]["go"]:.2f}. EGVLA1의 정지는 실기에서 근접 규칙(래치)이 맡는데 이 에피소드들에서는 래치가 걸리지 않았다(0%).</li>'
            f'<li><b>그라운딩</b>: OWLv2 검출 프레임 {v3["e1"]["has"]:.0f}%(v3)·{v6["e1"]["has"]:.0f}%(V6) vs Florence-2 100%. EGVLA1 프레임당 약 {M["v3"]["e1_ms"]:.0f}ms(이 서버 fp32, 그라운딩 재사용 {M["v3"]["e1_cached"]:.0f}% 포함).</li></ul>')


def block(tbl, H, st):
    f = FM / "egvla1_vs_egvla2.json"
    if not f.exists():
        return ""
    d = json.load(open(f))
    import ch77_anatomy as AN
    M = {ds: metrics(eps) for ds, eps in d.items() if eps}
    groups = [(DSN[ds].split("(")[0], ["EGVLA1", "EGVLA2"], [R, A]) for ds in M]
    rows = [("8-class 일치율(사람 대비, %)", [[M[ds]["e1"]["agree"], M[ds]["e2"]["agree"]] for ds in M], "high"),
            ("방향 오차 평균(°)", [[M[ds]["e1"]["ang"], M[ds]["e2"]["ang"]] for ds in M], "low"),
            ("방향 오차 중앙값(°)", [[M[ds]["e1"]["ang_med"], M[ds]["e2"]["ang_med"]] for ds in M], "low"),
            ("GO/STOP F1", [[M[ds]["e1"]["go"], M[ds]["e2"]["go"]] for ds in M], "high"),
            ("궤적 종점 차(cm, 속력 0.7 동일)", [[M[ds]["e1"]["fpe"], M[ds]["e2"]["fpe"]] for ds in M], "low"),
            ("클래스 전환 빈도(%/프레임)", [[M[ds]["e1"]["sw"], M[ds]["e2"]["sw"]] for ds in M], None),
            ("회전 출력 비율(%)", [[M[ds]["e1"]["rot"], M[ds]["e2"]["rot"]] for ds in M], None),
            ("그라운딩 성공 프레임(%)", [[M[ds]["e1"]["has"], M[ds]["e2"]["has"]] for ds in M], "high")]
    mt = Z.heat_table(groups, rows, fmt="{:.1f}", corner="지표 (데이터셋별 EGVLA1 vs EGVLA2)")
    ref = tbl(["데이터셋", "사람의 회전 비율", "사람의 클래스 전환 빈도", "EGVLA1 프레임당 시간(이 머신, 그라운딩 재사용 포함)", "EGVLA1 그라운딩 재사용", "EGVLA1 정지 래치 후 프레임"],
              [[DSN[ds], f'{M[ds]["human"]["rot"]:.1f}%', f'{M[ds]["human"]["sw"]:.1f}%', f'{M[ds]["e1_ms"]:.0f} ms', f'{M[ds]["e1_cached"]:.0f}%', f'{M[ds]["e1_latched"]:.0f}%'] for ds in M])
    vis = ""
    for ds, eps in d.items():
        vis += f'<div style="margin:16px 0 6px;color:#fcd34d;font-weight:800;font-size:0.86rem">{DSN[ds]}</div>'
        for e in eps[:6]:
            fr = e["frames"]; key = e["name"].replace(".h5", "")
            imgs = "".join(f'<img src="egvla2_fm/compare/{ds}_{key}_{k}.jpg" style="width:16%;border-radius:4px;margin-right:0.5%">' for k in range(6))
            da = lambda v: float(np.degrees(np.arctan2(v[1], v[0]))) if np.linalg.norm(v[:2]) >= 0.2 else None
            ts = AN.linechart([("사람", G, "", [(r["t"], da(r["human"])) for r in fr]), ("EGVLA1", R, "", [(r["t"], da(r["e1"])) for r in fr]),
                               ("EGVLA2", A, 'stroke-dasharray="5 3"', [(r["t"], da(r["e2"])) for r in fr])], w=330, h=170, xmax=e["n"] - 1, title="이동 방향(°, 0=전진, +=좌) — 정지·회전은 빈칸")
            ph, p1, p2 = (path([r[k] for r in fr], [r["dt"] for r in fr]) for k in ("human", "e1", "e2"))
            allp = np.vstack([ph, p1, p2]); lat0, lat1 = allp[:, 1].min(), allp[:, 1].max(); fw = max(allp[:, 0].max(), 1); w, h = 300, 220
            sc = min((w - 50) / max(lat1 - lat0, 60), (h - 50) / max(fw, 60)); Pt = lambda p: f"{w - 25 - (p[1] - lat0) * sc:.1f},{h - 25 - p[0] * sc:.1f}"
            tj = (f'<svg viewBox="0 0 {w} {h}" style="width:100%;max-width:{w}px;background:#0a1220;border:1px solid #1e293b;border-radius:8px"><text x="10" y="14" fill="#e2e8f0" font-size="10" font-weight="700">적분 궤적(속력 0.7 동일, 위=전진)</text>'
                  + "".join(f'<polyline points="{" ".join(Pt(p) for p in pp)}" fill="none" stroke="{c}" stroke-width="2.2" {dd}/>' for pp, c, dd in ((ph, G, ""), (p1, R, ""), (p2, A, 'stroke-dasharray="6 3"'))) + '</svg>')
            m1 = np.mean([r["e1cls"] == r["hcls"] for r in fr]) * 100; m2 = np.mean([r["e2cls"] == r["hcls"] for r in fr]) * 100
            vis += (f'<div style="background:#0d1420;border:1px solid #1e293b;border-radius:10px;padding:8px 10px;margin:8px 0"><div style="font-size:0.8rem;font-weight:800;color:#e2e8f0">{key} · {e["n"]}프레임 · '
                    f'8-class 일치 <span style="color:{R}">EGVLA1 {m1:.0f}%</span> / <span style="color:{A}">EGVLA2 {m2:.0f}%</span></div><div style="margin:4px 0">{imgs}</div>'
                    f'<div style="display:flex;gap:10px;flex-wrap:wrap"><div style="flex:1 1 320px">{ts}</div><div style="flex:1 1 280px">{tj}</div></div></div>')
    crit = tbl(["판단 기준(방향 오차 외)", "왜 중요한가", "지금 잴 수 있나", "현재 값/상태"], [
        ["<b>실기 성공률(SR)</b>", "최종 목표. 오프라인 오차와 연결이 약함", "폐루프만", "EGVLA1 92.0%(276/300), EGVLA2 미측정"],
        ["<b>충돌·수동 개입률(CR)</b>", "안전. 경로 모호성에서 평균 방향이 장애물로 향할 위험", "폐루프 기록", "기록 요청됨(R7)"],
        ["<b>정지 판단</b>", "도착·정지 실패는 바로 실패", "오프라인 GO/STOP F1(부분)", "위 표 GO/STOP F1, 폐루프 근접 규칙"],
        ["<b>그라운딩 가용성</b>", "레포 결론: gnd%≥80이면 98.8% 성공", "오프라인 has_bbox 비율", "위 표 그라운딩 성공 프레임"],
        ["<b>행동 안정성</b>", "흔들림·잦은 전환은 실기에서 비틀림·지연 누적", "전환 빈도·저크", "위 표 클래스 전환 빈도, CH77 평활 스윕"],
        ["<b>반응 지연</b>", "반응 지연이 이동 거리로 환산되면(연속 구동 가정, 892ms ≈ 18cm). 현재 루프는 추론 중 정지 상태라 실제 이동은 판단당 약 8cm", "벤치·젯슨 실측", "EGVLA1 젯슨 약 2.0s / EGVLA2 892ms"],
        ["<b>경로 모호성 처리</b>", "시작 구간 다봉 — 평균 방향이 안전한가", "샘플 분석·폐루프", "CH77 ⑥-13: 시작 47% 프레임에서 샘플 갈림"],
        ["<b>일반화</b>", "안 본 목표 위치·시작 위치", "LOPO·pos2/3 재생", "LOPO 16.9°, pos2/3 대기"],
        ["<b>교차 데이터셋 견고성</b>", "다른 수집(조명·운전 습관)에서도 동작하는가", "상대 데이터셋에서 평가(이 섹션)", "위 표의 상대 데이터셋 열"],
        ["<b>자원(메모리·전력)</b>", "경량화 논문의 핵심", "젯슨 측정", "파라미터 458M→231M, 전력 미측정"]])
    return (H("⑥-14 EGVLA1 vs EGVLA2 — 같은 프레임에서 두 모델이 낸 액션 비교와 방향 오차 이외의 판단 기준")
            + '<div style="font-size:0.8rem;line-height:1.6;margin:6px 0">두 데이터셋(각 모델이 학습에 안 쓴 에피소드, 목표 위치별 2개씩 10개)의 같은 프레임에서 <b>사람(초록)·EGVLA1(빨강)·EGVLA2(노랑)</b>가 낸 액션을 비교했다. '
              'EGVLA1은 운영 서버 클래스를 그대로 실행(OWL 임계값 0.20, 그라운딩 재사용 3, JPEG q80), EGVLA2는 서빙 정책(K=8, EMA 0.5, az=0). 사진 아래 화살표: 왼쪽부터 사람·EGVLA1·EGVLA2(위=전진, 왼쪽=좌, 원=정지, 호=회전).</div>'
            + Z.LEG + mt + ref + obs(M, st)
            + f'{st("주의", R)} 각 모델은 상대 데이터셋을 본 적이 없다(v3에서 EGVLA1, V6에서 EGVLA2는 교차 데이터셋). EGVLA2는 회전을 내지 않으므로(az=0) 사람이 회전한 프레임은 8-class 일치에서 불리하다. V6는 프레임 시각이 없어 0.2초 간격으로 적분했다. EGVLA1의 실기 92%에는 이 오프라인 실행에 없는 프리뷰 재시도 등이 일부 포함된다.'
            + H("방향 오차 외에 무엇으로 판단하나") + crit
            + H("실제 프레임에서 두 모델의 출력 (에피소드별)") + vis)

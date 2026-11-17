"""S1 v2 — 11/17 실측(near_area=0.18) 쌍(첫 STOPPED 박스 면적 ↔ 동환 cm)으로 면적→간격을 직접 보정한 정지 정책 시뮬레이션.
v1(stop_policy_sim.py)은 물체별 1점 앵커 + 접촉 면적 가정이었다. v2는 가정 대신 실측 5~10점 회귀를 쓴다.
성공 기준(사용자 지정): 바구니 5cm, 의자 17.5cm, 허용 ±5cm(임시), 접촉=간격<0.
한계: 쌍 매칭은 '동환 보고 순서 = 세션 시각 순서' 가정, 손 측정, 물체별 5~10점, 열린 루프, 같은 11/16 경로로 임계값을 고른다(LOO로 완화)."""
import csv, json, math, statistics as st
from pathlib import Path
import stop_policy_sim as S

ROOT = Path(__file__).resolve().parent.parent
RC = ROOT / "_received_1117_near018"
pairs = list(csv.DictReader(open(RC / "calibration_pairs_1117.csv", encoding="utf-8")))
sess17 = json.load(open(RC / "sessions_1117_near018_area.json"))


def lin(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    res = [y - (a + b * x) for x, y in zip(xs, ys)]
    return a, b, (sum(r * r for r in res) / max(len(res) - 2, 1)) ** 0.5


def spearman(xs, ys):
    rk = lambda v: [sorted(v).index(x) + 1 for x in v]
    rx, ry = rk(xs), rk(ys)
    mx, my = st.mean(rx), st.mean(ry)
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))


# 규칙 재현 검증(11/17: T=0.18, nf=2)
hit = tot = 0
for sid, v in sess17.items():
    ni = S._near_index(v["area_seq"], 0.18, False)
    tot += 1; hit += (ni == v["first_stopped_step"])
print(f"규칙 재현(T=0.18, 첫 STOPPED 프레임 일치): {hit}/{tot}")

out = {"rule_repro_1117": [hit, tot], "calib": {}}
for o in ("basket", "chair"):
    P = [(1 / math.sqrt(float(r["area_at_first_stopped"])), float(r["dh_cm(+safe/-bump)"])) for r in pairs if r["object"] == o]
    z, g = zip(*P)
    a, b, sd = lin(z, g)
    rho = spearman(list(z), list(g))
    zc = -a / b
    a_contact = 1 / zc ** 2
    t_area = 1 / ((S.TARGET[o] - a) / b) ** 2
    print(f"\n[{o}] n={len(P)} 회귀 gap = {a:.1f} + {b:.1f}·(1/√area) | 잔차SD {sd:.1f}cm | 순위상관 {rho:.2f} | 접촉 면적(간격0) {a_contact:.3f} | 목표 {S.TARGET[o]}cm 면적 {t_area:.3f}")
    out["calib"][o] = {"n": len(P), "alpha": round(a, 2), "beta": round(b, 2), "resid_sd_cm": round(sd, 1), "spearman": round(rho, 2), "area_at_gap0": round(a_contact, 3), "target_area": round(t_area, 3)}
    k, zc_ = b, zc
    out[o] = {}
    pols = {
        "P0a 현행 near=0.25": (lambda s: 0.25, {}),
        "P0b 오늘 near=0.18": (lambda s: 0.18, {}),
        "P1 물체별 near(LOO)": (lambda s: S.best_T(o, k, zc_, s), {}),
        "P2 다음스텝 예측+물체별(LOO)": (lambda s: S.best_T(o, k, zc_, s, pred=True), {"pred": True}),
    }
    S.LAG = 0
    for name, (pick, kw) in pols.items():
        rows = S.run_policy(o, k, zc_, pick, **kw)
        m = S.metrics(rows, o)
        # 보정 잔차(SD)까지 반영한 대역 안 확률: 간격 ~ N(추정, SD)
        pin = st.mean(0.5 * (math.erf((S.TARGET[o] + S.BAND - r["gap"]) / (sd * math.sqrt(2))) - math.erf((S.TARGET[o] - S.BAND - r["gap"]) / (sd * math.sqrt(2)))) for r in rows)
        pc = st.mean(0.5 * (1 + math.erf((0 - r["gap"]) / (sd * math.sqrt(2)))) for r in rows)
        m["in_band_with_noise"] = round(pin, 2); m["contact_with_noise"] = round(pc, 2)
        out[o][name] = m
        print(f"  {name:30s} 간격중앙 {m['gap_median']:6.1f}cm [{m['gap_min']:.0f}~{m['gap_max']:.0f}] 대역내 {m['in_band']:.0%}(잡음반영 {pin:.0%}) 접촉 {m['contact']:.0%}(잡음반영 {pc:.0%}) T중앙 {m['T_median']}")
json.dump(out, open(ROOT / "docs/v5/egvla2_fm/a3c/stop_policy_sim_v2.json", "w"), ensure_ascii=False, indent=1)

"""S1 — 정지 정책 오프라인 시뮬레이터 (plan_20261117_stop_policy.md).

입력: 11/16 a3-s2 실기 세션 45개의 로봇 기록 박스 면적 시계열과, 같은 프레임에서 a3-s2 헤드가 낸 출력 크기(replay_1116.json).
성공 기준(사용자 지정, 동환 11/17 손 측정 기반): 물체마다 '근사하게 정지하는 값 하나'
    바구니 5cm, 의자 17.5cm(이상적 15~20cm의 중간) — 허용 ±5cm(임시값), 접촉 = 간격<0.
면적→간격 변환(근사): 핀홀 모델 gap = k·(1/√area − 1/√area_접촉).
    k는 동환 11/17 시험(near_area=0.18)의 물체별 간격 중앙값 1점으로 보정, 접촉 면적은 범위로 두고 민감도를 본다.
한계(출력에도 표기): 열린 루프(정지 이후 경로 변화 없음), 보정은 물체별 1점·손 측정, 기록이 실제 정지에서 끝나므로 더 늦게 멈추는 정책은 외삽.
"""
from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
A = ROOT / "docs/v5/egvla2_fm/a3c"
R = json.load(open(A / "replay_1116.json"))
SESS = {k: v for k, v in R.items() if not k.startswith("_")}
HEAD_MODEL = "a3-s2"
HEAD_THR = 0.2
NF = 2
HEAD_PERSIST = 2          # 헤드 idle을 정지로 인정하는 연속 프레임 수(시뮬레이션 가정)

OBJ = lambda s: "basket" if "basket" in s else "chair" if "chair" in s else "desk"
TARGET = {"basket": 5.0, "chair": 17.5}
BAND = 5.0
# 동환 11/17 17:11 (near_area=0.18) 손 측정, 부딪침은 음수
OBS_GAP = {"basket": [5, 1, 5, 3, -3, 1, 2, -2, 5, -3], "chair": [25, 25, 30, 35, 25]}  # 의자 25~35 중 5개 값은 원문 25/25/35/25/25
OBS_GAP["chair"] = [30, 25, 35, 25, 25]
CONTACT_AREA = {"basket": (0.34, 0.40), "chair": (0.40, 0.50)}   # 바구니: 접촉 세션 마지막 면적 0.34~0.38 / 의자: 근거 없는 가정 범위


def series(sid: str):
    v = SESS[sid]
    a = list(v["area_rec"])
    h = [math.hypot(*c) for c in v["cmd"][HEAD_MODEL]]
    n = min(len(a), len(h))
    return a[:n], h[:n]


def extend(a: list[float], extra: int = 4) -> list[float]:
    """기록이 끝난 뒤 면적은 1/√area 선형(마지막 3스텝 기울기)으로 외삽."""
    if len(a) < 4:
        return a + [a[-1]] * extra
    z = [1 / math.sqrt(x) for x in a[-4:]]
    slope = (z[-1] - z[0]) / 3
    out = list(a)
    zc = z[-1]
    for _ in range(extra):
        zc = max(zc + slope, 0.9)
        out.append(1 / zc ** 2)
    return out


LAG = 0      # near 규칙이 걸린 뒤 실제로 멈추기까지의 지연(프레임). main()에서 11/16 기록으로 추정


def _near_index(a: list[float], T: float, pred: bool) -> int | None:
    run = 0
    for i, x in enumerate(a):
        if pred and i > 0:
            x = x + max(0.0, x - a[i - 1])
        run = run + 1 if x >= T else 0
        if run >= NF:
            return i
    return None


def stop_index(a: list[float], h: list[float], T: float, *, pred: bool = False, head: bool = True, head_floor: float = 0.0):
    """정지 프레임 인덱스와 사유.
    근거: 기록 세션은 실제 컨트롤러(near 0.25 + 헤드 idle)가 마지막 프레임 전에는 멈추지 않았음을 보여 준다.
    그래서 헤드·운영자에 의한 실제 종료는 '기록 마지막 프레임'에서만 일어난 정지 사건으로 취급하고(헤드를 끄거나 바닥값 미만이면 제외),
    near 규칙은 후보 임계값으로 다시 계산해 LAG만큼 늦춰 반영한다. 둘 중 이른 쪽."""
    last = len(h) - 1 if h else len(a) - 1
    cands = []
    ni = _near_index(a, T, pred)
    if ni is not None:
        cands.append((min(ni + LAG, len(a) - 1), "near"))
    if head and last >= 0 and h[last] < HEAD_THR and a[last] >= head_floor:
        cands.append((last, "head"))
    if not cands:
        return len(a) - 1, "end"
    return min(cands)


def sid_of(o: str) -> list[str]:
    return [k for k, v in SESS.items() if OBJ(v["instruction"]) == o]


def calibrate(o: str, a_c: float):
    """T=0.18 현행 정책(near+head)을 11/16 경로에 적용한 정지 면적 중앙값 ↔ 동환 간격 중앙값 → k."""
    ar = []
    for sid in sid_of(o):
        a, h = series(sid)
        a = extend(a)
        i, _ = stop_index(a, h, 0.18)
        ar.append(a[i])
    a_anchor = st.median(ar)
    g = st.median(OBS_GAP[o])
    zc = 1 / math.sqrt(a_c)
    k = g / (1 / math.sqrt(a_anchor) - zc)
    return k, zc, a_anchor


def gap_of(area: float, k: float, zc: float) -> float:
    return k * (1 / math.sqrt(area) - zc)


def area_of_gap(g: float, k: float, zc: float) -> float:
    return 1 / (g / k + zc) ** 2


def run_policy(o: str, k: float, zc: float, pick, **kw):
    """pick(sid_excluded)->T. LOO: 해당 세션을 뺀 나머지로 T를 고른다."""
    out = []
    for sid in sid_of(o):
        T = pick(sid)
        a, h = series(sid)
        a = extend(a)
        i, why = stop_index(a, h, T, **kw)
        out.append({"sid": sid, "T": round(T, 3), "why": why, "area": a[i], "gap": gap_of(a[i], k, zc), "extrap": i >= len(series(sid)[0])})
    return out


def best_T(o: str, k: float, zc: float, excl: str | None, **kw):
    grid = [round(0.06 + 0.01 * j, 2) for j in range(0, 33)]
    best, bs = None, 1e9
    for T in grid:
        gs = []
        for sid in sid_of(o):
            if sid == excl:
                continue
            a, h = series(sid)
            a = extend(a)
            i, _ = stop_index(a, h, T, **kw)
            gs.append(abs(gap_of(a[i], k, zc) - TARGET[o]))
        m = st.mean(gs)
        if m < bs:
            best, bs = T, m
    return best


def metrics(rows: list[dict], o: str) -> dict:
    g = [r["gap"] for r in rows]
    t = TARGET[o]
    return {
        "n": len(g),
        "gap_median": round(st.median(g), 1), "gap_min": round(min(g), 1), "gap_max": round(max(g), 1),
        "in_band": round(sum(abs(x - t) <= BAND for x in g) / len(g), 2),
        "contact": round(sum(x < 0 for x in g) / len(g), 2),
        "too_early": round(sum(x > t + BAND for x in g) / len(g), 2),
        "by_head": round(sum(r["why"] == "head" for r in rows) / len(rows), 2),
        "by_near": round(sum(r["why"] == "near" for r in rows) / len(rows), 2),
        "extrapolated": round(sum(r["extrap"] for r in rows) / len(rows), 2),
        "T_median": round(st.median([r["T"] for r in rows]), 3),
    }


def main() -> None:
    global LAG
    lags = []
    for sid in SESS:
        a, h = series(sid)
        ni = _near_index(a, 0.25, False)
        if ni is not None and ni <= len(a) - 1:
            lags.append(len(a) - 1 - ni)
    LAG = int(round(st.median(lags)))
    print('LAG(프레임) =', LAG, '표본', len(lags), '분포', sorted(set(lags)))
    res: dict = {"meta": {"head_model": HEAD_MODEL, "target_gap_cm": TARGET, "band_cm": BAND, "obs_gap_cm_1117": OBS_GAP,
                          "contact_area_range": CONTACT_AREA, "head_thr": HEAD_THR, "lag_frames": None, "near_frames": NF}, "objects": {}}
    pols = {
        "P0a 현행 near=0.25": lambda o, k, zc: (lambda s: 0.25, {}),
        "P0b 오늘 near=0.18": lambda o, k, zc: (lambda s: 0.18, {}),
        "P1 물체별 near(LOO)": lambda o, k, zc: (lambda s: best_T(o, k, zc, s), {}),
        "P2 다음스텝 예측+물체별 near(LOO)": lambda o, k, zc: (lambda s: best_T(o, k, zc, s, pred=True), {"pred": True}),
        "P4 P1+헤드바닥 0.15(LOO)": lambda o, k, zc: (lambda s: best_T(o, k, zc, s, head_floor=0.15), {"head_floor": 0.15}),
        "P5 헤드 끔+물체별 near(LOO)": lambda o, k, zc: (lambda s: best_T(o, k, zc, s, head=False), {"head": False}),
    }
    for o in ("basket", "chair"):
        res["objects"][o] = {}
        for ac in CONTACT_AREA[o]:
            k, zc, anc = calibrate(o, ac)
            tgt_area = area_of_gap(TARGET[o], k, zc)
            blk = {"anchor_stop_area": round(anc, 3), "k": round(k, 2), "target_area": round(tgt_area, 3), "policies": {}}
            for name, mk in pols.items():
                pick, kw = mk(o, k, zc)
                rows = run_policy(o, k, zc, pick, **kw)
                blk["policies"][name] = metrics(rows, o)
            res["objects"][o][f"a_contact={ac}"] = blk
    # 책상: 간격 보정 없음 → 면적만
    desk = {}
    for name, T in (("near=0.25", 0.25), ("near=0.18", 0.18)):
        ar = []
        for sid in sid_of("desk"):
            a, h = series(sid); a = extend(a); i, _ = stop_index(a, h, T); ar.append(a[i])
        desk[name] = {"stop_area_median": round(st.median(ar), 3), "range": [round(min(ar), 3), round(max(ar), 3)]}
    res["desk_area_only"] = desk
    json.dump(res, open(A / "stop_policy_sim.json", "w"), ensure_ascii=False, indent=1)
    for o, blks in res["objects"].items():
        for ac, b in blks.items():
            print(f"\n[{o}] {ac} 보정앵커 정지면적 {b['anchor_stop_area']} → 목표({TARGET[o]}cm) 면적 {b['target_area']}")
            for n, m in b["policies"].items():
                print(f"  {n:34s} 간격중앙 {m['gap_median']:6.1f}cm [{m['gap_min']:.0f}~{m['gap_max']:.0f}] 대역내 {m['in_band']:.0%} 접촉 {m['contact']:.0%} 너무일찍 {m['too_early']:.0%} | 헤드 {m['by_head']:.0%} near {m['by_near']:.0%} 외삽 {m['extrapolated']:.0%} T중앙 {m['T_median']}")
    print("\n책상(면적만):", desk)


if __name__ == "__main__":
    main()

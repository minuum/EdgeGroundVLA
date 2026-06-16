# -*- coding: utf-8 -*-
"""
시점(viewpoint) 분류별 grounding 경향 분석 (CPU only, grounding 캐시 기반).

basket 위치(HSV cx) × 거리(HSV area)로 9개 시점 버킷 분류 →
각 (모델 × 버킷)에서 tracking 오차 / full-frame율 / miss율 측정.
"객체 인식이 어떤 시점에서 무너지는가"를 체계적으로 점검.

산출: docs/v5/grounding_ablation/viewpoint_analysis.json
Usage: python3 scripts/analyze_grounding_viewpoints.py
"""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GDIR = ROOT / "docs/v5/grounding_ablation"
MODELS = ["base", "exp57", "exp58", "exp59"]


def pos_bucket(cx):
    return "L" if cx < 0.4 else ("R" if cx > 0.6 else "C")


def dist_bucket(area):
    return "far" if area < 0.05 else ("near" if area > 0.3 else "mid")


def main():
    hsv = json.loads((GDIR / "grounding_hsv.json").read_text())
    # ref[(ep, fi)] = (cx, area, viewpoint)
    ref = {}
    for ep in hsv:
        for fr in ep["frames"]:
            if fr["cx"] is not None:
                vp = f"{pos_bucket(fr['cx'])}-{dist_bucket(fr.get('area') or 0)}"
                ref[(ep["episode"], fr["frame_idx"])] = (fr["cx"], fr.get("area") or 0, vp)

    caches = {m: json.loads((GDIR / f"grounding_{m}.json").read_text()) for m in MODELS}
    # per (model, viewpoint): tracking err, fullframe, miss, n
    stat = {m: defaultdict(lambda: {"err": [], "full": 0, "miss": 0, "n": 0}) for m in MODELS}
    vp_n = defaultdict(int)

    for m in MODELS:
        for ep in caches[m]:
            for fr in ep["frames"]:
                key = (ep["episode"], fr["frame_idx"])
                if key not in ref:
                    continue
                cx_ref, _, vp = ref[key]
                s = stat[m][vp]; s["n"] += 1
                if m == MODELS[0]:
                    vp_n[vp] += 1
                if not fr["hit"] or fr["cx"] is None:
                    s["miss"] += 1; continue
                s["err"].append(abs(fr["cx"] - cx_ref))
                if (fr["area"] or 0) > 0.9:
                    s["full"] += 1

    order = [f"{p}-{d}" for p in "LCR" for d in ("far", "mid", "near")]
    out = {"viewpoints": {}, "n_per_vp": dict(vp_n)}
    print(f"{'viewpoint':<10}{'n':>5}", end="")
    for m in MODELS: print(f"{m:>22}", end="")
    print(); print(f"{'':<15}", end="")
    for m in MODELS: print(f"{'err / full / miss':>22}", end="")
    print("\n" + "-" * 105)
    for vp in order:
        if vp_n.get(vp, 0) == 0: continue
        print(f"{vp:<10}{vp_n[vp]:>5}", end="")
        out["viewpoints"][vp] = {"n": vp_n[vp]}
        for m in MODELS:
            s = stat[m][vp]
            err = np.mean(s["err"]) if s["err"] else None
            fr_full = s["full"] / max(s["n"], 1)
            fr_miss = s["miss"] / max(s["n"], 1)
            cell = f"{(f'{err:.2f}' if err is not None else '—')}/{fr_full*100:.0f}%/{fr_miss*100:.0f}%"
            print(f"{cell:>22}", end="")
            out["viewpoints"][vp][m] = {"cx_err": (float(err) if err is not None else None),
                                        "fullframe_rate": fr_full, "miss_rate": fr_miss, "n": s["n"]}
        print()
    print("\n(셀 = cx오차 / full-frame율 / miss율)  L=basket좌 C=중앙 R=우 · far/mid/near=거리")
    (GDIR / "viewpoint_analysis.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[SAVE] {GDIR}/viewpoint_analysis.json")


if __name__ == "__main__":
    main()

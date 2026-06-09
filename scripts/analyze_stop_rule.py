#!/usr/bin/env python3
"""
STOP 규칙 캘리브레이션 (plan_20260602_stop_arrival_rule.md S1)

도착 STOP 규칙:
  area_det 최근 W프레임 평균 > TH_AREA  AND  |cx_det-0.5| < TH_CX
  AND step >= MIN_STEPS  (AND rising 옵션)  → STOP (래치: 한 번 멈추면 유지)

이 규칙을 243ep PG2 데이터에 프레임별 시뮬레이션하여 (TH_AREA, TH_CX, W, MIN_STEPS, rising)
grid sweep → STOP 탐지 recall / 조기오발 / 타이밍 오차 산출.

산출: docs/v5/stop_rule_calibration.json

Usage:
  python3 scripts/analyze_stop_rule.py
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ANN  = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_pg2_cx.json"
OUT  = ROOT / "docs/v5/stop_rule_calibration.json"


def simulate(frames, th_area, th_cx, W, min_steps, rising):
    """규칙을 프레임별 적용 → 최초 STOP 트리거 인덱스(없으면 None)."""
    buf = []
    for i, fr in enumerate(frames):
        area = fr.get("area_det", 0.0)
        cx   = fr.get("cx_det", 0.5)
        buf.append(area)
        if len(buf) > W:
            buf = buf[-W:]
        area_avg = float(np.mean(buf))
        is_rising = (buf[-1] >= buf[0]) if len(buf) >= 2 else True
        centered  = abs(cx - 0.5) < th_cx
        if (i >= min_steps and area_avg > th_area and centered
                and (is_rising or not rising)):
            return i      # 최초 트리거
    return None


def gt_first_stop(frames):
    for i, fr in enumerate(frames):
        if fr.get("gt_class") == 0:
            return i
    return None


def main():
    data = json.loads(ANN.read_text())

    grid = []
    for th_area in (0.4, 0.5, 0.6, 0.7):
        for th_cx in (0.15, 0.20, 0.30, 1.0):       # 1.0 = cx 무시
            for W in (1, 3, 5):
                for min_steps in (0, 3):
                    for rising in (False, True):
                        grid.append((th_area, th_cx, W, min_steps, rising))

    results = []
    for (th_area, th_cx, W, min_steps, rising) in grid:
        # STOP 있는 ep: 트리거율 + 타이밍 오차 / 없는 ep: 조기오발
        timing_err_frames = []      # |rule_trig - gt_stop| (STOP ep)
        timing_err_phase  = []
        triggered_stop_ep = 0
        n_stop_ep = 0
        # STOP 없는 ep에서 트리거된 phase (조기성 판단)
        notrig_trigger_phase = []
        false_trigger_ep = 0
        n_nostop_ep = 0

        for ep in data:
            frames = ep["frames"]
            n = len(frames)
            if n == 0:
                continue
            trig = simulate(frames, th_area, th_cx, W, min_steps, rising)
            gstop = gt_first_stop(frames)

            if gstop is not None:
                n_stop_ep += 1
                if trig is not None:
                    triggered_stop_ep += 1
                    timing_err_frames.append(abs(trig - gstop))
                    timing_err_phase.append(abs(trig - gstop) / max(n - 1, 1))
            else:
                n_nostop_ep += 1
                if trig is not None:
                    false_trigger_ep += 1
                    notrig_trigger_phase.append(trig / max(n - 1, 1))

        rec = {
            "th_area": th_area, "th_cx": th_cx, "W": W,
            "min_steps": min_steps, "rising": rising,
            "stop_ep_trigger_rate": triggered_stop_ep / max(n_stop_ep, 1),
            "stop_ep_timing_err_frames_med": float(np.median(timing_err_frames)) if timing_err_frames else None,
            "stop_ep_timing_err_phase_med": float(np.median(timing_err_phase)) if timing_err_phase else None,
            # STOP 없는 ep에서 트리거 비율 + 그 트리거가 얼마나 일찍(phase) 일어났나
            "nostop_ep_trigger_rate": false_trigger_ep / max(n_nostop_ep, 1),
            "nostop_trigger_phase_med": float(np.median(notrig_trigger_phase)) if notrig_trigger_phase else None,
            # 조기성: phase<0.7에서 트리거된 ep 비율 (진짜 조기오발)
            "nostop_premature_rate": (float(np.mean(np.array(notrig_trigger_phase) < 0.7))
                                      if notrig_trigger_phase else 0.0),
        }
        results.append(rec)

    # 추천: STOP ep 트리거율 높고, 타이밍 오차 작고, 조기오발(premature) 낮은 것
    def score(r):
        if r["stop_ep_timing_err_phase_med"] is None:
            return -1e9
        return (r["stop_ep_trigger_rate"]
                - 2.0 * r["stop_ep_timing_err_phase_med"]
                - 3.0 * r["nostop_premature_rate"])
    best = max(results, key=score)

    OUT.write_text(json.dumps(
        {"dataset": str(ANN.relative_to(ROOT)), "n_grid": len(grid),
         "recommended": best, "grid": results}, indent=2, ensure_ascii=False))

    print(f"=== STOP 규칙 캘리브레이션 ({len(data)} ep, grid {len(grid)}) ===")
    print(f"STOP ep: {sum(1 for e in data if gt_first_stop(e['frames']) is not None)} / {len(data)}")
    print(f"\n[추천 config] {best['th_area']=} {best['th_cx']=} W={best['W']} "
          f"min_steps={best['min_steps']} rising={best['rising']}")
    print(f"  STOP ep 트리거율    : {best['stop_ep_trigger_rate']*100:.0f}%")
    print(f"  타이밍 오차(phase med): {best['stop_ep_timing_err_phase_med']:.3f}")
    print(f"  타이밍 오차(frame med): {best['stop_ep_timing_err_frames_med']:.1f}")
    print(f"  STOP없는 ep 트리거율 : {best['nostop_ep_trigger_rate']*100:.0f}% "
          f"(트리거 phase med={best['nostop_trigger_phase_med']})")
    print(f"  조기오발(phase<0.7)  : {best['nostop_premature_rate']*100:.0f}%")
    print(f"\n[SAVE] {OUT}")

    # 참고: TH_AREA별 대표값 (W=3, th_cx=0.2, min_steps=3, rising=False)
    print("\n[참고] W=3, th_cx=0.2, min_steps=3, rising=False:")
    for r in results:
        if r["W"]==3 and r["th_cx"]==0.20 and r["min_steps"]==3 and not r["rising"]:
            te = r["stop_ep_timing_err_phase_med"]
            print(f"  th_area={r['th_area']}: STOP트리거={r['stop_ep_trigger_rate']*100:.0f}% "
                  f"타이밍오차={te if te is None else round(te,3)} "
                  f"조기오발={r['nostop_premature_rate']*100:.0f}%")


if __name__ == "__main__":
    main()

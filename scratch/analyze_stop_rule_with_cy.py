#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
ANN  = ROOT / "docs/v5/bbox_frame_level/bbox_dataset_pg2_cx.json"

def simulate(frames, th_area, th_cx, th_cy, W, min_steps):
    """cy 게이트가 추가된 정지 시뮬레이션."""
    buf_area = []
    buf_cy = []
    for i, fr in enumerate(frames):
        area = fr.get("area_det", 0.0)
        cx   = fr.get("cx_det", 0.5)
        cy   = fr.get("cy_det", 0.0)
        
        buf_area.append(area)
        buf_cy.append(cy)
        if len(buf_area) > W:
            buf_area = buf_area[-W:]
            buf_cy = buf_cy[-W:]
            
        area_avg = float(np.mean(buf_area))
        cy_avg = float(np.mean(buf_cy))
        
        centered  = abs(cx - 0.5) < th_cx
        # cy_avg > th_cy 조건 추가
        if (i >= min_steps and area_avg > th_area and centered and cy_avg > th_cy):
            return i      # 최초 트리거
    return None

def gt_first_stop(frames):
    for i, fr in enumerate(frames):
        if fr.get("gt_class") == 0:
            return i
    return None

def main():
    data = json.loads(ANN.read_text())

    # 스윕 그리드 생성
    grid = []
    for th_area in (0.4, 0.5, 0.6):
        for th_cx in (0.2, 0.3, 1.0):
            for th_cy in (0.0, 0.40, 0.42, 0.45, 0.50):  # 0.0 = cy 게이트 무시
                for W in (3, 5):
                    for min_steps in (0, 3):
                        grid.append((th_area, th_cx, th_cy, W, min_steps))

    results = []
    for (th_area, th_cx, th_cy, W, min_steps) in grid:
        timing_err_frames = []
        timing_err_phase  = []
        triggered_stop_ep = 0
        n_stop_ep = 0
        
        notrig_trigger_phase = []
        false_trigger_ep = 0
        n_nostop_ep = 0

        for ep in data:
            frames = ep["frames"]
            n = len(frames)
            if n == 0:
                continue
            trig = simulate(frames, th_area, th_cx, th_cy, W, min_steps)
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
            "th_area": th_area, "th_cx": th_cx, "th_cy": th_cy, "W": W,
            "min_steps": min_steps,
            "stop_ep_trigger_rate": triggered_stop_ep / max(n_stop_ep, 1),
            "stop_ep_timing_err_frames_med": float(np.median(timing_err_frames)) if timing_err_frames else None,
            "stop_ep_timing_err_phase_med": float(np.median(timing_err_phase)) if timing_err_phase else None,
            "nostop_ep_trigger_rate": false_trigger_ep / max(n_nostop_ep, 1),
            "nostop_trigger_phase_med": float(np.median(notrig_trigger_phase)) if notrig_trigger_phase else None,
            "nostop_premature_rate": (float(np.mean(np.array(notrig_trigger_phase) < 0.7))
                                      if notrig_trigger_phase else 0.0),
        }
        results.append(rec)

    # 채점 함수: STOP ep 트리거율 높고, 오차가 적으며, STOP 없는 ep에서의 오발율(premature_rate)이 0에 가깝도록
    def score(r):
        if r["stop_ep_timing_err_phase_med"] is None:
            return -1e9
        return (r["stop_ep_trigger_rate"] 
                - 1.5 * r["stop_ep_timing_err_phase_med"] 
                - 4.0 * r["nostop_premature_rate"]
                - 0.5 * r["nostop_ep_trigger_rate"])  # 오발률 패널티 강화
                
    sorted_results = sorted(results, key=score, reverse=True)
    
    print(f"=== STOP 규칙 캘리브레이션 (+ cy 게이트 추가 스윕) ===")
    print(f"전체 에피소드: {len(data)}")
    print(f"STOP 라벨 존재 ep: {n_stop_ep} / STOP 없는 ep: {n_nostop_ep}\n")
    
    print("🏆 [상위 5개 최적 Config 비교]")
    for idx, best in enumerate(sorted_results[:5]):
        print(f"\n{idx+1}위 Config:")
        print(f"  설정: th_area={best['th_area']} \| th_cx={best['th_cx']} \| th_cy={best['th_cy']} \| W={best['W']} \| min_steps={best['min_steps']}")
        print(f"  - 진짜 도착 탐지율 (Recall)   : {best['stop_ep_trigger_rate']*100:.1f}%")
        print(f"  - 도착 타이밍 오차 (frame med) : {best['stop_ep_timing_err_frames_med']:.1f} frames")
        print(f"  - 도착 타이밍 오차 (phase med) : {best['stop_ep_timing_err_phase_med']:.3f}")
        print(f"  - 가짜 ep에서의 오발율 (False) : {best['nostop_ep_trigger_rate']*100:.1f}% (phase_med={best['nostop_trigger_phase_med']})")
        print(f"  - 조기 오발율 (phase < 0.7)  : {best['nostop_premature_rate']*100:.1f}%")

if __name__ == "__main__":
    main()

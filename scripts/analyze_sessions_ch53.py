#!/usr/bin/env python3
"""CH53 — 6/24·6/26 실세션 + v5_2 신규 데이터셋 분석."""
import h5py
import numpy as np
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent


def vel_to_label(v):
    vx, vy, vr = float(v[0]), float(v[1]), float(v[2])
    if abs(vx) < 0.05 and abs(vy) < 0.05 and abs(vr) < 0.05:
        return "STOP"
    if vr > 0.3:
        return "ROT_R"
    if vr < -0.3:
        return "ROT_L"
    if vx > 0.3 and vy < -0.3:
        return "FWD+L"
    if vx > 0.3 and vy > 0.3:
        return "FWD+R"
    if vx > 0.1:
        return "FWD"
    if vy < -0.1:
        return "LEFT"
    if vy > 0.1:
        return "RIGHT"
    return "OTHER"


def analyze_inference_session(sp):
    with h5py.File(sp, "r") as f:
        acts = f["actions"][:]
        bbox = f["grounding/bbox"][:]
        cached = f["grounding/cached"][:]
        attrs = dict(f.attrs)
        lat = f["grounding/latency_ms"][:] if "grounding/latency_ms" in f else np.array([])
        n = len(acts)
        areas = bbox[:, 2]
        has_bbox = bbox[:, 3]
        n_grounded = int((cached == 0).sum())
        n_cached = int((cached != 0).sum())
        skip_n_detected = int(np.max(cached)) if n > 0 else 1
        act_labels = [vel_to_label(a) for a in acts]
        cnt = Counter(act_labels)
        real_lat = lat[lat > 0] if len(lat) > 0 else np.array([])
        return {
            "session": sp.stem,
            "n_frames": n,
            "n_grounded": n_grounded,
            "n_cached": n_cached,
            "skip_n_detected": skip_n_detected,
            "area_min": float(areas.min()) if n > 0 else 0,
            "area_max": float(areas.max()) if n > 0 else 0,
            "area_mean": float(areas.mean()) if n > 0 else 0,
            "has_bbox_count": int((has_bbox > 0.5).sum()),
            "action_dist": dict(cnt),
            "lat_mean_ms": float(real_lat.mean()) if len(real_lat) > 0 else None,
            "lat_max_ms": float(real_lat.max()) if len(real_lat) > 0 else None,
        }


def analyze_v5_2():
    v5_2_root = ROOT / "ROS_action/mobile_vla_dataset_v5_2"
    eps = sorted(v5_2_root.glob("*.h5"))
    groups = {}
    for ep in eps:
        with h5py.File(ep, "r") as f:
            sc = dict(f.attrs).get("scenario", "unknown")
            acts = f["actions"][:]
            n = len(acts)
            try:
                instr = f["language_instruction"][0].decode()
            except Exception:
                instr = ""
            labels = [vel_to_label(a) for a in acts]
            if sc not in groups:
                groups[sc] = []
            groups[sc].append({
                "file": ep.name,
                "n_frames": n,
                "actions": labels,
                "instruction": instr,
            })
    summary = {}
    for sc, recs in sorted(groups.items()):
        all_acts = [a for r in recs for a in r["actions"]]
        c = Counter(all_acts)
        summary[sc] = {
            "n_episodes": len(recs),
            "avg_frames": float(np.mean([r["n_frames"] for r in recs])),
            "action_dist": dict(c),
            "sample_instruction": recs[0]["instruction"][:120] if recs else "",
        }
    return summary


if __name__ == "__main__":
    print("=== 6/24·6/26 추론 세션 분석 ===")
    sessions_dir = ROOT / "docs/inference_sessions"
    all_sessions = []
    for date in ["20260624", "20260626"]:
        sps = sorted(sessions_dir.glob(f"session_{date}*.h5"))
        for sp in sps:
            r = analyze_inference_session(sp)
            all_sessions.append(r)
            print(f"\n[{r['session']}]")
            print(f"  frames={r['n_frames']}  grounded={r['n_grounded']}  cached={r['n_cached']}  skip_n={r['skip_n_detected']}")
            print(f"  area: min={r['area_min']:.4f} max={r['area_max']:.4f} mean={r['area_mean']:.4f}")
            print(f"  has_bbox={r['has_bbox_count']}/{r['n_frames']}")
            print(f"  actions: {r['action_dist']}")
            if r["lat_mean_ms"]:
                print(f"  latency: mean={r['lat_mean_ms']:.0f}ms max={r['lat_max_ms']:.0f}ms")

    print("\n\n=== v5_2 신규 데이터셋 ===")
    v5_2 = analyze_v5_2()
    for sc, s in v5_2.items():
        print(f"[{sc}]  n_ep={s['n_episodes']}  avg_frames={s['avg_frames']:.1f}")
        print(f"  actions: {s['action_dist']}")
        print(f"  instr: {s['sample_instruction']}")

    out = {"inference_sessions": all_sessions, "v5_2_dataset": v5_2}
    out_path = ROOT / "docs/v5/ch53_session_analysis.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\n[저장] {out_path}")

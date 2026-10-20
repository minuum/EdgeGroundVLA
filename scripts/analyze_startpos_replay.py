#!/usr/bin/env python3
"""시작 위치별(pos1/pos2/pos3) 서버 재생 결과(replay_episodes_through_server.py의 JSON)를 합쳐 docs/v5/egvla2_fm/startpos_replay.json 으로 저장한다.
사용: python3 scripts/analyze_startpos_replay.py --pos1 /tmp/replay_pos1.json --pos2 /tmp/replay_pos2.json --pos3 /tmp/replay_pos3.json   (있는 것만)
기준선(같은 지표·같은 평활 설정 K=8, EMA 새 값 0.5, 안 본 에피소드): 5-fold fold0 방향 오차 10.24°, LOPO strong_right 18.31° (smoothing_sweep.json).
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
for k in ("pos1", "pos2", "pos3"):
    ap.add_argument(f"--{k}")
A = ap.parse_args()
sw = json.load(open(ROOT / "docs/v5/egvla2_fm/smoothing_sweep.json"))
out = dict(baseline=dict(fold0_angle=sw["fold0"]["K8|a0.5"]["angle"], lopo_angle=sw["lopo_strong_right"]["K8|a0.5"]["angle"],
                         fold0_f1=sw["fold0"]["K8|a0.5"]["go_f1"], lopo_f1=sw["lopo_strong_right"]["K8|a0.5"]["go_f1"],
                         fold0_jitter=sw["fold0"]["K8|a0.5"]["jitter"], lopo_jitter=sw["lopo_strong_right"]["K8|a0.5"]["jitter"]), starts={})
for k in ("pos1", "pos2", "pos3"):
    p = getattr(A, k)
    if p:
        d = json.load(open(p)); out["starts"][k] = dict(summary=d["summary"], episodes=d["episodes"])
(ROOT / "docs/v5/egvla2_fm/startpos_replay.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(json.dumps({k: v["summary"] for k, v in out["starts"].items()}, indent=1, ensure_ascii=False))

# -*- coding: utf-8 -*-
"""
STOP/근접 로직 ablation — Phase C (plan §10-C, Task #4).

발견(연구 단계): S7 "near-miss"는 grounding이 틀린 게 아니었다 — f53에서 area=0.297로
실제 GOAL_AREA(0.25) 임계값을 단독으로는 충분히 넘었지만, GOAL_CONSEC_FRAMES=3
(3프레임 연속) 조건을 못 채워(앞뒤 프레임 area 0.07 수준) proximity STOP이 한 번도
발동하지 않았다. 이후 바스켓을 완전히 놓침(NO_BBOX 연속). 즉 "detector가 틀렸다"가
아니라 "STOP 임계값/연속프레임 설계가 단발성 근접을 노이즈로 걸러내는 부작용"이 의심.

S6는 frame 24에서 학습된 STOP(class 0, raw model 예측)이 발동한 뒤 105프레임 끝까지
한 번도 안 풀린다(0/82 복귀) — proximity_override 없이도 동일 동작이라
"learned-with-latch" vs "proximity-without-latch"가 S6에서는 차이가 없다.
즉 grounding_hub.html §H4의 dead-zone 풀프레임 환각(f56/70/85)은 이미 STOP된
이후에 벌어진 일이라 로봇 동작에는 영향이 없었다.

이 스크립트는 운영 서버(stage2_v2_inference_server.py)를 건드리지 않고,
기존에 기록된 S6/S7 세션의 프레임별 bbox 히스토리(docs/v5/s6_cl_sim.json,
s7_cl_sim.json)에 대해 STOP 결정 로직만 **로컬로 재생(replay)**해 3가지
모드 + GOAL_CONSEC_FRAMES 스윕을 비교한다.

모드:
  - proximity   : 운영 기본값과 동일한 로직(stage2_v2_inference_server.py:499-517 재현)
  - learned     : 기록된 label=='STOP'(raw model 예측)이 한 번 뜨면 latch
  - hybrid      : proximity 조건 OR learned 조건, latch 없음(매 프레임 재평가)

Usage:
  .venv/bin/python3 scripts/eval/ablation_stop_mode.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SESSIONS = {
    "s6_dead_zone": ROOT / "docs/v5/s6_cl_sim.json",
    "s7_near_miss": ROOT / "docs/v5/s7_cl_sim.json",
}
GOAL_AREA = 0.25
GOAL_CX = 0.35
CONSEC_SWEEP = [1, 2, 3, 4, 5]


def replay_proximity(frames, consec_n):
    """운영 코드와 동일 로직: 직전 consec_n 프레임이 전부 근접조건 만족해야 발동."""
    hist = []
    first_stop = None
    for f in frames:
        b = f["bbox"]
        hist.append(b)
        last_n = hist[-consec_n:]
        if len(last_n) >= consec_n:
            near = sum(1 for h in last_n
                      if h.get("has_bbox") and h.get("area", 0) >= GOAL_AREA
                      and abs(h.get("cx", 0.5) - 0.5) <= GOAL_CX)
            if near >= consec_n:
                first_stop = f["frame"]
                break
    return first_stop


def replay_learned_latch(frames):
    """raw model label=='STOP'이 한 번 뜨면 그 프레임부터 latch."""
    for f in frames:
        if f["label"] == "STOP":
            return f["frame"]
    return None


def replay_hybrid_no_latch(frames, consec_n):
    """proximity 조건 OR learned 조건, latch 없음(매 프레임 독립 재평가)."""
    hist = []
    for f in frames:
        b = f["bbox"]
        hist.append(b)
        last_n = hist[-consec_n:]
        prox = False
        if len(last_n) >= consec_n:
            near = sum(1 for h in last_n
                      if h.get("has_bbox") and h.get("area", 0) >= GOAL_AREA
                      and abs(h.get("cx", 0.5) - 0.5) <= GOAL_CX)
            prox = near >= consec_n
        learned = f["label"] == "STOP"
        if prox or learned:
            return f["frame"]
    return None


def main():
    for name, path in SESSIONS.items():
        d = json.loads(path.read_text())
        frames = d["frames"]
        n = len(frames)
        print(f"\n{'=' * 60}\n[{name}] {d['session']} — {n}프레임\n{'=' * 60}")

        learned_first = replay_learned_latch(frames)
        print(f"  learned(latch)  : 첫 STOP frame={learned_first}")

        for c in CONSEC_SWEEP:
            prox_first = replay_proximity(frames, c)
            hybrid_first = replay_hybrid_no_latch(frames, c)
            tag = " ← 운영 기본값" if c == 3 else ""
            print(f"  proximity(n={c})  : 첫 STOP frame={prox_first}{tag}")
            print(f"  hybrid(n={c})     : 첫 STOP frame={hybrid_first}{tag}")

        # area 최대값 — 단발성 근접 스파이크가 있었는지 확인
        areas = [f["bbox"]["area"] for f in frames if f["bbox"]["has_bbox"]]
        if areas:
            mx = max(areas)
            mx_frame = next(f["frame"] for f in frames if f["bbox"]["area"] == mx)
            print(f"  [참고] area 최댓값={mx:.3f} (frame {mx_frame}) — "
                  f"GOAL_AREA({GOAL_AREA}) {'초과' if mx >= GOAL_AREA else '미달'}")


if __name__ == "__main__":
    main()

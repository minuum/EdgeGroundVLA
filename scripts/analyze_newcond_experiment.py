#!/usr/bin/env python3
"""심사위원1-②/3-①(환경 독립성) 대응 — 출발위치2·3 그리드 + 조도 ablation 결과 집계.

2026-09-14 교수님 논의로 확정된 실험(CH72-6, revision_briefing.html#realrobot-newcond)의
episode_log.csv 신규 행을 집계해 Wilson 95% CI와 함께 표로 뽑는다. 기존 100건(위치1,
Table 7)과 같은 형식으로 나란히 비교 가능하게 만드는 게 목표.

로깅 컨벤션(코드 변경 없이 기존 "경로" 필드 네이밍만 확장 — trackA_strong_right 등
기존 컨벤션과 동일한 방식):
  pos2_<direction>   — 신규 출발위치2, 5방향 각 n=5
  pos3_<direction>   — 신규 출발위치3, 5방향 각 n=5
  light_<direction>  — 기존 출발위치1에서 조도만 변경, 5방향 각 n=10
  <direction> ∈ {center, strong_left, weak_left, strong_right, weak_right}

Usage:
  python3 scripts/analyze_newcond_experiment.py
  python3 scripts/analyze_newcond_experiment.py --csv logs/episode_log.csv
"""
import argparse
import csv
from collections import defaultdict
from math import sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "logs" / "episode_log.csv"

DIRECTIONS = ["center", "strong_left", "weak_left", "strong_right", "weak_right"]
CONDITION_PREFIXES = {"pos2_": "출발위치2 (신규)", "pos3_": "출발위치3 (신규)", "light_": "조도 ablation (위치1)"}


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h) * 100, min(1.0, c + h) * 100


def load_rows(csv_path):
    with open(csv_path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def match_condition(path_type):
    for prefix, label in CONDITION_PREFIXES.items():
        if path_type.startswith(prefix):
            return prefix, label, path_type[len(prefix):]
    return None, None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"[!] {csv_path} 없음 — 아직 신규 실기 데이터가 기록되지 않았습니다.")
        return
    rows = load_rows(csv_path)

    buckets = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # prefix -> direction -> [succ, total]
    unmatched_new_style = 0
    for r in rows:
        prefix, label, direction = match_condition(r.get("경로", ""))
        if prefix is None:
            continue
        if direction not in DIRECTIONS:
            unmatched_new_style += 1
            continue
        succ = 1 if r.get("결과", "").strip() == "성공" else 0
        buckets[prefix][direction][0] += succ
        buckets[prefix][direction][1] += 1

    if not buckets:
        print("[!] pos2_/pos3_/light_ 접두어를 가진 행이 아직 없습니다.")
        print("    실기 기록 시 '경로' 필드를 예: pos2_center, pos3_weak_left, light_strong_right 로 남겨주세요.")
        return

    print("=" * 78)
    print("새 조건 실기 실험 집계 — 심사위원1-②/3-① 대응 (2026-09-14 교수님 논의 확정안)")
    print("=" * 78)

    for prefix, label in CONDITION_PREFIXES.items():
        by_dir = buckets.get(prefix)
        if not by_dir:
            print(f"\n[{label}] — 아직 기록된 데이터 없음")
            continue
        print(f"\n[{label}]")
        print(f"  {'방향':14s} {'성공/시행':>10s} {'성공률':>8s} {'Wilson 95% CI'}")
        tot_s, tot_n = 0, 0
        for d in DIRECTIONS:
            s, n = by_dir.get(d, [0, 0])
            if n == 0:
                continue
            tot_s += s
            tot_n += n
            lo, hi = wilson(s, n)
            print(f"  {d:14s} {s:>4d}/{n:<5d} {100*s/n:>6.1f}%  [{lo:5.1f}%, {hi:5.1f}%]")
        if tot_n:
            lo, hi = wilson(tot_s, tot_n)
            print(f"  {'합계':14s} {tot_s:>4d}/{tot_n:<5d} {100*tot_s/tot_n:>6.1f}%  [{lo:5.1f}%, {hi:5.1f}%]")

    print(f"\n참고: 기존 위치1(Table 7) 기준 95/100 (95.0%)")
    if unmatched_new_style:
        print(f"\n[경고] pos2_/pos3_/light_ 접두어는 있으나 방향명이 5가지 표준"
              f"({', '.join(DIRECTIONS)})과 안 맞는 행 {unmatched_new_style}개 — 오타 확인 필요")


if __name__ == "__main__":
    main()

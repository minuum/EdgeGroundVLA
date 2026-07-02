#!/usr/bin/env python3
"""
경로검증 탭 에피소드 기록(logs/episode_log.csv) + 추론 세션(docs/inference_sessions/*.h5)
분석 스크립트.

2026-07-02 대화에서 수동으로 하던 분석(오브젝트 위치별 성공률/FPE, 실패 메모
키워드 분류, 최근 세션 프레임별 grounding 덤프)을 스킬화한 것.
자세한 사용법은 .agent/skills/episode-analysis/SKILL.md 참조.

사용법:
    python3 scripts/analysis/analyze_episode_log.py                # 전체 집계 + 실패 분류
    python3 scripts/analysis/analyze_episode_log.py --group obj    # obj_* 만
    python3 scripts/analysis/analyze_episode_log.py --recent       # 최근 H5 세션 프레임별 덤프
    python3 scripts/analysis/analyze_episode_log.py --session 20260702_200616
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EPISODE_CSV = ROOT / "logs" / "episode_log.csv"
H5_DIR = ROOT / "docs" / "inference_sessions"

# 실패 메모 키워드 → 분류. 2026-07-02 분석에서 실제로 관찰된 패턴 기준.
# 순서 중요 — 먼저 매칭되는 카테고리로 분류됨.
FAILURE_PATTERNS = [
    ("프리뷰/회전 반복", ["도리도리", "빙글뱅글", "반복되는", "제자리"]),
    ("그라운딩 오탐/미탐", ["다른 객체", "다른객체", "인식", "엇나", "잘못", "이상한 객"]),
    ("경로 이탈(방향 확정 후)", ["오른쪽으로 가버", "왼쪽으로 가버", "직선방향으로", "너무 돌아버"]),
    ("각도/정밀도 부족", ["각도", "살짝"]),
    ("로딩/중단", ["로드되다가", "멈추는"]),
]

# 2026-07-02 분석 기준: FPE 0.2 미만은 대체로 성공, 이상은 대체로 실패.
FPE_SUCCESS_THRESHOLD = 0.20


def _classify_failure(note: str) -> str:
    for label, keywords in FAILURE_PATTERNS:
        if any(kw in note for kw in keywords):
            return label
    return "기타/미분류" if note.strip() and note.strip() != "—" else "메모 없음"


def _read_episodes() -> list[dict]:
    if not EPISODE_CSV.exists():
        raise SystemExit(f"episode_log.csv 없음: {EPISODE_CSV}")
    rows = []
    with open(EPISODE_CSV, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        for r in reader:
            if len(r) < 12:
                continue
            rows.append({
                "n": r[0], "path": r[1], "result": r[2],
                "steps": int(r[3]) if r[3] else 0,
                "lat_ms": float(r[4]) if r[4] else 0.0,
                "top_action": r[5], "gnd_pct": r[6], "area": r[7], "cx": r[8],
                "stop": r[9], "fpe": float(r[10]) if r[10] else None,
                "note": r[11], "date": r[12] if len(r) > 12 else "",
                # session_id는 2026-07-02 이후 기록에만 있음(구버전 행은 빈 문자열) —
                # 없으면 dump_session_frames()에서 날짜로 근사 매칭.
                "session_id": r[13] if len(r) > 13 else "",
            })
    return rows


def print_group_stats(rows: list[dict], group_prefix: str | None = None):
    filtered = [r for r in rows if group_prefix is None or r["path"].startswith(group_prefix)]
    if not filtered:
        print(f"[{group_prefix or '전체'}] 해당하는 에피소드 없음")
        return

    by_path = defaultdict(list)
    for r in filtered:
        by_path[r["path"]].append(r)

    print(f"\n{'경로/위치':<16} {'건수':>4} {'성공률':>8} {'평균FPE':>8} {'평균steps':>10} {'평균lat(ms)':>12}")
    print("-" * 64)
    for path, rs in sorted(by_path.items()):
        succ = sum(1 for r in rs if r["result"] == "성공")
        fpes = [r["fpe"] for r in rs if r["fpe"] is not None]
        avg_fpe = sum(fpes) / len(fpes) if fpes else float("nan")
        avg_steps = sum(r["steps"] for r in rs) / len(rs)
        avg_lat = sum(r["lat_ms"] for r in rs) / len(rs)
        print(f"{path:<16} {len(rs):>4} {succ}/{len(rs)} ({succ/len(rs)*100:>3.0f}%) "
              f"{avg_fpe:>8.2f} {avg_steps:>10.1f} {avg_lat:>12.0f}")

    # FPE 기준 성공/실패 분리 검증
    succ_fpes = [r["fpe"] for r in filtered if r["result"] == "성공" and r["fpe"] is not None]
    fail_fpes = [r["fpe"] for r in filtered if r["result"] == "실패" and r["fpe"] is not None]
    if succ_fpes and fail_fpes:
        print(f"\nFPE 분포 — 성공 평균 {sum(succ_fpes)/len(succ_fpes):.2f} "
              f"vs 실패 평균 {sum(fail_fpes)/len(fail_fpes):.2f} "
              f"(기준값 {FPE_SUCCESS_THRESHOLD} 사용 중, configs 아님 — 데이터 늘어나면 재검증 필요)")

    # 실패 메모 분류
    fails = [r for r in filtered if r["result"] == "실패"]
    if fails:
        by_cat = defaultdict(list)
        for r in fails:
            by_cat[_classify_failure(r["note"])].append(r)
        print(f"\n실패 {len(fails)}건 유형 분류:")
        for cat, rs in sorted(by_cat.items(), key=lambda x: -len(x[1])):
            print(f"  {cat:<20} {len(rs)}건  (예: #{rs[0]['n']} \"{rs[0]['note'][:40]}\")")


def _find_episode_notes(sid: str, rows: list[dict] | None = None) -> list[dict]:
    """세션 ID로 episode_log.csv에서 해당하는 기록(메모 포함)을 찾는다.
    2026-07-02 이후 기록은 session_id로 정확히 매칭, 그 이전 구버전 기록은
    session_id가 비어있어 세션 시작시각(±3분)으로 근사 매칭한다."""
    if rows is None:
        try:
            rows = _read_episodes()
        except SystemExit:
            return []
    exact = [r for r in rows if r.get("session_id") == sid]
    if exact:
        return exact
    try:
        from datetime import datetime as _dt
        sess_ts = _dt.strptime(sid, "%Y%m%d_%H%M%S")
    except ValueError:
        return []
    near = []
    for r in rows:
        try:
            row_ts = _dt.strptime(r["date"], "%Y-%m-%d %H:%M")
        except (ValueError, KeyError):
            continue
        if abs((row_ts - sess_ts).total_seconds()) <= 180:
            near.append(r)
    return near


def dump_session_frames(sid: str):
    import h5py
    h5p = H5_DIR / f"session_{sid}.h5"
    if not h5p.exists():
        raise SystemExit(f"세션 없음: {h5p}")
    with h5py.File(h5p, "r") as f:
        acts = f["actions"][()]
        bbox = f["grounding/bbox"][()]
        cached = f["grounding/cached"][()]
        lats = f["grounding/latency_ms"][()]
        attrs = dict(f.attrs)

    print(f"\n=== 세션 {sid} ===")
    print(f"attrs: {attrs}")

    notes = _find_episode_notes(sid)
    if notes:
        print("episode_log.csv 매칭 기록:")
        for r in notes:
            print(f"  #{r['n']} {r['path']} 결과={r['result']} FPE={r['fpe']} "
                  f"메모: \"{r['note']}\"")
    else:
        print("(episode_log.csv에 매칭되는 기록 없음 — 기록 저장 버튼을 안 눌렀을 수 있음)")
    print(f"{'#':>3} {'action':<10} {'cx':>6} {'cy':>6} {'area':>7} {'has':>5} "
          f"{'cached':>6} {'lat(ms)':>8}")
    amap = {
        (0.0, 0.0): "STOP", (1.15, 0.0): "FWD", (0.0, 1.15): "LEFT", (0.0, -1.15): "RIGHT",
        (1.15, 1.15): "FWD+L", (1.15, -1.15): "FWD+R",
    }
    for i in range(len(acts)):
        lx, ly = float(acts[i][0]), float(acts[i][1])
        az = float(acts[i][2]) if acts.shape[1] > 2 else 0.0
        if abs(az) > 0.05:
            lbl = "ROT_L" if az > 0 else "ROT_R"
        else:
            lbl = amap.get((round(lx, 2), round(ly, 2)), f"({lx:.2f},{ly:.2f})")
        cx, cy, area, has = bbox[i]
        cached_i = "cache" if cached[i] == 1 else ("live" if cached[i] == 0 else "none")
        print(f"{i:>3} {lbl:<10} {cx:>6.3f} {cy:>6.3f} {area:>7.4f} {bool(has)!s:>5} "
              f"{cached_i:>6} {lats[i]:>8.0f}")

    # 그라운딩 손실 구간 자동 탐지
    lost_start = None
    for i in range(len(acts)):
        has = bool(bbox[i][3])
        if not has and lost_start is None:
            lost_start = i
        elif has:
            lost_start = None
    if lost_start is not None and lost_start < len(acts) - 1:
        print(f"\n⚠️ 프레임 {lost_start}부터 세션 끝까지 grounding 미탐지 지속 "
              f"({len(acts) - lost_start}프레임) — fallback bbox로 방향 근거 없이 주행했을 가능성")


def _latest_session_id() -> str:
    files = sorted(glob.glob(str(H5_DIR / "session_*.h5")), key=os.path.getmtime, reverse=True)
    if not files:
        raise SystemExit("세션 H5 없음")
    name = Path(files[0]).stem  # session_YYYYMMDD_HHMMSS
    return name.replace("session_", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default=None,
                     help="경로 접두사 필터 (예: obj_, dist_, right_, left_, center_)")
    ap.add_argument("--session", default=None, help="특정 세션 ID 프레임별 덤프")
    ap.add_argument("--recent", action="store_true", help="가장 최근 세션 프레임별 덤프")
    args = ap.parse_args()

    if args.session:
        dump_session_frames(args.session)
        return
    if args.recent:
        dump_session_frames(_latest_session_id())
        return

    rows = _read_episodes()
    print(f"전체 에피소드: {len(rows)}건 (출처: {EPISODE_CSV})")
    if args.group:
        print_group_stats(rows, args.group)
    else:
        print_group_stats(rows, "obj_")
        print_group_stats(rows, "dist_")
        for prefix in ("right_", "center_", "left_"):
            print_group_stats(rows, prefix)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""EGVLA1 vs EGVLA2 비교 논문 실기 실험용 현장 기록 시트 생성기 & 자동 집계·논문 표 생성기.

사용법:
  1) 현장 기록 CSV 템플릿 생성 (기본 조건당 N=3회, 총 108행; --reps 5로 180행 가능):
     .venv/bin/python3 scripts/aggregate_paper_eg1_eg2.py --init-sheet --reps 3
  2) 실기 완료 후 JSON/H5 + 현장 CSV 자동 집계 및 논문용 Markdown/LaTeX 표 출력:
     .venv/bin/python3 scripts/aggregate_paper_eg1_eg2.py --sheet docs/v5/paper_eg1_eg2_field_sheet.csv --reports-dir _received_paper_reports
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SHEET = ROOT / "docs/v5/paper_eg1_eg2_field_sheet.csv"
DEFAULT_OUT_JSON = ROOT / "docs/v5/egvla2_fm/paper_eg1_eg2_results.json"

STARTS = ["pos1", "pos2", "pos3"]
TARGETS = ["G6", "G7", "G8"]
EG1_SPECS = [("EGVLA1", "exp73_owl_stage1v3_v6_mlp.pt", "바구니", "gray basket")]
EG2_SPECS = [
    ("EGVLA2", "A3C_AG28_s0.pt", "바구니", "go to the gray basket"),
    ("EGVLA2", "A3C_AG28_s0.pt", "의자", "go to the brown wooden chair"),
    ("EGVLA2", "A3C_AG28_s0.pt", "책상", "go to the small desk"),
]

HEADER = [
    "trial_id",
    "model",
    "ckpt",
    "object",
    "instruction",
    "start_pos",
    "target_pos",
    "rep",
    "session_id",
    "outcome",
    "contact_yn",
    "remain_dist_cm",
    "notes",
]


def init_sheet(out_csv: Path, reps: int = 3) -> None:
    """EGVLA1(9조건) + EGVLA2(27조건) × reps회 현장 기록 CSV 템플릿 생성."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    tid = 1
    for model, ckpt, obj, ins in EG1_SPECS + EG2_SPECS:
        for sp in STARTS:
            for tp in TARGETS:
                for r in range(1, reps + 1):
                    rows.append([f"T{tid:03d}", model, ckpt, obj, ins, sp, tp, r, "", "", "", "", ""])
                    tid += 1
    with out_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(rows)
    print(f"Generated field sheet: {out_csv} ({len(rows)} trials, reps={reps})")


def aggregate(sheet_path: Path, reports_dir: Path | None, out_json: Path) -> None:
    """현장 CSV와 session_*.json을 조인해 모델·객체별 및 9경로별 논문 지표 자동 집계."""
    with sheet_path.open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    # 완료된(outcome 입력된) 행만 집계
    done = [r for r in rows if r.get("outcome", "").strip()]
    print(f"총 시트 행: {len(rows)}건 | 기록 완료 행: {len(done)}건")
    if not done:
        print("[안내] 아직 outcome이 채워진 실기 행이 없어 스키마 검증만 완료했습니다.")
        return

    # 리포트 JSON 매핑 (session_id -> summary)
    rep_map = {}
    if reports_dir and reports_dir.exists():
        for jp in reports_dir.glob("session_*.json"):
            d = json.loads(jp.read_text(encoding="utf-8"))
            rep_map[d.get("session_id", jp.stem.replace("session_", ""))] = d

    by_group = defaultdict(list)
    by_route = defaultdict(list)

    for r in done:
        key = (r["model"], r["object"])
        rkey = (r["model"], r["object"], r["start_pos"], r["target_pos"])
        sid = r.get("session_id", "").strip()
        rep = rep_map.get(sid, {})
        lat = rep.get("summary", {}).get("avg_latency_ms")
        dist = None
        if r.get("remain_dist_cm", "").strip():
            try:
                dist = float(r["remain_dist_cm"])
            except ValueError:
                dist = None
        rec = dict(
            success=(r["outcome"].strip() == "성공"),
            contact=(r.get("contact_yn", "").strip().upper() == "Y"),
            dist_cm=dist,
            lat_ms=float(lat) if lat is not None else None,
        )
        by_group[key].append(rec)
        by_route[rkey].append(rec)

    summary = {}
    print("\n| 모델 | 객체 | N | 성공률 (%) | 접촉률 (%) | 잔여거리 (cm) | 평균 지연 (ms) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for (m, obj), items in sorted(by_group.items()):
        n = len(items)
        sr = sum(x["success"] for x in items) / n * 100
        cr = sum(x["contact"] for x in items) / n * 100
        dists = [x["dist_cm"] for x in items if x["dist_cm"] is not None]
        lats = [x["lat_ms"] for x in items if x["lat_ms"] is not None]
        d_str = f"{np.mean(dists):.1f} ± {np.std(dists):.1f}" if dists else "-"
        l_str = f"{np.mean(lats):.1f}" if lats else "-"
        print(f"| {m} | {obj} | {n} | {sr:.1f}% | {cr:.1f}% | {d_str} | {l_str} |")
        summary[f"{m}|{obj}"] = dict(
            n=n,
            success_rate=round(sr, 1),
            contact_rate=round(cr, 1),
            dist_mean_cm=round(float(np.mean(dists)), 1) if dists else None,
            dist_std_cm=round(float(np.std(dists)), 1) if dists else None,
            lat_mean_ms=round(float(np.mean(lats)), 1) if lats else None,
        )

    routes = {
        f"{m}|{obj}|{sp}|{tp}": round(sum(x["success"] for x in items) / len(items) * 100, 1)
        for (m, obj, sp, tp), items in sorted(by_route.items())
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"summary": summary, "routes": routes}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved aggregated JSON: {out_json}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-sheet", action="store_true", help="현장 기록 CSV 템플릿 생성")
    ap.add_argument("--reps", type=int, default=3, help="조건당 반복 횟수 (기본 3 -> 108행, 5 -> 180행)")
    ap.add_argument("--sheet", type=Path, default=DEFAULT_SHEET)
    ap.add_argument("--reports-dir", type=Path, default=None)
    ap.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    args = ap.parse_args()

    if args.init_sheet:
        init_sheet(args.sheet, reps=args.reps)
    else:
        aggregate(args.sheet, args.reports_dir, args.out_json)


if __name__ == "__main__":
    main()

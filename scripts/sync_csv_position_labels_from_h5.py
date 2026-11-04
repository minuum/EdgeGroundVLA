#!/usr/bin/env python3
"""H5 attrs(runtime_config)가 사후 일괄 치환된 session_id에 대해,
episode_log.csv의 '경로'/'런타임설정' 컬럼을 H5 기준으로 다시 맞춘다.

2026-11 pos4~6 x G6~8 2차 수집에서 H5 attrs만 pos4~6/G6~8로 치환되고
CSV는 원래 값(pos1/trackA_weak_*)으로 남아있던 불일치를 해소하기 위한
일회성 동기화 스크립트. H5가 source of truth.
"""
import csv
import glob
import json
import os
import shutil
import sys

CSV_PATH = "logs/episode_log.csv"
H5_DIR = "docs/inference_sessions"


def load_h5_runtime_config(session_id):
    import h5py

    path = os.path.join(H5_DIR, f"session_{session_id}.h5")
    if not os.path.exists(path):
        return None
    with h5py.File(path, "r") as f:
        raw = f.attrs.get("runtime_config")
        if raw is None:
            return None
        return json.loads(raw)


def main():
    with open(CSV_PATH, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        rows = list(reader)

    changed = 0
    checked = 0
    for row in rows:
        sid = row.get("session_id") or ""
        if not sid:
            continue
        h5_rc = load_h5_runtime_config(sid)
        if h5_rc is None:
            continue
        checked += 1
        try:
            csv_rc = json.loads(row.get("런타임설정") or "{}")
        except Exception:
            csv_rc = {}

        h5_sp = h5_rc.get("start_position")
        h5_tp = h5_rc.get("target_position")
        csv_sp = csv_rc.get("start_position")
        csv_tp = csv_rc.get("target_position")

        if h5_sp == csv_sp and h5_tp == csv_tp:
            continue

        csv_rc["start_position"] = h5_sp
        csv_rc["target_position"] = h5_tp
        row["런타임설정"] = json.dumps(csv_rc, ensure_ascii=False)
        if h5_tp:
            row["경로"] = h5_tp
        changed += 1

    print(f"checked={checked} changed={changed}")

    if changed == 0:
        print("no changes needed")
        return

    if "--apply" not in sys.argv:
        print("dry-run only — pass --apply to write changes")
        return

    backup_path = CSV_PATH + ".bak-pre-pos-sync"
    shutil.copy2(CSV_PATH, backup_path)
    print(f"backed up original to {backup_path}")

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {CSV_PATH}")


if __name__ == "__main__":
    main()

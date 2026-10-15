#!/usr/bin/env python3
"""EGVLA2 본학습 세트 `egvla2_v3_main` 정의(메타 JSON) 생성.

plan: docs/plans/plan_20261015_egvla2_300ep_pipeline.md §1
규칙: action_space==egvla2_continuous_v3, 읽기 가능, 프레임 수 일치, NaN 없음, 20프레임 이상, 제외 목록(meta/egvla2_v3_excluded.json) 미포함.
각 파일에 (위치, 경로) 셀과 '가이드 이후 여부'(속력 분석용)를 기록한다. 가이드 이후 = 10/14 이후 수집분(속력 가이드는 로봇 세션이 아직 미적용이라 현재는 참고용 분리).
실행: .venv/bin/python3 -I scripts/build_egvla2_main_set.py
"""
import json
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "ROS_action" / "mobile_vla_dataset_egvla2"
META = ROOT / "docs/v5/egvla2_fm/meta"
OUT = META / "egvla2_v3_main.json"


def main():
    excl_p = META / "egvla2_v3_excluded.json"
    excluded = set(json.loads(excl_p.read_text())) if excl_p.exists() else set()
    files, rejected = [], {}
    for p in sorted(DATA.glob("*.h5")):
        try:
            with h5py.File(p, "r") as f:
                if f.attrs.get("action_space") != "egvla2_continuous_v3":
                    continue
                n = len(f["actions_exec"])
                ok = (len(f["actions_raw"]) == n == len(f["actions_scaled"]) == len(f["images"]) == len(f["frame_timestamps"]))
                bad = (not ok) or n < 20 or any(np.isnan(np.asarray(f[k])).any() for k in ("actions_exec", "actions_raw"))
                if bad:
                    rejected[p.name] = "length/NaN/too short"
                    continue
                if p.name in excluded:
                    rejected[p.name] = "excluded list"
                    continue
                pos, path = str(f.attrs.get("cx_position", "")), str(f.attrs.get("cx_path", ""))
                files.append(dict(name=p.name, position=pos or "unlabeled", path=path or "unlabeled", frames=int(n),
                                  date=p.name.split("_")[1]))
        except Exception as e:  # noqa: BLE001
            rejected[p.name] = f"read error: {e}"
    # 300ep 정량 기준: 15칸(5위치×3경로) 각 PER_CELL건 = 300. 셀당 초과분·위치 라벨 없는 파일은 학습에서 빼고 reserve로 따로 지정(시간 순 앞 PER_CELL건 유지).
    per_cell, cells, keep, reserve = 20, {}, [], []
    for d in sorted(files, key=lambda x: x["name"]):
        (cells.setdefault((d["position"], d["path"]), [])).append(d)
    for (pos, path), v in sorted(cells.items()):
        if pos == "unlabeled" or path == "unlabeled":
            reserve += [dict(d, reason="위치 라벨 없음(10/13 파일럿)") for d in v]
        else:
            keep += v[:per_cell]
            reserve += [dict(d, reason=f"셀 초과분(>{per_cell}, 시간 순 뒤쪽)") for d in v[per_cell:]]
    short = {f"{p}|{q}": per_cell - sum(1 for d in keep if (d["position"], d["path"]) == (p, q))
             for p in ("center", "weak_left", "strong_left", "weak_right", "strong_right") for q in ("left_curve", "straight", "right_curve")}
    short = {k: v for k, v in short.items() if v > 0}
    files = keep
    for d in files:
        d["post_guide"] = False  # 가이드(§12.3) 적용 이후 수집분이 생기면 이 목록에서 갱신
    OUT.write_text(json.dumps(dict(set="egvla2_v3_main", n=len(files), frames=sum(d["frames"] for d in files),
                                   rejected=rejected, per_cell=per_cell, short_cells=short, reserve=reserve, files=files), indent=1, ensure_ascii=False))
    print(f"reserve {len(reserve)}건, 20건 미달 셀 {short}")
    print(f"{len(files)} files, {sum(d['frames'] for d in files)} frames, rejected {len(rejected)} -> {OUT}")


if __name__ == "__main__":
    main()
